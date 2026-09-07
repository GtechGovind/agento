"""One turn: running it, streaming it, and making it durable.

:meth:`TurnHandle.stream` *is* the execution. Nothing runs until you iterate it,
and it can only be iterated once — the generator and the run are the same thing.

Two consumption patterns, both supported by the same method::

    # Stream to a client as it happens
    async for event in turn.stream():
        await websocket.send_json(event.model_dump())

    # Or run it in the background and return immediately
    task = asyncio.create_task(turn.drain())
    return {"turn_id": turn.id}          # the client polls or subscribes later

**Persist before yield.** Every durable event is written to the store *before* it
is handed to you. Explicitly closing the stream finalizes its committed work. Continued execution
after a client disconnect requires a host-owned background task.

**One terminal write.** The turn's final state is written here, in a ``finally``,
on completion, error, cancellation, or explicit generator closure. A turn that ends without a terminal state would sit "running"
forever and block the session, so cleanup attempts this write. Process death or a storage outage still needs
host-supervised recovery.

If the store reports the turn already reached a terminal state — because
something else cancelled it — that state wins and is reported back. A late
completion never overwrites a cancellation.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from ..core.events import (
    ArtifactCreated,
    ContextCompacted,
    McpAuthRequired,
    McpInitialized,
    ModelMessage,
    ModelMessageDelta,
    StreamEvent,
    ThreadCreated,
    ThreadDone,
    ToolResult,
    TurnCreated,
    TurnDone,
    TurnState,
    TurnStateCancelled,
    TurnStateDone,
    TurnStateError,
)
from ..core.runtime.internal_events import AppendContext, ReplaceContext, SetState, ThreadFinished
from ..core.runtime.orchestrator import Orchestrator
from ..errors import TurnNotRunningError
from .store.base import Page, SessionStore, TurnRecord, TurnSnapshot

__all__ = ["TurnHandle"]

_DURABLE_EVENTS = (
    ToolResult,
    ThreadCreated,
    ThreadDone,
    McpInitialized,
    McpAuthRequired,
    ArtifactCreated,
    ContextCompacted,
)


class TurnHandle:
    """A turn — durable, and executable exactly once.

    Returned by :meth:`~agento.session.session_handle.SessionHandle.create_turn`
    (executable) and :meth:`~agento.session.session_handle.SessionHandle.get_turn`
    (read-only).
    """

    def __init__(
        self,
        *,
        store: SessionStore,
        record: TurnRecord,
        orchestrator: Orchestrator | None = None,
        resolver: Any = None,
        cancel: asyncio.Event | None = None,
        initial_events: list[Any] | None = None,
    ) -> None:
        self._store = store
        self._record = record
        self._orchestrator = orchestrator
        self._resolver = resolver
        self._cancel = cancel or asyncio.Event()
        self._started = False
        self._initial_events = list(initial_events or [])
        self._persisted_ids: set[str] = set()

    # ------------------------------------------------------------------ #
    # Identity                                                           #
    # ------------------------------------------------------------------ #

    @property
    def id(self) -> str:
        """The turn id."""
        return self._record.turn_id

    @property
    def session_id(self) -> str:
        return self._record.session_id

    @property
    def previous_turn_id(self) -> str | None:
        return self._record.previous_turn_id

    @property
    def state(self) -> TurnState:
        """The turn's state, as of the last write."""
        return self._record.state

    @property
    def record(self) -> TurnRecord:
        """The full stored record."""
        return self._record

    @property
    def is_executable(self) -> bool:
        """Whether :meth:`stream` can run this turn.

        False for a turn loaded from storage: execution belongs to the process
        that created it.
        """
        return self._orchestrator is not None

    def cancel(self, reason: str = "client-cancelled") -> None:
        """Ask the run to stop.

        Cooperative: the loop checks between steps, so an in-flight model call or
        tool completes first and the turn is left in a resumable state.
        """
        self._cancel_reason = reason
        self._cancel.set()

    # ------------------------------------------------------------------ #
    # Execution                                                          #
    # ------------------------------------------------------------------ #

    async def stream(self) -> AsyncIterator[StreamEvent]:
        """Run the turn, yielding events as they happen.

        Yields:
            :class:`~agento.core.events.TurnCreated` first, then model messages,
            deltas, tool results, sub-agent lifecycle events, and finally
            :class:`~agento.core.events.TurnDone`.

        Raises:
            RuntimeError: Called twice, or on a turn loaded from storage.
        """
        if self._started:
            raise RuntimeError(
                f"Turn {self.id!r} has already been run. stream() is the execution, so it can "
                "only be consumed once. Use list_events() to re-read what happened."
            )
        if self._orchestrator is None:
            raise RuntimeError(
                f"Turn {self.id!r} was loaded from storage and cannot be run. Only a turn from "
                "create_turn() is executable."
            )
        self._started = True

        orchestrator = self._orchestrator
        failure: BaseException | None = None
        frozen_state: TurnState | None = None

        execution = orchestrator.execute(self._cancel)
        try:
            created = TurnCreated(
                turn_id=self.id,
                previous_turn_id=self._record.previous_turn_id,
                input=list(self._record.input),
                created_at=self._record.created_at.isoformat(),
            )
            await self._checkpoint([created, *self._initial_events])
            yield created
            for initial in self._initial_events:
                yield initial

            async for event in execution:
                try:
                    public = await self._persist(event)
                except TurnNotRunningError as exc:
                    # Something else finished this turn while it was running.
                    # Stop, and report the state that actually won.
                    frozen_state = exc.state
                    break
                if public is not None:
                    yield public

        except (GeneratorExit, asyncio.CancelledError) as exc:
            failure = exc
            raise
        except Exception as exc:
            failure = exc
        except BaseException as exc:
            failure = exc
            raise
        finally:
            try:
                await execution.aclose()
                state = frozen_state or self._terminal_state(orchestrator, failure)
                done = TurnDone(state=state)
                if frozen_state is None:
                    try:
                        snapshot = self._snapshot(orchestrator)
                        await self._store.update_turn(
                            self.session_id, self.id, state=state,
                            snapshot=snapshot, events=[done],
                        )
                        self._record.snapshot = snapshot
                    except TurnNotRunningError as exc:
                        state = exc.state
                        done = TurnDone(state=state)
                self._record.state = state
            finally:
                await self._close_resolver()
        # Never yield inside finally: aclose() and task cancellation must unwind.
        yield done

    async def drain(self) -> TurnState:
        """Run the turn to completion without consuming the events yourself.

        Everything is still persisted, so a client can read it back with
        :meth:`list_events` or a live subscription. Use this when the HTTP
        response should return before the agent finishes::

            task = asyncio.create_task(turn.drain())

        Returns:
            The turn's terminal state.
        """
        async for _ in self.stream():
            pass
        return self._record.state

    async def wait(self) -> TurnState:
        """Alias for :meth:`drain`, for readability at call sites."""
        return await self.drain()

    async def final_output(self) -> str:
        """Run the turn and return the agent's final message as text.

        The convenience path for a script that wants an answer rather than a
        stream. Returns an empty string if the turn ended paused or errored —
        check :attr:`state` when that matters.
        """
        from ..core.messages import text_of

        state = await self.drain()
        output = getattr(state, "output", None)
        return text_of(getattr(output, "content", None)) if output else ""

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #

    async def _persist(self, event: Any) -> StreamEvent | None:
        """Write one event's effects, and return it if the consumer should see it.

        Returning ``None`` means the event was internal bookkeeping: its effect is
        stored, but there is nothing meaningful to show.
        """
        orchestrator = self._orchestrator
        assert orchestrator is not None

        if isinstance(event, ModelMessageDelta):
            # Never persisted: the complete ModelMessage carries everything.
            return event

        if isinstance(event, ModelMessage):
            # Complete messages were committed by the preceding AppendContext.
            # The initial empty placeholder is transient, like token deltas.
            return event

        if isinstance(event, AppendContext):
            await self._checkpoint(list(event.events))
            return None

        if isinstance(event, ReplaceContext):
            await self._checkpoint([event.event] if event.event is not None else [])
            return None

        if isinstance(event, SetState):
            await self._save_snapshot(orchestrator)
            return None

        if isinstance(event, ThreadFinished):
            # The orchestrator translates a sub-agent's finish into ThreadDone,
            # and the root's into the turn's terminal state.
            return None

        if isinstance(event, _DURABLE_EVENTS):
            if event.id not in self._persisted_ids:
                await self._checkpoint([event])
            return event

        return event

    def _snapshot(self, orchestrator: Orchestrator) -> TurnSnapshot:
        """Current thread state, plus observed MCP session ids for diagnostics."""
        sessions = self._resolver.mcp_sessions() if self._resolver is not None else {}
        return TurnSnapshot(threads=orchestrator.snapshots(), mcp_sessions=sessions)

    async def _save_snapshot(self, orchestrator: Orchestrator) -> None:
        await self._checkpoint([])

    async def _checkpoint(self, events: list[Any]) -> None:
        assert self._orchestrator is not None
        snapshot = self._snapshot(self._orchestrator)
        fresh = [event for event in events if event.id not in self._persisted_ids]
        await self._store.update_turn(self.session_id, self.id, snapshot=snapshot, events=fresh)
        self._persisted_ids.update(event.id for event in fresh)
        self._record.snapshot = snapshot

    def _terminal_state(
        self, orchestrator: Orchestrator, failure: BaseException | None
    ) -> TurnState:
        """Decide how the turn ended.

        Order matters. Cancellation is checked first because a cancelled turn
        often *also* raises ``CancelledError``, and reporting it as an error
        would misrepresent something the user asked for.
        """
        metrics = orchestrator.metrics().to_turn_metrics()

        if self._cancel.is_set():
            return TurnStateCancelled(
                reason=getattr(self, "_cancel_reason", "client-cancelled"), metrics=metrics
            )

        if isinstance(failure, GeneratorExit):
            return TurnStateCancelled(reason="stream-closed", metrics=metrics)

        if isinstance(failure, asyncio.CancelledError):
            return TurnStateCancelled(reason="task-cancelled", metrics=metrics)

        if failure is not None:
            message = str(failure).strip() or f"{type(failure).__name__} (no message)"
            return TurnStateError(message=message, metrics=metrics)

        outcome = orchestrator.outcome
        if outcome.error:
            return TurnStateError(message=outcome.error, metrics=metrics)

        return TurnStateDone(
            output=outcome.output,
            required_actions=list(outcome.required_actions),
            metrics=metrics,
        )

    async def _close_resolver(self) -> None:
        if self._resolver is None:
            return
        resolver, self._resolver = self._resolver, None
        try:
            await resolver.aclose()
        except Exception:  # pragma: no cover - cleanup must never fail a turn
            pass

    # ------------------------------------------------------------------ #
    # Reading                                                            #
    # ------------------------------------------------------------------ #

    async def list_events(
        self, *, limit: int = 100, cursor: str | None = None, order: str = "asc"
    ) -> Page:
        """Read this turn's persisted events.

        Works during the run as well as after it, which is what lets a second
        client follow along.
        """
        return await self._store.list_turn_events(
            self.session_id, self.id, limit=limit, cursor=cursor, order=order
        )

    async def refresh(self) -> TurnHandle:
        """Re-read this turn from the store."""
        record = await self._store.get_turn(self.session_id, self.id)
        if record is not None:
            self._record = record
        return self

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"TurnHandle({self.id!r}, state={self._record.state.status})"
