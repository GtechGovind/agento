"""A session — a conversation, and the turns that make it up.

:meth:`SessionHandle.create_turn` is where a turn is prepared: threads are
rebuilt from the previous turn's snapshot, the new input is validated and
appended, and only then is anything written to the store. Execution happens
afterwards, when you iterate the returned
:class:`~agento.session.turn_handle.TurnHandle`.

That order is deliberate. Validation runs *before* the write, so a rejected input
— a user message sent while an approval is pending, an approval for a call that
does not exist — leaves no half-created turn in your database. Either the turn
exists and is valid, or nothing happened.

**Turns form a tree.** ``previous_turn_id`` says where a turn continues from:

``"auto"`` (default)
    Continue from the session tip. What you want almost always.
``"none"``
    Start a fresh root turn, with no history. A new conversation in the same
    session record.
a turn id
    Branch from that point. The turns after it stay in the store, untouched;
    the session simply continues down the new branch. This is how you implement
    "edit and resend".
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from .._ids import new_id
from ..core.events import ToolApproval, ToolReply, TurnInput, UserMessage
from ..core.messages import text_of
from ..core.runtime.agent_thread import AgentThread, ThreadSnapshot
from ..core.runtime.orchestrator import Orchestrator
from .agent import Agent
from .resolver import ResourceResolver
from .store.base import Page, SessionRecord, SessionStore, TurnRecord, TurnSnapshot
from .turn_handle import TurnHandle

__all__ = ["SessionHandle"]

MAX_ANCESTORS = 20
"""How many recent ancestors a turn records. Older ones are reachable by
following those turns' own lists."""

MAX_TITLE_LENGTH = 60

TurnInputLike = str | TurnInput | Sequence[str | TurnInput] | None
"""What ``create_turn(input=...)`` accepts: a string, one input item, a list of
either, or nothing (to resume a paused turn)."""


def _normalize_input(value: TurnInputLike) -> list[TurnInput]:
    """Accept the convenient forms and produce a list of input items."""
    if value is None:
        return []
    if isinstance(value, str):
        return [UserMessage(content=value)]
    if isinstance(value, (UserMessage, ToolApproval, ToolReply)):
        return [value]
    items: list[TurnInput] = []
    for entry in value:
        if isinstance(entry, str):
            items.append(UserMessage(content=entry))
        else:
            items.append(entry)
    return items


def _derive_title(inputs: Sequence[TurnInput]) -> str | None:
    """A session title from the first user message.

    Only used for the first turn, and never overwrites a title you set.
    """
    for item in inputs:
        if isinstance(item, UserMessage):
            text = text_of(item.content).strip()
            if text:
                return text[:MAX_TITLE_LENGTH]
    return None


class SessionHandle:
    """One conversation.

    Args:
        store: Where sessions and turns live.
        record: The stored session.
        agent: The live agent definition, with its tools. Kept in memory because
            functions cannot be serialized — a session reloaded in a fresh
            process gets its agent from
            :class:`~agento.session.agento.Agento`'s registry.
        runtime: The owning :class:`~agento.session.agento.Agento`, which
            supplies model, connector, skill and artifact configuration.
    """

    def __init__(
        self,
        *,
        store: SessionStore,
        record: SessionRecord,
        agent: Agent | None = None,
        runtime: Any = None,
    ) -> None:
        self._store = store
        self._record = record
        self._agent = agent
        self._runtime = runtime

    # ------------------------------------------------------------------ #
    # Identity                                                           #
    # ------------------------------------------------------------------ #

    @property
    def id(self) -> str:
        return self._record.session_id

    @property
    def title(self) -> str | None:
        return self._record.title

    @property
    def metadata(self) -> dict[str, str]:
        """Your identifiers. Visible to every tool through the tool context."""
        return self._record.metadata

    @property
    def last_turn_id(self) -> str | None:
        """The session tip."""
        return self._record.last_turn_id

    @property
    def record(self) -> SessionRecord:
        return self._record

    @property
    def agent(self) -> Agent:
        """The live agent.

        Raises:
            ConfigurationError: The session was loaded from storage and no live
                agent is available. Register it on ``Agento(agents=...)`` or pass
                ``agent=`` when loading the session.
        """
        if self._agent is None:
            from ..errors import ConfigurationError

            raise ConfigurationError(
                f"Session {self.id!r} has no live agent. Agent definitions include Python "
                "functions, which cannot be stored, so a session loaded in a new process needs "
                "its agent supplied — register it with Agento(agents=[...]) or pass "
                "agent=... to sessions.get()."
            )
        return self._agent

    # ------------------------------------------------------------------ #
    # Turns                                                              #
    # ------------------------------------------------------------------ #

    async def create_turn(
        self,
        input: TurnInputLike = None,
        *,
        previous_turn_id: str = "auto",
        turn_id: str | None = None,
        custom: dict[str, Any] | None = None,
        cancel: asyncio.Event | None = None,
    ) -> TurnHandle:
        """Prepare a turn. Nothing runs until you iterate the result.

        Args:
            input: A string, an input item, or a list. Omit to resume a turn that
                is waiting on nothing in particular.
            previous_turn_id: ``"auto"`` (the tip), ``"none"`` (a fresh root), or
                a turn id to branch from.
            turn_id: Supply your own id — useful for idempotency when a request
                may be retried.
            custom: Stored alongside the turn.
            cancel: Your own cancellation event, if you want to hold the handle.

        Returns:
            An executable :class:`~agento.session.turn_handle.TurnHandle`.

        Raises:
            InvalidSendInputError: The input is not valid for the conversation's
                current state — no turn record is persisted. Uploaded artifacts
                may already exist and need host retention/cleanup.
        """
        agent = self.agent
        inputs = _normalize_input(input)
        from ..errors import SessionNotFoundError

        refreshed = await self._store.get_session(self.id)
        if refreshed is None:
            raise SessionNotFoundError(self.id)
        self._record = refreshed
        expected_tip = (refreshed.last_turn_id,)

        previous = await self._resolve_previous(previous_turn_id)
        cancel_event = cancel or asyncio.Event()
        new_turn_id = turn_id or new_id()

        resolver = self._runtime.build_resolver(
            session_id=self.id,
            turn_id=new_turn_id,
            metadata=self._record.metadata,
        )

        try:
            threads = await self._build_threads(agent, previous, resolver)
            orchestrator = Orchestrator(
                threads,
                sub_agent_factory=resolver.sub_agent_factory(agent),
                tracer=self._runtime.tracer,
            )

            # Validate and append before anything is written. A failure here
            # leaves the store untouched.
            prepared_events: list[Any] = []
            from ..core.runtime.internal_events import AppendContext

            async for event in orchestrator.send(inputs):
                if isinstance(event, AppendContext):
                    prepared_events.extend(event.events)

            record = TurnRecord(
                turn_id=new_turn_id,
                session_id=self.id,
                previous_turn_id=previous.turn_id if previous else None,
                first_turn_id=previous.first_turn_id if previous else new_turn_id,
                ancestor_ids=(
                    [*previous.ancestor_ids, previous.turn_id][-MAX_ANCESTORS:] if previous else []
                ),
                input=inputs,
                snapshot=TurnSnapshot(
                    threads=orchestrator.snapshots(),
                    mcp_sessions=(previous.snapshot.mcp_sessions if previous else {}),
                ),
                custom=dict(custom or {}),
            )
            await self._store.create_turn(record, expected_tip=expected_tip)

            if not self._record.last_turn_id or not self._record.title:
                title = _derive_title(inputs)
                if title:
                    await self._store.update_session(self.id, set_title_if_absent=title)

            refreshed = await self._store.get_session(self.id)
            if refreshed is not None:
                self._record = refreshed

            return TurnHandle(
                store=self._store,
                record=record,
                orchestrator=orchestrator,
                resolver=resolver,
                cancel=cancel_event,
                initial_events=prepared_events,
            )
        except BaseException:
            # Anything acquired above is owned by this method until the handle
            # exists; from then on the handle's finally owns it.
            await resolver.aclose()
            raise

    async def run(self, input: TurnInputLike = None, **kwargs: Any) -> str:
        """Create a turn, run it, and return the final text.

        The one-liner for scripts and simple request handlers::

            answer = await session.run("What changed in the last deploy?")
        """
        turn = await self.create_turn(input, **kwargs)
        return await turn.final_output()

    async def stream(self, input: TurnInputLike = None, **kwargs: Any) -> Any:
        """Create a turn and return its event stream.

        ``async for event in await session.stream("hello")`` — the same as calling
        :meth:`create_turn` and then ``stream()``.
        """
        turn = await self.create_turn(input, **kwargs)
        return turn.stream()

    async def get_turn(self, turn_id: str) -> TurnHandle | None:
        """Load a turn from storage. Not executable."""
        record = await self._store.get_turn(self.id, turn_id)
        if record is None:
            return None
        return TurnHandle(store=self._store, record=record)

    async def list_turns(self, *, limit: int = 50, cursor: str | None = None) -> Page:
        """List this session's turns, newest first."""
        return await self._store.list_turns(self.id, limit=limit, cursor=cursor)

    async def list_events(self, *, limit: int = 100, cursor: str | None = None) -> Page:
        """Read the session's events across turns, newest first.

        Follows the active branch only, so a conversation that was edited and
        resent shows the branch it is actually on.
        """
        return await self._store.list_session_events(self.id, limit=limit, cursor=cursor)

    async def cancel_active_turn(self, reason: str = "client-cancelled") -> None:
        """Mark a running tip turn cancelled.

        For stopping a turn in *another* process, where you do not hold the
        handle. In the same process, call ``turn.cancel()`` — that stops the work;
        this only records the outcome.
        """
        from ..core.events import TurnStateCancelled

        refreshed = await self._store.get_session(self.id)
        if refreshed is not None:
            self._record = refreshed
        turn_id = self._record.last_turn_id
        if turn_id is None:
            return
        record = await self._store.get_turn(self.id, turn_id)
        if record is None or record.state.status != "running":
            return
        await self._store.update_turn(
            self.id, turn_id, state=TurnStateCancelled(reason=reason)
        )

    # ------------------------------------------------------------------ #
    # Session mutation                                                   #
    # ------------------------------------------------------------------ #

    async def set_title(self, title: str) -> None:
        """Set the session title."""
        await self._store.update_session(self.id, title=title)
        self._record.title = title

    async def update_metadata(self, metadata: dict[str, str]) -> None:
        """Replace the session metadata. Visible to tools from the next turn."""
        await self._store.update_session(self.id, metadata=metadata)
        self._record.metadata = dict(metadata)

    async def delete(self) -> None:
        """Delete this session and everything under it."""
        await self._store.delete_session(self.id)

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    async def _resolve_previous(self, previous_turn_id: str) -> TurnRecord | None:
        """Resolve a parent without changing another turn's state."""
        if previous_turn_id == "none":
            return None

        turn_id = (
            self._record.last_turn_id if previous_turn_id == "auto" else previous_turn_id
        )
        if turn_id is None:
            return None

        record = await self._store.get_turn(self.id, turn_id)
        if record is None:
            from ..errors import TurnNotFoundError

            raise TurnNotFoundError(turn_id)

        if record.state.status == "running":
            from ..errors import PreviousTurnRunningError

            raise PreviousTurnRunningError(turn_id)

        return record

    async def _build_threads(
        self,
        agent: Agent,
        previous: TurnRecord | None,
        resolver: ResourceResolver,
    ) -> dict[str, AgentThread]:
        """Rebuild every thread the previous turn left behind.

        A turn that ended mid-delegation had live sub-agent threads; they are
        rebuilt too, so the next turn resumes the fan-out rather than losing it.
        """
        resume = previous.snapshot.mcp_sessions if previous else {}

        if previous is None or not previous.snapshot.threads:
            main = await resolver.build_thread(agent, resume_mcp=resume)
            return {"main": main}

        threads: dict[str, AgentThread] = {}
        for thread_id, raw in previous.snapshot.threads.items():
            snapshot = (
                raw if isinstance(raw, ThreadSnapshot) else ThreadSnapshot.model_validate(raw)
            )
            threads[thread_id] = await resolver.build_thread(
                agent,
                thread_id=thread_id,
                title=(snapshot.agent_info.name if snapshot.agent_info else thread_id),
                context=snapshot.context,
                usage=snapshot.usage,
                parent=snapshot.parent,
                agent_info=snapshot.agent_info,
                completion=snapshot.completion,
                capability_state=snapshot.capability_state,
                resume_mcp=resume,
            )

        if "main" not in threads:  # pragma: no cover - a snapshot always has main
            threads["main"] = await resolver.build_thread(agent, resume_mcp=resume)
        return threads

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"SessionHandle({self.id!r}, title={self._record.title!r})"
