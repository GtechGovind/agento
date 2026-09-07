"""Running a root agent and its sub-agents together.

A turn is usually one thread. It stops being one the moment the agent delegates:
:class:`~agento.core.runtime.orchestrator.Orchestrator` owns the set of live
threads, runs the ones that can make progress *in parallel*, and routes each
child's result back into the parent's conversation.

**How a sub-agent joins back.** When the model calls ``create_sub_agent``, the
tool does not return a value — it yields a
:class:`~agento.core.runtime.internal_events.CreateSubAgent`, and the parent's
tool call stays **open**. An open tool call is exactly the loop's own
"I am waiting" state, so the parent naturally halts. When the child finishes, the
orchestrator writes the child's summary as the result of that call, and the
parent resumes on its next pass. There is no separate join mechanism, no
bookkeeping to get out of sync — the message list is the state.

**What runs in parallel.** Only *leaves*: threads that are not the parent of any
live thread. A parent with running children has nothing to do until they report
back. Leaves run in batches of :data:`MAX_PARALLEL_SUB_AGENTS`, which bounds
concurrent model spend when an agent fans out aggressively.

Cancellation is cooperative and checked between steps, so a stopped turn always
leaves each thread in a valid, resumable state.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence
from contextlib import aclosing
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..events import (
    ActionRequired,
    ApprovalRequired,
    ClientToolRequired,
    McpAuthRequired,
    McpServerAuth,
    ModelMessage,
    ThreadCreated,
    ThreadDone,
    ThreadParent,
    ThreadStateDone,
    ThreadStateError,
    ToolResult,
)
from .agent_thread import AgentThread
from .context_utils import open_tool_call_ids
from .internal_events import CreateSubAgent, ThreadFinished
from .metrics import ThreadMetrics

__all__ = ["ExecutionOutcome", "Orchestrator", "SubAgentFactory"]

MAX_PARALLEL_SUB_AGENTS = 5
"""How many threads run at once. Bounds concurrent model spend during a fan-out."""

SubAgentFactory = Callable[..., Awaitable[AgentThread]]
"""Builds a sub-agent thread.

Called with ``parent_definition``, ``request``, ``thread_id`` and ``parent``.
Supplied by the session layer, which is what knows how to resolve a model
override and rebuild the tool sets.
"""


class ExecutionOutcome(BaseModel):
    """What a turn amounted to.

    Attributes:
        output: The root agent's final message, or ``None`` when the turn ended
            paused before producing one.
        required_actions: What the host must resolve to continue. Non-empty means
            paused, not failed.
        error: Set when the root agent errored.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output: ModelMessage | None = None
    required_actions: list[Any] = Field(default_factory=list)
    error: str | None = None


class Orchestrator:
    """Runs a set of agent threads to completion.

    Args:
        threads: Live threads, keyed by id. Must contain exactly one root (a
            thread with no parent).
        sub_agent_factory: Builds a child thread on demand.
        tracer: Where spans go.
    """

    def __init__(
        self,
        threads: dict[str, AgentThread],
        *,
        sub_agent_factory: SubAgentFactory | None = None,
        tracer: Any = None,
    ) -> None:
        from ...tracing import NOOP_TRACER

        self.threads = threads
        self._sub_agent_factory = sub_agent_factory
        self._tracer = tracer or NOOP_TRACER
        self._finished_metrics = ThreadMetrics()
        self.outcome = ExecutionOutcome()

    # ------------------------------------------------------------------ #
    # Introspection                                                      #
    # ------------------------------------------------------------------ #

    @property
    def root(self) -> AgentThread:
        """The root thread."""
        for thread in self.threads.values():
            if thread.parent is None:
                return thread
        raise RuntimeError("Orchestrator has no root thread")

    def metrics(self) -> ThreadMetrics:
        """Totals across every thread, live and finished.

        Safe to call mid-run. A sub-agent moves from live to finished in one
        step, so it is never counted twice or missed.
        """
        total = ThreadMetrics()
        total.add(self._finished_metrics)
        for thread in self.threads.values():
            total.add(thread.metrics)
        return total

    def snapshots(self) -> dict[str, Any]:
        """Every live thread's durable state."""
        return {thread_id: thread.snapshot() for thread_id, thread in self.threads.items()}

    def _active_threads(self) -> list[AgentThread]:
        """Leaves — threads with no live children."""
        parents = {
            thread.parent.thread_id for thread in self.threads.values() if thread.parent is not None
        }
        return [thread for tid, thread in self.threads.items() if tid not in parents]

    # ------------------------------------------------------------------ #
    # Input                                                              #
    # ------------------------------------------------------------------ #

    async def send(self, inputs: Sequence[Any]) -> AsyncIterator[Any]:
        """Route a batch of input to the threads it belongs to.

        Approvals and tool replies carry a ``thread_id`` and go to that thread —
        a sub-agent can be the one waiting on an approval. User messages only
        ever go to the root, and only when no sub-agent is running: a fresh
        instruction arriving mid-delegation has no well-defined place in the
        conversation.
        """
        by_thread: dict[str, list[Any]] = {thread_id: [] for thread_id in self.threads}

        user_messages = [item for item in inputs if getattr(item, "type", "") == "user.message"]
        directed = [item for item in inputs if getattr(item, "type", "") != "user.message"]

        for item in directed:
            thread_id = getattr(item, "thread_id", None)
            if thread_id not in by_thread:
                from ...errors import InvalidSendInputError

                raise InvalidSendInputError(f"Unknown thread_id: {thread_id!r}")
            by_thread[thread_id].append(item)

        if user_messages:
            if len(self.threads) > 1:
                from ...errors import InvalidSendInputError

                raise InvalidSendInputError(
                    "Cannot send a user message while sub-agents are running. Let the current "
                    "turn finish first (send an empty input to resume it)."
                )
            by_thread[self.root.thread_id].extend(user_messages)

        # Validate everything before appending anything, so a rejected batch
        # leaves no thread half-updated.
        for thread_id, batch in by_thread.items():
            self.threads[thread_id].validate_input(batch)

        for thread_id, batch in by_thread.items():
            async for event in self.threads[thread_id].send(batch):
                yield event

    # ------------------------------------------------------------------ #
    # Execution                                                          #
    # ------------------------------------------------------------------ #

    async def execute(self, cancel: asyncio.Event | None = None) -> AsyncGenerator[Any, None]:
        """Run every thread to completion, or until the turn must pause.

        Yields:
            Public and internal events from all threads, interleaved.

        After the generator finishes, :attr:`outcome` holds the turn's result.
        """
        stop = False
        pending_auth: list[McpServerAuth] = []
        required_actions: list[ActionRequired] = []
        output: ModelMessage | None = None
        error: str | None = None

        with self._tracer.span("agento.turn", thread_count=len(self.threads)):
            while self.threads and not stop:
                if cancel is not None and cancel.is_set():
                    break

                active = self._active_threads()
                if not active:  # pragma: no cover - a parent always has a live child
                    break

                for start in range(0, len(active), MAX_PARALLEL_SUB_AGENTS):
                    if stop:
                        break
                    batch = active[start : start + MAX_PARALLEL_SUB_AGENTS]
                    generators = [thread.execute(cancel) for thread in batch]

                    async with aclosing(_merge(generators)) as merged_stream:
                        async for event in merged_stream:
                            if isinstance(event, McpAuthRequired):
                                pending_auth.extend(event.mcp_servers)
                                stop = True
                                continue

                            async for produced in self._handle(event, cancel):
                                yield produced

                            if isinstance(event, (ApprovalRequired, ClientToolRequired)):
                                required_actions.append(event)
                                stop = True
                            elif isinstance(event, ThreadFinished) and event.parent is None:
                                if event.status == "error":
                                    error = event.error
                                else:
                                    output = event.output
                                stop = True

        if pending_auth:
            merged = _merge_auth(pending_auth)
            yield merged
            required_actions.append(merged)

        self.outcome = ExecutionOutcome(
            output=output, required_actions=required_actions, error=error
        )

    async def _handle(self, event: Any, cancel: asyncio.Event | None) -> AsyncIterator[Any]:
        """Translate one thread event, spawning or retiring threads as needed."""
        if isinstance(event, CreateSubAgent):
            async for produced in self._spawn(event, cancel):
                yield produced
            return

        if isinstance(event, ThreadFinished):
            async for produced in self._finish(event):
                yield produced
            return

        yield event

    async def _spawn(self, event: CreateSubAgent, cancel: asyncio.Event | None) -> AsyncIterator[Any]:
        """Build and register a sub-agent thread."""
        if self._sub_agent_factory is None:
            raise RuntimeError(
                "A sub-agent was requested but no sub_agent_factory is configured. "
                "Disable sub-agents on the agent, or supply a factory."
            )

        from ..._ids import new_thread_id

        parent = ThreadParent(thread_id=event.thread_id, tool_call_id=event.tool_call_id)
        child_id = new_thread_id()
        child = await self._sub_agent_factory(
            parent_definition=self.threads[event.thread_id].definition,
            request=event.agent_info,
            thread_id=child_id,
            parent=parent,
        )

        self.threads[child_id] = child
        yield ThreadCreated(
            thread_id=child_id,
            parent=parent,
            agent_info=event.agent_info,
            title=event.agent_info.name,
        )

    async def _finish(self, event: ThreadFinished) -> AsyncIterator[Any]:
        """Retire a finished thread; deliver a sub-agent's result to its parent."""
        if event.parent is None:
            # The root finishing ends the turn; the turn layer writes the
            # terminal state, so nothing is emitted here.
            return

        thread = self.threads.get(event.thread_id)
        parent_thread = self.threads.get(event.parent.thread_id)

        # Deliver the child's summary as the result of the parent's open call.
        # This *is* the join: closing that call is what lets the parent resume.
        # Guarded on the call still being open, because a resumed turn may
        # already carry the result from the turn that was interrupted.
        if (
            parent_thread is not None
            and event.send_to_parent is not None
            and event.parent.tool_call_id in open_tool_call_ids(parent_thread.context)
        ):
            result = ToolResult(
                thread_id=event.parent.thread_id,
                tool_call_id=event.send_to_parent.tool_call_id,
                content=event.send_to_parent.content,
            )
            append = parent_thread.deliver_tool_message(event.send_to_parent)
            append.events = [result]
            yield append
            yield result

        state = (
            ThreadStateError(error=event.error or "Sub-agent failed", output=event.output)
            if event.status == "error"
            else ThreadStateDone(output=event.output)  # type: ignore[arg-type]
        )
        if thread is not None:
            # Move metrics and drop the thread in one step so a concurrent
            # metrics() call cannot see it twice or miss it.
            self._finished_metrics.add(thread.metrics)
            self.threads.pop(event.thread_id, None)

        yield ThreadDone(
            thread_id=event.thread_id,
            parent=event.parent,
            title=event.title,
            state=state,
        )



def _merge_auth(servers: Sequence[McpServerAuth]) -> McpAuthRequired:
    """Collapse per-thread auth requirements into one event, de-duplicated."""
    by_id: dict[str, McpServerAuth] = {}
    for server in servers:
        by_id.setdefault(server.id, server)
    return McpAuthRequired(thread_id=None, mcp_servers=list(by_id.values()))


async def _merge(generators: Sequence[AsyncIterator[Any]]) -> AsyncGenerator[Any, None]:
    """Interleave several async generators into one stream.

    Each source is pumped by its own task into a shared queue, so a slow thread
    never blocks a fast one. On early exit — a caller that stops consuming, or an
    exception — every pump is cancelled and every generator closed, so no task is
    left running against a stream nobody reads.
    """
    if len(generators) == 1:
        try:
            async for item in generators[0]:
                yield item
        finally:
            close = getattr(generators[0], "aclose", None)
            if close is not None:
                await close()
        return

    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    async def pump(generator: AsyncIterator[Any]) -> None:
        try:
            async for item in generator:
                ack = asyncio.Event()
                await queue.put(("item", (item, ack)))
                await ack.wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - forwarded to the consumer
            await queue.put(("error", exc))
        finally:
            await queue.put(("done", None))

    tasks = [asyncio.create_task(pump(generator)) for generator in generators]
    remaining = len(tasks)

    try:
        while remaining:
            kind, payload = await queue.get()
            if kind == "done":
                remaining -= 1
            elif kind == "error":
                raise payload
            else:
                item, ack = payload
                yield item
                ack.set()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for generator in generators:
            aclose = getattr(generator, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:  # pragma: no cover - best effort cleanup
                    pass
