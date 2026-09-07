"""Coordinate thread execution, host input, and durable child-to-parent joins."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence
from contextlib import aclosing
from itertools import islice
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..events import (
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
from ..messages import LLMToolMessage
from .agent_thread import AgentThread
from .context_utils import open_tool_call_ids
from .internal_events import AppendContext, CreateSubAgent, ThreadFinished
from .metrics import ThreadMetrics

__all__ = ["ExecutionOutcome", "Orchestrator", "SubAgentFactory"]

MAX_PARALLEL_SUB_AGENTS = 5
SubAgentFactory = Callable[..., Awaitable[AgentThread]]


class ExecutionOutcome(BaseModel):
    """Result of execution, including host actions required before continuation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    output: ModelMessage | None = None
    required_actions: list[Any] = Field(default_factory=list)
    error: str | None = None


class Orchestrator:
    """Own the live thread registry for a turn containing one root agent.

    Each scheduling round selects leaves from the registry. A paused batch is
    allowed to finish its current streams, but prevents later batches from
    starting. The supplied factory resolves definitions and resources for newly
    requested children; this class owns their registration and retirement.
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
        self.outcome = ExecutionOutcome()
        self._finished_metrics = ThreadMetrics()
        self._sub_agent_factory = sub_agent_factory
        self._tracer = tracer or NOOP_TRACER

    @property
    def root(self) -> AgentThread:
        candidate = next((thread for thread in self.threads.values() if thread.parent is None), None)
        if candidate is None:
            raise RuntimeError("Orchestrator has no root thread")
        return candidate

    def snapshots(self) -> dict[str, Any]:
        return {key: value.snapshot() for key, value in self.threads.items()}

    def metrics(self) -> ThreadMetrics:
        combined = self._finished_metrics.model_copy(deep=True)
        for live in self.threads.values():
            combined.add(live.metrics)
        return combined

    def _active_threads(self) -> list[AgentThread]:
        candidates = dict(self.threads)
        for thread in self.threads.values():
            if thread.parent is not None:
                candidates.pop(thread.parent.thread_id, None)
        return list(candidates.values())

    async def send(self, inputs: Sequence[Any]) -> AsyncIterator[Any]:
        """Validate a complete input batch before mutating any thread journal."""
        from ...errors import InvalidSendInputError

        routed: dict[str, list[Any]] = {key: [] for key in self.threads}
        user_inputs = []
        for item in inputs:
            if getattr(item, "type", None) == "user.message":
                user_inputs.append(item)
                continue
            destination = getattr(item, "thread_id", None)
            if destination not in routed:
                raise InvalidSendInputError(f"Unknown thread_id: {destination!r}")
            routed[destination].append(item)
        if user_inputs:
            if len(routed) > 1:
                raise InvalidSendInputError(
                    "Cannot send a user message while sub-agents are running. "
                    "Resume their work with an empty input before sending a new message."
                )
            routed[self.root.thread_id].extend(user_inputs)

        deliveries = [(self.threads[key], batch) for key, batch in routed.items()]
        for thread, batch in deliveries:
            thread.validate_input(batch)
        for thread, batch in deliveries:
            async for event in thread.send(batch):
                yield event

    async def execute(self, cancel: asyncio.Event | None = None) -> AsyncGenerator[Any, None]:
        """Yield execution events and set ``outcome`` after all selected work drains."""
        result = ExecutionOutcome()
        auth_requests: list[McpServerAuth] = []
        paused = False
        with self._tracer.span("agento.turn", thread_count=len(self.threads)):
            while self.threads and not paused and not (cancel is not None and cancel.is_set()):
                async for recovered in self._recover_joins():
                    yield recovered
                ready = iter(self._active_threads())
                batch = list(islice(ready, MAX_PARALLEL_SUB_AGENTS))
                if not batch:
                    break
                while batch:
                    async with aclosing(_merge([thread.execute(cancel) for thread in batch])) as events:
                        async for event in events:
                            if isinstance(event, McpAuthRequired):
                                auth_requests.extend(event.mcp_servers)
                                paused = True
                                continue
                            if isinstance(event, (ApprovalRequired, ClientToolRequired)):
                                result.required_actions.append(event)
                                paused = True
                            elif isinstance(event, ThreadFinished) and event.parent is None:
                                if event.status == "error":
                                    result.error = event.error
                                else:
                                    result.output = event.output
                                paused = True
                            async for forwarded in self._handle(event, cancel):
                                yield forwarded
                    if paused:
                        break
                    batch = list(islice(ready, MAX_PARALLEL_SUB_AGENTS))

        if auth_requests:
            auth = _merge_auth(auth_requests)
            yield auth
            result.required_actions.append(auth)
        self.outcome = result

    async def _handle(self, event: Any, cancel: asyncio.Event | None) -> AsyncIterator[Any]:
        if isinstance(event, CreateSubAgent):
            stream = self._spawn(event, cancel)
        elif isinstance(event, ThreadFinished):
            stream = self._finish(event)
        else:
            yield event
            return
        async for forwarded in stream:
            yield forwarded

    async def _spawn(self, event: CreateSubAgent, cancel: asyncio.Event | None) -> AsyncIterator[Any]:
        from ..._ids import new_thread_id

        factory = self._sub_agent_factory
        if factory is None:
            raise RuntimeError("Sub-agent creation requires a configured sub_agent_factory.")
        relation = ThreadParent(thread_id=event.thread_id, tool_call_id=event.tool_call_id)
        identifier = new_thread_id()
        self.threads[identifier] = await factory(
            parent_definition=self.threads[event.thread_id].definition,
            request=event.agent_info,
            thread_id=identifier,
            parent=relation,
        )
        yield ThreadCreated(
            thread_id=identifier, parent=relation, agent_info=event.agent_info, title=event.agent_info.name,
        )

    async def _finish(self, event: ThreadFinished) -> AsyncIterator[Any]:
        if event.parent is None:
            return
        recipient = self.threads.get(event.parent.thread_id)
        reply = event.send_to_parent
        checkpoint: AppendContext | None = None
        public: ToolResult | None = None
        if recipient is not None and reply is not None:
            if event.parent.tool_call_id in open_tool_call_ids(recipient.context):
                public = ToolResult(
                    thread_id=event.parent.thread_id, tool_call_id=reply.tool_call_id, content=reply.content,
                )
                checkpoint = recipient.deliver_tool_message(reply)
        state: ThreadStateDone | ThreadStateError
        if event.status == "error":
            state = ThreadStateError(error=event.error or "Sub-agent failed", output=event.output)
        else:
            state = ThreadStateDone(output=event.output)  # type: ignore[arg-type]
        done = ThreadDone(thread_id=event.thread_id, parent=event.parent, title=event.title, state=state)
        self._retire(event.thread_id)
        if checkpoint is not None and public is not None:
            # The snapshot, parent result, and terminal child event form one
            # checkpoint. Public delivery still presents the result first.
            checkpoint.events = [public, done]
            yield checkpoint
            yield public
        yield done

    def _retire(self, thread_id: str) -> None:
        retired = self.threads.pop(thread_id, None)
        if retired is not None:
            self._finished_metrics.add(retired.metrics)

    async def _recover_joins(self) -> AsyncIterator[Any]:
        """Retire children already answered in snapshots written before atomic joins.

        The parent's result proves delivery, so running the child again would
        repeat completed work. Replay terminal details when a completion exists.
        Otherwise preserve the result and checkpoint retirement without inventing
        the historic success/error status that the snapshot did not retain.
        """
        for child in tuple(self.threads.values()):
            relation = child.parent
            if relation is None:
                continue
            parent = self.threads.get(relation.thread_id)
            if parent is None or relation.tool_call_id in open_tool_call_ids(parent.context):
                continue
            answered = any(
                isinstance(message, LLMToolMessage) and message.tool_call_id == relation.tool_call_id
                for message in parent.context
            )
            if not answered:
                continue
            completion = child.snapshot().completion
            if completion is not None:
                event = ThreadFinished(
                    thread_id=child.thread_id, parent=relation, title=child.title,
                    status=completion.status, output=completion.output, error=completion.error,
                )
                async for done in self._finish(event):
                    yield done
            else:
                self._retire(child.thread_id)
                yield AppendContext(thread_id=relation.thread_id)


def _merge_auth(servers: Sequence[McpServerAuth]) -> McpAuthRequired:
    identifiers: set[str] = set()
    unique = []
    for server in servers:
        if server.id not in identifiers:
            identifiers.add(server.id)
            unique.append(server)
    return McpAuthRequired(thread_id=None, mcp_servers=unique)


class _StreamCursor:
    """A single producer task and its next result slot."""

    def __init__(self, source: AsyncIterator[Any]) -> None:
        self.source = source
        self.ready: asyncio.Future[tuple[bool, Any]] = asyncio.get_running_loop().create_future()
        self.resume = asyncio.Event()
        self.started = False

    def request_next(self) -> asyncio.Future[tuple[bool, Any]]:
        self.ready = asyncio.get_running_loop().create_future()
        self.resume.set()
        return self.ready

    async def close(self) -> None:
        close = getattr(self.source, "aclose", None)
        if close is not None:
            try:
                await close()
            except Exception:  # pragma: no cover - continue closing the remaining sources
                pass

    async def run(self) -> None:
        self.started = True
        try:
            while True:
                try:
                    value = await anext(self.source)
                except StopAsyncIteration:
                    self.ready.set_result((False, None))
                    return
                self.ready.set_result((True, value))
                await self.resume.wait()
                self.resume.clear()
        except asyncio.CancelledError:
            if not self.ready.done():
                self.ready.cancel()
            raise
        except Exception as error:
            self.ready.set_exception(error)
        finally:
            await self.close()


async def _merge(generators: Sequence[AsyncIterator[Any]]) -> AsyncGenerator[Any, None]:
    """Wait on independent result slots, acknowledging each event after its yield.

    Each iterator remains on one producer task, including during close, so its
    context managers keep the same task/context throughout execution. Result
    slots hold at most one event per source; no producer advances while its
    published event is waiting for the consumer to checkpoint it.
    """
    cursors = [_StreamCursor(source) for source in generators]
    workers = [asyncio.create_task(cursor.run()) for cursor in cursors]
    pending = {cursor.ready: cursor for cursor in cursors}
    try:
        while pending:
            completed, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for slot in completed:
                cursor = pending.pop(slot)
                present, value = slot.result()
                if present:
                    yield value
                    pending[cursor.request_next()] = cursor
    finally:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        for cursor in cursors:
            if not cursor.started:
                await cursor.close()
            if cursor.ready.done() and not cursor.ready.cancelled():
                cursor.ready.exception()
