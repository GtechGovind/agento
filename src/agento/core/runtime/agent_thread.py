"""Execute one conversation through queued model, tool, and input operations.

AgentThread owns the durable conversation. Each execute() call creates a short-
lived driver whose queue contains only the next operation. Operations publish a
checkpoint before exposing completed output; the session layer commits that
checkpoint. Closing an outer stream closes the active operation and provider
iterator. No background execution survives the stream's owner.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..._ids import new_event_id
from ...errors import CapabilityStateError, InvalidSendInputError
from ..capabilities.base import AppendContext as CapabilityAppend
from ..capabilities.base import (
    Capability,
    CapabilityOutput,
    ContextUsage,
    EmitEvent,
    ExecutionContext,
)
from ..capabilities.base import ReplaceContext as CapabilityReplace
from ..capabilities.base import SetState as CapabilitySetState
from ..events import (
    AgentInfo,
    ApprovalRequired,
    ClientToolRequired,
    InputTokenBreakdown,
    McpAuthRequired,
    McpInitialized,
    MessageUsage,
    ModelMessage,
    ModelMessageDelta,
    PublicToolCall,
    ThreadParent,
    ToolApproval,
    ToolCallRef,
    ToolReply,
    ToolResult,
    TurnInput,
    UserMessage,
)
from ..instructions import ROOT_AGENT_IDENTITY, SUB_AGENT_IDENTITY, InstructionBuilder
from ..llm.accumulate import StreamAccumulator
from ..llm.base import LLMRequest
from ..messages import (
    ApprovalRecord,
    ContextMessage,
    EnrichedToolCall,
    LLMAssistantMessage,
    LLMToolMessage,
    LLMUserMessage,
    Usage,
    text_of,
    to_wire_message,
    unknown_tool_info,
)
from ..tokens import estimate_tokens, estimate_tokens_for_json
from ..tools.context import ToolContext
from ..tools.execute import ExecutionResult, execute_tool_calls
from ..tools.registry import ToolRegistry, build_registry
from .context_utils import (
    INTERNAL_MESSAGE_GUIDANCE,
    closable_open_tool_call_ids,
    estimate_context_usage,
    is_llm_message,
    last_assistant_message,
    open_tool_call_ids,
    pending_approval_calls,
    pending_client_side_calls,
    scan_approvals,
)
from .internal_events import (
    AppendContext,
    CreateSubAgent,
    ReplaceContext,
    SetState,
    SubAgentCompletion,
    ThreadFinished,
)
from .metrics import ThreadMetrics
from .tool_call_repair import ToolCallRepair
from .user_input import process_user_message

__all__ = ["AgentDefinition", "AgentThread", "ThreadSnapshot"]

DEFAULT_ITERATION_LIMIT = 100
ThreadState = Literal["llm-call-required", "tool-response-required", "user-input-required"]


class AgentDefinition(BaseModel):
    """Resolved clients, tools, and model settings used by a conversation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: Any
    instructions: str | None = None
    initial_messages: list[LLMUserMessage] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    response_format: dict[str, Any] | None = None
    iteration_limit: int = DEFAULT_ITERATION_LIMIT
    tool_sets: list[Any] = Field(default_factory=list)
    name: str | None = None


class ThreadSnapshot(BaseModel):
    """Serializable conversation data consumed when constructing the next turn."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    thread_id: str
    context: list[ContextMessage] = Field(default_factory=list)
    usage: ContextUsage = Field(default_factory=ContextUsage)
    parent: ThreadParent | None = None
    agent_info: AgentInfo | None = None
    completion: SubAgentCompletion | None = None
    capability_state: dict[str, Any] = Field(default_factory=dict)


@dataclass
class _Journal:
    messages: list[ContextMessage]
    usage: ContextUsage
    completion: SubAgentCompletion | None
    values: dict[str, Any] = field(default_factory=dict)


@asynccontextmanager
async def _managed_stream(stream: AsyncIterator[Any]) -> AsyncIterator[AsyncIterator[Any]]:
    """Own an iterator, including adapters that implement only __anext__."""
    try:
        yield stream
    finally:
        close = getattr(stream, "aclose", None)
        if close is not None:
            await close()


class AgentThread:
    """A mutable conversation with one active send or execute consumer.

    Construction restores capability state and prepares instructions. send()
    validates and records input; execute() advances the conversation until an
    answer, external action, child task, cancellation, or error stops progress.
    The public context list is an inspection surface; use send() to add input.
    """

    def __init__(
        self,
        definition: AgentDefinition,
        *,
        thread_id: str = "main",
        title: str = "main",
        capabilities: Sequence[Capability] = (),
        context: Sequence[ContextMessage] | None = None,
        usage: ContextUsage | None = None,
        parent: ThreadParent | None = None,
        agent_info: AgentInfo | None = None,
        completion: SubAgentCompletion | None = None,
        capability_state: dict[str, Any] | None = None,
        session_id: str | None = None,
        turn_id: str | None = None,
        metadata: dict[str, str] | None = None,
        artifacts: Any = None,
        tracer: Any = None,
    ) -> None:
        from ...tracing import NOOP_TRACER

        self.definition, self.thread_id, self.title = definition, thread_id, title
        self.parent, self.agent_info = parent, agent_info
        self.session_id, self.turn_id = session_id, turn_id
        self.metadata = dict(metadata or {})
        self.artifacts = artifacts
        self.tracer = tracer if tracer is not None else NOOP_TRACER
        self.capabilities = [ToolCallRepair(), *capabilities]
        self._journal = _Journal(list(context or ()), usage or ContextUsage(), completion)
        self._counters = ThreadMetrics()
        self._tools: ToolRegistry | None = None
        self._in_use = False
        self._input_prepared = False

        claimed = [cap.state_key for cap in self.capabilities if cap.state_key is not None]
        if len(claimed) != len(set(claimed)):
            raise CapabilityStateError(f"Duplicate capability state key on thread {thread_id!r}")
        self._declared_keys = {key for key in claimed if key}
        for cap in self.capabilities:
            key = cap.state_key
            if key in self._declared_keys and key in (capability_state or {}):
                assert key is not None
                value = (capability_state or {})[key]
                cap.load_state(value)
                self._journal.values[key] = value

        prompt = InstructionBuilder.system_prompt(SUB_AGENT_IDENTITY if parent else ROOT_AGENT_IDENTITY)
        sections = prompt.begin_section("agent-capabilities")
        sections.add_section("internal-messages", INTERNAL_MESSAGE_GUIDANCE)
        for cap in self.capabilities:
            cap.build_instructions(sections)
        if parent is None and definition.instructions:
            prompt.add_section("user-instructions", definition.instructions, escape=True)
        self._prompt = prompt.build()

    @property
    def instructions(self) -> str:
        return self._prompt

    @property
    def context(self) -> list[ContextMessage]:
        return self._journal.messages

    @property
    def metrics(self) -> ThreadMetrics:
        return self._counters

    def snapshot(self) -> ThreadSnapshot:
        return ThreadSnapshot(
            thread_id=self.thread_id,
            parent=self.parent,
            agent_info=self.agent_info,
            context=list(self.context),
            usage=self._journal.usage,
            completion=self._journal.completion,
            capability_state=dict(self._journal.values),
        )

    def is_awaiting_user_input(self) -> bool:
        return bool(pending_approval_calls(self.context) or pending_client_side_calls(self.context))

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        if self._in_use:
            raise RuntimeError(f"Thread {self.thread_id!r} is already running; it permits one consumer")
        self._in_use = True
        try:
            yield
        finally:
            self._in_use = False

    def validate_input(self, inputs: Sequence[TurnInput]) -> None:
        """Reject invalid or incomplete responses without modifying the journal."""
        unanswered = open_tool_call_ids(self.context)
        blocks_user = unanswered - closable_open_tool_call_ids(self.context)
        decisions_due = {call.id for call in pending_approval_calls(self.context)}
        results_due = {call.id for call in pending_client_side_calls(self.context)}
        for position, item in enumerate(inputs):
            problem: str | None = None
            if isinstance(item, ToolReply):
                if item.tool_call_id in unanswered:
                    unanswered.remove(item.tool_call_id)
                    blocks_user.discard(item.tool_call_id)
                    results_due.discard(item.tool_call_id)
                else:
                    problem = f"no open tool call with id {item.tool_call_id!r}"
            elif isinstance(item, ToolApproval):
                if item.tool_call_id in decisions_due:
                    decisions_due.remove(item.tool_call_id)
                else:
                    problem = f"no approval is pending for tool_call_id {item.tool_call_id!r}"
            elif isinstance(item, UserMessage):
                if not item.content or isinstance(item.content, str) and not item.content.strip():
                    problem = "user message content is empty"
                elif blocks_user:
                    problem = "A user message cannot be sent while an approval or tool result is pending"
            else:
                problem = f"unsupported input type {type(item).__name__}"
            if problem is not None:
                raise InvalidSendInputError(f"input[{position}]: {problem}")
        missing = decisions_due | results_due
        if missing:
            raise InvalidSendInputError(
                "Every pending approval and client-side tool call must be answered in the same "
                f"batch. Still unanswered: {', '.join(sorted(missing))}"
            )

    async def send(self, inputs: Sequence[TurnInput]) -> AsyncIterator[AppendContext]:
        """Apply pre-send hooks and emit input checkpoints in provider-valid order."""
        if not inputs and not self.is_awaiting_user_input():
            return
        with self._exclusive():
            async with _managed_stream(self._hooks("pre_send")) as stream:
                async for event in stream:
                    if isinstance(event, AppendContext):
                        yield event
            self._input_prepared = True
            self.validate_input(inputs)

            responses: list[ContextMessage] = [
                ApprovalRecord(tool_call_id=item.tool_call_id, decision=item.decision, reason=item.reason)
                for item in inputs if isinstance(item, ToolApproval)
            ]
            responses.extend(
                LLMToolMessage(tool_call_id=item.tool_call_id, content=item.content)
                for item in inputs if isinstance(item, ToolReply)
            )
            new_messages: list[ContextMessage] = []
            attachments: list[Any] = []
            for item in inputs:
                if isinstance(item, UserMessage):
                    prepared = await process_user_message(item, artifacts=self.artifacts)
                    new_messages.extend(prepared.messages)
                    attachments.extend(prepared.events)
            if responses:
                yield self._record(responses)
            if new_messages:
                yield self._record(new_messages, public=attachments)

    async def execute(self, cancel: asyncio.Event | None = None) -> AsyncIterator[Any]:
        """Publish checkpoints and output until the next execution boundary.

        Cooperative cancellation is observed before model/tool work and after
        model streaming. Waiting-for-input events for an already committed
        assistant message still reach the consumer. Exceptions become terminal
        error events; task cancellation and generator closure remain observable.
        """
        with self._exclusive():
            try:
                async with _managed_stream(_Invocation(self, cancel).events()) as stream:
                    async for event in stream:
                        yield event
            except Exception as exc:
                message = str(exc).strip() or f"{type(exc).__name__} (no message)"
                yield self._finished(error=message)

    def deliver_tool_message(self, message: LLMToolMessage) -> AppendContext:
        """Record a child or host result using the normal checkpoint format."""
        return self._record([message])

    def _record(
        self, messages: Sequence[ContextMessage], *, public: Sequence[Any] = (),
        usage: ContextUsage | None = None, completion: SubAgentCompletion | None = None,
    ) -> AppendContext:
        appended = list(messages)
        self._journal.usage = (
            usage if usage is not None
            else self._journal.usage.merged_with(estimate_context_usage(appended))
        )
        self.context.extend(appended)
        if completion is not None:
            self._journal.completion = completion
        return AppendContext(
            thread_id=self.thread_id, messages=appended, events=list(public),
            usage=self._journal.usage, completion=completion,
        )

    def _view(self) -> ExecutionContext:
        return ExecutionContext(
            context=self.context, usage=self._journal.usage, thread_id=self.thread_id,
            session_id=self.session_id, turn_id=self.turn_id, metadata=self.metadata,
            artifacts=self.artifacts, agent_name=self.definition.name, is_sub_agent=self.parent is not None,
        )

    def _accept(self, output: CapabilityOutput, *, owner: Capability) -> list[Any]:
        """Apply a capability command and return its checkpoint-first publication."""
        if isinstance(output, EmitEvent):
            return [output.event]
        if isinstance(output, CapabilitySetState):
            if output.key not in self._declared_keys:
                raise CapabilityStateError(f"Undeclared capability state key {output.key!r}")
            if output.key != owner.state_key:
                raise CapabilityStateError(
                    f"Capability with state key {owner.state_key!r} cannot update {output.key!r}"
                )
            self._journal.values[output.key] = output.value
            return [SetState(thread_id=self.thread_id, key=output.key, value=output.value)]
        if isinstance(output, CapabilityAppend):
            append = self._record(output.messages, public=output.events, usage=output.usage)
            return [append, *output.events]
        if isinstance(output, CapabilityReplace):
            self._journal.messages = list(output.messages)
            self._journal.usage = output.usage
            self.metrics.total_compactions += 1
            if output.model_usage is not None:
                self.metrics.add_usage(output.model_usage)
            checkpoint = ReplaceContext(
                thread_id=self.thread_id, messages=list(output.messages), usage=output.usage,
                event=output.event, model_usage=output.model_usage,
            )
            return [checkpoint, output.event] if output.event is not None else [checkpoint]
        raise TypeError(f"Unsupported capability output: {type(output).__name__}")

    async def _hooks(self, hook: str) -> AsyncIterator[Any]:
        for cap in self.capabilities:
            async with _managed_stream(getattr(cap, hook)(self._view())) as commands:
                async for command in commands:
                    for event in self._accept(command, owner=cap):
                        yield event

    async def _registry(self) -> ToolRegistry:
        if self._tools is None:
            self._tools = await build_registry(
                user_sets=list(self.definition.tool_sets),
                builtin_sets=[tools for cap in self.capabilities for tools in cap.tool_sets()],
            )
        return self._tools

    def _request(self, registry: ToolRegistry) -> LLMRequest:
        history = [*self.definition.initial_messages, *self.context]
        messages = [to_wire_message(message) for message in history if is_llm_message(message)]
        if self.instructions:
            messages.insert(0, {"role": "system", "content": self.instructions})
        for cap in self.capabilities:
            prepared = cap.prepare_request(messages)
            if prepared is not None:
                messages = prepared
        return LLMRequest(
            messages=messages, params=dict(self.definition.params),
            response_format=self.definition.response_format, tools=registry.schemas or None,
        )

    def _usage_event(self, usage: Usage, request: LLMRequest, registry: ToolRegistry) -> MessageUsage:
        personal = 0 if self.parent else estimate_tokens(self.definition.instructions)
        harness = max(0, estimate_tokens(self.instructions) - personal)
        user_tools = 0
        for schema in request.tools or ():
            resolved = registry.resolve(schema.get("function", {}).get("name", ""))
            size = estimate_tokens_for_json(schema)
            if resolved is not None and registry.is_builtin(resolved.tool_set.name):
                harness += size
            else:
                user_tools += size
        return MessageUsage(
            **usage.model_dump(),
            input_tokens_breakdown=InputTokenBreakdown(
                harness=harness, instructions=personal, tool_definitions=user_tools,
                messages=max(0, usage.input_tokens - harness - personal - user_tools),
            ),
        )

    async def _resolved_message(self, raw: Any, registry: ToolRegistry) -> LLMAssistantMessage:
        message = raw.model_dump(exclude_none=True, exclude={"tool_calls"})
        calls: list[EnrichedToolCall] = []
        for call in raw.tool_calls or ():
            target = registry.resolve(call.function.name)
            info = unknown_tool_info(call.function.name)
            if target is not None:
                try:
                    payload = json.loads(call.function.arguments or "{}")
                except (TypeError, ValueError):
                    payload = {}
                info = await target.tool_set.tool_info(
                    target.original_name, payload if isinstance(payload, dict) else {},
                    resolve_underlying=True,
                )
            details = call.model_dump()
            details["tool_info"] = info
            calls.append(EnrichedToolCall.model_validate(details))
        return LLMAssistantMessage(**message, tool_calls=calls or None)

    def _finished(
        self, *, output: ModelMessage | None = None, error: str | None = None,
        completion: SubAgentCompletion | None = None,
    ) -> ThreadFinished:
        details: dict[str, Any] = {"status": "error" if error is not None else "done",
                                   "output": output, "error": error}
        if completion is not None:
            details.update(completion.model_dump())
        elif self.parent is not None:
            details["send_to_parent"] = LLMToolMessage(
                tool_call_id=self.parent.tool_call_id,
                content=error if error is not None else text_of(output.content if output else None),
            )
        return ThreadFinished(thread_id=self.thread_id, title=self.title, parent=self.parent, **details)

    def __repr__(self) -> str:
        return f"AgentThread({self.thread_id!r}, messages={len(self.context)})"


class _Invocation:
    """Own transient scheduling; queued operations never outlive their consumer."""

    def __init__(self, thread: AgentThread, cancel: asyncio.Event | None) -> None:
        self.thread = thread
        self.cancel = cancel
        self.work: deque[Callable[[], AsyncIterator[Any]]] = deque([self.prepare])
        self.previous: ThreadState | None = None
        self.message_id = ""
        self.tools: ToolRegistry | None = None

    @property
    def cancelled(self) -> bool:
        return self.cancel is not None and self.cancel.is_set()

    async def events(self) -> AsyncIterator[Any]:
        completion = self.thread._journal.completion
        if completion is not None:
            yield self.thread._finished(completion=completion)
            return
        while self.work:
            operation = self.work.popleft()
            async with _managed_stream(operation()) as stream:
                async for event in stream:
                    yield event

    async def prepare(self) -> AsyncIterator[Any]:
        if not self.thread._input_prepared:
            async with _managed_stream(self.thread._hooks("pre_send")) as stream:
                async for event in stream:
                    yield event
        self.thread._input_prepared = False
        self.tools = await self.thread._registry()
        if self.tools.initialized:
            yield McpInitialized(thread_id=self.thread.thread_id, mcp_servers=list(self.tools.initialized))
        if self.tools.auth_required:
            yield McpAuthRequired(thread_id=None, mcp_servers=list(self.tools.auth_required))
        else:
            self.schedule()

    def schedule(self) -> None:
        """Select work from unanswered obligations, retaining the state contract."""
        open_calls = open_tool_call_ids(self.thread.context)
        waiting = self.thread.is_awaiting_user_input()
        choice: ThreadState = (
            "user-input-required" if waiting else "tool-response-required" if open_calls else "llm-call-required"
        )
        if self.previous is not None:
            model_boundary = "llm-call-required"
            valid = self.previous != choice and (
                self.previous != "tool-response-required" or choice == model_boundary
            )
            if not valid:
                raise RuntimeError(f"Illegal thread transition: {self.previous} -> {choice}")
        self.previous = choice
        routes = {"llm-call-required": self.generate, "tool-response-required": self.resolve,
                  "user-input-required": self.wait_for_input}
        self.work.append(routes[choice])

    async def wait_for_input(self) -> AsyncIterator[Any]:
        thread = self.thread
        for event_type, pending in (
            (ClientToolRequired, pending_client_side_calls(thread.context)),
            (ApprovalRequired, pending_approval_calls(thread.context)),
        ):
            if pending:
                yield event_type(
                    thread_id=thread.thread_id,
                    tool_calls=[ToolCallRef(id=call.id, source_event_id=self.message_id) for call in pending],
                )

    async def generate(self) -> AsyncIterator[Any]:
        thread = self.thread
        if self.cancelled:
            return
        if thread.metrics.iterations >= thread.definition.iteration_limit:
            yield thread._finished(error=f"Reached the iteration limit of {thread.definition.iteration_limit}")
            return
        thread.metrics.iterations += 1
        async with _managed_stream(thread._hooks("pre_llm")) as stream:
            async for event in stream:
                yield event
        assert self.tools is not None
        request = thread._request(self.tools)
        self.message_id = new_event_id()
        yield ModelMessage(id=self.message_id, thread_id=thread.thread_id)

        accumulator = StreamAccumulator(source=getattr(thread.definition.llm, "model", None))
        usage: MessageUsage | None = None
        with thread.tracer.span("agento.llm", model=getattr(thread.definition.llm, "model", "")) as span:
            async with _managed_stream(thread.definition.llm.stream(request)) as chunks:
                async for chunk in chunks:
                    accumulator.add(chunk)
                    if chunk.usage is not None:
                        usage = thread._usage_event(chunk.usage, request, self.tools)
                    yield ModelMessageDelta(
                        id=self.message_id, thread_id=thread.thread_id, content=chunk.content,
                        reasoning_content=chunk.reasoning_content, finish_reason=chunk.finish_reason,
                        tool_calls=[call.model_dump(exclude_none=True) for call in chunk.tool_calls]
                        if chunk.tool_calls else None, usage=usage,
                    )
            response = accumulator.result()
            span.set_output(text_of(response.message.content)[:2000])
        if self.cancelled:
            return

        assistant = await thread._resolved_message(response.message, self.tools)
        public = ModelMessage(
            id=self.message_id, thread_id=thread.thread_id, content=assistant.content,
            finish_reason=response.finish_reason,
            usage=usage or thread._usage_event(response.usage, request, self.tools),
            tool_calls=[PublicToolCall(id=call.id, type=call.type, function=call.function,
                                       tool_info=call.tool_info.to_public())
                        for call in assistant.tool_calls or ()] or None,
        )
        ended = not assistant.tool_calls or response.finish_reason == "length"
        error = "The model response exceeded its output token limit" if response.finish_reason == "length" else None
        terminal = thread._finished(output=public, error=error) if ended else None
        completion = None
        if terminal is not None and thread.parent is not None:
            assert terminal.send_to_parent is not None
            completion = SubAgentCompletion(
                output=public, status=terminal.status, error=terminal.error,
                send_to_parent=terminal.send_to_parent,
            )
        thread.metrics.add_usage(response.usage)
        yield thread._record(
            [assistant], public=[public], completion=completion,
            usage=ContextUsage(prompt_tokens=response.usage.input_tokens,
                               completion_tokens=response.usage.output_tokens),
        )
        yield public
        if terminal is None:
            self.schedule()
        else:
            yield terminal

    async def resolve(self) -> AsyncIterator[Any]:
        if self.cancelled:
            return
        thread = self.thread
        assert self.tools is not None
        assistant = last_assistant_message(thread.context)
        assert assistant is not None
        unanswered = open_tool_call_ids(thread.context)
        result = await execute_tool_calls(
            [call for call in assistant.tool_calls or () if call.id in unanswered], self.tools,
            approvals={key: value for key, value in scan_approvals(thread.context).items() if key in unanswered},
            base_context=ToolContext(
                session_id=thread.session_id, turn_id=thread.turn_id, thread_id=thread.thread_id,
                agent_name=thread.definition.name, metadata=thread.metadata, artifacts=thread.artifacts,
            ),
        )
        thread.metrics.total_tool_calls += len(result.results)
        thread.metrics.total_sub_agents += len(result.sub_agents)
        publications = await self.result_publications(result)
        for event in publications:
            yield event
        if result.auth_required:
            yield McpAuthRequired(thread_id=None, mcp_servers=list(result.auth_required))
            return
        async with _managed_stream(thread._hooks("post_tool_call")) as stream:
            async for event in stream:
                yield event
        if result.sub_agents:
            for child in result.sub_agents:
                yield CreateSubAgent(thread_id=thread.thread_id, tool_call_id=child.tool_call_id,
                                     agent_info=child.agent_info)
        else:
            self.schedule()

    async def result_publications(self, batch: ExecutionResult) -> list[Any]:
        """Stage rewrites, then publish one checkpoint ahead of result events."""
        thread = self.thread
        expected = tuple(result.tool_call.id for result in batch.results)
        extra = list(batch.events)
        if batch.initialized:
            extra.append(McpInitialized(thread_id=thread.thread_id, mcp_servers=list(batch.initialized)))
        for cap in thread.capabilities:
            commands = await cap.process_tool_results(batch.results, thread._view())
            for command in commands:
                extra.extend(thread._accept(command, owner=cap))
        if expected != tuple(result.tool_call.id for result in batch.results):
            raise RuntimeError("A capability changed tool result identities or order; rewrite content only")
        public = [ToolResult(
            thread_id=thread.thread_id, tool_call_id=result.message.tool_call_id,
            content=result.message.content, is_error=result.failed,
            artifact_id=result.artifact_id, created_at=result.completed_at,
        ) for result in batch.results]
        if not batch.results:
            return extra
        checkpoint = thread._record(
            [result.message for result in batch.results],
            public=[event for event in extra if hasattr(event, "id")] + public,
        )
        return [checkpoint, *extra, *public]
