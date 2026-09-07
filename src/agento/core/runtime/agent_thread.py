"""The agent loop.

One :class:`AgentThread` is one agent's conversation: its messages, its tools,
its capabilities. The root agent is a thread; every sub-agent is another thread.
:meth:`AgentThread.execute` is the loop itself.

The loop is a three-state machine, and the state is **derived from the messages**
rather than stored beside them:

``llm-call-required``
    No tool call is outstanding. Call the model.
``tool-response-required``
    The model asked for tools and they can be run. Run them.
``user-input-required``
    An outstanding call needs a human or the host application. Stop and say so.

Deriving state from the message list is what makes a turn resumable. To continue
a conversation the runtime loads the messages and asks the same question again;
there is no status field that could disagree with the history, and no migration
to write when the loop gains a new behaviour.

**Persist before yield.** Every state change is yielded as an internal event
*before* the thread applies it to its own memory. A consumer that persists each
event as it arrives can therefore be ahead of the in-memory thread, never behind
— so a crash mid-turn loses nothing that was already announced.

Nothing raises out of :meth:`execute`. A provider outage, a bad tool, a bug in a
capability — all of it becomes a
:class:`~agento.core.runtime.internal_events.ThreadFinished` with ``status
= "error"``, because a turn that dies without a terminal event leaves the session
permanently mid-flight.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..._ids import new_event_id
from ...errors import CapabilityStateError, InvalidSendInputError
from ..capabilities.base import (
    AppendContext as CapabilityAppend,
)
from ..capabilities.base import (
    Capability,
    CapabilityOutput,
    ContextUsage,
    EmitEvent,
    ExecutionContext,
)
from ..capabilities.base import (
    ReplaceContext as CapabilityReplace,
)
from ..capabilities.base import (
    SetState as CapabilitySetState,
)
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
    InternalToolInfo,
    LLMAssistantMessage,
    LLMToolMessage,
    LLMUserMessage,
    Usage,
    text_of,
    to_wire_message,
    unknown_tool_info,
)
from ..tokens import estimate_tokens, estimate_tokens_for_json
from ..tools.base import ToolSet
from ..tools.context import ToolContext
from ..tools.execute import execute_tool_calls
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

# Which transitions are legal. Enforced so that a bug in a capability shows up as
# a loud, specific error instead of an agent that quietly loops or stalls.
_VALID_TRANSITIONS: dict[ThreadState, set[ThreadState]] = {
    "llm-call-required": {"tool-response-required", "user-input-required"},
    "tool-response-required": {"llm-call-required"},
    # From user-input-required: an approval decision moves to executing the tool;
    # a client-side result resolves the call outright and goes back to the model.
    "user-input-required": {"tool-response-required", "llm-call-required"},
}


class AgentDefinition(BaseModel):
    """Everything a thread needs to run, with names already resolved to objects.

    This is the *runtime* form of an agent. The declarative, serializable form is
    :class:`agento.session.agent.Agent`; a resolver turns one into the other by
    looking up the model client, connecting MCP servers and loading skills.

    Attributes:
        llm: The bound model client.
        instructions: The agent's system prompt. Sub-agents get ``None`` — their
            task arrives as a user message instead.
        initial_messages: Messages injected at the start of every conversation,
            before any user input. Sub-agents receive their task this way.
        params: Provider parameters — ``temperature``, ``max_tokens``,
            ``reasoning_effort``.
        response_format: Structured-output spec, passed through untouched.
        iteration_limit: Maximum model calls in one turn. The backstop against a
            runaway loop.
        tool_sets: The agent's own tools. Capability tools are added separately.
        name: Optional agent name, for tracing and tool context.
    """

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
    """A thread's complete durable state.

    Persisted at the end of a turn and used to rebuild the thread at the start of
    the next. Everything needed to resume is here — which is the point.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    thread_id: str
    context: list[ContextMessage] = Field(default_factory=list)
    usage: ContextUsage = Field(default_factory=ContextUsage)
    parent: ThreadParent | None = None
    agent_info: AgentInfo | None = None
    completion: SubAgentCompletion | None = None
    capability_state: dict[str, Any] = Field(default_factory=dict)


class AgentThread:
    """One agent's conversation and the loop that advances it.

    Args:
        definition: What to run.
        thread_id: ``"main"`` for the root agent.
        title: Human-readable label.
        capabilities: Behaviours hooked onto the loop. A
            :class:`~agento.core.runtime.tool_call_repair.ToolCallRepair` is
            always installed first.
        context: Messages from a previous turn.
        usage: Context size from a previous turn.
        parent: Set on a sub-agent thread.
        agent_info: The delegation request, on a sub-agent thread.
        completion: A sub-agent's already-known outcome. When present the thread
            replays it instead of re-running — so resuming a turn does not
            re-execute delegated work that already happened.
        capability_state: Durable state to restore.
        session_id: For tool context and tracing.
        turn_id: For tool context and tracing.
        metadata: The session's metadata, exposed to tools.
        artifacts: The artifact store, if configured.
        tracer: Where spans go.
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

        self.definition = definition
        self.thread_id = thread_id
        self.title = title
        self.parent = parent
        self.agent_info = agent_info
        self.session_id = session_id
        self.turn_id = turn_id
        self.metadata = dict(metadata or {})
        self.artifacts = artifacts
        self.tracer = tracer or NOOP_TRACER

        self._context: list[ContextMessage] = list(context or [])
        self._usage = usage or ContextUsage()
        self._precomputed_completion = completion
        self._metrics = ThreadMetrics()
        self._registry: ToolRegistry | None = None
        self._busy = False
        self._pre_send_ran = False
        self._current_state: ThreadState | None = None
        self._pending_events: list[Any] = []

        # ToolCallRepair is always first: the conversation must be structurally
        # valid before any other capability looks at it.
        self.capabilities: list[Capability] = [ToolCallRepair(), *capabilities]
        self._state_keys = {
            capability.state_key for capability in self.capabilities if capability.state_key
        }
        self._assert_unique_state_keys()
        self._capability_state = self._restore_state(capability_state or {})

        self._instructions = self._build_instructions()

    # ------------------------------------------------------------------ #
    # Construction helpers                                               #
    # ------------------------------------------------------------------ #

    def _assert_unique_state_keys(self) -> None:
        seen: set[str] = set()
        for capability in self.capabilities:
            key = capability.state_key
            if key is None:
                continue
            if key in seen:
                raise CapabilityStateError(
                    f"Two capabilities on thread {self.thread_id!r} both declare state key {key!r}"
                )
            seen.add(key)

    def _restore_state(self, stored: dict[str, Any]) -> dict[str, Any]:
        """Hand each capability its own state back, ignoring orphaned keys.

        A key whose capability is no longer attached is dropped rather than kept:
        turning a capability off should not leave its state to be silently
        resurrected later.
        """
        claimed = {key: value for key, value in stored.items() if key in self._state_keys}
        for capability in self.capabilities:
            key = capability.state_key
            if key is not None and key in claimed:
                capability.load_state(claimed[key])
        return claimed

    def _build_instructions(self) -> str:
        """Assemble the system prompt once, at construction.

        A sub-agent gets :data:`SUB_AGENT_IDENTITY` and **no** user instructions:
        its task arrives as a user message, and re-applying the parent's persona
        on top would blur the delegation.
        """
        identity = SUB_AGENT_IDENTITY if self.parent else ROOT_AGENT_IDENTITY
        builder = InstructionBuilder.system_prompt(identity)

        capabilities = builder.begin_section("agent-capabilities")
        capabilities.add_section("internal-messages", INTERNAL_MESSAGE_GUIDANCE)
        for capability in self.capabilities:
            capability.build_instructions(capabilities)

        if not self.parent and self.definition.instructions:
            builder.add_section("user-instructions", self.definition.instructions, escape=True)

        return builder.build()

    @property
    def instructions(self) -> str:
        """The assembled system prompt. Useful for debugging and tests."""
        return self._instructions

    @property
    def context(self) -> list[ContextMessage]:
        """The thread's messages. Read-only in practice — mutate through the loop."""
        return self._context

    @property
    def metrics(self) -> ThreadMetrics:
        """Running totals for this thread."""
        return self._metrics

    def snapshot(self) -> ThreadSnapshot:
        """Capture everything needed to rebuild this thread next turn."""
        return ThreadSnapshot(
            thread_id=self.thread_id,
            context=list(self._context),
            usage=self._usage,
            parent=self.parent,
            agent_info=self.agent_info,
            completion=self._precomputed_completion,
            capability_state=dict(self._capability_state),
        )

    # ------------------------------------------------------------------ #
    # Input                                                              #
    # ------------------------------------------------------------------ #

    def is_awaiting_user_input(self) -> bool:
        """Whether the thread is paused on a human or the host application."""
        return bool(pending_approval_calls(self._context) or pending_client_side_calls(self._context))

    def validate_input(self, inputs: Sequence[TurnInput]) -> None:
        """Check a batch against the thread's state without changing anything.

        Called before a turn is persisted, so an invalid batch fails cleanly and
        leaves no half-created turn behind.

        Raises:
            InvalidSendInputError: The batch is not valid here.
        """
        open_ids = open_tool_call_ids(self._context)
        # A dangling call that ToolCallRepair will close does not block a new
        # user message — otherwise an interrupted turn would wedge the session.
        blocking = open_ids - closable_open_tool_call_ids(self._context)
        pending_approvals = {call.id for call in pending_approval_calls(self._context)}
        pending_client = {call.id for call in pending_client_side_calls(self._context)}

        for index, item in enumerate(inputs):
            if isinstance(item, UserMessage):
                if _is_empty_content(item.content):
                    raise InvalidSendInputError(f"input[{index}]: user message content is empty")
                if blocking:
                    raise InvalidSendInputError(
                        "A user message cannot be sent while the agent is waiting for an approval "
                        "or a client-side tool result. Answer those first."
                    )
            elif isinstance(item, ToolApproval):
                if item.tool_call_id not in pending_approvals:
                    raise InvalidSendInputError(
                        f"input[{index}]: no approval is pending for tool_call_id "
                        f"{item.tool_call_id!r}"
                    )
                pending_approvals.discard(item.tool_call_id)
            elif isinstance(item, ToolReply):
                if item.tool_call_id not in open_ids:
                    raise InvalidSendInputError(
                        f"input[{index}]: no open tool call with id {item.tool_call_id!r}"
                    )
                open_ids.discard(item.tool_call_id)
                blocking.discard(item.tool_call_id)
                pending_client.discard(item.tool_call_id)
            else:  # pragma: no cover - guarded by the union
                raise InvalidSendInputError(f"input[{index}]: unsupported input type {type(item).__name__}")

        if pending_approvals or pending_client:
            missing = sorted(pending_approvals | pending_client)
            raise InvalidSendInputError(
                "Every pending approval and client-side tool call must be answered in the same "
                f"batch. Still unanswered: {', '.join(missing)}"
            )

    async def send(self, inputs: Sequence[TurnInput]) -> AsyncIterator[AppendContext]:
        """Append a batch of input to the thread.

        Runs ``pre_send`` capabilities first (so a broken conversation is
        repaired before new input lands), validates the batch, then appends.

        Yields:
            :class:`~agento.core.runtime.internal_events.AppendContext` events —
            the caller persists them, and the thread applies them as they pass.
        """
        if not inputs and not self.is_awaiting_user_input():
            return
        self._assert_not_busy()

        self._busy = True
        try:
            async for event in self._run_hooks("pre_send"):
                if isinstance(event, AppendContext):
                    yield event
            self._pre_send_ran = True

            self.validate_input(inputs)

            approvals: list[ApprovalRecord] = []
            tool_messages: list[LLMToolMessage] = []
            user_messages: list[ContextMessage] = []
            events: list[Any] = []

            for item in inputs:
                if isinstance(item, ToolApproval):
                    approvals.append(
                        ApprovalRecord(
                            tool_call_id=item.tool_call_id,
                            decision=item.decision,
                            reason=item.reason,
                        )
                    )
                elif isinstance(item, ToolReply):
                    tool_messages.append(
                        LLMToolMessage(tool_call_id=item.tool_call_id, content=item.content)
                    )
                elif isinstance(item, UserMessage):
                    processed = await process_user_message(item, artifacts=self.artifacts)
                    user_messages.extend(processed.messages)
                    events.extend(processed.events)

            # Approvals and client-side results first: they close out the
            # previous exchange before any new user message opens a new one.
            if approvals or tool_messages:
                yield self._append([*approvals, *tool_messages])
            if user_messages:
                yield self._append(user_messages, events=events)
        finally:
            self._busy = False

    # ------------------------------------------------------------------ #
    # Execution                                                          #
    # ------------------------------------------------------------------ #

    async def execute(self, cancel: asyncio.Event | None = None) -> AsyncIterator[Any]:
        """Run the loop until it stops.

        Args:
            cancel: Set this to stop after the current step. Checked before every
                model call and every tool execution — never in the middle of one,
                so a cancelled turn is always left in a valid state.

        Yields:
            Public events for the caller, interleaved with internal events for
            the persistence layer.
        """
        self._assert_not_busy()
        self._busy = True
        self._current_state = None

        try:
            if self._precomputed_completion is not None:
                # This thread already finished in an earlier turn. Replay the
                # outcome rather than re-running delegated work.
                yield self._replay_completion(self._precomputed_completion)
                return

            if not self._pre_send_ran:
                async for event in self._run_hooks("pre_send"):
                    yield event
            self._pre_send_ran = False

            registry = await self._ensure_registry()

            if registry.initialized:
                yield McpInitialized(thread_id=self.thread_id, mcp_servers=list(registry.initialized))

            if registry.auth_required:
                yield McpAuthRequired(thread_id=None, mcp_servers=list(registry.auth_required))
                return

            model_message_id = ""

            while True:
                for event in self._drain_pending_events():
                    yield event

                state = self._derive_state()

                if state == "llm-call-required":
                    if _is_cancelled(cancel):
                        return
                    if self._metrics.iterations >= self.definition.iteration_limit:
                        yield self._error_finished(
                            f"Reached the iteration limit of {self.definition.iteration_limit} "
                            "without finishing. Ask again, or raise iteration_limit."
                        )
                        return
                    self._metrics.iterations += 1

                    should_exit = False
                    async for item in self._step_model_call(registry, cancel):
                        if isinstance(item, _StepOutcome):
                            should_exit = item.exit
                            model_message_id = item.model_message_id or model_message_id
                        else:
                            yield item
                    if should_exit:
                        return

                elif state == "tool-response-required":
                    if _is_cancelled(cancel):
                        return
                    should_exit = False
                    async for item in self._step_tool_calls(registry):
                        if isinstance(item, _StepOutcome):
                            should_exit = item.exit
                        else:
                            yield item
                    if should_exit:
                        return

                else:  # user-input-required
                    # Deliberately not cancellable: the assistant message is
                    # already committed, so the caller must be told what it is
                    # waiting for or the turn ends with an unexplained pause.
                    for event in self._step_user_input(model_message_id):
                        yield event
                    return

        except Exception as exc:  # noqa: BLE001 - a turn must always terminate cleanly
            yield self._error_finished(_describe(exc))
        finally:
            self._busy = False

    # -- steps ---------------------------------------------------------- #

    async def _step_model_call(
        self, registry: ToolRegistry, cancel: asyncio.Event | None
    ) -> AsyncIterator[Any]:
        """One model call: hooks, stream, assemble, decide what happens next."""
        async for event in self._run_hooks("pre_llm"):
            yield event

        request = self._build_request(registry)
        message_id = new_event_id()

        # Placeholder that opens the delta stream. Deltas share this id, and the
        # complete message that follows reuses it, so a client can create one
        # bubble and fill it in.
        yield ModelMessage(id=message_id, thread_id=self.thread_id)

        accumulator = StreamAccumulator(source=getattr(self.definition.llm, "model", None))
        usage: MessageUsage | None = None

        with self.tracer.span("agento.llm", model=getattr(self.definition.llm, "model", "")) as span:
            async for chunk in self.definition.llm.stream(request):
                accumulator.add(chunk)
                if chunk.usage is not None:
                    usage = self._attribute_usage(request, chunk.usage, registry)
                yield ModelMessageDelta(
                    id=message_id,
                    thread_id=self.thread_id,
                    content=chunk.content,
                    reasoning_content=chunk.reasoning_content,
                    tool_calls=[call.model_dump(exclude_none=True) for call in chunk.tool_calls]
                    if chunk.tool_calls
                    else None,
                    finish_reason=chunk.finish_reason,
                    usage=usage,
                )
            span.set_output(text_of(accumulator.result().message.content)[:2000])

        if _is_cancelled(cancel):
            # A partially-streamed response is not a message. Emitting it would
            # persist something the model never actually finished saying.
            yield _StepOutcome(exit=True, model_message_id=message_id)
            return

        response = accumulator.result()
        if usage is None:
            usage = self._attribute_usage(request, response.usage, registry)

        assistant = await self._enrich(response.message, registry)
        public = self._to_public_message(assistant, message_id, response.finish_reason, usage)

        completion = self._completion_for(assistant, public, response.finish_reason)

        self._metrics.add_usage(response.usage)
        yield self._append(
            [assistant],
            events=[public],
            usage=ContextUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
            ),
            completion=completion,
        )
        yield public

        if response.finish_reason == "length":
            yield self._error_finished(
                "The model's response was cut off by the output token limit. "
                "Raise max_tokens, or ask for a shorter answer.",
                output=public,
            )
            yield _StepOutcome(exit=True, model_message_id=message_id)
            return

        if not assistant.tool_calls:
            yield ThreadFinished(
                thread_id=self.thread_id,
                status="done",
                title=self.title,
                output=public,
                parent=self.parent,
                send_to_parent=completion.send_to_parent if completion else None,
            )
            yield _StepOutcome(exit=True, model_message_id=message_id)
            return

        yield _StepOutcome(exit=False, model_message_id=message_id)

    async def _step_tool_calls(self, registry: ToolRegistry) -> AsyncIterator[Any]:
        """Execute the open tool calls and fold the results back in."""
        assistant = last_assistant_message(self._context)
        if assistant is None:  # pragma: no cover - unreachable via the state machine
            yield _StepOutcome(exit=True)
            return

        open_ids = open_tool_call_ids(self._context)
        calls = [call for call in (assistant.tool_calls or []) if call.id in open_ids]
        approvals = scan_approvals(self._context)

        outcome = await execute_tool_calls(
            calls,
            registry,
            approvals={call_id: decision for call_id, decision in approvals.items() if call_id in open_ids},
            base_context=ToolContext(
                session_id=self.session_id,
                turn_id=self.turn_id,
                thread_id=self.thread_id,
                agent_name=self.definition.name,
                metadata=self.metadata,
                artifacts=self.artifacts,
            ),
        )

        self._metrics.total_tool_calls += len(outcome.results)
        self._metrics.total_sub_agents += len(outcome.sub_agents)

        pending_events: list[Any] = list(outcome.events)
        if outcome.initialized:
            pending_events.append(McpInitialized(thread_id=self.thread_id, mcp_servers=list(outcome.initialized)))

        # Capabilities may rewrite result content here — this is where a huge
        # payload becomes a preview plus an artifact id.
        execution = self._execution_context()
        expected = [result.tool_call.id for result in outcome.results]
        for capability in self.capabilities:
            for output in await capability.process_tool_results(outcome.results, execution):
                async for event in self._apply(output):
                    pending_events.append(event)
        if [result.tool_call.id for result in outcome.results] != expected:
            raise RuntimeError(
                "A capability added or removed tool results. Results may be rewritten in place, "
                "but every tool call must keep exactly one result."
            )

        public_results = [
            ToolResult(
                thread_id=self.thread_id,
                tool_call_id=result.message.tool_call_id,
                content=result.message.content,
                is_error=result.failed,
                artifact_id=result.artifact_id,
                created_at=result.completed_at,
            ) for result in outcome.results
        ]

        if outcome.results:
            yield self._append(
                [result.message for result in outcome.results],
                events=[event for event in pending_events if hasattr(event, "id")] + public_results,
            )
        for event in pending_events:
            yield event
        for public_result in public_results:
            yield public_result

        if outcome.auth_required:
            yield McpAuthRequired(thread_id=None, mcp_servers=list(outcome.auth_required))
            yield _StepOutcome(exit=True)
            return

        async for event in self._run_hooks("post_tool_call"):
            yield event

        if outcome.sub_agents:
            for request in outcome.sub_agents:
                yield CreateSubAgent(
                    thread_id=self.thread_id,
                    tool_call_id=request.tool_call_id,
                    agent_info=request.agent_info,
                )
            # The orchestrator takes over: this thread waits until its children
            # report back, which they do by closing the open tool call.
            yield _StepOutcome(exit=True)
            return

        yield _StepOutcome(exit=False)

    def _step_user_input(self, model_message_id: str) -> Iterable[Any]:
        """Announce what the run is waiting for, then stop."""
        client_side = pending_client_side_calls(self._context)
        if client_side:
            yield ClientToolRequired(
                thread_id=self.thread_id,
                tool_calls=[
                    ToolCallRef(id=call.id, source_event_id=model_message_id) for call in client_side
                ],
            )

        approvals = pending_approval_calls(self._context)
        if approvals:
            yield ApprovalRequired(
                thread_id=self.thread_id,
                tool_calls=[
                    ToolCallRef(id=call.id, source_event_id=model_message_id) for call in approvals
                ],
            )

    # ------------------------------------------------------------------ #
    # Request building                                                   #
    # ------------------------------------------------------------------ #

    def _build_request(self, registry: ToolRegistry) -> LLMRequest:
        """Assemble the provider request for this iteration."""
        messages: list[dict[str, Any]] = []
        if self._instructions:
            messages.append({"role": "system", "content": self._instructions})
        for initial_message in self.definition.initial_messages:
            messages.append(to_wire_message(initial_message))
        for message in self._context:
            if is_llm_message(message):
                messages.append(to_wire_message(message))

        # Ephemeral, per-request edits. Not stored, so they cannot accumulate.
        for capability in self.capabilities:
            replaced = capability.prepare_request(messages)
            if replaced is not None:
                messages = replaced

        return LLMRequest(
            messages=messages,
            tools=registry.schemas or None,
            response_format=self.definition.response_format,
            params=dict(self.definition.params),
        )

    def _attribute_usage(
        self, request: LLMRequest, usage: Usage, registry: ToolRegistry
    ) -> MessageUsage:
        """Split reported input tokens across the things that produced them.

        Estimates, and honest about it — the provider gives one number and agento
        apportions it. Still the fastest way to answer "why is this agent
        expensive?", which is usually "forty tool schemas you never use".
        """
        instruction_tokens = (
            estimate_tokens(self.definition.instructions) if not self.parent else 0
        )
        harness_tokens = max(0, estimate_tokens(self._instructions) - instruction_tokens)

        agent_tool_tokens = 0
        builtin_tool_tokens = 0
        for schema in request.tools or []:
            tokens = estimate_tokens_for_json(schema)
            name = schema.get("function", {}).get("name", "")
            mapped = registry.resolve(name)
            if mapped is not None and registry.is_builtin(mapped.tool_set.name):
                builtin_tool_tokens += tokens
            else:
                agent_tool_tokens += tokens

        accounted = harness_tokens + builtin_tool_tokens + instruction_tokens + agent_tool_tokens
        return MessageUsage(
            **usage.model_dump(),
            input_tokens_breakdown=InputTokenBreakdown(
                harness=harness_tokens + builtin_tool_tokens,
                instructions=instruction_tokens,
                tool_definitions=agent_tool_tokens,
                messages=max(0, usage.input_tokens - accounted),
            ),
        )

    async def _ensure_registry(self) -> ToolRegistry:
        """Build the tool registry once per turn."""
        if self._registry is None:
            builtin_sets: list[ToolSet] = []
            for capability in self.capabilities:
                builtin_sets.extend(capability.tool_sets())
            self._registry = await build_registry(
                builtin_sets=builtin_sets,
                user_sets=list(self.definition.tool_sets),
            )
        return self._registry

    async def _enrich(self, message: Any, registry: ToolRegistry) -> LLMAssistantMessage:
        """Attach resolved tool info to each of the model's tool calls.

        This is where the loop learns that a call needs approval, must be run by
        the host, or spawns a sub-agent — and the reason it happens once, here,
        rather than being re-derived later, is that the answer must stay stable
        for the life of the conversation even if the agent's configuration
        changes between turns.
        """
        raw_calls = message.tool_calls or []
        if not raw_calls:
            data = message.model_dump(exclude_none=True)
            data.pop("tool_calls", None)
            return LLMAssistantMessage(**data)

        enriched: list[EnrichedToolCall] = []
        for call in raw_calls:
            mapped = registry.resolve(call.function.name)
            if mapped is None:
                info: InternalToolInfo = unknown_tool_info(call.function.name)
            else:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                info = await mapped.tool_set.tool_info(
                    mapped.original_name,
                    arguments if isinstance(arguments, dict) else {},
                    resolve_underlying=True,
                )
            enriched.append(EnrichedToolCall(**call.model_dump(), tool_info=info))

        data = message.model_dump(exclude_none=True)
        data["tool_calls"] = enriched
        return LLMAssistantMessage(**data)

    def _to_public_message(
        self,
        assistant: LLMAssistantMessage,
        message_id: str,
        finish_reason: Any,
        usage: MessageUsage | None,
    ) -> ModelMessage:
        """Project the stored assistant message onto the client-visible event."""
        return ModelMessage(
            id=message_id,
            thread_id=self.thread_id,
            content=assistant.content,
            tool_calls=[
                PublicToolCall(
                    id=call.id,
                    type=call.type,
                    function=call.function,
                    tool_info=call.tool_info.to_public(),
                )
                for call in (assistant.tool_calls or [])
            ]
            or None,
            finish_reason=finish_reason,
            usage=usage,
        )

    def _completion_for(
        self, assistant: LLMAssistantMessage, public: ModelMessage, finish_reason: Any
    ) -> SubAgentCompletion | None:
        """Build a sub-agent's completion record, when this message ends its work."""
        if self.parent is None:
            return None
        if finish_reason == "length":
            error = "The sub-agent's response was cut off by the output token limit."
            return SubAgentCompletion(
                status="error",
                output=public,
                error=error,
                send_to_parent=LLMToolMessage(
                    tool_call_id=self.parent.tool_call_id, content=error
                ),
            )
        if not assistant.tool_calls:
            return SubAgentCompletion(
                status="done",
                output=public,
                send_to_parent=LLMToolMessage(
                    tool_call_id=self.parent.tool_call_id,
                    content=text_of(assistant.content),
                ),
            )
        return None

    def _replay_completion(self, completion: SubAgentCompletion) -> ThreadFinished:
        """Re-emit a sub-agent outcome recorded on an earlier turn."""
        return ThreadFinished(
            thread_id=self.thread_id,
            status=completion.status,
            title=self.title,
            output=completion.output,
            error=completion.error,
            parent=self.parent,
            send_to_parent=completion.send_to_parent,
        )

    # ------------------------------------------------------------------ #
    # State plumbing                                                     #
    # ------------------------------------------------------------------ #

    def _derive_state(self) -> ThreadState:
        """Work out what to do next, and check the transition is legal."""
        open_ids = open_tool_call_ids(self._context)
        if not open_ids:
            state: ThreadState = "llm-call-required"
        elif pending_approval_calls(self._context) or pending_client_side_calls(self._context):
            state = "user-input-required"
        else:
            state = "tool-response-required"

        if self._current_state is not None and state not in _VALID_TRANSITIONS[self._current_state]:
            raise RuntimeError(f"Illegal thread transition: {self._current_state} -> {state}")
        self._current_state = state
        return state

    def _execution_context(self) -> ExecutionContext:
        return ExecutionContext(
            thread_id=self.thread_id,
            context=self._context,
            usage=self._usage,
            session_id=self.session_id,
            turn_id=self.turn_id,
            metadata=self.metadata,
            artifacts=self.artifacts,
            agent_name=self.definition.name,
            is_sub_agent=self.parent is not None,
        )

    async def _run_hooks(self, hook: str) -> AsyncIterator[Any]:
        """Run one hook across every capability, applying what they yield."""
        execution = self._execution_context()
        for capability in self.capabilities:
            iterator = getattr(capability, hook)(execution)
            async for output in iterator:
                async for event in self._apply(output):
                    yield event

    async def _apply(self, output: CapabilityOutput) -> AsyncIterator[Any]:
        """Translate one capability output into internal events and apply it."""
        if isinstance(output, CapabilityAppend):
            yield self._append(output.messages, events=output.events, usage=output.usage)
        elif isinstance(output, CapabilityReplace):
            event = ReplaceContext(
                thread_id=self.thread_id,
                messages=list(output.messages),
                usage=output.usage,
                event=output.event,
                model_usage=output.model_usage,
            )
            if output.model_usage is not None:
                self._metrics.add_usage(output.model_usage)
            self._metrics.total_compactions += 1
            self._context = list(output.messages)
            self._usage = output.usage
            yield event
            if output.event is not None:
                yield output.event
        elif isinstance(output, CapabilitySetState):
            if output.key not in self._state_keys:
                raise CapabilityStateError(
                    f"No capability on thread {self.thread_id!r} declares state key {output.key!r}"
                )
            self._capability_state[output.key] = output.value
            yield SetState(thread_id=self.thread_id, key=output.key, value=output.value)
        elif isinstance(output, EmitEvent):
            yield output.event

    def _append(
        self,
        messages: Sequence[ContextMessage],
        *,
        events: Sequence[Any] = (),
        usage: ContextUsage | None = None,
        completion: SubAgentCompletion | None = None,
    ) -> AppendContext:
        """Build an append event and apply it to the thread.

        The event is constructed and the in-memory state updated together, but the
        caller yields the event before anything downstream sees the new state —
        the persist-before-yield ordering this module depends on.
        """
        new_usage = usage or self._usage.merged_with(estimate_context_usage(messages))
        event = AppendContext(
            thread_id=self.thread_id,
            messages=list(messages),
            events=list(events),
            usage=new_usage,
            completion=completion,
        )
        self._context.extend(messages)
        self._usage = new_usage
        if completion is not None:
            self._precomputed_completion = completion
        return event

    def deliver_tool_message(self, message: LLMToolMessage) -> AppendContext:
        """Close an open tool call with a result produced outside this thread.

        Used by the orchestrator to hand a sub-agent's summary back to its
        parent. The parent's tool call was left open while the child ran; this
        closes it, which returns the parent to ``llm-call-required`` on its next
        pass.
        """
        return self._append([message])

    def _drain_pending_events(self) -> list[Any]:
        events, self._pending_events = self._pending_events, []
        return events

    def _error_finished(self, message: str, output: ModelMessage | None = None) -> ThreadFinished:
        """Terminal error event, including the tool message a parent is waiting for."""
        return ThreadFinished(
            thread_id=self.thread_id,
            status="error",
            title=self.title,
            error=message,
            output=output,
            parent=self.parent,
            send_to_parent=(
                LLMToolMessage(tool_call_id=self.parent.tool_call_id, content=message)
                if self.parent
                else None
            ),
        )

    def _assert_not_busy(self) -> None:
        if self._busy:
            raise RuntimeError(
                f"Thread {self.thread_id!r} is already running. An AgentThread has one "
                "consumer at a time."
            )

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"AgentThread({self.thread_id!r}, messages={len(self._context)})"


class _StepOutcome:
    """Internal signal from a step generator: stop, or keep going."""

    __slots__ = ("exit", "model_message_id")

    def __init__(self, exit: bool, model_message_id: str = "") -> None:
        self.exit = exit
        self.model_message_id = model_message_id


def _is_cancelled(cancel: asyncio.Event | None) -> bool:
    return cancel is not None and cancel.is_set()


def _is_empty_content(content: Any) -> bool:
    if isinstance(content, str):
        return not content.strip()
    return not content


def _describe(exc: BaseException) -> str:
    """A message worth showing a user.

    Providers and transports reject with all sorts of objects; ``str(exc)`` on
    some of them is empty, which would surface as a blank error.
    """
    text = str(exc).strip()
    return text if text else f"{type(exc).__name__} (no message)"
