"""The event stream — agento's real public contract.

Everything an agent does is observable as a sequence of events. Running a turn is
``async for event in turn.stream()``, and these models are what comes out. If you
are integrating agento into an application, this module is the one to read first.

Two kinds of event travel down the stream:

**Durable events** are written to the session store before they are yielded, so
they can be replayed later with ``turn.list_events()`` or
``session.list_events()``. Every event here is durable except one.

**Transient events** — :class:`ModelMessageDelta` — are the token-by-token
stream. They are never persisted, because the complete :class:`ModelMessage` that
follows contains everything they carried. Render deltas for responsiveness;
treat the ``ModelMessage`` as the truth.

Ordering is defined by ``event.id``: ids are monotonic ULIDs, so lexicographic
order *is* creation order, both within a stream and after a reload from storage.

A note on the three "the run is paused" events. :class:`ApprovalRequired`,
:class:`ClientToolRequired` and :class:`McpAuthRequired` all mean the same thing
structurally — the loop cannot continue without something only the host can
provide. They arrive just before :class:`TurnDone`, and they are repeated in
``TurnDone.state.required_actions`` so a caller that only inspects the terminal
state still sees them. You answer by starting the next turn with the matching
input item (:class:`ToolApproval`, :class:`ToolReply`, or a fresh user message
once the MCP server is authorized).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .._ids import new_event_id
from .messages import (
    ApprovalDecision,
    FinishReason,
    ToolCall,
    ToolInfo,
    Usage,
    UserContentPart,
)

__all__ = [
    "ActionRequired",
    "AgentInfo",
    "ApprovalRequired",
    "ArtifactCreated",
    "ClientToolRequired",
    "ContextCompacted",
    "Event",
    "InputTokenBreakdown",
    "McpAuthRequired",
    "McpInitialized",
    "McpServerAuth",
    "McpServerInit",
    "ModelMessage",
    "ModelMessageDelta",
    "PublicToolCall",
    "StreamEvent",
    "ThreadCreated",
    "ThreadDone",
    "ThreadParent",
    "ThreadState",
    "ThreadStateDone",
    "ThreadStateError",
    "ToolApproval",
    "ToolCallRef",
    "ToolReply",
    "ToolResult",
    "TurnCreated",
    "TurnDone",
    "TurnInput",
    "TurnMetrics",
    "TurnState",
    "TurnStateCancelled",
    "TurnStateDone",
    "TurnStateError",
    "TurnStateRunning",
    "UserMessage",
    "now_iso",
]


def now_iso() -> str:
    """The current UTC time as an ISO-8601 string with a ``Z`` suffix."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class _Event(BaseModel):
    """Shared shape for every event.

    Attributes:
        id: Monotonic ULID. Also the ordering key.
        created_at: ISO-8601 UTC timestamp.
        thread_id: Which agent thread produced this. ``"main"`` for the root
            agent, a generated id for a sub-agent, and ``None`` for events that
            belong to the run as a whole rather than to any one thread.
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(default_factory=new_event_id)
    created_at: str = Field(default_factory=now_iso)
    thread_id: str | None = None


# --------------------------------------------------------------------------- #
# Input items — what you send *into* a turn                                     #
# --------------------------------------------------------------------------- #


class UserMessage(BaseModel):
    """A message from the user.

    ``content`` is either a plain string or a list of parts when you need to
    attach files or images::

        UserMessage(content="hello")
        UserMessage(content=[TextPart(text="summarize this"),
                             FilePart(name="q3.csv", data="data:text/csv;base64,...")])

    ``session.create_turn(input="hello")`` is shorthand for the first form.
    """

    type: Literal["user.message"] = "user.message"
    content: str | list[UserContentPart]


class ToolApproval(BaseModel):
    """Answer to an :class:`ApprovalRequired` event.

    Args:
        thread_id: The thread from the ``ApprovalRequired`` event.
        tool_call_id: The specific call being approved or denied.
        decision: ``"allow"`` runs the tool; ``"deny"`` returns the refusal to the
            model as the tool's result, so it can react rather than stall.
        reason: Shown to the model when denying. Worth filling in — "needs a
            manager's sign-off" produces much better behaviour than silence.

    A resuming turn must answer **every** pending approval and client-side tool
    call in one batch; a partial batch is rejected with
    :class:`~agento.errors.InvalidSendInputError`.
    """

    type: Literal["user.tool_approval"] = "user.tool_approval"
    thread_id: str
    tool_call_id: str
    decision: ApprovalDecision = "allow"
    reason: str | None = None


class ToolReply(BaseModel):
    """Answer to a :class:`ClientToolRequired` event.

    The host executed the tool (or asked the user the question) and is handing
    back the result as the tool's output.

    Args:
        thread_id: The thread from the ``ClientToolRequired`` event.
        tool_call_id: The call being answered.
        content: The result, as the model should see it.
    """

    type: Literal["user.tool_response"] = "user.tool_response"
    thread_id: str
    tool_call_id: str
    content: str


TurnInput = Annotated[
    UserMessage | ToolApproval | ToolReply,
    Field(discriminator="type"),
]
"""One item of turn input.

A batch must be homogeneous: either all user messages, or all approvals and tool
replies. Mixing them is rejected, because "here is my answer to your question,
and also a new request" has no well-defined ordering against the pending call.
"""


# --------------------------------------------------------------------------- #
# Supporting shapes                                                            #
# --------------------------------------------------------------------------- #


class PublicToolCall(ToolCall):
    """A tool call as it appears on an event, with redacted tool info."""

    tool_info: ToolInfo | None = None


class InputTokenBreakdown(BaseModel):
    """Where a model call's input tokens went.

    These are estimates, and they exist to answer "why is this agent expensive?".
    ``harness`` covers agento's own prompt scaffolding and built-in tools;
    ``instructions`` is your agent's system prompt; ``tool_definitions`` is the
    JSON schemas of the tools you attached; ``messages`` is the conversation.
    """

    harness: int = 0
    skills: int = 0
    instructions: int = 0
    tool_definitions: int = 0
    messages: int = 0


class MessageUsage(Usage):
    """Per-call usage, with the attribution breakdown attached."""

    input_tokens_breakdown: InputTokenBreakdown = Field(default_factory=InputTokenBreakdown)


class ThreadParent(BaseModel):
    """Links a sub-agent thread back to the call that created it."""

    thread_id: str
    tool_call_id: str


class AgentInfo(BaseModel):
    """The request that created a sub-agent.

    Attributes:
        name: Short label the parent gave the sub-agent, e.g. ``"pr-reviewer"``.
        input: The self-contained task. A sub-agent sees no conversation history,
            so this is all the context it gets.
        model: Optional model override, when the parent chose a different model
            for the delegated work.
    """

    type: Literal["dynamic"] = "dynamic"
    name: str
    input: str
    model: str | None = None


class ToolCallRef(BaseModel):
    """Points at a pending tool call and the message that requested it."""

    id: str
    source_event_id: str


class McpServerInit(BaseModel):
    """An MCP server that connected during this turn."""

    id: str
    name: str
    session_id: str | None = None
    transport: str | None = None


class McpServerAuth(BaseModel):
    """An MCP server that needs authorization before its tools can be used."""

    id: str
    name: str
    auth_url: str


class TurnMetrics(BaseModel):
    """Totals for a turn, summed across the root agent and every sub-agent."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cache_read_tokens: int | None = None
    total_cache_write_tokens: int | None = None
    total_reasoning_tokens: int | None = None
    total_cost_usd: float | None = None
    iterations: int = 0
    total_tool_calls: int = 0
    total_sub_agents: int = 0
    total_compactions: int = 0


# --------------------------------------------------------------------------- #
# Events                                                                       #
# --------------------------------------------------------------------------- #


class ModelMessage(_Event):
    """A complete assistant message.

    Emitted twice per model call, and the difference matters:

    * **At the start**, as a placeholder carrying only ``id`` — this is what
      opens a delta stream, and every following :class:`ModelMessageDelta` shares
      that id.
    * **At the end**, fully populated. This is the durable record.

    So a client can create a message bubble on the first, append deltas to it,
    and replace its contents with the second.
    """

    type: Literal["model.message"] = "model.message"
    content: str | list[Any] | None = None
    tool_calls: list[PublicToolCall] | None = None
    finish_reason: FinishReason | None = None
    usage: MessageUsage | None = None


class ModelMessageDelta(_Event):
    """One streaming chunk. Never persisted.

    ``id`` matches the :class:`ModelMessage` this delta belongs to.
    """

    type: Literal["model.message.delta"] = "model.message.delta"
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: FinishReason | None = None
    usage: MessageUsage | None = None


class ToolResult(_Event):
    """The result of one executed tool call.

    ``content`` is what the model will see. If the result was large enough to be
    offloaded, this is already the shortened version and ``artifact_id`` points
    at the full payload in the artifact store.
    """

    type: Literal["tool.result"] = "tool.result"
    tool_call_id: str
    content: str
    is_error: bool = False
    artifact_id: str | None = None


class ThreadCreated(_Event):
    """A sub-agent thread started."""

    type: Literal["thread.created"] = "thread.created"
    parent: ThreadParent
    agent_info: AgentInfo
    title: str


class ThreadStateDone(BaseModel):
    """A thread finished normally."""

    status: Literal["done"] = "done"
    output: ModelMessage


class ThreadStateError(BaseModel):
    """A thread failed."""

    status: Literal["error"] = "error"
    error: str
    output: ModelMessage | None = None


ThreadState = Annotated[
    ThreadStateDone | ThreadStateError,
    Field(discriminator="status"),
]


class ThreadDone(_Event):
    """A sub-agent thread reached a terminal state.

    The root thread does not emit this; its outcome is the turn's outcome.
    """

    type: Literal["thread.done"] = "thread.done"
    parent: ThreadParent | None = None
    title: str
    state: ThreadState


class ApprovalRequired(_Event):
    """One or more tool calls need a human decision before they run.

    Answer with :class:`ToolApproval` items on the next turn.
    """

    type: Literal["tool.approval_required"] = "tool.approval_required"
    tool_calls: list[ToolCallRef]


class ClientToolRequired(_Event):
    """One or more tool calls must be executed by the host application.

    This covers both genuine client-side tools you registered and the built-in
    ``ask_user_question``. Answer with :class:`ToolReply` items on the next turn.
    """

    type: Literal["tool.response_required"] = "tool.response_required"
    tool_calls: list[ToolCallRef]


class McpInitialized(_Event):
    """MCP servers connected. Carries observed session ids for diagnostics."""

    type: Literal["mcp.initialize"] = "mcp.initialize"
    mcp_servers: list[McpServerInit]


class McpAuthRequired(_Event):
    """MCP servers need authorization; each entry carries the URL to send the user to."""

    type: Literal["mcp.auth_required"] = "mcp.auth_required"
    mcp_servers: list[McpServerAuth]


class ArtifactCreated(_Event):
    """A payload was written to the artifact store.

    Happens when a tool result is too large for context, or when a user uploads a
    non-inline file. The agent can read it back with the built-in
    ``read_artifact`` tool; your application can fetch it from the store directly.
    """

    type: Literal["artifact.created"] = "artifact.created"
    artifact_id: str
    name: str
    size_bytes: int
    mime_type: str | None = None
    source_tool: str | None = None


class ContextCompacted(_Event):
    """The conversation was summarized and the older history replaced.

    Purely informational, but worth surfacing: after this the agent is working
    from a summary, so a user who asks "what did I say at the start?" may get a
    condensed answer even though the full event log is still intact.
    """

    type: Literal["context.compacted"] = "context.compacted"
    reason: Literal["compaction"] = "compaction"
    messages_before: int
    tokens_before: int
    usage: Usage | None = None


class TurnCreated(_Event):
    """The turn started. Always the first event of a stream."""

    type: Literal["turn.created"] = "turn.created"
    turn_id: str
    previous_turn_id: str | None = None
    input: list[TurnInput] = Field(default_factory=list)


class TurnStateRunning(BaseModel):
    """The turn is executing."""

    status: Literal["running"] = "running"


class TurnStateDone(BaseModel):
    """The turn finished.

    "Done" includes *paused*: if ``required_actions`` is non-empty the agent did
    not fail, it is waiting for you. Check that list before treating the run as
    complete.

    Attributes:
        output: The final assistant message, or ``None`` when the turn ended
            paused with no closing message.
        required_actions: Approvals, client-side tool calls or MCP authorizations
            still outstanding.
    """

    status: Literal["done"] = "done"
    output: ModelMessage | None = None
    required_actions: list[ActionRequired] = Field(default_factory=list)
    completed_at: str = Field(default_factory=now_iso)
    metrics: TurnMetrics | None = None


class TurnStateCancelled(BaseModel):
    """The turn was cancelled before completing."""

    status: Literal["cancelled"] = "cancelled"
    reason: str
    completed_at: str = Field(default_factory=now_iso)
    metrics: TurnMetrics | None = None


class TurnStateError(BaseModel):
    """The turn failed."""

    status: Literal["error"] = "error"
    message: str
    completed_at: str = Field(default_factory=now_iso)
    metrics: TurnMetrics | None = None


TurnState = Annotated[
    TurnStateRunning | TurnStateDone | TurnStateCancelled | TurnStateError,
    Field(discriminator="status"),
]


class TurnDone(_Event):
    """The turn reached a terminal state. Always the last event of a stream."""

    type: Literal["turn.done"] = "turn.done"
    state: TurnState


ActionRequired = Annotated[
    ApprovalRequired | ClientToolRequired | McpAuthRequired,
    Field(discriminator="type"),
]
"""Something the host must resolve before the agent can continue."""


Event = Annotated[
    TurnCreated | TurnDone | ModelMessage | ToolResult | ThreadCreated | ThreadDone | ApprovalRequired | ClientToolRequired | McpInitialized | McpAuthRequired | ArtifactCreated | ContextCompacted,
    Field(discriminator="type"),
]
"""Every durable event. This is what a session store persists and replays."""


StreamEvent = (TurnCreated | TurnDone | ModelMessage | ModelMessageDelta | ToolResult | ThreadCreated | ThreadDone | ApprovalRequired | ClientToolRequired | McpInitialized | McpAuthRequired | ArtifactCreated | ContextCompacted)
"""Everything ``turn.stream()`` yields — the durable events plus live deltas."""


# Late rebuild: TurnStateDone references ActionRequired, which is defined after it.
TurnStateDone.model_rebuild()
TurnDone.model_rebuild()
