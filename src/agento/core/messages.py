"""The message and tool-call types that make up an agent's context.

These models are the internal, durable representation of a conversation. They
are deliberately close to the OpenAI chat-completions shape, because that shape
is what every provider adapter (and LiteLLM in particular) already speaks — but
they carry extra fields that agento needs and the wire format does not have.

The important extra is :class:`InternalToolInfo`, attached to every tool call the
model makes. When the model emits a call to ``search_issues``, the raw wire data
tells us only the name and the JSON arguments. agento additionally needs to know
*which* tool set it came from, whether it requires human approval, whether the
host application must execute it, and whether it spawns a sub-agent. Those facts
drive the loop's state machine, so they are resolved once at the moment the
assistant message is built and then travel with the message for the rest of the
session.

Two rules keep this honest:

* Internal fields never reach the model. :func:`to_wire_message` strips them.
* Internal fields never reach the client either. Events carry the redacted
  :class:`ToolInfo` instead (see :mod:`agento.core.events`).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ApprovalDecision",
    "ApprovalRecord",
    "AssistantContentPart",
    "ContextMessage",
    "EnrichedToolCall",
    "FilePart",
    "FinishReason",
    "FunctionCall",
    "ImagePart",
    "InternalToolInfo",
    "LLMAssistantMessage",
    "LLMToolMessage",
    "LLMUserMessage",
    "RawAssistantMessage",
    "RedactedThinkingBlock",
    "TextPart",
    "ThinkingBlock",
    "ToolCall",
    "ToolInfo",
    "ToolKind",
    "Usage",
    "UserContentPart",
    "empty_usage",
    "text_of",
    "to_wire_message",
]


class _Model(BaseModel):
    """Base for every message model.

    ``extra="allow"`` is deliberate: providers keep inventing fields (thought
    signatures, citation blocks, provider-specific metadata), and silently
    dropping them on a round trip through the store would break multi-turn replay
    for the providers that require them back.
    """

    model_config = ConfigDict(extra="allow")


# --------------------------------------------------------------------------- #
# Content parts                                                                #
# --------------------------------------------------------------------------- #


class TextPart(_Model):
    """A run of plain text inside a multi-part message."""

    type: Literal["text"] = "text"
    text: str


class ImagePart(_Model):
    """An image, as a URL or a ``data:`` URI.

    Args:
        url: ``https://…`` or ``data:image/png;base64,…``.
    """

    type: Literal["image"] = "image"
    url: str


class FilePart(_Model):
    """A file attached to a user message.

    Args:
        name: Filename shown to the model.
        data: A ``data:<mime>;base64,<payload>`` URI. The MIME type is parsed
            out of the URI, so it must be present.

    How a file is handled depends on its MIME type. Images and PDFs are passed
    inline to models that accept them. Anything else is written to the configured
    :class:`~agento.artifacts.base.ArtifactStore`, and the model is told the
    artifact id so it can read the content back through a tool.
    """

    type: Literal["file"] = "file"
    name: str
    data: str


UserContentPart = Annotated[
    TextPart | ImagePart | FilePart,
    Field(discriminator="type"),
]
"""One element of a structured user message."""


class RefusalPart(_Model):
    """A provider-generated refusal, kept distinct from ordinary assistant text."""

    type: Literal["refusal"] = "refusal"
    refusal: str


AssistantContentPart = Annotated[
    TextPart | RefusalPart,
    Field(discriminator="type"),
]


# --------------------------------------------------------------------------- #
# Reasoning                                                                    #
# --------------------------------------------------------------------------- #


class ThinkingBlock(_Model):
    """A block of model reasoning.

    Args:
        thinking: The reasoning text.
        signature: An opaque provider signature. Anthropic requires the signature
            to be echoed back on subsequent turns for extended-thinking
            conversations, which is why these blocks are stored in context rather
            than discarded after streaming.
    """

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str | None = None


class RedactedThinkingBlock(_Model):
    """Reasoning the provider redacted. The opaque payload must still be replayed."""

    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: str


ThinkingBlockUnion = Annotated[
    ThinkingBlock | RedactedThinkingBlock,
    Field(discriminator="type"),
]


# --------------------------------------------------------------------------- #
# Tool calls                                                                   #
# --------------------------------------------------------------------------- #

ToolKind = Literal["builtin", "mcp", "local", "unknown"]
"""Where a tool came from.

``builtin``  agento's own tools (ask_user_question, read_skill, create_sub_agent…)
``mcp``      a tool on a remote MCP server
``local``    a Python function the host registered
``unknown``  the model hallucinated a name that maps to nothing
"""


class FunctionCall(_Model):
    """The name and raw JSON arguments of a tool invocation."""

    name: str
    arguments: str = "{}"


class ToolCall(_Model):
    """A tool call exactly as the model emitted it."""

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ToolInfo(_Model):
    """Public description of the tool behind a call. Safe to show a client.

    Deliberately omits the approval / client-side / sub-agent flags, which are
    orchestration details rather than something a UI should key off.
    """

    kind: ToolKind
    name: str
    source_id: str | None = None
    source_name: str | None = None


class InternalToolInfo(_Model):
    """Everything the loop knows about the tool behind a call.

    Attributes:
        kind: Origin of the tool.
        source_id: Stable id of the owning tool set.
        source_name: Display name of the owning tool set.
        original_tool_name: The tool's real name, before agento sanitized or
            de-duplicated it for the model.
        requires_approval: The loop must pause and ask a human before executing.
        is_client_side: agento cannot execute this; the host supplies the result.
        is_deferred: Reached indirectly through the deferred-tools ``call_tool``
            wrapper rather than as a preloaded tool.
        creates_subagent: Executing this spawns a child thread instead of
            returning a result.
    """

    kind: ToolKind
    source_id: str = ""
    source_name: str = ""
    original_tool_name: str
    requires_approval: bool = False
    is_client_side: bool = False
    is_deferred: bool = False
    creates_subagent: bool = False

    def to_public(self) -> ToolInfo:
        """Project onto the client-visible :class:`ToolInfo`."""
        return ToolInfo(
            kind=self.kind,
            name=self.original_tool_name,
            source_id=self.source_id or None,
            source_name=self.source_name or None,
        )


class EnrichedToolCall(ToolCall):
    """A tool call plus the resolved information about its tool."""

    tool_info: InternalToolInfo


def unknown_tool_info(tool_name: str) -> InternalToolInfo:
    """Info for a tool name the model invented. Executing it returns an error."""
    return InternalToolInfo(kind="unknown", original_tool_name=tool_name)


# --------------------------------------------------------------------------- #
# Messages                                                                     #
# --------------------------------------------------------------------------- #


class LLMUserMessage(_Model):
    """A user turn in the conversation."""

    role: Literal["user"] = "user"
    content: str | list[UserContentPart]


class RawAssistantMessage(_Model):
    """A model turn exactly as the provider produced it.

    Its tool calls carry no ``tool_info``, because at the moment a response is
    assembled nothing has yet looked up which tool set each name belongs to. The
    runtime enriches this into an :class:`LLMAssistantMessage` immediately
    afterwards, and only the enriched form is stored.

    Attributes:
        content: Text, structured parts, or ``None`` for a tool-only completion.
        tool_calls: Calls the model wants executed. ``None`` (not ``[]``) when
            there are none — several providers reject an empty array on replay.
        thinking_blocks: Reasoning, retained for multi-turn replay.
        reasoning_content: Flat reasoning text, convenient for display. Redundant
            with ``thinking_blocks`` and not sent back to the provider.
        source: Which model produced this, for debugging mixed-model sessions.
    """

    role: Literal["assistant"] = "assistant"
    content: str | list[AssistantContentPart] | None = None
    tool_calls: list[ToolCall] | None = None
    thinking_blocks: list[ThinkingBlockUnion] | None = None
    reasoning_content: str | None = None
    source: str | None = None


class LLMAssistantMessage(RawAssistantMessage):
    """A model turn with every tool call resolved to a known tool.

    This is the form that lives in a thread's context and survives to the next
    turn, because the loop's state machine reads ``tool_info`` to decide whether
    a call needs approval, needs the host to execute it, or spawns a sub-agent.
    """

    tool_calls: list[EnrichedToolCall] | None = None  # type: ignore[assignment]


class LLMToolMessage(_Model):
    """The result of one tool call, keyed back to the call by id."""

    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str


ApprovalDecision = Literal["allow", "deny"]


class ApprovalRecord(_Model):
    """A human's approval decision, recorded in context but never sent to the model.

    It lives in the message list so that a turn resumed from storage can tell
    which pending approvals have already been answered — the decision has to be
    as durable as the tool call it answers.
    """

    type: Literal["approval.decision"] = "approval.decision"
    tool_call_id: str
    decision: ApprovalDecision
    reason: str | None = None


LLMContextMessage = (LLMUserMessage | LLMAssistantMessage | LLMToolMessage)
"""A message that is actually sent to the model."""

ContextMessage = (LLMUserMessage | LLMAssistantMessage | LLMToolMessage | ApprovalRecord)
"""Anything stored in a thread's context, including non-model bookkeeping."""


# --------------------------------------------------------------------------- #
# Usage                                                                        #
# --------------------------------------------------------------------------- #


class Usage(_Model):
    """Token accounting for one model call.

    Every field except the first three is optional because provider support
    varies; ``None`` means "not reported", which is different from zero.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_usd: float | None = None


def empty_usage() -> Usage:
    """A zeroed :class:`Usage`."""
    return Usage()


FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "error"]
"""Why the model stopped.

``length`` is the one the loop treats specially: it means the response was
truncated, so the turn ends in an error rather than silently continuing from a
half-formed message.
"""


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def text_of(content: str | list[Any] | None) -> str:
    """Flatten any message content into plain text.

    Used for sub-agent results, session titles and log lines — anywhere a single
    string is needed and structure is not.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, BaseModel):
            data = item.model_dump()
            parts.append(str(data.get("text") or data.get("refusal") or ""))
        elif isinstance(item, dict):
            parts.append(str(item.get("text") or item.get("refusal") or ""))
    return "".join(parts)


def to_wire_message(message: ContextMessage) -> dict[str, Any]:
    """Convert a stored context message into the dict a provider expects.

    This is the single place where agento's internal extras are removed:

    * ``tool_info`` is stripped from each tool call — it is agento's bookkeeping,
      and providers reject unknown fields inside ``tool_calls``.
    * ``reasoning_content`` is dropped; it duplicates ``thinking_blocks`` and is
      display-only.
    * :class:`ApprovalRecord` returns ``{}`` — the caller filters those out
      beforehand, and this is the belt-and-braces guard.
    * ``tool_calls: []`` becomes an omitted key.

    Args:
        message: A message from a thread's context.

    Returns:
        A plain dict ready to place in a provider request.
    """
    if isinstance(message, ApprovalRecord):
        return {}

    data = message.model_dump(exclude_none=True)
    data.pop("reasoning_content", None)
    data.pop("source", None)
    if isinstance(data.get("content"), list):
        content = []
        for part in data["content"]:
            if part.get("type") == "image":
                content.append({"type": "image_url", "image_url": {"url": part["url"]}})
            elif part.get("type") == "file":
                content.append({"type": "file", "file": {
                    "filename": part["name"], "file_data": part["data"],
                }})
            else:
                content.append(part)
        data["content"] = content

    tool_calls = data.get("tool_calls")
    if isinstance(tool_calls, list):
        if not tool_calls:
            data.pop("tool_calls", None)
        else:
            data["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": call.get("type", "function"),
                    "function": {
                        "name": call["function"]["name"],
                        "arguments": call["function"].get("arguments", "{}"),
                    },
                }
                for call in tool_calls
            ]
    return data
