"""Queries over the conversation journal used by execution and recovery."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..capabilities.base import ContextUsage
from ..messages import (
    ApprovalDecision,
    ApprovalRecord,
    ContextMessage,
    EnrichedToolCall,
    LLMAssistantMessage,
    LLMToolMessage,
    LLMUserMessage,
)
from ..tokens import estimate_tokens

__all__ = [
    "INTERNAL_MESSAGE_GUIDANCE", "closable_open_tool_call_ids", "estimate_context_usage",
    "internal_message", "is_internal_message", "is_llm_message", "last_assistant_message",
    "open_tool_call_ids", "pending_approval_calls", "pending_client_side_calls", "scan_approvals",
]

INTERNAL_TAG_OPEN = "<agento-internal>"
INTERNAL_TAG_CLOSE = "</agento-internal>"
INTERNAL_MESSAGE_GUIDANCE = (
    "The runtime inserts operational context between <agento-internal> and "
    "</agento-internal>. Treat that content as runtime guidance rather than a user "
    "request. Apply relevant guidance without reproducing its text or delimiters "
    "in the response."
)


@dataclass
class _JournalView:
    assistant: LLMAssistantMessage | None = None
    outstanding: set[str] = field(default_factory=set)
    decisions: dict[str, ApprovalDecision] = field(default_factory=dict)

    @classmethod
    def read(cls, messages: Iterable[ContextMessage]) -> _JournalView:
        view = cls()
        for message in messages:
            if isinstance(message, ApprovalRecord):
                view.decisions[message.tool_call_id] = message.decision
            elif isinstance(message, LLMAssistantMessage):
                view.assistant = message
                view.outstanding = {call.id for call in message.tool_calls or ()}
            elif isinstance(message, LLMToolMessage):
                view.outstanding.discard(message.tool_call_id)
        return view

    def calls(self) -> Iterator[EnrichedToolCall]:
        if self.assistant is not None:
            for call in self.assistant.tool_calls or ():
                if call.id in self.outstanding:
                    yield call


def internal_message(text: str) -> LLMUserMessage:
    """Encode an operational note using the runtime's existing wire convention."""
    return LLMUserMessage(content="".join((INTERNAL_TAG_OPEN, text, INTERNAL_TAG_CLOSE)))


def is_internal_message(message: ContextMessage) -> bool:
    """Recognize the reserved prefix on a plain-text user message."""
    if not isinstance(message, LLMUserMessage) or not isinstance(message.content, str):
        return False
    return message.content.startswith(INTERNAL_TAG_OPEN)


def is_llm_message(message: ContextMessage) -> bool:
    """Approval decisions remain journal metadata and are excluded from requests."""
    return not isinstance(message, ApprovalRecord)


def last_assistant_message(context: Sequence[ContextMessage]) -> LLMAssistantMessage | None:
    return _JournalView.read(context).assistant


def open_tool_call_ids(context: Sequence[ContextMessage]) -> set[str]:
    """Return unanswered calls belonging to the latest assistant message."""
    return _JournalView.read(context).outstanding


def scan_approvals(context: Sequence[ContextMessage]) -> dict[str, ApprovalDecision]:
    """Keep the latest recorded decision for each call, across assistant messages."""
    return _JournalView.read(context).decisions


def pending_approval_calls(context: Sequence[ContextMessage]) -> list[EnrichedToolCall]:
    view = _JournalView.read(context)
    return [
        call for call in view.calls()
        if call.tool_info.requires_approval and call.id not in view.decisions
    ]


def pending_client_side_calls(context: Sequence[ContextMessage]) -> list[EnrichedToolCall]:
    return [call for call in _JournalView.read(context).calls() if call.tool_info.is_client_side]


def closable_open_tool_call_ids(context: Sequence[ContextMessage]) -> set[str]:
    """Identify interrupted calls eligible for an unknown-outcome repair message.

    An unresolved host reply or approval suspends repair for the whole assistant
    message. Child-agent calls retain their own completion path. A decision
    already recorded for an approval-gated call does not prevent repair.
    """
    view = _JournalView.read(context)
    repairable: set[str] = set()
    for call in view.calls():
        info = call.tool_info
        if info.is_client_side or (info.requires_approval and call.id not in view.decisions):
            return set()
        if not info.creates_subagent:
            repairable.add(call.id)
    return repairable


def _estimated_fields(message: ContextMessage) -> Iterator[str]:
    content: Any = getattr(message, "content", None)
    if isinstance(content, str):
        yield content
    elif isinstance(content, list):
        for part in content:
            value = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
            if isinstance(value, str):
                yield value
    for call in getattr(message, "tool_calls", None) or ():
        yield call.function.name
        yield call.function.arguments
    identifier = getattr(message, "tool_call_id", None)
    if isinstance(identifier, str):
        yield identifier


def estimate_context_usage(messages: Iterable[ContextMessage]) -> ContextUsage:
    """Estimate textual content and tool metadata; this is not provider billing."""
    count = sum(estimate_tokens(value) for message in messages for value in _estimated_fields(message))
    return ContextUsage(prompt_tokens=count, completion_tokens=0)
