"""Reading a thread's context.

The loop's state machine is derived entirely from the message list — there is no
separate status field to fall out of sync with it. That makes resuming a turn
from storage trivial (load the messages, ask again) but means the questions below
have to be answered precisely.

The central one is **which tool calls are still open**. A call is open when the
last assistant message issued it and no tool message has answered it yet. Only
the *last* assistant message can have open calls, because the loop never proceeds
past an assistant message until every call it made is answered — so the scan
walks backwards and stops there.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
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
    "INTERNAL_MESSAGE_GUIDANCE",
    "closable_open_tool_call_ids",
    "estimate_context_usage",
    "internal_message",
    "is_internal_message",
    "is_llm_message",
    "last_assistant_message",
    "open_tool_call_ids",
    "pending_approval_calls",
    "pending_client_side_calls",
    "scan_approvals",
]


INTERNAL_TAG_OPEN = "<agento-internal>"
INTERNAL_TAG_CLOSE = "</agento-internal>"

INTERNAL_MESSAGE_GUIDANCE = (
    "Messages wrapped in agento-internal tags are system notes, not words from the user. "
    "The Agent must take them into account but must never mention the tags, quote them, or "
    "produce content containing them."
)


def internal_message(text: str) -> LLMUserMessage:
    """Wrap a system note as a user message the model will not echo.

    Some things have to reach the model mid-conversation without looking like the
    user said them — "the earlier history was summarized", "these files were
    uploaded". Providers only accept ``system`` at the start, so agento uses a
    tagged user message and tells the model about the convention in the system
    prompt.
    """
    return LLMUserMessage(content=f"{INTERNAL_TAG_OPEN}{text}{INTERNAL_TAG_CLOSE}")


def is_internal_message(message: ContextMessage) -> bool:
    """Whether a message is one of agento's internal notes."""
    return (
        isinstance(message, LLMUserMessage)
        and isinstance(message.content, str)
        and message.content.startswith(INTERNAL_TAG_OPEN)
    )


def is_llm_message(message: ContextMessage) -> bool:
    """Whether a message is sent to the model.

    False for :class:`~agento.core.messages.ApprovalRecord`, which is bookkeeping
    the loop needs and the model must never see.
    """
    return not isinstance(message, ApprovalRecord)


def last_assistant_message(context: Sequence[ContextMessage]) -> LLMAssistantMessage | None:
    """The most recent assistant message, if any."""
    for message in reversed(context):
        if isinstance(message, LLMAssistantMessage):
            return message
    return None


def open_tool_call_ids(context: Sequence[ContextMessage]) -> set[str]:
    """Tool calls issued but not yet answered.

    Walks backwards collecting answered ids until it reaches the last assistant
    message, then returns that message's calls minus the answered ones.
    """
    answered: set[str] = set()
    for message in reversed(context):
        if isinstance(message, LLMToolMessage):
            answered.add(message.tool_call_id)
            continue
        if isinstance(message, LLMAssistantMessage):
            issued = {call.id for call in (message.tool_calls or [])}
            return issued - answered
    return set()


def scan_approvals(context: Sequence[ContextMessage]) -> dict[str, ApprovalDecision]:
    """Every approval decision recorded so far, by tool call id."""
    decisions: dict[str, ApprovalDecision] = {}
    for message in context:
        if isinstance(message, ApprovalRecord):
            decisions[message.tool_call_id] = message.decision
    return decisions


def pending_approval_calls(context: Sequence[ContextMessage]) -> list[EnrichedToolCall]:
    """Open calls that need approval and have not received a decision."""
    open_ids = open_tool_call_ids(context)
    if not open_ids:
        return []
    decisions = scan_approvals(context)
    assistant = last_assistant_message(context)
    return [
        call
        for call in (assistant.tool_calls or [] if assistant else [])
        if call.id in open_ids and call.tool_info.requires_approval and call.id not in decisions
    ]


def pending_client_side_calls(context: Sequence[ContextMessage]) -> list[EnrichedToolCall]:
    """Open calls the host application must answer."""
    open_ids = open_tool_call_ids(context)
    if not open_ids:
        return []
    assistant = last_assistant_message(context)
    return [
        call
        for call in (assistant.tool_calls or [] if assistant else [])
        if call.id in open_ids and call.tool_info.is_client_side
    ]


def closable_open_tool_call_ids(context: Sequence[ContextMessage]) -> set[str]:
    """Open calls that can be safely closed with a placeholder result.

    A turn interrupted mid-execution — cancelled, crashed, timed out — leaves
    tool calls with no results, and providers reject that conversation outright.
    Those calls are *closable*: writing an "outcome unknown" result makes
    the history valid again without claiming the action never happened.

    Three kinds are **not** closable, because something is still legitimately
    expected to answer them:

    * approval-gated calls — a human is being asked,
    * client-side calls — the host is being asked,
    * sub-agent calls — a child thread is running.

    If the last assistant message contains *any* of those, nothing is closed: the
    turn is properly paused, not broken.
    """
    assistant = last_assistant_message(context)
    if assistant is None or not assistant.tool_calls:
        return set()

    if pending_approval_calls(context) or pending_client_side_calls(context):
        return set()

    open_ids = open_tool_call_ids(context)
    return {
        call.id
        for call in assistant.tool_calls
        if call.id in open_ids and not call.tool_info.creates_subagent
    }


def estimate_context_usage(messages: Iterable[ContextMessage]) -> ContextUsage:
    """Estimate the token cost of some messages.

    Used between model calls, when the provider has not yet given a real count.
    Deliberately counts tool call names, arguments and ids as well as content —
    a message with ten tool calls is not free.
    """
    tokens = 0
    for message in messages:
        content: Any = getattr(message, "content", None)
        if isinstance(content, str):
            tokens += estimate_tokens(content)
        elif isinstance(content, list):
            for part in content:
                text = getattr(part, "text", None)
                if text is None and isinstance(part, dict):
                    text = part.get("text")
                if isinstance(text, str):
                    tokens += estimate_tokens(text)

        for call in getattr(message, "tool_calls", None) or []:
            tokens += estimate_tokens(call.function.name)
            tokens += estimate_tokens(call.function.arguments)

        tool_call_id = getattr(message, "tool_call_id", None)
        if isinstance(tool_call_id, str):
            tokens += estimate_tokens(tool_call_id)

    return ContextUsage(prompt_tokens=tokens, completion_tokens=0)
