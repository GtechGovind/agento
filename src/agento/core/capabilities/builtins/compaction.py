"""Context compaction — surviving a conversation longer than the window.

Every long agent run eventually approaches the model's context limit. The naive
responses are both bad: truncating the oldest messages loses the original task,
and refusing to continue strands the user mid-job.

Compaction takes a third route. When the conversation crosses a threshold, agento
asks the model to write a **structured summary** of everything so far — the
original request, the decisions taken, the files touched, the errors hit and how
they were fixed, what remains — and replaces the working history with that
summary. The agent keeps going with room to think.

**When it triggers.** At 80% of the model's context length, when
:class:`~agento.core.llm.base.ModelProperties` reports one. Otherwise at 50,000
input tokens, which is a conservative guess that works on nearly every model. An
explicit ``threshold_tokens`` overrides both.

**What it costs.** One extra model call, and the older messages leave the agent's
working memory. The durable event log still has every message — ``list_events()``
returns the whole conversation — but the *agent* is now working from the summary.
That is a real trade: it is why the summary prompt asks for verbatim user
messages and specific detail rather than a tidy paragraph.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from ...events import ContextCompacted
from ...llm.base import LLMRequest
from ...messages import (
    ApprovalRecord,
    ContextMessage,
    LLMAssistantMessage,
    LLMToolMessage,
    LLMUserMessage,
    text_of,
)
from ..base import Capability, CapabilityOutput, ExecutionContext, ReplaceContext

__all__ = ["ContextCompaction", "DEFAULT_THRESHOLD_TOKENS", "DEFAULT_CONTEXT_RATIO"]

DEFAULT_THRESHOLD_TOKENS = 50_000
"""Used when the model's context length is unknown."""

DEFAULT_CONTEXT_RATIO = 0.8
"""Fraction of a known context window at which to compact."""

# Roughly the size of the summarization prompt below. Counted against the
# threshold so the compaction call itself cannot be what overflows the window.
_PROMPT_TOKENS = 950

SUMMARY_PROMPT = """\
Summarize the conversation above so that another agent could pick up the work
without having seen it. Be thorough about specifics — this summary replaces the
conversation, so anything you leave out is lost.

Write your summary under these headings:

1. Original request and intent
   What the user asked for, in their terms, including any later refinements.

2. Key decisions and constraints
   Choices made and why, constraints stated, approaches ruled out.

3. Work completed
   What was actually done. Name files, records, identifiers and values exactly.
   Include the content of anything created or changed where it matters.

4. Errors encountered and how they were resolved
   Include feedback from the user, especially corrections. These are the details
   most often lost, and most expensive to lose.

5. Every user message
   List each message the user sent, verbatim or near-verbatim. Do not
   paraphrase away their wording.

6. Outstanding work
   What remains, in enough detail to act on.

7. Current state
   What was happening immediately before this summary, and the obvious next
   step — but only if it follows directly from the user's most recent request.

Write the summary only. No preamble, no commentary."""

CONTINUATION_NOTE = (
    "The message above is a summary of the conversation so far; the earlier messages have been "
    "removed to free context. Do not repeat work described there as already done. If a result was "
    "already produced, present it rather than recomputing it."
)


def resolve_threshold(
    *,
    configured: int | None,
    context_length: int | None,
    max_output_tokens: int | None = None,
) -> int:
    """Work out the token count at which to compact.

    Args:
        configured: An explicit threshold, which always wins.
        context_length: The model's total window, if known.
        max_output_tokens: Reserved output budget, if configured. Subtracted from
            the window so compaction leaves room for the reply.

    Returns:
        A token threshold.
    """
    if configured is not None:
        return configured
    if context_length is None:
        return DEFAULT_THRESHOLD_TOKENS

    ratio_threshold = int(context_length * DEFAULT_CONTEXT_RATIO)
    if max_output_tokens:
        input_budget = context_length - max_output_tokens
        if input_budget > 0:
            # Never let an oversized output reservation push the threshold so low
            # that compaction fires on every call.
            return min(ratio_threshold, input_budget)
    return ratio_threshold


def _render_for_summary(context: Sequence[ContextMessage]) -> str:
    """Flatten the conversation into text for the summarizer.

    Tool calls and results are included — they are usually where the concrete
    facts live, and a summary that omits them tends to be pleasant and useless.
    """
    lines: list[str] = []
    for index, message in enumerate(context):
        if isinstance(message, ApprovalRecord):
            lines.append(f"[{index}] <approval tool_call_id={message.tool_call_id}>: {message.decision}")
        elif isinstance(message, LLMUserMessage):
            lines.append(f"[{index}] <user>: {text_of(message.content)}")
        elif isinstance(message, LLMToolMessage):
            lines.append(f"[{index}] <tool-result id={message.tool_call_id}>: {message.content}")
        elif isinstance(message, LLMAssistantMessage):
            for block in message.thinking_blocks or []:
                thinking = getattr(block, "thinking", None)
                if thinking:
                    lines.append(f"[{index}] <assistant-reasoning>: {thinking}")
            content = text_of(message.content)
            if content:
                lines.append(f"[{index}] <assistant>: {content}")
            for call in message.tool_calls or []:
                lines.append(
                    f"[{index}] <tool-call id={call.id} name={call.function.name}>: "
                    f"{call.function.arguments}"
                )
    return "\n".join(lines)


class ContextCompaction(Capability):
    """Summarizes and replaces the conversation when it grows too large.

    Args:
        llm: The client used for the summarization call. Normally the agent's own
            model; pass a cheaper one if summaries are a cost concern.
        threshold_tokens: Explicit trigger. Overrides the derived threshold.
        context_length: The model's window, if the client does not report one.
        max_output_tokens: Output budget to reserve.
        min_messages: Never compact a conversation shorter than this. Prevents a
            pathological loop when a single enormous message is what crossed the
            threshold — summarizing it would not shrink anything.
    """

    name = "context_compaction"

    def __init__(
        self,
        llm: Any,
        *,
        threshold_tokens: int | None = None,
        context_length: int | None = None,
        max_output_tokens: int | None = None,
        min_messages: int = 3,
    ) -> None:
        self._llm = llm
        properties = getattr(llm, "properties", None)
        self._threshold = resolve_threshold(
            configured=threshold_tokens,
            context_length=context_length or getattr(properties, "context_length", None),
            max_output_tokens=max_output_tokens
            or getattr(properties, "max_output_tokens", None),
        )
        self._min_messages = min_messages

    @property
    def threshold_tokens(self) -> int:
        """The resolved trigger point, for tests and diagnostics."""
        return self._threshold

    async def pre_llm(self, context: ExecutionContext) -> AsyncIterator[CapabilityOutput]:
        current = context.usage.total()
        if current + _PROMPT_TOKENS < self._threshold:
            return
        if len(context.context) < self._min_messages:
            return

        transcript = _render_for_summary(context.context)
        response = await self._llm.complete(
            LLMRequest(
                messages=[
                    {"role": "user", "content": transcript},
                    {"role": "user", "content": SUMMARY_PROMPT},
                ],
                params={},
            )
        )

        summary = text_of(response.message.content)
        if not summary.strip():
            # A failed summary must not destroy the conversation. Leaving the
            # context alone means the next call may fail on length — which is
            # recoverable — whereas replacing it with nothing is not.
            return

        from ...._ids import new_event_id
        from ...runtime.context_utils import estimate_context_usage, internal_message

        replacement: list[ContextMessage] = [
            LLMAssistantMessage(content=summary),
            internal_message(CONTINUATION_NOTE),
        ]

        yield ReplaceContext(
            messages=replacement,
            # Sized from the replacement itself, not from the summarization
            # call's input tokens — those cover the whole transcript we just
            # discarded, and reusing them would leave the thread looking as full
            # as before and liable to compact again on the next iteration.
            usage=estimate_context_usage(replacement),
            event=ContextCompacted(
                id=new_event_id(),
                thread_id=context.thread_id,
                messages_before=len(context.context),
                tokens_before=current,
                usage=response.usage,
            ),
            model_usage=response.usage,
        )
