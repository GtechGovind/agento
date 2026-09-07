"""Repairing a conversation left broken by an interrupted turn.

Providers enforce a structural rule: every tool call in an assistant message must
have a matching tool result. Break it and the request is rejected — not once, but
on every subsequent request, because the offending messages stay in the history.
A session in that state is permanently stuck.

A turn can end mid-execution for reasons nobody chose: the user hit stop, the
process was redeployed, a tool timed out. That leaves calls with no results.

:class:`ToolCallRepair` runs as a ``pre_send`` capability — before any new input
lands — and writes a short placeholder result for each dangling call. The model
reads it and is instructed to reconcile the unknown outcome before retrying.

It is careful about *which* calls it closes. Calls waiting on a human, on the host
application, or on a running sub-agent are legitimately open, and closing them
would discard work the user is in the middle of. Those are left alone; only calls
nothing is coming back for get repaired. See
:func:`~agento.core.runtime.context_utils.closable_open_tool_call_ids`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from ..capabilities.base import AppendContext, Capability, CapabilityOutput, ExecutionContext
from ..messages import LLMToolMessage
from .context_utils import closable_open_tool_call_ids, estimate_context_usage

__all__ = ["ToolCallRepair"]

_PLACEHOLDER = json.dumps(
    {"error": "Outcome unknown: the previous run ended without a durable tool result. "
     "The action may have completed. Reconcile its status before retrying; "
     "do not repeat side effects without an idempotency key or human confirmation."}
)


class ToolCallRepair(Capability):
    """Closes dangling tool calls before new input is accepted.

    Installed on every thread automatically, ahead of any capability you add —
    the conversation must be structurally valid before anything else inspects it.
    """

    name = "tool_call_repair"

    async def pre_send(self, context: ExecutionContext) -> AsyncIterator[CapabilityOutput]:
        dangling = closable_open_tool_call_ids(context.context)
        if not dangling:
            return

        messages = [
            LLMToolMessage(tool_call_id=call_id, content=_PLACEHOLDER) for call_id in sorted(dangling)
        ]
        yield AppendContext(
            messages=list(messages),
            usage=context.usage.merged_with(estimate_context_usage(messages)),
        )
