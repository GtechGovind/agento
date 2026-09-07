"""Executing a batch of tool calls.

A model can ask for several tools at once, and they should run at once — a turn
that fans out to five independent lookups should take as long as the slowest, not
the sum. So this module runs the batch with ``asyncio.gather`` and then sorts the
outcomes into the five buckets the loop cares about.

One invariant governs everything here:

    **Every tool call must produce a tool message.**

Providers reject a conversation containing an assistant message whose tool call
has no matching result — the next request would fail, permanently, for the rest
of the session. So even the failure paths synthesize a message: an unknown tool
name, a source that needs authorization, a tool that raised. The only exceptions
are the calls that do not *resolve* at all — approval-gated, client-side and
sub-agent calls — which stay deliberately open because something else is going to
answer them.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from ..events import AgentInfo, McpServerAuth, McpServerInit, now_iso
from ..messages import ApprovalDecision, EnrichedToolCall, LLMToolMessage
from .base import (
    ApprovalRequiredOutcome,
    AuthRequiredOutcome,
    ClientSideRequiredOutcome,
    CreateSubAgentOutcome,
    ToolSuccess,
)
from .context import ToolContext, use_tool_context
from .registry import MappedTool, ToolRegistry

__all__ = ["ExecutionResult", "SubAgentRequest", "ToolCallResult", "execute_tool_calls"]

AUTH_PENDING_MESSAGE = (
    "This tool's server needs to be authorized before it can be used. "
    "Authorization has been requested from the user; try again once it is granted."
)


class SubAgentRequest(NamedTuple):
    """A call that asked for a sub-agent instead of returning a value."""

    tool_call_id: str
    agent_info: AgentInfo


class ToolCallResult(BaseModel):
    """One resolved tool call.

    Attributes:
        tool_call: The original call.
        message: The tool message to append to context. Capabilities may rewrite
            its ``content`` — that is how large-result offloading replaces a
            50,000-token payload with a preview and an artifact id.
        outcome: The raw outcome, before any rewriting.
        mapped: Which tool set handled it. ``None`` for an unknown tool name.
        completed_at: ISO-8601 completion time.
        artifact_id: Set by offloading when the full payload was stored.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    tool_call: EnrichedToolCall
    message: LLMToolMessage
    outcome: ToolSuccess
    mapped: Any = None
    completed_at: str = Field(default_factory=now_iso)
    artifact_id: str | None = None

    @property
    def failed(self) -> bool:
        """Whether the tool reported an error."""
        return self.outcome.is_error


class ExecutionResult(BaseModel):
    """Everything that came out of executing one batch.

    Attributes:
        results: Calls that resolved. One tool message each.
        sub_agents: Calls that asked for a sub-agent. Left open until the child
            thread finishes and reports back.
        approval_required: Calls that need a human. Left open.
        client_side: Calls the host must execute. Left open.
        auth_required: Servers needing authorization. These calls *do* get a
            placeholder message, so the conversation stays valid and the model is
            told to retry after authorization.
        initialized: Servers that connected during this batch.
        events: Extra events sources asked to surface.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    results: list[ToolCallResult] = Field(default_factory=list)
    sub_agents: list[SubAgentRequest] = Field(default_factory=list)
    approval_required: list[EnrichedToolCall] = Field(default_factory=list)
    client_side: list[EnrichedToolCall] = Field(default_factory=list)
    auth_required: list[McpServerAuth] = Field(default_factory=list)
    initialized: list[McpServerInit] = Field(default_factory=list)
    events: list[Any] = Field(default_factory=list)


def _parse_arguments(raw: str) -> dict[str, Any]:
    """Parse a tool call's JSON arguments, tolerantly.

    A model can emit malformed JSON. Returning ``{}`` lets the tool's own
    validation produce a helpful "field required" message the model can act on,
    which recovers far more reliably than a parse error would.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def execute_tool_calls(
    tool_calls: Sequence[EnrichedToolCall],
    registry: ToolRegistry,
    *,
    approvals: dict[str, ApprovalDecision] | None = None,
    base_context: ToolContext | None = None,
) -> ExecutionResult:
    """Run a batch of tool calls concurrently and classify the outcomes.

    Args:
        tool_calls: The calls to execute — normally the *open* calls from the
            last assistant message.
        registry: This turn's tool registry, for resolving exposed names.
        approvals: Decisions already made, keyed by tool call id. A call whose id
            appears here is executed with that decision instead of pausing again.
        base_context: Session/turn identity to expose to each tool through
            :func:`~agento.core.tools.context.current_tool_context`. Per-call
            fields are filled in for each call.

    Returns:
        An :class:`ExecutionResult`.
    """
    approvals = approvals or {}
    result = ExecutionResult()
    if not tool_calls:
        return result

    outcomes = await asyncio.gather(
        *(_execute_one(call, registry, approvals, base_context) for call in tool_calls)
    )

    for call, mapped, outcome in outcomes:
        if isinstance(outcome, CreateSubAgentOutcome):
            result.sub_agents.append(SubAgentRequest(call.id, outcome.agent_info))
            continue

        if isinstance(outcome, ApprovalRequiredOutcome):
            result.approval_required.append(call)
            continue

        if isinstance(outcome, ClientSideRequiredOutcome):
            result.client_side.append(call)
            continue

        if isinstance(outcome, AuthRequiredOutcome):
            result.auth_required.extend(outcome.servers)
            # Placeholder result: the conversation must stay well-formed even
            # though the real call never happened.
            result.results.append(
                ToolCallResult(
                    tool_call=call,
                    message=LLMToolMessage(tool_call_id=call.id, content=AUTH_PENDING_MESSAGE),
                    outcome=ToolSuccess(content=AUTH_PENDING_MESSAGE, is_error=True),
                    mapped=mapped,
                )
            )
            continue

        if outcome.initialized is not None:
            result.initialized.append(outcome.initialized)
        if outcome.events:
            result.events.extend(outcome.events)

        result.results.append(
            ToolCallResult(
                tool_call=call,
                message=LLMToolMessage(tool_call_id=call.id, content=outcome.content),
                outcome=outcome,
                mapped=mapped,
            )
        )

    return result


async def _execute_one(
    call: EnrichedToolCall,
    registry: ToolRegistry,
    approvals: dict[str, ApprovalDecision],
    base_context: ToolContext | None,
) -> tuple[EnrichedToolCall, MappedTool | None, Any]:
    """Execute a single call, converting any exception into an error result."""
    mapped = registry.resolve(call.function.name)
    if mapped is None:
        return (
            call,
            None,
            ToolSuccess(
                content=json.dumps(
                    {
                        "error": f"Unknown tool: {call.function.name}",
                        "hint": "This tool is not available. Use one of the tools listed above.",
                    }
                ),
                is_error=True,
            ),
        )

    decision = approvals.get(call.id)
    context = (base_context.model_copy() if base_context else ToolContext()).model_copy(
        update={
            "tool_call_id": call.id,
            "tool_name": mapped.original_name,
            "approval": decision,
        }
    )

    try:
        with use_tool_context(context):
            outcome = await mapped.tool_set.call_tool(
                mapped.original_name, _parse_arguments(call.function.arguments), decision
            )
    except Exception as exc:
        # A tool set raising is a bug or an outage, not a model mistake — but it
        # still must not kill the turn. Report it to the model and let it adapt.
        outcome = ToolSuccess(
            content=json.dumps({"error": f"Tool execution failed: {type(exc).__name__}: {exc}"}),
            is_error=True,
        )

    return call, mapped, outcome
