"""Ambient context for a running tool.

A real application's tools usually need to know *who* they are running for — the
session, the tenant, the authenticated user — and threading that through the
model is impossible: the model chooses the arguments, and you do not want it
inventing a customer id.

agento solves this the way web frameworks solve "current request": a
:class:`ToolContext` is placed in a :class:`contextvars.ContextVar` for the
duration of each tool call. Two ways to read it:

**Ask for it by type.** Annotate a parameter and agento injects it, leaving it
out of the schema the model sees::

    @tool
    async def list_my_orders(status: str, ctx: ToolContext) -> str:
        user_id = ctx.metadata["user_id"]
        return json.dumps(await db.orders(user_id, status))

The model sees a one-argument tool. ``ctx`` is filled in by the runtime.

**Or read it anywhere.** Useful deeper in a call stack, where passing the context
down would be noise::

    from agento import current_tool_context

    async def audit(action: str) -> None:
        ctx = current_tool_context()
        log.info("action", action=action, session=ctx.session_id if ctx else None)

Because it is a context variable, concurrency is safe: parallel tool calls each
see their own context, including across ``await`` boundaries.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ToolContext", "current_tool_context", "use_tool_context"]


class ToolContext(BaseModel):
    """What agento knows about the call currently executing.

    Attributes:
        session_id: The session this call belongs to.
        turn_id: The turn.
        thread_id: The agent thread — ``"main"`` for the root agent, a generated
            id for a sub-agent. Useful for telling delegated work apart.
        tool_call_id: The specific call.
        tool_name: The tool's own name.
        agent_name: The agent's name, when it has one.
        metadata: The owning session's ``metadata`` dict. This is the intended
            place for your application's identifiers — put ``user_id`` or
            ``tenant`` on the session and every tool can read them here.
        artifacts: The configured artifact store, if any. A tool that produces
            something large can write it here and return a short reference
            instead of flooding the context window.
        approval: The human's decision, on a call that was approval-gated. Only
            set on the retry after approval.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    session_id: str | None = None
    turn_id: str | None = None
    thread_id: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    agent_name: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    artifacts: Any = None
    approval: Any = None


_current: contextvars.ContextVar[ToolContext | None] = contextvars.ContextVar(
    "agento_tool_context", default=None
)


def current_tool_context() -> ToolContext | None:
    """The context of the tool call on this task, or ``None`` outside one."""
    return _current.get()


@contextmanager
def use_tool_context(context: ToolContext | None) -> Iterator[None]:
    """Bind a context for the duration of a block.

    The runtime wraps every tool execution in this. Tests can use it to exercise
    a context-reading tool directly::

        with use_tool_context(ToolContext(session_id="s1")):
            await my_tool(status="open")
    """
    token = _current.set(context)
    try:
        yield
    finally:
        _current.reset(token)
