"""Tools the host application executes, not agento.

Some tools cannot run inside the harness by definition. Asking the user a
question is the obvious one; so is anything that needs the browser tab the user
is looking at, a native file picker, or a credential that must never leave the
client.

A client-side tool advertises a normal schema, so the model calls it like any
other. But the loop never executes it: it recognises ``is_client_side`` on the
tool info *before* dispatch, pauses the whole turn, and emits
:class:`~agento.core.events.ClientToolRequired`. Your application does the work
and starts the next turn with a :class:`~agento.core.events.ToolReply`, whose
content becomes the tool's result.

Defining one is the same as any other tool, minus a body::

    from agento.core.tools import ClientSideToolSet, tool

    @tool
    async def pick_file(prompt: str) -> str:
        \"\"\"Ask the user to choose a file from their computer.

        Args:
            prompt: What to show in the picker.
        \"\"\"
        raise NotImplementedError  # never called; the host answers this

    agent = agento.Agent(
        model="openai/gpt-4o",
        tools=[ClientSideToolSet("ui", [pick_file])],
    )

The body is unreachable, so ``raise NotImplementedError`` is the honest thing to
write there. agento never calls it — and if a bug ever led to a direct call, the
set returns a clear error rather than running your placeholder.
"""

from __future__ import annotations

from typing import Any

from ..messages import ApprovalDecision, InternalToolInfo
from .base import ClientSideRequiredOutcome, ToolOutcome
from .local import LocalToolSet

__all__ = ["ClientSideToolSet"]


class ClientSideToolSet(LocalToolSet):
    """A :class:`~agento.core.tools.local.LocalToolSet` whose tools the host runs.

    Identical to ``LocalToolSet`` in how it advertises tools; different in that
    every call returns :class:`~agento.core.tools.base.ClientSideRequiredOutcome`
    instead of a result.
    """

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        approval: ApprovalDecision | None = None,
    ) -> ToolOutcome:
        """Always defer to the host.

        In practice the loop pauses before ever reaching this, because
        ``is_client_side`` on the tool info is what drives the state machine.
        This is the safety net for the impossible case.
        """
        return ClientSideRequiredOutcome(tool_info=await self.tool_info(name, arguments))

    async def tool_info(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        resolve_underlying: bool = False,
    ) -> InternalToolInfo:
        info = await super().tool_info(name, arguments, resolve_underlying)
        return info.model_copy(update={"is_client_side": True})
