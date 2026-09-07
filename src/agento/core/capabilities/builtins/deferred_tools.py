"""Deferred tool loading — many connectors without the token bill.

Every tool an agent is given costs context on **every single model call**: name,
description, and a full JSON Schema. Three MCP connectors with thirty tools each
can spend twenty thousand tokens before the user has typed anything, and most of
those tools will not be used.

Deferred loading inverts it. A deferred server contributes one line to the prompt
— its name and description. When the agent decides it needs that server, it
discovers the tools itself:

1. ``list_tools(server)`` — what is available, names only.
2. ``get_tool_info(server, tool)`` — description and schemas for the one it wants.
3. ``call_tool(server, tool, input)`` — invoke it.

Three extra round trips, in exchange for a prompt that stays small no matter how
many connectors are attached. For a server the agent uses on most turns, preload
instead (``preload=True``); for the long tail, defer. That is why ``preload``
defaults to ``False``.

**Approval still applies.** ``call_tool`` looks *through* itself when asked what
it is about to run, so a destructive tool reached this way pauses for approval
exactly as it would if it had been preloaded. Deferring is a context optimization
and never a way around policy.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from ...instructions import InstructionBuilder
from ...messages import ApprovalDecision, InternalToolInfo
from ...tools.base import (
    AuthRequiredOutcome,
    ToolSet,
    error_result,
    text_result,
)
from ...tools.local import LocalToolSet, Tool
from ..base import Capability

__all__ = ["DeferredTools"]

LIST_TOOLS = "list_tools"
GET_TOOL_INFO = "get_tool_info"
CALL_TOOL = "call_tool"

_MAX_DESCRIPTION_CHARS = 240


class _DeferredToolSet(LocalToolSet):
    """The three discovery tools, wrapping the agent's deferred sources."""

    def __init__(self, sources: Sequence[ToolSet]) -> None:
        self._sources: dict[str, ToolSet] = {source.name: source for source in sources}
        super().__init__(
            "deferred_tools",
            [
                self._build_list_tools(),
                self._build_get_tool_info(),
                self._build_call_tool(),
            ],
            description="Discover and call tools on servers that are not preloaded.",
            kind="builtin",
        )

    # -- helpers ------------------------------------------------------------ #

    async def _tools_of(self, server: str) -> Any:
        source = self._sources.get(server)
        if source is None:
            known = ", ".join(sorted(self._sources)) or "none"
            return error_result(
                json.dumps({"error": f"Unknown server {server!r}", "available_servers": known})
            )
        listing = await source.list_tools()
        if isinstance(listing, AuthRequiredOutcome):
            return listing
        return listing.tools

    # -- tools -------------------------------------------------------------- #

    def _build_list_tools(self) -> Tool:
        async def list_tools(server: str) -> Any:
            """List the tool names available on a server.

            Returns names only. Use get_tool_info to see what a tool does and
            what arguments it takes before calling it.

            Args:
                server: The server to inspect.
            """
            tools = await self._tools_of(server)
            if not isinstance(tools, list):
                return tools
            return text_result(
                json.dumps({"server": server, "tools": [tool.name for tool in tools]})
            )

        return Tool(list_tools, name=LIST_TOOLS)

    def _build_get_tool_info(self) -> Tool:
        async def get_tool_info(server: str, tool: str) -> Any:
            """Get a tool's description and argument schema.

            Call this before calling a tool for the first time. Do not guess a
            tool's arguments from its name.

            Args:
                server: The server the tool is on.
                tool: The tool name, from list_tools.
            """
            tools = await self._tools_of(server)
            if not isinstance(tools, list):
                return tools
            found = next((item for item in tools if item.name == tool), None)
            if found is None:
                return error_result(
                    json.dumps(
                        {
                            "error": f"Server {server!r} has no tool {tool!r}",
                            "available_tools": [item.name for item in tools],
                        }
                    )
                )
            return text_result(
                json.dumps(
                    {
                        "name": found.name,
                        "description": found.description,
                        "input_schema": found.input_schema,
                        "output_schema": found.output_schema,
                    }
                )
            )

        return Tool(get_tool_info, name=GET_TOOL_INFO)

    def _build_call_tool(self) -> Tool:
        async def call_tool(
            server: str,
            tool: str,
            input: dict[str, Any] | None = None,
        ) -> Any:
            """Call a tool on a server.

            Call get_tool_info for this tool first, so the arguments match its
            schema.

            Args:
                server: The server the tool is on.
                tool: The tool name.
                input: Arguments for the tool, matching its input schema.
            """
            source = self._sources.get(server)
            if source is None:
                return error_result(json.dumps({"error": f"Unknown server {server!r}"}))
            return await source.call_tool(tool, input or {})

        return Tool(
            call_tool,
            name=CALL_TOOL,
            input_schema={
                "type": "object",
                "properties": {
                    "server": {"type": "string", "description": "The server the tool is on."},
                    "tool": {"type": "string", "description": "The tool name."},
                    "input": {
                        "type": "object",
                        "description": "Arguments matching the tool's input schema.",
                        "additionalProperties": True,
                    },
                },
                "required": ["server", "tool"],
                "additionalProperties": False,
            },
        )

    # -- policy pass-through ------------------------------------------------ #

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        approval: ApprovalDecision | None = None,
    ) -> Any:
        """Route ``call_tool`` with the approval decision attached.

        The generic path would swallow the decision, and the underlying tool set
        needs it — otherwise an approved destructive call would pause forever.
        """
        if name != CALL_TOOL:
            return await super().call_tool(name, arguments, approval)

        server = arguments.get("server")
        tool_name = arguments.get("tool")
        source = self._sources.get(str(server))
        if source is None:
            return error_result(json.dumps({"error": f"Unknown server {server!r}"}))
        payload = arguments.get("input") or {}
        return await source.call_tool(
            str(tool_name), payload if isinstance(payload, dict) else {}, approval
        )

    async def tool_info(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        resolve_underlying: bool = False,
    ) -> InternalToolInfo:
        """Report the *underlying* tool when asked about a ``call_tool`` call.

        This is what keeps approval honest through the wrapper. Without it every
        deferred call would look like a harmless call to ``call_tool`` and the
        destructive tool inside would run unchallenged.
        """
        base = await super().tool_info(name, arguments, resolve_underlying)
        if name != CALL_TOOL:
            return base
        if not resolve_underlying or not arguments:
            return base.model_copy(update={"is_deferred": True})

        source = self._sources.get(str(arguments.get("server")))
        if source is None:
            return base.model_copy(update={"is_deferred": True})

        payload = arguments.get("input") or {}
        underlying = await source.tool_info(
            str(arguments.get("tool")), payload if isinstance(payload, dict) else {}, True
        )
        return underlying.model_copy(update={"is_deferred": True})


class DeferredTools(Capability):
    """Adds the discovery tools for tool sets that are not preloaded.

    Args:
        sources: The agent's tool sets. Only those with ``preload=False`` are
            advertised as deferred; a fully preloaded set needs no discovery.
    """

    name = "deferred_tools"

    def __init__(self, sources: Sequence[ToolSet]) -> None:
        self._all = list(sources)
        self._deferred = [source for source in sources if not source.preload]
        self._tools = _DeferredToolSet(self._all) if self._deferred else None

    def tool_sets(self) -> Sequence[ToolSet]:
        return [self._tools] if self._tools is not None else []

    def build_instructions(self, builder: InstructionBuilder) -> None:
        if not self._deferred:
            return
        listing = "\n".join(
            f"- {source.name}: {source.description[:_MAX_DESCRIPTION_CHARS]}"
            if source.description
            else f"- {source.name}"
            for source in self._deferred
        )
        builder.add_section(
            "deferred-tools",
            "The tools on these servers are NOT loaded. Their schemas are not in this prompt:\n"
            f"{listing}\n\n"
            f"To use one: {LIST_TOOLS}(server) to see what is there, {GET_TOOL_INFO}(server, tool) "
            f"to learn its arguments, then {CALL_TOOL}(server, tool, input) to run it. "
            "Never guess a tool's name or arguments — check first.",
        )
