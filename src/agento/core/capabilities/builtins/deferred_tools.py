"""Expose connector discovery through a small, policy-aware tool gateway.

The gateway publishes a fixed schema catalogue. Connector listings are read at
request time, and execution and approval metadata are delegated to the selected
source. Adding a source therefore does not preload its individual tool schemas.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from ...instructions import InstructionBuilder
from ...messages import ApprovalDecision, InternalToolInfo
from ...tools.base import (
    AuthRequiredOutcome,
    ToolListing,
    ToolOutcome,
    ToolSchema,
    ToolSet,
    error_result,
    text_result,
)
from ..base import Capability

__all__ = ["DeferredTools"]

LIST_TOOLS = "list_tools"
GET_TOOL_INFO = "get_tool_info"
CALL_TOOL = "call_tool"


class _ServerArgument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    server: str


class _ToolArgument(_ServerArgument):
    tool: str


_DESCRIPTIONS = {
    LIST_TOOLS: "Read the names of tools exposed by a connector. Request get_tool_info for a tool's schema.",
    GET_TOOL_INFO: "Inspect one connector tool's description and schemas before preparing its arguments.",
    CALL_TOOL: "Execute a connector tool using arguments matching its inspected schema.",
}


def _schema(operation: str) -> ToolSchema:
    fields: dict[str, Any] = {"server": {"type": "string", "description": "Connector identifier."}}
    required = ["server"]
    if operation != LIST_TOOLS:
        fields["tool"] = {"type": "string", "description": "Tool identifier returned by the connector."}
        required.append("tool")
    if operation == CALL_TOOL:
        fields["input"] = {
            "type": "object", "additionalProperties": True,
            "description": "Values for the target tool's arguments.",
        }
    return ToolSchema(
        name=operation,
        description=_DESCRIPTIONS[operation],
        input_schema={"type": "object", "properties": fields, "required": required, "additionalProperties": False},
    )


class _DiscoveryGateway:
    name = "deferred_tools"
    id = "deferred_tools"
    description = "Connector catalogue, schema lookup, and delegated execution."
    preload = True
    has_preloaded_tools = True

    def __init__(self, sources: Sequence[ToolSet]) -> None:
        self._connections = {source.name: source for source in sources}

    def allowed_tool_names(self) -> list[str]:
        return list(_DESCRIPTIONS)

    async def list_tools(self) -> ToolListing:
        return ToolListing(tools=[_schema(operation) for operation in _DESCRIPTIONS])

    def _destination(self, arguments: dict[str, Any]) -> tuple[ToolSet | None, str, dict[str, Any]]:
        source = self._connections.get(str(arguments.get("server")))
        payload = arguments.get("input")
        return source, str(arguments.get("tool")), payload if isinstance(payload, dict) else {}

    async def tool_info(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        resolve_underlying: bool = False,
    ) -> InternalToolInfo:
        if name == CALL_TOOL and arguments and resolve_underlying:
            source, target, payload = self._destination(arguments)
            if source is not None:
                info = await source.tool_info(target, payload, True)
                return info.model_copy(update={"is_deferred": True})
        return InternalToolInfo(
            kind="builtin", source_id=self.id, source_name=self.name,
            original_tool_name=name, is_deferred=name == CALL_TOOL,
        )

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        approval: ApprovalDecision | None = None,
    ) -> ToolOutcome:
        if name == CALL_TOOL:
            source, target, payload = self._destination(arguments)
            if source is not None:
                return await source.call_tool(target, payload, approval)
            return error_result(json.dumps({"error": f"Unknown server {arguments.get('server')!r}"}))
        if name not in _DESCRIPTIONS:
            return error_result(json.dumps({"error": f"Unknown tool: {name}"}))
        if approval == "deny":
            return error_result(json.dumps({"error": "User denied this tool call."}))

        model = _ServerArgument if name == LIST_TOOLS else _ToolArgument
        try:
            query = model.model_validate(arguments)
        except ValidationError as error:
            details = []
            for issue in error.errors():
                details.append({"field": ".".join(map(str, issue["loc"])), "problem": issue["msg"]})
            return error_result(json.dumps({"error": "Invalid arguments", "details": details}))

        source = self._connections.get(query.server)
        if source is None:
            return error_result(json.dumps({
                "error": f"Unknown server {query.server!r}",
                "available_servers": ", ".join(sorted(self._connections)) or "none",
            }))
        try:
            catalogue = await source.list_tools()
            if isinstance(catalogue, AuthRequiredOutcome):
                return catalogue
            names = [item.name for item in catalogue.tools]
            if name == LIST_TOOLS:
                return text_result(json.dumps({"server": query.server, "tools": names}), initialized=catalogue.initialized)
            target_name = str(arguments["tool"])
            by_name = {item.name: item for item in reversed(catalogue.tools)}
            if target_name not in by_name:
                return error_result(json.dumps({
                    "error": f"Server {query.server!r} has no tool {target_name!r}", "available_tools": names,
                }), initialized=catalogue.initialized)
            target_schema = by_name[target_name]
            return text_result(json.dumps(target_schema.model_dump(
                include={"name", "description", "input_schema", "output_schema"},
            )), initialized=catalogue.initialized)
        except Exception as error:
            return error_result(json.dumps({"error": f"{type(error).__name__}: {error}"}))


class DeferredTools(Capability):
    """Advertise deferred sources while keeping their execution policy intact."""

    name = "deferred_tools"

    def __init__(self, sources: Sequence[ToolSet]) -> None:
        self._catalogues = tuple(source for source in sources if not source.preload)
        self._gateway = _DiscoveryGateway(sources) if self._catalogues else None

    def tool_sets(self) -> Sequence[ToolSet]:
        return () if self._gateway is None else (self._gateway,)

    def build_instructions(self, builder: InstructionBuilder) -> None:
        if self._gateway is None:
            return
        section = builder.begin_section("deferred-tools")
        section.add_content(
            "Connector tool schemas are available on demand. Inspect a connector with list_tools(server), "
            "then request get_tool_info(server, tool) for the selected tool. Submit its arguments through "
            "call_tool(server, tool, input). Use the discovered names and schema; forwarded calls retain "
            "the connector's normal approval requirements."
        )
        section.add_content("\n".join(
            f"- {source.name}" + (f": {source.description[:240]}" if source.description else "")
            for source in self._catalogues
        ))
