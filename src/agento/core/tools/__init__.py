"""Tools — everything the agent can call.

Four kinds, all behind one interface:

* :class:`~agento.core.tools.local.LocalToolSet` — Python functions, via
  :func:`~agento.core.tools.local.tool`. Start here.
* :class:`~agento.core.tools.client_side.ClientSideToolSet` — tools your
  application executes; the loop pauses and asks.
* :class:`~agento.core.tools.remote_mcp.RemoteMCP` — a remote MCP server.
* agento's own built-ins, in :mod:`agento.core.capabilities.builtins`.

Around any of them, :class:`~agento.core.tools.policy.PolicyToolSet` applies one
agent's enable/disable/preload/approval policy.

``RemoteMCP`` is imported lazily so the ``mcp`` package is only required if you
actually use one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import (
    ApprovalRequiredOutcome,
    AuthRequiredOutcome,
    ClientSideRequiredOutcome,
    CreateSubAgentOutcome,
    ToolAnnotations,
    ToolListing,
    ToolListOutcome,
    ToolOutcome,
    ToolSchema,
    ToolSet,
    ToolSource,
    ToolSuccess,
    error_result,
    text_result,
)
from .client_side import ClientSideToolSet
from .context import ToolContext, current_tool_context, use_tool_context
from .execute import ExecutionResult, SubAgentRequest, ToolCallResult, execute_tool_calls
from .local import LocalToolSet, Tool, function_tools, tool
from .policy import PolicyToolSet, ToolSelectors
from .registry import MappedTool, ToolRegistry, build_registry, sanitize_tool_name
from .selectors import (
    APPROVAL_TAGS,
    DEFAULT_DISABLE_TOOLS,
    DEFAULT_ENABLE_TOOLS,
    DEFAULT_PRELOAD_TOOLS,
    DEFAULT_REQUIRE_APPROVAL_FOR_TOOLS,
    SELECTION_TAGS,
    TAG_ALL,
    TAG_DESTRUCTIVE,
    TAG_READ_ONLY,
    TAG_WRITE,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .remote_mcp import RemoteMCP

__all__ = [
    "APPROVAL_TAGS",
    "ApprovalRequiredOutcome",
    "AuthRequiredOutcome",
    "ClientSideRequiredOutcome",
    "ClientSideToolSet",
    "CreateSubAgentOutcome",
    "DEFAULT_DISABLE_TOOLS",
    "DEFAULT_ENABLE_TOOLS",
    "DEFAULT_PRELOAD_TOOLS",
    "DEFAULT_REQUIRE_APPROVAL_FOR_TOOLS",
    "ExecutionResult",
    "LocalToolSet",
    "MappedTool",
    "PolicyToolSet",
    "RemoteMCP",
    "SELECTION_TAGS",
    "SubAgentRequest",
    "TAG_ALL",
    "TAG_DESTRUCTIVE",
    "TAG_READ_ONLY",
    "TAG_WRITE",
    "Tool",
    "ToolAnnotations",
    "ToolCallResult",
    "ToolContext",
    "ToolListOutcome",
    "ToolListing",
    "ToolOutcome",
    "ToolRegistry",
    "ToolSchema",
    "ToolSelectors",
    "ToolSet",
    "ToolSource",
    "ToolSuccess",
    "build_registry",
    "current_tool_context",
    "error_result",
    "execute_tool_calls",
    "function_tools",
    "sanitize_tool_name",
    "text_result",
    "tool",
    "use_tool_context",
]


def __getattr__(name: str) -> Any:
    """Import the MCP adapter on first use so ``mcp`` stays optional."""
    if name == "RemoteMCP":
        from .remote_mcp import RemoteMCP as _RemoteMCP

        globals()["RemoteMCP"] = _RemoteMCP
        return _RemoteMCP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
