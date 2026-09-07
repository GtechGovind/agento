"""Per-agent tool policy.

One MCP connection can be shared by several agents in a process, but each agent
may expose a different slice of it and gate different tools behind approval. So
the connection itself stays policy-free (a
:class:`~agento.core.tools.base.ToolSource`) and each agent wraps it in a
:class:`PolicyToolSet`, which is where enable / disable / preload / approval live.

The wrapper is also the **enforcement point**, not just a filter. A tool the
policy excluded is refused at :meth:`PolicyToolSet.call_tool` even if the model
somehow names it — which it can, through a deferred-tools ``call_tool`` wrapper
or a hallucinated name — so a disabled tool is genuinely unreachable rather than
merely hidden.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from ...errors import McpConnectionError
from ..messages import ApprovalDecision, InternalToolInfo
from .base import (
    ApprovalRequiredOutcome,
    AuthRequiredOutcome,
    ToolAnnotations,
    ToolListing,
    ToolListOutcome,
    ToolOutcome,
    ToolSchema,
    ToolSource,
    error_result,
)
from .selectors import (
    DEFAULT_DISABLE_TOOLS,
    DEFAULT_ENABLE_TOOLS,
    DEFAULT_PRELOAD_TOOLS,
    DEFAULT_REQUIRE_APPROVAL_FOR_TOOLS,
    is_tool_allowed,
    literal_names,
    matches_any_selector,
    selectors_include_all,
)

__all__ = ["PolicyToolSet", "ToolSelectors"]


class ToolSelectors:
    """The four selector lists that define one agent's view of a tool source.

    Args:
        enable: Tools the agent may use. Default ``["@all"]``.
        disable: Subtracted from ``enable``. Default none.
        preload: Which tools' schemas go into the system prompt when the source
            is not fully preloaded. Default none.
        require_approval: Which tools pause for a human. Default
            ``["@write", "@destructive"]``. Pass ``[]`` to disable approval
            entirely for this source — an explicit empty list is honoured, which
            is why the default is expressed as ``None`` rather than as ``[]``.
    """

    __slots__ = ("enable", "disable", "preload", "require_approval")

    def __init__(
        self,
        enable: Sequence[str] | None = None,
        disable: Sequence[str] | None = None,
        preload: Sequence[str] | None = None,
        require_approval: Sequence[str] | None = None,
    ) -> None:
        self.enable = list(enable if enable is not None else DEFAULT_ENABLE_TOOLS)
        self.disable = list(disable if disable is not None else DEFAULT_DISABLE_TOOLS)
        self.preload = list(preload if preload is not None else DEFAULT_PRELOAD_TOOLS)
        self.require_approval = list(
            require_approval if require_approval is not None else DEFAULT_REQUIRE_APPROVAL_FOR_TOOLS
        )

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return (
            f"ToolSelectors(enable={self.enable}, disable={self.disable}, "
            f"preload={self.preload}, require_approval={self.require_approval})"
        )


class PolicyToolSet:
    """One agent's policy-applied view of a :class:`ToolSource`.

    Args:
        source: The underlying provider of tools.
        selectors: This agent's policy.
        preload: When true every enabled tool is preloaded and ``selectors.preload``
            is irrelevant. When false only tools matching ``selectors.preload``
            are eager, and the rest are discovered through deferred tools.
    """

    def __init__(
        self,
        source: ToolSource,
        selectors: ToolSelectors | None = None,
        *,
        preload: bool = False,
    ) -> None:
        self._source = source
        self._selectors = selectors or ToolSelectors()
        self._preload = preload
        # Filled in by the first successful list_tools(). Before that, policy
        # questions are answered from the raw selectors.
        self._resolved_allowed: list[str] | None = None
        self._annotations: dict[str, ToolAnnotations | None] = {}

    # -- identity ----------------------------------------------------------- #

    @property
    def name(self) -> str:
        return self._source.name

    @property
    def id(self) -> str:
        return self._source.id

    @property
    def description(self) -> str:
        return self._source.description

    @property
    def source(self) -> ToolSource:
        """The wrapped, policy-free source."""
        return self._source

    @property
    def preload(self) -> bool:
        return self._preload

    @property
    def has_preloaded_tools(self) -> bool:
        """Whether anything from this set reaches the system prompt.

        The runtime checks this before calling :meth:`list_tools` during setup.
        When it is false the call is skipped entirely — which avoids a network
        round trip, and more importantly avoids triggering an OAuth prompt for a
        server the agent may never touch this turn.
        """
        return self._preload or bool(self._selectors.preload)

    # -- listing ------------------------------------------------------------ #

    async def list_tools(self) -> ToolListOutcome:
        """List the tools this agent may use, with ``preload`` set per tool.

        Raises:
            McpConnectionError: A tool named literally in ``enable`` does not
                exist on the source. Failing loudly here is deliberate: silently
                ignoring a typo would quietly remove an ability the agent's
                author believed it had.
        """
        listing = await self._source.list_tools()
        if isinstance(listing, AuthRequiredOutcome):
            return listing

        available = {schema.name for schema in listing.tools}
        missing = [name for name in literal_names(self._selectors.enable) if name not in available]
        if missing:
            raise McpConnectionError(
                f"Tools not found on '{self.name}': {', '.join(missing)}",
                422,
            )

        filtered: list[ToolSchema] = []
        for schema in listing.tools:
            self._annotations[schema.name] = schema.annotations
            if not is_tool_allowed(
                schema.name, schema.annotations, self._selectors.enable, self._selectors.disable
            ):
                continue
            filtered.append(
                schema.model_copy(
                    update={"preload": self._is_preloaded(schema.name, schema.annotations)}
                )
            )

        self._resolved_allowed = [schema.name for schema in filtered]
        return ToolListing(tools=filtered, initialized=listing.initialized)

    def _is_preloaded(self, name: str, annotations: ToolAnnotations | None) -> bool:
        if self._preload:
            return True
        return matches_any_selector(name, annotations, self._selectors.preload)

    def allowed_tool_names(self) -> list[str] | None:
        """Names this agent may call, or ``None`` when unrestricted.

        After :meth:`list_tools` this is exact. Before it, ``@all`` with no
        disables is reported as unrestricted and anything else falls back to the
        literal names, which is the safe direction to be wrong in.
        """
        if self._resolved_allowed is not None:
            return list(self._resolved_allowed)
        if selectors_include_all(self._selectors.enable) and not self._selectors.disable:
            return None
        return literal_names(self._selectors.enable)

    # -- calling ------------------------------------------------------------ #

    def requires_approval(self, name: str, annotations: ToolAnnotations | None) -> bool:
        """Whether calling ``name`` needs a human decision first."""
        return matches_any_selector(name, annotations, self._selectors.require_approval)

    async def _annotations_for(self, name: str) -> ToolAnnotations | None:
        """Annotations for a tool, listing the source once if needed."""
        if name in self._annotations:
            return self._annotations[name]
        listing = await self._source.list_tools()
        if isinstance(listing, AuthRequiredOutcome):
            return None
        for schema in listing.tools:
            self._annotations.setdefault(schema.name, schema.annotations)
        return self._annotations.get(name)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        approval: ApprovalDecision | None = None,
    ) -> ToolOutcome:
        """Enforce policy, then delegate to the source.

        The order matters. Authorization is checked before anything else, so an
        unauthorized server reports that rather than a spurious "tool not
        allowed" caused by annotations we could not read. Then the allow-list.
        Then approval — and only then does the underlying tool actually run.
        """
        listing = await self._source.list_tools()
        if isinstance(listing, AuthRequiredOutcome):
            return listing
        for schema in listing.tools:
            self._annotations.setdefault(schema.name, schema.annotations)

        annotations = self._annotations.get(name)

        allowed = self.allowed_tool_names()
        if allowed is not None and name not in allowed:
            if not is_tool_allowed(name, annotations, self._selectors.enable, self._selectors.disable):
                raise McpConnectionError(
                    f"Tool '{name}' is not enabled on '{self.name}'",
                    403,
                )

        info = await self._source.tool_info(name, arguments)
        needs_approval = info.requires_approval or self.requires_approval(name, annotations)

        if needs_approval and approval is None:
            return ApprovalRequiredOutcome(
                tool_info=info.model_copy(update={"requires_approval": True})
            )

        if approval == "deny":
            return error_result(json.dumps({"error": "User denied this tool call."}))

        outcome = await self._source.call_tool(name, arguments, approval)

        # The listing above may have been the call that opened the connection;
        # carry that fact forward so the runtime still emits mcp.initialize.
        if listing.initialized is not None and hasattr(outcome, "initialized"):
            if getattr(outcome, "initialized", None) is None:
                return outcome.model_copy(update={"initialized": listing.initialized})
        return outcome

    async def tool_info(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        resolve_underlying: bool = False,
    ) -> InternalToolInfo:
        """Describe a tool, with this agent's approval decision applied."""
        info = await self._source.tool_info(name, arguments, resolve_underlying)
        annotations = await self._annotations_for(name)
        return info.model_copy(
            update={
                "requires_approval": info.requires_approval or self.requires_approval(name, annotations)
            }
        )

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"PolicyToolSet({self.name!r}, preload={self._preload})"
