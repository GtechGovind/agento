"""The tool interface.

Everything the agent can call — a Python function you wrote, a tool on a remote
MCP server, ``ask_user_question``, ``create_sub_agent``, ``read_skill`` — reaches
the loop through the same small interface: a :class:`ToolSet` that can list its
tools and call one.

The part worth understanding is that **calling a tool does not always produce a
result.** Five things can come back, and the loop treats each differently:

``ToolSuccess``
    Normal case. The content goes into context as a tool message.

``ApprovalRequired``
    The tool is gated. The loop pauses the whole turn, emits
    :class:`~agento.core.events.ApprovalRequired`, and waits for a human. When
    the next turn supplies the decision, the same call is retried with it.

``ClientSideRequired``
    agento cannot run this one — the host application must. The loop pauses and
    emits :class:`~agento.core.events.ClientToolRequired`.

``CreateSubAgent``
    The call spawns a child thread instead of returning. The orchestrator builds
    the sub-agent and the parent's tool call stays open until the child finishes.

``AuthRequired``
    A remote MCP server needs authorization first. The turn ends with an
    ``mcp.auth_required`` action so the host can send the user through OAuth.

That union is why :meth:`ToolSet.call_tool` returns :data:`ToolOutcome` rather
than a string. Everything else in this module supports it.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ..events import AgentInfo, McpServerAuth, McpServerInit
from ..messages import ApprovalDecision, InternalToolInfo

__all__ = [
    "ApprovalRequiredOutcome",
    "AuthRequiredOutcome",
    "ClientSideRequiredOutcome",
    "CreateSubAgentOutcome",
    "ToolAnnotations",
    "ToolListing",
    "ToolListOutcome",
    "ToolOutcome",
    "ToolSchema",
    "ToolSet",
    "ToolSource",
    "ToolSuccess",
    "error_result",
    "text_result",
]


class ToolAnnotations(BaseModel):
    """Behavioural hints about a tool, following MCP's annotation vocabulary.

    These drive tool *selection* and *approval*: a policy can say "expose only
    read-only tools" or "require approval for anything destructive" without
    knowing a single tool name, which is what makes those policies portable
    across MCP servers you did not write.

    A tool with no annotations is treated as neither read-only nor destructive:
    it will not match ``@read-only`` and will not be auto-gated by
    ``@destructive``. Annotate your own tools; you get better defaults for free.
    """

    model_config = ConfigDict(extra="allow")

    title: str | None = None
    read_only: bool | None = None
    destructive: bool | None = None
    idempotent: bool | None = None
    open_world: bool | None = None

    @classmethod
    def from_mcp(cls, raw: Any) -> ToolAnnotations | None:
        """Build from an MCP server's camelCase annotation object."""
        if raw is None:
            return None
        get = raw.get if isinstance(raw, dict) else lambda key, default=None: getattr(raw, key, default)
        return cls(
            title=get("title"),
            read_only=get("readOnlyHint"),
            destructive=get("destructiveHint"),
            idempotent=get("idempotentHint"),
            open_world=get("openWorldHint"),
        )


class ToolSchema(BaseModel):
    """A tool as advertised to the model.

    Attributes:
        name: The tool's own name, before agento sanitizes or de-duplicates it
            for the model.
        description: What the tool does. This is the single biggest lever on
            whether the model uses it correctly.
        input_schema: JSON Schema for the arguments object.
        output_schema: JSON Schema for the result, when the source declares one.
            Optional, but valuable — the agent can inspect it before calling.
        annotations: Behavioural hints; see :class:`ToolAnnotations`.
        preload: Whether this tool's schema goes into the system prompt up front.
            Set by policy, not by the tool. ``False`` means the agent must
            discover it through the deferred-tools interface first.
    """

    model_config = ConfigDict(extra="allow")

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    output_schema: dict[str, Any] | None = None
    annotations: ToolAnnotations | None = None
    preload: bool = True


class ToolListing(BaseModel):
    """A successful ``list_tools``."""

    tools: list[ToolSchema] = Field(default_factory=list)
    initialized: McpServerInit | None = None
    """Set on the call that actually opened a connection, so the runtime can
    emit ``mcp.initialize`` and persist the session id for the next turn."""


class AuthRequiredOutcome(BaseModel):
    """The source needs authorization before it can be used."""

    kind: str = "auth_required"
    servers: list[McpServerAuth] = Field(default_factory=list)


ToolListOutcome = (ToolListing | AuthRequiredOutcome)
"""What ``list_tools`` can return."""


class ToolSuccess(BaseModel):
    """A tool ran and produced a result.

    Attributes:
        content: What the model will see. Already a string — structured results
            are JSON-encoded by the source.
        is_error: The tool ran but failed. The content still goes to the model,
            because a model that can see the error can usually fix its arguments
            and retry; hiding it just produces a confused retry loop.
        is_structured: The content is JSON rather than prose. Used to decide
            whether offloading a large result is worth it.
        initialized: Connection info, when this call opened the connection.
        events: Extra events the source wants surfaced on the stream.
    """

    model_config = ConfigDict(extra="allow")

    kind: str = "success"
    content: str
    is_error: bool = False
    is_structured: bool = False
    initialized: McpServerInit | None = None
    events: list[Any] = Field(default_factory=list)


class ApprovalRequiredOutcome(BaseModel):
    """The call is gated on a human decision."""

    kind: str = "approval_required"
    tool_info: InternalToolInfo


class ClientSideRequiredOutcome(BaseModel):
    """The host application must execute this call and supply the result."""

    kind: str = "client_side_required"
    tool_info: InternalToolInfo


class CreateSubAgentOutcome(BaseModel):
    """The call spawns a sub-agent rather than returning a value."""

    kind: str = "create_sub_agent"
    agent_info: AgentInfo


ToolOutcome = (ToolSuccess | ApprovalRequiredOutcome | ClientSideRequiredOutcome | CreateSubAgentOutcome | AuthRequiredOutcome)
"""Everything ``call_tool`` can return."""


def text_result(content: str, **kwargs: Any) -> ToolSuccess:
    """Shorthand for a successful text result."""
    return ToolSuccess(content=content, **kwargs)


def error_result(message: str, **kwargs: Any) -> ToolSuccess:
    """Shorthand for a failed call.

    Note this is still a :class:`ToolSuccess` — the *call* completed, the *tool*
    failed. The distinction matters: the model sees the message and can react.
    """
    return ToolSuccess(content=message, is_error=True, **kwargs)


@runtime_checkable
class ToolSource(Protocol):
    """A provider of tools, with no per-agent policy applied.

    One source can be shared by several agents in a process — a remote MCP
    connection, for instance — with each agent wrapping it in its own
    :class:`~agento.core.tools.policy.PolicyToolSet` to apply its own
    enable/disable/approval rules. That is why the policy lives in the wrapper
    rather than here.
    """

    @property
    def name(self) -> str:
        """Stable name. This is how an agent refers to the source, and what the
        model sees in ``mcp server: <name>`` on each tool description."""
        ...

    @property
    def id(self) -> str:
        """Stable identifier. Often equal to ``name``."""
        ...

    @property
    def description(self) -> str:
        """One line about what this source offers. Shown to the agent when the
        source's tools are deferred, so it can decide whether to look inside."""
        ...

    async def list_tools(self) -> ToolListOutcome:
        """List every tool this source offers, unfiltered."""
        ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        approval: ApprovalDecision | None = None,
    ) -> ToolOutcome:
        """Execute one tool.

        Args:
            name: The tool's own name.
            arguments: Parsed arguments.
            approval: A human's decision, present only when this call is a retry
                after an approval pause.
        """
        ...

    async def tool_info(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        resolve_underlying: bool = False,
    ) -> InternalToolInfo:
        """Describe a tool without calling it.

        Args:
            name: The tool's own name.
            arguments: Present when the caller has them, which lets a wrapper
                like deferred-tools look *through* itself to the real target.
            resolve_underlying: Ask wrappers to report the wrapped tool rather
                than themselves.
        """
        ...


@runtime_checkable
class ToolSet(ToolSource, Protocol):
    """A :class:`ToolSource` with one agent's policy applied.

    Attributes:
        preload: Every tool from this set goes into the system prompt.
        has_preloaded_tools: At least one tool is preloaded. When false the
            runtime skips calling ``list_tools`` during setup entirely, which
            matters because that call can be a network round trip and, for an
            OAuth-protected server, can trigger an authorization prompt the user
            never needed.
    """

    @property
    def preload(self) -> bool: ...

    @property
    def has_preloaded_tools(self) -> bool: ...

    def allowed_tool_names(self) -> list[str] | None:
        """Names this agent may call, or ``None`` for unrestricted.

        Advisory — :meth:`call_tool` enforces the real policy. This exists so
        components that need an allow-list up front can have one.
        """
        ...
