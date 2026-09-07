"""The agent definition — what an agent *is*, declaratively.

An :class:`Agent` is a plain pydantic model. It names a model, some instructions,
some tools, some connectors, some skills, and a handful of runtime switches. It
holds no connections, no clients and no state, which means you can build one at
import time, store it in your own database as JSON, load it back, and hand it to
:class:`~agento.session.agento.Agento` to run.

The two forms of "tool" are worth understanding, because they behave differently:

**Live objects** (``tools=[...]``) — decorated functions or tool sets. Attached
directly, no lookup, but not serializable. Excluded from ``model_dump()``, so an
agent stored and reloaded keeps everything *except* these, and they must be
re-supplied by the code that constructs the agent. That is fine for the normal
case where an agent is defined in your codebase.

**References** (``mcp_servers=[...]``, ``skills=[...]``) — names resolved at turn
time against what you configured on ``Agento``. Fully serializable, which is what
lets an agent definition live in a database and still reach a connector holding
live OAuth credentials.

::

    agent = agento.Agent(
        name="support",
        model="openai/gpt-4o",
        instructions="You help customers with billing questions. Be exact about amounts.",
        tools=[lookup_invoice, issue_refund],
        mcp_servers=[agento.MCPServerRef(name="stripe", enable_tools=["@read-only"])],
        skills=["refund-policy"],
    )
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..core.tools.selectors import (
    DEFAULT_DISABLE_TOOLS,
    DEFAULT_ENABLE_TOOLS,
    DEFAULT_PRELOAD_TOOLS,
    DEFAULT_REQUIRE_APPROVAL_FOR_TOOLS,
)

__all__ = [
    "Agent",
    "CompactionConfig",
    "GenerativeUIConfig",
    "LargeToolResponseConfig",
    "MCPServerConfig",
    "MCPServerRef",
    "RuntimeConfig",
    "SubAgentConfig",
]


class SubAgentConfig(BaseModel):
    """Delegation settings.

    Attributes:
        enabled: Whether the agent may spawn sub-agents.
        model_choices: Optional label → description map. When set, the agent
            picks a label per delegation and :class:`Agento` resolves it to a
            real model — how you let an agent send cheap work to a cheap model.
        models: Label → model name, used to resolve the above.
    """

    enabled: bool = True
    model_choices: dict[str, str] = Field(default_factory=dict)
    models: dict[str, str] = Field(default_factory=dict)


class CompactionConfig(BaseModel):
    """Context compaction settings.

    Attributes:
        enabled: Whether to compact at all. Turning this off means a long
            conversation eventually fails on context length.
        threshold_tokens: Explicit trigger. Defaults to 80% of the model's
            context window, or 50,000 tokens when that is unknown.
    """

    enabled: bool = True
    threshold_tokens: int | None = None


class LargeToolResponseConfig(BaseModel):
    """Offloading settings for oversized tool results."""

    enabled: bool = True
    individual_token_threshold: int = 6_000
    total_token_threshold: int = 10_000
    preview_chars: int = 400


class GenerativeUIConfig(BaseModel):
    """Generative UI settings.

    Attributes:
        enabled: Whether the agent may emit ``openui`` blocks.
        preload: Put the full specification in the system prompt (~4,000 tokens)
            rather than behind a tool call.
    """

    enabled: bool = False
    preload: bool = False


class RuntimeConfig(BaseModel):
    """Which harness behaviours are switched on.

    The defaults are the ones worth having: compaction and offloading protect the
    context window, sub-agents and questions make the agent more capable, and
    generative UI is off because most applications do not render it.

    Attributes:
        iteration_limit: Maximum model calls in one turn. The backstop against a
            loop that never terminates.
        current_datetime: Give the agent a clock.
        ask_user_questions: Let it ask the user for a decision mid-run.
        sub_agents: Delegation.
        compaction: Summarize a conversation that outgrows the window.
        large_tool_response: Offload oversized tool results.
        generative_ui: The openui language.
    """

    iteration_limit: int = 100
    current_datetime: bool = True
    ask_user_questions: bool = True
    sub_agents: SubAgentConfig = Field(default_factory=SubAgentConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    large_tool_response: LargeToolResponseConfig = Field(default_factory=LargeToolResponseConfig)
    generative_ui: GenerativeUIConfig = Field(default_factory=GenerativeUIConfig)


class MCPServerRef(BaseModel):
    """An agent's reference to a configured MCP server, with its policy.

    The server's URL and credentials live on :class:`Agento`; this is only the
    agent's view of it. That separation is what keeps agent definitions
    serializable and free of secrets.

    Attributes:
        name: The configured server's name.
        enable_tools: What the agent may use. Tags or literal names.
        disable_tools: Subtracted from ``enable_tools``.
        preload_tools: Tools whose schemas go in the prompt while the rest stay
            deferred. Only meaningful when ``preload`` is false.
        require_approval_for_tools: What pauses for a human. Pass ``[]`` to turn
            approval off for this server entirely.
        preload: Put every enabled tool's schema in the prompt. Defaults to
            false — see :mod:`agento.core.capabilities.builtins.deferred_tools`
            for why.
    """

    name: str
    enable_tools: list[str] = Field(default_factory=lambda: list(DEFAULT_ENABLE_TOOLS))
    disable_tools: list[str] = Field(default_factory=lambda: list(DEFAULT_DISABLE_TOOLS))
    preload_tools: list[str] = Field(default_factory=lambda: list(DEFAULT_PRELOAD_TOOLS))
    require_approval_for_tools: list[str] = Field(
        default_factory=lambda: list(DEFAULT_REQUIRE_APPROVAL_FOR_TOOLS)
    )
    preload: bool = False


class MCPServerConfig(BaseModel):
    """How to reach an MCP server. Configured once on :class:`Agento`.

    Attributes:
        url: The server's endpoint.
        headers: Static headers, or an async callable re-invoked per operation —
            use the callable form for tokens that expire, or to signal that
            authorization is required.
        description: Shown to the agent when this server's tools are deferred.
            With deferred loading this sentence is the only thing the agent knows
            before deciding whether to look inside, so it earns real thought.
        transport: ``"auto"``, ``"streamable-http"`` or ``"sse"``.
        request_timeout: Seconds for one operation.
        connect_timeout: Seconds to connect.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    url: str
    headers: dict[str, str] | Any | None = None
    description: str = ""
    transport: str = "auto"
    request_timeout: float = 60.0
    connect_timeout: float = 30.0


class Agent(BaseModel):
    """A complete agent definition.

    Attributes:
        name: Optional name, used in traces, tool context and sub-agent labels.
        model: Model identifier, resolved by whatever you configured as
            ``Agento(llm=...)`` — e.g. ``"openai/gpt-4o"`` for LiteLLM.
        instructions: The system prompt. The single biggest lever you have. Say
            what this agent is for and how it should behave; do not restate tool
            documentation, which agento already injects.
        messages: Messages injected at the start of every session, before user
            input. Useful for a standing brief that is not a persona.
        tools: Live tools — decorated functions or tool sets. Not serialized.
        mcp_servers: References to configured MCP servers.
        skills: Skill names to attach, resolved against the configured source.
        response_format: Structured-output spec, passed to the provider
            untouched.
        params: Provider parameters — ``temperature``, ``max_tokens``,
            ``reasoning_effort``.
        config: Which harness behaviours are on.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str | None = None
    model: str
    instructions: str | None = None
    messages: list[str] = Field(default_factory=list)
    # Live objects: excluded from serialization because a function has no JSON
    # form. An agent reloaded from storage must be given these again.
    tools: list[Any] = Field(default_factory=list, exclude=True)
    mcp_servers: list[MCPServerRef] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    response_format: dict[str, Any] | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    config: RuntimeConfig = Field(default_factory=RuntimeConfig)

    def with_(self, **changes: Any) -> Agent:
        """A copy with fields replaced.

        Handy for variants of one agent::

            careful = agent.with_(params={"temperature": 0})
            fast = agent.with_(model="openai/gpt-4o-mini")
        """
        return self.model_copy(update=changes)
