"""Turning an agent definition into a running thread.

An :class:`~agento.session.agent.Agent` is declarative: a model *name*, connector
*names*, skill *names*. Something has to turn those into a model client, live MCP
connections and loaded skill metadata, and that something is the resolver.

It exists as its own object, once per turn, for three reasons:

* **Connections are shared within a turn.** An agent and its five sub-agents that
  all reference ``"github"`` get one connection, not six — the resolver caches by
  name, and the cache lives exactly as long as the turn.
* **Cleanup has an owner.** :meth:`ResourceResolver.aclose` closes what it
  opened. The turn handler calls it in a ``finally``, so a turn that fails
  halfway never leaks a connection.
* **Overriding is easy.** Subclass it to resolve models from your own registry,
  build tool sets differently, or add capabilities of your own to every agent.

It is also where an agent's ``config`` becomes a list of capabilities — the one
place that knows ``compaction.enabled`` means "attach
:class:`~agento.core.capabilities.builtins.compaction.ContextCompaction`".
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from ..core.capabilities.base import Capability
from ..core.capabilities.builtins import (
    AskUserQuestion,
    ContextCompaction,
    CurrentDateTime,
    DeferredTools,
    GenerativeUI,
    LargeToolResponse,
    Skills,
    SubAgents,
)
from ..core.events import AgentInfo, ThreadParent
from ..core.messages import LLMUserMessage
from ..core.runtime.agent_thread import AgentDefinition, AgentThread
from ..core.tools.base import ToolSet
from ..core.tools.local import function_tools
from ..core.tools.policy import PolicyToolSet, ToolSelectors
from ..errors import ConfigurationError
from .agent import Agent, MCPServerConfig, MCPServerRef

__all__ = ["LLMResolver", "ResourceResolver"]

LLMResolver = Any
"""What ``Agento(llm=...)`` accepts.

* an :class:`~agento.core.llm.base.LLM` — used for every model name;
* a mapping of model name → ``LLM``;
* a callable ``(model_name) -> LLM`` (optionally async) — how the provider
  factories work.
"""


class ResourceResolver:
    """Builds threads for one turn, and owns what it opens.

    Args:
        llm: How to resolve a model name. See :data:`LLMResolver`.
        mcp: Configured MCP servers, by name.
        skills: Where skills come from.
        artifacts: Where oversized content goes.
        tracer: Where spans go.
        session_id: Passed to threads and tool context.
        turn_id: Passed to threads and tool context.
        metadata: The session's metadata, exposed to tools.
        extra_capabilities: Added to every thread, after the built-ins. The
            supported way to extend every agent in an application at once.
    """

    def __init__(
        self,
        *,
        llm: LLMResolver,
        mcp: Mapping[str, MCPServerConfig] | None = None,
        skills: Any = None,
        artifacts: Any = None,
        tracer: Any = None,
        session_id: str | None = None,
        turn_id: str | None = None,
        metadata: Mapping[str, str] | None = None,
        extra_capabilities: Sequence[Capability] = (),
    ) -> None:
        from ..tracing import NOOP_TRACER

        self._llm = llm
        self._mcp = dict(mcp or {})
        self._skills = skills
        self._artifacts = artifacts
        self._tracer = tracer or NOOP_TRACER
        self._session_id = session_id
        self._turn_id = turn_id
        self._metadata = dict(metadata or {})
        self._extra_capabilities = list(extra_capabilities)

        self._llm_cache: dict[str, Any] = {}
        self._sources: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Models                                                             #
    # ------------------------------------------------------------------ #

    async def resolve_llm(self, model: str) -> Any:
        """Resolve a model name to a client, caching per turn.

        Raises:
            ConfigurationError: The name cannot be resolved.
        """
        if model in self._llm_cache:
            return self._llm_cache[model]

        resolver = self._llm
        client: Any

        if isinstance(resolver, Mapping):
            if model not in resolver:
                raise ConfigurationError(
                    f"No model client configured for {model!r}. Known models: "
                    f"{', '.join(sorted(resolver)) or 'none'}."
                )
            client = resolver[model]
        elif callable(resolver) and not hasattr(resolver, "stream"):
            client = resolver(model)
            if isinstance(client, Awaitable):
                client = await client
        elif hasattr(resolver, "stream"):
            # A single client for everything. Warn-free, but the agent's model
            # name is then only a label.
            client = resolver
        else:
            raise ConfigurationError(
                "Agento(llm=...) must be an LLM, a mapping of name to LLM, or a callable "
                f"returning one. Got {type(resolver).__name__}."
            )

        if client is None:
            raise ConfigurationError(f"Model {model!r} resolved to None.")
        self._llm_cache[model] = client
        return client

    # ------------------------------------------------------------------ #
    # Tools                                                              #
    # ------------------------------------------------------------------ #

    def _local_tool_sets(self, agent: Agent) -> list[ToolSet]:
        """Wrap the agent's live tools into policy-applied sets.

        Loose functions are bundled into one set. A tool set you built yourself
        is wrapped with default policy — unless it is already a
        :class:`~agento.core.tools.policy.PolicyToolSet`, in which case your
        policy is used as given. That exception matters: re-wrapping would
        silently reinstate the default approval rules over an explicit choice
        such as ``require_approval=[]``.
        """
        if not agent.tools:
            return []

        loose: list[Any] = []
        sets: list[ToolSet] = []
        for entry in agent.tools:
            if isinstance(entry, PolicyToolSet):
                sets.append(entry)
            elif hasattr(entry, "list_tools"):
                sets.append(PolicyToolSet(entry, ToolSelectors(), preload=True))
            else:
                loose.append(entry)

        if loose:
            sets.append(
                PolicyToolSet(
                    function_tools(*loose, name=agent.name or "tools"),
                    ToolSelectors(),
                    preload=True,
                )
            )
        return sets

    async def _mcp_tool_set(self, ref: MCPServerRef, *, resume: Mapping[str, str]) -> ToolSet:
        """Connect (or reuse) an MCP server and apply the agent's policy."""
        config = self._mcp.get(ref.name)
        if config is None:
            raise ConfigurationError(
                f"Agent references MCP server {ref.name!r}, which is not configured. "
                f"Configured servers: {', '.join(sorted(self._mcp)) or 'none'}."
            )

        source = self._sources.get(ref.name)
        if source is None:
            from ..core.tools.remote_mcp import RemoteMCP

            source = RemoteMCP(
                ref.name,
                config.url,
                headers=config.headers,
                description=config.description,
                transport=config.transport,
                session_id=resume.get(ref.name),
                request_timeout=config.request_timeout,
                connect_timeout=config.connect_timeout,
            )
            self._sources[ref.name] = source

        return PolicyToolSet(
            source,
            ToolSelectors(
                enable=ref.enable_tools,
                disable=ref.disable_tools,
                preload=ref.preload_tools,
                require_approval=ref.require_approval_for_tools,
            ),
            preload=ref.preload,
        )

    def mcp_sessions(self) -> dict[str, str]:
        """Live MCP session ids for diagnostics; reconnect initializes a new session."""
        sessions: dict[str, str] = {}
        for name, source in self._sources.items():
            session_id = getattr(source, "session_id", None)
            if session_id:
                sessions[name] = session_id
        return sessions

    # ------------------------------------------------------------------ #
    # Definitions and capabilities                                       #
    # ------------------------------------------------------------------ #

    async def build_definition(
        self,
        agent: Agent,
        *,
        agent_info: AgentInfo | None = None,
        resume_mcp: Mapping[str, str] | None = None,
    ) -> AgentDefinition:
        """Resolve an agent into a runnable definition.

        Args:
            agent: The declarative agent.
            agent_info: Set when building a sub-agent. Its ``input`` becomes the
                thread's opening user message, and its ``model`` (if any) is
                resolved instead of the agent's own.
            resume_mcp: Server name → session id from the previous turn.
        """
        model = agent.model
        if agent_info is not None and agent_info.model:
            model = agent.config.sub_agents.models.get(agent_info.model, agent_info.model)

        client = await self.resolve_llm(model)

        tool_sets: list[ToolSet] = list(self._local_tool_sets(agent))
        for ref in agent.mcp_servers:
            tool_sets.append(await self._mcp_tool_set(ref, resume=resume_mcp or {}))

        if agent_info is not None:
            # A sub-agent gets its brief as a user message and no system prompt
            # of its own; its identity comes from SUB_AGENT_IDENTITY.
            initial = [LLMUserMessage(content=agent_info.input)]
            instructions = None
            response_format = None
        else:
            initial = [LLMUserMessage(content=text) for text in agent.messages]
            instructions = agent.instructions
            response_format = agent.response_format

        return AgentDefinition(
            llm=client,
            instructions=instructions,
            initial_messages=initial,
            params=dict(agent.params),
            response_format=response_format,
            iteration_limit=agent.config.iteration_limit,
            tool_sets=tool_sets,
            name=agent.name,
        )

    async def build_capabilities(
        self,
        agent: Agent,
        definition: AgentDefinition,
        *,
        is_sub_agent: bool = False,
    ) -> list[Capability]:
        """Turn an agent's ``config`` into capabilities.

        Interactive capabilities — asking the user, spawning further sub-agents —
        are withheld from sub-agents: a delegated thread has no user to ask, and
        letting children spawn children makes fan-out unbounded.
        """
        config = agent.config
        capabilities: list[Capability] = []

        if config.current_datetime:
            capabilities.append(CurrentDateTime())

        if config.compaction.enabled:
            capabilities.append(
                ContextCompaction(
                    definition.llm, threshold_tokens=config.compaction.threshold_tokens
                )
            )

        if config.ask_user_questions and not is_sub_agent:
            capabilities.append(AskUserQuestion())

        sub_agents_on = config.sub_agents.enabled and not is_sub_agent
        if sub_agents_on:
            capabilities.append(SubAgents(model_choices=config.sub_agents.model_choices))

        if config.large_tool_response.enabled:
            capabilities.append(
                LargeToolResponse(
                    individual_token_threshold=config.large_tool_response.individual_token_threshold,
                    total_token_threshold=config.large_tool_response.total_token_threshold,
                    preview_chars=config.large_tool_response.preview_chars,
                    sub_agents_available=sub_agents_on,
                )
            )

        if agent.skills:
            if self._skills is None:
                raise ConfigurationError(
                    f"Agent requests skills {agent.skills} but no skill source is configured. "
                    "Pass skills=... to Agento."
                )
            resolved = await self._skills.list_skills(agent.skills)
            found = {skill.name for skill in resolved}
            missing = [name for name in agent.skills if name not in found]
            if missing:
                raise ConfigurationError(
                    f"Unknown skill(s): {', '.join(missing)}. Available: "
                    f"{', '.join(sorted(found)) or 'none'}."
                )
            capabilities.append(Skills(self._skills, resolved))

        if config.generative_ui.enabled and not is_sub_agent:
            capabilities.append(GenerativeUI(preload=config.generative_ui.preload))

        # Deferred discovery goes last so it sees the final tool set list.
        deferred = DeferredTools(definition.tool_sets)
        if deferred.tool_sets():
            capabilities.append(deferred)

        capabilities.extend(self._extra_capabilities)
        return capabilities

    # ------------------------------------------------------------------ #
    # Threads                                                            #
    # ------------------------------------------------------------------ #

    async def build_thread(
        self,
        agent: Agent,
        *,
        thread_id: str = "main",
        title: str = "main",
        context: Sequence[Any] | None = None,
        usage: Any = None,
        parent: ThreadParent | None = None,
        agent_info: AgentInfo | None = None,
        completion: Any = None,
        capability_state: dict[str, Any] | None = None,
        resume_mcp: Mapping[str, str] | None = None,
    ) -> AgentThread:
        """Build one thread, ready to run or resume."""
        definition = await self.build_definition(
            agent, agent_info=agent_info, resume_mcp=resume_mcp
        )
        capabilities = await self.build_capabilities(
            agent, definition, is_sub_agent=parent is not None
        )
        return AgentThread(
            definition,
            thread_id=thread_id,
            title=title,
            capabilities=capabilities,
            context=context,
            usage=usage,
            parent=parent,
            agent_info=agent_info,
            completion=completion,
            capability_state=capability_state,
            session_id=self._session_id,
            turn_id=self._turn_id,
            metadata=self._metadata,
            artifacts=self._artifacts,
            tracer=self._tracer,
        )

    def sub_agent_factory(self, agent: Agent) -> Callable[..., Awaitable[AgentThread]]:
        """A factory the orchestrator uses to build children on demand."""

        async def build(
            *,
            parent_definition: AgentDefinition,
            request: AgentInfo,
            thread_id: str,
            parent: ThreadParent,
        ) -> AgentThread:
            thread = await self.build_thread(
                agent,
                thread_id=thread_id,
                title=request.name,
                parent=parent,
                agent_info=request,
            )
            # Children inherit the parent's live tool sets rather than
            # reconnecting: one connection per turn, and identical policy.
            thread.definition.tool_sets = list(parent_definition.tool_sets)
            return thread

        return build

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    async def aclose(self) -> None:
        """Release everything opened during the turn. Idempotent, never raises."""
        sources, self._sources = self._sources, {}
        for source in sources.values():
            close = getattr(source, "aclose", None)
            if close is None:
                continue
            try:
                await close()
            except Exception:  # pragma: no cover - best effort
                pass
