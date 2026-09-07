"""``Agento`` — the object you construct once and keep.

Everything shared across agents lives here: how model names resolve to clients,
which MCP servers exist and how to reach them, where skills come from, where
oversized content goes, where sessions are stored, where traces go. An
:class:`~agento.session.agent.Agent` names things; ``Agento`` is what those names
mean.

The smallest useful application::

    import agento

    app = agento.Agento(llm=agento.LiteLLMProvider())
    agent = agento.Agent(model="openai/gpt-4o", instructions="Be concise.")

    print(await app.run(agent, "What is an agent harness?"))

A real one::

    app = agento.Agento(
        llm=agento.LiteLLMProvider(),
        mcp={
            "github": agento.MCPServerConfig(
                url="https://api.githubcopilot.com/mcp/",
                headers={"Authorization": f"Bearer {token}"},
                description="Issues, pull requests, repository files and CI status.",
            ),
        },
        skills=agento.FileSkillSource("./skills"),
        artifacts=agento.LocalArtifactStore("./.agento/artifacts"),
        store=agento.SQLSessionStore("sqlite+aiosqlite:///./agento.db"),
        agents=[support_agent, research_agent],
    )

    session = await app.sessions.create(agent=support_agent, metadata={"user_id": "u_42"})
    async for event in await session.stream("Why was invoice 8812 refunded?"):
        ...

Registering agents (``agents=[...]``) matters for durability: an agent's tools
are Python functions and cannot be stored, so a session reloaded in a new process
finds its live agent by name here.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..core.capabilities.base import Capability
from .agent import Agent, MCPServerConfig
from .resolver import ResourceResolver
from .sessions import Sessions
from .store.base import SessionStore

__all__ = ["Agento"]


class Agento:
    """Shared configuration and the entry point for running agents.

    Args:
        llm: How model names resolve to clients. An
            :class:`~agento.core.llm.base.LLM` (used for everything), a mapping
            of name → client, or a callable returning one —
            :class:`~agento.core.llm.litellm_client.LiteLLMProvider` is the usual
            choice.
        store: Where sessions and turns are kept. Defaults to an in-memory store,
            which is fine for scripts and tests and wrong for anything that must
            survive a restart.
        mcp: Configured MCP servers, by name. Agents reference these names.
        skills: Where skills come from — usually a
            :class:`~agento.skills.filesystem.FileSkillSource`.
        artifacts: Where oversized tool results and large uploads go. Without
            one, large results are truncated to a preview instead of stored.
        tracer: Where spans go. No-op by default.
        agents: Agents to register by name, so sessions loaded from storage can
            find their live definitions.
        capabilities: Extra capabilities added to every agent this runtime runs.
            The place to put a behaviour your whole application needs.
    """

    def __init__(
        self,
        *,
        llm: Any,
        store: SessionStore | None = None,
        mcp: Mapping[str, MCPServerConfig] | None = None,
        skills: Any = None,
        artifacts: Any = None,
        tracer: Any = None,
        agents: Iterable[Agent] = (),
        capabilities: Sequence[Capability] = (),
    ) -> None:
        from ..tracing import NOOP_TRACER
        from .store.memory import MemorySessionStore

        self.llm = llm
        self.store: SessionStore = store or MemorySessionStore()
        self.mcp = dict(mcp or {})
        self.skills = skills
        self.artifacts = artifacts
        self.tracer = tracer or NOOP_TRACER
        self.capabilities = list(capabilities)

        self._agents: dict[str, Agent] = {}
        for agent in agents:
            self.register_agent(agent)

        self.sessions = Sessions(self)
        """Create, load and list sessions."""

    # ------------------------------------------------------------------ #
    # Agents                                                             #
    # ------------------------------------------------------------------ #

    def register_agent(self, agent: Agent) -> Agent:
        """Register an agent by name so stored sessions can find it again.

        Raises:
            ValueError: The agent has no name. Registration is by name, so an
                unnamed agent has nothing to register under.
        """
        if not agent.name:
            raise ValueError(
                "Only named agents can be registered — Agent(name=..., ...). The name is how a "
                "session loaded from storage finds its live definition again."
            )
        self._agents[agent.name] = agent
        return agent

    def get_agent(self, name: str) -> Agent | None:
        """Look up a registered agent."""
        return self._agents.get(name)

    @property
    def agents(self) -> dict[str, Agent]:
        """Registered agents, by name."""
        return dict(self._agents)

    # ------------------------------------------------------------------ #
    # Running                                                            #
    # ------------------------------------------------------------------ #

    async def run(
        self,
        agent: Agent,
        input: Any = None,
        *,
        metadata: Mapping[str, str] | None = None,
        session_id: str | None = None,
    ) -> str:
        """Run one turn in a throwaway session and return the final text.

        The shortest path from an agent to an answer. For anything multi-turn,
        create a session and keep it.
        """
        session = await self.sessions.create(
            agent=agent, metadata=metadata, session_id=session_id
        )
        return await session.run(input)

    async def stream(
        self,
        agent: Agent,
        input: Any = None,
        *,
        metadata: Mapping[str, str] | None = None,
    ) -> Any:
        """Run one turn in a throwaway session and return its event stream."""
        session = await self.sessions.create(agent=agent, metadata=metadata)
        return await session.stream(input)

    def run_sync(self, agent: Agent, input: Any = None, **kwargs: Any) -> str:
        """Blocking version of :meth:`run`, for scripts and sync frameworks.

        Raises:
            RuntimeError: Called from inside a running event loop, where it would
                deadlock. Use ``await app.run(...)`` there.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(agent, input, **kwargs))
        raise RuntimeError(
            "run_sync() cannot be called from inside an event loop — it would deadlock. "
            "Use 'await app.run(...)' instead."
        )

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def build_resolver(
        self,
        *,
        session_id: str | None = None,
        turn_id: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> ResourceResolver:
        """Build the per-turn resolver.

        Override in a subclass to use your own resolver — a different model
        registry, custom tool sets, or capabilities computed per session.
        """
        return ResourceResolver(
            llm=self.llm,
            mcp=self.mcp,
            skills=self.skills,
            artifacts=self.artifacts,
            tracer=self.tracer,
            session_id=session_id,
            turn_id=turn_id,
            metadata=metadata,
            extra_capabilities=self.capabilities,
        )

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return (
            f"Agento(store={type(self.store).__name__}, mcp={sorted(self.mcp)}, "
            f"agents={sorted(self._agents)})"
        )
