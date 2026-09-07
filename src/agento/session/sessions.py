"""Creating and loading sessions.

Reached as ``app.sessions``. Four operations, and the third is the interesting
one::

    session = await app.sessions.create(agent=agent, metadata={"user_id": "u_42"})
    session = await app.sessions.get(session_id)

    # Bind a conversation to something in your own system
    session, created = await app.sessions.get_or_create_by_external_id(
        "slack:C123:1699999999.123456", agent=agent
    )

``get_or_create_by_external_id`` is what makes agento fit an existing application.
A Slack thread, a support ticket, a document — you already have an identifier for
the conversation, and you want the agent's memory keyed to it without maintaining
a second mapping table. It is also race-safe: two events for the same thread
arriving at once produce one session, not two.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .._ids import new_id
from ..errors import SessionExternalIdConflictError
from .agent import Agent
from .session_handle import SessionHandle
from .store.base import Page, SessionRecord

__all__ = ["Sessions"]


class Sessions:
    """Session lifecycle, bound to one :class:`~agento.session.agento.Agento`."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime
        self._store = runtime.store

    async def create(
        self,
        *,
        agent: Agent,
        session_id: str | None = None,
        external_id: str | None = None,
        title: str | None = None,
        metadata: Mapping[str, str] | None = None,
        custom: Mapping[str, Any] | None = None,
    ) -> SessionHandle:
        """Start a new conversation.

        Args:
            agent: The agent that will run it. Stored in serialized form, and
                kept live on the handle so its Python tools stay attached.
            session_id: Supply your own id if you have one.
            external_id: Your key for this conversation. Unique when set.
            title: A title. Otherwise derived from the first user message.
            metadata: Your identifiers. Every tool can read these through
                :class:`~agento.core.tools.context.ToolContext` — the intended
                way to give tools a user or tenant without the model choosing it.
            custom: Anything else stored alongside.

        Returns:
            A :class:`~agento.session.session_handle.SessionHandle`.
        """
        record = SessionRecord(
            session_id=session_id or new_id(),
            agent={**agent.model_dump(mode="json"), "requires_local_tools": bool(agent.tools)},
            agent_name=agent.name,
            external_id=external_id,
            title=title,
            metadata=dict(metadata or {}),
            custom=dict(custom or {}),
        )
        await self._store.create_session(record)
        if agent.name:
            # Registering here means a later get() in this process finds the live
            # definition without the caller having to pass it again.
            self._runtime._agents.setdefault(agent.name, agent)  # noqa: SLF001
        return SessionHandle(
            store=self._store, record=record, agent=agent, runtime=self._runtime
        )

    async def get(self, session_id: str, *, agent: Agent | None = None) -> SessionHandle | None:
        """Load a session, or ``None`` if there is no such session.

        Args:
            session_id: The session to load.
            agent: The live agent, if it is not registered on the runtime. A
                session's stored agent has no Python tools — they cannot be
                serialized — so running a turn needs the live definition from
                somewhere.
        """
        record = await self._store.get_session(session_id)
        if record is None:
            return None
        return self._bind(record, agent)

    async def get_by_external_id(
        self, external_id: str, *, agent: Agent | None = None
    ) -> SessionHandle | None:
        """Load a session by your own key, or ``None``."""
        record = await self._store.get_session_by_external_id(external_id)
        if record is None:
            return None
        return self._bind(record, agent)

    async def get_or_create_by_external_id(
        self,
        external_id: str,
        *,
        agent: Agent,
        metadata: Mapping[str, str] | None = None,
        custom: Mapping[str, Any] | None = None,
    ) -> tuple[SessionHandle, bool]:
        """Get the session for a key, creating it if it does not exist.

        Race-safe: if another worker wins the create, this returns that session
        rather than failing — which is what you want when two events for the same
        Slack thread arrive together.

        Returns:
            ``(session, created)``.
        """
        existing = await self.get_by_external_id(external_id, agent=agent)
        if existing is not None:
            return existing, False

        try:
            session = await self.create(
                agent=agent, external_id=external_id, metadata=metadata, custom=custom
            )
            return session, True
        except SessionExternalIdConflictError:
            winner = await self.get_by_external_id(external_id, agent=agent)
            if winner is None:  # pragma: no cover - only if the winner was deleted
                raise
            return winner, False

    async def list(self, *, limit: int = 50, cursor: str | None = None) -> Page:
        """List sessions, most recently updated first."""
        return await self._store.list_sessions(limit=limit, cursor=cursor)

    async def delete(self, session_id: str) -> None:
        """Delete a session and everything under it."""
        await self._store.delete_session(session_id)

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _bind(self, record: SessionRecord, agent: Agent | None) -> SessionHandle:
        """Attach a live agent to a stored session.

        Preference order: the one passed in, then the runtime's registry by name,
        then a definition reconstructed from storage — which works for an agent
        whose tools are all MCP servers and skills, and fails clearly (when a
        turn is created) for one with Python tools.
        """
        live = agent
        if live is None and record.agent_name:
            live = self._runtime.get_agent(record.agent_name)
        if live is None and record.agent and not record.agent.get("requires_local_tools"):
            try:
                live = Agent.model_validate(record.agent)
            except Exception:
                live = None
        return SessionHandle(
            store=self._store, record=record, agent=live, runtime=self._runtime
        )
