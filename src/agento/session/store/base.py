"""Persistence — the contract, and the records that cross it.

agento keeps two things durable: the **event log** (what happened, append-only,
replayable) and the **turn snapshot** (enough state to resume the conversation
next turn). Everything else is derived.

The protocol is intentionally small — twelve methods — so that pointing agento at
your existing database is an afternoon rather than a project. Two implementations
ship: :class:`~agento.session.store.memory.MemorySessionStore` and
:class:`~agento.session.store.sql.SQLSessionStore` (SQLAlchemy: SQLite,
Postgres, MySQL).

Three invariants a store must uphold:

1. **Turn creation is atomic with the session tip.** Inserting a turn and
   advancing ``session.last_turn_id`` must be one unit. Two concurrent turns must
   not leave a tip pointing at a turn that does not exist.
2. **The first terminal state wins.** Once a turn is ``done``, ``cancelled`` or
   ``error``, later writes must be rejected with
   :class:`~agento.errors.TurnNotRunningError` carrying the state that won —
   which is how a cancellation is not overwritten by a late completion.
3. **Events order by id.** Event ids are monotonic ULIDs; ordering by id
   lexicographically is ordering by creation. Do not sort by timestamp.

Turns form a **tree**, not a list: ``previous_turn_id`` is a parent pointer, so
branching a conversation from an earlier point is just creating a turn whose
parent is not the tip. Listing walks the ancestor chain from a given turn, so
sibling branches stay invisible to each other.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ...core.events import TurnInput, TurnMetrics, TurnState, TurnStateRunning

__all__ = [
    "Page",
    "SessionEventItem",
    "SessionMetrics",
    "SessionRecord",
    "SessionStore",
    "TurnRecord",
    "TurnSnapshot",
    "utc_now",
]


def utc_now() -> datetime:
    """Timezone-aware current time. Naive datetimes are a source of subtle bugs."""
    return datetime.now(timezone.utc)


class SessionMetrics(BaseModel):
    """Rolled-up totals for a session."""

    total_turns: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0


class SessionRecord(BaseModel):
    """One conversation.

    Attributes:
        session_id: Unique id.
        agent: The agent definition, serialized. Live tools are absent by
            construction (see :class:`~agento.session.agent.Agent`), so a session
            reloaded in a fresh process needs its agent re-supplied to run.
        agent_name: The agent's name, for looking a live definition back up.
        title: Human-readable title. Set once, from the first user message,
            unless you set it yourself.
        external_id: Your own key for this conversation — a ticket id, a Slack
            thread. Unique when set, and the basis of
            :meth:`~agento.session.sessions.Sessions.get_or_create_by_external_id`.
        last_turn_id: The session tip. What ``previous_turn_id="auto"`` resolves
            to.
        metadata: Your identifiers. Exposed to every tool through
            :class:`~agento.core.tools.context.ToolContext`, which is the
            intended way to give tools a user or tenant.
        metrics: Rolled-up totals.
        custom: Anything else you want stored alongside.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    agent: dict[str, Any] = Field(default_factory=dict)
    agent_name: str | None = None
    title: str | None = None
    external_id: str | None = None
    last_turn_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, str] = Field(default_factory=dict)
    metrics: SessionMetrics = Field(default_factory=SessionMetrics)
    custom: dict[str, Any] = Field(default_factory=dict)


class TurnSnapshot(BaseModel):
    """Everything needed to rebuild a turn's threads.

    Attributes:
        threads: Thread id → :class:`~agento.core.runtime.agent_thread.ThreadSnapshot`.
            Always contains ``"main"``; sub-agent threads appear while they run
            and are removed when they finish.
        mcp_sessions: Server name → last observed session id for diagnostics.
            The current MCP adapter initializes a fresh session on reconnect.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    threads: dict[str, Any] = Field(default_factory=dict)
    mcp_sessions: dict[str, str] = Field(default_factory=dict)


class TurnRecord(BaseModel):
    """One exchange within a session.

    Attributes:
        turn_id: Unique id.
        session_id: The owning session.
        previous_turn_id: Parent turn. ``None`` for a root turn. Turns form a
            tree, so this is a parent pointer rather than a linked list.
        first_turn_id: The root of this branch, for cheap branch identification.
        ancestor_ids: Recent ancestors, newest last. May be truncated — a reader
            needing the full chain follows older turns' own ``ancestor_ids``.
        state: Running, done, cancelled or error.
        input: What started the turn.
        snapshot: Thread state.
        custom: Anything you want stored alongside.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    turn_id: str
    session_id: str
    previous_turn_id: str | None = None
    first_turn_id: str = ""
    ancestor_ids: list[str] = Field(default_factory=list)
    state: TurnState = Field(default_factory=TurnStateRunning)
    input: list[TurnInput] = Field(default_factory=list)
    snapshot: TurnSnapshot = Field(default_factory=TurnSnapshot)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    custom: dict[str, Any] = Field(default_factory=dict)

    @property
    def metrics(self) -> TurnMetrics | None:
        """This turn's metrics, once it has reached a terminal state."""
        return getattr(self.state, "metrics", None)


class SessionEventItem(BaseModel):
    """One row of a session-wide event feed."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    turn_id: str
    event: Any


class Page(BaseModel):
    """A page of results.

    ``next_cursor`` is opaque: pass it back unchanged to get the next page, and
    ``None`` means there are no more.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    items: list[Any] = Field(default_factory=list)
    next_cursor: str | None = None

    def __iter__(self) -> Any:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


@runtime_checkable
class SessionStore(Protocol):
    """Where sessions, turns and events are kept."""

    # -- sessions ----------------------------------------------------------- #

    async def create_session(self, record: SessionRecord) -> None:
        """Insert a session.

        Raises:
            SessionAlreadyExistsError: The id is taken.
            SessionExternalIdConflictError: The ``external_id`` is taken.
        """
        ...

    async def get_session(self, session_id: str) -> SessionRecord | None:
        """Fetch a session, or ``None``."""
        ...

    async def get_session_by_external_id(self, external_id: str) -> SessionRecord | None:
        """Fetch by your own key, or ``None``."""
        ...

    async def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        metadata: dict[str, str] | None = None,
        metrics: SessionMetrics | None = None,
        last_turn_id: str | None = None,
        agent: dict[str, Any] | None = None,
        custom: dict[str, Any] | None = None,
        set_title_if_absent: str | None = None,
    ) -> None:
        """Patch a session. Only the fields given are changed.

        Args:
            set_title_if_absent: Set the title only if there is none. First write
                wins, so a derived title never overwrites one you chose.

        Raises:
            SessionNotFoundError: No such session.
        """
        ...

    async def delete_session(self, session_id: str) -> None:
        """Delete a session and everything under it. Missing is not an error."""
        ...

    async def list_sessions(
        self, *, limit: int = 50, cursor: str | None = None
    ) -> Page:
        """List sessions, most recently updated first."""
        ...

    # -- turns -------------------------------------------------------------- #

    async def create_turn(
        self, record: TurnRecord, *, expected_tip: tuple[str | None] | None = None
    ) -> None:
        """Insert a turn and advance the session tip, atomically.

        ``expected_tip=(id,)`` compares the current tip; ``(None,)`` requires
        an empty session. ``None`` omits comparison for direct store callers.
        A running session tip is rejected even for a fresh root or branch.

        Raises:
            TurnAlreadyExistsError: The id is taken.
            PreviousTurnRunningError: The parent turn is still running.
            SessionNotFoundError: No such session.
        """
        ...

    async def get_turn(self, session_id: str, turn_id: str) -> TurnRecord | None:
        """Fetch a turn, or ``None``."""
        ...

    async def list_turns(
        self, session_id: str, *, limit: int = 50, cursor: str | None = None
    ) -> Page:
        """List a session's turns, newest first."""
        ...

    async def update_turn(
        self,
        session_id: str,
        turn_id: str,
        *,
        state: TurnState | None = None,
        snapshot: TurnSnapshot | None = None,
        custom: dict[str, Any] | None = None,
        events: Sequence[Any] = (),
    ) -> None:
        """Patch a turn.

        All supplied fields, events, and metric changes commit together or
        change nothing. Every mutation of an already-terminal turn must fail,
        including repeats of the same terminal state.

        Raises:
            TurnNotFoundError: No such turn.
            TurnNotRunningError: Already terminal. Carries the winning state.
        """
        ...

    # -- events ------------------------------------------------------------- #

    async def append_events(self, session_id: str, turn_id: str, events: Sequence[Any]) -> None:
        """Append events to a turn's log. Append-only; ids are ULIDs."""
        ...

    async def list_turn_events(
        self,
        session_id: str,
        turn_id: str,
        *,
        limit: int = 100,
        cursor: str | None = None,
        order: str = "asc",
    ) -> Page:
        """Read one turn's events, ordered by id."""
        ...

    async def list_session_events(
        self, session_id: str, *, limit: int = 100, cursor: str | None = None
    ) -> Page:
        """Read a session's events across turns, newest first.

        Follows the ancestor chain from the session tip, so a branched
        conversation shows only the active branch.
        """
        ...
