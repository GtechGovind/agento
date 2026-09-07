"""A durable session store on SQLAlchemy.

One implementation, three databases: SQLite for local use, Postgres for a
deployment, MySQL if that is what you have. Written against SQLAlchemy Core
rather than the ORM, so there is no session/identity-map machinery between agento
and the tables, and the SQL is easy to read.

::

    pip install "agento[sqlite]"      # or agento[postgres]

    store = agento.SQLSessionStore("sqlite+aiosqlite:///./agento.db")
    await store.create_tables()

    app = agento.Agento(llm=..., store=store)

Three tables:

``agento_sessions``
    One row per conversation.
``agento_turns``
    One row per turn, with its thread snapshot as JSON.
``agento_events``
    Append-only event log, primary-keyed by the event's ULID, which is also the
    ordering key — so reading a turn's history is an index scan.

**Atomicity.** ``create_turn`` inserts the turn and advances the session tip in
one transaction, so the tip can never point at a turn that does not exist.
``update_turn`` re-reads the row inside its transaction before writing a terminal
state, so the first terminal write wins and a late completion cannot overwrite a
cancellation.

**Snapshots are rewritten, not appended.** Each context change rewrites the
turn's snapshot JSON. Simple and correct, and fine at the scale an embedded agent
runs at; a deployment with very long turns and heavy concurrency may want a
store that appends messages instead, which the twelve-method protocol leaves you
free to write.

.. note::
   This is one of three adapters that could not be runtime-tested where agento
   was written (PyPI was unreachable). The SQL is ordinary and the contract is
   covered by the shared store test suite — run
   ``pytest tests/test_store_contract.py --sql`` against a real database once
   SQLAlchemy is installed.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import Any

from ...core.events import Event, TurnInput, TurnState
from ...errors import (
    InvalidPageTokenError,
    PreviousTurnRunningError,
    SessionAlreadyExistsError,
    SessionExternalIdConflictError,
    SessionNotFoundError,
    SessionStoreConflictError,
    TurnAlreadyExistsError,
    TurnNotFoundError,
    TurnNotRunningError,
)
from .base import (
    Page,
    SessionEventItem,
    SessionMetrics,
    SessionRecord,
    TurnRecord,
    TurnSnapshot,
    utc_now,
)

__all__ = ["SQLSessionStore"]


def _import_sqlalchemy() -> Any:
    try:
        import sqlalchemy as sa
        from sqlalchemy.ext.asyncio import create_async_engine
    except ImportError as exc:  # pragma: no cover - depends on install
        raise ImportError(
            "SQLSessionStore requires SQLAlchemy and an async driver. Install with:\n"
            '    pip install "agento[sqlite]"      # SQLite\n'
            '    pip install "agento[postgres]"    # PostgreSQL'
        ) from exc
    return sa, create_async_engine


def _encode(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")


def _decode(cursor: str | None) -> str | None:
    if not cursor:
        return None
    padding = "=" * (-len(cursor) % 4)
    try:
        return base64.urlsafe_b64decode(cursor + padding).decode()
    except Exception:
        raise InvalidPageTokenError(f"Malformed page cursor: {cursor!r}") from None


class SQLSessionStore:
    """Sessions, turns and events in a SQL database.

    Args:
        url: An async SQLAlchemy URL — ``sqlite+aiosqlite:///./agento.db``,
            ``postgresql+asyncpg://user:pass@host/db``.
        engine: An engine you already have. Overrides ``url``.
        table_prefix: Prefix for the three table names, if you share a schema.
        echo: Log SQL. Useful once, while working out what a query does.
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        engine: Any = None,
        table_prefix: str = "agento_",
        echo: bool = False,
    ) -> None:
        sa, create_async_engine = _import_sqlalchemy()
        self._sa = sa

        if engine is None:
            if url is None:
                raise ValueError("SQLSessionStore needs either a url or an engine.")
            engine = create_async_engine(url, echo=echo)
        self.engine = engine

        self._metadata = sa.MetaData()
        JSON = sa.JSON

        self.sessions = sa.Table(
            f"{table_prefix}sessions",
            self._metadata,
            sa.Column("session_id", sa.String(64), primary_key=True),
            sa.Column("agent", JSON, nullable=False, default=dict),
            sa.Column("agent_name", sa.String(255), nullable=True, index=True),
            sa.Column("title", sa.Text, nullable=True),
            sa.Column("external_id", sa.String(255), nullable=True, unique=True),
            sa.Column("last_turn_id", sa.String(64), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, index=True),
            sa.Column("meta", JSON, nullable=False, default=dict),
            sa.Column("metrics", JSON, nullable=False, default=dict),
            sa.Column("custom", JSON, nullable=False, default=dict),
        )

        self.turns = sa.Table(
            f"{table_prefix}turns",
            self._metadata,
            sa.Column("turn_id", sa.String(64), primary_key=True),
            sa.Column("session_id", sa.String(64), nullable=False, index=True),
            sa.Column("previous_turn_id", sa.String(64), nullable=True),
            sa.Column("first_turn_id", sa.String(64), nullable=True),
            sa.Column("ancestor_ids", JSON, nullable=False, default=list),
            sa.Column("state", JSON, nullable=False),
            sa.Column("input", JSON, nullable=False, default=list),
            sa.Column("snapshot", JSON, nullable=False, default=dict),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, index=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("custom", JSON, nullable=False, default=dict),
        )

        self.events = sa.Table(
            f"{table_prefix}events",
            self._metadata,
            # The ULID is the primary key and the ordering key at once, so
            # reading a turn's history in order is a single index scan.
            sa.Column("event_id", sa.String(64), primary_key=True),
            sa.Column("session_id", sa.String(64), nullable=False, index=True),
            sa.Column("turn_id", sa.String(64), nullable=False, index=True),
            sa.Column("payload", JSON, nullable=False),
        )

        from pydantic import TypeAdapter

        self._turn_state: TypeAdapter[TurnState] = TypeAdapter(TurnState)
        self._turn_input = TypeAdapter(list[TurnInput])
        self._event: TypeAdapter[Any] = TypeAdapter(Event)

    # ------------------------------------------------------------------ #
    # Schema                                                             #
    # ------------------------------------------------------------------ #

    async def create_tables(self) -> None:
        """Create the tables if they are missing.

        Fine to call at startup every time. For anything with a real deployment
        process, generate the DDL once with :meth:`schema_sql` and manage it with
        your own migrations instead.
        """
        async with self.engine.begin() as connection:
            await connection.run_sync(self._metadata.create_all)

    async def drop_tables(self) -> None:
        """Drop the tables. For tests."""
        async with self.engine.begin() as connection:
            await connection.run_sync(self._metadata.drop_all)

    def schema_sql(self, dialect: str = "postgresql") -> str:
        """The CREATE TABLE statements, for your migration tool."""
        from sqlalchemy.schema import CreateTable

        engine = self._sa.create_mock_engine(f"{dialect}://", lambda *a, **kw: None)
        return "\n\n".join(
            str(CreateTable(table).compile(dialect=engine.dialect)).strip() + ";"
            for table in self._metadata.sorted_tables
        )

    async def dispose(self) -> None:
        """Close the connection pool."""
        await self.engine.dispose()

    # ------------------------------------------------------------------ #
    # Row conversion                                                     #
    # ------------------------------------------------------------------ #

    def _to_session(self, row: Any) -> SessionRecord:
        return SessionRecord(
            session_id=row.session_id,
            agent=row.agent or {},
            agent_name=row.agent_name,
            title=row.title,
            external_id=row.external_id,
            last_turn_id=row.last_turn_id,
            created_at=row.created_at,
            updated_at=row.updated_at,
            metadata=row.meta or {},
            metrics=SessionMetrics.model_validate(row.metrics or {}),
            custom=row.custom or {},
        )

    def _to_turn(self, row: Any) -> TurnRecord:
        return TurnRecord(
            turn_id=row.turn_id,
            session_id=row.session_id,
            previous_turn_id=row.previous_turn_id,
            first_turn_id=row.first_turn_id or "",
            ancestor_ids=row.ancestor_ids or [],
            state=self._turn_state.validate_python(row.state),
            input=self._turn_input.validate_python(row.input or []),
            snapshot=TurnSnapshot.model_validate(row.snapshot or {}),
            created_at=row.created_at,
            updated_at=row.updated_at,
            custom=row.custom or {},
        )

    def _to_event(self, payload: Any) -> Any:
        """Rebuild a typed event, falling back to the raw dict.

        A payload written by a newer version of agento with an event type this
        one does not know should not break a history read — the caller gets the
        dict and can decide what to do with it.
        """
        try:
            return self._event.validate_python(payload)
        except Exception:
            return payload

    # ------------------------------------------------------------------ #
    # Sessions                                                           #
    # ------------------------------------------------------------------ #

    async def create_session(self, record: SessionRecord) -> None:
        try:
            sa = self._sa
            async with self.engine.begin() as connection:
                existing = await connection.execute(
                    sa.select(self.sessions.c.session_id).where(
                        self.sessions.c.session_id == record.session_id
                    )
                )
                if existing.first() is not None:
                    raise SessionAlreadyExistsError(record.session_id)

                if record.external_id is not None:
                    clash = await connection.execute(
                        sa.select(self.sessions.c.session_id).where(
                            self.sessions.c.external_id == record.external_id
                        )
                    )
                    if clash.first() is not None:
                        raise SessionExternalIdConflictError(record.external_id)

                await connection.execute(
                    self.sessions.insert().values(
                        session_id=record.session_id,
                        agent=record.agent,
                        agent_name=record.agent_name,
                        title=record.title,
                        external_id=record.external_id,
                        last_turn_id=record.last_turn_id,
                        created_at=record.created_at,
                        updated_at=record.updated_at,
                        meta=record.metadata,
                        metrics=record.metrics.model_dump(),
                        custom=record.custom,
                    )
                )
        except self._sa.exc.IntegrityError as exc:
            if await self.get_session(record.session_id) is not None:
                raise SessionAlreadyExistsError(record.session_id) from exc
            if record.external_id is not None and await self.get_session_by_external_id(record.external_id):
                raise SessionExternalIdConflictError(record.external_id) from exc
            raise

    async def get_session(self, session_id: str) -> SessionRecord | None:
        sa = self._sa
        async with self.engine.connect() as connection:
            result = await connection.execute(
                sa.select(self.sessions).where(self.sessions.c.session_id == session_id)
            )
            row = result.first()
        return self._to_session(row) if row else None

    async def get_session_by_external_id(self, external_id: str) -> SessionRecord | None:
        sa = self._sa
        async with self.engine.connect() as connection:
            result = await connection.execute(
                sa.select(self.sessions).where(self.sessions.c.external_id == external_id)
            )
            row = result.first()
        return self._to_session(row) if row else None

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
        sa = self._sa
        values: dict[str, Any] = {"updated_at": utc_now()}
        if title is not None:
            values["title"] = title
        if metadata is not None:
            values["meta"] = metadata
        if metrics is not None:
            values["metrics"] = metrics.model_dump()
        if last_turn_id is not None:
            values["last_turn_id"] = last_turn_id
        if agent is not None:
            values["agent"] = agent
        if custom is not None:
            values["custom"] = custom

        async with self.engine.begin() as connection:
            row = (
                await connection.execute(
                    sa.select(self.sessions.c.title).where(self.sessions.c.session_id == session_id)
                )
            ).first()
            if row is None:
                raise SessionNotFoundError(session_id)
            # First write wins: a derived title never displaces one you chose.
            if set_title_if_absent is not None and not row.title:
                values["title"] = set_title_if_absent

            await connection.execute(
                self.sessions.update()
                .where(self.sessions.c.session_id == session_id)
                .values(**values)
            )

    async def delete_session(self, session_id: str) -> None:
        async with self.engine.begin() as connection:
            await connection.execute(
                self.events.delete().where(self.events.c.session_id == session_id)
            )
            await connection.execute(
                self.turns.delete().where(self.turns.c.session_id == session_id)
            )
            await connection.execute(
                self.sessions.delete().where(self.sessions.c.session_id == session_id)
            )

    async def list_sessions(self, *, limit: int = 50, cursor: str | None = None) -> Page:
        sa = self._sa
        after = _decode(cursor)
        query = sa.select(self.sessions).order_by(
            self.sessions.c.updated_at.desc(), self.sessions.c.session_id.desc()
        )
        async with self.engine.connect() as connection:
            if after is not None:
                anchor = (
                    await connection.execute(
                        sa.select(self.sessions.c.updated_at).where(
                            self.sessions.c.session_id == after
                        )
                    )
                ).first()
                if anchor is not None:
                    query = query.where(
                        sa.tuple_(self.sessions.c.updated_at, self.sessions.c.session_id)
                        < sa.tuple_(anchor.updated_at, after)
                    )
            rows = (await connection.execute(query.limit(limit + 1))).fetchall()

        has_more = len(rows) > limit
        page = rows[:limit]
        return Page(
            items=[self._to_session(row) for row in page],
            next_cursor=_encode(page[-1].session_id) if has_more and page else None,
        )

    # ------------------------------------------------------------------ #
    # Turns                                                              #
    # ------------------------------------------------------------------ #

    async def create_turn(
        self, record: TurnRecord, *, expected_tip: tuple[str | None] | None = None
    ) -> None:
        sa = self._sa
        async with self.engine.begin() as connection:
            await self._lock_session(connection, record.session_id)
            session_row = (
                await connection.execute(
                    sa.select(self.sessions.c.metrics, self.sessions.c.last_turn_id).where(
                        self.sessions.c.session_id == record.session_id
                    )
                )
            ).first()
            if session_row is None:
                raise SessionNotFoundError(record.session_id)

            if expected_tip is not None and session_row.last_turn_id != expected_tip[0]:
                raise SessionStoreConflictError("Session tip changed; reload and retry")

            existing = (
                await connection.execute(
                    sa.select(self.turns.c.turn_id).where(self.turns.c.turn_id == record.turn_id)
                )
            ).first()
            if existing is not None:
                raise TurnAlreadyExistsError(record.turn_id)

            if session_row.last_turn_id is not None:
                active = (await connection.execute(sa.select(self.turns.c.state).where(
                    self.turns.c.turn_id == session_row.last_turn_id,
                    self.turns.c.session_id == record.session_id,
                ))).first()
                if active is not None and (active.state or {}).get("status") == "running":
                    raise PreviousTurnRunningError(session_row.last_turn_id)
            if record.previous_turn_id is not None:
                previous = (
                    await connection.execute(
                        sa.select(self.turns.c.state).where(
                            self.turns.c.turn_id == record.previous_turn_id,
                            self.turns.c.session_id == record.session_id
                        )
                    )
                ).first()
                if previous is not None and (previous.state or {}).get("status") == "running":
                    raise PreviousTurnRunningError(record.previous_turn_id)

            await connection.execute(
                self.turns.insert().values(
                    turn_id=record.turn_id,
                    session_id=record.session_id,
                    previous_turn_id=record.previous_turn_id,
                    first_turn_id=record.first_turn_id,
                    ancestor_ids=record.ancestor_ids,
                    state=record.state.model_dump(mode="json"),
                    input=self._turn_input.dump_python(record.input, mode="json"),
                    snapshot=record.snapshot.model_dump(mode="json"),
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                    custom=record.custom,
                )
            )

            # Same transaction as the insert: the tip can never point at a turn
            # that does not exist.
            metrics = dict(session_row.metrics or {})
            metrics["total_turns"] = int(metrics.get("total_turns", 0)) + 1
            await connection.execute(
                self.sessions.update()
                .where(self.sessions.c.session_id == record.session_id)
                .values(last_turn_id=record.turn_id, metrics=metrics, updated_at=utc_now())
            )

    async def get_turn(self, session_id: str, turn_id: str) -> TurnRecord | None:
        sa = self._sa
        async with self.engine.connect() as connection:
            row = (
                await connection.execute(
                    sa.select(self.turns).where(
                        self.turns.c.session_id == session_id, self.turns.c.turn_id == turn_id
                    )
                )
            ).first()
        return self._to_turn(row) if row else None

    async def list_turns(
        self, session_id: str, *, limit: int = 50, cursor: str | None = None
    ) -> Page:
        sa = self._sa
        after = _decode(cursor)
        query = (
            sa.select(self.turns)
            .where(self.turns.c.session_id == session_id)
            .order_by(self.turns.c.created_at.desc(), self.turns.c.turn_id.desc())
        )
        async with self.engine.connect() as connection:
            if after is not None:
                anchor = (
                    await connection.execute(
                        sa.select(self.turns.c.created_at).where(self.turns.c.turn_id == after)
                    )
                ).first()
                if anchor is not None:
                    query = query.where(
                        sa.tuple_(self.turns.c.created_at, self.turns.c.turn_id)
                        < sa.tuple_(anchor.created_at, after)
                    )
            rows = (await connection.execute(query.limit(limit + 1))).fetchall()

        has_more = len(rows) > limit
        page = rows[:limit]
        return Page(
            items=[self._to_turn(row) for row in page],
            next_cursor=_encode(page[-1].turn_id) if has_more and page else None,
        )

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
        sa = self._sa
        async with self.engine.begin() as connection:
            await self._lock_session(connection, session_id)
            row = (
                await connection.execute(
                    sa.select(self.turns.c.state).where(
                        self.turns.c.session_id == session_id, self.turns.c.turn_id == turn_id
                    )
                )
            ).first()
            if row is None:
                raise TurnNotFoundError(turn_id)

            values: dict[str, Any] = {"updated_at": utc_now()}

            current = row.state or {}
            if current.get("status", "running") != "running":
                raise TurnNotRunningError(turn_id, self._turn_state.validate_python(current))
            if state is not None:
                values["state"] = state.model_dump(mode="json")

            if snapshot is not None:
                values["snapshot"] = snapshot.model_dump(mode="json")
            if custom is not None:
                values["custom"] = custom

            await connection.execute(
                self.turns.update()
                .where(self.turns.c.session_id == session_id, self.turns.c.turn_id == turn_id)
                .values(**values)
            )

            if state is not None and state.status != "running":
                await self._fold_metrics(connection, session_id, state)

            if events:
                await connection.execute(self.events.insert(), self._event_rows(session_id, turn_id, events))

    async def _lock_session(self, connection: Any, session_id: str) -> None:
        await connection.execute(
            self.sessions.update().where(self.sessions.c.session_id == session_id)
            .values(updated_at=self.sessions.c.updated_at)
        )

    async def _fold_metrics(self, connection: Any, session_id: str, state: TurnState) -> None:
        """Roll a finished turn's totals into its session."""
        sa = self._sa
        metrics = getattr(state, "metrics", None)
        if metrics is None:
            return
        row = (
            await connection.execute(
                sa.select(self.sessions.c.metrics).where(self.sessions.c.session_id == session_id)
            )
        ).first()
        if row is None:
            return
        totals = dict(row.metrics or {})
        totals["total_tokens"] = int(totals.get("total_tokens", 0)) + (metrics.total_tokens or 0)
        totals["total_cost_usd"] = float(totals.get("total_cost_usd", 0.0)) + (
            metrics.total_cost_usd or 0.0
        )
        await connection.execute(
            self.sessions.update()
            .where(self.sessions.c.session_id == session_id)
            .values(metrics=totals)
        )

    # ------------------------------------------------------------------ #
    # Events                                                             #
    # ------------------------------------------------------------------ #

    async def append_events(self, session_id: str, turn_id: str, events: Sequence[Any]) -> None:
        if not events:
            return
        async with self.engine.begin() as connection:
            exists = (await connection.execute(self._sa.select(self.turns.c.turn_id).where(
                self.turns.c.session_id == session_id, self.turns.c.turn_id == turn_id
            ))).first()
            if exists is None:
                raise TurnNotFoundError(turn_id)
            await connection.execute(self.events.insert(), self._event_rows(session_id, turn_id, events))

    @staticmethod
    def _event_rows(session_id: str, turn_id: str, events: Sequence[Any]) -> list[dict[str, Any]]:
        return [
            {
                "event_id": getattr(event, "id", None) or "",
                "session_id": session_id,
                "turn_id": turn_id,
                "payload": event.model_dump(mode="json")
                if hasattr(event, "model_dump")
                else dict(event),
            }
            for event in events
        ]

    async def list_turn_events(
        self,
        session_id: str,
        turn_id: str,
        *,
        limit: int = 100,
        cursor: str | None = None,
        order: str = "asc",
    ) -> Page:
        sa = self._sa
        after = _decode(cursor)
        ascending = order != "desc"
        column = self.events.c.event_id
        query = (
            sa.select(self.events)
            .where(self.events.c.session_id == session_id, self.events.c.turn_id == turn_id)
            .order_by(column.asc() if ascending else column.desc())
        )
        if after is not None:
            query = query.where(column > after if ascending else column < after)

        async with self.engine.connect() as connection:
            rows = (await connection.execute(query.limit(limit + 1))).fetchall()

        has_more = len(rows) > limit
        page = rows[:limit]
        return Page(
            items=[self._to_event(row.payload) for row in page],
            next_cursor=_encode(page[-1].event_id) if has_more and page else None,
        )

    async def list_session_events(
        self, session_id: str, *, limit: int = 100, cursor: str | None = None
    ) -> Page:
        sa = self._sa
        session = await self.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)

        # Walk the active branch from the tip, so a forked conversation shows
        # only the branch the session is on.
        chain: list[str] = []
        current = session.last_turn_id
        seen: set[str] = set()
        async with self.engine.connect() as connection:
            while current and current not in seen:
                seen.add(current)
                chain.append(current)
                row = (
                    await connection.execute(
                        sa.select(self.turns.c.previous_turn_id).where(
                            self.turns.c.turn_id == current
                        )
                    )
                ).first()
                current = row.previous_turn_id if row else None

            if not chain:
                return Page(items=[], next_cursor=None)

            position = {turn_id: index for index, turn_id in enumerate(chain)}
            query = (
                sa.select(self.events)
                .where(
                    self.events.c.session_id == session_id,
                    self.events.c.turn_id.in_(chain),
                )
                .order_by(self.events.c.event_id.desc())
            )
            after = _decode(cursor)
            if after is not None:
                query = query.where(self.events.c.event_id < after)

            rows = (await connection.execute(query.limit(limit + 1))).fetchall()

        # Newest turn first, then newest event within it.
        ordered = sorted(rows, key=lambda row: (position.get(row.turn_id, 0), _invert(row.event_id)))
        has_more = len(ordered) > limit
        page = ordered[:limit]
        return Page(
            items=[
                SessionEventItem(turn_id=row.turn_id, event=self._to_event(row.payload))
                for row in page
            ],
            next_cursor=_encode(page[-1].event_id) if has_more and page else None,
        )

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"SQLSessionStore({self.engine.url!r})"


def _invert(event_id: str) -> tuple[int, ...]:
    """Sort key that reverses a string's order, for descending event order."""
    return tuple(-ord(character) for character in event_id)
