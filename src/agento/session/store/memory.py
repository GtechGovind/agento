"""An in-process session store.

The reference implementation, and the one every contract test runs against. Fast,
dependency-free, and gone when the process exits.

Use it for tests, notebooks, single-run scripts, and any agent whose
conversations do not need to outlive the process. Use
:class:`~agento.session.store.sql.SQLSessionStore` for anything else.

It is deliberately strict rather than forgiving — it raises the same conflicts a
real database would, on the same conditions. A store that quietly tolerated a
duplicate id or a double terminal write would let a bug pass here and fail in
production.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Sequence
from typing import Any

from ...core.events import TurnState
from ...errors import (
    PreviousTurnRunningError,
    SessionAlreadyExistsError,
    SessionExternalIdConflictError,
    SessionNotFoundError,
    SessionStoreConflictError,
    SessionStoreInvariantError,
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

__all__ = ["MemorySessionStore"]


def _encode_cursor(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")


def _decode_cursor(cursor: str | None) -> str | None:
    if not cursor:
        return None
    padding = "=" * (-len(cursor) % 4)
    try:
        return base64.urlsafe_b64decode(cursor + padding).decode()
    except Exception:
        from ...errors import InvalidPageTokenError

        raise InvalidPageTokenError(f"Malformed page cursor: {cursor!r}") from None


class MemorySessionStore:
    """Sessions, turns and events held in dictionaries.

    A single lock serializes writes, which is what makes create-turn atomic with
    the session tip — the same guarantee a SQL store gets from a transaction.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionRecord] = {}
        self._by_external: dict[str, str] = {}
        self._turns: dict[str, dict[str, TurnRecord]] = {}
        self._events: dict[str, dict[str, list[Any]]] = {}
        self._lock = asyncio.Lock()

    # -- sessions ----------------------------------------------------------- #

    async def create_session(self, record: SessionRecord) -> None:
        async with self._lock:
            if record.session_id in self._sessions:
                raise SessionAlreadyExistsError(record.session_id)
            if record.external_id is not None:
                if record.external_id in self._by_external:
                    raise SessionExternalIdConflictError(record.external_id)
                self._by_external[record.external_id] = record.session_id
            self._sessions[record.session_id] = record.model_copy(deep=True)
            self._turns[record.session_id] = {}
            self._events[record.session_id] = {}

    async def get_session(self, session_id: str) -> SessionRecord | None:
        record = self._sessions.get(session_id)
        return record.model_copy(deep=True) if record else None

    async def get_session_by_external_id(self, external_id: str) -> SessionRecord | None:
        session_id = self._by_external.get(external_id)
        return await self.get_session(session_id) if session_id else None

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
        async with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                raise SessionNotFoundError(session_id)
            if title is not None:
                record.title = title
            if set_title_if_absent is not None and not record.title:
                record.title = set_title_if_absent
            if metadata is not None:
                record.metadata = dict(metadata)
            if metrics is not None:
                record.metrics = metrics
            if last_turn_id is not None:
                record.last_turn_id = last_turn_id
            if agent is not None:
                record.agent = agent
            if custom is not None:
                record.custom = dict(custom)
            record.updated_at = utc_now()

    async def delete_session(self, session_id: str) -> None:
        async with self._lock:
            record = self._sessions.pop(session_id, None)
            if record is not None and record.external_id:
                self._by_external.pop(record.external_id, None)
            self._turns.pop(session_id, None)
            self._events.pop(session_id, None)

    async def list_sessions(self, *, limit: int = 50, cursor: str | None = None) -> Page:
        ordered = sorted(
            self._sessions.values(), key=lambda record: (record.updated_at, record.session_id), reverse=True
        )
        after = _decode_cursor(cursor)
        if after is not None:
            index = next(
                (i for i, record in enumerate(ordered) if record.session_id == after), len(ordered) - 1
            )
            ordered = ordered[index + 1 :]
        page = ordered[:limit]
        next_cursor = _encode_cursor(page[-1].session_id) if len(ordered) > limit and page else None
        return Page(items=[record.model_copy(deep=True) for record in page], next_cursor=next_cursor)

    # -- turns -------------------------------------------------------------- #

    async def create_turn(
        self, record: TurnRecord, *, expected_tip: tuple[str | None] | None = None
    ) -> None:
        async with self._lock:
            session = self._sessions.get(record.session_id)
            if session is None:
                raise SessionNotFoundError(record.session_id)
            if expected_tip is not None and session.last_turn_id != expected_tip[0]:
                raise SessionStoreConflictError("Session tip changed; reload and retry")
            turns = self._turns.setdefault(record.session_id, {})
            if record.turn_id in turns:
                raise TurnAlreadyExistsError(record.turn_id)

            active = turns.get(session.last_turn_id or "")
            if active is not None and active.state.status == "running":
                raise PreviousTurnRunningError(active.turn_id)
            if record.previous_turn_id is not None:
                previous = turns.get(record.previous_turn_id)
                if previous is not None and previous.state.status == "running":
                    raise PreviousTurnRunningError(record.previous_turn_id)

            turns[record.turn_id] = record.model_copy(deep=True)
            self._events.setdefault(record.session_id, {})[record.turn_id] = []

            # Atomic with the insert: the tip can never point at a turn that
            # does not exist.
            session.last_turn_id = record.turn_id
            session.metrics.total_turns += 1
            session.updated_at = utc_now()

    async def get_turn(self, session_id: str, turn_id: str) -> TurnRecord | None:
        record = self._turns.get(session_id, {}).get(turn_id)
        return record.model_copy(deep=True) if record else None

    async def list_turns(
        self, session_id: str, *, limit: int = 50, cursor: str | None = None
    ) -> Page:
        turns = sorted(
            self._turns.get(session_id, {}).values(),
            key=lambda record: (record.created_at, record.turn_id),
            reverse=True,
        )
        after = _decode_cursor(cursor)
        if after is not None:
            index = next((i for i, record in enumerate(turns) if record.turn_id == after), len(turns) - 1)
            turns = turns[index + 1 :]
        page = turns[:limit]
        next_cursor = _encode_cursor(page[-1].turn_id) if len(turns) > limit and page else None
        return Page(items=[record.model_copy(deep=True) for record in page], next_cursor=next_cursor)

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
        async with self._lock:
            record = self._turns.get(session_id, {}).get(turn_id)
            if record is None:
                raise TurnNotFoundError(turn_id)
            if record.state.status != "running":
                raise TurnNotRunningError(turn_id, record.state)
            log = self._events[session_id][turn_id]
            self._validate_events(events)
            if state is not None:
                record.state = state.model_copy(deep=True)
                if state.status != "running":
                    self._fold_metrics(session_id, state)
            if snapshot is not None:
                record.snapshot = snapshot.model_copy(deep=True)
            if custom is not None:
                record.custom = dict(custom)
            log.extend(event.model_copy(deep=True) for event in events)
            record.updated_at = utc_now()

    def _fold_metrics(self, session_id: str, state: TurnState) -> None:
        """Roll a finished turn's totals into the session."""
        session = self._sessions.get(session_id)
        metrics = getattr(state, "metrics", None)
        if session is None or metrics is None:
            return
        session.metrics.total_tokens += metrics.total_tokens
        session.metrics.total_cost_usd += metrics.total_cost_usd or 0.0

    # -- events ------------------------------------------------------------- #

    async def append_events(self, session_id: str, turn_id: str, events: Sequence[Any]) -> None:
        async with self._lock:
            if turn_id not in self._turns.get(session_id, {}):
                raise TurnNotFoundError(turn_id)
            self._validate_events(events)
            self._events[session_id][turn_id].extend(event.model_copy(deep=True) for event in events)

    def _validate_events(self, events: Sequence[Any]) -> None:
        ids = [event.id for event in events]
        existing = {event.id for turns in self._events.values() for log in turns.values() for event in log}
        if len(set(ids)) != len(ids) or existing.intersection(ids):
            raise SessionStoreInvariantError("Duplicate event ID")

    async def list_turn_events(
        self,
        session_id: str,
        turn_id: str,
        *,
        limit: int = 100,
        cursor: str | None = None,
        order: str = "asc",
    ) -> Page:
        events = list(self._events.get(session_id, {}).get(turn_id, []))
        events.sort(key=lambda event: getattr(event, "id", ""), reverse=order == "desc")

        after = _decode_cursor(cursor)
        if after is not None:
            index = next(
                (i for i, event in enumerate(events) if getattr(event, "id", "") == after),
                len(events) - 1,
            )
            events = events[index + 1 :]

        page = events[:limit]
        next_cursor = (
            _encode_cursor(getattr(page[-1], "id", "")) if len(events) > limit and page else None
        )
        return Page(items=[event.model_copy(deep=True) for event in page], next_cursor=next_cursor)

    async def list_session_events(
        self, session_id: str, *, limit: int = 100, cursor: str | None = None
    ) -> Page:
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)

        # Walk the active branch from the tip backwards, so a forked
        # conversation shows only the branch the session is actually on.
        chain: list[str] = []
        turns = self._turns.get(session_id, {})
        current = session.last_turn_id
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            chain.append(current)
            record = turns.get(current)
            current = record.previous_turn_id if record else None

        items: list[SessionEventItem] = []
        for turn_id in chain:
            events = sorted(
                self._events.get(session_id, {}).get(turn_id, []),
                key=lambda event: getattr(event, "id", ""),
                reverse=True,
            )
            items.extend(SessionEventItem(turn_id=turn_id, event=event.model_copy(deep=True)) for event in events)

        after = _decode_cursor(cursor)
        if after is not None:
            index = next(
                (i for i, item in enumerate(items) if getattr(item.event, "id", "") == after),
                len(items) - 1,
            )
            items = items[index + 1 :]

        page = items[:limit]
        next_cursor = (
            _encode_cursor(getattr(page[-1].event, "id", "")) if len(items) > limit and page else None
        )
        return Page(items=page, next_cursor=next_cursor)

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"MemorySessionStore({len(self._sessions)} sessions)"
