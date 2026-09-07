"""The session store contract.

Every test here is written against the :class:`~agento.session.store.base.SessionStore`
protocol rather than any one implementation, so the same suite validates a store
you write yourself::

    from test_store import run_contract
    await run_contract(MyStore())

The SQL store is exercised through the same function when SQLAlchemy and
aiosqlite are installed, and skipped otherwise.
"""

from __future__ import annotations

from typing import Any

import agento
from agento.session.store.base import SessionRecord, TurnRecord, TurnSnapshot


async def _make_session(store: Any, **kwargs: Any) -> SessionRecord:
    record = SessionRecord(session_id=kwargs.pop("session_id", "s1"), **kwargs)
    await store.create_session(record)
    return record


async def run_contract(store: Any) -> None:
    """Run every store invariant against ``store``."""
    await _sessions_round_trip(store)
    await _external_id_is_unique(store)
    await _turns_advance_the_tip(store)
    await _terminal_state_is_write_once(store)
    await _previous_running_turn_is_rejected(store)
    await _events_order_by_id(store)
    await _session_feed_follows_the_branch(store)
    await _delete_removes_everything(store)


async def _sessions_round_trip(store: Any) -> None:
    await _make_session(store, session_id="rt1", title="First", metadata={"k": "v"})

    loaded = await store.get_session("rt1")
    assert loaded is not None
    assert loaded.title == "First"
    assert loaded.metadata == {"k": "v"}

    await store.update_session("rt1", title="Renamed")
    assert (await store.get_session("rt1")).title == "Renamed"

    # set_title_if_absent never overwrites.
    await store.update_session("rt1", set_title_if_absent="Ignored")
    assert (await store.get_session("rt1")).title == "Renamed"

    assert await store.get_session("missing") is None


async def _external_id_is_unique(store: Any) -> None:
    await _make_session(store, session_id="ex1", external_id="ticket-1")
    assert (await store.get_session_by_external_id("ticket-1")).session_id == "ex1"

    try:
        await _make_session(store, session_id="ex2", external_id="ticket-1")
    except agento.errors.SessionExternalIdConflictError:
        pass
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("A duplicate external_id must be rejected")


async def _turns_advance_the_tip(store: Any) -> None:
    await _make_session(store, session_id="t1")
    await store.create_turn(TurnRecord(turn_id="turn-1", session_id="t1", first_turn_id="turn-1"))

    session = await store.get_session("t1")
    assert session.last_turn_id == "turn-1"
    assert session.metrics.total_turns == 1

    turn = await store.get_turn("t1", "turn-1")
    assert turn is not None
    assert turn.state.status == "running"

    try:
        await store.create_turn(TurnRecord(turn_id="turn-1", session_id="t1"))
    except agento.errors.TurnAlreadyExistsError:
        pass
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("A duplicate turn id must be rejected")


async def _terminal_state_is_write_once(store: Any) -> None:
    await _make_session(store, session_id="t2")
    await store.create_turn(TurnRecord(turn_id="turn-2", session_id="t2"))

    await store.update_turn("t2", "turn-2", state=agento.TurnStateCancelled(reason="stopped"))

    try:
        await store.update_turn("t2", "turn-2", state=agento.TurnStateDone(output=None))
    except agento.errors.TurnNotRunningError as exc:
        # The winning state is reported back, so a caller can surface it.
        assert exc.state.status == "cancelled"
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("A late completion must not overwrite a cancellation")

    assert (await store.get_turn("t2", "turn-2")).state.status == "cancelled"


async def _previous_running_turn_is_rejected(store: Any) -> None:
    await _make_session(store, session_id="t3")
    await store.create_turn(TurnRecord(turn_id="a", session_id="t3"))

    try:
        await store.create_turn(TurnRecord(turn_id="b", session_id="t3", previous_turn_id="a"))
    except agento.errors.PreviousTurnRunningError:
        pass
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("Chaining from a running turn must be rejected")


async def _events_order_by_id(store: Any) -> None:
    await _make_session(store, session_id="e1")
    await store.create_turn(TurnRecord(turn_id="turn-e", session_id="e1"))

    events = [agento.ToolResult(tool_call_id=f"c{i}", content=str(i)) for i in range(5)]
    await store.append_events("e1", "turn-e", events)

    page = await store.list_turn_events("e1", "turn-e", limit=10)
    assert [e.content for e in page.items] == ["0", "1", "2", "3", "4"]

    descending = await store.list_turn_events("e1", "turn-e", limit=10, order="desc")
    assert [e.content for e in descending.items] == ["4", "3", "2", "1", "0"]

    first = await store.list_turn_events("e1", "turn-e", limit=2)
    assert [e.content for e in first.items] == ["0", "1"]
    assert first.next_cursor is not None
    second = await store.list_turn_events("e1", "turn-e", limit=2, cursor=first.next_cursor)
    assert [e.content for e in second.items] == ["2", "3"]


async def _session_feed_follows_the_branch(store: Any) -> None:
    await _make_session(store, session_id="b1")
    await store.create_turn(TurnRecord(turn_id="root", session_id="b1", first_turn_id="root"))
    await store.append_events("b1", "root", [agento.ToolResult(tool_call_id="r", content="root")])
    await store.update_turn("b1", "root", state=agento.TurnStateDone(output=None))

    await store.create_turn(
        TurnRecord(turn_id="branch-a", session_id="b1", previous_turn_id="root")
    )
    await store.append_events("b1", "branch-a", [agento.ToolResult(tool_call_id="a", content="a")])
    await store.update_turn("b1", "branch-a", state=agento.TurnStateDone(output=None))

    # A second branch from root becomes the tip; branch-a should drop out.
    await store.create_turn(
        TurnRecord(turn_id="branch-b", session_id="b1", previous_turn_id="root")
    )
    await store.append_events("b1", "branch-b", [agento.ToolResult(tool_call_id="b", content="b")])

    feed = await store.list_session_events("b1", limit=50)
    turn_ids = {item.turn_id for item in feed.items}
    assert turn_ids == {"root", "branch-b"}
    # Newest turn first.
    assert feed.items[0].turn_id == "branch-b"


async def _delete_removes_everything(store: Any) -> None:
    await _make_session(store, session_id="d1")
    await store.create_turn(TurnRecord(turn_id="turn-d", session_id="d1"))
    await store.append_events("d1", "turn-d", [agento.ToolResult(tool_call_id="c", content="x")])

    await store.delete_session("d1")

    assert await store.get_session("d1") is None
    assert await store.get_turn("d1", "turn-d") is None
    # Deleting again is a no-op, not an error.
    await store.delete_session("d1")


# --------------------------------------------------------------------------- #
# Implementations                                                              #
# --------------------------------------------------------------------------- #


async def test_memory_store_contract() -> None:
    await run_contract(agento.MemorySessionStore())


async def test_snapshot_round_trips() -> None:
    store = agento.MemorySessionStore()
    await _make_session(store, session_id="snap")
    await store.create_turn(TurnRecord(turn_id="t", session_id="snap"))

    snapshot = TurnSnapshot(
        threads={"main": {"thread_id": "main", "context": [], "capability_state": {"k": 1}}},
        mcp_sessions={"github": "sess-1"},
    )
    await store.update_turn("snap", "t", snapshot=snapshot)

    loaded = await store.get_turn("snap", "t")
    assert loaded.snapshot.mcp_sessions == {"github": "sess-1"}
    assert loaded.snapshot.threads["main"]["capability_state"] == {"k": 1}


async def test_sql_store_contract() -> None:
    """The same contract against SQLite, when SQLAlchemy is installed."""
    try:
        import aiosqlite  # type: ignore[import-not-found]  # noqa: F401
        import sqlalchemy  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        from helpers import skip

        skip("SQLAlchemy and aiosqlite are not installed")
        return

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.db"
        store = agento.SQLSessionStore(f"sqlite+aiosqlite:///{path}")
        await store.create_tables()
        try:
            await run_contract(store)
        finally:
            await store.dispose()
