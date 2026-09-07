"""Conversations that survive a restart.

    python examples/05_persistence.py

Uses SQLite through :class:`agento.SQLSessionStore` when SQLAlchemy is installed,
and falls back to the in-memory store otherwise — the code is identical either
way, which is the point of the store being an interface.

Also shows ``get_or_create_by_external_id``: binding a conversation to a key you
already have (a Slack thread, a ticket) so you do not maintain a second mapping
table.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from _shared import make_runtime

import agento

DB_PATH = Path("./.agento-example.db")


def build_store() -> tuple[object, str]:
    """A SQLite store when possible, otherwise memory."""
    try:
        import aiosqlite  # type: ignore[import-not-found]  # noqa: F401
        import sqlalchemy  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        print("[SQLAlchemy not installed — using the in-memory store]\n")
        return agento.MemorySessionStore(), "memory"
    return agento.SQLSessionStore(f"sqlite+aiosqlite:///{DB_PATH}"), "sqlite"


async def main() -> None:
    store, kind = build_store()
    if kind == "sqlite":
        await store.create_tables()  # type: ignore[attr-defined]

    app, model = make_runtime(
        [
            agento.say("Noted — you prefer metric units."),
            agento.say("You told me you prefer metric units."),
        ],
        store=store,
    )

    agent = agento.Agent(
        name="assistant",
        model=model,
        instructions="Remember what the user tells you about their preferences.",
    )
    app.register_agent(agent)

    # A conversation keyed to something in your own system.
    session, created = await app.sessions.get_or_create_by_external_id(
        "slack:C0123:1699999999.123456", agent=agent, metadata={"user_id": "u_42"}
    )
    print(f"session {session.id} ({'created' if created else 'existing'}) on {kind}")

    await (await session.create_turn("I prefer metric units, remember that.")).drain()

    # --- simulate a restart: nothing in memory, everything from the store ---
    fresh_app = agento.Agento(llm=app.llm, store=store, agents=[agent])
    reloaded = await fresh_app.sessions.get(session.id)
    assert reloaded is not None

    answer = await reloaded.run("what did I tell you about units?")
    print("after reload:", answer)

    turns = await reloaded.list_turns()
    print(f"turns stored: {len(turns.items)}")

    events = await reloaded.list_events(limit=100)
    print(f"events stored: {len(events.items)}")

    if kind == "sqlite":
        await store.dispose()  # type: ignore[attr-defined]
        print(f"\ndatabase written to {DB_PATH} — delete it to start over")


if __name__ == "__main__":
    asyncio.run(main())
