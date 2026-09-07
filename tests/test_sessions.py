"""Sessions, turns, persistence and resumption."""

from __future__ import annotations

import asyncio

from helpers import build_agent, build_app, collect, kinds, of_type, say

import agento


@agento.tool
async def note(text: str) -> str:
    """Record a note.

    Args:
        text: What to record.
    """
    return f"noted: {text}"


async def test_multi_turn_conversation_keeps_history() -> None:
    app, llm = build_app([say("Hello."), say("Yes, I remember.")])
    agent = build_agent()
    session = await app.sessions.create(agent=agent)

    await (await session.create_turn("hi, I am Govind")).drain()
    await (await session.create_turn("do you remember my name?")).drain()

    # The second call saw the first exchange.
    second = llm.requests[1].messages
    contents = [str(m.get("content")) for m in second]
    assert any("Govind" in c for c in contents)
    assert any(c == "Hello." for c in contents)


async def test_events_are_persisted_before_they_are_yielded() -> None:
    app, _ = build_app([say(tool_calls=[("note", {"text": "x"})]), say("done")])
    agent = build_agent(tools=[note])
    session = await app.sessions.create(agent=agent)
    turn = await session.create_turn("note something")

    seen_live = await collect(turn.stream())
    stored = (await turn.list_events(limit=100)).items

    assert kinds(stored) == [
        "TurnCreated",
        "ModelMessage",
        "ToolResult",
        "ModelMessage",
        "TurnDone",
    ]
    # Deltas are streamed but never stored.
    assert of_type(seen_live, agento.ModelMessageDelta)
    assert not of_type(stored, agento.ModelMessageDelta)


async def test_session_title_comes_from_the_first_message_only() -> None:
    app, _ = build_app([say("a"), say("b")])
    session = await app.sessions.create(agent=build_agent())

    await (await session.create_turn("Help me plan a migration")).drain()
    assert session.title == "Help me plan a migration"

    await (await session.create_turn("second message")).drain()
    assert session.title == "Help me plan a migration"


async def test_explicit_title_is_never_overwritten() -> None:
    app, _ = build_app([say("a")])
    session = await app.sessions.create(agent=build_agent(), title="Chosen title")
    await (await session.create_turn("something else entirely")).drain()
    assert session.title == "Chosen title"


async def test_session_reload_resumes_the_conversation() -> None:
    """A session loaded fresh from the store continues where it left off."""
    app, llm = build_app([say("First answer."), say("Second answer.")])
    agent = build_agent()
    app.register_agent(agent)

    session = await app.sessions.create(agent=agent)
    session_id = session.id
    await (await session.create_turn("first question")).drain()

    reloaded = await app.sessions.get(session_id)
    assert reloaded is not None
    await (await reloaded.create_turn("second question")).drain()

    contents = [str(m.get("content")) for m in llm.requests[1].messages]
    assert any("first question" in c for c in contents)
    assert any("First answer." in c for c in contents)


async def test_branching_from_an_earlier_turn() -> None:
    app, _ = build_app([say("one"), say("two"), say("three")])
    session = await app.sessions.create(agent=build_agent())

    first = await session.create_turn("q1")
    await first.drain()
    second = await session.create_turn("q2")
    await second.drain()

    # Branch from the first turn rather than the tip.
    branch = await session.create_turn("q2 rewritten", previous_turn_id=first.id)
    await branch.drain()

    assert branch.previous_turn_id == first.id
    # All three turns still exist; the session simply moved onto the branch.
    assert len((await session.list_turns()).items) == 3
    assert session.last_turn_id == branch.id

    # The session feed follows the active branch only.
    feed = await session.list_events(limit=100)
    turn_ids = {item.turn_id for item in feed.items}
    assert second.id not in turn_ids
    assert {first.id, branch.id} <= turn_ids


async def test_new_root_turn_starts_fresh() -> None:
    app, llm = build_app([say("one"), say("two")])
    session = await app.sessions.create(agent=build_agent())
    await (await session.create_turn("remember: alpha")).drain()

    await (await session.create_turn("what did I say?", previous_turn_id="none")).drain()

    contents = [str(m.get("content")) for m in llm.requests[1].messages]
    assert not any("alpha" in c for c in contents)


async def test_get_or_create_by_external_id() -> None:
    app, _ = build_app([say("hi"), say("hello again")])
    agent = build_agent()

    session, created = await app.sessions.get_or_create_by_external_id("slack:C1:1699", agent=agent)
    assert created is True

    same, created_again = await app.sessions.get_or_create_by_external_id(
        "slack:C1:1699", agent=agent
    )
    assert created_again is False
    assert same.id == session.id


async def test_metadata_reaches_tools() -> None:
    seen: dict[str, str] = {}

    @agento.tool
    async def whoami(ctx: agento.ToolContext) -> str:
        """Report the caller."""
        seen.update(ctx.metadata)
        seen["session"] = ctx.session_id or ""
        seen["thread"] = ctx.thread_id or ""
        return "ok"

    app, _ = build_app([say(tool_calls=["whoami"]), say("done")])
    agent = build_agent(tools=[whoami])
    session = await app.sessions.create(agent=agent, metadata={"user_id": "u_42"})
    await (await session.create_turn("who am I?")).drain()

    assert seen["user_id"] == "u_42"
    assert seen["session"] == session.id
    assert seen["thread"] == "main"


async def test_cancellation_leaves_a_resumable_turn() -> None:
    app, llm = build_app(
        [say(tool_calls=[("note", {"text": "one"})])] * 5, on_exhausted="repeat"
    )
    agent = build_agent(tools=[note])
    session = await app.sessions.create(agent=agent)
    turn = await session.create_turn("keep going")

    seen = 0
    async for _ in turn.stream():
        seen += 1
        if seen == 6:
            turn.cancel()

    assert turn.state.status == "cancelled"

    # The next turn resumes from the cancelled one's context rather than losing it.
    llm.push(say("Picking up where we left off."))
    resumed = await session.create_turn("carry on")
    await resumed.drain()
    assert resumed.state.status == "done"
    assert resumed.previous_turn_id == turn.id


async def test_run_returns_the_final_text() -> None:
    app, _ = build_app([say("The answer is 42.")])
    agent = build_agent()
    assert await app.run(agent, "what is the answer?") == "The answer is 42."


async def test_drain_runs_in_the_background() -> None:
    app, _ = build_app([say("background answer")])
    agent = build_agent()
    session = await app.sessions.create(agent=agent)
    turn = await session.create_turn("go")

    task = asyncio.create_task(turn.drain())
    state = await task

    assert state.status == "done"
    assert len((await turn.list_events(limit=50)).items) >= 3


async def test_loaded_turn_is_not_executable() -> None:
    app, _ = build_app([say("hi")])
    session = await app.sessions.create(agent=build_agent())
    turn = await session.create_turn("hi")
    await turn.drain()

    loaded = await session.get_turn(turn.id)
    assert loaded is not None
    assert loaded.is_executable is False
    try:
        await collect(loaded.stream())
    except RuntimeError as exc:
        assert "loaded from storage" in str(exc)
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("A stored turn should not be runnable")
