"""Human-in-the-loop: tool approval and client-side tools."""

from __future__ import annotations

from helpers import build_agent, build_app, collect, first_of, kinds, say

import agento


@agento.tool(read_only=True)
async def read_row(table: str) -> str:
    """Read a row.

    Args:
        table: Table name.
    """
    return f"row from {table}"


@agento.tool(destructive=True)
async def drop_table(table: str) -> str:
    """Drop a table.

    Args:
        table: Table name.
    """
    return f"dropped {table}"


def _agent() -> agento.Agent:
    return build_agent(tools=[read_row, drop_table])


async def test_destructive_tool_pauses_for_approval() -> None:
    app, _ = build_app([say(tool_calls=[("drop_table", {"table": "users"})]), say("Done.")])
    session = await app.sessions.create(agent=_agent())
    turn = await session.create_turn("drop the users table")

    events = await collect(turn.stream())

    required = first_of(events, agento.ApprovalRequired)
    assert len(required.tool_calls) == 1
    assert turn.state.status == "done"
    # "done" here means paused, not finished — the required action says so.
    assert len(turn.state.required_actions) == 1
    # The tool did not run.
    assert not [e for e in events if isinstance(e, agento.ToolResult)]


async def test_approval_allows_the_tool_to_run() -> None:
    app, _ = build_app([say(tool_calls=[("drop_table", {"table": "users"})]), say("Table dropped.")])
    session = await app.sessions.create(agent=_agent())
    first = await session.create_turn("drop the users table")
    events = await collect(first.stream())
    pending = first_of(events, agento.ApprovalRequired)

    second = await session.create_turn(
        [
            agento.ToolApproval(
                thread_id=pending.thread_id,
                tool_call_id=pending.tool_calls[0].id,
                decision="allow",
            )
        ]
    )
    events = await collect(second.stream())

    assert first_of(events, agento.ToolResult).content == "dropped users"
    assert second.state.output.content == "Table dropped."


async def test_denial_is_reported_to_the_model() -> None:
    app, _ = build_app(
        [say(tool_calls=[("drop_table", {"table": "users"})]), say("Understood, leaving it alone.")]
    )
    session = await app.sessions.create(agent=_agent())
    first = await session.create_turn("drop the users table")
    pending = first_of(await collect(first.stream()), agento.ApprovalRequired)

    second = await session.create_turn(
        [
            agento.ToolApproval(
                thread_id=pending.thread_id,
                tool_call_id=pending.tool_calls[0].id,
                decision="deny",
                reason="production database",
            )
        ]
    )
    events = await collect(second.stream())

    result = first_of(events, agento.ToolResult)
    assert result.is_error is True
    assert "denied" in result.content.lower()
    assert second.state.output.content == "Understood, leaving it alone."


async def test_read_only_tool_does_not_pause() -> None:
    app, _ = build_app([say(tool_calls=[("read_row", {"table": "users"})]), say("Here it is.")])
    session = await app.sessions.create(agent=_agent())
    turn = await session.create_turn("read a row")

    events = await collect(turn.stream())

    assert "ApprovalRequired" not in kinds(events)
    assert first_of(events, agento.ToolResult).content == "row from users"


async def test_user_message_is_rejected_while_an_approval_is_pending() -> None:
    """A new instruction cannot jump ahead of a decision the agent is waiting on."""
    app, _ = build_app([say(tool_calls=[("drop_table", {"table": "users"})]), say("Done.")])
    session = await app.sessions.create(agent=_agent())
    first = await session.create_turn("drop the users table")
    await collect(first.stream())

    try:
        await session.create_turn("actually, never mind")
    except agento.InvalidSendInputError as exc:
        assert "approval" in str(exc)
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("A user message during a pending approval should be rejected")


async def test_client_side_tool_pauses_and_resumes() -> None:
    @agento.tool
    async def pick_file(prompt: str) -> str:
        """Ask the user to choose a file.

        Args:
            prompt: What to show.
        """
        raise NotImplementedError

    app, _ = build_app(
        [say(tool_calls=[("pick_file", {"prompt": "Choose a CSV"})]), say("Got it, reading q3.csv.")]
    )
    agent = build_agent(tools=[agento.ClientSideToolSet("ui", [pick_file])])
    session = await app.sessions.create(agent=agent)

    first = await session.create_turn("open a file")
    events = await collect(first.stream())
    pending = first_of(events, agento.ClientToolRequired)
    assert not [e for e in events if isinstance(e, agento.ToolResult)]

    second = await session.create_turn(
        [
            agento.ToolReply(
                thread_id=pending.thread_id,
                tool_call_id=pending.tool_calls[0].id,
                content="q3.csv",
            )
        ]
    )
    await collect(second.stream())
    assert second.state.output.content == "Got it, reading q3.csv."


async def test_partial_answers_are_rejected() -> None:
    """Every pending call must be answered in one batch, or none is."""
    app, _ = build_app(
        [
            say(
                tool_calls=[
                    ("drop_table", {"table": "a"}),
                    ("drop_table", {"table": "b"}),
                ]
            ),
            say("Done."),
        ]
    )
    session = await app.sessions.create(agent=_agent())
    first = await session.create_turn("drop both tables")
    pending = first_of(await collect(first.stream()), agento.ApprovalRequired)
    assert len(pending.tool_calls) == 2

    try:
        await session.create_turn(
            [
                agento.ToolApproval(
                    thread_id=pending.thread_id,
                    tool_call_id=pending.tool_calls[0].id,
                    decision="allow",
                )
            ]
        )
    except agento.InvalidSendInputError as exc:
        assert "same batch" in str(exc)
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("A partial approval batch should be rejected")


async def test_approval_can_be_disabled_per_server() -> None:
    """An explicit empty list means no approval, even for destructive tools."""
    app, _ = build_app([say(tool_calls=[("drop_table", {"table": "users"})]), say("Done.")])
    agent = build_agent(
        tools=[
            agento.PolicyToolSet(
                agento.LocalToolSet("db", [drop_table]),
                agento.ToolSelectors(require_approval=[]),
                preload=True,
            )
        ]
    )
    session = await app.sessions.create(agent=agent)
    turn = await session.create_turn("drop it")

    events = await collect(turn.stream())

    assert "ApprovalRequired" not in kinds(events)
    assert first_of(events, agento.ToolResult).content == "dropped users"
