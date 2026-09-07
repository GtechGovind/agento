"""The agent loop: model calls, tool execution, and how a turn ends."""

from __future__ import annotations

from helpers import build_agent, build_app, collect, first_of, kinds, say, streamed_text

import agento


@agento.tool
async def get_weather(city: str) -> str:
    """Look up the weather.

    Args:
        city: City name.
    """
    return f"31C in {city}"


@agento.tool
async def explode(what: str) -> str:
    """A tool that raises.

    Args:
        what: Ignored.
    """
    raise RuntimeError("tool blew up")


async def test_simple_answer() -> None:
    app, _ = build_app([say("Hello there.")])
    agent = build_agent(instructions="Be brief.")
    session = await app.sessions.create(agent=agent)
    turn = await session.create_turn("hi")

    events = await collect(turn.stream())

    assert kinds(events) == ["TurnCreated", "ModelMessage", "ModelMessage", "TurnDone"]
    assert streamed_text(events) == "Hello there."
    assert turn.state.status == "done"
    assert turn.state.output.content == "Hello there."
    assert turn.state.required_actions == []


async def test_tool_call_round_trip() -> None:
    app, llm = build_app(
        [say(tool_calls=[("get_weather", {"city": "Delhi"})]), say("It is 31C in Delhi.")]
    )
    agent = build_agent(tools=[get_weather])
    session = await app.sessions.create(agent=agent)
    turn = await session.create_turn("weather in Delhi?")

    events = await collect(turn.stream())

    result = first_of(events, agento.ToolResult)
    assert result.content == "31C in Delhi"
    assert result.is_error is False
    assert turn.state.output.content == "It is 31C in Delhi."
    assert turn.state.metrics.total_tool_calls == 1
    assert turn.state.metrics.iterations == 2
    # The tool reached the model with its docstring as the description.
    schema = next(t for t in llm.requests[0].tools if t["function"]["name"] == "get_weather")
    assert "Look up the weather" in schema["function"]["description"]
    assert schema["function"]["parameters"]["required"] == ["city"]


async def test_tool_exception_becomes_a_result() -> None:
    """A raising tool must not kill the turn — the model should see the error."""
    app, _ = build_app(
        [say(tool_calls=[("explode", {"what": "x"})]), say("That tool failed; trying another way.")]
    )
    agent = build_agent(tools=[explode])
    session = await app.sessions.create(agent=agent)
    turn = await session.create_turn("go")

    events = await collect(turn.stream())

    result = first_of(events, agento.ToolResult)
    assert result.is_error is True
    assert "tool blew up" in result.content
    assert turn.state.status == "done"


async def test_unknown_tool_is_reported_not_fatal() -> None:
    app, _ = build_app([say(tool_calls=[("nonexistent", {})]), say("Sorry, I cannot do that.")])
    agent = build_agent(tools=[get_weather])
    session = await app.sessions.create(agent=agent)
    turn = await session.create_turn("go")

    events = await collect(turn.stream())

    result = first_of(events, agento.ToolResult)
    assert "Unknown tool" in result.content
    assert turn.state.status == "done"


async def test_invalid_arguments_are_returned_to_the_model() -> None:
    """Validation failure should read as a correctable message, not an exception."""
    app, _ = build_app(
        [say(tool_calls=[("get_weather", {})]), say("I need a city name.")]
    )
    agent = build_agent(tools=[get_weather])
    session = await app.sessions.create(agent=agent)
    turn = await session.create_turn("weather?")

    events = await collect(turn.stream())

    result = first_of(events, agento.ToolResult)
    assert "Invalid arguments" in result.content
    assert "city" in result.content


async def test_iteration_limit_ends_the_turn() -> None:
    """A model that never stops calling tools must be stopped by the limit."""
    app, _ = build_app(
        [say(tool_calls=[("get_weather", {"city": "Delhi"})])] * 10, on_exhausted="repeat"
    )
    agent = build_agent(
        tools=[get_weather],
        config=agento.RuntimeConfig(
            iteration_limit=3,
            current_datetime=False,
            ask_user_questions=False,
            sub_agents=agento.SubAgentConfig(enabled=False),
            compaction=agento.CompactionConfig(enabled=False),
            large_tool_response=agento.LargeToolResponseConfig(enabled=False),
        ),
    )
    session = await app.sessions.create(agent=agent)
    turn = await session.create_turn("loop forever")

    await collect(turn.stream())

    assert turn.state.status == "error"
    assert "iteration limit" in turn.state.message
    assert turn.state.metrics.iterations == 3


async def test_truncated_response_is_an_error() -> None:
    app, _ = build_app([say("A very long answer that got cut", finish_reason="length")])
    agent = build_agent()
    session = await app.sessions.create(agent=agent)
    turn = await session.create_turn("write an essay")

    await collect(turn.stream())

    assert turn.state.status == "error"
    assert "token limit" in turn.state.message


async def test_provider_failure_becomes_an_error_state() -> None:
    """A provider outage must terminate the turn cleanly, not raise out of it."""
    app, _ = build_app([say(error="upstream 503")])
    agent = build_agent()
    session = await app.sessions.create(agent=agent)
    turn = await session.create_turn("hello")

    events = await collect(turn.stream())

    assert turn.state.status == "error"
    assert "upstream 503" in turn.state.message
    # The stream still ended properly rather than blowing up mid-iteration.
    assert kinds(events)[-1] == "TurnDone"


async def test_parallel_tool_calls_all_resolve() -> None:
    app, _ = build_app(
        [
            say(
                tool_calls=[
                    ("get_weather", {"city": "Delhi"}),
                    ("get_weather", {"city": "Mumbai"}),
                    ("get_weather", {"city": "Chennai"}),
                ]
            ),
            say("All three collected."),
        ]
    )
    agent = build_agent(tools=[get_weather])
    session = await app.sessions.create(agent=agent)
    turn = await session.create_turn("weather in three cities")

    events = await collect(turn.stream())

    results = [e for e in events if isinstance(e, agento.ToolResult)]
    assert len(results) == 3
    assert {r.content for r in results} == {
        "31C in Delhi",
        "31C in Mumbai",
        "31C in Chennai",
    }
    assert turn.state.metrics.total_tool_calls == 3


async def test_stream_is_single_use() -> None:
    app, _ = build_app([say("hi")])
    agent = build_agent()
    session = await app.sessions.create(agent=agent)
    turn = await session.create_turn("hi")

    await collect(turn.stream())
    try:
        await collect(turn.stream())
    except RuntimeError as exc:
        assert "already been run" in str(exc)
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("A second stream() should raise")
