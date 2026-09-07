"""Delegation: sub-agents, isolation, and joining back."""

from __future__ import annotations

from helpers import build_app, collect, first_of, of_type, say

import agento
from agento.core.instructions import SUB_AGENT_IDENTITY

SUB_MARKER = SUB_AGENT_IDENTITY


@agento.tool(read_only=True)
async def search(q: str) -> str:
    """Search.

    Args:
        q: Query.
    """
    return f"raw results for {q}: " + ("noise " * 30)


def _agent(**overrides: object) -> agento.Agent:
    config = agento.RuntimeConfig(
        current_datetime=False,
        ask_user_questions=False,
        compaction=agento.CompactionConfig(enabled=False),
        large_tool_response=agento.LargeToolResponseConfig(enabled=False),
        **overrides,  # type: ignore[arg-type]
    )
    return agento.Agent(
        name="lead", model="scripted/test-model", instructions="Delegate.", tools=[search], config=config
    )


def _is_sub(request: object) -> bool:
    messages = getattr(request, "messages", [])
    return bool(messages) and SUB_MARKER in str(messages[0].get("content", ""))


def _assistant_turns(request: object) -> int:
    return sum(1 for m in getattr(request, "messages", []) if m.get("role") == "assistant")


async def test_sub_agent_runs_and_reports_back() -> None:
    def script(request: object) -> object:
        if _is_sub(request):
            if _assistant_turns(request) == 0:
                return say(tool_calls=[("search", {"q": "harnesses"})])
            return say("Summary: a harness runs the loop.")
        if _assistant_turns(request) == 0:
            return say(
                tool_calls=[
                    ("create_sub_agent", {"name": "researcher", "input": "Research harnesses."})
                ]
            )
        return say("The researcher says: a harness runs the loop.")

    app, _ = build_app([script] * 20, on_exhausted="repeat")
    session = await app.sessions.create(agent=_agent())
    turn = await session.create_turn("research harnesses")

    events = await collect(turn.stream())

    created = first_of(events, agento.ThreadCreated)
    assert created.agent_info.name == "researcher"

    done = first_of(events, agento.ThreadDone)
    assert done.state.status == "done"

    # The parent's open tool call was closed with the child's summary.
    parent_results = [
        e for e in of_type(events, agento.ToolResult) if e.thread_id == "main"
    ]
    assert parent_results[-1].content == "Summary: a harness runs the loop."

    assert turn.state.output.content == "The researcher says: a harness runs the loop."
    assert turn.state.metrics.total_sub_agents == 1


async def test_sub_agent_context_is_isolated() -> None:
    """The child sees its brief and nothing else — that is the whole point."""
    seen: dict[str, list[str]] = {"sub": []}

    def script(request: object) -> object:
        if _is_sub(request):
            seen["sub"] = [
                str(m.get("content", "")) for m in request.messages if m.get("role") == "user"  # type: ignore[attr-defined]
            ]
            return say("done")
        if _assistant_turns(request) == 0:
            return say(
                tool_calls=[("create_sub_agent", {"name": "worker", "input": "Do the thing."})]
            )
        return say("finished")

    app, _ = build_app([script] * 20, on_exhausted="repeat")
    session = await app.sessions.create(agent=_agent())
    turn = await session.create_turn("a secret the child must not see")
    await collect(turn.stream())

    assert seen["sub"] == ["Do the thing."]
    assert not any("secret" in message for message in seen["sub"])


async def test_sub_agents_cannot_spawn_further_sub_agents() -> None:
    """Children get no delegation tool, so fan-out stays bounded."""
    tools_seen: dict[str, list[str]] = {}

    def script(request: object) -> object:
        names = [t["function"]["name"] for t in (request.tools or [])]  # type: ignore[attr-defined]
        if _is_sub(request):
            tools_seen["sub"] = names
            return say("done")
        tools_seen["root"] = names
        if _assistant_turns(request) == 0:
            return say(tool_calls=[("create_sub_agent", {"name": "w", "input": "work"})])
        return say("finished")

    app, _ = build_app([script] * 20, on_exhausted="repeat")
    session = await app.sessions.create(agent=_agent())
    await collect((await session.create_turn("go")).stream())

    assert "create_sub_agent" in tools_seen["root"]
    assert "create_sub_agent" not in tools_seen["sub"]


async def test_sub_agent_failure_is_reported_to_the_parent() -> None:
    def script(request: object) -> object:
        if _is_sub(request):
            return say(error="sub-agent provider failure")
        if _assistant_turns(request) == 0:
            return say(tool_calls=[("create_sub_agent", {"name": "w", "input": "work"})])
        return say("I could not complete the research.")

    app, _ = build_app([script] * 20, on_exhausted="repeat")
    session = await app.sessions.create(agent=_agent())
    turn = await session.create_turn("go")

    events = await collect(turn.stream())

    done = first_of(events, agento.ThreadDone)
    assert done.state.status == "error"
    # The parent still resumed and answered rather than hanging.
    assert turn.state.status == "done"
    assert turn.state.output.content == "I could not complete the research."


async def test_parallel_sub_agents() -> None:
    def script(request: object) -> object:
        if _is_sub(request):
            return say("partial result")
        if _assistant_turns(request) == 0:
            return say(
                tool_calls=[
                    ("create_sub_agent", {"name": "one", "input": "task one"}),
                    ("create_sub_agent", {"name": "two", "input": "task two"}),
                    ("create_sub_agent", {"name": "three", "input": "task three"}),
                ]
            )
        return say("All three reported.")

    app, _ = build_app([script] * 30, on_exhausted="repeat")
    session = await app.sessions.create(agent=_agent())
    turn = await session.create_turn("fan out")

    events = await collect(turn.stream())

    assert len(of_type(events, agento.ThreadCreated)) == 3
    assert len(of_type(events, agento.ThreadDone)) == 3
    assert turn.state.metrics.total_sub_agents == 3
    assert turn.state.output.content == "All three reported."


async def test_model_choices_are_offered_and_resolved() -> None:
    chosen: dict[str, str] = {}

    def script(request: object) -> object:
        if _is_sub(request):
            return say("done")
        if _assistant_turns(request) == 0:
            schema = next(
                t for t in request.tools if t["function"]["name"] == "create_sub_agent"  # type: ignore[attr-defined]
            )
            chosen["enum"] = schema["function"]["parameters"]["properties"]["model"]["enum"]
            return say(
                tool_calls=[
                    ("create_sub_agent", {"name": "w", "input": "work", "model": "fast"})
                ]
            )
        return say("finished")

    app, _ = build_app([script] * 20, on_exhausted="repeat")
    agent = _agent(
        sub_agents=agento.SubAgentConfig(
            enabled=True,
            model_choices={"fast": "quick lookups", "thorough": "deep analysis"},
            models={"fast": "scripted/test-model", "thorough": "scripted/test-model"},
        )
    )
    session = await app.sessions.create(agent=agent)
    turn = await session.create_turn("go")

    events = await collect(turn.stream())

    assert sorted(chosen["enum"]) == ["fast", "thorough"]
    assert first_of(events, agento.ThreadCreated).agent_info.model == "fast"
