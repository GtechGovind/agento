"""Context management: compaction, offloading, deferred tools, skills."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from helpers import build_agent, build_app, collect, first_of, of_type, say

import agento


@agento.tool(read_only=True)
async def big_dump(rows: int = 2000) -> str:
    """Return a large payload.

    Args:
        rows: How many rows.
    """
    return json.dumps([{"id": i, "note": "x" * 40} for i in range(rows)])


@agento.tool(read_only=True)
async def small(q: str) -> str:
    """Return a small payload.

    Args:
        q: Query.
    """
    return f"ok: {q}"


def _config(**overrides: object) -> agento.RuntimeConfig:
    base: dict[str, object] = {
        "current_datetime": False,
        "ask_user_questions": False,
        "sub_agents": agento.SubAgentConfig(enabled=False),
        "compaction": agento.CompactionConfig(enabled=False),
        "large_tool_response": agento.LargeToolResponseConfig(enabled=False),
    }
    base.update(overrides)
    return agento.RuntimeConfig(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Compaction                                                                   #
# --------------------------------------------------------------------------- #


async def test_compaction_triggers_and_replaces_history() -> None:
    heavy = agento.Usage(input_tokens=90_000, output_tokens=10, total_tokens=90_010)
    app, llm = build_app(
        [say("first", usage=heavy), say("A STRUCTURED SUMMARY"), say("second")],
        properties=agento.ModelProperties(context_length=100_000),
    )
    agent = agento.Agent(
        name="a", model="m", instructions="hi", config=_config(compaction=agento.CompactionConfig())
    )
    session = await app.sessions.create(agent=agent)
    await (await session.create_turn("hello")).drain()

    turn = await session.create_turn("and now?")
    events = await collect(turn.stream())

    compacted = first_of(events, agento.ContextCompacted)
    assert compacted.tokens_before >= 90_000
    assert turn.state.metrics.total_compactions == 1
    assert turn.state.output.content == "second"

    # The conversation the model saw after compaction is the summary, not the
    # original history.
    final_messages = llm.requests[-1].messages
    assert any("A STRUCTURED SUMMARY" in str(m.get("content")) for m in final_messages)


async def test_compaction_does_not_loop() -> None:
    heavy = agento.Usage(input_tokens=90_000, output_tokens=10, total_tokens=90_010)
    app, llm = build_app(
        [say("first", usage=heavy), say("SUMMARY"), say("second"), say("third")],
        properties=agento.ModelProperties(context_length=100_000),
    )
    agent = agento.Agent(name="a", model="m", config=_config(compaction=agento.CompactionConfig()))
    session = await app.sessions.create(agent=agent)
    await (await session.create_turn("one")).drain()
    await (await session.create_turn("two")).drain()

    third = await session.create_turn("three")
    events = await collect(third.stream())

    assert not of_type(events, agento.ContextCompacted)
    assert third.state.output.content == "third"


async def test_compaction_threshold_from_model_context_length() -> None:
    from agento.core.capabilities.builtins.compaction import resolve_threshold

    assert resolve_threshold(configured=None, context_length=200_000) == 160_000
    assert resolve_threshold(configured=None, context_length=None) == 50_000
    assert resolve_threshold(configured=1234, context_length=200_000) == 1234
    # An output reservation shrinks the input budget.
    assert (
        resolve_threshold(configured=None, context_length=100_000, max_output_tokens=40_000)
        == 60_000
    )


# --------------------------------------------------------------------------- #
# Large tool responses                                                         #
# --------------------------------------------------------------------------- #


async def test_large_result_is_offloaded_to_an_artifact() -> None:
    store = agento.MemoryArtifactStore()
    app, _ = build_app(
        [say(tool_calls=[("big_dump", {"rows": 2000})]), say("Stored and summarized.")],
        artifacts=store,
    )
    agent = agento.Agent(
        name="a",
        model="m",
        tools=[big_dump],
        config=_config(large_tool_response=agento.LargeToolResponseConfig()),
    )
    session = await app.sessions.create(agent=agent)
    turn = await session.create_turn("dump it")

    events = await collect(turn.stream())

    created = first_of(events, agento.ArtifactCreated)
    assert created.size_bytes > 50_000

    result = first_of(events, agento.ToolResult)
    assert "too large" in result.content
    assert created.artifact_id in result.content
    assert len(result.content) < 3_000
    assert result.artifact_id == created.artifact_id

    stored = await store.read(created.artifact_id)
    assert len(stored) == created.size_bytes


async def test_small_results_pass_through_untouched() -> None:
    store = agento.MemoryArtifactStore()
    app, _ = build_app(
        [say(tool_calls=[("small", {"q": "hi"})]), say("done")], artifacts=store
    )
    agent = agento.Agent(
        name="a",
        model="m",
        tools=[small],
        config=_config(large_tool_response=agento.LargeToolResponseConfig()),
    )
    session = await app.sessions.create(agent=agent)
    turn = await session.create_turn("go")

    events = await collect(turn.stream())

    assert first_of(events, agento.ToolResult).content == "ok: hi"
    assert not of_type(events, agento.ArtifactCreated)
    assert await store.list() == []


async def test_offloading_without_a_store_truncates_instead() -> None:
    app, _ = build_app([say(tool_calls=[("big_dump", {"rows": 2000})]), say("done")])
    agent = agento.Agent(
        name="a",
        model="m",
        tools=[big_dump],
        config=_config(large_tool_response=agento.LargeToolResponseConfig()),
    )
    session = await app.sessions.create(agent=agent)
    turn = await session.create_turn("go")

    events = await collect(turn.stream())

    result = first_of(events, agento.ToolResult)
    assert "No artifact store" in result.content
    assert len(result.content) < 3_000


async def test_read_and_search_artifact_tools() -> None:
    store = agento.MemoryArtifactStore()
    artifact = await store.write(
        name="log.txt",
        content=b"line one\nline two has ERROR\nline three\n",
        mime_type="text/plain",
    )

    from agento.core.capabilities.builtins.large_tool_response import (
        _read_artifact,
        _search_artifact,
    )

    context = agento.ToolContext(artifacts=store)

    read = json.loads(await _read_artifact(artifact.id, context, 0, 8))
    assert read["content"] == "line one"
    assert read["has_more"] is True

    found = json.loads(await _search_artifact(artifact.id, "ERROR", context))
    assert len(found["matches"]) == 1
    assert found["matches"][0]["line"] == 2

    missing = json.loads(await _read_artifact("nope", context))
    assert "error" in missing


# --------------------------------------------------------------------------- #
# Deferred tools                                                               #
# --------------------------------------------------------------------------- #


async def test_deferred_tools_stay_out_of_the_prompt() -> None:
    deferred = agento.PolicyToolSet(
        agento.LocalToolSet("catalog", [small, big_dump], description="A big catalogue."),
        agento.ToolSelectors(),
        preload=False,
    )
    app, llm = build_app(
        [
            say(tool_calls=[("list_tools", {"server": "catalog"})]),
            say(tool_calls=[("call_tool", {"server": "catalog", "tool": "small", "input": {"q": "x"}})]),
            say("done"),
        ]
    )
    agent = build_agent(tools=[deferred])
    session = await app.sessions.create(agent=agent)
    turn = await session.create_turn("use the catalogue")

    events = await collect(turn.stream())

    exposed = [t["function"]["name"] for t in llm.requests[0].tools]
    assert "small" not in exposed
    assert {"list_tools", "get_tool_info", "call_tool"} <= set(exposed)
    assert "A big catalogue." in llm.requests[0].messages[0]["content"]

    results = of_type(events, agento.ToolResult)
    assert "small" in results[0].content          # list_tools
    assert results[1].content == "ok: x"          # call_tool reached the real tool


async def test_approval_survives_the_deferred_wrapper() -> None:
    """A destructive tool called through call_tool must still pause."""

    @agento.tool(destructive=True)
    async def wipe(table: str) -> str:
        """Wipe a table.

        Args:
            table: Table name.
        """
        return "wiped"

    deferred = agento.PolicyToolSet(
        agento.LocalToolSet("db", [wipe]), agento.ToolSelectors(), preload=False
    )
    app, _ = build_app(
        [
            say(
                tool_calls=[
                    ("call_tool", {"server": "db", "tool": "wipe", "input": {"table": "users"}})
                ]
            ),
            say("done"),
        ]
    )
    agent = build_agent(tools=[deferred])
    session = await app.sessions.create(agent=agent)
    turn = await session.create_turn("wipe users")

    events = await collect(turn.stream())

    assert of_type(events, agento.ApprovalRequired)
    assert not of_type(events, agento.ToolResult)


# --------------------------------------------------------------------------- #
# Skills                                                                       #
# --------------------------------------------------------------------------- #


def _write_skill(root: Path, name: str, description: str, body: str) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n", encoding="utf-8"
    )


async def test_skills_are_advertised_and_read_on_demand() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_skill(root, "refund-policy", "How to handle a refund request.", "1. Check eligibility.")
        (root / "refund-policy" / "limits.md").write_text("max 500", encoding="utf-8")

        app, llm = build_app(
            [say(tool_calls=[("read_skill", {"name": "refund-policy"})]), say("Following the policy.")],
            skills=agento.FileSkillSource(root),
        )
        agent = build_agent(skills=["refund-policy"])
        session = await app.sessions.create(agent=agent)
        turn = await session.create_turn("refund order 12")

        events = await collect(turn.stream())

        prompt = llm.requests[0].messages[0]["content"]
        # The description is advertised; the body is not.
        assert "How to handle a refund request." in prompt
        assert "Check eligibility" not in prompt
        assert "limits.md" in prompt

        assert "1. Check eligibility." in first_of(events, agento.ToolResult).content


async def test_unknown_skill_fails_clearly() -> None:
    with TemporaryDirectory() as tmp:
        app, _ = build_app([say("hi")], skills=agento.FileSkillSource(tmp))
        agent = build_agent(skills=["does-not-exist"])
        session = await app.sessions.create(agent=agent)
        try:
            await session.create_turn("hello")
        except agento.ConfigurationError as exc:
            assert "does-not-exist" in str(exc)
        else:  # pragma: no cover - the guard must fire
            raise AssertionError("An unknown skill should fail at turn creation")


async def test_skill_file_read_cannot_escape_its_directory() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_skill(root, "s", "A skill.", "body")
        (root / "secret.txt").write_text("do not read me", encoding="utf-8")

        source = agento.FileSkillSource(root)
        try:
            await source.read_resource("s", "../secret.txt")
        except (ValueError, KeyError) as exc:
            assert "escape" in str(exc) or "no resource" in str(exc).lower()
        else:  # pragma: no cover - the guard must fire
            raise AssertionError("Path traversal should be refused")
