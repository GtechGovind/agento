"""Tool definition: schemas, validation, policy and naming."""

from __future__ import annotations

import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel

import agento
from agento.core.tools.registry import build_registry, sanitize_tool_name


class Priority(str, Enum):
    LOW = "low"
    HIGH = "high"


class Filter(BaseModel):
    field: str
    value: str


async def test_schema_is_derived_from_signature_and_docstring() -> None:
    @agento.tool
    async def find(
        query: str,
        limit: int = 10,
        priority: Priority = Priority.LOW,
        mode: Literal["fast", "deep"] = "fast",
        filters: list[Filter] | None = None,
    ) -> str:
        """Search the catalogue.

        Args:
            query: What to search for.
            limit: Maximum results.
            priority: How urgent this is.
            mode: Search depth.
            filters: Extra filters to apply.
        """
        return "ok"

    schema = find.schema
    assert schema.name == "find"
    assert schema.description == "Search the catalogue."

    properties = schema.input_schema["properties"]
    assert properties["query"]["description"] == "What to search for."
    assert properties["limit"]["default"] == 10
    assert properties["mode"]["enum"] == ["fast", "deep"]
    assert schema.input_schema["required"] == ["query"]
    # A nested model becomes a proper sub-schema rather than an opaque object.
    assert "$defs" in schema.input_schema


async def test_arguments_are_validated_before_the_function_runs() -> None:
    calls: list[int] = []

    @agento.tool
    async def double(n: int) -> str:
        """Double a number.

        Args:
            n: The number.
        """
        calls.append(n)
        return str(n * 2)

    good = await double.execute({"n": 21})
    assert good.content == "42"

    bad = await double.execute({"n": "not a number"})
    assert bad.is_error is True
    assert "n" in bad.content
    # The body never ran with a bad argument.
    assert calls == [21]


async def test_sync_tools_are_supported() -> None:
    @agento.tool
    def add(a: int, b: int) -> str:
        """Add two numbers.

        Args:
            a: First.
            b: Second.
        """
        return str(a + b)

    result = await add.execute({"a": 2, "b": 3})
    assert result.content == "5"


async def test_structured_returns_are_json_encoded() -> None:
    @agento.tool
    async def rows() -> list[dict[str, int]]:
        """Return rows."""
        return [{"id": 1}, {"id": 2}]

    result = await rows.execute({})
    assert result.is_structured is True
    assert json.loads(result.content) == [{"id": 1}, {"id": 2}]


async def test_pydantic_returns_are_serialized() -> None:
    class Answer(BaseModel):
        value: int

    @agento.tool
    async def answer() -> Answer:
        """Return a model."""
        return Answer(value=42)

    result = await answer.execute({})
    assert json.loads(result.content) == {"value": 42}


async def test_tool_remains_callable_directly() -> None:
    @agento.tool
    async def greet(name: str) -> str:
        """Greet someone.

        Args:
            name: Who.
        """
        return f"hi {name}"

    assert await greet(name="Govind") == "hi Govind"


async def test_context_parameter_is_hidden_from_the_model() -> None:
    @agento.tool
    async def whoami(scope: str, ctx: agento.ToolContext) -> str:
        """Report the caller.

        Args:
            scope: Anything.
        """
        return ctx.session_id or "none"

    assert list(whoami.schema.input_schema["properties"]) == ["scope"]

    from agento.core.tools.context import use_tool_context

    with use_tool_context(agento.ToolContext(session_id="s-1")):
        result = await whoami.execute({"scope": "x"})
    assert result.content == "s-1"


async def test_selectors_filter_by_annotation() -> None:
    @agento.tool(read_only=True)
    async def peek() -> str:
        """Read."""
        return "ok"

    @agento.tool(destructive=True)
    async def nuke() -> str:
        """Destroy."""
        return "ok"

    source = agento.LocalToolSet("db", [peek, nuke])

    read_only = agento.PolicyToolSet(
        source, agento.ToolSelectors(enable=["@read-only"]), preload=True
    )
    listing = await read_only.list_tools()
    assert [t.name for t in listing.tools] == ["peek"]

    # And it is enforced, not merely hidden.
    try:
        await read_only.call_tool("nuke", {})
    except agento.McpConnectionError as exc:
        assert exc.status_code == 403
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("A disabled tool must not be callable")


async def test_missing_named_tool_fails_loudly() -> None:
    @agento.tool
    async def present() -> str:
        """A tool."""
        return "ok"

    policy = agento.PolicyToolSet(
        agento.LocalToolSet("s", [present]),
        agento.ToolSelectors(enable=["present", "typo_tool"]),
        preload=True,
    )
    try:
        await policy.list_tools()
    except agento.McpConnectionError as exc:
        assert "typo_tool" in str(exc)
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("A misspelled tool name should be reported")


async def test_names_are_sanitized_and_deduplicated() -> None:
    assert sanitize_tool_name("normal_name") == "normal_name"
    assert sanitize_tool_name("weird.name/here") == "weird_name_here"
    assert len(sanitize_tool_name("x" * 200)) == 64

    @agento.tool(name="search")
    async def search_a(q: str) -> str:
        """Search A.

        Args:
            q: Query.
        """
        return "a"

    @agento.tool(name="search")
    async def search_b(q: str) -> str:
        """Search B.

        Args:
            q: Query.
        """
        return "b"

    registry = await build_registry(
        user_sets=[
            agento.PolicyToolSet(
                agento.LocalToolSet("alpha", [search_a]), agento.ToolSelectors(), preload=True
            ),
            agento.PolicyToolSet(
                agento.LocalToolSet("beta", [search_b]), agento.ToolSelectors(), preload=True
            ),
        ]
    )

    names = [schema["function"]["name"] for schema in registry.schemas]
    assert names == ["search", "search1"]
    # Both remain reachable, each pointing at its own set.
    assert registry.resolve("search").tool_set.name == "alpha"
    assert registry.resolve("search1").tool_set.name == "beta"
    # The set name is in the description, so the model can tell them apart.
    assert "mcp server: alpha" in registry.schemas[0]["function"]["description"]


async def test_builtin_sets_claim_names_first() -> None:
    @agento.tool(name="get_current_datetime")
    async def impostor() -> str:
        """A user tool with a built-in's name."""
        return "user"

    from agento.core.capabilities.builtins import CurrentDateTime

    registry = await build_registry(
        builtin_sets=list(CurrentDateTime().tool_sets()),
        user_sets=[
            agento.PolicyToolSet(
                agento.LocalToolSet("mine", [impostor]), agento.ToolSelectors(), preload=True
            )
        ],
    )

    assert registry.resolve("get_current_datetime").tool_set.name == "datetime"
    assert registry.resolve("get_current_datetime1").tool_set.name == "mine"
