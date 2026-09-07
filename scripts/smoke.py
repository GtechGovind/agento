#!/usr/bin/env python3
"""Verify the three adapters that need real dependencies.

agento's core is tested with a scripted model and an in-memory store, which needs
nothing installed. Three pieces cannot be tested that way, because they exist to
talk to something real:

* **LiteLLM** — a live model call, streaming, with a tool.
* **MCP** — a live connection to a remote server, listing and calling a tool.
* **SQLAlchemy** — a durable session in SQLite, written and read back.

This script exercises each one. Run it once in your environment before relying on
them::

    pip install "agento[all]"
    export OPENAI_API_KEY=...            # or ANTHROPIC_API_KEY / GEMINI_API_KEY
    python scripts/smoke.py

Each check is skipped, not failed, when its prerequisites are missing — so
running it with only a database configured still tells you the store works.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import agento  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

MODEL_BY_KEY = {
    "OPENAI_API_KEY": "openai/gpt-4o-mini",
    "ANTHROPIC_API_KEY": "anthropic/claude-sonnet-4-5",
    "GEMINI_API_KEY": "gemini/gemini-2.0-flash",
}

results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    colour = {"ok": GREEN, "skip": YELLOW, "fail": RED}[status]
    print(f"{colour}{status.upper():5}{RESET} {name}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
    results.append((name, status, detail))


def pick_model() -> str | None:
    for variable, model in MODEL_BY_KEY.items():
        if os.environ.get(variable):
            return model
    return None


# --------------------------------------------------------------------------- #
# 1. LiteLLM                                                                   #
# --------------------------------------------------------------------------- #


async def check_litellm() -> None:
    """A real streamed completion, with a tool call and usage."""
    name = "LiteLLM adapter"
    try:
        import litellm  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        record(name, "skip", 'pip install "agento[litellm]"')
        return

    model = pick_model()
    if model is None:
        record(name, "skip", "no OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY")
        return

    @agento.tool(read_only=True)
    async def get_population(city: str) -> str:
        """Return a city's population.

        Args:
            city: City name.
        """
        return f'{{"city": "{city}", "population": 32000000}}'

    app = agento.Agento(llm=agento.LiteLLMProvider())
    agent = agento.Agent(
        name="smoke",
        model=model,
        instructions="Answer in one short sentence. Use the tool rather than guessing.",
        tools=[get_population],
    )

    session = await app.sessions.create(agent=agent)
    turn = await session.create_turn("What is the population of Delhi? Use the tool.")

    deltas = 0
    tool_ran = False
    async for event in turn.stream():
        if isinstance(event, agento.ModelMessageDelta) and event.content:
            deltas += 1
        elif isinstance(event, agento.ToolResult):
            tool_ran = True

    if turn.state.status != "done":
        raise RuntimeError(f"turn ended {turn.state.status}: {getattr(turn.state, 'message', '')}")

    metrics = turn.state.metrics
    if metrics.total_input_tokens == 0:
        raise RuntimeError("no usage reported — check stream_options handling")

    detail = (
        f"{model} · {deltas} deltas · tool={'yes' if tool_ran else 'no'} · "
        f"{metrics.total_tokens} tokens"
        + (f" · ${metrics.total_cost_usd:.5f}" if metrics.total_cost_usd else "")
    )
    record(name, "ok", detail)


# --------------------------------------------------------------------------- #
# 2. MCP                                                                       #
# --------------------------------------------------------------------------- #


async def check_mcp() -> None:
    """A real connection to a remote MCP server.

    Set ``SMOKE_MCP_URL`` (and ``SMOKE_MCP_HEADER`` as ``Name: value`` if it needs
    auth). A public, no-auth server such as ``https://mcp.deepwiki.com/mcp`` works.
    """
    name = "MCP adapter"
    try:
        import mcp  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        record(name, "skip", 'pip install "agento[mcp]"')
        return

    url = os.environ.get("SMOKE_MCP_URL")
    if not url:
        record(name, "skip", "set SMOKE_MCP_URL to a remote MCP server")
        return

    headers: dict[str, str] = {}
    raw_header = os.environ.get("SMOKE_MCP_HEADER")
    if raw_header and ":" in raw_header:
        key, value = raw_header.split(":", 1)
        headers[key.strip()] = value.strip()

    from agento.core.tools.remote_mcp import RemoteMCP

    server = RemoteMCP("smoke", url, headers=headers, connect_timeout=30)
    try:
        listing = await server.list_tools()
        if hasattr(listing, "servers"):
            raise RuntimeError("server reports that authorization is required")
        tools = listing.tools
        if not tools:
            raise RuntimeError("server returned no tools")
        detail = f"{len(tools)} tools · session={server.session_id or 'stateless'}"

        # Call the first read-only tool, if there is one and it takes no required
        # arguments — enough to prove the round trip without side effects.
        for schema in tools:
            required = schema.input_schema.get("required") or []
            read_only = schema.annotations and schema.annotations.read_only
            if read_only and not required:
                outcome = await server.call_tool(schema.name, {})
                detail += f" · called {schema.name} ({len(outcome.content)} chars)"
                break
        record(name, "ok", detail)
    finally:
        await server.aclose()


# --------------------------------------------------------------------------- #
# 3. SQLAlchemy store                                                          #
# --------------------------------------------------------------------------- #


async def check_sql_store() -> None:
    """A durable session in SQLite, written and read back in a fresh runtime."""
    name = "SQLAlchemy store"
    try:
        import aiosqlite  # type: ignore[import-not-found]  # noqa: F401
        import sqlalchemy  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        record(name, "skip", 'pip install "agento[sqlite]"')
        return

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "smoke.db"
        store = agento.SQLSessionStore(f"sqlite+aiosqlite:///{path}")
        await store.create_tables()
        try:
            # Full contract suite, the same one the memory store passes.
            sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
            from test_store import run_contract  # type: ignore[import-not-found]

            await run_contract(store)

            # And a real turn, end to end, reloaded from a fresh runtime.
            llm = agento.ScriptedLLM([agento.say("stored answer"), agento.say("second answer")])
            app = agento.Agento(llm=llm, store=store)
            agent = agento.Agent(name="smoke", model="scripted", instructions="hi")
            app.register_agent(agent)

            session = await app.sessions.create(agent=agent)
            await (await session.create_turn("first")).drain()

            reloaded_app = agento.Agento(llm=llm, store=store, agents=[agent])
            reloaded = await reloaded_app.sessions.get(session.id)
            if reloaded is None:
                raise RuntimeError("session did not survive reload")
            answer = await reloaded.run("second")

            events = await reloaded.list_events(limit=100)
            detail = f"contract passed · reloaded · {len(events.items)} events · {answer!r}"
            record(name, "ok", detail)
        finally:
            await store.dispose()


# --------------------------------------------------------------------------- #


async def main() -> int:
    print(f"agento {agento.__version__} — adapter smoke test\n")

    for check in (check_litellm, check_mcp, check_sql_store):
        try:
            await check()
        except Exception:
            record(check.__doc__.splitlines()[0] if check.__doc__ else check.__name__, "fail")
            print(f"{DIM}{traceback.format_exc()}{RESET}")

    print()
    ok = sum(1 for _, status, _ in results if status == "ok")
    skipped = sum(1 for _, status, _ in results if status == "skip")
    failed = sum(1 for _, status, _ in results if status == "fail")
    print(f"{ok} ok, {skipped} skipped, {failed} failed")

    if skipped:
        print(f"\n{DIM}Skipped checks need a package or an environment variable — see the "
              f"detail column.{RESET}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
