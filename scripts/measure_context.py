"""Measure actual runtime request context with synthetic, offline fixtures.

Requires tiktoken (the extras extra). Counts JSON-serialized messages and tools,
not provider billing tokens, answer quality, latency, or complete task cost.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import tiktoken

import agento


def config(*, offload: bool = False) -> agento.RuntimeConfig:
    return agento.RuntimeConfig(
        current_datetime=False,
        ask_user_questions=False,
        sub_agents=agento.SubAgentConfig(enabled=False),
        compaction=agento.CompactionConfig(enabled=False),
        large_tool_response=agento.LargeToolResponseConfig(enabled=offload),
    )


def serialized(request: agento.LLMRequest) -> str:
    return json.dumps(
        {"messages": request.messages, "tools": request.tools or []},
        ensure_ascii=False, separators=(",", ":"),
    )


def counts(requests: list[agento.LLMRequest]) -> list[int]:
    encoder = tiktoken.get_encoding("o200k_base")
    return [len(encoder.encode(serialized(request), disallowed_special=())) for request in requests]


def comparison(before: int, after: int) -> dict[str, int | float]:
    return {"before": before, "after": after,
            "reduction_percent": round(100 * (before - after) / before, 2)}


async def offloading(rows: int) -> dict[str, Any]:
    subjects = ["Delivery address correction", "Invoice copy requested",
                "Replacement part availability", "Shipment tracking update"]
    statuses = ["open", "waiting", "resolved"]
    payload = json.dumps([
        {"ticket_id": f"TKT-{i:06d}", "customer_id": f"CUST-{i % 137:04d}",
         "subject": subjects[i % len(subjects)], "status": statuses[i % len(statuses)],
         "priority": "high" if i % 7 == 0 else "normal",
         "last_update": f"2026-09-{1 + i % 7:02d}T09:00:00Z"}
        for i in range(rows)
    ])

    @agento.tool(read_only=True)
    async def search_tickets() -> str:
        """Return matching support tickets for inspection."""
        return payload

    runs: dict[bool, list[int]] = {}
    retained = False
    stored_bytes = 0
    for enabled in (False, True):
        artifacts = agento.MemoryArtifactStore()
        llm = agento.ScriptedLLM([
            agento.say(tool_calls=["search_tickets"]), agento.say("Inspection complete."),
        ])
        app = agento.Agento(llm=llm, artifacts=artifacts)
        agent = agento.Agent(name="measurement", model="scripted/measurement", tools=[search_tickets],
                             config=config(offload=enabled))
        await app.run(agent, "Inspect the matching support tickets.")
        assert len(llm.requests) == 2
        runs[enabled] = counts(llm.requests)
        if enabled:
            records = await artifacts.list()
            if records:
                saved = await artifacts.read(records[0].id)
                assert saved == payload.encode()
                stored_bytes = len(saved)
                retained = True
            else:
                assert payload in serialized(llm.requests[-1]) or any(
                    m.get("content") == payload for m in llm.requests[-1].messages
                )
    return {
        "rows": rows, "payload_bytes": len(payload.encode()),
        "payload_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "offloaded": retained, "stored_bytes": stored_bytes,
        "next_request": comparison(runs[False][-1], runs[True][-1]),
        "two_request_input_total": comparison(sum(runs[False]), sum(runs[True])),
        "baseline_request_tokens": runs[False], "enabled_request_tokens": runs[True],
    }


async def deferred(tool_count: int) -> dict[str, Any]:
    async def search_records(
        query: str, status: Literal["open", "closed", "all"] = "all", limit: int = 20,
    ) -> str:
        """Search a collection of business records.

        Args:
            query: Search words or an exact record identifier.
            status: Filter by record state, or include all states.
            limit: Maximum number of matching records to return.
        """
        raise AssertionError("This scenario measures discovery before any business tool is called")

    tools = [agento.Tool(search_records, name=f"search_collection_{i:03d}", read_only=True,
                         description=f"Search collection {i:03d} for matching business records. "
                         "Returns identifiers, titles, and status for the selected records.")
             for i in range(tool_count)]
    runs: dict[bool, int] = {}
    exposed: dict[bool, int] = {}
    for preload in (True, False):
        source = agento.PolicyToolSet(
            agento.LocalToolSet("catalog", tools, description="Search the business record collections."),
            agento.ToolSelectors(), preload=preload,
        )
        llm = agento.ScriptedLLM([agento.say("Ready.")])
        app = agento.Agento(llm=llm)
        await app.run(agento.Agent(name="measurement", model="scripted/measurement",
                                  tools=[source], config=config()), "What can you help with?")
        assert len(llm.requests) == 1
        names = {t["function"]["name"] for t in llm.requests[0].tools or []}
        if preload:
            assert names == {tool.name for tool in tools}
        else:
            assert names == {"list_tools", "get_tool_info", "call_tool"}
        runs[preload] = counts(llm.requests)[0]
        exposed[preload] = len(names)
    return {"catalog_tools": tool_count, "first_request": comparison(runs[True], runs[False]),
            "preloaded_schemas": exposed[True], "deferred_entrypoint_schemas": exposed[False]}


async def measure() -> dict[str, Any]:
    return {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "agento": version("agento"),
        "pydantic": version("pydantic"), "tiktoken": version("tiktoken"),
        "encoding": "o200k_base", "unit": "tokens in compact JSON of runtime messages plus tools",
        "method": "Paired real runtime executions with synthetic data and ScriptedLLM; "
                  "unrelated optional capabilities disabled equally; helper schemas and instructions included.",
        "limits": "No provider calls, billing, answer-quality, latency, or full-task savings measured. "
                  "Offloading measures the next request and two-request input total before selective reads. "
                  "Deferral measures only the first request before discovery. Generated artifact IDs "
                  "can change token counts slightly. Negative reductions mean additional context.",
        "offloading": [await offloading(rows) for rows in (20, 200, 2000)],
        "deferred_tools": [await deferred(size) for size in (1, 10, 100)],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write results here instead of standard output")
    args = parser.parse_args()
    result = json.dumps(asyncio.run(measure()), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
        print(f"Wrote context measurements to {args.output}")
    else:
        print(result, end="")
