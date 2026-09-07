"""What agento does when a tool returns far too much.

    python examples/04_context_management.py

A tool here returns a 300 KB JSON payload — enough to swamp a context window and
degrade every model call after it for the rest of the session.

agento writes it to the artifact store instead, and replaces it in the
conversation with an id, a size and a preview. The agent then reads what it needs
with ``read_artifact`` and ``search_artifact``, which is grep over the stored
content.

Skills appear here too: the agent is given ``refund-policy`` but only its
description is in the prompt. The body is read on demand.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from _shared import make_runtime

import agento

HERE = Path(__file__).parent


@agento.tool(read_only=True)
async def export_transactions(days: int = 90) -> str:
    """Export raw transactions as JSON.

    Args:
        days: How many days of history to export.
    """
    return json.dumps(
        [
            {
                "id": f"txn_{i}",
                "amount": round(10 + (i % 97) * 1.37, 2),
                "status": "refunded" if i % 50 == 0 else "settled",
                "note": "routine transaction " * 3,
            }
            for i in range(3000)
        ]
    )


async def main() -> None:
    artifacts = agento.MemoryArtifactStore()

    app, model = make_runtime(
        [
            agento.say(tool_calls=[("export_transactions", {"days": 90})]),
            agento.say(tool_calls=[("read_skill", {"name": "refund-policy"})]),
            agento.say("60 of the 3000 transactions were refunded, all within policy."),
        ],
        artifacts=artifacts,
        skills=agento.FileSkillSource(HERE / "skills"),
    )

    agent = agento.Agent(
        name="analyst",
        model=model,
        instructions="You analyse transaction exports. Never print raw data; summarize.",
        tools=[export_transactions],
        skills=["refund-policy"],
    )

    session = await app.sessions.create(agent=agent)
    turn = await session.create_turn("how many refunds in the last 90 days, and were they in policy?")

    async for event in turn.stream():
        if isinstance(event, agento.ArtifactCreated):
            print(f"→ stored {event.name}: {event.size_bytes:,} bytes as {event.artifact_id}")
        elif isinstance(event, agento.ToolResult):
            print(f"← {event.content[:160].replace(chr(10), ' ')}...")
        elif isinstance(event, agento.ModelMessageDelta) and event.content:
            print(event.content, end="", flush=True)
    print("\n")

    stored = await artifacts.list()
    print(f"artifacts kept: {len(stored)}")
    for artifact in stored:
        print(f"  {artifact.id}  {artifact.name}  {artifact.size_bytes:,} bytes")

    # Nothing near 300 KB ever entered the conversation.
    print("\ncontext cost of that tool result:", turn.state.metrics.total_input_tokens, "input tokens")


if __name__ == "__main__":
    asyncio.run(main())
