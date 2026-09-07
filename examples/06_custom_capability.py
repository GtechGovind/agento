"""Writing your own capability.

    python examples/06_custom_capability.py

Everything agento does beyond "call the model, run the tools" is a capability:
compaction, offloading, sub-agents, skills. They use no private API, so anything
they can do, you can do.

Two here:

``BudgetGuard``
    Tracks spend across turns using durable capability state, and injects a
    warning into the prompt when the agent is close to the limit. Shows
    ``state_key`` / ``load_state`` / ``SetState``, and ``prepare_request`` for an
    edit that applies to one call without becoming part of the history.

``ToolAudit``
    Records every tool result. Shows ``process_tool_results``, the same hook that
    large-result offloading uses.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from _shared import make_runtime

import agento


class BudgetGuard(agento.Capability):
    """Warns the agent as it approaches a token budget, and remembers across turns."""

    name = "budget_guard"
    # Declaring a key is what makes state durable. Anything written under it is
    # persisted with the turn and handed back on the next one.
    state_key = "example.budget"

    def __init__(self, limit_tokens: int) -> None:
        self._limit = limit_tokens
        self._spent = 0

    def load_state(self, value: Any) -> None:
        """Restore the running total from the previous turn."""
        self._spent = int(value or 0)

    def build_instructions(self, builder: agento.core.instructions.InstructionBuilder) -> None:
        builder.add_section(
            "budget",
            f"This conversation has a budget of {self._limit:,} tokens. Prefer short, direct "
            "answers, and do not re-read information already in the conversation.",
        )

    async def pre_llm(self, context: agento.ExecutionContext) -> AsyncIterator[Any]:
        """Record the current spend before each model call."""
        self._spent = context.usage.total()
        yield agento.SetState(key=self.state_key, value=self._spent)

    def prepare_request(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        """Append a warning for this one call, without storing it.

        ``prepare_request`` is the ephemeral hook: the edit reaches the model and
        then disappears, so a reminder like this cannot pile up in the history.
        """
        if self._spent < self._limit * 0.8:
            return None
        return [
            *messages,
            {
                "role": "user",
                "content": (
                    f"[budget: {self._spent:,} of {self._limit:,} tokens used — "
                    "wrap up and answer now]"
                ),
            },
        ]


class ToolAudit(agento.Capability):
    """Records every tool result. A hook into the same place offloading uses."""

    name = "tool_audit"

    def __init__(self) -> None:
        self.records: list[tuple[str, int, bool]] = []

    async def process_tool_results(
        self, results: list[Any], context: agento.ExecutionContext
    ) -> list[Any]:
        for result in results:
            self.records.append(
                (result.tool_call.function.name, len(result.message.content), result.failed)
            )
        return []


@agento.tool(read_only=True)
async def lookup(term: str) -> str:
    """Look something up.

    Args:
        term: What to look up.
    """
    return f"definition of {term}: a thing that does a thing"


async def main() -> None:
    audit = ToolAudit()

    app, model = make_runtime(
        [
            agento.say(tool_calls=[("lookup", {"term": "harness"})]),
            agento.say("A harness is the runtime around a model."),
        ],
    )
    # Capabilities on the runtime apply to every agent it runs.
    app.capabilities = [BudgetGuard(limit_tokens=50_000), audit]

    agent = agento.Agent(
        name="assistant", model=model, instructions="Answer briefly.", tools=[lookup]
    )
    session = await app.sessions.create(agent=agent)

    answer = await session.run("what is a harness?")
    print("answer:", answer)
    print("audited tool calls:", audit.records)

    # The budget capability's state survived into the stored turn.
    turn_id = session.last_turn_id
    turn = await session.get_turn(turn_id) if turn_id else None
    if turn is not None:
        main_thread = turn.record.snapshot.threads.get("main", {})
        state = (
            main_thread.get("capability_state")
            if isinstance(main_thread, dict)
            else main_thread.capability_state
        )
        print("durable capability state:", state)


if __name__ == "__main__":
    asyncio.run(main())
