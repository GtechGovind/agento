"""The smallest useful agento program.

    python examples/01_hello.py

Shows the two ways to run an agent: ``app.run()`` for an answer, and
``turn.stream()`` for the events as they happen.
"""

from __future__ import annotations

import asyncio

from _shared import make_runtime

import agento


async def main() -> None:
    app, model = make_runtime(
        [
            agento.say("An agent harness is the runtime around a model."),
            agento.say("Because running an agent well means streaming, tools, approvals and state."),
        ]
    )

    agent = agento.Agent(
        name="assistant",
        model=model,
        instructions="You are concise. Two sentences at most.",
    )

    # 1. Just give me the answer.
    answer = await app.run(agent, "In one line: what is an agent harness?")
    print("run():", answer, "\n")

    # 2. Or watch it happen, token by token.
    session = await app.sessions.create(agent=agent)
    turn = await session.create_turn("And why would I want one?")

    print("stream(): ", end="", flush=True)
    async for event in turn.stream():
        if isinstance(event, agento.ModelMessageDelta) and event.content:
            print(event.content, end="", flush=True)
    print("\n")

    print("state:  ", turn.state.status)
    print("tokens: ", turn.state.metrics.total_tokens)


if __name__ == "__main__":
    asyncio.run(main())
