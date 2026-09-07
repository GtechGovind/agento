"""Giving an agent tools.

    python examples/02_tools.py

Three things to notice:

* The JSON Schema comes from the type hints, and the per-argument descriptions
  come from the docstring. Those descriptions are what the model reads.
* ``ToolContext`` is injected, not passed by the model, so a tool can know which
  user it is acting for without the model being able to choose.
* ``destructive=True`` turns on the approval gate — see ``03_approval.py``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Literal

from _shared import make_runtime

import agento

ORDERS = {
    "8812": {"id": "8812", "customer": "u_42", "total": 149.00, "status": "delivered"},
    "8813": {"id": "8813", "customer": "u_42", "total": 39.50, "status": "refunded"},
    "9001": {"id": "9001", "customer": "u_99", "total": 12.00, "status": "shipped"},
}


@agento.tool(read_only=True)
async def list_orders(
    ctx: agento.ToolContext,
    status: Literal["delivered", "shipped", "refunded", "any"] = "any",
) -> str:
    """List the current customer's orders.

    Args:
        status: Only orders in this state, or "any" for all of them.
    """
    # The customer comes from the session, never from the model.
    customer = ctx.metadata.get("user_id")
    orders = [
        order
        for order in ORDERS.values()
        if order["customer"] == customer and (status == "any" or order["status"] == status)
    ]
    return json.dumps(orders)


@agento.tool(read_only=True)
async def get_order(order_id: str) -> str:
    """Look up one order by id.

    Args:
        order_id: The order number, e.g. "8812".
    """
    order = ORDERS.get(order_id)
    return json.dumps(order) if order else json.dumps({"error": "no such order"})


async def main() -> None:
    app, model = make_runtime(
        [
            agento.say(tool_calls=[("list_orders", {"status": "any"})]),
            agento.say("You have two orders: 8812 (delivered, $149) and 8813 (refunded, $39.50)."),
        ]
    )

    agent = agento.Agent(
        name="support",
        model=model,
        instructions=(
            "You help customers with their orders. Be specific about amounts and dates. "
            "Never guess an order id — look it up."
        ),
        tools=[list_orders, get_order],
    )

    session = await app.sessions.create(agent=agent, metadata={"user_id": "u_42"})
    turn = await session.create_turn("what have I ordered recently?")

    async for event in turn.stream():
        if isinstance(event, agento.ModelMessage) and event.tool_calls:
            for call in event.tool_calls:
                print(f"→ calling {call.function.name}({call.function.arguments})")
        elif isinstance(event, agento.ToolResult):
            print(f"← {event.content[:100]}")
        elif isinstance(event, agento.ModelMessageDelta) and event.content:
            print(event.content, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
