"""Human in the loop: pausing before something irreversible.

    python examples/03_approval.py

A tool marked ``destructive=True`` does not run when the model asks for it. The
turn ends with an ``ApprovalRequired`` event and a ``required_actions`` entry;
you decide, and the next turn carries the decision.

The same shape covers ``ask_user_question`` and any client-side tool: the run
pauses, you supply something, the run continues.
"""

from __future__ import annotations

import asyncio

from _shared import make_runtime

import agento

REFUNDED: list[str] = []


@agento.tool(read_only=True)
async def get_order(order_id: str) -> str:
    """Look up an order.

    Args:
        order_id: The order number.
    """
    return f'{{"id": "{order_id}", "total": 149.00, "status": "delivered"}}'


@agento.tool(destructive=True)
async def issue_refund(order_id: str, amount: float) -> str:
    """Refund money to the customer. This cannot be undone.

    Args:
        order_id: The order to refund.
        amount: How much to refund, in the order's currency.
    """
    REFUNDED.append(order_id)
    return f"refunded {amount:.2f} on order {order_id}"


async def main() -> None:
    app, model = make_runtime(
        [
            agento.say(tool_calls=[("issue_refund", {"order_id": "8812", "amount": 149.0})]),
            agento.say("Refunded $149.00 on order 8812. It will arrive in 5-10 business days."),
        ]
    )

    agent = agento.Agent(
        name="support",
        model=model,
        instructions="You handle refunds. Confirm the amount before refunding.",
        tools=[get_order, issue_refund],
    )

    session = await app.sessions.create(agent=agent, metadata={"user_id": "u_42"})

    # --- first turn: the agent asks to refund, and is stopped ---------------
    turn = await session.create_turn("refund order 8812 in full")
    pending: agento.ApprovalRequired | None = None

    async for event in turn.stream():
        if isinstance(event, agento.ApprovalRequired):
            pending = event

    if pending is None:
        print("The agent did not ask to do anything irreversible.")
        return

    print(f"paused: {len(pending.tool_calls)} call(s) awaiting approval")
    print("refunds issued so far:", REFUNDED)  # still empty — nothing ran

    # --- you decide ---------------------------------------------------------
    # In a real application this is a button, a Slack action, a policy check.
    decision = "allow"

    resumed = await session.create_turn(
        [
            agento.ToolApproval(
                thread_id=pending.thread_id,
                tool_call_id=call.id,
                decision=decision,
                reason=None if decision == "allow" else "outside the refund window",
            )
            for call in pending.tool_calls
        ]
    )

    async for event in resumed.stream():
        if isinstance(event, agento.ToolResult):
            print("←", event.content)
        elif isinstance(event, agento.ModelMessageDelta) and event.content:
            print(event.content, end="", flush=True)
    print()
    print("refunds issued:", REFUNDED)


if __name__ == "__main__":
    asyncio.run(main())
