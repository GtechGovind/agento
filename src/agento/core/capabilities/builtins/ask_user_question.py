"""Lets the agent ask the user a question mid-run.

Without this, an agent facing an ambiguity has two bad options: guess, or stop
and describe the ambiguity in prose and hope the user replies usefully.
``ask_user_question`` gives it a third: pause the run, hand the host a structured
question with mutually exclusive options, and continue with the answer.

The tool is **client-side** — agento never executes it. When the model calls it,
the loop emits :class:`~agento.core.events.ClientToolRequired` and the turn ends
paused. Your application renders the question however it likes (chips, a select,
a Slack block) and starts the next turn with a
:class:`~agento.core.events.ToolReply` carrying the chosen answer::

    async for event in turn.stream():
        if isinstance(event, agento.ClientToolRequired):
            pending = event

    answer = await ask_the_human(...)
    turn = await session.create_turn(input=[
        agento.ToolReply(
            thread_id=pending.thread_id,
            tool_call_id=pending.tool_calls[0].id,
            content=answer,
        )
    ])

The tool description carries the guidance that keeps this from being annoying:
ask only when the answer changes what happens next, make options genuinely
distinct, never add an "Other" option (free text is always available).

Not given to sub-agents — a delegated thread has no user to ask.
"""

from __future__ import annotations

from collections.abc import Sequence

from ...tools.base import ToolSet
from ...tools.client_side import ClientSideToolSet
from ...tools.local import tool
from ..base import Capability

__all__ = ["AskUserQuestion"]


@tool(name="ask_user_question", read_only=True)
async def _ask_user_question(question: str, options: list[str] | None = None) -> str:
    """Ask the user a question when you need a decision before continuing.

    Only ask when the answer meaningfully changes what you do next. If you can
    make a reasonable choice and say what you assumed, do that instead.

    Rules:
    - Give 0-5 options. Every option must be a distinct, mutually exclusive
      outcome that leads somewhere different.
    - The user picks exactly one. If a valid answer is a combination, add it as
      its own option.
    - Never add "Other", "Something else", or an option whose purpose is to let
      the user type their own answer — a free-text box is always shown.
    - If one option is clearly best given what you know, put it first and end its
      text with exactly " (Recommended)".

    Args:
        question: The question, in one clear sentence.
        options: 0-5 mutually exclusive choices.
    """
    raise NotImplementedError("ask_user_question is answered by the host application")


class AskUserQuestion(Capability):
    """Adds the client-side ``ask_user_question`` tool."""

    name = "ask_user_question"

    def __init__(self) -> None:
        self._tools = ClientSideToolSet(
            "ask_user",
            [_ask_user_question],
            description="Ask the user a clarifying question.",
            kind="builtin",
        )

    def tool_sets(self) -> Sequence[ToolSet]:
        return [self._tools]
