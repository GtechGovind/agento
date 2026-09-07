"""Delegation — letting the agent spawn focused sub-agents.

The problem this solves is context, not parallelism (though it gives you that
too). Some work is *expensive to watch*: searching the web, reading a large
codebase, checking twelve pull requests. Done inline, every intermediate tool
result lands in the main conversation, and by the time the agent has the answer
its context is full of material it no longer needs — which measurably degrades
the reasoning that follows.

A sub-agent runs that work somewhere else. It gets a fresh context, the same
tools, and one self-contained instruction. It does the searching, the reading,
the twelve lookups — and returns **one summary**. The parent's conversation gains
a paragraph instead of forty tool results.

Two consequences worth designing around:

* A sub-agent **cannot see the conversation**. It gets the ``input`` string and
  nothing else, so that string must carry every fact it needs. The tool
  description says this at length, because a vague delegation is the single most
  common way this feature disappoints.
* A sub-agent **cannot ask the user anything**. Interactive capabilities are not
  given to child threads.

``model_choices`` lets an agent pick a cheaper or stronger model per task::

    SubAgents(model_choices={
        "fast":     "A small, quick model. Use for lookups and extraction.",
        "thorough": "A strong reasoning model. Use for analysis and synthesis.",
    })

The model then chooses a label per delegation, and the session layer maps it to a
real model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ...events import AgentInfo
from ...instructions import InstructionBuilder
from ...messages import InternalToolInfo
from ...tools.base import CreateSubAgentOutcome, ToolSet
from ...tools.local import LocalToolSet, Tool
from ..base import Capability

__all__ = ["SubAgents", "SUB_AGENT_TOOL_NAME", "large_result_delegation_hint"]

SUB_AGENT_TOOL_NAME = "create_sub_agent"


def large_result_delegation_hint() -> str:
    """Guidance shared with the large-tool-response capability.

    Both need to say "delegate this", and saying it in two slightly different
    ways in one prompt is worse than saying it once.
    """
    return (
        "For work that produces a lot of unstructured output — web searches, documentation "
        "lookups, broad scans — delegate to a sub-agent and ask it to return only the relevant "
        "summary, with citations where they aid verification."
    )


_DESCRIPTION_BASE = (
    "Delegate a well-defined piece of work to a sub-agent. Do not delegate the whole task — "
    "delegate a part of it whose intermediate steps you do not need to see.\n\n"
    "The sub-agent has the same tools you do, but it CANNOT see this conversation, the user's "
    "original message, or anything you have already learned. Its entire world is the `input` "
    "string. Write that input as a complete brief: the context it needs, the exact task, any "
    "constraints, what to return, and what has already been done so it does not repeat work.\n\n"
    "The sub-agent returns a single summary. Nothing else from its run reaches you."
)


class _SubAgentToolSet(LocalToolSet):
    """Tool set whose calls create threads rather than return values."""

    async def tool_info(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        resolve_underlying: bool = False,
    ) -> InternalToolInfo:
        info = await super().tool_info(name, arguments, resolve_underlying)
        # The loop reads this flag to leave the parent's tool call open while the
        # child runs, and to know the call must not be auto-repaired.
        return info.model_copy(update={"creates_subagent": True})


class SubAgents(Capability):
    """Adds ``create_sub_agent``.

    Args:
        model_choices: Optional label → description map offered to the model as
            an extra ``model`` argument. The session layer maps the chosen label
            to a real model when building the child.
        max_parallel: Advisory hint included in the prompt. The hard limit is
            :data:`~agento.core.runtime.orchestrator.MAX_PARALLEL_SUB_AGENTS`.
    """

    name = "sub_agents"

    def __init__(
        self,
        *,
        model_choices: dict[str, str] | None = None,
        max_parallel: int = 5,
    ) -> None:
        self._model_choices = dict(model_choices or {})
        self._max_parallel = max_parallel
        self._tools = _SubAgentToolSet(
            "sub_agents",
            [self._build_tool()],
            description="Delegate work to a sub-agent.",
            kind="builtin",
        )

    def _build_tool(self) -> Tool:
        choices = self._model_choices

        if choices:
            listing = "\n".join(f"- {label}: {text}" for label, text in choices.items())
            description = (
                f"{_DESCRIPTION_BASE}\n\nChoose a model for the delegated work:\n{listing}"
            )
            allowed = sorted(choices)

            async def create_sub_agent_with_model(name: str, input: str, model: str) -> Any:
                if model not in allowed:
                    from ...tools.base import error_result

                    return error_result(
                        f"Unknown model {model!r}. Choose one of: {', '.join(allowed)}."
                    )
                return CreateSubAgentOutcome(
                    agent_info=AgentInfo(name=name, input=input, model=model)
                )

            return Tool(
                create_sub_agent_with_model,
                name=SUB_AGENT_TOOL_NAME,
                description=description,
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Short label for this sub-task, e.g. 'pr-review'.",
                        },
                        "input": {
                            "type": "string",
                            "description": "The complete, self-contained brief for the sub-agent.",
                        },
                        "model": {
                            "type": "string",
                            "enum": allowed,
                            "description": "Which model the sub-agent should use.",
                        },
                    },
                    "required": ["name", "input", "model"],
                    "additionalProperties": False,
                },
            )

        async def create_sub_agent(name: str, input: str) -> Any:
            return CreateSubAgentOutcome(agent_info=AgentInfo(name=name, input=input))

        return Tool(
            create_sub_agent,
            name=SUB_AGENT_TOOL_NAME,
            description=_DESCRIPTION_BASE,
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short label for this sub-task, e.g. 'pr-review'.",
                    },
                    "input": {
                        "type": "string",
                        "description": "The complete, self-contained brief for the sub-agent.",
                    },
                },
                "required": ["name", "input"],
                "additionalProperties": False,
            },
        )

    def tool_sets(self) -> Sequence[ToolSet]:
        return [self._tools]

    def build_instructions(self, builder: InstructionBuilder) -> None:
        paragraphs = [
            f"The Agent can delegate work with the {SUB_AGENT_TOOL_NAME} tool. Delegation is for "
            "work whose intermediate steps the Agent does not need to see: many tool calls, broad "
            "exploration, or anything that would otherwise fill the context window.",
            large_result_delegation_hint(),
            f"Up to {self._max_parallel} sub-agents run at once, so independent sub-tasks should "
            "be delegated together rather than one after another.",
            "A sub-agent starts with no knowledge of this conversation. Every delegation must be "
            "self-contained.",
        ]
        builder.add_section("sub-agents", "\n\n".join(paragraphs))
