"""Shared helpers for the test suite.

Deliberately free of pytest fixtures so that every test module runs unchanged
under pytest *and* under ``python tests/run_tests.py``, which needs no
third-party packages at all. That matters more than it sounds: it means the
suite is runnable in a bare checkout, before anything is installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import agento
from agento import ScriptedLLM, say


def build_app(
    script: Sequence[Any] | None = None,
    *,
    llm: ScriptedLLM | None = None,
    on_exhausted: str = "error",
    properties: Any = None,
    **kwargs: Any,
) -> tuple[agento.Agento, ScriptedLLM]:
    """An :class:`agento.Agento` driven by a scripted model.

    Args:
        script: Responses the model returns, in order.
        llm: Use this client instead of building one.
        on_exhausted: ``"error"`` (default), ``"repeat"`` or ``"stop"``.
        properties: Model limits, for compaction tests.
        kwargs: Passed to :class:`agento.Agento` — ``artifacts``, ``skills``,
            ``store``, ``mcp``.

    Returns:
        ``(app, llm)`` — keep the llm to inspect requests or extend the script.
    """
    if llm is None:
        llm = ScriptedLLM(
            list(script or []),
            on_exhausted=on_exhausted,
            properties=properties,
        )
    return agento.Agento(llm=llm, **kwargs), llm


def build_agent(**kwargs: Any) -> agento.Agent:
    """An agent with the noisier capabilities off, so tests assert on one thing.

    Sub-agents, questions and the clock each add tools to the prompt; leaving
    them on by default would make every tool-list assertion incidental.
    """
    config = kwargs.pop("config", None) or agento.RuntimeConfig(
        current_datetime=False,
        ask_user_questions=False,
        sub_agents=agento.SubAgentConfig(enabled=False),
        compaction=agento.CompactionConfig(enabled=False),
        large_tool_response=agento.LargeToolResponseConfig(enabled=False),
    )
    kwargs.setdefault("name", "test-agent")
    kwargs.setdefault("model", "scripted/test-model")
    return agento.Agent(config=config, **kwargs)


async def collect(stream: Any) -> list[Any]:
    """Drain an event stream into a list."""
    return [event async for event in stream]


def kinds(events: Sequence[Any], *, skip_deltas: bool = True) -> list[str]:
    """Event class names, for readable assertions."""
    return [
        type(event).__name__
        for event in events
        if not (skip_deltas and type(event).__name__ == "ModelMessageDelta")
    ]


def of_type(events: Sequence[Any], cls: type) -> list[Any]:
    """Every event of one type."""
    return [event for event in events if isinstance(event, cls)]


def first_of(events: Sequence[Any], cls: type) -> Any:
    """The first event of a type.

    Raises:
        AssertionError: There is none — with the event list in the message, so a
            failure says what actually happened.
    """
    for event in events:
        if isinstance(event, cls):
            return event
    raise AssertionError(f"No {cls.__name__} in stream. Got: {kinds(events)}")


def streamed_text(events: Sequence[Any]) -> str:
    """Concatenate the streamed delta content."""
    return "".join(
        event.content
        for event in events
        if isinstance(event, agento.ModelMessageDelta) and event.content
    )


class Skip(Exception):
    """Raised by :func:`skip` to mark a test as not applicable.

    Defined here rather than in the runner so that both the runner and the test
    modules import the *same* class — a runner executing as ``__main__`` and a
    test importing it by module name would otherwise get two distinct classes,
    and the skip would surface as a failure.
    """


def skip(reason: str) -> None:
    """Skip the current test, e.g. when an optional dependency is missing."""
    import sys

    if "pytest" in sys.modules:
        sys.modules["pytest"].skip(reason)
    raise Skip(reason)


__all__ = [
    "Skip",
    "build_agent",
    "build_app",
    "collect",
    "first_of",
    "kinds",
    "of_type",
    "say",
    "skip",
    "streamed_text",
]
