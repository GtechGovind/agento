"""Capabilities — how behaviour is added to the agent loop.

The loop in :mod:`agento.core.runtime.agent_thread` is deliberately small: call
the model, run the tools, repeat. Everything else agento does — summarizing a
long conversation, offloading a huge tool result, delegating to a sub-agent,
teaching the model to read a skill — is a **capability** hooked onto that loop.

That is what makes agento extensible rather than merely configurable. A
capability is a class with up to five hooks, and your own capability is a
first-class citizen alongside the built-ins.

The hooks, in the order the loop reaches them::

    send()  ─── pre_send ──┐
                           │   repair dangling state before new input lands
    ┌──────────────────────┘
    │
    ├─ pre_llm ──────────── just before each model call
    │                       (compaction lives here)
    ├─ prepare_request ───── last-chance edit of the outgoing message array
    │                       (not stored; applies to this one call only)
    │
    │  ... model call, tool execution ...
    │
    ├─ process_tool_results  results in hand, before they enter context
    │                       (large-result offloading lives here)
    └─ post_tool_call ────── after results are in context

Plus two non-hook contributions:

* :meth:`Capability.tool_sets` — tools the capability provides.
* :meth:`Capability.build_instructions` — a section in the system prompt.

**Hooks yield instructions, they do not mutate.** A hook is an async generator
yielding :data:`CapabilityOutput` values, and the loop applies them. The loop applies each transition in memory, then the turn layer checkpoints
that state before publishing its durable event.

Durable state: set :attr:`Capability.state_key` and yield
:class:`SetState`. The value is persisted per thread and handed back to
:meth:`Capability.load_state` on the next turn — how a capability remembers
something across turns without inventing its own storage.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..events import ContextCompacted
from ..instructions import InstructionBuilder
from ..messages import ContextMessage, Usage
from ..tools.base import ToolSet

__all__ = [
    "AppendContext",
    "Capability",
    "CapabilityOutput",
    "ContextUsage",
    "EmitEvent",
    "ExecutionContext",
    "JsonValue",
    "ReplaceContext",
    "SetState",
]

JsonValue = bool | int | float | str | list[Any] | dict[str, Any] | None
"""Anything a capability may persist. JSON-serializable by definition — durable
state is stored as JSON, so ``None`` clears a value and ``undefined`` does not
exist."""


class ContextUsage(BaseModel):
    """Live size of a thread's context, for the *next* model call.

    Distinct from billing usage: this is the running estimate the loop uses to
    decide when to compact. ``prompt_tokens`` is refreshed from the provider's
    real count after each call and estimated in between.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0

    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def merged_with(self, other: ContextUsage) -> ContextUsage:
        return ContextUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


class ExecutionContext(BaseModel):
    """Read-only view of the thread, handed to every hook.

    Attributes:
        thread_id: ``"main"`` for the root agent; a generated id for a sub-agent.
        context: The thread's messages. Treat as read-only — yield outputs to
            change it.
        usage: Current context size.
        session_id: The owning session.
        turn_id: The current turn.
        metadata: The session's metadata dict, so a capability can read your
            application's identifiers.
        artifacts: The configured artifact store, if any.
        agent_name: The agent's name, when it has one.
        is_sub_agent: Whether this thread is delegated work. Interactive
            capabilities check this — a sub-agent cannot ask the user anything.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    thread_id: str
    context: list[ContextMessage] = Field(default_factory=list)
    usage: ContextUsage = Field(default_factory=ContextUsage)
    session_id: str | None = None
    turn_id: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    artifacts: Any = None
    agent_name: str | None = None
    is_sub_agent: bool = False


class AppendContext(BaseModel):
    """Add messages to the end of the thread's context.

    Args:
        messages: What to append.
        events: Events to emit alongside — for anything the user should see.
        usage: Replacement context size. Omit and the loop estimates the delta.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: str = "append"
    messages: list[ContextMessage] = Field(default_factory=list)
    events: list[Any] = Field(default_factory=list)
    usage: ContextUsage | None = None


class ReplaceContext(BaseModel):
    """Replace the thread's context wholesale.

    Used by compaction, and destructive by nature — the replaced messages are
    gone from the agent's working memory, though the durable event log still has
    every one of them.

    Args:
        messages: The new context.
        usage: Its size. Required, since it cannot be derived from a delta.
        event: The event announcing the change.
        model_usage: Tokens spent producing the replacement, if a model was used.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: str = "replace"
    messages: list[ContextMessage]
    usage: ContextUsage
    event: ContextCompacted | None = None
    model_usage: Usage | None = None


class EmitEvent(BaseModel):
    """Emit an event without touching context."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: str = "event"
    event: Any


class SetState(BaseModel):
    """Persist durable state under this capability's declared key.

    Args:
        key: Must equal the capability's :attr:`Capability.state_key`. The loop
            rejects a key no capability declared, which catches typos and stops
            one capability from writing over another's state.
        value: JSON-serializable. ``None`` clears.
    """

    kind: str = "state"
    key: str
    value: JsonValue = None


CapabilityOutput = (AppendContext | ReplaceContext | EmitEvent | SetState)
"""What a hook may yield."""


async def _empty() -> AsyncIterator[CapabilityOutput]:
    """An async iterator with nothing in it — the default for every hook."""
    return
    yield  # pragma: no cover - unreachable, makes this an async generator


class Capability:
    """Base class for everything that extends the loop.

    Subclass and override only the hooks you need; the rest stay inert. A
    capability that only adds tools overrides :meth:`tool_sets`; one that only
    adds guidance overrides :meth:`build_instructions`.

    A minimal, complete example — a capability that reminds the agent of a
    deadline before every model call::

        class DeadlineReminder(Capability):
            name = "deadline"

            def __init__(self, deadline: str) -> None:
                self._deadline = deadline

            def build_instructions(self, builder):
                builder.add_section("deadline", f"The deadline is {self._deadline}.")

    Attributes:
        name: Identifier, used in logs and errors.
        state_key: Declare a key here to use durable cross-turn state. Keys
            beginning ``agento.`` are reserved for the built-ins.
    """

    name: str = "capability"
    state_key: str | None = None

    # -- static contributions ----------------------------------------------- #

    def tool_sets(self) -> Sequence[ToolSet]:
        """Tool sets this capability provides.

        These are registered before the agent's own, so a built-in keeps its
        natural name if a user tool happens to collide.
        """
        return ()

    def build_instructions(self, builder: InstructionBuilder) -> None:
        """Add a section to the system prompt.

        Called once when the thread is built. Add nothing and no section appears
        — empty sections are dropped, so a conditional capability costs nothing
        on the runs where it has nothing to say.
        """
        return None

    # -- durable state ------------------------------------------------------ #

    def load_state(self, value: JsonValue) -> None:
        """Restore state persisted on a previous turn.

        Called during thread construction, before any hook runs, and only when
        :attr:`state_key` is set and a value exists.
        """
        return None

    # -- hooks -------------------------------------------------------------- #

    def pre_send(self, context: ExecutionContext) -> AsyncIterator[CapabilityOutput]:
        """Before new input is appended to the thread.

        The place to repair state left by a previous turn — a turn cancelled
        mid-tool-call leaves a dangling call that would make the next request
        invalid, and this hook is where that gets fixed.
        """
        return _empty()

    def pre_llm(self, context: ExecutionContext) -> AsyncIterator[CapabilityOutput]:
        """Before every model call.

        Runs on each iteration of the loop, not once per turn. Changes here are
        durable. Context compaction lives here.
        """
        return _empty()

    def prepare_request(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        """Last chance to edit the outgoing message array.

        Unlike :meth:`pre_llm`, this is **ephemeral**: the edit applies to this
        one request and is never stored. Use it for anything that should reach
        the model without becoming part of the conversation — a just-in-time
        reminder, a cache-control marker, redaction.

        Args:
            messages: The provider-shaped messages about to be sent.

        Returns:
            A replacement array, or ``None`` to leave it alone.
        """
        return None

    async def process_tool_results(
        self,
        results: list[Any],
        context: ExecutionContext,
    ) -> list[CapabilityOutput]:
        """Inspect or rewrite tool results before they enter context.

        This hook may **mutate** ``results`` in place — specifically each
        result's ``message.content`` — which is how large-result offloading
        replaces a huge payload with a preview. It must not add or remove
        results: every tool call still needs exactly one message, or the next
        provider request is invalid.

        Args:
            results: :class:`~agento.core.tools.execute.ToolCallResult` objects.
            context: The thread snapshot.

        Returns:
            Outputs to apply, usually just events.
        """
        return []

    def post_tool_call(self, context: ExecutionContext) -> AsyncIterator[CapabilityOutput]:
        """After tool results have entered context, before the next model call."""
        return _empty()

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"{type(self).__name__}(name={self.name!r})"
