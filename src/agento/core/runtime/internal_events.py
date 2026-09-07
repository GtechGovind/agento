"""Events that never leave the harness.

:meth:`~agento.core.runtime.agent_thread.AgentThread.execute` yields a mix of
public events (which reach your ``async for``) and the internal ones defined
here, which the orchestrator and the turn handler consume and translate.

They exist because durability and streaming want different shapes. A caller
wants "the model said this". The session store wants "append these two messages
to this thread and update its token count". :class:`AppendContext` is the second
thing; the ``model.message`` event travelling beside it is the first. Keeping
them separate is what lets the turn handler persist state *before* yielding the
event that reflects it — so a crash mid-stream can leave storage ahead of the
consumer, never behind.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..capabilities.base import ContextUsage
from ..events import AgentInfo, ContextCompacted, ModelMessage, ThreadParent
from ..messages import ContextMessage, LLMToolMessage, Usage

__all__ = [
    "AppendContext",
    "CreateSubAgent",
    "InternalEvent",
    "ReplaceContext",
    "SetState",
    "SubAgentCompletion",
    "ThreadFinished",
]


class AppendContext(BaseModel):
    """Append messages to a thread's context.

    Attributes:
        messages: What to append.
        events: Durable events to persist alongside — the assistant message that
            these context messages represent, typically.
        usage: The thread's context size after appending.
        completion: Set when this append is a sub-agent's final message, so a
            resumed turn can replay the child's outcome without re-running it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: str = "internal.context.append"
    thread_id: str
    messages: list[ContextMessage] = Field(default_factory=list)
    events: list[Any] = Field(default_factory=list)
    usage: ContextUsage | None = None
    completion: SubAgentCompletion | None = None


class ReplaceContext(BaseModel):
    """Replace a thread's context wholesale (compaction)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: str = "internal.context.replace"
    thread_id: str
    messages: list[ContextMessage]
    usage: ContextUsage
    event: ContextCompacted | None = None
    model_usage: Usage | None = None


class SetState(BaseModel):
    """Persist a capability's durable state for a thread."""

    type: str = "internal.state.set"
    thread_id: str
    key: str
    value: Any = None


class CreateSubAgent(BaseModel):
    """A thread asked for a sub-agent.

    The orchestrator builds the child and registers it. The parent's tool call
    stays open until the child finishes — that open call *is* the join.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: str = "internal.subagent.create"
    thread_id: str
    tool_call_id: str
    agent_info: AgentInfo


class SubAgentCompletion(BaseModel):
    """A sub-agent's final outcome, and the message that closes the parent's call.

    Stored on the child thread so a turn resumed from storage can replay the
    result rather than re-running the delegated work — which would be both slow
    and, for a sub-agent with side effects, wrong.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: str = "done"
    output: ModelMessage
    error: str | None = None
    send_to_parent: LLMToolMessage


class ThreadFinished(BaseModel):
    """A thread reached a terminal state.

    For a sub-agent the orchestrator delivers ``send_to_parent`` to the parent
    and drops the child. For the root thread this ends the turn.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: str = "internal.thread.finished"
    thread_id: str
    status: str = "done"
    title: str = ""
    output: ModelMessage | None = None
    error: str | None = None
    parent: ThreadParent | None = None
    send_to_parent: LLMToolMessage | None = None


InternalEvent = (AppendContext | ReplaceContext | SetState | CreateSubAgent | ThreadFinished)

AppendContext.model_rebuild()
