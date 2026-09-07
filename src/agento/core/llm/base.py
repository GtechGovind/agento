"""The LLM interface, and the streaming shapes every adapter normalizes to.

agento never talks to a provider directly. It talks to something satisfying
:class:`LLM`, which has exactly two methods. That is the whole contract, and it
is small on purpose: swapping providers, routing through a gateway, adding a
cache, recording and replaying a session for tests, or driving the loop from a
scripted fake are all "write a class with two methods".

**Streaming is the primary path.** Adapters yield :class:`StreamChunk` objects —
normalized deltas — and agento assembles the final assistant message itself with
:class:`~agento.core.llm.accumulate.StreamAccumulator`. Putting assembly in the
core rather than in each adapter means tool-call fragment stitching, reasoning
block collection and usage merging are implemented and tested once, and a new
adapter only has to translate the provider's chunk shape.

**Model identity belongs to the client, not the request.** An :class:`LLM`
instance is already bound to a model, so :class:`LLMRequest` has no ``model``
field. This is what stops callers from inventing a model string that the bound
client would ignore.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ..messages import (
    FinishReason,
    RawAssistantMessage,
    ThinkingBlockUnion,
    Usage,
)

__all__ = [
    "BaseLLM",
    "LLM",
    "LLMRequest",
    "LLMResponse",
    "ModelProperties",
    "StreamChunk",
    "ToolCallDelta",
]


class ModelProperties(BaseModel):
    """What agento knows about the bound model's limits.

    Optional, but worth providing: ``context_length`` is what lets context
    compaction trigger at 80% of the real window instead of falling back to a
    fixed 50,000-token guess.
    """

    context_length: int | None = None
    max_output_tokens: int | None = None


class LLMRequest(BaseModel):
    """One model call.

    Attributes:
        messages: Provider-shaped message dicts, already stripped of agento's
            internal fields by :func:`~agento.core.messages.to_wire_message`.
        tools: OpenAI-style function-tool schemas, or ``None`` when the agent has
            no tools this call.
        response_format: Structured-output spec, passed through untouched
            (``{"type": "json_object"}``, ``{"type": "json_schema", ...}``).
        params: Everything else, forwarded to the provider as-is —
            ``temperature``, ``max_tokens``, ``top_p``, ``reasoning_effort``,
            and any provider-specific keys. agento does not validate these; the
            provider is the authority on what it accepts.
    """

    model_config = ConfigDict(extra="allow")

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    response_format: dict[str, Any] | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class ToolCallDelta(BaseModel):
    """A fragment of a tool call arriving mid-stream.

    Providers stream tool calls piecemeal: the ``id`` and ``name`` arrive in the
    first fragment for a given ``index``, then ``arguments`` accumulates across
    many fragments as the JSON is generated. ``index`` is the join key.
    """

    model_config = ConfigDict(extra="allow")

    index: int = 0
    id: str | None = None
    name: str | None = None
    arguments: str | None = None


class StreamChunk(BaseModel):
    """One normalized delta from a streaming completion.

    Every field is optional; a chunk carries whichever parts the provider sent.
    Adapters should not synthesize empty chunks — if a provider frame contains
    nothing agento cares about, skip it.

    Attributes:
        content: New visible text.
        reasoning_content: New reasoning text, for display.
        thinking_blocks: Complete reasoning blocks, when the provider emits them
            whole. These are retained in context because some providers require
            them (with signatures) on subsequent turns.
        tool_calls: Tool-call fragments.
        finish_reason: Set on the frame that ends the response.
        usage: Token accounting, usually on the final frame.
    """

    model_config = ConfigDict(extra="allow")

    content: str | None = None
    reasoning_content: str | None = None
    thinking_blocks: list[ThinkingBlockUnion] | None = None
    tool_calls: list[ToolCallDelta] | None = None
    finish_reason: FinishReason | None = None
    usage: Usage | None = None

    def is_empty(self) -> bool:
        """True when this chunk carries nothing worth emitting downstream."""
        return (
            not self.content
            and not self.reasoning_content
            and not self.thinking_blocks
            and not self.tool_calls
            and self.finish_reason is None
            and self.usage is None
        )


class LLMResponse(BaseModel):
    """A completed model call.

    Attributes:
        message: The assembled assistant message. Its ``tool_calls`` are *raw* —
            they carry no ``tool_info`` yet, because only the runtime knows which
            tool set each name belongs to. The runtime enriches them immediately
            afterwards.
        usage: Token accounting, zeroed if the provider reported none.
        finish_reason: ``None`` when the provider never said.
    """

    model_config = ConfigDict(extra="allow")

    message: RawAssistantMessage
    usage: Usage = Field(default_factory=Usage)
    finish_reason: FinishReason | None = None


@runtime_checkable
class LLM(Protocol):
    """A model client bound to one model.

    Implement this to plug in a provider agento does not ship an adapter for, to
    route through your own gateway, or to fake a model in tests.
    """

    @property
    def model(self) -> str:
        """Identifier for the bound model. Used only for logging and tracing."""
        ...

    @property
    def properties(self) -> ModelProperties:
        """Known limits of the bound model. Return an empty instance if unknown."""
        ...

    def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Stream a completion.

        Args:
            request: The call to make.

        Yields:
            Normalized deltas, in order.

        Raises:
            Any provider error. The runtime catches these and turns them into an
            error event rather than letting them escape the turn.
        """
        ...

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Run a completion without streaming.

        Used for internal, non-user-facing calls — context compaction is the main
        one. :class:`BaseLLM` provides a working default that drains
        :meth:`stream`, so adapters only override this when the provider has a
        genuinely cheaper non-streaming path.
        """
        ...


class BaseLLM:
    """Convenience base class for adapters.

    Supplies a :meth:`complete` implementation that drains :meth:`stream` through
    the shared accumulator, so an adapter is complete once it implements
    :meth:`stream` and sets ``_model``.

    Args:
        model: Identifier for the bound model.
        properties: Known limits of the model, if any.
    """

    def __init__(self, model: str, properties: ModelProperties | None = None) -> None:
        self._model = model
        self._properties = properties or ModelProperties()

    @property
    def model(self) -> str:
        return self._model

    @property
    def properties(self) -> ModelProperties:
        return self._properties

    def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:  # pragma: no cover - abstract
        raise NotImplementedError

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Drain :meth:`stream` and assemble the result."""
        from .accumulate import StreamAccumulator

        accumulator = StreamAccumulator()
        async for chunk in self.stream(request):
            accumulator.add(chunk)
        return accumulator.result()

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"{type(self).__name__}(model={self._model!r})"
