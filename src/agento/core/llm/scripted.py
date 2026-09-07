"""A scripted, deterministic LLM for tests and examples.

The agent loop is where all the interesting behaviour lives — approval pauses,
sub-agent fan-out, compaction, offloading, iteration limits, cancellation — and
none of it should need a network call to test. :class:`ScriptedLLM` is handed a
list of responses and returns them in order, so every one of those behaviours
becomes an ordinary deterministic unit test.

It is also the fastest way to understand the loop. This is a complete agent run
with a tool call in it, no API key required::

    llm = ScriptedLLM([
        say(tool_calls=[("get_weather", {"city": "Delhi"})]),
        say("It's 31°C in Delhi."),
    ])

The class is part of the public API (``agento.ScriptedLLM``) precisely so your
own tests can drive your agents this way.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from ..messages import FinishReason, Usage
from .base import BaseLLM, LLMRequest, ModelProperties, StreamChunk, ToolCallDelta

__all__ = ["ScriptedLLM", "ScriptedResponse", "ScriptedToolCall", "say"]


class ScriptedToolCall(BaseModel):
    """One tool call the scripted model should emit.

    Args:
        name: Tool name, as the model would see it after agento's sanitizing.
        arguments: Arguments as a dict (serialized for you) or a raw JSON string
            when you want to test malformed-argument handling.
        id: Call id. Generated if omitted.
    """

    name: str
    arguments: dict[str, Any] | str = Field(default_factory=dict)
    id: str | None = None

    def arguments_json(self) -> str:
        if isinstance(self.arguments, str):
            return self.arguments
        import json

        return json.dumps(self.arguments)


class ScriptedResponse(BaseModel):
    """One complete model response.

    Args:
        content: The visible text, if any.
        tool_calls: Tool calls to emit. A response with tool calls does not end
            the run — the loop executes them and calls the model again, which
            consumes the next script entry.
        reasoning: Reasoning text, streamed as ``reasoning_content``.
        finish_reason: Defaults to ``"tool_calls"`` when tool calls are present,
            otherwise ``"stop"``. Set ``"length"`` to test truncation handling.
        usage: Token accounting. Set ``input_tokens`` high to trip compaction.
        error: When set, streaming raises ``RuntimeError`` with this message —
            for testing provider-failure paths.
    """

    content: str | None = None
    tool_calls: list[ScriptedToolCall] = Field(default_factory=list)
    reasoning: str | None = None
    finish_reason: FinishReason | None = None
    usage: Usage | None = None
    error: str | None = None

    def resolved_finish_reason(self) -> FinishReason:
        if self.finish_reason is not None:
            return self.finish_reason
        return "tool_calls" if self.tool_calls else "stop"


def say(
    content: str | None = None,
    *,
    tool_calls: Sequence[ScriptedToolCall | tuple[str, dict[str, Any]] | str] = (),
    reasoning: str | None = None,
    finish_reason: FinishReason | None = None,
    usage: Usage | None = None,
    error: str | None = None,
) -> ScriptedResponse:
    """Build a :class:`ScriptedResponse` with minimal ceremony.

    Tool calls accept three shorthands::

        say(tool_calls=["ping"])                        # no arguments
        say(tool_calls=[("search", {"q": "agents"})])   # name + arguments
        say(tool_calls=[ScriptedToolCall(name=..., arguments=..., id="fixed")])
    """
    normalized: list[ScriptedToolCall] = []
    for index, call in enumerate(tool_calls):
        if isinstance(call, ScriptedToolCall):
            normalized.append(call)
        elif isinstance(call, str):
            normalized.append(ScriptedToolCall(name=call, id=f"call_{index}"))
        else:
            name, arguments = call
            normalized.append(ScriptedToolCall(name=name, arguments=arguments, id=f"call_{index}"))
    return ScriptedResponse(
        content=content,
        tool_calls=normalized,
        reasoning=reasoning,
        finish_reason=finish_reason,
        usage=usage,
        error=error,
    )


ScriptEntry = ScriptedResponse | str | Callable[[LLMRequest], "ScriptedResponse | str"]
"""A script entry: a response, a bare string (shorthand for text), or a callable
that inspects the request and decides — useful for asserting on what the loop
actually sent."""


class ScriptedLLM(BaseLLM):
    """An :class:`~agento.core.llm.base.LLM` that replays a fixed script.

    Args:
        script: Responses to return, in order.
        model: Reported model name.
        properties: Reported model limits. Set ``context_length`` to control when
            compaction triggers in a test.
        chunk_size: Characters per streamed content chunk. The default of 8
            produces several deltas per response, which exercises the streaming
            path; set it very large to emit text in one chunk.
        on_exhausted: What to do when the script runs out. ``"error"`` (default)
            raises, which surfaces a test that under-specified its script;
            ``"repeat"`` replays the last entry; ``"stop"`` returns an empty
            ``stop`` response.

    Attributes:
        requests: Every :class:`LLMRequest` received, in order. Assert against
            this to check what the loop actually sent — system prompt contents,
            which tools were exposed, whether history was compacted.
    """

    def __init__(
        self,
        script: Sequence[ScriptEntry] | None = None,
        *,
        model: str = "scripted/test-model",
        properties: ModelProperties | None = None,
        chunk_size: int = 8,
        on_exhausted: str = "error",
    ) -> None:
        super().__init__(model, properties or ModelProperties(context_length=200_000))
        self._script: list[ScriptEntry] = list(script or [])
        self._index = 0
        self._chunk_size = max(1, chunk_size)
        self._on_exhausted = on_exhausted
        self.requests: list[LLMRequest] = []

    # -- script management -------------------------------------------------- #

    def push(self, *responses: ScriptEntry) -> ScriptedLLM:
        """Append more responses. Returns ``self`` for chaining."""
        self._script.extend(responses)
        return self

    def reset(self) -> None:
        """Rewind to the start of the script and clear recorded requests."""
        self._index = 0
        self.requests.clear()

    @property
    def call_count(self) -> int:
        """How many model calls have been made."""
        return len(self.requests)

    def _next(self, request: LLMRequest) -> ScriptedResponse:
        if self._index >= len(self._script):
            if self._on_exhausted == "repeat" and self._script:
                entry: ScriptEntry = self._script[-1]
            elif self._on_exhausted == "stop":
                return ScriptedResponse(content="", finish_reason="stop")
            else:
                raise AssertionError(
                    f"ScriptedLLM script exhausted after {self._index} response(s); "
                    "the agent asked for another completion. Add another entry, or pass "
                    "on_exhausted='stop'."
                )
        else:
            entry = self._script[self._index]
            self._index += 1

        if callable(entry):
            entry = entry(request)
        if isinstance(entry, str):
            return ScriptedResponse(content=entry)
        return entry

    # -- LLM protocol ------------------------------------------------------- #

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Replay the next scripted response as a sequence of chunks."""
        self.requests.append(request)
        response = self._next(request)

        if response.error:
            raise RuntimeError(response.error)

        if response.reasoning:
            for piece in _split(response.reasoning, self._chunk_size):
                yield StreamChunk(reasoning_content=piece)

        if response.content:
            for piece in _split(response.content, self._chunk_size):
                yield StreamChunk(content=piece)

        for index, call in enumerate(response.tool_calls):
            # Split like a real provider does: identity first, then the
            # arguments in fragments, so the accumulator's stitching is exercised.
            yield StreamChunk(
                tool_calls=[
                    ToolCallDelta(index=index, id=call.id or f"call_{index}", name=call.name, arguments="")
                ]
            )
            for piece in _split(call.arguments_json(), self._chunk_size):
                yield StreamChunk(tool_calls=[ToolCallDelta(index=index, arguments=piece)])

        usage = response.usage or _estimate_usage(request, response)
        yield StreamChunk(finish_reason=response.resolved_finish_reason(), usage=usage)


def _split(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [text]


def _estimate_usage(request: LLMRequest, response: ScriptedResponse) -> Usage:
    """Plausible token numbers, so metrics and compaction thresholds behave."""
    from ..tokens import estimate_tokens_for_json

    input_tokens = estimate_tokens_for_json(request.messages) + estimate_tokens_for_json(request.tools or [])
    output_tokens = estimate_tokens_for_json(
        {"content": response.content, "tool_calls": [c.model_dump() for c in response.tool_calls]}
    )
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )
