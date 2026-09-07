"""Assembling a complete assistant message out of streaming chunks.

Every adapter yields :class:`~agento.core.llm.base.StreamChunk` deltas; exactly
one piece of code turns those back into a whole message, and this is it. Keeping
assembly here rather than in each adapter means the fiddly parts — stitching
tool-call argument fragments, de-duplicating reasoning blocks, deciding which
``usage`` frame wins — are written and tested once.

The fiddly parts, specifically:

* **Tool calls arrive by index, not by id.** The first fragment for index 0
  carries the id and name; twenty more fragments carry slices of the JSON
  arguments. We key on index and concatenate.
* **Some providers re-send usage.** A later frame with real numbers should
  replace an earlier zeroed one, but a later zeroed frame must not erase real
  numbers. We keep the frame with the largest token total.
* **Reasoning blocks may be re-sent as they grow.** Anthropic streams a thinking
  block and then re-sends it with a signature attached. We keep the last version
  of each block position rather than appending duplicates.
"""

from __future__ import annotations

from typing import Any

from ..messages import (
    FinishReason,
    FunctionCall,
    RawAssistantMessage,
    ToolCall,
    Usage,
)
from .base import LLMResponse, StreamChunk

__all__ = ["StreamAccumulator"]


class _PartialToolCall:
    """A tool call being built up across fragments."""

    __slots__ = ("id", "name", "arguments")

    def __init__(self) -> None:
        self.id: str | None = None
        self.name: str | None = None
        self.arguments: list[str] = []

    def merge(self, id_: str | None, name: str | None, arguments: str | None) -> None:
        # Ids and names arrive once, usually on the first fragment. Never let a
        # later empty value clobber one we already have.
        if id_:
            self.id = id_
        if name:
            self.name = name
        if arguments:
            self.arguments.append(arguments)

    def build(self, fallback_index: int) -> ToolCall:
        return ToolCall(
            id=self.id or f"call_{fallback_index}",
            function=FunctionCall(
                name=self.name or "unknown",
                arguments="".join(self.arguments) or "{}",
            ),
        )


class StreamAccumulator:
    """Folds a stream of chunks into one :class:`LLMResponse`.

    Usage::

        acc = StreamAccumulator()
        async for chunk in llm.stream(request):
            acc.add(chunk)
            ...            # forward the chunk to the client too
        response = acc.result()

    The runtime does exactly this: it forwards each chunk as a delta event *and*
    feeds it here, so the live stream and the durable message come from the same
    source and cannot disagree.
    """

    __slots__ = (
        "_content",
        "_reasoning",
        "_tool_calls",
        "_thinking",
        "_finish_reason",
        "_usage",
        "_source",
    )

    def __init__(self, source: str | None = None) -> None:
        self._content: list[str] = []
        self._reasoning: list[str] = []
        self._tool_calls: dict[int, _PartialToolCall] = {}
        self._thinking: dict[int, Any] = {}
        self._finish_reason: FinishReason | None = None
        self._usage: Usage | None = None
        self._source = source

    def add(self, chunk: StreamChunk) -> None:
        """Fold one chunk in."""
        if chunk.content:
            self._content.append(chunk.content)
        if chunk.reasoning_content:
            self._reasoning.append(chunk.reasoning_content)

        if chunk.thinking_blocks:
            # Position-keyed, so a re-sent block (e.g. the same block with a
            # signature attached) replaces its earlier version instead of
            # appearing twice.
            for index, block in enumerate(chunk.thinking_blocks):
                self._thinking[index] = block

        if chunk.tool_calls:
            for delta in chunk.tool_calls:
                partial = self._tool_calls.setdefault(delta.index, _PartialToolCall())
                partial.merge(delta.id, delta.name, delta.arguments)

        if chunk.finish_reason is not None:
            self._finish_reason = chunk.finish_reason

        if chunk.usage is not None:
            self._usage = _better_usage(self._usage, chunk.usage)

    def result(self) -> LLMResponse:
        """Build the assembled response.

        Safe to call at any point, including after an aborted stream — you get
        whatever had arrived so far. The runtime relies on that when a turn is
        cancelled mid-response.
        """
        tool_calls: list[Any] | None = None
        if self._tool_calls:
            tool_calls = [
                self._tool_calls[index].build(index) for index in sorted(self._tool_calls)
            ]

        content = "".join(self._content)
        reasoning = "".join(self._reasoning)

        message = RawAssistantMessage(
            # None rather than "" for a tool-only completion: several providers
            # reject an empty-string assistant message on replay.
            content=content if content else None,
            tool_calls=tool_calls,
            thinking_blocks=[self._thinking[i] for i in sorted(self._thinking)] or None,
            reasoning_content=reasoning or None,
            source=self._source,
        )

        usage = self._usage or Usage()
        if usage.total_tokens == 0 and (usage.input_tokens or usage.output_tokens):
            usage = usage.model_copy(
                update={"total_tokens": usage.input_tokens + usage.output_tokens}
            )

        return LLMResponse(
            message=message,
            usage=usage,
            finish_reason=self._finish_reason,
        )

    @property
    def has_tool_calls(self) -> bool:
        """Whether any tool call has been seen so far."""
        return bool(self._tool_calls)


def _better_usage(current: Usage | None, incoming: Usage) -> Usage:
    """Pick the more informative of two usage frames.

    Providers differ: some send usage once at the end, some send a zeroed frame
    early and a real one later, some send partial frames throughout. Taking the
    frame with the largest total avoids a late empty frame erasing real numbers,
    while still letting a genuinely later, larger count win.
    """
    if current is None:
        return incoming
    if incoming.total_tokens or incoming.input_tokens or incoming.output_tokens:
        current_total = current.total_tokens or (current.input_tokens + current.output_tokens)
        incoming_total = incoming.total_tokens or (incoming.input_tokens + incoming.output_tokens)
        if incoming_total >= current_total:
            return incoming
    # Even a frame with no token counts can carry cost, which some gateways
    # report separately. Keep it rather than dropping the whole frame.
    if incoming.cost_usd is not None and current.cost_usd is None:
        return current.model_copy(update={"cost_usd": incoming.cost_usd})
    return current
