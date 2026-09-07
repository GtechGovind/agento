"""LiteLLM adapter — the recommended way to talk to a model provider.

LiteLLM already solves the problem agento would otherwise have to solve per
provider: one call shape for OpenAI, Anthropic, Gemini, Bedrock, Vertex, Azure,
Ollama, vLLM, OpenRouter and around a hundred more, with tool calls, reasoning
content, cache tokens and cost normalized on the way out. This module is the
thin translation from LiteLLM's chunk shape into agento's
:class:`~agento.core.llm.base.StreamChunk`.

Install with the extra::

    pip install "agento[litellm]"

Two ways to use it::

    # One fixed model
    app = agento.Agento(llm=agento.LiteLLMClient("openai/gpt-4o"))

    # Resolve whatever model an Agent names
    app = agento.Agento(llm=agento.LiteLLMProvider())
    agent = agento.Agent(model="anthropic/claude-sonnet-4-5", ...)

Credentials follow LiteLLM's own conventions — environment variables such as
``OPENAI_API_KEY`` and ``ANTHROPIC_API_KEY`` — or pass ``api_key`` / ``api_base``
explicitly to point at a gateway.

.. note::
   This file is one of three adapters that could not be runtime-tested where
   agento was written (PyPI was unreachable). It is written defensively —
   every field is read through ``getattr``/``dict`` fallbacks, because LiteLLM
   returns pydantic objects for some providers and plain dicts for others — but
   run ``python scripts/smoke.py`` once against a real key before relying on it.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Callable
from typing import Any

from ..messages import FinishReason, ThinkingBlock, Usage
from .base import BaseLLM, LLMRequest, ModelProperties, StreamChunk, ToolCallDelta

__all__ = ["LiteLLMClient", "LiteLLMProvider"]

_VALID_FINISH_REASONS = {"stop", "length", "tool_calls", "content_filter"}


def _import_litellm() -> Any:
    if sys.version_info < (3, 11):
        raise ImportError("The LiteLLM adapter requires Python 3.11+. On Python 3.10, use the OpenAI adapter.")
    try:
        import litellm
    except ImportError as exc:  # pragma: no cover - depends on install
        raise ImportError(
            "LiteLLMClient requires the 'litellm' package. Install it with:\n"
            '    pip install "agento[litellm]"'
        ) from exc
    return litellm


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """Read the first present attribute or key from a LiteLLM object.

    LiteLLM sometimes yields pydantic models and sometimes plain dicts depending
    on the provider and the code path, so every read goes through here.
    """
    for name in names:
        if obj is None:
            return default
        if isinstance(obj, dict):
            if name in obj and obj[name] is not None:
                return obj[name]
        else:
            value = getattr(obj, name, None)
            if value is not None:
                return value
    return default


def _normalize_finish_reason(raw: Any) -> FinishReason | None:
    if not raw:
        return None
    value = str(raw)
    if value in _VALID_FINISH_REASONS:
        return value  # type: ignore[return-value]
    # Some providers say "max_tokens" / "end_turn" / "STOP"; map onto the
    # canonical set so the loop's truncation handling still fires.
    lowered = value.lower()
    if lowered in {"max_tokens", "max_output_tokens", "token_limit"}:
        return "length"
    if lowered in {"end_turn", "stop_sequence", "complete"}:
        return "stop"
    if lowered in {"tool_use", "function_call"}:
        return "tool_calls"
    return "stop"


def _normalize_usage(raw: Any, cost: float | None = None) -> Usage | None:
    """Translate a LiteLLM usage object into agento's :class:`Usage`."""
    if raw is None and cost is None:
        return None
    if raw is None:
        return Usage(cost_usd=cost)

    prompt = int(_get(raw, "prompt_tokens", "input_tokens", default=0) or 0)
    completion = int(_get(raw, "completion_tokens", "output_tokens", default=0) or 0)
    total = int(_get(raw, "total_tokens", default=prompt + completion) or (prompt + completion))

    prompt_details = _get(raw, "prompt_tokens_details")
    completion_details = _get(raw, "completion_tokens_details")

    cache_read = _get(prompt_details, "cached_tokens")
    if cache_read is None:
        cache_read = _get(raw, "cache_read_input_tokens")
    cache_write = _get(raw, "cache_creation_input_tokens", "_cache_creation_input_tokens")
    reasoning = _get(completion_details, "reasoning_tokens")

    return Usage(
        input_tokens=prompt,
        output_tokens=completion,
        total_tokens=total,
        cache_read_tokens=int(cache_read) if cache_read is not None else None,
        cache_write_tokens=int(cache_write) if cache_write is not None else None,
        reasoning_tokens=int(reasoning) if reasoning is not None else None,
        cost_usd=cost,
    )


def _extract_cost(chunk: Any) -> float | None:
    """Pull LiteLLM's computed response cost off a chunk, if it set one."""
    hidden = getattr(chunk, "_hidden_params", None)
    if isinstance(hidden, dict):
        cost = hidden.get("response_cost")
        if isinstance(cost, (int, float)):
            return float(cost)
    return None


def _extract_thinking(delta: Any) -> list[Any] | None:
    """Collect reasoning blocks, whatever shape the provider used."""
    blocks = _get(delta, "thinking_blocks")
    if not blocks:
        return None
    out: list[Any] = []
    for block in blocks:
        block_type = _get(block, "type", default="thinking")
        if block_type == "thinking":
            text = _get(block, "thinking", "text", default="")
            out.append(ThinkingBlock(thinking=str(text), signature=_get(block, "signature")))
        else:
            # Redacted blocks are opaque; keep them verbatim so they can be
            # replayed to the provider on the next turn.
            out.append(block if isinstance(block, dict) else dict(block))
    return out or None


class LiteLLMClient(BaseLLM):
    """An :class:`~agento.core.llm.base.LLM` bound to one LiteLLM model.

    Args:
        model: A LiteLLM model string — ``"openai/gpt-4o"``,
            ``"anthropic/claude-sonnet-4-5"``, ``"gemini/gemini-2.0-flash"``,
            ``"openai/my-model"`` with an ``api_base`` for a self-hosted endpoint.
        api_key: Overrides the provider's environment variable.
        api_base: Point at a gateway or a self-hosted endpoint.
        properties: Model limits. Looked up from LiteLLM's model registry when
            omitted, which is what lets compaction target the real context window.
        default_params: Parameters merged into every request beneath the agent's
            own — a place for ``temperature`` or ``max_tokens`` defaults.
        extra: Additional keyword arguments passed straight to
            ``litellm.acompletion`` (``custom_llm_provider``, ``api_version``,
            ``vertex_project``, headers, and so on).
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        properties: ModelProperties | None = None,
        default_params: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        self._litellm = _import_litellm()
        super().__init__(model, properties or _lookup_properties(self._litellm, model))
        self._api_key = api_key
        self._api_base = api_base
        self._default_params = dict(default_params or {})
        self._extra = extra

    def _build_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": request.messages,
            **self._extra,
            **self._default_params,
            # The agent's own params win over the client's defaults.
            **request.params,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if request.tools:
            kwargs["tools"] = request.tools
        if request.response_format:
            kwargs["response_format"] = request.response_format
        return kwargs

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Stream a completion through ``litellm.acompletion``."""
        kwargs = self._build_kwargs(request)
        kwargs["stream"] = True
        # Without this most providers omit usage entirely on streamed responses,
        # which would leave metrics and compaction blind.
        kwargs.setdefault("stream_options", {"include_usage": True})

        response = await self._litellm.acompletion(**kwargs)

        async for raw_chunk in response:
            chunk = self._translate(raw_chunk)
            if chunk is not None and not chunk.is_empty():
                yield chunk

    def _translate(self, raw_chunk: Any) -> StreamChunk | None:
        choices = _get(raw_chunk, "choices", default=[]) or []
        choice = choices[0] if choices else None
        delta = _get(choice, "delta")

        tool_calls: list[ToolCallDelta] | None = None
        raw_tool_calls = _get(delta, "tool_calls")
        if raw_tool_calls:
            tool_calls = []
            for index, call in enumerate(raw_tool_calls):
                function = _get(call, "function")
                tool_calls.append(
                    ToolCallDelta(
                        index=int(_get(call, "index", default=index) or 0),
                        id=_get(call, "id"),
                        name=_get(function, "name"),
                        arguments=_get(function, "arguments"),
                    )
                )

        usage = _normalize_usage(_get(raw_chunk, "usage"), _extract_cost(raw_chunk))

        return StreamChunk(
            content=_get(delta, "content"),
            reasoning_content=_get(delta, "reasoning_content"),
            thinking_blocks=_extract_thinking(delta),
            tool_calls=tool_calls,
            finish_reason=_normalize_finish_reason(_get(choice, "finish_reason")),
            usage=usage,
        )

    async def complete(self, request: LLMRequest) -> Any:
        """Non-streaming completion.

        Overrides the base implementation because LiteLLM's non-streaming path
        reports usage more reliably than the streamed one on several providers,
        and compaction cares about that number.
        """
        from ..messages import FunctionCall, RawAssistantMessage, ToolCall
        from .base import LLMResponse

        kwargs = self._build_kwargs(request)
        kwargs["stream"] = False
        response = await self._litellm.acompletion(**kwargs)

        choices = _get(response, "choices", default=[]) or []
        choice = choices[0] if choices else None
        message = _get(choice, "message")

        tool_calls = None
        raw_tool_calls = _get(message, "tool_calls")
        if raw_tool_calls:
            tool_calls = []
            for index, call in enumerate(raw_tool_calls):
                function = _get(call, "function")
                tool_calls.append(
                    ToolCall(
                        id=str(_get(call, "id", default=f"call_{index}")),
                        function=FunctionCall(
                            name=str(_get(function, "name", default="unknown")),
                            arguments=str(_get(function, "arguments", default="{}")),
                        ),
                    )
                )

        assistant = RawAssistantMessage(
            content=_get(message, "content"),
            tool_calls=tool_calls,
            thinking_blocks=_extract_thinking(message),
            reasoning_content=_get(message, "reasoning_content"),
            source=self._model,
        )
        return LLMResponse(
            message=assistant,
            usage=_normalize_usage(_get(response, "usage"), _extract_cost(response)) or Usage(),
            finish_reason=_normalize_finish_reason(_get(choice, "finish_reason")),
        )


def _lookup_properties(litellm_module: Any, model: str) -> ModelProperties:
    """Ask LiteLLM's model registry for the model's limits.

    Best effort: unknown models simply get empty properties, and compaction
    falls back to its fixed threshold.
    """
    try:  # pragma: no cover - depends on the optional dependency
        info = litellm_module.get_model_info(model)
        return ModelProperties(
            context_length=info.get("max_input_tokens") or info.get("max_tokens"),
            max_output_tokens=info.get("max_output_tokens"),
        )
    except Exception:
        return ModelProperties()


class LiteLLMProvider:
    """Resolves any model name to a :class:`LiteLLMClient`.

    Pass one of these as ``Agento(llm=...)`` and every agent's ``model`` string is
    resolved through LiteLLM, so a single application can mix providers freely::

        app = agento.Agento(llm=agento.LiteLLMProvider())
        fast = agento.Agent(model="openai/gpt-4o-mini", ...)
        deep = agento.Agent(model="anthropic/claude-sonnet-4-5", ...)

    Clients are cached per model name, so repeated turns reuse one instance.

    Args:
        api_key: Applied to every client.
        api_base: Applied to every client — point the whole app at a gateway.
        default_params: Merged into every request.
        extra: Forwarded to each :class:`LiteLLMClient`.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        default_params: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        self._api_key = api_key
        self._api_base = api_base
        self._default_params = dict(default_params or {})
        self._extra = extra
        self._cache: dict[str, LiteLLMClient] = {}

    def __call__(self, model: str) -> LiteLLMClient:
        client = self._cache.get(model)
        if client is None:
            client = LiteLLMClient(
                model,
                api_key=self._api_key,
                api_base=self._api_base,
                default_params=self._default_params,
                **self._extra,
            )
            self._cache[model] = client
        return client


# Type alias documenting what Agento accepts for ``llm=``.
LLMResolver = Callable[[str], Any]
