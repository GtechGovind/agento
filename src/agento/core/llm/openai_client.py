"""Adapter for the ``openai`` SDK and any OpenAI-compatible endpoint.

Use this instead of :mod:`~agento.core.llm.litellm_client` when your application
only ever talks to one OpenAI-compatible endpoint — OpenAI itself, Azure
OpenAI, vLLM, Ollama's compat layer, OpenRouter, TrueFoundry's gateway, LM
Studio — and you would rather not pull LiteLLM in. It is a smaller dependency
and a shorter code path; it just does not do multi-provider translation.

Install with::

    pip install "agento[openai]"

Then::

    app = agento.Agento(llm=agento.OpenAIClient("gpt-4o"))

    # or against any compatible server
    app = agento.Agento(llm=agento.OpenAIClient(
        "qwen2.5-coder",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    ))

.. note::
   Like the LiteLLM adapter, this file could not be runtime-tested where agento
   was written. Verify it once with ``python scripts/smoke.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ..messages import FinishReason, Usage
from .base import BaseLLM, LLMRequest, ModelProperties, StreamChunk, ToolCallDelta

__all__ = ["OpenAIClient", "OpenAIProvider"]

_VALID_FINISH_REASONS = {"stop", "length", "tool_calls", "content_filter"}


def _import_openai() -> Any:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover - depends on install
        raise ImportError(
            "OpenAIClient requires the 'openai' package. Install it with:\n"
            '    pip install "agento[openai]"'
        ) from exc
    return AsyncOpenAI


def _finish_reason(raw: Any) -> FinishReason | None:
    if not raw:
        return None
    value = str(raw)
    return value if value in _VALID_FINISH_REASONS else "stop"  # type: ignore[return-value]


def _usage(raw: Any) -> Usage | None:
    if raw is None:
        return None
    prompt = int(getattr(raw, "prompt_tokens", 0) or 0)
    completion = int(getattr(raw, "completion_tokens", 0) or 0)
    total = int(getattr(raw, "total_tokens", prompt + completion) or (prompt + completion))

    prompt_details = getattr(raw, "prompt_tokens_details", None)
    completion_details = getattr(raw, "completion_tokens_details", None)
    cached = getattr(prompt_details, "cached_tokens", None) if prompt_details else None
    reasoning = getattr(completion_details, "reasoning_tokens", None) if completion_details else None

    return Usage(
        input_tokens=prompt,
        output_tokens=completion,
        total_tokens=total,
        cache_read_tokens=int(cached) if cached is not None else None,
        reasoning_tokens=int(reasoning) if reasoning is not None else None,
    )


class OpenAIClient(BaseLLM):
    """An :class:`~agento.core.llm.base.LLM` backed by the ``openai`` SDK.

    Args:
        model: The model id as the endpoint knows it.
        api_key: Falls back to ``OPENAI_API_KEY``.
        base_url: Point at a compatible server instead of api.openai.com.
        properties: Model limits. Supply ``context_length`` so context compaction
            can size itself to the real window.
        default_params: Merged beneath each request's own params.
        client: An already-configured ``AsyncOpenAI`` instance, if you have one
            (custom transport, proxy, retry policy).
        extra: Forwarded to the ``AsyncOpenAI`` constructor.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        properties: ModelProperties | None = None,
        default_params: dict[str, Any] | None = None,
        client: Any | None = None,
        **extra: Any,
    ) -> None:
        super().__init__(model, properties)
        if client is None:
            async_openai = _import_openai()
            kwargs: dict[str, Any] = dict(extra)
            if api_key is not None:
                kwargs["api_key"] = api_key
            if base_url is not None:
                kwargs["base_url"] = base_url
            client = async_openai(**kwargs)
        self._client = client
        self._default_params = dict(default_params or {})

    def _build_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {key: value for key, value in message.items() if key != "thinking_blocks"}
                for message in request.messages
            ],
            **self._default_params,
            **request.params,
        }
        if request.tools:
            kwargs["tools"] = request.tools
        if request.response_format:
            kwargs["response_format"] = request.response_format
        return kwargs

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion."""
        kwargs = self._build_kwargs(request)
        kwargs["stream"] = True
        kwargs.setdefault("stream_options", {"include_usage": True})

        stream = await self._client.chat.completions.create(**kwargs)
        async for raw in stream:
            choices = getattr(raw, "choices", None) or []
            choice = choices[0] if choices else None
            delta = getattr(choice, "delta", None)

            tool_calls: list[ToolCallDelta] | None = None
            raw_calls = getattr(delta, "tool_calls", None) if delta else None
            if raw_calls:
                tool_calls = []
                for index, call in enumerate(raw_calls):
                    function = getattr(call, "function", None)
                    tool_calls.append(
                        ToolCallDelta(
                            index=int(getattr(call, "index", index) or 0),
                            id=getattr(call, "id", None),
                            name=getattr(function, "name", None) if function else None,
                            arguments=getattr(function, "arguments", None) if function else None,
                        )
                    )

            chunk = StreamChunk(
                content=getattr(delta, "content", None) if delta else None,
                # Reasoning models on compatible gateways commonly expose this.
                reasoning_content=getattr(delta, "reasoning_content", None) if delta else None,
                tool_calls=tool_calls,
                finish_reason=_finish_reason(getattr(choice, "finish_reason", None) if choice else None),
                usage=_usage(getattr(raw, "usage", None)),
            )
            if not chunk.is_empty():
                yield chunk


class OpenAIProvider:
    """Resolves model names to :class:`OpenAIClient` instances, with caching.

    Args:
        api_key: Applied to every client.
        base_url: Applied to every client.
        properties: Applied to every client when a per-model lookup is not
            available. Set ``context_length`` if all your models share a window.
        default_params: Merged into every request.
        extra: Forwarded to each client.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        properties: ModelProperties | None = None,
        default_params: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._properties = properties
        self._default_params = dict(default_params or {})
        self._extra = extra
        self._cache: dict[str, OpenAIClient] = {}

    def __call__(self, model: str) -> OpenAIClient:
        client = self._cache.get(model)
        if client is None:
            client = OpenAIClient(
                model,
                api_key=self._api_key,
                base_url=self._base_url,
                properties=self._properties,
                default_params=self._default_params,
                **self._extra,
            )
            self._cache[model] = client
        return client
