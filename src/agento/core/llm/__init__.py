"""Model clients.

:class:`~agento.core.llm.base.LLM` is the interface — two methods, no provider
assumptions. Everything else here is either an adapter for a real provider or a
tool for building one.

Adapters are imported lazily: naming :class:`LiteLLMClient` in this module does
not import ``litellm`` until you actually construct one, so a minimal install
stays minimal and an import error tells you exactly which extra to install.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .accumulate import StreamAccumulator
from .base import LLM, BaseLLM, LLMRequest, LLMResponse, ModelProperties, StreamChunk, ToolCallDelta
from .scripted import ScriptedLLM, ScriptedResponse, ScriptedToolCall, say

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .litellm_client import LiteLLMClient, LiteLLMProvider
    from .openai_client import OpenAIClient, OpenAIProvider

__all__ = [
    "LLM",
    "BaseLLM",
    "LLMRequest",
    "LLMResponse",
    "LiteLLMClient",
    "LiteLLMProvider",
    "ModelProperties",
    "OpenAIClient",
    "OpenAIProvider",
    "ScriptedLLM",
    "ScriptedResponse",
    "ScriptedToolCall",
    "StreamAccumulator",
    "StreamChunk",
    "ToolCallDelta",
    "say",
]

_LAZY = {
    "LiteLLMClient": ".litellm_client",
    "LiteLLMProvider": ".litellm_client",
    "OpenAIClient": ".openai_client",
    "OpenAIProvider": ".openai_client",
}


def __getattr__(name: str) -> Any:
    """Import provider adapters on first use, not at import time."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
