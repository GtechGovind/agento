"""Token estimation.

The harness needs token counts in three places, none of which require exactness:

* deciding when to compact the conversation,
* deciding when a tool result is too large to put in context,
* attributing a model call's input tokens across instructions, tools, skills and
  messages so a caller can see where their context window went.

All three are thresholds and reports, not billing. So agento uses ``tiktoken``
when it is installed (accurate for OpenAI-family tokenizers, and close enough
for everything else) and falls back to a characters-per-token heuristic
otherwise. The fallback is intentionally slightly *pessimistic* — it is better to
compact one message early than one message late.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["estimate_tokens", "estimate_tokens_for_json"]

# Empirically ~4 characters per token for English prose across common
# tokenizers; JSON and code trend a little denser, hence the conservative round.
_CHARS_PER_TOKEN = 3.8

_encoder: Any | None = None
_encoder_loaded = False


def _get_encoder() -> Any | None:
    """Load and memoize a tiktoken encoder, or return None if unavailable.

    Loading is deferred to first use because importing tiktoken costs real time
    and many hosts never need it.
    """
    global _encoder, _encoder_loaded
    if _encoder_loaded:
        return _encoder
    _encoder_loaded = True
    try:  # pragma: no cover - depends on an optional dependency
        import tiktoken

        _encoder = tiktoken.get_encoding("o200k_base")
    except Exception:
        _encoder = None
    return _encoder


def estimate_tokens(text: str | None) -> int:
    """Estimate the number of tokens in a string.

    Args:
        text: The text to measure. ``None`` and empty strings count as zero.

    Returns:
        An approximate token count. Never negative.
    """
    if not text:
        return 0
    encoder = _get_encoder()
    if encoder is not None:  # pragma: no cover - optional dependency path
        try:
            return len(encoder.encode(text, disallowed_special=()))
        except Exception:
            pass
    return int(len(text) / _CHARS_PER_TOKEN) + 1


def estimate_tokens_for_json(value: object) -> int:
    """Estimate tokens for an arbitrary JSON-serializable value.

    Used for tool schemas, where what actually reaches the model is the
    serialized JSON rather than any Python representation.
    """
    try:
        return estimate_tokens(json.dumps(value, separators=(",", ":"), default=str))
    except (TypeError, ValueError):
        return estimate_tokens(str(value))
