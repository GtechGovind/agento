"""Shared setup for the examples.

Every example runs with or without an API key:

* With one — ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ``GEMINI_API_KEY`` — it
  talks to a real model through LiteLLM.
* Without one, it falls back to a scripted stand-in, so you can read the output
  and follow the flow before signing up for anything.

Your own code will not need this file; it is here so the examples are runnable
in any environment.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import agento

MODEL_BY_KEY = {
    "OPENAI_API_KEY": "openai/gpt-4o-mini",
    "ANTHROPIC_API_KEY": "anthropic/claude-sonnet-4-5",
    "GEMINI_API_KEY": "gemini/gemini-2.0-flash",
}


def real_model() -> str | None:
    """The first model we have credentials for, or None."""
    if os.environ.get("AGENTO_OFFLINE") == "1":
        return None
    for variable, model in MODEL_BY_KEY.items():
        if os.environ.get(variable):
            return model
    return None


def make_runtime(fallback_script: Sequence[Any] = (), **kwargs: Any) -> tuple[Any, str]:
    """Build an :class:`agento.Agento` and the model name to use.

    Args:
        fallback_script: Responses for the offline stand-in.
        kwargs: Passed through to ``Agento`` — ``artifacts``, ``skills``, ``store``.

    Returns:
        ``(app, model_name)``.
    """
    model = real_model()
    if model is not None:
        return agento.Agento(llm=agento.LiteLLMProvider(), **kwargs), model

    print("[no API key found — using a scripted model so this example still runs]\n")
    llm = agento.ScriptedLLM(list(fallback_script), on_exhausted="stop")
    return agento.Agento(llm=llm, **kwargs), "scripted/demo"
