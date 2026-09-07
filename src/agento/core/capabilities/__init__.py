"""Capabilities — the extension point of the agent loop.

:class:`~agento.core.capabilities.base.Capability` is the base class; everything
in :mod:`agento.core.capabilities.builtins` is an instance of it, including the
ones you might assume are built into the loop itself.
"""

from .base import (
    AppendContext,
    Capability,
    CapabilityOutput,
    ContextUsage,
    EmitEvent,
    ExecutionContext,
    ReplaceContext,
    SetState,
)

__all__ = [
    "AppendContext",
    "Capability",
    "CapabilityOutput",
    "ContextUsage",
    "EmitEvent",
    "ExecutionContext",
    "ReplaceContext",
    "SetState",
]
