"""Turning tool sets into the tool list a model actually receives.

Between "the agent has three MCP servers and some Python functions" and "here is
a ``tools`` array for the provider" sit three problems:

1. **Names must be legal.** Providers accept ``[a-zA-Z0-9_-]`` and cap the
   length. MCP servers routinely ship names with dots and slashes.
2. **Names must be unique.** Two servers can both expose ``search``. Whoever
   claims the name first keeps it; the second gets a numeric suffix.
3. **Names must be reversible.** When the model calls ``search1``, the loop has
   to know which tool set that was, and what the tool is really called there.

:class:`ToolRegistry` holds both directions of that mapping for the duration of a
turn, alongside the provider-shaped schemas.

Ordering is deterministic — built-in sets first, then user sets, alphabetically,
with each set's tools sorted by name. That matters for more than tidiness: a
stable tool order means a stable prompt prefix, which is what lets providers
serve most of a long conversation from their prompt cache.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any, NamedTuple

from ...errors import ToolNameCollisionError
from ..events import McpServerAuth, McpServerInit
from .base import AuthRequiredOutcome, ToolSchema, ToolSet

__all__ = ["MappedTool", "ToolRegistry", "build_registry", "sanitize_tool_name"]

_VALID_NAME = re.compile(r"^[a-zA-Z0-9_-]+$")
_ILLEGAL_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
MAX_TOOL_NAME_LENGTH = 64
MAX_DUPLICATE_SUFFIX = 50


def sanitize_tool_name(name: str) -> str:
    """Make a tool name safe for a provider's ``tools`` array.

    Replaces illegal characters with underscores and truncates to 64 characters.
    Already-legal names pass through untouched, so the common case keeps the name
    the tool author chose.
    """
    if _VALID_NAME.match(name) and len(name) <= MAX_TOOL_NAME_LENGTH:
        return name
    return _ILLEGAL_CHARS.sub("_", name)[:MAX_TOOL_NAME_LENGTH]


def _unique_name(name: str, taken: set[str]) -> str:
    """A sanitized name not already in ``taken``.

    Raises:
        ToolNameCollisionError: More than 50 tools collide on one name, which
            means something is structurally wrong rather than merely unlucky.
    """
    candidate = sanitize_tool_name(name)
    if candidate not in taken:
        return candidate
    stem = candidate[: MAX_TOOL_NAME_LENGTH - 2]
    for suffix in range(1, MAX_DUPLICATE_SUFFIX + 1):
        attempt = f"{stem}{suffix}"
        if attempt not in taken:
            return attempt
    raise ToolNameCollisionError(
        f"More than {MAX_DUPLICATE_SUFFIX} tools collide on the name {name!r}"
    )


class MappedTool(NamedTuple):
    """Which tool set owns an exposed tool, and what it is called there."""

    tool_set: ToolSet
    original_name: str


class ToolRegistry:
    """The tool surface for one turn.

    Attributes:
        schemas: Provider-shaped tool definitions, ready to put on a request.
        mapping: Exposed name → :class:`MappedTool`.
        initialized: MCP servers that connected while building this registry.
        auth_required: Servers that need authorization before they can be used.
    """

    __slots__ = ("schemas", "mapping", "initialized", "auth_required", "_builtin_sets")

    def __init__(
        self,
        schemas: list[dict[str, Any]],
        mapping: dict[str, MappedTool],
        initialized: list[McpServerInit],
        auth_required: list[McpServerAuth],
        builtin_sets: set[str],
    ) -> None:
        self.schemas = schemas
        self.mapping = mapping
        self.initialized = initialized
        self.auth_required = auth_required
        self._builtin_sets = builtin_sets

    def resolve(self, exposed_name: str) -> MappedTool | None:
        """Look up the tool behind a name the model used."""
        return self.mapping.get(exposed_name)

    def is_builtin(self, tool_set_name: str) -> bool:
        """Whether a set is one of agento's own rather than the host's.

        Used for token attribution: agento's own tools count as harness overhead
        rather than against the agent's declared tools.
        """
        return tool_set_name in self._builtin_sets

    def __len__(self) -> int:
        return len(self.schemas)

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"ToolRegistry({len(self.schemas)} tools)"


def _to_provider_schema(exposed_name: str, schema: ToolSchema, set_name: str) -> dict[str, Any]:
    """Render one tool in the provider's function-tool shape.

    The set name is prefixed onto the description because with several connectors
    attached the model otherwise has no way to tell two similarly-named tools
    apart — and picking the wrong ``search`` is a failure mode that looks like
    the agent being stupid rather than the prompt being ambiguous.
    """
    description = schema.description or ""
    if set_name:
        description = f"mcp server: {set_name}\n{description}".rstrip()

    parameters = dict(schema.input_schema or {})
    parameters.setdefault("type", "object")
    parameters.setdefault("properties", {})

    return {
        "type": "function",
        "function": {
            "name": exposed_name,
            "description": description,
            "parameters": parameters,
        },
    }


async def build_registry(
    *,
    builtin_sets: Sequence[ToolSet] = (),
    user_sets: Sequence[ToolSet] = (),
) -> ToolRegistry:
    """Build the turn's tool registry from the agent's tool sets.

    Args:
        builtin_sets: agento's own sets — ask-user-question, sub-agents,
            skills, deferred tools, and so on. Listed first so they claim their
            natural names before any user tool can take them.
        user_sets: The agent's own sets.

    Returns:
        A :class:`ToolRegistry`. Sets that need authorization contribute nothing
        to ``schemas`` and are reported in ``auth_required`` instead, so the turn
        can end with an actionable ``mcp.auth_required`` rather than an error.

    Sets with nothing preloaded are skipped entirely — no ``list_tools`` call is
    made for them. That is what makes deferred loading cheap: an agent with ten
    connectors attached and none preloaded performs zero network calls at setup.
    """
    schemas: list[dict[str, Any]] = []
    mapping: dict[str, MappedTool] = {}
    initialized: list[McpServerInit] = []
    auth_required: list[McpServerAuth] = []
    taken: set[str] = set()

    ordered = [*_sorted(builtin_sets), *_sorted(s for s in user_sets if s.preload or s.has_preloaded_tools)]

    for tool_set in ordered:
        listing = await tool_set.list_tools()
        if isinstance(listing, AuthRequiredOutcome):
            auth_required.extend(listing.servers)
            continue
        if listing.initialized is not None:
            initialized.append(listing.initialized)

        for schema in sorted(listing.tools, key=lambda item: item.name):
            if not schema.preload:
                # Deferred: discoverable through the deferred-tools interface,
                # but its schema stays out of the prompt.
                continue
            exposed = _unique_name(schema.name, taken)
            taken.add(exposed)
            mapping[exposed] = MappedTool(tool_set, schema.name)
            schemas.append(_to_provider_schema(exposed, schema, tool_set.name))

    return ToolRegistry(
        schemas=schemas,
        mapping=mapping,
        initialized=initialized,
        auth_required=auth_required,
        builtin_sets={s.name for s in builtin_sets},
    )


def _sorted(sets: Iterable[ToolSet]) -> list[ToolSet]:
    return sorted(sets, key=lambda item: item.name)
