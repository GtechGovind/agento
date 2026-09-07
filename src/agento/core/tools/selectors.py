"""Tool selectors — naming groups of tools without listing them.

An agent that attaches a connector with sixty tools should not have to enumerate
them to say "read-only access only" or "ask me before anything destructive". A
*selector* is either a literal tool name or one of four tags that match on the
tool's annotations:

=================  ===============================================
``@all``           every tool
``@read-only``     ``read_only is True``
``@write``         ``read_only is False`` and not destructive
``@destructive``   ``destructive is True``
=================  ===============================================

Selectors appear in four places on an :class:`~agento.session.agent.MCPServerRef`:

``enable_tools``
    What the agent may use at all. Default ``["@all"]``.
``disable_tools``
    Subtracted from the enabled set. Default none. Subtraction wins, so
    ``enable=["@all"], disable=["delete_repo"]`` does what it looks like.
``preload_tools``
    Which tools' schemas go into the system prompt while the rest stay deferred.
    Default none.
``require_approval_for_tools``
    What pauses for a human. Default ``["@write", "@destructive"]`` — a safe
    default that stays safe as a server adds tools.

**Unannotated tools are exempt from tag matching** (except ``@all``). A server
that ships no annotations therefore gets no automatic approval gate, which is
worth knowing: for such servers, name the tools explicitly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .base import ToolAnnotations

__all__ = [
    "DEFAULT_DISABLE_TOOLS",
    "DEFAULT_ENABLE_TOOLS",
    "DEFAULT_PRELOAD_TOOLS",
    "DEFAULT_REQUIRE_APPROVAL_FOR_TOOLS",
    "TAG_ALL",
    "TAG_DESTRUCTIVE",
    "TAG_READ_ONLY",
    "TAG_WRITE",
    "APPROVAL_TAGS",
    "SELECTION_TAGS",
    "is_tool_allowed",
    "literal_names",
    "matches_any_selector",
    "selectors_include_all",
]

TAG_ALL = "@all"
TAG_READ_ONLY = "@read-only"
TAG_WRITE = "@write"
TAG_DESTRUCTIVE = "@destructive"

_ALL_TAGS = frozenset({TAG_ALL, TAG_READ_ONLY, TAG_WRITE, TAG_DESTRUCTIVE})

SELECTION_TAGS = (TAG_ALL, TAG_READ_ONLY)
"""Tags valid on ``enable_tools`` / ``disable_tools`` / ``preload_tools``."""

APPROVAL_TAGS = (TAG_ALL, TAG_WRITE, TAG_DESTRUCTIVE)
"""Tags valid on ``require_approval_for_tools``."""

DEFAULT_ENABLE_TOOLS: list[str] = [TAG_ALL]
DEFAULT_DISABLE_TOOLS: list[str] = []
DEFAULT_PRELOAD_TOOLS: list[str] = []
DEFAULT_REQUIRE_APPROVAL_FOR_TOOLS: list[str] = [TAG_WRITE, TAG_DESTRUCTIVE]


def _is_tag(selector: str) -> bool:
    return selector in _ALL_TAGS


def _matches_tag(tag: str, annotations: ToolAnnotations | None) -> bool:
    if tag == TAG_ALL:
        return True
    if annotations is None:
        # No annotations means no behavioural claim, so every tag but @all
        # abstains rather than guessing.
        return False
    if tag == TAG_READ_ONLY:
        return annotations.read_only is True
    if tag == TAG_WRITE:
        return annotations.read_only is False and annotations.destructive is not True
    if tag == TAG_DESTRUCTIVE:
        return annotations.destructive is True
    return False


def matches_any_selector(
    tool_name: str,
    annotations: ToolAnnotations | None,
    selectors: Sequence[str] | None,
) -> bool:
    """Whether a tool matches at least one selector.

    Args:
        tool_name: The tool's own name.
        annotations: Its behavioural hints, if any.
        selectors: Tags and/or literal names. Empty or ``None`` matches nothing.
    """
    if not selectors:
        return False
    for selector in selectors:
        if _is_tag(selector):
            if _matches_tag(selector, annotations):
                return True
        elif selector == tool_name:
            return True
    return False


def is_tool_allowed(
    tool_name: str,
    annotations: ToolAnnotations | None,
    enable: Sequence[str],
    disable: Sequence[str],
) -> bool:
    """Apply enable-then-disable. Disable always wins."""
    if not matches_any_selector(tool_name, annotations, enable):
        return False
    return not matches_any_selector(tool_name, annotations, disable)


def literal_names(selectors: Iterable[str] | None) -> list[str]:
    """The non-tag entries of a selector list.

    Used to validate up front that every explicitly named tool actually exists
    on the source — a typo in a tool name should fail loudly at setup rather
    than silently narrow the agent's abilities.
    """
    if not selectors:
        return []
    return [selector for selector in selectors if not _is_tag(selector)]


def selectors_include_all(selectors: Sequence[str] | None) -> bool:
    """Whether ``@all`` is present."""
    return bool(selectors) and TAG_ALL in (selectors or [])
