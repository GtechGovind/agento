"""Every exception agento can raise.

All of them derive from :class:`AgentoError`, so a host application can wrap a
whole agent run in one ``except AgentoError`` and still tell the cases apart by
the ``code`` attribute. Errors that a caller is expected to *handle* (bad input,
an unconfigured model) are distinct classes; errors that are genuinely internal
bugs are plain ``RuntimeError``.

The ``code`` values are stable strings, safe to map onto HTTP statuses or your
own error taxonomy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core.events import TurnState

__all__ = [
    "AgentoError",
    "InvalidSendInputError",
    "InvalidFileInputError",
    "ToolNameCollisionError",
    "McpConnectionError",
    "CapabilityStateError",
    "ConfigurationError",
    "ArtifactStoreRequiredError",
    "SessionNotFoundError",
    "SessionAlreadyExistsError",
    "SessionExternalIdConflictError",
    "TurnNotFoundError",
    "TurnAlreadyExistsError",
    "TurnNotRunningError",
    "PreviousTurnRunningError",
    "SessionStoreConflictError",
    "SessionStoreInvariantError",
    "InvalidPageTokenError",
]


class AgentoError(Exception):
    """Base class for everything agento raises.

    Args:
        code: Stable machine-readable identifier for the failure kind.
        message: Human-readable description.
    """

    code: str = "agento_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


# --------------------------------------------------------------------------- #
# Input / configuration                                                        #
# --------------------------------------------------------------------------- #


class InvalidSendInputError(AgentoError):
    """The batch of items sent into a turn is not valid for the thread's state.

    Raised, for example, when a user message arrives while the agent is still
    waiting for a tool approval, or when a batch resolves only some of the
    pending approvals.
    """

    code = "invalid_send_input"


class InvalidFileInputError(AgentoError):
    """A file content part could not be decoded (bad data URI, empty payload,
    path traversal in the filename)."""

    code = "invalid_file_input"


class ToolNameCollisionError(AgentoError):
    """Two tools from different sources sanitize to the same name and the
    de-duplication suffix budget was exhausted."""

    code = "tool_name_collision"


class McpConnectionError(AgentoError):
    """A remote MCP server could not be reached, or refused a call.

    ``status_code`` mirrors the upstream meaning so a host can translate it:
    401 auth, 403 tool not allowed, 422 unprocessable, 502 transport failure.
    """

    code = "mcp_connection_failed"

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class CapabilityStateError(AgentoError):
    """A capability wrote durable state under a key it did not declare, or two
    capabilities declared the same key on one thread."""

    code = "capability_state_error"


class ConfigurationError(AgentoError):
    """The runtime is missing something the agent asked for — an unregistered
    model name, an unknown MCP server, a skill that no source provides."""

    code = "configuration_error"


class ArtifactStoreRequiredError(AgentoError):
    """An operation needed an artifact store and none is configured."""

    code = "artifact_store_required"


# --------------------------------------------------------------------------- #
# Session store                                                                #
# --------------------------------------------------------------------------- #


class SessionStoreError(AgentoError):
    """Base for every persistence failure."""

    code = "session_store_error"


class SessionStoreNotFoundError(SessionStoreError):
    """A row the caller referenced does not exist."""

    code = "not_found"


class SessionStoreConflictError(SessionStoreError):
    """A write lost a race, or violated a uniqueness constraint."""

    code = "conflict"


class SessionStoreInvariantError(SessionStoreError):
    """The store was asked to do something structurally impossible (e.g. append
    to a thread that the turn does not contain). Indicates a bug, not bad input."""

    code = "invariant_violation"


class SessionNotFoundError(SessionStoreNotFoundError):
    """No session with that id."""

    code = "session_not_found"

    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session not found: {session_id}")
        self.session_id = session_id


class SessionAlreadyExistsError(SessionStoreConflictError):
    """A session with that id already exists."""

    code = "session_already_exists"

    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session already exists: {session_id}")
        self.session_id = session_id


class SessionExternalIdConflictError(SessionStoreConflictError):
    """Another session already claims that ``external_id``."""

    code = "session_external_id_conflict"

    def __init__(self, external_id: str) -> None:
        super().__init__(f"external_id already in use: {external_id}")
        self.external_id = external_id


class TurnNotFoundError(SessionStoreNotFoundError):
    """No turn with that id in this session."""

    code = "turn_not_found"

    def __init__(self, turn_id: str) -> None:
        super().__init__(f"Turn not found: {turn_id}")
        self.turn_id = turn_id


class TurnAlreadyExistsError(SessionStoreConflictError):
    """A turn with that id already exists."""

    code = "turn_already_exists"

    def __init__(self, turn_id: str) -> None:
        super().__init__(f"Turn already exists: {turn_id}")
        self.turn_id = turn_id


class TurnNotRunningError(SessionStoreConflictError):
    """The turn reached a terminal state before this write landed.

    Carries the state that actually won, so the caller can surface it instead of
    overwriting a cancellation with a late 'done'.
    """

    code = "turn_not_running"

    def __init__(self, turn_id: str, state: TurnState) -> None:
        super().__init__(f"Turn is no longer running: {turn_id}")
        self.turn_id = turn_id
        self.state = state


class PreviousTurnRunningError(SessionStoreConflictError):
    """A new turn tried to chain from a turn that is still executing. Cancel or
    await the previous turn first."""

    code = "previous_turn_running"

    def __init__(self, turn_id: str) -> None:
        super().__init__(f"Previous turn is still running: {turn_id}")
        self.turn_id = turn_id


class InvalidPageTokenError(SessionStoreError):
    """A pagination cursor is malformed or was issued for a different query."""

    code = "invalid_page_token"
