"""Persistence.

:class:`~agento.session.store.base.SessionStore` is the contract — twelve methods.
:class:`~agento.session.store.memory.MemorySessionStore` is the reference
implementation; :class:`~agento.session.store.sql.SQLSessionStore` covers SQLite,
Postgres and MySQL through SQLAlchemy and is imported lazily.
"""

from typing import Any

from .base import (
    Page,
    SessionEventItem,
    SessionMetrics,
    SessionRecord,
    SessionStore,
    TurnRecord,
    TurnSnapshot,
)
from .memory import MemorySessionStore

__all__ = [
    "MemorySessionStore",
    "Page",
    "SQLSessionStore",
    "SessionEventItem",
    "SessionMetrics",
    "SessionRecord",
    "SessionStore",
    "TurnRecord",
    "TurnSnapshot",
]


def __getattr__(name: str) -> Any:
    """Import the SQLAlchemy store on first use."""
    if name == "SQLSessionStore":
        from .sql import SQLSessionStore as _SQLSessionStore

        globals()[name] = _SQLSessionStore
        return _SQLSessionStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
