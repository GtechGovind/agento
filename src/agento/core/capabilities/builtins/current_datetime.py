"""Tells the agent what time it is.

Models have no clock. Left to itself a model will answer "what's today's date?"
from its training data, which is confidently wrong, and will silently miscompute
anything relative — "last quarter", "overdue", "in three days".

One small tool fixes it. On by default, because the failure it prevents is
invisible: nothing errors, the answer is just wrong.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone

from ...tools.base import ToolSet
from ...tools.local import LocalToolSet, tool
from ..base import Capability

__all__ = ["CurrentDateTime"]


@tool(read_only=True, name="get_current_datetime")
async def _get_current_datetime() -> str:
    """Get the current date and time in UTC.

    Call this before any reasoning that depends on the current date — deadlines,
    ages, "recent", "this quarter" — rather than assuming a date.
    """
    now = datetime.now(timezone.utc)
    return json.dumps(
        {
            "iso": now.isoformat().replace("+00:00", "Z"),
            "unix_ms": int(now.timestamp() * 1000),
            "weekday": now.strftime("%A"),
            "timezone": "UTC",
        }
    )


class CurrentDateTime(Capability):
    """Adds ``get_current_datetime``."""

    name = "current_datetime"

    def __init__(self) -> None:
        self._tools = LocalToolSet(
            "datetime",
            [_get_current_datetime],
            description="The current date and time.",
            kind="builtin",
        )

    def tool_sets(self) -> Sequence[ToolSet]:
        return [self._tools]
