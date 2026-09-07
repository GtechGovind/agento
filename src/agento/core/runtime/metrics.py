"""Per-thread and per-turn counters.

Two rules make these numbers trustworthy:

* **Counted before yielding.** Usage is folded in before the corresponding event
  leaves the generator, so a consumer that stops mid-stream still gets accurate
  totals for the work that actually happened.
* **Each thread counted once.** When a sub-agent finishes, the orchestrator moves
  its metrics into a "finished" bucket and drops the live thread in the same
  step, so a totals query during a fan-out can never double-count.

Optional fields stay ``None`` when the provider never reported them, which is
different from zero: "this provider does not report cache tokens" and "nothing
was cached" are not the same fact.
"""

from __future__ import annotations

from pydantic import BaseModel

from ..events import TurnMetrics
from ..messages import Usage

__all__ = ["ThreadMetrics"]


def _add_optional(current: int | None, incoming: int | None) -> int | None:
    if current is None and incoming is None:
        return None
    return (current or 0) + (incoming or 0)


def _add_optional_float(current: float | None, incoming: float | None) -> float | None:
    if current is None and incoming is None:
        return None
    return (current or 0.0) + (incoming or 0.0)


class ThreadMetrics(BaseModel):
    """Running totals for one agent thread."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cache_read_tokens: int | None = None
    total_cache_write_tokens: int | None = None
    total_reasoning_tokens: int | None = None
    total_cost_usd: float | None = None
    iterations: int = 0
    total_tool_calls: int = 0
    total_sub_agents: int = 0
    total_compactions: int = 0

    def add_usage(self, usage: Usage) -> None:
        """Fold one model call's usage in."""
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.total_tokens += usage.total_tokens or (usage.input_tokens + usage.output_tokens)
        self.total_cache_read_tokens = _add_optional(self.total_cache_read_tokens, usage.cache_read_tokens)
        self.total_cache_write_tokens = _add_optional(self.total_cache_write_tokens, usage.cache_write_tokens)
        self.total_reasoning_tokens = _add_optional(self.total_reasoning_tokens, usage.reasoning_tokens)
        self.total_cost_usd = _add_optional_float(self.total_cost_usd, usage.cost_usd)

    def add(self, other: ThreadMetrics) -> None:
        """Fold another thread's totals in — used to roll sub-agents up."""
        self.total_input_tokens += other.total_input_tokens
        self.total_output_tokens += other.total_output_tokens
        self.total_tokens += other.total_tokens
        self.total_cache_read_tokens = _add_optional(
            self.total_cache_read_tokens, other.total_cache_read_tokens
        )
        self.total_cache_write_tokens = _add_optional(
            self.total_cache_write_tokens, other.total_cache_write_tokens
        )
        self.total_reasoning_tokens = _add_optional(
            self.total_reasoning_tokens, other.total_reasoning_tokens
        )
        self.total_cost_usd = _add_optional_float(self.total_cost_usd, other.total_cost_usd)
        self.iterations += other.iterations
        self.total_tool_calls += other.total_tool_calls
        self.total_sub_agents += other.total_sub_agents
        self.total_compactions += other.total_compactions

    def to_turn_metrics(self) -> TurnMetrics:
        """Project onto the public, turn-level shape."""
        return TurnMetrics(
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            total_tokens=self.total_tokens,
            total_cache_read_tokens=self.total_cache_read_tokens,
            total_cache_write_tokens=self.total_cache_write_tokens,
            total_reasoning_tokens=self.total_reasoning_tokens,
            total_cost_usd=self.total_cost_usd,
            iterations=self.iterations,
            total_tool_calls=self.total_tool_calls,
            total_sub_agents=self.total_sub_agents,
            total_compactions=self.total_compactions,
        )
