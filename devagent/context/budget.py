"""Token accounting per component, and the four-limit budget tracker.

Estimates use a characters-per-token heuristic; the actual usage reported by
the model corrects the cumulative numbers after every call.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.types import Budget, Usage

CHARS_PER_TOKEN = 3.6


def estimate_tokens(text: Any) -> int:
    if text is None:
        return 0
    if not isinstance(text, str):
        try:
            text = json.dumps(text, ensure_ascii=False, default=str)
        except Exception:
            text = str(text)
    return max(1, int(len(text) / CHARS_PER_TOKEN)) if text else 0


@dataclass
class BudgetTracker:
    budget: Budget
    started_at: float = field(default_factory=time.monotonic)
    steps: int = 0
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def record(self, usage: Usage, cost_usd: float) -> None:
        self.usage.add(usage)
        self.cost_usd += cost_usd

    def check(self, *, about_to_step: bool = False) -> Optional[str]:
        """Return a human-readable reason if any limit is spent, else None."""
        next_step = self.steps + (1 if about_to_step else 0)
        if next_step > self.budget.max_steps:
            return f"step limit reached ({self.budget.max_steps} steps)"
        if self.usage.total >= self.budget.max_tokens:
            return f"token limit reached ({self.usage.total} >= {self.budget.max_tokens} tokens)"
        if self.cost_usd >= self.budget.max_cost_usd:
            return f"cost limit reached (${self.cost_usd:.3f} >= ${self.budget.max_cost_usd:.2f})"
        if self.elapsed() >= self.budget.max_wall_seconds:
            return f"wall-clock limit reached ({int(self.elapsed())}s >= {self.budget.max_wall_seconds}s)"
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "usage": self.usage.as_dict(),
            "cost_usd": round(self.cost_usd, 6),
            "elapsed_s": round(self.elapsed(), 1),
            "limits": vars(self.budget),
        }
