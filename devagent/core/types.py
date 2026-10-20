"""The only types that cross subsystem boundaries.

Everything else in the codebase is built on these dataclasses. Keep them small,
frozen where they are messages between subsystems, and free of behaviour that
belongs to a particular subsystem.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional

Verdict = Literal["allow", "deny", "approve"]
PlanStatus = Literal["pending", "in_progress", "done", "blocked"]


@dataclass(frozen=True)
class ToolCall:
    """An intention produced by the model. It has no effect until the policy
    engine allows it and the sandbox executes it."""

    id: str
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """What the model sees after a tool call. Truncation is always explicit."""

    call_id: str
    ok: bool
    content: str
    truncated: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    """The policy engine's answer for one ToolCall.

    ``rule`` names the rule that decided; it goes into the trace so that the
    question "which rule let it" is always answerable.
    """

    verdict: Verdict
    rule: str
    reason: str
    tier: int = -1


class RunOutcome(str, Enum):
    COMPLETED = "completed"
    STOPPED_AT_LIMIT = "stopped_at_limit"
    DENIED = "denied"  # policy denials halted progress
    ABORTED = "aborted"  # user pressed ctrl-c or rejected an approval
    FAILED = "failed"  # infrastructure error


@dataclass
class Budget:
    """Four separate limits because they fail for different reasons."""

    max_steps: int = 25
    max_tokens: int = 400_000  # cumulative across the run
    max_cost_usd: float = 1.00
    max_wall_seconds: int = 600


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
        }


@dataclass
class ModelResponse:
    """Provider-neutral model output.

    ``content`` is a list of Anthropic-style blocks::

        {"type": "text", "text": "..."}
        {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
    """

    content: list[dict[str, Any]]
    stop_reason: str  # end_turn | tool_use | max_tokens | <provider specific>
    usage: Usage
    latency_ms: int
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def text(self) -> str:
        return "\n".join(b.get("text", "") for b in self.content if b.get("type") == "text").strip()

    def tool_calls(self) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for b in self.content:
            if b.get("type") == "tool_use":
                calls.append(ToolCall(id=str(b["id"]), name=str(b["name"]), args=dict(b.get("input") or {})))
        return calls


@dataclass
class PlanStep:
    title: str
    status: PlanStatus = "pending"
    note: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"title": self.title, "status": self.status, "note": self.note}


@dataclass
class RunResult:
    outcome: RunOutcome
    final_text: str
    steps: int
    usage: Usage
    cost_usd: float
    wall_seconds: float
    stop_reason: str = ""  # human readable: which limit, which denial, etc.
    tool_calls: int = 0
    tool_errors: int = 0
    tests_passed: Optional[bool] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "final_text": self.final_text,
            "steps": self.steps,
            "usage": self.usage.as_dict(),
            "cost_usd": round(self.cost_usd, 6),
            "wall_seconds": round(self.wall_seconds, 2),
            "stop_reason": self.stop_reason,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "tests_passed": self.tests_passed,
        }
