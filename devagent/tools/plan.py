"""update_plan: the agent's own visible todo list. Lightweight planning without a planner."""
from __future__ import annotations

from ..core.types import PlanStep
from .base import ToolContext, ToolError, ToolOutput, tool

VALID_STATUS = ("pending", "in_progress", "done", "blocked")
STATUS_MARK = {"pending": "[ ]", "in_progress": "[>]", "done": "[x]", "blocked": "[!]"}


def render_plan(steps: list[PlanStep]) -> str:
    if not steps:
        return "(no plan)"
    return "\n".join(
        f"{STATUS_MARK[s.status]} {i}. {s.title}" + (f"  - {s.note}" if s.note else "") for i, s in enumerate(steps, start=1)
    )


def _normalize_status(raw: object) -> str:
    status = str(raw or "pending").strip().lower().replace("-", "_").replace(" ", "_")
    if status == "completed":
        return "done"
    if status in ("inprogress", "doing", "active"):
        return "in_progress"
    return status


@tool
def update_plan(ctx: ToolContext, steps: list[dict]) -> ToolOutput:
    """Replace your visible plan with an updated list of steps.

    Call it when you start a multi-step task and again whenever a step changes
    status or the plan changes shape. Keep at most one step in_progress.

    Args:
        steps: Ordered list of objects with "title" (short imperative text), "status" (one of pending, in_progress, done, blocked) and optional "note".
    """
    if not isinstance(steps, list) or not steps:
        raise ToolError('update_plan: steps must be a non-empty list like [{"title": "Read parser", "status": "in_progress"}].')
    if len(steps) > 30:
        raise ToolError("update_plan: keep the plan under 30 steps; group small items together.")
    parsed: list[PlanStep] = []
    for i, raw in enumerate(steps, start=1):
        if not isinstance(raw, dict):
            raise ToolError(f"update_plan: step {i} is not an object. Each step needs a title and a status.")
        title = str(raw.get("title", "")).strip()
        status = _normalize_status(raw.get("status"))
        if not title:
            raise ToolError(f"update_plan: step {i} has no title.")
        if status not in VALID_STATUS:
            raise ToolError(f"update_plan: step {i} has status '{raw.get('status')}'; use one of {', '.join(VALID_STATUS)}.")
        parsed.append(PlanStep(title=title, status=status, note=str(raw.get("note", "")).strip()))  # type: ignore[arg-type]
    in_progress = sum(1 for s in parsed if s.status == "in_progress")
    if ctx.runtime is not None:
        ctx.runtime.plan = parsed
    text = "Plan updated:\n" + render_plan(parsed)
    if in_progress > 1:
        text += f"\n(note: {in_progress} steps are in_progress; keep it to one so progress is readable)"
    return ToolOutput(text, meta={"steps": len(parsed), "done": sum(1 for s in parsed if s.status == "done")})
