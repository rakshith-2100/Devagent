"""delegate: a read-only subagent as a tool.

The child runs in-process with a fresh context and a restricted toolset, and
returns a structured summary. The parent never sees the child's exploration,
which is the context isolation win. One agent owns the edits; children may only
read and report.
"""
from __future__ import annotations

from typing import Optional

from ..policy.rules import READ_ONLY_TOOLS
from .base import ToolContext, ToolError, ToolOutput, tool

DEFAULT_DELEGATE_TOOLS = ["read_file", "list_dir", "grep_repo", "find_files"]


@tool
def delegate(ctx: ToolContext, brief: str, tools: Optional[list[str]] = None, max_steps: int = 15) -> ToolOutput:
    """Hand a bounded, read-only investigation to a subagent and get back a written summary.

    Good uses: "find every place X is constructed and summarise how each is used",
    "work out how the build system runs tests", "review this diff for security
    problems". The subagent cannot edit files or run commands. Write a real
    brief: objective, scope, what is out of scope, and the output shape you want.

    Args:
        brief: The full brief for the subagent. Include objective, scope, out-of-scope, and required output shape.
        tools: Read-only tools to grant. Defaults to read_file, list_dir, grep_repo, find_files.
        max_steps: Step budget for the subagent (3-40). Defaults to 15.
    """
    if not brief or len(brief.strip()) < 40:
        raise ToolError("delegate: the brief is too short to act on. Give objective, scope, out-of-scope and the output shape.")
    requested = list(tools) if tools else list(DEFAULT_DELEGATE_TOOLS)
    bad = sorted(set(requested) - set(READ_ONLY_TOOLS))
    if bad:
        raise ToolError(
            f"delegate: only read-only tools may be delegated; not allowed: {', '.join(bad)}. "
            f"Allowed: {', '.join(sorted(READ_ONLY_TOOLS))}."
        )
    max_steps = max(3, min(int(max_steps), 40))
    rt = ctx.runtime
    if rt is None or getattr(rt, "delegate", None) is None:
        raise ToolError("delegate: subagents are not enabled in this run (set DEVAGENT_ENABLE_DELEGATE=1).")
    report = rt.delegate(brief.strip(), requested, max_steps)
    return ToolOutput(report, meta={"tools": requested, "max_steps": max_steps})
