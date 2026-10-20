"""git_diff: so the agent can review its own work before declaring done."""
from __future__ import annotations

from typing import Optional

from .base import ToolContext, ToolError, ToolOutput, tool

DIFF_CAP = 15_000


@tool
def git_diff(ctx: ToolContext, staged: bool = False, path: Optional[str] = None) -> ToolOutput:
    """Show the current changes in the worktree as a unified diff, plus any new untracked files.

    Use it before finishing to review your own change as a reviewer would.

    Args:
        staged: Show staged changes (git diff --cached) instead of unstaged ones.
        path: Restrict the diff to one file or directory.
    """
    args = ["diff", "--no-color", "--no-ext-diff"]
    if staged:
        args.append("--cached")
    if path:
        rel = ctx.sandbox.relpath(ctx.sandbox.resolve(path))
        args += ["--", rel]
    res = ctx.sandbox.run_git(*args)
    if res.returncode != 0:
        raise ToolError(f"git_diff failed: {res.stderr.strip() or 'not a git repository?'}")
    diff = res.stdout
    status = ctx.sandbox.run_git("status", "--porcelain", "--untracked-files=all")
    untracked = [line[3:] for line in status.stdout.splitlines() if line.startswith("??")]
    stat = ctx.sandbox.run_git("diff", "--stat", "--no-color", *(["--cached"] if staged else []))
    parts = []
    if stat.stdout.strip():
        parts.append("Summary:\n" + stat.stdout.rstrip())
    if untracked:
        shown = untracked[:40]
        more = f"\n  ... {len(untracked) - len(shown)} more (add a .gitignore for generated folders)" if len(untracked) > len(shown) else ""
        parts.append("New untracked files (not in the diff below):\n" + "\n".join(f"  {u}" for u in shown) + more)
    truncated = False
    if diff.strip():
        if len(diff) > DIFF_CAP:
            diff = diff[:DIFF_CAP] + f"\n... [diff truncated at {DIFF_CAP} chars; pass path=... to see one file]"
            truncated = True
        parts.append(diff.rstrip())
    if not parts:
        return ToolOutput("No changes in the worktree." + (" (staged)" if staged else ""))
    return ToolOutput("\n\n".join(parts), truncated=truncated, meta={"untracked": len(untracked)})
