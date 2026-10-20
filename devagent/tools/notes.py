"""append_notes: context offloading into NOTES.md in the worktree."""
from __future__ import annotations

from .base import ToolContext, ToolError, ToolOutput, tool


@tool
def append_notes(ctx: ToolContext, text: str) -> ToolOutput:
    """Append a note to NOTES.md in the working root.

    Use it for facts you have established, approaches that failed and why, and
    where you are in the task. Notes are shown to you at the start of every
    step and survive context compaction.

    Args:
        text: The note to append, in Markdown. Be specific: paths, names, exact errors.
    """
    text = (text or "").strip()
    if not text:
        raise ToolError("append_notes: text is empty.")
    if len(text) > 4000:
        raise ToolError("append_notes: keep each note under 4000 characters; summarise and append the rest separately.")
    rt = ctx.runtime
    if rt is None or getattr(rt, "notes", None) is None:
        raise ToolError("append_notes: notes are not available in this context.")
    rt.notes.append(text)
    return ToolOutput(f"Appended to NOTES.md ({rt.notes.line_count()} lines total).")
