"""Summarise-and-continue at around 70 per cent of the window.

The compaction prompt insists on one element people omit: approaches already
tried and why they failed. An agent that forgets a failed fix will try it again
immediately.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from ..prompts import render_prompt
from .budget import estimate_tokens

RESULT_PREVIEW_CHARS = 1500


def messages_tokens(messages: Sequence[BaseMessage]) -> int:
    total = 0
    for m in messages:
        total += estimate_tokens(_content_text(m))
        if isinstance(m, AIMessage) and m.tool_calls:
            total += estimate_tokens(m.tool_calls)
    return total


def _content_text(m: BaseMessage) -> str:
    c = m.content
    if isinstance(c, str):
        return c
    parts: list[str] = []
    for block in c:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            parts.append(str(block.get("text") or json.dumps(block, default=str)))
    return "\n".join(parts)


def render_transcript(messages: Sequence[BaseMessage], *, result_preview: int = RESULT_PREVIEW_CHARS) -> str:
    out: list[str] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            out.append(f"[user]\n{_content_text(m)}")
        elif isinstance(m, AIMessage):
            text = _content_text(m)
            if text:
                out.append(f"[assistant]\n{text}")
            for tc in m.tool_calls or []:
                out.append(f"[tool_call {tc.get('name')}] {json.dumps(tc.get('args', {}), ensure_ascii=False, default=str)[:600]}")
        elif isinstance(m, ToolMessage):
            text = _content_text(m)
            if len(text) > result_preview:
                text = text[:result_preview] + f" ... [{len(text) - result_preview} more chars]"
            status = getattr(m, "status", "success")
            out.append(f"[tool_result {status}]\n{text}")
    return "\n\n".join(out)


def split_for_compaction(messages: Sequence[BaseMessage], keep_recent: int) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """Split so that the kept tail starts with an assistant turn (the summary
    becomes the preceding user turn, which keeps alternation valid)."""
    msgs = list(messages)
    if len(msgs) <= keep_recent + 2:
        return [], msgs
    idx = len(msgs) - keep_recent
    while idx > 1 and not isinstance(msgs[idx], AIMessage):
        idx -= 1
    if idx <= 1:
        return [], msgs
    return msgs[:idx], msgs[idx:]


def compact_messages(
    messages: Sequence[BaseMessage],
    *,
    goal: str,
    keep_recent: int,
    summarise: Callable[[str], str],
) -> tuple[list[BaseMessage], dict[str, Any]]:
    """Return (new_messages, info). ``summarise`` is a function from prompt text to summary text."""
    older, recent = split_for_compaction(messages, keep_recent)
    if not older:
        return list(messages), {"compacted": False, "reason": "too few messages to compact"}
    before = messages_tokens(messages)
    prompt = render_prompt("compaction", goal=goal, conversation=render_transcript(older))
    summary = summarise(prompt).strip()
    head = HumanMessage(
        content=(
            "[Context was compacted to save space. What follows is a summary of the work so far; "
            "the most recent exchanges are preserved after it.]\n\n"
            f"{summary}\n\n[Original goal, verbatim:]\n{goal}"
        )
    )
    new_messages: list[BaseMessage] = [head, *recent]
    after = messages_tokens(new_messages)
    return new_messages, {"compacted": True, "before_tokens": before, "after_tokens": after, "dropped_messages": len(older), "summary": summary}
