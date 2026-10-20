"""The model judge covers the one thing tests cannot see: cause versus symptom.

Give it the diff, the original failure and the task; ask binary questions with
quoted evidence. Calibrate it against your own labels before trusting it.
"""
from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from ..core.model import _text_of
from ..core.planner import extract_json
from ..prompts import render_prompt

QUESTIONS = ("root_cause", "minimal", "tests_untouched", "honest_report")


def judge_attempt(model: BaseChatModel, *, task: str, original_failure: str, diff: str, report: str) -> dict[str, Any]:
    prompt = render_prompt(
        "judge",
        task=task,
        original_failure=original_failure[-6000:],
        diff=diff[-12000:] if diff.strip() else "(empty diff)",
        report=report or "(no report)",
    )
    response = model.invoke([HumanMessage(content=prompt)], purpose="judge")
    data = extract_json(_text_of(response.content)) or {}
    answers = {}
    for q in QUESTIONS:
        entry = data.get(q) or {}
        if isinstance(entry, dict):
            answers[q] = {"verdict": bool(entry.get("verdict", False)), "evidence": str(entry.get("evidence", ""))}
        else:
            answers[q] = {"verdict": bool(entry), "evidence": ""}
    overall = bool(data.get("overall", answers["root_cause"]["verdict"] and answers["tests_untouched"]["verdict"] and answers["honest_report"]["verdict"]))
    return {"ok": overall, "answers": answers, "notes": str(data.get("notes", "")), "raw": _text_of(response.content)[:2000]}
