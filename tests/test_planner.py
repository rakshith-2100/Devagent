from __future__ import annotations

import copy
import json
from pathlib import Path

from devagent.core.config import Config
from devagent.core.planner import extract_json, normalize_plan, run_plan_mode
from devagent.core.session import create_runtime
from devagent.core.types import RunOutcome
from devagent.policy.approval import auto_reject

from .test_graph import FIX_EDIT

PLAN = {
    "summary": "Guard divide against zero.",
    "steps": [{"title": "Add a zero check to divide()", "detail": "raise ValueError in src/calc.py", "files": ["src/calc.py"]}],
    "verification": "run_tests",
    "risks": [],
}
REPLAN_DONE = {"finished": True, "summary": "all done; tests pass", "remaining": [], "reason": "unchanged", "failed_approaches": []}


def fenced(obj) -> dict:
    return {"text": "Here is the plan:\n```json\n" + json.dumps(obj) + "\n```"}


def test_extract_json_variants() -> None:
    assert extract_json("```json\n{\"a\": 1}\n```")["a"] == 1
    assert extract_json("noise {\"a\": {\"b\": 2}} trailing")["a"]["b"] == 2
    assert extract_json("no json here") is None
    plan = normalize_plan({"steps": ["one", {"title": "two", "files": ["x"]}, {"bad": 1}]}, "goal")
    assert [s["title"] for s in plan["steps"]] == ["one", "two"]
    assert normalize_plan(None, "the goal")["steps"][0]["title"] == "the goal"


def test_plan_mode_end_to_end(temp_repo: Path, config: Config) -> None:
    script = [
        {"tool_calls": [{"name": "read_file", "args": {"path": "src/calc.py"}}]},  # planner explores
        fenced(PLAN),  # planner answers
        {"tool_calls": [FIX_EDIT]},  # executor step 1
        {"tool_calls": [{"name": "run_tests", "args": {}}]},
        {"text": "Added the zero check; tests pass."},
        fenced(REPLAN_DONE),  # replanner
    ]
    rt = create_runtime(temp_repo, copy.deepcopy(config), script=script)
    reviews = []

    def review(payload):
        reviews.append(payload)
        return {"action": "accept"}

    rr = run_plan_mode(rt, "Fix the failing divide test", approval_handler=auto_reject, plan_review_handler=review)
    assert rr.outcome == RunOutcome.COMPLETED, rr
    assert reviews and reviews[0]["plan"]["steps"][0]["title"] == "Add a zero check to divide()"
    assert "all done" in rr.final_text and "Added the zero check" in rr.final_text
    assert rr.tests_passed is True
    assert "raise ValueError" in (rt.root / "src" / "calc.py").read_text(encoding="utf-8")
    # the planner child only had read-only tools
    planner_call = rt.model.calls[0]
    assert "planning stage" in str(planner_call[-1].content)
    assert (rt.promptlog.dir / "planner" / "transcript.md").is_file()
    # usage from planner + executor + replanner all counted
    assert rr.usage.input_tokens == 6000
    # executor step prompt carried the plan and the step
    step_goal = str(rt.model.calls[2][-1].content)
    assert "CURRENT STEP: Add a zero check to divide()" in step_goal and "Fix the failing divide test" in step_goal


def test_plan_mode_edit_and_reject(temp_repo: Path, config: Config) -> None:
    script = [fenced(PLAN), {"text": "did the edited step"}, fenced(REPLAN_DONE)]
    rt = create_runtime(temp_repo, copy.deepcopy(config), script=script)
    rr = run_plan_mode(rt, "goal", approval_handler=auto_reject, plan_review_handler=lambda p: {"action": "edit", "steps": ["Only inspect"]})
    assert rr.outcome == RunOutcome.COMPLETED
    assert "CURRENT STEP: Only inspect" in str(rt.model.calls[1][-1].content)

    rt = create_runtime(temp_repo, copy.deepcopy(config), script=[fenced(PLAN)])
    rr = run_plan_mode(rt, "goal", approval_handler=auto_reject, plan_review_handler=lambda p: {"action": "reject"})
    assert rr.outcome == RunOutcome.ABORTED and "rejected" in rr.stop_reason


def test_plan_mode_replanner_adds_a_step(temp_repo: Path, config: Config) -> None:
    replan_more = {"finished": False, "summary": "one more", "remaining": ["Run the full suite"], "reason": "verification missing", "failed_approaches": ["tried X"]}
    script = [fenced(PLAN), {"text": "step one done"}, fenced(replan_more), {"text": "suite green"}, fenced(REPLAN_DONE)]
    rt = create_runtime(temp_repo, copy.deepcopy(config), script=script)
    rr = run_plan_mode(rt, "goal", approval_handler=auto_reject, plan_review_handler=lambda p: {"action": "accept"})
    assert rr.outcome == RunOutcome.COMPLETED
    second_step_goal = str(rt.model.calls[3][-1].content)
    assert "CURRENT STEP: Run the full suite" in second_step_goal and "tried X" in second_step_goal
