"""Plan mode (``devagent --plan``) as a LangGraph graph.

    START -> plan -> review (interrupt) -> execute_step -> replan -> execute_step ... -> END

Plan mode exists for human oversight, not speed: the review step is the
feature. A planner run (read-only tools) produces a numbered plan, the user
approves or edits it, an executor run works through the steps, and a replanner
call after each step decides whether to continue, change the plan, or stop.
The replanner must preserve the goal verbatim and carry forward which
approaches already failed.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Optional, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ..context.budget import BudgetTracker
from ..policy.approval import ApprovalHandler, auto_reject
from ..policy.rules import READ_ONLY_TOOLS
from ..prompts import render_prompt
from ..trace.promptlog import PromptLogger
from ..trace.tracer import Tracer
from .graph import message_text, run_agent
from .model import usage_from_message
from .runtime import AgentRuntime
from .types import Budget, RunOutcome, RunResult, Usage

PlanReviewHandler = Callable[[dict[str, Any]], dict[str, Any]]


class PlanState(TypedDict, total=False):
    goal: str
    plan: dict[str, Any]
    aborted: bool
    current: int
    completed: list[dict[str, Any]]
    remaining: list[str]
    failed_approaches: list[str]
    finished: bool
    summary: str
    outcome: str
    stop_reason: str
    iterations: int


_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json(text: str) -> Optional[dict[str, Any]]:
    """First JSON object in a model reply: fenced if present, else the outermost braces."""
    if not text:
        return None
    m = _FENCE.search(text)
    candidates = [m.group(1)] if m else []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for cand in candidates:
        try:
            data = json.loads(cand)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return None


def normalize_plan(data: Optional[dict[str, Any]], goal: str) -> dict[str, Any]:
    steps_in = (data or {}).get("steps") or []
    steps: list[dict[str, Any]] = []
    for s in steps_in:
        if isinstance(s, str):
            steps.append({"title": s, "detail": "", "files": []})
        elif isinstance(s, dict) and s.get("title"):
            steps.append({"title": str(s["title"]), "detail": str(s.get("detail", "")), "files": list(s.get("files") or [])})
    if not steps:
        steps = [{"title": goal, "detail": "The planner produced no usable steps; executing the goal as a single step.", "files": []}]
    return {
        "summary": str((data or {}).get("summary", "")),
        "steps": steps,
        "verification": str((data or {}).get("verification", "run the test suite")),
        "risks": list((data or {}).get("risks") or []),
    }


def _planner_runtime(rt: AgentRuntime) -> AgentRuntime:
    child_id = f"{rt.run_id}-plan"
    return AgentRuntime(
        run_id=child_id,
        repo_root=rt.repo_root,
        root=rt.root,
        branch=rt.branch,
        config=rt.config,
        sandbox=rt.sandbox,
        registry=rt.registry.subset(sorted(n for n in rt.registry.names() if n in READ_ONLY_TOOLS)),
        policy=rt.policy,
        tracer=Tracer(rt.tracer.path.with_name(f"{child_id}.jsonl"), child_id),
        promptlog=PromptLogger(rt.promptlog.dir / "planner", enabled=rt.config.log_prompts),
        notes=rt.notes,
        model=rt.model,
        budget=BudgetTracker(Budget(max_steps=min(12, rt.config.budget.max_steps), max_tokens=rt.config.budget.max_tokens, max_cost_usd=rt.config.budget.max_cost_usd, max_wall_seconds=rt.config.budget.max_wall_seconds)),
        static_system=rt.static_system,
        repo_map=rt.repo_map,
        test_hint=rt.test_hint,
        purpose="planner",
        observer=rt.observer,
        parent_run_id=rt.run_id,
    )


def build_plan_graph(rt: AgentRuntime, *, approval_handler: ApprovalHandler, totals: dict[str, Any], checkpointer: Any = None):
    def add_totals(rr: RunResult) -> None:
        totals["usage"].add(rr.usage)
        totals["cost_usd"] += rr.cost_usd
        totals["steps"] += rr.steps

    def plan_node(state: PlanState) -> dict[str, Any]:
        child = _planner_runtime(rt)
        child.promptlog.log_static(child.static_system, child.registry.schemas())
        rt.observer.on_event("planning (read-only exploration)")
        rr = run_agent(
            child,
            render_prompt("planner", goal=state["goal"]),
            approval_handler=auto_reject,
            thread_id=child.run_id,
            reset_budget=False,
            isolated=True,
        )
        add_totals(rr)
        plan = normalize_plan(extract_json(rr.final_text), state["goal"])
        if rr.outcome != RunOutcome.COMPLETED:
            plan["risks"].append(f"planner stopped early: {rr.stop_reason}")
        return {"plan": plan, "remaining": [s["title"] for s in plan["steps"]], "completed": [], "failed_approaches": [], "current": 0, "iterations": 0}

    def review_node(state: PlanState) -> dict[str, Any]:
        answer = interrupt({"kind": "plan_review", "plan": state["plan"], "goal": state["goal"]})
        answer = answer if isinstance(answer, dict) else {"action": str(answer)}
        action = str(answer.get("action", "reject"))
        if action == "reject":
            return {"aborted": True, "outcome": RunOutcome.ABORTED.value, "stop_reason": "the user rejected the plan"}
        plan = dict(state["plan"])
        if action == "edit" and answer.get("steps"):
            plan = normalize_plan({**plan, "steps": answer["steps"]}, state["goal"])
        return {"plan": plan, "remaining": [s["title"] for s in plan["steps"]], "aborted": False}

    def route_after_review(state: PlanState) -> str:
        return END if state.get("aborted") else "execute_step"

    def execute_step_node(state: PlanState) -> dict[str, Any]:
        remaining = list(state.get("remaining", []))
        if not remaining:
            return {"finished": True, "outcome": RunOutcome.COMPLETED.value, "summary": state.get("summary", "")}
        title = remaining[0]
        detail = next((s.get("detail", "") for s in state["plan"]["steps"] if s.get("title") == title), "")
        completed = list(state.get("completed", []))
        done_lines = "\n".join(f"- {c['title']}: {c['outcome']}" for c in completed) or "(none yet)"
        failed = "\n".join(f"- {f}" for f in state.get("failed_approaches", [])) or "(none)"
        step_goal = (
            f"Overall goal (for context only; do not do more than the current step):\n{state['goal']}\n\n"
            f"Approved plan:\n" + "\n".join(f"{i + 1}. {s['title']}" for i, s in enumerate(state["plan"]["steps"])) + "\n\n"
            f"Steps already completed:\n{done_lines}\n\n"
            f"Approaches already tried that failed (do not repeat):\n{failed}\n\n"
            f"CURRENT STEP: {title}\n{detail}\n\n"
            f"Do only this step. Verify it ({state['plan'].get('verification', 'run the tests')}) when it makes sense, "
            "then stop and report what you did and what you observed."
        )
        rt.observer.on_event(f"executing step {len(completed) + 1}: {title}")
        rr = run_agent(
            rt,
            step_goal,
            approval_handler=approval_handler,
            thread_id=f"{rt.run_id}-step{len(completed) + 1}",
            reset_budget=True,
            isolated=True,
        )
        add_totals(rr)
        completed.append({"title": title, "outcome": rr.outcome.value, "report": rr.final_text, "stop_reason": rr.stop_reason, "tests_passed": rr.tests_passed})
        out: dict[str, Any] = {"completed": completed, "remaining": remaining[1:], "iterations": int(state.get("iterations", 0)) + 1}
        if rr.outcome == RunOutcome.ABORTED:
            out.update(outcome=RunOutcome.ABORTED.value, stop_reason=f"aborted during step '{title}'", finished=False, aborted=True)
        elif rr.outcome == RunOutcome.FAILED:
            out.update(outcome=RunOutcome.FAILED.value, stop_reason=f"step '{title}' failed: {rr.stop_reason}", finished=False, aborted=True)
        return out

    def route_after_step(state: PlanState) -> str:
        if state.get("aborted") or state.get("finished"):
            return END
        return "replan"

    def replan_node(state: PlanState) -> dict[str, Any]:
        completed = state.get("completed", [])
        completed_text = "\n\n".join(f"### {c['title']} (outcome: {c['outcome']})\n{c['report'] or '(no report)'}" for c in completed) or "(none)"
        prompt = render_prompt(
            "replanner",
            goal=state["goal"],
            completed=completed_text,
            failed="\n".join(f"- {f}" for f in state.get("failed_approaches", [])) or "(none)",
            remaining="\n".join(f"- {r}" for r in state.get("remaining", [])) or "(none)",
        )
        response = rt.model.invoke([HumanMessage(content=prompt)], purpose="replanner")
        usage = usage_from_message(response)
        totals["usage"].add(usage)
        totals["cost_usd"] += rt.cost_of(usage)
        data = extract_json(message_text(response)) or {}
        finished = bool(data.get("finished", False))
        remaining = [str(r) for r in (data.get("remaining") or [])] if not finished else []
        failed = [str(f) for f in (data.get("failed_approaches") or state.get("failed_approaches", []))]
        summary = str(data.get("summary", ""))
        rt.tracer.note("replan", finished=finished, remaining=remaining, reason=str(data.get("reason", "")))
        rt.observer.on_event(f"replan: {'finished' if finished else f'{len(remaining)} step(s) remaining'} - {data.get('reason', '')}")
        out: dict[str, Any] = {"finished": finished, "remaining": remaining, "failed_approaches": failed, "summary": summary}
        # Steps added by the replanner need a detail entry so execute_step can find one.
        known = {s["title"] for s in state["plan"]["steps"]}
        new_steps = [{"title": r, "detail": "(added by the replanner)", "files": []} for r in remaining if r not in known]
        if new_steps:
            out["plan"] = {**state["plan"], "steps": state["plan"]["steps"] + new_steps}
        if finished:
            out["outcome"] = RunOutcome.COMPLETED.value
        elif int(state.get("iterations", 0)) >= max(2 * len(state["plan"]["steps"]), 4):
            out.update(finished=True, outcome=RunOutcome.STOPPED_AT_LIMIT.value, stop_reason="replanner kept adding steps; iteration cap reached")
        return out

    def route_after_replan(state: PlanState) -> str:
        return END if state.get("finished") else "execute_step"

    g: StateGraph = StateGraph(PlanState)
    g.add_node("plan", plan_node)
    g.add_node("review", review_node)
    g.add_node("execute_step", execute_step_node)
    g.add_node("replan", replan_node)
    g.add_edge(START, "plan")
    g.add_edge("plan", "review")
    g.add_conditional_edges("review", route_after_review, {END: END, "execute_step": "execute_step"})
    g.add_conditional_edges("execute_step", route_after_step, {END: END, "replan": "replan"})
    g.add_conditional_edges("replan", route_after_replan, {END: END, "execute_step": "execute_step"})
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def run_plan_mode(
    rt: AgentRuntime,
    goal: str,
    *,
    approval_handler: ApprovalHandler,
    plan_review_handler: PlanReviewHandler,
) -> RunResult:
    totals: dict[str, Any] = {"usage": Usage(), "cost_usd": 0.0, "steps": 0}
    graph = build_plan_graph(rt, approval_handler=approval_handler, totals=totals)
    config = {"configurable": {"thread_id": f"{rt.run_id}-planmode"}, "recursion_limit": 200}
    started = time.monotonic()
    rt.tracer.note("plan_mode_start", goal=goal)
    outcome = RunOutcome.FAILED
    stop_reason = ""
    final_text = ""
    try:
        result = graph.invoke({"goal": goal, "completed": [], "remaining": [], "failed_approaches": [], "iterations": 0}, config)
        while result.get("__interrupt__"):
            payload = result["__interrupt__"][0].value
            if isinstance(payload, dict) and payload.get("kind") == "plan_review":
                answer = plan_review_handler(payload)
            else:
                raise RuntimeError(f"Unexpected interrupt in plan mode: {payload!r}")
            result = graph.invoke(Command(resume=answer), config)
        outcome = RunOutcome(result.get("outcome") or RunOutcome.FAILED.value)
        stop_reason = str(result.get("stop_reason", ""))
        reports = "\n\n".join(f"## Step: {c['title']} ({c['outcome']})\n{c['report']}" for c in result.get("completed", []))
        final_text = (result.get("summary") or "").strip()
        final_text = (final_text + "\n\n" + reports).strip() if reports else final_text
    except KeyboardInterrupt:
        outcome, stop_reason = RunOutcome.ABORTED, "interrupted with ctrl-c"
    except Exception as e:  # noqa: BLE001
        outcome, stop_reason = RunOutcome.FAILED, f"{type(e).__name__}: {e}"
        rt.tracer.note("plan_mode_exception", error=str(e))
    rr = RunResult(
        outcome=outcome,
        final_text=final_text,
        steps=int(totals["steps"]),
        usage=totals["usage"],
        cost_usd=float(totals["cost_usd"]),
        wall_seconds=time.monotonic() - started,
        stop_reason=stop_reason,
        tool_calls=rt.tool_calls,
        tool_errors=rt.tool_errors,
        tests_passed=rt.tests_passed,
    )
    rt.tracer.note("plan_mode_end", outcome=outcome.value, cost_usd=rr.cost_usd, steps=rr.steps)
    rt.observer.on_run_end(rr)
    return rr
