"""The think-act-observe loop as a LangGraph ``StateGraph``.

    START -> agent -> policy -> execute -> agent -> ... -> finalize -> END

* ``agent``    calls the model (after budget checks and optional compaction).
* ``policy``   turns every proposed tool call into a Decision; anything that
               needs approval raises a LangGraph ``interrupt()`` so the caller
               can ask the user and resume. This node has no side effects
               because LangGraph re-runs it from the top on resume.
* ``execute``  runs the allowed calls in the sandbox and turns refusals into
               ordinary tool results the model can read.
* ``finalize`` settles the RunOutcome.

The loop produces intentions, never effects: nothing touches a file until the
policy engine has decided and the execute node has run it through the sandbox.
"""
from __future__ import annotations

import contextvars
import time
import traceback
from dataclasses import asdict
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.types import Command, interrupt

from ..context.budget import BudgetTracker, estimate_tokens
from ..context.compaction import compact_messages, messages_tokens
from ..policy.approval import ApprovalHandler, ApprovalRequest, ApprovalResponse, auto_reject, build_request
from ..prompts import render_prompt
from ..trace.promptlog import PromptLogger
from ..trace.tracer import Tracer
from .model import ModelError, _text_of, usage_from_message
from .runtime import AgentRuntime
from .types import Budget, Decision, RunOutcome, RunResult, ToolCall, Usage

MAX_REFUSED_STEPS = 3  # consecutive steps where every call was refused -> DENIED


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    goal: str
    step: int
    outcome: Optional[str]
    stop_reason: str
    final_text: str
    refused_steps: int
    pending: list[dict[str, Any]]
    aborted: bool
    plan: list[dict[str, Any]]


def message_text(m: BaseMessage) -> str:
    return _text_of(m.content)


def _summarise_with(rt: AgentRuntime, prompt: str) -> str:
    """Compaction uses the bare model (no tools) and counts against the budget."""
    response = rt.model.invoke([HumanMessage(content=prompt)], purpose="compaction")
    usage = usage_from_message(response)
    rt.budget.record(usage, rt.cost_of(usage))
    return message_text(response)


def build_agent_graph(rt: AgentRuntime, *, checkpointer: Any = None):
    # ------------------------------------------------------------- agent
    def agent_node(state: AgentState) -> dict[str, Any]:
        step = int(state.get("step", 0)) + 1
        limit = rt.budget.check(about_to_step=True)
        if limit:
            return {"outcome": RunOutcome.STOPPED_AT_LIMIT.value, "stop_reason": limit}

        messages: list[BaseMessage] = list(state.get("messages", []))
        prelude: list[BaseMessage] = []
        last = messages[-1] if messages else None
        if isinstance(last, AIMessage) and last.tool_calls:
            # A previous run stopped before these calls executed; close them so the API accepts the history.
            for tc in last.tool_calls:
                tm = ToolMessage(
                    content="(not executed: the previous run stopped before this call ran)",
                    tool_call_id=tc["id"],
                    name=tc["name"],
                    status="error",
                )
                prelude.append(tm)
                messages.append(tm)

        tool_schemas = rt.registry.schemas()
        fixed = estimate_tokens(rt.static_system) + estimate_tokens(rt.repo_map) + estimate_tokens(tool_schemas)
        replace_all = False
        if messages_tokens(messages) + fixed > rt.config.context_window_tokens * rt.config.compaction_ratio:
            new_messages, info = compact_messages(
                messages,
                goal=state.get("goal", ""),
                keep_recent=rt.config.keep_recent_messages,
                summarise=lambda prompt: _summarise_with(rt, prompt),
            )
            if info.get("compacted"):
                rt.tracer.compaction(
                    step=step,
                    before_tokens=info["before_tokens"],
                    after_tokens=info["after_tokens"],
                    dropped_messages=info["dropped_messages"],
                )
                rt.promptlog.log_compaction(info["before_tokens"], info["after_tokens"], info["summary"])
                rt.observer.on_compaction(info)
                messages = new_messages
                replace_all = True

        rt.observer.on_step(step, rt.purpose)
        system = rt.system_messages()
        breakdown = {
            "system": estimate_tokens(rt.static_system),
            "context": estimate_tokens(system[1].content),
            "tools": estimate_tokens(tool_schemas),
            "history": messages_tokens(messages),
            "largest_tool_result": max((estimate_tokens(m.content) for m in messages if isinstance(m, ToolMessage)), default=0),
        }
        started = time.monotonic()
        response = rt.bound_model().invoke([*system, *messages], purpose=rt.purpose)
        if not isinstance(response, AIMessage):  # pragma: no cover - defensive
            response = AIMessage(content=str(response))
        latency_ms = int(response.response_metadata.get("latency_ms") or (time.monotonic() - started) * 1000)
        usage = usage_from_message(response)
        reported = response.response_metadata.get("cost_usd")  # OpenRouter returns the real price
        cost = float(reported) if reported is not None else rt.cost_of(usage)
        rt.budget.record(usage, cost)
        rt.budget.steps = step
        stop_reason = str(response.response_metadata.get("stop_reason", ""))
        rt.tracer.model_call(
            step=step,
            purpose=rt.purpose,
            in_tokens=usage.input_tokens,
            out_tokens=usage.output_tokens,
            cache_read=usage.cache_read_tokens,
            cache_write=usage.cache_write_tokens,
            latency_ms=latency_ms,
            stop_reason=stop_reason,
            breakdown=breakdown,
            cost_usd=cost,
            tool_calls=len(response.tool_calls),
        )
        text = message_text(response)
        rt.observer.on_assistant(step, text, list(response.tool_calls))

        if replace_all:
            msg_update: list[Any] = [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages, response]
        else:
            msg_update = [*prelude, response]
        out: dict[str, Any] = {"messages": msg_update, "step": step}
        if not response.tool_calls:
            out["outcome"] = RunOutcome.COMPLETED.value
            out["final_text"] = text
            out["stop_reason"] = "model finished" + (" (output hit max_tokens)" if stop_reason == "max_tokens" else "")
        return out

    def route_after_agent(state: AgentState) -> str:
        return "finalize" if state.get("outcome") else "policy"

    # ------------------------------------------------------------ policy
    def policy_node(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {"pending": [], "aborted": False}
        agent_reason = message_text(last)
        pctx = rt.policy_context(state.get("goal", ""))
        pending: list[dict[str, Any]] = []
        asks: list[tuple[int, ApprovalRequest]] = []
        for tc in last.tool_calls:
            call = ToolCall(id=str(tc["id"]), name=str(tc["name"]), args=dict(tc.get("args") or {}))
            decision = rt.policy.decide(call, pctx)
            entry = {"call": {"id": call.id, "name": call.name, "args": call.args}, "decision": asdict(decision), "approval": None}
            if decision.verdict == "approve":
                asks.append((len(pending), build_request(call, decision, agent_reason, rt.sandbox)))
            pending.append(entry)
        aborted = False
        if asks:
            answers = interrupt({"kind": "approvals", "step": state.get("step", 0), "requests": [req.as_dict() for _, req in asks]})
            if not isinstance(answers, (list, tuple)):
                answers = [answers]
            answers = list(answers) + ["reject"] * (len(asks) - len(answers))
            for (idx, _req), ans in zip(asks, answers):
                resp = ApprovalResponse.from_dict(ans)
                pending[idx]["approval"] = resp.as_dict()
                if resp.action == "abort":
                    aborted = True
        return {"pending": pending, "aborted": aborted}

    # ----------------------------------------------------------- execute
    def execute_node(state: AgentState) -> dict[str, Any]:
        step = int(state.get("step", 0))
        pending = state.get("pending", []) or []
        aborted = bool(state.get("aborted"))
        results: list[ToolMessage] = []
        refused = 0
        plan_update: Optional[list[dict[str, Any]]] = None

        def refuse(call: ToolCall, content: str) -> None:
            nonlocal refused
            refused += 1
            rt.tool_errors += 1
            results.append(ToolMessage(content=content, tool_call_id=call.id, name=call.name, status="error"))
            rt.tracer.tool_result(step=step, call_id=call.id, name=call.name, ok=False, size=len(content), truncated=False, duration_ms=0, preview=content)
            rt.promptlog.log_tool_result(call.id, call.name, False, content, False, 0)

        for entry in pending:
            c = entry["call"]
            call = ToolCall(id=c["id"], name=c["name"], args=dict(c.get("args") or {}))
            d = entry["decision"]
            decision = Decision(d["verdict"], d["rule"], d["reason"], d.get("tier", -1))
            rt.tracer.tool_call(step=step, call_id=call.id, name=call.name, args=call.args)
            rt.tracer.decision(step=step, call_id=call.id, verdict=decision.verdict, rule=decision.rule, tier=decision.tier, reason=decision.reason)
            rt.promptlog.log_decision(call.id, call.name, decision.verdict, decision.rule, decision.reason)
            rt.observer.on_tool_call(step, call, decision)

            if aborted:
                refuse(call, "REJECTED: the user aborted the run before this call executed.")
                continue
            if decision.verdict == "deny":
                refuse(call, f"DENIED by policy ({decision.rule}): {decision.reason}")
                continue
            if decision.verdict == "approve":
                resp = ApprovalResponse.from_dict(entry.get("approval") or "reject")
                rt.tracer.approval(step=step, call_id=call.id, action=resp.action, note=resp.note)
                rt.promptlog.log_approval(call.id, resp.action, resp.note)
                if resp.action == "reject":
                    refuse(
                        call,
                        f"REJECTED by the user: {resp.note or 'no reason given'}. Do not retry this exact action; "
                        "adapt your approach, or explain in your final reply why it is required.",
                    )
                    continue
                if resp.action == "edit" and isinstance(resp.args, dict):
                    call = ToolCall(id=call.id, name=call.name, args=dict(resp.args))

            ctx = rt.tool_context()
            ctx.approved = decision.verdict == "approve"  # reached only after the user accepted
            result = rt.registry.execute(call, ctx)
            rt.tool_calls += 1
            if not result.ok:
                rt.tool_errors += 1
            duration = int(result.meta.get("duration_ms", 0))
            rt.tracer.tool_result(
                step=step,
                call_id=call.id,
                name=call.name,
                ok=result.ok,
                size=len(result.content),
                truncated=result.truncated,
                duration_ms=duration,
                preview=result.content,
            )
            rt.promptlog.log_tool_result(call.id, call.name, result.ok, result.content, result.truncated, duration)
            rt.observer.on_tool_result(step, call, result)
            if call.name == "update_plan" and result.ok:
                plan_update = [s.as_dict() for s in rt.plan]
                rt.promptlog.log_plan(plan_update)
                rt.observer.on_plan(rt.plan)
            results.append(
                ToolMessage(content=result.content, tool_call_id=call.id, name=call.name, status="success" if result.ok else "error")
            )

        all_refused = bool(pending) and refused == len(pending)
        refused_steps = int(state.get("refused_steps", 0)) + 1 if all_refused else 0
        out: dict[str, Any] = {"messages": results, "pending": [], "refused_steps": refused_steps, "aborted": False}
        if plan_update is not None:
            out["plan"] = plan_update
        if aborted:
            out["outcome"] = RunOutcome.ABORTED.value
            out["stop_reason"] = "the user aborted at an approval prompt"
        elif refused_steps >= MAX_REFUSED_STEPS:
            out["outcome"] = RunOutcome.DENIED.value
            out["stop_reason"] = f"every proposed action was refused for {refused_steps} consecutive steps"
        return out

    def route_after_execute(state: AgentState) -> str:
        return "finalize" if state.get("outcome") else "agent"

    # ---------------------------------------------------------- finalize
    def finalize_node(state: AgentState) -> dict[str, Any]:
        outcome = state.get("outcome") or RunOutcome.FAILED.value
        final_text = state.get("final_text", "")
        if not final_text:
            for m in reversed(state.get("messages", [])):
                if isinstance(m, AIMessage) and message_text(m).strip():
                    final_text = message_text(m)
                    break
        return {"outcome": outcome, "final_text": final_text, "pending": [], "aborted": False}

    g: StateGraph = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("policy", policy_node)
    g.add_node("execute", execute_node)
    g.add_node("finalize", finalize_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent, {"policy": "policy", "finalize": "finalize"})
    g.add_edge("policy", "execute")
    g.add_conditional_edges("execute", route_after_execute, {"agent": "agent", "finalize": "finalize"})
    g.add_edge("finalize", END)
    # The runtime's checkpointer (SQLite under .devagent/) makes the session resumable;
    # child runtimes (delegates, the planner) have none and stay in memory.
    return g.compile(checkpointer=checkpointer or getattr(rt, "checkpointer", None) or InMemorySaver())


# ------------------------------------------------------------------ driver


def answer_interrupt(payload: Any, approval_handler: ApprovalHandler) -> list[dict[str, Any]]:
    """Turn an approvals interrupt payload into the list of responses the policy node expects."""
    if not isinstance(payload, dict) or payload.get("kind") != "approvals":
        raise RuntimeError(f"Unexpected interrupt payload: {payload!r}")
    answers: list[dict[str, Any]] = []
    aborted = False
    for raw in payload.get("requests", []):
        if aborted:
            answers.append(ApprovalResponse("abort", note="run aborted").as_dict())
            continue
        resp = approval_handler(ApprovalRequest.from_dict(raw))
        if resp.action == "abort":
            aborted = True
        answers.append(resp.as_dict())
    return answers


def reset_run_counters(rt: AgentRuntime) -> None:
    rt.budget = BudgetTracker(rt.config.budget)
    rt.tool_calls = 0
    rt.tool_errors = 0
    rt.tests_passed = None
    rt.test_runs = 0
    rt.consecutive_test_failures = 0


def run_agent(
    rt: AgentRuntime,
    goal: str,
    *,
    approval_handler: ApprovalHandler,
    graph: Any = None,
    thread_id: Optional[str] = None,
    reset_budget: bool = True,
    isolated: bool = False,
) -> RunResult:
    """Drive the graph to completion, answering approval interrupts through ``approval_handler``.

    Never raises on tool failure (those are observations). Infrastructure
    failures become ``RunOutcome.FAILED`` with the error in ``stop_reason``.
    ``isolated=True`` runs in a fresh context so a run started from inside
    another graph's node (a delegate) does not inherit that graph's config.
    """
    if reset_budget:
        reset_run_counters(rt)
    graph = graph or build_agent_graph(rt)
    thread = thread_id or rt.run_id
    config = {"configurable": {"thread_id": thread}, "recursion_limit": rt.config.budget.max_steps * 4 + 20}
    model_id = getattr(rt.model, "model_id", "") or getattr(rt.model, "_llm_type", "")
    rt.tracer.run_start(
        goal=goal,
        repo=str(rt.repo_root),
        root=str(rt.root),
        branch=rt.branch,
        model=str(model_id),
        config=rt.config.redacted(),
        parent_run_id=rt.parent_run_id,
    )
    rt.promptlog.log_goal(goal)
    started = time.monotonic()
    # On a reused thread, a previous run may have stopped with tool calls that never executed.
    # Close them before the new goal so the API accepts the history.
    closers: list[BaseMessage] = []
    try:
        snapshot = graph.get_state(config)
        prior = list((snapshot.values or {}).get("messages", [])) if snapshot else []
    except Exception:  # noqa: BLE001 - a fresh thread or a checkpointer without state
        prior = []
    if prior and isinstance(prior[-1], AIMessage) and prior[-1].tool_calls:
        closers = [
            ToolMessage(
                content="(not executed: the previous run stopped before this call ran)",
                tool_call_id=tc["id"],
                name=tc["name"],
                status="error",
            )
            for tc in prior[-1].tool_calls
        ]
    initial: dict[str, Any] = {
        "messages": [*closers, HumanMessage(content=goal)],
        "goal": goal,
        "step": 0,
        "outcome": None,
        "stop_reason": "",
        "final_text": "",
        "refused_steps": 0,
        "pending": [],
        "aborted": False,
    }

    def invoke(payload: Any) -> dict[str, Any]:
        if isolated:
            return contextvars.Context().run(graph.invoke, payload, config)
        return graph.invoke(payload, config)

    outcome = RunOutcome.FAILED
    stop_reason = ""
    final_text = ""
    try:
        result = invoke(initial)
        while result.get("__interrupt__"):
            answers = answer_interrupt(result["__interrupt__"][0].value, approval_handler)
            result = invoke(Command(resume=answers))
        outcome = RunOutcome(result.get("outcome") or RunOutcome.FAILED.value)
        stop_reason = str(result.get("stop_reason", ""))
        final_text = str(result.get("final_text", ""))
    except KeyboardInterrupt:
        outcome = RunOutcome.ABORTED
        stop_reason = "interrupted with ctrl-c"
    except ModelError as e:
        outcome = RunOutcome.FAILED
        stop_reason = f"model error: {e}"
        rt.tracer.note("model_error", error=str(e))
    except Exception as e:  # noqa: BLE001 - infrastructure failure is reported, not hidden
        outcome = RunOutcome.FAILED
        stop_reason = f"{type(e).__name__}: {e}"
        rt.tracer.note("exception", error=str(e), traceback=traceback.format_exc())

    wall = time.monotonic() - started
    usage = Usage(**rt.budget.usage.as_dict())
    rr = RunResult(
        outcome=outcome,
        final_text=final_text,
        steps=rt.budget.steps,
        usage=usage,
        cost_usd=rt.budget.cost_usd,
        wall_seconds=wall,
        stop_reason=stop_reason,
        tool_calls=rt.tool_calls,
        tool_errors=rt.tool_errors,
        tests_passed=rt.tests_passed,
    )
    rt.tracer.run_end(
        outcome=outcome.value,
        steps=rr.steps,
        cost_usd=rr.cost_usd,
        wall_s=wall,
        tests_passed=rr.tests_passed,
        stop_reason=stop_reason,
        usage=usage.as_dict(),
    )
    rt.promptlog.finish({**rr.as_dict(), "run_id": rt.run_id, "branch": rt.branch, "root": str(rt.root), "files_changed": sorted(rt.files_changed)})
    rt.observer.on_run_end(rr)
    return rr


# ---------------------------------------------------------------- delegate


def make_delegate(rt: AgentRuntime):
    """Build the ``delegate`` hook: a read-only child agent with its own context and a restricted toolset."""

    def _delegate(brief: str, tools: list[str], max_steps: int) -> str:
        rt.delegate_count += 1
        n = rt.delegate_count
        child_id = f"{rt.run_id}-d{n:02d}"
        remaining = Budget(
            max_steps=max_steps,
            max_tokens=max(1000, rt.config.budget.max_tokens - rt.budget.usage.total),
            max_cost_usd=max(0.01, rt.config.budget.max_cost_usd - rt.budget.cost_usd),
            max_wall_seconds=max(30, int(rt.config.budget.max_wall_seconds - rt.budget.elapsed())),
        )
        child = AgentRuntime(
            run_id=child_id,
            repo_root=rt.repo_root,
            root=rt.root,
            branch=rt.branch,
            config=rt.config,
            sandbox=rt.sandbox,
            registry=rt.registry.subset(tools),
            policy=rt.policy,
            tracer=Tracer(rt.tracer.path.with_name(f"{child_id}.jsonl"), child_id),
            promptlog=PromptLogger(rt.promptlog.dir / f"delegate-{n:02d}", enabled=rt.config.log_prompts),
            notes=rt.notes,
            model=rt.model,
            budget=BudgetTracker(remaining),
            static_system=render_prompt("delegate", brief=brief),
            repo_map=rt.repo_map,
            test_hint=rt.test_hint,
            purpose="delegate",
            observer=rt.observer,
            parent_run_id=rt.run_id,
        )
        child.promptlog.log_static(child.static_system, child.registry.schemas())
        rt.tracer.note(f"delegate #{n} started", child_run_id=child_id, tools=tools, max_steps=max_steps)
        rt.observer.on_event(f"delegate #{n}: {brief[:80]}...")
        rr = run_agent(
            child,
            "Carry out the brief in your instructions and reply with the result.",
            approval_handler=auto_reject,
            thread_id=child_id,
            reset_budget=False,
            isolated=True,
        )
        rt.budget.record(rr.usage, rr.cost_usd)
        rt.tracer.note(f"delegate #{n} finished", child_run_id=child_id, outcome=rr.outcome.value, steps=rr.steps, cost_usd=rr.cost_usd)
        header = f"[delegate #{n}: outcome={rr.outcome.value}, steps={rr.steps}, cost=${rr.cost_usd:.4f}]"
        if rr.outcome != RunOutcome.COMPLETED:
            return f"{header}\nThe subagent stopped early ({rr.stop_reason}). Partial report:\n{rr.final_text or '(none)'}"
        return f"{header}\n{rr.final_text}"

    return _delegate
