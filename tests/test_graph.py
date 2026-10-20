from __future__ import annotations

import copy
import json
from pathlib import Path

from devagent.core.config import Config
from devagent.core.graph import build_agent_graph, make_delegate, run_agent
from devagent.core.session import create_runtime
from devagent.core.types import RunOutcome
from devagent.policy.approval import ApprovalResponse, auto_accept, auto_reject
from devagent.trace.tracer import Tracer

FIX_EDIT = {
    "name": "edit_file",
    "args": {
        "path": "src/calc.py",
        "old_string": "    return a / b\n",
        "new_string": "    if b == 0:\n        raise ValueError('division by zero')\n    return a / b\n",
    },
}
HAPPY_SCRIPT = [
    {"text": "Let me look at the code.", "tool_calls": [{"name": "read_file", "args": {"path": "src/calc.py"}}]},
    {"text": "Fixing divide.", "tool_calls": [FIX_EDIT]},
    {"tool_calls": [{"name": "run_tests", "args": {}}]},
    {"text": "Fixed `divide` in src/calc.py to raise ValueError on zero. Tests pass."},
]


def make_rt(repo: Path, config: Config, script: list, **overrides):
    cfg = copy.deepcopy(config)
    for k, v in overrides.items():
        if hasattr(cfg.budget, k):
            setattr(cfg.budget, k, v)
        else:
            setattr(cfg, k, v)
    return create_runtime(repo, cfg, script=copy.deepcopy(script))


def events_of(rt) -> list[dict]:
    return Tracer.read(rt.tracer.path)


def test_happy_path_edits_in_worktree_only(temp_repo: Path, config: Config) -> None:
    rt = make_rt(temp_repo, config, HAPPY_SCRIPT)
    rr = run_agent(rt, "The divide test is failing. Fix it.", approval_handler=auto_reject)
    assert rr.outcome == RunOutcome.COMPLETED, rr
    assert rr.steps == 4 and rr.tool_calls == 3 and rr.tool_errors == 0
    assert rr.tests_passed is True
    assert "Tests pass" in rr.final_text
    assert rt.files_changed == {"src/calc.py"}
    # isolation: the worktree changed, the user's checkout did not
    assert "raise ValueError" in (rt.root / "src" / "calc.py").read_text(encoding="utf-8")
    assert "raise ValueError" not in (temp_repo / "src" / "calc.py").read_text(encoding="utf-8")
    assert rt.root.parent.name == ".devagent" and rt.branch.startswith("devagent/")
    # trace
    kinds = [e["t"] for e in events_of(rt)]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    assert kinds.count("model_call") == 4 and kinds.count("tool_call") == 3 and kinds.count("tool_result") == 3
    decisions = [e for e in events_of(rt) if e["t"] == "decision"]
    assert all(d["verdict"] == "allow" for d in decisions) and decisions[0]["rule"] == "tier0_default"
    model_calls = [e for e in events_of(rt) if e["t"] == "model_call"]
    assert set(model_calls[0]["breakdown"]) >= {"system", "context", "tools", "history"}
    end = [e for e in events_of(rt) if e["t"] == "run_end"][0]
    assert end["outcome"] == "completed" and end["tests_passed"] is True
    # prompt/output logs
    logdir = rt.promptlog.dir
    assert (logdir / "system_prompt.txt").is_file() and (logdir / "tools.json").is_file()
    assert (logdir / "calls" / "01-agent.request.json").is_file() and (logdir / "calls" / "04-agent.response.json").is_file()
    req = json.loads((logdir / "calls" / "02-agent.request.json").read_text(encoding="utf-8"))
    assert req["system"][0]["text"].startswith("You are devagent") and "Repository map" in req["system"][1]["text"]
    assert req["toolConfig"]["tools"][0]["toolSpec"]["name"] == "read_file"
    transcript = (logdir / "transcript.md").read_text(encoding="utf-8")
    assert "edit_file" in transcript and "Finished" in transcript
    summary = json.loads((logdir / "summary.json").read_text(encoding="utf-8"))
    assert summary["outcome"] == "completed" and summary["files_changed"] == ["src/calc.py"]


def test_system_prompt_lists_tools_and_context(temp_repo: Path, config: Config) -> None:
    rt = make_rt(temp_repo, config, [{"text": "nothing to do"}])
    run_agent(rt, "hello", approval_handler=auto_reject)
    sent = rt.model.calls[0]
    assert sent[0].content.startswith("You are devagent")
    assert "src/calc.py" in sent[1].content and "divide(a, b)" in sent[1].content  # repo map with symbols
    assert "python -m pytest" in sent[1].content  # test hint


def test_approval_accept_reject_abort(temp_repo: Path, config: Config) -> None:
    script = [{"text": "Running tests the long way.", "tool_calls": [{"name": "run_command", "args": {"command": "python -m pytest -q"}}]}, {"text": "done"}]
    # accept
    rt = make_rt(temp_repo, config, script)
    seen = []

    def accept(req):
        seen.append(req)
        return ApprovalResponse("accept")

    rr = run_agent(rt, "run the tests", approval_handler=accept)
    assert rr.outcome == RunOutcome.COMPLETED
    assert len(seen) == 1 and seen[0].call.name == "run_command" and seen[0].agent_reason == "Running tests the long way."
    assert seen[0].decision.rule == "approve_run_command_default"
    approvals = [e for e in events_of(rt) if e["t"] == "approval"]
    assert approvals[0]["action"] == "accept"
    tool_msg = rt.model.calls[1][-1]
    assert tool_msg.content.startswith("$ python -m pytest")
    # reject
    rt = make_rt(temp_repo, config, script)
    rr = run_agent(rt, "run the tests", approval_handler=lambda req: ApprovalResponse("reject", note="use run_tests"))
    assert rr.outcome == RunOutcome.COMPLETED
    assert rt.model.calls[1][-1].content.startswith("REJECTED by the user: use run_tests")
    assert rr.tool_calls == 0 and rr.tool_errors == 1
    # edit
    rt = make_rt(temp_repo, config, script)
    rr = run_agent(rt, "run", approval_handler=lambda req: ApprovalResponse("edit", args={"command": "python --version"}))
    assert "$ python --version" in rt.model.calls[1][-1].content
    # abort
    rt = make_rt(temp_repo, config, script)
    rr = run_agent(rt, "run the tests", approval_handler=lambda req: ApprovalResponse("abort"))
    assert rr.outcome == RunOutcome.ABORTED and "aborted" in rr.stop_reason


def test_approved_tier3_command_actually_runs(temp_repo: Path, config: Config) -> None:
    """The sandbox must honour the user's approval; before this test it re-denied approved tier-3 commands."""
    script = [{"tool_calls": [{"name": "run_command", "args": {"command": "python -c print(12345)"}}]}, {"text": "done"}]
    rt = make_rt(temp_repo, config, script, mode="yolo")
    rr = run_agent(rt, "x", approval_handler=auto_accept)
    assert rr.outcome == RunOutcome.COMPLETED and rr.tool_calls == 1 and rr.tool_errors == 0
    out = rt.model.calls[1][-1].content
    assert "12345" in out and "exit code 0" in out
    # ...but a rejected one still does not run
    rt = make_rt(temp_repo, config, script, mode="yolo")
    run_agent(rt, "x", approval_handler=auto_reject)
    assert rt.model.calls[1][-1].content.startswith("REJECTED by the user")
    # ...and a network tool is denied even if something claims approval
    from devagent.sandbox.local import CommandNotAllowedError, LocalSandbox

    try:
        LocalSandbox(temp_repo).run("curl http://x", approved=True)
    except CommandNotAllowedError:
        pass
    else:  # pragma: no cover
        raise AssertionError("network tool ran despite the denylist")


def test_yolo_mode_runs_allowlisted_commands_without_asking(temp_repo: Path, config: Config) -> None:
    script = [{"tool_calls": [{"name": "run_command", "args": {"command": "python --version"}}]}, {"text": "done"}]
    rt = make_rt(temp_repo, config, script, mode="yolo")
    asked = []
    rr = run_agent(rt, "x", approval_handler=lambda req: asked.append(req) or ApprovalResponse("reject"))
    assert rr.outcome == RunOutcome.COMPLETED and not asked and rr.tool_calls == 1


def test_policy_denial_is_an_observation(temp_repo: Path, config: Config) -> None:
    script = [
        {"tool_calls": [{"name": "read_file", "args": {"path": "../../outside.txt"}}, {"name": "run_command", "args": {"command": "curl http://x"}}]},
        {"text": "I could not do that; here is what I know."},
    ]
    rt = make_rt(temp_repo, config, script, mode="yolo")
    rr = run_agent(rt, "x", approval_handler=auto_accept)
    assert rr.outcome == RunOutcome.COMPLETED
    results = rt.model.calls[1][-2:]
    assert results[0].content.startswith("DENIED by policy (deny_path_escape)")
    assert results[1].content.startswith("DENIED by policy (deny_network_tool)")
    assert rr.tool_errors == 2 and rr.tool_calls == 0


def test_denied_streak_ends_run(temp_repo: Path, config: Config) -> None:
    curl = {"tool_calls": [{"name": "run_command", "args": {"command": "curl http://x"}}]}
    rt = make_rt(temp_repo, config, [curl, curl, curl, {"text": "never reached"}])
    rr = run_agent(rt, "x", approval_handler=auto_reject)
    assert rr.outcome == RunOutcome.DENIED and rr.steps == 3


def test_step_budget(temp_repo: Path, config: Config) -> None:
    rt = make_rt(temp_repo, config, HAPPY_SCRIPT, max_steps=2)
    rr = run_agent(rt, "x", approval_handler=auto_reject)
    assert rr.outcome == RunOutcome.STOPPED_AT_LIMIT and "step limit" in rr.stop_reason and rr.steps == 2


def test_cost_budget(temp_repo: Path, config: Config) -> None:
    rt = make_rt(temp_repo, config, HAPPY_SCRIPT, max_cost_usd=0.000001)
    rr = run_agent(rt, "x", approval_handler=auto_reject)
    assert rr.outcome == RunOutcome.STOPPED_AT_LIMIT and "cost limit" in rr.stop_reason and rr.steps == 1


def test_compaction_summarises_older_turns(temp_repo: Path, config: Config) -> None:
    read = {"tool_calls": [{"name": "read_file", "args": {"path": "src/calc.py"}}]}
    summary = {"text": "## Goal\nfix divide\n## Approaches tried and why they failed\nnone yet"}
    rt = make_rt(temp_repo, config, [read, read, summary, {"text": "done"}], context_window_tokens=100, compaction_ratio=0.5, keep_recent_messages=2)
    rr = run_agent(rt, "fix divide", approval_handler=auto_reject)
    assert rr.outcome == RunOutcome.COMPLETED
    comp = [e for e in events_of(rt) if e["t"] == "compaction"]
    assert comp and comp[0]["dropped_messages"] >= 3
    last_call = rt.model.calls[-1]
    assert any("[Context was compacted" in str(m.content) for m in last_call)
    assert any("Approaches tried" in str(m.content) for m in last_call)
    assert (rt.promptlog.dir / "transcript.md").read_text(encoding="utf-8").count("Context compaction") == 1


def test_multi_turn_keeps_history_on_one_thread(temp_repo: Path, config: Config) -> None:
    rt = make_rt(temp_repo, config, [{"text": "first answer"}, {"text": "second answer"}])
    graph = build_agent_graph(rt)
    run_agent(rt, "first goal", approval_handler=auto_reject, graph=graph)
    run_agent(rt, "second goal", approval_handler=auto_reject, graph=graph)
    contents = [str(m.content) for m in rt.model.calls[1]]
    assert any("first goal" in c for c in contents) and any("first answer" in c for c in contents)
    assert contents[-1] == "second goal"


def test_dangling_tool_calls_are_closed_on_next_turn(temp_repo: Path, config: Config) -> None:
    """An infrastructure crash between the model call and execution leaves tool calls
    without results in the checkpoint. The next run on the thread must close them."""
    rt = make_rt(temp_repo, config, [HAPPY_SCRIPT[0], {"text": "ok"}])
    graph = build_agent_graph(rt)
    original = rt.policy.decide

    def boom(*_a, **_k):
        raise RuntimeError("policy crashed")

    rt.policy.decide = boom  # type: ignore[method-assign]
    rr = run_agent(rt, "g1", approval_handler=auto_reject, graph=graph)
    assert rr.outcome == RunOutcome.FAILED and "policy crashed" in rr.stop_reason
    rt.policy.decide = original  # type: ignore[method-assign]
    rr = run_agent(rt, "g2", approval_handler=auto_reject, graph=graph)
    assert rr.outcome == RunOutcome.COMPLETED
    msgs = rt.model.calls[-1]
    tool_msgs = [m for m in msgs if m.type == "tool"]
    assert tool_msgs and "not executed" in tool_msgs[0].content


def test_delegate_runs_a_read_only_child(temp_repo: Path, config: Config) -> None:
    brief = "Objective: find where divide() is defined. Scope: src/ only. Out of scope: tests. Output: one line 'path:line'."
    script = [
        {"tool_calls": [{"name": "delegate", "args": {"brief": brief, "max_steps": 3}}]},
        {"tool_calls": [{"name": "grep_repo", "args": {"pattern": "def divide"}}]},  # child
        {"text": "src/calc.py:17"},  # child final
        {"text": "divide is at src/calc.py:17"},  # parent final
    ]
    rt = make_rt(temp_repo, config, script, enable_delegate=True)
    rt.delegate = make_delegate(rt)
    rr = run_agent(rt, "where is divide?", approval_handler=auto_reject)
    assert rr.outcome == RunOutcome.COMPLETED
    child_result = rt.model.calls[3][-1]
    assert child_result.content.startswith("[delegate #1: outcome=completed") and "src/calc.py:17" in child_result.content
    assert rr.steps == 2  # parent steps only
    assert rr.usage.input_tokens == 4000  # child spend counted against the parent (4 scripted calls)
    assert rt.tracer.path.with_name(f"{rt.run_id}-d01.jsonl").is_file()
    assert (rt.promptlog.dir / "delegate-01" / "transcript.md").is_file()
    # the child's system prompt is the delegate brief, with the read-only toolset
    child_system = rt.model.calls[1][0].content
    assert "read-only subagent" in child_system and brief in child_system


def test_delegate_with_write_tool_is_denied(temp_repo: Path, config: Config) -> None:
    script = [{"tool_calls": [{"name": "delegate", "args": {"brief": "x" * 60, "tools": ["edit_file"]}}]}, {"text": "ok"}]
    rt = make_rt(temp_repo, config, script, enable_delegate=True)
    rt.delegate = make_delegate(rt)
    run_agent(rt, "x", approval_handler=auto_reject)
    assert rt.model.calls[1][-1].content.startswith("DENIED by policy (deny_delegate_scope)")


def test_no_worktree_mode(temp_repo: Path, config: Config) -> None:
    rt = make_rt(temp_repo, config, [{"text": "hi"}], use_worktree=False)
    assert rt.root == temp_repo.resolve() and "no worktree" in rt.branch
    rr = run_agent(rt, "x", approval_handler=auto_reject)
    assert rr.outcome == RunOutcome.COMPLETED
