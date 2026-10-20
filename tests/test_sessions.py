from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

from devagent.core.config import Config
from devagent.core.graph import build_agent_graph, run_agent
from devagent.core.session import close_runtime, create_runtime
from devagent.core.sessions import find_session, list_sessions, record_turn, restore_from_checkpoint
from devagent.core.types import RunOutcome
from devagent.policy.approval import auto_reject


def _turn(rt, goal: str, cfg: Config):
    graph = build_agent_graph(rt)
    rr = run_agent(rt, goal, approval_handler=auto_reject, graph=graph)
    record_turn(rt.repo_root, cfg, rt.session_meta, goal, rr.outcome.value)
    return graph, rr


def test_session_is_persisted_and_resumed_with_full_history(temp_repo: Path, config: Config) -> None:
    cfg = copy.deepcopy(config)
    rt1 = create_runtime(temp_repo, cfg, script=[{"text": "first answer"}])
    _, rr1 = _turn(rt1, "first goal", cfg)
    assert rr1.outcome == RunOutcome.COMPLETED
    assert (temp_repo / ".devagent" / "sessions.sqlite").is_file()
    meta_file = temp_repo / ".devagent" / "sessions" / f"{rt1.run_id}.json"
    assert meta_file.is_file()
    saved = json.loads(meta_file.read_text(encoding="utf-8"))
    assert saved["turns"] == 1 and saved["title"] == "first goal" and saved["branch"] == rt1.branch
    first_root, first_branch, first_id = rt1.root, rt1.branch, rt1.run_id
    close_runtime(rt1)

    sessions = list_sessions(temp_repo, cfg)
    assert [s.run_id for s in sessions] == [first_id]
    meta = find_session(temp_repo, cfg, "last")
    assert meta is not None and meta.run_id == first_id

    rt2 = create_runtime(temp_repo, cfg, script=[{"text": "second answer"}], resume=meta)
    assert (rt2.run_id, rt2.root, rt2.branch) == (first_id, first_root, first_branch)
    graph2 = build_agent_graph(rt2)
    assert restore_from_checkpoint(rt2, graph2) == 2  # user + assistant from turn 1
    rr2 = run_agent(rt2, "second goal", approval_handler=auto_reject, graph=graph2)
    assert rr2.outcome == RunOutcome.COMPLETED
    sent = [str(m.content) for m in rt2.model.calls[0]]
    assert any("first goal" in c for c in sent) and any("first answer" in c for c in sent)
    assert sent[-1] == "second goal"
    # the log directory is shared and numbering continued instead of overwriting
    assert (rt2.promptlog.dir / "calls" / "01-agent.request.json").is_file()
    assert (rt2.promptlog.dir / "calls" / "02-agent.request.json").is_file()
    record_turn(rt2.repo_root, cfg, rt2.session_meta, "second goal", rr2.outcome.value)
    assert find_session(temp_repo, cfg, "1").turns == 2
    close_runtime(rt2)


def test_find_session_keys(temp_repo: Path, config: Config) -> None:
    cfg = copy.deepcopy(config)
    a = create_runtime(temp_repo, cfg, script=[{"text": "a"}])
    _turn(a, "task a", cfg)
    close_runtime(a)
    b = create_runtime(temp_repo, cfg, script=[{"text": "b"}])
    _turn(b, "task b", cfg)
    close_runtime(b)
    assert find_session(temp_repo, cfg, "last").run_id == b.run_id
    assert find_session(temp_repo, cfg, "2").run_id == a.run_id
    assert find_session(temp_repo, cfg, a.run_id).run_id == a.run_id
    assert find_session(temp_repo, cfg, a.run_id[-6:]).run_id == a.run_id
    assert find_session(temp_repo, cfg, "nope") is None
    assert find_session(temp_repo, cfg, "9") is None


def test_resume_restores_plan_and_recreates_missing_worktree(temp_repo: Path, config: Config) -> None:
    cfg = copy.deepcopy(config)
    script = [
        {"tool_calls": [{"name": "update_plan", "args": {"steps": [{"title": "Read", "status": "done"}, {"title": "Fix", "status": "in_progress"}]}}]},
        {"text": "paused here"},
    ]
    rt = create_runtime(temp_repo, cfg, script=script)
    _turn(rt, "fix the bug", cfg)
    root, branch = rt.root, rt.branch
    close_runtime(rt)
    # simulate the user deleting the worktree directory (the branch survives)
    subprocess.run(["git", "worktree", "remove", "--force", str(root)], cwd=str(temp_repo), check=True, capture_output=True)
    assert not root.exists()

    meta = find_session(temp_repo, cfg, "last")
    rt2 = create_runtime(temp_repo, cfg, script=[{"text": "resumed"}], resume=meta)
    assert rt2.root == root and root.is_dir() and rt2.branch == branch
    graph = build_agent_graph(rt2)
    restore_from_checkpoint(rt2, graph)
    assert [(s.title, s.status) for s in rt2.plan] == [("Read", "done"), ("Fix", "in_progress")]
    close_runtime(rt2)


def test_project_memory_is_injected_into_context(temp_repo: Path, config: Config) -> None:
    cfg = copy.deepcopy(config)
    (temp_repo / ".devagent").mkdir(exist_ok=True)
    (temp_repo / ".devagent" / "memory.md").write_text("Always keep functions under 20 lines.", encoding="utf-8")
    rt = create_runtime(temp_repo, cfg, script=[{"text": "ok"}])
    assert "Always keep functions under 20 lines." in rt.dynamic_context()
    _turn(rt, "hello", cfg)
    assert "Always keep functions under 20 lines." in str(rt.model.calls[0][1].content)
    close_runtime(rt)
