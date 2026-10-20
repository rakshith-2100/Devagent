"""``devagent eval``: one command gives you pass^k, cost and steps per case.

Each case points at a fixture repo whose broken state gets committed into a
fresh temporary git repository. The agent runs in a worktree of that repo with
approvals auto-rejected (non-interactive), the code graders check the result,
and the optional judge reads the diff.
"""
from __future__ import annotations

import copy
import json
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from rich.console import Console
from rich.table import Table

from ..core.config import Config
from ..core.graph import make_delegate, run_agent
from ..core.runtime import NullObserver
from ..core.session import create_runtime
from ..policy.approval import auto_reject
from .graders import grade, run_tests_now
from .judge import judge_attempt

EVALS_DIR = Path(__file__).parent


@dataclass
class EvalCase:
    id: str
    fixture: Path
    task: str
    expect: dict[str, Any] = field(default_factory=dict)
    graders: list[str] = field(default_factory=lambda: ["code"])
    runs: int = 3
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path, base: Path) -> "EvalCase":
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        fixture = Path(data["fixture"])
        if not fixture.is_absolute():
            fixture = (base / fixture).resolve()
        return cls(
            id=str(data.get("id", path.stem)),
            fixture=fixture,
            task=str(data["task"]),
            expect=dict(data.get("expect") or {}),
            graders=list(data.get("graders") or ["code"]),
            runs=int(data.get("runs", 3)),
            tags=list(data.get("tags") or []),
        )


def load_cases(cases_dir: Optional[Path] = None) -> list[EvalCase]:
    cases_dir = cases_dir or (EVALS_DIR / "cases")
    base = cases_dir.parent if cases_dir.parent.joinpath("fixtures").is_dir() else EVALS_DIR
    return [EvalCase.from_yaml(p, base) for p in sorted(cases_dir.glob("*.yaml"))]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True, text=True)


def prepare_fixture(case: EvalCase, workdir: Path, k: int) -> Path:
    """Copy the fixture into a fresh git repository with the broken state committed."""
    repo = workdir / f"{case.id}-run{k}"
    shutil.copytree(case.fixture, repo, ignore=shutil.ignore_patterns(".devagent", "__pycache__", ".pytest_cache", ".git"))
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "evals@devagent.local")
    _git(repo, "config", "user.name", "devagent evals")
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"fixture: {case.id} (broken state)")
    return repo


def run_case_once(case: EvalCase, cfg: Config, workdir: Path, k: int, *, script: Optional[list[Any]], use_judge: bool) -> dict[str, Any]:
    repo = prepare_fixture(case, workdir, k)
    case_cfg = copy.deepcopy(cfg)
    case_cfg.use_worktree = True
    if "max_steps" in case.expect:
        case_cfg.budget.max_steps = int(case.expect["max_steps"]) + 2  # the grader enforces the real cap
    if "max_cost_usd" in case.expect:
        case_cfg.budget.max_cost_usd = float(case.expect["max_cost_usd"]) * 1.5
    rt = create_runtime(repo, case_cfg, observer=NullObserver(), script=copy.deepcopy(script) if script is not None else None)
    if case_cfg.enable_delegate:
        rt.delegate = make_delegate(rt)
    _passed_before, original_failure = run_tests_now(rt.root)
    started = time.monotonic()
    rr = run_agent(rt, case.task, approval_handler=auto_reject)
    grades = grade(case.expect, root=rt.root, trace_path=rt.tracer.path, result=rr)
    judge: Optional[dict[str, Any]] = None
    if use_judge and "judge" in case.graders:
        diff = rt.sandbox.run_git("diff").stdout
        try:
            judge = judge_attempt(rt.model, task=case.task, original_failure=original_failure, diff=diff, report=rr.final_text)
        except Exception as e:  # noqa: BLE001 - the judge must never crash the harness
            judge = {"ok": False, "error": str(e)}
    passed = all(g["ok"] for g in grades.values()) and (judge is None or judge.get("ok", False))
    return {
        "case": case.id,
        "run": k,
        "passed": passed,
        "grades": grades,
        "judge": judge,
        "result": rr.as_dict(),
        "wall_s": round(time.monotonic() - started, 1),
        "trace": str(rt.tracer.path),
        "logs": str(rt.promptlog.dir),
        "worktree": str(rt.root),
    }


def run_evals(
    cfg: Config,
    *,
    console: Console,
    case_ids: Optional[list[str]] = None,
    tags: Optional[list[str]] = None,
    runs_override: Optional[int] = None,
    use_judge: bool = False,
    script: Optional[list[Any]] = None,
    cases_dir: Optional[Path] = None,
    output_root: Optional[Path] = None,
    keep_workdir: bool = True,
) -> int:
    cases = load_cases(cases_dir)
    if case_ids:
        cases = [c for c in cases if c.id in set(case_ids)]
    if tags:
        cases = [c for c in cases if set(tags) & set(c.tags)]
    if not cases:
        console.print("[red]no matching eval cases[/red]")
        return 1
    out_dir = (output_root or Path.cwd()) / cfg.devagent_dir / "evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    workdir = Path(tempfile.mkdtemp(prefix=f"devagent-evals-{stamp}-"))
    console.print(f"[dim]fixtures under {workdir}[/dim]")

    all_results: list[dict[str, Any]] = []
    for case in cases:
        runs = runs_override or case.runs
        for k in range(1, runs + 1):
            console.print(f"[bold]{case.id}[/bold] run {k}/{runs} ...", end=" ")
            try:
                res = run_case_once(case, cfg, workdir, k, script=script, use_judge=use_judge)
            except Exception as e:  # noqa: BLE001 - one broken run must not kill the suite
                res = {"case": case.id, "run": k, "passed": False, "error": f"{type(e).__name__}: {e}", "grades": {}, "result": {}}
            all_results.append(res)
            mark = "[green]pass[/green]" if res["passed"] else "[red]FAIL[/red]"
            failed = [n for n, g in res.get("grades", {}).items() if not g.get("ok")]
            extra = f" ({', '.join(failed)})" if failed else (f" ({res['error']})" if res.get("error") else "")
            console.print(f"{mark}{extra}")

    table = Table(title=f"eval results ({stamp})")
    for col in ("case", "runs", "pass^k", "pass rate", "steps avg", "cost avg", "wall p50", "tags"):
        table.add_column(col)
    overall_ok = True
    summary: dict[str, Any] = {}
    for case in cases:
        rs = [r for r in all_results if r["case"] == case.id]
        n = len(rs)
        passes = sum(1 for r in rs if r["passed"])
        pass_k = passes == n
        overall_ok = overall_ok and pass_k
        steps = [r["result"].get("steps", 0) for r in rs if r.get("result")]
        costs = [r["result"].get("cost_usd", 0.0) for r in rs if r.get("result")]
        walls = [r.get("wall_s", 0.0) for r in rs]
        summary[case.id] = {"runs": n, "passes": passes, "pass_k": pass_k, "steps_avg": statistics.mean(steps) if steps else 0, "cost_avg": statistics.mean(costs) if costs else 0}
        table.add_row(
            case.id,
            str(n),
            "[green]yes[/green]" if pass_k else "[red]no[/red]",
            f"{passes}/{n}",
            f"{statistics.mean(steps):.1f}" if steps else "-",
            f"${statistics.mean(costs):.3f}" if costs else "-",
            f"{statistics.median(walls):.0f}s" if walls else "-",
            ",".join(case.tags),
        )
    console.print(table)
    report_path = out_dir / f"{stamp}.json"
    report_path.write_text(json.dumps({"stamp": stamp, "config": cfg.redacted(), "summary": summary, "runs": all_results}, indent=2, default=str), encoding="utf-8")
    console.print(f"[dim]report: {report_path}[/dim]")
    if not keep_workdir:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0 if overall_ok else 1
