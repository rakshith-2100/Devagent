"""Code graders: free, consistent, and they cover almost everything that matters.

did the tests pass, which files changed, was a forbidden tool used, how many
steps, what did it cost, what was the outcome enum.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.types import RunResult
from ..sandbox.local import LocalSandbox
from ..tools.tests import detect_test_command
from ..trace.tracer import Tracer

IGNORED_CHANGES = {"NOTES.md"}


def changed_files(root: Path) -> set[str]:
    sb = LocalSandbox(root)
    res = sb.run_git("status", "--porcelain", "--untracked-files=all")
    changed: set[str] = set()
    for line in res.stdout.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        path = path.strip('"').replace("\\", "/")
        if path not in IGNORED_CHANGES:
            changed.add(path)
    return changed


def run_tests_now(root: Path, timeout: int = 180) -> tuple[bool, str]:
    sb = LocalSandbox(root)
    argv, _runner = detect_test_command(root)
    if not argv:
        return False, "no test runner detected"
    res = sb.run(argv, timeout=timeout, check_allowlist=False)
    out = (res.stdout + "\n" + res.stderr).strip()
    return (res.returncode == 0 and not res.timed_out), out[-4000:]


def tools_used(trace_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in Tracer.read(trace_path):
        if e.get("t") == "tool_call":
            counts[e["name"]] = counts.get(e["name"], 0) + 1
    return counts


def grade(expect: dict[str, Any], *, root: Path, trace_path: Path, result: RunResult) -> dict[str, dict[str, Any]]:
    grades: dict[str, dict[str, Any]] = {}

    if "tests_pass" in expect:
        passed, output = run_tests_now(root)
        grades["tests_pass"] = {"ok": passed == bool(expect["tests_pass"]), "detail": output.splitlines()[-1] if output else ""}

    changed = changed_files(root)
    if "files_changed" in expect:
        want = set(expect["files_changed"])
        missing = sorted(want - changed)
        extra = sorted(changed - want)
        grades["files_changed"] = {
            "ok": not missing,
            "detail": f"changed={sorted(changed)}" + (f" missing={missing}" if missing else "") + (f" extra={extra}" if extra else ""),
        }
    if "files_unchanged" in expect:
        touched = sorted(set(expect["files_unchanged"]) & changed)
        grades["files_unchanged"] = {"ok": not touched, "detail": f"touched={touched}" if touched else "ok"}

    used = tools_used(trace_path)
    if "forbidden_tools" in expect:
        bad = {t: n for t, n in used.items() if t in set(expect["forbidden_tools"])}
        grades["forbidden_tools"] = {"ok": not bad, "detail": f"used={bad}" if bad else "none used"}
    if "max_steps" in expect:
        grades["max_steps"] = {"ok": result.steps <= int(expect["max_steps"]), "detail": f"{result.steps} <= {expect['max_steps']}"}
    if "max_cost_usd" in expect:
        grades["max_cost_usd"] = {"ok": result.cost_usd <= float(expect["max_cost_usd"]), "detail": f"${result.cost_usd:.4f} <= ${expect['max_cost_usd']}"}
    if "outcome" in expect:
        grades["outcome"] = {"ok": result.outcome.value == str(expect["outcome"]), "detail": result.outcome.value}
    return grades
