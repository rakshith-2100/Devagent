"""run_tests: task-shaped, not a shell call. Returns pass/fail counts plus only
the failing output."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from .base import ToolContext, ToolError, ToolOutput, tool

FAIL_OUTPUT_CAP = 8000
PYTEST_ARGV = ["python", "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider", "--no-header"]

# Matches both "===== 1 failed, 4 passed in 0.16s =====" and the quiet-mode "1 failed, 4 passed in 0.16s".
_PYTEST_SUMMARY = re.compile(r"^(?:=+\s)?((?:\d+ [a-z]+(?:, )?)+|no tests ran) in [\d.]+s(?: \([^)]*\))?(?:\s=+)?\s*$", re.MULTILINE)
_COUNT = re.compile(r"(\d+) (passed|failed|error|errors|skipped|xfailed|xpassed|warnings?|deselected)")


def detect_test_command(root: Path) -> tuple[list[str], str]:
    """Return (argv, runner_name) for the repository under ``root``."""
    if (root / "package.json").is_file():
        try:
            pkg = json.loads((root / "package.json").read_text(encoding="utf-8"))
            if "test" in (pkg.get("scripts") or {}):
                return ["npm", "test", "--silent", "--"], "npm"
        except (OSError, json.JSONDecodeError):
            pass
    if (root / "go.mod").is_file():
        return ["go", "test", "./..."], "go"
    if (root / "Cargo.toml").is_file():
        return ["cargo", "test", "--quiet"], "cargo"
    py_markers = ("pytest.ini", "pyproject.toml", "setup.cfg", "setup.py", "conftest.py", "tox.ini", "requirements.txt")
    if any((root / m).is_file() for m in py_markers) or (root / "tests").is_dir() or (root / "test").is_dir():
        return list(PYTEST_ARGV), "pytest"
    if any(root.glob("test_*.py")) or any(root.glob("*_test.py")) or any(root.glob("**/test_*.py")):
        return list(PYTEST_ARGV), "pytest"
    makefile = root / "Makefile"
    if makefile.is_file() and "test:" in makefile.read_text(encoding="utf-8", errors="replace"):
        return ["make", "test"], "make"
    return [], ""


def test_hint(root: Path) -> str:
    argv, _runner = detect_test_command(root)
    return " ".join(argv) if argv else "no test runner detected; run_tests will say so"


def _parse_pytest(stdout: str) -> tuple[dict[str, int], str]:
    counts: dict[str, int] = {}
    last = None
    for last in _PYTEST_SUMMARY.finditer(stdout):
        pass
    summary = last.group(1) if last else ""
    for n, kind in _COUNT.findall(summary):
        kind = {"errors": "error", "warnings": "warning"}.get(kind, kind)
        counts[kind] = counts.get(kind, 0) + int(n)
    return counts, summary


def _failing_section(stdout: str) -> str:
    """Only the failing output: from the first FAILURES/ERRORS header to the summary line."""
    start = None
    for marker in ("= FAILURES =", "= ERRORS =", "FAILED ", "ERROR "):
        idx = stdout.find(marker)
        if idx != -1 and (start is None or idx < start):
            start = idx
    if start is None:
        return ""
    section = stdout[start:]
    m = _PYTEST_SUMMARY.search(section)
    if m:
        section = section[: m.start()].rstrip()
    return section


@tool
def run_tests(ctx: ToolContext, target: Optional[str] = None, timeout: int = 120) -> ToolOutput:
    """Run the project's test suite (or one target) and report pass/fail counts with only the failing output.

    The runner is detected from the repository (pytest, npm test, go test,
    cargo test, make test). Use this instead of run_command for tests.

    Args:
        target: Optional test file, directory, or node id, e.g. "tests/test_calc.py" or "tests/test_calc.py::test_divide".
        timeout: Seconds before the run is killed (10-900). Defaults to 120.
    """
    root = ctx.sandbox.root
    argv, runner = detect_test_command(root)
    if not argv:
        raise ToolError(
            "run_tests: no test runner detected (looked for pytest markers, package.json scripts.test, go.mod, Cargo.toml, Makefile). "
            "If tests exist, tell the user which command runs them."
        )
    timeout = max(10, min(int(timeout), 900))
    if target:
        file_part, _, node_part = target.partition("::")
        rel_target = ctx.sandbox.relpath(ctx.sandbox.resolve(file_part))
        if node_part:
            rel_target += "::" + node_part
        argv = argv + [rel_target]
    res = ctx.sandbox.run(argv, timeout=timeout, check_allowlist=False)
    combined = res.stdout + ("\n" + res.stderr if res.stderr.strip() else "")

    if res.timed_out:
        passed = False
        headline = f"TIMED OUT after {timeout}s"
        detail = combined[-FAIL_OUTPUT_CAP:]
    elif runner == "pytest":
        counts, summary = _parse_pytest(res.stdout)
        failed = counts.get("failed", 0) + counts.get("error", 0)
        if res.returncode == 5:
            passed = False
            headline = "NO TESTS COLLECTED"
            detail = combined[-FAIL_OUTPUT_CAP:]
        elif not summary and res.returncode != 0:
            passed = False
            headline = f"RUNNER ERROR (exit {res.returncode})"
            detail = combined[-FAIL_OUTPUT_CAP:]
        else:
            passed = res.returncode == 0 and failed == 0
            headline = ("PASSED" if passed else "FAILED") + f": {summary or 'no summary line'}"
            detail = _failing_section(res.stdout) if not passed else ""
    else:
        passed = res.returncode == 0
        headline = ("PASSED" if passed else "FAILED") + f" (exit {res.returncode})"
        detail = "" if passed else combined[-FAIL_OUTPUT_CAP:]

    truncated = False
    if len(detail) > FAIL_OUTPUT_CAP:
        detail = detail[:FAIL_OUTPUT_CAP] + f"\n... [failing output truncated at {FAIL_OUTPUT_CAP} chars; run a single target to see less]"
        truncated = True

    rt = ctx.runtime
    if rt is not None:
        rt.tests_passed = passed
        rt.test_runs += 1
        rt.consecutive_test_failures = 0 if passed else rt.consecutive_test_failures + 1

    text = f"Test command: {' '.join(res.argv)}\nResult: {headline} in {res.duration_ms / 1000:.1f}s"
    if detail:
        text += "\n\nFailing output:\n" + detail
    if rt is not None and not passed and rt.consecutive_test_failures >= ctx.config.max_test_attempts:
        text += (
            f"\n\nNOTE: tests have failed {rt.consecutive_test_failures} times in a row. Stop iterating: "
            "write down what you tried in append_notes and report to the user instead of attempting another variation."
        )
    return ToolOutput(
        text, truncated=truncated or res.truncated, meta={"passed": passed, "returncode": res.returncode, "runner": runner}
    )
