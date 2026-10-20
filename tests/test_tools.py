from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest

from devagent.context.notes import NotesFile
from devagent.core.config import Config
from devagent.core.types import ToolCall
from devagent.sandbox.local import LocalSandbox
from devagent.tools import default_registry
from devagent.tools.base import ToolContext, ToolError
from devagent.tools.fs import edit_file, list_dir, read_file, write_file
from devagent.tools.git import git_diff
from devagent.tools.notes import append_notes
from devagent.tools.plan import update_plan
from devagent.tools.search import find_files, glob_to_regex, grep_repo
from devagent.tools.tests import run_tests


@dataclass
class FakeRuntime:
    notes: NotesFile
    plan: list = field(default_factory=list)
    files_changed: set = field(default_factory=set)
    tests_passed: Optional[bool] = None
    test_runs: int = 0
    consecutive_test_failures: int = 0


@pytest.fixture
def tctx(temp_repo: Path, config: Config) -> ToolContext:
    return ToolContext(sandbox=LocalSandbox(temp_repo), config=config, runtime=FakeRuntime(notes=NotesFile(temp_repo / "NOTES.md")))


def test_read_file_numbered_and_ranges(tctx: ToolContext) -> None:
    out = read_file(tctx, "src/calc.py")
    assert out.content.startswith("src/calc.py (lines 1-")
    assert "| def divide(a, b):" in out.content
    assert not out.truncated
    part = read_file(tctx, "src/calc.py", start_line=2, end_line=3)
    assert "(lines 2-3 of" in part.content and part.truncated


def test_read_file_caps_and_signals(tctx: ToolContext) -> None:
    big = tctx.sandbox.root / "big.txt"
    big.write_text("\n".join(f"line {i}" for i in range(1, 1001)), encoding="utf-8")
    out = read_file(tctx, "big.txt")
    assert out.truncated
    assert "(lines 1-400 of 1000)" in out.content
    assert "600 more lines" in out.content and "start_line=401" in out.content


def test_read_file_errors_are_instructions(tctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="find_files"):
        read_file(tctx, "nope.py")
    with pytest.raises(ToolError, match="list_dir"):
        read_file(tctx, "src")


def test_list_dir(tctx: ToolContext) -> None:
    out = list_dir(tctx, ".")
    assert "src/" in out.content and "tests/" in out.content and "calc.py" in out.content
    out = list_dir(tctx, "src", depth=1)
    assert out.content.startswith("src/")


def test_write_file_new_only(tctx: ToolContext) -> None:
    out = write_file(tctx, "src/new_mod.py", "x = 1\n")
    assert "Created src/new_mod.py" in out.content
    assert "src/new_mod.py" in tctx.runtime.files_changed
    with pytest.raises(ToolError, match="edit_file"):
        write_file(tctx, "src/new_mod.py", "x = 2\n")


def test_edit_file_semantics(tctx: ToolContext) -> None:
    out = edit_file(tctx, "src/calc.py", "    return a / b\n", "    if b == 0:\n        raise ValueError('division by zero')\n    return a / b\n")
    assert out.content.startswith("Replaced 1 occurrence(s) in src/calc.py")
    assert "+    if b == 0:" in out.content
    text = (tctx.sandbox.root / "src" / "calc.py").read_text(encoding="utf-8")
    assert "raise ValueError" in text
    with pytest.raises(ToolError, match="not found"):
        edit_file(tctx, "src/calc.py", "this text is not there", "x")
    with pytest.raises(ToolError, match="appears 4 times"):
        edit_file(tctx, "src/calc.py", "    return a ", "    return  a ")
    out = edit_file(tctx, "src/calc.py", "    return a ", "    return  a ", replace_all=True)
    assert "Replaced 4 occurrence(s)" in out.content


def test_edit_file_identical_strings_rejected(tctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="identical"):
        edit_file(tctx, "src/calc.py", "def add", "def add")


def test_edit_file_hints_at_nearby_line(tctx: ToolContext) -> None:
    with pytest.raises(ToolError) as info:
        edit_file(tctx, "src/calc.py", "def divide(a, b):\n    return b / a\n", "x")
    assert "near line" in str(info.value) and "read_file(" in str(info.value)


def test_edit_file_preserves_crlf(tctx: ToolContext) -> None:
    p = tctx.sandbox.root / "win.txt"
    p.write_bytes(b"one\r\ntwo\r\nthree\r\n")
    edit_file(tctx, "win.txt", "two\n", "2\n")
    assert p.read_bytes() == b"one\r\n2\r\nthree\r\n"


def test_grep_and_find(tctx: ToolContext) -> None:
    out = grep_repo(tctx, r"def divide")
    assert "src/calc.py:" in out.content and "1 match(es)" in out.content
    out = grep_repo(tctx, r"def ", glob="tests/*.py")
    assert "tests/test_calc.py" in out.content and "src/calc.py" not in out.content
    out = grep_repo(tctx, "zzz_not_present")
    assert out.content.startswith("No matches")
    out = grep_repo(tctx, "def", max_results=2)
    assert out.truncated and "Showing 2 of" in out.content
    out = find_files(tctx, "*.py")
    assert "src/calc.py" in out.content and "tests/test_calc.py" in out.content
    out = find_files(tctx, "test_")
    assert "tests/test_calc.py" in out.content and "src/calc.py" not in out.content
    out = find_files(tctx, "src/**/*.py")
    assert "src/calc.py" in out.content and "tests/test_calc.py" not in out.content


def test_glob_to_regex() -> None:
    assert glob_to_regex("*.py").search("a/b/c.py")
    assert not glob_to_regex("*.py").search("a/b/c.pyc")
    assert glob_to_regex("src/**/*.ts").search("src/x.ts")
    assert glob_to_regex("src/**/*.ts").search("src/a/b/x.ts")
    assert not glob_to_regex("src/**/*.ts").search("lib/x.ts")


def test_update_plan_and_notes(tctx: ToolContext) -> None:
    out = update_plan(tctx, [{"title": "Read code", "status": "done"}, {"title": "Fix", "status": "in_progress"}])
    assert "[x] 1. Read code" in out.content and "[>] 2. Fix" in out.content
    assert len(tctx.runtime.plan) == 2
    with pytest.raises(ToolError, match="status"):
        update_plan(tctx, [{"title": "x", "status": "maybe"}])
    with pytest.raises(ToolError):
        update_plan(tctx, [])
    out = append_notes(tctx, "divide() lives in src/calc.py")
    assert "NOTES.md" in out.content
    assert "divide()" in (tctx.sandbox.root / "NOTES.md").read_text(encoding="utf-8")


def test_run_tests_reports_only_failures(tctx: ToolContext) -> None:
    out = run_tests(tctx)
    assert "Result: FAILED" in out.content
    assert "test_divide_by_zero" in out.content
    assert "test_add" not in out.content.split("Failing output:")[1]
    assert tctx.runtime.tests_passed is False and tctx.runtime.consecutive_test_failures == 1
    edit_file(tctx, "src/calc.py", "    return a / b\n", "    if b == 0:\n        raise ValueError('division by zero')\n    return a / b\n")
    out = run_tests(tctx)
    assert "Result: PASSED" in out.content and tctx.runtime.tests_passed is True
    out = run_tests(tctx, target="tests/test_calc.py::test_add")
    assert "PASSED" in out.content


def test_run_tests_three_strikes_note(tctx: ToolContext) -> None:
    for _ in range(3):
        out = run_tests(tctx)
    assert "failed 3 times in a row" in out.content


def test_git_diff(tctx: ToolContext) -> None:
    assert git_diff(tctx).content.startswith("No changes")
    edit_file(tctx, "src/calc.py", "    return a / b\n", "    return a // b\n")
    write_file(tctx, "src/extra.py", "y = 2\n")
    out = git_diff(tctx)
    assert "src/calc.py" in out.content and "-    return a / b" in out.content
    assert "New untracked files" in out.content and "src/extra.py" in out.content


def test_registry_execute_wraps_errors(tctx: ToolContext) -> None:
    reg = default_registry()
    res = reg.execute(ToolCall("1", "read_file", {"path": "missing.py"}), tctx)
    assert not res.ok and "does not exist" in res.content
    res = reg.execute(ToolCall("2", "read_file", {}), tctx)
    assert not res.ok and "missing required argument" in res.content
    res = reg.execute(ToolCall("3", "read_file", {"path": "src/calc.py", "bogus": 1}), tctx)
    assert not res.ok and "unknown argument" in res.content
    res = reg.execute(ToolCall("4", "nope", {}), tctx)
    assert not res.ok and "Unknown tool" in res.content
    res = reg.execute(ToolCall("5", "run_command", {"command": "curl http://x"}), tctx)
    assert not res.ok and "network" in res.content.lower()
    res = reg.execute(ToolCall("6", "read_file", {"path": "src/calc.py", "start_line": "2", "end_line": "3"}), tctx)
    assert res.ok and "(lines 2-3" in res.content
    assert res.meta["duration_ms"] >= 0 and res.meta["bytes"] > 0
