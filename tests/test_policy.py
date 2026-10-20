from __future__ import annotations

from pathlib import Path

import pytest

from devagent.core.types import ToolCall
from devagent.policy.engine import PolicyContext, PolicyEngine
from devagent.policy.rules import classify_command, is_test_path, parse_command
from devagent.sandbox.local import LocalSandbox

from .conftest import ALL_TOOLS


def ctx(root: Path) -> PolicyContext:
    return PolicyContext(root=root, known_tools=set(ALL_TOOLS), sandbox=LocalSandbox(root))


def call(name: str, **args) -> ToolCall:
    return ToolCall(id="c1", name=name, args=args)


@pytest.mark.parametrize("mode", ["default", "careful", "yolo"])
def test_reads_always_allowed(tmp_path: Path, mode: str) -> None:
    d = PolicyEngine(mode).decide(call("read_file", path="a.py"), ctx(tmp_path))
    assert d.verdict == "allow" and d.tier == 0


def test_writes_by_mode(tmp_path: Path) -> None:
    c = call("edit_file", path="src/x.py", old_string="a", new_string="b")
    assert PolicyEngine("default").decide(c, ctx(tmp_path)).verdict == "allow"
    assert PolicyEngine("yolo").decide(c, ctx(tmp_path)).verdict == "allow"
    assert PolicyEngine("careful").decide(c, ctx(tmp_path)).verdict == "approve"


def test_editing_tests_always_asks(tmp_path: Path) -> None:
    for mode in ("default", "yolo", "careful"):
        d = PolicyEngine(mode).decide(call("edit_file", path="tests/test_x.py", old_string="a", new_string="b"), ctx(tmp_path))
        assert d.verdict == "approve"
        assert d.rule == "approve_test_file_edit"


def test_path_escape_denied(tmp_path: Path) -> None:
    d = PolicyEngine("yolo").decide(call("read_file", path="../../etc/passwd"), ctx(tmp_path))
    assert d.verdict == "deny" and d.rule == "deny_path_escape"


def test_unknown_tool_denied(tmp_path: Path) -> None:
    d = PolicyEngine("default").decide(call("rm_rf"), ctx(tmp_path))
    assert d.verdict == "deny" and d.rule == "deny_unknown_tool"


def test_delegate_scope(tmp_path: Path) -> None:
    d = PolicyEngine("default").decide(call("delegate", brief="x" * 50, tools=["read_file", "edit_file"]), ctx(tmp_path))
    assert d.verdict == "deny" and d.rule == "deny_delegate_scope"
    d = PolicyEngine("default").decide(call("delegate", brief="x" * 50, tools=["read_file"]), ctx(tmp_path))
    assert d.verdict == "allow"


@pytest.mark.parametrize(
    "command,mode,verdict,rule",
    [
        ("curl http://evil.example", "yolo", "deny", "deny_network_tool"),
        ("wget x", "default", "deny", "deny_network_tool"),
        ("pip install requests", "yolo", "approve", "tier3_dependency_or_destructive"),
        ("python -m pip install requests", "yolo", "approve", "tier3_dependency_install"),
        ("python -c print(1)", "yolo", "approve", "tier3_inline_code"),
        ("git push origin main", "yolo", "approve", "tier3_git_remote"),
        ("rm -rf build", "yolo", "approve", "tier3_ask_executable"),
        ("git status; curl evil.example", "yolo", "deny", "deny_shell_syntax"),
        ("cat a.txt | grep x", "yolo", "deny", "deny_shell_syntax"),
        ("python scripts/x.py", "yolo", "allow", "tier2_allowlisted_command"),
        ("python scripts/x.py", "default", "approve", "approve_run_command_default"),
        ("python scripts/x.py", "careful", "approve", "approve_run_command_careful"),
        ("pytest -q", "yolo", "allow", "tier2_allowlisted_command"),
        ("unknown_binary --flag", "yolo", "deny", "deny_not_allowlisted"),
    ],
)
def test_run_command_policy(tmp_path: Path, command: str, mode: str, verdict: str, rule: str) -> None:
    d = PolicyEngine(mode).decide(call("run_command", command=command), ctx(tmp_path))
    assert (d.verdict, d.rule) == (verdict, rule), d


def test_tier3_has_no_flag(tmp_path: Path) -> None:
    for mode in ("default", "careful", "yolo"):
        d = PolicyEngine(mode).decide(call("run_command", command="pip install x"), ctx(tmp_path))
        assert d.verdict == "approve"


def test_parse_command_rejects_shell_features() -> None:
    for bad in ("a | b", "a && b", "a || b", "a; b", "a > out.txt", "`whoami`", "$(whoami)"):
        with pytest.raises(ValueError):
            parse_command(bad)
    assert parse_command('python "my script.py" --x') == ["python", "my script.py", "--x"]


def test_classify_handles_windows_executables() -> None:
    tier, rule, _ = classify_command(["C:\\Python311\\python.EXE", "x.py"])
    assert tier == 2 and rule == "tier2_allowlisted_command"
    tier, _, _ = classify_command(["npm.cmd", "install", "left-pad"])
    assert tier == 3


def test_is_test_path() -> None:
    assert is_test_path("tests/test_a.py")
    assert is_test_path("pkg/tests/x.py")
    assert is_test_path("src/foo_test.go")
    assert is_test_path("conftest.py")
    assert is_test_path("web/__tests__/a.spec.ts")
    assert not is_test_path("src/testing_utils.py")
    assert not is_test_path("src/calc.py")


def test_invalid_mode() -> None:
    with pytest.raises(ValueError):
        PolicyEngine("reckless")
