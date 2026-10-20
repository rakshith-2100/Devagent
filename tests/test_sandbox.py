from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from devagent.sandbox.local import CommandNotAllowedError, LocalSandbox, PathEscapeError, create_worktree, git_root, remove_worktree


def test_resolve_confines(tmp_path: Path) -> None:
    sb = LocalSandbox(tmp_path)
    (tmp_path / "src").mkdir()
    assert sb.resolve("src") == (tmp_path / "src").resolve()
    assert sb.resolve(".") == tmp_path.resolve()
    assert sb.resolve("") == tmp_path.resolve()
    with pytest.raises(PathEscapeError):
        sb.resolve("../outside.txt")
    with pytest.raises(PathEscapeError):
        sb.resolve(str(tmp_path.parent / "x"))
    # a sibling directory that shares the prefix must not count as inside
    with pytest.raises(PathEscapeError):
        sb.resolve(str(tmp_path) + "_sibling/x")


def test_symlink_escape_is_resolved_before_check(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    link = root / "link.txt"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this system")
    sb = LocalSandbox(root)
    with pytest.raises(PathEscapeError):
        sb.resolve("link.txt")


def test_clean_environment(tmp_path: Path) -> None:
    sb = LocalSandbox(tmp_path)
    env = sb.build_env()
    assert "PATH" in env
    assert "SECRET_FOR_SANDBOX_TEST" not in env
    assert env["PYTHONIOENCODING"] == "utf-8"
    res = sb.run([sys.executable, "-c", "import os; print(os.environ.get('SECRET_FOR_SANDBOX_TEST'))"], check_allowlist=False)
    assert res.stdout.strip() == "None"


def test_allowlist_enforced_at_sandbox_level(tmp_path: Path) -> None:
    sb = LocalSandbox(tmp_path)
    with pytest.raises(CommandNotAllowedError):
        sb.run("curl http://example.com")
    with pytest.raises(CommandNotAllowedError):
        sb.run("python -c print(1)")  # tier 3 (inline code) cannot run without approval path
    with pytest.raises(ValueError):
        sb.run("echo hi && curl x")


def test_timeout_and_output_cap(tmp_path: Path) -> None:
    sb = LocalSandbox(tmp_path, output_cap_chars=50)
    res = sb.run([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.5, check_allowlist=False)
    assert res.timed_out and not res.ok
    res = sb.run([sys.executable, "-c", "print('x' * 500)"], check_allowlist=False)
    assert res.truncated and "truncated" in res.stdout


def test_missing_executable(tmp_path: Path) -> None:
    sb = LocalSandbox(tmp_path)
    res = sb.run(["definitely-not-a-real-binary-xyz"], check_allowlist=False)
    assert res.returncode == 127 and "not found" in res.stderr


def test_worktree_lifecycle(temp_repo: Path) -> None:
    assert git_root(temp_repo) == temp_repo.resolve()
    wt, branch = create_worktree(temp_repo, "r_test")
    assert wt.is_dir() and (wt / "src" / "calc.py").is_file()
    assert branch == "devagent/r_test"
    assert wt.parent.name == ".devagent"
    exclude = (temp_repo / ".git" / "info" / "exclude").read_text(encoding="utf-8")
    assert ".devagent/" in exclude
    (wt / "src" / "calc.py").write_text("changed", encoding="utf-8")
    assert "changed" not in (temp_repo / "src" / "calc.py").read_text(encoding="utf-8")
    remove_worktree(temp_repo, wt, delete_branch=True, branch=branch)
    assert not wt.exists()
