from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from devagent.core.config import Config
from devagent.evals.runner import EVALS_DIR

ALL_TOOLS = {
    "read_file",
    "list_dir",
    "grep_repo",
    "find_files",
    "git_diff",
    "edit_file",
    "write_file",
    "update_plan",
    "append_notes",
    "run_tests",
    "run_command",
    "delegate",
}


def git(repo: Path, *args: str) -> str:
    res = subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True, text=True, encoding="utf-8")
    return res.stdout


def make_repo(dst: Path, fixture: str = "calc") -> Path:
    shutil.copytree(EVALS_DIR / "fixtures" / fixture, dst, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".devagent"))
    git(dst, "init", "-q")
    git(dst, "config", "user.email", "t@example.com")
    git(dst, "config", "user.name", "tests")
    git(dst, "config", "core.autocrlf", "false")
    git(dst, "add", "-A")
    git(dst, "commit", "-q", "-m", "broken fixture")
    return dst


@pytest.fixture
def temp_repo(tmp_path: Path) -> Path:
    """A tiny git repo (the calc fixture) with one failing test."""
    return make_repo(tmp_path / "repo")


@pytest.fixture
def config() -> Config:
    cfg = Config()
    cfg.provider = "mock"
    cfg.log_prompts = True
    cfg.budget.max_wall_seconds = 300
    return cfg


@pytest.fixture(autouse=True)
def _no_real_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("BEDROCK_API_KEY", "AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SECRET_FOR_SANDBOX_TEST", "do-not-leak")
    _ = os.environ
