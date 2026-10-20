"""Phase 1 sandbox: a git worktree plus three enforcement layers.

1. Path confinement: every path is resolved (symlinks included) before being
   compared against the worktree root.
2. Command allowlist: matched on parsed argv, never on the raw string.
3. Clean environment: the subprocess environment is built explicitly rather
   than inherited, so API keys and cloud credentials never reach a
   model-written command.

Every command gets a timeout and captured output, with stdout capped and the
cap signalled.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from ..policy.rules import COMMAND_ALLOWLIST, classify_command, parse_command


class SandboxError(Exception):
    """Base class; the message is written for the model to read."""


class PathEscapeError(SandboxError):
    pass


class CommandNotAllowedError(SandboxError):
    pass


@dataclass
class CommandResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    duration_ms: int = 0
    truncated: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


# Environment variables that are safe and often necessary for tools to run.
_ENV_PASSTHROUGH = (
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "COMSPEC",
    "WINDIR",
    "TEMP",
    "TMP",
    "TMPDIR",
    "HOME",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "LANG",
    "LC_ALL",
    "TERM",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "PYTHONPATH",
    "NODE_PATH",
    "GOPATH",
    "GOROOT",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "JAVA_HOME",
)


def _normcase(p: Path) -> str:
    return os.path.normcase(str(p))


def _as_text(data: object) -> str:
    if data is None:
        return ""
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data).decode("utf-8", "replace")
    return str(data)


class LocalSandbox:
    def __init__(
        self,
        root: Path | str,
        *,
        allowlist: Iterable[str] = COMMAND_ALLOWLIST,
        output_cap_chars: int = 20_000,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise SandboxError(f"Sandbox root does not exist: {self.root}")
        self.allowlist = set(allowlist)
        self.output_cap_chars = output_cap_chars

    # ----------------------------------------------------------------- paths
    def resolve(self, path: str | os.PathLike[str] | None) -> Path:
        """Resolve ``path`` relative to the root and confine it.

        Resolution happens before the check, which is what closes the symlink
        escape. The error message tells the model what root it is confined to.
        """
        raw = str(path).strip() if path is not None else ""
        if raw in ("", "."):
            return self.root
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()
        if not self.is_inside(resolved):
            raise PathEscapeError(
                f"Path '{raw}' resolves to '{resolved}', which is outside the working root "
                f"'{self.root}'. Only paths inside the working root are allowed. "
                f"Use a path relative to the root, for example 'src/module.py'."
            )
        return resolved

    def is_inside(self, resolved: Path) -> bool:
        root = _normcase(self.root)
        target = _normcase(resolved)
        return target == root or target.startswith(root.rstrip("\\/") + os.sep)

    def relpath(self, resolved: Path) -> str:
        try:
            return resolved.relative_to(self.root).as_posix()
        except ValueError:
            return str(resolved)

    # ----------------------------------------------------------- environment
    def build_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        for key in _ENV_PASSTHROUGH:
            v = os.environ.get(key)
            if v is not None:
                env[key] = v
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["CI"] = "1"  # many test runners and CLIs go non-interactive under CI
        env["NO_COLOR"] = "1"
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["DEVAGENT_SANDBOX"] = "1"
        return env

    # -------------------------------------------------------------- commands
    def run(
        self,
        command: str | list[str],
        *,
        timeout: float = 60,
        check_allowlist: bool = True,
        approved: bool = False,
        cwd: Optional[Path | str] = None,
        stdin_text: Optional[str] = None,
    ) -> CommandResult:
        argv = parse_command(command) if isinstance(command, str) else list(command)
        if not argv:
            raise CommandNotAllowedError("Empty command.")
        if check_allowlist:
            tier, _rule, reason = classify_command(argv)
            # Denied executables (network tools) never run. Tier 3 runs only when the
            # user approved this specific call; the policy engine is what asks them.
            if tier < 0 or (tier >= 3 and not approved):
                raise CommandNotAllowedError(reason)
        workdir = self.resolve(cwd) if cwd is not None else self.root
        env = self.build_env()
        # Resolve the executable ourselves: with shell=False Windows will not find
        # .cmd/.bat shims such as npm.cmd or npx.cmd from a bare name.
        found = shutil.which(argv[0], path=env.get("PATH"))
        exec_argv = [found, *argv[1:]] if found else argv
        started = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(
                exec_argv,
                cwd=str(workdir),
                env=env,
                input=stdin_text,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                shell=False,
            )
            stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as e:
            timed_out = True
            stdout = _as_text(e.stdout)
            stderr = _as_text(e.stderr)
            rc = -1
        except FileNotFoundError:
            return CommandResult(
                argv=argv,
                returncode=127,
                stdout="",
                stderr=f"Executable not found on PATH: {argv[0]}",
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        duration_ms = int((time.monotonic() - started) * 1000)
        stdout, stderr, truncated = self._cap_output(stdout, stderr)
        return CommandResult(
            argv=argv,
            returncode=rc,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration_ms=duration_ms,
            truncated=truncated,
        )

    def _cap_output(self, stdout: str, stderr: str) -> tuple[str, str, bool]:
        truncated = False
        if len(stdout) > self.output_cap_chars:
            stdout = stdout[: self.output_cap_chars] + f"\n... [stdout truncated at {self.output_cap_chars} chars]"
            truncated = True
        if len(stderr) > self.output_cap_chars:
            stderr = stderr[: self.output_cap_chars] + f"\n... [stderr truncated at {self.output_cap_chars} chars]"
            truncated = True
        return stdout, stderr, truncated

    def run_git(self, *args: str, timeout: float = 60, cwd: Optional[Path] = None) -> CommandResult:
        """Internal git calls (diff, ls-files). Not subject to the model allowlist."""
        return self.run(["git", *args], timeout=timeout, check_allowlist=False, cwd=cwd)

    # ------------------------------------------------------------ file lists
    def tracked_and_untracked_files(self) -> Optional[list[str]]:
        """Files git knows about plus untracked-but-not-ignored ones. None if not a git repo."""
        try:
            res = self.run_git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
        except SandboxError:
            return None
        if res.returncode != 0:
            return None
        return [p for p in res.stdout.split("\0") if p]


# ------------------------------------------------------------------ worktrees


def _git(repo: Path, *args: str, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def git_root(path: Path | str) -> Optional[Path]:
    try:
        res = _git(Path(path), "rev-parse", "--show-toplevel")
    except (FileNotFoundError, subprocess.TimeoutExpired, NotADirectoryError):
        return None
    if res.returncode != 0:
        return None
    return Path(res.stdout.strip()).resolve()


def new_run_id() -> str:
    return "r_" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _ensure_excluded(repo_root: Path, devagent_dir: str) -> None:
    """Keep .devagent/ out of git status without touching the user's .gitignore."""
    exclude = repo_root / ".git" / "info" / "exclude"
    try:
        if not exclude.parent.is_dir():
            exclude.parent.mkdir(parents=True, exist_ok=True)
        existing = exclude.read_text(encoding="utf-8") if exclude.is_file() else ""
        line = f"{devagent_dir}/"
        if line not in existing.splitlines():
            with exclude.open("a", encoding="utf-8") as f:
                if existing and not existing.endswith("\n"):
                    f.write("\n")
                f.write(line + "\n")
    except OSError:
        pass


def create_worktree(repo_root: Path, run_id: str, *, devagent_dir: str = ".devagent") -> tuple[Path, str]:
    """``git worktree add .devagent/run-<id> -b devagent/<id>``. Returns (path, branch)."""
    repo_root = Path(repo_root).resolve()
    _ensure_excluded(repo_root, devagent_dir)
    wt_dir = repo_root / devagent_dir / f"run-{run_id}"
    wt_dir.parent.mkdir(parents=True, exist_ok=True)
    branch = f"devagent/{run_id}"
    head = _git(repo_root, "rev-parse", "--verify", "HEAD")
    if head.returncode != 0:
        raise SandboxError(
            "The repository has no commits yet; devagent needs at least one commit to create a worktree. "
            "Commit something first, or run with --no-worktree."
        )
    res = _git(repo_root, "worktree", "add", str(wt_dir), "-b", branch)
    if res.returncode != 0:
        raise SandboxError(f"git worktree add failed: {res.stderr.strip() or res.stdout.strip()}")
    return wt_dir.resolve(), branch


def restore_worktree(repo_root: Path, worktree: Path, branch: str) -> Path:
    """Re-attach an existing devagent branch at its worktree path (after the directory was removed)."""
    repo_root = Path(repo_root).resolve()
    _git(repo_root, "worktree", "prune")
    exists = _git(repo_root, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}")
    if exists.returncode != 0:
        raise SandboxError(f"Cannot resume: branch '{branch}' no longer exists in {repo_root}.")
    worktree.parent.mkdir(parents=True, exist_ok=True)
    res = _git(repo_root, "worktree", "add", str(worktree), branch)
    if res.returncode != 0:
        raise SandboxError(f"git worktree add failed while resuming: {res.stderr.strip() or res.stdout.strip()}")
    return worktree.resolve()


def remove_worktree(repo_root: Path, worktree: Path, *, delete_branch: bool = False, branch: str = "") -> None:
    repo_root = Path(repo_root).resolve()
    _git(repo_root, "worktree", "remove", "--force", str(worktree))
    if delete_branch and branch:
        _git(repo_root, "branch", "-D", branch)
