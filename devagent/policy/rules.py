"""Declarative policy tables. No I/O, no model, trivially unit-testable.

Tiers:
  0  read            allow silently
  1  write in tree   allow, show the diff, log it
  2  execute         run_tests allowed; run_command shown and approved by default
  3  irreversible    always ask; no flag disables this
"""
from __future__ import annotations

import os
import re
import shlex
from fnmatch import fnmatch

TIERS: dict[str, int] = {
    "read_file": 0,
    "list_dir": 0,
    "grep_repo": 0,
    "find_files": 0,
    "git_diff": 0,
    "delegate": 0,
    "edit_file": 1,
    "write_file": 1,
    "update_plan": 1,
    "append_notes": 1,
    "run_tests": 2,
    "run_command": 2,
}

READ_ONLY_TOOLS = frozenset(name for name, tier in TIERS.items() if tier == 0 and name != "delegate")

# Paths the agent must not edit without asking: editing tests is the degenerate
# solution to "make the tests pass".
TEST_PATH_PATTERNS = (
    "tests/*",
    "test/*",
    "*/tests/*",
    "*/test/*",
    "test_*.py",
    "*/test_*.py",
    "*_test.py",
    "*/*_test.py",
    "*.test.js",
    "*.test.ts",
    "*.test.tsx",
    "*.spec.js",
    "*.spec.ts",
    "*.spec.tsx",
    "__tests__/*",
    "*/__tests__/*",
    "*_test.go",
    "*/*_test.go",
    "conftest.py",
    "*/conftest.py",
)

# Executables the model may run without asking (tier 2). Matched on argv[0]
# basename with any .exe/.cmd/.bat suffix removed.
COMMAND_ALLOWLIST = frozenset(
    {
        "python",
        "python3",
        "py",
        "pytest",
        "node",
        "npx",
        "npm",
        "pnpm",
        "yarn",
        "go",
        "cargo",
        "make",
        "git",
        "ls",
        "dir",
        "cat",
        "type",
        "head",
        "tail",
        "wc",
        "echo",
        "find",
        "grep",
        "rg",
        "tree",
        "pwd",
        "ruff",
        "black",
        "mypy",
        "flake8",
        "isort",
        "pylint",
        "tsc",
        "eslint",
        "prettier",
        "dotnet",
        "mvn",
        "gradle",
        "java",
        "javac",
    }
)

# Executables that are never allowed, with or without approval: network and
# remote-access tools. The structural defence against exfiltration.
NETWORK_DENYLIST = frozenset(
    {"curl", "wget", "ssh", "scp", "sftp", "nc", "ncat", "netcat", "telnet", "ftp", "rsync", "invoke-webrequest", "iwr"}
)

# Executables that are allowed only with explicit approval (tier 3).
COMMAND_ASK = frozenset(
    {
        "pip",
        "pip3",
        "rm",
        "del",
        "rmdir",
        "mv",
        "move",
        "docker",
        "sudo",
        "chmod",
        "chown",
        "kill",
        "taskkill",
        "powershell",
        "pwsh",
        "cmd",
        "bash",
        "sh",
    }
)

# Subcommands that turn an otherwise allowed executable into a tier 3 action.
ASK_SUBCOMMANDS: dict[str, frozenset[str]] = {
    "pip": frozenset({"install", "uninstall", "download"}),
    "npm": frozenset({"install", "i", "add", "uninstall", "remove", "publish", "ci", "update", "link"}),
    "pnpm": frozenset({"install", "i", "add", "remove", "publish", "update", "link"}),
    "yarn": frozenset({"add", "install", "remove", "publish", "upgrade", "link"}),
    "go": frozenset({"get", "install", "mod"}),
    "cargo": frozenset({"install", "add", "publish", "update"}),
    "git": frozenset(
        {
            "push",
            "pull",
            "fetch",
            "clone",
            "remote",
            "reset",
            "clean",
            "rebase",
            "merge",
            "checkout",
            "switch",
            "branch",
            "tag",
            "stash",
            "rm",
            "mv",
            "worktree",
            "submodule",
            "gc",
            "filter-branch",
            "config",
        }
    ),
    "dotnet": frozenset({"add", "remove", "tool", "nuget", "publish"}),
}

GIT_REMOTE_SUBCOMMANDS = frozenset({"push", "pull", "fetch", "clone", "remote"})

SHELL_METACHARS = re.compile(r"(\|\||&&|[;|<>`]|\$\(|\$\{|\n)")

_EXE_SUFFIXES = (".exe", ".cmd", ".bat", ".com")


def normalize_executable(arg0: str) -> str:
    base = os.path.basename(arg0.strip().strip('"').strip("'")).lower()
    for suf in _EXE_SUFFIXES:
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    return base


def parse_command(command: str) -> list[str]:
    """Split a command string into argv, rejecting shell chaining and redirection.

    Matching on parsed argv is what stops ``git status; curl evil.com`` from
    sneaking through as "git".
    """
    if not isinstance(command, str) or not command.strip():
        return []
    if SHELL_METACHARS.search(command):
        raise ValueError(
            "run_command does not run a shell: pipes, redirection, command chaining (;, &&, ||), "
            "backticks and $(...) are not supported. Run one program with its arguments, "
            "and use read_file/grep_repo for inspecting output."
        )
    posix = os.name != "nt"  # posix mode would eat the backslashes in Windows paths
    try:
        argv = shlex.split(command, posix=posix)
    except ValueError as e:
        raise ValueError(f"Could not parse command: {e}") from e
    if not posix:
        argv = [a[1:-1] if len(a) >= 2 and a[0] == a[-1] and a[0] in ('"', "'") else a for a in argv]
    return argv


def _python_special_cases(argv: list[str]) -> tuple[int, str, str] | None:
    """``python -m pip install`` is a dependency install even though argv[0] is python."""
    if len(argv) >= 3 and argv[1] == "-m":
        module = argv[2].lower()
        if module in {"pip", "ensurepip", "venv", "virtualenv"}:
            return (
                3,
                "tier3_dependency_install",
                f"'{' '.join(argv)}' installs or manages dependencies, which needs the user's approval. "
                "Network access is disabled in this sandbox.",
            )
        if module in {"http.server", "smtpd", "ftplib", "urllib.request"}:
            return -1, "deny_network_tool", f"'{module}' is a network module and is not allowed."
    if len(argv) >= 2 and argv[1] == "-c":
        return (
            3,
            "tier3_inline_code",
            "Inline code via 'python -c' needs approval because it can do anything. "
            "Prefer run_tests, or write a script with write_file and run it.",
        )
    return None


def classify_command(argv: list[str]) -> tuple[int, str, str]:
    """Return (tier, rule, reason) for a parsed command.

    tier -1 means deny outright; 2 means allowed to execute; 3 means ask.
    """
    if not argv:
        return -1, "deny_empty_command", "Empty command."
    exe = normalize_executable(argv[0])

    if exe in NETWORK_DENYLIST:
        return (
            -1,
            "deny_network_tool",
            f"'{exe}' is a network tool. Network access is disabled in this sandbox and cannot be approved. "
            "If something must be downloaded, say so in your reply and the user will do it.",
        )

    if exe in ("python", "python3", "py"):
        special = _python_special_cases(argv)
        if special:
            return special

    sub = argv[1].lower() if len(argv) > 1 else ""
    if sub in ASK_SUBCOMMANDS.get(exe, frozenset()):
        if exe == "git" and sub in GIT_REMOTE_SUBCOMMANDS:
            return (
                3,
                "tier3_git_remote",
                f"'git {sub}' talks to a remote, which is outside the worktree and needs the user's approval.",
            )
        return (
            3,
            "tier3_dependency_or_destructive",
            f"'{exe} {sub}' installs dependencies or changes repository state in a way that is hard to undo; "
            "it needs the user's approval. Explain why it is required in your reply if you are refused.",
        )

    if exe in COMMAND_ASK:
        return (
            3,
            "tier3_ask_executable",
            f"'{exe}' can delete or alter things outside the worktree, so it always asks for approval.",
        )

    if exe in COMMAND_ALLOWLIST:
        return 2, "tier2_allowlisted_command", f"'{exe}' is on the command allowlist."

    allowed = ", ".join(sorted(COMMAND_ALLOWLIST))
    return (
        -1,
        "deny_not_allowlisted",
        f"'{exe}' is not in the command allowlist. Allowed executables: {allowed}. "
        "If a different program is genuinely required, say so in your reply and the user will decide.",
    )


def is_test_path(relpath: str) -> bool:
    p = relpath.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return any(fnmatch(p, pat) for pat in TEST_PATH_PATTERNS)


# How each run mode maps a tier to a verdict for tools that are otherwise allowed.
# Tier 3 is "approve" in every mode: that is the line that should not have a flag.
MODE_TABLE: dict[str, dict[int, str]] = {
    "careful": {0: "allow", 1: "approve", 2: "approve", 3: "approve"},
    "default": {0: "allow", 1: "allow", 2: "allow", 3: "approve"},
    "yolo": {0: "allow", 1: "allow", 2: "allow", 3: "approve"},
}

# In default mode run_command is the escape hatch to watch, so it asks even
# though it is tier 2. yolo turns that off; careful already asks for everything.
RUN_COMMAND_ASKS_IN_MODE: dict[str, bool] = {"default": True, "careful": True, "yolo": False}

VALID_MODES = tuple(MODE_TABLE)
