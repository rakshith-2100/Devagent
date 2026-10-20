"""run_command: the allowlisted escape hatch, and the thing to watch in traces."""
from __future__ import annotations

from ..sandbox.local import CommandNotAllowedError
from .base import ToolContext, ToolError, ToolOutput, tool


def format_command_result(
    argv: list[str], returncode: int, stdout: str, stderr: str, duration_ms: int, timed_out: bool
) -> str:
    parts = [f"$ {' '.join(argv)}"]
    if timed_out:
        parts.append(
            f"TIMED OUT after {duration_ms / 1000:.1f}s (process killed). Raise timeout only if the command genuinely needs it."
        )
    else:
        parts.append(f"exit code {returncode} ({duration_ms / 1000:.1f}s)")
    if stdout.strip():
        parts.append("--- stdout ---\n" + stdout.rstrip())
    if stderr.strip():
        parts.append("--- stderr ---\n" + stderr.rstrip())
    if not stdout.strip() and not stderr.strip():
        parts.append("(no output)")
    return "\n".join(parts)


@tool
def run_command(ctx: ToolContext, command: str, timeout: int = 60) -> ToolOutput:
    """Run one allowlisted program inside the working root and return its output.

    No shell: pipes, redirection and command chaining are rejected. Network
    tools are never allowed. Dependency installs and destructive commands stop
    and ask the user. Prefer run_tests, read_file and grep_repo whenever they
    can do the job; this tool is the escape hatch.

    Args:
        command: The program and its arguments, e.g. "python scripts/check.py --fast".
        timeout: Seconds before the process is killed (1-600). Defaults to 60.
    """
    timeout = max(1, min(int(timeout), 600))
    try:
        res = ctx.sandbox.run(command, timeout=timeout, approved=bool(ctx.approved))
    except (CommandNotAllowedError, ValueError) as e:
        raise ToolError(f"run_command: {e}") from e
    text = format_command_result(res.argv, res.returncode, res.stdout, res.stderr, res.duration_ms, res.timed_out)
    return ToolOutput(text, truncated=res.truncated, meta={"returncode": res.returncode, "timed_out": res.timed_out})
