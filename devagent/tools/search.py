"""Search tools: grep_repo (content) and find_files (names). Kept separate
because the model confuses "find by name" with "find by content"."""
from __future__ import annotations

import re
import shutil
import subprocess
from typing import Optional

from .base import ToolContext, ToolError, ToolOutput, tool
from .fs import repo_files

MAX_LINE_PREVIEW = 300
MAX_FILE_BYTES = 2_000_000


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a glob with ``**`` into a regex over posix relative paths."""
    pattern = pattern.replace("\\", "/")
    anchored = "/" in pattern.rstrip("/")
    out = ""
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "*":
            if pattern[i : i + 3] == "**/":
                out += "(?:.*/)?"
                i += 3
                continue
            if pattern[i : i + 2] == "**":
                out += ".*"
                i += 2
                continue
            out += "[^/]*"
        elif c == "?":
            out += "[^/]"
        elif c in ".+^$()[]{}|":
            out += "\\" + c
        else:
            out += c
        i += 1
    if anchored:
        return re.compile("^" + out + "$")
    return re.compile("(^|/)" + out + "$")


def _matches_glob(rel: str, pattern: Optional[str]) -> bool:
    if not pattern:
        return True
    return bool(glob_to_regex(pattern).search(rel))


def _rg_available() -> Optional[str]:
    return shutil.which("rg")


def _scan_with_rg(ctx: ToolContext, rg: str, pattern: str, glob: Optional[str], case_sensitive: bool, literal: bool, max_results: int):
    argv = [rg, "--no-heading", "--line-number", "--color", "never", "--no-messages"]
    if literal:
        argv.append("-F")
    if not case_sensitive:
        argv.append("-i")
    if glob:
        argv += ["-g", glob]
    argv += ["-e", pattern, "."]
    proc = subprocess.run(
        argv,
        cwd=str(ctx.sandbox.root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        env=ctx.sandbox.build_env(),
    )
    matches: list[tuple[str, int, str]] = []
    total = 0
    seen_files: set[str] = set()
    for line in proc.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        path = parts[0].replace("\\", "/")
        if path.startswith("./"):
            path = path[2:]
        total += 1
        seen_files.add(path)
        if len(matches) < max_results:
            try:
                matches.append((path, int(parts[1]), parts[2]))
            except ValueError:
                continue
    return matches, total, len(seen_files)


def _scan_with_python(ctx: ToolContext, regex: re.Pattern[str], glob: Optional[str], max_results: int):
    matches: list[tuple[str, int, str]] = []
    total = 0
    files_with_matches = 0
    for rel in repo_files(ctx):
        if not _matches_glob(rel, glob):
            continue
        p = ctx.sandbox.root / rel
        try:
            if p.stat().st_size > MAX_FILE_BYTES:
                continue
            data = p.read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:8000]:
            continue
        text = data.decode("utf-8", errors="replace")
        found = False
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                total += 1
                found = True
                if len(matches) < max_results:
                    matches.append((rel, lineno, line))
        if found:
            files_with_matches += 1
    return matches, total, files_with_matches


@tool
def grep_repo(
    ctx: ToolContext, pattern: str, glob: Optional[str] = None, max_results: int = 40, case_sensitive: bool = False
) -> ToolOutput:
    """Search file contents with a regular expression.

    Returns file path, line number and the matching line, plus the total match
    count so you know whether to narrow the search. Binary files, ignored paths
    and files over 2 MB are skipped.

    Args:
        pattern: Regular expression (Python/ripgrep syntax). Escape literal dots and parentheses.
        glob: Restrict to paths matching this glob, e.g. "*.py" or "src/**/*.ts".
        max_results: Maximum matching lines to return (1-200). Defaults to 40.
        case_sensitive: Match case exactly. Defaults to false.
    """
    if not pattern:
        raise ToolError("grep_repo: pattern is empty.")
    max_results = max(1, min(int(max_results), 200))
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
        literal = False
    except re.error:
        regex = re.compile(re.escape(pattern), flags)
        literal = True

    rg = _rg_available()
    matches: list[tuple[str, int, str]] = []
    total = files_with_matches = 0
    if rg:
        try:
            matches, total, files_with_matches = _scan_with_rg(ctx, rg, pattern, glob, case_sensitive, literal, max_results)
        except (subprocess.TimeoutExpired, OSError):
            rg = None
    if not rg:
        matches, total, files_with_matches = _scan_with_python(ctx, regex, glob, max_results)

    if total == 0:
        note = " (pattern was not a valid regex, so it was matched literally)" if literal else ""
        scope = f" in files matching {glob}" if glob else ""
        return ToolOutput(
            f"No matches for /{pattern}/{scope}{note}. Try a shorter pattern, a different spelling, or find_files for file names."
        )
    lines = []
    for path, lineno, content in matches:
        content = content.strip()
        if len(content) > MAX_LINE_PREVIEW:
            content = content[:MAX_LINE_PREVIEW] + " ..."
        lines.append(f"{path}:{lineno}: {content}")
    header = f"{total} match(es) in {files_with_matches} file(s) for /{pattern}/" + (f" (glob {glob})" if glob else "")
    truncated = total > len(matches)
    footer = (
        f"\nShowing {len(matches)} of {total}. Narrow with glob=... or a more specific pattern, or raise max_results."
        if truncated
        else ""
    )
    return ToolOutput(header + "\n" + "\n".join(lines) + footer, truncated=truncated, meta={"total": total})


@tool
def find_files(ctx: ToolContext, glob: str, max_results: int = 100) -> ToolOutput:
    """Find files by name or path pattern.

    Use this to locate files when you know (part of) the name; use grep_repo to
    search file contents.

    Args:
        glob: Pattern such as "*.py", "test_*.py", "src/**/*.ts" or "**/config*". A bare name matches anywhere in the tree.
        max_results: Maximum paths to return (1-500). Defaults to 100.
    """
    if not glob:
        raise ToolError('find_files: glob is empty. Example: find_files(glob="*.py").')
    max_results = max(1, min(int(max_results), 500))
    pattern = glob if any(ch in glob for ch in "*?[") else f"*{glob}*"
    regex = glob_to_regex(pattern)
    hits = [rel for rel in repo_files(ctx) if regex.search(rel)]
    if not hits:
        bare = glob.strip("*")
        return ToolOutput(f'No files match {glob}. Try list_dir to see the layout, or a looser pattern like "**/*{bare}*".')
    shown = hits[:max_results]
    out = f"{len(hits)} file(s) match {glob}:\n" + "\n".join(shown)
    truncated = len(hits) > len(shown)
    if truncated:
        out += f"\n... {len(hits) - len(shown)} more. Narrow the pattern or raise max_results."
    return ToolOutput(out, truncated=truncated, meta={"total": len(hits)})
