"""Filesystem tools: read_file, list_dir, write_file, edit_file."""
from __future__ import annotations

import difflib
import os
from pathlib import Path
from typing import Optional

from .base import ToolContext, ToolError, ToolOutput, tool

READ_CAP_LINES = 400
MAX_LINE_CHARS = 2000
LIST_CAP_ENTRIES = 300
DEFAULT_IGNORES = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".idea",
    ".vscode",
    ".devagent",
    ".tox",
    ".eggs",
}


def _read_text(p: Path) -> str:
    data = p.read_bytes()
    if b"\x00" in data[:8000]:
        raise ToolError(f"read_file: '{p.name}' looks like a binary file; only text files can be read.")
    return data.decode("utf-8", errors="replace")


def _detect_eol(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _write_text(p: Path, text: str) -> None:
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def repo_files(ctx: ToolContext, under: Optional[Path] = None) -> list[str]:
    """Relative paths of files the agent may care about: git-aware when possible."""
    listed = ctx.sandbox.tracked_and_untracked_files()
    if listed is None:
        listed = []
        root = ctx.sandbox.root
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in DEFAULT_IGNORES)
            for fn in filenames:
                listed.append(Path(dirpath, fn).relative_to(root).as_posix())
    else:
        listed = [p for p in listed if not any(part in DEFAULT_IGNORES for part in p.split("/")[:-1])]
    if under is not None and under != ctx.sandbox.root:
        prefix = ctx.sandbox.relpath(under).rstrip("/") + "/"
        listed = [p for p in listed if p.startswith(prefix)]
    return sorted(set(listed))


@tool
def read_file(ctx: ToolContext, path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> ToolOutput:
    """Read a text file and return its lines with line numbers.

    Returns at most 400 lines per call and says how many remain. Use start_line
    and end_line to read a specific region of a long file.

    Args:
        path: File path relative to the working root, e.g. "src/app.py".
        start_line: First line to return (1-based). Defaults to 1.
        end_line: Last line to return, inclusive. Defaults to start_line + 399.
    """
    p = ctx.sandbox.resolve(path)
    rel = ctx.sandbox.relpath(p)
    if not p.exists():
        raise ToolError(f"read_file: '{path}' does not exist. Use find_files(glob=...) or list_dir(path=...) to locate it.")
    if p.is_dir():
        raise ToolError(f"read_file: '{path}' is a directory. Use list_dir(path=\"{path}\") instead.")
    text = _read_text(p)
    lines = text.splitlines()
    total = len(lines)
    if total == 0:
        return ToolOutput(f"{rel} is empty (0 lines).", meta={"lines": 0})
    start = max(1, int(start_line) if start_line else 1)
    if start > total:
        raise ToolError(f"read_file: start_line {start} is past the end of '{rel}', which has {total} lines.")
    end = int(end_line) if end_line else start + READ_CAP_LINES - 1
    end = min(end, total, start + READ_CAP_LINES - 1)
    if end < start:
        raise ToolError(f"read_file: end_line ({end_line}) must be >= start_line ({start}).")
    body_lines = []
    for i, line in enumerate(lines[start - 1 : end], start=start):
        if len(line) > MAX_LINE_CHARS:
            line = line[:MAX_LINE_CHARS] + f" ... [line truncated, {len(line)} chars]"
        body_lines.append(f"{i:>5}| {line}")
    header = f"{rel} (lines {start}-{end} of {total})"
    footer = ""
    if end < total:
        footer = f"\n... {total - end} more lines. Call read_file(path=\"{rel}\", start_line={end + 1}) to continue."
    return ToolOutput(
        header + "\n" + "\n".join(body_lines) + footer,
        truncated=end < total,
        meta={"lines": total, "shown": end - start + 1},
    )


@tool
def list_dir(ctx: ToolContext, path: str = ".", depth: int = 2) -> ToolOutput:
    """List files and directories under a path as an indented tree.

    Respects .gitignore when the repository is a git repo. Directories deeper
    than ``depth`` are shown collapsed with a count of what they contain.

    Args:
        path: Directory relative to the working root. Defaults to the root.
        depth: How many levels to expand (1-6). Defaults to 2.
    """
    base = ctx.sandbox.resolve(path)
    if not base.exists():
        raise ToolError(f"list_dir: '{path}' does not exist.")
    if not base.is_dir():
        raise ToolError(f"list_dir: '{path}' is a file; use read_file(path=\"{path}\").")
    depth = max(1, min(int(depth), 6))
    files = repo_files(ctx, base)
    base_rel = ctx.sandbox.relpath(base)
    prefix_len = 0 if base == ctx.sandbox.root else len(base_rel) + 1
    tree: dict = {}
    for rel in files:
        parts = rel[prefix_len:].split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part + "/", {})
        node[parts[-1]] = None

    lines: list[str] = []
    truncated = False

    def count_leaves(node: dict) -> int:
        return sum(1 if v is None else count_leaves(v) for v in node.values())

    def walk(node: dict, level: int, indent: str) -> None:
        nonlocal truncated
        entries = sorted(node.items(), key=lambda kv: (kv[1] is None, kv[0].lower()))
        for name, child in entries:
            if len(lines) >= LIST_CAP_ENTRIES:
                truncated = True
                return
            if child is None:
                lines.append(f"{indent}{name}")
            elif level < depth:
                lines.append(f"{indent}{name}")
                walk(child, level + 1, indent + "  ")
            else:
                lines.append(f"{indent}{name} ({count_leaves(child)} files)")

    walk(tree, 1, "")
    header = f"{base_rel if base != ctx.sandbox.root else '.'}/ ({len(files)} files total, depth {depth})"
    out = header + "\n" + "\n".join(lines) if lines else header + "\n(empty)"
    if truncated:
        out += f"\n... listing capped at {LIST_CAP_ENTRIES} entries. Narrow with path=\"<subdir>\" or use find_files(glob=...)."
    return ToolOutput(out, truncated=truncated, meta={"files": len(files)})


@tool
def write_file(ctx: ToolContext, path: str, content: str) -> ToolOutput:
    """Create a new text file. Fails if the file already exists.

    Overwriting an existing file is not allowed here; use edit_file for changes
    to existing files so the diff stays reviewable.

    Args:
        path: File path relative to the working root. Parent directories are created.
        content: The full text content of the new file.
    """
    p = ctx.sandbox.resolve(path)
    rel = ctx.sandbox.relpath(p)
    if p.exists():
        raise ToolError(
            f"write_file: '{rel}' already exists. Use edit_file(path=\"{rel}\", old_string=..., new_string=...) "
            "to change it; read it first with read_file if you have not."
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    _write_text(p, content)
    n = len(content.splitlines())
    if ctx.runtime is not None:
        ctx.runtime.files_changed.add(rel)
    return ToolOutput(f"Created {rel} ({n} lines).", meta={"lines": n})


def _not_found_message(rel: str, old: str, text: str) -> str:
    base = f"edit_file: old_string not found in {rel}. The file may have changed since you read it."
    stripped_old = "\n".join(l.rstrip() for l in old.splitlines())
    stripped_text = "\n".join(l.rstrip() for l in text.splitlines())
    if stripped_old and stripped_old != old and stripped_old in stripped_text:
        return base + " The text matches except for trailing whitespace; copy the exact lines from read_file output."
    first = old.splitlines()[0].strip() if old.splitlines() else ""
    if first:
        for i, line in enumerate(text.splitlines(), start=1):
            if first in line:
                lo, hi = max(1, i - 5), i + 20
                return (
                    base
                    + f" The first line of old_string appears near line {i}; re-read with "
                    + f"read_file(path=\"{rel}\", start_line={lo}, end_line={hi}) and retry with the exact current text."
                )
    return base + f" Re-read the relevant section with read_file(path=\"{rel}\") and retry with the exact current text."


@tool
def edit_file(ctx: ToolContext, path: str, old_string: str, new_string: str, replace_all: bool = False) -> ToolOutput:
    """Replace an exact string in a file.

    The call succeeds only if old_string appears exactly once (or replace_all is
    true). Include enough surrounding lines to make the match unique. Read the
    file first; if the match fails, the file has changed since you last read it.

    Args:
        path: File path relative to the working root.
        old_string: The exact current text to replace, including indentation.
        new_string: The replacement text.
        replace_all: Replace every occurrence instead of requiring a unique match.
    """
    p = ctx.sandbox.resolve(path)
    rel = ctx.sandbox.relpath(p)
    if not p.exists():
        raise ToolError(f"edit_file: '{rel}' does not exist. Use write_file to create new files.")
    if p.is_dir():
        raise ToolError(f"edit_file: '{rel}' is a directory.")
    if old_string == "":
        raise ToolError("edit_file: old_string is empty. Provide the exact text to replace; to create a file use write_file.")
    if old_string == new_string:
        raise ToolError("edit_file: old_string and new_string are identical; nothing to do.")
    original = _read_text(p)
    eol = _detect_eol(original)
    text = original.replace("\r\n", "\n")
    old = old_string.replace("\r\n", "\n")
    new = new_string.replace("\r\n", "\n")
    count = text.count(old)
    if count == 0:
        raise ToolError(_not_found_message(rel, old, text))
    if count > 1 and not replace_all:
        raise ToolError(
            f"edit_file: old_string appears {count} times in {rel}. Include more surrounding context to make it unique, "
            "or pass replace_all=true to replace every occurrence."
        )
    updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    diff = "\n".join(
        difflib.unified_diff(text.splitlines(), updated.splitlines(), fromfile=f"a/{rel}", tofile=f"b/{rel}", lineterm="", n=2)
    )
    _write_text(p, updated.replace("\n", eol) if eol == "\r\n" else updated)
    if ctx.runtime is not None:
        ctx.runtime.files_changed.add(rel)
    replaced = count if replace_all else 1
    if len(diff) > 6000:
        diff = diff[:6000] + "\n... [diff preview truncated]"
    return ToolOutput(f"Replaced {replaced} occurrence(s) in {rel}.\n{diff}", meta={"replaced": replaced})
