"""A compact repository overview kept near the front of the context.

The directory tree with gitignored paths removed, and for each source file one
line of its top-level symbols. Python is parsed with ``ast``; a few other
languages get a regex pass. Target two to four thousand tokens.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable, Optional

from .budget import estimate_tokens

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", ".idea", ".vscode", ".devagent", ".tox", ".eggs", "vendor", "third_party"}
SOURCE_EXT = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt", ".rb", ".cs", ".php", ".c", ".h", ".cpp", ".hpp"}
MAX_SYMBOLS_PER_FILE = 12
MAX_FILE_BYTES = 400_000

_JS_SYMBOL = re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function\*?\s+(\w+)|class\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>)", re.MULTILINE)
_GO_SYMBOL = re.compile(r"^func\s+(?:\([^)]*\)\s*)?(\w+)|^type\s+(\w+)\s", re.MULTILINE)
_RUST_SYMBOL = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:fn|struct|enum|trait|impl(?:<[^>]*>)?)\s+(\w+)", re.MULTILINE)
_JAVA_SYMBOL = re.compile(r"^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?(?:abstract\s+)?(?:class|interface|enum|record)\s+(\w+)", re.MULTILINE)
_RUBY_SYMBOL = re.compile(r"^\s*(?:class|module|def)\s+([\w.:?!]+)", re.MULTILINE)


def _python_symbols(text: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_")]
            suffix = f"({', '.join(methods[:6])}{', ...' if len(methods) > 6 else ''})" if methods else ""
            out.append(f"class {node.name}{suffix}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args if a.arg not in ("self", "cls")]
            out.append(f"{node.name}({', '.join(args)})")
    return out


def _regex_symbols(pattern: re.Pattern[str], text: str) -> list[str]:
    names: list[str] = []
    for m in pattern.finditer(text):
        name = next((g for g in m.groups() if g), None)
        if name and name not in names:
            names.append(name)
    return names


def file_symbols(path: Path) -> list[str]:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return []
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    ext = path.suffix.lower()
    if ext == ".py":
        syms = _python_symbols(text)
    elif ext in (".js", ".jsx", ".ts", ".tsx"):
        syms = _regex_symbols(_JS_SYMBOL, text)
    elif ext == ".go":
        syms = _regex_symbols(_GO_SYMBOL, text)
    elif ext == ".rs":
        syms = _regex_symbols(_RUST_SYMBOL, text)
    elif ext in (".java", ".kt", ".cs"):
        syms = _regex_symbols(_JAVA_SYMBOL, text)
    elif ext == ".rb":
        syms = _regex_symbols(_RUBY_SYMBOL, text)
    else:
        syms = []
    if len(syms) > MAX_SYMBOLS_PER_FILE:
        syms = syms[:MAX_SYMBOLS_PER_FILE] + [f"... +{len(syms) - MAX_SYMBOLS_PER_FILE}"]
    return syms


def _walk(root: Path) -> list[str]:
    out: list[str] = []
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts[:-1]):
            continue
        out.append(p.relative_to(root).as_posix())
    return out


def _sort_key(rel: str) -> tuple[int, str]:
    """Source first, tests after, docs and config last; shallower paths first within a group."""
    lower = rel.lower()
    ext = Path(rel).suffix.lower()
    if any(seg in ("tests", "test", "__tests__") for seg in lower.split("/")[:-1]) or Path(rel).name.startswith("test_"):
        group = 1
    elif ext in SOURCE_EXT:
        group = 0
    elif ext in (".md", ".rst", ".txt"):
        group = 3
    else:
        group = 2
    return (group, rel.count("/"), rel)


def build_repo_map(root: Path | str, files: Optional[Iterable[str]] = None, *, max_tokens: int = 3000, max_files: int = 400) -> str:
    root = Path(root)
    rel_files = sorted(files) if files is not None else _walk(root)
    rel_files = [f for f in rel_files if not any(part in SKIP_DIRS for part in f.split("/")[:-1])]
    total = len(rel_files)
    if total == 0:
        return "(empty repository)"
    ordered = sorted(rel_files, key=_sort_key)[:max_files]

    lines: list[str] = [f"{total} files. Format: path  [top-level symbols]"]
    used = estimate_tokens(lines[0])
    shown = 0
    cut = False
    for rel in ordered:
        p = root / rel
        syms = file_symbols(p) if p.suffix.lower() in SOURCE_EXT else []
        line = f"{rel}" + (f"  [{'; '.join(syms)}]" if syms else "")
        cost = estimate_tokens(line)
        if used + cost > max_tokens:
            cut = True
            break
        lines.append(line)
        used += cost
        shown += 1
    if cut or shown < total:
        lines.append(f"... {total - shown} more files not shown. Use list_dir(path=...) or find_files(glob=...) to see them.")
    return "\n".join(lines)
