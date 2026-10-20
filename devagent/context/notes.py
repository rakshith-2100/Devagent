"""NOTES.md: the agent's scratchpad in the worktree, read at the start of every
step and appended to as it works."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path


class NotesFile:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.is_file()

    def read(self, max_chars: int = 12_000) -> str:
        if not self.path.is_file():
            return "(no notes yet)"
        text = self.path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return "(no notes yet)"
        if len(text) > max_chars:
            return "... [older notes omitted]\n" + text[-max_chars:]
        return text

    def append(self, text: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        block = f"\n### {stamp}\n{text.strip()}\n"
        if not self.path.is_file():
            self.path.write_text("# NOTES\n\nWorking notes kept by devagent during this run.\n" + block, encoding="utf-8")
        else:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(block)

    def line_count(self) -> int:
        if not self.path.is_file():
            return 0
        return len(self.path.read_text(encoding="utf-8", errors="replace").splitlines())
