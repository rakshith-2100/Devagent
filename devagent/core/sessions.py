"""Persistent sessions: the conversation lives in the project's .devagent folder.

    .devagent/sessions.sqlite        LangGraph checkpoints, one thread per session, every step
    .devagent/sessions/<run_id>.json a small index record: title, worktree, branch, turns, outcome
    .devagent/memory.md              optional project memory written by the user; injected into context

A session is resumed by re-opening the same checkpoint thread on the same
worktree, so the model sees everything that happened before, including a run
that was interrupted half-way.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .config import Config
from .types import PlanStep

SESSIONS_SUBDIR = "sessions"
DB_NAME = "sessions.sqlite"
MEMORY_NAME = "memory.md"
MAX_GOALS_KEPT = 50


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def devagent_base(repo_root: Path | str, config: Config) -> Path:
    return Path(repo_root) / config.devagent_dir


@dataclass
class SessionMeta:
    run_id: str
    repo_root: str
    root: str
    branch: str
    created: str
    updated: str
    title: str = ""
    mode: str = "default"
    provider: str = ""
    model: str = ""
    turns: int = 0
    last_outcome: str = ""
    goals: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SessionMeta":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def new(cls, *, run_id: str, repo_root: Path, root: Path, branch: str, config: Config) -> "SessionMeta":
        now = _now()
        return cls(
            run_id=run_id,
            repo_root=str(repo_root),
            root=str(root),
            branch=branch,
            created=now,
            updated=now,
            mode=config.mode,
            provider=config.provider,
            model=config.effective_model(),
        )


def sessions_dir(repo_root: Path | str, config: Config) -> Path:
    p = devagent_base(repo_root, config) / SESSIONS_SUBDIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_session(repo_root: Path | str, config: Config, meta: SessionMeta) -> Path:
    path = sessions_dir(repo_root, config) / f"{meta.run_id}.json"
    path.write_text(json.dumps(meta.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def list_sessions(repo_root: Path | str, config: Config) -> list[SessionMeta]:
    d = devagent_base(repo_root, config) / SESSIONS_SUBDIR
    if not d.is_dir():
        return []
    out: list[SessionMeta] = []
    for p in d.glob("*.json"):
        try:
            out.append(SessionMeta.from_dict(json.loads(p.read_text(encoding="utf-8"))))
        except (OSError, ValueError, TypeError):
            continue
    return sorted(out, key=lambda m: m.updated, reverse=True)


def find_session(repo_root: Path | str, config: Config, key: str) -> Optional[SessionMeta]:
    """``key`` is 'last', a 1-based number from the list (newest first), a run id, or a unique suffix."""
    sessions = list_sessions(repo_root, config)
    if not sessions:
        return None
    key = (key or "").strip()
    if key in ("", "last", "latest"):
        return sessions[0]
    if key.isdigit():
        idx = int(key)
        return sessions[idx - 1] if 1 <= idx <= len(sessions) else None
    exact = [s for s in sessions if s.run_id == key]
    if exact:
        return exact[0]
    suffix = [s for s in sessions if s.run_id.endswith(key)]
    return suffix[0] if len(suffix) == 1 else None


def record_turn(repo_root: Path | str, config: Config, meta: SessionMeta, goal: str, outcome: str) -> SessionMeta:
    meta.turns += 1
    meta.updated = _now()
    meta.last_outcome = outcome
    if not meta.title:
        meta.title = " ".join(goal.split())[:80]
    meta.goals = (meta.goals + [" ".join(goal.split())[:200]])[-MAX_GOALS_KEPT:]
    save_session(repo_root, config, meta)
    return meta


def open_checkpointer(repo_root: Path | str, config: Config) -> tuple[Any, Optional[Path]]:
    """SQLite checkpointer in the project; falls back to memory if the package is missing."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:  # pragma: no cover - optional dependency missing
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver(), None
    base = devagent_base(repo_root, config)
    base.mkdir(parents=True, exist_ok=True)
    path = base / DB_NAME
    conn = sqlite3.connect(str(path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver, path


def close_checkpointer(checkpointer: Any) -> None:
    conn = getattr(checkpointer, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def read_project_memory(repo_root: Path | str, config: Config, max_chars: int = 8000) -> str:
    path = devagent_base(repo_root, config) / MEMORY_NAME
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return text[:max_chars] + ("\n... [truncated]" if len(text) > max_chars else "")


def restore_from_checkpoint(rt: Any, graph: Any) -> int:
    """Pull the plan back out of the checkpoint and return how many messages the thread holds."""
    try:
        snapshot = graph.get_state({"configurable": {"thread_id": rt.run_id}})
    except Exception:  # noqa: BLE001 - a thread that was never checkpointed
        return 0
    values = getattr(snapshot, "values", None) or {}
    plan = values.get("plan") or []
    rt.plan = [PlanStep(title=str(s.get("title", "")), status=s.get("status", "pending"), note=str(s.get("note", ""))) for s in plan if isinstance(s, dict)]
    return len(values.get("messages") or [])
