"""Append-only JSONL tracer. One file per run, one object per span, no database."""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Tracer:
    def __init__(self, path: Path | str, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._t0 = time.monotonic()

    # ------------------------------------------------------------ primitives
    def emit(self, t: str, **fields: Any) -> dict[str, Any]:
        event: dict[str, Any] = {
            "t": t,
            "run_id": self.run_id,
            "ts": _now_iso(),
            "elapsed_ms": int((time.monotonic() - self._t0) * 1000),
        }
        event.update(fields)
        line = json.dumps(event, ensure_ascii=False, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        return event

    # ----------------------------------------------------------- span types
    def run_start(
        self,
        *,
        goal: str,
        repo: str,
        root: str,
        branch: str,
        model: str,
        config: dict[str, Any],
        parent_run_id: Optional[str] = None,
    ) -> None:
        self.emit(
            "run_start", goal=goal, repo=repo, root=root, branch=branch, model=model, config=config, parent_run_id=parent_run_id
        )

    def model_call(
        self,
        *,
        step: int,
        purpose: str,
        in_tokens: int,
        out_tokens: int,
        cache_read: int,
        cache_write: int,
        latency_ms: int,
        stop_reason: str,
        breakdown: dict[str, int],
        cost_usd: float,
        tool_calls: int,
    ) -> None:
        self.emit(
            "model_call",
            step=step,
            purpose=purpose,
            in_tokens=in_tokens,
            out_tokens=out_tokens,
            cache_read=cache_read,
            cache_write=cache_write,
            latency_ms=latency_ms,
            stop_reason=stop_reason,
            breakdown=breakdown,
            cost_usd=round(cost_usd, 6),
            tool_calls=tool_calls,
        )

    def tool_call(self, *, step: int, call_id: str, name: str, args: dict[str, Any]) -> None:
        self.emit("tool_call", step=step, id=call_id, name=name, args=args)

    def decision(self, *, step: int, call_id: str, verdict: str, rule: str, tier: int, reason: str) -> None:
        self.emit("decision", step=step, id=call_id, verdict=verdict, rule=rule, tier=tier, reason=reason)

    def approval(self, *, step: int, call_id: str, action: str, note: str = "") -> None:
        self.emit("approval", step=step, id=call_id, action=action, note=note)

    def tool_result(
        self, *, step: int, call_id: str, name: str, ok: bool, size: int, truncated: bool, duration_ms: int, preview: str = ""
    ) -> None:
        self.emit(
            "tool_result",
            step=step,
            id=call_id,
            name=name,
            ok=ok,
            bytes=size,
            truncated=truncated,
            duration_ms=duration_ms,
            preview=preview[:200],
        )

    def compaction(self, *, step: int, before_tokens: int, after_tokens: int, dropped_messages: int) -> None:
        self.emit(
            "compaction", step=step, before_tokens=before_tokens, after_tokens=after_tokens, dropped_messages=dropped_messages
        )

    def note(self, text: str, **fields: Any) -> None:
        self.emit("note", text=text, **fields)

    def run_end(
        self,
        *,
        outcome: str,
        steps: int,
        cost_usd: float,
        wall_s: float,
        tests_passed: Optional[bool],
        stop_reason: str,
        usage: dict[str, int],
    ) -> None:
        self.emit(
            "run_end",
            outcome=outcome,
            steps=steps,
            cost_usd=round(cost_usd, 6),
            wall_s=round(wall_s, 2),
            tests_passed=tests_passed,
            stop_reason=stop_reason,
            usage=usage,
        )

    # --------------------------------------------------------------- reading
    @staticmethod
    def read(path: Path | str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        p = Path(path)
        if not p.is_file():
            return events
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                events.append({"t": "corrupt_line", "raw": line[:200]})
        return events

    @staticmethod
    def iter_runs(traces_dir: Path | str) -> list[Path]:
        d = Path(traces_dir)
        if not d.is_dir():
            return []
        return sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
