"""Human-readable prompt/output logs for evaluation.

The JSONL trace is for programs; this is for people. One directory per run::

    .devagent/logs/<run_id>/
        system_prompt.txt        the static system prompt
        tools.json               the tool schemas the model saw
        transcript.md            the whole run, readable top to bottom
        calls/NN-<purpose>.request.json   exact request body sent to the model
        calls/NN-<purpose>.response.json  exact response body received
        calls/NN-<purpose>.context.txt    the dynamic context block (repo map, notes)
        summary.json             outcome, usage, cost

Everything is append-only during the run so a crash still leaves a usable log.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def _short(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [{len(text) - limit} more chars; see the request/response json]"


class PromptLogger:
    def __init__(self, directory: Path | str, *, enabled: bool = True, preview_chars: int = 4000) -> None:
        self.dir = Path(directory)
        self.enabled = enabled
        self.preview_chars = preview_chars
        self._lock = threading.Lock()
        self._call_no = 0
        if self.enabled:
            (self.dir / "calls").mkdir(parents=True, exist_ok=True)
            # A resumed session keeps numbering where it left off instead of overwriting earlier calls.
            numbers = [p.name.split("-", 1)[0] for p in (self.dir / "calls").glob("*.request.json")]
            self._call_no = max((int(n) for n in numbers if n.isdigit()), default=0)
            self._append(
                "transcript.md", f"# devagent run {self.dir.name}\n\nStarted {datetime.now().isoformat(timespec='seconds')}\n"
            )

    # ------------------------------------------------------------ low level
    def _append(self, name: str, text: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            with (self.dir / name).open("a", encoding="utf-8") as f:
                f.write(text)
                if not text.endswith("\n"):
                    f.write("\n")

    def _write(self, name: str, text: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            (self.dir / name).write_text(text, encoding="utf-8")

    def _dump(self, name: str, data: Any) -> None:
        self._write(name, json.dumps(data, indent=2, ensure_ascii=False, default=str))

    # ---------------------------------------------------------------- events
    def log_static(self, system_prompt: str, tools: list[dict[str, Any]]) -> None:
        self._write("system_prompt.txt", system_prompt)
        self._dump("tools.json", tools)
        names = ", ".join(t.get("name", "?") for t in tools)
        self._append(
            "transcript.md",
            f"\n## System prompt\n\nSaved to `system_prompt.txt` ({len(system_prompt)} chars). Tools: {names}\n",
        )

    def log_goal(self, goal: str) -> None:
        self._append("transcript.md", f"\n## Goal\n\n{goal}\n")

    def log_event(self, text: str) -> None:
        self._append("transcript.md", f"\n> {text}\n")

    def next_call_no(self) -> int:
        with self._lock:
            self._call_no += 1
            return self._call_no

    def log_model_request(self, call_no: int, purpose: str, body: dict[str, Any], *, dynamic_context: str = "") -> None:
        self._dump(f"calls/{call_no:02d}-{purpose}.request.json", body)
        if dynamic_context:
            self._write(f"calls/{call_no:02d}-{purpose}.context.txt", dynamic_context)
        n_msgs = len(body.get("messages", []))
        self._append(
            "transcript.md",
            f"\n### Model call {call_no} ({purpose})\n\nRequest: {n_msgs} messages -> `calls/{call_no:02d}-{purpose}.request.json`\n",
        )

    def log_model_response(
        self,
        call_no: int,
        purpose: str,
        raw: dict[str, Any],
        *,
        text: str,
        tool_calls: list[dict[str, Any]],
        usage: dict[str, int],
        latency_ms: int,
        stop_reason: str,
    ) -> None:
        self._dump(f"calls/{call_no:02d}-{purpose}.response.json", raw)
        parts = [f"Response: stop={stop_reason} latency={latency_ms}ms usage={usage}"]
        if text:
            parts.append("\n**Assistant:**\n\n" + _short(text, self.preview_chars))
        for tc in tool_calls:
            args = json.dumps(tc.get("args", {}), ensure_ascii=False, default=str)
            parts.append(
                f"\n**Tool call** `{tc.get('name')}` ({tc.get('id')}):\n\n```json\n{_short(args, self.preview_chars)}\n```"
            )
        self._append("transcript.md", "\n".join(parts) + "\n")

    def log_decision(self, call_id: str, name: str, verdict: str, rule: str, reason: str) -> None:
        tail = f" - {reason}" if verdict != "allow" else ""
        self._append("transcript.md", f"- policy `{name}` ({call_id}): **{verdict}** via `{rule}`{tail}")

    def log_approval(self, call_id: str, action: str, note: str = "") -> None:
        self._append("transcript.md", f"- approval ({call_id}): **{action}**" + (f" - {note}" if note else ""))

    def log_tool_result(self, call_id: str, name: str, ok: bool, content: str, truncated: bool, duration_ms: int) -> None:
        status = "ok" if ok else "ERROR"
        flag = " (truncated)" if truncated else ""
        self._append(
            "transcript.md",
            f"\n**Tool result** `{name}` ({call_id}) {status}{flag} {duration_ms}ms:\n\n```\n{_short(content, self.preview_chars)}\n```\n",
        )

    def log_compaction(self, before_tokens: int, after_tokens: int, summary: str) -> None:
        self._append(
            "transcript.md",
            f"\n### Context compaction\n\n~{before_tokens} -> ~{after_tokens} tokens\n\n{_short(summary, self.preview_chars)}\n",
        )

    def log_plan(self, steps: list[dict[str, Any]]) -> None:
        lines = [f"- [{s.get('status', 'pending')}] {s.get('title', '')}" for s in steps]
        self._append("transcript.md", "\n**Plan updated:**\n\n" + "\n".join(lines) + "\n")

    def finish(self, summary: dict[str, Any]) -> None:
        self._dump("summary.json", summary)
        self._append(
            "transcript.md",
            f"\n## Finished\n\n```json\n{json.dumps(summary, indent=2, ensure_ascii=False, default=str)}\n```\n",
        )

    @property
    def transcript_path(self) -> Optional[Path]:
        return (self.dir / "transcript.md") if self.enabled else None
