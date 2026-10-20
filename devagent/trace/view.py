"""``devagent trace --last``: a readable timeline from the JSONL trace."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .tracer import Tracer


def list_runs(traces_dir: Path | str) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in Tracer.iter_runs(traces_dir):
        events = Tracer.read(path)
        start = next((e for e in events if e.get("t") == "run_start"), {})
        end = next((e for e in reversed(events) if e.get("t") == "run_end"), {})
        steps = max((e.get("step", 0) for e in events if isinstance(e.get("step"), int)), default=0)
        runs.append(
            {
                "run_id": path.stem,
                "path": str(path),
                "goal": start.get("goal", ""),
                "model": start.get("model", ""),
                "outcome": end.get("outcome", "(running or crashed)"),
                "steps": steps,
                "cost_usd": end.get("cost_usd"),
                "wall_s": end.get("wall_s"),
                "started": start.get("ts", ""),
            }
        )
    return runs


def find_run(traces_dir: Path | str, run_id: Optional[str] = None) -> Optional[Path]:
    paths = list(Tracer.iter_runs(traces_dir))
    if not paths:
        return None
    if run_id is None:
        return paths[-1]
    for p in paths:
        if p.stem == run_id or p.stem.endswith(run_id):
            return p
    return None


def _fmt_args(args: dict[str, Any], limit: int = 100) -> str:
    s = json.dumps(args, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def render_timeline(events: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for e in events:
        t = e.get("t")
        step = e.get("step")
        prefix = f"[{step:>2}] " if isinstance(step, int) else "     "
        if t == "run_start":
            lines.append(f"RUN {e.get('run_id')}  model={e.get('model')}  branch={e.get('branch')}")
            lines.append(f"     goal: {e.get('goal')}")
        elif t == "model_call":
            bd = e.get("breakdown") or {}
            bd_s = " ".join(f"{k}={v}" for k, v in bd.items())
            lines.append(
                f"{prefix}model({e.get('purpose')}) in={e.get('in_tokens')} out={e.get('out_tokens')} "
                f"cache_r={e.get('cache_read')} {e.get('latency_ms')}ms stop={e.get('stop_reason')} "
                f"${e.get('cost_usd')}  [{bd_s}]"
            )
        elif t == "tool_call":
            lines.append(f"{prefix}  -> {e.get('name')}({_fmt_args(e.get('args') or {})})  id={e.get('id')}")
        elif t == "decision":
            mark = {"allow": "allow", "deny": "DENY", "approve": "ASK"}.get(e.get("verdict"), e.get("verdict"))
            extra = "" if e.get("verdict") == "allow" else f" - {str(e.get('reason', ''))[:120]}"
            lines.append(f"{prefix}     policy {mark} via {e.get('rule')}{extra}")
        elif t == "approval":
            lines.append(f"{prefix}     user {e.get('action')} {e.get('note', '')}")
        elif t == "tool_result":
            status = "ok" if e.get("ok") else "ERR"
            trunc = " truncated" if e.get("truncated") else ""
            preview = str(e.get("preview", ""))[:80]
            lines.append(f"{prefix}  <- {e.get('name')} {status} {e.get('bytes')}B {e.get('duration_ms')}ms{trunc}  {preview!r}")
        elif t == "compaction":
            lines.append(
                f"{prefix}compaction {e.get('before_tokens')} -> {e.get('after_tokens')} tokens, dropped {e.get('dropped_messages')} messages"
            )
        elif t == "note":
            lines.append(f"{prefix}note: {e.get('text')}")
        elif t == "run_end":
            lines.append(
                f"END outcome={e.get('outcome')} steps={e.get('steps')} cost=${e.get('cost_usd')} "
                f"wall={e.get('wall_s')}s tests_passed={e.get('tests_passed')} {e.get('stop_reason', '')}"
            )
    return "\n".join(lines)


def render_step(events: list[dict[str, Any]], step: int) -> str:
    chosen = [e for e in events if e.get("step") == step]
    if not chosen:
        return f"No events for step {step}."
    return "\n".join(json.dumps(e, indent=2, ensure_ascii=False, default=str) for e in chosen)


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    tool_calls = [e for e in events if e.get("t") == "tool_call"]
    results = [e for e in events if e.get("t") == "tool_result"]
    errors_by_tool: dict[str, int] = {}
    calls_by_tool: dict[str, int] = {}
    for e in tool_calls:
        calls_by_tool[e["name"]] = calls_by_tool.get(e["name"], 0) + 1
    for e in results:
        if not e.get("ok"):
            errors_by_tool[e["name"]] = errors_by_tool.get(e["name"], 0) + 1
    decisions = [e for e in events if e.get("t") == "decision"]
    verdicts: dict[str, int] = {}
    for e in decisions:
        verdicts[e["verdict"]] = verdicts.get(e["verdict"], 0) + 1
    model_calls = [e for e in events if e.get("t") == "model_call"]
    return {
        "model_calls": len(model_calls),
        "input_tokens": sum(e.get("in_tokens", 0) for e in model_calls),
        "output_tokens": sum(e.get("out_tokens", 0) for e in model_calls),
        "cost_usd": round(sum(e.get("cost_usd", 0.0) for e in model_calls), 6),
        "tool_calls_by_tool": calls_by_tool,
        "tool_errors_by_tool": errors_by_tool,
        "verdicts": verdicts,
    }
