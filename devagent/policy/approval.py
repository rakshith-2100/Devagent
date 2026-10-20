"""The approval request contract.

An approval request says what will happen in plain terms, shows the full
arguments or the actual diff, gives the agent's stated reason, and says whether
it can be undone. The response offers accept, reject, and edit, because most
rejections are corrections.

Requests and responses round-trip through plain dicts so they can be carried in
a LangGraph ``interrupt()`` payload and a checkpoint.
"""
from __future__ import annotations

import difflib
import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

from ..core.types import Decision, ToolCall

ACTIONS = ("accept", "reject", "edit", "abort")


@dataclass
class ApprovalRequest:
    call: ToolCall
    decision: Decision
    summary: str
    details: str
    agent_reason: str
    reversible: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "approval",
            "call": {"id": self.call.id, "name": self.call.name, "args": self.call.args},
            "decision": asdict(self.decision),
            "summary": self.summary,
            "details": self.details,
            "agent_reason": self.agent_reason,
            "reversible": self.reversible,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ApprovalRequest":
        c = d["call"]
        dec = d["decision"]
        return cls(
            call=ToolCall(id=c["id"], name=c["name"], args=dict(c.get("args") or {})),
            decision=Decision(dec["verdict"], dec["rule"], dec["reason"], dec.get("tier", -1)),
            summary=d["summary"],
            details=d["details"],
            agent_reason=d.get("agent_reason", ""),
            reversible=bool(d.get("reversible", False)),
        )


@dataclass
class ApprovalResponse:
    action: str  # accept | reject | edit | abort
    args: Optional[dict[str, Any]] = None  # for edit
    note: str = ""

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"Unknown approval action '{self.action}'. Expected one of {ACTIONS}.")

    def as_dict(self) -> dict[str, Any]:
        return {"action": self.action, "args": self.args, "note": self.note}

    @classmethod
    def from_dict(cls, d: Any) -> "ApprovalResponse":
        if isinstance(d, ApprovalResponse):
            return d
        if isinstance(d, str):
            return cls(action=d)
        if isinstance(d, dict):
            return cls(action=str(d.get("action", "reject")), args=d.get("args"), note=str(d.get("note", "")))
        return cls(action="reject", note="unrecognised approval response")


ApprovalHandler = Callable[[ApprovalRequest], ApprovalResponse]


def auto_accept(_req: ApprovalRequest) -> ApprovalResponse:
    return ApprovalResponse("accept", note="auto-accepted")


def auto_reject(_req: ApprovalRequest) -> ApprovalResponse:
    return ApprovalResponse("reject", note="auto-rejected (non-interactive run)")


# ----------------------------------------------------------------- previews


def _preview_edit(call: ToolCall, sandbox: Any) -> tuple[str, str]:
    path = str(call.args.get("path", ""))
    old = str(call.args.get("old_string", ""))
    new = str(call.args.get("new_string", ""))
    replace_all = bool(call.args.get("replace_all", False))
    details = ""
    try:
        p = sandbox.resolve(path)
        text = p.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
        count = text.count(old)
        if count == 0:
            details = "(old_string not found in the current file; the edit would fail)\n"
        else:
            updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
            diff = difflib.unified_diff(
                text.splitlines(), updated.splitlines(), fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="", n=3
            )
            details = "\n".join(diff)
    except Exception as e:  # preview must never block the request
        details = f"(could not build diff preview: {e})\n--- old_string ---\n{old}\n--- new_string ---\n{new}"
    summary = f"Edit {path}" + (" (all occurrences)" if replace_all else "")
    return summary, details


def _preview_write(call: ToolCall) -> tuple[str, str]:
    path = str(call.args.get("path", ""))
    content = str(call.args.get("content", ""))
    lines = content.splitlines()
    head = "\n".join(lines[:60])
    if len(lines) > 60:
        head += f"\n... ({len(lines) - 60} more lines)"
    return f"Create {path} ({len(lines)} lines)", head


def build_request(call: ToolCall, decision: Decision, agent_reason: str, sandbox: Any = None) -> ApprovalRequest:
    name = call.name
    reversible = True
    if name == "edit_file":
        summary, details = _preview_edit(call, sandbox)
    elif name == "write_file":
        summary, details = _preview_write(call)
    elif name == "run_command":
        summary = f"Run command: {call.args.get('command', '')}"
        details = f"timeout: {call.args.get('timeout', 60)}s\nrule: {decision.rule}\n{decision.reason}"
        reversible = False
    elif name == "run_tests":
        summary = f"Run tests{(' for ' + str(call.args.get('target'))) if call.args.get('target') else ''}"
        details = decision.reason
    elif name == "delegate":
        summary = "Spawn a read-only subagent"
        details = json.dumps(call.args, indent=2)
    else:
        summary = f"{name}"
        details = json.dumps(call.args, indent=2, default=str)
    return ApprovalRequest(
        call=call,
        decision=decision,
        summary=summary,
        details=details,
        agent_reason=agent_reason.strip() or "(the agent gave no reason)",
        reversible=reversible,
    )
