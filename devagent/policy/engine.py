"""ToolCall -> Decision. One pure function over the tables in ``rules``.

The engine never executes anything. It reads the call, consults the tables,
and names the rule that decided so the trace can record it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..core.types import Decision, ToolCall
from . import rules


@dataclass
class PolicyContext:
    root: Path
    known_tools: set[str]
    sandbox: Any = None  # LocalSandbox, used only for path resolution
    goal: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class PolicyEngine:
    def __init__(self, mode: str = "default") -> None:
        if mode not in rules.VALID_MODES:
            raise ValueError(f"Unknown mode '{mode}'. Valid modes: {', '.join(rules.VALID_MODES)}")
        self.mode = mode

    # ------------------------------------------------------------------ api
    def decide(self, call: ToolCall, ctx: PolicyContext) -> Decision:
        name = call.name
        if name not in ctx.known_tools or name not in rules.TIERS:
            return Decision(
                "deny",
                "deny_unknown_tool",
                f"Unknown tool '{name}'. Available tools: {', '.join(sorted(ctx.known_tools))}.",
            )
        tier = rules.TIERS[name]

        # Path confinement: resolve before checking. Applies to every tool with a path argument.
        rel = self._relpath_or_none(call, ctx)
        if rel is None and "path" in call.args and call.args.get("path") not in (None, "", "."):
            return Decision(
                "deny",
                "deny_path_escape",
                f"Path '{call.args.get('path')}' is outside the working root. Only paths inside the root are allowed.",
                tier=3,
            )

        if name in ("edit_file", "write_file") and rel is not None and rules.is_test_path(rel):
            return Decision(
                "approve",
                "approve_test_file_edit",
                f"'{rel}' is a test file. Editing tests to make them pass is the failure mode this guard exists for; "
                "the user must approve test changes explicitly.",
                tier=3,
            )

        if name == "run_command":
            return self._decide_command(call, tier)

        if name == "delegate":
            requested = call.args.get("tools") or []
            if isinstance(requested, list):
                bad = sorted(set(map(str, requested)) - set(rules.READ_ONLY_TOOLS))
                if bad:
                    return Decision(
                        "deny",
                        "deny_delegate_scope",
                        f"delegate may only receive read-only tools; not allowed: {', '.join(bad)}. "
                        f"Allowed: {', '.join(sorted(rules.READ_ONLY_TOOLS))}.",
                        tier=0,
                    )

        verdict = rules.MODE_TABLE[self.mode][tier]
        rule = f"tier{tier}_{self.mode}"
        reason = {
            0: "Read-only tool.",
            1: "Write inside the worktree; reversible via git.",
            2: "Execution inside the sandbox with a clean environment and timeout.",
            3: "Irreversible or external action.",
        }[tier]
        return Decision(verdict, rule, reason, tier=tier)

    # -------------------------------------------------------------- helpers
    def _relpath_or_none(self, call: ToolCall, ctx: PolicyContext) -> Optional[str]:
        raw = call.args.get("path")
        if raw in (None, "", "."):
            return "." if raw is not None else None
        if ctx.sandbox is None:
            return str(raw)
        try:
            resolved = ctx.sandbox.resolve(str(raw))
        except Exception:
            return None
        return ctx.sandbox.relpath(resolved)

    def _decide_command(self, call: ToolCall, tier: int) -> Decision:
        command = call.args.get("command", "")
        try:
            argv = rules.parse_command(str(command))
        except ValueError as e:
            return Decision("deny", "deny_shell_syntax", str(e), tier=tier)
        cmd_tier, rule, reason = rules.classify_command(argv)
        if cmd_tier < 0:
            return Decision("deny", rule, reason, tier=tier)
        if cmd_tier >= 3:
            return Decision("approve", rule, reason, tier=3)
        if self.mode == "careful" or rules.RUN_COMMAND_ASKS_IN_MODE.get(self.mode, True):
            return Decision("approve", f"approve_run_command_{self.mode}", reason, tier=2)
        return Decision("allow", rule, reason, tier=2)
