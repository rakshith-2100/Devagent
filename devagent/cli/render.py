"""Terminal rendering: step output, diffs, the live plan, approval prompts."""
from __future__ import annotations

import json
from typing import Any, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from ..core.types import Decision, PlanStep, RunResult, ToolCall, ToolResult
from ..policy.approval import ApprovalRequest, ApprovalResponse
from ..tools.plan import STATUS_MARK

VERDICT_STYLE = {"allow": "green", "deny": "red", "approve": "yellow"}
OUTCOME_STYLE = {"completed": "green", "stopped_at_limit": "yellow", "denied": "red", "aborted": "red", "failed": "red bold"}


def _short(text: str, limit: int = 600) -> str:
    text = text.rstrip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [{len(text) - limit} more chars]"


class RichObserver:
    """Implements the Observer protocol with rich output."""

    def __init__(self, console: Optional[Console] = None, *, verbose: bool = False) -> None:
        self.console = console or Console()
        self.verbose = verbose

    def on_step(self, step: int, purpose: str) -> None:
        label = f"step {step}" if purpose == "agent" else f"{purpose} step {step}"
        self.console.rule(f"[dim]{label}[/dim]", style="dim")

    def on_assistant(self, step: int, text: str, tool_calls: list[dict[str, Any]]) -> None:
        if text.strip():
            self.console.print(Markdown(text))
        for tc in tool_calls:
            args = json.dumps(tc.get("args", {}), ensure_ascii=False, default=str)
            if len(args) > 160 and not self.verbose:
                args = args[:157] + "..."
            self.console.print(f"  [cyan]->[/cyan] [bold]{tc.get('name')}[/bold] {args}")

    def on_tool_call(self, step: int, call: ToolCall, decision: Decision) -> None:
        if decision.verdict != "allow" or self.verbose:
            style = VERDICT_STYLE.get(decision.verdict, "white")
            self.console.print(f"     [dim]policy[/dim] [{style}]{decision.verdict}[/{style}] [dim]({decision.rule})[/dim]")

    def on_tool_result(self, step: int, call: ToolCall, result: ToolResult) -> None:
        status = "[green]ok[/green]" if result.ok else "[red]error[/red]"
        trunc = " [yellow](truncated)[/yellow]" if result.truncated else ""
        ms = result.meta.get("duration_ms", 0)
        self.console.print(f"  [cyan]<-[/cyan] {call.name} {status}{trunc} [dim]{ms}ms[/dim]")
        if call.name == "edit_file" and result.ok and "\n" in result.content:
            diff = result.content.split("\n", 1)[1]
            self.console.print(Syntax(_short(diff, 3000), "diff", theme="ansi_dark", word_wrap=True))
        elif not result.ok or self.verbose or call.name in ("run_tests", "run_command"):
            self.console.print(Text(_short(result.content, 1200 if self.verbose else 600), style="dim"))

    def on_plan(self, steps: list[PlanStep]) -> None:
        self.console.print(render_plan_panel(steps))

    def on_compaction(self, info: dict[str, Any]) -> None:
        self.console.print(f"[yellow]context compacted[/yellow] ~{info.get('before_tokens')} -> ~{info.get('after_tokens')} tokens")

    def on_event(self, text: str) -> None:
        self.console.print(f"[dim]{text}[/dim]")

    def on_run_end(self, result: RunResult) -> None:
        pass  # the REPL prints the report


def render_plan_panel(steps: list[PlanStep]) -> Panel:
    lines = []
    for i, s in enumerate(steps, start=1):
        style = {"done": "green", "in_progress": "cyan bold", "blocked": "red", "pending": "white"}[s.status]
        note = f" [dim]- {escape(s.note)}[/dim]" if s.note else ""
        lines.append(f"[{style}]{escape(STATUS_MARK[s.status])} {i}. {escape(s.title)}[/{style}]{note}")
    return Panel("\n".join(lines) or "(empty)", title="plan", border_style="blue", expand=False)


def render_report(
    console: Console,
    result: RunResult,
    *,
    branch: str,
    root: str,
    files_changed: list[str],
    transcript: Optional[str],
    show_text: bool = False,
) -> None:
    style = OUTCOME_STYLE.get(result.outcome.value, "white")
    console.rule(f"[{style}]{result.outcome.value}[/{style}]")
    if show_text and result.final_text.strip():
        console.print(Markdown(result.final_text))
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_row("outcome", f"[{style}]{result.outcome.value}[/{style}]" + (f"  [dim]{result.stop_reason}[/dim]" if result.stop_reason else ""))
    table.add_row("steps", f"{result.steps}  tool calls {result.tool_calls}  tool errors {result.tool_errors}")
    table.add_row("tokens", f"in {result.usage.input_tokens}  out {result.usage.output_tokens}  cache read {result.usage.cache_read_tokens}")
    table.add_row("cost", f"${result.cost_usd:.4f} (estimate)  wall {result.wall_seconds:.0f}s")
    table.add_row("tests", {True: "[green]passed[/green]", False: "[red]failing[/red]", None: "[dim]not run[/dim]"}[result.tests_passed])
    table.add_row("files", ", ".join(files_changed) if files_changed else "[dim]none changed[/dim]")
    table.add_row("branch", f"{branch}  [dim]{root}[/dim]")
    if transcript:
        table.add_row("logs", f"[dim]{transcript}[/dim]")
    console.print(table)


# ------------------------------------------------------------ approvals


def interactive_approval(console: Console):
    """Return an ApprovalHandler that asks on the terminal with [y/n/e/a]."""

    def handler(req: ApprovalRequest) -> ApprovalResponse:
        d = req.decision
        header = Text()
        header.append("APPROVAL NEEDED  ", style="bold yellow")
        header.append(req.summary, style="bold")
        body = [
            f"[dim]rule:[/dim] {d.rule}  [dim]tier:[/dim] {d.tier}  [dim]reversible:[/dim] {'yes' if req.reversible else 'no'}",
            f"[dim]why the policy asks:[/dim] {d.reason}",
            f"[dim]agent's stated reason:[/dim] {req.agent_reason}",
        ]
        console.print(Panel("\n".join(body), title=header, border_style="yellow", expand=False))
        if req.details:
            lexer = "diff" if req.call.name == "edit_file" else ("json" if req.details.lstrip().startswith("{") else "text")
            console.print(Syntax(_short(req.details, 4000), lexer, theme="ansi_dark", word_wrap=True))
        while True:
            try:
                answer = console.input("[bold]accept[/bold] [y] / reject [n] / edit args [e] / abort run [a]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return ApprovalResponse("abort", note="input closed")
            if answer in ("y", "yes"):
                return ApprovalResponse("accept")
            if answer in ("n", "no"):
                note = console.input("reason for the agent (optional): ").strip()
                return ApprovalResponse("reject", note=note)
            if answer in ("a", "abort", "q"):
                return ApprovalResponse("abort", note="user aborted")
            if answer in ("e", "edit"):
                console.print("[dim]current args:[/dim]")
                console.print(Syntax(json.dumps(req.call.args, indent=2, ensure_ascii=False), "json", theme="ansi_dark"))
                raw = console.input("new args as JSON (empty to cancel): ").strip()
                if not raw:
                    continue
                try:
                    args = json.loads(raw)
                    if not isinstance(args, dict):
                        raise ValueError("must be an object")
                except ValueError as e:
                    console.print(f"[red]invalid JSON: {e}[/red]")
                    continue
                return ApprovalResponse("edit", args=args, note="edited by user")
            console.print("[dim]please answer y, n, e or a[/dim]")

    return handler


def interactive_plan_review(console: Console):
    def handler(payload: dict[str, Any]) -> dict[str, Any]:
        plan = payload.get("plan", {})
        lines = [f"[bold]{plan.get('summary', '')}[/bold]", ""]
        for i, s in enumerate(plan.get("steps", []), start=1):
            files = f" [dim]({', '.join(s.get('files', []))})[/dim]" if s.get("files") else ""
            lines.append(f"{i}. {s.get('title')}{files}")
            if s.get("detail"):
                lines.append(f"   [dim]{s['detail']}[/dim]")
        lines.append("")
        lines.append(f"[dim]verification:[/dim] {plan.get('verification', '')}")
        for r in plan.get("risks", []):
            lines.append(f"[yellow]risk:[/yellow] {r}")
        console.print(Panel("\n".join(lines), title="proposed plan", border_style="blue"))
        while True:
            try:
                answer = console.input("[bold]approve[/bold] [y] / edit steps [e] / reject [n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return {"action": "reject"}
            if answer in ("y", "yes"):
                return {"action": "accept"}
            if answer in ("n", "no"):
                return {"action": "reject"}
            if answer in ("e", "edit"):
                console.print("[dim]enter one step per line; finish with an empty line[/dim]")
                steps: list[str] = []
                while True:
                    try:
                        line = console.input(f"{len(steps) + 1}. ").strip()
                    except (EOFError, KeyboardInterrupt):
                        break
                    if not line:
                        break
                    steps.append(line)
                if steps:
                    return {"action": "edit", "steps": steps}
                continue
            console.print("[dim]please answer y, e or n[/dim]")

    return handler
