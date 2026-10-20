"""devagent entry point: REPL, one-shot runs, trace viewer, evals, doctor."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.table import Table

from ..core.config import GLOBAL_ENV_TEMPLATE, Config, global_env_path, load_settings
from ..core.graph import build_agent_graph, make_delegate, run_agent
from ..core.model import ModelError, load_script
from ..core.planner import run_plan_mode
from ..core.session import close_runtime, create_runtime, devagent_dir
from ..core.sessions import SessionMeta, find_session, list_sessions, record_turn, restore_from_checkpoint
from ..core.types import RunOutcome
from ..sandbox.local import git_root
from ..trace.tracer import Tracer
from ..trace.view import find_run, list_runs, render_step, render_timeline, summarize
from .render import RichObserver, interactive_approval, interactive_plan_review, render_plan_panel, render_report

HELP = """commands:
  <text>          run the agent on a task
  /plan           show the agent's current plan
  /trace          print this session's trace timeline
  /diff           show the worktree diff
  /notes          print NOTES.md
  /budget         show the limits in force
  /mode <m>       switch run mode: default | careful | yolo
  /sessions       list this repository's sessions (newest first)
  /resume [n|id]  switch to a previous session (its worktree and full conversation)
  /help           this help
  /quit           exit (the worktree, branch and session are kept; resume with devagent --resume)
"""


COMMANDS = ("run", "trace", "eval", "repomap", "doctor", "logs", "init", "build-sandbox")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--env", default=".env", help="path to a .env file (default: ./.env)")

    p = argparse.ArgumentParser(
        prog="devagent",
        description="A coding agent that works in an isolated git worktree. "
        "With no subcommand, 'run' is assumed: devagent [path] [--goal ...] [--plan] [--yolo|--careful]",
    )
    sub = p.add_subparsers(dest="command")

    r = sub.add_parser("run", parents=[common], help="start the REPL, or run one task with --goal (default command)")
    r.add_argument("path", nargs="?", default=".", help="repository path (default: current directory)")
    r.add_argument("--goal", "-g", help="run one task non-interactively and exit")
    r.add_argument("--plan", action="store_true", help="plan mode: propose a plan for approval, then execute it step by step")
    r.add_argument("--resume", nargs="?", const="pick", metavar="SESSION", help="resume a previous session of this repository: a run id, a number from the list, or 'last'; with no value, choose from a list")
    r.add_argument("--continue", "-c", dest="continue_last", action="store_true", help="resume the most recent session of this repository")
    mode = r.add_mutually_exclusive_group()
    mode.add_argument("--careful", action="store_true", help="ask before every write and every command")
    mode.add_argument("--yolo", action="store_true", help="auto-approve allowlisted commands (never tier 3)")
    r.add_argument("--provider", choices=["bedrock", "openrouter", "mock"], help="model provider (overrides DEVAGENT_MODEL_PROVIDER)")
    r.add_argument("--model", help="model id for the active provider (overrides BEDROCK_MODEL_ID / OPENROUTER_MODEL)")
    r.add_argument("--region", help="AWS region (overrides AWS_REGION)")
    r.add_argument("--max-steps", type=int)
    r.add_argument("--max-cost", type=float, help="USD")
    r.add_argument("--max-wall", type=int, help="seconds")
    r.add_argument("--no-worktree", action="store_true", help="work directly in the repository (no isolation)")
    r.add_argument("--sandbox", choices=["local", "docker"], help="where commands run (overrides DEVAGENT_SANDBOX)")
    r.add_argument("--delegate", action="store_true", help="enable the delegate tool (read-only subagents)")
    r.add_argument("--mock", metavar="SCRIPT.json", help="use a scripted model from a JSON file (offline dry run)")
    r.add_argument("--verbose", "-v", action="store_true")

    t = sub.add_parser("trace", parents=[common], help="view traces")
    t.add_argument("path", nargs="?", default=".")
    t.add_argument("--last", action="store_true", help="the most recent run (default)")
    t.add_argument("--run", help="a run id (or unique suffix)")
    t.add_argument("--step", type=int, help="expand one step")
    t.add_argument("--list", action="store_true", help="list runs")
    t.add_argument("--summary", action="store_true", help="aggregate numbers for the run")

    e = sub.add_parser("eval", parents=[common], help="run the eval harness")
    e.add_argument("path", nargs="?", default=".", help="repository (used for .devagent output)")
    e.add_argument("--case", action="append", help="case id (repeatable)")
    e.add_argument("--tag", action="append", help="only cases with this tag (repeatable)")
    e.add_argument("--runs", type=int, help="override runs per case")
    e.add_argument("--judge", action="store_true", help="also run the model judge")
    e.add_argument("--mock", metavar="SCRIPT.json", help="scripted model for harness dry runs")
    e.add_argument("--cases-dir", help="directory of YAML cases (default: built-in)")

    m = sub.add_parser("repomap", parents=[common], help="print the repository map the agent sees")
    m.add_argument("path", nargs="?", default=".")

    d = sub.add_parser("doctor", parents=[common], help="check configuration and environment")
    d.add_argument("path", nargs="?", default=".")

    lg = sub.add_parser("logs", parents=[common], help="print the path of the latest prompt/output log")
    lg.add_argument("path", nargs="?", default=".")

    sub.add_parser("init", parents=[common], help="create the user-level settings file ~/.devagent/.env")

    b = sub.add_parser("build-sandbox", parents=[common], help="build the Docker sandbox image (needs Docker running)")
    b.add_argument("--image", help="image tag (default: DEVAGENT_DOCKER_IMAGE or devagent-sandbox:latest)")
    return p


def normalize_argv(argv: list[str]) -> list[str]:
    """``devagent some/path --goal x`` means ``devagent run some/path --goal x``."""
    for a in argv:
        if a in ("-h", "--help"):
            return argv
        if a in COMMANDS:
            return argv
        if not a.startswith("-"):
            break
    return ["run", *argv]


def config_from_args(args: argparse.Namespace) -> Config:
    load_settings(getattr(args, "env", ".env"))  # ./.env first, then ~/.devagent/.env
    overrides: dict[str, Any] = {}
    if getattr(args, "sandbox", None):
        overrides["sandbox"] = args.sandbox
    if getattr(args, "provider", None):
        overrides["provider"] = args.provider
    if getattr(args, "model", None):
        overrides["model_id"] = args.model
        overrides["openrouter_model"] = args.model
    if getattr(args, "region", None):
        overrides["aws_region"] = args.region
    if getattr(args, "careful", False):
        overrides["mode"] = "careful"
    if getattr(args, "yolo", False):
        overrides["mode"] = "yolo"
    if getattr(args, "no_worktree", False):
        overrides["use_worktree"] = False
    if getattr(args, "delegate", False):
        overrides["enable_delegate"] = True
    if getattr(args, "mock", None):
        overrides["provider"] = "mock"
    cfg = Config.from_env(**overrides)
    if getattr(args, "max_steps", None):
        cfg.budget.max_steps = args.max_steps
    if getattr(args, "max_cost", None):
        cfg.budget.max_cost_usd = args.max_cost
    if getattr(args, "max_wall", None):
        cfg.budget.max_wall_seconds = args.max_wall
    return cfg


# ---------------------------------------------------------------- commands


def cmd_trace(args: argparse.Namespace, console: Console) -> int:
    cfg = config_from_args(args)
    repo = git_root(Path(args.path)) or Path(args.path).resolve()
    traces = devagent_dir(repo, cfg) / "traces"
    if args.list:
        table = Table(title=f"runs in {traces}")
        for col in ("run id", "outcome", "steps", "cost", "wall", "goal"):
            table.add_column(col)
        for r in list_runs(traces):
            table.add_row(r["run_id"], str(r["outcome"]), str(r["steps"]), str(r["cost_usd"]), str(r["wall_s"]), (r["goal"] or "")[:60])
        console.print(table)
        return 0
    path = find_run(traces, args.run)
    if path is None:
        console.print(f"[red]no traces found in {traces}[/red]")
        return 1
    events = Tracer.read(path)
    if args.step is not None:
        console.print(render_step(events, args.step))
    elif args.summary:
        console.print_json(json.dumps(summarize(events)))
    else:
        console.print(f"[dim]{path}[/dim]")
        console.print(render_timeline(events))
    return 0


def cmd_repomap(args: argparse.Namespace, console: Console) -> int:
    from ..context.repomap import build_repo_map
    from ..sandbox.local import LocalSandbox

    cfg = config_from_args(args)
    root = Path(args.path).resolve()
    sb = LocalSandbox(root)
    console.print(build_repo_map(root, sb.tracked_and_untracked_files(), max_tokens=cfg.repomap_max_tokens, max_files=cfg.repomap_max_files))
    return 0


def cmd_logs(args: argparse.Namespace, console: Console) -> int:
    cfg = config_from_args(args)
    repo = git_root(Path(args.path)) or Path(args.path).resolve()
    logs = devagent_dir(repo, cfg) / "logs"
    runs = sorted((p for p in logs.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime) if logs.is_dir() else []
    if not runs:
        console.print(f"[red]no logs in {logs}[/red]")
        return 1
    console.print(str(runs[-1] / "transcript.md"))
    return 0


def cmd_doctor(args: argparse.Namespace, console: Console) -> int:
    cfg = config_from_args(args)
    repo = git_root(Path(args.path))
    rows = [
        ("python", sys.version.split()[0]),
        ("repository", str(repo) if repo else f"{Path(args.path).resolve()} (not a git repo: worktree isolation unavailable)"),
        ("provider", cfg.provider),
        ("model", cfg.effective_model()),
        ("region", cfg.aws_region),
        ("credentials", "bearer token set" if cfg.bearer_token else ("AWS access keys set" if cfg.aws_access_key_id else "NONE (set BEDROCK_API_KEY in .env)")),
        ("mode", cfg.mode),
        ("budget", f"{cfg.budget.max_steps} steps, {cfg.budget.max_tokens} tokens, ${cfg.budget.max_cost_usd}, {cfg.budget.max_wall_seconds}s"),
        ("prompt cache", "on" if cfg.prompt_cache else "off"),
        ("logs", "on" if cfg.log_prompts else "off"),
        ("delegate", "on" if cfg.enable_delegate else "off"),
    ]
    try:
        import langgraph.version as lgv

        rows.append(("langgraph", lgv.__version__))
    except Exception:  # pragma: no cover
        rows.append(("langgraph", "missing"))
    import shutil

    rows.append(("ripgrep", shutil.which("rg") or "not found (python fallback for grep_repo)"))
    rows.append(("git", shutil.which("git") or "NOT FOUND"))
    gpath = global_env_path()
    rows.append(("settings", f"./.env{' (found)' if Path(args.env).is_file() else ' (none)'}; {gpath}{' (found)' if gpath.is_file() else ' (none; run devagent init)'}"))
    from ..sandbox.docker import DockerSandbox

    ok, detail = DockerSandbox.daemon_available()
    if ok:
        image_ok = DockerSandbox(Path(args.path).resolve(), image=cfg.docker_image).image_present() if Path(args.path).is_dir() else False
        rows.append(("docker", f"engine {detail}; image {cfg.docker_image} {'present' if image_ok else 'missing (run devagent build-sandbox)'}"))
    else:
        rows.append(("docker", f"not available ({detail})"))
    rows.append(("sandbox", cfg.sandbox))
    table = Table(show_header=False, box=None)
    for k, v in rows:
        table.add_row(f"[bold]{k}[/bold]", v)
    console.print(table)
    return 0


def cmd_init(args: argparse.Namespace, console: Console) -> int:
    path = global_env_path()
    if path.is_file():
        console.print(f"settings file already exists: {path}")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(GLOBAL_ENV_TEMPLATE, encoding="utf-8")
    console.print(f"created {path}\nAdd your API key there; a .env in a repository overrides it.")
    return 0


def cmd_build_sandbox(args: argparse.Namespace, console: Console) -> int:
    from ..sandbox.docker import DockerSandbox
    from ..sandbox.local import SandboxError

    cfg = config_from_args(args)
    image = args.image or cfg.docker_image
    ok, detail = DockerSandbox.daemon_available()
    if not ok:
        console.print(f"[red]Docker engine not reachable ({detail}). Start Docker Desktop and retry.[/red]")
        return 2
    try:
        DockerSandbox.build_image(image, log=lambda line: console.print(f"[dim]{line}[/dim]"))
    except SandboxError as e:
        console.print(f"[red]{e}[/red]")
        return 1
    console.print(f"[green]sandbox image ready:[/green] {image}\nUse it with: devagent --sandbox docker   (or DEVAGENT_SANDBOX=docker)")
    return 0


def cmd_eval(args: argparse.Namespace, console: Console) -> int:
    from ..evals.runner import run_evals

    cfg = config_from_args(args)
    script = load_script(args.mock) if args.mock else None
    return run_evals(
        cfg,
        console=console,
        case_ids=args.case,
        tags=args.tag,
        runs_override=args.runs,
        use_judge=args.judge,
        script=script,
        cases_dir=Path(args.cases_dir) if args.cases_dir else None,
        output_root=Path(args.path).resolve(),
    )


# ------------------------------------------------------------------ REPL


def render_sessions_table(repo_root: Path, cfg: Config, *, current: Optional[str] = None) -> Table:
    table = Table(title=f"sessions in {repo_root}")
    for col in ("#", "session", "updated", "turns", "last outcome", "title"):
        table.add_column(col)
    for i, s in enumerate(list_sessions(repo_root, cfg), start=1):
        mark = " *" if s.run_id == current else ""
        table.add_row(str(i), s.run_id + mark, s.updated.replace("T", " "), str(s.turns), s.last_outcome or "-", (s.title or "(no turns yet)")[:60])
    return table


def pick_session(console: Console, repo_root: Path, cfg: Config) -> Optional[SessionMeta]:
    sessions = list_sessions(repo_root, cfg)
    if not sessions:
        return None
    console.print(render_sessions_table(repo_root, cfg))
    try:
        answer = console.input("session number (enter to cancel): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    return find_session(repo_root, cfg, answer) if answer else None


def _announce_resume(console: Console, rt: Any, graph: Any, meta: SessionMeta) -> None:
    n = restore_from_checkpoint(rt, graph)
    console.print(
        f"[green]resumed[/green] {rt.run_id}: {n} message(s), {meta.turns} turn(s), "
        f"last outcome {meta.last_outcome or '-'}; worktree {rt.root}"
    )
    if meta.goals:
        console.print(f"  [dim]last task: {meta.goals[-1][:100]}[/dim]")
    if rt.plan:
        console.print(render_plan_panel(rt.plan))


def _print_banner(console: Console, rt: Any, cfg: Config) -> None:
    console.print(f"[bold]devagent[/bold] run {rt.run_id}")
    console.print(f"  repo    {rt.repo_root}")
    console.print(f"  root    {rt.root}")
    console.print(f"  branch  {rt.branch}")
    where = cfg.aws_region if cfg.provider == "bedrock" else cfg.openrouter_base_url if cfg.provider == "openrouter" else "offline"
    console.print(f"  model   {cfg.effective_model()} ({cfg.provider}, {where})  mode {cfg.mode}")
    console.print(f"  logs    {rt.promptlog.dir if cfg.log_prompts else 'disabled'}")
    console.print("[dim]type a task, or /help[/dim]")


def _run_goal(rt: Any, cfg: Config, goal: str, console: Console, *, plan_mode: bool, graph: Any = None) -> RunOutcome:
    approve = interactive_approval(console)
    if plan_mode:
        result = run_plan_mode(rt, goal, approval_handler=approve, plan_review_handler=interactive_plan_review(console))
    else:
        # The same graph (and checkpoint thread) for the whole session, so a follow-up such as
        # "continue" after a failed or stopped run sees everything that happened before.
        result = run_agent(rt, goal, approval_handler=approve, graph=graph)
    if getattr(rt, "session_meta", None) is not None:
        record_turn(rt.repo_root, cfg, rt.session_meta, goal, result.outcome.value)
    transcript = str(rt.promptlog.transcript_path) if rt.promptlog.transcript_path else None
    render_report(
        console,
        result,
        branch=rt.branch,
        root=str(rt.root),
        files_changed=sorted(rt.files_changed),
        transcript=transcript,
        show_text=plan_mode,  # single runs already streamed the final message as the last assistant turn
    )
    return result.outcome


def cmd_main(args: argparse.Namespace, console: Console) -> int:
    cfg = config_from_args(args)
    script = load_script(args.mock) if args.mock else None
    observer = RichObserver(console, verbose=args.verbose)
    resume_meta: Optional[SessionMeta] = None
    if getattr(args, "continue_last", False) or getattr(args, "resume", None):
        repo_root = git_root(Path(args.path)) or Path(args.path).resolve()
        key = "last" if args.continue_last else str(args.resume)
        resume_meta = pick_session(console, repo_root, cfg) if key == "pick" else find_session(repo_root, cfg, key)
        if resume_meta is None:
            what = "no sessions yet" if key in ("pick", "last") else f"no session matching {key!r}"
            console.print(f"[red]{what} in {repo_root}[/red]")
            return 1
    try:
        rt = create_runtime(args.path, cfg, observer=observer, script=script, resume=resume_meta)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]could not start: {e}[/red]")
        return 2
    if cfg.enable_delegate:
        rt.delegate = make_delegate(rt)
    _print_banner(console, rt, cfg)
    graph = build_agent_graph(rt)
    if resume_meta is not None:
        _announce_resume(console, rt, graph, resume_meta)

    if cfg.provider == "bedrock" and not (cfg.bearer_token or cfg.aws_access_key_id):
        console.print("[yellow]no Bedrock credentials found; model calls will fail. Set BEDROCK_API_KEY in .env (see .env.example) or use --mock.[/yellow]")

    if args.goal:
        outcome = _run_goal(rt, cfg, args.goal, console, plan_mode=args.plan, graph=graph)
        return 0 if outcome == RunOutcome.COMPLETED else 1

    plan_mode = args.plan
    while True:
        try:
            line = console.input("[bold cyan]devagent>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, rest = line[1:].partition(" ")
            cmd = cmd.lower()
            if cmd in ("quit", "exit", "q"):
                break
            if cmd == "help":
                console.print(HELP)
            elif cmd == "plan":
                if rest.strip() in ("on", "off"):
                    plan_mode = rest.strip() == "on"
                    console.print(f"plan mode {'on' if plan_mode else 'off'}")
                else:
                    console.print(render_plan_panel(rt.plan))
            elif cmd == "sessions":
                console.print(render_sessions_table(rt.repo_root, cfg, current=rt.run_id))
            elif cmd == "resume":
                meta = find_session(rt.repo_root, cfg, rest.strip()) if rest.strip() else pick_session(console, rt.repo_root, cfg)
                if meta is None:
                    console.print("[red]no such session; /sessions lists them[/red]")
                elif meta.run_id == rt.run_id:
                    console.print("already in that session")
                else:
                    close_runtime(rt)
                    try:
                        rt = create_runtime(args.path, cfg, observer=observer, script=script, resume=meta)
                    except Exception as e:  # noqa: BLE001
                        console.print(f"[red]could not resume: {e}[/red]")
                        return 2
                    if cfg.enable_delegate:
                        rt.delegate = make_delegate(rt)
                    graph = build_agent_graph(rt)
                    _announce_resume(console, rt, graph, meta)
            elif cmd == "trace":
                console.print(render_timeline(Tracer.read(rt.tracer.path)))
            elif cmd == "diff":
                res = rt.sandbox.run_git("diff", "--stat")
                console.print(res.stdout or "(no changes)")
                res = rt.sandbox.run_git("status", "--short")
                console.print(res.stdout or "")
            elif cmd == "notes":
                console.print(rt.notes.read())
            elif cmd == "budget":
                console.print_json(json.dumps(rt.budget.as_dict()))
            elif cmd == "mode":
                from ..policy.engine import PolicyEngine

                try:
                    rt.policy = PolicyEngine(rest.strip())
                    cfg.mode = rest.strip()
                    console.print(f"mode set to {cfg.mode}")
                except ValueError as e:
                    console.print(f"[red]{e}[/red]")
            else:
                console.print(f"[red]unknown command /{cmd}[/red]\n{HELP}")
            continue
        try:
            _run_goal(rt, cfg, line, console, plan_mode=plan_mode, graph=graph)
        except ModelError as e:
            console.print(f"[red]{e}[/red]")
    console.print(
        f"[dim]branch {rt.branch} kept at {rt.root}. Review with: git diff main...{rt.branch}\n"
        f"Resume this session with: devagent --resume {rt.run_id}   (or devagent -c for the latest)[/dim]"
    )
    close_runtime(rt)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    console = Console()
    parser = build_parser()
    args = parser.parse_args(normalize_argv(list(sys.argv[1:] if argv is None else argv)))
    if os.name == "nt":
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
    if args.command == "trace":
        return cmd_trace(args, console)
    if args.command == "eval":
        return cmd_eval(args, console)
    if args.command == "repomap":
        return cmd_repomap(args, console)
    if args.command == "doctor":
        return cmd_doctor(args, console)
    if args.command == "logs":
        return cmd_logs(args, console)
    if args.command == "init":
        return cmd_init(args, console)
    if args.command == "build-sandbox":
        return cmd_build_sandbox(args, console)
    return cmd_main(args, console)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
