"""Session setup: worktree, sandbox, tracer, prompt logs, repo map, runtime."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from ..context.budget import BudgetTracker
from ..context.notes import NotesFile
from ..context.repomap import build_repo_map
from ..policy.engine import PolicyEngine
from ..prompts import load_prompt
from ..sandbox.local import LocalSandbox, SandboxError, create_worktree, git_root, new_run_id, remove_worktree, restore_worktree
from .sessions import SessionMeta, close_checkpointer, open_checkpointer, read_project_memory, save_session
from ..tools import default_registry
from ..tools.tests import test_hint
from ..trace.promptlog import PromptLogger
from ..trace.tracer import Tracer
from .config import Config
from .model import build_chat_model, summarize_raw_response
from .runtime import AgentRuntime, NullObserver


def devagent_dir(repo_root: Path, config: Config) -> Path:
    return repo_root / config.devagent_dir


def create_runtime(
    repo_path: Path | str,
    config: Config,
    *,
    model: Optional[BaseChatModel] = None,
    observer: Any = None,
    run_id: Optional[str] = None,
    script: Optional[list[Any]] = None,
    resume: Optional[SessionMeta] = None,
) -> AgentRuntime:
    """Create the worktree and everything a run needs. Raises on infrastructure problems.

    With ``resume`` the runtime re-attaches to that session: same run id (so the
    checkpoint thread, trace and logs continue) and the same worktree.
    """
    start = Path(repo_path).resolve()
    if not start.is_dir():
        raise FileNotFoundError(f"{start} is not a directory")
    repo_root = git_root(start) or start
    if resume is not None:
        run_id = resume.run_id
    run_id = run_id or new_run_id()
    base = devagent_dir(repo_root, config)
    base.mkdir(parents=True, exist_ok=True)

    is_git = git_root(start) is not None
    if resume is not None and resume.branch.startswith("devagent/") and is_git:
        root, branch = Path(resume.root), resume.branch
        if not root.is_dir():
            root = restore_worktree(repo_root, root, branch)
    elif resume is not None:
        root, branch = repo_root, resume.branch
    elif config.use_worktree and is_git:
        root, branch = create_worktree(repo_root, run_id, devagent_dir=config.devagent_dir)
    else:
        root, branch = repo_root, "(working directly in the repository; no worktree)"

    sandbox = _make_sandbox(root, config, run_id)
    tracer = Tracer(base / "traces" / f"{run_id}.jsonl", run_id)
    promptlog = PromptLogger(base / "logs" / run_id, enabled=config.log_prompts)
    notes = NotesFile(root / "NOTES.md")

    if model is None:
        model = build_chat_model(
            config,
            request_hook=_request_hook(promptlog),
            response_hook=_response_hook(promptlog),
            script=script,
        )
    else:
        # attach hooks if the model supports them (scripted models in tests)
        for attr, hook in (("request_hook", _request_hook(promptlog)), ("response_hook", _response_hook(promptlog))):
            if hasattr(model, attr) and getattr(model, attr) is None:
                try:
                    setattr(model, attr, hook)
                except Exception:
                    pass

    files = sandbox.tracked_and_untracked_files()
    repo_map = build_repo_map(root, files, max_tokens=config.repomap_max_tokens, max_files=config.repomap_max_files)
    registry = default_registry(enable_delegate=config.enable_delegate)
    static_system = load_prompt("system")
    rt = AgentRuntime(
        run_id=run_id,
        repo_root=repo_root,
        root=root,
        branch=branch,
        config=config,
        sandbox=sandbox,
        registry=registry,
        policy=PolicyEngine(config.mode),
        tracer=tracer,
        promptlog=promptlog,
        notes=notes,
        model=model,
        budget=BudgetTracker(config.budget),
        static_system=static_system,
        repo_map=repo_map,
        test_hint=test_hint(root),
        observer=observer or NullObserver(),
    )
    rt.checkpointer, rt.checkpoint_path = open_checkpointer(repo_root, config)
    rt.project_memory = read_project_memory(repo_root, config)
    rt.session_meta = resume or SessionMeta.new(run_id=run_id, repo_root=repo_root, root=root, branch=branch, config=config)
    if resume is None:
        save_session(repo_root, config, rt.session_meta)
    promptlog.log_static(static_system, registry.schemas())
    return rt


def _make_sandbox(root: Path, config: Config, run_id: str) -> LocalSandbox:
    if config.sandbox == "docker":
        from ..sandbox.docker import DockerSandbox

        ok, detail = DockerSandbox.daemon_available()
        if not ok:
            raise SandboxError(
                f"DEVAGENT_SANDBOX=docker but the Docker engine is not reachable ({detail}). "
                "Start Docker Desktop, or run with --sandbox local."
            )
        sandbox = DockerSandbox(root, image=config.docker_image, memory=config.docker_memory, cpus=config.docker_cpus, run_id=run_id)
        if not sandbox.image_present():
            raise SandboxError(f"Sandbox image '{config.docker_image}' not found. Build it once with: devagent build-sandbox")
        return sandbox
    if config.sandbox != "local":
        raise SandboxError(f"Unknown DEVAGENT_SANDBOX '{config.sandbox}'. Use 'local' or 'docker'.")
    return LocalSandbox(root)


def _request_hook(promptlog: PromptLogger):
    def hook(purpose: str, body: dict[str, Any]) -> None:
        # Never mutate ``body``: it is the exact payload that goes on the wire.
        n = promptlog.next_call_no()
        promptlog.log_model_request(n, purpose, {k: v for k, v in body.items() if not k.startswith("_")})

    return hook


def _response_hook(promptlog: PromptLogger):
    def hook(purpose: str, raw: dict[str, Any], latency_ms: int) -> None:
        n = promptlog._call_no  # a response is logged right after its request, so this pairs them
        text, tool_calls, usage, stop_reason = summarize_raw_response(raw)
        promptlog.log_model_response(
            n,
            purpose,
            {k: v for k, v in raw.items() if not k.startswith("_")},
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            latency_ms=latency_ms,
            stop_reason=stop_reason,
        )

    return hook


def close_runtime(rt: AgentRuntime, *, remove_worktree_dir: bool = False) -> None:
    """By default the worktree and branch are kept for review; the session stays resumable."""
    close_checkpointer(rt.checkpointer)
    if remove_worktree_dir and rt.root != rt.repo_root:
        remove_worktree(rt.repo_root, rt.root, delete_branch=False)
