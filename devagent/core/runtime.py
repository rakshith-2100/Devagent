"""Per-run runtime objects that the graph nodes close over.

LangGraph state must be serialisable for checkpoints; sandboxes, tracers and
model clients are not. They live here, and ``build_agent_graph`` captures the
runtime in closures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage

from ..context.budget import BudgetTracker
from ..context.notes import NotesFile
from ..policy.engine import PolicyContext, PolicyEngine
from ..prompts import render_prompt
from ..sandbox.local import LocalSandbox
from ..tools.base import ToolContext, ToolRegistry
from ..trace.promptlog import PromptLogger
from ..trace.tracer import Tracer
from .config import Config
from .types import Decision, PlanStep, RunResult, ToolCall, ToolResult


class Observer(Protocol):
    """What the CLI renders. Every method is optional; see ``NullObserver``."""

    def on_step(self, step: int, purpose: str) -> None: ...
    def on_assistant(self, step: int, text: str, tool_calls: list[dict[str, Any]]) -> None: ...
    def on_tool_call(self, step: int, call: ToolCall, decision: Decision) -> None: ...
    def on_tool_result(self, step: int, call: ToolCall, result: ToolResult) -> None: ...
    def on_plan(self, steps: list[PlanStep]) -> None: ...
    def on_compaction(self, info: dict[str, Any]) -> None: ...
    def on_event(self, text: str) -> None: ...
    def on_run_end(self, result: RunResult) -> None: ...


class NullObserver:
    def on_step(self, step: int, purpose: str) -> None:
        pass

    def on_assistant(self, step: int, text: str, tool_calls: list[dict[str, Any]]) -> None:
        pass

    def on_tool_call(self, step: int, call: ToolCall, decision: Decision) -> None:
        pass

    def on_tool_result(self, step: int, call: ToolCall, result: ToolResult) -> None:
        pass

    def on_plan(self, steps: list[PlanStep]) -> None:
        pass

    def on_compaction(self, info: dict[str, Any]) -> None:
        pass

    def on_event(self, text: str) -> None:
        pass

    def on_run_end(self, result: RunResult) -> None:
        pass


DelegateFn = Callable[[str, list[str], int], str]


@dataclass
class AgentRuntime:
    run_id: str
    repo_root: Path
    root: Path  # the worktree (or the repo itself with --no-worktree)
    branch: str
    config: Config
    sandbox: LocalSandbox
    registry: ToolRegistry
    policy: PolicyEngine
    tracer: Tracer
    promptlog: PromptLogger
    notes: NotesFile
    model: BaseChatModel
    budget: BudgetTracker
    static_system: str
    repo_map: str
    test_hint: str = ""
    purpose: str = "agent"
    observer: Any = field(default_factory=NullObserver)
    parent_run_id: Optional[str] = None

    # persistence (set by create_runtime)
    checkpointer: Any = None  # LangGraph checkpointer; the CLI builds the graph with it
    checkpoint_path: Any = None
    session_meta: Any = None  # SessionMeta record under .devagent/sessions/
    project_memory: str = ""  # .devagent/memory.md, user-maintained, injected into the context

    # mutable per-run state that tools update
    plan: list[PlanStep] = field(default_factory=list)
    files_changed: set[str] = field(default_factory=set)
    tests_passed: Optional[bool] = None
    test_runs: int = 0
    consecutive_test_failures: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    delegate: Optional[DelegateFn] = None
    delegate_count: int = 0
    _bound_model: Any = field(default=None, repr=False)

    # ------------------------------------------------------------ helpers
    def dynamic_context(self) -> str:
        return render_prompt(
            "context",
            root=str(self.root),
            branch=self.branch,
            mode=self.config.mode,
            test_hint=self.test_hint,
            repo_map=self.repo_map,
            notes=self.notes.read(),
            project_memory=self.project_memory or "(none)",
        )

    def system_messages(self) -> list[SystemMessage]:
        """Cache-friendly order: static prompt first, volatile context second."""
        return [SystemMessage(content=self.static_system), SystemMessage(content=self.dynamic_context())]

    def bound_model(self) -> Any:
        if self._bound_model is None:
            schemas = self.registry.schemas()
            self._bound_model = self.model.bind_tools(schemas) if schemas else self.model
        return self._bound_model

    def tool_context(self) -> ToolContext:
        return ToolContext(sandbox=self.sandbox, config=self.config, runtime=self)

    def policy_context(self, goal: str = "") -> PolicyContext:
        return PolicyContext(root=self.root, known_tools=set(self.registry.names()), sandbox=self.sandbox, goal=goal)

    def cost_of(self, usage: Any) -> float:
        return self.config.cost_usd(usage.input_tokens, usage.output_tokens, usage.cache_read_tokens, usage.cache_write_tokens)
