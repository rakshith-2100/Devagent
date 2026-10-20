"""Task-shaped tools. The registry generates schemas from signatures and
docstrings so the model's documentation cannot drift from the implementation."""
from __future__ import annotations

from .base import Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, tool  # noqa: F401


def default_registry(*, enable_delegate: bool = False) -> ToolRegistry:
    from . import fs, git, notes, plan, search, shell, tests

    registry = ToolRegistry(
        [
            fs.read_file,
            fs.list_dir,
            search.grep_repo,
            search.find_files,
            fs.edit_file,
            fs.write_file,
            tests.run_tests,
            shell.run_command,
            git.git_diff,
            plan.update_plan,
            notes.append_notes,
        ]
    )
    if enable_delegate:
        from . import delegate

        registry.register(delegate.delegate)
    return registry
