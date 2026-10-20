from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from devagent.core.config import Config
from devagent.evals.runner import load_cases, run_evals

from .test_graph import HAPPY_SCRIPT


def test_cases_load() -> None:
    cases = load_cases()
    ids = {c.id for c in cases}
    assert {"fix-divide-by-zero-001", "fix-slugify-002", "injection-readme-003"} <= ids
    for c in cases:
        assert c.fixture.is_dir(), c.fixture
        assert c.expect.get("tests_pass") is True


def test_runner_passes_with_scripted_fix(tmp_path: Path, config: Config) -> None:
    console = Console(record=True, width=120)
    rc = run_evals(
        config,
        console=console,
        case_ids=["fix-divide-by-zero-001"],
        runs_override=2,
        script=HAPPY_SCRIPT,
        output_root=tmp_path,
        keep_workdir=False,
    )
    out = console.export_text()
    assert rc == 0, out
    assert "pass" in out
    reports = list((tmp_path / ".devagent" / "evals").glob("*.json"))
    assert len(reports) == 1
    data = json.loads(reports[0].read_text(encoding="utf-8"))
    case = data["summary"]["fix-divide-by-zero-001"]
    assert case["runs"] == 2 and case["pass_k"] is True
    grades = data["runs"][0]["grades"]
    assert all(g["ok"] for g in grades.values()), grades
    assert set(grades) == {"tests_pass", "files_changed", "files_unchanged", "forbidden_tools", "max_steps", "max_cost_usd", "outcome"}


def test_runner_fails_when_agent_edits_tests(tmp_path: Path, config: Config) -> None:
    cheat = [
        {"tool_calls": [{"name": "edit_file", "args": {"path": "tests/test_calc.py", "old_string": "    with pytest.raises(ValueError):\n        divide(1, 0)\n", "new_string": "    assert True\n"}}]},
        {"text": "done"},
    ]
    console = Console(record=True, width=120)
    # auto-reject means the test edit is refused by the approval gate; the suite stays red -> case fails
    rc = run_evals(config, console=console, case_ids=["fix-divide-by-zero-001"], runs_override=1, script=cheat, output_root=tmp_path, keep_workdir=False)
    assert rc == 1
    out = console.export_text()
    assert "tests_pass" in out
