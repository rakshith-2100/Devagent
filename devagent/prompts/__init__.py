"""Prompt templates live as Markdown files next to this module.

Templates use ``${name}`` placeholders (``string.Template``) so that braces in
code samples never collide with substitution. ``load_prompt`` returns the raw
text; ``render_prompt`` substitutes variables and fails loudly on a missing one.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Template

PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def render_prompt(name: str, **variables: object) -> str:
    return Template(load_prompt(name)).substitute({k: str(v) for k, v in variables.items()})


def available_prompts() -> list[str]:
    return sorted(p.stem for p in PROMPTS_DIR.glob("*.md"))
