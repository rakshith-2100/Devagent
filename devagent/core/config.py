"""Configuration: environment variables, an optional .env file, and CLI flags.

Nothing here talks to the network. ``Config.from_env`` is the single place where
environment lookups happen so that the rest of the code is testable with a
plain dataclass.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

from .types import Budget

Mode = str  # "default" | "careful" | "yolo"

DEFAULT_MODEL_ID = "anthropic.claude-opus-5-5"
QUOTE_CHARS = ('"', "'")
SECRET_FIELDS = {"bearer_token", "aws_access_key_id", "aws_secret_access_key", "aws_session_token", "openrouter_api_key"}
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def load_dotenv_file(path: Path | str = ".env", *, override: bool = False) -> dict[str, str]:
    """Tiny .env loader (KEY=VALUE lines, comment lines, optional quotes). No dependency."""
    p = Path(path)
    loaded: dict[str, str] = {}
    if not p.is_file():
        return loaded
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in QUOTE_CHARS:
            value = value[1:-1]
        if key and (override or key not in os.environ):
            os.environ[key] = value
            loaded[key] = value
    return loaded


def global_env_path() -> Path:
    """User-level settings, read after the project's .env: ~/.devagent/.env"""
    return Path.home() / ".devagent" / ".env"


GLOBAL_ENV_TEMPLATE = """# devagent user-level settings. A .env in the current repository takes precedence.
# Provider: bedrock | openrouter | mock
DEVAGENT_MODEL_PROVIDER=openrouter

# OpenRouter (OpenAI-compatible)
OPENROUTER_API_KEY=
OPENROUTER_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free

# Amazon Bedrock (either a Bedrock API key or IAM access keys)
BEDROCK_API_KEY=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-opus-5-5

# Sandbox: local | docker  (docker needs: devagent build-sandbox)
DEVAGENT_SANDBOX=local

# Budgets
DEVAGENT_MAX_STEPS=25
DEVAGENT_MAX_COST_USD=1.00
DEVAGENT_MAX_WALL_SECONDS=600
"""


def load_settings(project_env: Path | str = ".env") -> list[Path]:
    """Load settings in precedence order: process env > ./.env > ~/.devagent/.env.

    Returns the files that were actually found.
    """
    found: list[Path] = []
    for path in (Path(project_env), global_env_path()):
        if path.is_file():
            load_dotenv_file(path)  # never overrides what is already set
            found.append(path)
    return found


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v not in (None, "") else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v not in (None, "") else default
    except ValueError:
        return default


def _env_opt_float(name: str) -> Optional[float]:
    v = os.getenv(name)
    if v in (None, ""):
        return None
    try:
        return float(v)
    except ValueError:
        return None


@dataclass
class Config:
    # --- model / provider -------------------------------------------------
    provider: str = "bedrock"  # bedrock | mock
    aws_region: str = "us-east-1"
    model_id: str = DEFAULT_MODEL_ID
    bearer_token: Optional[str] = None  # BEDROCK_API_KEY or AWS_BEARER_TOKEN_BEDROCK
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    max_output_tokens: int = 8192
    temperature: Optional[float] = None  # omitted unless set; current Claude models reject it
    request_timeout_s: float = 180.0
    max_retries: int = 5  # retries per model call on 429/5xx/overloaded; backoff 2..60 s
    prompt_cache: bool = False  # insert Converse cachePoint blocks
    extra_request_fields: dict[str, Any] = field(default_factory=dict)  # additionalModelRequestFields

    # --- pricing (USD per million tokens) used only for the cost budget -----
    price_in_per_m: float = 4.0
    price_out_per_m: float = 20.0
    price_cache_read_per_m: float = 0.4
    price_cache_write_per_m: float = 5.0

    # --- context ------------------------------------------------------------
    context_window_tokens: int = 200_000
    compaction_ratio: float = 0.70
    keep_recent_messages: int = 8
    repomap_max_tokens: int = 3000
    repomap_max_files: int = 400

    # --- behaviour ----------------------------------------------------------
    mode: Mode = "default"
    budget: Budget = field(default_factory=Budget)
    enable_delegate: bool = False
    use_worktree: bool = True
    devagent_dir: str = ".devagent"
    log_prompts: bool = True
    max_test_attempts: int = 3

    # --- sandbox ------------------------------------------------------------
    sandbox: str = "local"  # local | docker
    docker_image: str = "devagent-sandbox:latest"
    docker_memory: str = "2g"
    docker_cpus: str = "2"

    # --- OpenAI-compatible providers (OpenRouter) ---------------------------
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL

    def effective_model(self) -> str:
        """The model id the active provider will actually call."""
        if self.provider == "openrouter":
            return self.openrouter_model
        if self.provider == "mock":
            return "scripted"
        return self.model_id

    @classmethod
    def from_env(cls, **overrides: Any) -> "Config":
        cfg = cls()
        cfg.provider = os.getenv("DEVAGENT_MODEL_PROVIDER", cfg.provider).strip().lower()
        cfg.aws_region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or cfg.aws_region
        cfg.model_id = os.getenv("BEDROCK_MODEL_ID", cfg.model_id)
        cfg.openrouter_api_key = os.getenv("OPENROUTER_API_KEY") or None
        cfg.openrouter_model = os.getenv("OPENROUTER_MODEL", cfg.openrouter_model)
        cfg.openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", cfg.openrouter_base_url)
        cfg.bearer_token = os.getenv("BEDROCK_API_KEY") or os.getenv("AWS_BEARER_TOKEN_BEDROCK") or None
        cfg.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID") or None
        cfg.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY") or None
        cfg.aws_session_token = os.getenv("AWS_SESSION_TOKEN") or None
        cfg.max_output_tokens = _env_int("DEVAGENT_MAX_OUTPUT_TOKENS", cfg.max_output_tokens)
        cfg.temperature = _env_opt_float("DEVAGENT_TEMPERATURE")
        cfg.request_timeout_s = _env_float("DEVAGENT_REQUEST_TIMEOUT", cfg.request_timeout_s)
        cfg.max_retries = _env_int("DEVAGENT_MAX_RETRIES", cfg.max_retries)
        cfg.prompt_cache = _env_bool("BEDROCK_PROMPT_CACHE", cfg.prompt_cache)
        extra = os.getenv("BEDROCK_EXTRA_FIELDS")
        if extra:
            try:
                cfg.extra_request_fields = json.loads(extra)
            except json.JSONDecodeError as e:  # pragma: no cover - config error path
                raise ValueError(f"BEDROCK_EXTRA_FIELDS is not valid JSON: {e}") from e

        cfg.price_in_per_m = _env_float("DEVAGENT_PRICE_IN_PER_M", cfg.price_in_per_m)
        cfg.price_out_per_m = _env_float("DEVAGENT_PRICE_OUT_PER_M", cfg.price_out_per_m)
        cfg.price_cache_read_per_m = _env_float("DEVAGENT_PRICE_CACHE_READ_PER_M", cfg.price_cache_read_per_m)
        cfg.price_cache_write_per_m = _env_float("DEVAGENT_PRICE_CACHE_WRITE_PER_M", cfg.price_cache_write_per_m)

        cfg.context_window_tokens = _env_int("DEVAGENT_CONTEXT_WINDOW", cfg.context_window_tokens)
        cfg.compaction_ratio = _env_float("DEVAGENT_COMPACTION_RATIO", cfg.compaction_ratio)
        cfg.keep_recent_messages = _env_int("DEVAGENT_KEEP_RECENT", cfg.keep_recent_messages)
        cfg.repomap_max_tokens = _env_int("DEVAGENT_REPOMAP_TOKENS", cfg.repomap_max_tokens)

        cfg.mode = os.getenv("DEVAGENT_MODE", cfg.mode)
        cfg.budget = Budget(
            max_steps=_env_int("DEVAGENT_MAX_STEPS", cfg.budget.max_steps),
            max_tokens=_env_int("DEVAGENT_MAX_TOKENS", cfg.budget.max_tokens),
            max_cost_usd=_env_float("DEVAGENT_MAX_COST_USD", cfg.budget.max_cost_usd),
            max_wall_seconds=_env_int("DEVAGENT_MAX_WALL_SECONDS", cfg.budget.max_wall_seconds),
        )
        cfg.enable_delegate = _env_bool("DEVAGENT_ENABLE_DELEGATE", cfg.enable_delegate)
        cfg.use_worktree = _env_bool("DEVAGENT_USE_WORKTREE", cfg.use_worktree)
        cfg.log_prompts = _env_bool("DEVAGENT_LOG_PROMPTS", cfg.log_prompts)
        cfg.sandbox = os.getenv("DEVAGENT_SANDBOX", cfg.sandbox).strip().lower()
        cfg.docker_image = os.getenv("DEVAGENT_DOCKER_IMAGE", cfg.docker_image)
        cfg.docker_memory = os.getenv("DEVAGENT_DOCKER_MEMORY", cfg.docker_memory)
        cfg.docker_cpus = os.getenv("DEVAGENT_DOCKER_CPUS", cfg.docker_cpus)
        cfg.max_test_attempts = _env_int("DEVAGENT_MAX_TEST_ATTEMPTS", cfg.max_test_attempts)

        known = {f.name for f in fields(cls)}
        for k, v in overrides.items():
            if v is None:
                continue
            if k not in known:
                raise TypeError(f"Unknown config field: {k}")
            setattr(cfg, k, v)
        return cfg

    # -- helpers -------------------------------------------------------------
    def cost_usd(self, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0) -> float:
        return (
            input_tokens * self.price_in_per_m
            + output_tokens * self.price_out_per_m
            + cache_read * self.price_cache_read_per_m
            + cache_write * self.price_cache_write_per_m
        ) / 1_000_000

    def redacted(self) -> dict[str, Any]:
        """A dict safe to write into traces: secrets removed."""
        d: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name in SECRET_FIELDS:
                d[f.name] = "<set>" if v else None
            elif f.name == "budget":
                d[f.name] = vars(v)
            else:
                d[f.name] = v
        return d
