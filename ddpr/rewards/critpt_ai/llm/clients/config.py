"""Completion settings and environment configuration."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...schemas import CompletionResult


class ClientConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["bedrock", "openai", "ddsr"] = "bedrock"
    provider: Literal["vllm", "openai", "aliyun", "bedrock"] = "vllm"
    model: str = "openai.gpt-5.6-sol"
    region: str = "us-east-1"
    max_tokens: int = Field(default=16000, ge=1)
    reasoning_effort: str = "high"
    timeout_seconds: float = Field(default=600.0, gt=0.0, allow_inf_nan=False)
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key_env: str | None = "OPENAI_API_KEY"
    http_max_attempts: int = Field(default=3, ge=1, le=10)
    http_backoff_seconds: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    extra_body: dict[str, Any] = Field(default_factory=dict)


def client_from_env(prefix: str) -> dict[str, Any]:
    load_dotenv()
    raw = os.getenv(f"{prefix}_EXTRA_BODY_JSON", "").strip()
    extra = json.loads(raw) if raw else {}
    if not isinstance(extra, dict):
        raise ValueError(f"{prefix}_EXTRA_BODY_JSON must decode to a JSON object")
    values = {
        "backend": os.getenv(f"{prefix}_BACKEND", "bedrock").strip().casefold(),
        "base_url": os.getenv(f"{prefix}_BASE_URL", "http://127.0.0.1:8000/v1"),
        "api_key_env": os.getenv(f"{prefix}_API_KEY_ENV", "OPENAI_API_KEY").strip()
        or None,
        "http_max_attempts": int(os.getenv(f"{prefix}_HTTP_MAX_ATTEMPTS", "3")),
        "http_backoff_seconds": float(
            os.getenv(f"{prefix}_HTTP_BACKOFF_SECONDS", "1.0")
        ),
        "extra_body": extra,
    }

    for field in (
        "provider",
        "model",
        "region",
        "max_tokens",
        "reasoning_effort",
        "timeout_seconds",
    ):
        value = os.getenv(f"{prefix}_{field.upper()}")
        if value is not None:
            values[field] = value
    return values


Completion = Callable[[list[dict[str, str]], ClientConfig], str | CompletionResult]


def load_dotenv(path: Path | None = None) -> None:
    source = path or Path.cwd() / ".env"
    if not source.exists():
        return
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)
