"""Shared transport settings for the two independently prompted Judges."""

from __future__ import annotations

import json
import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .backends import GenerationConfig, OpenAIChatBackend, OpenAIChatBackendConfig
from .bedrock import run_bedrock_responses
from .schemas import CompletionResult


class JudgeBackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["bedrock", "openai"] = "bedrock"
    model: str = "openai.gpt-5.6-sol"
    region: str = "us-east-1"
    max_tokens: int = Field(default=16000, ge=1)
    reasoning_effort: str = "high"
    timeout_seconds: float = Field(default=600.0, gt=0.0)
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key_env: str | None = "OPENAI_API_KEY"
    http_max_attempts: int = Field(default=3, ge=1, le=10)
    http_backoff_seconds: float = Field(default=1.0, ge=0.0)
    extra_body: dict[str, Any] = Field(default_factory=dict)


def transport_from_env(prefix: str) -> dict[str, Any]:
    raw = os.getenv(f"{prefix}_EXTRA_BODY_JSON", "").strip()
    extra = json.loads(raw) if raw else {}
    if not isinstance(extra, dict):
        raise ValueError(f"{prefix}_EXTRA_BODY_JSON must decode to a JSON object")
    return {
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


def complete_judge(
    messages: list[dict[str, str]], config: JudgeBackendConfig
) -> CompletionResult:
    if config.backend == "bedrock":
        run = run_bedrock_responses(
            messages=messages,
            model=config.model,
            region=config.region,
            max_tokens=config.max_tokens,
            timeout_seconds=config.timeout_seconds,
            reasoning_effort=config.reasoning_effort,
            json_mode=True,
        )
        return CompletionResult(
            content=run.content,
            reasoning_content=run.reasoning_content,
            response_id=run.response_id,
            finish_reason=run.finish_reason,
            usage=run.usage,
        )
    backend = OpenAIChatBackend(
        OpenAIChatBackendConfig(
            base_url=config.base_url,
            api_key_env=config.api_key_env,
            max_attempts=config.http_max_attempts,
            initial_backoff_seconds=config.http_backoff_seconds,
        )
    )
    return backend.complete(
        messages,
        GenerationConfig(
            model=config.model,
            max_tokens=config.max_tokens,
            timeout_seconds=config.timeout_seconds,
            temperature=0.0,
            extra_body={
                **config.extra_body,
                "response_format": {"type": "json_object"},
            },
        ),
    )
