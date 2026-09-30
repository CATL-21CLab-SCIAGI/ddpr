"""Default synchronous model transport for physics assessments."""

from __future__ import annotations

from ..schemas import CompletionResult
from .bedrock import run_bedrock_responses
from .config import JudgeBackendConfig
from .openai import GenerationConfig, OpenAIChatBackend, OpenAIChatBackendConfig


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
