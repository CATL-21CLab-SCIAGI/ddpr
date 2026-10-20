from __future__ import annotations

import json
import os
import socket
import time
from collections.abc import Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...schemas import CompletionResult, ProviderCallRecord

DEFAULT_MAX_TOKENS = 96000
DEFAULT_TIMEOUT_SECONDS = 1500.0
_RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}
_RESERVED_CHAT_FIELDS = {
    "messages",
    "model",
    "max_tokens",
    "stream",
    "temperature",
    "top_p",
    "seed",
}


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1)
    timeout_seconds: float = Field(
        default=DEFAULT_TIMEOUT_SECONDS,
        gt=0.0,
        allow_inf_nan=False,
    )
    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    seed: int | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_extra_body(self) -> "GenerationConfig":
        conflicts = sorted(_RESERVED_CHAT_FIELDS.intersection(self.extra_body))
        if conflicts:
            raise ValueError(
                "extra_body may not override reserved fields: " + ", ".join(conflicts)
            )
        return self


class OpenAIChatBackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = "http://127.0.0.1:8000/v1"
    api_key_env: str | None = "OPENAI_API_KEY"
    max_attempts: int = Field(default=3, ge=1, le=10)
    initial_backoff_seconds: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)


class BackendInfrastructureError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        provider_calls: list[ProviderCallRecord],
    ) -> None:
        super().__init__(message)
        self.provider_calls = provider_calls


class BackendTimeLimit(TimeoutError):
    def __init__(
        self,
        message: str,
        *,
        provider_calls: list[ProviderCallRecord],
    ) -> None:
        super().__init__(message)
        self.provider_calls = provider_calls


class OpenAIChatBackend:
    name = "openai-chat"

    def __init__(
        self,
        config: OpenAIChatBackendConfig | None = None,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config or OpenAIChatBackendConfig()
        self.environment = environment if environment is not None else os.environ

    @property
    def endpoint(self) -> str:
        base = self.config.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"

    def complete(
        self,
        messages: list[dict[str, str]],
        generation: GenerationConfig,
    ) -> CompletionResult:
        payload: dict[str, Any] = {
            "model": generation.model,
            "messages": messages,
            "max_tokens": generation.max_tokens,
            "stream": False,
            **generation.extra_body,
        }
        if generation.temperature is not None:
            payload["temperature"] = generation.temperature
        if generation.top_p is not None:
            payload["top_p"] = generation.top_p
        if generation.seed is not None:
            payload["seed"] = generation.seed

        body = json.dumps(
            payload,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        calls: list[ProviderCallRecord] = []
        overall_started = time.perf_counter()
        deadline = overall_started + generation.timeout_seconds
        for call_index in range(1, self.config.max_attempts + 1):
            remaining_seconds = deadline - time.perf_counter()
            if remaining_seconds <= 0:
                raise BackendTimeLimit(
                    "OpenAI-compatible request exceeded "
                    f"{generation.timeout_seconds:g} seconds across retries.",
                    provider_calls=calls,
                )
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
            api_key = self._api_key()
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            request = Request(
                self.endpoint,
                data=body,
                headers=headers,
                method="POST",
            )
            started = time.perf_counter()
            try:
                with urlopen(
                    request,
                    timeout=remaining_seconds,
                ) as response:
                    response_body = response.read().decode("utf-8")
                    request_id = response.headers.get("x-request-id")
            except HTTPError as exc:
                latency = time.perf_counter() - started
                error_body = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code in _RETRYABLE_HTTP_CODES
                calls.append(
                    ProviderCallRecord(
                        call_index=call_index,
                        status=(
                            "retryable_error" if retryable else "non_retryable_error"
                        ),
                        latency_seconds=latency,
                        http_status=exc.code,
                        request_id=(exc.headers or {}).get("x-request-id"),
                        error_type="HTTPError",
                        error_message=_safe_error_message(error_body),
                    )
                )
                if retryable and call_index < self.config.max_attempts:
                    self._backoff(call_index, deadline=deadline)
                    continue
                raise BackendInfrastructureError(
                    f"OpenAI-compatible request failed with HTTP {exc.code}: "
                    f"{_safe_error_message(error_body)}",
                    provider_calls=calls,
                ) from exc
            except (TimeoutError, socket.timeout) as exc:
                latency = time.perf_counter() - started
                calls.append(
                    ProviderCallRecord(
                        call_index=call_index,
                        status="time_limit",
                        latency_seconds=latency,
                        error_type=type(exc).__name__,
                        error_message=str(exc) or "request timed out",
                    )
                )
                raise BackendTimeLimit(
                    "OpenAI-compatible request exceeded "
                    f"{generation.timeout_seconds:g} seconds.",
                    provider_calls=calls,
                ) from exc
            except URLError as exc:
                latency = time.perf_counter() - started
                if (
                    isinstance(
                        exc.reason,
                        (TimeoutError, socket.timeout),
                    )
                    or "timed out" in str(exc.reason).casefold()
                ):
                    calls.append(
                        ProviderCallRecord(
                            call_index=call_index,
                            status="time_limit",
                            latency_seconds=latency,
                            error_type=type(exc.reason).__name__,
                            error_message=str(exc.reason),
                        )
                    )
                    raise BackendTimeLimit(
                        "OpenAI-compatible request exceeded "
                        f"{generation.timeout_seconds:g} seconds.",
                        provider_calls=calls,
                    ) from exc
                calls.append(
                    ProviderCallRecord(
                        call_index=call_index,
                        status="retryable_error",
                        latency_seconds=latency,
                        error_type=type(exc.reason).__name__,
                        error_message=str(exc.reason),
                    )
                )
                if call_index < self.config.max_attempts:
                    self._backoff(call_index, deadline=deadline)
                    continue
                raise BackendInfrastructureError(
                    f"OpenAI-compatible request failed: {exc.reason}",
                    provider_calls=calls,
                ) from exc

            latency = time.perf_counter() - started
            try:
                result = json.loads(response_body)
            except json.JSONDecodeError as exc:
                calls.append(
                    ProviderCallRecord(
                        call_index=call_index,
                        status="non_retryable_error",
                        latency_seconds=latency,
                        request_id=request_id,
                        error_type="JSONDecodeError",
                        error_message="response was not valid JSON",
                    )
                )
                raise BackendInfrastructureError(
                    "OpenAI-compatible response was not valid JSON.",
                    provider_calls=calls,
                ) from exc
            calls.append(
                ProviderCallRecord(
                    call_index=call_index,
                    status="success",
                    latency_seconds=latency,
                    request_id=request_id,
                )
            )
            try:
                return _parse_chat_completion(
                    result,
                    calls=calls,
                    latency_seconds=time.perf_counter() - overall_started,
                )
            except BackendInfrastructureError as exc:
                calls[-1].status = "non_retryable_error"
                calls[-1].error_type = type(exc).__name__
                calls[-1].error_message = str(exc)
                raise

        raise AssertionError("unreachable")

    def _api_key(self) -> str | None:
        if self.config.api_key_env is None:
            return None
        value = self.environment.get(self.config.api_key_env)
        return value if value else None

    def _backoff(self, call_index: int, *, deadline: float) -> None:
        delay = self.config.initial_backoff_seconds * (2 ** (call_index - 1))
        if delay > 0:
            time.sleep(min(delay, max(0.0, deadline - time.perf_counter())))


def _parse_chat_completion(
    payload: Any,
    *,
    calls: list[ProviderCallRecord],
    latency_seconds: float,
) -> CompletionResult:
    if not isinstance(payload, dict):
        raise BackendInfrastructureError(
            "OpenAI-compatible response must be a JSON object.",
            provider_calls=calls,
        )
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise BackendInfrastructureError(
            "OpenAI-compatible response must contain at least one choice.",
            provider_calls=calls,
        )
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise BackendInfrastructureError(
            "OpenAI-compatible choice must contain one message object.",
            provider_calls=calls,
        )
    message = choice["message"]
    return CompletionResult(
        content=_message_text(message.get("content")),
        reasoning_content=_message_text(
            message.get("reasoning_content", message.get("reasoning"))
        ),
        model=_optional_string(payload.get("model")),
        response_id=_optional_string(payload.get("id")),
        finish_reason=_optional_string(choice.get("finish_reason")),
        usage=payload.get("usage") if isinstance(payload.get("usage"), dict) else None,
        provider="openai-chat",
        latency_seconds=latency_seconds,
        provider_calls=calls,
    )


def _message_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return None
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str):
            parts.append(text)
    joined = "".join(parts)
    return joined or None


def _safe_error_message(body: str) -> str:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body[:1000]
    if isinstance(payload, dict):
        error = payload.get("error", payload)
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"][:1000]
        return str(error)[:1000]
    return str(payload)[:1000]


def _optional_string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None
