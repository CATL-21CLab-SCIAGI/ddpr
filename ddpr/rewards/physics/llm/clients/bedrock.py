from __future__ import annotations

import json
import math
import os
import signal
import threading
import time
from contextlib import contextmanager
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict

from .config import load_dotenv

BEDROCK_API_KEY_ENV = "AWS_BEDROCK_API_KEY"
AWS_BEDROCK_BEARER_ENV = "AWS_BEARER_TOKEN_BEDROCK"
_RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


class BedrockRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str | None = None
    reasoning_content: str | None = None
    response_id: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


class _BedrockWallClockTimeout(TimeoutError):
    pass


def require_bedrock_api_key() -> str:
    load_dotenv()
    api_key = os.getenv(BEDROCK_API_KEY_ENV) or os.getenv(AWS_BEDROCK_BEARER_ENV)
    if not api_key:
        raise RuntimeError(
            f"Bedrock API key not found. Set {BEDROCK_API_KEY_ENV} in "
            ".env or the process environment."
        )
    return api_key


def run_bedrock_responses(
    *,
    messages: list[dict[str, str]],
    model: str,
    region: str,
    max_tokens: int,
    timeout_seconds: float,
    reasoning_effort: str,
    json_mode: bool,
    max_attempts: int = 3,
    initial_backoff_seconds: float = 1.0,
) -> BedrockRunResult:
    payload: dict[str, Any] = {
        "model": model,
        "input": messages,
        "max_output_tokens": max_tokens,
        "store": False,
        "reasoning": {"effort": reasoning_effort},
    }
    if json_mode:
        payload["text"] = {"format": {"type": "json_object"}}
    response = post_bedrock_json(
        url=(f"https://bedrock-mantle.{region}.api.aws/openai/v1/responses"),
        payload=payload,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        initial_backoff_seconds=initial_backoff_seconds,
    )
    output = response.get("output", [])
    if not isinstance(output, list):
        raise RuntimeError("Bedrock response output must be a list.")
    return BedrockRunResult(
        content=_responses_text(output),
        reasoning_content=_responses_reasoning(output),
        response_id=_optional_string(response.get("id")),
        finish_reason=_responses_finish_reason(response),
        usage=_json_safe(response.get("usage")),
    )


def post_bedrock_json(
    *,
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    max_attempts: int = 3,
    initial_backoff_seconds: float = 1.0,
) -> dict[str, Any]:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    if not math.isfinite(initial_backoff_seconds) or initial_backoff_seconds < 0:
        raise ValueError("initial_backoff_seconds must be finite and nonnegative")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    body = json.dumps(
        payload,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    deadline = time.monotonic() + timeout_seconds

    for attempt in range(1, max_attempts + 1):
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise TimeoutError(
                f"Bedrock request exceeded {timeout_seconds:g} seconds across retries."
            )
        request = Request(
            url,
            data=body,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {require_bedrock_api_key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with _wall_clock_deadline(remaining_seconds):
                with urlopen(request, timeout=remaining_seconds) as response:
                    response_body = response.read().decode("utf-8")
                    request_id = response.headers.get("x-amzn-requestid")
        except TimeoutError as exc:
            raise TimeoutError(
                f"Bedrock request exceeded {timeout_seconds:g} seconds."
            ) from exc
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if (
                _is_retryable_http_error(exc.code, error_body)
                and attempt < max_attempts
            ):
                _bounded_backoff(
                    attempt, deadline=deadline, initial_seconds=initial_backoff_seconds
                )
                continue
            raise RuntimeError(
                f"Bedrock request failed with HTTP {exc.code}: "
                f"{_safe_error_message(error_body)}"
            ) from exc
        except URLError as exc:
            if (
                isinstance(exc.reason, TimeoutError)
                or "timed out" in str(exc.reason).casefold()
            ):
                raise TimeoutError(
                    f"Bedrock request exceeded {timeout_seconds:g} seconds."
                ) from exc
            if attempt < max_attempts:
                _bounded_backoff(
                    attempt, deadline=deadline, initial_seconds=initial_backoff_seconds
                )
                continue
            raise RuntimeError(f"Bedrock request failed: {exc.reason}") from exc

        try:
            result = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Bedrock response was not valid JSON.") from exc
        if not isinstance(result, dict):
            raise RuntimeError("Bedrock response must be a JSON object.")
        if request_id:
            result.setdefault("_request_id", request_id)
        return result

    raise AssertionError("unreachable")


def _bounded_backoff(attempt: int, *, deadline: float, initial_seconds: float) -> None:
    delay = initial_seconds * 2 ** (attempt - 1)
    time.sleep(min(delay, max(0.0, deadline - time.monotonic())))


@contextmanager
def _wall_clock_deadline(timeout_seconds: float):
    can_use_alarm = (
        threading.current_thread() is threading.main_thread()
        and hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
    )
    if not can_use_alarm or signal.getitimer(signal.ITIMER_REAL)[0] > 0:
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def expire(_signum: int, _frame: Any) -> None:
        raise _BedrockWallClockTimeout

    signal.signal(signal.SIGALRM, expire)
    try:
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        yield
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        finally:
            signal.signal(signal.SIGALRM, previous_handler)


def _is_retryable_http_error(code: int, body: str) -> bool:
    if code in _RETRYABLE_HTTP_CODES:
        return True
    if code != 400:
        return False
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    error = payload.get("error", payload)
    if not isinstance(error, dict):
        return False
    error_code = str(error.get("code", "")).casefold()
    error_type = str(error.get("type", "")).casefold()
    message = str(error.get("message", "")).casefold()
    return (
        "validation_error" in {error_code, error_type}
        and "internal server error" in message
    )


def _safe_error_message(body: str) -> str:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body[:1000]
    if isinstance(payload, dict):
        return str(payload.get("error", payload))[:1000]
    return str(payload)[:1000]


def _responses_text(output: list[Any]) -> str | None:
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for block in item.get("content", []):
            if isinstance(block, dict) and block.get("type") == "output_text":
                parts.append(str(block.get("text", "")))
    text = "".join(parts)
    return text or None


def _responses_reasoning(output: list[Any]) -> str | None:
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        for summary in item.get("summary", []):
            if isinstance(summary, dict) and summary.get("text"):
                parts.append(str(summary["text"]))
    return "\n".join(parts) or None


def _responses_finish_reason(response: dict[str, Any]) -> str | None:
    status = _optional_string(response.get("status"))
    incomplete = response.get("incomplete_details")
    if status == "incomplete" and isinstance(incomplete, dict):
        return _optional_string(incomplete.get("reason")) or status
    return status


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return str(value)
