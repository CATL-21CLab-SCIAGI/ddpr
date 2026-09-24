from __future__ import annotations

import multiprocessing
import os
import signal
import threading
from collections.abc import Callable
from typing import Any

from .primitives import (
    GradingPlan,
    GradingResult,
    extract_gradeable_answer,
    grade_with_plan,
)

DEFAULT_GRADER_TIMEOUT_SECONDS = 120.0


class GraderTimeout(BaseException):
    """Bypass broad exception handlers inside mathematical evaluation."""


class AnswerExtractionError(ValueError):
    pass


def grader_timeout_seconds() -> float:
    seconds = float(
        os.getenv(
            "AUTO_CRITPT_GRADER_TIMEOUT_SECONDS", str(DEFAULT_GRADER_TIMEOUT_SECONDS)
        )
    )
    if seconds <= 0:
        raise ValueError("AUTO_CRITPT_GRADER_TIMEOUT_SECONDS must be positive")
    return seconds


def evaluate_answer_locally(
    response: str,
    plan: GradingPlan,
    extract: Callable[..., dict[str, Any]] = extract_gradeable_answer,
    grade: Callable[..., GradingResult] = grade_with_plan,
) -> tuple[dict[str, Any], GradingResult]:
    answer = extract_answer_locally(response, plan, extract)
    return answer, grade(answer, plan)


def extract_answer_locally(
    response: str,
    plan: GradingPlan,
    extract: Callable[..., dict[str, Any]] = extract_gradeable_answer,
) -> dict[str, Any]:
    try:
        return extract(response, grading_plan=plan)
    except ValueError as exc:
        raise AnswerExtractionError(str(exc)) from exc


def _worker(sender: Any, function: Callable[..., Any], args: tuple[Any, ...]) -> None:
    try:
        sender.send(("success", function(*args)))
    except BaseException as exc:
        kind = "extraction_error" if isinstance(exc, AnswerExtractionError) else "error"
        sender.send((kind, str(exc)))
    finally:
        sender.close()


def run_with_deadline(
    function: Callable[..., Any],
    *args: Any,
    timeout_seconds: float | None = None,
) -> Any:
    """Bound extraction and comparison, including calls from worker threads."""
    seconds = grader_timeout_seconds() if timeout_seconds is None else timeout_seconds
    if seconds <= 0:
        raise GraderTimeout
    use_alarm = (
        threading.current_thread() is threading.main_thread()
        and hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
        and signal.getitimer(signal.ITIMER_REAL)[0] == 0
    )
    if use_alarm:
        previous = signal.getsignal(signal.SIGALRM)

        def expire(_signum: int, _frame: Any) -> None:
            raise GraderTimeout

        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            return function(*args)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker,
        args=(sender, function, args),
        daemon=True,
    )
    started = False
    try:
        process.start()
        started = True
        sender.close()
        if not receiver.poll(seconds):
            raise GraderTimeout
        try:
            kind, payload = receiver.recv()
        except EOFError as exc:
            raise RuntimeError("Local grader process exited without a result") from exc
        if kind == "extraction_error":
            raise AnswerExtractionError(payload)
        if kind != "success":
            raise RuntimeError(payload)
        return payload
    finally:
        receiver.close()
        sender.close()
        if started:
            process.join(timeout=0.1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
