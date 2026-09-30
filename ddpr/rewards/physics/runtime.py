from __future__ import annotations

import math
import multiprocessing
import os
import signal
import threading
from collections.abc import Callable
from typing import Any

DEFAULT_GRADER_TIMEOUT = 120.0  # Seconds.


class GraderTimeout(BaseException):
    """Bypass broad exception handlers inside mathematical evaluation."""


class AnswerExtractionError(ValueError):
    pass


def grader_timeout() -> float:
    """Read the configured grading timeout in seconds."""
    seconds = float(
        os.getenv("AUTO_CRITPT_GRADER_TIMEOUT_SECONDS", str(DEFAULT_GRADER_TIMEOUT))
    )
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(
            "AUTO_CRITPT_GRADER_TIMEOUT_SECONDS must be finite and positive"
        )
    return seconds


def _worker(sender: Any, fn: Callable[..., Any], args: tuple[Any, ...]) -> None:
    try:
        sender.send(("success", fn(*args)))
    except BaseException as exc:
        kind = "extraction_error" if isinstance(exc, AnswerExtractionError) else "error"
        sender.send((kind, str(exc)))
    finally:
        sender.close()


def run_with_deadline(
    fn: Callable[..., Any],
    *args: Any,
    timeout: float | None = None,
) -> Any:
    """Run ``fn`` within ``timeout`` seconds, including from worker threads."""
    seconds = grader_timeout() if timeout is None else timeout
    if not math.isfinite(seconds):
        raise ValueError("timeout must be finite")
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
        try:
            signal.setitimer(signal.ITIMER_REAL, seconds)
            return fn(*args)
        finally:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            finally:
                signal.signal(signal.SIGALRM, previous)

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker,
        args=(sender, fn, args),
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
