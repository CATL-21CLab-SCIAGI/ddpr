import signal
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("sympy")

from ddpr.rewards.physics.runtime import (
    GraderTimeout,
    grader_timeout,
    run_with_deadline,
)


@pytest.mark.parametrize("threaded", [False, True])
def test_local_execution_deadline_in_training_threads(threaded):
    def run():
        with pytest.raises(GraderTimeout):
            run_with_deadline(time.sleep, 10, timeout=0.05)

    if threaded:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(run).result(timeout=5)
    else:
        run()


@pytest.mark.parametrize("seconds", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("threaded", [False, True])
def test_nonfinite_deadline_does_not_change_signal_handler(
    monkeypatch, seconds, threaded
):
    def forbidden(*args):
        pytest.fail("Invalid timeout must not change signal handlers")

    monkeypatch.setattr(signal, "signal", forbidden)

    def run():
        with pytest.raises(ValueError, match="must be finite"):
            run_with_deadline(int, timeout=seconds)

    if threaded:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(run).result(timeout=5)
    else:
        run()


@pytest.mark.parametrize("seconds", ["nan", "inf", "-inf", "0", "-1"])
def test_invalid_timeout_configuration(monkeypatch, seconds):
    monkeypatch.setenv("AUTO_CRITPT_GRADER_TIMEOUT_SECONDS", seconds)
    with pytest.raises(ValueError, match="must be finite and positive"):
        grader_timeout()


@pytest.mark.parametrize("seconds", [0, -1])
def test_expired_budget_does_not_run_function(seconds):
    def forbidden():
        pytest.fail("Expired grading budget must not execute work")

    with pytest.raises(GraderTimeout):
        run_with_deadline(forbidden, timeout=seconds)


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="Requires SIGALRM")
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_timer_setup_failure_restores_signal_handler(monkeypatch, cleanup_fails):
    previous = signal.getsignal(signal.SIGALRM)
    monkeypatch.setattr(signal, "getitimer", lambda _timer: (0, 0))

    def fail_timer(_timer, seconds):
        if seconds or cleanup_fails:
            raise OSError("Timer unavailable")

    monkeypatch.setattr(signal, "setitimer", fail_timer)
    try:
        with pytest.raises(OSError, match="Timer unavailable"):
            run_with_deadline(int, timeout=1)
        assert signal.getsignal(signal.SIGALRM) == previous
    finally:
        signal.signal(signal.SIGALRM, previous)
