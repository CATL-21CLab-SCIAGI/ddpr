import io
import signal
from urllib.error import HTTPError

import pytest

pytest.importorskip("pydantic")

from ddpr.rewards.physics.llm.clients import bedrock, client, openai
from ddpr.rewards.physics.llm.equivalence import EquivalenceJudgeConfig
from ddpr.rewards.physics.llm.semantic import SemanticJudgeConfig


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize(
    "config,field",
    [
        (EquivalenceJudgeConfig, "timeout_seconds"),
        (SemanticJudgeConfig, "timeout_seconds"),
        (EquivalenceJudgeConfig, "http_backoff_seconds"),
        (openai.GenerationConfig, "timeout_seconds"),
        (openai.OpenAIChatBackendConfig, "initial_backoff_seconds"),
    ],
)
def test_nonfinite_client_settings(config, field, value):
    kwargs = {field: value}
    if config is openai.GenerationConfig:
        kwargs["model"] = "test"
    with pytest.raises(ValueError):
        config(**kwargs)


@pytest.mark.parametrize("backend", ["bedrock", "openai"])
@pytest.mark.parametrize("attempts,status", [(1, 503), (3, 503), (3, 401)])
def test_judge_retry_settings(monkeypatch, backend, attempts, status):
    calls = []
    sleeps = []
    module = bedrock if backend == "bedrock" else openai

    def fail(request, timeout):
        calls.append(request)
        raise HTTPError(request.full_url, status, "test error", {}, io.BytesIO(b"{}"))

    monkeypatch.setenv("AWS_BEDROCK_API_KEY", "test-key")
    monkeypatch.setattr(module, "urlopen", fail)
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    with pytest.raises(RuntimeError, match=f"HTTP {status}"):
        client.complete_judge(
            [{"role": "user", "content": "test"}],
            EquivalenceJudgeConfig(
                backend=backend,
                http_max_attempts=attempts,
                http_backoff_seconds=0.25,
            ),
        )
    expected = attempts if status == 503 else 1
    assert len(calls) == expected
    assert sleeps == [0.25 * 2**index for index in range(expected - 1)]


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="Requires SIGALRM")
def test_bedrock_timer_setup_failure_restores_handler(monkeypatch):
    previous = signal.getsignal(signal.SIGALRM)
    monkeypatch.setattr(signal, "getitimer", lambda _timer: (0, 0))

    def fail_timer(*args):
        raise OSError("Timer unavailable")

    monkeypatch.setattr(signal, "setitimer", fail_timer)
    try:
        with (
            pytest.raises(OSError, match="Timer unavailable"),
            bedrock._wall_clock_deadline(1),
        ):
            pytest.fail("Timer setup must fail first")
        assert signal.getsignal(signal.SIGALRM) == previous
    finally:
        signal.signal(signal.SIGALRM, previous)


@pytest.mark.parametrize("backend", ["bedrock", "openai"])
@pytest.mark.parametrize("body", [b"invalid json", b"[]"])
def test_malformed_response_is_a_client_failure(monkeypatch, backend, body):
    class Response(io.BytesIO):
        def __init__(self, body):
            super().__init__(body)
            self.headers = {}

    module = bedrock if backend == "bedrock" else openai
    monkeypatch.setenv("AWS_BEDROCK_API_KEY", "test-key")
    monkeypatch.setattr(module, "urlopen", lambda *a, **kw: Response(body))
    with pytest.raises(RuntimeError, match="JSON") as raised:
        client.complete_judge([], EquivalenceJudgeConfig(backend=backend))
    if backend == "openai":
        assert raised.value.provider_calls[-1].status == "non_retryable_error"
