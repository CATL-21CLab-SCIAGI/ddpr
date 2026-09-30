"""DDSR-Bench backend, using real HTTP parsing with mocked responses."""

import asyncio
import json

import pytest

pytest.importorskip("ddsr_bench.generation.client")
pytest.importorskip("pydantic")

import httpx
from ddsr_bench.generation.client import CLIENTS, ClientError, VLLMClient

from ddpr.rewards.physics import load_problem
from ddpr.rewards.physics.llm.clients.client import DdsrBackend
from ddpr.rewards.physics.llm.clients.config import ClientConfig


@pytest.mark.parametrize("inside_loop", [False, True])
def test_completion(monkeypatch, inside_loop):
    requests, clients = [], []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "request-1",
                "model": "judge",
                "choices": [
                    {
                        "message": {
                            "content": '{"verdict":"correct"}',
                            "reasoning_content": "private reasoning",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 8},
            },
        )

    def create(**kwargs):
        client = VLLMClient(**kwargs, transport=httpx.MockTransport(respond))
        clients.append(client)
        return client

    monkeypatch.setitem(CLIENTS, "vllm", create)
    monkeypatch.setenv("TEST_JUDGE_KEY", "test-key")
    config = ClientConfig(model="judge", api_key_env="TEST_JUDGE_KEY", max_tokens=64)
    complete = DdsrBackend()
    messages = [{"role": "user", "content": "Assess this answer"}]

    async def invoke():
        return complete.complete(messages, config)

    result = (
        asyncio.run(invoke()) if inside_loop else complete.complete(messages, config)
    )
    assert result.content == '{"verdict":"correct"}'
    assert result.reasoning_content == "private reasoning"
    assert result.response_id == "request-1"
    assert result.model == "judge"
    assert result.finish_reason == "stop"
    assert result.usage == {"completion_tokens": 8}
    assert result.provider == "ddsr-bench/vllm"
    assert clients[0].http.is_closed
    assert clients[0].retries == config.http_max_attempts - 1
    payload = json.loads(requests[0].content)
    assert payload["messages"] == messages
    assert payload["max_tokens"] == 64
    assert requests[0].headers["authorization"] == "Bearer test-key"


def test_timeout_closes_client(monkeypatch):
    clients = []

    async def respond(request):
        await asyncio.sleep(1)
        raise AssertionError("timeout should cancel request")

    def create(**kwargs):
        client = VLLMClient(**kwargs, transport=httpx.MockTransport(respond))
        clients.append(client)
        return client

    monkeypatch.setitem(CLIENTS, "vllm", create)
    with pytest.raises(TimeoutError):
        DdsrBackend().complete([], ClientConfig(api_key_env=None, timeout_seconds=0.01))
    assert clients[0].http.is_closed


def test_unsupported_configuration():
    with pytest.raises(ValueError, match="extra_body"):
        DdsrBackend().complete([], ClientConfig(extra_body={"unsupported": True}))


@pytest.mark.parametrize(
    "metadata,error",
    [
        ({"task_path": "relative"}, ValueError),
        ({"task_path": 1}, TypeError),
        ({"task_path": "/task", "problem": {}}, ValueError),
        ({"task_path": "/task", "problem_path": "/problem"}, ValueError),
    ],
)
def test_task_metadata(metadata, error):
    with pytest.raises(error):
        load_problem(metadata)


def test_semantic_response_parsing(monkeypatch):
    from ddpr.rewards.physics.llm.semantic import LLMSemanticJudge, SemanticJudgeConfig

    verdict = {
        "verdict": "incorrect",
        "confidence": 0.99,
        "solver_answer_summary": "Wrong answer",
        "assessment": "Disagrees",
        "substantive_errors": ["Wrong value"],
    }

    def respond(request):
        return httpx.Response(
            200,
            json={
                "model": "judge",
                "id": "judgment-1",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(verdict),
                            "reasoning_content": '{"verdict":"correct"}',
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    monkeypatch.setitem(
        CLIENTS,
        "vllm",
        lambda **kwargs: VLLMClient(**kwargs, transport=httpx.MockTransport(respond)),
    )
    judge = LLMSemanticJudge(
        config=SemanticJudgeConfig(
            backend="ddsr", provider="vllm", model="judge", api_key_env=None
        ),
    )
    result = judge.judge(
        "wrong",
        public_problem={
            "problem_statement": "A question",
            "answer_type": "text",
            "answer_items": ["answer"],
        },
        reference_answer="right",
        reference_solution="explanation",
    )
    assert result.verdict == "incorrect"
    assert result.response_id == "judgment-1"


@pytest.mark.parametrize("backend", ["vllm", "openai", "aliyun"])
@pytest.mark.parametrize("status", [200, 401, 503])
def test_default_dispatch(monkeypatch, backend, status):
    from ddpr.rewards.physics.llm.clients.client import complete_judge

    provider = CLIENTS[backend]
    requests, clients = [], []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            status,
            json={
                "model": "judge",
                "choices": [
                    {"message": {"content": "visible"}, "finish_reason": "stop"}
                ],
            },
        )

    def create(**kwargs):
        client = provider(**kwargs, transport=httpx.MockTransport(respond))
        clients.append(client)
        return client

    monkeypatch.setitem(CLIENTS, backend, create)
    config = ClientConfig(
        backend="ddsr", provider=backend, model="judge", api_key_env=None
    )
    if status == 200:
        result = complete_judge([], config)
        assert result.content == "visible"
        assert result.provider == f"ddsr-bench/{backend}"
    else:
        with pytest.raises(ClientError) as raised:
            complete_judge([], config)
        assert str(status) in str(raised.value)
    # DDSR-Bench avoids retrying a POST after a potentially paid generation.
    assert len(requests) == 1
    assert clients[0].http.is_closed
    payload = json.loads(requests[0].content)
    budget_key = "max_completion_tokens" if backend == "openai" else "max_tokens"
    assert payload[budget_key] == config.max_tokens
