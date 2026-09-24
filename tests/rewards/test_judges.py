import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("sympy")

from ddpr.rewards.physics import grade_response
from ddpr.rewards.physics._core import bedrock, judge_backend
from ddpr.rewards.physics._core.grading_runtime import GraderTimeout, run_with_deadline
from ddpr.rewards.physics.equivalence import (
    EquivalenceJudgeConfig,
    ExpressionEquivalenceJudge,
)
from ddpr.rewards.physics.schemas import CompletionResult
from ddpr.rewards.physics.semantic import LLMSemanticJudge, SemanticJudgeConfig


def problem():
    return {
        "problem_id": "freestyle",
        "problem_statement": "Compute the energy.",
        "public_solver_output_instructions": "Give the value and explain.",
        "answer_type": "exact",
        "reference_answer": "2",
        "reference_solution": "E=1+1=2.",
    }


def correct():
    return json.dumps(
        {
            "verdict": "correct",
            "confidence": 0.95,
            "solver_answer_summary": "E=2",
            "assessment": "Agrees with reference.",
            "substantive_errors": [],
        }
    )


def test_semantic_schema_repair_is_bounded():
    responses = iter(["invalid", correct()])
    prompts = []

    def complete(messages, _config):
        prompts.append(messages)
        return next(responses)

    judge = LLMSemanticJudge(
        config=SemanticJudgeConfig(model="mock"), completion=complete
    )
    assert grade_response(problem(), "E=2", semantic_judge=judge).reward == 1
    assert len(prompts) == 2
    assert prompts[1][-2] == {"role": "assistant", "content": "invalid"}
    assert "schema" in prompts[1][-1]["content"]


@pytest.mark.parametrize("mode", ["malformed", "api"])
def test_semantic_failures_remain_unresolved(mode):
    calls = []

    def complete(messages, _config):
        calls.append(messages)
        if mode == "api":
            raise TimeoutError("mock timeout")
        return "invalid"

    judge = LLMSemanticJudge(
        config=SemanticJudgeConfig(model="mock"), completion=complete
    )
    report = grade_response(problem(), "E=2", semantic_judge=judge)
    assert report.reward is None
    assert report.grading_status == "grader_error"
    assert len(calls) == (1 if mode == "api" else 2)


def test_openai_equivalence_transport(monkeypatch):
    captured = {}

    def complete(self, messages, generation):
        captured.update(config=self.config, messages=messages, generation=generation)
        return CompletionResult(
            content=json.dumps(
                {
                    "verdict": "not_equivalent",
                    "confidence": 0.99,
                    "rationale": "Different expressions.",
                    "assumptions_used": [],
                }
            )
        )

    monkeypatch.setattr(judge_backend.OpenAIChatBackend, "complete", complete)
    judge = ExpressionEquivalenceJudge(
        config=EquivalenceJudgeConfig(
            backend="openai",
            model="local-judge",
            base_url="http://localhost:8000/v1",
            api_key_env=None,
        )
    )
    source = {
        "problem_id": "auto",
        "problem_statement": "Find f(x).",
        "public_solver_output_instructions": "Return answer(x).",
        "answer_contract": {"answer_type": "symbolic"},
        "grading_plan": {
            "calls": [
                {
                    "primitive": "symbolic",
                    "target": "answer",
                    "expected": "x+1",
                    "args": {"variables": ["x"]},
                }
            ]
        },
    }
    report = grade_response(
        source,
        "```python\ndef answer(x):\n    return x+2\n```",
        equivalence_judge=judge,
    )
    assert report.reward == 0
    assert captured["config"].base_url == "http://localhost:8000/v1"
    assert captured["generation"].extra_body["response_format"] == {
        "type": "json_object"
    }
    assert "Find f(x)." in json.dumps(captured["messages"])


def test_bedrock_semantic_transport(monkeypatch):
    captured = {}

    class Response:
        def __init__(self):
            self.headers = {"x-amzn-requestid": "mock-request"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "id": "mock-response",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": correct()},
                            ],
                        }
                    ],
                }
            ).encode()

    def urlopen(request, timeout):
        captured.update(
            url=request.full_url,
            body=json.loads(request.data),
            authorization=request.headers["Authorization"],
            timeout=timeout,
        )
        return Response()

    monkeypatch.setenv("AWS_BEDROCK_API_KEY", "mock-key")
    monkeypatch.setattr(bedrock, "urlopen", urlopen)
    judge = LLMSemanticJudge(
        config=SemanticJudgeConfig(
            model="openai.gpt-5.6-sol",
            region="us-east-1",
            reasoning_effort="high",
        )
    )
    assert grade_response(problem(), "E=2", semantic_judge=judge).reward == 1
    assert captured["authorization"] == "Bearer mock-key"
    assert "us-east-1" in captured["url"]
    assert captured["body"]["reasoning"] == {"effort": "high"}
    assert captured["body"]["store"] is False


@pytest.mark.parametrize("threaded", [False, True])
def test_local_execution_deadline_in_training_threads(threaded):
    def run():
        with pytest.raises(GraderTimeout):
            run_with_deadline(time.sleep, 10, timeout_seconds=0.05)

    if threaded:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(run).result(timeout=5)
    else:
        run()
