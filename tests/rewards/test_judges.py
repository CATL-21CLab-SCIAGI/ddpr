import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("sympy")

from ddpr.rewards.physics import grade_response
from ddpr.rewards.physics.execution.runtime import GraderTimeout, run_with_deadline
from ddpr.rewards.physics.llm.equivalence import (
    EquivalenceJudgeConfig,
    ExpressionEquivalenceJudge,
)
from ddpr.rewards.physics.llm.semantic import LLMSemanticJudge, SemanticJudgeConfig
from ddpr.rewards.physics.schemas import (
    CompletionResult,
    FreestyleProblemGradingInput,
    GradingCall,
    PrimitiveGrade,
)
from ddpr.rewards.physics.transport import bedrock, client


@pytest.mark.parametrize(
    "config_type,prefix,tokens,timeout",
    [
        (EquivalenceJudgeConfig, "AUTO_CRITPT_GRADER_JUDGE", 16000, 600),
        (SemanticJudgeConfig, "AUTO_CRITPT_GRADER_SEMANTIC_JUDGE", 32000, 1800),
    ],
)
def test_judge_environment(monkeypatch, tmp_path, config_type, prefix, tokens, timeout):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        os,
        "environ",
        {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("AUTO_CRITPT_GRADER_")
        },
    )
    defaults = config_type.from_env()
    assert defaults == config_type()
    assert defaults.max_tokens == tokens
    assert defaults.timeout_seconds == timeout

    (tmp_path / ".env").write_text(
        f"{prefix}_MODEL=dotenv-model\n{prefix}_REGION=test-region\n",
        encoding="utf-8",
    )
    overrides = {
        "MODEL": "environment-model",
        "BACKEND": " OPENAI ",
        "MAX_TOKENS": "1234",
        "TIMEOUT_SECONDS": "45.5",
        "REASONING_EFFORT": "low",
        "API_KEY_ENV": "",
        "EXTRA_BODY_JSON": '{"enable_thinking": true}',
    }
    for suffix, value in overrides.items():
        monkeypatch.setenv(f"{prefix}_{suffix}", value)
    configured = config_type.from_env()
    assert configured.model == "environment-model"
    assert configured.region == "test-region"
    assert configured.backend == "openai"
    assert configured.max_tokens == 1234
    assert configured.timeout_seconds == 45.5
    assert configured.reasoning_effort == "low"
    assert configured.api_key_env is None
    assert configured.extra_body == {"enable_thinking": True}

    monkeypatch.setenv(f"{prefix}_MAX_TOKENS", "0")
    with pytest.raises(ValueError):
        config_type.from_env()


@pytest.mark.parametrize(
    "confidence,resolution", [(0.89, "uncertain"), (0.9, "equivalent"), ("high", None)]
)
def test_equivalence_confidence_contract(confidence, resolution):
    from pydantic import ValidationError

    def complete(messages, _config):
        assert "confidence must be a number from 0 to 1" in messages[1]["content"]
        return json.dumps(
            {
                "verdict": "equivalent",
                "confidence": confidence,
                "rationale": "Mock comparison",
                "assumptions_used": [],
            }
        )

    judge = ExpressionEquivalenceJudge(
        config=EquivalenceJudgeConfig(model="mock"), completion=complete
    )
    kwargs = {
        "call": GradingCall(primitive="exact", target="answer", expected="2"),
        "actual": "1+1",
        "deterministic_grade": PrimitiveGrade(
            primitive="exact", target="answer", passed=False, score=0.0
        ),
    }
    if resolution is None:
        with pytest.raises(ValidationError):
            judge.judge(**kwargs)
    else:
        assert judge.judge(**kwargs).resolution == resolution


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


@pytest.mark.parametrize("kind", ["semantic", "equivalence"])
@pytest.mark.parametrize("structured", [False, True])
def test_shared_completion_contract(monkeypatch, kind, structured):
    from ddpr.rewards.physics.llm import equivalence, semantic

    def forbidden(*args, **kwargs):
        pytest.fail("Injected completion must bypass the default transport")

    monkeypatch.setattr(semantic, "complete_judge", forbidden)
    monkeypatch.setattr(equivalence, "complete_judge", forbidden)
    calls = []
    payload = (
        correct()
        if kind == "semantic"
        else json.dumps(
            {
                "verdict": "equivalent",
                "confidence": 0.99,
                "rationale": "Mock equivalence",
                "assumptions_used": [],
            }
        )
    )

    def complete(messages, config):
        calls.append((messages, config))
        if not structured:
            return payload
        return CompletionResult(
            content=payload,
            model="served-model",
            response_id="request-1",
            usage={"total_tokens": 12},
        )

    if kind == "semantic":
        judge = LLMSemanticJudge(
            config=SemanticJudgeConfig(model="configured-model"), completion=complete
        )
        result = judge.judge(
            FreestyleProblemGradingInput.model_validate(problem()), "E=2"
        )
        assert result.verdict == "correct"
    else:
        judge = ExpressionEquivalenceJudge(
            config=EquivalenceJudgeConfig(model="configured-model"), completion=complete
        )
        result = judge.judge(
            call=GradingCall(primitive="exact", target="answer", expected="2"),
            actual="1+1",
            deterministic_grade=PrimitiveGrade(
                primitive="exact", target="answer", passed=False, score=0.0
            ),
        )
        assert result.resolution == "equivalent"
    assert len(calls) == 1
    assert calls[0][1] is judge.config
    assert [message["role"] for message in calls[0][0]] == ["system", "user"]
    assert result.model == ("served-model" if structured else "configured-model")
    assert result.response_id == ("request-1" if structured else None)
    assert result.usage == ({"total_tokens": 12} if structured else None)


def test_semantic_schema_repair_is_bounded():
    responses = iter(["invalid", correct()])
    prompts = []

    def complete(messages, _config):
        prompts.append(messages)
        return next(responses)

    judge = LLMSemanticJudge(
        config=SemanticJudgeConfig(model="mock"), completion=complete
    )
    assert (
        grade_response(problem(), "E=2", semantic_judge=judge).grading_status == "pass"
    )
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
    assert "reward" not in report.model_dump()
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

    monkeypatch.setattr(client.OpenAIChatBackend, "complete", complete)
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
    assert report.grading_status == "fail"
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
    assert (
        grade_response(problem(), "E=2", semantic_judge=judge).grading_status == "pass"
    )
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
