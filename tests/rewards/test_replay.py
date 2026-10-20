"""Opt-in saved-judgment replay; no network or corpus answer execution."""

import ast
import hashlib
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("ddsr_bench")
pytest.importorskip("pydantic")
pytest.importorskip("sympy")

import httpx
from ddsr_bench.generation.client import CLIENTS, VLLMClient

from ddpr.rewards.critpt_ai import (
    CritptAiReward,
    CritptAiRewardError,
    answer,
    grade_response,
)
from ddpr.rewards.critpt_ai.critpt_ai.inputs import load_problem
from ddpr.rewards.critpt_ai.llm.equivalence import (
    EquivalenceJudgeConfig,
    LLMEquivalenceJudge,
)
from ddpr.rewards.critpt_ai.llm.semantic import LLMSemanticJudge, SemanticJudgeConfig
from ddpr.rewards.critpt_ai.rule.compare import grade_with_plan
from ddpr.rewards.critpt_ai.schemas import UnexecutedAnswer


@pytest.fixture
def replay(monkeypatch):
    root = os.getenv("DDPR_TEST_CRITPT_AI_REPLAY")
    tasks = os.getenv("DDPR_TEST_CRITPT_TASKS")
    if not root or not tasks:
        pytest.skip("Set DDPR_TEST_CRITPT_AI_REPLAY and DDPR_TEST_CRITPT_TASKS")

    def forbidden(*args, **kwargs):
        pytest.fail("Network and corpus execution are forbidden during replay")

    monkeypatch.setattr(answer, "_execute_answer_code", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    return Path(root), Path(tasks)


def _transport(monkeypatch, saved):
    pending = iter(saved)
    consumed = []

    def respond(request):
        expected = next(pending)
        assert json.loads(request.content)["messages"] == expected["messages"]
        result = expected["completion"]
        consumed.append(expected)
        return httpx.Response(
            200,
            json={
                "id": result["response_id"],
                "model": result["model"],
                "choices": [
                    {
                        "message": {
                            "content": result["content"],
                            "reasoning_content": result["reasoning_content"],
                        },
                        "finish_reason": result["finish_reason"],
                    }
                ],
                "usage": result["usage"],
            },
        )

    monkeypatch.setitem(
        CLIENTS,
        "vllm",
        lambda **kwargs: VLLMClient(**kwargs, transport=httpx.MockTransport(respond)),
    )
    return consumed


def test_rule_reference_replay(replay, monkeypatch):
    root, tasks = replay
    originals = sorted(root.glob("ai-*.json"))
    originals = [p for p in originals if not p.name.endswith(".completion.json")]
    assert originals, "No saved rule assessments"
    for path in originals:
        saved = json.loads(path.read_text())
        task = tasks / "rule" / saved["task"]
        reference_path = task / "tests/reference.json"
        assert (
            hashlib.sha256(reference_path.read_bytes()).hexdigest()
            == saved["reference_sha256"]
        )
        reference = json.loads(reference_path.read_text())
        assert reference["reference_check"] == saved["reference_check"]
        problem = load_problem(task)
        fn = next(
            n
            for n in ast.parse(reference["code"]).body
            if isinstance(n, ast.FunctionDef) and n.name == "answer"
        )
        returned = fn.body[-1]
        assert isinstance(returned, ast.Return)
        try:
            value = ast.literal_eval(returned.value)
        except (ValueError, TypeError):
            value = UnexecutedAnswer(
                expression=ast.unparse(returned.value),
                answer_code=reference["code"],
                reason="Static audit: Python answer function was not executed",
            )
        result = grade_with_plan({"answer": value}, problem.grading_plan)
        if saved["method"] == "rules":
            assert result.passed
            assert not grade_with_plan({}, problem.grading_plan).passed
            continue
        completion = json.loads(path.with_suffix(".completion.json").read_text())
        consumed = _transport(
            monkeypatch, [{"messages": saved["prompts"][0], "completion": completion}]
        )
        judgment = LLMEquivalenceJudge(
            config=EquivalenceJudgeConfig(
                backend="ddsr",
                provider="vllm",
                model="replay",
                api_key_env=None,
            )
        ).judge(
            call=problem.grading_plan.calls[0],
            answer=value,
            deterministic_grade=result.calls[0],
            public_problem=problem.model_dump(),
        )
        assert judgment.resolution == saved["resolution"] == "equivalent"
        assert len(consumed) == 1


@pytest.mark.parametrize("case", ["reference", "wrong", "abstention"])
def test_semantic_reward_replay(replay, monkeypatch, case):
    root, tasks = replay
    summary = json.loads((root / "semantic-summary.json").read_text())
    expected = next(row for row in summary["results"] if row["case"] == case)
    task = tasks / "llm" / summary["task"]
    problem = load_problem(task)
    response_path = root / f"{case}.response.txt"
    assert (
        hashlib.sha256(response_path.read_bytes()).hexdigest()
        == expected["response_sha256"]
    )
    response = response_path.read_text()
    saved = json.loads((root / f"{case}.completions.json").read_text())
    judge = LLMSemanticJudge(
        config=SemanticJudgeConfig(
            backend="ddsr",
            provider="vllm",
            model="replay",
            api_key_env=None,
        )
    )
    consumed = _transport(monkeypatch, saved * 2)
    report = grade_response(problem, response, semantic_judge=judge, attempt_id=case)
    score = CritptAiReward(semantic_judge=judge)(
        response=response,
        sample_id=case,
        reference_completion="UNVERIFIED_TEACHER",
        metadata={"task_path": str(task.resolve())},
    )
    assert report.grading_status == expected["status"]
    assert report.semantic_judgment.attempt_id == case
    assert report.problem_id == problem.problem_id
    assert report.response_sha256 == expected["response_sha256"]
    assert score == expected["reward"]
    assert len(consumed) == len(saved) * 2

    # An unavailable judge must remain unresolved, never become a wrong-answer reward.
    def unavailable(request):
        return httpx.Response(503, json={"error": "Synthetic outage"})

    monkeypatch.setitem(
        CLIENTS,
        "vllm",
        lambda **kwargs: VLLMClient(
            **kwargs, transport=httpx.MockTransport(unavailable)
        ),
    )
    with pytest.raises(CritptAiRewardError) as failure:
        CritptAiReward(semantic_judge=judge)(
            response=response,
            sample_id=case,
            reference_completion="UNVERIFIED_TEACHER",
            metadata={"task_path": str(task.resolve())},
        )
    assert failure.value.sample_id == case
