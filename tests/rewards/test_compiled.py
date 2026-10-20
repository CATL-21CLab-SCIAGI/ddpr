"""Opt-in checks with private, compiled DDSR-Bench CritPt-AI tasks."""

import ast
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("sympy")
pytest.importorskip("ddsr_bench.benchmarks.critpt.evaluation.ai")

from ddsr_bench.benchmarks.critpt.data.loader import load_ai_task
from ddsr_bench.benchmarks.critpt.evaluation.ai.grader import parse_problem
from ddsr_bench.benchmarks.critpt.evaluation.ai.llm.semantic import (
    LLMSemanticJudge,
    SemanticJudgeConfig,
)
from ddsr_bench.benchmarks.critpt.evaluation.ai.rule.compare import (
    grade_with_plan,
)

from ddpr.rewards.critpt_ai import CritptAiReward, CritptAiRewardError, grade_response


def _tasks():
    root = os.getenv("DDPR_TEST_CRITPT_TASKS")
    if root is None:
        return [pytest.param(None, marks=pytest.mark.skip(reason="No compiled tasks"))]
    tasks = sorted(path.parent for path in Path(root).rglob("instruction.md"))
    if not tasks:
        raise ValueError("DDPR_TEST_CRITPT_TASKS contains no compiled tasks")
    selected = [
        task
        for task in tasks
        if json.loads((task / "tests/reference.json").read_text(encoding="utf-8")).get(
            "reference_check"
        )
        in {"matched", "unresolved", "different"}
    ]
    if not selected:
        raise ValueError(
            "No compiled tasks have matched/unresolved/different reference_check"
        )
    return selected


@pytest.mark.parametrize("task", _tasks())
def test_compiled_literal_reference(task):
    problem = load_ai_task(task)
    if problem.spec.grader != "rule":
        pytest.skip("Freestyle references require semantic assessment")
    reference = json.loads((task / "tests/reference.json").read_text(encoding="utf-8"))
    # Read a return literal; never import or invoke the saved answer function.
    functions = [
        node
        for node in ast.parse(reference["code"]).body
        if isinstance(node, ast.FunctionDef) and node.name == "answer"
    ]
    assert len(functions) == 1
    returned = functions[0].body[-1]
    if not isinstance(returned, ast.Return):
        pytest.skip("Reference requires assessment beyond a return literal")
    try:
        actual = ast.literal_eval(returned.value)
    except (ValueError, TypeError):
        pytest.skip("Nonliteral reference: separate static LLM assessment required")
    result = grade_with_plan({"answer": actual}, parse_problem(problem).grading_plan)
    assert result.passed, result.model_dump()
    # A missing target must not inherit the reference's passing result.
    assert not grade_with_plan({}, parse_problem(problem).grading_plan).passed


@pytest.mark.parametrize("task", _tasks())
def test_compiled_problem(task):
    problem = load_ai_task(task)
    from ddpr.rewards.critpt_ai import load_problem as load_reward_problem

    metadata = {"task_path": str(task.resolve())}
    assert load_reward_problem(metadata) == problem
    if problem.spec.grader == "rule":
        # Check real contracts without executing downloaded answer code.
        report = grade_response(problem, "", llm_fallback=False)
        assert report.extraction_status == "missing_answer_function"
        assert (
            CritptAiReward(llm_fallback=False)(
                response="",
                sample_id=task.name,
                reference_completion="UNVERIFIED_TEACHER",
                metadata=metadata,
            )
            == 0.0
        )
        return

    for verdict, expected in [
        ("correct", 1.0),
        ("incorrect", 0.0),
        ("uncertain", None),
    ]:
        prompts = []

        def complete(messages, _config, *, verdict=verdict, prompts=prompts):
            prompts.append(messages)
            return json.dumps(
                {
                    "verdict": verdict,
                    "confidence": 0.99,
                    "solver_answer_summary": "Synthetic test response",
                    "assessment": "Mocked judgment; no correctness claim",
                    "substantive_errors": [],
                }
            )

        scorer = CritptAiReward(
            semantic_judge=LLMSemanticJudge(
                config=SemanticJudgeConfig(model="mock"), completion=complete
            )
        )
        kwargs = {
            "response": "Synthetic test response",
            "sample_id": task.name,
            "reference_completion": "UNVERIFIED_TEACHER",
            "metadata": metadata,
        }
        if expected is None:
            with pytest.raises(CritptAiRewardError):
                scorer(**kwargs)
        else:
            assert scorer(**kwargs) == expected
        payload = json.loads(prompts[0][1]["content"])
        assert (
            payload["private_reference"]["reference_answer"] == problem.answer.snippet
        )
        assert "UNVERIFIED_TEACHER" not in json.dumps(prompts)
