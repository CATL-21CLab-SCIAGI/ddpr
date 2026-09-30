"""Opt-in checks with private, compiled DDSR-Bench CritPt-AI tasks."""

import ast
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("sympy")

from ddpr.rewards.physics import PhysicsReward, PhysicsRewardError, grade_response
from ddpr.rewards.physics.critpt_ai.inputs import load_problem
from ddpr.rewards.physics.llm.semantic import LLMSemanticJudge, SemanticJudgeConfig
from ddpr.rewards.physics.rule.compare import grade_with_plan


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
    problem = load_problem(task)
    if problem.problem_format != "auto":
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
    result = grade_with_plan({"answer": actual}, problem.grading_plan)
    assert result.passed, result.model_dump()
    # A missing target must not inherit the reference's passing result.
    assert not grade_with_plan({}, problem.grading_plan).passed


@pytest.mark.parametrize("task", _tasks())
def test_compiled_problem(task):
    problem = load_problem(task)
    from ddpr.rewards.physics import load_problem as load_reward_problem

    metadata = {"task_path": str(task.resolve())}
    assert load_reward_problem(metadata) == problem
    if problem.problem_format == "auto":
        # Check real contracts without executing downloaded answer code.
        report = grade_response(problem, "", judge_enabled=False)
        assert report.extraction_status == "missing_answer_function"
        assert (
            PhysicsReward(judge_enabled=False)(
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

        scorer = PhysicsReward(
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
            with pytest.raises(PhysicsRewardError):
                scorer(**kwargs)
        else:
            assert scorer(**kwargs) == expected
        payload = json.loads(prompts[0][1]["content"])
        assert (
            payload["private_reference"]["reference_answer"] == problem.reference_answer
        )
        assert "UNVERIFIED_TEACHER" not in json.dumps(prompts)
