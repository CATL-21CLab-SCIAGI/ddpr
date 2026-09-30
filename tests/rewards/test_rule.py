"""Rules consume extracted values; execution returns serializable answers."""

from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("sympy")

from ddpr.rewards.physics.execution import answers
from ddpr.rewards.physics.execution.runtime import (
    extract_answer_locally,
    run_with_deadline,
)
from ddpr.rewards.physics.rule.plan import grade_with_plan
from ddpr.rewards.physics.schemas import GradingPlan, UnexecutedAnswer


@pytest.mark.parametrize(
    "kind,actual,expected,args,passed",
    [
        ("floating_numeric", 1.25, 1.25, {"abs_tol": 0.001}, True),
        ("floating_numeric", 1.3, 1.25, {"abs_tol": 0.001}, False),
        ("exact", "sqrt(12)/4", "sqrt(3)/2", {}, True),
        ("exact", "sqrt(2)", "sqrt(3)/2", {}, False),
        ("symbolic", "x+x", "2*x", {"variables": ["x"]}, True),
        ("symbolic", "x+1", "2*x", {"variables": ["x"]}, False),
        ("categorical", {"A", "B"}, ["B", "A"], {}, True),
        ("categorical", {"A"}, ["B", "A"], {}, False),
    ],
)
def test_extracted_values_do_not_execute_answer_code(
    monkeypatch, kind, actual, expected, args, passed
):
    def forbidden(*args, **kwargs):
        pytest.fail("Comparing extracted values must not execute an answer function")

    monkeypatch.setattr(answers, "_execute_answer_code", forbidden)
    call = {"primitive": kind, "target": "answer", "expected": expected, "args": args}
    if kind == "floating_numeric":
        call["precision"] = {"kind": "decimal_places", "digits": 3}
    plan = GradingPlan(calls=[call])
    result = grade_with_plan({"answer": actual}, plan)
    assert result.passed is passed
    assert result.score == float(passed)
    assert result.calls[0].target == "answer"


def test_inert_answer_survives_worker_transfer():
    # Trusted fixture, not a downloaded corpus answer.
    response = (
        "```python\nimport sympy as sp\ndef answer(x):\n    return sp.fresnels(x)\n```"
    )
    plan = GradingPlan(
        calls=[
            {
                "primitive": "symbolic",
                "target": "answer",
                "expected": "fresnels(x)",
                "args": {"variables": ["x"]},
            }
        ]
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        extracted = pool.submit(
            run_with_deadline,
            extract_answer_locally,
            response,
            plan,
            timeout_seconds=10,
        ).result(timeout=15)
    value = extracted["answer"]
    assert isinstance(value, UnexecutedAnswer)
    assert value.expression == "sp.fresnels(x)"
    result = grade_with_plan(extracted, plan)
    assert not result.passed
    assert result.calls[0].equivalence_candidate
    assert result.calls[0].deterministic_skip_reason == value.reason
