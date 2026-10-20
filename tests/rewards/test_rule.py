"""Rules consume extracted values; execution returns serializable answers."""

from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("sympy")

from ddpr.rewards.critpt_ai import answer
from ddpr.rewards.critpt_ai.answer import extract_answer_locally
from ddpr.rewards.critpt_ai.rule.compare import grade_with_plan
from ddpr.rewards.critpt_ai.rule.symbolic import grade_symbolic
from ddpr.rewards.critpt_ai.runtime import (
    run_with_deadline,
)
from ddpr.rewards.critpt_ai.schemas import GradingPlan, UnexecutedAnswer


@pytest.mark.parametrize(
    "expression,passed",
    [
        ("x/10**20", True),
        ("(x-2/7)*(x-5/11)*(x-11/17)", True),
        ("floor(x)", True),
    ],
)
def test_identical_symbolic_answers(expression, passed):
    result = grade_symbolic(expression, expression, variables=["x"])
    assert result.passed is passed


def test_constant_answer_with_valid_symbolic_reference():
    assert not grade_symbolic("sin(x)**2+cos(x)**2", "x", variables=["x"]).passed


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

    monkeypatch.setattr(answer, "_execute_answer_code", forbidden)
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
            timeout=10,
        ).result(timeout=15)
    value = extracted["answer"]
    assert isinstance(value, UnexecutedAnswer)
    assert value.expression == "sp.fresnels(x)"
    result = grade_with_plan(extracted, plan)
    assert not result.passed
    assert result.calls[0].equivalence_candidate
    assert result.calls[0].deterministic_skip_reason == value.reason


@pytest.mark.parametrize("actual", ["x", "symbol"])
def test_submitted_assumptions_cannot_change_reference(actual):
    import sympy as sp

    value = sp.Symbol("x", positive=True) if actual == "symbol" else actual
    assert not grade_symbolic(value, "sqrt(x**2)").passed
    assert grade_symbolic(
        value,
        "sqrt(x**2)",
        variables=["x"],
        variable_assumptions={"x": {"positive": True}},
    ).passed


@pytest.mark.parametrize("reverse", [False, True])
def test_conflicting_aliases_are_rejected(reverse):
    from ddpr.rewards.critpt_ai.rule.categorical import grade_categorical

    items = [("A", ["B"]), ("B", [])]
    aliases = dict(reversed(items) if reverse else items)
    with pytest.raises(ValueError, match="conflicting"):
        grade_categorical("A", "B", allowed_aliases=aliases)


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf"), 0])
def test_invalid_plan_weights(weight):
    with pytest.raises(ValueError):
        GradingPlan(
            calls=[
                {
                    "primitive": "exact",
                    "target": "answer",
                    "expected": "2",
                    "weight": weight,
                }
            ]
        )


def test_valid_weighted_comparison():
    plan = GradingPlan(
        composition="weighted",
        calls=[
            {
                "primitive": "exact",
                "target": "a",
                "expected": "2",
                "weight": 3,
                "required": False,
            },
            {
                "primitive": "exact",
                "target": "b",
                "expected": "2",
                "weight": 1,
                "required": False,
            },
        ],
    )
    result = grade_with_plan({"a": 2, "b": 3}, plan)
    assert result.passed and result.score == 0.75


def test_piecewise_boolean_conditions():
    result = grade_symbolic(
        answer="Piecewise((x, (x-y > 0) & (x+y < 1)), (2*x, True))",
        target="Piecewise((x, (2*x > 2*y) & (2*x+2*y < 2)), (2*x, True))",
        variables=["x", "y"],
        variable_assumptions={"x": {"real": True}, "y": {"real": True}},
    )
    assert result.passed
