import asyncio
import importlib.util
import json
import pickle
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("sympy")

from ddpr.rewards import load_reward
from ddpr.rewards.physics import (
    PhysicsReward,
    PhysicsRewardError,
    grade_response,
    load_problem,
)
from ddpr.rewards.physics._core import evaluator, primitives
from ddpr.rewards.physics._core.grading_runtime import GraderTimeout
from ddpr.rewards.physics.equivalence import (
    EquivalenceJudgeConfig,
    ExpressionEquivalenceJudge,
)
from ddpr.rewards.physics.semantic import LLMSemanticJudge, SemanticJudgeConfig


def auto_problem(kind="exact", expected="sqrt(3)/2", args=None):
    call = {"primitive": kind, "target": "answer", "expected": expected}
    if args is not None:
        call["args"] = args
    if kind == "floating_numeric":
        call["precision"] = {"kind": "decimal_places", "digits": 4}
    return {
        "problem_id": "canonical-problem",
        "problem_statement": "Compute the requested value, with x positive and real.",
        "public_solver_output_instructions": "Return a fenced Python answer() function.",
        "answer_contract": {"answer_type": kind},
        "grading_plan": {"calls": [call]},
        "reference_solution": "PRIVATE_DERIVATION_NOT_NEEDED_BY_AUTO_JUDGE",
    }


def freestyle_problem():
    return {
        "problem_id": "freestyle-problem",
        "problem_statement": "Give the energy at x=1 and whether the state is stable.",
        "public_solver_output_instructions": "Give both results and explain briefly.",
        "answer_type": "exact",
        "answer_items": [
            {"item_id": "energy", "answer_type": "exact", "target": "Energy at x=1"},
            {"item_id": "state", "answer_type": "categorical", "target": "Stability"},
        ],
        "reference_answer": "E=2; stable.",
        "reference_solution": "Evaluating E=1+x*x gives 2; its minimum is stable.",
        "blueprint": {"private": "UNNEEDED_BLUEPRINT"},
    }


def block(expression):
    return (
        "```python\nimport sympy as sp\nx = sp.Symbol('x', positive=True)\n"
        f"def answer():\n    return {expression}\n```"
    )


def forbidden_judge():
    def complete(_messages):
        pytest.fail("This answer must not call the equivalence Judge")

    return ExpressionEquivalenceJudge(
        config=EquivalenceJudgeConfig(model="test"), completion=complete
    )


def equivalence(verdict="equivalent", prompts=None):
    def complete(messages):
        if prompts is not None:
            prompts.append(messages)
        return json.dumps(
            {
                "verdict": verdict,
                "confidence": 0.99,
                "rationale": "Mock comparison, without solving the original problem.",
                "assumptions_used": [],
            }
        )

    return ExpressionEquivalenceJudge(
        config=EquivalenceJudgeConfig(model="test"), completion=complete
    )


def semantic(verdict="correct", prompts=None):
    def complete(messages, _config):
        if prompts is not None:
            prompts.append(messages)
        return json.dumps(
            {
                "verdict": verdict,
                "confidence": 0.99,
                "solver_answer_summary": "Both requested items were compared.",
                "assessment": "Mock semantic comparison.",
                "substantive_errors": [],
            }
        )

    return LLMSemanticJudge(
        config=SemanticJudgeConfig(model="test"), completion=complete
    )


def score(scorer, problem, response, teacher="UNVERIFIED_TEACHER_SENTINEL"):
    return scorer(
        response=response,
        sample_id="training-sample:1",
        reference_completion=teacher,
        metadata={"problem": problem},
    )


@pytest.mark.parametrize(
    "kind,expected,args,expression,passed",
    [
        ("floating_numeric", 1.25, {"abs_tol": 1e-4}, "1.25001", True),
        ("floating_numeric", 1.25, {"abs_tol": 1e-4}, "1.251", False),
        ("exact", "sqrt(3)/2", {}, "sp.sqrt(12)/4", True),
        ("exact", "sqrt(3)/2", {}, "sp.sqrt(2)", False),
        ("symbolic", "(x+1)**2", {"variables": ["x"]}, "x*x+2*x+1", True),
        ("symbolic", "(x+1)**2", {"variables": ["x"]}, "x*x+1", False),
        ("categorical", ["A", "C"], {"choices": ["A", "B", "C"]}, "{'C', 'A'}", True),
        ("categorical", ["A", "C"], {"choices": ["A", "B", "C"]}, "{'A'}", False),
    ],
)
def test_four_auto_types(kind, expected, args, expression, passed):
    problem = auto_problem(kind, expected, args)
    scorer = PhysicsReward(judge_enabled=False)
    assert score(scorer, problem, block(expression)) == float(passed)


@pytest.mark.parametrize(
    "expression,expected",
    [("sp.acosh(2)", "log(2+sqrt(3))"), ("sp.root(8, 3)", "2")],
)
def test_expanded_vocabulary_passes_without_judge(expression, expected):
    report = grade_response(
        auto_problem(expected=expected),
        block(expression),
        equivalence_judge=forbidden_judge(),
    )
    assert report.reward == 1
    assert not report.equivalence_judgments


def test_variable_assumptions_and_categorical_aliases():
    problem = auto_problem(
        "symbolic",
        "x",
        {"variables": ["x"], "variable_assumptions": {"x": {"positive": True}}},
    )
    assert (
        grade_response(problem, block("sp.sqrt(x*x)"), judge_enabled=False).reward == 1
    )
    problem = auto_problem(
        "categorical",
        ["A"],
        {"choices": ["A", "B"], "allowed_aliases": {"alpha": "A"}},
    )
    assert grade_response(problem, block("{'alpha'}"), judge_enabled=False).reward == 1


@pytest.mark.parametrize(
    "response,status",
    [
        ("The answer is 1", "missing_answer_function"),
        ("```python\ndef answer(:\n    return 1\n```", "parse_failure"),
        ("```python\nimport os\ndef answer():\n    return 1\n```", "unsafe_code"),
        (block("1/0"), "runtime_failure"),
        (block("{'answer': [1]}"), "invalid_answer_format"),
    ],
)
def test_invalid_student_response_is_zero(response, status):
    report = grade_response(
        auto_problem(), response, equivalence_judge=forbidden_judge()
    )
    assert report.reward == 0
    assert report.extraction_status == status


def test_floating_expression_is_not_silently_rounded():
    report = grade_response(
        auto_problem("floating_numeric", 1.25, {"abs_tol": 0.01}),
        block("sp.Rational(5,4)"),
        equivalence_judge=forbidden_judge(),
    )
    assert report.reward == 0
    assert report.extraction_status == "invalid_answer_format"


def test_unsupported_sympy_is_inert_but_can_reach_judge(monkeypatch):
    def forbidden(*args):
        pytest.fail("Unsupported code was executed")

    monkeypatch.setattr(primitives, "_execute_answer_code", forbidden)
    prompts = []
    scorer = PhysicsReward(equivalence_judge=equivalence(prompts=prompts))
    problem = auto_problem("symbolic", "fresnels(x)", {"variables": ["x"]})
    assert score(scorer, problem, block("sp.fresnels(x)")) == 1
    rendered = json.dumps(prompts)
    assert problem["problem_statement"] in rendered
    assert "fresnels(x)" in rendered
    assert "UNVERIFIED_TEACHER_SENTINEL" not in rendered
    assert "PRIVATE_DERIVATION_NOT_NEEDED_BY_AUTO_JUDGE" not in rendered
    assert "do not solve the original task" in rendered


@pytest.mark.parametrize("failure", ["uncertain", "disabled", "api", "timeout"])
def test_unresolved_is_exception_not_zero(monkeypatch, failure):
    if failure == "timeout":

        def timeout(*args, **kwargs):
            raise GraderTimeout

        monkeypatch.setattr(evaluator, "run_with_deadline", timeout)
    judge = equivalence("uncertain")
    if failure == "api":

        def unavailable(_messages):
            raise TimeoutError("mock provider timeout")

        judge = ExpressionEquivalenceJudge(
            config=EquivalenceJudgeConfig(model="test"),
            completion=unavailable,
        )
    scorer = PhysicsReward(judge_enabled=failure != "disabled", equivalence_judge=judge)
    problem = auto_problem("symbolic", "x", {"variables": ["x"]})
    with pytest.raises(PhysicsRewardError) as raised:
        score(scorer, problem, block("sp.fresnels(x)"))
    error = raised.value
    assert error.report.reward is None
    assert error.report.grading_status in {"grader_error", "grader_uncertain"}
    restored = pickle.loads(pickle.dumps(error))
    assert restored.sample_id == "training-sample:1"
    assert restored.report == error.report


def test_exact_numeric_conflict_is_unresolved():
    with pytest.raises(PhysicsRewardError) as raised:
        score(
            PhysicsReward(equivalence_judge=equivalence("equivalent")),
            auto_problem(expected="2"),
            block("sp.Integer(3)"),
        )
    assert raised.value.report.exact_numeric_comparisons[0].status == "different"
    assert raised.value.report.grading_status == "grader_uncertain"


@pytest.mark.parametrize("verdict,expected", [("correct", 1.0), ("incorrect", 0.0)])
def test_freestyle_multiple_items_and_teacher_ignored(verdict, expected):
    prompts = []
    scorer = PhysicsReward(
        judge_enabled=False, semantic_judge=semantic(verdict, prompts)
    )
    assert score(scorer, freestyle_problem(), "E=2; stable.") == expected
    payload = json.loads(prompts[0][1]["content"])
    assert len(payload["answer_items"]) == 2
    assert payload["private_reference"]["reference_answer"] == "E=2; stable."
    assert "UNVERIFIED_TEACHER_SENTINEL" not in json.dumps(prompts)
    assert "UNNEEDED_BLUEPRINT" not in json.dumps(prompts)


def test_freestyle_uncertain_and_empty_response():
    scorer = PhysicsReward(semantic_judge=semantic("uncertain"))
    with pytest.raises(PhysicsRewardError):
        score(scorer, freestyle_problem(), "Maybe stable")
    assert score(scorer, freestyle_problem(), "") == 0


def test_reference_completion_cannot_override_ground_truth():
    scorer = PhysicsReward(judge_enabled=False)
    problem = auto_problem(expected="1")
    assert score(scorer, problem, block("1"), teacher=block("2")) == 1
    assert score(scorer, problem, block("2"), teacher=block("2")) == 0
    with pytest.raises(ValueError, match="exactly one"):
        scorer(
            response=block("1"),
            sample_id=None,
            reference_completion=block("1"),
            metadata={},
        )


def test_inline_and_file_inputs(tmp_path):
    problem = auto_problem()
    path = tmp_path / "problem.json"
    path.write_text(json.dumps(problem))
    assert load_problem({"problem": problem}) == load_problem(
        {"problem_path": str(path)}
    )
    legacy = freestyle_problem()
    del legacy["answer_items"]
    assert len(load_problem({"problem": legacy}).answer_items) == 1


@pytest.mark.parametrize(
    "metadata,error",
    [
        ({}, ValueError),
        ({"problem": {}, "problem_path": "/x"}, ValueError),
        ({"problem_path": "relative.json"}, ValueError),
        ({"problem": []}, TypeError),
        ({"problem_path": 42}, TypeError),
        ([], TypeError),
    ],
)
def test_bad_metadata_is_not_a_training_score(metadata, error):
    with pytest.raises(error):
        load_problem(metadata)


def test_bad_or_composite_contract_is_not_a_training_score():
    problem = auto_problem()
    problem["answer_contract"]["answer_type"] = "composite"
    with pytest.raises(ValueError, match="not supported"):
        grade_response(problem, block("1"))
    problem = auto_problem()
    problem["grading_plan"]["calls"] *= 2
    with pytest.raises(ValueError, match="not supported"):
        grade_response(problem, block("1"))
    problem = auto_problem()
    problem["answer_contract"]["answer_type"] = "symbolic"
    with pytest.raises(ValueError, match="must match"):
        grade_response(problem, block("1"))


def test_zero_argument_reward_loads_with_environment(monkeypatch):
    monkeypatch.setenv("DDPR_PHYSICS_EQUIVALENCE_JUDGE", "false")
    load_reward.cache_clear()
    assert not load_reward("ddpr.rewards.physics.PhysicsReward").judge_enabled
    monkeypatch.setenv("DDPR_PHYSICS_EQUIVALENCE_JUDGE", "invalid")
    with pytest.raises(ValueError, match="must be true or false"):
        PhysicsReward()
    load_reward.cache_clear()


def test_core_reward_import_does_not_require_physics_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "from ddpr.rewards import SmokeReward; print(SmokeReward)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "SmokeReward" in result.stdout


def test_slime_reward_calls_real_grader_from_worker_thread(monkeypatch):
    from ddpr.backends.slime import reward

    monkeypatch.setattr(
        reward, "load_reward", lambda _path: PhysicsReward(judge_enabled=False)
    )
    problem = auto_problem(expected="1")
    samples = [
        SimpleNamespace(
            response=block(value),
            sample_id=value,
            reference_completion="IGNORED",
            metadata={"problem": problem},
        )
        for value in ("1", "2")
    ]
    assert asyncio.run(reward.reward(SimpleNamespace(ddpr_reward="test"), samples)) == [
        1.0,
        0.0,
    ]


def test_five_named_comparison_modules():
    from ddpr.rewards.physics.categorical import grade_categorical
    from ddpr.rewards.physics.exact import grade_exact
    from ddpr.rewards.physics.floating import grade_floating_numeric
    from ddpr.rewards.physics.symbolic import grade_symbolic

    assert grade_exact("sqrt(4)", "2").passed
    assert grade_symbolic("x+x", "2*x", {"variables": ["x"]}).passed
    assert grade_floating_numeric(1.0, 1.0, {"abs_tol": 0.01}).passed
    assert grade_categorical({"A", "B"}, ["B", "A"]).passed


def test_swift_adapter_with_mocked_framework(monkeypatch):
    framework = ModuleType("swift.rewards")
    framework.ORM = object
    framework.orms = {}
    monkeypatch.setitem(sys.modules, "swift.rewards", framework)
    source = Path(__file__).parents[2] / "ddpr/backends/swift/reward.py"
    spec = importlib.util.spec_from_file_location("physics_swift_adapter_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("DDPR_REWARD", "ddpr.rewards.physics.PhysicsReward")
    monkeypatch.setattr(
        module, "load_reward", lambda _path: PhysicsReward(judge_enabled=False)
    )
    adapter = module.RewardAdapter()
    assert adapter(
        completions=[block("1"), block("2")],
        sample_id=["first", "second"],
        reference_completion=[block("2"), block("2")],
        metadata=[{"problem": auto_problem(expected="1")}] * 2,
    ) == [1.0, 0.0]
