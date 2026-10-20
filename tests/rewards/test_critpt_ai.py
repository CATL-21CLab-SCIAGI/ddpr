import asyncio
import builtins
import importlib.util
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("sympy")
pytest.importorskip("ddsr_bench.benchmarks.critpt.evaluation.ai")

from ddsr_bench.benchmarks.critpt.evaluation.ai import answer, grader
from ddsr_bench.benchmarks.critpt.evaluation.ai.io.schemas import (
    GradeReport,
)
from ddsr_bench.benchmarks.critpt.evaluation.ai.llm.equivalence import (
    EquivalenceJudgeConfig,
    LLMEquivalenceJudge,
)
from ddsr_bench.benchmarks.critpt.evaluation.ai.llm.semantic import (
    LLMSemanticJudge,
    SemanticJudgeConfig,
)
from ddsr_bench.benchmarks.critpt.evaluation.ai.runtime import GraderTimeout

from ddpr.rewards import load_reward
from ddpr.rewards.critpt_ai import (
    CritptAiReward,
    CritptAiRewardError,
    grade_response,
    load_problem,
)


@pytest.mark.parametrize(
    "status,extraction,expected",
    [
        ("pass", "success", 1.0),
        ("fail", "success", 0.0),
        ("resource_limit", "success", None),
        ("not_graded", "missing_answer_function", 0.0),
        ("not_graded", "parse_failure", 0.0),
        ("not_graded", "invalid_answer_format", 0.0),
        ("not_graded", "unsafe_code", 0.0),
        ("not_graded", "runtime_failure", 0.0),
        ("not_graded", "not_attempted", None),
        ("grader_error", "success", None),
        ("grader_uncertain", "success", None),
        ("infrastructure_error", "not_attempted", None),
    ],
)
def test_reward_mapping(monkeypatch, status, extraction, expected):
    from ddpr.rewards.critpt_ai import reward

    report = GradeReport(
        problem_id="test",
        answer_type="exact",
        raw_response="",
        response_sha256="",
        extraction_status=extraction,
        grading_status=status,
        passed=status == "pass",
    )
    monkeypatch.setattr(reward, "grade_response", lambda *args, **kwargs: report)
    assert "reward" not in report.model_dump()
    if expected is None:
        with pytest.raises(CritptAiRewardError) as raised:
            score(CritptAiReward(), auto_problem(), "")
        assert raised.value.report is report
    else:
        assert score(CritptAiReward(), auto_problem(), "") == expected


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
    def complete(_messages, _config):
        pytest.fail("This answer must not call the equivalence Judge")

    return LLMEquivalenceJudge(
        config=EquivalenceJudgeConfig(model="test"), completion=complete
    )


def equivalence(verdict="equivalent", prompts=None):
    def complete(messages, _config):
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

    return LLMEquivalenceJudge(
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
    scorer = CritptAiReward(judge_enabled=False)
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
    assert report.grading_status == "pass"
    assert not report.equivalence_judgments


def test_variable_assumptions_and_categorical_aliases():
    problem = auto_problem(
        "symbolic",
        "x",
        {"variables": ["x"], "variable_assumptions": {"x": {"positive": True}}},
    )
    assert (
        grade_response(
            problem, block("sp.sqrt(x*x)"), judge_enabled=False
        ).grading_status
        == "pass"
    )
    problem = auto_problem(
        "categorical",
        ["A"],
        {"choices": ["A", "B"], "allowed_aliases": {"alpha": "A"}},
    )
    assert (
        grade_response(problem, block("{'alpha'}"), judge_enabled=False).grading_status
        == "pass"
    )


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
    assert report.grading_status == "not_graded"
    assert report.extraction_status == status


@pytest.mark.parametrize("failed_import", [1, 2])
def test_missing_sympy_is_not_a_zero_reward(monkeypatch, failed_import):
    original_import = builtins.__import__
    imports = 0

    def import_module(name, globals=None, *args, **kwargs):
        nonlocal imports
        if name == "sympy" and (globals or {}).get("__name__") == answer.__name__:
            imports += 1
            if imports == failed_import:
                raise ImportError("SymPy dependency unavailable")
        return original_import(name, globals, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    with pytest.raises(CritptAiRewardError) as raised:
        score(
            CritptAiReward(judge_enabled=False), auto_problem("exact", "2"), block("2")
        )
    assert imports == failed_import
    assert raised.value.report.grading_status == "grader_error"
    assert "SymPy dependency unavailable" in raised.value.report.error_message


@pytest.mark.parametrize(
    "kind,expression", [("exact", "2"), ("symbolic", "2*x"), ("symbolic", "x+1")]
)
@pytest.mark.parametrize("error", [RuntimeError, TypeError, ValueError])
def test_comparison_errors_are_not_zero_rewards(monkeypatch, kind, expression, error):
    def fail(*args, **kwargs):
        raise error("comparison failed internally")

    monkeypatch.setattr("sympy.simplify", fail)
    problem = (
        auto_problem("symbolic", "2*x", {"variables": ["x"]})
        if kind == "symbolic"
        else auto_problem("exact", "2")
    )
    response = block(expression)
    with pytest.raises(CritptAiRewardError) as raised:
        score(CritptAiReward(judge_enabled=False), problem, response)
    assert raised.value.report.grading_status == "grader_error"
    assert raised.value.report.extraction_status == "success"
    assert raised.value.report.extracted_answer is not None
    assert "comparison failed internally" in raised.value.report.error_message


def test_floating_expression_is_not_silently_rounded():
    report = grade_response(
        auto_problem("floating_numeric", 1.25, {"abs_tol": 0.01}),
        block("sp.Rational(5,4)"),
        equivalence_judge=forbidden_judge(),
    )
    assert report.grading_status == "not_graded"
    assert report.extraction_status == "invalid_answer_format"


def test_unsupported_sympy_is_inert_but_can_reach_judge(monkeypatch):
    def forbidden(*args):
        pytest.fail("Unsupported code was executed")

    monkeypatch.setattr(answer, "_execute_answer_code", forbidden)
    prompts = []
    scorer = CritptAiReward(equivalence_judge=equivalence(prompts=prompts))
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

        monkeypatch.setattr(grader, "run_with_deadline", timeout)
    judge = equivalence("uncertain")
    if failure == "api":

        def unavailable(_messages, _config):
            raise TimeoutError("mock provider timeout")

        judge = LLMEquivalenceJudge(
            config=EquivalenceJudgeConfig(model="test"),
            completion=unavailable,
        )
    scorer = CritptAiReward(
        judge_enabled=failure != "disabled", equivalence_judge=judge
    )
    problem = auto_problem("symbolic", "x", {"variables": ["x"]})
    with pytest.raises(CritptAiRewardError) as raised:
        score(scorer, problem, block("sp.fresnels(x)"))
    error = raised.value
    assert "reward" not in error.report.model_dump()
    assert error.report.grading_status in {"grader_error", "grader_uncertain"}
    restored = pickle.loads(pickle.dumps(error))
    assert restored.sample_id == "training-sample:1"
    assert restored.report == error.report


def test_exact_numeric_conflict_is_unresolved():
    with pytest.raises(CritptAiRewardError) as raised:
        score(
            CritptAiReward(equivalence_judge=equivalence("equivalent")),
            auto_problem(expected="2"),
            block("sp.Integer(3)"),
        )
    assert raised.value.report.exact_numeric_comparisons[0].status == "different"
    assert raised.value.report.grading_status == "grader_uncertain"


@pytest.mark.parametrize("verdict,expected", [("correct", 1.0), ("incorrect", 0.0)])
def test_freestyle_multiple_items_and_teacher_ignored(verdict, expected):
    prompts = []
    scorer = CritptAiReward(
        judge_enabled=False, semantic_judge=semantic(verdict, prompts)
    )
    assert score(scorer, freestyle_problem(), "E=2; stable.") == expected
    payload = json.loads(prompts[0][1]["content"])
    assert len(payload["answer_items"]) == 2
    assert payload["private_reference"]["reference_answer"] == "E=2; stable."
    assert "UNVERIFIED_TEACHER_SENTINEL" not in json.dumps(prompts)
    assert "UNNEEDED_BLUEPRINT" not in json.dumps(prompts)


def test_freestyle_uncertain_and_empty_response():
    scorer = CritptAiReward(semantic_judge=semantic("uncertain"))
    with pytest.raises(CritptAiRewardError):
        score(scorer, freestyle_problem(), "Maybe stable")
    assert score(scorer, freestyle_problem(), "") == 0


@pytest.mark.parametrize("verdict", ["correct", "incorrect", "uncertain"])
def test_report_round_trip_preserves_judgment(verdict):
    report = grade_response(
        freestyle_problem(), "E=2; stable.", semantic_judge=semantic(verdict)
    )
    assert GradeReport.model_validate_json(report.model_dump_json()) == report
    assert pickle.loads(pickle.dumps(report)) == report


def test_reference_completion_cannot_override_ground_truth():
    scorer = CritptAiReward(judge_enabled=False)
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


@pytest.mark.parametrize("response", [None, 42, [], b"answer"])
def test_nontext_response_is_rejected_before_execution(monkeypatch, response):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid input reached execution")

    monkeypatch.setattr(grader, "run_with_deadline", forbidden)
    with pytest.raises(TypeError, match="response must be text"):
        grade_response(auto_problem(), response)


def test_grading_can_be_relocated_without_ddpr(tmp_path):
    source = Path(grader.__file__).parent
    target = tmp_path / "standalone_critpt_ai"
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("reward.py", "__pycache__", ".DS_Store"),
    )
    # Integration supplies its own exports, excluding DDPR's reward adapter.
    (target / "__init__.py").write_text("", encoding="utf-8")
    script = """
import json
import sys
sys.path.insert(0, sys.argv[1])
from standalone_critpt_ai.grader import grade_response
from standalone_critpt_ai.io.inputs import parse_grading_problem
problem = parse_grading_problem(json.loads(sys.argv[2]))
report = grade_response(problem, "", judge_enabled=False)
assert report.extraction_status == "missing_answer_function"
assert "reward" not in report.model_dump()
assert not any(name == "ddpr" or name.startswith("ddpr.") for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script, str(tmp_path), json.dumps(auto_problem())],
        check=True,
        capture_output=True,
        text=True,
    )


def test_zero_argument_reward_loads_with_environment(monkeypatch):
    monkeypatch.setenv("DDPR_CRITPT_AI_EQUIVALENCE_JUDGE", "false")
    load_reward.cache_clear()
    assert not load_reward("ddpr.rewards.critpt_ai.CritptAiReward").judge_enabled
    monkeypatch.setenv("DDPR_CRITPT_AI_EQUIVALENCE_JUDGE", "invalid")
    with pytest.raises(ValueError, match="must be true or false"):
        CritptAiReward()
    load_reward.cache_clear()


def test_core_reward_import_does_not_require_critpt_ai_dependencies():
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
        reward, "load_reward", lambda _path: CritptAiReward(judge_enabled=False)
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


def test_rule_comparisons():
    from ddsr_bench.benchmarks.critpt.evaluation.ai.rule.categorical import (
        grade_categorical,
    )
    from ddsr_bench.benchmarks.critpt.evaluation.ai.rule.numeric import (
        grade_exact_numeric,
        grade_floating_numeric,
    )
    from ddsr_bench.benchmarks.critpt.evaluation.ai.rule.symbolic import (
        grade_symbolic,
    )

    assert grade_exact_numeric(answer="sqrt(4)", target="2").passed
    assert grade_symbolic(answer="x+x", target="2*x", variables=["x"]).passed
    assert grade_floating_numeric(answer=1.0, target=1.0, abs_tol=0.01).passed
    assert grade_categorical(answer={"A", "B"}, target=["B", "A"]).passed


def test_swift_adapter_with_mocked_framework(monkeypatch):
    framework = ModuleType("swift.rewards")
    framework.ORM = object
    framework.orms = {}
    monkeypatch.setitem(sys.modules, "swift.rewards", framework)
    source = Path(__file__).parents[2] / "ddpr/backends/swift/reward.py"
    spec = importlib.util.spec_from_file_location(
        "critpt_ai_swift_adapter_test", source
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("DDPR_REWARD", "ddpr.rewards.critpt_ai.CritptAiReward")
    monkeypatch.setattr(
        module, "load_reward", lambda _path: CritptAiReward(judge_enabled=False)
    )
    adapter = module.RewardAdapter()
    assert adapter(
        completions=[block("1"), block("2")],
        sample_id=["first", "second"],
        reference_completion=[block("2"), block("2")],
        metadata=[{"problem": auto_problem(expected="1")}] * 2,
    ) == [1.0, 0.0]


@pytest.mark.parametrize(
    "kind,expected,args",
    [
        ("exact", "x", None),
        ("exact", "", None),
        ("floating_numeric", 1.0, {"abs_tol": -1}),
        ("floating_numeric", float("inf"), {"abs_tol": 0.1}),
        ("symbolic", "1", {"variables": ["x"]}),
    ],
)
def test_invalid_references_are_not_zero_rewards(kind, expected, args):
    problem = auto_problem(kind, expected, args)
    response = block("1.0" if kind == "floating_numeric" else "2")
    with pytest.raises(CritptAiRewardError) as raised:
        score(CritptAiReward(judge_enabled=False), problem, response)
    assert raised.value.report.grading_status == "grader_error"


@pytest.mark.parametrize("timeout", ["0", "nan", "inf", "invalid"])
def test_invalid_numeric_timeout_is_a_grader_error(monkeypatch, timeout):
    monkeypatch.setenv("AUTO_CRITPT_NUMERIC_CHECK_TIMEOUT_SECONDS", timeout)
    with pytest.raises(CritptAiRewardError) as raised:
        score(
            CritptAiReward(equivalence_judge=forbidden_judge()),
            auto_problem(expected="2"),
            block("3"),
        )
    assert raised.value.report.grading_status == "grader_error"
    assert raised.value.report.extraction_status == "success"
    assert (
        "Expression equivalence assessment failed" in raised.value.report.error_message
    )


def test_judge_configuration_failure_is_reported(monkeypatch):
    monkeypatch.setenv("AUTO_CRITPT_GRADER_JUDGE_BACKEND", "invalid")
    report = grade_response(
        auto_problem("symbolic", "x", {"variables": ["x"]}),
        block("sp.fresnels(x)"),
    )
    assert report.grading_status == "grader_error"
    assert report.extraction_status == "text_only"
    assert "backend" in report.error_message


@pytest.mark.parametrize("judge_enabled", [False, True])
def test_comparison_timeout_preserves_fallback(monkeypatch, judge_enabled):
    def timeout(*args):
        raise GraderTimeout

    monkeypatch.setattr(grader, "grade_with_plan", timeout)
    report = grade_response(
        auto_problem("symbolic", "x+1", {"variables": ["x"]}),
        block("x+1"),
        judge_enabled=judge_enabled,
        equivalence_judge=equivalence("equivalent"),
    )
    assert report.extraction_status == "success"
    assert report.grading_status == ("pass" if judge_enabled else "grader_error")
    if judge_enabled:
        assert report.deterministic_skip_reasons == {"answer": "comparison_timeout"}


@pytest.mark.parametrize("target", ["", " ", "\n"])
def test_empty_grading_target_is_invalid_metadata(target):
    problem = auto_problem(expected="1")
    problem["grading_plan"]["calls"][0]["target"] = target
    with pytest.raises(ValueError, match="target"):
        load_problem({"problem": problem})


def test_legacy_freestyle_accepts_mapping_input():
    from types import MappingProxyType

    problem = freestyle_problem()
    del problem["answer_items"]
    assert load_problem({"problem": MappingProxyType(problem)}) == load_problem(
        {"problem": problem}
    )


@pytest.mark.parametrize("freestyle", [False, True])
def test_native_report_formatting(freestyle):
    import hashlib

    from ddsr_bench.benchmarks.critpt.evaluation.ai.io.outputs import (
        format_report,
    )
    from sympy import Rational

    problem = load_problem(
        {"problem": freestyle_problem() if freestyle else auto_problem()}
    )
    response = "A mathematical response: α = ½"
    report = format_report(
        problem,
        response,
        extraction_status="not_applicable" if freestyle else "success",
        grading_status="pass",
        passed=True,
        score=1.0,
        extracted_answer={"answer": Rational(1, 2)},
    )
    assert report.problem_id == problem.problem_id
    assert report.problem_format == ("freestyle" if freestyle else "auto")
    assert report.answer_type == problem.answer_type
    assert report.raw_response == response
    assert report.response_sha256 == hashlib.sha256(response.encode()).hexdigest()
    assert report.extracted_answer == {"answer": "1/2"}
    assert GradeReport.model_validate_json(report.model_dump_json()) == report


@pytest.mark.parametrize(
    "kind,target,kwargs,response",
    [
        ("categorical", "B", {"choices": ["A"]}, "'A'"),
        ("symbolic", "sin(x)**2+cos(x)**2", {"variables": ["x"]}, "2"),
        (
            "symbolic",
            "sin(x)**2+cos(x)**2",
            {"variables": ["x"]},
            "sp.sin(x)**2+sp.cos(x)**2",
        ),
        (
            "symbolic",
            "sin(x)**2+cos(x)**2",
            {"variables": ["x"]},
            "sp.cos(x)**2+sp.sin(x)**2+1",
        ),
    ],
)
def test_invalid_reference_cannot_be_an_incorrect_answer(
    kind, target, kwargs, response
):
    with pytest.raises(CritptAiRewardError) as raised:
        score(
            CritptAiReward(judge_enabled=False),
            auto_problem(kind, target, kwargs),
            block(response),
        )
    assert raised.value.report.grading_status == "grader_error"
