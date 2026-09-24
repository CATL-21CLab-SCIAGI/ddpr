from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Mapping
from typing import Any

from .equivalence import (
    EquivalenceJudge,
    EquivalenceJudgeConfig,
    ExpressionEquivalenceJudge,
    judge_equivalence,
)
from .grading_runtime import (
    AnswerExtractionError,
    extract_answer_locally,
    grader_timeout_seconds,
    run_with_deadline,
)
from .grading_runtime import (
    GraderTimeout as _GraderTimeout,
)
from .inputs import parse_grading_problem, validate_judge_mode
from .primitives import (
    ExactNumericComparison,
    PrimitiveGrade,
    UnexecutedAnswer,
    compare_exact_numerically,
    compose_grades,
    extract_gradeable_answer,
    grade_with_plan,
)
from .schemas import (
    FreestyleProblemGradingInput,
    GradeReport,
    GradingInput,
    ProblemFormatSelection,
)
from .semantic import LLMSemanticJudge, SemanticJudgeConfig


def grade_response(
    problem: GradingInput | Mapping[str, Any],
    response: str,
    *,
    equivalence_judge: EquivalenceJudge | None = None,
    judge_config: EquivalenceJudgeConfig | None = None,
    judge_enabled: bool = True,
    problem_format: ProblemFormatSelection = "detect",
    semantic_judge: LLMSemanticJudge | None = None,
    semantic_judge_config: SemanticJudgeConfig | None = None,
    attempt_id: str = "saved-response",
) -> GradeReport:
    selected_problem = parse_grading_problem(problem, problem_format=problem_format)
    validate_judge_mode(selected_problem, judge_enabled=judge_enabled)
    if isinstance(selected_problem, FreestyleProblemGradingInput):
        return _grade_freestyle_response(
            selected_problem,
            response,
            semantic_judge=semantic_judge,
            semantic_judge_config=semantic_judge_config,
            attempt_id=attempt_id,
        )
    response_hash = hashlib.sha256(response.encode("utf-8")).hexdigest()
    extracted_answer = None
    try:
        deadline = time.monotonic() + grader_timeout_seconds()
        extracted_answer = run_with_deadline(
            extract_answer_locally,
            response,
            selected_problem.grading_plan,
            extract_gradeable_answer,
            timeout_seconds=deadline - time.monotonic(),
        )
        deterministic = run_with_deadline(
            grade_with_plan,
            extracted_answer,
            selected_problem.grading_plan,
            timeout_seconds=deadline - time.monotonic(),
        )
    except AnswerExtractionError as exc:
        return GradeReport(
            problem_id=selected_problem.problem_id,
            answer_type=selected_problem.answer_contract.answer_type,
            raw_response=response,
            response_sha256=response_hash,
            extraction_status=_classify_extraction_error(str(exc)),
            grading_status="not_graded",
            passed=False,
            error_message=str(exc),
        )

    except _GraderTimeout:
        plan = selected_problem.grading_plan
        if (
            extracted_answer is None
            or not judge_enabled
            or any(call.primitive not in {"exact", "symbolic"} for call in plan.calls)
            or any(call.target not in extracted_answer for call in plan.calls)
        ):
            return GradeReport(
                problem_id=selected_problem.problem_id,
                answer_type=selected_problem.answer_type,
                raw_response=response,
                response_sha256=response_hash,
                extracted_answer=_json_safe(extracted_answer),
                extraction_status="success"
                if extracted_answer is not None
                else "not_attempted",
                grading_status="grader_error",
                passed=False,
                error_message="Deterministic grader exceeded the configured local resource limit.",
            )
        deterministic = compose_grades(
            plan,
            [
                PrimitiveGrade(
                    primitive=call.primitive,
                    target=call.target,
                    passed=False,
                    score=0.0,
                    reason="Deterministic comparison exceeded its local resource limit.",
                    equivalence_candidate=True,
                    deterministic_skip_reason="comparison_timeout",
                )
                for call in plan.calls
            ],
        )
    except Exception as exc:
        return GradeReport(
            problem_id=selected_problem.problem_id,
            answer_type=selected_problem.answer_contract.answer_type,
            raw_response=response,
            response_sha256=response_hash,
            extraction_status="not_attempted",
            grading_status="grader_error",
            passed=False,
            error_message=str(exc),
        )

    invalid_scalar_targets = [
        call.target
        for call in selected_problem.grading_plan.calls
        if call.primitive in {"exact", "symbolic"}
        and isinstance(
            extracted_answer.get(call.target), (dict, list, tuple, set, frozenset)
        )
    ]
    if invalid_scalar_targets:
        return GradeReport(
            problem_id=selected_problem.problem_id,
            answer_type=selected_problem.answer_type,
            raw_response=response,
            response_sha256=response_hash,
            extracted_answer=_json_safe(extracted_answer),
            extraction_status="invalid_answer_format",
            grading_status="not_graded",
            passed=False,
            deterministic_grading_results=[
                grade.model_dump(mode="json") for grade in deterministic.calls
            ],
            error_message=(
                "Expected one scalar mathematical answer, not a container, for target(s): "
                + ", ".join(invalid_scalar_targets)
            ),
        )

    text_answers = [
        value
        for value in extracted_answer.values()
        if isinstance(value, UnexecutedAnswer)
    ]
    answer_fields = {
        "extracted_answer": None if text_answers else _json_safe(extracted_answer),
        "extraction_status": "text_only" if text_answers else "success",
        "candidate_answer_code": text_answers[0].answer_code if text_answers else None,
        "deterministic_skip_reasons": {
            grade.target: grade.deterministic_skip_reason
            for grade in deterministic.calls
            if grade.deterministic_skip_reason is not None
        },
    }
    deterministic_results = [
        result.model_dump(mode="json")
        for result in deterministic.calls
        if result.deterministic_skip_reason is None
    ]
    final_grades = list(deterministic.calls)
    judgments = []
    exact_numeric_comparisons = []
    uncertain_indexes: set[int] = set()
    selected_judge = equivalence_judge
    for index, (call, grade) in enumerate(
        zip(
            selected_problem.grading_plan.calls,
            deterministic.calls,
            strict=True,
        )
    ):
        if grade.passed or not grade.equivalence_candidate:
            continue
        numeric_comparison = None
        if call.primitive == "exact" and grade.deterministic_skip_reason in {
            None,
            "comparison_timeout",
        }:
            numeric_comparison = _compare_exact_with_deadline(
                extracted_answer[call.target],
                call.expected,
                target=call.target,
                significant_digits=16,
            )
            exact_numeric_comparisons.append(numeric_comparison)
        if not judge_enabled:
            if grade.deterministic_skip_reason is not None:
                uncertain_indexes.add(index)
            continue
        if selected_judge is None:
            selected_judge = ExpressionEquivalenceJudge(config=judge_config)
        try:
            judgment = judge_equivalence(
                selected_judge,
                call=call,
                actual=extracted_answer[call.target],
                deterministic_grade=grade,
                public_problem={
                    "problem_statement": selected_problem.problem_statement,
                    "public_solver_output_instructions": selected_problem.public_solver_output_instructions,
                },
            )
        except Exception as exc:
            return GradeReport(
                problem_id=selected_problem.problem_id,
                answer_type=selected_problem.answer_contract.answer_type,
                raw_response=response,
                response_sha256=response_hash,
                **answer_fields,
                grading_status="grader_error",
                passed=False,
                score=None,
                deterministic_grading_results=deterministic_results,
                equivalence_judgments=judgments,
                exact_numeric_comparisons=exact_numeric_comparisons,
                grading_results=deterministic_results,
                error_message=("Expression equivalence Judge failed: " + str(exc)),
            )
        if numeric_comparison is not None:
            judgment = _reconcile_numeric_evidence(
                judgment,
                numeric_comparison.status,
            )
        judgments.append(judgment)
        if judgment.resolution == "equivalent":
            final_grades[index] = grade.model_copy(
                update={
                    "passed": True,
                    "score": 1.0,
                    "reason": (
                        "LLM equivalence Judge confirmed equivalence: "
                        + judgment.rationale
                    ),
                }
            )
        elif judgment.resolution == "uncertain":
            uncertain_indexes.add(index)
            final_grades[index] = grade.model_copy(
                update={
                    "reason": (
                        "LLM equivalence Judge was uncertain: " + judgment.rationale
                    )
                }
            )

    final = compose_grades(selected_problem.grading_plan, final_grades)
    grading_status = "pass" if final.passed else "fail"
    if uncertain_indexes and not final.passed:
        optimistic = [
            (
                grade.model_copy(update={"passed": True, "score": 1.0})
                if index in uncertain_indexes
                else grade
            )
            for index, grade in enumerate(final_grades)
        ]
        if compose_grades(selected_problem.grading_plan, optimistic).passed:
            grading_status = "grader_uncertain"

    return GradeReport(
        problem_id=selected_problem.problem_id,
        answer_type=selected_problem.answer_contract.answer_type,
        raw_response=response,
        response_sha256=response_hash,
        **answer_fields,
        grading_status=grading_status,
        passed=final.passed,
        score=None if uncertain_indexes else final.score,
        deterministic_grading_results=deterministic_results,
        equivalence_judgments=judgments,
        exact_numeric_comparisons=exact_numeric_comparisons,
        grading_results=[
            {
                **result.model_dump(mode="json"),
                "grading_status": (
                    "uncertain"
                    if index in uncertain_indexes
                    else "pass"
                    if result.passed
                    else "fail"
                ),
            }
            for index, result in enumerate(final_grades)
        ],
    )


def _grade_freestyle_response(
    problem: FreestyleProblemGradingInput,
    response: str,
    *,
    semantic_judge: LLMSemanticJudge | None,
    semantic_judge_config: SemanticJudgeConfig | None,
    attempt_id: str,
) -> GradeReport:
    common = {
        "problem_id": problem.problem_id,
        "problem_format": "freestyle",
        "answer_type": problem.answer_type,
        "raw_response": response,
        "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
        "extraction_status": "not_applicable",
    }
    if not response.strip():
        return GradeReport(
            **common,
            grading_status="fail",
            passed=False,
            score=0.0,
            error_message="Solver supplied no final response.",
        )
    try:
        selected_judge = semantic_judge or LLMSemanticJudge(
            config=semantic_judge_config
        )
        judgment = selected_judge.judge(problem, response, attempt_id=attempt_id)
    except Exception as exc:
        return GradeReport(
            **common,
            grading_status="grader_error",
            passed=False,
            error_message="SemanticJudge failed: " + str(exc),
        )
    status = {
        "correct": "pass",
        "incorrect": "fail",
        "uncertain": "grader_uncertain",
    }[judgment.verdict]
    return GradeReport(
        **common,
        grading_status=status,
        passed=status == "pass",
        score=None if status == "grader_uncertain" else float(status == "pass"),
        semantic_judgment=judgment,
    )


def _reconcile_numeric_evidence(judgment: Any, status: str) -> Any:
    conflict = (judgment.resolution == "equivalent" and status == "different") or (
        judgment.resolution == "not_equivalent" and status == "match"
    )
    if not conflict:
        return judgment
    return judgment.model_copy(
        update={
            "resolution": "uncertain",
            "rationale": (
                judgment.rationale
                + " The independent 16-digit numerical cross-check conflicts "
                "with this verdict."
            ),
        }
    )


def _numeric_comparison(
    actual: Any, expected: Any, target: str, digits: int
) -> ExactNumericComparison:
    return compare_exact_numerically(
        actual, expected, target=target, significant_digits=digits
    )


def _compare_exact_with_deadline(
    actual: Any,
    expected: Any,
    *,
    target: str,
    significant_digits: int,
) -> ExactNumericComparison:
    seconds = float(os.getenv("AUTO_CRITPT_NUMERIC_CHECK_TIMEOUT_SECONDS", "30"))
    if seconds <= 0:
        raise ValueError("AUTO_CRITPT_NUMERIC_CHECK_TIMEOUT_SECONDS must be positive")
    try:
        return run_with_deadline(
            _numeric_comparison,
            actual,
            expected,
            target,
            significant_digits,
            timeout_seconds=seconds,
        )
    except _GraderTimeout:
        reason = "Numeric exact-value comparison exceeded the configured local resource limit."
    except Exception as exc:
        reason = f"Numeric exact-value comparison unavailable: {exc}"
    return ExactNumericComparison(
        target=target,
        status="unavailable",
        significant_digits=significant_digits,
        reason=reason,
    )


def _classify_extraction_error(message: str) -> str:
    lowered = message.casefold()
    if "no final python answer function block" in lowered:
        return "missing_answer_function"
    if (
        "direct finite python float literal" in lowered
        or "direct finite python float literals" in lowered
    ):
        return "invalid_answer_format"
    if any(
        marker in lowered
        for marker in (
            "invalid syntax",
            "exactly one answer",
            "callable answer",
            "*args",
            "**kwargs",
        )
    ):
        return "parse_failure"
    if any(
        marker in lowered
        for marker in (
            "unsupported syntax",
            "may only import",
            "may only call",
            "may not use",
            "may not access",
            "may not modify",
            "may only access",
            "not allowed",
        )
    ):
        return "unsafe_code"
    if any(
        marker in lowered
        for marker in (
            "could not execute",
            "could not call",
            "sympy is not installed",
        )
    ):
        return "runtime_failure"
    return "parse_failure"


def _json_safe(value: Any) -> Any:
    try:
        import sympy as sp
    except ImportError:  # pragma: no cover
        sp = None
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if sp is not None and isinstance(value, sp.Basic):
        return str(value)
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted((_json_safe(item) for item in value), key=repr)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)
