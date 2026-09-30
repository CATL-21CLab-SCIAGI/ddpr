from __future__ import annotations

from typing import Any

from ..schemas import GradingPlan, GradingResult, PrimitiveGrade
from .categorical import grade_categorical
from .numeric import grade_exact, grade_floating_numeric
from .symbolic import grade_symbolic


def grade_with_plan(
    answer_block: dict[str, Any],
    grading_plan: GradingPlan,
) -> GradingResult:
    grades: list[PrimitiveGrade] = []
    for call in grading_plan.calls:
        if call.target not in answer_block:
            grades.append(
                PrimitiveGrade(
                    primitive=call.primitive,
                    target=call.target,
                    passed=False,
                    score=0.0,
                    reason="target missing from answer block",
                )
            )
            continue
        actual = answer_block[call.target]
        if call.primitive == "floating_numeric":
            grade = grade_floating_numeric(actual, call.expected, call.args)
        elif call.primitive == "exact":
            grade = grade_exact(actual, call.expected, call.args)
        elif call.primitive == "symbolic":
            grade = grade_symbolic(actual, call.expected, call.args)
        else:
            grade = grade_categorical(actual, call.expected, call.args)
        grades.append(grade.model_copy(update={"target": call.target}))

    return compose_grades(grading_plan, grades)


def compose_grades(
    grading_plan: GradingPlan,
    grades: list[PrimitiveGrade],
) -> GradingResult:
    if len(grades) != len(grading_plan.calls):
        raise ValueError("grades must align with grading_plan calls")
    required_failed = any(
        call.required and not grade.passed
        for call, grade in zip(grading_plan.calls, grades, strict=True)
    )
    total_weight = sum(max(call.weight, 0.0) for call in grading_plan.calls)
    if total_weight <= 0:
        score = 1.0 if all(grade.passed for grade in grades) else 0.0
    else:
        score = (
            sum(
                max(call.weight, 0.0) * grade.score
                for call, grade in zip(grading_plan.calls, grades, strict=True)
            )
            / total_weight
        )
    passed = (not required_failed) and (
        score >= 1.0 if grading_plan.composition == "all_required" else score > 0
    )
    return GradingResult(passed=passed, score=score, calls=grades)
