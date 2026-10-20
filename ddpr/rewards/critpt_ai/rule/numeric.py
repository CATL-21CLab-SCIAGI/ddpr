from __future__ import annotations

import math
from typing import Any

from ..schemas import ExactNumericComparison, PrimitiveGrade, unsupported_grading_args
from .utils import (
    _grade,
    _parse_expression,
    _text_only_grade,
    _validate_closed_form_expression,
    _validate_exact_expression,
)


def grade_floating_numeric(
    answer: Any,
    target: Any,
    **kwargs: Any,
) -> PrimitiveGrade:
    unsupported_args = unsupported_grading_args("floating_numeric", kwargs)
    if unsupported_args:
        raise ValueError(f"unsupported floating_numeric args: {unsupported_args}")
    if not ({"abs_tol", "rel_tol"} & kwargs.keys()):
        raise ValueError("floating_numeric grading requires abs_tol or rel_tol")
    raw_abs_tol = kwargs.get("abs_tol", 0.0)
    raw_rel_tol = kwargs.get("rel_tol", 0.0)
    if any(
        isinstance(value, bool) or not isinstance(value, int | float)
        for value in (raw_abs_tol, raw_rel_tol)
    ):
        raise ValueError("floating tolerances must be JSON numbers")
    abs_tol = float(raw_abs_tol)
    rel_tol = float(raw_rel_tol)
    if (
        not math.isfinite(abs_tol)
        or not math.isfinite(rel_tol)
        or abs_tol < 0
        or rel_tol < 0
    ):
        raise ValueError("floating tolerances must be finite and non-negative")
    if type(target) is not float or not math.isfinite(target):
        raise ValueError("expected value must be a finite Python float")
    if type(answer) is not float or not math.isfinite(answer):
        return _grade(
            "floating_numeric",
            False,
            "actual value must be a finite Python float",
        )

    passed = math.isclose(answer, target, abs_tol=abs_tol, rel_tol=rel_tol)
    reason = "floating values match" if passed else "floating values differ"
    return _grade("floating_numeric", passed, reason)


def grade_exact_numeric(
    answer: Any,
    target: Any,
    **kwargs: Any,
) -> PrimitiveGrade:
    unsupported_args = unsupported_grading_args("exact", kwargs)
    if unsupported_args:
        raise ValueError(f"unsupported exact args: {unsupported_args}")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("exact expected value must be a non-empty expression string")
    import sympy as sp

    reference_fallback = _text_only_grade("exact", target)
    if reference_fallback is not None:
        return _text_only_grade("exact", answer) or reference_fallback
    expected_expr = _parse_expression(target, sp)
    _validate_closed_form_expression(expected_expr, sp, label="expected")
    _validate_exact_expression(expected_expr, sp, label="expected")
    fallback = _text_only_grade("exact", answer)
    if fallback is not None:
        return fallback
    try:
        actual_expr = _parse_expression(answer, sp)
        _validate_closed_form_expression(actual_expr, sp, label="actual")
        _validate_exact_expression(actual_expr, sp, label="actual")
    except (TypeError, ValueError) as exc:
        return _grade("exact", False, str(exc))

    difference = sp.simplify(actual_expr - expected_expr)
    passed = difference == 0
    if not passed:
        passed = difference.equals(0) is True

    reason = "exact values are equivalent" if passed else "exact values differ"
    return _grade(
        "exact",
        bool(passed),
        reason,
        equivalence_candidate=not bool(passed),
    )


def evaluate_numeric_agreement(
    answer: Any,
    target: Any,
    *,
    field_name: str,
    significant_digits: int = 16,
) -> ExactNumericComparison:
    """Evaluate numerical agreement between an answer and its reference target.

    field_name identifies the output being assessed (for example, "energy"); it
    labels the result's serialized `target` field and does not affect comparison.
    A numerical match is supporting evidence, not proof of exact equality.
    """

    if not 1 <= significant_digits <= 50:
        raise ValueError("significant_digits must be between 1 and 50")
    try:
        import sympy as sp

        actual_expr = _parse_expression(answer, sp)
        expected_expr = _parse_expression(target, sp)
        _validate_closed_form_expression(actual_expr, sp, label="actual")
        _validate_closed_form_expression(expected_expr, sp, label="expected")
        _validate_exact_expression(actual_expr, sp, label="actual")
        _validate_exact_expression(expected_expr, sp, label="expected")

        working_digits = max(32, significant_digits + 16)
        actual_numeric = sp.N(actual_expr, working_digits)
        expected_numeric = sp.N(expected_expr, working_digits)
        if (
            actual_numeric.is_real is not True
            or expected_numeric.is_real is not True
            or actual_numeric.is_finite is not True
            or expected_numeric.is_finite is not True
        ):
            raise ValueError("numerical cross-check requires finite real constants")

        absolute_actual = sp.Abs(actual_numeric)
        absolute_expected = sp.Abs(expected_numeric)
        scale = (
            absolute_actual
            if absolute_actual >= absolute_expected
            else absolute_expected
        )
        difference = sp.Abs(actual_numeric - expected_numeric)
        relative_difference = (
            sp.Integer(0) if scale == 0 else sp.N(difference / scale, working_digits)
        )
        threshold = sp.Float(5, working_digits) * sp.Pow(
            10,
            -significant_digits,
        )
        matched = bool(relative_difference <= threshold)
        return ExactNumericComparison(
            target=field_name,
            status="match" if matched else "different",
            significant_digits=significant_digits,
            actual_decimal=str(sp.N(actual_numeric, significant_digits + 2)),
            expected_decimal=str(sp.N(expected_numeric, significant_digits + 2)),
            relative_difference=str(sp.N(relative_difference, significant_digits + 2)),
            reason=(
                f"values agree to {significant_digits} significant digits"
                if matched
                else f"values differ within a {significant_digits}-digit check"
            ),
        )
    except Exception as exc:
        return ExactNumericComparison(
            target=field_name,
            status="unavailable",
            significant_digits=significant_digits,
            reason=f"numeric exact-value comparison unavailable: {exc}",
        )
