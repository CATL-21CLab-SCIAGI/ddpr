from __future__ import annotations

from typing import Any

from ..schemas import PrimitiveGrade, PrimitiveName, UnexecutedAnswer
from .expressions import (
    UnsupportedAnswerOperation,
    _validate_expression_call_names,
)


def _grade(
    primitive: PrimitiveName,
    target: str,
    passed: bool,
    reason: str,
    *,
    equivalence_candidate: bool = False,
) -> PrimitiveGrade:
    return PrimitiveGrade(
        primitive=primitive,
        target=target,
        passed=passed,
        score=1.0 if passed else 0.0,
        equivalence_candidate=equivalence_candidate,
        reason=reason,
    )


def _text_only_grade(primitive: PrimitiveName, actual: Any) -> PrimitiveGrade | None:
    reason = actual.reason if isinstance(actual, UnexecutedAnswer) else None
    if isinstance(actual, str):
        try:
            _validate_expression_call_names(actual)
        except UnsupportedAnswerOperation as exc:
            reason = str(exc)
        except ValueError:
            return None
    if reason is None:
        return None
    return PrimitiveGrade(
        primitive=primitive,
        target="",
        passed=False,
        score=0.0,
        reason="Local evaluation skipped: " + reason,
        equivalence_candidate=True,
        deterministic_skip_reason=reason,
    )
