"""Format grading evidence as an Auto-CritPt-compatible report."""

from __future__ import annotations

import hashlib
from typing import Any

from .schemas import GradeReport, GradingInput


def format_report(problem: GradingInput, response: str, **kwargs: Any) -> GradeReport:
    """Attach native problem metadata and serialize extracted mathematical values."""
    if "extracted_answer" in kwargs:
        kwargs["extracted_answer"] = _json_safe(kwargs["extracted_answer"])
    return GradeReport(
        problem_id=problem.problem_id,
        problem_format=problem.problem_format,
        answer_type=problem.answer_type,
        raw_response=response,
        response_sha256=hashlib.sha256(response.encode("utf-8")).hexdigest(),
        **kwargs,
    )


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
