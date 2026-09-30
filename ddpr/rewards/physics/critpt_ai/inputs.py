from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .schemas import (
    FreestyleProblemGradingInput,
    GradingInput,
    ProblemFormat,
    ProblemFormatSelection,
    ProblemGradingInput,
    PublicProblemInput,
)


def resolve_problem_format(
    problem: PublicProblemInput | Mapping[str, Any],
    problem_format: ProblemFormatSelection = "detect",
) -> ProblemFormat:
    if problem_format not in {"detect", "auto", "freestyle"}:
        raise ValueError("problem_format must be detect, auto, or freestyle")
    if isinstance(problem, (ProblemGradingInput, FreestyleProblemGradingInput)):
        detected = problem.problem_format
    else:
        payload = (
            problem.model_dump() if isinstance(problem, PublicProblemInput) else problem
        )
        if not isinstance(payload, Mapping):
            raise TypeError("problem must be a JSON object or problem input model")
        auto = "grading_plan" in payload or "answer_contract" in payload
        freestyle = "answer_type" in payload or "answer_items" in payload
        if auto and freestyle:
            raise ValueError("problem mixes Auto and Freestyle grading fields")
        detected = "auto" if auto else ("freestyle" if freestyle else None)
    if problem_format != "detect":
        if detected is not None and detected != problem_format:
            raise ValueError("problem_format conflicts with the supplied problem")
        return problem_format
    return detected or "auto"


def parse_grading_problem(
    problem: GradingInput | Mapping[str, Any],
    *,
    problem_format: ProblemFormatSelection = "detect",
) -> GradingInput:
    selected_format = resolve_problem_format(problem, problem_format)
    if isinstance(problem, (ProblemGradingInput, FreestyleProblemGradingInput)):
        return problem
    schema = (
        FreestyleProblemGradingInput
        if selected_format == "freestyle"
        else ProblemGradingInput
    )
    return schema.model_validate(problem)
