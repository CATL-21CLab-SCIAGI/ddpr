"""Map physics grading results to DDPR training rewards."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ddpr.rewards.base import Reward

from .critpt_ai.inputs import parse_grading_problem
from .critpt_ai.schemas import GradeReport, GradingInput
from .grader import grade_response
from .llm.equivalence import EquivalenceJudge
from .llm.semantic import LLMSemanticJudge


class PhysicsRewardError(RuntimeError):
    """No reliable reward is available; the report distinguishes why."""

    def __init__(self, report: GradeReport, sample_id: str | None = None) -> None:
        self.report = report
        self.sample_id = sample_id
        super().__init__(
            f"Physics reward unresolved for {sample_id or report.problem_id}: "
            f"{report.grading_status}. {report.error_message or 'See grading report.'}"
        )

    def __reduce__(self):
        return type(self), (self.report, self.sample_id)


def load_problem(metadata: Mapping[str, Any]) -> GradingInput:
    """Load exactly one canonical problem from private training metadata."""
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    keys = [key for key in ("problem", "problem_path") if key in metadata]
    if len(keys) != 1:
        raise ValueError("metadata must provide exactly one of problem or problem_path")
    if keys[0] == "problem":
        payload = metadata["problem"]
    else:
        path = metadata["problem_path"]
        if not isinstance(path, str):
            raise TypeError("metadata.problem_path must be an absolute path string")
        source = Path(path)
        if not source.is_absolute():
            raise ValueError("metadata.problem_path must be absolute on every worker")
        with source.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError("the canonical problem must be a JSON object")
    return parse_grading_problem(payload)


def _judge_enabled() -> bool:
    value = os.getenv("DDPR_PHYSICS_EQUIVALENCE_JUDGE", "true").strip().casefold()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError("DDPR_PHYSICS_EQUIVALENCE_JUDGE must be true or false")
    return value in {"true", "1"}


@dataclass(frozen=True, slots=True)
class PhysicsReward(Reward):
    """Binary correctness reward; verifier errors are never converted to zero."""

    judge_enabled: bool = field(default_factory=_judge_enabled)
    equivalence_judge: EquivalenceJudge | None = None
    semantic_judge: LLMSemanticJudge | None = None

    def __call__(
        self,
        *,
        response: str,
        sample_id: str | None,
        reference_completion: str,
        metadata: Mapping[str, Any],
    ) -> float:
        # reference_completion is an unverified teacher output, not grading truth.
        report = grade_response(
            load_problem(metadata),
            response,
            judge_enabled=self.judge_enabled,
            equivalence_judge=self.equivalence_judge,
            semantic_judge=self.semantic_judge,
            attempt_id=sample_id or "student-response",
        )
        if report.grading_status == "pass":
            return 1.0
        if report.grading_status == "fail":
            return 0.0
        if report.grading_status == "not_graded" and report.extraction_status in {
            "missing_answer_function",
            "parse_failure",
            "invalid_answer_format",
            "unsafe_code",
            "runtime_failure",
        }:
            return 0.0
        raise PhysicsRewardError(report=report, sample_id=sample_id)
