"""Map CritPt-AI grading results to DDPR training rewards."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ddsr_bench.benchmarks.critpt.data.loader import load_ai_task
from ddsr_bench.benchmarks.critpt.data.schemas import AIProblem
from ddsr_bench.benchmarks.critpt.evaluation.ai.grader import grade_response
from ddsr_bench.benchmarks.critpt.evaluation.ai.io.inputs import (
    parse_grading_problem,
)
from ddsr_bench.benchmarks.critpt.evaluation.ai.io.schemas import (
    GradeReport,
    GradingInput,
)
from ddsr_bench.benchmarks.critpt.evaluation.ai.llm.equivalence import (
    EquivalenceJudge,
)
from ddsr_bench.benchmarks.critpt.evaluation.ai.llm.semantic import (
    LLMSemanticJudge,
)

from ddpr.rewards.base import Reward


class CritptAiRewardError(RuntimeError):
    """No reliable reward is available; the report distinguishes why."""

    def __init__(self, report: GradeReport, sample_id: str | None = None) -> None:
        self.report = report
        self.sample_id = sample_id
        super().__init__(
            f"CritPt-AI reward unresolved for {sample_id or report.problem_id}: "
            f"{report.grading_status}. {report.error_message or 'See grading report.'}"
        )

    def __reduce__(self):
        return type(self), (self.report, self.sample_id)


def load_problem(metadata: Mapping[str, Any]) -> AIProblem | GradingInput:
    """Load a prepared task, or a native Auto/Freestyle compatibility record."""
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    keys = [key for key in ("problem", "problem_path", "task_path") if key in metadata]
    if len(keys) != 1:
        raise ValueError(
            "metadata must provide exactly one of problem, problem_path, or task_path"
        )
    if keys[0] == "problem":
        payload = metadata["problem"]
    else:
        key = keys[0]
        path = metadata[key]
        if not isinstance(path, str):
            raise TypeError(f"metadata.{key} must be an absolute path string")
        source = Path(path)
        if not source.is_absolute():
            raise ValueError(f"metadata.{key} must be absolute on every worker")
        if key == "task_path":
            return load_ai_task(source)
        with source.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError("the canonical problem must be a JSON object")
    return parse_grading_problem(payload)


def _llm_fallback() -> bool:
    value = os.getenv("DDPR_CRITPT_AI_EQUIVALENCE_JUDGE", "true").strip().casefold()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError("DDPR_CRITPT_AI_EQUIVALENCE_JUDGE must be true or false")
    return value in {"true", "1"}


@dataclass(frozen=True, slots=True)
class CritptAiReward(Reward):
    """Binary correctness reward; verifier errors are never converted to zero."""

    llm_fallback: bool = field(default_factory=_llm_fallback)
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
            llm_fallback=self.llm_fallback,
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
        raise CritptAiRewardError(report=report, sample_id=sample_id)
