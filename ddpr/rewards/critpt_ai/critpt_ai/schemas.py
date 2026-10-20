"""Native CritPt-AI problem contracts and grading reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ..schemas import (
    EquivalenceJudgment,
    ExactNumericComparison,
    GradingPlan,
    PrecisionSpec,
    PrimitiveName,
    SemanticJudgment,
)

AnswerType = Literal[
    "floating_numeric",
    "exact",
    "symbolic",
    "categorical",
    "composite",
]


ProblemFormat = Literal["auto", "freestyle"]


ProblemFormatSelection = Literal["detect", "auto", "freestyle"]


SemanticAnswerType = Literal["floating_numeric", "exact", "symbolic", "categorical"]


ExtractionStatus = Literal[
    "not_attempted",
    "not_applicable",
    "success",
    "text_only",
    "missing_answer_function",
    "parse_failure",
    "invalid_answer_format",
    "unsafe_code",
    "runtime_failure",
]


GradingStatus = Literal[
    "not_graded",
    "pass",
    "fail",
    "grader_error",
    "grader_uncertain",
    "resource_limit",
    "infrastructure_error",
]


class PublicProblemInput(BaseModel):
    """The only Problem fields that may reach a Solver model."""

    model_config = ConfigDict(extra="ignore")

    problem_id: str
    problem_statement: str
    public_solver_output_instructions: str

    @field_validator(
        "problem_id",
        "problem_statement",
        "public_solver_output_instructions",
    )
    @classmethod
    def validate_nonempty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("public problem text fields must be non-empty")
        return normalized


class AnswerPartContractSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    answer_type: PrimitiveName
    expected_precision: PrecisionSpec | None = None


class AnswerContractSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    answer_type: AnswerType
    parts: dict[str, AnswerPartContractSummary] = Field(default_factory=dict)


class ProblemGradingInput(PublicProblemInput):
    """Minimal private Problem projection needed for grading."""

    answer_contract: AnswerContractSummary
    grading_plan: GradingPlan

    @model_validator(mode="after")
    def validate_explicit_composite_parts(self) -> Self:
        # Older minimal grading inputs have only an answer type and a plan.
        parts = self.answer_contract.parts
        if not parts:
            return self
        targets = ("part_a", "part_b", "part_c")
        if (
            self.answer_type != "composite"
            or len(parts) not in {2, 3}
            or set(parts) != set(targets[: len(parts)])
        ):
            raise ValueError(
                "composite requires 2-3 flat part_a/part_b/part_c contracts"
            )
        calls = self.grading_plan.calls
        if len(calls) != len(parts) or {call.target for call in calls} != set(parts):
            raise ValueError("composite contracts and grading calls must match exactly")
        if self.grading_plan.composition != "all_required" or any(
            not call.required or call.weight != 1 for call in calls
        ):
            raise ValueError("composite requires equal-weight all_required grading")
        for call in calls:
            part = parts[call.target]
            if (
                part.answer_type != call.primitive
                or part.expected_precision != call.precision
            ):
                raise ValueError(
                    f"{call.target} contract type/precision differs from grading call"
                )
        return self

    @property
    def answer_type(self) -> AnswerType:
        return self.answer_contract.answer_type

    @property
    def problem_format(self) -> ProblemFormat:
        return "auto"


class AnswerItem(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    item_id: str = Field(min_length=1)
    answer_type: SemanticAnswerType
    target: str = Field(min_length=1)


class FreestyleProblemGradingInput(PublicProblemInput):
    """Only public task and private comparison anchors, without author artifacts."""

    answer_type: SemanticAnswerType
    answer_items: list[AnswerItem] = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    reference_solution: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_items(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and "answer_items" not in value:
            value = {
                **value,
                "answer_items": [
                    {
                        "item_id": "answer",
                        "answer_type": value.get("answer_type"),
                        "target": "The complete final result requested by the problem.",
                    }
                ],
            }
        return value

    @model_validator(mode="after")
    def validate_items(self) -> Self:
        ids = [item.item_id for item in self.answer_items]
        if len(ids) != len(set(ids)):
            raise ValueError("answer item ids must be unique")
        if self.answer_type not in {item.answer_type for item in self.answer_items}:
            raise ValueError("answer_type must occur in answer_items")
        if not self.reference_answer.strip() or not self.reference_solution.strip():
            raise ValueError("reference answer and solution must be non-empty")
        return self

    @property
    def problem_format(self) -> ProblemFormat:
        return "freestyle"

    def public_view(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "problem_statement": self.problem_statement,
            "answer_type": self.answer_type,
            "answer_items": [
                item.model_dump(mode="json") for item in self.answer_items
            ],
            "public_solver_output_instructions": self.public_solver_output_instructions,
        }


GradingInput = ProblemGradingInput | FreestyleProblemGradingInput


class GradeReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem_id: str
    problem_format: ProblemFormat = "auto"
    answer_type: AnswerType
    raw_response: str
    response_sha256: str
    extracted_answer: Any | None = None
    extraction_status: ExtractionStatus
    grading_status: GradingStatus
    candidate_answer_code: str | None = None
    deterministic_skip_reasons: dict[str, str] = Field(default_factory=dict)
    passed: bool
    semantic_judgment: SemanticJudgment | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    deterministic_grading_results: list[dict[str, Any]] = Field(default_factory=list)
    equivalence_judgments: list[EquivalenceJudgment] = Field(default_factory=list)
    exact_numeric_comparisons: list[ExactNumericComparison] = Field(
        default_factory=list
    )
    grading_results: list[dict[str, Any]] = Field(default_factory=list)
    error_message: str | None = None
