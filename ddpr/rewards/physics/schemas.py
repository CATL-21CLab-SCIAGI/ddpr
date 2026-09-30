from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

PrimitiveName = Literal["floating_numeric", "exact", "symbolic", "categorical"]


SUPPORTED_GRADING_ARGS: dict[PrimitiveName, frozenset[str]] = {
    "floating_numeric": frozenset({"abs_tol", "rel_tol"}),
    "exact": frozenset(),
    "symbolic": frozenset(
        {
            "domain_constraints",
            "variables",
            "variable_assumptions",
            "equivalent_up_to_constant_factor",
        }
    ),
    "categorical": frozenset(
        {
            "allowed_aliases",
            "allowed_choices",
            "case_sensitive",
            "choices",
        }
    ),
}


def unsupported_grading_args(
    primitive: PrimitiveName,
    args: dict[str, Any],
) -> list[str]:
    """Return grading-call argument names the primitive does not implement."""

    return sorted(set(args) - SUPPORTED_GRADING_ARGS[primitive])


class PrecisionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["significant_digits", "decimal_places"]
    digits: int = Field(ge=0, le=15)

    @model_validator(mode="after")
    def validate_digits(self) -> Self:
        if self.kind == "significant_digits" and self.digits < 1:
            raise ValueError("significant_digits precision must use at least one digit")
        return self


class GradingCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primitive: PrimitiveName
    target: str
    expected: Any
    args: dict[str, Any] = Field(default_factory=dict)
    precision: PrecisionSpec | None = None
    weight: float = 1.0
    required: bool = True

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        unsupported_args = unsupported_grading_args(self.primitive, self.args)
        if unsupported_args:
            raise ValueError(
                f"{self.primitive} grading call has unsupported args: "
                + ", ".join(unsupported_args)
            )
        if self.primitive == "floating_numeric" and self.precision is None:
            raise ValueError("floating_numeric grading calls require precision")
        if self.primitive != "floating_numeric" and self.precision is not None:
            raise ValueError(
                "precision is only valid for floating_numeric grading calls"
            )
        return self

    @model_serializer(mode="wrap")
    def serialize_call(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, Any]:
        data = handler(self)
        if self.precision is None:
            data.pop("precision", None)
        return data


class GradingPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calls: list[GradingCall] = Field(min_length=1)
    composition: Literal["all_required", "weighted"] = "all_required"


class PrimitiveGrade(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primitive: PrimitiveName
    target: str
    passed: bool
    score: float
    reason: str = ""
    equivalence_candidate: bool = False
    deterministic_skip_reason: str | None = None


class ExactNumericComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str
    status: Literal["match", "different", "unavailable"]
    significant_digits: int = Field(default=16, ge=1, le=50)
    actual_decimal: str | None = None
    expected_decimal: str | None = None
    relative_difference: str | None = None
    reason: str


class GradingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    score: float
    calls: list[PrimitiveGrade]


@dataclass(frozen=True)
class UnexecutedAnswer:
    expression: str
    answer_code: str
    reason: str

    def __str__(self) -> str:
        return self.expression


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
ProviderCallStatus = Literal[
    "success",
    "retryable_error",
    "non_retryable_error",
    "time_limit",
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
        if isinstance(value, dict) and "answer_items" not in value:
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


class ProviderCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_index: int = Field(ge=1)
    status: ProviderCallStatus
    latency_seconds: float = Field(ge=0.0)
    http_status: int | None = None
    request_id: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class CompletionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str | None = None
    reasoning_content: str | None = None
    model: str | None = None
    response_id: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    provider: str | None = None
    latency_seconds: float | None = Field(default=None, ge=0.0)
    provider_calls: list[ProviderCallRecord] = Field(default_factory=list)


class EquivalenceJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primitive: Literal["exact", "symbolic"]
    target: str
    verdict: Literal["equivalent", "not_equivalent", "uncertain"]
    resolution: Literal["equivalent", "not_equivalent", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    assumptions_used: list[str] = Field(default_factory=list)
    model: str
    raw_response: str
    response_id: str | None = None
    usage: dict[str, Any] | None = None


class SemanticJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt_id: str
    verdict: Literal["correct", "incorrect", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    solver_answer_summary: str
    assessment: str
    substantive_errors: list[str] = Field(default_factory=list)
    model: str
    raw_response: str
    response_id: str | None = None
    usage: dict[str, Any] | None = None
    prompt_messages: list[dict[str, str]] = Field(min_length=1)


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
