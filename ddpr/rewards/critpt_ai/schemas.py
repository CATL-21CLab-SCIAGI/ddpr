from __future__ import annotations

import math
from collections.abc import Mapping
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
    weight: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    required: bool = True

    @field_validator("target")
    @classmethod
    def validate_target(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("grading target must be non-empty")
        return value

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

    @model_validator(mode="after")
    def validate_weights(self) -> Self:
        total = sum(call.weight for call in self.calls)
        if not math.isfinite(total) or total <= 0:
            raise ValueError("grading weights must have a finite positive total")
        return self


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


ProviderCallStatus = Literal[
    "success",
    "retryable_error",
    "non_retryable_error",
    "time_limit",
]


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


class EquivalenceResponse(BaseModel):
    """LLM-supplied assessment fields, without request metadata."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["equivalent", "not_equivalent", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    assumptions_used: list[str] = Field(default_factory=list)

    @field_validator("verdict", "rationale", mode="before")
    @classmethod
    def strip_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("assumptions_used", mode="before")
    @classmethod
    def normalize_assumptions_used(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, Mapping):
            return [
                f"{key}: {item}"
                for key, item in sorted(
                    value.items(),
                    key=lambda pair: str(pair[0]),
                )
            ]
        return value

    @field_validator("assumptions_used")
    @classmethod
    def strip_assumptions(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value]


class EquivalenceJudgment(EquivalenceResponse):
    primitive: Literal["exact", "symbolic"]
    target: str
    resolution: Literal["equivalent", "not_equivalent", "uncertain"]
    model: str
    raw_response: str
    response_id: str | None = None
    usage: dict[str, Any] | None = None


class SemanticResponse(BaseModel):
    """LLM-supplied assessment fields, without request metadata."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["correct", "incorrect", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    solver_answer_summary: str = Field(min_length=1)
    assessment: str = Field(min_length=1)
    substantive_errors: list[str] = Field(default_factory=list)

    @field_validator("verdict", "solver_answer_summary", "assessment", mode="before")
    @classmethod
    def strip_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("substantive_errors")
    @classmethod
    def strip_errors(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value]


class SemanticJudgment(SemanticResponse):
    attempt_id: str
    model: str
    raw_response: str
    response_id: str | None = None
    usage: dict[str, Any] | None = None
    prompt_messages: list[dict[str, str]] = Field(min_length=1)
