from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..schemas import (
    FreestyleProblemGradingInput,
    SemanticJudgment,
)
from ..transport.client import complete_judge
from ..transport.config import (
    Completion,
    JudgeBackendConfig,
    load_dotenv,
    transport_from_env,
)
from .parsing import extract_json_object, normalize_result


class SemanticJudgeConfig(JudgeBackendConfig):
    model_config = ConfigDict(extra="forbid")

    model: str = "openai.gpt-5.6-sol"
    region: str = "us-east-1"
    max_tokens: int = Field(default=32000, ge=1, le=64000)
    reasoning_effort: str = "high"
    timeout_seconds: float = Field(default=1800.0, gt=0.0)

    @classmethod
    def from_env(cls) -> "SemanticJudgeConfig":
        load_dotenv()
        return cls(
            **transport_from_env("AUTO_CRITPT_GRADER_SEMANTIC_JUDGE"),
            model=os.getenv(
                "AUTO_CRITPT_GRADER_SEMANTIC_JUDGE_MODEL", "openai.gpt-5.6-sol"
            ),
            region=os.getenv("AUTO_CRITPT_GRADER_SEMANTIC_JUDGE_REGION", "us-east-1"),
            max_tokens=int(
                os.getenv("AUTO_CRITPT_GRADER_SEMANTIC_JUDGE_MAX_TOKENS", "32000")
            ),
            reasoning_effort=os.getenv(
                "AUTO_CRITPT_GRADER_SEMANTIC_JUDGE_REASONING_EFFORT",
                "high",
            ),
            timeout_seconds=float(
                os.getenv(
                    "AUTO_CRITPT_GRADER_SEMANTIC_JUDGE_TIMEOUT_SECONDS",
                    "1800",
                )
            ),
        )


class _JudgeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: str
    confidence: float = Field(ge=0.0, le=1.0)
    solver_answer_summary: str = Field(min_length=1)
    assessment: str = Field(min_length=1)
    substantive_errors: list[str] = Field(default_factory=list)


class LLMSemanticJudge:
    def __init__(
        self,
        *,
        config: SemanticJudgeConfig | None = None,
        completion: Completion | None = None,
    ) -> None:
        self.config = config or SemanticJudgeConfig.from_env()
        self._completion = completion

    def judge(
        self,
        problem: FreestyleProblemGradingInput,
        response: str,
        *,
        attempt_id: str = "saved-response",
    ) -> SemanticJudgment:
        if not response.strip():
            raise ValueError("SemanticJudge requires a completed Solver response")

        messages = build_semantic_judge_prompt(problem, response, attempt_id=attempt_id)
        current_messages = messages
        complete = self._completion or complete_judge
        last_error: Exception | None = None
        for schema_attempt in (1, 2):
            result = normalize_result(complete(current_messages, self.config))
            try:
                payload = extract_json_object(
                    result.content or "",
                    source="SemanticJudge response",
                )
                parsed_response = _JudgeResponse.model_validate(payload)
                if parsed_response.verdict not in {
                    "correct",
                    "incorrect",
                    "uncertain",
                }:
                    raise ValueError("verdict must be correct, incorrect, or uncertain")
            except Exception as exc:
                last_error = exc
                if schema_attempt == 2:
                    break
                current_messages = [
                    *messages,
                    {"role": "assistant", "content": result.content or ""},
                    {
                        "role": "user",
                        "content": (
                            "Your response did not match the requested JSON "
                            f"schema: {exc}. Return a corrected JSON object only."
                        ),
                    },
                ]
                continue

            return SemanticJudgment(
                attempt_id=attempt_id,
                verdict=parsed_response.verdict,
                confidence=parsed_response.confidence,
                solver_answer_summary=parsed_response.solver_answer_summary,
                assessment=parsed_response.assessment,
                substantive_errors=parsed_response.substantive_errors,
                model=result.model or self.config.model,
                raw_response=result.content or "",
                response_id=result.response_id,
                usage=result.usage,
                prompt_messages=current_messages,
            )
        raise ValueError(
            "SemanticJudge response remained invalid after one schema repair: "
            f"{last_error}"
        )


def build_semantic_judge_prompt(
    problem: FreestyleProblemGradingInput,
    response: str,
    *,
    attempt_id: str = "saved-response",
) -> list[dict[str, str]]:
    system = """You are Freestyle_CritPt SemanticJudge.

Judge whether one Solver response substantively answers the public theoretical-
physics problem. The reference solution is authoritative evidence but not a
required wording or derivation path. Accept mathematically and scientifically
equivalent answers, alternate notation, and alternate valid reasoning. Ignore
style and verbosity unless missing reasoning makes the claimed result impossible
to assess.

Judge every declared answer item and require all items to be correct. Within an
item, accept any mathematically equivalent representation of the requested
object. Check all answer-changing assumptions, regimes, branches, signs, units,
precision, and option selections requested by the problem. Extra intermediate
results in the reference are not additional required outputs. Conversely, do not
waive an explicitly requested result because its item groups several objects.
Use correct when the response is substantively correct and complete, incorrect
when it contains a concrete answer-changing error or omission, and uncertain
only when correctness or equivalence cannot be resolved from the supplied
comparison anchors. Do not diagnose, repair, or route possible defects in the
problem or reference; other audit stages own those tasks. Do not use tools or
demand a machine-oriented answer serialization.

Return one JSON object only with verdict, confidence, solver_answer_summary,
assessment, and substantive_errors."""
    payload: dict[str, Any] = {
        "public_task": problem.public_view(),
        "semantic_answer_type": problem.answer_type,
        "answer_items": [item.model_dump(mode="json") for item in problem.answer_items],
        "private_reference": {
            "reference_answer": problem.reference_answer,
            "reference_solution": problem.reference_solution,
        },
        "solver_attempt": {
            "attempt_id": attempt_id,
            "response": response,
        },
        "response_schema": {
            "verdict": "correct | incorrect | uncertain",
            "confidence": "number from 0 to 1",
            "solver_answer_summary": "concise summary",
            "assessment": "substantive comparison",
            "substantive_errors": ["concrete answer-changing error, if any"],
        },
    }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(payload, indent=2, ensure_ascii=False),
        },
    ]
