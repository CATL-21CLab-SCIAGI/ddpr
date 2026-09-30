from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import Field

from ..schemas import SemanticJudgment, SemanticResponse
from .clients.client import complete_judge
from .clients.config import (
    ClientConfig,
    Completion,
    client_from_env,
)
from .utils import extract_json_object, normalize_result

SEMANTIC_ENV_PREFIX = "AUTO_CRITPT_GRADER_SEMANTIC_JUDGE"


class SemanticJudgeConfig(ClientConfig):
    max_tokens: int = Field(default=32000, ge=1, le=64000)
    timeout_seconds: float = Field(default=1800.0, gt=0.0, allow_inf_nan=False)

    @classmethod
    def from_env(cls) -> "SemanticJudgeConfig":
        return cls(**client_from_env(SEMANTIC_ENV_PREFIX))


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
        answer: str,
        *,
        public_problem: Mapping[str, Any],
        reference_answer: str,
        reference_solution: str,
        attempt_id: str = "saved-response",
    ) -> SemanticJudgment:
        """Assess an answer against reference text.

        public_problem supplies public task fields, including answer_type and
        answer_items; native format normalization belongs to the caller.
        """
        if not answer.strip():
            raise ValueError("SemanticJudge requires a completed Solver response")

        messages = _messages(
            answer,
            public_problem=public_problem,
            reference_answer=reference_answer,
            reference_solution=reference_solution,
            attempt_id=attempt_id,
        )
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
                parsed_response = SemanticResponse.model_validate(payload)
            except (ValueError, TypeError) as exc:
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
                **parsed_response.model_dump(),
                attempt_id=attempt_id,
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


def _messages(
    answer: str,
    *,
    public_problem: Mapping[str, Any],
    reference_answer: str,
    reference_solution: str,
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
        "public_task": dict(public_problem),
        "semantic_answer_type": public_problem["answer_type"],
        "answer_items": public_problem["answer_items"],
        "private_reference": {
            "reference_answer": reference_answer,
            "reference_solution": reference_solution,
        },
        "solver_attempt": {
            "attempt_id": attempt_id,
            "response": answer,
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
