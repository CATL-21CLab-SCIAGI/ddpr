from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
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


def load_problem(task_dir: str | Path) -> GradingInput:
    """Adapt a compiled task to private grading input without executing task code.

    Preserve the original reference problem ID; the compiled task ID identifies
    generation trials. Keep the returned private fields out of student prompts.
    """
    from ddsr_bench.benchmarks.critpt.data.schemas import AIProblemSpec, CritPtSource
    from ddsr_bench.benchmarks.critpt.data.utils import require_text
    from ddsr_bench.benchmarks.critpt.evaluation.prepare import decode_instruction
    from ddsr_bench.benchmarks.critpt.generation.prompts import load_prompts
    from ddsr_bench.benchmarks.utils import read_json

    task = Path(task_dir)
    problem = decode_instruction((task / "instruction.md").read_text(encoding="utf-8"))
    if not isinstance(problem, AIProblemSpec):
        raise TypeError("AI judging requires a critpt-ai task")
    path = task / "tests" / "reference.json"
    reference = read_json(path, allow_nan=False)
    instructions = problem.answer_instructions
    if not instructions or not instructions.strip():
        instructions = load_prompts(CritPtSource.AI)["ANSWER_INSTRUCTIONS"]
    if problem.code_template:
        instructions += f"\n\n```python\n{problem.code_template.rstrip()}\n```"
    record = {
        "problem_id": require_text(reference, "problem_id", path),
        "problem_statement": problem.statement,
        "public_solver_output_instructions": instructions,
    }
    if problem.grader == "rule":
        for key in ("answer_contract", "grading_plan"):
            value = reference.get(key)
            if not isinstance(value, dict):
                raise TypeError(f"{path} requires an object at {key!r}")
            record[key] = value
    elif problem.grader == "llm":
        for key in ("answer_type", "reference_solution"):
            record[key] = require_text(reference, key, path)
        record["reference_answer"] = require_text(reference, "snippet", path)
        items = reference.get("answer_items")
        if not isinstance(items, list) or not items:
            raise ValueError(f"{path} requires nonempty answer_items")
        record["answer_items"] = items
    else:
        raise ValueError(f"unknown answer grader: {problem.grader}")
    return parse_grading_problem(record)


def load_problems(task_dirs: Sequence[str | Path]) -> list[GradingInput]:
    """Load selected tasks in caller order, retaining duplicates and validation errors."""
    if isinstance(task_dirs, (str, bytes, Path)):
        raise TypeError("task_dirs must be a sequence of task directory paths")
    return [load_problem(task_dir) for task_dir in task_dirs]
