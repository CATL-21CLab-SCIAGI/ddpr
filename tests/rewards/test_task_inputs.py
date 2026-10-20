import json
from dataclasses import replace

import pytest

pytest.importorskip("ddsr_bench")
pytest.importorskip("pydantic")

from ddsr_bench.benchmarks.critpt.data.schemas import (
    AIAnswerSpec,
    AIProblem,
    AIProblemSpec,
    CritPtSource,
)
from ddsr_bench.benchmarks.critpt.evaluation.prepare import compile_problem
from ddsr_bench.benchmarks.critpt.generation.prompts import load_prompts
from ddsr_bench.benchmarks.utils import Resources

from ddpr.rewards.critpt_ai.critpt_ai.inputs import load_problem, load_problems


@pytest.fixture(params=["rule", "llm"])
def prepared(tmp_path, request, monkeypatch):
    import sys

    # The temporary sibling judge adapter must never be a dependency.
    monkeypatch.setitem(
        sys.modules, "ddsr_bench.benchmarks.critpt.evaluation.judge", None
    )
    from ddsr_bench.benchmarks.critpt.generation import prompts

    # Reject dependencies on the experimental instruction helper too.
    def forbidden(*args, **kwargs):
        raise AssertionError("Do not use the temporary answer_instructions helper")

    monkeypatch.setattr(prompts, "answer_instructions", forbidden, raising=False)
    rule = request.param == "rule"
    spec = AIProblemSpec(
        id="ai-prepared-id",
        type="main",
        index=None,
        statement="Find the value.",
        code_template="def answer():\n    return ...\n" if rule else "",
        source=CritPtSource.AI,
        source_path=tmp_path / "missing" / "problem.json",
        grader=request.param,
        answer_instructions=None,
    )
    answer = AIAnswerSpec(
        code="",
        snippet="SECRET",
        testcases=None,
        problem_id="native-id",
        reference_solution="SECRET_SOLUTION",
        reasoning_plan={"unused": "PRIVATE"},
        source_ids=["PRIVATE"],
        answer_contract={"answer_type": "exact"} if rule else None,
        grading_plan=(
            {
                "calls": [
                    {"primitive": "exact", "target": "answer", "expected": "SECRET"}
                ]
            }
            if rule
            else None
        ),
        answer_type=None if rule else "exact",
        answer_items=(
            None
            if rule
            else [{"item_id": "value", "answer_type": "exact", "target": "value"}]
        ),
    )
    task = compile_problem(
        AIProblem(spec, answer), tmp_path / "tasks", Resources("unused", 1, 1024, 120)
    )
    return task, spec


def test_input(prepared):
    task, spec = prepared
    before = (task / "instruction.md").read_bytes()
    record = load_problem(task).model_dump()
    assert record["problem_id"] == "native-id"
    assert record["problem_statement"] == spec.statement
    instructions = record["public_solver_output_instructions"]
    assert instructions.startswith(load_prompts(CritPtSource.AI)["ANSWER_INSTRUCTIONS"])
    assert "SECRET" not in instructions
    if spec.code_template:
        assert spec.code_template.rstrip() in instructions
    assert "PRIVATE" not in json.dumps(record)
    assert (task / "instruction.md").read_bytes() == before
    assert not spec.source_path.exists()  # No dependency on the original corpus.
    common = {"problem_id", "problem_statement", "public_solver_output_instructions"}
    if spec.grader == "rule":
        assert common | {"answer_contract", "grading_plan"} <= set(record)
        assert record["grading_plan"]["calls"][0]["expected"] == "SECRET"
    else:
        assert common | {
            "answer_type",
            "answer_items",
            "reference_answer",
            "reference_solution",
        } <= set(record)
        assert record["reference_answer"] == "SECRET"
    custom = replace(spec, answer_instructions="Custom output contract.")
    from ddsr_bench.benchmarks.critpt.evaluation.prepare import encode_instruction

    (task / "instruction.md").write_text(encode_instruction(custom))
    assert load_problem(task).public_solver_output_instructions.startswith("Custom")


def test_problem_list(prepared, tmp_path):
    import shutil

    task, _ = prepared
    other = tmp_path / "second-task"
    shutil.copytree(task, other)
    reference = other / "tests/reference.json"
    data = json.loads(reference.read_text())
    data["problem_id"] = "second-id"
    reference.write_text(json.dumps(data))
    problems = load_problems([other, str(task), other])
    assert [problem.problem_id for problem in problems] == [
        "second-id",
        "native-id",
        "second-id",
    ]
    assert problems[1] == load_problem(task)
    assert load_problems([]) == []
    with pytest.raises(TypeError, match="sequence"):
        load_problems(str(task))
    reference.write_text("{}")
    with pytest.raises(ValueError):
        load_problems([task, other])


def test_missing_reference(prepared):
    task, _ = prepared
    (task / "tests/reference.json").unlink()
    with pytest.raises(ValueError, match="cannot read"):
        load_problem(task)


def test_invalid_reference(prepared):
    task, spec = prepared
    path = task / "tests/reference.json"
    data = json.loads(path.read_text())
    data.pop("grading_plan" if spec.grader == "rule" else "answer_items")
    path.write_text(json.dumps(data))
    with pytest.raises((TypeError, ValueError)):
        load_problem(task)


def test_official(prepared):
    task, _ = prepared
    path = task / "instruction.md"
    data = json.loads(path.read_text())
    data.update(source="critpt-official")
    data.pop("answer_instructions")
    path.write_text(json.dumps(data))
    with pytest.raises(TypeError, match="critpt-ai"):
        load_problem(task)
