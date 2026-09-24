# Physics rewards

This module scores an already-generated student response against a canonical
native Auto_CritPt or Freestyle_CritPt problem. It does not run a solver, generate
graders, format free text into Python, audit problems, or modify training data.
The mathematical and Judge implementations are a pinned, grading-only snapshot
of Auto_CritPt_Grader 0.6.0; see [_core/UPSTREAM.md](_core/UPSTREAM.md).

## Input

Install `ddpr[physics]` and set
`DDPR_REWARD=ddpr.rewards.physics.PhysicsReward` for Slime or Swift.
Pass exactly one source in the existing sample metadata:

```python
metadata = {"problem": native_problem_json}
# Or, on shared storage mounted at the same path on every worker:
metadata = {"problem_path": "/mnt/nas/corpus/example/problem.json"}
```

Paths are read when scoring; keep those files immutable for reproducible runs.
Do not use teacher `reference_completion` as a fallback. That field is accepted
but never read by this reward or included in its Judge prompts. `sample_id` is
only an attempt identifier and need not equal the canonical problem's ID.

Both formats need nonempty `problem_id`, `problem_statement` and
`public_solver_output_instructions`. Auto additionally needs `answer_contract`
and `grading_plan`; Freestyle needs `answer_type`, `reference_answer`,
`reference_solution` and usually `answer_items`. Legacy Freestyle inputs without
items are normalized to one complete-answer item. Extra author/research fields
are ignored. Use native problem JSON, not the separately exported CritPt schema.

Private metadata must remain separate from the public student prompt. The
existing Slime/Swift adapters preserve this separation; no backend API changes
are necessary. Exporters still need to attach the canonical problem to metadata;
this module does not infer it from teacher text or a sample ID.

## Scoring

| Strategy | Input response | Comparison |
| --- | --- | --- |
| floating | Fenced Python `answer()` | Finite float literal; problem tolerances |
| exact | Fenced Python `answer()` | Exact scalar equivalence |
| symbolic | Fenced Python `answer(...)` | Symbolic equivalence with declared variables/assumptions |
| categorical | Fenced Python `answer()` | Choice or choice set, including declared aliases |
| semantic | Full natural-language response | Freestyle SemanticJudge, all requested items |

Auto retains the original restricted answer-code execution and shared local
extraction/comparison deadline. It does not freely execute submitted Python.
Readable unsupported SymPy expressions may go to the expression Judge as inert
text. Dangerous/malformed code is still rejected. The exact/symbolic Judge gets
the public task, submitted result and expected expression, and must not solve
the task or fill in a missing answer. Exact numeric cross-checks use 16 digits as
conflict evidence, not a replacement for symbolic proof.

The five named modules expose the existing comparison/Judge implementations;
they do not duplicate algorithms. Low-level `grade_exact`, `grade_symbolic`, etc.
operate on extracted values. Use `grade_response` for untrusted raw responses so
the normal extraction, timeout and optional Judge handling are applied.

Auto composite/multiple grading targets raise `ValueError` for now. Freestyle's
multiple natural-language answer items are supported and are not Auto composite.
There is no partial-credit policy added by this integration.

## Results and errors

`PhysicsReward` returns a float: 1.0 for pass, 0.0 for fail or an invalid student
answer. If the grading core cannot determine correctness, it raises
`PhysicsRewardError`; its `.report` is a serializable Pydantic `GradeReport` and
`.sample_id` identifies the affected attempt. The exception is pickle-safe.
Bad/missing problem metadata and configuration errors propagate separately.
Neither adapter catches these exceptions or converts them to zero. A training
controller must explicitly retry, skip, or stop on them.

For diagnostics outside the trainer:

```python
from ddpr.rewards.physics import grade_response, load_problem

problem = load_problem(metadata)
report = grade_response(problem, student_response)
print(report.reward)  # 0, 1, or None if unresolved
print(report.model_dump_json(indent=2))
```

Reports contain private references/Judge inputs and student text. Store them as
private evaluation artifacts, not as future policy prompts. The reward itself
does not write shared files or keep a mutable last-report buffer.

## Judge configuration

Defaults match the standalone grader. Both Judges use Bedrock
`openai.gpt-5.6-sol`, high reasoning, `us-east-1`; configure explicitly if your
account/model availability differs. Credentials come from the worker environment
or a local `.env`, never from problem metadata.

- Bedrock: `AWS_BEDROCK_API_KEY` (legacy `AWS_BEARER_TOKEN_BEDROCK` also supported).
- Auto fallback: `AUTO_CRITPT_GRADER_JUDGE_*` (16k output tokens, 600s by default).
- Freestyle: `AUTO_CRITPT_GRADER_SEMANTIC_JUDGE_*` (32k tokens, 1800s by default).
- Shared suffixes: `BACKEND`, `MODEL`, `REGION`, `MAX_TOKENS`, `REASONING_EFFORT`,
  `TIMEOUT_SECONDS`, `BASE_URL`, `API_KEY_ENV`, `HTTP_MAX_ATTEMPTS`,
  `HTTP_BACKOFF_SECONDS`, `EXTRA_BODY_JSON`.
- Auto confidence threshold: `AUTO_CRITPT_GRADER_JUDGE_MIN_CONFIDENCE` (0.90).
- `DDPR_PHYSICS_EQUIVALENCE_JUDGE=false` disables only Auto's fallback. An
  unsupported expression without a Judge remains unresolved, not a wrong answer.
  Freestyle still uses SemanticJudge.
- Local limits: `AUTO_CRITPT_GRADER_TIMEOUT_SECONDS` (120s) and
  `AUTO_CRITPT_NUMERIC_CHECK_TIMEOUT_SECONDS` (30s).

The original environment names are retained for configuration parity. For an
OpenAI-compatible Judge, set `*_BACKEND=openai`, `*_MODEL`, `*_BASE_URL`, and
`*_API_KEY_ENV` naming the environment variable containing that service's key.
Provider-specific OpenAI reasoning options can be supplied through
`*_EXTRA_BODY_JSON`; the dedicated `REASONING_EFFORT` setting is used by Bedrock.
Set both prefixes when switching both Judges. No LLM call is made on module
import or reward construction; Auto deterministic passes do not call a Judge.

## Verification

```bash
python -m pip install -e '.[dev,physics]'
python -m pytest -q tests/rewards tests/test_rewards.py
python -m ruff check .
python -m ruff format --check .
```

Tests use mocked Judge completions/transports. They cover wrong answers versus
unresolved failures, teacher isolation, legacy inputs, inert SymPy fallback,
native threading deadlines, the Slime reward adapter, and Swift's adapter with a
mocked framework API. Full framework/GPU
training and live Judge quality are separate checks, not implied by these tests.

Integration validation on 2026-09-24: the full CPU suite passed (119 tests;
29 optional framework/environment checks skipped), and Ruff lint/format checks
passed. A read-only sample of 110 native Auto/Freestyle problems across v1, v2,
v2 pinned/guidance and v3 loaded successfully. Eight saved Auto responses from
v2/v3, covering all four primitives, produced identical full grading reports to
the upstream implementation with live judging disabled. The retained 116 core
function/class ASTs also matched upstream. A wheel build included the grading
core and its license without depending on the standalone grader package.
