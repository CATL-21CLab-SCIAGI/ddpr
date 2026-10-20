# CritPt-AI rewards

This module scores an already-generated student response against a canonical
native Auto_CritPt or Freestyle_CritPt problem. It does not run a solver, generate
graders, format free text into Python, audit problems, or modify training data.
The mathematical and Judge implementations are a pinned, grading-only snapshot
of Auto_CritPt_Grader 0.6.0; see [UPSTREAM.md](UPSTREAM.md).

## Input

Install `ddpr[critpt-ai]` and set
`DDPR_REWARD=ddpr.rewards.critpt_ai.CritptAiReward` for Slime or Swift.
Pass exactly one source in the existing sample metadata:

```python
metadata = {"problem": native_problem_json}
# Or, on shared storage mounted at the same path on every worker:
metadata = {"problem_path": "/mnt/nas/corpus/example/problem.json"}
# Or a compiled Harbor CritPt-AI task, using DDSR-Bench:
metadata = {"task_path": "/mnt/nas/tasks/critpt-ai/example"}
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

Rule comparisons live in `rule/`, with dispatch in `compare.py` and mathematical
parsing and result helpers in `utils.py`. Shared grading plans and assessment
models live in `schemas.py`. Answer extraction and execution live in `answer.py`,
checks in `validation.py`, and deadlines in `runtime.py`. LLM policies live in
`llm/`, with provider calls in `llm/clients/`. `critpt_ai/` owns native problem
formats, parsing and report formatting. `grader.py` coordinates assessments.
Training metadata loading and numerical reward mapping stay in `reward.py`.
Low-level comparisons operate on extracted values. Use `grade_response` for raw
responses to apply extraction, timeouts and optional Judge handling. These code
restrictions and deadlines do not provide container isolation.

Rule functions use `grade_*(answer, target, **kwargs)`, for example
`grade_symbolic(answer, target, variables=["x"])`. Here `target` is the reference
value. The serialized plan retains `expected` for that value and `target` for the
answer-field name; dispatch unpacks its `args` dictionary as keyword options.
The graders are `grade_exact_numeric`, `grade_floating_numeric`, `grade_symbolic`
and `grade_categorical`. `evaluate_numeric_agreement` provides supporting numerical
evidence only; it does not assign a grade.

Auto composite/multiple grading targets raise `ValueError` for now. Freestyle's
multiple natural-language answer items are supported and are not Auto composite.
There is no partial-credit policy added by this integration.

## Results and errors

`CritptAiReward` returns a float: 1.0 for pass, 0.0 for fail or an invalid student
answer. Resource limits remain unresolved. If the grading core cannot determine
correctness, it raises `CritptAiRewardError`; its `.report` is a serializable Pydantic `GradeReport` and
`.sample_id` identifies the affected attempt. The exception is pickle-safe.
Invalid problem metadata raises input-validation errors. Grader failures appear
in the report; errors configuring the reward itself propagate directly.
Neither adapter catches these exceptions or converts them to zero. A training
controller must explicitly retry, skip, or stop on them.

For diagnostics outside the trainer:

`grader.grade_response` returns assessments; `CritptAiReward` owns training score
mapping. Use `grading_status` and `extraction_status` for diagnostics.

```python
from ddpr.rewards.critpt_ai import grade_response, load_problem

problem = load_problem(metadata)
report = grade_response(problem, student_response)
print(report.grading_status)  # pass, fail, or a diagnostic status
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
- `DDPR_CRITPT_AI_EQUIVALENCE_JUDGE=false` disables only Auto's fallback. An
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

For explicit configuration, import `LLMEquivalenceJudge` and
`EquivalenceJudgeConfig` from `ddpr.rewards.critpt_ai.llm.equivalence`, or
`LLMSemanticJudge` and `SemanticJudgeConfig` from
`ddpr.rewards.critpt_ai.llm.semantic`, and pass the configured instance to
`CritptAiReward(equivalence_judge=...)` or `CritptAiReward(semantic_judge=...)`.

Both judges accept `completion=complete`, where the synchronous callable has the
signature `complete(messages, config) -> str | CompletionResult`. It replaces
provider calls without changing prompts, parsing or grading policy. Return
`CompletionResult` from `critpt_ai.schemas` to retain model, request ID and usage.
The caller owns any injected client's lifecycle.

The added `ddsr` backend uses DDSR-Bench generation clients through the normal
client dispatcher. Existing `openai` and `bedrock` behavior remains unchanged.
Install the sibling checkout and optional dependencies:

```bash
python -m pip install -e ../ddsr-bench -e '.[critpt-ai,ddsr]'
```

For example, configure the Freestyle judge on every worker:

```bash
export AUTO_CRITPT_GRADER_SEMANTIC_JUDGE_BACKEND=ddsr
export AUTO_CRITPT_GRADER_SEMANTIC_JUDGE_PROVIDER=vllm
export AUTO_CRITPT_GRADER_SEMANTIC_JUDGE_MODEL=your-judge-model
export AUTO_CRITPT_GRADER_SEMANTIC_JUDGE_BASE_URL=http://localhost:8000/v1
export AUTO_CRITPT_GRADER_SEMANTIC_JUDGE_API_KEY_ENV=
export AUTO_CRITPT_GRADER_SEMANTIC_JUDGE_REASONING_EFFORT=
```

Use the `AUTO_CRITPT_GRADER_JUDGE_` prefix for the equivalence judge.
With `BACKEND=ddsr`, `PROVIDER` selects `vllm`, `openai`, `aliyun` or `bedrock`.
DDSR-Bench owns HTTP requests, response parsing and retries; DDPR bridges its
async client to synchronous grading and keeps reasoning separate from visible
judgments. Each request closes its client. `HTTP_BACKOFF_SECONDS` does not apply
to this backend, and nonempty `EXTRA_BODY_JSON` is rejected. JSON judgments
are requested through prompts, without forcing provider-specific JSON mode.

`BACKEND=bedrock` retains its Responses API implementation;
`BACKEND=ddsr` with `PROVIDER=bedrock` uses DDSR-Bench Chat Completions.
The external Auto-CritPt problem/report interfaces remain unchanged.

For `task_path`, DDPR adapts the compiled instructions and private references
in `critpt_ai/inputs.py`, using DDSR-Bench's task decoder and prompt definitions.
It does not execute corpus code or require the original corpus. For offline
grading, `critpt_ai.inputs.load_problems(task_dirs)` loads an explicit list of
task directories in order; each entry uses the same validation.

## Verification

```bash
python -m pip install -e '.[dev,critpt-ai]'
python -m pytest -q tests/rewards tests/test_rewards.py
python -m ruff check .
python -m ruff format --check .
```

Tests use mocked Judge completions/transports. They cover wrong answers versus
unresolved failures, teacher isolation, legacy inputs, inert SymPy fallback,
native threading deadlines, the Slime reward adapter, and Swift's adapter with a
mocked framework API. Full framework/GPU training and live Judge quality are separate checks, not implied by these tests.

With the sibling DDSR-Bench checkout available, test compiled CritPt-AI inputs:

```bash
PYTHONPATH=../ddsr-bench \
DDPR_TEST_CRITPT_TASKS=../ddsr-bench/tasks/critpt-ai \
python -m pytest -q tests/rewards/test_compiled.py
```

These checks load private prepared-task references through DDPR's input adapter.
Tasks with `reference_check` equal to `matched`, `unresolved` or `different` are
selected; known matches remain part of the regression coverage.
They check input compatibility and rule grading of literal reference returns,
including a missing-answer negative control. Nonliteral returns are explicitly
skipped by the literal test. No corpus function is called and no model API is
used by these tests.

A separate static audit on 2026-09-30 accepted all ten compiled rule references:
six previously `matched` and four `unresolved`. Three used literal rule checks;
seven used Qwen3.8-Max equivalence assessment with code supplied only as text.
No `different` case was available. Live controls rejected a wrong expression,
accepted a semantic reference answer, and rejected an abstention. These results
check assessment behavior, not answer-code execution or general judge accuracy.
Private reports are retained under `data/physics-check/static-reference-20260930/`
and excluded from Git.

Saved-response checks on the same date replayed all ten rule references with
unchanged results. A live Qwen3.8-Max check through `grade_response` accepted one
Freestyle reference and rejected a wrong answer and an abstention. Replaying
those identical requests through `CritptAiReward` returned 1, 0 and 0. Artifacts
are under `data/physics-check/saved-response-20260930/`. These checks execute no
corpus code; the full rule path, which runs answer functions, remains unverified
on these references.

To replay the saved assessments through the current DDSR-Bench client adapter:

```bash
PYTHONPATH=../ddsr-bench \
DDPR_TEST_CRITPT_TASKS=../ddsr-bench/tasks/critpt-ai \
DDPR_TEST_CRITPT_AI_REPLAY=data/physics-check/saved-response-20260930 \
python -m pytest -q tests/rewards/test_replay.py
```

This offline check verifies input hashes and exact judge messages before replaying
saved completions through a mocked HTTP transport. It checks static rule-reference
outcomes, semantic reports/rewards and unresolved provider failures. Network access
and corpus execution are blocked; it does not establish live provider availability
or rule execution isolation.
