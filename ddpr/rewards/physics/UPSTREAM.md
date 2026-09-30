# Upstream grading snapshot

Source: `CATL-21CLab-SCIAGI/Auto_CritPt_Grader`, Apache-2.0.
Commit: `565d1b0bfab353e17a40a38fe656c3ab0caee660` (version 0.6.0).
Original directory: `src/auto_critpt_grader/`.
The accompanying LICENSE is copied from that repository.

This package contains the standard grading core, not the per-problem Python
grader generator. Relative imports make it independent of the original package.
The implementation retains upstream answer-code restrictions, confidence
handling and HTTP transports. Comparison, error-handling and prompt changes
are listed below.

Integration changes:

- Native Auto/Freestyle input parsing and report formatting live in `critpt_ai/`.
  `inputs.py` normalizes native problems; `outputs.py` attaches report metadata
  and serializes extracted values. `critpt_ai/schemas.py` owns native contracts
  and reports. Root `grader.py` owns grading orchestration; root
  `schemas.py` retains grading plans, precision, assessment and completion types.
  Shared answer extraction/execution lives in `answer.py`, with answer and
  grading-plan validation in root `validation.py`. The public reward and
  `grade_response` exports are preserved.
- Shared models live in `schemas.py`; LLM policies live in `llm/` and model clients in
  `llm/clients/`. Upstream `backends.py`, `judge_backend.py` and `llm_io.py` are now
  `llm/clients/openai.py`, `llm/clients/client.py` and `llm/utils.py`.
  Shared `ClientConfig` is in `llm/clients/config.py`; each assessment module
  owns its environment prefix.
  Shared environment loading preserves each judge's defaults and prefix.
  Both LLM policies accept the same injected `(messages, config)` completion
  callable, returning text or `CompletionResult`. Provider result conversion
  belongs to clients; prompts, confidence rules and schema repair are preserved.
- Both transports honor the configured HTTP attempt count and backoff. Transport
  timeouts and backoffs must be finite; Bedrock restores signal handlers when
  timer setup fails. Malformed OpenAI response envelopes are recorded as failed
  calls rather than successful requests.
- The equivalence prompt now explicitly requests numeric confidence in `[0, 1]`.
  Live Qwen responses otherwise used `"high"`, which the schema correctly rejects.
  The confidence threshold and validation remain unchanged.
- LLM implementations are named `LLMEquivalenceJudge` and `LLMSemanticJudge`,
  accept submitted content as `answer`, and build prompts with private `_messages`
  helpers. Verdicts use literal schema types; required explanation text is stripped
  before length validation. JSON parsing rejects duplicate keys and multiple
  objects. Semantic schema repair catches validation/parsing errors rather than
  unexpected internal exceptions; its one-repair limit is unchanged.
  Semantic assessment receives explicit public task and reference data, without
  importing native problem models. Its prompt payload and scoring policy remain
  unchanged. Response fields are defined once in `schemas.py`; recorded judgments extend
  them with request metadata. Text normalization does not alter raw responses.
- `primitives.py` is split across `rule/`, `answer.py` and `schemas.py`.
  Dispatch and grade composition live in `rule/compare.py`, following the sibling
  DDSR-Bench grading layout without sharing its different comparison contracts.
  `rule/utils.py` contains mathematical parsing and result helpers. AST validation
  and the restricted SymPy namespace live in `validation.py`, shared by
  expression parsing and answer execution. Answer-function structure and
  grading-plan checks are shared there too. The unused vocabulary-list wrapper
  and the former `common.py` and `expressions.py` modules are removed.
  `grading_runtime.py` is now `runtime.py`. Timeouts must be finite;
  alarm cleanup restores the previous handler even when timer setup fails.
  Missing SymPy is a grader failure, not an invalid-answer score. Execution
  parameters use `grading_plan`, `fn`, `extract_fn` and `timeout` (seconds);
  serialized fields and environment names remain unchanged.
  Removed its unused combined extraction/grading function; `grader.py` owns
  that sequence. Equivalence judges use the declared `judge` protocol directly,
  without signature inspection for older implementations.
- Symbolic constancy checks use simplification instead of three numerical samples,
  which rejected valid small-scale expressions and functions equal at those points.
  Simplification remains bounded by the caller's grading deadline.
  Unexpected exact/symbolic comparison errors propagate to the grading report as
  unresolved errors instead of becoming incorrect-answer scores.
- Empty grading targets are invalid inputs. Legacy Freestyle normalization accepts
  mappings consistently with the loader. A resource-limit report is unresolved
  in `PhysicsReward`, not an incorrect-answer score.
- Reference and configuration failures raise rather than penalizing answers.
  Unsupported reference operations retain the text-only LLM fallback. Symbol
  assumptions come from the problem contract, never from submitted expressions.
  Conflicting categorical aliases are rejected regardless of dictionary order.
  References outside categorical choices or simplifying to symbolic constants
  are grader errors, including when submitted and reference expressions match.
  Plans require finite nonnegative weights with a finite positive total; grade
  composition no longer silently clamps invalid weights. The private `_grade`
  helper leaves target assignment to comparison dispatch.
- Rule comparison functions accept `answer`, `target` and keyword options rather
  than `actual`, `expected` and a positional argument dictionary. Serialized
  grading-plan fields remain unchanged. The numeric diagnostic uses `field_name`
  for the answer-field name, distinct from its reference value `target`.
  `grade_exact` is renamed to `grade_exact_numeric`, matching
  `grade_floating_numeric`; stored primitive names remain unchanged.
  The supporting numerical assessment is named `evaluate_numeric_agreement`.
- Equivalence-judge initialization and invalid numerical timeout settings produce
  grader-error reports. Deterministic comparison failures retain successful
  extraction evidence instead of reporting extraction as unattempted.
- `grader.py` combines the former public adapter and `evaluator.py` orchestration,
  avoiding a second normalization pass. `critpt_ai/inputs.py` owns native parsing;
  the unused internal judge-mode guard and `_core` package are removed.
  `reward.py` owns training metadata loading and score mapping.
  Removed the report's `schema_version`
  and derived `reward` fields and reward validator. The same score mapping
  now lives in `PhysicsReward`.
- Removed `run_and_grade`, its completion normalizer and unused grading helper
  from evaluator.py. DDPR supplies the student's response; no solver is bundled.
- Removed solver/batch report schemas and the solver-only Bedrock backend class
  and completion protocol. OpenAI/Bedrock Judge transports remain.
- Applied DDPR formatting and import ordering. Other upstream style differences
  are lint-exempt to avoid changing error types or numerical behavior on import.
- DDPR's public adapter excludes Auto composite/multi-target problems. Internal
  plan composition code is retained for source parity, not exposed as a reward.

The former floating/exact/symbolic/categorical re-export modules are removed;
comparison implementations live in `rule/numeric.py`, `rule/symbolic.py` and
`rule/categorical.py`. SemanticJudge and the optional Auto
equivalence Judge live in `llm/semantic.py` and `llm/equivalence.py`.
For raw responses, always use
`physics.grade_response` or `PhysicsReward` so extraction and deadlines apply.

Future upstream syncs should review these omissions, run the DDPR reward tests,
and compare saved-response results before changing this revision. Do not copy
the CLI, model runner, code generation or campaign modules into this directory.
