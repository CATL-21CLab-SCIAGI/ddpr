# Upstream grading snapshot

Source: `CATL-21CLab-SCIAGI/Auto_CritPt_Grader`, Apache-2.0.
Commit: `565d1b0bfab353e17a40a38fe656c3ab0caee660` (version 0.6.0).
Original directory: `src/auto_critpt_grader/`.
The accompanying LICENSE is copied from that repository.

This package contains the standard grading core, not the per-problem Python
grader generator. Relative imports make it independent of the original package.
Mathematical comparisons, answer-code restrictions, confidence handling, HTTP
transports and grading error classification are preserved. Prompt changes are
listed below.

Integration changes:

- Shared models live in `schemas.py`; LLM policies live in `llm/` and transports in
  `transport/`. Upstream `backends.py`, `judge_backend.py` and `llm_io.py` are now
  `transport/openai.py`, `transport/client.py` and `llm/parsing.py`.
  Transport configuration is in `transport/config.py`.
  Shared environment loading preserves each judge's defaults and prefix.
  Both LLM policies accept the same injected `(messages, config)` completion
  callable, returning text or `CompletionResult`. Provider result conversion
  belongs to transport; prompts, confidence rules and schema repair are preserved.
- The equivalence prompt now explicitly requests numeric confidence in `[0, 1]`.
  Live Qwen responses otherwise used `"high"`, which the schema correctly rejects.
  The confidence threshold and validation remain unchanged.
- `primitives.py` is split across `rule/`, `execution/answers.py` and `schemas.py`.
  Comparison and extraction definitions are unchanged. The shared expression
  vocabulary lives in `rule/expressions.py`; answer execution imports it.
  `grading_runtime.py` is now `execution/runtime.py`, with unchanged deadlines.
  Removed its unused combined extraction/grading function; `grader.py` owns
  that sequence. Equivalence judges use the declared `judge` protocol directly,
  without signature inspection for older implementations.
- `grader.py` combines the former public adapter and `evaluator.py` orchestration,
  avoiding a second normalization pass. `inputs.py` owns native problem parsing;
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
