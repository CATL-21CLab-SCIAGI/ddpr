# Upstream grading snapshot

Source: `CATL-21CLab-SCIAGI/Auto_CritPt_Grader`, Apache-2.0.
Commit: `565d1b0bfab353e17a40a38fe656c3ab0caee660` (version 0.6.0).
Original directory: `src/auto_critpt_grader/`.
The accompanying LICENSE is copied from that repository.

This directory contains the standard grading core, not the per-problem Python
grader generator. Relative imports make it independent of the original package.
Mathematical comparisons, answer-code restrictions, Judge prompts, confidence
handling, HTTP transports and grading error classification are preserved.

Integration changes:

- Removed `run_and_grade`, its completion normalizer and unused grading helper
  from evaluator.py. DDPR supplies the student's response; no solver is bundled.
- Removed solver/batch report schemas and the solver-only Bedrock backend class
  and completion protocol. OpenAI/Bedrock Judge transports remain.
- Applied DDPR formatting and import ordering. Other upstream style differences
  are lint-exempt to avoid changing error types or numerical behavior on import.
- DDPR's public adapter excludes Auto composite/multi-target problems. Internal
  plan composition code is retained for source parity, not exposed as a reward.

The public floating/exact/symbolic/categorical modules re-export the comparison
functions from one shared implementation. SemanticJudge and the optional Auto
equivalence Judge have separate public modules. For raw responses, always use
`physics.grade_response` or `PhysicsReward` so extraction and deadlines apply.

Future upstream syncs should review these omissions, run the DDPR reward tests,
and compare saved-response results before changing this revision. Do not copy
the CLI, model runner, code generation or campaign modules into this directory.
