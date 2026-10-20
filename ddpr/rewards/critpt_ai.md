# CritPt-AI reward

`CritptAiReward` adapts DDSR-Bench's CritPt-AI grader for Slime and Swift.

Install DDSR-Bench with the `llm_fallback` API from
[PR #11](https://github.com/CATL-21CLab-SCIAGI/ddsr-bench/pull/11) and select the
shared implementation. Older `0.1.0` checkouts satisfy the version constraint but
lack this API; update both projects together. From the DDPR project root:

```bash
python -m pip install -e ../ddsr-bench -e '.[critpt-ai]'
export DDPR_REWARD=ddpr.rewards.critpt_ai.CritptAiReward
```

Each record must supply exactly one canonical grading source in private metadata:
`metadata.problem` containing the native Auto/Freestyle `problem.json` object, or
`metadata.problem_path` containing its absolute path, or `metadata.task_path`
pointing to a compiled CritPt-AI task (requires DDSR-Bench). Paths must be readable
at the same absolute location on every worker and remain immutable during a run.
Teacher `reference_completion` stays in the shared interface but is ignored.
Keep the student's prompt public: do not put grading plans or references into
`prompt`/`messages`. Neither backend adds private metadata to the policy prompt.

The reward is 1.0 for correct and 0.0 for incorrect responses. Judge uncertainty,
API failures and grader errors raise `CritptAiRewardError` with a private `.report`
and `.sample_id`; they are not zero rewards. Dataset/configuration errors also
surface as errors.
Auto supports floating numeric, exact, symbolic and categorical answers, with
optional exact/symbolic equivalence judging. Freestyle uses SemanticJudge and
supports multiple answer items. Auto composite is not connected yet.

Set `DDPR_CRITPT_AI_EQUIVALENCE_JUDGE=false` to disable exact/symbolic LLM fallback;
Freestyle still requires its semantic judge. Python callers use
`CritptAiReward(llm_fallback=False)`; the old `judge_enabled` keyword is rejected.
The default is `True`. For judge configuration and execution limits, see
[DDSR-Bench's grader guide](https://github.com/CATL-21CLab-SCIAGI/ddsr-bench/blob/main/ddsr_bench/benchmarks/critpt/evaluation/ai/README.md).

Export judge settings and credentials before launching. Slime forwards
`AUTO_CRITPT_*`, `DDPR_CRITPT_AI_*`, standard OpenAI/Aliyun/Bedrock keys and
credentials named by judge `*_API_KEY_ENV` settings through Ray. Swift inherits
the exported environment. A driver-only `.env` file is insufficient; protect
Ray's dashboard and job data, which can contain worker credentials.

The Python grader executes rule answers locally; this adapter requires neither
Docker nor the standalone Auto-CritPt project. For diagnostics,
`ddpr.rewards.critpt_ai` also exports `grade_response` and metadata-based
`load_problem`; judge classes belong to DDSR-Bench.
GPU training with this reward has not been validated. Existing backend adapters
score serially within each reward process;
LLM judging may be substantially slower and more expensive than a numeric check.
