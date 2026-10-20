# CritPt-AI rewards

`CritptAiReward` adapts DDSR-Bench's CritPt-AI grader to Slime and Swift.
It owns training metadata loading and reward mapping; grading rules, LLM judges,
answer processing and report schemas live in DDSR-Bench.

## Setup

Install the DDSR-Bench checkout containing the merged CritPt-AI integration
(PR #10) in the same environment as DDPR:

```bash
python -m pip install -e ../ddsr-bench -e '.[critpt-ai]'
export DDPR_REWARD=ddpr.rewards.critpt_ai.CritptAiReward
```

The grader guide at
`ddsr_bench/benchmarks/critpt/evaluation/ai/README.md` in the DDSR-Bench
checkout owns judge configuration and execution limits.
Set `DDPR_CRITPT_AI_EQUIVALENCE_JUDGE=false` to disable rule-grading LLM fallback;
Freestyle still needs its semantic judge. Python callers must replace
`CritptAiReward(judge_enabled=...)` with `CritptAiReward(llm_fallback=...)`;
the old keyword is no longer accepted. The default remains `True`, and the
existing environment variable is unchanged. Existing OpenAI and Bedrock clients
and the additive DDSR client configuration are preserved.

Export judge settings and credentials before launching either backend. Slime
forwards `AUTO_CRITPT_*`, `DDPR_CRITPT_AI_*`, the standard OpenAI/Aliyun/Bedrock
keys and keys named by judge `*_API_KEY_ENV` settings through Ray's job
environment. Swift workers inherit the exported environment.
Do not rely on a driver-only `.env` file. Protect Ray's dashboard and job data,
which can contain worker credentials.

Use the existing RL smoke launcher for either backend with `DDPR_REWARD` set
as above and an export containing canonical task metadata. This overrides the
synthetic smoke reward. Judge API calls may be incurred if enabled.

## Input and scores

Supply exactly one private metadata source:

```python
metadata = {"problem": native_problem_json}
metadata = {"problem_path": "/mnt/nas/corpus/example/problem.json"}
metadata = {"task_path": "/mnt/nas/tasks/critpt-ai/example"}
```

Paths must be absolute and readable at the same location on every worker.
`task_path` uses DDSR-Bench's unified CritPt loader; `problem` and `problem_path`
retain compatibility with native Auto/Freestyle records.
Keep files immutable during a run. References and grading plans must never enter
student prompts. The teacher `reference_completion` argument is ignored.
`sample_id` identifies the attempt and need not equal the canonical problem ID.

The reward is 1.0 for a pass and 0.0 for a fail or invalid student answer.
Unresolved/resource-limited grading raises `CritptAiRewardError`; its `.report`
holds diagnostics and `.sample_id` identifies the attempt. Invalid metadata
raises validation errors. Backend adapters do not turn these errors into zero.
Reports contain private references and must remain private artifacts.

For diagnostics, `ddpr.rewards.critpt_ai` continues to export `grade_response`
and the metadata-based `load_problem`; judge classes come from DDSR-Bench.

## Verification

```bash
DDPR_TEST_CRITPT_TASKS=../ddsr-bench/tasks/critpt-ai \
python -m pytest -q tests/rewards tests/test_rewards.py
```

Set `DDPR_TEST_CRITPT_AI_REPLAY=data/physics-check/saved-response-20260930`
to replay existing private judge completions through mocked HTTP. Replay checks
exact prompts and input hashes, semantic scores, static rule assessments and
unresolved provider failures. It blocks network calls and corpus execution.
Set `DDPR_TEST_RAY=1` to run the local Ray worker configuration check in
`tests/backends/slime/test_reward.py`. It uses test credentials and no model calls.
Core grader tests live in DDSR-Bench. These checks do not establish live
provider quality, isolated rule execution, or GPU training with this reward.
The Python grader executes rule answers locally; Docker is not required by
this reward adapter.
