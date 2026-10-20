# ddpr

Training for data-driven physics reasoning.

ddpr adapts ddsr-bench's `sft.jsonl` in memory for supervised fine-tuning (SFT)
and reinforcement learning (RL) with Slime or MS-Swift. SFT supervises only the
completion; RL generates new responses and passes teacher references to rewards.
Both backends have passed Qwen3.8-27B SFT and RL (GRPO) optimizer-update checks
on H20: Slime with selected decoder layers trainable, and Swift with LoRA.
RL optimizer checks use synthetic rewards. Optional Auto/Freestyle CritPt-AI
rewards are available and tested separately on CPU.
Both backends accept caller-supplied rewards through their native hooks.

## Installation

Use Python 3.12 or newer. From this project's root:

```bash
python -m pip install -e .
```

Install the chosen training framework and its model dependencies separately.
ddpr does not depend on ddsr-bench's Python package or install training frameworks.
The similarly named PyPI package `slime` is not the training framework.

Use the [image table and DSW/DLC setup guide](envs/README.md#images-by-gpu)
to select the custom image and verify the installed frameworks before training.

## Training

Choose a backend, provide the model and exported data, then run its SFT or RL
launcher. Start with a smoke job to check integration on your container and GPUs.
Smoke jobs inherit the regular training settings with a smaller workload.

| Backend | Training | Smoke checks |
| --- | --- | --- |
| Slime | [SFT](ddpr/backends/slime/README.md#sft-launcher), [RL](ddpr/backends/slime/README.md#rl-launcher) | [SFT and RL](ddpr/backends/slime/README.md#sft-and-rl-smoke-launchers) |
| MS-Swift | [SFT](ddpr/backends/swift/README.md#sft-launcher), [RL](ddpr/backends/swift/README.md#rl-launcher) | [SFT and RL](ddpr/backends/swift/README.md#sft-and-rl-smoke-launchers) |

## Input

Both backends read the same JSONL export without a converted file:

```json
{"id":"trial:1","prompt":[{"role":"user","content":"Question"}],"completion":[{"role":"assistant","content":"Answer"}],"metadata":{"benchmark":"cmphysbench"}}
```

Use text-only system/user/assistant messages and exactly one assistant completion.
Empty completion text is skipped in SFT and RL; malformed records and empty
datasets raise errors. Tools and multimodal inputs are unsupported. `id` and
`metadata` are optional; adapters preserve them as `sample_id` and metadata.
Backend guides describe prompt-turn constraints and token limits.

## Reward contract

SFT does not call a reward. For ordinary RL, supply a shared reward class or a
native backend hook. Both RL smoke launchers default to
[`SmokeReward`](ddpr/rewards/smoke.py), which assigns synthetic text-hash scores;
it does not evaluate physics. Identical responses receive identical scores and
may produce zero advantages.

To implement a shared reward, subclass [`Reward`](ddpr/rewards/base.py). For
example, put this toy formatting check in `my_rewards.py` and supply a nonempty
`metadata.required_suffix` in each input record:

```python
from collections.abc import Mapping
from typing import Any

from ddpr.rewards import Reward


class FormatReward(Reward):
    def __call__(
        self,
        *,
        response: str,
        sample_id: str | None,
        reference_completion: str,
        metadata: Mapping[str, Any],
    ) -> float:
        return float(response.rstrip().endswith(metadata["required_suffix"]))
```

For `{"required_suffix": "END"}`, a response ending in `END` scores 1; others
score 0. This checks formatting only. `response` is the newly generated text;
`sample_id` is the exported ID or `None`, and `reference_completion` is the
unverified teacher answer, unused in this example. Return a finite scalar with
higher scores meaning better responses. Let verifier failures surface rather
than silently converting them to correctness scores.

Make the module importable on every worker (for example, install your package),
then select the same class before either backend's RL launcher:

```bash
export DDPR_REWARD=my_rewards.FormatReward
```

Classes must be constructible without arguments and are cached per process.
Slime and Swift adapters pass the fields above and preserve response order.
`DDPR_REWARD` overrides native reward settings, including in smoke jobs; unset it
to use native hooks in regular RL or the default synthetic reward in smoke jobs.

### Auto/Freestyle CritPt-AI reward

Install DDSR-Bench with the `llm_fallback` API from
[PR #11](https://github.com/CATL-21CLab-SCIAGI/ddsr-bench/pull/11) and select the
shared implementation. Older `0.1.0` checkouts satisfy the version constraint but
lack this API; update both projects together:

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
and `.sample_id`;
they are not zero rewards. Dataset/configuration errors also surface as errors.
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
GPU training with this reward has not been
validated. Existing backend adapters score serially within each reward process;
LLM judging may be substantially slower and more expensive than a numeric check.

## Future work

Planned extensions, not yet implemented:

- **Additional backends such as Miles:** reuse shared data adapters and add
  backend-specific launch configurations and integration tests.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code layout,
conventions and checks.
