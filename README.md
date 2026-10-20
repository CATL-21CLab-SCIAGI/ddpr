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

Install the optional grading dependencies and select the shared implementation:

```bash
python -m pip install -e ../ddsr-bench -e '.[critpt-ai]'
export DDPR_REWARD=ddpr.rewards.critpt_ai.CritptAiReward
```

Each record must supply exactly one canonical grading source in private metadata:
`metadata.problem` containing the native Auto/Freestyle `problem.json` object, or
`metadata.problem_path` containing its absolute path, or `metadata.task_path`
pointing to a compiled CritPt-AI task (requires DDSR-Bench). Paths must be readable
on every worker.
Teacher `reference_completion` stays in the shared interface but is ignored.
Keep the student's prompt public: do not put grading plans or references into
`prompt`/`messages`. Neither backend adds private metadata to the policy prompt.

The reward is 1.0 for correct and 0.0 for incorrect responses. Judge uncertainty,
API failures and grader errors raise `CritptAiRewardError` with a detailed report;
they are not zero rewards. Dataset/configuration errors also surface as errors.
Auto supports floating numeric, exact, symbolic and categorical answers, with
optional exact/symbolic equivalence judging. Freestyle uses SemanticJudge and
supports multiple answer items. Auto composite is not connected yet.

See [CritPt-AI reward usage and configuration](ddpr/rewards/critpt_ai/README.md).
Grading is supplied by the DDSR-Bench checkout containing the CritPt-AI evaluator;
the standalone Auto-CritPt project is not a runtime dependency. GPU training with this reward has not been
validated. Existing backend adapters score serially within each reward process;
LLM judging may be substantially slower and more expensive than a numeric check.

## Future work

Planned extensions, not yet implemented:

- **Additional backends such as Miles:** reuse shared data adapters and add
  backend-specific launch configurations and integration tests.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code layout,
conventions and checks.
