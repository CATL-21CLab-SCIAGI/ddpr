# ddpr

Training for data-driven physics reasoning.

ddpr adapts ddsr-bench's `sft.jsonl` in memory for supervised fine-tuning (SFT)
with Slime or MS-Swift, preserving context and supervising only the completion.
Both backends have passed Qwen3.8-27B SFT and RL (GRPO) optimizer-update checks
on H20: Slime with selected decoder layers trainable, and Swift with LoRA.
RL checks use synthetic rewards; the physics reward is not yet defined.
Both backends accept caller-supplied rewards through their native hooks.

## Installation

Use Python 3.12 or newer. From this project's root:

```bash
python -m pip install -e .
```

Install the chosen training framework and its model dependencies separately
using its own instructions. ddpr does not depend on ddsr-bench's Python package, and does
not install the similarly named PyPI package `slime`.

See the [Slime SFT/RL guide](ddpr/backends/slime/README.md) or
[Swift SFT/RL guide](ddpr/backends/swift/README.md) for training setup.
Use the [image table and DSW/DLC setup guide](envs/README.md#images-by-gpu)
to select the custom image and verify the installed frameworks before training.
For Slime SFT with configurable duration and checkpoints, use the
[SFT launcher](ddpr/backends/slime/README.md#sft-launcher).

## Run the SFT smoke job

From the project root on a DSW instance using the Slime image:

```bash
python scripts/envs/setup.py slime
export DDPR_MODEL=/path/to/Qwen3.8-27B
export DDPR_SFT_DATA=/path/to/sft.jsonl
export DDPR_ACTOR_GPUS=4
export DDPR_OUTPUT_DIR=/path/to/fresh/smoke-output
bash scripts/train/slime/qwen38_27b_sft_smoke.sh
```

This reuses the regular SFT configuration for one update on two samples, saves
a checkpoint, and checks per-rank gradients and sampled weight changes.
See the [Slime smoke guide](ddpr/backends/slime/README.md#sft-and-rl-smoke-launchers)
for RL, configuration and verification limits.

## Reward contract

Select the same reward class for either backend:

```bash
export DDPR_REWARD=ddpr.rewards.SmokeReward
# Or: export DDPR_REWARD=my_package.rewards.PhysicsReward
```

Both RL smoke wrappers default to [`SmokeReward`](ddpr/rewards/smoke.py), a small
`Reward` implementation assigning deterministic text-hash scores. It is purely
synthetic: it does not evaluate physics or compare against teacher answers.
Identical responses receive identical scores and may produce zero advantages.

Implement `Reward.__call__` with generated text, sample ID, metadata and the
unverified teacher completion; return a finite scalar, higher being better.
The class must be importable on every worker and constructible without arguments.
Slime and Swift adapters handle invocation and preserve response order.
`DDPR_REWARD` takes precedence over native backend reward settings; unset it
to use native hooks. Concrete physics verification remains to be defined.

## Future work

Planned extensions, not yet implemented:

- **Physics rewards:** integrate verification for generated responses.
  Teacher completions are not automatically verified references.
- **Additional backends such as Miles:** reuse shared data adapters and add
  backend-specific launch configurations and integration tests.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code layout,
conventions and checks.
