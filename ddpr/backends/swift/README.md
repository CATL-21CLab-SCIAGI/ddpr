# Swift SFT and RL

Use the [DSW/DLC guide](../../../envs/README.md) for images and setup.
Install ddpr and your reward implementation in every worker's environment.

## Input and supervision

Read ddsr-bench's `sft.jsonl` using the shared [input format](../../../README.md#input).
Swift requires a final user prompt turn for both SFT and RL.

SFT preserves history with prompt loss disabled and supervises the completion
plus template suffix. RL uses GRPO and exposes the teacher answer only as
`reference_completion` to rewards. Both preserve `sample_id` and metadata;
teacher answers are unverified. Swift may cache datasets/tokens, but no converted
JSONL is needed.

## Qwen3.8-27B

Run from the project root. The launchers use `qwen3_5` model type and `qwen3_8`
template, defaulting to `Qwen/Qwen3.8-27B` through ModelScope. `DDPR_MODEL` can
select a local checkpoint. Trailing CLI options select tuning and parallelism,
for example `--tuner_type lora` or `--deepspeed zero3`; full tuning is the default.

### SFT launcher

```bash
export DDPR_SFT_DATA=/path/to/sft.jsonl
export DDPR_OUTPUT_DIR=/path/to/checkpoints
bash scripts/train/swift/qwen38_27b_sft.sh
```

Defaults are one epoch, one sample per device, eight accumulation steps and
checkpoint saving every 100 updates. `DDPR_MAX_LENGTH` defaults to 8192 tokens
for the full conversation. Strict loading rejects oversized samples rather than
dropping or truncating them. Resume with `--resume_from_checkpoint /path/to/checkpoint`.

### RL launcher

Set model, data and output as above, then select a shared reward:

```bash
export DDPR_REWARD=my_rewards.PhysicsReward
bash scripts/train/swift/qwen38_27b_rl.sh
```

Defaults are one epoch, four generations per prompt and Transformers generation
with vLLM disabled. `DDPR_MAX_LENGTH` defaults to 8192 and
`DDPR_MAX_COMPLETION_LENGTH` to 2048. Swift's default GRPO policy truncates
oversized prompts from the left; choose enough space to preserve the problem.

### SFT and RL smoke launchers

```bash
# Reuse model/data settings above; use fresh output directories.
unset DDPR_REWARD  # Use the synthetic reward for the RL smoke job.
DDPR_OUTPUT_DIR=data/smoke/swift/sft bash scripts/train/swift/qwen38_27b_sft_smoke.sh
DDPR_OUTPUT_DIR=data/smoke/swift/rl bash scripts/train/swift/qwen38_27b_rl_smoke.sh
```

Both wrap regular training, fixing one update, a 2048-token limit and no checkpoint
saving; arguments and logs remain. RL fixes two generations, a per-device batch
of two, one accumulation step and a 16-token response cap. Capped responses remain
eligible for learning. RL defaults to synthetic `SmokeReward`; set `DDPR_REWARD`
to test your own class. Other training settings are inherited.

Check `logging.jsonl` for completion of step 1 and training metrics. These wrappers
do not assert weight changes or checkpoint persistence; the separate integration
tests verify updates and SFT resume.

## Reward and launch integration

Select a shared class through `DDPR_REWARD`; see the
[reward contract](../../../README.md#reward-contract). The adapter registers it
as `ddpr_reward` and preserves batch order. For native Swift rewards, unset
`DDPR_REWARD`, set `DDPR_REWARD_FUNCS` to registered names, and optionally set
`DDPR_REWARD_PLUGIN` to their registration file.

Custom launchers can source [sft.sh](../../../scripts/train/swift/sft.sh) or
[rl.sh](../../../scripts/train/swift/rl.sh), passing `DDPR_SFT_ARGS` to `swift sft`
or `DDPR_RL_ARGS` to `swift rlhf`. The plugin registers `ddpr_sft` and `ddpr_rl`;
no ddsr-bench installation is needed.

## Verification

See [CONTRIBUTING](../../../CONTRIBUTING.md#development-setup) for local tests and
[frameworks.json](../../../envs/frameworks.json) for pinned revisions.

- **H20:** LoRA SFT/RL updates and checkpoints passed on one GPU per task, with
  BF16, SDPA, rank 8/alpha 16 and synthetic rewards. Evidence:
  `data/gpu-check/swift-20260923/` (excluded from Git).
- **Local:** tiny-model SFT/RL and smoke checks passed, including the shared reward
  adapter and SFT resume. These do not establish Qwen3.8 GPU capacity.
- **Pending:** latest GPU smoke/reward integration, distributed/full-parameter
  training, vLLM, B300 and DLC execution. Physics verification is not implemented.

The tested image has dependency conflicts and no recoverable Swift Git commit,
so the strict environment check fails despite successful LoRA runs. With zero
data-loader workers, pass `--dataloader_persistent_workers false`.
Network-backed loading stalled; successful runs used a temporary model copy in
`/dev/shm`, with checkpoints on persistent storage.
