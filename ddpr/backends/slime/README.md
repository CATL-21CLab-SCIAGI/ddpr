# Slime SFT and RL

Use the [DSW/DLC guide](../../../envs/README.md) for images and setup.
Install ddpr and your reward implementation in every worker's environment.

## Input and supervision

Read ddsr-bench's `sft.jsonl` directly; no converted file is needed:

```json
{"id":"trial:1","prompt":[{"role":"user","content":"Question"}],"completion":[{"role":"assistant","content":"Answer"}],"metadata":{"benchmark":"cmphysbench"}}
```

Messages must be text-only system/user/assistant turns with exactly one assistant
completion. Empty completion text is skipped and counted; malformed records and
empty datasets raise errors. Tools and multimodal inputs are unsupported.

SFT preserves history and supervises only the completion and template suffix.
Its “rollout” prepares offline data, without generation or reward scoring.
RL uses GRPO, requires a final user turn, and keeps the teacher completion outside
the prompt. Both preserve `sample_id` and metadata; teacher references are unverified.

## Qwen3.8-27B

Run from the project root. Set `DDPR_MODEL` to the local model/tokenizer directory.
`DDPR_SLIME_ROOT` and `DDPR_MEGATRON_ROOT` default to `/root/slime` and
`/root/Megatron-LM`. Trailing CLI arguments override regular launcher settings.

The launchers use the checkpoint's `qwen3_5` architecture and Slime's `qwen3_5`
loss mask. Other tested mask modes fail or lose context with this tokenizer.
The unchanged template can insert an empty thinking block; no separate reasoning
trace is added.

### SFT launcher

```bash
export DDPR_MODEL=/path/to/Qwen3.8-27B
export DDPR_SFT_DATA=/path/to/sft.jsonl
export DDPR_OUTPUT_DIR=/path/to/checkpoints
export DDPR_ACTOR_GPUS=4
export DDPR_NUM_ROLLOUT=1000
bash scripts/train/slime/qwen38_27b_sft.sh
```

Each rollout is one update. The single-node launcher uses one pipeline stage per
actor GPU. `DDPR_BATCH_SIZE` defaults to 8, `DDPR_MAX_LENGTH` to 8192 tokens per
full conversation, and `DDPR_SAVE_INTERVAL` to 100 updates. Small datasets repeat
to fill batches; oversized samples raise errors. Choose settings that fit your GPUs.

Resume with `DDPR_LOAD` pointing to the checkpoint root. `DDPR_NUM_ROLLOUT` is
the total target. When extending it, pass `--override-opt_param-scheduler`; to
retain the saved schedule, use `--use-checkpoint-opt_param-scheduler`. Keep data,
batching and parallelism consistent. Optimizer state and dataset cursor are saved.

### RL launcher

Set model, data and output as above, then configure reward and GPU allocation:

```bash
export DDPR_REWARD=my_rewards.PhysicsReward
export DDPR_ACTOR_GPUS=2
export DDPR_ROLLOUT_GPUS=2
export DDPR_ROLLOUT_GPUS_PER_ENGINE=2
bash scripts/train/slime/qwen38_27b_rl.sh
```

This uses separate training/generation GPUs on one node. Defaults are one update,
8 prompts, 4 responses per prompt and prompt/response limits of 4096/2048 tokens.
Oversized prompts raise errors. Checkpoints are saved each update; SFT's resume
settings also apply. These allocations are examples, not capacity guarantees.

### SFT and RL smoke launchers

```bash
# Reuse model, data and GPU settings above; use fresh output directories.
DDPR_OUTPUT_DIR=data/smoke/slime/sft bash scripts/train/slime/qwen38_27b_sft_smoke.sh
DDPR_OUTPUT_DIR=data/smoke/slime/rl bash scripts/train/slime/qwen38_27b_rl_smoke.sh
```

Both wrap regular training, fixing one update, batch size 2 and a 2048-token
sequence limit. RL uses one prompt and two responses capped at 16 tokens.
Unset `DDPR_REWARD` to use its default synthetic `SmokeReward`.
Workload flags are fixed; optimizer, topology and trainable parameters are inherited.
Full-model training remains the default.

Smoke jobs save checkpoints and require nonzero gradients and sampled weight
changes on every training rank. Evidence goes to `$DDPR_OUTPUT_DIR/checks/`;
checks inspect up to 4096 values per trainable tensor.

## Reward and launch integration

Select a shared class through `DDPR_REWARD`; see the
[reward contract](../../../README.md#reward-contract). The adapter handles
single/grouped samples and serializes verifier calls within each worker.
For a native async hook, unset `DDPR_REWARD` and set
`DDPR_REWARD_FUNCTION=module.function`.

Custom launchers can source [sft.sh](../../../scripts/train/slime/sft.sh) or
[rl.sh](../../../scripts/train/slime/rl.sh) and pass `DDPR_SFT_ARGS` or `DDPR_RL_ARGS`
to `python -m ddpr.backends.slime.train`. This delegates to Slime while carrying
reward selection in Ray arguments. Set `DDPR_LOSS_MASK_TYPE=qwen3_5` for SFT and
keep its global-dataset/template settings unchanged. SFT template keyword
overrides and generation-based evaluation are unsupported. RL evaluation uses
Slime's separate loader; unfinished retry-buffer generations are not restored.

## Verification

See [CONTRIBUTING](../../../CONTRIBUTING.md#development-setup) for local tests and
[frameworks.json](../../../envs/frameworks.json) for pinned revisions.

- **H20:** selected-layer SFT/RL updates passed on two GPUs, along with RL resume
  and selected rollout-weight comparisons. Only layers 31 and 63 were trainable;
  rewards were synthetic. Evidence: `data/gpu-check/slime-rank-update-20260924/`
  (excluded from Git).
- **Pending:** GPU reruns of refactored launchers/reward adapters, default full-model
  RL topology, SFT resume, B300 and DLC execution. The earlier L20D-reported run
  does not establish B300 support.

For frozen-prefix training, set `DDPR_RECOMPUTE_GRANULARITY=selective`.
Full recomputation in the tested Megatron version can suppress gradients when
checkpoint inputs do not require gradients. Full-model training keeps full
recomputation. These checks do not validate physics reward quality.
