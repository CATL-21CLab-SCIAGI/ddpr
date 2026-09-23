# Swift SFT and RL

For standalone commands, see [Qwen3.8-27B](#qwen38-27b) below.

## Input and supervision

Read ddsr-bench's exported `sft.jsonl` directly, using the same
[record format](../slime/README.md#input-and-supervision) as the Slime backend.
The adapter accepts text-only system/user/assistant messages, a prompt ending
with a user message, and exactly one nonempty assistant completion.
Tool definitions and multimodal inputs are unsupported.

SFT preserves earlier assistant messages as context with `loss: false` and
supervises the completion with `loss: true`, including applicable template
suffix tokens. RL currently uses GRPO to generate fresh responses from the
prompt and exposes the teacher answer as `reference_completion` to rewards.
Both retain the exported ID as `sample_id` and preserve `metadata`.

Export views and quality selection belong to ddsr-bench. ddpr does not filter
for correctness or add separately recorded provider reasoning. Teacher answers
are not automatically verified references. Adaptation happens during loading;
Swift may create dataset/tokenization caches, but no converted JSONL is needed.

## Launch integration

Set up and check the [Swift GPU environment](../../../configs/environments/README.md#swift-gpu)
before launching on a new host.

Install ddpr and MS-Swift in the training environment. No ddsr-bench installation
is needed. Source the [SFT argument fragment](../../../configs/jobs/swift/sft.sh)
from a model-specific launcher:

```bash
export DDPR_SFT_DATA=/absolute/path/to/sft.jsonl
source /absolute/path/to/ddpr/configs/jobs/swift/sft.sh
```

Pass `"${DDPR_SFT_ARGS[@]}"` to `swift sft` alongside model, checkpoint,
optimizer, parallelism, batch-size and training-duration settings. The fragment
is not a standalone launcher. `plugin.py` registers the `ddpr_sft` and `ddpr_rl`
dataset views through Swift's external plugin interface.

For RL, source the [RL fragment](../../../configs/jobs/swift/rl.sh) and
pass `"${DDPR_RL_ARGS[@]}"` to `swift rlhf`. Set `DDPR_REWARD_FUNCS` to
space-separated reward names; `DDPR_REWARD_PLUGIN` optionally loads the Python
file registering them in `swift.rewards.orms`. Reward calls receive generated
`completions` plus batched `sample_id`, `metadata` and `reference_completion`.
No default physics reward is bundled.

## Qwen3.8-27B

The launchers use Swift's `qwen3_5` model type, matching the checkpoint's
architecture, and its dedicated `qwen3_8` template. These are separate from
Slime's loss-mask modes. The template supplies conversation formatting and
special tokens; no separate reasoning trace is added.

They default to `Qwen/Qwen3.8-27B` through ModelScope. `DDPR_MODEL` can name a
local checkpoint. Choose GPU parallelism and memory settings for the host;
trailing Swift options can select settings such as `--deepspeed zero3`.

### SFT launcher

Run from the project root in a training environment:

```bash
export DDPR_SFT_DATA=/absolute/path/to/sft.jsonl
export DDPR_OUTPUT_DIR=/absolute/path/to/checkpoints
bash configs/jobs/swift/qwen38_27b_sft.sh
```

Defaults are one epoch, one sample per device, eight accumulation steps and
checkpoint saving every 100 updates. `DDPR_MAX_LENGTH` defaults to 8192 tokens
for the full prompt, completion and template. With the launcher's `--strict true`,
oversized samples raise an error even though `--truncation_strategy delete` is
selected; they are neither dropped nor truncated. Choose a limit from actual
sample lengths and memory capacity.

Trailing arguments override settings, for example `--tuner_type lora` or
`--max_steps 10`. Resume with `--resume_from_checkpoint /path/to/checkpoint`.

### RL launcher

For RL with GRPO, additionally configure rewards:

```bash
export DDPR_REWARD_PLUGIN=/absolute/path/to/rewards.py
export DDPR_REWARD_FUNCS=physics_reward
bash configs/jobs/swift/qwen38_27b_rl.sh
```

The launcher uses Transformers generation and saves checkpoints under
`DDPR_OUTPUT_DIR`. `DDPR_MAX_LENGTH` defaults to 8192;
`DDPR_MAX_COMPLETION_LENGTH` separately limits generated responses to 2048
by default. Trailing Swift arguments can override training settings.
Swift's default GRPO length policy truncates oversized prompts from the left.
Choose a sufficient prompt limit to retain the full physics problem and history.

### SFT smoke launcher

The [smoke launcher](../../../configs/jobs/swift/qwen38_27b_sft_smoke.sh) reuses
regular SFT settings, fixes one optimizer update and a 2048-token full-sample
limit, and disables checkpoint saving:

```bash
export DDPR_SFT_DATA=/absolute/path/to/sft.jsonl
export DDPR_OUTPUT_DIR=/absolute/path/to/smoke-output
bash configs/jobs/swift/qwen38_27b_sft_smoke.sh
```

The output directory still holds arguments and training logs. By default it
uses one sample per device and eight accumulation steps; the effective batch
also depends on worker count. Use suitable short samples. Device, precision
and parallelism options can be passed through, while the smoke duration,
length limit and disabled checkpoint saving are fixed.

## Verification

### Interface checks

Use the [local development environment](../../../CONTRIBUTING.md#development-setup)
for dataset, reward-interface and real-tokenizer checks. Weights are not needed;
prefer ModelScope when downloading missing tokenizer files.

```bash
DDPR_INSTALLED_SWIFT=1 \
DDPR_QWEN38_TOKENIZER=/path/to/local/Qwen3.8-27B-tokenizer \
python -m pytest -q tests/backends/swift
```

Checks cover registry loading, history, completion-only labels, IDs, metadata,
reward arguments and overlength errors in eager and lazy loading.
The existing eight-sample CMPhysBench export loaded in
both views. Rerun checks when changing Swift or the student tokenizer.

### Training checks

Add `DDPR_SWIFT_CPU_TRAINING=1` to exercise the SFT, SFT smoke and RL launchers.
Tests create a tiny random Qwen3 model with its matching template and use a
synthetic GRPO reward. Regular launchers must update weights and save a
checkpoint; the smoke launcher must complete one update without a checkpoint.
SFT also resumes from the first checkpoint of a two-update run over distinct
records and must match the uninterrupted run's final weights exactly, with
optimizer steps restored.

These checks passed locally on 2026-09-23. They validate training integration,
not Qwen3.8-27B training or physics accuracy. The separate tokenizer test checks
the actual `qwen3_8` template. GPU SFT and RL updates remain unverified.

### Tested revisions

- MS-Swift: `ae700468052af1e461857b6f41eee65aa1641f45` (4.6.0.dev0).
- Transformers: `5.12.1`; TRL: `0.29.1`; PyTorch: `2.14.0`.
- Qwen3.8 tokenizer: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

Upstream reference: [dataset registry](https://github.com/modelscope/ms-swift/blob/ae700468052af1e461857b6f41eee65aa1641f45/swift/dataset/register.py),
[Qwen models](https://github.com/modelscope/ms-swift/blob/ae700468052af1e461857b6f41eee65aa1641f45/swift/model/models/qwen.py),
and [reward interface](https://github.com/modelscope/ms-swift/blob/ae700468052af1e461857b6f41eee65aa1641f45/swift/rl_core/grpo_algorithm.py).
