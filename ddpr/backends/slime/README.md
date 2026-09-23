# Slime SFT and RL

For the tested command-line job, see the [project quick start](../../../README.md#run-the-sft-smoke-job).

## Input and supervision

Read ddsr-bench's exported `sft.jsonl` directly. Each line is a record like this
(formatted here for readability):

```json
{
  "id": "trial:formatting",
  "prompt": [
    {"role": "user", "content": "Question"},
    {"role": "assistant", "content": "Derivation"},
    {"role": "user", "content": "Format the answer"}
  ],
  "completion": [{"role": "assistant", "content": "Final code"}],
  "metadata": {"benchmark": "critpt", "stage": "formatting"}
}
```

The adapter accepts text-only system/user/assistant messages and exactly one
nonempty assistant completion. It preserves context and metadata, masks all
prompt messages, and supervises the completion plus applicable template suffix
tokens. In this example, the earlier derivation receives no loss.

ddpr's SFT data source loads `prompt`, `completion` and `metadata` into
`sample.prompt`, `sample.label` and `sample.metadata`. It preserves the exported
`id` as `sample.sample_id`, without adding it to metadata.

Export views and quality selection belong to ddsr-bench. ddpr does not filter
for correctness, add derived examples, or include separately recorded provider
reasoning. Tool definitions and multimodal inputs are unsupported.

For RL, the prompt must end with a user message. The adapter formats it once
with a generation prefix and keeps the teacher completion outside the prompt.
Reward functions receive `sample.sample_id`, `sample.reference_completion`,
unchanged `sample.metadata`, and the newly generated `sample.response`.
`sample.label` remains unset; the teacher reference is not verified ground truth.

## Launch integration

Set up and check the [Slime GPU environment](../../../configs/environments/README.md#slime-gpu)
before launching on a new host.

Install ddpr in every Slime worker's Python environment, using an editable
installation with accessible source or a wheel. No ddsr-bench installation is
needed. Replace a Slime SFT launcher's data/loss arguments with the
[sourceable fragment](../../../configs/jobs/slime/sft.sh):

```bash
export DDPR_SFT_DATA=/absolute/path/to/sft.jsonl
export DDPR_LOSS_MASK_TYPE=qwen3_5  # verified for Qwen3.8-27B
source /absolute/path/to/ddpr/configs/jobs/slime/sft.sh
```

Pass `"${DDPR_SFT_ARGS[@]}"` to the training command alongside model, checkpoint,
optimizer, parallelism, batch-size and training-duration settings. The fragment
is not a standalone launcher.

Slime calls its SFT data-preparation interface a “rollout.” Here it only loads
and tokenizes exported samples: `sft_loss` trains on completions, advantage
computation is disabled, and the reward field is an unused zero placeholder.
No policy generation or reward scoring occurs.

Keep the global dataset enabled and leave `--apply-chat-template` unset so the
mask generator receives the original messages. The fragment sets one sample
per prompt and disables generation workers with `--debug-train-only`.
Generation-based evaluation is unsupported by this adapter.
Chat-template keyword overrides and tool columns are also unsupported by the
SFT mask generator and are rejected.

`--seq-length` limits each tokenized prompt plus completion, including template
tokens. Oversized samples raise an error rather than being truncated. Errors
identify Slime's sample index, which may differ from the JSONL line after
shuffling.

For RL, source [rl.sh](../../../configs/jobs/slime/rl.sh) and pass
`"${DDPR_RL_ARGS[@]}"` to the training command. Set `DDPR_SFT_DATA` as above
and `DDPR_REWARD_FUNCTION` to an importable async reward function:

```python
async def reward(args, sample, **kwargs):
    return await score_physics_response(
        response=sample.response,
        sample_id=sample.sample_id,
        reference=sample.reference_completion,
        metadata=sample.metadata,
    )
```

`score_physics_response` represents your verifier; none is bundled.
With `--group-rm`, the reward hook instead takes a list of samples and returns
one reward per sample. Install the reward module in the rollout environment.
The fragment selects GRPO with native Slime generation, policy loss and advantage
computation. It must not be combined with the SFT fragment.

Both tasks use ddpr's data source to validate records before grouping and
shuffling, preserve IDs and fill batches across as many dataset epochs as needed.
Oversized RL prompts fail with a file and line number; they are not truncated
or dropped. The source reuses Slime's retry buffer and cursor checkpoints.
Partial generations in the retry buffer are not persisted across a process
restart by upstream Slime.
Slime evaluation datasets use a separate loader and do not receive this adapter.

## Qwen3.8-27B

Use `qwen3_5` masking for the verified `Qwen/Qwen3.8-27B` tokenizer. It preserves
the full conversation and supervises the final completion. On this template,
`qwen` fails and `qwen3` produces incorrect tokenization and no supervised tokens
in the tested cases. Slime's `distill_qwen` mode drops intermediate context and
is rejected, including automatic selection through `qwen`.

The unchanged template inserts default reasoning instructions and an empty
thinking block when `reasoning_content` is absent. Slime masks the final
`<think>` opener but supervises the closing `</think>`, answer and `<|im_end|>`.
This trains visible answers without a separate reasoning trace.

The launcher's Qwen3.5 model specification is a separate choice: the mounted
Qwen3.8 checkpoint declares the `qwen3_5` architecture.

### SFT launcher

Use [qwen38_27b_sft.sh](../../../configs/jobs/slime/qwen38_27b_sft.sh) for
training with checkpoints. From the project root:

```bash
export DDPR_SFT_DATA=/absolute/path/to/sft.jsonl
export DDPR_OUTPUT_DIR=/absolute/path/to/checkpoints
export DDPR_ACTOR_GPUS=4  # choose for the target host
export DDPR_NUM_ROLLOUT=1000
bash configs/jobs/slime/qwen38_27b_sft.sh
```

The launcher runs one SFT update per rollout, using one pipeline stage per
actor GPU on a single node. `DDPR_BATCH_SIZE` defaults to eight samples per
update; smaller datasets repeat across epochs to fill the batch. Choose a batch
compatible with the parallelism. `DDPR_MAX_LENGTH` defaults to 8192 tokens per
full conversation; oversized samples raise an error. Reassess memory for these
lengths and batches.
Trailing Slime arguments can override optimizer and parallelism settings.

Checkpoints include optimizer state and the dataset cursor, saved every
`DDPR_SAVE_INTERVAL` updates (default 100) and at the final update. To resume,
keep the same export, model, batching and parallelism settings, then run:

```bash
export DDPR_LOAD="$DDPR_OUTPUT_DIR"
export DDPR_NUM_ROLLOUT=2000
bash configs/jobs/slime/qwen38_27b_sft.sh
```

The update count is the total target, not additional updates after resumption.
`DDPR_MODEL` remains the Hugging Face format model/tokenizer directory;
`DDPR_LOAD` points to the saved Slime checkpoint root. `DDPR_SLIME_ROOT` and
`DDPR_MEGATRON_ROOT` override `/root/slime` and `/root/Megatron-LM`.
Checkpoint/resume wiring was checked against upstream code; a GPU save/resume
run remains unverified.

### RL launcher

The [RL launcher](../../../configs/jobs/slime/qwen38_27b_rl.sh) uses separate
training and generation GPUs on one node. Set `DDPR_ACTOR_GPUS`,
`DDPR_ROLLOUT_GPUS` and `DDPR_ROLLOUT_GPUS_PER_ENGINE` from the target host's
allocation; no GPU capacity is assumed. Training uses one pipeline stage per
actor GPU. The selected counts must fit the model and divide its layers/heads.

```bash
export DDPR_SFT_DATA=/absolute/path/to/sft.jsonl
export DDPR_OUTPUT_DIR=/absolute/path/to/checkpoints
export DDPR_REWARD_FUNCTION=my_rewards.reward
# Set the three GPU allocation variables described above before launching.
bash configs/jobs/slime/qwen38_27b_rl.sh
```

Defaults are one rollout update, eight prompts and four responses per prompt,
a 4096-token prompt limit and a 2048-token generation limit. The full sequence
limit is 6144. Trailing Slime arguments can override duration, lengths and batch
settings; keep those settings consistent. Checkpoints are saved each rollout.
`DDPR_LOAD` selects a checkpoint to resume; `DDPR_MODEL` selects the Hugging Face
format model directory and `DDPR_MEGATRON_ROOT` overrides `/root/Megatron-LM`.
This launcher has not yet been validated with GPU generation or optimization.

### SFT smoke launcher

The fixed [smoke launcher](../../../configs/jobs/slime/qwen38_27b_sft_smoke.sh)
runs one update on eight samples, with four GPUs, a 2048-token full-sample
limit and no checkpoint saving:

```bash
export DDPR_SFT_DATA=/absolute/path/to/sft.jsonl
bash configs/jobs/slime/qwen38_27b_sft_smoke.sh
```

Use the regular launcher for configurable training. See [training checks](#training-checks)
for the recorded GPU smoke result.

## Verification

### Interface checks

Run tokenizer and mask checks with local tokenizer files; weights are not
needed. Run from the project root with Slime and the
[development dependencies](../../../CONTRIBUTING.md#development-setup) installed.
Prefer ModelScope when downloading missing model artifacts.

```bash
DDPR_QWEN38_TOKENIZER=/path/to/local/Qwen3.8-27B-tokenizer \
python -m pytest -q tests/backends/slime/test_masks.py
```

On the configured DSW, test the installed Slime parser, loader and adapter:

```bash
cd /mnt/workspace/Projects/ddpr
DDPR_INSTALLED_SLIME=1 \
DDPR_QWEN38_TOKENIZER=/mnt/model/Qwen/Qwen3.8-27B \
python -m pytest -q
```

These tests cover token IDs, completion masks, conversation history, metadata
and argument defaults. They do not perform optimization. Rerun compatibility
checks when changing Slime or the student tokenizer.

CPU checks against the local Slime revision below verify the native interfaces
and ddpr's data-source adapter:

```bash
DDPR_QWEN38_TOKENIZER=/path/to/local/Qwen3.8-27B-tokenizer \
python -m pytest -q tests/backends/slime/test_rl.py tests/backends/slime/test_integration.py -k 'rl or native'
```

These require an installed Slime checkout and the local development dependencies.
They verify SFT ID preservation and complete batches, RL prompt-only chat
templating, separate teacher references, independent sample groups, buffered
retries, dataset-cursor restoration and custom rewards.
They do not run generation or optimization.

Both fragments use `--data-source-path`; RL retains native generation and buffering.
Tests also cover IDs and references through serialization, small datasets
spanning several epochs, deterministic cursor restoration and invalid settings.

### Training checks

An earlier revision of the [launcher](../../../configs/jobs/slime/qwen38_27b_sft_smoke.sh)
completed one SFT Adam update on eight fresh CMPhysBench samples on 2026-09-14.
Subsequent data-source changes have CPU coverage but have not been rerun on GPU.
The training configuration uses four
pipeline stages, microbatches of one, activation recomputation, and no CPU
optimizer offload or checkpoint saving.

Its 2,048-token per-sample limit accommodates the longest smoke sample (1,004
tokens). Choose a limit from full tokenized sample lengths for other datasets
and reassess memory requirements.

- Loss: `0.5177125935`; gradient norm: `12.8978897616`; learning rate: `1e-5`.
- Batch preparation: 0.54 seconds; training: 568 seconds including initial
  kernel compilation, not a steady-state throughput measurement.
- Validation: 41 tests passed; training exited successfully.
- Log: `data/sft-smoke/cmphysbench-aliyun-20260914/train-4gpu.log`, retained
  locally and on DSW, excluded from Git.

The environment exposed four devices named NVIDIA L20D. Physical VRAM capacity
was not verified; runtime memory reports are not hardware specifications.

### Tested revisions

- DSW Slime: `06ffdbe22be068b52f9ed0fc318c473f7030197e`.
- Local Slime reference: `4c193f1f37509cca70f0e88807a9305b70f63f4e`.
- Initial Qwen3.8 tokenizer check: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

Upstream reference: [dataset loader](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/utils/data.py),
[SFT preparation](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/rollout/sft_rollout.py),
[loss masks](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/utils/mask_utils.py),
and [SFT launch example](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/scripts/run-qwen3-4B-base-sft.sh).
