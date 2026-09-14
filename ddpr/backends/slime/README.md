# Slime SFT

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

Slime loads `prompt`, `completion` and `metadata` into `sample.prompt`,
`sample.label` and `sample.metadata`. Its loader drops the top-level `id`;
calling ddpr's `load_sample(record)` directly preserves it in `SftSample.id`.
Neither path moves the ID into metadata.

Export views and quality selection belong to ddsr-bench. ddpr does not filter
for correctness, add derived examples, or include separately recorded provider
reasoning. Tool definitions and multimodal inputs are unsupported.

## Launch integration

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

`--seq-length` limits each tokenized prompt plus completion, including template
tokens. Oversized samples raise an error rather than being truncated. Errors
identify Slime's sample index, which may differ from the JSONL line after
shuffling.

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

## Verification

Run tokenizer and mask checks with local tokenizer files; weights are not
needed. Run from the project root with the development dependencies,
`transformers` and `tokenizers` installed. Prefer ModelScope when downloading
missing model artifacts.

```bash
DDPR_SLIME_ROOT=/path/to/slime \
DDPR_QWEN38_TOKENIZER=/path/to/local/Qwen3.8-27B-tokenizer \
python -m pytest -q tests/backends/slime/test_upstream_masks.py
```

On the configured DSW, test the installed Slime parser, loader and adapter:

```bash
cd /mnt/workspace/Projects/ddpr
DDPR_INSTALLED_SLIME=1 \
DDPR_SLIME_ROOT=/root/slime \
DDPR_QWEN38_TOKENIZER=/mnt/model/Qwen/Qwen3.8-27B \
python -m pytest -q
```

These tests cover token IDs, completion masks, conversation history, metadata
and argument defaults. They do not perform optimization. Rerun compatibility
checks when changing Slime or the student tokenizer.

### One-step Qwen3.8 smoke test

The [launcher](../../../configs/jobs/slime/qwen38_27b_smoke.sh) completed one SFT
Adam update on eight fresh CMPhysBench samples on 2026-09-14. It uses four
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
