# Slime SFT

## Input and supervision

Use the `sft.jsonl` already exported by ddsr-bench. A record contains:

```json
{"id":"trial:formatting","prompt":[{"role":"user","content":"Question"},{"role":"assistant","content":"Derivation"},{"role":"user","content":"Format the answer"}],"completion":[{"role":"assistant","content":"Final code"}],"metadata":{"benchmark":"critpt","stage":"formatting"}}
```

Slime's dataset loader maps `prompt` to `sample.prompt`, `completion` to
`sample.label`, and `metadata` to `sample.metadata`. ddpr combines the two
message lists in memory, masks every context message with `step_loss_mask=0`,
and supervises the completion with `step_loss_mask=1`. In this example the
earlier derivation remains in context but only the final code receives loss.
Slime's template may also supervise completion terminator tokens.

The adapter accepts text-only system/user/assistant messages and exactly one
nonempty assistant completion, matching ddsr-bench's current export. It retains
metadata and does not mutate the loaded prompt or completion. The standard
Slime loader does not retain the top-level `id`; existing trajectory/problem/
stage metadata remains available. `load_sample(record)` preserves the top-level
`id` in `SftSample.id`, or uses `None` when it is absent. The Slime adapter passes
a mapping of its loaded fields to this same entry point. The loader does not
move the ID into metadata.

Export view selection (`native`, `full`, and benchmark-specific views) remains
ddsr-bench's responsibility. ddpr neither adds derived examples nor filters
by quality. Select appropriate examples before training: static format
validation alone does not establish correctness. Separately recorded provider
reasoning is not automatically added to the visible completion.

## Launch integration

Install ddpr in the Python environment used by every Slime worker. For editable
installation the source directory must be accessible on each node; alternatively
install a built ddpr wheel on each worker. No ddsr-bench installation is needed.

Use Slime's model-specific SFT launch script and replace its SFT arguments with
the [sourceable fragment](../../../configs/jobs/slime/sft.sh):

```bash
export DDPR_SFT_DATA=/absolute/path/to/sft.jsonl
export DDPR_LOSS_MASK_TYPE=qwen3  # example only; match the student tokenizer
source /absolute/path/to/ddpr/configs/jobs/slime/sft.sh
```

Pass `"${DDPR_SFT_ARGS[@]}"` to the script's training command, together with its
model/checkpoint, optimizer, parallelism and batch settings. Configure epochs
or rollout count explicitly. The fragment is not a standalone launch command.
It performs no cleanup, downloads or model generation.

Leave `--apply-chat-template` unset: Slime's mask generator must receive the
original message list. Keep the global dataset enabled, one sample per prompt,
and SFT loss with advantage computation disabled. Do not configure an RL reward
key or generation-based evaluation with this adapter.

The adapter supports Slime's `qwen`, `qwen3`, and `qwen3_5` multi-turn mask paths.
It rejects `distill_qwen`, including its automatic selection under `qwen`,
because that path keeps only the first prompt message. A matching mode name
does not guarantee compatibility with every checkpoint: verify rendered text,
token IDs, and supervised spans using the intended student tokenizer.

Slime's prompt-length filter skips unrendered message lists. ddpr checks full
tokenized length against `args.seq_length` when configured and fails rather
than silently truncating a target. Set a suitable sequence length in the launch
script. Invalid data reports Slime's sample index, not a JSONL line number,
because the rollout receives already loaded, potentially shuffled examples.

## Compatibility reference

Reviewed against THUDM/slime commit
`4c193f1f37509cca70f0e88807a9305b70f63f4e`:

- [Dataset loader](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/utils/data.py)
- [SFT batch preparation](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/rollout/sft_rollout.py)
- [Mask generation](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/utils/mask_utils.py)
- [Official SFT launch example](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/scripts/run-qwen3-4B-base-sft.sh)

ddpr prepares the same token IDs, response length, reward placeholder, and
response loss-mask fields as Slime's built-in SFT rollout. Slime remains
responsible for packing batches, loss computation, optimization and checkpointing.
Use the recorded revision for reproducibility and rerun compatibility checks
when updating Slime.
