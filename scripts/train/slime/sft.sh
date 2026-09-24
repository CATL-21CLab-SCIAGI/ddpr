# Source this fragment from a model-specific Slime launch script.
# Model, checkpoints, batch sizes, sequence length, optimizer and GPU settings
# belong to that launch script. No processes are started by this file.
DDPR_SFT_ARGS=(
    --prompt-data "${DDPR_SFT_DATA:?Set DDPR_SFT_DATA to the exported sft.jsonl}"
    --input-key prompt
    --label-key completion
    --metadata-key metadata
    --tool-key ""
    --data-source-path ddpr.backends.slime.plugin.RolloutDataSource
    --rollout-function-path ddpr.backends.slime.sft.generate_rollout
    --loss-mask-type "${DDPR_LOSS_MASK_TYPE:?Set the student loss mask type}"
    --n-samples-per-prompt 1
    --loss-type sft_loss
    --calculate-per-token-loss
    --disable-compute-advantages-and-returns
    --debug-train-only
    --rollout-shuffle
)
