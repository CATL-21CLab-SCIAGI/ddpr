# Source this fragment from a model-specific Slime launch script.
# Model, checkpoints, batch sizes, optimizer and GPU settings belong there.
# No processes are started by this file.
DDPR_RL_ARGS=(
    --prompt-data "${DDPR_SFT_DATA:?Set DDPR_SFT_DATA to the exported sft.jsonl}"
    --input-key prompt --label-key completion --metadata-key metadata
    --data-source-path ddpr.backends.slime.plugin.RolloutDataSource
    --rollout-function-path slime.rollout.sglang_rollout.generate_rollout
    --apply-chat-template --rollout-shuffle
    --loss-type policy_loss --advantage-estimator grpo
    --custom-rm-path "${DDPR_REWARD_FUNCTION:?Set DDPR_REWARD_FUNCTION to an async module.function}"
)
