# Source this fragment from a model-specific Swift launch script.
# Model, checkpoints, generation limits, optimizer and GPU settings belong there.
# No processes are started by this file.
export DDPR_SFT_DATA="${DDPR_SFT_DATA:?Set DDPR_SFT_DATA to the exported sft.jsonl}"
DDPR_SWIFT_PLUGINS=("$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/../../../ddpr/backends/swift/plugin.py")
if [[ -n "${DDPR_REWARD_PLUGIN:-}" ]]; then
    DDPR_SWIFT_PLUGINS+=("${DDPR_REWARD_PLUGIN}")
fi
: "${DDPR_REWARD_FUNCS:?Set DDPR_REWARD_FUNCS to registered Swift reward names}"
read -r -a DDPR_REWARDS <<< "${DDPR_REWARD_FUNCS}"
if (( ${#DDPR_REWARDS[@]} == 0 )); then
    echo "DDPR_REWARD_FUNCS must contain at least one reward name" >&2
    return 1
fi
DDPR_RL_ARGS=(
    --external_plugins "${DDPR_SWIFT_PLUGINS[@]}"
    --rlhf_type grpo --dataset ddpr_rl
    --strict true --remove_unused_columns false
    --reward_funcs "${DDPR_REWARDS[@]}"
)
