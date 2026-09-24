# Source this fragment from a model-specific Swift launch script.
# Model, checkpoints, batch sizes, sequence length, optimizer and GPU settings
# belong to that launch script. No processes are started by this file.
export DDPR_SFT_DATA="${DDPR_SFT_DATA:?Set DDPR_SFT_DATA to the exported sft.jsonl}"
DDPR_SWIFT_PLUGIN="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/../../../ddpr/backends/swift/plugin.py"
DDPR_SFT_ARGS=(
    --external_plugins "${DDPR_SWIFT_PLUGIN}"
    --dataset ddpr_sft
    --strict true --remove_unused_columns false
    --loss_scale default
)
