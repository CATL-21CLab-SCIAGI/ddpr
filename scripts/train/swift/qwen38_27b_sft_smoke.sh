#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Reuse the regular SFT configuration, fixing duration and checkpoint behavior.
# Trailing user options still select the device, precision and parallelism.
exec bash "${SCRIPT_DIR}/qwen38_27b_sft.sh" "$@" \
    --max_steps 1 --max_length 2048 --save_strategy no
