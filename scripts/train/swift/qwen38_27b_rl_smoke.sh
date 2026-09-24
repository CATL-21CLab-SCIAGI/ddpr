#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export DDPR_REWARD="${DDPR_REWARD:-ddpr.rewards.SmokeReward}"

# Reuse regular RL settings, fixing only the smoke workload.
# Keep capped responses eligible for the short integration test.
exec bash "${SCRIPT_DIR}/qwen38_27b_rl.sh" "$@" \
    --max_steps 1 --max_length 2048 --max_completion_length 16 \
    --num_generations 2 --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 --overlong_filter false --save_strategy no
