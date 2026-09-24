#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/tests/backends/slime:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DDPR_REWARD="${DDPR_REWARD:-ddpr.rewards.SmokeReward}"

# Inherit training settings; fix only the smoke workload and verification.
exec bash "${SCRIPT_DIR}/qwen38_27b_rl.sh" "$@" \
    --num-rollout 1 --rollout-batch-size 1 --n-samples-per-prompt 2 \
    --global-batch-size 2 --micro-batch-size 1 --seq-length 2048 \
    --rollout-max-prompt-len 1536 --rollout-max-response-len 16 \
    --save-interval 1 --group-rm \
    --custom-megatron-before-train-step-hook-path smoke_checks.before_step
