#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DDPR_SLIME_ROOT="${DDPR_SLIME_ROOT:-/root/slime}"
DDPR_MODEL="${DDPR_MODEL:-/mnt/model/Qwen/Qwen3.8-27B}"
export DDPR_LOSS_MASK_TYPE=qwen3_5
export PYTHONPATH="${DDPR_SLIME_ROOT}:/root/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

source "${SCRIPT_DIR}/sft.sh"
source "${DDPR_SLIME_ROOT}/scripts/models/qwen3.5-27B.sh"

# Ray starts locally when needed. This script does not stop existing services.
# Four pipeline stages partition the model and Adam state across four GPUs.
python "${DDPR_SLIME_ROOT}/train.py" \
    "${MODEL_ARGS[@]}" "${DDPR_SFT_ARGS[@]}" \
    --hf-checkpoint "${DDPR_MODEL}" --load "${DDPR_MODEL}" \
    --actor-num-nodes 1 --actor-num-gpus-per-node 4 \
    --num-rollout 1 --rollout-batch-size 8 --global-batch-size 8 \
    --micro-batch-size 1 --seq-length 2048 \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 4 \
    --context-parallel-size 1 \
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
    --optimizer adam --lr 1e-5 --lr-decay-style constant --weight-decay 0.1 \
    --adam-beta1 0.9 --adam-beta2 0.98 \
    --use-distributed-optimizer --use-precision-aware-optimizer \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
    --attention-backend flash
