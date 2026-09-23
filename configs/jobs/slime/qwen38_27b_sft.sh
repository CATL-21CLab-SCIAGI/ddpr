#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DDPR_SLIME_ROOT="${DDPR_SLIME_ROOT:-/root/slime}"
DDPR_MODEL="${DDPR_MODEL:-/mnt/model/Qwen/Qwen3.8-27B}"
: "${DDPR_OUTPUT_DIR:?Set DDPR_OUTPUT_DIR to the checkpoint directory}"
: "${DDPR_ACTOR_GPUS:?Set DDPR_ACTOR_GPUS for one-node training}"
: "${DDPR_NUM_ROLLOUT:?Set DDPR_NUM_ROLLOUT to the total number of SFT updates}"
DDPR_BATCH_SIZE="${DDPR_BATCH_SIZE:-8}"
export DDPR_LOSS_MASK_TYPE=qwen3_5
export PYTHONPATH="${DDPR_SLIME_ROOT}:${DDPR_MEGATRON_ROOT:-/root/Megatron-LM}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

source "${SCRIPT_DIR}/sft.sh"
source "${DDPR_SLIME_ROOT}/scripts/models/qwen3.5-27B.sh"

# One pipeline stage per actor GPU; no generation workers are started.
python "${DDPR_SLIME_ROOT}/train.py" \
    "${MODEL_ARGS[@]}" "${DDPR_SFT_ARGS[@]}" \
    --hf-checkpoint "${DDPR_MODEL}" --load "${DDPR_LOAD:-${DDPR_MODEL}}" \
    --save "${DDPR_OUTPUT_DIR}" --save-interval "${DDPR_SAVE_INTERVAL:-100}" \
    --actor-num-nodes 1 --actor-num-gpus-per-node "${DDPR_ACTOR_GPUS}" \
    --num-gpus-per-node "${DDPR_ACTOR_GPUS}" \
    --num-rollout "${DDPR_NUM_ROLLOUT}" \
    --rollout-batch-size "${DDPR_BATCH_SIZE}" --global-batch-size "${DDPR_BATCH_SIZE}" \
    --micro-batch-size 1 --seq-length "${DDPR_MAX_LENGTH:-8192}" \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size "${DDPR_ACTOR_GPUS}" \
    --context-parallel-size 1 \
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
    --optimizer adam --lr 1e-5 --lr-decay-style constant --weight-decay 0.1 \
    --adam-beta1 0.9 --adam-beta2 0.98 \
    --use-distributed-optimizer --use-precision-aware-optimizer \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
    --attention-backend flash \
    "$@"
