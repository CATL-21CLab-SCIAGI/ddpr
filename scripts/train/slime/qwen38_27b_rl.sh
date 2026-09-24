#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DDPR_SLIME_ROOT="${DDPR_SLIME_ROOT:-/root/slime}"
DDPR_MODEL="${DDPR_MODEL:-/mnt/model/Qwen/Qwen3.8-27B}"
: "${DDPR_OUTPUT_DIR:?Set DDPR_OUTPUT_DIR to the checkpoint directory}"
: "${DDPR_ACTOR_GPUS:?Set DDPR_ACTOR_GPUS for one-node training}"
: "${DDPR_ROLLOUT_GPUS:?Set DDPR_ROLLOUT_GPUS for separate generation GPUs}"
: "${DDPR_ROLLOUT_GPUS_PER_ENGINE:?Set DDPR_ROLLOUT_GPUS_PER_ENGINE}"
export PYTHONPATH="${DDPR_SLIME_ROOT}:${DDPR_MEGATRON_ROOT:-/root/Megatron-LM}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

source "${SCRIPT_DIR}/rl.sh"
source "${DDPR_SLIME_ROOT}/scripts/models/qwen3.5-27B.sh"

# Selective recomputation preserves gradients when the embedding/prefix is frozen.
RECOMPUTE_ARGS=(--recompute-granularity "${DDPR_RECOMPUTE_GRANULARITY:-full}")
if [[ "${DDPR_RECOMPUTE_GRANULARITY:-full}" == full ]]; then
    RECOMPUTE_ARGS+=(--recompute-method uniform --recompute-num-layers 1)
fi

# Separate actor and generation GPUs; reassess memory on the target host.
python -m ddpr.backends.slime.train \
    "${MODEL_ARGS[@]}" "${DDPR_RL_ARGS[@]}" \
    --hf-checkpoint "${DDPR_MODEL}" --load "${DDPR_LOAD:-${DDPR_MODEL}}" \
    --save "${DDPR_OUTPUT_DIR}" --save-interval 1 \
    --actor-num-nodes 1 --actor-num-gpus-per-node "${DDPR_ACTOR_GPUS}" \
    --rollout-num-gpus "${DDPR_ROLLOUT_GPUS}" \
    --rollout-num-gpus-per-engine "${DDPR_ROLLOUT_GPUS_PER_ENGINE}" \
    --num-gpus-per-node "$((DDPR_ACTOR_GPUS + DDPR_ROLLOUT_GPUS))" \
    --num-rollout "${DDPR_NUM_ROLLOUT:-1}" --rollout-batch-size 8 --n-samples-per-prompt 4 \
    --global-batch-size 32 --micro-batch-size 1 \
    --rollout-max-prompt-len 4096 --rollout-max-response-len 2048 --seq-length 6144 \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size "${DDPR_ACTOR_GPUS}" \
    --context-parallel-size 1 \
    "${RECOMPUTE_ARGS[@]}" \
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 \
    --adam-beta1 0.9 --adam-beta2 0.98 \
    --use-distributed-optimizer --use-precision-aware-optimizer \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
    --attention-backend flash \
    "$@"
