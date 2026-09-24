#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/sft.sh"
: "${DDPR_OUTPUT_DIR:?Set DDPR_OUTPUT_DIR to the checkpoint output directory}"

swift sft "${DDPR_SFT_ARGS[@]}" \
    --model "${DDPR_MODEL:-Qwen/Qwen3.8-27B}" \
    --model_type qwen3_5 --template qwen3_8 --use_hf false \
    --torch_dtype bfloat16 --report_to none --split_dataset_ratio 0 \
    --output_dir "${DDPR_OUTPUT_DIR}" \
    --tuner_type full --learning_rate 1e-5 \
    --num_train_epochs 1 --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 --gradient_checkpointing true \
    --max_length "${DDPR_MAX_LENGTH:-8192}" --truncation_strategy delete \
    --logging_steps 1 --save_steps 100 \
    "$@"
