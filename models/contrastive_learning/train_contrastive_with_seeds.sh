# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# =========================
# Configuration
# =========================

BASE_MODEL_NAME_OR_PATH="meta-llama/Llama-3.1-8B-Instruct"

RUN_NAMES=(
  "llama31_8b_baseline"
  "llama31_8b_vt"
)

SEEDS=(
  "a"
  "b"
  "c"
)

# =========================
# Train contrastive models
# =========================

for RUN_NAME in "${RUN_NAMES[@]}"; do
  echo
  echo "##################################################"
  echo "Starting run: ${RUN_NAME}"
  echo "##################################################"

  for SEED in "${SEEDS[@]}"; do
    echo
    echo "=================================================="
    echo "Starting contrastive training"
    echo "Run:  ${RUN_NAME}"
    echo "Seed: ${SEED}"
    echo "=================================================="

    SFT_ADAPTER_PATH="./outputs_final/${RUN_NAME}_${SEED}"
    TOKENIZER_PATH="${SFT_ADAPTER_PATH}"

    CONTRASTIVE_DATA_DIR="../../data/contrastive_merged_1${SEED}"
    TRAIN_FILE="${CONTRASTIVE_DATA_DIR}/contrastive_train.jsonl"
    VALID_FILE="${CONTRASTIVE_DATA_DIR}/contrastive_val.jsonl"

    OUTPUT_DIR="./outputs_final/${RUN_NAME}_contrastive_${SEED}"

    echo "  SFT adapter:      ${SFT_ADAPTER_PATH}"
    echo "  Tokenizer:        ${TOKENIZER_PATH}"
    echo "  Training file:    ${TRAIN_FILE}"
    echo "  Validation file:  ${VALID_FILE}"
    echo "  Output directory: ${OUTPUT_DIR}"

    python "${SCRIPT_DIR}/train_contrastive_from_sft.py" \
      --base_model_name_or_path "${BASE_MODEL_NAME_OR_PATH}" \
      --adapter_path "${SFT_ADAPTER_PATH}" \
      --tokenizer_name_or_path "${TOKENIZER_PATH}" \
      --train_file "${TRAIN_FILE}" \
      --valid_file "${VALID_FILE}" \
      --output_dir "${OUTPUT_DIR}" \
      --load_in_4bit \
      --bf16 \
      --gradient_checkpointing \
      --max_context_length 4096 \
      --max_target_length 256 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 8 \
      --learning_rate 5e-5 \
      --num_train_epochs 1

    echo
    echo "Contrastive training completed successfully:"
    echo "  Run:  ${RUN_NAME}"
    echo "  Seed: ${SEED}"
  done

  echo
  echo "All seeds completed successfully for ${RUN_NAME}."
done

echo
echo "All contrastive runs completed successfully."