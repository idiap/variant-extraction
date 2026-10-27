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

# Change this value to rename all training and evaluation outputs.
RUN_NAME="llama31_8b_baseline"

SEEDS=(
  "a"
  "b"
  "c"
)

BRONCO_TEST_JSONL="../../data/processed_bronco/full.jsonl"

# =========================
# Train and evaluate each seed
# =========================

for SEED in "${SEEDS[@]}"; do
  echo
  echo "=================================================="
  echo "Starting seed: ${SEED}"
  echo "=================================================="

  # Seed-specific datasets.
  DATA_DIR="../../data/processed_merged_1${SEED}"
  TRAIN_FILE="${DATA_DIR}/train.jsonl"
  VALID_FILE="${DATA_DIR}/val.jsonl"
  SEED_TEST_JSONL="${DATA_DIR}/test.jsonl"

  # Seed-specific model output and adapter.
  TRAIN_OUTPUT_DIR="./outputs_final/${RUN_NAME}_${SEED}"
  ADAPTER_PATH="${TRAIN_OUTPUT_DIR}"

  # Seed-specific evaluation output directories.
  BRONCO_EVAL_OUTPUT_DIR="../../evaluation_final/${RUN_NAME}_bronco_${SEED}"
  SEED_TEST_EVAL_OUTPUT_DIR="../../evaluation_final/${RUN_NAME}_test_${SEED}"

  # =========================
  # Training
  # =========================

  echo
  echo "Starting training for seed ${SEED}..."
  echo "  Training file:    ${TRAIN_FILE}"
  echo "  Validation file:  ${VALID_FILE}"
  echo "  Output directory: ${TRAIN_OUTPUT_DIR}"

  # To use Accelerate, replace the next command with:
  # accelerate launch "${SCRIPT_DIR}/train_baseline_ner.py" \
  python "${SCRIPT_DIR}/train_baseline_ner.py" \
    --model_name "${BASE_MODEL_NAME_OR_PATH}" \
    --train_file "${TRAIN_FILE}" \
    --valid_file "${VALID_FILE}" \
    --output_dir "${TRAIN_OUTPUT_DIR}" \
    --bf16 \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --eval_steps 500 \
    --save_steps 10000

  echo "Training completed successfully for seed ${SEED}."

  # =========================
  # Evaluation datasets
  # =========================

  TEST_JSONL_FILES=(
    "${BRONCO_TEST_JSONL}"
    "${SEED_TEST_JSONL}"
  )

  EVAL_OUTPUT_DIRS=(
    "${BRONCO_EVAL_OUTPUT_DIR}"
    "${SEED_TEST_EVAL_OUTPUT_DIR}"
  )

  # =========================
  # Evaluation
  # =========================

  for i in "${!TEST_JSONL_FILES[@]}"; do
    TEST_JSONL="${TEST_JSONL_FILES[$i]}"
    EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIRS[$i]}"

    echo
    echo "Starting evaluation for seed ${SEED}:"
    echo "  Test file:        ${TEST_JSONL}"
    echo "  Adapter path:     ${ADAPTER_PATH}"
    echo "  Output directory: ${EVAL_OUTPUT_DIR}"

    python "${SCRIPT_DIR}/evaluate_baseline_ner.py" \
      --test-jsonl "${TEST_JSONL}" \
      --output-dir "${EVAL_OUTPUT_DIR}" \
      --base-model-name-or-path "${BASE_MODEL_NAME_OR_PATH}" \
      --adapter-path "${ADAPTER_PATH}"

    echo "Evaluation completed for seed ${SEED}: ${TEST_JSONL}"
  done

  echo
  echo "Seed ${SEED} completed successfully."
done

echo
echo "All training and evaluation runs completed successfully."