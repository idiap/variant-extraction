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
RUN_NAME="llama31_8b_vt"

SEEDS=("a" "b" "c")

BRONCO_TEST_JSONL="../../data/processed_bronco/full.jsonl"
RESOURCES_DIR="../../resources"
HGNC_DATE="2026-05-05"
ADD_GENE_NAMES="None"
ADD_DNA_TOKENS=0

for SEED in "${SEEDS[@]}"; do
  echo
  echo "=================================================="
  echo "Starting VT seed: ${SEED}"
  echo "=================================================="

  DATA_DIR="../../data/processed_merged_1${SEED}"
  TRAIN_FILE="${DATA_DIR}/train.jsonl"
  VALID_FILE="${DATA_DIR}/val.jsonl"
  SEED_TEST_JSONL="${DATA_DIR}/test.jsonl"

  TRAIN_OUTPUT_DIR="./outputs_final/${RUN_NAME}_${SEED}"
  WARMUP_ADAPTER_PATH="./outputs_final/${RUN_NAME}_warmup"
  ADAPTER_PATH="${TRAIN_OUTPUT_DIR}"
  TOKENIZER_PATH="${TRAIN_OUTPUT_DIR}"

  BRONCO_EVAL_OUTPUT_DIR="../../evaluation_final/${RUN_NAME}_bronco_${SEED}"
  SEED_TEST_EVAL_OUTPUT_DIR="../../evaluation_final/${RUN_NAME}_test_${SEED}"

  echo "Starting VT training for seed ${SEED}..."

  # To use Accelerate, replace the next command with:
  # accelerate launch "${SCRIPT_DIR}/train_vt_ner.py" \
  python "${SCRIPT_DIR}/train_vt_ner.py" \
    --model_name "${BASE_MODEL_NAME_OR_PATH}" \
    --train_file "${TRAIN_FILE}" \
    --valid_file "${VALID_FILE}" \
    --output_dir "${TRAIN_OUTPUT_DIR}" \
    --add_gene_names "${ADD_GENE_NAMES}" \
    --resources_dir "${RESOURCES_DIR}" \
    --hgnc_date "${HGNC_DATE}" \
    --add_dna_tokens "${ADD_DNA_TOKENS}" \
    --bf16 \
    --use_trainable_token_indices \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --eval_steps 500 \
    --save_steps 10000

  echo "VT training completed for seed ${SEED}."

  TEST_JSONL_FILES=("${BRONCO_TEST_JSONL}" "${SEED_TEST_JSONL}")
  EVAL_OUTPUT_DIRS=("${BRONCO_EVAL_OUTPUT_DIR}" "${SEED_TEST_EVAL_OUTPUT_DIR}")

  for i in "${!TEST_JSONL_FILES[@]}"; do
    TEST_JSONL="${TEST_JSONL_FILES[$i]}"
    EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIRS[$i]}"

    echo
    echo "Evaluating seed ${SEED}"
    echo "  Test file: ${TEST_JSONL}"
    echo "  Output:    ${EVAL_OUTPUT_DIR}"

    python "${SCRIPT_DIR}/evaluate_vt_ner_model.py" \
      --test-jsonl "${TEST_JSONL}" \
      --output-dir "${EVAL_OUTPUT_DIR}" \
      --base-model-name-or-path "${BASE_MODEL_NAME_OR_PATH}" \
      --adapter-path "${ADAPTER_PATH}" \
      --tokenizer-path "${TOKENIZER_PATH}"
  done

done

echo
echo "All VT seed training and benchmark runs completed successfully."
