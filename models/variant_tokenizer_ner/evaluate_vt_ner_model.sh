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

for SEED in "${SEEDS[@]}"; do
  echo
  echo "=================================================="
  echo "Starting VT seed: ${SEED}"
  echo "=================================================="

  DATA_DIR="../../data/processed_merged_1${SEED}"
  SEED_TEST_JSONL="${DATA_DIR}/test.jsonl"

  TRAIN_OUTPUT_DIR="./outputs_final/${RUN_NAME}_${SEED}"
  ADAPTER_PATH="${TRAIN_OUTPUT_DIR}"
  TOKENIZER_PATH="${TRAIN_OUTPUT_DIR}"

  BRONCO_EVAL_OUTPUT_DIR="../../evaluation_final/${RUN_NAME}_bronco_${SEED}"
  SEED_TEST_EVAL_OUTPUT_DIR="../../evaluation_final/${RUN_NAME}_test_${SEED}"

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