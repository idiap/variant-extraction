# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

TEST_JSONL="../../data/processed_bronco/full.jsonl"
OUTPUT_DIR="../../evaluation/llama31_8b_baseline_bronco"
BASE_MODEL_NAME_OR_PATH="meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH="./outputs_final/llama31_8b_baseline"

python "${SCRIPT_DIR}/evaluate_baseline_ner.py" \
  --test-jsonl "${TEST_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --base-model-name-or-path "${BASE_MODEL_NAME_OR_PATH}" \
  --adapter-path "${ADAPTER_PATH}"