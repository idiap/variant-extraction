# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#!/usr/bin/env bash

set -euo pipefail

# =============================================================================
# User configuration
# =============================================================================

# Choose one of: zero_shot, baseline, vt, or any custom label.
MODEL_NAME="vt"

# Base Hugging Face model or local base-model path.
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"

# Adapter path.
# - For a single adapter evaluation, set this to one adapter directory.
# - For checkpoint-loop evaluation, set this to the adapter root that contains checkpoint-* dirs.
# - Leave empty for zero-shot/base-model evaluation.
ADAPTER="./outputs/llama31_8b_vt"

# If true, evaluate every checkpoint-* directory under ADAPTER.
# If false, evaluate ADAPTER itself once.
# Ignored when ADAPTER is empty.
LOOP_CHECKPOINTS=true

# Optional checkpoint glob, relative to ADAPTER.
# The default matches Hugging Face Trainer checkpoints: checkpoint-100, checkpoint-200, etc.
CHECKPOINT_GLOB="checkpoint-*"

# Tokenizer path.
# Leave empty to use the conditional defaults below:
#   - zero-shot / no adapter: BASE_MODEL
#   - baseline adapter: BASE_MODEL
#   - vt adapter, single run: ADAPTER
#   - vt adapter, checkpoint loop: current checkpoint directory
# You can override this with a fixed tokenizer path if needed.
TOKENIZER=""

# Evaluation size.
NUM_E2E_EXAMPLES=500
NUM_PPL_EXAMPLES=1000
MAX_NEW_TOKENS=80

# Output directory.
OUTPUT_DIR="results"

# Runtime options.
DTYPE="bfloat16"
PROMPT_STYLE="llama3"
SEED=42
TEMPERATURE=0.0
TOP_P=1.0

# Optional switches. Set to true or false.
LOAD_IN_4BIT=false
MERGE_ADAPTER=false
SKIP_GRAMMAR=false
SKIP_PPL=false

# Optional local perplexity file. Leave empty to use WikiText-2.
PPL_TEXT_FILE=""

# Cache directories.
E2E_CACHE_DIR="data/e2e_nlg"
WIKITEXT_CACHE_DIR="data/wikitext2"

# Extra raw arguments appended to the Python command.
# Example: EXTRA_ARGS=(--grammar_device cuda:0)
EXTRA_ARGS=()

# Python script to run.
SCRIPT="eval_single_model.py"


# =============================================================================
# Helpers
# =============================================================================

bool_is_true() {
  [[ "${1}" == "true" || "${1}" == "1" || "${1}" == "yes" || "${1}" == "y" ]]
}

step_from_checkpoint_path() {
  local path="$1"
  local base
  base="$(basename "${path}")"
  if [[ "${base}" =~ ^checkpoint-([0-9]+)$ ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
  else
    printf '%s' "${base}"
  fi
}

resolve_tokenizer() {
  local model_name="$1"
  local adapter_path="$2"
  local default_tokenizer="$3"

  if [[ -n "${default_tokenizer}" ]]; then
    printf '%s' "${default_tokenizer}"
    return
  fi

  if [[ -z "${adapter_path}" ]]; then
    printf '%s' "${BASE_MODEL}"
    return
  fi

  case "${model_name}" in
    baseline)
      printf '%s' "${BASE_MODEL}"
      ;;
    vt)
      printf '%s' "${adapter_path}"
      ;;
    *)
      printf '%s' "${adapter_path}"
      ;;
  esac
}

run_one_eval() {
  local run_model_name="$1"
  local adapter_path="$2"
  local tokenizer_path="$3"
  local output_csv="$4"
  local summary_csv="$5"

  local cmd=(
    python "${SCRIPT}"
    --model_name "${run_model_name}"
    --base_model "${BASE_MODEL}"
    --tokenizer "${tokenizer_path}"
    --num_e2e_examples "${NUM_E2E_EXAMPLES}"
    --num_ppl_examples "${NUM_PPL_EXAMPLES}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --dtype "${DTYPE}"
    --prompt_style "${PROMPT_STYLE}"
    --seed "${SEED}"
    --temperature "${TEMPERATURE}"
    --top_p "${TOP_P}"
    --e2e_cache_dir "${E2E_CACHE_DIR}"
    --wikitext_cache_dir "${WIKITEXT_CACHE_DIR}"
    --output_csv "${output_csv}"
    --summary_csv "${summary_csv}"
  )

  if [[ -n "${adapter_path}" ]]; then
    cmd+=(--adapter "${adapter_path}")
  fi

  if bool_is_true "${LOAD_IN_4BIT}"; then
    cmd+=(--load_in_4bit)
  fi

  if bool_is_true "${MERGE_ADAPTER}"; then
    cmd+=(--merge_adapter)
  fi

  if bool_is_true "${SKIP_GRAMMAR}"; then
    cmd+=(--skip_grammar)
  fi

  if bool_is_true "${SKIP_PPL}"; then
    cmd+=(--skip_ppl)
  fi

  if [[ -n "${PPL_TEXT_FILE}" ]]; then
    cmd+=(--ppl_text_file "${PPL_TEXT_FILE}")
  fi

  cmd+=("${EXTRA_ARGS[@]}")

  printf '\nRunning command:\n'
  printf ' %q' "${cmd[@]}"
  printf '\n\n'

  "${cmd[@]}"
}

combine_summaries() {
  local output_dir="$1"
  local combined_csv="$2"

  python - "$output_dir" "$combined_csv" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

output_dir = Path(sys.argv[1])
combined_csv = Path(sys.argv[2])

summary_files = sorted(output_dir.glob("*_summary.csv"))
summary_files = [p for p in summary_files if p.name != combined_csv.name]

frames = []
for path in summary_files:
    df = pd.read_csv(path)
    stem = path.name[:-len("_summary.csv")]
    match = re.search(r"_step_([^_]+)$", stem)
    step = match.group(1) if match else ""
    df.insert(0, "run", stem)
    df.insert(1, "step", step)
    df.insert(2, "summary_file", str(path))
    frames.append(df)

if frames:
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(combined_csv, index=False)
    print(f"Saved combined checkpoint summary to: {combined_csv}")
else:
    print(f"No summary files found under: {output_dir}")
PY
}


# =============================================================================
# Execution
# =============================================================================

mkdir -p "${OUTPUT_DIR}"

if [[ -z "${ADAPTER}" ]]; then
  # Zero-shot/base model evaluation.
  RUN_MODEL_NAME="${MODEL_NAME:-zero_shot}"
  RUN_TOKENIZER="$(resolve_tokenizer "${RUN_MODEL_NAME}" "" "${TOKENIZER}")"
  OUTPUT_CSV="${OUTPUT_DIR}/${RUN_MODEL_NAME}_results.csv"
  SUMMARY_CSV="${OUTPUT_DIR}/${RUN_MODEL_NAME}_summary.csv"

  run_one_eval "${RUN_MODEL_NAME}" "" "${RUN_TOKENIZER}" "${OUTPUT_CSV}" "${SUMMARY_CSV}"
  exit 0
fi

if bool_is_true "${LOOP_CHECKPOINTS}"; then
  if [[ ! -d "${ADAPTER}" ]]; then
    echo "ERROR: ADAPTER must be a directory when LOOP_CHECKPOINTS=true: ${ADAPTER}" >&2
    exit 1
  fi

  mapfile -t CHECKPOINTS < <(
    find "${ADAPTER}" -mindepth 1 -maxdepth 1 -type d -name "${CHECKPOINT_GLOB}" \
      | sort -V
  )

  if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
    echo "ERROR: no checkpoint directories matching ${ADAPTER}/${CHECKPOINT_GLOB}" >&2
    exit 1
  fi

  echo "Found ${#CHECKPOINTS[@]} checkpoints under ${ADAPTER}"

  for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    STEP="$(step_from_checkpoint_path "${CHECKPOINT}")"
    RUN_MODEL_NAME="${MODEL_NAME}_step_${STEP}"
    RUN_TOKENIZER="$(resolve_tokenizer "${MODEL_NAME}" "${CHECKPOINT}" "${TOKENIZER}")"
    OUTPUT_CSV="${OUTPUT_DIR}/${RUN_MODEL_NAME}_results.csv"
    SUMMARY_CSV="${OUTPUT_DIR}/${RUN_MODEL_NAME}_summary.csv"

    echo ""
    echo "=============================================================================="
    echo "Evaluating checkpoint: ${CHECKPOINT}"
    echo "Step: ${STEP}"
    echo "Model name: ${RUN_MODEL_NAME}"
    echo "Tokenizer: ${RUN_TOKENIZER}"
    echo "=============================================================================="

    run_one_eval "${RUN_MODEL_NAME}" "${CHECKPOINT}" "${RUN_TOKENIZER}" "${OUTPUT_CSV}" "${SUMMARY_CSV}"
  done

  combine_summaries "${OUTPUT_DIR}" "${OUTPUT_DIR}/${MODEL_NAME}_checkpoint_summary.csv"
else
  # Single adapter evaluation.
  RUN_MODEL_NAME="${MODEL_NAME}"
  RUN_TOKENIZER="$(resolve_tokenizer "${MODEL_NAME}" "${ADAPTER}" "${TOKENIZER}")"
  OUTPUT_CSV="${OUTPUT_DIR}/${RUN_MODEL_NAME}_results.csv"
  SUMMARY_CSV="${OUTPUT_DIR}/${RUN_MODEL_NAME}_summary.csv"

  run_one_eval "${RUN_MODEL_NAME}" "${ADAPTER}" "${RUN_TOKENIZER}" "${OUTPUT_CSV}" "${SUMMARY_CSV}"
fi
