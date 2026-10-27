# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#!/usr/bin/env bash

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

# Choose one: zero_shot, baseline, vt
MODEL_TYPE="vt"

# Human-readable run name. Leave empty to auto-generate.
# Examples: zeroshot, baseline_final, vt_all_ckpts
RUN_NAME=""

# Base model used for zero-shot and as the parent model for LoRA adapters.
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"

# Adapter path.
# - Empty for zero_shot.
# - For one fine-tuned model: path to adapter directory.
# - For checkpoint loop: root directory containing checkpoint-* directories.
ADAPTER_ROOT="./outputs/llama31_8b_vt"

# Evaluate all checkpoint-* directories under ADAPTER_ROOT.
# Ignored for zero_shot.
LOOP_CHECKPOINTS=true
CHECKPOINT_GLOB="checkpoint-*"

# Tokenizer override. Leave empty for defaults:
# - zero_shot: BASE_MODEL
# - baseline: BASE_MODEL
# - vt single/checkpoint: adapter/checkpoint directory
TOKENIZER=""

# Evaluation settings.
NUM_EXAMPLES=1000
WIKITEXT_SPLIT="test"
PPL_MAX_LENGTH=1024
PPL_STRIDE=512

# Save model continuations for each WikiText input prefix.
# This adds generation time. Set false for pure perplexity speed.
GENERATE_OUTPUTS=true
MAX_NEW_TOKENS=64
TEMPERATURE=0.0
TOP_P=1.0

# Runtime settings.
DTYPE="bfloat16"
LOAD_IN_4BIT=false
MERGE_ADAPTER=false
SEED=42

# Cache/output.
WIKITEXT_CACHE_DIR="data/wikitext2"
OUTPUT_ROOT="."

# Python script.
SCRIPT="eval_wikitext2_perplexity.py"

# Extra args passed directly to Python.
EXTRA_ARGS=()


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
  local model_type="$1"
  local adapter_path="$2"

  if [[ -n "${TOKENIZER}" ]]; then
    printf '%s' "${TOKENIZER}"
    return
  fi

  if [[ "${model_type}" == "zero_shot" ]]; then
    printf '%s' "${BASE_MODEL}"
  elif [[ "${model_type}" == "baseline" ]]; then
    printf '%s' "${BASE_MODEL}"
  elif [[ "${model_type}" == "vt" ]]; then
    printf '%s' "${adapter_path}"
  else
    printf '%s' "${BASE_MODEL}"
  fi
}

run_one() {
  local model_type="$1"
  local run_name="$2"
  local adapter_path="$3"
  local tokenizer_path="$4"
  local output_dir="$5"

  local cmd=(
    python "${SCRIPT}"
    --model_type "${model_type}"
    --run_name "${run_name}"
    --base_model "${BASE_MODEL}"
    --tokenizer "${tokenizer_path}"
    --num_examples "${NUM_EXAMPLES}"
    --wikitext_split "${WIKITEXT_SPLIT}"
    --wikitext_cache_dir "${WIKITEXT_CACHE_DIR}"
    --ppl_max_length "${PPL_MAX_LENGTH}"
    --ppl_stride "${PPL_STRIDE}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top_p "${TOP_P}"
    --dtype "${DTYPE}"
    --seed "${SEED}"
    --output_dir "${output_dir}"
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
  if bool_is_true "${GENERATE_OUTPUTS}"; then
    cmd+=(--generate_outputs)
  fi

  cmd+=("${EXTRA_ARGS[@]}")

  echo
  echo "Running:"
  printf ' %q' "${cmd[@]}"
  echo
  echo
  "${cmd[@]}"
}

combine_checkpoint_summaries() {
  local parent_dir="$1"
  local output_csv="$2"

  python - "$parent_dir" "$output_csv" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

parent = Path(sys.argv[1])
out = Path(sys.argv[2])

summary_files = sorted(parent.glob("results_*/summary.csv"))
rows = []
for path in summary_files:
    df = pd.read_csv(path)
    if df.empty:
        continue
    row = df.iloc[0].to_dict()
    run_dir = path.parent.name
    row["run_dir"] = str(path.parent)
    row["summary_file"] = str(path)
    if "checkpoint_step" not in row or pd.isna(row.get("checkpoint_step")):
        m = re.search(r"checkpoint-(\d+)|step_(\d+)", str(path))
        if m:
            row["checkpoint_step"] = int(m.group(1) or m.group(2))
    rows.append(row)

if not rows:
    print(f"No summary.csv files found under {parent}/results_*/summary.csv")
    raise SystemExit(0)

combined = pd.DataFrame(rows)
if "checkpoint_step" in combined.columns:
    combined["checkpoint_step_numeric"] = pd.to_numeric(combined["checkpoint_step"], errors="coerce")
    combined = combined.sort_values(["checkpoint_step_numeric", "run_name"], na_position="first")
combined.to_csv(out, index=False)
print(f"Saved checkpoint perplexity curve to: {out}")
PY
}


# =============================================================================
# Execution
# =============================================================================

mkdir -p "${OUTPUT_ROOT}"

if [[ "${MODEL_TYPE}" == "zero_shot" ]]; then
  RUN="${RUN_NAME:-zeroshot}"
  OUT_DIR="${OUTPUT_ROOT}/results_${RUN}"
  TOK="$(resolve_tokenizer "zero_shot" "")"
  run_one "zero_shot" "${RUN}" "" "${TOK}" "${OUT_DIR}"
  exit 0
fi

if [[ -z "${ADAPTER_ROOT}" ]]; then
  echo "ERROR: ADAPTER_ROOT is required for MODEL_TYPE=${MODEL_TYPE}" >&2
  exit 1
fi

if bool_is_true "${LOOP_CHECKPOINTS}"; then
  mapfile -t CHECKPOINTS < <(
    find "${ADAPTER_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "${CHECKPOINT_GLOB}" | sort -V
  )

  if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
    echo "ERROR: no checkpoint directories matching ${ADAPTER_ROOT}/${CHECKPOINT_GLOB}" >&2
    exit 1
  fi

  echo "Found ${#CHECKPOINTS[@]} checkpoints under ${ADAPTER_ROOT}"

  for CKPT in "${CHECKPOINTS[@]}"; do
    STEP="$(step_from_checkpoint_path "${CKPT}")"
    BASE_RUN="${RUN_NAME:-${MODEL_TYPE}}"
    RUN="${BASE_RUN}_step_${STEP}"
    OUT_DIR="${OUTPUT_ROOT}/results_${RUN}"
    TOK="$(resolve_tokenizer "${MODEL_TYPE}" "${CKPT}")"

    echo
    echo "=============================================================================="
    echo "Evaluating ${MODEL_TYPE} checkpoint: ${CKPT}"
    echo "Step: ${STEP}"
    echo "Tokenizer: ${TOK}"
    echo "Output: ${OUT_DIR}"
    echo "=============================================================================="

    run_one "${MODEL_TYPE}" "${RUN}" "${CKPT}" "${TOK}" "${OUT_DIR}"
  done

  CURVE_NAME="${RUN_NAME:-${MODEL_TYPE}}_wikitext2_perplexity_by_step.csv"
  combine_checkpoint_summaries "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/${CURVE_NAME}"
else
  RUN="${RUN_NAME:-${MODEL_TYPE}}"
  OUT_DIR="${OUTPUT_ROOT}/results_${RUN}"
  TOK="$(resolve_tokenizer "${MODEL_TYPE}" "${ADAPTER_ROOT}")"
  run_one "${MODEL_TYPE}" "${RUN}" "${ADAPTER_ROOT}" "${TOK}" "${OUT_DIR}"
fi
