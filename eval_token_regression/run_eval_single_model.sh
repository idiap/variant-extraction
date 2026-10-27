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

# Choose exactly one of: zero_shot, baseline, vt, or any custom label.
MODEL_NAME="vt"

# Base Hugging Face model or local base-model path.
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"

# Adapter path.
# Leave empty for zero-shot/base-model evaluation.
# For baseline or VT, set this to the adapter directory.
ADAPTER="./outputs/llama31_8b_vt"

# Tokenizer path.
# Leave empty to use the conditional defaults below:
#   - zero-shot / no adapter: BASE_MODEL
#   - baseline adapter: BASE_MODEL
#   - vt adapter: ADAPTER
TOKENIZER=""

# Evaluation size.
NUM_E2E_EXAMPLES=500
NUM_PPL_EXAMPLES=1000
MAX_NEW_TOKENS=80

# Output directory and files.
OUTPUT_DIR="results"
OUTPUT_CSV="${OUTPUT_DIR}/${MODEL_NAME}_results.csv"
SUMMARY_CSV="${OUTPUT_DIR}/${MODEL_NAME}_summary.csv"

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
# Conditional logic
# =============================================================================

if [[ -z "${ADAPTER}" ]]; then
  # No adapter means zero-shot/base-model evaluation.
  MODEL_NAME="${MODEL_NAME:-zero_shot}"
  TOKENIZER="${TOKENIZER:-${BASE_MODEL}}"
else
  case "${MODEL_NAME}" in
    baseline)
      # Baseline adapter normally uses the original/base tokenizer.
      TOKENIZER="${TOKENIZER:-${BASE_MODEL}}"
      ;;
    vt)
      # VT adapter normally uses the expanded tokenizer saved with the adapter.
      TOKENIZER="${TOKENIZER:-${ADAPTER}}"
      ;;
    *)
      # For custom adapter labels, default to adapter tokenizer.
      TOKENIZER="${TOKENIZER:-${ADAPTER}}"
      ;;
  esac
fi

mkdir -p "${OUTPUT_DIR}"

cmd=(
  python "${SCRIPT}"
  --model_name "${MODEL_NAME}"
  --base_model "${BASE_MODEL}"
  --tokenizer "${TOKENIZER}"
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
  --output_csv "${OUTPUT_CSV}"
  --summary_csv "${SUMMARY_CSV}"
)

if [[ -n "${ADAPTER}" ]]; then
  cmd+=(--adapter "${ADAPTER}")
fi

if [[ "${LOAD_IN_4BIT}" == true ]]; then
  cmd+=(--load_in_4bit)
fi

if [[ "${MERGE_ADAPTER}" == true ]]; then
  cmd+=(--merge_adapter)
fi

if [[ "${SKIP_GRAMMAR}" == true ]]; then
  cmd+=(--skip_grammar)
fi

if [[ "${SKIP_PPL}" == true ]]; then
  cmd+=(--skip_ppl)
fi

if [[ -n "${PPL_TEXT_FILE}" ]]; then
  cmd+=(--ppl_text_file "${PPL_TEXT_FILE}")
fi

cmd+=("${EXTRA_ARGS[@]}")

printf 'Running command:\n'
printf ' %q' "${cmd[@]}"
printf '\n\n'

"${cmd[@]}"
