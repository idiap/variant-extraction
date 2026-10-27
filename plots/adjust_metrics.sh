# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#!/usr/bin/env bash
set -euo pipefail

EVAL_DIR="../evaluation"
SCRIPT="adjust_metrics.py"

MODEL="qwen3_8b"

STRATEGIES=(
  "baseline"
  "vt"
  "baseline_contrastive"
)

SUBSETS=(
  "tmvar"
  "varchat"
  "variantscape"
  "bronco"
  "hered"
)

for STRATEGY in "${STRATEGIES[@]}"; do
  for SUBSET in "${SUBSETS[@]}"; do
    RUN_DIR="${EVAL_DIR}/${MODEL}_${STRATEGY}_${SUBSET}"

    if [[ ! -d "$RUN_DIR" ]]; then
      echo "[SKIP] Missing directory: $RUN_DIR"
      continue
    fi

    METRICS_JSON="${RUN_DIR}/metrics.json"
    ERRORS_CSV="${RUN_DIR}/per_example_errors.csv"
    OUTPUT_JSON="${RUN_DIR}/metrics_adj.json"
    ACCEPTED_CSV="${RUN_DIR}/accepted_textual_variant_adjustments.csv"

    if [[ ! -f "$METRICS_JSON" ]]; then
      echo "[SKIP] Missing metrics.json: $RUN_DIR"
      continue
    fi

    if [[ ! -f "$ERRORS_CSV" ]]; then
      echo "[SKIP] Missing per_example_errors.csv: $RUN_DIR"
      continue
    fi

    echo "[RUN] ${MODEL}_${STRATEGY}_${SUBSET}"

    python3.10 "$SCRIPT" \
      --metrics_json "$METRICS_JSON" \
      --errors_csv "$ERRORS_CSV" \
      --output_json "$OUTPUT_JSON" \
      --accepted_cases_csv "$ACCEPTED_CSV"

    echo "[OK] Wrote: $OUTPUT_JSON"
  done
done

echo "Done."