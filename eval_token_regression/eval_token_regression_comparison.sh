# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

python eval_token_regression_comparison.py \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --baseline_model ./outputs/llama31_8b_baseline \
  --vt_model ./outputs/llama31_8b_vt \
  --num_e2e_examples 500 \
  --num_ppl_examples 1000 \
  --max_new_tokens 80 \
  --output_csv token_regression_results.csv \
  --summary_csv token_regression_summary.csv
