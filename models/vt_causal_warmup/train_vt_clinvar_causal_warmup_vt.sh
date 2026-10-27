# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#!/usr/bin/env bash
set -euo pipefail

python train_vt_clinvar_causal_warmup.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --clinvar_summary_file ../../resources/variant_summary.txt.gz \
  --output_dir ./outputs_final/llama31_8b_vt_warmup \
  --decomposition_init_added_tokens \
  --sample_size 1000000 \
  --clinvar_column Name \
  --max_seq_length 256 \
  --resources_dir ../../resources \
  --add_gene_names None \
  --add_dna_tokens 0 \
  --per_device_train_batch_size 64 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --num_train_epochs 2 \
  --bf16 \
  --no-load_in_4bit \
  --eval_steps 5000 \
  --save_steps 31000