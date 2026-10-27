# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#accelerate config

#accelerate launch train_vt_ner.py \
python train_vt_ner.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --train_file ../../data/processed_merged_1/train.jsonl \
  --valid_file ../../data/processed_merged_1/val.jsonl \
  --output_dir ./outputs/llama31_8b_vt \
  --warmup_adapter_path ./outputs/llama31_8b_vt_warmup \
  --add_gene_names None \
  --resources_dir ../../resources \
  --hgnc_date 2026-05-05 \
  --add_dna_tokens 0 \
  --bf16 \
  --use_trainable_token_indices \
  --max_seq_length 4096 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --eval_steps 500 \
  --save_steps 10000

# --decomposition_init_added_tokens