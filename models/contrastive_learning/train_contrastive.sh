# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

python train_contrastive_from_sft.py \
  --base_model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --adapter_path ./outputs/llama31_8b_baseline \
  --tokenizer_name_or_path ./outputs/llama31_8b_baseline \
  --train_file ../../data/contrastive_merged_1/contrastive_train.jsonl \
  --valid_file ../../data/contrastive_merged_1/contrastive_val.jsonl \
  --output_dir ./outputs/llama31_8b_baseline_contrastive \
  --load_in_4bit \
  --bf16 \
  --gradient_checkpointing \
  --max_context_length 4096 \
  --max_target_length 256 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 1