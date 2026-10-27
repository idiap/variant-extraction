# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#accelerate config

#accelerate launch train_baseline_ner.py \
python train_baseline_ner.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --train_file ../../data/processed_merged_1/train.jsonl \
  --valid_file ../../data/processed_merged_1/val.jsonl \
  --output_dir ./outputs/llama31_8b_baseline \
  --bf16 \
  --max_seq_length 4096 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --eval_steps 500 \
  --save_steps 10000

# LLMs
# meta-llama/Llama-3.1-8B-Instruct
# Qwen/Qwen2.5-7B-Instruct
# Qwen/Qwen2.5-14B-Instruct
# mistralai/Mistral-7B-Instruct-v0.3