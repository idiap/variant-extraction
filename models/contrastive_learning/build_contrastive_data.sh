# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

python build_contrastive_data_text_output.py \
  --train_files ../../data/processed_merged_1c/train.jsonl \
  --valid_files ../../data/processed_merged_1c/val.jsonl \
  --output_dir ../../data/contrastive_merged_1c \
  --negatives_per_example 8 \
  --include_no_variant_examples \
  --max_no_variant_examples 1000 \
  --seed 999


# --clinvar_summary_file ../../resources/variant_summary.txt.gz \