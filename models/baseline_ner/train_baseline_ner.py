# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#!/usr/bin/env python3
"""
train_baseline_ner_corrected.py

Baseline QLoRA SFT training script for genomic variant extraction.

Main design choice:
  - Use each model's native chat_template.
  - Convert the original conversational JSONL into prompt/completion format.
  - Use TRL SFTTrainer with completion_only_loss=True.

This avoids hardcoding Llama chat tokens and is therefore suitable for comparing
multiple instruction-model families such as Llama, Qwen, Gemma, Phi, and Mistral,
provided the tokenizer has a valid native chat_template.

Expected input JSONL format:
{
  "messages": [
    {"role": "system", "content": "..."},        # optional; overwritten here
    {"role": "user", "content": "Title: ...\nAbstract: ..."},
    {"role": "assistant", "content": "Variant: V600E, Gene: BRAF"}
  ]
}
"""

import argparse
import inspect
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

try:
    from trl import SFTConfig
except Exception:  # pragma: no cover
    SFTConfig = None


SYSTEM_PROMPT = (
    "Extract only specific genetic variants from biomedical title and abstract text. "
    "Return only lines in the required format. "
    "Allowed outputs are:\n"
    "Variant: <mutation>, Gene: <gene>\n"
    "or exactly:\n"
    "No variant\n"
    "Do not include explanations, markdown, JSON, or extra text."
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base instruction model.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Path to train.jsonl.",
    )
    parser.add_argument(
        "--valid_file",
        type=str,
        required=True,
        help="Path to valid.jsonl.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save LoRA adapter and tokenizer.",
    )

    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)

    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bf16. Recommended for H100. Usually not for RTX 3090.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16. Recommended for RTX 3090.",
    )
    parser.add_argument(
        "--debug_print_batch",
        action="store_true",
        help="Print one tokenized training batch and show which tokens receive loss.",
    )

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer(model_name: str):
    tokenizer_kwargs = {
        "use_fast": True,
        "trust_remote_code": True,
    }

    # Recent Mistral/Ministral tokenizers may require this flag. It is harmless to
    # keep this model-specific instead of passing it to every tokenizer.
    name = model_name.lower()
    if "mistral" in name or "ministral" in name:
        tokenizer_kwargs["fix_mistral_regex"] = True

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

    if tokenizer.chat_template is None:
        raise ValueError(
            f"Tokenizer for {model_name} has no chat_template. Use an instruction/chat "
            "checkpoint or implement an explicit prompt/completion formatting function."
        )

    return tokenizer


def normalize_example(example):
    """
    Convert original conversational data into TRL prompt/completion format.

    Output format:
    {
      "prompt": [system message, user message],
      "completion": [assistant message]
    }

    With SFTConfig(completion_only_loss=True), TRL computes loss only on the
    completion, while still formatting the prompt with the model's native chat_template.
    """

    messages = example["messages"]
    messages = [m for m in messages if m["role"] != "system"]

    if len(messages) != 2:
        raise ValueError(f"Expected user + assistant messages, got: {messages}")

    if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        raise ValueError(f"Expected user then assistant, got: {messages}")

    user_text = messages[0]["content"].strip()
    assistant_text = messages[1]["content"].strip()

    if not user_text:
        raise ValueError(f"Empty user text: {messages}")

    if not assistant_text:
        raise ValueError(f"Empty assistant label: {messages}")

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "completion": [
            {"role": "assistant", "content": assistant_text},
        ],
    }


def make_training_args(args):
    """
    Use SFTConfig because completion_only_loss and max_length are SFT-specific.
    Do not silently fall back to TrainingArguments, because that would likely train
    on a different objective.
    """

    if SFTConfig is None:
        raise RuntimeError(
            "Your installed TRL version does not expose SFTConfig. "
            "Upgrade TRL or implement manual answer-only masking."
        )

    base_kwargs = dict(
        output_dir=args.output_dir,
        seed=args.seed,
        data_seed=args.seed,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        report_to="none",
        remove_unused_columns=False,
    )

    # transformers changed evaluation_strategy -> eval_strategy in recent versions.
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        base_kwargs["eval_strategy"] = "steps"
    else:
        base_kwargs["evaluation_strategy"] = "steps"
    base_kwargs["eval_steps"] = args.eval_steps

    sft_sig = inspect.signature(SFTConfig.__init__)
    sft_kwargs = dict(base_kwargs)

    if "max_length" in sft_sig.parameters:
        sft_kwargs["max_length"] = args.max_seq_length
    if "packing" in sft_sig.parameters:
        sft_kwargs["packing"] = False
    if "completion_only_loss" in sft_sig.parameters:
        sft_kwargs["completion_only_loss"] = True
    if "assistant_only_loss" in sft_sig.parameters:
        sft_kwargs["assistant_only_loss"] = False

    return SFTConfig(**sft_kwargs)


def build_lora_config(args) -> LoraConfig:
    modules_to_save = ["lm_head"]

    model_name = args.model_name.lower()

    if "gemma" in model_name:
        target_modules = [
            "q_proj.linear",
            "k_proj.linear",
            "v_proj.linear",
            "o_proj.linear",
            "gate_proj.linear",
            "up_proj.linear",
            "down_proj.linear",
        ]
    else:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )


def print_trainable_parameters(model):
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    pct = 100 * trainable / total if total else 0
    print(f"Trainable parameters: {trainable:,} / {total:,} ({pct:.4f}%)")


def print_chat_template_preview(tokenizer):
    print("\nActive tokenizer chat_template preview:")
    template = tokenizer.chat_template
    print(template[:1500] if template else "None")
    print()


def debug_print_one_batch(trainer, tokenizer, max_tokens: int = 256):
    print("\nDebugging one tokenized training batch. Tokens marked LOSS contribute to loss.")
    batch = next(iter(trainer.get_train_dataloader()))
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())

    for i, (tok, lab) in enumerate(zip(tokens, labels.tolist())):
        if i >= max_tokens:
            print(f"... truncated after {max_tokens} tokens")
            break
        flag = "LOSS" if lab != -100 else "----"
        print(f"{i:04d}\t{flag}\t{tok}\t{lab}")
    print()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = Path(args.output_dir)

    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --bf16 or --fp16.")

    if not args.bf16 and not args.fp16:
        print("Warning: neither --bf16 nor --fp16 selected. Defaulting to fp16.")
        args.fp16 = True

    tokenizer = load_tokenizer(args.model_name)
    print("Using the model's native chat_template with prompt/completion SFT.")
    print_chat_template_preview(tokenizer)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    vocab_size = len(tokenizer)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False

    peft_config = build_lora_config(args)

    raw_datasets = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            "validation": args.valid_file,
        },
    )

    train_dataset = raw_datasets["train"].map(
        normalize_example,
        remove_columns=raw_datasets["train"].column_names,
    )
    valid_dataset = raw_datasets["validation"].map(
        normalize_example,
        remove_columns=raw_datasets["validation"].column_names,
    )

    training_args = make_training_args(args)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print_trainable_parameters(trainer.model)

    if args.debug_print_batch:
        debug_print_one_batch(trainer, tokenizer)

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    baseline_metadata = {
        "ner_name": "baseline",
        "description": "baseline model without added tokens",
        "base_model_name": args.model_name,
        "vocab_size": vocab_size,
        "use_trainable_token_indices": False,
        "train_full_embeddings": False,
        "modules_to_save": peft_config.modules_to_save,
        "sft_format": "prompt_completion_native_chat_template_completion_only_loss",
    }
    with open(output_dir / "baseline_metadata.json", "w", encoding="utf-8") as f:
        json.dump(baseline_metadata, f, indent=2, ensure_ascii=False)
    with open(output_dir / "log_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    print(f"Saved LoRA adapter and tokenizer to: {args.output_dir}")
    print(f"Saved baseline metadata to: {output_dir / 'baseline_metadata.json'}")
    print(f"Saved log history to: {output_dir / 'log_history.json'}")


if __name__ == "__main__":
    main()
