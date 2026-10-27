# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#!/usr/bin/env python3
"""
train_vt_ner.py

Variant Tokenizer (vt) QLoRA SFT training script for genomic variant extraction.

Compared with the standard-tokenizer NER script, this script:
  1. Adds HGVS / variant-nomenclature-aware tokens to the tokenizer.
  2. Resizes the model embedding matrix.
  3. Makes the new token input embeddings trainable through PEFT trainable_token_indices
     when available.
  4. Trains/saves lm_head through modules_to_save so the model can emit the new tokens.
  5. Saves the modified tokenizer alongside the LoRA adapter.

Expected data format: JSONL conversational SFT, one example per line:
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
from typing import Iterable, List, Tuple
import requests
import pandas as pd

import numpy as np
import torch
from datasets import load_dataset
from itertools import product
from peft import LoraConfig
from tokenizers import AddedToken
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
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

# =========================
# Debug generation settings
# =========================
# These diagnostics help interpret mean token accuracy during training.
# They periodically generate from validation examples and print target vs prediction.
DEBUG_GENERATION_EVERY_N_STEPS = 100
DEBUG_GENERATION_NUM_NEGATIVE_EXAMPLES = 3
DEBUG_GENERATION_NUM_POSITIVE_EXAMPLES = 5
DEBUG_GENERATION_MAX_NEW_TOKENS = 128

# Print dataset diagnostics before training.
PRINT_LABEL_DISTRIBUTION = True
PRINT_TOKENIZATION_STATS = True
TOKENIZATION_STATS_N_EXAMPLES = 100


# HGVS / variant-nomenclature-aware token inventory.
# Keep this deliberately compact. Do not enumerate full variants like c.2138C>G.
HGVS_AWARE_TOKENS = [
    # HGVS coordinate prefixes
    "c.", "p.", "g.", "m.", "n.", "r.",

    # HGVS operations / structural atoms
    "delins", "del", "dup", "ins", "inv", "con", "fs", "ext", "Ter",

    # rsID lexical prefix
    "rs",

    # Chromosome lexical prefix often seen in genomic coordinates
    "chr",

    # Three-letter amino-acid symbols
    "Ala", "Arg", "Asn", "Asp", "Cys", "Gln", "Glu", "Gly", "His", "Ile",
    "Leu", "Lys", "Met", "Phe", "Pro", "Ser", "Thr", "Trp", "Tyr", "Val",

    # A few common compact protein-change suffixes / frameshift endings
    "*", "=", "_", "+", "-",
]


def set_training_compatible_chat_template(tokenizer):
    """
    Chat template compatible with TRL assistant-token training.

    The important part is:
      {% generation %} ... {% endgeneration %}

    TRL uses these markers to know which tokens belong to the assistant answer.
    """

    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|start_header_id|>system<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '<|start_header_id|>user<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
        "{% generation %}"
        "{{ message['content'] | trim + '<|eot_id|>' }}"
        "{% endgeneration %}"
        "{{ '<|eot_id|>' }}"
        "{% endif %}"
        "{% endfor %}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train vt NER model with HGVS-aware tokenizer tokens.")

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
        help="Where to save vt LoRA adapter and modified tokenizer.",
    )

    parser.add_argument(
        "--add_gene_names",
        type=str,
        default=None
        # None --> no gene names added as tokens
        # hgnc --> add all hgnc gene symbols
        # most_freq --> add only most
    )
    parser.add_argument(
        "--resources_dir",
        type=str,
        default="../resources",
    )
    parser.add_argument(
        "--hgnc_date",
        type=str,
        default="2026-05-05",
    )
    parser.add_argument(
        "--add_dna_tokens",
        type=int,
        default=0
        # 0 --> no DNA tokens,
        # >= 1 --> = kmer
        # -1 --> BPE
    )
    parser.add_argument(
        "--bpe_tokens_file",
        type=str,
        default="../../data/tokenizers/clinvar_allele_bpe/selected_bpe_dna_tokens.txt"
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
        "--use_trainable_token_indices",
        action="store_true",
        help=(
            "Use PEFT trainable_token_indices for the newly added input embeddings. "
            "Recommended if your PEFT version supports it."
        ),
    )
    parser.add_argument(
        "--train_full_embeddings",
        action="store_true",
        help=(
            "Fallback mode: train/save full embed_tokens instead of only new token rows. "
            "Uses more memory and disk."
        ),
    )

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def build_added_tokens(raw_tokens: Iterable[str]) -> List[AddedToken]:
    """
    Add tokens as ordinary tokens, not special tokens.

    single_word=False is intentional: HGVS atoms frequently appear inside compact strings,
    e.g. c.2138C>G, p.Arg713Trp, rs121913529.
    """
    tokens = dedupe_keep_order(raw_tokens)
    return [
        AddedToken(tok, single_word=False, lstrip=False, rstrip=False, normalized=False)
        for tok in tokens
    ]


def add_hgnc_gene_names(tokens_to_add, resources_dir=None, hgnc_date=None):
    resources_dir = Path(resources_dir or "../resources")
    hgnc_date = hgnc_date or "2026-05-05"

    hgnc_file = resources_dir / "hgnc_complete_set.tsv"

    if not hgnc_file.exists():
        url = (
            "https://storage.googleapis.com/public-download-files/"
            f"hgnc/archive/archive/monthly/tsv/hgnc_complete_set_{hgnc_date}.tsv"
        )

        response = requests.get(url)
        response.raise_for_status()

        hgnc_file.write_bytes(response.content)

    df = pd.read_csv(hgnc_file, sep="\t")

    gene_symbols = (
        df["symbol"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    tokens_to_add.extend(gene_symbols)

    print(f"Adding {len(gene_symbols)} HGNC gene symbols")

    return tokens_to_add


def add_most_frequent_clinvar_genes(tokens_to_add, resources_dir=None):
    resources_dir = Path(resources_dir or "../resources")
    top_genes_path = resources_dir / "top_genes.txt"
    
    top_genes = []
    with top_genes_path.open("r", encoding="utf-8") as f:
        for line in f:
            gene = line.strip().upper()
            if not gene:
                continue
            top_genes.append(gene)

    tokens_to_add.extend(top_genes)
    return tokens_to_add


def add_dna_kmers(tokens_to_add, kmer=None):
    kmer = kmer or 3

    print(f"Adding DNA kmers with kmer = {kmer}")

    all_dna_kmer = []
    # extend with all kmers from 1 to kmer
    for k in range(kmer):
        dna_kmers = [
            "".join(kmer_tuple)
            for kmer_tuple in product("ACGT", repeat=(k+1))
        ]
        all_dna_kmer.extend(dna_kmers)

    tokens_to_add.extend(all_dna_kmer)

    return tokens_to_add


def add_dna_bpe(tokens_to_add, bpe_tokens_file):
    bpe_tokens_file = Path(bpe_tokens_file)
    bpe_tokens = []
    with bpe_tokens_file.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip().upper()
            if not t:
                continue
            bpe_tokens.append(t)

    tokens_to_add.extend(bpe_tokens)
    return tokens_to_add


def add_variant_tokens(tokenizer, args) -> Tuple[int, List[str], List[int]]:
    old_vocab_size = len(tokenizer)
    
    tokens_to_add = HGVS_AWARE_TOKENS
    if args.add_gene_names:
        if args.add_gene_names.lower() == "hgnc":
            tokens_to_add = add_hgnc_gene_names(
                tokens_to_add,
                resources_dir=args.resources_dir,
                hgnc_date=args.hgnc_date,
            )
        elif args.add_gene_names.lower() == "most_freq":
            tokens_to_add = add_most_frequent_clinvar_genes(tokens_to_add,
                                                            resources_dir=args.resources_dir)

    if args.add_dna_tokens > 0:
        kmer = args.add_dna_tokens
        tokens_to_add = add_dna_kmers(
            tokens_to_add,
            kmer=kmer,
        )
    elif args.add_dna_tokens == -1: # BPE
        tokens_to_add = add_dna_bpe(tokens_to_add, bpe_tokens_file=args.bpe_tokens_file)

    added_tokens = build_added_tokens(raw_tokens=tokens_to_add)
    num_added = tokenizer.add_tokens(added_tokens, special_tokens=False)

    actually_added_tokens = []
    actually_added_ids = []
    for tok in HGVS_AWARE_TOKENS:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tok_id, int) and tok_id >= old_vocab_size:
            actually_added_tokens.append(tok)
            actually_added_ids.append(tok_id)

    return num_added, actually_added_tokens, actually_added_ids


def normalize_example(example):
    """
    Ensures each example has a stable system prompt.
    """
    messages = example["messages"]

    # Remove existing system message if present.
    messages = [m for m in messages if m["role"] != "system"]

    if len(messages) != 2:
        raise ValueError(f"Expected user + assistant messages, got: {messages}")

    if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        raise ValueError(f"Expected user then assistant, got: {messages}")

    assistant_text = messages[1]["content"].strip()

    if not assistant_text:
        raise ValueError(f"Empty assistant label: {messages}")

    example["messages"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": messages[0]["content"].strip()},
        {"role": "assistant", "content": assistant_text},
    ]

    return example


def make_training_args(args):
    """
    Prefer TRL SFTConfig when available because max_length and assistant_only_loss
    are SFT-specific. Fall back to transformers.TrainingArguments for older stacks.
    """
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

    if SFTConfig is not None:
        sft_sig = inspect.signature(SFTConfig.__init__)
        sft_kwargs = dict(base_kwargs)
        if "max_length" in sft_sig.parameters:
            sft_kwargs["max_length"] = args.max_seq_length
        if "packing" in sft_sig.parameters:
            sft_kwargs["packing"] = False
        if "assistant_only_loss" in sft_sig.parameters:
            # Best for conversational datasets when the chat template supports assistant masks.
            sft_kwargs["assistant_only_loss"] = True
        return SFTConfig(**sft_kwargs)

    return TrainingArguments(**base_kwargs)


def build_lora_config(args, added_token_ids: List[int]) -> LoraConfig:
    modules_to_save = ["lm_head"]

    if args.train_full_embeddings:
        modules_to_save.append("embed_tokens")

    lora_kwargs = dict(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        modules_to_save=modules_to_save,
    )

    lora_sig = inspect.signature(LoraConfig.__init__)

    if args.use_trainable_token_indices:
        if "trainable_token_indices" not in lora_sig.parameters:
            raise RuntimeError(
                "Your installed PEFT version does not support trainable_token_indices. "
                "Either upgrade peft or rerun with --train_full_embeddings instead."
            )
        if not added_token_ids:
            print("Warning: no new tokens were added, so trainable_token_indices will be empty.")
        lora_kwargs["trainable_token_indices"] = {"embed_tokens": added_token_ids}

    elif not args.train_full_embeddings:
        raise RuntimeError(
            "Choose how to train new token embeddings: either pass "
            "--use_trainable_token_indices or --train_full_embeddings. "
            "Recommended: --use_trainable_token_indices with a recent PEFT version."
        )

    return LoraConfig(**lora_kwargs)


def print_trainable_parameters(model):
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    pct = 100 * trainable / total if total else 0
    print(f"Trainable parameters: {trainable:,} / {total:,} ({pct:.4f}%)")



def build_llama3_inference_prompt(messages: list[dict]) -> str:
    """
    Build an inference prompt from a training example.

    Uses only system + user messages.
    Does not include the assistant target.

    This intentionally avoids tokenizer.apply_chat_template() during free generation,
    because the training template contains `{% generation %}` markers for loss masking.
    """

    system = ""
    user = ""

    for msg in messages:
        if msg["role"] == "system":
            system = str(msg["content"]).strip()
        elif msg["role"] == "user":
            user = str(msg["content"]).strip()

    parts = ["<|begin_of_text|>"]

    if system:
        parts.append(
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}"
            "<|eot_id|>"
        )

    parts.append(
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}"
        "<|eot_id|>"
    )

    parts.append(
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    return "".join(parts)


def get_eos_token_ids(tokenizer):
    """
    Return EOS IDs suitable for Llama-3.x chat generation.

    Llama-3.x chat turns often terminate with <|eot_id|>, so include it when present.
    """
    eos_ids = []

    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            eos_ids.extend(tokenizer.eos_token_id)
        else:
            eos_ids.append(tokenizer.eos_token_id)

    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id >= 0 and eot_id != tokenizer.unk_token_id:
        eos_ids.append(eot_id)

    eos_ids = list(dict.fromkeys(eos_ids))

    if not eos_ids:
        return None

    return eos_ids if len(eos_ids) > 1 else eos_ids[0]


def cast_lm_head_for_generation(model, target_dtype: torch.dtype):
    """
    PEFT modules_to_save can keep lm_head in a different dtype from the
    hidden states produced during generation. This fixes debug generation
    dtype mismatches such as Half != BFloat16.
    """

    base = model

    # PeftModel usually exposes the wrapped model as base_model.model.
    candidates = []

    if hasattr(base, "get_output_embeddings"):
        candidates.append(base.get_output_embeddings())

    if hasattr(base, "base_model"):
        candidates.append(base.base_model)

        if hasattr(base.base_model, "model"):
            inner = base.base_model.model
            if hasattr(inner, "get_output_embeddings"):
                candidates.append(inner.get_output_embeddings())
            if hasattr(inner, "lm_head"):
                candidates.append(inner.lm_head)

    if hasattr(base, "lm_head"):
        candidates.append(base.lm_head)

    seen = set()
    for module in candidates:
        if module is None:
            continue

        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)

        try:
            module.to(dtype=target_dtype)
        except Exception:
            pass


def select_debug_examples_by_label(dataset, n_negative: int = 3, n_positive: int = 3):
    """
    Return a balanced list of validation examples for debug generation:
    first n_negative No-variant examples and first n_positive variant examples.
    """

    negatives = []
    positives = []

    for ex in dataset:
        assistant = ""
        for msg in ex["messages"]:
            if msg["role"] == "assistant":
                assistant = str(msg["content"]).strip()
                break

        if assistant.casefold() == "no variant":
            if len(negatives) < n_negative:
                negatives.append(ex)
        else:
            if len(positives) < n_positive:
                positives.append(ex)

        if len(negatives) >= n_negative and len(positives) >= n_positive:
            break

    debug_examples = negatives + positives

    print(
        f"Selected debug examples: "
        f"{len(negatives)} negative, {len(positives)} positive"
    )

    return debug_examples


class GenerationDebugCallback(TrainerCallback):
    """
    Periodically generates from validation examples during training.

    This is useful because mean token accuracy is teacher-forced and can be misleading.
    The callback shows actual free-generation behavior.
    """

    def __init__(
        self,
        tokenizer,
        eval_dataset,
        every_n_steps: int = 100,
        num_negative_examples: int = 3,
        num_positive_examples: int = 3,
        max_new_tokens: int = 128,
    ):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.every_n_steps = every_n_steps
        self.num_negative_examples = num_negative_examples
        self.num_positive_examples = num_positive_examples
        self.max_new_tokens = max_new_tokens

        self.debug_examples = select_debug_examples_by_label(
            eval_dataset,
            n_negative=num_negative_examples,
            n_positive=num_positive_examples,
        )

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control

        if state.global_step == 0:
            return control

        if state.global_step % self.every_n_steps != 0:
            return control

        print("\n" + "=" * 100)
        print(f"DEBUG GENERATION AT STEP {state.global_step}")
        print("=" * 100)

        was_training = model.training
        model.eval()

        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.do_sample = False
            model.generation_config.temperature = None
            model.generation_config.top_p = None

        device = next(model.parameters()).device
        target_dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32

        # Fix PEFT modules_to_save dtype mismatch during debug generation.
        cast_lm_head_for_generation(model, target_dtype)

        eos_token_id = get_eos_token_ids(self.tokenizer)

        # Avoid decoder-only padding warnings by generating one example at a time.
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        with torch.no_grad():
            n = len(self.debug_examples)

            for i in range(n):
                example = self.debug_examples[i]
                messages = example["messages"]

                target = ""
                for msg in messages:
                    if msg["role"] == "assistant":
                        target = str(msg["content"]).strip()
                        break

                prompt = build_llama3_inference_prompt(messages)

                encoded = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=False,
                    padding=False,
                    truncation=True,
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}

                prompt_width = encoded["input_ids"].shape[1]

                generation_kwargs = dict(
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                if eos_token_id is not None:
                    generation_kwargs["eos_token_id"] = eos_token_id

                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=target_dtype):
                        output_ids = model.generate(
                            **encoded,
                            **generation_kwargs,
                        )
                else:
                    output_ids = model.generate(
                        **encoded,
                        **generation_kwargs,
                    )

                generated_ids = output_ids[0][prompt_width:]
                prediction = self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                ).strip()

                print(f"\n--- Validation example {i} ---")
                print("TARGET:")
                print(target)
                print("\nPREDICTION:")
                print(prediction)
                print("\nGENERATED TOKEN IDS:")
                print(generated_ids.detach().cpu().tolist()[:100])

        if was_training:
            model.train()

        print("=" * 100 + "\n")

        return control


def print_label_distribution(dataset, name: str):
    n_total = len(dataset)
    n_no_variant = 0
    n_variant = 0

    for ex in dataset:
        assistant = ""
        for msg in ex["messages"]:
            if msg["role"] == "assistant":
                assistant = str(msg["content"]).strip()
                break

        if assistant.casefold() == "no variant":
            n_no_variant += 1
        else:
            n_variant += 1

    print(f"\n{name} label distribution")
    print(f"Total:      {n_total}")
    print(f"No variant: {n_no_variant} ({n_no_variant / max(n_total, 1):.2%})")
    print(f"Variant:    {n_variant} ({n_variant / max(n_total, 1):.2%})")


def print_tokenization_stats(dataset, tokenizer, name: str, n: int = 100):
    """
    Prints full-example and assistant-target token lengths.

    This helps explain why the vt tokenizer may obtain higher mean token accuracy:
    added tokens can shorten assistant targets and make many tokens easier to predict.
    """
    n = min(n, len(dataset))
    if n == 0:
        print(f"\n{name} tokenization stats: empty dataset")
        return

    full_lengths = []
    assistant_lengths = []
    no_variant_assistant_lengths = []
    variant_assistant_lengths = []

    for i in range(n):
        messages = dataset[i]["messages"]

        try:
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            full_text = "\n".join(
                f"{msg['role']}: {str(msg['content']).strip()}" for msg in messages
            )

        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        full_lengths.append(len(full_ids))

        assistant = ""
        for msg in messages:
            if msg["role"] == "assistant":
                assistant = str(msg["content"]).strip()
                break

        assistant_ids = tokenizer(assistant, add_special_tokens=False)["input_ids"]
        assistant_len = len(assistant_ids)
        assistant_lengths.append(assistant_len)

        if assistant.casefold() == "no variant":
            no_variant_assistant_lengths.append(assistant_len)
        else:
            variant_assistant_lengths.append(assistant_len)

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    print(f"\n{name} tokenization stats over {n} examples")
    print(f"Full length mean:                 {mean(full_lengths):.2f}")
    print(f"Full length min/max:              {min(full_lengths)} / {max(full_lengths)}")
    print(f"Assistant length mean:            {mean(assistant_lengths):.2f}")
    print(f"Assistant length min/max:         {min(assistant_lengths)} / {max(assistant_lengths)}")
    print(f"No-variant assistant mean length: {mean(no_variant_assistant_lengths):.2f}")
    print(f"Variant assistant mean length:    {mean(variant_assistant_lengths):.2f}")


def print_formatted_training_sample(dataset, tokenizer, name: str = "Training"):
    """
    Prints one formatted sample to verify that the assistant answer is followed by <|eot_id|>.
    """
    if len(dataset) == 0:
        print(f"\n{name} formatted sample: empty dataset")
        return

    messages = dataset[0]["messages"]

    print(f"\n{name} formatted sample")
    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        print(formatted[:4000])
        if "<|eot_id|>" not in formatted:
            print("WARNING: formatted sample does not contain <|eot_id|>.")
    except Exception as exc:
        print(f"Could not apply chat template for formatted sample: {exc}")
        print(messages)


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

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=True,
    )

    if 'Llama-3' in args.model_name:
        set_training_compatible_chat_template(tokenizer)
        print("Using training-compatible Llama-3 chat template with generation markers.")

    # Many causal LMs do not define a pad token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    old_vocab_size = len(tokenizer)
    num_added, added_token_strings, added_token_ids = add_variant_tokens(tokenizer, args)
    new_vocab_size = len(tokenizer)

    print(f"Original tokenizer size: {old_vocab_size}")
    print(f"New tokenizer size:      {new_vocab_size}")
    print(f"Number of tokens added:  {num_added}")
    #print(f"Actually added tokens:   {added_token_strings}")
    #print(f"Actually added token IDs:{added_token_ids}")

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

    # Resize embeddings after adding tokens.
    # pad_to_multiple_of=8 improves Tensor Core friendliness.
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    # Important for gradient checkpointing / training.
    model.config.use_cache = False

    peft_config = build_lora_config(args, added_token_ids)

    raw_datasets = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            "validation": args.valid_file,
        },
    )

    train_dataset = raw_datasets["train"].map(normalize_example)
    valid_dataset = raw_datasets["validation"].map(normalize_example)

    if PRINT_LABEL_DISTRIBUTION:
        print_label_distribution(train_dataset, "Train")
        print_label_distribution(valid_dataset, "Validation")

    if PRINT_TOKENIZATION_STATS:
        print_tokenization_stats(train_dataset, tokenizer, "Train", n=TOKENIZATION_STATS_N_EXAMPLES)
        print_tokenization_stats(valid_dataset, tokenizer, "Validation", n=TOKENIZATION_STATS_N_EXAMPLES)
        print_formatted_training_sample(train_dataset, tokenizer, name="Train")

    training_args = make_training_args(args)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.add_callback(
        GenerationDebugCallback(
            tokenizer=tokenizer,
            eval_dataset=valid_dataset,
            every_n_steps=DEBUG_GENERATION_EVERY_N_STEPS,
            num_negative_examples=DEBUG_GENERATION_NUM_NEGATIVE_EXAMPLES,
            num_positive_examples=DEBUG_GENERATION_NUM_POSITIVE_EXAMPLES,
            max_new_tokens=DEBUG_GENERATION_MAX_NEW_TOKENS,
        )
    )

    print_trainable_parameters(trainer.model)

    trainer.train()

    # Save adapter + PEFT trainable token/lm_head state + tokenizer.
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    vt_metadata = {
        "ner_name": "vt",
        "description": "variant tokenizer model with HGVS-aware added tokens",
        "base_model_name": args.model_name,
        "old_vocab_size": old_vocab_size,
        "new_vocab_size": new_vocab_size,
        "num_added": num_added,
        "added_token_strings": added_token_strings,
        "added_token_ids": added_token_ids,
        "use_trainable_token_indices": args.use_trainable_token_indices,
        "train_full_embeddings": args.train_full_embeddings,
        "modules_to_save": peft_config.modules_to_save,
    }
    with open(output_dir / "vt_tokenizer_metadata.json", "w", encoding="utf-8") as f:
        json.dump(vt_metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved vt LoRA adapter and modified tokenizer to: {args.output_dir}")
    print(f"Saved vt metadata to: {output_dir / 'vt_tokenizer_metadata.json'}")


if __name__ == "__main__":
    main()
