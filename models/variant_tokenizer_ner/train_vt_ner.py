# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#!/usr/bin/env python3
"""
train_vt_ner_corrected.py

Variant Tokenizer QLoRA SFT training script for genomic variant extraction.

Compared with the baseline script, this script:
  1. Adds HGVS / variant-nomenclature-aware tokens to the tokenizer.
  2. Optionally adds gene-symbol tokens, DNA k-mers, or ClinVar/BPE DNA tokens.
  3. Resizes the model embedding matrix.
  4. Makes newly added input-token embeddings trainable through PEFT
     trainable_token_indices when available, or trains full embeddings as fallback.
  5. Saves lm_head through modules_to_save so the model can emit added tokens.
  6. Saves the modified tokenizer alongside the LoRA adapter.

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
from itertools import product
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from datasets import load_dataset
from peft import LoraConfig
from tokenizers import AddedToken
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
try:
    from safetensors.torch import safe_open
except Exception:
    safe_open = None


SYSTEM_PROMPT = (
    "Extract only specific genetic variants from biomedical title and abstract text. "
    "Return only lines in the required format. "
    "Allowed outputs are:\n"
    "Variant: <mutation>, Gene: <gene>\n"
    "or exactly:\n"
    "No variant\n"
    "Do not include explanations, markdown, JSON, or extra text."
)


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


def parse_args():
    parser = argparse.ArgumentParser(description="Train VT NER model with HGVS-aware tokenizer tokens.")

    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base instruction model.",
    )
    parser.add_argument(
        "--warmup_adapter_path",
        type=str,
        default=None,
        help=(
            "Optional path to a ClinVar causal-LM warm-up full model directory. "
            "If provided, the script loads the tokenizer and full model from this directory "
            "and starts NER LoRA SFT from the warmed-up VT model. If omitted, training starts "
            "from the base model as before."
        ),
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
        help="Where to save VT LoRA adapter and modified tokenizer.",
    )
    parser.add_argument(
        "--add_gene_names",
        type=str,
        default=None,
        help="None, 'hgnc', or 'most_freq'.",
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
        default=0,
        help="0: none, positive integer: add all DNA k-mers up to k, -1: add BPE tokens.",
    )
    parser.add_argument(
        "--bpe_tokens_file",
        type=str,
        default="../../data/tokenizers/clinvar_allele_bpe/selected_bpe_dna_tokens.txt",
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
    parser.add_argument(
        "--debug_print_batch",
        action="store_true",
        help="Print one tokenized training batch and show which tokens receive loss.",
    )
    parser.add_argument(
        "--decomposition_init_added_tokens",
        action="store_true",
        help=(
            "Initialize each newly added VT token from the mean of the base tokenizer "
            "decomposition embeddings before LoRA/PEFT wrapping. Only applies when "
            "starting from the base model, not when --warmup_adapter_path is used."
        ),
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


def infer_adapter_vocab_size(adapter_path: str | Path) -> int | None:
    """
    Infer saved vocab size from adapter weights.

    This is useful because resize_token_embeddings(..., pad_to_multiple_of=8)
    can make the model embedding matrix larger than len(tokenizer). If the adapter
    saved lm_head or embed_tokens with the padded size, the base model must be
    resized to that same size before loading the adapter.
    """
    adapter_path = Path(adapter_path)

    candidate_keys = [
        "base_model.model.lm_head.modules_to_save.default.weight",
        "base_model.model.lm_head.weight",
        "base_model.model.model.embed_tokens.modules_to_save.default.weight",
        "base_model.model.model.embed_tokens.weight",
        "lm_head.modules_to_save.default.weight",
        "lm_head.weight",
        "model.embed_tokens.modules_to_save.default.weight",
        "model.embed_tokens.weight",
    ]

    safetensors_path = adapter_path / "adapter_model.safetensors"
    if safetensors_path.exists() and safe_open is not None:
        with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for key in candidate_keys:
                if key in keys:
                    return f.get_tensor(key).shape[0]

            # Fallback: inspect any saved lm_head/embed_tokens matrix.
            for key in keys:
                if ("lm_head" in key or "embed_tokens" in key) and key.endswith(".weight"):
                    tensor = f.get_tensor(key)
                    if len(tensor.shape) == 2:
                        return tensor.shape[0]

    bin_path = adapter_path / "adapter_model.bin"
    if bin_path.exists():
        state = torch.load(bin_path, map_location="cpu")
        for key in candidate_keys:
            if key in state:
                return state[key].shape[0]

        for key, tensor in state.items():
            if ("lm_head" in key or "embed_tokens" in key) and key.endswith(".weight"):
                if len(tensor.shape) == 2:
                    return tensor.shape[0]

    return None


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
    resources_dir.mkdir(parents=True, exist_ok=True)

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
    gene_symbols = df["symbol"].dropna().astype(str).unique().tolist()

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
            if gene:
                top_genes.append(gene)

    tokens_to_add.extend(top_genes)
    print(f"Adding {len(top_genes)} most-frequent gene symbols")
    return tokens_to_add


def add_dna_kmers(tokens_to_add, kmer=None):
    kmer = kmer or 3
    print(f"Adding DNA k-mers up to k = {kmer}")

    all_dna_kmers = []
    for k in range(1, kmer + 1):
        dna_kmers = ["".join(kmer_tuple) for kmer_tuple in product("ACGT", repeat=k)]
        all_dna_kmers.extend(dna_kmers)

    tokens_to_add.extend(all_dna_kmers)
    return tokens_to_add


def add_dna_bpe(tokens_to_add, bpe_tokens_file):
    bpe_tokens_file = Path(bpe_tokens_file)
    bpe_tokens = []
    with bpe_tokens_file.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip().upper()
            if t:
                bpe_tokens.append(t)

    tokens_to_add.extend(bpe_tokens)
    print(f"Adding {len(bpe_tokens)} BPE DNA tokens from {bpe_tokens_file}")
    return tokens_to_add


def add_variant_tokens(tokenizer, args) -> Tuple[int, List[str], List[int]]:
    old_vocab_size = len(tokenizer)

    # Must be a fresh list. Do not mutate the global HGVS_AWARE_TOKENS.
    tokens_to_add = list(HGVS_AWARE_TOKENS)

    if args.add_gene_names and args.add_gene_names != "None":
        gene_mode = args.add_gene_names.lower()
        if gene_mode == "hgnc":
            tokens_to_add = add_hgnc_gene_names(
                tokens_to_add,
                resources_dir=args.resources_dir,
                hgnc_date=args.hgnc_date,
            )
        elif gene_mode == "most_freq":
            tokens_to_add = add_most_frequent_clinvar_genes(
                tokens_to_add,
                resources_dir=args.resources_dir,
            )
        else:
            raise ValueError("--add_gene_names must be one of: None, hgnc, most_freq")

    if args.add_dna_tokens > 0:
        tokens_to_add = add_dna_kmers(tokens_to_add, kmer=args.add_dna_tokens)
    elif args.add_dna_tokens == -1:
        tokens_to_add = add_dna_bpe(tokens_to_add, bpe_tokens_file=args.bpe_tokens_file)

    tokens_to_add = dedupe_keep_order(tokens_to_add)
    added_tokens = build_added_tokens(raw_tokens=tokens_to_add)
    num_added = tokenizer.add_tokens(added_tokens, special_tokens=False)

    # Important: collect IDs for all tokens added in this configuration, not only HGVS tokens.
    actually_added_tokens = []
    actually_added_ids = []
    for tok in tokens_to_add:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tok_id, int) and tok_id >= old_vocab_size:
            actually_added_tokens.append(tok)
            actually_added_ids.append(tok_id)

    return num_added, actually_added_tokens, actually_added_ids


def _clean_decomposition_ids(token_ids: list[int], old_vocab_size: int) -> list[int]:
    """Keep only valid pre-existing token IDs."""
    return [int(tid) for tid in token_ids if isinstance(tid, int) and 0 <= int(tid) < old_vocab_size]


def get_base_token_decomposition_ids(
    base_tokenizer,
    token: str,
    old_vocab_size: int,
) -> list[int]:
    """
    Tokenize an added token string with the original base tokenizer.

    After VT tokens are added, the modified tokenizer will often encode the string
    as the new single token. For decomposition initialization we instead need the
    original model's decomposition, so this function uses a separate base tokenizer
    loaded before VT token addition.
    """
    candidates = [token]

    # Some tokenizers encode standalone strings differently depending on a leading
    # space. The first non-empty valid decomposition is used.
    if not token.startswith(" "):
        candidates.append(" " + token)

    for text in candidates:
        ids = base_tokenizer(text, add_special_tokens=False)["input_ids"]
        ids = _clean_decomposition_ids(ids, old_vocab_size)
        if ids:
            return ids

    return []


def initialize_added_tokens_from_base_decomposition(
    model,
    base_tokenizer,
    added_token_strings: list[str],
    added_token_ids: list[int],
    old_vocab_size: int,
) -> dict:
    """
    Initialize newly added VT token rows from their base-tokenizer decomposition.

    For each added token, we tokenize the token string with the original tokenizer
    and copy the mean of those existing embedding rows into the new row. The same
    is done for the output embedding / lm_head when it is present and has matching
    vocabulary dimension.

    This keeps new tokens closer to the pretrained embedding manifold than the
    default random/average initialization used by resize_token_embeddings().
    """
    if not added_token_strings or not added_token_ids:
        return {
            "enabled": True,
            "initialized_input_rows": 0,
            "initialized_output_rows": 0,
            "skipped": [],
        }

    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()

    input_weight = input_emb.weight
    output_weight = output_emb.weight if output_emb is not None else None

    initialized_input = 0
    initialized_output = 0
    skipped = []

    with torch.no_grad():
        for token, token_id in zip(added_token_strings, added_token_ids):
            token_id = int(token_id)
            decomp_ids = get_base_token_decomposition_ids(
                base_tokenizer=base_tokenizer,
                token=token,
                old_vocab_size=old_vocab_size,
            )

            if not decomp_ids:
                skipped.append({"token": token, "token_id": token_id, "reason": "empty_decomposition"})
                continue

            if token_id >= input_weight.shape[0]:
                skipped.append({"token": token, "token_id": token_id, "reason": "input_id_out_of_range"})
                continue

            decomp_tensor = torch.tensor(decomp_ids, device=input_weight.device, dtype=torch.long)
            mean_input = input_weight.index_select(0, decomp_tensor).mean(dim=0)
            input_weight[token_id].copy_(mean_input.to(dtype=input_weight.dtype))
            initialized_input += 1

            if output_weight is not None and token_id < output_weight.shape[0]:
                out_decomp_tensor = torch.tensor(decomp_ids, device=output_weight.device, dtype=torch.long)
                mean_output = output_weight.index_select(0, out_decomp_tensor).mean(dim=0)
                output_weight[token_id].copy_(mean_output.to(dtype=output_weight.dtype))
                initialized_output += 1

    stats = {
        "enabled": True,
        "initialized_input_rows": initialized_input,
        "initialized_output_rows": initialized_output,
        "skipped": skipped[:50],
        "n_skipped": len(skipped),
    }

    print("Decomposition initialization stats:")
    print(json.dumps(stats, indent=2, ensure_ascii=False)[:4000])
    return stats


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


def build_lora_config(args, added_token_ids: List[int]) -> LoraConfig:
    modules_to_save = ["lm_head"]

    if args.train_full_embeddings:
        modules_to_save.append("embed_tokens")

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

    lora_kwargs = dict(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
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

    if args.warmup_adapter_path is None:
        # Current behavior: start from base tokenizer and add VT tokens.
        tokenizer = load_tokenizer(args.model_name)
        tokenizer_source = args.model_name
        warmup_adapter_path = None

        print("Starting from base model/tokenizer and adding VT tokens.")
        print("Using the model's native chat_template with prompt/completion SFT.")
        print_chat_template_preview(tokenizer)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "right"

        old_vocab_size = len(tokenizer)
        base_tokenizer_for_decomp = None
        if args.decomposition_init_added_tokens:
            # Keep an unmodified copy of the original tokenizer so added tokens can
            # be decomposed into the model's native/pretrained tokens.
            base_tokenizer_for_decomp = load_tokenizer(args.model_name)

        num_added, added_token_strings, added_token_ids = add_variant_tokens(tokenizer, args)
        new_vocab_size = len(tokenizer)

    else:
        # Continue from ClinVar warm-up full model/tokenizer.
        warmup_adapter_path = Path(args.warmup_adapter_path)
        if not warmup_adapter_path.exists():
            raise FileNotFoundError(f"Warm-up adapter path does not exist: {warmup_adapter_path}")

        print(f"Continuing from warm-up full model/tokenizer: {warmup_adapter_path}")

        tokenizer = AutoTokenizer.from_pretrained(
            warmup_adapter_path,
            use_fast=True,
            trust_remote_code=True,
        )
        tokenizer_source = str(warmup_adapter_path)

        if tokenizer.chat_template is None:
            raise ValueError(
                f"Tokenizer loaded from {warmup_adapter_path} has no chat_template."
            )

        print("Using tokenizer saved by the ClinVar warm-up stage.")
        print_chat_template_preview(tokenizer)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "right"

        base_tokenizer_for_decomp = None

        # Tokens were already added during warm-up. Do not add them again.
        old_vocab_size = None
        new_vocab_size = len(tokenizer)
        num_added = 0

        metadata_path = warmup_adapter_path / "vt_clinvar_embedding_only_warmup_metadata.json"
        if metadata_path.exists():
            warmup_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            added_token_strings = warmup_metadata.get("added_token_strings", [])
            added_token_ids = warmup_metadata.get("added_token_ids", [])
        else:
            raise FileNotFoundError(
                f"Warm-up metadata not found: {metadata_path}. "
                "Cannot recover added_token_ids for trainable_token_indices."
            )

    print(f"Tokenizer source:        {tokenizer_source}")
    print(f"Tokenizer size:          {len(tokenizer)}")
    print(f"Number of tokens added in this stage: {num_added}")
    print(f"Tracked new token IDs:   {len(added_token_ids)}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )

    model_source = str(warmup_adapter_path) if args.warmup_adapter_path is not None else args.model_name

    print(f"Loading model from: {model_source}")
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.warmup_adapter_path is None:
        # Fresh VT SFT: resize base model after adding VT tokens.
        print(f"Resizing model embeddings to tokenizer size: {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    else:
        # Warmed VT SFT: model already includes the warmed tokenizer embeddings.
        current_vocab_size = model.get_input_embeddings().weight.shape[0]
        print(f"Loaded warmed full model with embedding size: {current_vocab_size}")
        print(f"Loaded warmed tokenizer size: {len(tokenizer)}")

        if current_vocab_size < len(tokenizer):
            raise ValueError(
                f"Warmed model embedding size {current_vocab_size} is smaller than "
                f"tokenizer size {len(tokenizer)}. The warm-up checkpoint/tokenizer are inconsistent."
            )

    decomposition_init_stats = {"enabled": False}
    if args.decomposition_init_added_tokens:
        if args.warmup_adapter_path is not None:
            print(
                "Warning: --decomposition_init_added_tokens is ignored when "
                "--warmup_adapter_path is provided, because the warmed model already "
                "contains initialized/trained VT embeddings."
            )
        else:
            if base_tokenizer_for_decomp is None or old_vocab_size is None:
                raise RuntimeError("Missing base tokenizer/old vocab size for decomposition initialization.")
            decomposition_init_stats = initialize_added_tokens_from_base_decomposition(
                model=model,
                base_tokenizer=base_tokenizer_for_decomp,
                added_token_strings=added_token_strings,
                added_token_ids=added_token_ids,
                old_vocab_size=old_vocab_size,
            )

    # In both cases, create a new NER LoRA adapter for this SFT stage.
    peft_config = build_lora_config(args, added_token_ids)
    model_is_already_peft = False

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

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
    )

    # Always pass peft_config: even when starting from the warmed full model,
    # this SFT stage creates a new NER LoRA adapter.
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    trainer = SFTTrainer(**trainer_kwargs)

    print_trainable_parameters(trainer.model)

    if args.debug_print_batch:
        debug_print_one_batch(trainer, tokenizer)

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    modules_to_save = None
    if peft_config is not None:
        modules_to_save = peft_config.modules_to_save
    elif hasattr(trainer.model, "peft_config"):
        try:
            active_adapter = trainer.model.active_adapter
            modules_to_save = trainer.model.peft_config[active_adapter].modules_to_save
        except Exception:
            modules_to_save = None

    vt_metadata = {
        "ner_name": "vt",
        "description": "variant tokenizer model with HGVS-aware and optional gene/DNA/BPE added tokens",
        "base_model_name": args.model_name,
        "started_from_warmup": args.warmup_adapter_path is not None,
        "warmup_adapter_path": str(args.warmup_adapter_path) if args.warmup_adapter_path else None,
        "tokenizer_source": tokenizer_source,
        "old_vocab_size": old_vocab_size,
        "new_vocab_size": len(tokenizer),
        "num_added_this_stage": num_added,
        "added_token_strings": added_token_strings,
        "added_token_ids": added_token_ids,
        "use_trainable_token_indices": args.use_trainable_token_indices,
        "train_full_embeddings": args.train_full_embeddings,
        "modules_to_save": modules_to_save,
        "decomposition_init_added_tokens": args.decomposition_init_added_tokens,
        "decomposition_init_stats": decomposition_init_stats,
        "sft_format": "prompt_completion_native_chat_template_completion_only_loss",
    }
    with open(output_dir / "vt_tokenizer_metadata.json", "w", encoding="utf-8") as f:
        json.dump(vt_metadata, f, indent=2, ensure_ascii=False)
    with open(output_dir / "log_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    print(f"Saved VT LoRA adapter and modified tokenizer to: {args.output_dir}")
    print(f"Saved VT metadata to: {output_dir / 'vt_tokenizer_metadata.json'}")
    print(f"Saved log history to: {output_dir / 'log_history.json'}")


if __name__ == "__main__":
    main()
