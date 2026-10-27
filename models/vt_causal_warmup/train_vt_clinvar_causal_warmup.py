# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#!/usr/bin/env python3
"""
train_vt_clinvar_only_warmup.py

Embedding-only causal-LM warm-up for VariantTokenizer added tokens using ClinVar
variant strings.

Purpose
=======
This script is meant to run BEFORE task-specific VT NER SFT when you want to
adapt only the newly added token embeddings and keep the rest of the base model
frozen.

It does four things:
  1. Loads the base causal/instruction model and tokenizer.
  2. Adds the same VT tokens used by the VT NER training script.
  3. Freezes every model parameter except the newly added token rows in the
     input embedding matrix and, when untied, the output embedding/lm_head
     matrix.
  4. Runs a short causal language-modeling warm-up on ClinVar variant names.

What is trainable?
==================
Only rows whose token IDs correspond to newly added tokenizer tokens are allowed
to update. All other embedding rows and all transformer/language-model weights
are kept frozen by gradient masks.

If the model ties input and output embeddings, updating the added rows of the
input embedding matrix also updates the same tied rows used for output logits.

If the model has an untied output embedding/lm_head matrix, this script also
allows only the added rows of that output matrix to update by default. Disable
that with --no-train_output_embedding_rows if you want the strictest possible
input-embedding-only warm-up.

Input
=====
ClinVar variant_summary.txt or variant_summary.txt.gz, using the Name column by
default.

Output
======
A full model checkpoint, the modified tokenizer, metadata, trainer log history,
and a compact added_token_embeddings.pt file containing only the added token
embedding rows.

Note
====
This is intentionally not a PEFT/LoRA script. It does not create a PEFT adapter.
"""

from __future__ import annotations

import argparse
import gzip
import inspect
import json
import random
import re
from itertools import product
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from datasets import Dataset
from tokenizers import AddedToken
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
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

    # Common compact protein-change suffixes / frameshift endings
    "*", "=", "_", "+", "-",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Causal-LM warm-up for VariantTokenizer added tokens on ClinVar "
            "variant names, training only newly added embedding rows."
        )
    )

    # Model/data/output.
    p.add_argument("--model_name", required=True, help="Base causal/instruction model.")
    p.add_argument("--clinvar_summary_file", required=True, help="Path to ClinVar variant_summary.txt or .gz.")
    p.add_argument("--output_dir", required=True, help="Where to save the warmed model, tokenizer, and metadata.")
    p.add_argument("--clinvar_column", default="Name", help="ClinVar column to use. Default: Name.")
    p.add_argument(
        "--text_template",
        default="{name}",
        help="Template for each warm-up sequence. Available placeholder: {name}.",
    )
    p.add_argument("--max_rows", type=int, default=None, help="Optional cap on number of ClinVar rows read.")
    p.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Randomly subsample this many cleaned/deduplicated ClinVar examples.",
    )
    p.add_argument(
        "--shuffle_texts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle cleaned/deduplicated ClinVar examples before optional subsampling. Default: true.",
    )
    p.add_argument("--valid_fraction", type=float, default=0.01, help="Validation fraction after deduplication.")
    p.add_argument("--min_text_chars", type=int, default=4, help="Drop examples shorter than this.")

    # Same tokenizer extensions as VT NER training.
    p.add_argument("--add_gene_names", default=None, help="None, 'hgnc', or 'most_freq'.")
    p.add_argument("--resources_dir", default="../resources")
    p.add_argument("--hgnc_date", default="2026-05-05")
    p.add_argument(
        "--add_dna_tokens",
        type=int,
        default=0,
        help="0: none, positive integer: add all DNA k-mers up to k, -1: add BPE tokens.",
    )
    p.add_argument(
        "--bpe_tokens_file",
        default="../../data/tokenizers/clinvar_allele_bpe/selected_bpe_dna_tokens.txt",
    )

    # Embedding-only behavior.
    p.add_argument(
        "--train_output_embedding_rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If output embeddings/lm_head are untied from input embeddings, train only the newly "
            "added rows there too. Default: true. Use --no-train_output_embedding_rows for "
            "input-embedding-only training."
        ),
    )
    p.add_argument(
        "--decomposition_init_added_tokens",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Initialize each newly added token embedding from the mean of the base tokenizer "
            "decomposition embeddings before embedding-only warm-up. Default: false."
        ),
    )

    # Training hyperparameters.
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)

    # Precision/loading.
    # Embedding-only training is simplest when the frozen base is not quantized.
    p.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)

    return p.parse_args()


def set_seed(seed: int) -> None:
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


def load_tokenizer(model_name: str):
    tokenizer_kwargs = {"use_fast": True, "trust_remote_code": True}
    name = model_name.lower()
    if "mistral" in name or "ministral" in name:
        tokenizer_kwargs["fix_mistral_regex"] = True

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_added_tokens(raw_tokens: Iterable[str]) -> List[AddedToken]:
    tokens = dedupe_keep_order(raw_tokens)
    return [
        AddedToken(tok, single_word=False, lstrip=False, rstrip=False, normalized=False)
        for tok in tokens
    ]


def add_hgnc_gene_names(tokens_to_add: list[str], resources_dir=None, hgnc_date=None) -> list[str]:
    resources_dir = Path(resources_dir or "../resources")
    hgnc_date = hgnc_date or "2026-05-05"
    resources_dir.mkdir(parents=True, exist_ok=True)
    hgnc_file = resources_dir / "hgnc_complete_set.tsv"

    if not hgnc_file.exists():
        url = (
            "https://storage.googleapis.com/public-download-files/"
            f"hgnc/archive/archive/monthly/tsv/hgnc_complete_set_{hgnc_date}.tsv"
        )
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        hgnc_file.write_bytes(response.content)

    df = pd.read_csv(hgnc_file, sep="\t")
    gene_symbols = df["symbol"].dropna().astype(str).unique().tolist()
    tokens_to_add.extend(gene_symbols)
    print(f"Adding {len(gene_symbols)} HGNC gene symbols")
    return tokens_to_add


def add_most_frequent_clinvar_genes(tokens_to_add: list[str], resources_dir=None) -> list[str]:
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


def add_dna_kmers(tokens_to_add: list[str], kmer=None) -> list[str]:
    kmer = kmer or 3
    print(f"Adding DNA k-mers up to k = {kmer}")
    all_dna_kmers = []
    for k in range(1, kmer + 1):
        all_dna_kmers.extend("".join(t) for t in product("ACGT", repeat=k))
    tokens_to_add.extend(all_dna_kmers)
    return tokens_to_add


def add_dna_bpe(tokens_to_add: list[str], bpe_tokens_file) -> list[str]:
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
    tokens_to_add = list(HGVS_AWARE_TOKENS)

    if args.add_gene_names and args.add_gene_names != "None":
        gene_mode = args.add_gene_names.lower()
        if gene_mode == "hgnc":
            tokens_to_add = add_hgnc_gene_names(tokens_to_add, args.resources_dir, args.hgnc_date)
        elif gene_mode == "most_freq":
            tokens_to_add = add_most_frequent_clinvar_genes(tokens_to_add, args.resources_dir)
        else:
            raise ValueError("--add_gene_names must be one of: None, hgnc, most_freq")

    if args.add_dna_tokens > 0:
        tokens_to_add = add_dna_kmers(tokens_to_add, kmer=args.add_dna_tokens)
    elif args.add_dna_tokens == -1:
        tokens_to_add = add_dna_bpe(tokens_to_add, args.bpe_tokens_file)

    tokens_to_add = dedupe_keep_order(tokens_to_add)
    num_added = tokenizer.add_tokens(build_added_tokens(tokens_to_add), special_tokens=False)

    actually_added_tokens = []
    actually_added_ids = []
    for tok in tokens_to_add:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tok_id, int) and tok_id >= old_vocab_size:
            actually_added_tokens.append(tok)
            actually_added_ids.append(tok_id)

    return num_added, actually_added_tokens, actually_added_ids


def open_maybe_gzip(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def normalize_text(value: str) -> str:
    value = "" if value is None else str(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def read_clinvar_names(
    path: Path,
    column: str = "Name",
    template: str = "{name}",
    max_rows: int | None = None,
    min_text_chars: int = 4,
) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"ClinVar summary file not found: {path}")

    rows = []
    seen = set()
    n_read = 0

    with open_maybe_gzip(path) as f:
        reader = pd.read_csv(f, sep="\t", dtype=str, chunksize=100_000, low_memory=False)
        for chunk in reader:
            if column not in chunk.columns:
                raise ValueError(
                    f"Column {column!r} not found in {path}. Available columns include: "
                    f"{list(chunk.columns)[:30]}"
                )

            # Replace transcript with gene symbol and remove parentheses.
            if column == "Name" and "GeneSymbol" in chunk.columns:
                split_name = chunk[column].fillna("").astype(str).str.split(":", n=1)
                suffix = split_name.str[1].fillna(split_name.str[0])
                chunk[column] = (
                    chunk["GeneSymbol"].fillna("").astype(str) + ":" + suffix
                ).str.replace(r"[()]", "", regex=True)

            for raw_name in chunk[column].fillna("").astype(str).tolist():
                name = normalize_text(raw_name)
                if len(name) < min_text_chars:
                    continue

                text = normalize_text(template.format(name=name))
                if len(text) < min_text_chars:
                    continue

                key = text.casefold()
                if key in seen:
                    continue
                seen.add(key)
                rows.append(text)
                n_read += 1

                if max_rows is not None and n_read >= max_rows:
                    return rows

    return rows


def make_dataset(texts: list[str], valid_fraction: float, seed: int):
    ds = Dataset.from_dict({"text": texts})
    if valid_fraction <= 0 or len(ds) < 2:
        return ds, None
    split = ds.train_test_split(test_size=valid_fraction, seed=seed, shuffle=True)
    return split["train"], split["test"]


def tokenize_dataset(ds: Dataset, tokenizer, max_seq_length: int) -> Dataset:
    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            add_special_tokens=True,
        )

    return ds.map(tokenize_batch, batched=True, remove_columns=ds.column_names)


def initialize_added_tokens_from_base_decomposition(
    model,
    base_tokenizer,
    added_token_strings: list[str],
    added_token_ids: list[int],
    old_vocab_size: int,
) -> dict:
    """
    Initialize each added token row from the mean embedding of how the original
    base tokenizer decomposes that token string.

    This is a direct initialization, not a gradient update. It must run after
    resize_token_embeddings(...), because the new rows must already exist, and
    before freezing/gradient masking/training starts.
    """
    if len(added_token_strings) != len(added_token_ids):
        raise ValueError(
            "added_token_strings and added_token_ids must have the same length: "
            f"{len(added_token_strings)} vs {len(added_token_ids)}"
        )

    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None or not hasattr(input_embeddings, "weight"):
        raise RuntimeError("Could not locate model input embeddings.")

    output_embeddings = model.get_output_embeddings()

    input_weight = input_embeddings.weight
    output_weight = None
    output_tied = False
    if output_embeddings is not None and hasattr(output_embeddings, "weight"):
        output_weight = output_embeddings.weight
        output_tied = output_weight.data_ptr() == input_weight.data_ptr()

    stats = {
        "enabled": True,
        "initialized_input_rows": 0,
        "initialized_output_rows": 0,
        "output_embeddings_tied": output_tied,
        "skipped": [],
    }

    with torch.no_grad():
        for token_string, token_id in zip(added_token_strings, added_token_ids):
            if token_id < old_vocab_size:
                stats["skipped"].append(
                    {"token": token_string, "token_id": token_id, "reason": "not_new_token_id"}
                )
                continue

            decomposition_ids = base_tokenizer(
                token_string,
                add_special_tokens=False,
            )["input_ids"]

            # Defensive filter: decomposition must use only original vocabulary rows.
            decomposition_ids = [int(i) for i in decomposition_ids if int(i) < old_vocab_size]

            if not decomposition_ids:
                stats["skipped"].append(
                    {"token": token_string, "token_id": token_id, "reason": "empty_base_decomposition"}
                )
                continue

            decomp_index = torch.as_tensor(
                decomposition_ids,
                dtype=torch.long,
                device=input_weight.device,
            )
            mean_input = input_weight.index_select(0, decomp_index).mean(dim=0)
            input_weight[token_id].copy_(mean_input.to(dtype=input_weight.dtype))
            stats["initialized_input_rows"] += 1

            if output_weight is not None and not output_tied:
                decomp_index_out = decomp_index.to(output_weight.device)
                mean_output = output_weight.index_select(0, decomp_index_out).mean(dim=0)
                output_weight[token_id].copy_(mean_output.to(dtype=output_weight.dtype))
                stats["initialized_output_rows"] += 1
            elif output_weight is not None and output_tied:
                # Tied output already sees the input row update.
                stats["initialized_output_rows"] += 1

    stats["n_skipped"] = len(stats["skipped"])
    return stats


def compute_dtype(args) -> torch.dtype:
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float32


def make_training_args(args) -> TrainingArguments:
    kwargs = dict(
        output_dir=args.output_dir,
        seed=args.seed,
        data_seed=args.seed,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=0.3,
        report_to="none",
        remove_unused_columns=False,
    )

    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
    kwargs["eval_steps"] = args.eval_steps

    return TrainingArguments(**kwargs)


def _make_row_mask(weight: torch.Tensor, row_ids: list[int]) -> torch.Tensor:
    mask = torch.zeros_like(weight)
    index = torch.as_tensor(row_ids, dtype=torch.long, device=weight.device)
    mask.index_fill_(0, index, 1)
    return mask


def _register_row_gradient_mask(weight: torch.nn.Parameter, row_ids: list[int], label: str) -> None:
    mask = _make_row_mask(weight, row_ids)

    def mask_grad(grad: torch.Tensor) -> torch.Tensor:
        return grad * mask.to(device=grad.device, dtype=grad.dtype)

    weight.register_hook(mask_grad)
    print(f"Gradient mask registered for {label}: {len(row_ids)} trainable rows")


def freeze_all_but_added_token_embedding_rows(
    model,
    added_token_ids: list[int],
    train_output_embedding_rows: bool = True,
) -> dict:
    """
    Freeze the full model, then allow gradients only for newly added token rows
    in the input embeddings and, optionally, untied output embeddings/lm_head.

    Important: the full embedding Parameter is marked requires_grad=True because
    PyTorch cannot set requires_grad per row. Row-level freezing is enforced by
    gradient hooks that zero all non-added rows.
    """
    if not added_token_ids:
        raise ValueError("No added token IDs were provided; refusing to train embeddings.")

    for param in model.parameters():
        param.requires_grad = False

    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None or not hasattr(input_embeddings, "weight"):
        raise RuntimeError("Could not locate model input embeddings.")

    input_weight = input_embeddings.weight
    input_weight.requires_grad = True
    _register_row_gradient_mask(input_weight, added_token_ids, "input embeddings")

    output_policy = "none"
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None and hasattr(output_embeddings, "weight"):
        output_weight = output_embeddings.weight
        if output_weight is input_weight:
            output_policy = "tied_to_input_embeddings"
        elif train_output_embedding_rows:
            output_weight.requires_grad = True
            _register_row_gradient_mask(output_weight, added_token_ids, "untied output embeddings/lm_head")
            output_policy = "masked_added_rows_only"
        else:
            output_weight.requires_grad = False
            output_policy = "frozen_untied_output_embeddings"

    return {
        "train_input_embedding_rows": True,
        "train_output_embedding_rows": train_output_embedding_rows,
        "output_embedding_policy": output_policy,
        "num_added_token_rows_trainable": len(added_token_ids),
    }


def print_trainable_parameters(model) -> None:
    trainable = 0
    total = 0
    trainable_names = []
    for name, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            trainable_names.append(name)
    pct = 100 * trainable / total if total else 0.0
    print(f"Parameters with requires_grad=True: {trainable:,} / {total:,} ({pct:.4f}%)")
    print("Trainable Parameter objects:")
    for name in trainable_names:
        print(f"  - {name}")
    print("Note: gradient masks restrict these Parameter objects to newly added token rows only.")


def save_added_token_embedding_rows(
    model,
    output_dir: Path,
    added_token_ids: list[int],
    added_token_strings: list[str],
    old_vocab_size: int,
    new_vocab_size: int,
    base_model_name: str,
) -> None:
    input_weight = model.get_input_embeddings().weight.detach().cpu()
    payload = {
        "base_model_name": base_model_name,
        "old_vocab_size": old_vocab_size,
        "new_vocab_size": new_vocab_size,
        "added_token_ids": added_token_ids,
        "added_token_strings": added_token_strings,
        "input_embeddings": input_weight[added_token_ids].clone(),
    }

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None and hasattr(output_embeddings, "weight"):
        output_weight = output_embeddings.weight.detach().cpu()
        if output_weight.data_ptr() == input_weight.data_ptr():
            payload["output_embeddings"] = "tied_to_input_embeddings"
        else:
            payload["output_embeddings"] = output_weight[added_token_ids].clone()

    torch.save(payload, output_dir / "added_token_embeddings.pt")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --bf16 or --fp16.")
    if not args.bf16 and not args.fp16:
        print("Warning: neither --bf16 nor --fp16 selected. Defaulting to fp16.")
        args.fp16 = True
    if args.load_in_4bit:
        print(
            "Warning: --load_in_4bit is enabled. Embedding-only training is usually "
            "simpler and safer with --no-load_in_4bit."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = load_tokenizer(args.model_name)

    # Keep an unmodified tokenizer for decomposition-based initialization.
    # The main tokenizer is mutated by add_variant_tokens(...), so this separate
    # tokenizer preserves the original base-model tokenization of strings such
    # as "delins", "p.", or "Arg".
    base_tokenizer_for_decomp = None
    if args.decomposition_init_added_tokens:
        print("Loading a second unmodified base tokenizer for decomposition initialization.")
        base_tokenizer_for_decomp = load_tokenizer(args.model_name)

    old_vocab_size = len(tokenizer)
    num_added, added_token_strings, added_token_ids = add_variant_tokens(tokenizer, args)
    new_vocab_size = len(tokenizer)

    print(f"Original tokenizer size: {old_vocab_size}")
    print(f"New tokenizer size:      {new_vocab_size}")
    print(f"Number of tokens added:  {num_added}")
    print(f"Tracked new token IDs:   {len(added_token_ids)}")

    if not added_token_ids:
        raise RuntimeError(
            "No new tokens were added to the tokenizer. There are no additional token embeddings to train."
        )

    print(f"Reading ClinVar examples from: {args.clinvar_summary_file}")
    texts = read_clinvar_names(
        Path(args.clinvar_summary_file),
        column=args.clinvar_column,
        template=args.text_template,
        max_rows=args.max_rows,
        min_text_chars=args.min_text_chars,
    )
    if not texts:
        raise RuntimeError("No ClinVar warm-up texts were extracted.")
    print(f"Warm-up examples after cleaning/deduplication: {len(texts):,}")

    if args.shuffle_texts:
        rng = random.Random(args.seed)
        rng.shuffle(texts)

    if args.sample_size is not None:
        if args.sample_size <= 0:
            raise ValueError("--sample_size must be positive.")
        original_n = len(texts)
        texts = texts[: min(args.sample_size, len(texts))]
        print(f"Subsampled ClinVar warm-up examples: {original_n:,} -> {len(texts):,}")

    print("Example warm-up strings:")
    for t in texts[:5]:
        print(f"  - {t}")

    train_ds, valid_ds = make_dataset(texts, args.valid_fraction, args.seed)
    train_tok = tokenize_dataset(train_ds, tokenizer, args.max_seq_length)
    valid_tok = tokenize_dataset(valid_ds, tokenizer, args.max_seq_length) if valid_ds is not None else None

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype(args),
        )

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=None if args.load_in_4bit else compute_dtype(args),
        quantization_config=quantization_config,
        trust_remote_code=True,
    )

    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    decomposition_init_stats = {"enabled": False}
    if args.decomposition_init_added_tokens:
        if base_tokenizer_for_decomp is None:
            raise RuntimeError("base_tokenizer_for_decomp was not initialized.")
        print("Initializing added-token embeddings from base-tokenizer decompositions.")
        decomposition_init_stats = initialize_added_tokens_from_base_decomposition(
            model=model,
            base_tokenizer=base_tokenizer_for_decomp,
            added_token_strings=added_token_strings,
            added_token_ids=added_token_ids,
            old_vocab_size=old_vocab_size,
        )
        print("Decomposition initialization stats:")
        print(json.dumps(decomposition_init_stats, indent=2, ensure_ascii=False))

    freeze_policy = freeze_all_but_added_token_embedding_rows(
        model,
        added_token_ids=added_token_ids,
        train_output_embedding_rows=args.train_output_embedding_rows,
    )

    training_args = make_training_args(args)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        data_collator=collator,
        processing_class=tokenizer,
    )

    print_trainable_parameters(trainer.model)
    trainer.train()

    # Save a full checkpoint because this is not a PEFT adapter.
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    save_added_token_embedding_rows(
        model=trainer.model,
        output_dir=output_dir,
        added_token_ids=added_token_ids,
        added_token_strings=added_token_strings,
        old_vocab_size=old_vocab_size,
        new_vocab_size=new_vocab_size,
        base_model_name=args.model_name,
    )

    metadata = {
        "stage": "clinvar_causal_lm_embedding_only_warmup",
        "description": "Causal-LM warm-up of only VT added-token embedding rows on ClinVar variant names.",
        "base_model_name": args.model_name,
        "clinvar_summary_file": args.clinvar_summary_file,
        "clinvar_column": args.clinvar_column,
        "text_template": args.text_template,
        "old_vocab_size": old_vocab_size,
        "new_vocab_size": new_vocab_size,
        "num_added": num_added,
        "added_token_strings": added_token_strings,
        "added_token_ids": added_token_ids,
        "decomposition_init_added_tokens": args.decomposition_init_added_tokens,
        "decomposition_init_stats": decomposition_init_stats,
        "freeze_policy": freeze_policy,
        "num_warmup_examples": len(texts),
        "max_seq_length": args.max_seq_length,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "sample_size": args.sample_size,
        "shuffle_texts": args.shuffle_texts,
        "load_in_4bit": args.load_in_4bit,
        "bf16": args.bf16,
        "fp16": args.fp16,
    }

    (output_dir / "vt_clinvar_embedding_only_warmup_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2),
        encoding="utf-8",
    )

    print(f"Saved full embedding-only warm-up model and tokenizer to: {args.output_dir}")
    print(f"Saved compact embedding rows to: {output_dir / 'added_token_embeddings.pt'}")
    print(f"Saved metadata to: {output_dir / 'vt_clinvar_embedding_only_warmup_metadata.json'}")


if __name__ == "__main__":
    main()
