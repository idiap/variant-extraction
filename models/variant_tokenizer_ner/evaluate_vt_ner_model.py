# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#!/usr/bin/env python3
"""
Evaluate the vt NER model: variant-tokenizer fine-tuned causal LLM.

Expected test.jsonl format:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Title: ...\nAbstract: ..."},
    {"role": "assistant", "content": "Variant: V600E, Gene: BRAF"}
  ]
}

This script is the tokenizer-modified counterpart of the baseline NER evaluator.

Important vt-specific behavior:
- Loads the tokenizer from ADAPTER_PATH, not from the base model.
- The tokenizer has extra HGVS / variant-nomenclature-aware tokens.
- Loads the base model.
- Resizes the base model embeddings to len(vt_tokenizer).
- Then loads the LoRA adapter.

Important inference behavior:
- Does NOT rely on tokenizer.apply_chat_template().
- Builds the Llama-3.x prompt manually.
- Uses batch size 1 by default to avoid padding-related generation artifacts.
- Computes the generated-token slice from each example's real input width.
- Uses <|eot_id|> as an additional EOS token when available.

It computes:
- binary detection metrics, equivalent to Human != "0" vs model != "0"
- variant-level extraction metrics
- gene-variant pair extraction metrics
- output-format validity

It writes:
- metrics.json
- predictions.csv
- confusion_matrix.csv
- per_example_errors.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# =========================
# Evaluation configuration
# =========================

USE_LORA_ADAPTER = True

# Batch size 1 avoids padding entirely. After confirming outputs, you can try 2 or 4.
BATCH_SIZE = 32
MAX_NEW_TOKENS = 128
DO_SAMPLE = False
TEMPERATURE = None
TOP_P = None

LOAD_IN_4BIT = True
USE_BF16 = True
USE_FP16 = False

CASE_INSENSITIVE_MATCHING = True
INCLUDE_UNKNOWN_GENE_IN_PAIR_METRICS = True

PRINT_FIRST_N_PREDICTIONS = 5
PRINT_FIRST_N_PROMPTS = 1



# =========================
# Data structures
# =========================

@dataclass
class Example:
    idx: int
    system: str
    user: str
    target: str


@dataclass
class ParsedOutput:
    is_no_variant: bool
    valid_format: bool
    variants: set[str]
    pairs: set[tuple[str, str]]


# =========================
# JSONL loading
# =========================

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing test file: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return rows


def messages_to_example(row: dict, idx: int) -> Example:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"Row {idx} has no messages list")

    system = ""
    user = None
    assistant = None

    for msg in messages:
        role = msg.get("role")
        content = str(msg.get("content", "")).strip()
        if role == "system":
            system = content
        elif role == "user":
            user = content
        elif role == "assistant":
            assistant = content

    if user is None or assistant is None:
        raise ValueError(f"Row {idx} must contain user and assistant messages")

    return Example(idx=idx, system=system, user=user, target=assistant)


def load_examples(path: Path) -> list[Example]:
    rows = load_jsonl(path)
    return [messages_to_example(row, idx=i) for i, row in enumerate(rows)]


# =========================
# Model loading / generation
# =========================

def compute_dtype() -> torch.dtype:
    if USE_BF16:
        return torch.bfloat16
    if USE_FP16:
        return torch.float16
    return torch.float32


def get_token_id(tokenizer, token: str) -> int | None:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        return None
    if isinstance(token_id, int) and token_id >= 0 and token_id != tokenizer.unk_token_id:
        return token_id
    return None


def build_eos_token_ids(tokenizer) -> int | list[int]:
    """
    Llama-3.x chat fine-tunes often stop at <|eot_id|>, not only at eos_token_id.
    Include both eos_token_id and <|eot_id|> when possible.
    """
    eos_ids: list[int] = []

    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            eos_ids.extend(tokenizer.eos_token_id)
        else:
            eos_ids.append(tokenizer.eos_token_id)

    eot_id = get_token_id(tokenizer, "<|eot_id|>")
    if eot_id is not None:
        eos_ids.append(eot_id)

    eos_ids = list(dict.fromkeys(eos_ids))

    if not eos_ids:
        raise ValueError("Could not determine eos_token_id or <|eot_id|> token id.")

    return eos_ids[0] if len(eos_ids) == 1 else eos_ids


def load_model_and_tokenizer(
    base_model_name_or_path: str,
    adapter_path: Path,
    tokenizer_path: Path,
):
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Missing vt tokenizer directory: {tokenizer_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        trust_remote_code=True,
    )

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.init_kwargs["padding_side"] = "left"
    tokenizer.init_kwargs["truncation_side"] = "left"

    print(f"Loaded vt tokenizer from: {tokenizer_path}")
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    print(f"tokenizer.padding_side: {tokenizer.padding_side}")
    print(f"tokenizer.truncation_side: {tokenizer.truncation_side}")
    print(f"tokenizer.pad_token: {repr(tokenizer.pad_token)}")
    print(f"tokenizer.pad_token_id: {tokenizer.pad_token_id}")
    print(f"tokenizer.eos_token: {repr(tokenizer.eos_token)}")
    print(f"tokenizer.eos_token_id: {tokenizer.eos_token_id}")
    print(f"<|eot_id|> id: {get_token_id(tokenizer, '<|eot_id|>')}")

    metadata_path = tokenizer_path / "vt_tokenizer_metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            print("Loaded vt tokenizer metadata:")
            print(json.dumps(metadata, indent=2)[:2000])
        except Exception as exc:
            print(f"Warning: could not read {metadata_path}: {exc}")

    quantization_config = None
    if LOAD_IN_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype(),
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        device_map="auto",
        torch_dtype=None if LOAD_IN_4BIT else compute_dtype(),
        quantization_config=quantization_config,
        trust_remote_code=True,
    )

    old_vocab_size = model.get_input_embeddings().weight.shape[0]
    new_vocab_size = len(tokenizer)

    if old_vocab_size != new_vocab_size:
        print(f"Resizing model embeddings: {old_vocab_size} -> {new_vocab_size}")
        model.resize_token_embeddings(new_vocab_size, pad_to_multiple_of=8)
    else:
        print("Model embeddings already match tokenizer size.")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    if USE_LORA_ADAPTER:
        if not adapter_path.exists():
            raise FileNotFoundError(f"Missing LoRA adapter directory: {adapter_path}")
        print(f"Loading vt LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    return model, tokenizer


def build_llama3_prompt(example: Example) -> str:
    """
    Manual Llama-3.x chat prompt.

    This avoids tokenizer.apply_chat_template(), because the tokenizer saved during
    training may contain a training-only template with `{% generation %}` markers.
    The final string must end with the assistant header, not with <|eot_id|>.
    """
    system = example.system.strip()
    user = example.user.strip()

    parts: list[str] = ["<|begin_of_text|>"]

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

    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def build_prompt(example: Example, tokenizer) -> str:
    return build_llama3_prompt(example)


def batched(iterable: list[Example], batch_size: int) -> Iterable[list[Example]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


@torch.no_grad()
def generate_predictions(model, tokenizer, examples: list[Example]) -> list[str]:
    predictions: list[str] = []

    eos_token_id = build_eos_token_ids(tokenizer)

    generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": eos_token_id,
        "use_cache": True,
    }
    if TEMPERATURE is not None:
        generation_kwargs["temperature"] = TEMPERATURE
    if TOP_P is not None:
        generation_kwargs["top_p"] = TOP_P

    total_batches = math.ceil(len(examples) / BATCH_SIZE)

    for batch_idx, batch in enumerate(batched(examples, BATCH_SIZE), start=1):
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"

        prompts = [build_prompt(ex, tokenizer) for ex in batch]

        if batch_idx <= PRINT_FIRST_N_PROMPTS:
            print("\n=== DEBUG PROMPT ===")
            print(prompts[0][-3000:])
            print("=== END DEBUG PROMPT ===\n")

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=(len(batch) > 1),
            truncation=True,
            add_special_tokens=False,
        )

        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )

        # outputs = prompt + generated tokens. Since batch size 1 has no padding,
        # and larger batches use left padding, the safe slicing point is tensor width.
        prompt_width = input_ids.shape[1]

        for local_idx, output_ids in enumerate(outputs):
            new_tokens = output_ids[prompt_width:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            text = clean_generated_text(text)
            predictions.append(text)

            global_idx = len(predictions) - 1
            if global_idx < PRINT_FIRST_N_PREDICTIONS:
                print("\n=== DEBUG PREDICTION ===")
                print(f"Example idx: {batch[local_idx].idx}")
                print(f"Target:\n{batch[local_idx].target}")
                print(f"Prediction raw cleaned:\n{text!r}")
                print("Generated token ids:", new_tokens.detach().cpu().tolist()[:50])
                print("=== END DEBUG PREDICTION ===\n")

        print(f"Generated batch {batch_idx} / {total_batches}")

    return predictions


def clean_generated_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    stop_markers = [
        "\nUser:",
        "\nSystem:",
        "\nAssistant:",
        "<|eot_id|>",
        "<|end_of_text|>",
        "<|start_header_id|>",
    ]
    for marker in stop_markers:
        if marker in text:
            text = text.split(marker, 1)[0].strip()

    return text


# =========================
# Parsing outputs
# =========================

def canonical_text(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" '\"`.;")
    if CASE_INSENSITIVE_MATCHING:
        value = value.casefold()
    return value


def canonical_gene(value: str) -> str:
    value = canonical_text(value)
    return value.upper() if not CASE_INSENSITIVE_MATCHING else value


def canonical_variant(value: str) -> str:
    value = canonical_text(value)
    value = re.sub(r"\s+", "", value)
    return value


def is_no_variant_text(text: str) -> bool:
    t = canonical_text(text)
    return t in {"0", "no variant", "none", "nan", ""}


def parse_output(text: str) -> ParsedOutput:
    raw = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()

    if is_no_variant_text(raw):
        return ParsedOutput(
            is_no_variant=True,
            valid_format=True,
            variants=set(),
            pairs=set(),
        )

    variants: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    valid_lines = 0
    invalid_lines = 0

    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue

        match = re.match(
            r"^Variant:\s*(?P<variant>.+?)\s*,\s*Gene:\s*(?P<gene>.+?)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if not match:
            invalid_lines += 1
            continue

        variant = canonical_variant(match.group("variant"))
        gene = canonical_gene(match.group("gene"))

        if not variant:
            invalid_lines += 1
            continue

        variants.add(variant)

        if gene and (INCLUDE_UNKNOWN_GENE_IN_PAIR_METRICS or gene != canonical_gene("Unknown")):
            pairs.add((variant, gene))

        valid_lines += 1

    valid_format = valid_lines > 0 and invalid_lines == 0

    return ParsedOutput(
        is_no_variant=False if variants else True,
        valid_format=valid_format,
        variants=variants,
        pairs=pairs,
    )


# =========================
# Metrics
# =========================

def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def binary_confusion(y_true: list[int], y_pred: list[int]) -> dict[str, int | float]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred length mismatch")

    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        elif t == 0 and p == 0:
            tn += 1
        else:
            raise ValueError(f"Unexpected labels: true={t}, pred={p}")

    scores = prf(tp, fp, fn)
    accuracy = safe_div(tp + tn, tp + fp + fn + tn)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": scores["precision"],
        "recall": scores["recall"],
        "f1": scores["f1"],
        "accuracy": accuracy,
    }


def set_extraction_metrics(true_sets: list[set], pred_sets: list[set]) -> dict[str, int | float]:
    tp = fp = fn = 0

    for true_set, pred_set in zip(true_sets, pred_sets):
        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)

    scores = prf(tp, fp, fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": scores["precision"],
        "recall": scores["recall"],
        "f1": scores["f1"],
    }


def format_validity(parsed_predictions: list[ParsedOutput]) -> dict[str, int | float]:
    valid = sum(p.valid_format for p in parsed_predictions)
    total = len(parsed_predictions)
    return {
        "valid_outputs": valid,
        "invalid_outputs": total - valid,
        "valid_output_rate": safe_div(valid, total),
    }


# =========================
# Reporting
# =========================

def write_predictions_csv(
    path: Path,
    examples: list[Example],
    predictions: list[str],
    parsed_targets: list[ParsedOutput],
    parsed_predictions: list[ParsedOutput],
) -> None:
    rows = []
    for ex, pred, t, p in zip(examples, predictions, parsed_targets, parsed_predictions):
        rows.append({
            "idx": ex.idx,
            "user": ex.user,
            "target": ex.target,
            "prediction": pred,
            "true_binary": int(not t.is_no_variant),
            "pred_binary": int(not p.is_no_variant),
            "target_variants": " | ".join(sorted(t.variants)),
            "pred_variants": " | ".join(sorted(p.variants)),
            "target_pairs": " | ".join(f"{v}::{g}" for v, g in sorted(t.pairs)),
            "pred_pairs": " | ".join(f"{v}::{g}" for v, g in sorted(p.pairs)),
            "prediction_valid_format": p.valid_format,
        })
    pd.DataFrame(rows).to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def write_confusion_matrix_csv(path: Path, confusion: dict[str, int | float]) -> None:
    matrix = pd.DataFrame(
        [
            {"": "True 0", "Pred 0": confusion["tn"], "Pred 1": confusion["fp"]},
            {"": "True 1", "Pred 0": confusion["fn"], "Pred 1": confusion["tp"]},
        ]
    )
    matrix.to_csv(path, index=False)


def write_errors_csv(
    path: Path,
    examples: list[Example],
    predictions: list[str],
    parsed_targets: list[ParsedOutput],
    parsed_predictions: list[ParsedOutput],
) -> None:
    rows = []
    for ex, pred, t, p in zip(examples, predictions, parsed_targets, parsed_predictions):
        binary_error = int(not t.is_no_variant) != int(not p.is_no_variant)
        variant_error = t.variants != p.variants
        pair_error = t.pairs != p.pairs
        format_error = not p.valid_format

        if not (binary_error or variant_error or pair_error or format_error):
            continue

        rows.append({
            "idx": ex.idx,
            "error_binary": binary_error,
            "error_variant_set": variant_error,
            "error_pair_set": pair_error,
            "error_format": format_error,
            "user": ex.user,
            "target": ex.target,
            "prediction": pred,
            "missing_variants": " | ".join(sorted(t.variants - p.variants)),
            "extra_variants": " | ".join(sorted(p.variants - t.variants)),
            "missing_pairs": " | ".join(f"{v}::{g}" for v, g in sorted(t.pairs - p.pairs)),
            "extra_pairs": " | ".join(f"{v}::{g}" for v, g in sorted(p.pairs - t.pairs)),
        })

    pd.DataFrame(rows).to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


# =========================
# Main
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a VT-tokenizer NER model.")
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model-name-or-path", required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_jsonl = args.test_jsonl
    output_dir = args.output_dir
    base_model_name_or_path = args.base_model_name_or_path
    adapter_path = args.adapter_path
    tokenizer_path = args.tokenizer_path or adapter_path

    metrics_json = output_dir / "metrics.json"
    predictions_csv = output_dir / "predictions.csv"
    confusion_matrix_csv = output_dir / "confusion_matrix.csv"
    per_example_errors_csv = output_dir / "per_example_errors.csv"

    print("Current Working Directory:", os.getcwd())
    print("Test JSONL:", test_jsonl)
    print("Base model:", base_model_name_or_path)
    print("Tokenizer path:", tokenizer_path)
    print("Use LoRA adapter:", USE_LORA_ADAPTER)
    if USE_LORA_ADAPTER:
        print("Adapter path:", adapter_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    examples = load_examples(test_jsonl)
    print(f"Loaded test examples: {len(examples)}")

    model, tokenizer = load_model_and_tokenizer(
        base_model_name_or_path=base_model_name_or_path,
        adapter_path=adapter_path,
        tokenizer_path=tokenizer_path,
    )
    predictions = generate_predictions(model, tokenizer, examples)

    parsed_targets = [parse_output(ex.target) for ex in examples]
    parsed_predictions = [parse_output(pred) for pred in predictions]
    y_true = [int(not p.is_no_variant) for p in parsed_targets]
    y_pred = [int(not p.is_no_variant) for p in parsed_predictions]

    binary = binary_confusion(y_true, y_pred)
    variant_metrics = set_extraction_metrics([p.variants for p in parsed_targets], [p.variants for p in parsed_predictions])
    pair_metrics = set_extraction_metrics([p.pairs for p in parsed_targets], [p.pairs for p in parsed_predictions])
    validity = format_validity(parsed_predictions)

    metrics = {
        "model": {
            "model_type": "vt_ner",
            "base_model_name_or_path": base_model_name_or_path,
            "tokenizer_path": str(tokenizer_path),
            "use_lora_adapter": USE_LORA_ADAPTER,
            "adapter_path": str(adapter_path) if USE_LORA_ADAPTER else None,
            "load_in_4bit": LOAD_IN_4BIT,
            "use_bf16": USE_BF16,
            "use_fp16": USE_FP16,
            "tokenizer_vocab_size": len(tokenizer),
            "batch_size": BATCH_SIZE,
            "manual_llama3_prompt": True,
        },
        "data": {
            "test_jsonl": str(test_jsonl),
            "n_examples": len(examples),
            "n_positive_true": sum(y_true),
            "n_negative_true": len(y_true) - sum(y_true),
            "case_insensitive_matching": CASE_INSENSITIVE_MATCHING,
            "include_unknown_gene_in_pair_metrics": INCLUDE_UNKNOWN_GENE_IN_PAIR_METRICS,
        },
        "binary_detection": binary,
        "variant_extraction_micro": variant_metrics,
        "gene_variant_pair_extraction_micro": pair_metrics,
        "format_validity": validity,
    }

    metrics_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    write_predictions_csv(predictions_csv, examples, predictions, parsed_targets, parsed_predictions)
    write_confusion_matrix_csv(confusion_matrix_csv, binary)
    write_errors_csv(per_example_errors_csv, examples, predictions, parsed_targets, parsed_predictions)

    print("\n=== Binary detection metrics ===")
    print(json.dumps(binary, indent=2))
    print("\n=== Variant extraction micro metrics ===")
    print(json.dumps(variant_metrics, indent=2))
    print("\n=== Gene-variant pair extraction micro metrics ===")
    print(json.dumps(pair_metrics, indent=2))
    print("\n=== Format validity ===")
    print(json.dumps(validity, indent=2))
    print("\nWrote:")
    print(f"- {metrics_json}")
    print(f"- {predictions_csv}")
    print(f"- {confusion_matrix_csv}")
    print(f"- {per_example_errors_csv}")


if __name__ == "__main__":
    main()