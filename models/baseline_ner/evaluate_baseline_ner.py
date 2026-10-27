# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#!/usr/bin/env python3
"""
Evaluate a fine-tuned causal LLM on genetic-variant extraction using test.jsonl.

Expected test.jsonl format:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Title: ...\nAbstract: ..."},
    {"role": "assistant", "content": "Variant: V600E, Gene: BRAF"}
  ]
}

The script loads a base model plus a LoRA adapter by default and computes:
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
# Constants: edit if needed
# =========================

TEST_JSONL = Path("../../data/processed_merged_1/test.jsonl")
OUTPUT_DIR = Path("../../evaluation/llama31_8b_baseline_test")

# Same base model used for training.
BASE_MODEL_NAME_OR_PATH = "meta-llama/Llama-3.1-8B-Instruct"

# Directory produced by the training script, containing adapter_config.json.
# Example: /data/outputs/llama31_8b_variant_ner_qlora
ADAPTER_PATH = Path("./outputs/llama31_8b_baseline")

# If you have merged and saved a full model instead of LoRA adapters, set:
# USE_LORA_ADAPTER = False
# BASE_MODEL_NAME_OR_PATH = "/path/to/merged/model"
USE_LORA_ADAPTER = True

# Evaluation generation settings.
BATCH_SIZE = 32
MAX_NEW_TOKENS = 128
DO_SAMPLE = False
TEMPERATURE = None
TOP_P = None

# Quantized loading is usually best for RTX 3090 and also fine for H100 evaluation.
LOAD_IN_4BIT = True
USE_BF16 = True   # Good for H100. On RTX 3090, set this False and USE_FP16 True.
USE_FP16 = False

# For exact matching, we normalize whitespace and case by default. This avoids
# false mismatches such as "v600e" vs "V600E" or extra spaces.
CASE_INSENSITIVE_MATCHING = True

# Whether examples with "Gene: Unknown" should participate in pair-level metrics.
# Keep True if your training labels contain Unknown and you want to evaluate it.
INCLUDE_UNKNOWN_GENE_IN_PAIR_METRICS = True

METRICS_JSON = OUTPUT_DIR / "metrics.json"
PREDICTIONS_CSV = OUTPUT_DIR / "predictions.csv"
CONFUSION_MATRIX_CSV = OUTPUT_DIR / "confusion_matrix.csv"
PER_EXAMPLE_ERRORS_CSV = OUTPUT_DIR / "per_example_errors.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a base causal LLM with a LoRA adapter on a JSONL test set."
    )
    parser.add_argument(
        "--test-jsonl",
        type=Path,
        default=TEST_JSONL,
        help=f"Path to the test JSONL file (default: {TEST_JSONL})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory for evaluation outputs (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--base-model-name-or-path",
        default=BASE_MODEL_NAME_OR_PATH,
        help=f"Hugging Face model ID or local base-model path (default: {BASE_MODEL_NAME_OR_PATH})",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=ADAPTER_PATH,
        help=f"Path to the LoRA adapter directory (default: {ADAPTER_PATH})",
    )
    return parser.parse_args()


def configure_from_args(args: argparse.Namespace) -> None:
    global TEST_JSONL, OUTPUT_DIR, BASE_MODEL_NAME_OR_PATH, ADAPTER_PATH
    global METRICS_JSON, PREDICTIONS_CSV, CONFUSION_MATRIX_CSV, PER_EXAMPLE_ERRORS_CSV

    TEST_JSONL = args.test_jsonl
    OUTPUT_DIR = args.output_dir
    BASE_MODEL_NAME_OR_PATH = args.base_model_name_or_path
    ADAPTER_PATH = args.adapter_path

    METRICS_JSON = OUTPUT_DIR / "metrics.json"
    PREDICTIONS_CSV = OUTPUT_DIR / "predictions.csv"
    CONFUSION_MATRIX_CSV = OUTPUT_DIR / "confusion_matrix.csv"
    PER_EXAMPLE_ERRORS_CSV = OUTPUT_DIR / "per_example_errors.csv"


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


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME_OR_PATH,
        use_fast=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization_config = None
    if LOAD_IN_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype(),
        )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME_OR_PATH,
        device_map="auto",
        torch_dtype=None if LOAD_IN_4BIT else compute_dtype(),
        quantization_config=quantization_config,
        trust_remote_code=True,
    )

    if USE_LORA_ADAPTER:
        if not ADAPTER_PATH.exists():
            raise FileNotFoundError(f"Missing LoRA adapter directory: {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))

    model.eval()
    return model, tokenizer


def build_prompt(example: Example, tokenizer) -> str:
    messages = []
    if example.system:
        messages.append({"role": "system", "content": example.system})
    messages.append({"role": "user", "content": example.user})

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Fallback for non-chat-tokenizer models.
    system_part = f"System: {example.system}\n" if example.system else ""
    return f"{system_part}User: {example.user}\nAssistant:"


def batched(iterable: list[Example], batch_size: int) -> Iterable[list[Example]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


@torch.no_grad()
def generate_predictions(model, tokenizer, examples: list[Example]) -> list[str]:
    predictions: list[str] = []

    generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if TEMPERATURE is not None:
        generation_kwargs["temperature"] = TEMPERATURE
    if TOP_P is not None:
        generation_kwargs["top_p"] = TOP_P

    for batch_idx, batch in enumerate(batched(examples, BATCH_SIZE), start=1):
        prompts = [build_prompt(ex, tokenizer) for ex in batch]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

        outputs = model.generate(**encoded, **generation_kwargs)

        input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        for output_ids, input_len in zip(outputs, input_lengths):
            # With left padding, slicing at input_len is not always correct because
            # output_ids includes padded prompt length. Use full encoded width instead.
            prompt_width = encoded["input_ids"].shape[1]
            new_tokens = output_ids[prompt_width:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            predictions.append(clean_generated_text(text))

        print(f"Generated batch {batch_idx} / {math.ceil(len(examples) / BATCH_SIZE)}")

    return predictions


def clean_generated_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Remove occasional chat-role continuations if a model emits them.
    stop_markers = [
        "\nUser:",
        "\nSystem:",
        "<|eot_id|>",
        "<|end_of_text|>",
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
    # Remove spaces inside variant mentions: "c. 2138 C > G" -> "c.2138c>g".
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

        # Required format: Variant: <mutation>, Gene: <gene>
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
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


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

def main() -> None:
    args = parse_args()
    configure_from_args(args)

    print("Current Working Directory:", os.getcwd())
    print("Test JSONL:", TEST_JSONL)
    print("Base model:", BASE_MODEL_NAME_OR_PATH)
    print("Use LoRA adapter:", USE_LORA_ADAPTER)
    if USE_LORA_ADAPTER:
        print("Adapter path:", ADAPTER_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    examples = load_examples(TEST_JSONL)
    print(f"Loaded test examples: {len(examples)}")

    model, tokenizer = load_model_and_tokenizer()
    predictions = generate_predictions(model, tokenizer, examples)

    parsed_targets = [parse_output(ex.target) for ex in examples]
    parsed_predictions = [parse_output(pred) for pred in predictions]

    y_true = [int(not p.is_no_variant) for p in parsed_targets]
    y_pred = [int(not p.is_no_variant) for p in parsed_predictions]

    binary = binary_confusion(y_true, y_pred)
    variant_metrics = set_extraction_metrics(
        [p.variants for p in parsed_targets],
        [p.variants for p in parsed_predictions],
    )
    pair_metrics = set_extraction_metrics(
        [p.pairs for p in parsed_targets],
        [p.pairs for p in parsed_predictions],
    )
    validity = format_validity(parsed_predictions)

    metrics = {
        "model": {
            "base_model_name_or_path": BASE_MODEL_NAME_OR_PATH,
            "use_lora_adapter": USE_LORA_ADAPTER,
            "adapter_path": str(ADAPTER_PATH) if USE_LORA_ADAPTER else None,
            "load_in_4bit": LOAD_IN_4BIT,
            "use_bf16": USE_BF16,
            "use_fp16": USE_FP16,
        },
        "data": {
            "test_jsonl": str(TEST_JSONL),
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

    METRICS_JSON.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    write_predictions_csv(PREDICTIONS_CSV, examples, predictions, parsed_targets, parsed_predictions)
    write_confusion_matrix_csv(CONFUSION_MATRIX_CSV, binary)
    write_errors_csv(PER_EXAMPLE_ERRORS_CSV, examples, predictions, parsed_targets, parsed_predictions)

    print("\n=== Binary detection metrics ===")
    print(json.dumps(binary, indent=2))

    print("\n=== Variant extraction micro metrics ===")
    print(json.dumps(variant_metrics, indent=2))

    print("\n=== Gene-variant pair extraction micro metrics ===")
    print(json.dumps(pair_metrics, indent=2))

    print("\n=== Format validity ===")
    print(json.dumps(validity, indent=2))

    print("\nWrote:")
    print(f"- {METRICS_JSON}")
    print(f"- {PREDICTIONS_CSV}")
    print(f"- {CONFUSION_MATRIX_CSV}")
    print(f"- {PER_EXAMPLE_ERRORS_CSV}")


if __name__ == "__main__":
    main()