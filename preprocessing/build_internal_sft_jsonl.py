# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#!/usr/bin/env python3
"""
build_internal_sft_jsonl.py

Preprocess NER_internal_data.jsonl into the same conversational SFT JSONL format
used by the other variant-NER datasets.

Input format observed:
    The file is not strict JSONL. It contains multiple pretty-printed JSON-like
    objects, one after another, with trailing commas before closing braces, e.g.:

    {
      "text": "...",
      "variants": [
        {"gene": "BRAF", "variant": "p.V600E"},
        ...
      ],
    }
    {
      ...
    }

This script therefore includes a tolerant parser that:
    1. splits the file into top-level JSON-like objects,
    2. removes trailing commas before } or ],
    3. parses each cleaned object with json.loads().

Output:
    OUTPUT_DIR/train.jsonl
    OUTPUT_DIR/val.jsonl
    OUTPUT_DIR/test.jsonl
    OUTPUT_DIR/internal_dataset_stats.json
    OUTPUT_DIR/internal_flat_examples.csv
    OUTPUT_DIR/internal_flat_examples.jsonl

SFT output format:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Title: ...\nAbstract: ..."},
    {"role": "assistant", "content": "Variant: p.V600E, Gene: BRAF"}
  ],
  "metadata": {...}
}
"""

from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


# =========================
# Constants: edit if needed
# =========================

RAW_INPUT_PATH = Path("../data/raw/NER_internal_data.jsonl")
OUTPUT_DIR = Path("../data/processed_internal")

TRAIN_JSONL = OUTPUT_DIR / "train.jsonl"
VAL_JSONL = OUTPUT_DIR / "val.jsonl"
TEST_JSONL = OUTPUT_DIR / "test.jsonl"

STATS_JSON = OUTPUT_DIR / "internal_dataset_stats.json"
FLAT_CSV = OUTPUT_DIR / "internal_flat_examples.csv"
FLAT_JSONL = OUTPUT_DIR / "internal_flat_examples.jsonl"

RANDOM_SEED = 42
TRAIN_FRACTION = 0.80
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10

SOURCE_NAME = "internal_ner"
MISSING_GENE_VALUE = "Unknown"

# The uploaded source has only a text field, not separate title/abstract fields.
# To keep the same SFT format as your other datasets, the text is placed in Abstract.
DEFAULT_TITLE = ""

DEDUPLICATE_OUTPUT_LINES = True
SAVE_FLAT_INSPECTION_FILES = True

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
# Data structures
# =========================

@dataclass
class FlatExample:
    source_id: str
    title: str
    abstract: str
    text_chunk: str
    target: str
    n_variants: int
    variant_texts: str
    gene_texts: str


# =========================
# Utilities
# =========================

def normalize_space(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip()


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        item = normalize_space(item)
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def remove_trailing_commas(json_like: str) -> str:
    """Remove commas immediately before } or ]."""
    return re.sub(r",\s*([}\]])", r"\1", json_like)


def split_top_level_objects(text: str) -> list[str]:
    """
    Split a concatenation of top-level JSON-like objects into object strings.

    This is quote-aware, so braces inside strings should not split objects.
    """
    objects: list[str] = []
    start = None
    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start:i + 1])
                start = None
            elif depth < 0:
                raise ValueError(f"Unbalanced braces near character {i}")

    if depth != 0:
        raise ValueError("Unbalanced braces: reached end of file before closing top-level object")

    return objects


def load_tolerant_json_objects(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    raw = path.read_text(encoding="utf-8")
    object_strings = split_top_level_objects(raw)

    records = []
    for idx, obj_text in enumerate(object_strings):
        cleaned = remove_trailing_commas(obj_text)
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            preview = cleaned[:1000]
            raise ValueError(
                f"Could not parse object {idx}. Error: {exc}\nPreview:\n{preview}"
            ) from exc

        if not isinstance(obj, dict):
            raise ValueError(f"Object {idx} is not a JSON object: {type(obj)}")

        records.append(obj)

    return records


# =========================
# Conversion
# =========================

def normalize_variant_record(record: dict) -> tuple[str, str]:
    """Return (variant, gene)."""
    variant = normalize_space(record.get("variant", ""))
    gene = normalize_space(record.get("gene", "")) or MISSING_GENE_VALUE
    return variant, gene


def build_target(variants: Any) -> tuple[str, list[str], list[str]]:
    if not isinstance(variants, list) or not variants:
        return "No variant", [], []

    lines: list[str] = []
    variant_texts: list[str] = []
    gene_texts: list[str] = []

    for item in variants:
        if not isinstance(item, dict):
            continue

        variant, gene = normalize_variant_record(item)
        if not variant:
            continue

        variant_texts.append(variant)
        gene_texts.append(gene)
        lines.append(f"Variant: {variant}, Gene: {gene}")

    if DEDUPLICATE_OUTPUT_LINES:
        lines = dedupe_keep_order(lines)
        variant_texts = dedupe_keep_order(variant_texts)
        gene_texts = dedupe_keep_order(gene_texts)

    if not lines:
        return "No variant", variant_texts, gene_texts

    return "\n".join(lines), variant_texts, gene_texts


def record_to_flat_example(record: dict, idx: int) -> FlatExample | None:
    text = normalize_space(record.get("text", ""))
    if not text:
        return None

    target, variant_texts, gene_texts = build_target(record.get("variants", []))

    return FlatExample(
        source_id=str(record.get("id", record.get("pmid", idx))),
        title=normalize_space(record.get("title", DEFAULT_TITLE)),
        abstract=text,
        text_chunk=text,
        target=target,
        n_variants=0 if target == "No variant" else len(target.splitlines()),
        variant_texts=" | ".join(variant_texts),
        gene_texts=" | ".join(gene_texts),
    )


def flat_example_to_sft_row(ex: FlatExample) -> dict:
    user_content = f"Title: {ex.title}\nAbstract: {ex.abstract}"

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": ex.target,
            },
        ],
        "metadata": {
            "source": SOURCE_NAME,
            "source_id": ex.source_id,
            "n_variants": ex.n_variants,
            "variant_texts": ex.variant_texts,
            "gene_texts": ex.gene_texts,
        },
    }


# =========================
# Splitting / writing
# =========================

def deterministic_split(rows: list[FlatExample]) -> tuple[list[FlatExample], list[FlatExample], list[FlatExample]]:
    rng = random.Random(RANDOM_SEED)
    rows = list(rows)
    rng.shuffle(rows)

    n = len(rows)
    n_train = int(n * TRAIN_FRACTION)
    n_val = int(n * VAL_FRACTION)

    train = rows[:n_train]
    val = rows[n_train:n_train + n_val]
    test = rows[n_train + n_val:]

    return train, val, test


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_positive(rows: list[FlatExample]) -> int:
    return sum(1 for r in rows if r.target != "No variant")


def count_negative(rows: list[FlatExample]) -> int:
    return sum(1 for r in rows if r.target == "No variant")


def write_flat_files(rows: list[FlatExample]) -> None:
    if not SAVE_FLAT_INSPECTION_FILES:
        return

    dict_rows = [asdict(r) for r in rows]

    pd.DataFrame(dict_rows).to_csv(
        FLAT_CSV,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    with FLAT_JSONL.open("w", encoding="utf-8") as f:
        for row in dict_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================
# Main
# =========================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading input file: {RAW_INPUT_PATH}")
    records = load_tolerant_json_objects(RAW_INPUT_PATH)
    print(f"Parsed source records: {len(records)}")

    flat_examples: list[FlatExample] = []
    dropped_empty_text = 0

    for idx, record in enumerate(records):
        ex = record_to_flat_example(record, idx=idx)
        if ex is None:
            dropped_empty_text += 1
            continue
        flat_examples.append(ex)

    if not flat_examples:
        raise RuntimeError("No examples were created from the input file.")

    train_examples, val_examples, test_examples = deterministic_split(flat_examples)

    train_rows = [flat_example_to_sft_row(ex) for ex in train_examples]
    val_rows = [flat_example_to_sft_row(ex) for ex in val_examples]
    test_rows = [flat_example_to_sft_row(ex) for ex in test_examples]

    write_jsonl(TRAIN_JSONL, train_rows)
    write_jsonl(VAL_JSONL, val_rows)
    write_jsonl(TEST_JSONL, test_rows)
    write_flat_files(flat_examples)

    stats = {
        "raw_input_path": str(RAW_INPUT_PATH),
        "source_name": SOURCE_NAME,
        "output_dir": str(OUTPUT_DIR),
        "random_seed": RANDOM_SEED,
        "train_fraction": TRAIN_FRACTION,
        "val_fraction": VAL_FRACTION,
        "test_fraction": TEST_FRACTION,
        "parsed_source_records": len(records),
        "dropped_empty_text": dropped_empty_text,
        "total_examples": len(flat_examples),
        "positive_examples": count_positive(flat_examples),
        "negative_examples": count_negative(flat_examples),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "test_examples": len(test_examples),
        "train_positive_examples": count_positive(train_examples),
        "val_positive_examples": count_positive(val_examples),
        "test_positive_examples": count_positive(test_examples),
        "train_negative_examples": count_negative(train_examples),
        "val_negative_examples": count_negative(val_examples),
        "test_negative_examples": count_negative(test_examples),
        "missing_gene_value": MISSING_GENE_VALUE,
        "deduplicate_output_lines": DEDUPLICATE_OUTPUT_LINES,
        "train_jsonl": str(TRAIN_JSONL),
        "val_jsonl": str(VAL_JSONL),
        "test_jsonl": str(TEST_JSONL),
        "flat_csv": str(FLAT_CSV) if SAVE_FLAT_INSPECTION_FILES else None,
        "flat_jsonl": str(FLAT_JSONL) if SAVE_FLAT_INSPECTION_FILES else None,
    }

    STATS_JSON.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDone.")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print("\nWrote:")
    print(f"- {TRAIN_JSONL}")
    print(f"- {VAL_JSONL}")
    print(f"- {TEST_JSONL}")
    print(f"- {STATS_JSON}")
    if SAVE_FLAT_INSPECTION_FILES:
        print(f"- {FLAT_CSV}")
        print(f"- {FLAT_JSONL}")


if __name__ == "__main__":
    main()
