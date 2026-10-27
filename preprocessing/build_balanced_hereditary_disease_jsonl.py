# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#!/usr/bin/env python3
"""
build_balanced_hereditary_disease_jsonl.py

Create a balanced-ish version of the hereditary disease SFT JSONL files.

Procedure, applied independently to each split:
    1. Read JSONL from ../data/processed_hereditary_disease_sft
    2. Keep all positive examples
    3. Randomly sample 10% of negative examples
    4. Write output JSONL to ../data/processed_hereditary_disease_balanced
    5. Write updated stats JSON

Positive definition:
    An example is positive if assistant content is not exactly "No variant".

Negative definition:
    An example is negative if assistant content is exactly "No variant".

Input:
    ../data/processed_hereditary_disease/train.jsonl
    ../data/processed_hereditary_disease/val.jsonl
    ../data/processed_hereditary_disease/test.jsonl

Output:
    ../data/processed_hereditary_disease_balanced/train.jsonl
    ../data/processed_hereditary_disease_balanced/val.jsonl
    ../data/processed_hereditary_disease_balanced/test.jsonl
    ../data/processed_hereditary_disease_balanced/hereditary_disease_balanced_stats.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


# =========================
# Constants
# =========================

INPUT_DIR = Path("../data/processed_hereditary_disease")
OUTPUT_DIR = Path("../data/processed_hereditary_disease_balanced")

SPLITS = ["train", "val", "test"]

INPUT_JSONL = {
    split: INPUT_DIR / f"{split}.jsonl"
    for split in SPLITS
}

OUTPUT_JSONL = {
    split: OUTPUT_DIR / f"{split}.jsonl"
    for split in SPLITS
}

STATS_JSON = OUTPUT_DIR / "hereditary_disease_balanced_stats.json"

NEGATIVE_SAMPLE_FRACTION = 0.10
RANDOM_SEED = 42

NEGATIVE_ASSISTANT_CONTENT = "No variant"

SHUFFLE_OUTPUT_ROWS = True


# =========================
# JSONL utilities
# =========================

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc

    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================
# Label utilities
# =========================

def get_assistant_content(row: dict[str, Any]) -> str:
    """
    Extract assistant message content from the conversational SFT row.

    Expected format:
        {
          "messages": [
            {"role": "system", ...},
            {"role": "user", ...},
            {"role": "assistant", "content": "..."}
          ],
          "metadata": {...}
        }
    """
    messages = row.get("messages")

    if not isinstance(messages, list):
        raise ValueError(f"Row has no valid messages list: {row}")

    assistant_messages = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]

    if not assistant_messages:
        raise ValueError(f"Row has no assistant message: {row}")

    content = assistant_messages[-1].get("content", "")

    if content is None:
        return ""

    return str(content).strip()


def is_negative(row: dict[str, Any]) -> bool:
    return get_assistant_content(row) == NEGATIVE_ASSISTANT_CONTENT


def is_positive(row: dict[str, Any]) -> bool:
    return not is_negative(row)


# =========================
# Balancing
# =========================

def sample_negatives(
    negatives: list[dict[str, Any]],
    fraction: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if not 0 <= fraction <= 1:
        raise ValueError("NEGATIVE_SAMPLE_FRACTION must be between 0 and 1.")

    n_to_keep = int(len(negatives) * fraction)

    if n_to_keep <= 0:
        return []

    return rng.sample(negatives, n_to_keep)


def balance_split(
    rows: list[dict[str, Any]],
    split_name: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Keep all positives and sample 10% negatives for one split.
    """
    positives = [row for row in rows if is_positive(row)]
    negatives = [row for row in rows if is_negative(row)]

    rng = random.Random(f"{seed}_{split_name}")

    sampled_negatives = sample_negatives(
        negatives=negatives,
        fraction=NEGATIVE_SAMPLE_FRACTION,
        rng=rng,
    )

    balanced_rows = positives + sampled_negatives

    if SHUFFLE_OUTPUT_ROWS:
        rng.shuffle(balanced_rows)

    original_total = len(rows)
    original_positive = len(positives)
    original_negative = len(negatives)

    balanced_total = len(balanced_rows)
    balanced_positive = len(positives)
    balanced_negative = len(sampled_negatives)

    stats = {
        "original_total": original_total,
        "original_positive": original_positive,
        "original_negative": original_negative,
        "original_positive_fraction": (
            original_positive / original_total if original_total else 0
        ),
        "original_negative_fraction": (
            original_negative / original_total if original_total else 0
        ),
        "negative_sample_fraction_requested": NEGATIVE_SAMPLE_FRACTION,
        "negative_sample_fraction_actual": (
            balanced_negative / original_negative if original_negative else 0
        ),
        "balanced_total": balanced_total,
        "balanced_positive": balanced_positive,
        "balanced_negative": balanced_negative,
        "balanced_positive_fraction": (
            balanced_positive / balanced_total if balanced_total else 0
        ),
        "balanced_negative_fraction": (
            balanced_negative / balanced_total if balanced_total else 0
        ),
        "dropped_negative": original_negative - balanced_negative,
        "dropped_positive": 0,
        "input_path": str(INPUT_JSONL[split_name]),
        "output_path": str(OUTPUT_JSONL[split_name]),
    }

    return balanced_rows, stats


# =========================
# Validation
# =========================

def validate_expected_files() -> None:
    missing = [
        str(path)
        for path in INPUT_JSONL.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing input JSONL files:\n" + "\n".join(missing)
        )


def validate_no_positive_dropped(
    original_rows: list[dict[str, Any]],
    balanced_rows: list[dict[str, Any]],
) -> None:
    """
    Defensive check: ensure all original positive rows are present in output.

    Uses JSON string identity for exact row matching.
    """
    original_positive_keys = {
        json.dumps(row, sort_keys=True, ensure_ascii=False)
        for row in original_rows
        if is_positive(row)
    }

    balanced_positive_keys = {
        json.dumps(row, sort_keys=True, ensure_ascii=False)
        for row in balanced_rows
        if is_positive(row)
    }

    missing = original_positive_keys - balanced_positive_keys

    if missing:
        raise RuntimeError(
            f"{len(missing)} positive examples were dropped unexpectedly."
        )


# =========================
# Main
# =========================

def main() -> None:
    validate_expected_files()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_stats: dict[str, Any] = {
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "splits": {},
        "negative_sample_fraction": NEGATIVE_SAMPLE_FRACTION,
        "random_seed": RANDOM_SEED,
        "shuffle_output_rows": SHUFFLE_OUTPUT_ROWS,
        "positive_definition": (
            "assistant content is not exactly 'No variant'"
        ),
        "negative_definition": (
            "assistant content is exactly 'No variant'"
        ),
    }

    total_original = 0
    total_original_positive = 0
    total_original_negative = 0

    total_balanced = 0
    total_balanced_positive = 0
    total_balanced_negative = 0

    for split_name in SPLITS:
        input_path = INPUT_JSONL[split_name]
        output_path = OUTPUT_JSONL[split_name]

        print(f"Reading {split_name}: {input_path}")

        original_rows = read_jsonl(input_path)

        balanced_rows, split_stats = balance_split(
            rows=original_rows,
            split_name=split_name,
            seed=RANDOM_SEED,
        )

        validate_no_positive_dropped(
            original_rows=original_rows,
            balanced_rows=balanced_rows,
        )

        write_jsonl(output_path, balanced_rows)

        all_stats["splits"][split_name] = split_stats

        total_original += split_stats["original_total"]
        total_original_positive += split_stats["original_positive"]
        total_original_negative += split_stats["original_negative"]

        total_balanced += split_stats["balanced_total"]
        total_balanced_positive += split_stats["balanced_positive"]
        total_balanced_negative += split_stats["balanced_negative"]

        print(
            f"Wrote {split_name}: "
            f"{split_stats['balanced_total']} rows "
            f"({split_stats['balanced_positive']} positive, "
            f"{split_stats['balanced_negative']} negative) -> {output_path}"
        )

    all_stats["totals"] = {
        "original_total": total_original,
        "original_positive": total_original_positive,
        "original_negative": total_original_negative,
        "original_positive_fraction": (
            total_original_positive / total_original if total_original else 0
        ),
        "original_negative_fraction": (
            total_original_negative / total_original if total_original else 0
        ),
        "balanced_total": total_balanced,
        "balanced_positive": total_balanced_positive,
        "balanced_negative": total_balanced_negative,
        "balanced_positive_fraction": (
            total_balanced_positive / total_balanced if total_balanced else 0
        ),
        "balanced_negative_fraction": (
            total_balanced_negative / total_balanced if total_balanced else 0
        ),
        "dropped_negative": total_original_negative - total_balanced_negative,
        "dropped_positive": 0,
    }

    STATS_JSON.write_text(
        json.dumps(all_stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nDone.")
    print(json.dumps(all_stats, indent=2, ensure_ascii=False))
    print("\nWrote:")
    for split_name in SPLITS:
        print(f"- {OUTPUT_JSONL[split_name]}")
    print(f"- {STATS_JSON}")


if __name__ == "__main__":
    main()