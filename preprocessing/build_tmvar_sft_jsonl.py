# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#!/usr/bin/env python3
"""
build_tmvar_sft_jsonl.py

Preprocess tmVar v3 into the same conversational SFT JSONL format used for
variant NER fine-tuning.

Input source:
    Hugging Face datasets loader from a local tmVar script:
        load_dataset(path=DATASET_SCRIPT_PATH, name=DATASET_NAME)

Output:
    OUTPUT_DIR/train.jsonl
    OUTPUT_DIR/val.jsonl
    OUTPUT_DIR/test.jsonl
    OUTPUT_DIR/tmvar_dataset_stats.json
    OUTPUT_DIR/tmvar_flat_examples.csv        optional inspection file
    OUTPUT_DIR/tmvar_flat_examples.jsonl      optional inspection file

Expected SFT example format:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Title: ...\nAbstract: ..."},
    {"role": "assistant", "content": "Variant: V600E, Gene: BRAF"}
  ],
  "metadata": {"pmid": "...", "source": "tmvar_v3"}
}

Entity mapping:
    Variant: DNAMutation, OtherMutation, ProteinMutation, SNP
    Gene:    Gene

Gene assignment strategy:
    1. If entity offsets are available, assign each variant to the nearest gene
       mention in the same article.
    2. If offsets are not available and exactly one unique gene exists, use it.
    3. Otherwise use MISSING_GENE_VALUE.

This script does not assume that tmVar provides explicit variant-gene relations.
It creates weak gene associations from mention proximity. If you need gold
variant-gene linking, replace assign_gene_to_variant() with a relation-aware method.
"""

from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset


# =========================
# Constants: edit if needed
# =========================

DATASET_SCRIPT_PATH = "./tmvar/tmvar_v3.py"
DATASET_NAME = "tmvar_v3_source"

OUTPUT_DIR = Path("../data/processed_tmvar")

TRAIN_JSONL = OUTPUT_DIR / "train.jsonl"
VAL_JSONL = OUTPUT_DIR / "val.jsonl"
TEST_JSONL = OUTPUT_DIR / "test.jsonl"

STATS_JSON = OUTPUT_DIR / "tmvar_dataset_stats.json"
FLAT_CSV = OUTPUT_DIR / "tmvar_flat_examples.csv"
FLAT_JSONL = OUTPUT_DIR / "tmvar_flat_examples.jsonl"

RANDOM_SEED = 42

TRAIN_FRACTION = 0.80
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10

SOURCE_NAME = "tmvar_v3"

# Use all splits from the loaded DatasetDict by default.
# To reproduce your exploration script exactly, set: SOURCE_SPLITS_TO_USE = ["test"]
SOURCE_SPLITS_TO_USE: list[str] | None = None

VARIANT_ENTITY_TYPES = {
    "DNAMutation",
    "OtherMutation",
    "ProteinMutation",
    "SNP",
}

GENE_ENTITY_TYPES = {"Gene"}
SELECTED_ENTITY_TYPES = VARIANT_ENTITY_TYPES | GENE_ENTITY_TYPES

MISSING_GENE_VALUE = "Unknown"
KEEP_VARIANTS_WITH_UNKNOWN_GENE = True
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
class EntityMention:
    text: str
    entity_type: str
    start: int | None = None
    end: int | None = None


@dataclass
class FlatExample:
    pmid: str
    title: str
    abstract: str
    text_chunk: str
    target: str
    n_variants: int
    n_genes: int
    variant_texts: str
    gene_texts: str
    source_split: str


# =========================
# Basic utilities
# =========================

def normalize_space(text: Any) -> str:
    text = "" if text is None else str(text)
    return re.sub(r"\s+", " ", text).strip()


def first_text(value: Any) -> str:
    """Return the first text value from list-like BigBio/tmVar fields."""
    if value is None:
        return ""
    if isinstance(value, list):
        if not value:
            return ""
        return normalize_space(value[0])
    return normalize_space(value)


def get_article_pmid(article: dict) -> str:
    for key in ["pmid", "document_id", "id"]:
        if key in article and article[key] is not None:
            return str(article[key])
    return ""


def get_title_and_abstract(article: dict) -> tuple[str, str]:
    """
    Robust extraction for your observed structure:
        article["passages"][0]["text"][0] -> title
        article["passages"][1]["text"][0] -> abstract
    """
    passages = article.get("passages", []) or []

    title = ""
    abstract = ""

    # Prefer passage metadata/type if available.
    for passage in passages:
        text = first_text(passage.get("text", ""))
        if not text:
            continue

        passage_type = ""
        for key in ["type", "section_type", "infons"]:
            val = passage.get(key)
            if isinstance(val, str):
                passage_type += " " + val.lower()
            elif isinstance(val, dict):
                passage_type += " " + " ".join(str(v).lower() for v in val.values())

        if "title" in passage_type and not title:
            title = text
        elif ("abstract" in passage_type or "paragraph" in passage_type) and not abstract:
            abstract = text

    # Fallback to positional convention used in your exploration.
    if not title and len(passages) >= 1:
        title = first_text(passages[0].get("text", ""))
    if not abstract and len(passages) >= 2:
        abstract = first_text(passages[1].get("text", ""))

    # If there is only one passage, treat it as abstract-like text.
    if not abstract and title and len(passages) == 1:
        abstract = title
        title = ""

    return title, abstract


def extract_entity_offsets(entity: dict) -> tuple[int | None, int | None]:
    """
    Tries to extract offsets from common BigBio/tmVar representations.

    Supported patterns include:
        entity["offsets"] = [[start, end]]
        entity["offsets"] = [{"start": start, "end": end}]
        entity["locations"] = [...]
        entity["start"] / entity["end"]
    """
    for key in ["offsets", "locations"]:
        offsets = entity.get(key)
        if isinstance(offsets, list) and offsets:
            first = offsets[0]

            if isinstance(first, list) and len(first) >= 2:
                try:
                    return int(first[0]), int(first[1])
                except Exception:
                    pass

            if isinstance(first, tuple) and len(first) >= 2:
                try:
                    return int(first[0]), int(first[1])
                except Exception:
                    pass

            if isinstance(first, dict):
                start = first.get("start", first.get("begin"))
                end = first.get("end", first.get("stop"))
                if start is not None and end is not None:
                    try:
                        return int(start), int(end)
                    except Exception:
                        pass

    start = entity.get("start", entity.get("begin"))
    end = entity.get("end", entity.get("stop"))
    if start is not None and end is not None:
        try:
            return int(start), int(end)
        except Exception:
            pass

    return None, None


def extract_entities(article: dict) -> list[EntityMention]:
    mentions: list[EntityMention] = []

    for entity in article.get("entities", []) or []:
        entity_type = entity.get("semantic_type_id", entity.get("type", ""))
        if entity_type not in SELECTED_ENTITY_TYPES:
            continue

        text = first_text(entity.get("text", ""))
        if not text:
            continue

        start, end = extract_entity_offsets(entity)

        mentions.append(
            EntityMention(
                text=text,
                entity_type=str(entity_type),
                start=start,
                end=end,
            )
        )

    return mentions


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        key = normalize_space(item)
        if key and key not in seen:
            out.append(key)
            seen.add(key)
    return out


def canonical_gene_for_output(gene: str) -> str:
    gene = normalize_space(gene)
    return gene if gene else MISSING_GENE_VALUE


def canonical_variant_for_output(variant: str) -> str:
    return normalize_space(variant)


# =========================
# Gene assignment
# =========================

def has_offsets(mention: EntityMention) -> bool:
    return mention.start is not None and mention.end is not None


def mention_midpoint(mention: EntityMention) -> float | None:
    if mention.start is None or mention.end is None:
        return None
    return (mention.start + mention.end) / 2.0


def assign_gene_to_variant(variant: EntityMention, genes: list[EntityMention]) -> str:
    """
    Assign a gene to a variant mention.

    This is a heuristic because tmVar entity annotations do not necessarily encode
    gold variant-gene relations.
    """
    if not genes:
        return MISSING_GENE_VALUE

    variant_mid = mention_midpoint(variant)

    genes_with_offsets = [g for g in genes if has_offsets(g)]
    if variant_mid is not None and genes_with_offsets:
        def distance(g: EntityMention) -> float:
            gene_mid = mention_midpoint(g)
            if gene_mid is None:
                return float("inf")
            return abs(variant_mid - gene_mid)

        nearest = min(genes_with_offsets, key=distance)
        return canonical_gene_for_output(nearest.text)

    unique_genes = dedupe_keep_order([canonical_gene_for_output(g.text) for g in genes])
    if len(unique_genes) == 1:
        return unique_genes[0]

    return MISSING_GENE_VALUE


def build_target(entities: list[EntityMention]) -> tuple[str, list[str], list[str]]:
    variants = [e for e in entities if e.entity_type in VARIANT_ENTITY_TYPES]
    genes = [e for e in entities if e.entity_type in GENE_ENTITY_TYPES]

    gene_texts = dedupe_keep_order([canonical_gene_for_output(g.text) for g in genes])
    variant_texts = dedupe_keep_order([canonical_variant_for_output(v.text) for v in variants])

    if not variants:
        return "No variant", variant_texts, gene_texts

    lines = []
    for variant in variants:
        variant_text = canonical_variant_for_output(variant.text)
        if not variant_text:
            continue

        gene = assign_gene_to_variant(variant, genes)
        if gene == MISSING_GENE_VALUE and not KEEP_VARIANTS_WITH_UNKNOWN_GENE:
            continue

        lines.append(f"Variant: {variant_text}, Gene: {gene}")

    if DEDUPLICATE_OUTPUT_LINES:
        lines = dedupe_keep_order(lines)

    if not lines:
        return "No variant", variant_texts, gene_texts

    return "\n".join(lines), variant_texts, gene_texts


# =========================
# Conversion
# =========================

def article_to_flat_example(article: dict, source_split: str) -> FlatExample | None:
    pmid = get_article_pmid(article)
    title, abstract = get_title_and_abstract(article)

    if not title and not abstract:
        return None

    text_chunk = normalize_space((title + " " + abstract).strip())

    entities = extract_entities(article)
    target, variant_texts, gene_texts = build_target(entities)

    return FlatExample(
        pmid=pmid,
        title=title,
        abstract=abstract,
        text_chunk=text_chunk,
        target=target,
        n_variants=0 if target == "No variant" else len(target.splitlines()),
        n_genes=len(gene_texts),
        variant_texts=" | ".join(variant_texts),
        gene_texts=" | ".join(gene_texts),
        source_split=source_split,
    )


def flat_example_to_sft_row(ex: FlatExample) -> dict:
    user_content = f"Title: {ex.title}\nAbstract: {ex.abstract}".strip()

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": ex.target},
        ],
        "metadata": {
            "pmid": ex.pmid,
            "source": SOURCE_NAME,
            "source_split": ex.source_split,
            "n_variants": ex.n_variants,
            "n_genes": ex.n_genes,
            "variant_texts": ex.variant_texts,
            "gene_texts": ex.gene_texts,
        },
    }


# =========================
# Split / write
# =========================

def deterministic_split(rows: list[FlatExample]) -> tuple[list[FlatExample], list[FlatExample], list[FlatExample]]:
    """Deterministic seeded 80/10/10 split."""
    if abs((TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION) - 1.0) > 1e-6:
        raise ValueError("TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION must equal 1.0")

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

    print(f"Loading tmVar dataset from: {DATASET_SCRIPT_PATH}")
    print(f"Dataset name: {DATASET_NAME}")

    dataset = load_dataset(
        path=DATASET_SCRIPT_PATH,
        name=DATASET_NAME,
    )

    available_splits = list(dataset.keys())
    if SOURCE_SPLITS_TO_USE is None:
        splits_to_use = available_splits
    else:
        missing = [s for s in SOURCE_SPLITS_TO_USE if s not in dataset]
        if missing:
            raise ValueError(f"Requested splits not found: {missing}. Available: {available_splits}")
        splits_to_use = SOURCE_SPLITS_TO_USE

    print(f"Available splits: {available_splits}")
    print(f"Using splits:      {splits_to_use}")

    flat_examples: list[FlatExample] = []
    dropped_empty_text = 0

    for split_name in splits_to_use:
        for article in dataset[split_name]:
            ex = article_to_flat_example(article, source_split=split_name)
            if ex is None:
                dropped_empty_text += 1
                continue
            flat_examples.append(ex)

    if not flat_examples:
        raise RuntimeError("No examples were created from tmVar.")

    train_examples, val_examples, test_examples = deterministic_split(flat_examples)

    train_rows = [flat_example_to_sft_row(ex) for ex in train_examples]
    val_rows = [flat_example_to_sft_row(ex) for ex in val_examples]
    test_rows = [flat_example_to_sft_row(ex) for ex in test_examples]

    write_jsonl(TRAIN_JSONL, train_rows)
    write_jsonl(VAL_JSONL, val_rows)
    write_jsonl(TEST_JSONL, test_rows)

    write_flat_files(flat_examples)

    stats = {
        "dataset_script_path": DATASET_SCRIPT_PATH,
        "dataset_name": DATASET_NAME,
        "source_name": SOURCE_NAME,
        "available_splits": available_splits,
        "splits_used": splits_to_use,
        "output_dir": str(OUTPUT_DIR),
        "random_seed": RANDOM_SEED,
        "train_fraction": TRAIN_FRACTION,
        "val_fraction": VAL_FRACTION,
        "test_fraction": TEST_FRACTION,
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
        "variant_entity_types": sorted(VARIANT_ENTITY_TYPES),
        "gene_entity_types": sorted(GENE_ENTITY_TYPES),
        "missing_gene_value": MISSING_GENE_VALUE,
        "keep_variants_with_unknown_gene": KEEP_VARIANTS_WITH_UNKNOWN_GENE,
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
