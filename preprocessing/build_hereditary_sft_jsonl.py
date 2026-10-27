# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#!/usr/bin/env python3

"""
The data source are the 400 hereditary-disease-related PubMed full articles manually annotated 
for genes, variants, diseases, and species, used in the study:

Huang, DL., Zeng, Q., Xiong, Y. et al. 
A Combined Manual Annotation and Deep-Learning Natural Language Processing Study on Accurate Entity Extraction in Hereditary Disease Related Biomedical Literature. 
Interdiscip Sci Comput Life Sci 16, 333–344 (2024). https://doi.org/10.1007/s12539-024-00605-2
https://github.com/dlhuang/Entity_Extraction_Hereditary_Disease_2023/tree/master/data


build_hereditary_disease_sft_jsonl.py

Convert BIO-style hereditary disease entity extraction .txt files into the same
conversational SFT JSONL format used by the previous tmVar preprocessing script.

Input:
    ../data/raw/Entity_Extraction_Hereditary_Disease_2023/PhaseII_data/train.txt
    ../data/raw/Entity_Extraction_Hereditary_Disease_2023/PhaseII_data/dev.txt
    ../data/raw/Entity_Extraction_Hereditary_Disease_2023/PhaseII_data/test.txt

Output:
    ../data/processed_hereditary_disease/train.jsonl
    ../data/processed_hereditary_disease/val.jsonl
    ../data/processed_hereditary_disease/test.jsonl
    ../data/processed_hereditary_disease/hereditary_disease_dataset_stats.json
    ../data/processed_hereditary_disease/hereditary_disease_flat_examples.csv
    ../data/processed_hereditary_disease/hereditary_disease_flat_examples.jsonl

Expected raw format:
    token TAG
    token TAG
    ...

    token TAG
    token TAG
    ...

Each empty line ends one text chunk.

Entity mapping:
    Gene:    B-gene / I-gene
    Variant: B-mutation / I-mutation

Assistant target format:
    Variant: <mutation>, Gene: <gene>

or exactly:
    No variant
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


# =========================
# Constants
# =========================

RAW_DIR = Path("../data/raw/Entity_Extraction_Hereditary_Disease_2023/PhaseII_data")
OUTPUT_DIR = Path("../data/processed_hereditary_disease")

INPUT_SPLITS = {
    "train": RAW_DIR / "train.txt",
    "val": RAW_DIR / "dev.txt",
    "test": RAW_DIR / "test.txt",
}

OUTPUT_JSONL = {
    "train": OUTPUT_DIR / "train.jsonl",
    "val": OUTPUT_DIR / "val.jsonl",
    "test": OUTPUT_DIR / "test.jsonl",
}

STATS_JSON = OUTPUT_DIR / "hereditary_disease_dataset_stats.json"
FLAT_CSV = OUTPUT_DIR / "hereditary_disease_flat_examples.csv"
FLAT_JSONL = OUTPUT_DIR / "hereditary_disease_flat_examples.jsonl"

SOURCE_NAME = "entity_extraction_hereditary_disease_2023_phaseII"

GENE_TYPE = "gene"
VARIANT_TYPE = "mutation"

RAW_GENE_TAG = "gene"
RAW_VARIANT_TAG = "mutation"

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
    token_start: int
    token_end: int
    char_start: int | None = None
    char_end: int | None = None


@dataclass
class FlatExample:
    example_id: str
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
# Text utilities
# =========================

def normalize_space(text: Any) -> str:
    text = "" if text is None else str(text)
    return re.sub(r"\s+", " ", text).strip()


def detokenize(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """
    Detokenize tokens while keeping approximate character spans for each token.

    Returns:
        text:
            Reconstructed text.
        spans:
            List of (char_start, char_end) for each token in the reconstructed text.

    The span tracking is useful for inspection/debugging, while gene assignment
    is done by token distance.
    """
    no_space_before = {
        ",", ".", ";", ":", "!", "?", "%", ")", "]", "}", "'", "''"
    }
    no_space_after = {
        "(", "[", "{", "``", "$", "#"
    }

    text_parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0

    for i, token in enumerate(tokens):
        if i == 0:
            prefix = ""
        else:
            prev = tokens[i - 1]
            if token in no_space_before:
                prefix = ""
            elif prev in no_space_after:
                prefix = ""
            else:
                prefix = " "

        if prefix:
            text_parts.append(prefix)
            cursor += len(prefix)

        start = cursor
        text_parts.append(token)
        cursor += len(token)
        end = cursor

        spans.append((start, end))

    text = "".join(text_parts)

    # Clean common Penn Treebank quotes while keeping text readable.
    text = text.replace("``", '"').replace("''", '"')

    return text, spans


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []

    for item in items:
        item = normalize_space(item)
        if not item:
            continue

        if item not in seen:
            out.append(item)
            seen.add(item)

    return out


# =========================
# BIO parsing
# =========================

def parse_tag(tag: str) -> tuple[str, str] | tuple[None, None]:
    """
    Parse a BIO tag.

    Examples:
        O           -> (None, None)
        B-gene      -> ("B", "gene")
        I-mutation  -> ("I", "mutation")
    """
    tag = normalize_space(tag)

    if tag == "O" or not tag:
        return None, None

    if "-" not in tag:
        return None, None

    prefix, entity_type = tag.split("-", 1)
    prefix = prefix.strip()
    entity_type = entity_type.strip()

    if prefix not in {"B", "I"}:
        return None, None

    return prefix, entity_type


def keep_entity_type(entity_type: str | None) -> bool:
    return entity_type in {RAW_GENE_TAG, RAW_VARIANT_TAG}


def read_bio_chunks(path: Path) -> list[tuple[list[str], list[str]]]:
    """
    Read a BIO file into chunks.

    Each returned item is:
        (tokens, tags)

    Empty lines close chunks.
    """
    chunks: list[tuple[list[str], list[str]]] = []

    tokens: list[str] = []
    tags: list[str] = []

    def flush() -> None:
        nonlocal tokens, tags

        if tokens:
            chunks.append((tokens, tags))

        tokens = []
        tags = []

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                flush()
                continue

            parts = line.rsplit(maxsplit=1)

            if len(parts) != 2:
                # Skip malformed lines defensively.
                continue

            token, tag = parts
            tokens.append(token)
            tags.append(tag)

    flush()

    return chunks


def extract_entities_from_bio(
    tokens: list[str],
    tags: list[str],
    token_char_spans: list[tuple[int, int]],
) -> list[EntityMention]:
    """
    Extract BIO entities from one chunk.

    Keeps only:
        gene
        mutation

    Other entity types are ignored.
    """
    entities: list[EntityMention] = []

    current_type: str | None = None
    current_tokens: list[str] = []
    current_start: int | None = None
    current_end: int | None = None

    def close_entity() -> None:
        nonlocal current_type, current_tokens, current_start, current_end

        if (
            current_type is not None
            and keep_entity_type(current_type)
            and current_tokens
            and current_start is not None
            and current_end is not None
        ):
            char_start = token_char_spans[current_start][0]
            char_end = token_char_spans[current_end - 1][1]

            entities.append(
                EntityMention(
                    text=normalize_space(" ".join(current_tokens)),
                    entity_type=current_type,
                    token_start=current_start,
                    token_end=current_end,
                    char_start=char_start,
                    char_end=char_end,
                )
            )

        current_type = None
        current_tokens = []
        current_start = None
        current_end = None

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        prefix, entity_type = parse_tag(tag)

        if prefix is None or entity_type is None or not keep_entity_type(entity_type):
            close_entity()
            continue

        if prefix == "B":
            close_entity()
            current_type = entity_type
            current_tokens = [token]
            current_start = i
            current_end = i + 1

        elif prefix == "I":
            if current_type == entity_type:
                current_tokens.append(token)
                current_end = i + 1
            else:
                # Defensive handling for invalid BIO:
                # treat stray I-type as a new B-type entity.
                close_entity()
                current_type = entity_type
                current_tokens = [token]
                current_start = i
                current_end = i + 1

    close_entity()

    return entities


def entities_to_dict(entities: list[EntityMention]) -> dict[str, list[str]]:
    """
    Compact dictionary for the DataFrame column requested in the previous step.

    Example:
        {
            "gene": ["BRCA1", "BRCA2"],
            "mutation": ["185/187delAG"]
        }
    """
    out: dict[str, list[str]] = defaultdict(list)

    for entity in entities:
        out[entity.entity_type].append(entity.text)

    return {
        entity_type: dedupe_keep_order(values)
        for entity_type, values in out.items()
    }


def parse_bio_txt_to_dataframe(path: Path, split_name: str) -> pd.DataFrame:
    """
    Build a DataFrame with:
        text
        entities

    Also includes internal helper columns used for JSONL conversion:
        example_id
        entity_mentions
        source_split
    """
    rows: list[dict[str, Any]] = []

    chunks = read_bio_chunks(path)

    for idx, (tokens, tags) in enumerate(chunks):
        text, token_char_spans = detokenize(tokens)
        entity_mentions = extract_entities_from_bio(tokens, tags, token_char_spans)

        rows.append(
            {
                "example_id": f"{split_name}_{idx:06d}",
                "text": text,
                "entities": entities_to_dict(entity_mentions),
                "entity_mentions": entity_mentions,
                "source_split": split_name,
            }
        )

    return pd.DataFrame(rows)


# =========================
# Gene assignment and target building
# =========================

def mention_midpoint_token(entity: EntityMention) -> float:
    return (entity.token_start + entity.token_end) / 2.0


def canonical_gene_for_output(gene: str) -> str:
    gene = normalize_space(gene)
    return gene if gene else MISSING_GENE_VALUE


def canonical_variant_for_output(variant: str) -> str:
    return normalize_space(variant)


def assign_gene_to_variant(
    variant: EntityMention,
    genes: list[EntityMention],
) -> str:
    """
    Assign a gene to a variant mention.

    Since the raw BIO files do not encode explicit variant-gene relations, this
    follows the same weak-association idea as the previous script: assign each
    variant to the nearest gene mention in the same text chunk.

    If there are no genes, returns Unknown.
    """
    if not genes:
        return MISSING_GENE_VALUE

    variant_mid = mention_midpoint_token(variant)

    nearest_gene = min(
        genes,
        key=lambda gene: abs(variant_mid - mention_midpoint_token(gene)),
    )

    return canonical_gene_for_output(nearest_gene.text)


def build_target(
    entities: list[EntityMention],
) -> tuple[str, list[str], list[str]]:
    variants = [
        entity
        for entity in entities
        if entity.entity_type == RAW_VARIANT_TAG
    ]

    genes = [
        entity
        for entity in entities
        if entity.entity_type == RAW_GENE_TAG
    ]

    variant_texts = dedupe_keep_order(
        [canonical_variant_for_output(v.text) for v in variants]
    )
    gene_texts = dedupe_keep_order(
        [canonical_gene_for_output(g.text) for g in genes]
    )

    if not variants:
        return "No variant", variant_texts, gene_texts

    lines: list[str] = []

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
# Conversion to SFT rows
# =========================

def dataframe_to_flat_examples(df: pd.DataFrame) -> list[FlatExample]:
    flat_examples: list[FlatExample] = []

    for row in df.itertuples(index=False):
        text_chunk = normalize_space(row.text)
        entities: list[EntityMention] = list(row.entity_mentions)

        target, variant_texts, gene_texts = build_target(entities)

        # The original tmVar script uses Title + Abstract.
        # This dataset has text chunks, so we keep the same user-message schema
        # by using an empty title and putting the chunk into Abstract.
        title = ""
        abstract = text_chunk

        n_variants = 0 if target == "No variant" else len(target.splitlines())

        flat_examples.append(
            FlatExample(
                example_id=row.example_id,
                title=title,
                abstract=abstract,
                text_chunk=text_chunk,
                target=target,
                n_variants=n_variants,
                n_genes=len(gene_texts),
                variant_texts=" | ".join(variant_texts),
                gene_texts=" | ".join(gene_texts),
                source_split=row.source_split,
            )
        )

    return flat_examples


def flat_example_to_sft_row(ex: FlatExample) -> dict[str, Any]:
    user_content = f"{ex.title}{ex.abstract}".strip()

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": ex.target},
        ],
        "metadata": {
            "example_id": ex.example_id,
            "source": SOURCE_NAME,
            "source_split": ex.source_split,
            "n_variants": ex.n_variants,
            "n_genes": ex.n_genes,
            "variant_texts": ex.variant_texts,
            "gene_texts": ex.gene_texts,
        },
    }


# =========================
# Write utilities
# =========================

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_positive(rows: list[FlatExample]) -> int:
    return sum(1 for row in rows if row.target != "No variant")


def count_negative(rows: list[FlatExample]) -> int:
    return sum(1 for row in rows if row.target == "No variant")


def write_flat_files(flat_examples: list[FlatExample]) -> None:
    if not SAVE_FLAT_INSPECTION_FILES:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dict_rows = [asdict(row) for row in flat_examples]

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

    all_flat_examples: list[FlatExample] = []
    split_stats: dict[str, dict[str, Any]] = {}

    for split_name, input_path in INPUT_SPLITS.items():
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        print(f"Reading {split_name}: {input_path}")

        df = parse_bio_txt_to_dataframe(
            path=input_path,
            split_name=split_name,
        )

        flat_examples = dataframe_to_flat_examples(df)
        sft_rows = [flat_example_to_sft_row(ex) for ex in flat_examples]

        output_path = OUTPUT_JSONL[split_name]
        write_jsonl(output_path, sft_rows)

        all_flat_examples.extend(flat_examples)

        split_stats[split_name] = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "examples": len(flat_examples),
            "positive_examples": count_positive(flat_examples),
            "negative_examples": count_negative(flat_examples),
            "total_variant_lines": sum(ex.n_variants for ex in flat_examples),
        }

        print(f"Wrote {len(sft_rows)} rows to {output_path}")

    write_flat_files(all_flat_examples)

    stats = {
        "source_name": SOURCE_NAME,
        "raw_dir": str(RAW_DIR),
        "output_dir": str(OUTPUT_DIR),
        "input_splits": {
            split_name: str(path)
            for split_name, path in INPUT_SPLITS.items()
        },
        "output_jsonl": {
            split_name: str(path)
            for split_name, path in OUTPUT_JSONL.items()
        },
        "entity_mapping": {
            "Gene": ["B-gene", "I-gene"],
            "Variant": ["B-mutation", "I-mutation"],
        },
        "ignored_entity_types": "All entity types except gene and mutation",
        "missing_gene_value": MISSING_GENE_VALUE,
        "keep_variants_with_unknown_gene": KEEP_VARIANTS_WITH_UNKNOWN_GENE,
        "deduplicate_output_lines": DEDUPLICATE_OUTPUT_LINES,
        "assignment_strategy": "Each variant is assigned to the nearest gene mention in the same chunk by token position.",
        "splits": split_stats,
        "total_examples": len(all_flat_examples),
        "total_positive_examples": count_positive(all_flat_examples),
        "total_negative_examples": count_negative(all_flat_examples),
        "total_variant_lines": sum(ex.n_variants for ex in all_flat_examples),
        "flat_csv": str(FLAT_CSV) if SAVE_FLAT_INSPECTION_FILES else None,
        "flat_jsonl": str(FLAT_JSONL) if SAVE_FLAT_INSPECTION_FILES else None,
    }

    STATS_JSON.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nDone.")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()