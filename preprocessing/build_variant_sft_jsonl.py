# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#!/usr/bin/env python3
"""
Build train/validation/test JSONL files for variant-extraction SFT.

Input CSV format expected:
- PaperTitle: article title
- Abstract: article abstract
- Human: human label
    - "0" means no specific variant -> assistant output "No variant"
    - otherwise comma-separated mentions, for example:
        AR T878A, AR L702H, AR H875Y, AR F877L
        c.6341del
        BRCA1 c.2138C>G
        W1692fs*3, R320W, C2385

Output JSONL format:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Title: ...\nAbstract: ..."},
    {"role": "assistant", "content": "Variant: ..., Gene: ..."}
  ]
}
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


# =========================
# Constants: edit if needed
# =========================

RAW_CSV_PATH = Path("../data/raw/LLM_evaluation_statistics.csv")
OUTPUT_DIR = Path("../data/processed")

TRAIN_JSONL = OUTPUT_DIR / "train.jsonl"
VAL_JSONL = OUTPUT_DIR / "val.jsonl"
TEST_JSONL = OUTPUT_DIR / "test.jsonl"
STATS_JSON = OUTPUT_DIR / "dataset_stats.json"

TITLE_COL = "PaperTitle"
ABSTRACT_COL = "Abstract"
LABEL_COL = "Human"

RANDOM_SEED = 42
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10

# If a variant has no explicit gene and no unique gene can be inferred,
# use this value in "Gene: <...>".
MISSING_GENE_VALUE = "Unknown"

# Gene columns present in your CSV. These are used only as a fallback when
# the Human label gives only the variant, e.g. "c.6341del".
GENE_COLUMNS = [
    "ATM",
    "BRCA1",
    "BRCA2",
    "PALB2",
    "CDK12",
    "CHEK2",
    "PPP2R2A",
    "RAD54L",
    "BRIP1",
    "BARD1",
    "CHEK1",
    "FANCL",
    "RAD51B",
    "RAD51C",
    "RAD51D",
]

# Gene-like tokens that may appear inside Human labels but are not necessarily
# represented by one-hot columns.
EXTRA_GENE_SYMBOLS = [
    "AR",
    "TP53",
    "BRAF",
    "SPOP",
    "SRC",
    "DDR2",
    "HNRNPU",
    "MDM2",
    "NRAS",
    "KRAS",
    "PIK3CA",
    "PTEN",
    "RB1",
]

KNOWN_GENE_SYMBOLS = sorted(
    set(GENE_COLUMNS + EXTRA_GENE_SYMBOLS),
    key=len,
    reverse=True,
)

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
# Parsing helpers
# =========================

def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def is_no_variant(label: str) -> bool:
    label = clean_text(label)
    return label == "" or label == "0" or label.lower() in {"no variant", "none", "nan"}


def normalize_label_text(label: str) -> str:
    label = clean_text(label)

    # Normalize common spacing issues.
    label = re.sub(r"\s+", " ", label)

    # Normalize comma spacing.
    label = re.sub(r"\s*,\s*", ", ", label)

    # Remove trailing punctuation that is not part of a variant.
    label = label.strip(" ;")

    return label


def split_mentions(label: str) -> list[str]:
    """
    Split Human label into candidate mentions.

    Main separator is comma. Also handles simple ' and ' joins when they connect
    two complete gene+variant mentions, e.g. 'SPOP F133 and SPOP F102'.
    """
    label = normalize_label_text(label)
    if is_no_variant(label):
        return []

    raw_parts: list[str] = []

    for part in label.split(","):
        part = part.strip()
        if not part:
            continue

        # Split "SPOP F133 and SPOP F102" into two mentions.
        # Avoid aggressive splitting because "delins" and text fragments can be fragile.
        and_parts = re.split(r"\s+\band\b\s+", part, flags=re.IGNORECASE)
        raw_parts.extend(and_parts)

    mentions = []
    for part in raw_parts:
        part = part.strip()
        part = re.sub(r"^\-+\s*", "", part)
        part = part.strip(" ;")
        if part and not is_no_variant(part):
            mentions.append(part)

    return mentions


def active_genes_from_row(row: pd.Series) -> list[str]:
    genes = []
    for gene in GENE_COLUMNS:
        if gene in row.index:
            value = row[gene]
            try:
                if int(value) == 1:
                    genes.append(gene)
            except Exception:
                pass
    return genes


def infer_gene_from_text(mention: str) -> tuple[str | None, str]:
    """
    Return (gene, variant) if the mention starts with a known gene symbol.

    Handles:
    - "AR T878A"       -> ("AR", "T878A")
    - "BRCA1-T300G"   -> ("BRCA1", "T300G")
    - "BRCA2 6174del" -> ("BRCA2", "6174del")
    """
    m = mention.strip()

    for gene in KNOWN_GENE_SYMBOLS:
        # Gene followed by whitespace, colon, slash, underscore, or hyphen.
        pattern = rf"^{re.escape(gene)}(?:\s+|[:/_-]+)(.+)$"
        match = re.match(pattern, m, flags=re.IGNORECASE)
        if match:
            variant = match.group(1).strip()
            if variant:
                return gene.upper(), variant

    return None, m


def infer_fallback_gene(row: pd.Series, title: str, abstract: str) -> str:
    """
    Infer gene only when reasonably unambiguous.

    Priority:
    1. exactly one active gene column
    2. exactly one known gene symbol appears in title/abstract
    3. Unknown
    """
    active = active_genes_from_row(row)
    if len(active) == 1:
        return active[0]

    text = f"{title} {abstract}"
    present = []
    for gene in KNOWN_GENE_SYMBOLS:
        if re.search(rf"\b{re.escape(gene)}\b", text, flags=re.IGNORECASE):
            present.append(gene.upper())

    present = sorted(set(present))
    if len(present) == 1:
        return present[0]

    return MISSING_GENE_VALUE


def mention_to_output_line(mention: str, row: pd.Series, title: str, abstract: str) -> str:
    gene, variant = infer_gene_from_text(mention)

    if gene is None:
        gene = infer_fallback_gene(row, title, abstract)

    variant = variant.strip()

    return f"Variant: {variant}, Gene: {gene}"


def build_assistant_output(row: pd.Series) -> str:
    label = clean_text(row[LABEL_COL])

    if is_no_variant(label):
        return "No variant"

    title = clean_text(row.get(TITLE_COL, ""))
    abstract = clean_text(row.get(ABSTRACT_COL, ""))

    lines = []
    seen = set()

    for mention in split_mentions(label):
        line = mention_to_output_line(mention, row, title, abstract)

        # Deduplicate while preserving order.
        key = line.lower()
        if key not in seen:
            seen.add(key)
            lines.append(line)

    if not lines:
        return "No variant"

    return "\n".join(lines)


def build_user_content(row: pd.Series) -> str:
    title = clean_text(row.get(TITLE_COL, ""))
    abstract = clean_text(row.get(ABSTRACT_COL, ""))

    return f"Title: {title}\nAbstract: {abstract}"


def build_jsonl_example(row: pd.Series) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_content(row)},
            {"role": "assistant", "content": build_assistant_output(row)},
        ]
    }


# =========================
# Splitting and writing
# =========================

def stratified_split_indices(df: pd.DataFrame) -> tuple[list[int], list[int], list[int]]:
    """
    Stratify by whether Human is no-variant or positive.
    This keeps positives/negatives distributed across train/val/test.
    """
    rng = random.Random(RANDOM_SEED)

    no_variant_indices = []
    positive_indices = []

    for idx, row in df.iterrows():
        label = clean_text(row[LABEL_COL])
        if is_no_variant(label):
            no_variant_indices.append(idx)
        else:
            positive_indices.append(idx)

    def split_group(indices: list[int]) -> tuple[list[int], list[int], list[int]]:
        indices = list(indices)
        rng.shuffle(indices)

        n = len(indices)
        n_train = int(round(n * TRAIN_FRAC))
        n_val = int(round(n * VAL_FRAC))

        train = indices[:n_train]
        val = indices[n_train:n_train + n_val]
        test = indices[n_train + n_val:]

        return train, val, test

    train_n, val_n, test_n = split_group(no_variant_indices)
    train_p, val_p, test_p = split_group(positive_indices)

    train = train_n + train_p
    val = val_n + val_p
    test = test_n + test_p

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def write_jsonl(path: Path, examples: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            count += 1
    return count


def count_positive_examples(df: pd.DataFrame, indices: list[int]) -> int:
    return sum(not is_no_variant(clean_text(df.loc[i, LABEL_COL])) for i in indices)


def main() -> None:
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(f"Input CSV not found: {RAW_CSV_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_CSV_PATH)

    required_cols = {ABSTRACT_COL, LABEL_COL}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if TITLE_COL not in df.columns:
        print(f"Warning: title column '{TITLE_COL}' not found. Titles will be empty.")

    # Drop rows with empty abstract; labels without input text are not useful for SFT.
    before = len(df)
    df = df[df[ABSTRACT_COL].notna() & df[ABSTRACT_COL].astype(str).str.strip().ne("")]
    df = df.reset_index(drop=True)
    dropped_empty_abstracts = before - len(df)

    train_idx, val_idx, test_idx = stratified_split_indices(df)

    train_examples = [build_jsonl_example(df.loc[i]) for i in train_idx]
    val_examples = [build_jsonl_example(df.loc[i]) for i in val_idx]
    test_examples = [build_jsonl_example(df.loc[i]) for i in test_idx]

    n_train = write_jsonl(TRAIN_JSONL, train_examples)
    n_val = write_jsonl(VAL_JSONL, val_examples)
    n_test = write_jsonl(TEST_JSONL, test_examples)

    stats = {
        "raw_csv_path": str(RAW_CSV_PATH),
        "output_dir": str(OUTPUT_DIR),
        "dropped_empty_abstracts": dropped_empty_abstracts,
        "total_examples": n_train + n_val + n_test,
        "train_examples": n_train,
        "val_examples": n_val,
        "test_examples": n_test,
        "train_positive_examples": count_positive_examples(df, train_idx),
        "val_positive_examples": count_positive_examples(df, val_idx),
        "test_positive_examples": count_positive_examples(df, test_idx),
        "missing_gene_value": MISSING_GENE_VALUE,
        "random_seed": RANDOM_SEED,
    }

    STATS_JSON.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Wrote: {TRAIN_JSONL}")
    print(f"Wrote: {VAL_JSONL}")
    print(f"Wrote: {TEST_JSONL}")
    print(f"Wrote: {STATS_JSON}")


if __name__ == "__main__":
    main()
