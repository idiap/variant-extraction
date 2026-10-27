# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#!/usr/bin/env python3
"""
build_bronco_sft_jsonl.py

Build a conversational SFT JSONL dataset from BRONCO full-text articles and
variant annotations.

Data source
============================
The BRONCO source contains 108 full text articles manually annotated for genomic variants:
BRONCO: Biomedical entity Relation ONcology COrpus for extracting gene-variant-disease-drug relations
https://doi.org/10.1093/database/baw043
http://infos.korea.ac.kr/bronco/

Chunking strategy
============================

The BRONCO source contains long full-text biomedical articles, while the variant
annotation file provides article-level character offsets for each annotated
variant mention. Directly using complete articles as model inputs is undesirable
because full texts are long, heterogeneous, and contain large regions irrelevant
to variant extraction. This script therefore constructs shorter training
instances using an annotation-centered chunking strategy.

Positive chunks are centered on annotated variant mentions rather than produced
by blind fixed-size segmentation. For each article, variant annotations are
sorted by character offset and grouped into local clusters. A cluster is expanded
with a configurable amount of left and right context, and nearby annotations are
merged into the same chunk when they are sufficiently close and the resulting
chunk does not exceed a maximum character length. This ensures that every
positive chunk contains at least one gold variant mention and preserves the
local textual context needed to extract the variant and its associated gene,
while avoiding full-article inputs.

Negative chunks are sampled from annotation-free regions of the same articles.
A candidate negative chunk is accepted only if it does not overlap any annotated
variant span. This provides the model with examples where no variant should be
returned, while preventing false-negative training instances caused by hidden
overlap with gold annotations.

The resulting examples are split into train, validation, and test sets using
stratified sampling over the binary positive/negative label, so that the class
proportion is approximately maintained in each split. Splitting is performed at
the chunk level by default. For stricter article-level separation, set
SPLIT_BY_ARTICLE = True.

Output format
=============

Each JSONL row follows the same conversational SFT format used for the other
data sources:

{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Title: ...\\nAbstract: ..."},
    {"role": "assistant", "content": "Variant: <mutation>, Gene: <gene>"}
  ],
  "metadata": {...}
}

For negative examples, the assistant content is exactly:

No variant
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


import pandas as pd


# =========================
# Paths
# =========================

RAW_DIR = Path("../data/raw/BRONCO")

FULLTEXT_PATH = RAW_DIR / "BRONCO_FullText_Tabbed_20150602.txt"
ANNOTATION_PATH = RAW_DIR / "BRONCO_MAPPED_final_20150914.txt_ExtractedMut.txt"

OUTPUT_DIR = Path("../data/processed_bronco")

TRAIN_JSONL = OUTPUT_DIR / "train.jsonl"
VAL_JSONL = OUTPUT_DIR / "val.jsonl"
TEST_JSONL = OUTPUT_DIR / "test.jsonl"

STATS_JSON = OUTPUT_DIR / "bronco_dataset_stats.json"
FLAT_CSV = OUTPUT_DIR / "bronco_flat_examples.csv"
FLAT_JSONL = OUTPUT_DIR / "bronco_flat_examples.jsonl"
CHUNKS_CSV = OUTPUT_DIR / "bronco_chunks.csv"
CHUNKS_JSONL = OUTPUT_DIR / "bronco_chunks.jsonl"


# =========================
# Dataset constants
# =========================

SOURCE_NAME = "bronco"

RANDOM_SEED = 42

TRAIN_FRACTION = 0.80
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10

# False = stratified chunk-level split.
# True = article-level split, preventing chunks from the same article appearing
# in multiple splits. This is stricter but may preserve class ratios less exactly.
SPLIT_BY_ARTICLE = True

# Positive chunking parameters.
MAX_CHARS = 1800
LEFT_CONTEXT = 500
RIGHT_CONTEXT = 500
MERGE_GAP = 700

# Boundary adjustment.
# Chunks are initially selected by character offsets, then expanded to nearby
# sentence-like boundaries without exceeding the full article.
BOUNDARY_SEARCH_CHARS = 250

# Negative chunking parameters.
NEGATIVE_CHUNK_CHARS = 1800
NEGATIVE_STRIDE_CHARS = 1800

# The number of negative chunks retained before the train/val/test split.
# None means retain all valid negative candidates.
# A float means retain that many negatives per positive, e.g. 1.0 = 1:1.
MAX_NEGATIVES_PER_POSITIVE: float | None = 0.3

SHUFFLE_OUTPUT_ROWS = True
SAVE_FLAT_INSPECTION_FILES = True

MISSING_GENE_VALUE = "Unknown"
DEDUPLICATE_TARGET_LINES = True

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
class Article:
    pmcid: str
    title: str
    body: str
    text: str


@dataclass
class VariantAnnotation:
    pmcid: str
    kind: str
    start: int
    end: int
    mention: str
    gene: str
    mutation_class: str
    mutation_type: str
    mutation_components: str
    normalized_mutation: str
    disease: str
    drugs: str
    cell_lines: str


@dataclass
class ChunkExample:
    example_id: str
    pmcid: str
    title: str
    text_chunk: str
    chunk_start: int
    chunk_end: int
    target: str
    label: str
    n_variants: int
    variant_texts: str
    gene_texts: str
    annotation_ids: str
    source: str


# =========================
# Basic utilities
# =========================

def normalize_space(value: Any) -> str:
    value = "" if value is None else str(value)
    return re.sub(r"\s+", " ", value).strip()


def clean_text(value: Any) -> str:
    """
    Normalize whitespace while preserving character offset stability as much as
    possible after reading the raw file.

    Important:
        Do not perform aggressive Unicode normalization or punctuation rewriting
        before slicing by offsets. The annotation offsets are expected to refer
        to the raw article string as distributed.
    """
    value = "" if value is None else str(value)
    return value.replace("\r\n", "\n").replace("\r", "\n")


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_int(value: Any, *, field_name: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"Could not parse integer field {field_name}: {value}") from exc
    
TEXT_ENCODINGS_TO_TRY = [
    "utf-8",
    "utf-8-sig",
    "latin-1",
    "cp1252",
]


def open_text_with_fallback(path: Path):
    """
    Open a text file using a small sequence of likely encodings.

    BRONCO contains legacy/non-UTF-8 bytes in the distributed text files.
    We first try UTF-8 for normal modern text, then fall back to encodings that
    can decode byte-preserving Western biomedical text.

    latin-1 never fails for byte values 0-255, so it is kept as a safe fallback.
    """
    last_error = None

    for encoding in TEXT_ENCODINGS_TO_TRY:
        try:
            f = path.open("r", encoding=encoding)
            # Force an initial decode attempt so errors happen here, not later.
            f.peek if False else None
            f.read(4096)
            f.seek(0)
            print(f"Opened {path} with encoding: {encoding}")
            return f
        except UnicodeDecodeError as exc:
            last_error = exc
            try:
                f.close()
            except Exception:
                pass
            continue

    # Final defensive fallback: preserve maximum content rather than crashing.
    print(
        f"Warning: falling back to utf-8 with errors='replace' for {path}. "
        f"Last decode error was: {last_error}"
    )
    return path.open("r", encoding="utf-8", errors="replace")


# =========================
# Reading input files
# =========================

def read_fulltext_file(path: Path) -> dict[str, Article]:
    """
    Read BRONCO_FullText_Tabbed_20150602.txt.

    Expected row structure:
        PMCID<TAB>Title<TAB>Full text

    The model input text is constructed as:
        title + " " + body

    This convention must match the coordinate system used by the annotation
    offsets. The script validates offsets later and reports mismatches.
    """
    if not path.exists():
        raise FileNotFoundError(f"Full-text file not found: {path}")

    articles: dict[str, Article] = {}

    with open_text_with_fallback(path) as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")

            if not line.strip():
                continue

            parts = line.split("\t", maxsplit=2)

            if len(parts) < 3:
                raise ValueError(
                    f"Malformed full-text row at line {line_number}. "
                    f"Expected at least 3 tab-separated fields."
                )

            pmcid = normalize_space(parts[0])
            title = clean_text(parts[1]).strip()
            body = clean_text(parts[2]).strip()
            text = f"{title} {body}".strip()

            if not pmcid:
                raise ValueError(f"Missing PMCID at full-text line {line_number}")

            articles[pmcid] = Article(
                pmcid=pmcid,
                title=title,
                body=body,
                text=text,
            )

    return articles


def parse_annotation_row(parts: list[str], line_number: int) -> VariantAnnotation:
    """
    Read BRONCO_MAPPED_final_20150914.txt_ExtractedMut.txt.

    Observed row structure:
        PMCID
        Var
        start
        end
        mention
        gene
        mutation_class
        mutation_type
        mutation_components
        normalized_mutation
        disease
        drugs
        cell_lines

    Extra columns, if present, are ignored.
    Missing trailing columns are filled with empty strings.
    """
    if len(parts) < 6:
        raise ValueError(
            f"Malformed annotation row at line {line_number}. "
            f"Expected at least 6 tab-separated fields."
        )

    padded = parts + [""] * max(0, 13 - len(parts))

    return VariantAnnotation(
        pmcid=normalize_space(padded[0]),
        kind=normalize_space(padded[1]),
        start=safe_int(padded[2], field_name="start"),
        end=safe_int(padded[3], field_name="end"),
        mention=normalize_space(padded[4]),
        gene=normalize_space(padded[5]),
        mutation_class=normalize_space(padded[6]),
        mutation_type=normalize_space(padded[7]),
        mutation_components=normalize_space(padded[8]),
        normalized_mutation=normalize_space(padded[9]),
        disease=normalize_space(padded[10]),
        drugs=normalize_space(padded[11]),
        cell_lines=normalize_space(padded[12]),
    )


def read_annotation_file(path: Path) -> list[VariantAnnotation]:
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    annotations: list[VariantAnnotation] = []

    with open_text_with_fallback(path) as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")

            if not line.strip():
                continue

            parts = line.split("\t")

            # Skip a header if one is present.
            if line_number == 1:
                first = normalize_space(parts[0]).lower()
                third = normalize_space(parts[2]).lower() if len(parts) > 2 else ""
                if first in {"pmcid", "pmc", "article_id", "id"} or third in {"start", "begin"}:
                    continue

            annotations.append(parse_annotation_row(parts, line_number))

    return annotations


# =========================
# Offset validation
# =========================

def validate_annotation_offsets(
    articles: dict[str, Article],
    annotations: list[VariantAnnotation],
) -> dict[str, Any]:
    """
    Validate whether text[start:end] matches the annotated mention.

    Some corpora have occasional encoding drift. We do not fail on mismatch by
    default, but we report counts and examples in the stats file.
    """
    checked = 0
    exact_matches = 0
    normalized_matches = 0
    mismatches: list[dict[str, Any]] = []
    missing_articles = 0
    out_of_bounds = 0

    for ann in annotations:
        article = articles.get(ann.pmcid)

        if article is None:
            missing_articles += 1
            continue

        text = article.text

        if ann.start < 0 or ann.end > len(text) or ann.start >= ann.end:
            out_of_bounds += 1
            if len(mismatches) < 20:
                mismatches.append(
                    {
                        "pmcid": ann.pmcid,
                        "start": ann.start,
                        "end": ann.end,
                        "mention": ann.mention,
                        "observed": None,
                        "reason": "out_of_bounds",
                    }
                )
            continue

        checked += 1
        observed = text[ann.start:ann.end]

        if observed == ann.mention:
            exact_matches += 1
        elif normalize_space(observed) == normalize_space(ann.mention):
            normalized_matches += 1
        else:
            if len(mismatches) < 20:
                mismatches.append(
                    {
                        "pmcid": ann.pmcid,
                        "start": ann.start,
                        "end": ann.end,
                        "mention": ann.mention,
                        "observed": observed,
                        "reason": "text_span_mismatch",
                    }
                )

    return {
        "annotations_total": len(annotations),
        "annotations_checked": checked,
        "exact_offset_matches": exact_matches,
        "normalized_offset_matches": normalized_matches,
        "offset_mismatches": checked - exact_matches - normalized_matches,
        "missing_article_annotations": missing_articles,
        "out_of_bounds_annotations": out_of_bounds,
        "example_mismatches": mismatches,
    }


# =========================
# Chunk boundary helpers
# =========================

def find_left_boundary(text: str, start: int) -> int:
    """
    Expand left to a nearby sentence-like boundary when possible.
    """
    search_start = max(0, start - BOUNDARY_SEARCH_CHARS)
    window = text[search_start:start]

    candidates = [
        window.rfind(". "),
        window.rfind("? "),
        window.rfind("! "),
        window.rfind("\n"),
    ]

    best = max(candidates)

    if best == -1:
        return start

    return search_start + best + 1


def find_right_boundary(text: str, end: int) -> int:
    """
    Expand right to a nearby sentence-like boundary when possible.
    """
    search_end = min(len(text), end + BOUNDARY_SEARCH_CHARS)
    window = text[end:search_end]

    candidates = []

    for pattern in [". ", "? ", "! ", "\n"]:
        idx = window.find(pattern)
        if idx != -1:
            candidates.append(idx + len(pattern))

    if not candidates:
        return end

    return end + min(candidates)


def overlaps_any_span(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    for span_start, span_end in spans:
        if start < span_end and end > span_start:
            return True

    return False


def annotation_id(ann: VariantAnnotation) -> str:
    return f"{ann.pmcid}:{ann.start}-{ann.end}:{ann.mention}:{ann.gene}"


def resolve_annotation_offsets_for_article(
    article: Article,
    annotations: list[VariantAnnotation],
    search_window: int = 3000,
) -> tuple[list[VariantAnnotation], list[VariantAnnotation]]:
    """
    Return annotations whose offsets are valid in article.text.

    If the original offset does not match the mention, try to find the mention
    near the original offset. If unresolved, return it separately so the article
    can be treated as unsafe for negative sampling.
    """
    resolved = []
    unresolved = []

    text = article.text

    for ann in annotations:
        mention = ann.mention

        if not mention:
            unresolved.append(ann)
            continue

        # Case 1: original offset is valid.
        if 0 <= ann.start < ann.end <= len(text):
            observed = text[ann.start:ann.end]
            if observed == mention or normalize_space(observed) == normalize_space(mention):
                resolved.append(ann)
                continue

        # Case 2: try to find mention near the provided offset.
        left = max(0, ann.start - search_window)
        right = min(len(text), ann.end + search_window)
        window = text[left:right]

        local_idx = window.find(mention)

        if local_idx != -1:
            new_start = left + local_idx
            new_end = new_start + len(mention)

            resolved.append(
                VariantAnnotation(
                    pmcid=ann.pmcid,
                    kind=ann.kind,
                    start=new_start,
                    end=new_end,
                    mention=ann.mention,
                    gene=ann.gene,
                    mutation_class=ann.mutation_class,
                    mutation_type=ann.mutation_type,
                    mutation_components=ann.mutation_components,
                    normalized_mutation=ann.normalized_mutation,
                    disease=ann.disease,
                    drugs=ann.drugs,
                    cell_lines=ann.cell_lines,
                )
            )
            continue

        # Case 3: if mention appears exactly once globally, use that.
        all_hits = [m.start() for m in re.finditer(re.escape(mention), text)]

        if len(all_hits) == 1:
            new_start = all_hits[0]
            new_end = new_start + len(mention)

            resolved.append(
                VariantAnnotation(
                    pmcid=ann.pmcid,
                    kind=ann.kind,
                    start=new_start,
                    end=new_end,
                    mention=ann.mention,
                    gene=ann.gene,
                    mutation_class=ann.mutation_class,
                    mutation_type=ann.mutation_type,
                    mutation_components=ann.mutation_components,
                    normalized_mutation=ann.normalized_mutation,
                    disease=ann.disease,
                    drugs=ann.drugs,
                    cell_lines=ann.cell_lines,
                )
            )
            continue

        unresolved.append(ann)

    return resolved, unresolved


# =========================
# Positive chunking
# =========================

def group_annotations_for_positive_chunks(
    annotations: list[VariantAnnotation],
) -> list[list[VariantAnnotation]]:
    """
    Group nearby variant annotations into local clusters.

    A new annotation is merged into the current group if:
        1. the gap from the previous annotation is <= MERGE_GAP
        2. the context-expanded group length remains <= MAX_CHARS

    This reduces redundant one-mention chunks and preserves local co-mentions.
    """
    if not annotations:
        return []

    annotations = sorted(annotations, key=lambda ann: (ann.start, ann.end))

    groups: list[list[VariantAnnotation]] = []
    current: list[VariantAnnotation] = [annotations[0]]

    for ann in annotations[1:]:
        current_start = min(a.start for a in current)
        candidate_end = max(ann.end, max(a.end for a in current))

        previous_end = max(a.end for a in current)
        gap = ann.start - previous_end

        expanded_start = max(0, current_start - LEFT_CONTEXT)
        expanded_end = candidate_end + RIGHT_CONTEXT
        expanded_len = expanded_end - expanded_start

        if gap <= MERGE_GAP and expanded_len <= MAX_CHARS:
            current.append(ann)
        else:
            groups.append(current)
            current = [ann]

    groups.append(current)

    return groups


def build_positive_chunks_for_article(
    article: Article,
    annotations: list[VariantAnnotation],
) -> list[ChunkExample]:
    chunks: list[ChunkExample] = []

    valid_annotations = [
        ann for ann in annotations
        if 0 <= ann.start < ann.end <= len(article.text)
    ]

    groups = group_annotations_for_positive_chunks(valid_annotations)

    for idx, group in enumerate(groups):
        raw_start = max(0, min(ann.start for ann in group) - LEFT_CONTEXT)
        raw_end = min(len(article.text), max(ann.end for ann in group) + RIGHT_CONTEXT)

        chunk_start = find_left_boundary(article.text, raw_start)
        chunk_end = find_right_boundary(article.text, raw_end)

        # Ensure all annotations remain inside after boundary adjustment.
        chunk_start = min(chunk_start, min(ann.start for ann in group))
        chunk_end = max(chunk_end, max(ann.end for ann in group))

        # If boundary adjustment made the chunk too large, fall back to raw bounds.
        if chunk_end - chunk_start > MAX_CHARS + BOUNDARY_SEARCH_CHARS:
            chunk_start = raw_start
            chunk_end = raw_end

        anns_inside = [
            ann for ann in valid_annotations
            if ann.start >= chunk_start and ann.end <= chunk_end
        ]

        target, variant_texts, gene_texts = build_target(anns_inside)

        if target == "No variant":
            continue

        text_chunk = article.text[chunk_start:chunk_end].strip()

        if not text_chunk:
            continue

        chunks.append(
            ChunkExample(
                example_id=f"{article.pmcid}_pos_{idx:05d}",
                pmcid=article.pmcid,
                title=article.title,
                text_chunk=text_chunk,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                target=target,
                label="positive",
                n_variants=len(target.splitlines()),
                variant_texts=" | ".join(variant_texts),
                gene_texts=" | ".join(gene_texts),
                annotation_ids=" | ".join(annotation_id(ann) for ann in anns_inside),
                source=SOURCE_NAME,
            )
        )

    return chunks


# =========================
# Negative chunking
# =========================

def build_negative_chunks_for_article(
    article: Article,
    annotations: list[VariantAnnotation],
) -> list[ChunkExample]:
    """
    Build annotation-free chunks.

    A candidate negative window is accepted only if:
      1. it does not overlap any valid annotated variant span;
      2. after boundary expansion, it still does not overlap any annotated span;
      3. the final text chunk does not contain any annotated variant mention string.

    The mention-string check is a defensive safeguard for BRONCO offset drift:
    if an annotation offset is misaligned with article.text, span-overlap checks
    alone can miss a true annotated variant and incorrectly create a negative
    chunk containing that variant.
    """

    spans = [
        (ann.start, ann.end)
        for ann in annotations
        if 0 <= ann.start < ann.end <= len(article.text)
    ]

    # Defensive mention-level blacklist.
    # Avoid very short mentions because strings like "A", "T", "G", "C", "del",
    # or "ins" can cause many false rejections.
    annotation_mentions = []
    for ann in annotations:
        mention = normalize_space(ann.mention)
        if not mention:
            continue

        # Keep only reasonably specific mentions.
        if len(mention) < 4:
            continue

        annotation_mentions.append(mention.casefold())

    annotation_mentions = dedupe_keep_order(annotation_mentions)

    chunks: list[ChunkExample] = []

    idx = 0
    start = 0

    while start < len(article.text):
        end = min(len(article.text), start + NEGATIVE_CHUNK_CHARS)

        if end - start < 200:
            break

        # First span-overlap check on the raw candidate window.
        if overlaps_any_span(start, end, spans):
            start += NEGATIVE_STRIDE_CHARS
            continue

        chunk_start = find_left_boundary(article.text, start)
        chunk_end = find_right_boundary(article.text, end)

        # Re-check after boundary expansion.
        if overlaps_any_span(chunk_start, chunk_end, spans):
            start += NEGATIVE_STRIDE_CHARS
            continue

        text_chunk = article.text[chunk_start:chunk_end].strip()

        if not text_chunk:
            start += NEGATIVE_STRIDE_CHARS
            continue

        text_chunk_norm = normalize_space(text_chunk).casefold()

        # Defensive check against offset drift:
        # reject the negative chunk if it literally contains an annotated mention.
        contains_annotated_mention = any(
            mention in text_chunk_norm
            for mention in annotation_mentions
        )

        if contains_annotated_mention:
            start += NEGATIVE_STRIDE_CHARS
            continue

        chunks.append(
            ChunkExample(
                example_id=f"{article.pmcid}_neg_{idx:05d}",
                pmcid=article.pmcid,
                title=article.title,
                text_chunk=text_chunk,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                target="No variant",
                label="negative",
                n_variants=0,
                variant_texts="",
                gene_texts="",
                annotation_ids="",
                source=SOURCE_NAME,
            )
        )
        idx += 1

        start += NEGATIVE_STRIDE_CHARS

    return chunks


def downsample_negatives_for_split(
    chunks: list[ChunkExample],
    rng: random.Random,
    max_negatives_per_positive: float = MAX_NEGATIVES_PER_POSITIVE,
) -> list[ChunkExample]:
    positive_chunks = [chunk for chunk in chunks if chunk.target != "No variant"]
    negative_chunks = [chunk for chunk in chunks if chunk.target == "No variant"]

    n_positive = len(positive_chunks)
    n_negative = len(negative_chunks)

    if n_positive == 0:
        # There is no positive anchor, so a negative:positive ratio is undefined.
        # To strictly preserve the requested ratio, keep zero negatives.
        print(
            "Warning: split has zero positive chunks; "
            "dropping negatives because negative:positive ratio is undefined."
        )
        return []

    max_negatives = int(round(n_positive * max_negatives_per_positive))
    max_negatives = min(max_negatives, n_negative)

    sampled_negatives = rng.sample(negative_chunks, k=max_negatives)

    balanced_chunks = positive_chunks + sampled_negatives
    rng.shuffle(balanced_chunks)

    return balanced_chunks


# =========================
# Target construction
# =========================

def canonical_variant_for_output(ann: VariantAnnotation) -> str:
    """
    Prefer the surface mention for extraction-style supervision.

    The normalized HGVS-like field is retained in metadata through annotation IDs
    and flat files, but the assistant target uses the text mention the model is
    expected to extract from the input.
    """
    return normalize_space(ann.mention)


def canonical_gene_for_output(ann: VariantAnnotation) -> str:
    gene = normalize_space(ann.gene)
    return gene if gene and gene != "-" else MISSING_GENE_VALUE


def build_target(
    annotations: list[VariantAnnotation],
) -> tuple[str, list[str], list[str]]:
    lines: list[str] = []
    variant_texts: list[str] = []
    gene_texts: list[str] = []

    for ann in sorted(annotations, key=lambda x: (x.start, x.end)):
        variant = canonical_variant_for_output(ann)
        gene = canonical_gene_for_output(ann)

        if not variant:
            continue

        lines.append(f"Variant: {variant}, Gene: {gene}")
        variant_texts.append(variant)
        gene_texts.append(gene)

    if DEDUPLICATE_TARGET_LINES:
        lines = dedupe_keep_order(lines)

    variant_texts = dedupe_keep_order(variant_texts)
    gene_texts = dedupe_keep_order(gene_texts)

    if not lines:
        return "No variant", variant_texts, gene_texts

    return "\n".join(lines), variant_texts, gene_texts


# =========================
# Conversion to SFT
# =========================

def chunk_to_sft_row(chunk: ChunkExample) -> dict[str, Any]:
    user_content = f"{chunk.text_chunk}".strip()

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": chunk.target},
        ],
        "metadata": {
            "example_id": chunk.example_id,
            "pmcid": chunk.pmcid,
            "source": chunk.source,
            "label": chunk.label,
            "chunk_start": chunk.chunk_start,
            "chunk_end": chunk.chunk_end,
            "n_variants": chunk.n_variants,
            "variant_texts": chunk.variant_texts,
            "gene_texts": chunk.gene_texts,
            "annotation_ids": chunk.annotation_ids,
        },
    }


# =========================
# Stratified split
# =========================

def split_list(
    items: list[Any],
    rng: random.Random,
) -> tuple[list[Any], list[Any], list[Any]]:
    if abs((TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION) - 1.0) > 1e-8:
        raise ValueError("TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION must equal 1.")

    items = list(items)
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * TRAIN_FRACTION)
    n_val = int(n * VAL_FRACTION)

    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]

    return train, val, test


def stratified_chunk_split(
    chunks: list[ChunkExample],
    rng: random.Random,
) -> tuple[list[ChunkExample], list[ChunkExample], list[ChunkExample]]:
    positives = [chunk for chunk in chunks if chunk.label == "positive"]
    negatives = [chunk for chunk in chunks if chunk.label == "negative"]

    train_pos, val_pos, test_pos = split_list(positives, rng)
    train_neg, val_neg, test_neg = split_list(negatives, rng)

    train = train_pos + train_neg
    val = val_pos + val_neg
    test = test_pos + test_neg

    if SHUFFLE_OUTPUT_ROWS:
        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)

    return train, val, test


def stratified_article_split(
    chunks: list[ChunkExample],
    rng: random.Random,
) -> tuple[list[ChunkExample], list[ChunkExample], list[ChunkExample]]:
    """
    Article-level split.

    This prevents leakage across splits at the article level. It approximates
    class balance by stratifying articles according to whether they contain at
    least one positive chunk.
    """
    chunks_by_article: dict[str, list[ChunkExample]] = defaultdict(list)

    for chunk in chunks:
        chunks_by_article[chunk.pmcid].append(chunk)

    positive_articles = []
    negative_articles = []

    for pmcid, article_chunks in chunks_by_article.items():
        if any(chunk.label == "positive" for chunk in article_chunks):
            positive_articles.append(pmcid)
        else:
            negative_articles.append(pmcid)

    train_pos_ids, val_pos_ids, test_pos_ids = split_list(positive_articles, rng)
    train_neg_ids, val_neg_ids, test_neg_ids = split_list(negative_articles, rng)

    train_ids = set(train_pos_ids + train_neg_ids)
    val_ids = set(val_pos_ids + val_neg_ids)
    test_ids = set(test_pos_ids + test_neg_ids)

    train = [chunk for chunk in chunks if chunk.pmcid in train_ids]
    val = [chunk for chunk in chunks if chunk.pmcid in val_ids]
    test = [chunk for chunk in chunks if chunk.pmcid in test_ids]

    if SHUFFLE_OUTPUT_ROWS:
        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)

    return train, val, test


# =========================
# Stats and inspection files
# =========================

def count_positive(chunks: list[ChunkExample]) -> int:
    return sum(1 for chunk in chunks if chunk.label == "positive")


def count_negative(chunks: list[ChunkExample]) -> int:
    return sum(1 for chunk in chunks if chunk.label == "negative")


def count_variant_lines(chunks: list[ChunkExample]) -> int:
    return sum(chunk.n_variants for chunk in chunks)


def split_stats(name: str, chunks: list[ChunkExample], path: Path) -> dict[str, Any]:
    total = len(chunks)
    pos = count_positive(chunks)
    neg = count_negative(chunks)

    return {
        "split": name,
        "path": str(path),
        "examples": total,
        "positive_examples": pos,
        "negative_examples": neg,
        "positive_fraction": pos / total if total else 0,
        "negative_fraction": neg / total if total else 0,
        "variant_lines": count_variant_lines(chunks),
        "articles": len({chunk.pmcid for chunk in chunks}),
    }


def write_inspection_files(chunks: list[ChunkExample]) -> None:
    if not SAVE_FLAT_INSPECTION_FILES:
        return

    dict_rows = [asdict(chunk) for chunk in chunks]

    pd.DataFrame(dict_rows).to_csv(
        FLAT_CSV,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    pd.DataFrame(dict_rows).to_csv(
        CHUNKS_CSV,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    with FLAT_JSONL.open("w", encoding="utf-8") as f:
        for row in dict_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with CHUNKS_JSONL.open("w", encoding="utf-8") as f:
        for row in dict_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================
# Main
# =========================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(RANDOM_SEED)

    print(f"Reading full texts: {FULLTEXT_PATH}")
    articles = read_fulltext_file(FULLTEXT_PATH)

    print(f"Reading annotations: {ANNOTATION_PATH}")
    annotations = read_annotation_file(ANNOTATION_PATH)

    annotations_by_article: dict[str, list[VariantAnnotation]] = defaultdict(list)

    for ann in annotations:
        annotations_by_article[ann.pmcid].append(ann)

    print("Validating annotation offsets...")
    offset_validation = validate_annotation_offsets(articles, annotations)

    all_chunks_before_downsampling: list[ChunkExample] = []

    missing_article_pmids = sorted(
        set(annotations_by_article) - set(articles)
    )

    unresolved_annotation_records = []

    for pmcid, article in articles.items():
        article_annotations = annotations_by_article.get(pmcid, [])

        resolved_annotations, unresolved_annotations = resolve_annotation_offsets_for_article(
            article=article,
            annotations=article_annotations,
        )

        if unresolved_annotations:
            unresolved_annotation_records.extend(
                {
                    "pmcid": ann.pmcid,
                    "start": ann.start,
                    "end": ann.end,
                    "mention": ann.mention,
                    "gene": ann.gene,
                }
                for ann in unresolved_annotations
            )

        positive_chunks = build_positive_chunks_for_article(
            article=article,
            annotations=resolved_annotations,
        )

        # Conservative safety rule:
        # If any annotation could not be resolved, do not create negative chunks
        # for this article, because we cannot guarantee annotation-free windows.
        if unresolved_annotations:
            negative_chunks = []
        else:
            negative_chunks = build_negative_chunks_for_article(
                article=article,
                annotations=resolved_annotations,
            )

        all_chunks_before_downsampling.extend(positive_chunks)
        all_chunks_before_downsampling.extend(negative_chunks)

    if SPLIT_BY_ARTICLE:
        train_chunks_raw, val_chunks_raw, test_chunks_raw = stratified_article_split(
            all_chunks_before_downsampling,
            rng,
        )
        split_strategy = "article_level_stratified_then_per_split_negative_downsampling"
    else:
        train_chunks_raw, val_chunks_raw, test_chunks_raw = stratified_chunk_split(
            all_chunks_before_downsampling,
            rng,
        )
        split_strategy = "chunk_level_stratified_then_per_split_negative_downsampling"


    train_chunks = downsample_negatives_for_split(
        train_chunks_raw,
        rng,
        max_negatives_per_positive=MAX_NEGATIVES_PER_POSITIVE,
    )

    val_chunks = downsample_negatives_for_split(
        val_chunks_raw,
        rng,
        max_negatives_per_positive=MAX_NEGATIVES_PER_POSITIVE,
    )

    test_chunks = downsample_negatives_for_split(
        test_chunks_raw,
        rng,
        max_negatives_per_positive=MAX_NEGATIVES_PER_POSITIVE,
    )

    all_chunks = train_chunks + val_chunks + test_chunks

    train_rows = [chunk_to_sft_row(chunk) for chunk in train_chunks]
    val_rows = [chunk_to_sft_row(chunk) for chunk in val_chunks]
    test_rows = [chunk_to_sft_row(chunk) for chunk in test_chunks]

    write_jsonl(TRAIN_JSONL, train_rows)
    write_jsonl(VAL_JSONL, val_rows)
    write_jsonl(TEST_JSONL, test_rows)

    write_inspection_files(all_chunks)

    stats = {
        "source_name": SOURCE_NAME,
        "fulltext_path": str(FULLTEXT_PATH),
        "annotation_path": str(ANNOTATION_PATH),
        "output_dir": str(OUTPUT_DIR),
        "random_seed": RANDOM_SEED,
        "split_strategy": split_strategy,
        "split_by_article": SPLIT_BY_ARTICLE,
        "train_fraction": TRAIN_FRACTION,
        "validation_fraction": VAL_FRACTION,
        "test_fraction": TEST_FRACTION,
        "chunking_strategy": {
            "description": (
                "Positive chunks are centered on clusters of gold variant annotations. "
                "Nearby annotations are merged when they are within MERGE_GAP characters "
                "and the context-expanded chunk remains below MAX_CHARS. Negative chunks "
                "are sampled from annotation-free full-text regions and are rejected if "
                "they overlap any gold variant span."
            ),
            "max_chars": MAX_CHARS,
            "left_context": LEFT_CONTEXT,
            "right_context": RIGHT_CONTEXT,
            "merge_gap": MERGE_GAP,
            "boundary_search_chars": BOUNDARY_SEARCH_CHARS,
            "negative_chunk_chars": NEGATIVE_CHUNK_CHARS,
            "negative_stride_chars": NEGATIVE_STRIDE_CHARS,
            "max_negatives_per_positive": MAX_NEGATIVES_PER_POSITIVE,
        },
        "offset_validation": offset_validation,
        "input_counts": {
            "articles": len(articles),
            "annotation_rows": len(annotations),
            "articles_with_annotations": len(annotations_by_article),
            "annotation_pmids_missing_from_fulltext": len(missing_article_pmids),
            "example_missing_article_pmids": missing_article_pmids[:20],
        },
        "chunk_counts_before_negative_downsampling": {
            "total_chunks": len(all_chunks_before_downsampling),
            "positive_chunks": count_positive(all_chunks_before_downsampling),
            "negative_chunks": count_negative(all_chunks_before_downsampling),
            "variant_lines": count_variant_lines(all_chunks_before_downsampling),
        },
        "chunk_counts_after_negative_downsampling": {
            "total_chunks": len(all_chunks),
            "positive_chunks": count_positive(all_chunks),
            "negative_chunks": count_negative(all_chunks),
            "variant_lines": count_variant_lines(all_chunks),
        },
        "split_counts_before_negative_downsampling": {
            "train": split_stats("train_raw", train_chunks_raw, TRAIN_JSONL),
            "val": split_stats("val_raw", val_chunks_raw, VAL_JSONL),
            "test": split_stats("test_raw", test_chunks_raw, TEST_JSONL),
        },
        "splits": {
            "train": split_stats("train", train_chunks, TRAIN_JSONL),
            "val": split_stats("val", val_chunks, VAL_JSONL),
            "test": split_stats("test", test_chunks, TEST_JSONL),
        },
        "output_files": {
            "train_jsonl": str(TRAIN_JSONL),
            "val_jsonl": str(VAL_JSONL),
            "test_jsonl": str(TEST_JSONL),
            "stats_json": str(STATS_JSON),
            "flat_csv": str(FLAT_CSV) if SAVE_FLAT_INSPECTION_FILES else None,
            "flat_jsonl": str(FLAT_JSONL) if SAVE_FLAT_INSPECTION_FILES else None,
            "chunks_csv": str(CHUNKS_CSV) if SAVE_FLAT_INSPECTION_FILES else None,
            "chunks_jsonl": str(CHUNKS_JSONL) if SAVE_FLAT_INSPECTION_FILES else None,
        },
    }

    STATS_JSON.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

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
        print(f"- {CHUNKS_CSV}")
        print(f"- {CHUNKS_JSONL}")


if __name__ == "__main__":
    main()