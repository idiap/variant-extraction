# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#!/usr/bin/env python3
"""
build_contrastive_data_clinvar_safe.py

Build contrastive examples from supervised NER JSONL files.

Input rows must contain messages with a user context and assistant target.
Output rows contain context, positive target, and explicit negative targets.

Safety rules for candidate negative pair (variant, gene):
  1. Reject if the pair is positive in the current supervised example.
  2. Reject if the pair is known as a positive anywhere in train+validation corpus.
  3. Reject if ClinVar variant_summary links that variant/synonym to that gene.

ClinVar support is optional:
  --clinvar_summary_file /path/to/variant_summary.txt

The ClinVar parser is deliberately robust:
  - reads tab-separated variant_summary files, plain text or .gz
  - auto-detects gene columns such as GeneSymbol / Gene(s)
  - auto-detects variant text columns such as Name / VariationName / HGVS fields
  - extracts aliases such as p.Val600Glu -> V600E and rs121913529
  - only stores ClinVar evidence for genes present in your supervised pools.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO

NO_VARIANT = "No variant"
UNKNOWN_GENE_VALUES = {"unknown", "unk", "none", "na", "n/a", ""}

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Sec": "U", "Pyl": "O", "Ter": "*", "Stop": "*", "Xaa": "X",
}

GENE_COLUMN_CANDIDATES = ["GeneSymbol", "Gene(s)", "Genes", "Gene", "Symbol"]
VARIANT_TEXT_COLUMN_CANDIDATES = [
    "Name", "VariationName", "HGVS_c", "HGVS_p", "HGVS",
    "Canonical SPDI", "CanonicalSPDI", "RS# (dbSNP)", "RS",
]


@dataclass
class SFTExample:
    source_file: str
    source_idx: int
    context: str
    target: str
    pairs: list[tuple[str, str]]
    is_no_variant: bool
    metadata: dict


@dataclass
class Pools:
    variants: list[str]
    genes: list[str]
    known_pairs: set[tuple[str, str]]
    gene_to_variants: dict[str, set[str]]
    variant_to_genes: dict[str, set[str]]


@dataclass
class ClinVarIndex:
    gene_to_variants: dict[str, set[str]]
    n_rows_read: int
    n_rows_with_relevant_gene: int
    n_pairs_indexed: int


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_files", nargs="+", required=True)
    p.add_argument("--valid_files", nargs="+", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--negatives_per_example", type=int, default=8)
    p.add_argument("--include_no_variant_examples", action="store_true")
    p.add_argument("--max_no_variant_examples", type=int, default=None)
    p.add_argument(
        "--clinvar_summary_file",
        type=str,
        default=None,
        help="Optional path to ClinVar variant_summary.txt or variant_summary.txt.gz.",
    )
    p.add_argument(
        "--clinvar_gene_columns",
        nargs="*",
        default=None,
        help="Optional explicit ClinVar gene columns. Defaults to autodetection.",
    )
    p.add_argument(
        "--clinvar_variant_columns",
        nargs="*",
        default=None,
        help="Optional explicit ClinVar variant text columns. Defaults to autodetection.",
    )
    p.add_argument(
        "--clinvar_strict_substring_check",
        action="store_true",
        help="Also reject when candidate variant appears inside compact ClinVar variant text for same gene.",
    )
    return p.parse_args()


def canonical_variant(x: str) -> str:
    x = str(x).strip().strip("`'\".;")
    return re.sub(r"\s+", "", x)


def variant_key(x: str) -> str:
    return canonical_variant(x).casefold()


def canonical_gene(x: str) -> str:
    x = str(x).strip().strip("`'\".;")
    return re.sub(r"\s+", " ", x)


def gene_key(x: str) -> str:
    return canonical_gene(x).upper()


def is_unknown_gene(gene: str) -> bool:
    return canonical_gene(gene).casefold() in UNKNOWN_GENE_VALUES


def is_no_variant_text(x: str) -> bool:
    return str(x).strip().casefold() in {"no variant", "none", "0", "nan", ""}


def format_pairs(pairs: Iterable[tuple[str, str]]) -> str:
    lines = []
    seen = set()
    for variant, gene in pairs:
        variant = canonical_variant(variant)
        gene = canonical_gene(gene) or "Unknown"
        if not variant:
            continue
        pair = (variant, gene)
        if pair in seen:
            continue
        seen.add(pair)
        lines.append(f"Variant: {variant}, Gene: {gene}")
    return "\n".join(lines) if lines else NO_VARIANT


def parse_target_pairs(target: str) -> list[tuple[str, str]]:
    if is_no_variant_text(target):
        return []
    pairs = []
    for line in str(target).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^Variant:\s*(?P<variant>.+?)\s*,\s*Gene:\s*(?P<gene>.+?)\s*$", line, flags=re.I)
        if not m:
            continue
        variant = canonical_variant(m.group("variant"))
        gene = canonical_gene(m.group("gene")) or "Unknown"
        if variant:
            pairs.append((variant, gene))
    seen = set()
    out = []
    for pair in pairs:
        if pair not in seen:
            out.append(pair)
            seen.add(pair)
    return out


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {e}") from e
    return rows


def load_sft_examples(paths: list[Path]) -> list[SFTExample]:
    examples = []
    for path in paths:
        if not path.exists():
            print(f"Warning: missing file, skipping: {path}")
            continue
        for idx, row in enumerate(read_jsonl(path)):
            messages = row.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"Missing messages list in {path} row {idx}")
            user_text = None
            assistant_text = None
            for msg in messages:
                if msg.get("role") == "user":
                    user_text = str(msg.get("content", "")).strip()
                elif msg.get("role") == "assistant":
                    assistant_text = str(msg.get("content", "")).strip()
            if user_text is None or assistant_text is None:
                raise ValueError(f"Missing user/assistant in {path} row {idx}")
            pairs = parse_target_pairs(assistant_text)
            is_no_variant = is_no_variant_text(assistant_text) or len(pairs) == 0
            examples.append(SFTExample(str(path), idx, user_text, NO_VARIANT if is_no_variant else format_pairs(pairs), pairs, is_no_variant, row.get("metadata", {})))
    if not examples:
        raise RuntimeError("No examples loaded.")
    return examples


def build_pools(examples: list[SFTExample]) -> Pools:
    known_pairs = set()
    gene_to_variants = defaultdict(set)
    variant_to_genes = defaultdict(set)
    for ex in examples:
        for variant, gene in ex.pairs:
            if is_unknown_gene(gene):
                continue
            pair = (canonical_variant(variant), canonical_gene(gene))
            known_pairs.add(pair)
            gene_to_variants[pair[1]].add(pair[0])
            variant_to_genes[pair[0]].add(pair[1])
    variants = sorted({v for v, _ in known_pairs})
    genes = sorted({g for _, g in known_pairs})
    if not variants or not genes:
        raise RuntimeError("No usable positive variant-gene pairs found.")
    return Pools(variants, genes, known_pairs, dict(gene_to_variants), dict(variant_to_genes))


# -------------------------
# ClinVar indexing
# -------------------------

def open_text_maybe_gzip(path: Path) -> TextIO:
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def split_gene_field(value: str) -> list[str]:
    value = str(value or "").strip()
    if not value:
        return []
    parts = re.split(r"[|;,/]+", value)
    out = []
    for p in parts:
        p = canonical_gene(p)
        if p and not is_unknown_gene(p) and p != "-":
            out.append(p)
    return out


def compact_text_key(x: str) -> str:
    return re.sub(r"[^a-zA-Z0-9>*_=+\-]+", "", str(x)).casefold()


def aa3_to_1_variant(match: re.Match) -> str:
    aa1 = AA3_TO_1.get(match.group("aa1"), match.group("aa1"))
    pos = match.group("pos")
    aa2 = AA3_TO_1.get(match.group("aa2"), match.group("aa2"))
    return f"{aa1}{pos}{aa2}"


def extract_variant_aliases_from_text(text: str) -> set[str]:
    aliases = set()
    if not text:
        return aliases
    raw = str(text)

    for m in re.finditer(r"\brs\d+\b", raw, flags=re.I):
        aliases.add(m.group(0))

    for m in re.finditer(
        r"\b(?:c|g|p|m|n|r)\.[A-Za-z0-9_>*=+\-?]+(?:delins|del|dup|ins|inv)?[A-Za-z0-9_>*=+\-?]*",
        raw,
        flags=re.I,
    ):
        aliases.add(m.group(0))

    for m in re.finditer(
        r"(?:p\.)?(?P<aa1>Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val|Sec|Pyl|Ter|Stop|Xaa)"
        r"(?P<pos>\d+)"
        r"(?P<aa2>Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val|Sec|Pyl|Ter|Stop|Xaa)",
        raw,
    ):
        long_form = m.group(0)
        aliases.add(long_form)
        aliases.add(long_form[2:] if long_form.startswith("p.") else long_form)
        aliases.add(aa3_to_1_variant(m))

    for m in re.finditer(r"(?:p\.)?[A-Z]\d+[A-Z*]", raw):
        aliases.add(m.group(0))
        if m.group(0).startswith("p."):
            aliases.add(m.group(0)[2:])

    for m in re.finditer(r"\b[A-Z]+\d+[A-Z>*=]+\b", raw):
        aliases.add(m.group(0))

    return {a for a in aliases if a}


def choose_columns(fieldnames: list[str], explicit: list[str] | None, candidates: list[str], mode: str) -> list[str]:
    if explicit:
        missing = [c for c in explicit if c not in fieldnames]
        if missing:
            raise ValueError(f"Requested ClinVar columns not found: {missing}")
        return explicit
    found = [c for c in candidates if c in fieldnames]
    if found:
        return found
    fuzzy = []
    for original in fieldnames:
        key = original.lower()
        if mode == "gene" and ("gene" in key or "symbol" in key):
            fuzzy.append(original)
        if mode == "variant" and any(x in key for x in ["name", "hgvs", "variant", "variation", "rs"]):
            fuzzy.append(original)
    return fuzzy


def build_clinvar_index(
    clinvar_summary_file: str | None,
    relevant_genes: set[str],
    gene_columns: list[str] | None = None,
    variant_columns: list[str] | None = None,
    strict_substring_check: bool = False,
) -> ClinVarIndex | None:
    if not clinvar_summary_file:
        return None

    path = Path(clinvar_summary_file)
    if not path.exists():
        raise FileNotFoundError(f"ClinVar summary file not found: {path}")

    relevant_gene_keys = {gene_key(g) for g in relevant_genes}
    gene_to_variants = defaultdict(set)
    n_rows = 0
    n_relevant = 0
    n_pairs = 0

    print(f"Reading ClinVar summary for strict negative filtering: {path}")

    with open_text_maybe_gzip(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            raise RuntimeError(f"Could not read header from ClinVar summary file: {path}")

        gene_cols = choose_columns(reader.fieldnames, gene_columns, GENE_COLUMN_CANDIDATES, mode="gene")
        var_cols = choose_columns(reader.fieldnames, variant_columns, VARIANT_TEXT_COLUMN_CANDIDATES, mode="variant")

        if not gene_cols:
            raise RuntimeError("Could not detect ClinVar gene columns. Pass --clinvar_gene_columns explicitly.")
        if not var_cols:
            raise RuntimeError("Could not detect ClinVar variant text columns. Pass --clinvar_variant_columns explicitly.")

        print(f"ClinVar gene columns: {gene_cols}")
        print(f"ClinVar variant text columns: {var_cols}")

        for row in reader:
            n_rows += 1
            genes = []
            for c in gene_cols:
                genes.extend(split_gene_field(row.get(c, "")))
            genes = sorted(set(genes))
            matched = [g for g in genes if gene_key(g) in relevant_gene_keys]
            if not matched:
                continue
            n_relevant += 1

            joined_text = " ".join(str(row.get(c, "") or "") for c in var_cols)
            aliases = extract_variant_aliases_from_text(joined_text)
            if strict_substring_check:
                aliases.add(compact_text_key(joined_text))
            alias_keys = {variant_key(a) for a in aliases if a}
            if not alias_keys:
                continue

            for g in matched:
                gk = gene_key(g)
                before = len(gene_to_variants[gk])
                gene_to_variants[gk].update(alias_keys)
                n_pairs += len(gene_to_variants[gk]) - before

    print(
        f"ClinVar index complete: rows={n_rows:,}, relevant_rows={n_relevant:,}, "
        f"indexed_gene_variant_aliases={n_pairs:,}"
    )
    return ClinVarIndex(dict(gene_to_variants), n_rows, n_relevant, n_pairs)


def clinvar_rejects_pair(
    variant: str,
    gene: str,
    clinvar_index: ClinVarIndex | None,
    strict_substring_check: bool = False,
) -> bool:
    if clinvar_index is None:
        return False
    aliases = clinvar_index.gene_to_variants.get(gene_key(gene), set())
    if not aliases:
        return False
    vk = variant_key(variant)
    if vk in aliases:
        return True
    if strict_substring_check:
        compact_v = compact_text_key(variant)
        return any(compact_v and compact_v in a for a in aliases)
    return False


def is_safe_negative_pair(
    variant: str,
    gene: str,
    current_positive_pairs: set[tuple[str, str]],
    pools: Pools,
    clinvar_index: ClinVarIndex | None = None,
    clinvar_strict_substring_check: bool = False,
) -> bool:
    variant = canonical_variant(variant)
    gene = canonical_gene(gene)
    if not variant or is_unknown_gene(gene):
        return False
    pair = (variant, gene)
    if pair in current_positive_pairs:
        return False
    if pair in pools.known_pairs:
        return False
    if variant in pools.gene_to_variants.get(gene, set()):
        return False
    if gene in pools.variant_to_genes.get(variant, set()):
        return False
    if clinvar_rejects_pair(variant, gene, clinvar_index, clinvar_strict_substring_check):
        return False
    return True


def add_negative_target(negatives: list[str], pairs: list[tuple[str, str]], current: set[tuple[str, str]], pools: Pools, clinvar_index: ClinVarIndex | None, clinvar_strict_substring_check: bool):
    safe = [(v, g) for v, g in pairs if is_safe_negative_pair(v, g, current, pools, clinvar_index, clinvar_strict_substring_check)]
    if not safe:
        return
    target = format_pairs(safe)
    if target != NO_VARIANT and target not in negatives:
        negatives.append(target)


def sample_safe_random_pair(rng: random.Random, pools: Pools, current: set[tuple[str, str]], clinvar_index: ClinVarIndex | None, clinvar_strict_substring_check: bool, max_attempts: int = 1000):
    for _ in range(max_attempts):
        v = rng.choice(pools.variants)
        g = rng.choice(pools.genes)
        if is_safe_negative_pair(v, g, current, pools, clinvar_index, clinvar_strict_substring_check):
            return v, g
    return None


def make_negatives(ex: SFTExample, pools: Pools, rng: random.Random, n: int, clinvar_index: ClinVarIndex | None, clinvar_strict_substring_check: bool) -> list[str]:
    negatives = []
    current = set(ex.pairs)

    if ex.is_no_variant:
        while len(negatives) < n:
            pair = sample_safe_random_pair(rng, pools, current, clinvar_index, clinvar_strict_substring_check)
            if pair is None:
                break
            add_negative_target(negatives, [pair], current, pools, clinvar_index, clinvar_strict_substring_check)
        return negatives

    for v, _ in ex.pairs:
        genes = pools.genes[:]
        rng.shuffle(genes)
        for wrong_gene in genes:
            if is_safe_negative_pair(v, wrong_gene, current, pools, clinvar_index, clinvar_strict_substring_check):
                add_negative_target(negatives, [(v, wrong_gene)], current, pools, clinvar_index, clinvar_strict_substring_check)
                break
        if len(negatives) >= n:
            return negatives[:n]

    for _, g in ex.pairs:
        if is_unknown_gene(g):
            continue
        variants = pools.variants[:]
        rng.shuffle(variants)
        for wrong_variant in variants:
            if is_safe_negative_pair(wrong_variant, g, current, pools, clinvar_index, clinvar_strict_substring_check):
                add_negative_target(negatives, [(wrong_variant, g)], current, pools, clinvar_index, clinvar_strict_substring_check)
                break
        if len(negatives) >= n:
            return negatives[:n]

    if len(ex.pairs) >= 2:
        variants = [v for v, _ in ex.pairs]
        genes = [g for _, g in ex.pairs]
        for _ in range(50):
            shuffled = genes[:]
            rng.shuffle(shuffled)
            cand = list(zip(variants, shuffled))
            if set(cand) != current:
                add_negative_target(negatives, cand, current, pools, clinvar_index, clinvar_strict_substring_check)
            if len(negatives) >= n:
                return negatives[:n]

    while len(negatives) < n:
        pair = sample_safe_random_pair(rng, pools, current, clinvar_index, clinvar_strict_substring_check)
        if pair is None:
            break
        add_negative_target(negatives, [pair], current, pools, clinvar_index, clinvar_strict_substring_check)
    return negatives[:n]


def build_rows(examples: list[SFTExample], pools: Pools, rng: random.Random, negatives_per_example: int, include_no_variant_examples: bool, max_no_variant_examples: int | None, clinvar_index: ClinVarIndex | None, clinvar_strict_substring_check: bool) -> list[dict]:
    rows = []
    n_no_variant = 0
    for ex in examples:
        if ex.is_no_variant:
            if not include_no_variant_examples:
                continue
            if max_no_variant_examples is not None and n_no_variant >= max_no_variant_examples:
                continue
            n_no_variant += 1
        negatives = make_negatives(ex, pools, rng, negatives_per_example, clinvar_index, clinvar_strict_substring_check)
        if not negatives:
            continue
        rows.append({
            "context": ex.context,
            "positive": ex.target,
            "negatives": negatives,
            "metadata": {
                "source_file": ex.source_file,
                "source_idx": ex.source_idx,
                "is_no_variant": ex.is_no_variant,
                "positive_pairs": [list(p) for p in ex.pairs],
                "n_negatives": len(negatives),
                "source_metadata": ex.metadata,
            },
        })
    return rows


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    train_paths = [Path(x) for x in args.train_files]
    valid_paths = [Path(x) for x in args.valid_files]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_examples = load_sft_examples(train_paths)
    valid_examples = load_sft_examples(valid_paths)
    pools = build_pools(train_examples + valid_examples)

    clinvar_index = build_clinvar_index(
        clinvar_summary_file=args.clinvar_summary_file,
        relevant_genes=set(pools.genes),
        gene_columns=args.clinvar_gene_columns,
        variant_columns=args.clinvar_variant_columns,
        strict_substring_check=args.clinvar_strict_substring_check,
    )

    train_rows = build_rows(train_examples, pools, random.Random(args.seed), args.negatives_per_example, args.include_no_variant_examples, args.max_no_variant_examples, clinvar_index, args.clinvar_strict_substring_check)
    valid_rows = build_rows(valid_examples, pools, random.Random(args.seed + 1), args.negatives_per_example, args.include_no_variant_examples, args.max_no_variant_examples, clinvar_index, args.clinvar_strict_substring_check)

    train_out = out / "contrastive_train.jsonl"
    valid_out = out / "contrastive_val.jsonl"
    stats_out = out / "contrastive_data_stats.json"
    write_jsonl(train_out, train_rows)
    write_jsonl(valid_out, valid_rows)

    stats = {
        "train_files": [str(p) for p in train_paths],
        "valid_files": [str(p) for p in valid_paths],
        "output_dir": str(out),
        "seed": args.seed,
        "negatives_per_example": args.negatives_per_example,
        "include_no_variant_examples": args.include_no_variant_examples,
        "max_no_variant_examples": args.max_no_variant_examples,
        "clinvar_summary_file": args.clinvar_summary_file,
        "clinvar_strict_substring_check": args.clinvar_strict_substring_check,
        "pool_stats": {"variants": len(pools.variants), "genes": len(pools.genes), "known_pairs": len(pools.known_pairs)},
        "clinvar_index_stats": None if clinvar_index is None else {
            "rows_read": clinvar_index.n_rows_read,
            "rows_with_relevant_gene": clinvar_index.n_rows_with_relevant_gene,
            "indexed_gene_variant_aliases": clinvar_index.n_pairs_indexed,
            "genes_indexed": len(clinvar_index.gene_to_variants),
        },
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "train_no_variant_rows": sum(r["metadata"]["is_no_variant"] for r in train_rows),
        "valid_no_variant_rows": sum(r["metadata"]["is_no_variant"] for r in valid_rows),
        "train_output": str(train_out),
        "valid_output": str(valid_out),
    }
    stats_out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
