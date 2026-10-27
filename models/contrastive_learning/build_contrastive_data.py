# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#!/usr/bin/env python3
"""
build_contrastive_data.py

Build contrastive examples from supervised NER JSONL files.

Input rows must contain messages with a user context and assistant target.
Output rows contain context, positive target, and explicit negative targets.

Safety rule: a candidate negative pair (variant, gene) is rejected if that pair
is known as a positive anywhere in the train+validation corpus.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

NO_VARIANT = "No variant"
UNKNOWN_GENE_VALUES = {"unknown", "unk", "none", "na", "n/a", ""}


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_files", nargs="+", required=True)
    p.add_argument("--valid_files", nargs="+", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--negatives_per_example", type=int, default=8)
    p.add_argument("--include_no_variant_examples", action="store_true")
    p.add_argument("--max_no_variant_examples", type=int, default=None)
    return p.parse_args()


def canonical_variant(x: str) -> str:
    x = str(x).strip().strip("`'\".;")
    return re.sub(r"\s+", "", x)


def canonical_gene(x: str) -> str:
    x = str(x).strip().strip("`'\".;")
    return re.sub(r"\s+", " ", x)


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


def is_safe_negative_pair(variant: str, gene: str, current_positive_pairs: set[tuple[str, str]], pools: Pools) -> bool:
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
    return True


def add_negative_target(negatives: list[str], pairs: list[tuple[str, str]], current: set[tuple[str, str]], pools: Pools):
    safe = [(v, g) for v, g in pairs if is_safe_negative_pair(v, g, current, pools)]
    if not safe:
        return
    target = format_pairs(safe)
    if target != NO_VARIANT and target not in negatives:
        negatives.append(target)


def sample_safe_random_pair(rng: random.Random, pools: Pools, current: set[tuple[str, str]], max_attempts: int = 1000):
    for _ in range(max_attempts):
        v = rng.choice(pools.variants)
        g = rng.choice(pools.genes)
        if is_safe_negative_pair(v, g, current, pools):
            return v, g
    return None


def make_negatives(ex: SFTExample, pools: Pools, rng: random.Random, n: int) -> list[str]:
    negatives = []
    current = set(ex.pairs)
    if ex.is_no_variant:
        while len(negatives) < n:
            pair = sample_safe_random_pair(rng, pools, current)
            if pair is None:
                break
            add_negative_target(negatives, [pair], current, pools)
        return negatives

    # Correct variant, wrong gene.
    for v, _ in ex.pairs:
        genes = pools.genes[:]
        rng.shuffle(genes)
        for wrong_gene in genes:
            if is_safe_negative_pair(v, wrong_gene, current, pools):
                add_negative_target(negatives, [(v, wrong_gene)], current, pools)
                break
        if len(negatives) >= n:
            return negatives[:n]

    # Correct gene, wrong variant.
    for _, g in ex.pairs:
        if is_unknown_gene(g):
            continue
        variants = pools.variants[:]
        rng.shuffle(variants)
        for wrong_variant in variants:
            if is_safe_negative_pair(wrong_variant, g, current, pools):
                add_negative_target(negatives, [(wrong_variant, g)], current, pools)
                break
        if len(negatives) >= n:
            return negatives[:n]

    # Shuffled genes within same example.
    if len(ex.pairs) >= 2:
        variants = [v for v, _ in ex.pairs]
        genes = [g for _, g in ex.pairs]
        for _ in range(50):
            shuffled = genes[:]
            rng.shuffle(shuffled)
            cand = list(zip(variants, shuffled))
            if set(cand) != current:
                add_negative_target(negatives, cand, current, pools)
            if len(negatives) >= n:
                return negatives[:n]

    # Random safe pairs.
    while len(negatives) < n:
        pair = sample_safe_random_pair(rng, pools, current)
        if pair is None:
            break
        add_negative_target(negatives, [pair], current, pools)
    return negatives[:n]


def build_rows(examples: list[SFTExample], pools: Pools, rng: random.Random, negatives_per_example: int, include_no_variant_examples: bool, max_no_variant_examples: int | None) -> list[dict]:
    rows = []
    n_no_variant = 0
    for ex in examples:
        if ex.is_no_variant:
            if not include_no_variant_examples:
                continue
            if max_no_variant_examples is not None and n_no_variant >= max_no_variant_examples:
                continue
            n_no_variant += 1
        negatives = make_negatives(ex, pools, rng, negatives_per_example)
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

    train_rows = build_rows(train_examples, pools, random.Random(args.seed), args.negatives_per_example, args.include_no_variant_examples, args.max_no_variant_examples)
    valid_rows = build_rows(valid_examples, pools, random.Random(args.seed + 1), args.negatives_per_example, args.include_no_variant_examples, args.max_no_variant_examples)

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
        "pool_stats": {"variants": len(pools.variants), "genes": len(pools.genes), "known_pairs": len(pools.known_pairs)},
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
