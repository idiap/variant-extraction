# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#!/usr/bin/env python3
"""
build_contrastive_data_text_output.py

Build contrastive examples for a generative biomedical NER/extraction model.

This version keeps the model-facing output interface identical to evaluation:

  Positive for variant-bearing text:
    context = original user text
    positive = "Variant: <right variant>, Gene: <right gene>"

  Positive for no-variant text:
    context = original user text
    positive = "No variant"

  Negatives for variant-bearing text include evaluator-style outputs with:
    - wrong variant, right gene
    - right variant, wrong gene
    - wrong variant, wrong gene
    - "No variant"

  Negatives for no-variant text are evaluator-style variant/gene outputs sampled
  from the supervised corpus.

Input rows must contain OpenAI-style messages with a user context and assistant
NER target, e.g.:

  {"messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "Title: ...\nAbstract: ..."},
      {"role": "assistant", "content": "Variant: V600E, Gene: BRAF"}
  ]}

By default, output files keep the grouped format used by the earlier script:

  {"context": ..., "positive": ..., "negatives": [...], "metadata": {...}}

Optionally, use --output_format dpo or both to also write pairwise preference
rows:

  {"prompt": ..., "chosen": ..., "rejected": ..., "metadata": {...}}

No "Unknown" gene fallback is used. A target line is accepted only if it has a
non-empty variant and a non-empty, non-unknown gene.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

NO_VARIANT = "No variant"
UNKNOWN_GENE_VALUES = {"unknown", "unk", "none", "na", "n/a", ""}


@dataclass
class SFTExample:
    source_file: str
    source_idx: int
    system: str
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_files", nargs="+", required=True)
    p.add_argument("--valid_files", nargs="+", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--negatives_per_example", type=int, default=8)
    p.add_argument("--include_no_variant_examples", action="store_true")
    p.add_argument("--max_no_variant_examples", type=int, default=None)
    p.add_argument(
        "--output_format",
        choices=["grouped", "dpo", "both"],
        default="grouped",
        help="grouped writes context/positive/negatives; dpo writes prompt/chosen/rejected; both writes both.",
    )
    p.add_argument(
        "--avoid_known_positive_pairs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Avoid generating rejected variant-gene pairs that are positives anywhere in train+validation.",
    )
    p.add_argument(
        "--no_variant_negative_for_positive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For positive examples, include 'No variant' as one rejected output.",
    )
    return p.parse_args()


# -------------------------
# Canonicalization / parsing
# -------------------------

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
    """Format pairs exactly as the evaluator expects. No Unknown fallback."""
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()

    for variant, gene in pairs:
        variant = canonical_variant(variant)
        gene = canonical_gene(gene)
        if not variant or is_unknown_gene(gene):
            continue
        pair = (variant, gene)
        if pair in seen:
            continue
        seen.add(pair)
        lines.append(f"Variant: {variant}, Gene: {gene}")

    return "\n".join(lines) if lines else NO_VARIANT


def parse_target_pairs(target: str) -> list[tuple[str, str]]:
    """Parse evaluator-style assistant targets into (variant, gene) pairs.

    Lines with missing/unknown genes are ignored instead of being converted to
    Gene: Unknown.
    """
    if is_no_variant_text(target):
        return []

    pairs: list[tuple[str, str]] = []
    for line in str(target).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(
            r"^Variant:\s*(?P<variant>.+?)\s*,\s*Gene:\s*(?P<gene>.+?)\s*$",
            line,
            flags=re.I,
        )
        if not m:
            continue

        variant = canonical_variant(m.group("variant"))
        gene = canonical_gene(m.group("gene"))
        if variant and not is_unknown_gene(gene):
            pairs.append((variant, gene))

    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in pairs:
        if pair not in seen:
            out.append(pair)
            seen.add(pair)
    return out


# -------------------------
# Loading
# -------------------------

def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
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
    examples: list[SFTExample] = []

    for path in paths:
        if not path.exists():
            print(f"Warning: missing file, skipping: {path}")
            continue

        for idx, row in enumerate(read_jsonl(path)):
            messages = row.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"Missing messages list in {path} row {idx}")

            system_text = ""
            user_text = None
            assistant_text = None

            for msg in messages:
                role = msg.get("role")
                content = str(msg.get("content", "")).strip()
                if role == "system":
                    system_text = content
                elif role == "user":
                    user_text = content
                elif role == "assistant":
                    assistant_text = content

            if user_text is None or assistant_text is None:
                raise ValueError(f"Missing user/assistant in {path} row {idx}")

            pairs = parse_target_pairs(assistant_text)
            explicit_no_variant = is_no_variant_text(assistant_text)
            is_no_variant = explicit_no_variant or len(pairs) == 0
            target = NO_VARIANT if is_no_variant else format_pairs(pairs)

            examples.append(
                SFTExample(
                    source_file=str(path),
                    source_idx=idx,
                    system=system_text,
                    context=user_text,
                    target=target,
                    pairs=pairs,
                    is_no_variant=is_no_variant,
                    metadata=row.get("metadata", {}),
                )
            )

    if not examples:
        raise RuntimeError("No examples loaded.")
    return examples


def build_pools(examples: list[SFTExample]) -> Pools:
    known_pairs: set[tuple[str, str]] = set()
    gene_to_variants: dict[str, set[str]] = defaultdict(set)
    variant_to_genes: dict[str, set[str]] = defaultdict(set)

    for ex in examples:
        for variant, gene in ex.pairs:
            if is_unknown_gene(gene):
                continue
            pair = (canonical_variant(variant), canonical_gene(gene))
            if not pair[0] or is_unknown_gene(pair[1]):
                continue
            known_pairs.add(pair)
            gene_to_variants[pair[1]].add(pair[0])
            variant_to_genes[pair[0]].add(pair[1])

    variants = sorted({v for v, _ in known_pairs})
    genes = sorted({g for _, g in known_pairs})

    if not variants or not genes:
        raise RuntimeError("No usable positive variant-gene pairs found.")

    return Pools(
        variants=variants,
        genes=genes,
        known_pairs=known_pairs,
        gene_to_variants=dict(gene_to_variants),
        variant_to_genes=dict(variant_to_genes),
    )


# -------------------------
# Negative generation
# -------------------------

def pair_is_allowed_negative(
    pair: tuple[str, str],
    current_pairs: set[tuple[str, str]],
    pools: Pools,
    avoid_known_positive_pairs: bool,
) -> bool:
    variant, gene = canonical_variant(pair[0]), canonical_gene(pair[1])
    if not variant or is_unknown_gene(gene):
        return False
    pair = (variant, gene)
    if pair in current_pairs:
        return False
    if avoid_known_positive_pairs and pair in pools.known_pairs:
        return False
    return True


def sample_variant(
    rng: random.Random,
    pools: Pools,
    forbidden_variants: set[str],
) -> str | None:
    candidates = [v for v in pools.variants if v not in forbidden_variants]
    if not candidates:
        return None
    return rng.choice(candidates)


def sample_gene(
    rng: random.Random,
    pools: Pools,
    forbidden_genes: set[str],
) -> str | None:
    candidates = [g for g in pools.genes if g not in forbidden_genes and not is_unknown_gene(g)]
    if not candidates:
        return None
    return rng.choice(candidates)


def replace_one_pair(
    pairs: list[tuple[str, str]],
    index: int,
    replacement: tuple[str, str],
) -> list[tuple[str, str]]:
    out = list(pairs)
    out[index] = replacement
    return out


def add_negative(negatives: list[str], target: str, positive: str) -> None:
    target = target.strip()
    if not target:
        return
    if target == positive:
        return
    if target not in negatives:
        negatives.append(target)


def make_positive_example_negatives(
    ex: SFTExample,
    pools: Pools,
    rng: random.Random,
    n: int,
    avoid_known_positive_pairs: bool,
    include_no_variant_negative: bool,
) -> list[str]:
    """Create evaluator-style rejected outputs for a variant-bearing context."""
    negatives: list[str] = []
    current_pairs = set(ex.pairs)
    current_variants = {v for v, _ in ex.pairs}
    current_genes = {g for _, g in ex.pairs}
    positive = format_pairs(ex.pairs)

    # 1) wrong variant, right gene: corrupt one existing pair, keep other pairs.
    order = list(range(len(ex.pairs)))
    rng.shuffle(order)
    for idx in order:
        _, right_gene = ex.pairs[idx]
        for _ in range(200):
            wrong_variant = sample_variant(rng, pools, current_variants)
            if wrong_variant is None:
                break
            replacement = (wrong_variant, right_gene)
            if pair_is_allowed_negative(replacement, current_pairs, pools, avoid_known_positive_pairs):
                add_negative(negatives, format_pairs(replace_one_pair(ex.pairs, idx, replacement)), positive)
                break
        if len(negatives) >= n:
            return negatives[:n]

    # 2) right variant, wrong gene: corrupt one existing pair, keep other pairs.
    order = list(range(len(ex.pairs)))
    rng.shuffle(order)
    for idx in order:
        right_variant, _ = ex.pairs[idx]
        for _ in range(200):
            wrong_gene = sample_gene(rng, pools, current_genes)
            if wrong_gene is None:
                break
            replacement = (right_variant, wrong_gene)
            if pair_is_allowed_negative(replacement, current_pairs, pools, avoid_known_positive_pairs):
                add_negative(negatives, format_pairs(replace_one_pair(ex.pairs, idx, replacement)), positive)
                break
        if len(negatives) >= n:
            return negatives[:n]

    # 3) wrong variant, wrong gene: corrupt one existing pair, keep other pairs.
    order = list(range(len(ex.pairs)))
    rng.shuffle(order)
    for idx in order:
        for _ in range(500):
            wrong_variant = sample_variant(rng, pools, current_variants)
            wrong_gene = sample_gene(rng, pools, current_genes)
            if wrong_variant is None or wrong_gene is None:
                break
            replacement = (wrong_variant, wrong_gene)
            if pair_is_allowed_negative(replacement, current_pairs, pools, avoid_known_positive_pairs):
                add_negative(negatives, format_pairs(replace_one_pair(ex.pairs, idx, replacement)), positive)
                break
        if len(negatives) >= n:
            return negatives[:n]

    # 4) false no-variant negative.
    if include_no_variant_negative:
        add_negative(negatives, NO_VARIANT, positive)
        if len(negatives) >= n:
            return negatives[:n]

    # Fill remaining slots with random wrong-variant/wrong-gene corruptions.
    while len(negatives) < n:
        idx = rng.randrange(len(ex.pairs))
        made_one = False
        for _ in range(500):
            wrong_variant = sample_variant(rng, pools, current_variants)
            wrong_gene = sample_gene(rng, pools, current_genes)
            if wrong_variant is None or wrong_gene is None:
                break
            replacement = (wrong_variant, wrong_gene)
            if pair_is_allowed_negative(replacement, current_pairs, pools, avoid_known_positive_pairs):
                add_negative(negatives, format_pairs(replace_one_pair(ex.pairs, idx, replacement)), positive)
                made_one = True
                break
        if not made_one:
            break

    return negatives[:n]


def make_no_variant_example_negatives(
    pools: Pools,
    rng: random.Random,
    n: int,
) -> list[str]:
    """For a true No-variant context, any variant/gene output is a negative.

    These negatives are sampled from real positive pairs elsewhere in the corpus,
    so they look realistic while remaining wrong for this no-variant text.
    """
    negatives: list[str] = []
    known_pairs = list(pools.known_pairs)
    rng.shuffle(known_pairs)

    for pair in known_pairs:
        add_negative(negatives, format_pairs([pair]), NO_VARIANT)
        if len(negatives) >= n:
            break

    return negatives


def make_negatives(
    ex: SFTExample,
    pools: Pools,
    rng: random.Random,
    n: int,
    avoid_known_positive_pairs: bool,
    include_no_variant_negative: bool,
) -> list[str]:
    if ex.is_no_variant:
        return make_no_variant_example_negatives(pools, rng, n)

    return make_positive_example_negatives(
        ex=ex,
        pools=pools,
        rng=rng,
        n=n,
        avoid_known_positive_pairs=avoid_known_positive_pairs,
        include_no_variant_negative=include_no_variant_negative,
    )


# -------------------------
# Row building / writing
# -------------------------

def build_grouped_rows(
    examples: list[SFTExample],
    pools: Pools,
    rng: random.Random,
    negatives_per_example: int,
    include_no_variant_examples: bool,
    max_no_variant_examples: int | None,
    avoid_known_positive_pairs: bool,
    include_no_variant_negative: bool,
) -> list[dict]:
    rows: list[dict] = []
    n_no_variant = 0

    for ex in examples:
        if ex.is_no_variant:
            if not include_no_variant_examples:
                continue
            if max_no_variant_examples is not None and n_no_variant >= max_no_variant_examples:
                continue
            n_no_variant += 1

        negatives = make_negatives(
            ex=ex,
            pools=pools,
            rng=rng,
            n=negatives_per_example,
            avoid_known_positive_pairs=avoid_known_positive_pairs,
            include_no_variant_negative=include_no_variant_negative,
        )
        if not negatives:
            continue

        rows.append(
            {
                "context": ex.context,
                "positive": ex.target,
                "negatives": negatives,
                "metadata": {
                    "source_file": ex.source_file,
                    "source_idx": ex.source_idx,
                    "is_no_variant": ex.is_no_variant,
                    "positive_pairs": [list(p) for p in ex.pairs],
                    "n_negatives": len(negatives),
                    "negative_types": (
                        ["sampled_positive_pair_elsewhere"] if ex.is_no_variant else
                        ["wrong_variant_right_gene", "right_variant_wrong_gene", "wrong_variant_wrong_gene", "no_variant"]
                    ),
                    "source_metadata": ex.metadata,
                },
            }
        )

    return rows


def grouped_to_dpo_rows(grouped_rows: list[dict], include_system: bool = False) -> list[dict]:
    """Expand grouped rows into one chosen/rejected row per negative.

    The prompt field is just the original user context, matching the contrastive
    anchor requested by the user. If your trainer expects chat messages instead,
    adapt this function only; keep chosen/rejected strings unchanged.
    """
    rows: list[dict] = []
    for row in grouped_rows:
        for neg_idx, rejected in enumerate(row["negatives"]):
            rows.append(
                {
                    "prompt": row["context"],
                    "chosen": row["positive"],
                    "rejected": rejected,
                    "metadata": {
                        **row.get("metadata", {}),
                        "negative_idx": neg_idx,
                    },
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_outputs(
    output_dir: Path,
    split: Literal["train", "val"],
    grouped_rows: list[dict],
    output_format: str,
) -> dict[str, str | int]:
    written: dict[str, str | int] = {}

    if output_format in {"grouped", "both"}:
        path = output_dir / f"contrastive_{split}.jsonl"
        write_jsonl(path, grouped_rows)
        written[f"{split}_grouped_output"] = str(path)
        written[f"{split}_grouped_rows"] = len(grouped_rows)

    if output_format in {"dpo", "both"}:
        dpo_rows = grouped_to_dpo_rows(grouped_rows)
        path = output_dir / f"contrastive_{split}_dpo.jsonl"
        write_jsonl(path, dpo_rows)
        written[f"{split}_dpo_output"] = str(path)
        written[f"{split}_dpo_rows"] = len(dpo_rows)

    return written


# -------------------------
# Main
# -------------------------

def main() -> None:
    args = parse_args()

    train_paths = [Path(x) for x in args.train_files]
    valid_paths = [Path(x) for x in args.valid_files]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_examples = load_sft_examples(train_paths)
    valid_examples = load_sft_examples(valid_paths)
    pools = build_pools(train_examples + valid_examples)

    train_rows = build_grouped_rows(
        examples=train_examples,
        pools=pools,
        rng=random.Random(args.seed),
        negatives_per_example=args.negatives_per_example,
        include_no_variant_examples=args.include_no_variant_examples,
        max_no_variant_examples=args.max_no_variant_examples,
        avoid_known_positive_pairs=args.avoid_known_positive_pairs,
        include_no_variant_negative=args.no_variant_negative_for_positive,
    )
    valid_rows = build_grouped_rows(
        examples=valid_examples,
        pools=pools,
        rng=random.Random(args.seed + 1),
        negatives_per_example=args.negatives_per_example,
        include_no_variant_examples=args.include_no_variant_examples,
        max_no_variant_examples=args.max_no_variant_examples,
        avoid_known_positive_pairs=args.avoid_known_positive_pairs,
        include_no_variant_negative=args.no_variant_negative_for_positive,
    )

    written = {}
    written.update(write_outputs(out, "train", train_rows, args.output_format))
    written.update(write_outputs(out, "val", valid_rows, args.output_format))

    stats = {
        "train_files": [str(p) for p in train_paths],
        "valid_files": [str(p) for p in valid_paths],
        "output_dir": str(out),
        "seed": args.seed,
        "negatives_per_example": args.negatives_per_example,
        "include_no_variant_examples": args.include_no_variant_examples,
        "max_no_variant_examples": args.max_no_variant_examples,
        "output_format": args.output_format,
        "avoid_known_positive_pairs": args.avoid_known_positive_pairs,
        "no_variant_negative_for_positive": args.no_variant_negative_for_positive,
        "pool_stats": {
            "variants": len(pools.variants),
            "genes": len(pools.genes),
            "known_pairs": len(pools.known_pairs),
        },
        "train_examples_loaded": len(train_examples),
        "valid_examples_loaded": len(valid_examples),
        "train_grouped_rows_built": len(train_rows),
        "valid_grouped_rows_built": len(valid_rows),
        "train_no_variant_rows": sum(r["metadata"]["is_no_variant"] for r in train_rows),
        "valid_no_variant_rows": sum(r["metadata"]["is_no_variant"] for r in valid_rows),
        **written,
    }

    stats_out = out / "contrastive_data_stats.json"
    stats_out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
