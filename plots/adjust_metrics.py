# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#!/usr/bin/env python3
"""
adjust_metrics_textual_variants.py

Post-process an existing metrics.json + per_example_errors.csv pair and create
metrics_adj.json where specific textual substitution/change predictions are
counted as true positives if the predicted textual variant is present in
the source text.

This is useful when train data treats mentions such as:
  - "A to C change"
  - "G to T substitution"
  - "valine to glutamic acid substitution"
  - "substitution of methionine for threonine at position 790"
  - "S194 was mutated to alanine"

as variant mentions, while the test-set gold logic does not.

Adjustment policy:
  - Accepted textual substitution/change predictions are moved from FP to TP.
  - Do NOT remove false negatives.
  - For variant and pair micro metrics, accepted predictions decrease FP and increase TP.
  - For binary metrics, a gold-negative row with any accepted textual prediction
    becomes an adjusted positive example, so binary FP becomes TP.
  - Keep format_validity unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


AMINO_ACID_WORDS = [
    "alanine", "arginine", "asparagine", "aspartic acid", "cysteine",
    "glutamine", "glutamic acid", "glycine", "histidine", "isoleucine",
    "leucine", "lysine", "methionine", "phenylalanine", "proline",
    "serine", "threonine", "tryptophan", "tyrosine", "valine",
    "aspartate", "glutamate",
    "ala", "arg", "asn", "asp", "cys", "gln", "glu", "gly", "his",
    "ile", "leu", "lys", "met", "phe", "pro", "ser", "thr", "trp",
    "tyr", "val",
]

NUCLEOTIDE_WORDS = [
    "adenine", "cytosine", "guanine", "thymine", "uracil",
    "purine", "pyrimidine",
    "a", "c", "g", "t", "u",
]

RESIDUE_TERMS = sorted(
    set(AMINO_ACID_WORDS + NUCLEOTIDE_WORDS),
    key=len,
    reverse=True,
)

ACTION_TERMS = [
    "change", "substitution", "mutation", "mutated", "replacement",
    "transition", "transversion",
]

POSITION_TERMS = [
    "codon", "position", "residue", "amino acid position", "nt", "nucleotide",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_json", required=True)
    p.add_argument("--errors_csv", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument(
        "--accepted_cases_csv",
        default=None,
        help="Optional CSV report listing accepted textual-variant false positives.",
    )
    p.add_argument(
        "--case_insensitive",
        action="store_true",
        default=True,
        help="Case-insensitive matching; enabled by default.",
    )
    return p.parse_args()


def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x)


def compact_for_matching(x: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", safe_str(x)).casefold()


def metric_key_variant(x: str) -> str:
    # Matches the style visible in per_example_errors.csv: lowercased and whitespace-free.
    return re.sub(r"\s+", "", safe_str(x).strip().strip("`'\".;")).casefold()


def parse_pipe_list(x) -> list[str]:
    s = safe_str(x).strip()
    if not s:
        return []
    return [item.strip() for item in s.split("|") if item.strip()]


def parse_extra_pairs(x) -> list[tuple[str, str, str]]:
    """
    Return (variant_key, gene_key, original_pair_string).
    per_example_errors.csv uses strings like:
      variant::gene | variant2::gene2
    """
    out = []
    for item in parse_pipe_list(x):
        if "::" in item:
            v, g = item.split("::", 1)
            out.append((metric_key_variant(v), metric_key_variant(g), item))
    return out


def parse_prediction_variant_lines(prediction: str) -> list[tuple[str, str]]:
    """
    Parse raw predicted variant/gene strings from model output.
    This intentionally ignores <think> blocks because only lines starting with
    Variant: are parsed.
    """
    out = []
    for line in safe_str(prediction).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = line.strip()
        if not line.lower().startswith("variant:"):
            continue
        m = re.match(r"^Variant:\s*(?P<variant>.+?)\s*,\s*Gene:\s*(?P<gene>.+?)\s*$", line, flags=re.I)
        if not m:
            continue
        out.append((m.group("variant").strip(), m.group("gene").strip()))
    return out


def residue_regex() -> str:
    escaped = [re.escape(x) for x in RESIDUE_TERMS]
    # For one-letter nucleotides/amino acids, word boundaries are important.
    return r"(?:%s)" % "|".join(escaped)


RES = residue_regex()
ACTION = r"(?:%s)" % "|".join(map(re.escape, ACTION_TERMS))
POSITION = r"(?:%s)" % "|".join(map(re.escape, POSITION_TERMS))


TEXTUAL_VARIANT_PATTERNS = [
    # "A to C change", "G to T substitution", "glycine to leucine"
    re.compile(rf"\b{RES}\b\s*(?:-|–|—|\s)+to\s+(?:-|–|—|\s)*\b{RES}\b(?:\s+\w+){{0,8}}(?:{ACTION}|{POSITION}|\d)", re.I),
    # "valine-to-glutamic acid substitution"
    re.compile(rf"\b{RES}\b\s*[-–—]\s*to\s*[-–—]?\s*\b{RES}\b(?:\s+\w+){{0,8}}(?:{ACTION}|{POSITION}|\d)", re.I),
    # "substitution of valine to phenylalanine", "substitution of methionine for threonine"
    re.compile(rf"\b{ACTION}\s+of\s+\b{RES}\b(?:\s+\w+){{0,3}}\s+(?:to|for|by)\s+\b{RES}\b", re.I),
    # "arginine for methionine amino acid substitution"
    re.compile(rf"\b{RES}\b(?:\s+\w+){{0,3}}\s+(?:to|for|by)\s+\b{RES}\b(?:\s+\w+){{0,8}}\b{ACTION}\b", re.I),
    # "S194 was mutated to alanine", "serine was mutated to alanine"
    re.compile(rf"\b(?:[A-Za-z]?\d+[A-Za-z]?|{RES})\b\s+was\s+mutated\s+to\s+\b{RES}\b", re.I),
    # "arginine (CGC) to histidine (CAC) at codon 156"
    re.compile(rf"\b{RES}\b\s*\([^)]{{1,20}}\)\s+to\s+\b{RES}\b\s*\([^)]{{1,20}}\)(?:\s+\w+){{0,8}}(?:{POSITION}|\d)", re.I),
]


def looks_like_textual_substitution_or_change(raw_variant: str) -> bool:
    v = safe_str(raw_variant).strip()
    if not v:
        return False

    # Exclude compact HGVS/protein substitutions like V600E, c.1799T>A, GGT>GTT.
    # Those should remain evaluated normally.
    compact = compact_for_matching(v)
    if re.fullmatch(r"[a-z]\d+[a-z*]", compact, flags=re.I):
        return False
    if re.fullmatch(r"[cgpmnr]\.\w+", compact, flags=re.I):
        return False
    if ">" in v and not re.search(r"\b(to|change|substitution|mutation|mutated)\b", v, re.I):
        return False

    return any(p.search(v) for p in TEXTUAL_VARIANT_PATTERNS)


def variant_is_present_in_text(raw_variant: str, source_text: str) -> bool:
    """
    Require the model's textual variant to be present in the actual user text.
    This uses compact matching to tolerate spaces/punctuation differences.
    """
    rv = compact_for_matching(raw_variant)
    st = compact_for_matching(source_text)
    if not rv or not st:
        return False
    if rv in st:
        return True

    # Some predictions add a leading article not present in text.
    stripped = re.sub(r"^(a|an|the)", "", rv)
    if stripped and stripped in st:
        return True

    return False


def compute_prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def adjust_metrics(metrics: dict, errors: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    accepted_records = []

    accepted_variant_fp_keys_by_idx: dict[int, set[str]] = {}
    accepted_pair_fp_keys_by_idx: dict[int, set[str]] = {}

    for _, row in errors.iterrows():
        idx = int(row["idx"])
        source_text = safe_str(row.get("user", ""))

        extra_variants = set(metric_key_variant(x) for x in parse_pipe_list(row.get("extra_variants", "")))
        extra_pairs = parse_extra_pairs(row.get("extra_pairs", ""))

        if not extra_variants and not extra_pairs:
            continue

        raw_predicted = parse_prediction_variant_lines(row.get("prediction", ""))

        for raw_variant, raw_gene in raw_predicted:
            v_key = metric_key_variant(raw_variant)

            if v_key not in extra_variants:
                # Only adjust predictions currently counted as extra/FP variants.
                continue

            if not looks_like_textual_substitution_or_change(raw_variant):
                continue

            if not variant_is_present_in_text(raw_variant, source_text):
                continue

            accepted_variant_fp_keys_by_idx.setdefault(idx, set()).add(v_key)

            # Accept all extra FP pairs whose variant side is this accepted variant.
            accepted_pair_strings = []
            for pv, pg, original_pair in extra_pairs:
                if pv == v_key:
                    accepted_pair_fp_keys_by_idx.setdefault(idx, set()).add(original_pair)
                    accepted_pair_strings.append(original_pair)

            accepted_records.append(
                {
                    "idx": idx,
                    "accepted_raw_variant": raw_variant,
                    "accepted_variant_key": v_key,
                    "raw_gene": raw_gene,
                    "accepted_extra_pairs": " | ".join(accepted_pair_strings),
                    "reason": "textual substitution/change prediction present in source text",
                }
            )

    accepted_variant_fp = sum(len(v) for v in accepted_variant_fp_keys_by_idx.values())
    accepted_pair_fp = sum(len(v) for v in accepted_pair_fp_keys_by_idx.values())

    # Binary adjustment: if the original target is No variant but the prediction
    # contains at least one accepted textual variant present in the source text,
    # then under the adjusted definition the example is positive and detected.
    # Therefore binary FP becomes binary TP, not TN.
    binary_fp_to_tp = 0
    for _, row in errors.iterrows():
        idx = int(row["idx"])
        if not bool(row.get("error_binary", False)):
            continue

        target = safe_str(row.get("target", "")).strip().casefold()
        if target != "no variant":
            continue

        if accepted_variant_fp_keys_by_idx.get(idx):
            binary_fp_to_tp += 1

    adjusted = json.loads(json.dumps(metrics))

    # Variant-level adjustment: accepted textual variants move FP -> TP.
    vm = adjusted["variant_extraction_micro"]
    vm["tp"] = int(vm["tp"]) + accepted_variant_fp
    vm["fp"] = max(0, int(vm["fp"]) - accepted_variant_fp)
    vm.update(compute_prf(int(vm["tp"]), int(vm["fp"]), int(vm["fn"])))

    # Pair-level adjustment: accepted textual variant-gene pairs move FP -> TP.
    pm = adjusted["gene_variant_pair_extraction_micro"]
    pm["tp"] = int(pm["tp"]) + accepted_pair_fp
    pm["fp"] = max(0, int(pm["fp"]) - accepted_pair_fp)
    pm.update(compute_prf(int(pm["tp"]), int(pm["fp"]), int(pm["fn"])))

    # Binary adjustment: FP -> TP.
    bm = adjusted["binary_detection"]
    bm["tp"] = int(bm["tp"]) + binary_fp_to_tp
    bm["fp"] = max(0, int(bm["fp"]) - binary_fp_to_tp)
    bm.update(compute_prf(int(bm["tp"]), int(bm["fp"]), int(bm["fn"])))
    total = int(bm["tp"]) + int(bm["fp"]) + int(bm["fn"]) + int(bm["tn"])
    bm["accuracy"] = (int(bm["tp"]) + int(bm["tn"])) / total if total else 0.0

    if "data" in adjusted:
        adjusted["data"]["n_positive_true_adjusted"] = int(adjusted["data"].get("n_positive_true", 0)) + binary_fp_to_tp
        adjusted["data"]["n_negative_true_adjusted"] = max(0, int(adjusted["data"].get("n_negative_true", 0)) - binary_fp_to_tp)

    adjusted["adjustment"] = {
        "description": (
            "False-positive textual substitution/change predictions were counted as "
            "true positives when the predicted textual variant was present in the source text."
        ),
        "accepted_variant_fp_removed": accepted_variant_fp,
        "accepted_pair_fp_removed": accepted_pair_fp,
        "binary_fp_converted_to_tp": binary_fp_to_tp,
        "n_examples_with_accepted_adjustments": len(accepted_variant_fp_keys_by_idx),
        "policy": {
            "adds_true_positives": True,
            "removes_false_negatives": False,
            "changes_format_validity": False,
        },
    }

    return adjusted, pd.DataFrame(accepted_records)


def main():
    args = parse_args()

    metrics = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
    errors = pd.read_csv(args.errors_csv)

    adjusted, accepted = adjust_metrics(metrics, errors)

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(adjusted, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.accepted_cases_csv:
        accepted_path = Path(args.accepted_cases_csv)
        accepted_path.parent.mkdir(parents=True, exist_ok=True)
        accepted.to_csv(accepted_path, index=False)

    print(json.dumps(adjusted["adjustment"], indent=2, ensure_ascii=False))
    print(f"Wrote adjusted metrics to: {out}")
    if args.accepted_cases_csv:
        print(f"Wrote accepted adjustment cases to: {args.accepted_cases_csv}")


if __name__ == "__main__":
    main()
