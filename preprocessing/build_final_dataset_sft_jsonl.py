# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import json
from pathlib import Path
from collections import defaultdict


def is_negative_example(obj: dict) -> bool:
    """Define label from assistant output."""
    assistant_content = obj["messages"][2]["content"].strip()
    return assistant_content == "No variant"


def safe_div(a, b):
    return a / b if b else 0.0


def merge_and_stats(dataset_info: dict[str, Path], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "input_dirs": {k: str(v) for k, v in dataset_info.items()},
        "output_dir": str(output_dir),
        "splits": {},
        "datasets": {},
        "totals": defaultdict(int),
    }

    for split in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        split_name = split.replace(".jsonl", "")
        out_file = output_dir / split

        split_stats = {
            "original_total": 0,
            "original_positive": 0,
            "original_negative": 0,
            "input_path": {},
            "output_path": str(out_file),
            "datasets": {}
        }

        with out_file.open("w", encoding="utf-8") as fout:

            for dataset_name, ds_dir in dataset_info.items():
                in_file = ds_dir / split
                split_stats["input_path"][dataset_name] = str(in_file)

                # init per-dataset stats inside split
                ds_split_stats = {
                    "total": 0,
                    "positive": 0,
                    "negative": 0,
                }

                if not in_file.exists():
                    print(f"Missing: {in_file}")
                    split_stats["datasets"][dataset_name] = ds_split_stats
                    continue

                with in_file.open("r", encoding="utf-8") as fin:
                    for line in fin:
                        if not line.strip():
                            continue

                        obj = json.loads(line)

                        negative = is_negative_example(obj)

                        # annotate provenance
                        obj.setdefault("metadata", {})
                        obj["metadata"]["source_dataset"] = dataset_name

                        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

                        # update split stats
                        split_stats["original_total"] += 1
                        ds_split_stats["total"] += 1

                        # update global totals
                        stats["totals"]["original_total"] += 1

                        if negative:
                            split_stats["original_negative"] += 1
                            ds_split_stats["negative"] += 1
                            stats["totals"]["original_negative"] += 1
                        else:
                            split_stats["original_positive"] += 1
                            ds_split_stats["positive"] += 1
                            stats["totals"]["original_positive"] += 1

                split_stats["datasets"][dataset_name] = ds_split_stats

        # add derived stats per split
        split_stats["original_positive_fraction"] = safe_div(
            split_stats["original_positive"], split_stats["original_total"]
        )
        split_stats["original_negative_fraction"] = safe_div(
            split_stats["original_negative"], split_stats["original_total"]
        )

        stats["splits"][split_name] = split_stats

        print(
            f"{split_name}: total={split_stats['original_total']} "
            f"pos={split_stats['original_positive']} "
            f"neg={split_stats['original_negative']}"
        )

    # finalize totals
    totals = stats["totals"]
    totals["original_positive_fraction"] = safe_div(
        totals["original_positive"], totals["original_total"]
    )
    totals["original_negative_fraction"] = safe_div(
        totals["original_negative"], totals["original_total"]
    )

    stats["totals"] = dict(totals)

    return stats


def main():
    # Update: want to held-out bronco dataset
    dataset_info = {
        "variantscape": Path("../data/processed"),
        "tmvar": Path("../data/processed_tmvar"),
        "internal": Path("../data/processed_internal"),
        #"bronco": Path("../data/processed_bronco"),
        "hereditary": Path("../data/processed_hereditary_disease_balanced"),
    }

    output_dir = Path("../data/processed_merged_1")
    stats_path = output_dir / "merge_stats.json"

    stats = merge_and_stats(dataset_info, output_dir)

    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\nSaved stats → {stats_path}")


if __name__ == "__main__":
    main()