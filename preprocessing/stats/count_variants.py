# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import json
from pathlib import Path


def count_variants_jsonl(path: Path, name: str) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing test file: {path}")

    counter = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                line = json.loads(line)
                if name != "variantscape":
                    n_variants = line["metadata"]["n_variants"]
                else:
                    content = line["messages"][2]["content"]
                    n_variants = content.count("Variant:") if content != "No variant" else 0
                counter += n_variants
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return counter


def main():
    variantscape_path = Path("../data/processed")
    tmvar_path = Path("../data/processed_tmvar")
    varchat_path = Path("../data/processed_varchat")

    path_dict = {
        'variantscape': variantscape_path,
        'tmvar': tmvar_path,
        'varchat': varchat_path
    }

    for name, path in path_dict.items():
        print(f"\n{name.capitalize()}")
        tot_variants = 0
        for ds_split in ['train.jsonl', 'test.jsonl', 'val.jsonl']:
            n_split_variants = count_variants_jsonl(path / ds_split, name)
            print(f"{ds_split}: {n_split_variants}")
            tot_variants += n_split_variants
        print("Tot: ", tot_variants)


if __name__ == "__main__":
    main()