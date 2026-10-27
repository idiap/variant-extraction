# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import pandas as pd
from collections import Counter
from pathlib import Path


CLINVAR_FILE = "../../resources/variant_summary.txt.gz"

TARGET_TOP_N = 100 # None
# Keep the most frequent genes until their cumulative count
# covers this fraction of all gene occurrences.
TARGET_PERCENTILE = 0.2

OUTPUT_FILE = Path("../../resources/top_genes.txt")


def main():
    df = pd.read_csv(
        CLINVAR_FILE,
        sep="\t",
        compression="gzip",
    )

    # Expand entries such as:
    # "CRIPAK;LOC126806945;UVSSA"
    # into:
    # ["CRIPAK", "LOC126806945", "UVSSA"]
    gene_symbols = []

    for value in df["GeneSymbol"].dropna():
        if "covers" in value:
            continue

        # Unique genes within this row
        genes = {
            gene.strip()
            for gene in value.split(";")
            if gene.strip()
        }

        gene_symbols.extend(genes)

    # Count how many times each gene appears.
    freq = Counter(gene_symbols)

    # Genes sorted from most frequent to least frequent.
    sorted_gene_counts = freq.most_common()

    if TARGET_TOP_N != None:
        selected_genes = [gene for gene, _ in sorted_gene_counts][:TARGET_TOP_N]

        print(
            f"Selected {len(selected_genes)} genes "
            f"of all occurrences."
        )
    else:
        # Total number of gene occurrences in the dataset.
        total_count = sum(freq.values())

        # Number of occurrences we want to cover.
        target_count = total_count * TARGET_PERCENTILE

        # Select genes until the cumulative frequency reaches
        # the desired percentile of all occurrences.
        cumulative_count = 0
        selected_genes = []

        for gene, count in sorted_gene_counts:
            selected_genes.append(gene)
            cumulative_count += count

            if cumulative_count >= target_count:
                break

        print(
            f"Selected {len(selected_genes)} genes "
            f"covering {cumulative_count / total_count:.2%} "
            f"of all occurrences."
        )

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for gene in selected_genes:
            f.write(gene + "\n")


if __name__ == "__main__":
    main()