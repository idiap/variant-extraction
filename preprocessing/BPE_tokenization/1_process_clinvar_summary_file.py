# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import pandas as pd
import os
import re

OUTPUT_DIR = "../../data/clinvar"
OUTPUT_FILE = f"{OUTPUT_DIR}/clinvar_alleles.txt"

def extract_alleles(name):
    m = re.search(r"c\.(.*?)\s*\(p\.", name)
    if not m:
        return []
    return re.findall(r"[ACGT]+", m.group(1))

def main():
    print("Reading file")

    df = pd.read_csv(
        "../../resources/variant_summary.txt.gz",
        sep="\t",
        compression="gzip",
    )

    #
    # Original approach: extract alleles from VCF REF/ALT columns
    #
    # df = df[['Type', 'ReferenceAlleleVCF', 'AlternateAlleleVCF']].dropna()
    # df = df[
    #     (df['ReferenceAlleleVCF'] != 'na')
    #     & (df['AlternateAlleleVCF'] != 'na')
    # ]
    #
    # df = df[df['Type'] != 'single nucleotide variant']
    #
    # alleles = (
    #     list(df['ReferenceAlleleVCF'])
    #     + list(df['AlternateAlleleVCF'])
    # )

    #
    # New approach: extract alleles directly from HGVS strings in Name
    #
    df = df[['Type', 'Name']].dropna()

    # Keep same filtering as before
    df = df[df['Type'] != 'single nucleotide variant']

    alleles = []
    for name in df['Name']:
        alleles.extend(extract_alleles(name))

    alleles = sorted(set(alleles))

    print(f"Num unique alleles: {len(alleles)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        print(f"Saving to {OUTPUT_FILE}")
        for allele in alleles:
            f.write(f"{allele}\n")

    print("Done.")


if __name__ == "__main__":
    main()