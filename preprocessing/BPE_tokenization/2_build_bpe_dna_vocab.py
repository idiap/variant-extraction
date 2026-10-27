# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from pathlib import Path
from collections import Counter

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import BpeTrainer


# =========================
# Constants
# =========================

# One allele per line, for example:
# TAACCCAAAACCA
# AGAAGT
# CAGACTCTCAGCCAGTCCCTT
ALLELES_TXT = Path("../../data/clinvar/clinvar_alleles.txt")

OUTPUT_DIR = Path("../../data/tokenizers/clinvar_allele_bpe")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BPE_TOKENIZER_JSON = OUTPUT_DIR / "allele_bpe_tokenizer.json"
BPE_VOCAB_JSON = OUTPUT_DIR / "allele_bpe_vocab.json"

SELECTED_TOKENS_TXT = OUTPUT_DIR / "selected_bpe_dna_tokens.txt"
SELECTED_TOKENS_JSON = OUTPUT_DIR / "selected_bpe_dna_tokens.json"

# Auxiliary BPE vocabulary size.
# This is not the number of tokens you will add to Llama.
AUX_BPE_VOCAB_SIZE = 2048

# Minimum usage count in the allele corpus for a token to be retained.
MIN_TOKEN_FREQUENCY = 10

# Maximum number of selected BPE tokens to save.
MAX_TOKENS_TO_SELECT = 100

# Keep DNA-like BPE tokens only.
DNA_TOKEN_RE = re.compile(r"^[ACGTN]+$")

# Avoid adding very short tokens that the base tokenizer already handles well.
MIN_TOKEN_LENGTH = 2

# Avoid very long one-off allele-specific tokens.
MAX_TOKEN_LENGTH = 6

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
]


# =========================
# Loading
# =========================

def load_alleles(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing allele file: {path}")

    alleles = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            allele = line.strip().upper()

            if not allele:
                continue

            # Keep simple DNA alleles only.
            # This filters symbolic alleles such as <DEL>, <DUP>, etc.
            if DNA_TOKEN_RE.fullmatch(allele):
                alleles.append(allele)

    alleles = sorted(set(alleles))

    print(f"Loaded unique DNA alleles: {len(alleles):,}")

    if not alleles:
        raise RuntimeError("No valid DNA alleles found.")

    return alleles


# =========================
# BPE training
# =========================

def train_auxiliary_bpe(alleles: list[str]) -> Tokenizer:
    tokenizer = Tokenizer(
        BPE(
            unk_token="[UNK]",
        )
    )

    # Split DNA strings into individual bases first.
    # BPE then learns merges such as AC, GTT, CCCA, etc.
    tokenizer.pre_tokenizer = None

    trainer = BpeTrainer(
        vocab_size=AUX_BPE_VOCAB_SIZE,
        min_frequency=1,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=["A", "C", "G", "T", "N"],
        show_progress=True,
    )

    tokenizer.train_from_iterator(
        alleles,
        trainer=trainer,
    )

    tokenizer.save(str(BPE_TOKENIZER_JSON))

    BPE_VOCAB_JSON.write_text(
        json.dumps(tokenizer.get_vocab(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved auxiliary BPE tokenizer to: {BPE_TOKENIZER_JSON}")
    print(f"Saved auxiliary BPE vocab to: {BPE_VOCAB_JSON}")

    return tokenizer


# =========================
# Token selection
# =========================

def count_bpe_token_usage(tokenizer: Tokenizer, alleles: list[str]) -> Counter:
    counts = Counter()

    for allele in alleles:
        encoded = tokenizer.encode(allele)

        for token in encoded.tokens:
            if token in SPECIAL_TOKENS:
                continue

            counts[token] += 1

    return counts


def is_good_dna_bpe_token(token: str) -> bool:
    if token in SPECIAL_TOKENS:
        return False

    if not DNA_TOKEN_RE.fullmatch(token):
        return False

    if len(token) < MIN_TOKEN_LENGTH:
        return False

    if len(token) > MAX_TOKEN_LENGTH:
        return False

    # Optional: remove long homopolymers like AAAAAAA.
    # Comment this block out if homopolymer runs are important for your task.
    if len(set(token)) == 1 and len(token) > 6:
        return False

    return True


def select_bpe_tokens(tokenizer: Tokenizer, alleles: list[str]) -> list[dict]:
    vocab = tokenizer.get_vocab()
    usage_counts = count_bpe_token_usage(tokenizer, alleles)

    candidates = []

    for token, token_id in vocab.items():
        if not is_good_dna_bpe_token(token):
            continue

        frequency = usage_counts[token]

        if frequency < MIN_TOKEN_FREQUENCY:
            continue

        candidates.append(
            {
                "token": token,
                "aux_bpe_id": token_id,
                "length": len(token),
                "frequency": frequency,
                "score": frequency * len(token),
            }
        )

    candidates.sort(
        key=lambda x: (x["score"], x["frequency"], x["length"]),
        reverse=True,
    )

    selected = candidates[:MAX_TOKENS_TO_SELECT]

    print(f"BPE vocab size: {len(vocab):,}")
    print(f"Candidate DNA BPE tokens: {len(candidates):,}")
    print(f"Selected DNA BPE tokens: {len(selected):,}")

    return selected


def save_selected_tokens(selected: list[dict]) -> None:
    with SELECTED_TOKENS_TXT.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(row["token"] + "\n")

    SELECTED_TOKENS_JSON.write_text(
        json.dumps(selected, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved selected token list to: {SELECTED_TOKENS_TXT}")
    print(f"Saved selected token metadata to: {SELECTED_TOKENS_JSON}")


# =========================
# Main
# =========================

def main() -> None:
    alleles = load_alleles(ALLELES_TXT)

    tokenizer = train_auxiliary_bpe(alleles)

    selected = select_bpe_tokens(tokenizer, alleles)

    save_selected_tokens(selected)

    print("\nTop selected tokens:")
    for row in selected[:50]:
        print(
            f"{row['token']}\t"
            f"freq={row['frequency']}\t"
            f"len={row['length']}\t"
            f"score={row['score']}"
        )


if __name__ == "__main__":
    main()