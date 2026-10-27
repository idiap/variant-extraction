# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#!/usr/bin/env python3
"""
Evaluate one LLM or PEFT adapter for linguistic generation quality.

This version is designed for LoRA/PEFT checkpoints where:
  - the baseline adapter may use the base tokenizer, and
  - the VT adapter may use an expanded tokenizer saved inside the adapter directory.

Key fix for VT adapters:
  Before calling PeftModel.from_pretrained(...), the script inspects the adapter
  weights and resizes the base model embeddings to the exact vocabulary size saved
  in the adapter checkpoint. This handles cases where len(tokenizer) is 128276 but
  adapter weights are padded to 128280 rows.

Metrics:
  1. neural_grammar_edits_per_100_tokens on E2E NLG generations
  2. slot_error_rate on E2E NLG generations
  3. perplexity_wikitext2 on Salesforce/wikitext, unless --skip_ppl is passed

The summary reports metrics for one requested model only; it does not compare models.

Dependencies:
  pip install torch transformers accelerate peft datasets pandas tqdm requests sentencepiece protobuf pyarrow safetensors

Optional:
  pip install bitsandbytes
  only needed when --load_in_4bit is used.
"""

from __future__ import annotations

import argparse
import difflib
import gc
import json
import math
import os
import random
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
import torch
from datasets import Dataset, load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

try:
    from transformers import BitsAndBytesConfig
except Exception:  # pragma: no cover
    BitsAndBytesConfig = None


# =============================================================================
# Reproducibility / text utilities
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9£$€%\.\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_token_count(text: str) -> int:
    return max(1, len(re.findall(r"\b\w+\b", text)))


def mixed_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def ensure_dir(path: str | Path) -> None:
    path = Path(path)
    if str(path):
        path.mkdir(parents=True, exist_ok=True)


def safe_mean(values: Iterable[float]) -> float:
    vals = [v for v in values if not math.isnan(float(v))]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


# =============================================================================
# Direct E2E NLG loading
# =============================================================================


def download_file(url: str, path: str | Path, timeout: int = 120) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def extract_zip(zip_path: str | Path, extract_dir: str | Path) -> None:
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)
    ensure_dir(extract_dir)
    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)


def load_e2e_dataset(split: str = "test", cache_dir: str = "data/e2e_nlg") -> Dataset:
    """
    Load the original E2E NLG CSV files directly from the official GitHub release.
    This avoids deprecated Hugging Face dataset scripts.
    """
    cache_dir_p = Path(cache_dir)
    ensure_dir(cache_dir_p)

    url = "https://github.com/tuetschek/e2e-dataset/releases/download/v1.0.0/e2e-dataset.zip"
    zip_path = cache_dir_p / "e2e-dataset.zip"
    extract_dir = cache_dir_p / "e2e-dataset"

    if not zip_path.exists():
        download_file(url, zip_path)

    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        extract_zip(zip_path, extract_dir)

    split_map = {
        "train": "trainset.csv",
        "validation": "devset.csv",
        "valid": "devset.csv",
        "dev": "devset.csv",
        "test": "testset_w_refs.csv",
    }
    if split not in split_map:
        raise ValueError(f"Unsupported E2E split '{split}'. Use train, dev/validation, or test.")

    target_file = split_map[split]
    csv_path: Optional[Path] = None
    for root, _, files in os.walk(extract_dir):
        if target_file in files:
            csv_path = Path(root) / target_file
            break

    if csv_path is None:
        available = []
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if file.endswith(".csv"):
                    available.append(str(Path(root) / file))
        raise FileNotFoundError(
            f"Could not find {target_file}. Available CSV files:\n" + "\n".join(available)
        )

    df = pd.read_csv(csv_path)
    rename_map = {}
    if "mr" in df.columns:
        rename_map["mr"] = "meaning_representation"
    if "ref" in df.columns:
        rename_map["ref"] = "human_reference"
    df = df.rename(columns=rename_map)

    if "meaning_representation" not in df.columns:
        raise KeyError(f"Could not find MR column. Columns are: {list(df.columns)}")

    df = df[df["meaning_representation"].notna()].reset_index(drop=True)
    return Dataset.from_pandas(df, preserve_index=False)


def extract_mr(example: Dict[str, Any]) -> str:
    for key in ["meaning_representation", "mr", "input", "source"]:
        if key in example and example[key] is not None:
            return str(example[key])
    raise KeyError(f"Could not find MR field. Example keys: {list(example.keys())}")


def parse_e2e_mr(mr: str) -> Dict[str, str]:
    pairs = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*([^\]]+?)\s*\]", mr)
    return {slot.strip(): value.strip() for slot, value in pairs}


def make_prompt(mr: str) -> str:
    return (
        "Generate one fluent English sentence that describes the following meaning representation.\n\n"
        f"Meaning representation:\n{mr}\n\n"
        "Rules:\n"
        "- Do not add information that is not present.\n"
        "- Do not omit important information.\n"
        "- Output only the sentence."
    )


def build_slot_value_inventory(dataset: Dataset) -> Dict[str, set]:
    inventory: Dict[str, set] = {}
    for ex in dataset:
        try:
            slots = parse_e2e_mr(extract_mr(ex))
        except Exception:
            continue
        for slot, value in slots.items():
            value_norm = normalize_text(value)
            if value_norm:
                inventory.setdefault(slot, set()).add(value_norm)
    return inventory


def compute_slot_error_rate(
    output: str,
    expected_slots: Dict[str, str],
    slot_inventory: Dict[str, set],
) -> Tuple[float, Dict[str, int]]:
    out_norm = normalize_text(output)
    missing = 0
    wrong = 0
    hallucinated = 0

    for slot, expected_value in expected_slots.items():
        expected_norm = normalize_text(expected_value)
        if not expected_norm:
            continue

        if expected_norm in out_norm:
            continue

        missing += 1
        known_values = slot_inventory.get(slot, set())
        alternatives = [v for v in known_values if v and v != expected_norm]
        if any(v in out_norm for v in alternatives):
            wrong += 1
            hallucinated += 1

    denom = max(1, len(expected_slots))
    ser = (missing + wrong + hallucinated) / denom
    return ser, {
        "missing": missing,
        "wrong": wrong,
        "hallucinated": hallucinated,
        "num_slots": len(expected_slots),
    }


# =============================================================================
# WikiText-2 loading
# =============================================================================


def load_wikitext_examples(
    num_examples: int,
    cache_dir: str = "data/wikitext2",
    ppl_text_file: Optional[str] = None,
) -> List[str]:
    if ppl_text_file:
        print(f"Loading perplexity text from local file: {ppl_text_file}")
        with open(ppl_text_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        return texts[:num_examples]

    print("Loading WikiText-2 from Salesforce/wikitext...")
    ds = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="test",
        cache_dir=cache_dir,
    )
    texts = [row["text"].strip() for row in ds if row["text"].strip()]
    return texts[:num_examples]


# =============================================================================
# Model loading helpers
# =============================================================================


@dataclass
class LoadedModel:
    name: str
    model: Any
    tokenizer: Any
    tokenizer_path: str
    adapter_path: Optional[str]
    target_vocab_size: int


def resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def compute_dtype(dtype: str) -> torch.dtype:
    return resolve_dtype(dtype)


def build_quantization_config(load_in_4bit: bool, dtype: str):
    if not load_in_4bit:
        return None
    if BitsAndBytesConfig is None:
        raise RuntimeError("BitsAndBytesConfig is unavailable. Install transformers with bitsandbytes support.")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype(dtype),
    )


def get_token_id(tokenizer: Any, token: str) -> Optional[int]:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        return None
    if isinstance(token_id, int) and token_id >= 0 and token_id != tokenizer.unk_token_id:
        return token_id
    return None


def build_eos_token_ids(tokenizer: Any) -> int | List[int]:
    eos_ids: List[int] = []
    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            eos_ids.extend(tokenizer.eos_token_id)
        else:
            eos_ids.append(tokenizer.eos_token_id)

    # Useful for Llama-3.x chat models.
    eot_id = get_token_id(tokenizer, "<|eot_id|>")
    if eot_id is not None:
        eos_ids.append(eot_id)

    eos_ids = list(dict.fromkeys(eos_ids))
    if not eos_ids:
        raise ValueError("Could not determine eos_token_id.")
    return eos_ids[0] if len(eos_ids) == 1 else eos_ids


def adapter_config_exists(path: Optional[str]) -> bool:
    return bool(path) and (Path(path) / "adapter_config.json").exists()


def _shape_from_safetensors(path: Path) -> Dict[str, Tuple[int, ...]]:
    try:
        from safetensors import safe_open
    except Exception as exc:
        raise RuntimeError("safetensors is required to inspect adapter_model.safetensors") from exc

    shapes: Dict[str, Tuple[int, ...]] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            shapes[key] = tuple(f.get_tensor(key).shape)
    return shapes


def _shape_from_torch_bin(path: Path) -> Dict[str, Tuple[int, ...]]:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    shapes: Dict[str, Tuple[int, ...]] = {}
    if isinstance(state, dict):
        for key, value in state.items():
            if torch.is_tensor(value):
                shapes[key] = tuple(value.shape)
    return shapes


def infer_adapter_vocab_size(adapter_path: Optional[str]) -> Optional[int]:
    """
    Inspect PEFT adapter weights to find the embedding/lm_head row count.

    This is the critical VT fix. Some tokenizers report len(tokenizer)=128276,
    while training saved resized/padded embeddings with 128280 rows because
    resize_token_embeddings(..., pad_to_multiple_of=8) was used. PeftModel then
    requires the base model to have 128280 rows before the adapter is loaded.
    """
    if not adapter_path:
        return None

    path = Path(adapter_path)
    if not path.exists():
        return None

    candidates = [
        path / "adapter_model.safetensors",
        path / "adapter_model.bin",
        path / "pytorch_model.bin",
    ]

    shapes: Dict[str, Tuple[int, ...]] = {}
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix == ".safetensors":
            shapes.update(_shape_from_safetensors(candidate))
        else:
            shapes.update(_shape_from_torch_bin(candidate))
        if shapes:
            break

    if not shapes:
        return None

    row_counts: List[int] = []
    interesting = (
        "embed_tokens",
        "lm_head",
        "wte",
        "word_embeddings",
        "token_adapter",
        "modules_to_save",
    )
    for key, shape in shapes.items():
        if len(shape) != 2:
            continue
        if any(name in key for name in interesting):
            rows, hidden = int(shape[0]), int(shape[1])
            # Avoid accidentally treating small LoRA matrices as vocab matrices.
            if rows > 10000 and hidden >= 512:
                row_counts.append(rows)

    if not row_counts:
        return None

    vocab_size = max(row_counts)
    print(f"Inferred adapter checkpoint vocab size from weights: {vocab_size}")
    return vocab_size


def load_tokenizer_for_eval(tokenizer_path: str, prompt_style: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.init_kwargs["padding_side"] = "left"
    tokenizer.init_kwargs["truncation_side"] = "left"

    print(f"Loaded tokenizer from: {tokenizer_path}")
    print(f"Tokenizer vocabulary size len(tokenizer): {len(tokenizer)}")
    print(f"tokenizer.pad_token: {repr(tokenizer.pad_token)}")
    print(f"tokenizer.pad_token_id: {tokenizer.pad_token_id}")
    print(f"tokenizer.eos_token: {repr(tokenizer.eos_token)}")
    print(f"tokenizer.eos_token_id: {tokenizer.eos_token_id}")
    if prompt_style == "llama3":
        print(f"<|eot_id|> id: {get_token_id(tokenizer, '<|eot_id|>')}")
    return tokenizer


def print_vt_metadata_if_present(tokenizer_path: str) -> None:
    metadata_path = Path(tokenizer_path) / "vt_tokenizer_metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            print("Loaded VT tokenizer metadata:")
            print(json.dumps(metadata, indent=2)[:2000])
        except Exception as exc:
            print(f"Warning: could not read {metadata_path}: {exc}")


def load_eval_model(
    name: str,
    base_model: str,
    adapter_path: Optional[str],
    tokenizer_path: str,
    dtype: str,
    load_in_4bit: bool,
    merge_adapter: bool,
    prompt_style: str,
) -> LoadedModel:
    """
    Load base model + optional LoRA adapter.

    Correct VT order:
      1. Load tokenizer from adapter/tokenizer path.
      2. Inspect adapter weights for saved/padded vocab size.
      3. Load base model.
      4. Resize embeddings to max(len(tokenizer), adapter_vocab_size).
      5. Load PeftModel.from_pretrained.
    """
    print(f"\n=== Loading {name} ===")
    tokenizer = load_tokenizer_for_eval(tokenizer_path, prompt_style)
    print_vt_metadata_if_present(tokenizer_path)

    adapter_vocab_size = infer_adapter_vocab_size(adapter_path) if adapter_config_exists(adapter_path) else None
    target_vocab_size = max(len(tokenizer), adapter_vocab_size or 0)

    quantization_config = build_quantization_config(load_in_4bit, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=None if load_in_4bit else resolve_dtype(dtype),
        quantization_config=quantization_config,
        trust_remote_code=True,
    )

    old_vocab_size = model.get_input_embeddings().weight.shape[0]
    if old_vocab_size != target_vocab_size:
        print(f"Resizing model embeddings before adapter load: {old_vocab_size} -> {target_vocab_size}")
        # Do not use len(tokenizer) here. Use target_vocab_size inferred from adapter weights.
        model.resize_token_embeddings(target_vocab_size, pad_to_multiple_of=8)
        actual_vocab_size = model.get_input_embeddings().weight.shape[0]
        print(f"Actual embedding rows after resize: {actual_vocab_size}")
        if adapter_vocab_size is not None and actual_vocab_size != adapter_vocab_size:
            raise RuntimeError(
                "Embedding resize did not match adapter checkpoint vocab size. "
                f"adapter_vocab_size={adapter_vocab_size}, actual_vocab_size={actual_vocab_size}. "
                "Remove pad_to_multiple_of or inspect your adapter checkpoint."
            )
    else:
        print("Model embeddings already match target vocab size.")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    if adapter_config_exists(adapter_path):
        print(f"Loading PEFT adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        if merge_adapter:
            if load_in_4bit:
                print("Warning: --merge_adapter with --load_in_4bit may not be supported for all models. Trying anyway.")
            print("Merging adapter into base model...")
            model = model.merge_and_unload()
    elif adapter_path:
        raise FileNotFoundError(f"Expected adapter_config.json under adapter path: {adapter_path}")
    else:
        print("No adapter path provided; evaluating base model only.")

    model.eval()
    return LoadedModel(
        name=name,
        model=model,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        adapter_path=adapter_path,
        target_vocab_size=target_vocab_size,
    )


def primary_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unload_model(loaded: LoadedModel) -> None:
    del loaded.model
    del loaded.tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Prompting / generation
# =============================================================================


def build_llama3_prompt_from_text(user_text: str, system_text: str = "") -> str:
    parts: List[str] = ["<|begin_of_text|>"]
    if system_text.strip():
        parts.append(
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_text.strip()}"
            "<|eot_id|>"
        )
    parts.append(
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_text.strip()}"
        "<|eot_id|>"
    )
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def format_prompt_for_model(
    tokenizer: Any,
    prompt: str,
    prompt_style: str,
    system_prompt: str = "",
) -> str:
    if prompt_style == "raw":
        return prompt
    if prompt_style == "llama3":
        return build_llama3_prompt_from_text(prompt, system_prompt)
    if prompt_style == "chat_template":
        if getattr(tokenizer, "chat_template", None):
            messages = []
            if system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt.strip()})
            messages.append({"role": "user", "content": prompt})
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt
    if prompt_style == "auto":
        if getattr(tokenizer, "chat_template", None):
            messages = []
            if system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt.strip()})
            messages.append({"role": "user", "content": prompt})
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt
    raise ValueError(f"Unsupported prompt_style: {prompt_style}")


def clean_generated_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    stop_markers = [
        "\nUser:",
        "\nSystem:",
        "\nAssistant:",
        "<|eot_id|>",
        "<|end_of_text|>",
        "<|start_header_id|>",
    ]
    for marker in stop_markers:
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text


@torch.no_grad()
def generate_text(
    loaded: LoadedModel,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    prompt_style: str,
    system_prompt: str,
) -> str:
    tokenizer = loaded.tokenizer
    model = loaded.model
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    formatted_prompt = format_prompt_for_model(tokenizer, prompt, prompt_style, system_prompt)
    encoded = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=4096,
        add_special_tokens=(prompt_style == "raw"),
    )
    input_ids = encoded["input_ids"].to(primary_device(model))
    attention_mask = encoded["attention_mask"].to(primary_device(model))

    do_sample = temperature > 0.0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": build_eos_token_ids(tokenizer),
        "use_cache": True,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_kwargs,
    )
    prompt_width = input_ids.shape[1]
    new_tokens = outputs[0][prompt_width:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return clean_generated_text(text)


# =============================================================================
# Java-free neural grammar scoring
# =============================================================================


class NeuralGrammarScorer:
    def __init__(
        self,
        model_name: str = "vennify/t5-base-grammar-correction",
        device: Optional[str] = None,
        max_input_length: int = 256,
        max_new_tokens: int = 256,
        num_beams: int = 4,
    ):
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        print(f"Loading neural grammar scorer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if device is None:
            # CPU avoids occupying GPU memory needed by the 8B evaluator.
            device = "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def correct(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        source = "grammar: " + text
        inputs = self.tokenizer(
            source,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        ).to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            do_sample=False,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def score(self, text: str) -> Tuple[float, int, int, str]:
        corrected = self.correct(text)
        original_tokens = mixed_tokenize(text)
        corrected_tokens = mixed_tokenize(corrected)
        matcher = difflib.SequenceMatcher(None, original_tokens, corrected_tokens)
        edits = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                edits += max(i2 - i1, j2 - j1)
        denom = max(1, len(original_tokens))
        return 100.0 * edits / denom, edits, denom, corrected


def init_grammar_scorer(
    skip_grammar: bool,
    grammar_model: str,
    grammar_device: Optional[str],
) -> Optional[NeuralGrammarScorer]:
    if skip_grammar:
        print("Skipping neural grammar scorer.")
        return None
    return NeuralGrammarScorer(model_name=grammar_model, device=grammar_device)


def neural_grammar_edits_per_100_tokens(
    text: str,
    scorer: Optional[NeuralGrammarScorer],
) -> Tuple[float, int, int, str]:
    if scorer is None:
        return float("nan"), 0, word_token_count(text), ""
    return scorer.score(text)


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_generation_on_e2e(
    loaded: LoadedModel,
    dataset: Dataset,
    slot_inventory: Dict[str, set],
    grammar_scorer: Optional[NeuralGrammarScorer],
    num_examples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    prompt_style: str,
    system_prompt: str,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    subset = dataset.select(range(min(num_examples, len(dataset))))

    for idx, ex in enumerate(tqdm(subset, desc=f"E2E generation: {loaded.name}")):
        mr = extract_mr(ex)
        slots = parse_e2e_mr(mr)
        prompt = make_prompt(mr)
        output = generate_text(
            loaded=loaded,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            prompt_style=prompt_style,
            system_prompt=system_prompt,
        )

        grammar_rate, grammar_edits, grammar_tokens, corrected_output = (
            neural_grammar_edits_per_100_tokens(output, grammar_scorer)
        )
        ser, ser_parts = compute_slot_error_rate(output, slots, slot_inventory)

        rows.append(
            {
                "model": loaded.name,
                "example_id": idx,
                "mr": mr,
                "output": output,
                "corrected_output": corrected_output,
                "neural_grammar_edits": grammar_edits,
                "grammar_token_count": grammar_tokens,
                "neural_grammar_edits_per_100_tokens": grammar_rate,
                "slot_error_rate": ser,
                "missing_slots": ser_parts["missing"],
                "wrong_slots": ser_parts["wrong"],
                "hallucinated_slots": ser_parts["hallucinated"],
                "num_slots": ser_parts["num_slots"],
                "generated_word_tokens": word_token_count(output),
                "tokenizer_path": loaded.tokenizer_path,
                "adapter_path": loaded.adapter_path,
                "target_vocab_size": loaded.target_vocab_size,
            }
        )

    df = pd.DataFrame(rows)
    metrics = {
        "neural_grammar_edits_per_100_tokens": safe_mean(df["neural_grammar_edits_per_100_tokens"].tolist()),
        "slot_error_rate": float(df["slot_error_rate"].mean()),
        "missing_slots_per_example": float(df["missing_slots"].mean()),
        "wrong_slots_per_example": float(df["wrong_slots"].mean()),
        "hallucinated_slots_per_example": float(df["hallucinated_slots"].mean()),
        "avg_generated_word_tokens": float(df["generated_word_tokens"].mean()),
    }
    return metrics, df


@torch.no_grad()
def compute_perplexity(
    loaded: LoadedModel,
    texts: List[str],
    max_length: int = 1024,
    stride: int = 512,
) -> float:
    tokenizer = loaded.tokenizer
    model = loaded.model

    full_text = "\n\n".join(texts)
    encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = encodings.input_ids.to(primary_device(model))

    seq_len = input_ids.size(1)
    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0

    for begin_loc in tqdm(range(0, seq_len, stride), desc=f"Perplexity: {loaded.name}"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        input_ids_slice = input_ids[:, begin_loc:end_loc]
        target_ids = input_ids_slice.clone()
        target_ids[:, :-trg_len] = -100

        outputs = model(input_ids_slice, labels=target_ids)
        loss = outputs.loss
        valid_tokens = (target_ids != -100).sum().item()
        nll_sum += loss.item() * valid_tokens
        n_tokens += valid_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / max(1, n_tokens)
    return float(math.exp(avg_nll))


def relative_change(after: float, before: float) -> float:
    if before == 0 or math.isnan(before) or math.isnan(after):
        return float("nan")
    return 100.0 * (after - before) / before


def evaluate_one_model(
    model_name: str,
    adapter_path: Optional[str],
    tokenizer_path: str,
    args: argparse.Namespace,
    e2e: Dataset,
    slot_inventory: Dict[str, set],
    wikitext_texts: Optional[List[str]],
    grammar_scorer: Optional[NeuralGrammarScorer],
) -> Tuple[Dict[str, float], pd.DataFrame]:
    loaded = load_eval_model(
        name=model_name,
        base_model=args.base_model,
        adapter_path=adapter_path,
        tokenizer_path=tokenizer_path,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        merge_adapter=args.merge_adapter,
        prompt_style=args.prompt_style,
    )

    print(f"Evaluating {model_name} generation...")
    metrics, df = evaluate_generation_on_e2e(
        loaded=loaded,
        dataset=e2e,
        slot_inventory=slot_inventory,
        grammar_scorer=grammar_scorer,
        num_examples=args.num_e2e_examples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        prompt_style=args.prompt_style,
        system_prompt=args.system_prompt,
    )

    if args.skip_ppl:
        metrics["perplexity_wikitext2"] = float("nan")
    else:
        if not wikitext_texts:
            raise ValueError("wikitext_texts is empty but --skip_ppl was not set.")
        print(f"Evaluating {model_name} perplexity...")
        metrics["perplexity_wikitext2"] = compute_perplexity(
            loaded,
            wikitext_texts,
            max_length=args.ppl_max_length,
            stride=args.ppl_stride,
        )

    unload_model(loaded)
    return metrics, df


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one model on E2E generation quality and WikiText-2 perplexity."
    )

    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")

    parser.add_argument(
        "--model_name",
        type=str,
        default="model",
        help="Name written to the output CSVs, e.g. zero_shot, baseline, or vt.",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Optional PEFT/LoRA adapter directory. Omit this for zero-shot/base-model evaluation.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Backward-compatible alias for --adapter. Use only for adapter directories.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help=(
            "Tokenizer path. Defaults to --adapter when an adapter is provided, "
            "otherwise defaults to --base_model. For a baseline adapter that uses "
            "the base tokenizer, pass --tokenizer equal to --base_model."
        ),
    )

    parser.add_argument("--num_e2e_examples", type=int, default=500)
    parser.add_argument("--num_ppl_examples", type=int, default=1000)

    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--load_in_4bit", action="store_true", help="Load the base model in 4-bit NF4.")
    parser.add_argument("--merge_adapter", action="store_true", help="Merge LoRA adapter after loading.")

    parser.add_argument(
        "--prompt_style",
        type=str,
        default="llama3",
        choices=["llama3", "chat_template", "auto", "raw"],
        help="Use llama3 for manual Llama-3 prompts. Use chat_template for Qwen/chat models.",
    )
    parser.add_argument("--system_prompt", type=str, default="")

    parser.add_argument("--skip_grammar", action="store_true")
    parser.add_argument("--grammar_model", type=str, default="vennify/t5-base-grammar-correction")
    parser.add_argument(
        "--grammar_device",
        type=str,
        default="cpu",
        help="Device for grammar scorer. Default cpu to preserve GPU memory.",
    )

    parser.add_argument("--skip_ppl", action="store_true")
    parser.add_argument("--ppl_text_file", type=str, default=None)
    parser.add_argument("--ppl_max_length", type=int, default=1024)
    parser.add_argument("--ppl_stride", type=int, default=512)

    parser.add_argument("--e2e_cache_dir", type=str, default="data/e2e_nlg")
    parser.add_argument("--wikitext_cache_dir", type=str, default="data/wikitext2")

    parser.add_argument("--output_csv", type=str, default="single_model_eval_results.csv")
    parser.add_argument("--summary_csv", type=str, default="single_model_eval_summary.csv")

    args = parser.parse_args()
    set_seed(args.seed)

    adapter_path = args.adapter or args.model
    tokenizer_path = args.tokenizer or adapter_path or args.base_model

    print("Configuration:")
    print(f"  model_name: {args.model_name}")
    print(f"  base_model: {args.base_model}")
    print(f"  adapter: {adapter_path}")
    print(f"  tokenizer: {tokenizer_path}")
    print(f"  prompt_style: {args.prompt_style}")
    print(f"  load_in_4bit: {args.load_in_4bit}")

    print("Loading E2E dataset...")
    e2e = load_e2e_dataset(split="test", cache_dir=args.e2e_cache_dir)
    e2e = e2e.select(range(min(args.num_e2e_examples, len(e2e))))

    print("Building slot inventory...")
    slot_inventory = build_slot_value_inventory(e2e)

    if args.skip_ppl:
        print("Skipping perplexity evaluation.")
        wikitext_texts = None
    else:
        print("Loading WikiText examples...")
        wikitext_texts = load_wikitext_examples(
            num_examples=args.num_ppl_examples,
            cache_dir=args.wikitext_cache_dir,
            ppl_text_file=args.ppl_text_file,
        )

    print("Initializing Java-free neural grammar scorer...")
    grammar_scorer = init_grammar_scorer(
        skip_grammar=args.skip_grammar,
        grammar_model=args.grammar_model,
        grammar_device=args.grammar_device,
    )

    metrics, detailed_df = evaluate_one_model(
        model_name=args.model_name,
        adapter_path=adapter_path,
        tokenizer_path=tokenizer_path,
        args=args,
        e2e=e2e,
        slot_inventory=slot_inventory,
        wikitext_texts=wikitext_texts,
        grammar_scorer=grammar_scorer,
    )

    detailed_df.to_csv(args.output_csv, index=False)

    metric_names = [
        "neural_grammar_edits_per_100_tokens",
        "slot_error_rate",
        "missing_slots_per_example",
        "wrong_slots_per_example",
        "hallucinated_slots_per_example",
        "avg_generated_word_tokens",
        "perplexity_wikitext2",
    ]

    summary_rows = []
    for metric in metric_names:
        summary_rows.append(
            {
                "model": args.model_name,
                "metric": metric,
                "value": metrics.get(metric, float("nan")),
                "direction": "diagnostic" if metric == "avg_generated_word_tokens" else "lower_is_better",
                "base_model": args.base_model,
                "adapter_path": adapter_path,
                "tokenizer_path": tokenizer_path,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.summary_csv, index=False)

    print("\n=== Summary ===")
    print(summary_df[["metric", "value", "direction"]].to_string(index=False))
    print(f"\nSaved detailed generations to: {args.output_csv}")
    print(f"Saved summary metrics to: {args.summary_csv}")

    print("\nInterpretation:")
    print("- neural_grammar_edits_per_100_tokens: lower usually means better grammar/fluency.")
    print("- slot_error_rate: lower means better semantic faithfulness to the meaning representation.")
    print("- perplexity_wikitext2: lower usually means better general language modeling quality.")
    print("- avg_generated_word_tokens is diagnostic, not inherently better or worse.")


if __name__ == "__main__":
    main()
