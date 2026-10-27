# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#!/usr/bin/env python3
"""
Evaluate WikiText-2 perplexity for either:
  - zero-shot/base model only
  - one PEFT/LoRA fine-tuned adapter: baseline or VT
  - one checkpoint directory

This script intentionally evaluates only WikiText-2 perplexity. It does not run E2E,
grammar scoring, or slot-error scoring.

Important VT behavior:
  - baseline adapters usually use the base tokenizer
  - VT adapters usually use the tokenizer saved in the adapter/checkpoint directory
  - VT adapter checkpoints may contain padded embedding/lm_head rows, e.g. 128280
    even when len(tokenizer) is 128276. The script inspects adapter weights and
    resizes the base model to the exact saved vocab size before loading PEFT.

Outputs under results_<run_name>/ by default:
  - predictions.csv: per WikiText segment with input, expected ground-truth token,
    model output continuation, per-example perplexity, and losses
  - summary.csv: aggregate perplexity for the run
  - metrics.json: metadata + aggregate metrics

Dependencies:
  pip install torch transformers accelerate peft datasets pandas tqdm pyarrow safetensors
Optional:
  pip install bitsandbytes
  only needed with --load_in_4bit
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except Exception:  # pragma: no cover
    BitsAndBytesConfig = None


# =============================================================================
# Basic helpers
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def build_quantization_config(load_in_4bit: bool, dtype: str):
    if not load_in_4bit:
        return None
    if BitsAndBytesConfig is None:
        raise RuntimeError("BitsAndBytesConfig unavailable. Install bitsandbytes/compatible transformers.")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=resolve_dtype(dtype),
    )


def primary_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def adapter_config_exists(path: Optional[str]) -> bool:
    return bool(path) and (Path(path) / "adapter_config.json").exists()


def checkpoint_step_from_path(path: Optional[str]) -> Optional[int]:
    if not path:
        return None
    match = re.search(r"checkpoint-(\d+)", str(path))
    return int(match.group(1)) if match else None


# =============================================================================
# Adapter vocab-size inspection: critical for VT checkpoints
# =============================================================================


def _shape_from_safetensors(path: Path) -> Dict[str, Tuple[int, ...]]:
    try:
        from safetensors import safe_open
    except Exception as exc:
        raise RuntimeError("Install safetensors to inspect adapter_model.safetensors") from exc

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
    Infer saved embedding/lm_head row count from a PEFT adapter checkpoint.

    This handles VT checkpoints where len(tokenizer) differs from the checkpoint's
    padded embedding size because training used resize_token_embeddings(...,
    pad_to_multiple_of=8).
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
            shapes = _shape_from_safetensors(candidate)
        else:
            shapes = _shape_from_torch_bin(candidate)
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
            if rows > 10000 and hidden >= 512:
                row_counts.append(rows)

    if not row_counts:
        return None

    vocab_size = max(row_counts)
    print(f"Inferred adapter checkpoint vocab size: {vocab_size}")
    return vocab_size


# =============================================================================
# Model loading
# =============================================================================


@dataclass
class LoadedModel:
    run_name: str
    model_type: str
    base_model: str
    adapter_path: Optional[str]
    tokenizer_path: str
    target_vocab_size: int
    model: Any
    tokenizer: Any



def load_tokenizer(tokenizer_path: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.init_kwargs["padding_side"] = "left"
    tokenizer.init_kwargs["truncation_side"] = "left"

    print(f"Loaded tokenizer from: {tokenizer_path}")
    print(f"len(tokenizer): {len(tokenizer)}")
    print(f"pad_token_id: {tokenizer.pad_token_id}")
    print(f"eos_token_id: {tokenizer.eos_token_id}")
    return tokenizer



def load_eval_model(
    run_name: str,
    model_type: str,
    base_model: str,
    adapter_path: Optional[str],
    tokenizer_path: str,
    dtype: str,
    load_in_4bit: bool,
    merge_adapter: bool,
) -> LoadedModel:
    print(f"\n=== Loading model for run: {run_name} ===")
    print(f"model_type: {model_type}")
    print(f"base_model: {base_model}")
    print(f"adapter_path: {adapter_path}")
    print(f"tokenizer_path: {tokenizer_path}")

    tokenizer = load_tokenizer(tokenizer_path)
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
        print(f"Resizing embeddings before adapter load: {old_vocab_size} -> {target_vocab_size}")
        model.resize_token_embeddings(target_vocab_size, pad_to_multiple_of=8)
        actual_vocab_size = model.get_input_embeddings().weight.shape[0]
        print(f"Actual embedding rows after resize: {actual_vocab_size}")
        if adapter_vocab_size is not None and actual_vocab_size != adapter_vocab_size:
            raise RuntimeError(
                "Embedding resize did not match adapter checkpoint vocab size: "
                f"adapter_vocab_size={adapter_vocab_size}, actual={actual_vocab_size}"
            )
    else:
        print("Model embeddings already match target vocab size.")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    if adapter_path:
        if not adapter_config_exists(adapter_path):
            raise FileNotFoundError(f"Expected adapter_config.json under: {adapter_path}")
        print(f"Loading PEFT adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        if merge_adapter:
            if load_in_4bit:
                print("Warning: merging adapter with --load_in_4bit may fail depending on PEFT/bitsandbytes versions.")
            print("Merging adapter into base model...")
            model = model.merge_and_unload()
    else:
        print("No adapter supplied: zero-shot/base-model evaluation.")

    model.eval()
    return LoadedModel(
        run_name=run_name,
        model_type=model_type,
        base_model=base_model,
        adapter_path=adapter_path,
        tokenizer_path=tokenizer_path,
        target_vocab_size=target_vocab_size,
        model=model,
        tokenizer=tokenizer,
    )


# =============================================================================
# WikiText-2 data preparation
# =============================================================================


@dataclass
class PPLExample:
    example_id: int
    text: str
    input_text: str
    ground_truth_output: str



def load_wikitext2_texts(num_examples: int, cache_dir: str, split: str = "test") -> List[str]:
    print("Loading WikiText-2 from Salesforce/wikitext...")
    ds = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split=split,
        cache_dir=cache_dir,
    )
    texts = [row["text"].strip() for row in ds if row["text"].strip()]
    return texts[:num_examples]



def make_ppl_examples(texts: List[str], input_ratio: float = 0.5, min_chars_for_split: int = 20) -> List[PPLExample]:
    examples: List[PPLExample] = []
    for idx, text in enumerate(texts):
        clean = text.strip()
        if not clean:
            continue
        if len(clean) < min_chars_for_split:
            input_text = clean
            ground_truth = ""
        else:
            split_idx = max(1, min(len(clean) - 1, int(len(clean) * input_ratio)))
            # Prefer splitting at whitespace after the initial target point.
            ws = clean.find(" ", split_idx)
            if ws != -1 and ws < len(clean) - 1:
                split_idx = ws
            input_text = clean[:split_idx].strip()
            ground_truth = clean[split_idx:].strip()
        examples.append(PPLExample(idx, clean, input_text, ground_truth))
    return examples


# =============================================================================
# Generation + perplexity
# =============================================================================


def get_eos_token_ids(tokenizer: Any) -> int | List[int]:
    eos_ids: List[int] = []
    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            eos_ids.extend(tokenizer.eos_token_id)
        else:
            eos_ids.append(tokenizer.eos_token_id)
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id >= 0 and eot_id != tokenizer.unk_token_id:
        eos_ids.append(eot_id)
    eos_ids = list(dict.fromkeys(eos_ids))
    return eos_ids[0] if len(eos_ids) == 1 else eos_ids


@torch.no_grad()
def generate_continuation(
    loaded: LoadedModel,
    input_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    tokenizer = loaded.tokenizer
    model = loaded.model
    encoded = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(primary_device(model))
    attention_mask = encoded["attention_mask"].to(primary_device(model))

    do_sample = temperature > 0.0
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": get_eos_token_ids(tokenizer),
        "use_cache": True,
    }
    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p

    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    new_tokens = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@torch.no_grad()
def compute_example_nll(loaded: LoadedModel, text: str, max_length: int, stride: int) -> Tuple[float, int, float]:
    """
    Return total negative log-likelihood, number of scored tokens, and perplexity
    for one text using sliding windows.
    """
    tokenizer = loaded.tokenizer
    model = loaded.model
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(primary_device(model))

    seq_len = input_ids.size(1)
    if seq_len <= 1:
        return 0.0, 0, float("nan")

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids_slice = input_ids[:, begin_loc:end_loc]
        target_ids = input_ids_slice.clone()
        target_ids[:, :-trg_len] = -100

        outputs = model(input_ids_slice, labels=target_ids)
        loss = outputs.loss
        valid_tokens = (target_ids != -100).sum().item()
        nll_sum += float(loss.item()) * valid_tokens
        n_tokens += int(valid_tokens)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = math.exp(nll_sum / max(1, n_tokens)) if n_tokens > 0 else float("nan")
    return nll_sum, n_tokens, ppl


@torch.no_grad()
def evaluate_wikitext2_perplexity(
    loaded: LoadedModel,
    examples: List[PPLExample],
    max_length: int,
    stride: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    generate_outputs: bool,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    total_nll = 0.0
    total_tokens = 0

    for ex in tqdm(examples, desc=f"WikiText-2 PPL: {loaded.run_name}"):
        nll, scored_tokens, example_ppl = compute_example_nll(loaded, ex.text, max_length, stride)
        total_nll += nll
        total_tokens += scored_tokens

        model_output = ""
        if generate_outputs:
            model_output = generate_continuation(
                loaded=loaded,
                input_text=ex.input_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        rows.append(
            {
                "run_name": loaded.run_name,
                "model_type": loaded.model_type,
                "example_id": ex.example_id,
                "input": ex.input_text,
                "expected_ground_truth_output": ex.ground_truth_output,
                "model_output": model_output,
                "full_ground_truth_text": ex.text,
                "nll_sum": nll,
                "scored_tokens": scored_tokens,
                "perplexity": example_ppl,
                "base_model": loaded.base_model,
                "adapter_path": loaded.adapter_path,
                "tokenizer_path": loaded.tokenizer_path,
                "target_vocab_size": loaded.target_vocab_size,
                "checkpoint_step": checkpoint_step_from_path(loaded.adapter_path),
            }
        )

    aggregate_ppl = math.exp(total_nll / max(1, total_tokens)) if total_tokens > 0 else float("nan")
    metrics = {
        "run_name": loaded.run_name,
        "model_type": loaded.model_type,
        "base_model": loaded.base_model,
        "adapter_path": loaded.adapter_path,
        "tokenizer_path": loaded.tokenizer_path,
        "target_vocab_size": loaded.target_vocab_size,
        "checkpoint_step": checkpoint_step_from_path(loaded.adapter_path),
        "num_examples": len(examples),
        "total_scored_tokens": total_tokens,
        "nll_sum": total_nll,
        "perplexity_wikitext2": aggregate_ppl,
    }
    return metrics, pd.DataFrame(rows)


# =============================================================================
# CLI
# =============================================================================


def default_run_name(model_type: str, adapter: Optional[str]) -> str:
    if model_type == "zero_shot":
        return "zeroshot"
    if adapter:
        step = checkpoint_step_from_path(adapter)
        if step is not None:
            return f"{model_type}_step_{step}"
    return model_type


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate only WikiText-2 perplexity for zero-shot or one LoRA/VT model.")

    parser.add_argument("--model_type", choices=["zero_shot", "baseline", "vt"], required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter", type=str, default=None, help="PEFT adapter/checkpoint path for baseline or VT.")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path. Defaults depend on model_type.")

    parser.add_argument("--num_examples", type=int, default=1000)
    parser.add_argument("--wikitext_cache_dir", type=str, default="data/wikitext2")
    parser.add_argument("--wikitext_split", type=str, default="test", choices=["train", "validation", "test"])

    parser.add_argument("--ppl_max_length", type=int, default=1024)
    parser.add_argument("--ppl_stride", type=int, default=512)

    parser.add_argument("--generate_outputs", action="store_true", help="Also generate continuations for the input prefix.")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--merge_adapter", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output_dir", type=str, default=None, help="Defaults to results_<run_name>.")

    args = parser.parse_args()
    set_seed(args.seed)

    if args.model_type == "zero_shot" and args.adapter:
        raise ValueError("--adapter must be omitted when --model_type zero_shot")
    if args.model_type in {"baseline", "vt"} and not args.adapter:
        raise ValueError(f"--adapter is required when --model_type {args.model_type}")

    run_name = args.run_name or default_run_name(args.model_type, args.adapter)
    output_dir = Path(args.output_dir or f"results_{run_name}")
    ensure_dir(output_dir)

    if args.tokenizer:
        tokenizer_path = args.tokenizer
    elif args.model_type == "vt":
        tokenizer_path = args.adapter
    else:
        tokenizer_path = args.base_model

    print("Configuration:")
    print(f"  run_name: {run_name}")
    print(f"  model_type: {args.model_type}")
    print(f"  base_model: {args.base_model}")
    print(f"  adapter: {args.adapter}")
    print(f"  tokenizer: {tokenizer_path}")
    print(f"  output_dir: {output_dir}")

    texts = load_wikitext2_texts(args.num_examples, args.wikitext_cache_dir, args.wikitext_split)
    examples = make_ppl_examples(texts)

    loaded = load_eval_model(
        run_name=run_name,
        model_type=args.model_type,
        base_model=args.base_model,
        adapter_path=args.adapter,
        tokenizer_path=tokenizer_path,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        merge_adapter=args.merge_adapter,
    )

    metrics, df = evaluate_wikitext2_perplexity(
        loaded=loaded,
        examples=examples,
        max_length=args.ppl_max_length,
        stride=args.ppl_stride,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        generate_outputs=args.generate_outputs,
    )

    predictions_csv = output_dir / "predictions.csv"
    summary_csv = output_dir / "summary.csv"
    metrics_json = output_dir / "metrics.json"

    df.to_csv(predictions_csv, index=False)
    pd.DataFrame([metrics]).to_csv(summary_csv, index=False)
    metrics_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== WikiText-2 Perplexity Summary ===")
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"\nSaved per-example results to: {predictions_csv}")
    print(f"Saved summary to: {summary_csv}")
    print(f"Saved metrics JSON to: {metrics_json}")

    del loaded.model
    del loaded.tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
