# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#!/usr/bin/env python3
"""
train_contrastive_from_sft.py

Continue training a saved supervised fine-tuned LoRA adapter with contrastive learning.
Works for both baseline and VariantTokenizer adapters.

Loads:
  - base model from --base_model_name_or_path
  - tokenizer from --tokenizer_name_or_path if supplied, otherwise --adapter_path
  - LoRA adapter from --adapter_path with is_trainable=True

For VariantTokenizer adapters, the tokenizer saved next to the adapter contains the
extended vocabulary. The base model is resized to len(tokenizer) before loading the adapter.

Input contrastive JSONL:
{
  "context": "Title: ...\nAbstract: ...",
  "positive": "Variant: x, Gene: A",
  "negatives": ["Variant: x, Gene: B", ...]
}
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:  # pragma: no cover
    safe_load_file = None
from peft import PeftModel, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup

CONTEXT_EMBED_PREFIX = "Represent this biomedical text for genetic variant extraction:\n"
TARGET_EMBED_PREFIX = "Represent this extracted variant-gene answer:\n"


@dataclass
class ContrastiveExample:
    context: str
    positive: str
    negatives: list[str]
    metadata: dict


class ContrastiveDataset(Dataset):
    def __init__(self, path: Path, max_negatives: int | None = None):
        self.examples = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                context = str(row["context"]).strip()
                positive = str(row["positive"]).strip()
                negatives = [str(x).strip() for x in row.get("negatives", []) if str(x).strip()]
                if max_negatives is not None:
                    negatives = negatives[:max_negatives]
                if context and positive and negatives:
                    self.examples.append(ContrastiveExample(context, positive, negatives, row.get("metadata", {})))
        if not self.examples:
            raise RuntimeError(f"No valid contrastive examples in {path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate(batch: list[ContrastiveExample]) -> dict:
    return {
        "contexts": [x.context for x in batch],
        "positives": [x.positive for x in batch],
        "negatives": [x.negatives for x in batch],
        "metadata": [x.metadata for x in batch],
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_name_or_path", required=True)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--tokenizer_name_or_path", default=None)
    p.add_argument("--train_file", required=True)
    p.add_argument("--valid_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--log_jsonl", default=None)
    p.add_argument("--max_context_length", type=int, default=2048)
    p.add_argument("--max_target_length", type=int, default=256)
    p.add_argument("--max_negatives", type=int, default=8)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--max_grad_norm", type=float, default=0.3)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_each_epoch", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dtype_from_args(args):
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float32


def load_tokenizer(path: str):
    kwargs = {"use_fast": True, "trust_remote_code": True}
    name = path.lower()
    if "mistral" in name or "ministral" in name:
        kwargs["fix_mistral_regex"] = True
    tok = AutoTokenizer.from_pretrained(path, **kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"
    return tok


def infer_adapter_saved_vocab_size(adapter_path: str) -> int | None:
    """
    Infer vocab dimension saved inside a PEFT adapter.

    Some models have padded embedding/lm_head matrices larger than len(tokenizer).
    Example: tokenizer length 151669, but model/lm_head vocab dimension 151936.
    If we resize down to len(tokenizer), adapter loading fails.
    """
    path = Path(adapter_path)
    candidate_files = [
        path / "adapter_model.safetensors",
        path / "adapter_model.bin",
        path / "pytorch_model.bin",
    ]

    state = None
    for f in candidate_files:
        if not f.exists():
            continue
        if f.suffix == ".safetensors":
            if safe_load_file is None:
                raise RuntimeError(
                    f"{f} exists but safetensors is not importable. "
                    "Install safetensors or use a .bin adapter."
                )
            state = safe_load_file(str(f), device="cpu")
        else:
            state = torch.load(str(f), map_location="cpu")
        break

    if state is None:
        return None

    preferred_substrings = [
        "lm_head.modules_to_save.default.weight",
        "lm_head.weight",
        "embed_tokens.modules_to_save.default.weight",
        "embed_tokens.weight",
    ]
    for needle in preferred_substrings:
        for key, value in state.items():
            if needle in key and hasattr(value, "shape") and len(value.shape) == 2:
                return int(value.shape[0])

    for key, value in state.items():
        if (
            hasattr(value, "shape")
            and len(value.shape) == 2
            and ("lm_head" in key or "embed_tokens" in key or "wte" in key)
        ):
            return int(value.shape[0])

    return None


def maybe_resize_for_tokenizer_and_adapter(model, tokenizer, adapter_path: str):
    """
    Resize only when necessary and never shrink below the adapter checkpoint size.

    Priority:
      1. adapter saved vocab size, if present
      2. tokenizer length, only if larger than current model embeddings
      3. current model embedding size
    """
    current_n = int(model.get_input_embeddings().weight.shape[0])
    tokenizer_n = int(len(tokenizer))
    adapter_n = infer_adapter_saved_vocab_size(adapter_path)

    target_n = current_n
    reason = "keeping current model vocab size"

    if adapter_n is not None:
        target_n = adapter_n
        reason = f"matching adapter saved vocab size {adapter_n}"
    elif tokenizer_n > current_n:
        target_n = tokenizer_n
        reason = f"expanding to tokenizer length {tokenizer_n}"

    if target_n != current_n:
        print(f"Resizing token embeddings: {current_n} -> {target_n} ({reason})")
        model.resize_token_embeddings(target_n, pad_to_multiple_of=8)
    else:
        print(
            f"Not resizing token embeddings: model={current_n}, "
            f"tokenizer={tokenizer_n}, adapter={adapter_n} ({reason})"
        )

    final_n = int(model.get_input_embeddings().weight.shape[0])
    if adapter_n is not None and final_n != adapter_n:
        raise RuntimeError(
            f"After resize, model vocab size is {final_n}, but adapter expects {adapter_n}. "
            "Check adapter_path/tokenizer_name_or_path consistency."
        )

    if tokenizer_n > final_n:
        raise RuntimeError(
            f"Tokenizer length {tokenizer_n} is larger than model embedding size {final_n}. "
            "This means tokenizer_name_or_path and adapter_path are inconsistent."
        )


def load_model_and_tokenizer(args):
    tokenizer_path = args.tokenizer_name_or_path or args.adapter_path
    tokenizer = load_tokenizer(tokenizer_path)

    qconf = None
    if args.load_in_4bit:
        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype_from_args(args),
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        quantization_config=qconf,
        device_map="auto",
        torch_dtype=None if args.load_in_4bit else dtype_from_args(args),
        trust_remote_code=True,
    )

    # Necessary for VT adapters, but do not shrink padded vocabularies.
    # Match the adapter's saved lm_head/embedding size when available.
    maybe_resize_for_tokenizer_and_adapter(model, tokenizer, args.adapter_path)

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model, tokenizer


def build_context_text(context: str) -> str:
    return CONTEXT_EMBED_PREFIX + context.strip()


def build_target_text(target: str) -> str:
    return TARGET_EMBED_PREFIX + target.strip()


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


def encode_texts(model, tokenizer, texts: list[str], max_length: int) -> torch.Tensor:
    device = next(model.parameters()).device
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length, add_special_tokens=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
    return mean_pool(out.hidden_states[-1], enc["attention_mask"])


def explicit_negative_infonce(context_emb: torch.Tensor, positive_emb: torch.Tensor, negative_embs: torch.Tensor, temperature: float):
    context_emb = F.normalize(context_emb, dim=-1)
    positive_emb = F.normalize(positive_emb, dim=-1)
    negative_embs = F.normalize(negative_embs, dim=-1)
    pos_logits = torch.sum(context_emb * positive_emb, dim=-1, keepdim=True)
    neg_logits = torch.einsum("bd,bnd->bn", context_emb, negative_embs)
    logits = torch.cat([pos_logits, neg_logits], dim=1) / temperature
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits, labels)
    with torch.no_grad():
        r1 = (logits.argmax(dim=1) == 0).float().mean().item()
        pos = pos_logits.mean().item()
        neg = neg_logits.mean().item()
    return loss, {"recall_at_1": r1, "mean_pos_cosine": pos, "mean_neg_cosine": neg}


def step_batch(model, tokenizer, batch, args):
    contexts = [build_context_text(x) for x in batch["contexts"]]
    positives = [build_target_text(x) for x in batch["positives"]]
    n_neg = min(len(x) for x in batch["negatives"])
    if args.max_negatives is not None:
        n_neg = min(n_neg, args.max_negatives)
    if n_neg <= 0:
        return None, None
    negatives_flat = []
    for negs in batch["negatives"]:
        negatives_flat.extend(build_target_text(x) for x in negs[:n_neg])
    context_emb = encode_texts(model, tokenizer, contexts, args.max_context_length)
    positive_emb = encode_texts(model, tokenizer, positives, args.max_target_length)
    negative_flat = encode_texts(model, tokenizer, negatives_flat, args.max_target_length)
    negative_embs = negative_flat.view(len(contexts), n_neg, -1)
    return explicit_negative_infonce(context_emb, positive_emb, negative_embs, args.temperature)


def train_epoch(model, tokenizer, loader, optimizer, scheduler, args, epoch, log_jsonl_path):
    model.train()
    total_loss = 0.0
    total_r1 = 0.0
    n = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, start=1):
        loss, metrics = step_batch(model, tokenizer, batch, args)
        if loss is None:
            continue
        (loss / args.gradient_accumulation_steps).backward()
        if step % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item()
        total_r1 += metrics["recall_at_1"]
        n += 1
        if step % args.logging_steps == 0:
            record = {
                "epoch": epoch,
                "step": step,
                "total_steps": len(loader),
                "train_loss_running": total_loss / max(n, 1),
                "train_recall_at_1_running": total_r1 / max(n, 1),
                "batch_mean_pos_cosine": metrics["mean_pos_cosine"],
                "batch_mean_neg_cosine": metrics["mean_neg_cosine"],
            }

            print(
                f"epoch={epoch} step={step}/{len(loader)} "
                f"loss={record['train_loss_running']:.4f} "
                f"r@1={record['train_recall_at_1_running']:.4f} "
                f"pos={record['batch_mean_pos_cosine']:.4f} "
                f"neg={record['batch_mean_neg_cosine']:.4f}"
            )

            with log_jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if n > 0 and len(loader) % args.gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    return {"train_loss": total_loss / max(n, 1), "train_recall_at_1": total_r1 / max(n, 1)}


@torch.no_grad()
def evaluate(model, tokenizer, loader, args):
    model.eval()
    total_loss = total_r1 = total_pos = total_neg = 0.0
    n = 0
    for batch in loader:
        loss, metrics = step_batch(model, tokenizer, batch, args)
        if loss is None:
            continue
        total_loss += loss.item()
        total_r1 += metrics["recall_at_1"]
        total_pos += metrics["mean_pos_cosine"]
        total_neg += metrics["mean_neg_cosine"]
        n += 1
    return {
        "eval_loss": total_loss / max(n, 1),
        "eval_recall_at_1": total_r1 / max(n, 1),
        "eval_mean_pos_cosine": total_pos / max(n, 1),
        "eval_mean_neg_cosine": total_neg / max(n, 1),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --bf16 or --fp16.")
    if not args.bf16 and not args.fp16:
        print("Warning: neither --bf16 nor --fp16 selected. Defaulting to fp16.")
        args.fp16 = True
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_jsonl_path = Path(args.log_jsonl) if args.log_jsonl else out / "train_step_logs.jsonl"
    train_ds = ContrastiveDataset(Path(args.train_file), max_negatives=args.max_negatives)
    val_ds = ContrastiveDataset(Path(args.valid_file), max_negatives=args.max_negatives)
    train_loader = DataLoader(train_ds, batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.per_device_eval_batch_size, shuffle=False, collate_fn=collate)
    print(f"Loaded contrastive train examples: {len(train_ds)}")
    print(f"Loaded contrastive valid examples: {len(val_ds)}")
    model, tokenizer = load_model_and_tokenizer(args)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    update_steps_per_epoch = max(1, math.ceil(len(train_loader) / args.gradient_accumulation_steps))
    total_steps = max(1, int(update_steps_per_epoch * args.num_train_epochs))
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    history = []
    best_eval_loss = float("inf")
    for epoch in range(1, int(args.num_train_epochs) + 1):
        train_metrics = train_epoch(model, tokenizer, train_loader, optimizer, scheduler, args, epoch, log_jsonl_path)
        eval_metrics = evaluate(model, tokenizer, val_loader, args)
        record = {"epoch": epoch, **train_metrics, **eval_metrics}
        history.append(record)
        print(json.dumps(record, indent=2))
        if eval_metrics["eval_loss"] < best_eval_loss:
            best_eval_loss = eval_metrics["eval_loss"]
            best_dir = out / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"Saved best adapter/tokenizer to {best_dir}")
        if args.save_each_epoch:
            epoch_dir = out / f"epoch_{epoch}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)
    final_dir = out / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    metadata = {
        "base_model_name_or_path": args.base_model_name_or_path,
        "adapter_path": args.adapter_path,
        "tokenizer_name_or_path": args.tokenizer_name_or_path or args.adapter_path,
        "train_file": args.train_file,
        "valid_file": args.valid_file,
        "temperature": args.temperature,
        "max_negatives": args.max_negatives,
        "history": history,
    }
    (out / "contrastive_training_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved final adapter/tokenizer to {final_dir}")
    print(f"Saved metadata to {out / 'contrastive_training_metadata.json'}")


if __name__ == "__main__":
    main()
