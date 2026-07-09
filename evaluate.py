#!/usr/bin/env python3
"""Batch evaluation harness: compute perplexity / cross-entropy loss on a JSONL
dataset using a trained checkpoint.

Each line of the input JSONL must contain a "text" field. The script tokenizes,
runs inference in batches, and reports mean loss and perplexity.
"""

import argparse
import json
import math
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Checkpoint directory.")
    ap.add_argument("--data", required=True, help="Input JSONL file.")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seq-length", type=int, default=2048)
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--device", default=None, help="Device override (cpu, cuda, mps, xpu).")
    ap.add_argument("--backend", default=None, help="Backend override for environment setup.")
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers is required for evaluate.py") from exc

    import torch
    import torch.nn.functional as F

    from backends import default_device
    from runtime import resolve_dtype

    dev = default_device(prefer=args.backend or args.device)
    dtype_str = resolve_dtype(dev, args.dtype)
    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype_str]

    print(f"[evaluate] loading model from {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(dev.torch_device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = []
    with open(args.data) as f:
        for i, line in enumerate(f):
            if args.max_samples is not None and i >= args.max_samples:
                break
            obj = json.loads(line)
            texts.append(obj["text"])

    total_loss = 0.0
    total_tokens = 0

    print(f"[evaluate] evaluating {len(texts)} samples ...")
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i:i + args.batch_size]
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.seq_length,
            )
            input_ids = enc["input_ids"].to(dev.torch_device)
            attention_mask = enc["attention_mask"].to(dev.torch_device)

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss.item()

            # Weight by number of non-ignored tokens.
            n_tokens = (labels != -100).sum().item()
            total_loss += loss * n_tokens
            total_tokens += n_tokens

    mean_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(mean_loss)
    print(f"[evaluate] mean_loss={mean_loss:.4f} perplexity={ppl:.2f} tokens={total_tokens}")


if __name__ == "__main__":
    main()
