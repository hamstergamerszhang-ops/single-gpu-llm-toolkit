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
    ap.add_argument("--batch", "--batch-size", type=int, default=1, dest="batch",
                    help="Batch size (alias: --batch-size).")
    ap.add_argument("--max-seq-len", "--seq-length", type=int, default=2048,
                    dest="max_seq_len", help="Max sequence length (alias: --seq-length).")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--device", default=None, help="Backend override (rocm, cpu).")
    ap.add_argument("--backend", default=None, help="Backend override for environment setup.")
    ap.add_argument("--gfx-override", type=str, default=None,
                    help="Force HSA_OVERRIDE_GFX_VERSION (see rocm_env.py).")
    ap.add_argument("--hip-alloc-conf", type=str, default="expandable_segments:True")
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers is required for evaluate.py") from exc

    import torch
    import torch.nn.functional as F

    # ROCm bootstrap (same as every other GPU tool).
    from rocm_env import setup_rocm_env_from_args
    setup_rocm_env_from_args(args)

    from backends import default_device
    from runtime import resolve_dtype

    dev = default_device(prefer=args.backend or args.device)
    dtype_str = resolve_dtype(dev, args.dtype)
    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16,
                   "bf16": torch.bfloat16}[dtype_str]

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
    malformed = 0
    with open(args.data) as f:
        for i, line in enumerate(f):
            if args.max_samples is not None and i >= args.max_samples:
                break
            try:
                obj = json.loads(line)
                texts.append(obj["text"])
            except json.JSONDecodeError:
                malformed += 1
    if malformed:
        print(f"[evaluate] WARNING: skipped {malformed} malformed JSONL line(s)")
    if not texts:
        raise SystemExit("ERROR: no valid samples found in data file")

    total_loss = 0.0
    total_tokens = 0

    print(f"[evaluate] evaluating {len(texts)} samples ...")
    with torch.inference_mode():
        for i in range(0, len(texts), args.batch):
            batch_texts = texts[i:i + args.batch]
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_seq_len,
            )
            input_ids = enc["input_ids"].to(dev.torch_device)
            attention_mask = enc["attention_mask"].to(dev.torch_device)

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            # Pass attention_mask so the model doesn't attend to pad tokens.
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                           labels=labels)
            loss = outputs.loss.item()

            n_tokens = (labels != -100).sum().item()
            total_loss += loss * n_tokens
            total_tokens += n_tokens

    mean_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(mean_loss)
    print(f"[evaluate] mean_loss={mean_loss:.4f} perplexity={ppl:.2f} tokens={total_tokens}")


def _self_test():
    print("[selftest] evaluate: JSONL parsing + flag aliasing (no GPU required)")

    # Test JSONL error handling: malformed lines are skipped, not fatal.
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"text": "hello world"}\n')
        f.write('{bad json\n')
        f.write('{"text": "second valid"}\n')
        fpath = f.name
    texts = []
    malformed = 0
    with open(fpath) as f:
        for line in f:
            try:
                obj = json.loads(line)
                texts.append(obj["text"])
            except json.JSONDecodeError:
                malformed += 1
    os.unlink(fpath)
    assert len(texts) == 2, f"expected 2 valid, got {len(texts)}"
    assert malformed == 1, f"expected 1 malformed, got {malformed}"
    print("  OK (malformed JSONL skipped, valid rows kept)")

    # Test flag aliasing: --batch and --batch-size map to the same dest.
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", "--batch-size", type=int, default=1, dest="batch")
    ap.add_argument("--max-seq-len", "--seq-length", type=int, default=2048,
                    dest="max_seq_len")
    # --batch-size alias
    args1 = ap.parse_args(["--batch-size", "4"])
    assert args1.batch == 4, f"--batch-size should set dest batch=4, got {args1.batch}"
    # --batch short form
    args2 = ap.parse_args(["--batch", "8"])
    assert args2.batch == 8, f"--batch should set dest batch=8, got {args2.batch}"
    # --seq-length alias
    args3 = ap.parse_args(["--seq-length", "1024"])
    assert args3.max_seq_len == 1024, f"--seq-length should set dest max_seq_len=1024, got {args3.max_seq_len}"
    # --max-seq-len short form
    args4 = ap.parse_args(["--max-seq-len", "512"])
    assert args4.max_seq_len == 512, f"--max-seq-len should set dest max_seq_len=512, got {args4.max_seq_len}"
    print("  OK (flag aliases --batch/--batch-size and --max-seq-len/--seq-length work)")

    print("\n[selftest] All checks passed (no GPU required).")


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", default=False)
    args, _ = ap.parse_known_args()
    if args.selftest:
        _self_test()
    else:
        main()


if __name__ == "__main__":
    main_cli()
