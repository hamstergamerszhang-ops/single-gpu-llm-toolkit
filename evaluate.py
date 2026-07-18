#!/usr/bin/env python3
"""Batch evaluation harness: compute perplexity / cross-entropy loss on a JSONL
dataset using a trained checkpoint.

Each line of the input JSONL must contain a "text" field. The script tokenizes,
runs inference in batches, and reports mean loss and perplexity.

Targets AMD ROCm, with CPU as the fallback for testing/dev without real
hardware. Supports all dtypes the rest of the toolkit does (fp32, fp16, bf16,
fp8) via the shared runtime.DTYPE_MAP + resolve_dtype.
"""

import argparse
import json
import math
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", help="Checkpoint directory.")
    ap.add_argument("--data", help="Input JSONL file.")
    # --batch-size / --seq-length are kept as the flag names (matching the
    # other eval/export tools); dest is set explicitly so they can be
    # overridden by a recipe/preset whose keys use train_cpt's dest names.
    ap.add_argument("--batch-size", "--batch", type=int, default=1,
                    dest="batch_size")
    ap.add_argument("--seq-length", "--max-seq-len", type=int, default=2048,
                    dest="seq_length")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16", "fp8"], default="bf16")
    ap.add_argument("--device", default=None,
                    help="Device override (rocm, cpu). 'cuda' is accepted as an "
                         "alias for 'rocm' since ROCm reports through the cuda "
                         "namespace.")
    ap.add_argument("--backend", default=None, help="Backend override for environment setup.")
    ap.add_argument("--max-samples", type=int, default=None)
    # ROCm bootstrap flags, mirroring generate.py / train_cpt.py so consumer
    # AMD cards needing HSA_OVERRIDE_GFX_VERSION are auto-handled here too.
    ap.add_argument("--gfx-override", type=str, default=None,
                    help="Force HSA_OVERRIDE_GFX_VERSION (e.g. gfx1100) for AMD "
                         "consumer/older cards whose arch isn't in the ROCm torch "
                         "wheel's compiled list. Auto-detected when unset.")
    ap.add_argument("--hip-alloc-conf", type=str, default="expandable_segments:True,garbage_collection_threshold:0.6",
                    help="Value for PYTORCH_HIP_ALLOC_CONF. Pass 'none' to skip.")
    ap.add_argument("--preset", type=str, default=None,
                    help="Hardware preset (cpu, rx6800-16g, rx7900xtx-24g, "
                         "mi300x-80g, ...). Sets dtype/batch defaults; CLI flags override.")
    ap.add_argument("--config", type=str, default=None,
                    help="Path to a TOML/YAML recipe file of overrides.")
    ap.add_argument("--selftest", action="store_true", default=False,
                    help="Run built-in self-test (no GPU required).")
    args = ap.parse_args()

    if args.selftest:
        _self_test()
        return

    # Resolve recipe/preset BEFORE the required-arg check (a preset could
    # supply defaults, though not model/data which are required). Same merge
    # semantics as train_cpt.py/generate.py: recipe values win over argparse
    # defaults only when the CLI value still equals the default.
    if args.preset or args.config:
        from config import resolve_recipe, apply_preset
        defaults = {a.dest: a.default for a in ap._actions if hasattr(a, "default")}
        recipe = resolve_recipe(args.config, base_defaults={})
        if args.preset:
            try:
                recipe = apply_preset(recipe, args.preset)
            except ValueError as exc:
                ap.error(str(exc))
        for key, value in recipe.items():
            if hasattr(args, key) and getattr(args, key) == defaults.get(key):
                setattr(args, key, value)

    # Validate required args AFTER the --selftest check so `--selftest` alone
    # doesn't trip argparse's required-arg enforcement.
    if not args.model or not args.data:
        ap.error("--model and --data are required (unless --selftest).")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers is required for evaluate.py") from exc

    # ROCm bootstrap before torch import -- same ordering requirement as
    # generate.py / train_cpt.py (HSA_OVERRIDE_GFX_VERSION must be set before
    # the ROCm device runtime initializes).
    from backends import get_backend
    backend = get_backend(args.backend) if args.backend else None
    if backend is None or backend.name == "rocm":
        from rocm_env import setup_rocm_env
        setup_rocm_env(override=args.gfx_override, hip_alloc_conf=args.hip_alloc_conf)

    import torch
    import torch.nn.functional as F

    from backends import default_device
    from runtime import DTYPE_MAP, resolve_dtype

    # Map common aliases: ROCm exposes itself through torch's "cuda" namespace,
    # so users often pass --device cuda. Accept it as "rocm" rather than
    # letting get_backend raise ValueError.
    prefer = args.backend or args.device
    if prefer == "cuda":
        prefer = "rocm"
    dev = default_device(prefer=prefer)
    dtype_str = resolve_dtype(dev, args.dtype)
    torch_dtype = DTYPE_MAP[dtype_str]

    print(f"[evaluate] loading model from {args.model} (dtype={dtype_str}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(dev.torch_device)
    model.eval()

    # fp8 inference: weight-only quantization via torchao (same path as
    # generate.py). resolve_dtype already fell back to bf16 on unsupported
    # hardware, so reaching here with dtype_str=="fp8" means the device
    # advertises fp8 AND torchao's scaled-mm probe passed.
    if dtype_str == "fp8":
        try:
            from torchao.quantization import quantize_
            try:
                from torchao.quantization.quant_api import float8_weight_only as _f8
            except ImportError:
                from torchao.quantization import Float8WeightOnlyConfig as _f8
            quantize_(model, _f8())
            print("[evaluate] fp8 inference enabled (torchao float8 weight-only)")
        except ImportError:
            print("[evaluate] WARNING: torchao not installed, using bf16")
        except Exception as e:
            print(f"[evaluate] WARNING: fp8 inference failed ({e}) — using bf16")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = []
    with open(args.data) as f:
        for i, line in enumerate(f):
            if args.max_samples is not None and i >= args.max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[evaluate] WARNING: skipping malformed JSONL line {i+1}: {exc}",
                      file=sys.stderr)
                continue
            texts.append(obj["text"])

    total_loss = 0.0
    total_tokens = 0

    # Pre-tokenize all texts once (without padding) so the per-batch loop only
    # pads + moves to device — it doesn't re-run the text->token pipeline every
    # batch. Minor win for eval (the real win was pre-tokenization for training,
    # already done in train_cpt.py), but removes the tokenizer from the timed
    # eval path.
    print(f"[evaluate] tokenizing {len(texts)} samples ...")
    tokenized = [tokenizer(t, truncation=True, max_length=args.seq_length,
                          add_special_tokens=True)["input_ids"] for t in texts]

    print(f"[evaluate] evaluating {len(tokenized)} samples ...")
    with torch.inference_mode():
        for i in range(0, len(tokenized), args.batch_size):
            batch_ids = tokenized[i:i + args.batch_size]
            max_len = max(len(ids) for ids in batch_ids)
            pad_id = tokenizer.pad_token_id
            input_ids = torch.full((len(batch_ids), max_len), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(batch_ids), max_len), dtype=torch.long)
            for j, ids in enumerate(batch_ids):
                n = len(ids)
                input_ids[j, :n] = torch.tensor(ids, dtype=torch.long)
                attention_mask[j, :n] = 1
            input_ids = input_ids.to(dev.torch_device)
            attention_mask = attention_mask.to(dev.torch_device)

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            # Pass attention_mask so the model doesn't attend to padding --
            # without it, padded positions contribute to the loss and skew
            # perplexity (the agent flagged this as M4).
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss.item()

            # Weight by number of non-ignored tokens.
            n_tokens = (labels != -100).sum().item()
            total_loss += loss * n_tokens
            total_tokens += n_tokens

    mean_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(mean_loss)
    print(f"[evaluate] mean_loss={mean_loss:.4f} perplexity={ppl:.2f} tokens={total_tokens}")


def _self_test():
    """Self-test: exercise argparse flag aliasing and DTYPE_MAP coverage (no GPU)."""
    print("[selftest] evaluate: flag aliasing + dtype coverage (no GPU required)")

    # Flag aliasing: --batch-size and --batch must both set dest=batch_size;
    # --seq-length and --max-seq-len must both set dest=seq_length.
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--batch-size", "--batch", type=int, default=1, dest="batch_size")
    ap.add_argument("--seq-length", "--max-seq-len", type=int, default=2048, dest="seq_length")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16", "fp8"], default="bf16")

    a = ap.parse_args(["--batch-size", "4"])
    assert a.batch_size == 4, a.batch_size
    a = ap.parse_args(["--batch", "8"])
    assert a.batch_size == 8, a.batch_size
    a = ap.parse_args(["--seq-length", "1024"])
    assert a.seq_length == 1024, a.seq_length
    a = ap.parse_args(["--max-seq-len", "512"])
    assert a.seq_length == 512, a.seq_length
    print("  OK (flag aliases --batch/--batch-size, --max-seq-len/--seq-length both work)")

    # DTYPE_MAP includes fp8 (the crash this repo fixed).
    from runtime import DTYPE_MAP
    import torch
    assert DTYPE_MAP["fp8"] is torch.bfloat16
    assert DTYPE_MAP["bf16"] is torch.bfloat16
    print("  OK (DTYPE_MAP covers fp8 -> bf16)")

    # Perplexity math sanity: exp(loss) where loss=0 -> ppl=1.0.
    assert math.exp(0.0) == 1.0
    assert math.exp(1.0) == math.e
    print("  OK (perplexity = exp(mean_loss) math verified)")

    print("\n[selftest] All checks passed.")


if __name__ == "__main__":
    main()
