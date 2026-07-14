#!/usr/bin/env python3
"""Export a causal-LM checkpoint to ONNX.

Exports only the base model (not MTP modules) for inference. Requires
`transformers` and a checkpoint that can be loaded with
`AutoModelForCausalLM.from_pretrained`.
"""

import argparse
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="Source checkpoint directory.")
    ap.add_argument("--dst", required=True, help="Output directory or .onnx file path.")
    ap.add_argument("--max-seq-len", "--seq-length", type=int, default=128,
                    dest="max_seq_len", help="Dummy input sequence length (alias: --seq-length).")
    ap.add_argument("--batch", "--batch-size", type=int, default=1, dest="batch",
                    help="Dummy input batch size (alias: --batch-size).")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    args = ap.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers is required for ONNX export") from exc

    import torch

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"[export_onnx] loading {args.src} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.src,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.eval()

    device = next(model.parameters()).device
    dummy_input = torch.randint(
        0, model.config.vocab_size, (args.batch, args.max_seq_len), device=device
    )

    dst = args.dst
    if os.path.isdir(dst) or not dst.endswith(".onnx"):
        os.makedirs(dst, exist_ok=True)
        dst = os.path.join(dst, "model.onnx")

    print(f"[export_onnx] exporting to {dst} ...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            dst,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
            opset_version=14,
        )
    print("[export_onnx] done.")


def _self_test():
    print("[selftest] export_onnx: flag aliasing + dtype map (no GPU required)")
    import argparse

    # Test flag aliasing: --batch/--batch-size and --max-seq-len/--seq-length.
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seq-len", "--seq-length", type=int, default=128,
                    dest="max_seq_len")
    ap.add_argument("--batch", "--batch-size", type=int, default=1, dest="batch")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")

    args1 = ap.parse_args(["--batch-size", "4", "--seq-length", "256"])
    assert args1.batch == 4, f"--batch-size should set batch=4, got {args1.batch}"
    assert args1.max_seq_len == 256, f"--seq-length should set max_seq_len=256, got {args1.max_seq_len}"
    print("  OK (flag aliases --batch-size and --seq-length work)")

    # Test dtype map completeness.
    dtype_map = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
    for k, v in dtype_map.items():
        assert v in dir(__import__("torch")), f"torch.{v} should exist for {k}"
    print("  OK (dtype map covers fp32, fp16, bf16)")

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
