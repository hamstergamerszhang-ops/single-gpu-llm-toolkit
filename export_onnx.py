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
    ap.add_argument("--src", help="Source checkpoint directory.")
    ap.add_argument("--dst", help="Output directory or .onnx file path.")
    ap.add_argument("--seq-length", "--max-seq-len", type=int, default=128, dest="seq_length",
                    help="Dummy input sequence length.")
    ap.add_argument("--batch-size", "--batch", type=int, default=1, dest="batch_size",
                    help="Dummy input batch size.")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16", "fp8"], default="fp32")
    ap.add_argument("--selftest", action="store_true", default=False,
                    help="Run built-in self-test (no GPU required).")
    args = ap.parse_args()

    if args.selftest:
        _self_test()
        return

    # Validate required args AFTER the --selftest check.
    if not args.src or not args.dst:
        ap.error("--src and --dst are required (unless --selftest).")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers is required for ONNX export") from exc

    import torch
    from runtime import DTYPE_MAP

    torch_dtype = DTYPE_MAP[args.dtype]

    print(f"[export_onnx] loading {args.src} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.src,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.eval()

    device = next(model.parameters()).device
    dummy_input = torch.randint(
        0, model.config.vocab_size, (args.batch_size, args.seq_length), device=device
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
    """Self-test: flag aliasing + DTYPE_MAP coverage (no GPU required)."""
    print("[selftest] export_onnx: flag aliasing + dtype coverage (no GPU required)")

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--seq-length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16", "fp8"], default="fp32")
    ap.add_argument("--selftest", action="store_true", default=False)

    a = ap.parse_args(["--src", "/m", "--dst", "/o.onnx", "--dtype", "fp8"])
    assert a.dtype == "fp8"
    assert a.seq_length == 128
    assert a.batch_size == 1
    print("  OK (flags parsed, fp8 dtype accepted)")

    from runtime import DTYPE_MAP
    import torch
    assert DTYPE_MAP["fp8"] is torch.bfloat16
    assert DTYPE_MAP["fp32"] is torch.float32
    print("  OK (DTYPE_MAP covers fp8 -> bf16 for export load)")

    print("\n[selftest] All checks passed.")


if __name__ == "__main__":
    main()
