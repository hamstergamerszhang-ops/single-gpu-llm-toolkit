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


if __name__ == "__main__":
    main()
