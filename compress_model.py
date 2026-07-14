#!/usr/bin/env python3
"""Compress (quantize) any model checkpoint.

Takes any HuggingFace-format checkpoint and produces a quantized version using
torchao's quantization APIs. Targets AMD ROCm, with CPU as the fallback for
testing/dev without real hardware.

Usage:
    python3 compress_model.py --src ./checkpoints/base_15b --dst ./checkpoints/base_15b_int4
    python3 compress_model.py --src ./checkpoints/model --dst ./checkpoints/model_int8 --quant int8
"""

import argparse
import json
import os
import shutil


def log(msg: str):
    print(f"[compress] {msg}", flush=True)


QUANT_OPTIONS = {
    "int8": {
        "import_path": "torchao.quantization",
        "import_name": "int8_weight_only",
        "size_reduction": "~2x",
        "quality": "no perceptible loss",
        "hardware": "all devices (dequantizes to bf16/fp16 for matmul)",
    },
    "int4": {
        "import_path": "torchao.quantization",
        "import_name": "int4_weight_only",
        "size_reduction": "~4x",
        "quality": "minimal loss",
        "hardware": "all devices (dequantizes to bf16/fp16 for matmul)",
    },
    "fp8": {
        "import_path": "torchao.quantization",
        "import_name": "float8_weight_only",
        "size_reduction": "~2x",
        "quality": "negligible loss",
        "hardware": "best on AMD GPUs with native fp8 (MI300X/MI300A/MI325X/MI350)",
    },
}


def detect_config_layout(cfg: dict) -> str:
    if "text_config" in cfg and isinstance(cfg["text_config"], dict):
        return "nested"
    return "flat"


def get_model_info(cfg: dict) -> dict:
    layout = detect_config_layout(cfg)
    tc = cfg.get("text_config", cfg) if layout == "nested" else cfg
    return {
        "layout": layout,
        "hidden_size": tc.get("hidden_size"),
        "num_layers": tc.get("num_hidden_layers"),
        "vocab_size": tc.get("vocab_size"),
        "model_type": cfg.get("model_type", "unknown"),
        "dtype": cfg.get("torch_dtype", "bfloat16"),
    }


def plan_quantization(quant: str, model_info: dict) -> dict:
    if quant not in QUANT_OPTIONS:
        raise SystemExit(f"ERROR: unknown quantization '{quant}'. Options: {list(QUANT_OPTIONS)}")
    opt = QUANT_OPTIONS[quant]
    return {
        "quant": quant,
        "import_path": opt["import_path"],
        "import_name": opt["import_name"],
        "size_reduction": opt["size_reduction"],
        "quality": opt["quality"],
        "hardware": opt["hardware"],
        "model_type": model_info["model_type"],
        "layout": model_info["layout"],
        "hidden_size": model_info["hidden_size"],
        "num_layers": model_info["num_layers"],
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", required=True, help="Source model dir (read-only).")
    ap.add_argument("--dst", required=True, help="Output dir for compressed model.")
    ap.add_argument("--quant", type=str, default="int4",
                    choices=list(QUANT_OPTIONS.keys()))
    ap.add_argument("--gfx-override", type=str, default=None)
    ap.add_argument("--hip-alloc-conf", type=str, default="expandable_segments:True")
    ap.add_argument("--backend", type=str, default=None,
                    choices=["rocm", "cpu"],
                    help="Compute backend to use (auto-detected if unset).")
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the quantization plan, write nothing.")
    args = ap.parse_args()

    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        raise SystemExit("ERROR: --src and --dst must differ.")

    src_cfg_path = os.path.join(args.src, "config.json")
    if not os.path.exists(src_cfg_path):
        raise SystemExit(f"ERROR: {src_cfg_path} not found")
    with open(src_cfg_path) as f:
        cfg = json.load(f)

    model_info = get_model_info(cfg)
    plan = plan_quantization(args.quant, model_info)

    log(f"source: {args.src}")
    log(f"  model_type: {model_info['model_type']}")
    log(f"  config layout: {model_info['layout']}")
    log(f"  hidden_size: {model_info['hidden_size']}")
    log(f"  num_layers: {model_info['num_layers']}")
    log(f"  dtype: {model_info['dtype']}")
    log(f"quantization: {args.quant}")
    log(f"  size reduction: {plan['size_reduction']}")
    log(f"  quality: {plan['quality']}")
    log(f"  hardware: {plan['hardware']}")

    if args.dry_run:
        log("DRY RUN — nothing written.")
        return

    # ROCm bootstrap before torch import.
    from backends import get_backend
    backend = get_backend(args.backend) if args.backend else None
    if backend is None or backend.name == "rocm":
        from rocm_env import setup_rocm_env
        from rocm_env import setup_rocm_env_from_args
        setup_rocm_env_from_args(args)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from backends import BackendDevice
    from runtime import resolve_dtype

    dev = BackendDevice(backend=args.backend, index=args.device_index)
    if not dev.backend.is_available():
        raise SystemExit(f"ERROR: backend {dev.name} is not available.")

    dtype_str = resolve_dtype(dev, model_info["dtype"].replace("bfloat16", "bf16").replace("float16", "fp16").replace("float32", "fp32"))
    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype_str]

    log(f"loading model from {args.src} on {dev} (dtype={dtype_str}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.src, torch_dtype=torch_dtype, trust_remote_code=True
    )
    model.to(dev.torch_device)
    tokenizer = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)

    try:
        from torchao.quantization import quantize_
        mod = __import__(plan["import_path"], fromlist=[plan["import_name"]])
        quant_obj = getattr(mod, plan["import_name"])
        quantize_(model, quant_obj())
    except ImportError:
        raise SystemExit(
            f"ERROR: torchao not installed — required for {args.quant} quantization."
        )
    except Exception as e:
        raise SystemExit(f"ERROR: quantization failed ({e}).")

    log(f"quantization complete ({args.quant})")

    os.makedirs(args.dst, exist_ok=True)
    model.save_pretrained(args.dst, safe_serialization=True)
    tokenizer.save_pretrained(args.dst)

    custom_src = os.path.join(args.src, "modeling_custom.py")
    if os.path.exists(custom_src):
        shutil.copy2(custom_src, os.path.join(args.dst, "modeling_custom.py"))

    log(f"saved compressed model -> {args.dst}")

    src_size = sum(os.path.getsize(os.path.join(args.src, f))
                   for f in os.listdir(args.src) if f.endswith(".safetensors"))
    dst_size = sum(os.path.getsize(os.path.join(args.dst, f))
                   for f in os.listdir(args.dst) if f.endswith(".safetensors"))
    if src_size > 0 and dst_size > 0:
        reduction = src_size / dst_size
        log(f"size: {src_size/1024**3:.2f}GB -> {dst_size/1024**3:.2f}GB "
            f"({reduction:.1f}x reduction)")
    log("done.")


def _self_test():
    print("[selftest] compress_model: config detection + quantization planning")

    nested_cfg = {"model_type": "gemma4", "text_config": {"hidden_size": 3072}}
    assert detect_config_layout(nested_cfg) == "nested"
    flat_cfg = {"model_type": "llama", "hidden_size": 4096}
    assert detect_config_layout(flat_cfg) == "flat"
    print("  OK (detect_config_layout: nested vs flat)")

    info_nested = get_model_info(nested_cfg)
    assert info_nested["layout"] == "nested"
    assert info_nested["hidden_size"] == 3072
    info_flat = get_model_info(flat_cfg)
    assert info_flat["layout"] == "flat"
    assert info_flat["hidden_size"] == 4096
    print("  OK (get_model_info: extracts fields from both layouts)")

    for q in ("int8", "int4", "fp8"):
        plan = plan_quantization(q, info_flat)
        assert plan["quant"] == q
    print("  OK (plan_quantization: int8/int4/fp8 all produce valid plans)")

    try:
        plan_quantization("int2", info_flat)
        assert False, "should have raised"
    except SystemExit:
        pass
    print("  OK (invalid quant type raises SystemExit)")

    print("\n[selftest] All checks passed.")


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
