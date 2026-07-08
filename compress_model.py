#!/usr/bin/env python3
"""Compress (quantize) any model checkpoint — works on any AMD GPU or CPU.

Takes any HuggingFace-format checkpoint and produces a quantized version using
torchao's quantization APIs. Supports multiple compression levels:

  - **int8 weight-only**: ~2x smaller, no quality loss in practice. Uses
    torchao's `int8_weight_only()`. Works on ALL AMD cards (the dequantized
    matmul runs in bf16/fp16 — no special hardware needed).
  - **int4 weight-only**: ~4x smaller, tiny quality loss. Uses torchao's
    `int4_weight_only()`. Works on ALL AMD cards.
  - **fp8 weight-only**: ~2x smaller, no quality loss. Uses torchao's
    `float8_weight_only()`. Best throughput on MI300X/MI325X (native fp8
    compute), but the weights are dequantized to bf16 for the matmul on other
    cards, so it works everywhere.

The tool auto-detects the model's config layout (nested text_config for
Gemma-4, flat for Llama/Mistral/Qwen), dtype, and architecture. It uses
rocm_env.setup_rocm_env() for the gfx override, so the "every AMD device"
guarantee carries over.

Usage:
    python3 compress_model.py --src ./checkpoints/base_15b --dst ./checkpoints/base_15b_int4
    python3 compress_model.py --src ./checkpoints/model --dst ./checkpoints/model_int8 --quant int8

Self-test (no GPU/model required — exercises config detection + quantization
plan logic):
    python3 compress_model.py --selftest
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
        "hardware": "all AMD cards (dequantizes to bf16 for matmul)",
    },
    "int4": {
        "import_path": "torchao.quantization",
        "import_name": "int4_weight_only",
        "size_reduction": "~4x",
        "quality": "minimal loss",
        "hardware": "all AMD cards (dequantizes to bf16 for matmul)",
    },
    "fp8": {
        "import_path": "torchao.quantization",
        "import_name": "float8_weight_only",
        "size_reduction": "~2x",
        "quality": "no loss",
        "hardware": "all AMD cards (best on MI300X/MI325X with native fp8)",
    },
}


def detect_config_layout(cfg: dict) -> str:
    """Detect whether the config is nested (Gemma-4: text_config) or flat
    (Llama/Mistral/Qwen). Returns 'nested' or 'flat'."""
    if "text_config" in cfg and isinstance(cfg["text_config"], dict):
        return "nested"
    return "flat"


def get_model_info(cfg: dict) -> dict:
    """Extract model info from config.json, handling both nested and flat
    layouts. Returns a dict with: layout, hidden_size, num_layers, vocab_size,
    model_type, dtype."""
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
    """Build a quantization plan. Returns a dict describing what will happen."""
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
                    choices=list(QUANT_OPTIONS.keys()),
                    help="Quantization type: int8 (~2x), int4 (~4x), fp8 (~2x, best on MI300X).")
    ap.add_argument("--gfx-override", type=str, default=None,
                    help="Force HSA_OVERRIDE_GFX_VERSION (see rocm_env.py).")
    ap.add_argument("--hip-alloc-conf", type=str, default="max_split_size_mb:128",
                    help="PYTORCH_HIP_ALLOC_CONF value (pass 'none' to skip).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the quantization plan, write nothing.")
    args = ap.parse_args()

    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        raise SystemExit("ERROR: --src and --dst must differ.")

    # Load config to detect model info.
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

    # ROCm env bootstrap (same as train_cpt.py).
    from rocm_env import setup_rocm_env
    hip_conf = None if args.hip_alloc_conf.lower() == "none" else args.hip_alloc_conf
    setup_rocm_env(override=args.gfx_override, hip_alloc_conf=hip_conf)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"loading model from {args.src} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.src, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.src)

    # Apply quantization.
    try:
        from torchao.quantization import quantize_
        # torchao exposes int8_weight_only / int4_weight_only /
        # float8_weight_only as EITHER a plain function (older torchao: call
        # it, get back a Quantizer) OR a Config class (newer torchao: call it
        # -- i.e. instantiate it -- to get a Config instance). Both a
        # function and a class are `callable()` in Python (instantiating a
        # class IS calling it), so `quant_obj()` is the correct call in
        # EITHER case -- there's no actual branch to take here. A prior
        # version had `if callable(quant_obj): ... else: ...` with both
        # branches doing the identical `quantize_(model, quant_obj())`,
        # which is dead code: `callable(quant_obj)` is always True for
        # either API shape, so the `else` branch could never execute. The
        # one shape this WOULDN'T handle -- quant_obj already being a bare
        # instance rather than a function/class -- doesn't occur for any of
        # this tool's three supported quant names, so it isn't guarded here;
        # if torchao ever exposes one of them that way, this call raises
        # TypeError ("X object is not callable"), which is caught by the
        # `except Exception` below and surfaces as a clear quantization error
        # rather than silently doing the wrong thing.
        mod = __import__(plan["import_path"], fromlist=[plan["import_name"]])
        quant_obj = getattr(mod, plan["import_name"])
        quantize_(model, quant_obj())
    except ImportError:
        raise SystemExit(
            f"ERROR: torchao not installed — required for {args.quant} quantization. "
            f"Install with 'pip install torchao'."
        )
    except Exception as e:
        raise SystemExit(f"ERROR: quantization failed ({e}). The model architecture "
                         f"may not be compatible with {args.quant}.")

    log(f"quantization complete ({args.quant})")

    # Save the compressed model.
    os.makedirs(args.dst, exist_ok=True)
    model.save_pretrained(args.dst, safe_serialization=True)
    tokenizer.save_pretrained(args.dst)

    # Copy any custom modeling code.
    custom_src = os.path.join(args.src, "modeling_custom.py")
    if os.path.exists(custom_src):
        shutil.copy2(custom_src, os.path.join(args.dst, "modeling_custom.py"))

    log(f"saved compressed model -> {args.dst}")

    # Report size reduction.
    src_size = sum(os.path.getsize(os.path.join(args.src, f))
                   for f in os.listdir(args.src) if f.endswith(".safetensors"))
    dst_size = sum(os.path.getsize(os.path.join(args.dst, f))
                   for f in os.listdir(args.dst) if f.endswith(".safetensors"))
    if src_size > 0:
        reduction = src_size / dst_size
        log(f"size: {src_size/1024**3:.2f}GB -> {dst_size/1024**3:.2f}GB "
            f"({reduction:.1f}x reduction)")
    log("done.")


def _self_test():
    print("[selftest] compress_model: config detection + quantization planning")

    # detect_config_layout: nested (Gemma-4) vs flat (Llama).
    nested_cfg = {"model_type": "gemma4", "text_config": {"hidden_size": 3072}}
    assert detect_config_layout(nested_cfg) == "nested"
    flat_cfg = {"model_type": "llama", "hidden_size": 4096}
    assert detect_config_layout(flat_cfg) == "flat"
    print("  OK (detect_config_layout: nested vs flat)")

    # get_model_info handles both layouts.
    info_nested = get_model_info(nested_cfg)
    assert info_nested["layout"] == "nested"
    assert info_nested["hidden_size"] == 3072
    assert info_nested["model_type"] == "gemma4"
    info_flat = get_model_info(flat_cfg)
    assert info_flat["layout"] == "flat"
    assert info_flat["hidden_size"] == 4096
    print("  OK (get_model_info: extracts fields from both layouts)")

    # plan_quantization: all three options.
    for q in ("int8", "int4", "fp8"):
        plan = plan_quantization(q, info_flat)
        assert plan["quant"] == q
        assert "size_reduction" in plan
        assert "hardware" in plan
    print("  OK (plan_quantization: int8/int4/fp8 all produce valid plans)")

    # Invalid quant raises.
    try:
        plan_quantization("int2", info_flat)
        assert False, "should have raised"
    except SystemExit:
        pass
    print("  OK (invalid quant type raises SystemExit)")

    print("\n[selftest] All checks passed (no GPU/model required — run with a "
          "real checkpoint on AMD hardware for actual quantization).")


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
