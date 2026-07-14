#!/usr/bin/env python3
"""Auto-detect multi-GPU and run a model with pipeline parallelism.

Detects how many AMD GPUs are available, distributes the model across them
using an explicit, layer-balanced device map (pipeline parallelism — different
layers on different GPUs, with activations passed between them), and runs
streaming generation. If only 1 GPU is detected, falls back to single-GPU
generation.

Why explicit device_map (not device_map="auto"): on a node of identical GPUs
(the common AMD case — e.g. 8x MI300X), HF's `device_map="auto"` uses a greedy
memory-fit that can leave one GPU underloaded and another holding the LM head +
embeddings + last layers, producing an unbalanced pipeline with a large bubble.
A layer-count-balanced split across N identical GPUs gives each GPU the same
number of transformer layers, which minimizes the pipeline bubble and is what
you actually want on homogeneous AMD hardware. `build_explicit_device_map()`
produces that explicit map; the plan it returns is passed directly to
`from_pretrained(..., device_map=plan)`.

NOTE: despite the above, the CLI default is `--device-map auto` (not
`explicit`), for maximum compatibility — `auto` works for any HF model without
needing to detect the layer prefix. `build_explicit_device_map` now
auto-detects the layer prefix via `get_model_info_from_config` (checking
`model_type` for "gemma" to pick `model.language_model.layers` vs the standard
`model.layers`), so `--device-map explicit` works for both Gemma-4 and
standard Llama/Mistral/Qwen layouts on homogeneous nodes where you want the
balanced split.

Pipeline parallelism (not tensor parallelism) is used because it works for ANY
HF model without architecture-specific sharding code. True tensor parallelism
(splitting each layer's weights across GPUs with all-gather) requires
architecture-specific sharding code (Megatron-style) and is not implemented
here. FSDP in `train_cpt.py --fsdp` is the closest training-time equivalent
(it shards params across GPUs, though via reduce-scatter/all-gather rather than
intra-layer splitting).

NOTE: this is for inference only (no training). For production multi-GPU
training, use train_cpt.py --fsdp (data + param parallelism) instead.

Usage:
    # Auto-detect GPUs, run interactive streaming generation:
    python3 tensor_parallel.py --model ./checkpoints/base_15b

    # From a prompt file:
    python3 tensor_parallel.py --model ./checkpoints/base_15b --input prompts.txt

    # Print the sharding plan without loading the model:
    python3 tensor_parallel.py --model ./checkpoints/base_15b --dry-run

    # With flash attention + explicit GPU count override:
    python3 tensor_parallel.py --model ./checkpoints/base_15b --flash-attn --num-gpus 4

Self-test (no GPU required — exercises GPU detection logic + device map builder):
    python3 tensor_parallel.py --selftest
"""

import argparse
import os


def log(msg: str):
    print(f"[tp] {msg}", flush=True)


def detect_gpu_count(backend_name: str | None = None):
    """Detect how many GPUs are available. Returns (count, arch_names).

    Uses the backend abstraction (rocm, cuda, xpu) instead of torch.cuda
    directly. On CPU-only boxes returns (0, []).
    """
    from backends import autodetect_backend, get_backend

    if backend_name is not None:
        backend = get_backend(backend_name)
    else:
        backend = autodetect_backend()

    if not backend.is_available():
        return 0, []
    count = backend.get_device_count()
    archs = []
    for i in range(count):
        arch = backend.get_arch_tag(i)
        if not arch:
            props = backend.get_device_properties(i)
            arch = getattr(props, "name", f"gpu{i}") if props else f"gpu{i}"
        archs.append(arch)
    return count, archs


def get_model_info_from_config(cfg_path: str) -> dict:
    """Load config.json and extract model info (handles nested/flat layouts).

    Also detects the layer prefix: Gemma-4 uses "model.language_model.layers"
    (Gemma4ForConditionalGeneration wraps a Gemma4Model whose .language_model
    holds the decoder), while standard Llama/Mistral/Qwen AND every other real
    Gemma generation (Gemma2, Gemma3/Gemma3-text) use flat "model.layers". The
    detection checks model_type against the two known Gemma-4 spellings
    (see prune_vocab.py's model_type normalization for why there are two).
    """
    import json
    with open(cfg_path) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    # Detect layer prefix: Gemma-4 (model_type "gemma4", or "gemma4_unified"
    # before prune_vocab.py's normalization runs) wraps the decoder under
    # model.language_model (Gemma4ForConditionalGeneration -> Gemma4Model ->
    # .language_model -> Gemma4TextModel holding the decoder layers).
    #
    # IMPORTANT: this must NOT be a `model_type.startswith("gemma")` prefix
    # match. Verified directly against the installed transformers library:
    # real Gemma2 (model_type="gemma2") and real Gemma3-text
    # (model_type="gemma3_text") both use FLAT model.layers with no
    # language_model wrapper at all (`hasattr(Gemma2ForCausalLM(...).model,
    # "language_model")` is False for both) -- only Gemma-4 nests. A prefix
    # match would silently misroute every real Gemma2/Gemma3 checkpoint to a
    # device_map with keys that don't exist on the model, which either
    # silently drops those layers from the map or crashes from_pretrained
    # depending on the transformers version. We check model_type rather than
    # a "language_model" config key because real Gemma-4 configs flatten
    # text_config params without a nested "language_model" sub-dict -- the
    # prefix comes from the module structure, not the config JSON. This
    # matches expand_model.py and mtp_head.py, which both default to
    # "model.language_model.layers" for Gemma-4.
    model_type = cfg.get("model_type", "")
    if model_type in ("gemma4", "gemma4_unified"):
        layer_prefix = "model.language_model.layers"
    else:
        layer_prefix = "model.layers"
    return {
        "model_type": cfg.get("model_type", "unknown"),
        "hidden_size": tc.get("hidden_size"),
        "num_layers": tc.get("num_hidden_layers"),
        "vocab_size": tc.get("vocab_size"),
        "layer_prefix": layer_prefix,
    }


def build_explicit_device_map(num_layers: int, num_gpus: int,
                              layer_prefix: str = "model.layers") -> dict:
    """Build an explicit, layer-balanced device map for pipeline parallelism.

    Returns a dict like {"model.embed_tokens": 0, "model.layers.0": 0, ...,
    "model.layers.31": 3, "model.norm": 3, "lm_head": 3} mapping module names
    to device indices (0..num_gpus-1), with transformer layers split as evenly
    as possible across GPUs.

    For a model with N layers across G GPUs, GPU i gets layers
    [i*N//G, (i+1)*N//G). Embeddings go to GPU 0; the final norm + LM head go
    to the last GPU (they touch the output). This minimizes the pipeline bubble
    on homogeneous hardware (e.g. 8x MI300X) compared to HF's device_map="auto"
    greedy memory fit, which can leave one GPU underloaded.

    `layer_prefix` controls the key naming. Standard Llama/Mistral/Qwen use
    "model.layers"; Gemma-4 (this repo's primary target) uses
    "model.language_model.layers" because Gemma4ForConditionalGeneration wraps
    a Gemma4Model whose .language_model holds the decoder. The embed_tokens
    and norm prefixes are derived from layer_prefix. Call get_model_info_from_config
    or pass --device-map auto if your model uses a different prefix.
    """
    if num_gpus <= 1 or num_layers <= 0:
        return {}
    # Derive embed_tokens / norm prefixes from the layer prefix.
    # "model.layers" -> "model.embed_tokens" / "model.norm"
    # "model.language_model.layers" -> "model.language_model.embed_tokens" / "model.language_model.norm"
    base = layer_prefix.rsplit(".layers", 1)[0]  # "model" or "model.language_model"
    embed_key = f"{base}.embed_tokens"
    norm_key = f"{base}.norm"
    device_map = {}
    device_map[embed_key] = 0
    for layer in range(num_layers):
        # Balanced split: GPU i gets layers [i*N//G, (i+1)*N//G).
        gpu = min(num_gpus - 1, (layer * num_gpus) // num_layers)
        device_map[f"{layer_prefix}.{layer}"] = gpu
    last_gpu = num_gpus - 1
    device_map[norm_key] = last_gpu
    device_map["lm_head"] = last_gpu
    return device_map


def plan_sharding(num_gpus: int, model_info: dict) -> dict:
    """Build a distribution plan for the model across num_gpus GPUs.

    Returns a dict describing the plan AND (for multi-GPU) the explicit
    device_map to pass to from_pretrained(). If num_gpus <= 1, returns a
    single-GPU plan with device_map=None.

    For multi-GPU, build_explicit_device_map() produces a layer-balanced split
    — better than device_map="auto" on homogeneous AMD nodes because it avoids
    the greedy memory-fit leaving one GPU underloaded.
    """
    num_layers = model_info.get("num_layers") or 0
    layer_prefix = model_info.get("layer_prefix", "model.layers")
    if num_gpus <= 1:
        return {
            "mode": "single_gpu",
            "num_gpus": 1,
            "model_type": model_info.get("model_type", "unknown"),
            "hidden_size": model_info.get("hidden_size"),
            "num_layers": num_layers,
            "device_map": None,
            "layer_prefix": layer_prefix,
        }
    device_map = build_explicit_device_map(num_layers, num_gpus, layer_prefix)
    # Per-GPU layer counts for the plan summary.
    gpu_layers = {}
    for layer in range(num_layers):
        gpu = min(num_gpus - 1, (layer * num_gpus) // num_layers)
        gpu_layers[gpu] = gpu_layers.get(gpu, 0) + 1
    return {
        "mode": "pipeline_parallel",
        "num_gpus": num_gpus,
        "model_type": model_info.get("model_type", "unknown"),
        "hidden_size": model_info.get("hidden_size"),
        "num_layers": num_layers,
        "device_map": device_map,
        "layers_per_gpu": gpu_layers,
        "layer_prefix": layer_prefix,
    }


def report_vram_per_gpu(backend_name: str | None = None):
    """Print peak VRAM usage per GPU after model load. Helps verify the
    pipeline split is balanced — if one GPU is at 70GB and another at 10GB,
    the device map is wrong and you should use --device-map auto."""
    from backends import autodetect_backend, get_backend

    if backend_name is not None:
        backend = get_backend(backend_name)
    else:
        backend = autodetect_backend()

    if not backend.is_available():
        return
    n = backend.get_device_count()
    for i in range(n):
        info = backend.memory_info(i)
        alloc = info.get("allocated_bytes", 0) / 1024**3
        peak = backend.max_memory_allocated(i) / 1024**3
        total = info.get("total_bytes", 0) / 1024**3
        arch = backend.get_arch_tag(i) or f"gpu{i}"
        log(f"GPU {i} ({arch}, {total:.0f}GB total): "
            f"alloc={alloc:.1f}GB  peak={peak:.1f}GB")


def _load_model(model_path: str, num_gpus: int, archs: list,
                gfx_override: str, hip_alloc_conf: str,
                device_map_mode: str, flash_attn: bool,
                backend_name: str | None = None):
    """Load the model once with pipeline parallelism across num_gpus GPUs.
    Returns (model, tokenizer, first_device). Call once, reuse for all prompts."""
    from backends import autodetect_backend, get_backend, BackendDevice
    from runtime import resolve_dtype, resolve_flash_attn

    backend = get_backend(backend_name) if backend_name else autodetect_backend()
    if backend.name == "rocm":
        from rocm_env import setup_rocm_env
        hip_conf = None if hip_alloc_conf.lower() == "none" else hip_alloc_conf
        setup_rocm_env(override=gfx_override, hip_alloc_conf=hip_conf)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_str = resolve_dtype(BackendDevice(backend=backend), "bf16")
    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype_str]

    load_kwargs = {"torch_dtype": torch_dtype, "trust_remote_code": True}
    if resolve_flash_attn(BackendDevice(backend=backend), flash_attn):
        load_kwargs["attn_implementation"] = "flash_attention_2"
        log("flash attention 2 enabled at load")

    # Determine the device map.
    if num_gpus <= 1 or device_map_mode == "single":
        log("single GPU mode — loading model normally")
        device_map = None
    elif device_map_mode == "auto":
        log(f"multi-GPU mode — HF device_map='auto' across {num_gpus} GPUs: {archs}")
        device_map = "auto"
    else:
        cfg_path = os.path.join(model_path, "config.json")
        model_info = {}
        if os.path.exists(cfg_path):
            model_info = get_model_info_from_config(cfg_path)
        num_layers = model_info.get("num_layers") or 0
        layer_prefix = model_info.get("layer_prefix", "model.layers")
        device_map = build_explicit_device_map(num_layers, num_gpus, layer_prefix)
        if not device_map:
            log("WARNING: could not build explicit device map; falling back to device_map='auto'")
            device_map = "auto"
        else:
            gpu_layers = {}
            for layer in range(num_layers):
                gpu = min(num_gpus - 1, (layer * num_gpus) // num_layers)
                gpu_layers[gpu] = gpu_layers.get(gpu, 0) + 1
            log(f"multi-GPU mode — explicit layer-balanced device map across "
                f"{num_gpus} GPUs: {archs} (prefix: {layer_prefix})")
            log(f"  layers per GPU: {dict(sorted(gpu_layers.items()))}")

    if device_map is not None:
        load_kwargs["device_map"] = device_map
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        # ROCm PyTorch exposes through the "cuda" device namespace, not "rocm".
        # backends/device.py:46 documents this mapping explicitly.
        torch_dev = "cuda" if backend.name == "rocm" else backend.name
        model.to(torch.device(torch_dev, 0))

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = True
    model.eval()

    if hasattr(model, "hf_device_map") and model.hf_device_map:
        devices = sorted(set(str(v) for v in model.hf_device_map.values()))
        log(f"model distributed across devices: {devices}")
    report_vram_per_gpu(backend_name)

    # Determine the first device (where embed_tokens lives) for input placement.
    # ROCm exposes through "cuda" namespace, not "rocm" (backends/device.py:46).
    torch_dev = "cuda" if backend.name == "rocm" else backend.name
    first_device = torch.device(torch_dev, 0)
    try:
        embed_device = model.get_input_embeddings().weight.device
        first_device = embed_device
    except Exception:
        pass

    return model, tokenizer, first_device


def _generate(model, tokenizer, first_device, prompt: str,
              max_new_tokens: int, temperature: float, top_p: float):
    """Generate for a single prompt using an already-loaded model."""
    from generate import stream_generate
    stream_generate(model, tokenizer, prompt, max_new_tokens,
                    temperature, top_p, repetition_penalty=1.0,
                    device=first_device)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, help="HF model dir/repo.")
    ap.add_argument("--input", type=str, default=None,
                    help="File of prompts (one per line). If omitted, uses --prompt.")
    ap.add_argument("--prompt", type=str, default="Hello, what is ROCm?",
                    help="Prompt text (used if --input is not given).")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--gfx-override", type=str, default=None,
                    help="Force HSA_OVERRIDE_GFX_VERSION (see rocm_env.py).")
    ap.add_argument("--hip-alloc-conf", type=str, default="expandable_segments:True",
                    help="PYTORCH_HIP_ALLOC_CONF value (pass 'none' to skip).")
    ap.add_argument("--device-map", type=str, default="auto",
                    choices=["explicit", "auto", "single"],
                    help="How to distribute the model across GPUs. 'auto' "
                         "(default) uses HF's device_map='auto' (greedy memory "
                         "fit) — the safe default that works with any module "
                         "layout. 'explicit' builds a layer-balanced device map "
                         "— best on homogeneous AMD nodes (8x MI300X). The "
                         "explicit map auto-detects the layer prefix (standard "
                         "model.layers or Gemma-4's model.language_model.layers), "
                         "so it works for both layouts. 'single' forces "
                         "single-GPU even if multiple are detected.")
    ap.add_argument("--num-gpus", type=int, default=None,
                    help="Override the auto-detected GPU count (e.g. use only 4 "
                         "of 8 GPUs). When set, only the first N GPUs are used.")
    ap.add_argument("--flash-attn", action="store_true", default=False,
                    help="Use Flash Attention 2 at load (requires flash-attn built "
                         "for ROCm). Falls back to standard attention if not installed.")
    ap.add_argument("--backend", type=str, default=None,
                    choices=["rocm", "cpu"],
                    help="Compute backend to use (auto-detected if unset).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Detect GPUs + print sharding plan, don't run.")
    args = ap.parse_args()

    # ROCm env setup BEFORE torch import (only for the ROCm backend).
    from backends import get_backend
    backend = get_backend(args.backend) if args.backend else None
    if backend is None or backend.name == "rocm":
        from rocm_env import setup_rocm_env
        from rocm_env import setup_rocm_env_from_args
        setup_rocm_env_from_args(args)

    # Detect GPUs via the backend abstraction.
    num_gpus, archs = detect_gpu_count(backend_name=args.backend)
    if args.num_gpus is not None:
        num_gpus = min(args.num_gpus, num_gpus) if num_gpus > 0 else args.num_gpus
        archs = archs[:num_gpus] if archs else []

    # Load model info for the sharding plan.
    cfg_path = os.path.join(args.model, "config.json")
    model_info = {}
    if os.path.exists(cfg_path):
        model_info = get_model_info_from_config(cfg_path)

    plan = plan_sharding(num_gpus, model_info)
    log(f"detected {num_gpus} GPU(s): {archs if archs else 'none'}")
    log(f"sharding plan: {plan['mode']} across {plan['num_gpus']} GPU(s)")
    if plan.get("layers_per_gpu"):
        log(f"  layers per GPU: {dict(sorted(plan['layers_per_gpu'].items()))}")

    if args.dry_run:
        if plan.get("device_map") and isinstance(plan["device_map"], dict):
            # Build the key names from the detected layer_prefix so this works
            # for both standard ("model.layers") and Gemma-4 ("model.language_model.layers") layouts.
            pfx = plan.get("layer_prefix", "model.layers")
            base = pfx.rsplit(".layers", 1)[0]  # "model" or "model.language_model"
            n_layers = model_info.get("num_layers") or 0
            last = n_layers - 1 if n_layers else "?"
            log(f"  device_map ({len(plan['device_map'])} entries): "
                f"{base}.embed_tokens->GPU {plan['device_map'].get(f'{base}.embed_tokens')}, "
                f"{pfx}.0->GPU {plan['device_map'].get(f'{pfx}.0')}, "
                f"{pfx}.{last}->GPU {plan['device_map'].get(f'{pfx}.{last}')}, "
                f"lm_head->GPU {plan['device_map'].get('lm_head')}")
        log("DRY RUN — nothing run.")
        return

    if num_gpus == 0:
        raise SystemExit("ERROR: no GPUs detected. This tool needs at least 1 GPU.")

    # Load the model ONCE, then generate for each prompt. Previously the model
    # was reloaded per prompt in --input mode — a pure inefficiency that made
    # multi-prompt runs N× slower (each load re-reads all shards from disk).
    model, tokenizer, first_device = _load_model(
        args.model, num_gpus, archs,
        args.gfx_override, args.hip_alloc_conf,
        args.device_map, args.flash_attn,
        backend_name=args.backend)

    if args.input:
        with open(args.input, encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        for p in prompts:
            log(f"prompt: {p[:80]}{'...' if len(p) > 80 else ''}")
            _generate(model, tokenizer, first_device, p,
                     args.max_new_tokens, args.temperature, args.top_p)
    else:
        _generate(model, tokenizer, first_device, args.prompt,
                 args.max_new_tokens, args.temperature, args.top_p)


def _self_test():
    print("[selftest] tensor_parallel: GPU detection + device map builder")

    # detect_gpu_count on a non-GPU box returns (0, []).
    count, archs = detect_gpu_count()
    print(f"  detect_gpu_count() on this host: {count} GPUs, archs={archs}")
    assert count >= 0
    if count == 0:
        assert archs == []
    print("  OK (detect_gpu_count returns valid results)")

    # build_explicit_device_map: 32 layers across 4 GPUs -> 8 layers each.
    dm = build_explicit_device_map(32, 4)
    assert dm["model.embed_tokens"] == 0
    assert dm["model.layers.0"] == 0
    assert dm["model.layers.7"] == 0   # layers 0-7 on GPU 0
    assert dm["model.layers.8"] == 1   # layers 8-15 on GPU 1
    assert dm["model.layers.15"] == 1
    assert dm["model.layers.16"] == 2  # layers 16-23 on GPU 2
    assert dm["model.layers.23"] == 2
    assert dm["model.layers.24"] == 3  # layers 24-31 on GPU 3
    assert dm["model.layers.31"] == 3
    assert dm["model.norm"] == 3       # final norm on last GPU
    assert dm["lm_head"] == 3          # lm_head on last GPU
    print("  OK (build_explicit_device_map: 32 layers / 4 GPUs -> 8 each, balanced)")

    # build_explicit_device_map: uneven split (33 layers across 4 GPUs).
    dm = build_explicit_device_map(33, 4)
    # 33 = 9 + 8 + 8 + 8 (first GPU gets the extra layer due to integer division).
    gpu_counts = {}
    for layer in range(33):
        gpu = dm[f"model.layers.{layer}"]
        gpu_counts[gpu] = gpu_counts.get(gpu, 0) + 1
    assert gpu_counts == {0: 9, 1: 8, 2: 8, 3: 8}, gpu_counts
    assert dm["model.layers.32"] == 3  # last layer on last GPU
    print(f"  OK (build_explicit_device_map: 33 layers / 4 GPUs -> {gpu_counts} (extra on first GPU))")

    # build_explicit_device_map: single GPU -> empty map (no sharding).
    assert build_explicit_device_map(32, 1) == {}
    assert build_explicit_device_map(0, 4) == {}
    print("  OK (build_explicit_device_map: single-GPU or 0-layers -> empty map)")

    # plan_sharding: single GPU.
    plan = plan_sharding(1, {"model_type": "gemma4", "hidden_size": 4096, "num_layers": 32})
    assert plan["mode"] == "single_gpu"
    assert plan["num_gpus"] == 1
    assert plan["device_map"] is None
    print("  OK (single-GPU plan, device_map=None)")

    # plan_sharding: multi-GPU (pipeline parallelism with explicit device_map).
    plan = plan_sharding(4, {"model_type": "gemma4", "hidden_size": 4096, "num_layers": 32})
    assert plan["mode"] == "pipeline_parallel"
    assert plan["num_gpus"] == 4
    assert isinstance(plan["device_map"], dict)
    assert len(plan["device_map"]) == 35  # 32 layers + embed + norm + lm_head
    assert plan["layers_per_gpu"] == {0: 8, 1: 8, 2: 8, 3: 8}
    print("  OK (multi-GPU plan with explicit device_map + balanced layers_per_gpu)")

    # plan_sharding: 0 GPUs -> single_gpu plan (graceful fallback).
    assert plan_sharding(0, {})["mode"] == "single_gpu"
    print("  OK (0 GPUs -> single_gpu plan)")

    # get_model_info_from_config: handles nested text_config.
    import tempfile, json
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"model_type": "gemma4", "text_config": {"hidden_size": 4096,
                   "num_hidden_layers": 32, "vocab_size": 256000}}, f)
        cfg_path = f.name
    info = get_model_info_from_config(cfg_path)
    assert info["model_type"] == "gemma4"
    assert info["hidden_size"] == 4096
    assert info["num_layers"] == 32
    assert info["vocab_size"] == 256000
    assert info["layer_prefix"] == "model.language_model.layers", \
        f"gemma4 should detect model.language_model.layers, got {info['layer_prefix']}"
    os.unlink(cfg_path)
    print("  OK (get_model_info_from_config: gemma4 -> layer_prefix=model.language_model.layers)")

    # get_model_info_from_config: non-gemma model_type -> model.layers.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"model_type": "llama", "text_config": {"hidden_size": 4096,
                   "num_hidden_layers": 32, "vocab_size": 32000}}, f)
        cfg_path = f.name
    info = get_model_info_from_config(cfg_path)
    assert info["layer_prefix"] == "model.layers", \
        f"llama should detect model.layers, got {info['layer_prefix']}"
    os.unlink(cfg_path)
    print("  OK (get_model_info_from_config: llama -> layer_prefix=model.layers)")

    # get_model_info_from_config: REAL Gemma2/Gemma3-text model_types must NOT
    # get the language_model prefix -- only Gemma-4 nests under
    # model.language_model. Verified directly against the installed
    # transformers library: Gemma2ForCausalLM and Gemma3ForCausalLM (text-only)
    # both expose a flat `.model.layers`, no `.model.language_model` at all.
    # A `model_type.startswith("gemma")` prefix match (a real bug caught and
    # fixed while reviewing this file) would have wrongly routed both of these
    # to model.language_model.layers, producing a device_map with keys that
    # don't exist on the actual model.
    for real_model_type in ("gemma2", "gemma3_text", "gemma3", "gemma"):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"model_type": real_model_type, "text_config": {
                       "hidden_size": 4096, "num_hidden_layers": 32,
                       "vocab_size": 32000}}, f)
            cfg_path = f.name
        info = get_model_info_from_config(cfg_path)
        assert info["layer_prefix"] == "model.layers", (
            f"real model_type={real_model_type!r} should detect flat "
            f"model.layers (only gemma4/gemma4_unified nest), got "
            f"{info['layer_prefix']}"
        )
        os.unlink(cfg_path)
    print("  OK (get_model_info_from_config: real gemma2/gemma3 model_types "
          "correctly stay on flat model.layers, NOT misrouted by a "
          "startswith('gemma') prefix match)")

    # build_explicit_device_map with gemma4 prefix (model.language_model.layers).
    dm = build_explicit_device_map(8, 2, layer_prefix="model.language_model.layers")
    assert dm["model.language_model.embed_tokens"] == 0
    assert dm["model.language_model.layers.0"] == 0
    assert dm["model.language_model.layers.3"] == 0
    assert dm["model.language_model.layers.4"] == 1
    assert dm["model.language_model.layers.7"] == 1
    assert dm["model.language_model.norm"] == 1
    assert dm["lm_head"] == 1
    print("  OK (build_explicit_device_map: gemma4 prefix -> model.language_model.* keys)")

    print("\n[selftest] All checks passed (no GPU required — run on a multi-GPU "
          "AMD box for actual pipeline parallelism).")


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
