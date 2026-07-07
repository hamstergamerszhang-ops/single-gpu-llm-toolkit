#!/usr/bin/env python3
"""Throughput / VRAM benchmark for single-GPU (or multi-GPU) training configs.

Loads a model, runs N warmup + M measured training steps with dummy input, and
reports tokens/sec, peak VRAM, and avg step time. Supports testing multiple
configs (batch, seqlen, dtype, flash-attn, compile) in one run and printing a
comparison table — gives AMD devs concrete numbers to compare configurations
on their actual hardware.

Usage:
    python3 benchmark.py --model ./checkpoints/base_expanded_15b \\
        --configs "batch=2,seqlen=1024,dtype=bf16;batch=4,seqlen=512,dtype=fp8"

    # With flash-attn + compile:
    python3 benchmark.py --model ./checkpoints/base_expanded_15b \\
        --configs "batch=2,seqlen=2048,dtype=bf16,flash=1,compile=1"

Self-test (no GPU/model required — exercises config parser + table formatter):
    python3 benchmark.py --selftest
"""

import argparse
import sys
import time


def log(msg: str):
    print(f"[bench] {msg}", flush=True)


def parse_configs(configs_str: str) -> list[dict]:
    """Parse a semicolon-separated list of configs like:
    "batch=2,seqlen=1024,dtype=bf16;batch=4,seqlen=512,dtype=fp8"
    Returns a list of dicts with keys: batch, seqlen, dtype, flash, compile.
    Defaults: dtype=bf16, flash=0, compile=0.
    """
    configs = []
    for part in configs_str.split(";"):
        part = part.strip()
        if not part:
            continue
        cfg = {"batch": 2, "seqlen": 1024, "dtype": "bf16", "flash": 0, "compile": 0}
        for kv in part.split(","):
            kv = kv.strip()
            if "=" not in kv:
                continue
            key, val = kv.split("=", 1)
            key = key.strip().lower()
            val = val.strip()
            if key in ("batch", "seqlen"):
                cfg[key] = int(val)
            elif key == "dtype":
                cfg["dtype"] = val
            elif key in ("flash", "compile"):
                cfg[key] = 1 if val in ("1", "true", "yes") else 0
        configs.append(cfg)
    return configs


def format_table(results: list[dict]) -> str:
    """Format benchmark results as a comparison table."""
    if not results:
        return "(no results)"
    header = f"{'Config':<40} {'tokens/s':>10} {'peak_VRAM':>12} {'step_ms':>10}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in results:
        cfg_str = (f"b={r['batch']} s={r['seqlen']} {r['dtype']}"
                   f"{' +flash' if r.get('flash') else ''}"
                   f"{' +compile' if r.get('compile') else ''}")
        vram_str = f"{r['peak_vram_gb']:.1f} GB" if r.get("peak_vram_gb") is not None else "N/A"
        tps_str = f"{r['tokens_per_sec']:,.0f}" if r.get("tokens_per_sec") is not None else "N/A"
        ms_str = f"{r['step_ms']:.1f}" if r.get("step_ms") is not None else "N/A"
        lines.append(f"{cfg_str:<40} {tps_str:>10} {vram_str:>12} {ms_str:>10}")
    return "\n".join(lines)


def run_benchmark(model_path: str, configs: list[dict], warmup: int, steps: int,
                  gfx_override: str, hip_alloc_conf: str):
    """Run the benchmark for each config. Returns a list of result dicts."""
    from rocm_env import setup_rocm_env
    setup_rocm_env(override=gfx_override, hip_alloc_conf=hip_alloc_conf)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("ERROR: no CUDA/ROCm device visible — benchmark needs a GPU.")

    results = []
    for i, cfg in enumerate(configs):
        label = (f"b={cfg['batch']} s={cfg['seqlen']} {cfg['dtype']}"
                 f"{' +flash' if cfg['flash'] else ''}"
                 f"{' +compile' if cfg['compile'] else ''}")
        log(f"config {i+1}/{len(configs)}: {label}")

        # Reload model fresh for each config (each fp8 conversion mutates the
        # model in place, so a stale bf16 model can't be reused across
        # configs). Always load in bf16 first, same as train_cpt.py — fp8 is
        # applied after load via torchao's Float8Linear conversion below, not
        # via torch_dtype at load time.
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Apply optimizations.
        if cfg["dtype"] == "fp8":
            try:
                from torchao.float8 import convert_to_float8_training
                convert_to_float8_training(model)
                log("  fp8 enabled (torchao)")
            except ImportError:
                log("  WARNING: torchao not installed, using bf16")
        if cfg["flash"]:
            try:
                import flash_attn  # noqa: F401 — just checking it's importable
                # Prefer the public set_attn_implementation() API (see
                # train_cpt.py's _apply_flash_attn for why): it validates the
                # requested implementation and propagates to nested sub-configs
                # (e.g. Gemma-4 nests under text_config) instead of silently
                # no-op'ing on architectures where the private attribute poke
                # doesn't take effect post-load.
                if hasattr(model, "set_attn_implementation"):
                    model.set_attn_implementation("flash_attention_2")
                else:
                    model.config._attn_implementation = "flash_attention_2"
                    if hasattr(model, "text_config"):
                        model.text_config._attn_implementation = "flash_attention_2"
                log("  flash-attn enabled")
            except ImportError:
                log("  WARNING: flash-attn not installed, using standard attn")
        if cfg["compile"]:
            try:
                model = torch.compile(model)
                log("  torch.compile enabled")
            except Exception as e:
                log(f"  WARNING: compile failed ({e}), using eager")

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        # Warmup.
        for _ in range(warmup):
            input_ids = torch.randint(0, tokenizer.vocab_size,
                                      (cfg["batch"], cfg["seqlen"]), device="cuda")
            labels = input_ids.clone()
            attn = torch.ones_like(input_ids)
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attn)
            optimizer.zero_grad(set_to_none=True)
            outputs.loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()

        for _ in range(steps):
            input_ids = torch.randint(0, tokenizer.vocab_size,
                                      (cfg["batch"], cfg["seqlen"]), device="cuda")
            labels = input_ids.clone()
            attn = torch.ones_like(input_ids)
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attn)
            optimizer.zero_grad(set_to_none=True)
            outputs.loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak_vram = torch.cuda.max_memory_allocated() / 1024**3

        total_tokens = cfg["batch"] * cfg["seqlen"] * steps
        tps = total_tokens / elapsed
        step_ms = (elapsed / steps) * 1000

        results.append({
            **cfg, "tokens_per_sec": tps, "peak_vram_gb": peak_vram,
            "step_ms": step_ms,
        })
        log(f"  tokens/s: {tps:,.0f}  peak_VRAM: {peak_vram:.1f}GB  "
            f"step: {step_ms:.1f}ms")

        # Free model before next config.
        del model, optimizer
        torch.cuda.empty_cache()

    return results


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, help="HF model dir/repo to benchmark.")
    ap.add_argument("--configs", required=True,
                    help="Semicolon-separated configs: "
                         "'batch=2,seqlen=1024,dtype=bf16;batch=4,seqlen=512,dtype=fp8'")
    ap.add_argument("--warmup", type=int, default=3,
                    help="Warmup steps before measurement (default 3).")
    ap.add_argument("--steps", type=int, default=10,
                    help="Measured steps (default 10).")
    ap.add_argument("--gfx-override", type=str, default=None,
                    help="Force HSA_OVERRIDE_GFX_VERSION (see rocm_env.py).")
    ap.add_argument("--hip-alloc-conf", type=str, default="max_split_size_mb:128",
                    help="PYTORCH_HIP_ALLOC_CONF value (pass 'none' to skip).")
    args = ap.parse_args()

    configs = parse_configs(args.configs)
    if not configs:
        raise SystemExit("ERROR: no configs parsed from --configs")
    log(f"benchmarking {len(configs)} config(s): warmup={args.warmup}, steps={args.steps}")

    hip_conf = None if args.hip_alloc_conf.lower() == "none" else args.hip_alloc_conf
    results = run_benchmark(args.model, configs, args.warmup, args.steps,
                            args.gfx_override, hip_conf)

    print("\n" + format_table(results))


def _self_test():
    print("[selftest] benchmark: config parser + table formatter (no GPU required)")

    # Config parser: parses semicolon-separated key=value pairs with defaults.
    configs = parse_configs("batch=2,seqlen=1024,dtype=bf16;batch=4,seqlen=512,dtype=fp8,flash=1")
    assert len(configs) == 2
    assert configs[0] == {"batch": 2, "seqlen": 1024, "dtype": "bf16", "flash": 0, "compile": 0}
    assert configs[1] == {"batch": 4, "seqlen": 512, "dtype": "fp8", "flash": 1, "compile": 0}
    print("  OK (parses batch, seqlen, dtype, flash, compile with defaults)")

    # Empty / malformed parts are skipped gracefully.
    configs = parse_configs("batch=2;;batch=8,seqlen=512,")
    assert len(configs) == 2
    assert configs[1]["batch"] == 8
    print("  OK (skips empty/malformed parts)")

    # Table formatter produces aligned output.
    results = [
        {"batch": 2, "seqlen": 1024, "dtype": "bf16", "flash": 0, "compile": 0,
         "tokens_per_sec": 12345, "peak_vram_gb": 78.2, "step_ms": 45.3},
        {"batch": 4, "seqlen": 512, "dtype": "fp8", "flash": 1, "compile": 0,
         "tokens_per_sec": 22891, "peak_vram_gb": 65.1, "step_ms": 24.1},
    ]
    table = format_table(results)
    assert "tokens/s" in table
    assert "peak_VRAM" in table
    assert "b=2 s=1024 bf16" in table
    assert "b=4 s=512 fp8 +flash" in table
    assert "12,345" in table
    assert "78.2 GB" in table
    print("  OK (table formatter produces aligned comparison output)")

    # Empty results don't crash.
    assert format_table([]) == "(no results)"
    print("  OK (empty results handled)")

    print("\n[selftest] All checks passed (no GPU required — run with a real "
          "model on AMD hardware for actual numbers).")


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
