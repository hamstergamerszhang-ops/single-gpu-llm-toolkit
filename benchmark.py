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


def run_gen_benchmark(model_path: str, prompt_len: int, gen_len: int,
                      warmup: int, steps: int,
                      gfx_override: str, hip_alloc_conf: str,
                      backend_name: str | None = None):
    """Run a generation benchmark: measures prefill latency, decode latency,
    time-to-first-token (TTFT), and time-per-output-token (TPOT).

    Returns a result dict.
    """
    from backends import autodetect_backend, get_backend, BackendDevice
    from runtime import resolve_dtype

    backend = get_backend(backend_name) if backend_name else autodetect_backend()
    if backend.name == "rocm":
        from rocm_env import setup_rocm_env
        hip_conf = None if hip_alloc_conf.lower() == "none" else hip_alloc_conf
        setup_rocm_env(override=gfx_override, hip_alloc_conf=hip_conf)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    from threading import Thread
    import queue as _queue

    dev = BackendDevice(backend=backend)
    if not dev.backend.is_available():
        raise SystemExit("ERROR: no compute device visible — benchmark needs a device.")

    dtype_str = resolve_dtype(dev, "bf16")
    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16,
                   "bf16": torch.bfloat16, "fp8": torch.bfloat16}[dtype_str]

    log(f"loading model for generation benchmark: {model_path} on {dev}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, trust_remote_code=True
    ).to(dev.torch_device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    input_ids = torch.full((1, prompt_len), tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
                           dtype=torch.long, device=dev.torch_device)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model.generate(input_ids, max_new_tokens=gen_len, use_cache=True,
                               do_sample=False)

    dev.synchronize()
    dev.reset_peak_memory_stats()

    total_times = []
    first_token_times = []

    for _ in range(steps):
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True,
            timeout=30.0)
        gen_kwargs = {
            "input_ids": input_ids, "max_new_tokens": gen_len,
            "use_cache": True, "do_sample": False, "streamer": streamer,
        }

        def _gen():
            with torch.inference_mode():
                model.generate(**gen_kwargs)
        thread = Thread(target=_gen, daemon=True)
        dev.synchronize()
        t0 = time.perf_counter()
        thread.start()
        first_token_time = None
        try:
            for _ in streamer:
                if first_token_time is None:
                    dev.synchronize()
                    first_token_time = time.perf_counter() - t0
        except _queue.Empty:
            pass
        thread.join(timeout=30.0)
        dev.synchronize()
        t1 = time.perf_counter()
        total_times.append(t1 - t0)
        first_token_times.append(first_token_time if first_token_time else (t1 - t0))

    peak_vram = dev.max_memory_allocated() / 1024**3
    avg_ttft = sum(first_token_times) / len(first_token_times) * 1000
    avg_total = sum(total_times) / len(total_times) * 1000
    # TPOT needs at least 2 generated tokens (1 for TTFT, >=1 for decode).
    # With gen_len < 2, TPOT is meaningless — report 0.
    if gen_len >= 2:
        avg_tpot = (avg_total - avg_ttft) / (gen_len - 1)
    else:
        avg_tpot = 0.0
    tokens_per_s = gen_len / (sum(total_times) / len(total_times))

    result = {
        "prompt_len": prompt_len, "gen_len": gen_len,
        "ttft_ms": avg_ttft, "tpot_ms": avg_tpot,
        "total_ms": avg_total, "tokens_per_sec": tokens_per_s,
        "peak_vram_gb": peak_vram,
    }
    log(f"  TTFT: {avg_ttft:.1f}ms  TPOT: {avg_tpot:.1f}ms  "
        f"total: {avg_total:.0f}ms  {tokens_per_s:,.0f} tok/s  "
        f"VRAM: {peak_vram:.1f}GB")

    del model
    dev.empty_cache()
    return result


def format_gen_table(result: dict) -> str:
    """Format generation benchmark results."""
    if not result:
        return "(no results)"
    lines = [
        f"{'Metric':<25} {'Value':>15}",
        "-" * 42,
        f"{'Prompt length':<25} {result['prompt_len']:>15}",
        f"{'Generate length':<25} {result['gen_len']:>15}",
        f"{'TTFT (ms)':<25} {result['ttft_ms']:>15.1f}",
        f"{'TPOT (ms)':<25} {result['tpot_ms']:>15.1f}",
        f"{'Total time (ms)':<25} {result['total_ms']:>15.0f}",
        f"{'Tokens/sec':<25} {result['tokens_per_sec']:>15,.0f}",
        f"{'Peak VRAM (GB)':<25} {result['peak_vram_gb']:>15.1f}",
    ]
    return "\n".join(lines)


def run_benchmark(model_path: str, configs: list[dict], warmup: int, steps: int,
                  gfx_override: str, hip_alloc_conf: str,
                  backend_name: str | None = None):
    """Run the benchmark for each config. Returns a list of result dicts."""
    from backends import autodetect_backend, get_backend, BackendDevice
    from runtime import resolve_dtype, resolve_flash_attn, resolve_compile

    backend = get_backend(backend_name) if backend_name else autodetect_backend()
    if backend.name == "rocm":
        from rocm_env import setup_rocm_env
        hip_conf = None if hip_alloc_conf.lower() == "none" else hip_alloc_conf
        setup_rocm_env(override=gfx_override, hip_alloc_conf=hip_conf)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = BackendDevice(backend=backend)
    if not dev.backend.is_available():
        raise SystemExit("ERROR: no compute device visible — benchmark needs a device.")

    results = []
    for i, cfg in enumerate(configs):
        label = (f"b={cfg['batch']} s={cfg['seqlen']} {cfg['dtype']}"
                 f"{' +flash' if cfg['flash'] else ''}"
                 f"{' +compile' if cfg['compile'] else ''}")
        log(f"config {i+1}/{len(configs)}: {label}")

        dtype_str = resolve_dtype(dev, cfg["dtype"])
        torch_dtype = {"fp32": torch.float32, "fp16": torch.float16,
                   "bf16": torch.bfloat16, "fp8": torch.bfloat16}[dtype_str]

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, trust_remote_code=True
        ).to(dev.torch_device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if dtype_str == "fp8":
            if not dev.supports_fp8():
                log("  WARNING: fp8 not supported on this device — using bf16")
            else:
                try:
                    from torchao.float8 import convert_to_float8_training
                    convert_to_float8_training(model)
                    log("  fp8 enabled (torchao)")
                except ImportError:
                    log("  WARNING: torchao not installed, using bf16")
                except Exception as e:
                    log(f"  WARNING: fp8 conversion failed ({e}), using bf16")
        if resolve_flash_attn(dev, bool(cfg["flash"])):
            try:
                if hasattr(model, "set_attn_implementation"):
                    model.set_attn_implementation("flash_attention_2")
                else:
                    model.config._attn_implementation = "flash_attention_2"
                    if hasattr(model, "text_config"):
                        model.text_config._attn_implementation = "flash_attention_2"
                log("  flash-attn enabled")
            except Exception as e:
                log(f"  WARNING: flash-attn failed ({e}), using standard attn")
        if resolve_compile(dev, bool(cfg["compile"])):
            try:
                model = torch.compile(model)
                log("  torch.compile enabled")
            except Exception as e:
                log(f"  WARNING: compile failed ({e}), using eager")

        model.train()
        model.config.use_cache = False
        try:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=1e-5, fused=True, foreach=False)
        except (ValueError, RuntimeError):
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        for _ in range(warmup):
            input_ids = torch.randint(0, tokenizer.vocab_size or len(tokenizer) or 1,
                                      (cfg["batch"], cfg["seqlen"]), device=dev.torch_device)
            labels = input_ids.clone()
            attn = torch.ones_like(input_ids)
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attn)
            optimizer.zero_grad(set_to_none=True)
            outputs.loss.backward()
            optimizer.step()

        dev.synchronize()
        dev.reset_peak_memory_stats()
        start = time.perf_counter()

        for _ in range(steps):
            input_ids = torch.randint(0, tokenizer.vocab_size or len(tokenizer) or 1,
                                      (cfg["batch"], cfg["seqlen"]), device=dev.torch_device)
            labels = input_ids.clone()
            attn = torch.ones_like(input_ids)
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attn)
            optimizer.zero_grad(set_to_none=True)
            outputs.loss.backward()
            optimizer.step()

        dev.synchronize()
        elapsed = time.perf_counter() - start
        peak_vram = dev.max_memory_allocated() / 1024**3

        total_tokens = cfg["batch"] * cfg["seqlen"] * steps
        tps = total_tokens / elapsed
        step_ms = (elapsed / steps) * 1000

        results.append({
            **cfg, "tokens_per_sec": tps, "peak_vram_gb": peak_vram,
            "step_ms": step_ms,
        })
        log(f"  tokens/s: {tps:,.0f}  peak_VRAM: {peak_vram:.1f}GB  "
            f"step: {step_ms:.1f}ms")

        if "outputs" in dir():
            del outputs
        if "input_ids" in dir():
            del input_ids
        if "labels" in dir():
            del labels
        if "attn" in dir():
            del attn
        del model, optimizer
        dev.empty_cache()

    return results


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, help="HF model dir/repo to benchmark.")
    ap.add_argument("--configs", required=False, default=None,
                    help="Semicolon-separated configs: "
                         "'batch=2,seqlen=1024,dtype=bf16;batch=4,seqlen=512,dtype=fp8' "
                         "(required unless --gen is used).")
    ap.add_argument("--warmup", type=int, default=3,
                    help="Warmup steps before measurement (default 3).")
    ap.add_argument("--steps", type=int, default=10,
                    help="Measured steps (default 10).")
    ap.add_argument("--gfx-override", type=str, default=None,
                    help="Force HSA_OVERRIDE_GFX_VERSION (see rocm_env.py).")
    ap.add_argument("--hip-alloc-conf", type=str, default="expandable_segments:True",
                    help="PYTORCH_HIP_ALLOC_CONF value (pass 'none' to skip).")
    ap.add_argument("--gen", action="store_true", default=False,
                    help="Run a generation benchmark instead of the training "
                         "benchmark. Measures TTFT (time-to-first-token), TPOT "
                         "(time-per-output-token), and tokens/sec for a single "
                         "prompt. Use --gen-prompt-len and --gen-len to control "
                         "the prompt and generation lengths.")
    ap.add_argument("--gen-prompt-len", type=int, default=512,
                    help="Prompt length (in tokens) for the generation benchmark "
                         "(default 512). Only used with --gen.")
    ap.add_argument("--gen-len", type=int, default=128,
                    help="Number of tokens to generate in the generation benchmark "
                         "(default 128). Only used with --gen.")
    ap.add_argument("--backend", type=str, default=None,
                    choices=["rocm", "cpu"],
                    help="Compute backend to use (auto-detected if unset).")
    args = ap.parse_args()

    from rocm_env import setup_rocm_env_from_args

    if args.gen:
        log(f"generation benchmark: prompt_len={args.gen_prompt_len}, "
            f"gen_len={args.gen_len}, warmup={args.warmup}, steps={args.steps}")
        result = run_gen_benchmark(args.model, args.gen_prompt_len, args.gen_len,
                                   args.warmup, args.steps,
                                   args.gfx_override, args.hip_alloc_conf,
                                   backend_name=args.backend)
        print("\n" + format_gen_table(result))
        return

    configs = parse_configs(args.configs)
    if not configs:
        raise SystemExit("ERROR: no configs parsed from --configs (or --configs "
                         "not set; use --gen for generation benchmarking).")
    log(f"benchmarking {len(configs)} config(s): warmup={args.warmup}, steps={args.steps}")

    results = run_benchmark(args.model, configs, args.warmup, args.steps,
                            args.gfx_override, args.hip_alloc_conf,
                            backend_name=args.backend)

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

    # Generation benchmark table formatter.
    gen_result = {
        "prompt_len": 512, "gen_len": 128,
        "ttft_ms": 45.2, "tpot_ms": 12.3,
        "total_ms": 1610, "tokens_per_sec": 79.5,
        "peak_vram_gb": 42.1,
    }
    gen_table = format_gen_table(gen_result)
    assert "TTFT" in gen_table
    assert "TPOT" in gen_table
    assert "45.2" in gen_table
    assert "80" in gen_table  # 79.5 rounds to 80 with ,.0f format
    print("  OK (generation table formatter produces aligned output)")

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
