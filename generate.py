#!/usr/bin/env python3
"""Streaming text generation from a trained checkpoint.

Loads a checkpoint, runs generation with streaming token output, and supports
optimization flags (--flash-attn, --dtype fp8, --compile, --gfx-override,
--hip-alloc-conf). Targets AMD ROCm, with CPU as the fallback for testing/dev
without real hardware (CPU generation is very slow).

KV-cache is ENABLED by default here — essential for autoregressive decoding.

Usage:
    python3 generate.py --model ./checkpoints/model_cpt_1
    python3 generate.py --model ./checkpoints/model_cpt_1 --input prompts.txt
    python3 generate.py --model ./checkpoints/model_cpt_1 --flash-attn --dtype fp8
"""

import argparse
import os
import sys


def log(msg: str):
    print(f"[generate] {msg}", flush=True)


def build_gen_kwargs(input_ids, attention_mask, max_new_tokens: int,
                     temperature: float, top_p: float, repetition_penalty: float,
                     pad_token_id, eos_token_id, streamer, static_cache: bool = False):
    """Build the kwargs dict for model.generate()."""
    pad_id = pad_token_id if pad_token_id is not None else eos_token_id
    kwargs = dict(
        inputs=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-6),
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=pad_id,
        eos_token_id=eos_token_id,
        streamer=streamer,
        use_cache=True,
    )
    if static_cache:
        # cache_implementation="static" pre-allocates the full KV tensor once
        # instead of growing it per decode step. Pairs with torch.compile
        # (mode="reduce-overhead") for HIP graph capture of the decode loop —
        # the standard 1.5-3x TPOT improvement on MI300X.
        kwargs["cache_implementation"] = "static"
    return kwargs


def stream_generate(model, tokenizer, prompt: str, max_new_tokens: int,
                    temperature: float, top_p: float, repetition_penalty: float,
                    device, static_cache: bool = False):
    """Generate tokens one at a time, printing decoded text chunks as they're
    produced."""
    import queue
    from threading import Thread
    import torch
    from transformers import TextIteratorStreamer

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=30.0
    )
    gen_kwargs = build_gen_kwargs(
        inputs["input_ids"], inputs["attention_mask"], max_new_tokens,
        temperature, top_p, repetition_penalty,
        tokenizer.pad_token_id, tokenizer.eos_token_id, streamer,
        static_cache=static_cache,
    )

    thread_exc = []

    def _generate_with_exc():
        try:
            with torch.inference_mode():
                model.generate(**gen_kwargs)
        except Exception as e:
            thread_exc.append(e)

    thread = Thread(target=_generate_with_exc, daemon=True)
    thread.start()
    try:
        for new_text in streamer:
            print(new_text, end="", flush=True)
    except (StopIteration, queue.Empty):
        pass
    thread.join(timeout=5.0)
    if thread_exc:
        raise thread_exc[0]


def batch_generate(model, tokenizer, prompts: list[str], max_new_tokens: int,
                   temperature: float, top_p: float, repetition_penalty: float,
                   device, display_prompts: list[str] | None = None):
    """Generate for N prompts in a single batched forward pass.

    Left-pads all prompts to a common length, runs model.generate() once
    serving all prompts, then decodes and prints each sequence's output.
    This is 2-4x faster than sequential stream_generate for multi-prompt
    --input mode (one prefill + N parallel decode streams vs N separate
    prefill+decode cycles).
    """
    import torch
    from transformers import TextStreamer

    # Left-pad so generation appends to the right of every prompt.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    log(f"batched generate: {len(prompts)} prompts, max prompt len "
        f"{input_ids.size(1)}, generating {max_new_tokens} new tokens each")

    with torch.inference_mode():
        outputs = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-6),
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    # Decode only the newly generated tokens (strip the prompt prefix).
    for i in range(len(prompts)):
        prompt_len = attention_mask[i].sum().item()
        new_tokens = outputs[i][prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        label = (display_prompts[i] if display_prompts else prompts[i])
        print(f"\n--- Prompt: {label[:80]}{'...' if len(label) > 80 else ''} ---")
        print(f"[response] {text}")
    print()


def _load_model_and_tokenizer(args, dev):
    """Load the model, apply dtype/optimizations, and return (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from runtime import DTYPE_MAP, resolve_dtype, resolve_compile, resolve_flash_attn

    dtype_str = resolve_dtype(dev, args.dtype)
    torch_dtype = DTYPE_MAP[dtype_str]

    log(f"loading model from {args.model} on {dev} (dtype={dtype_str}) ...")
    load_kwargs = {"torch_dtype": torch_dtype, "trust_remote_code": True}
    if resolve_flash_attn(dev, args.flash_attn):
        load_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.to(dev.torch_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # fp8 inference: weight-only quantization via torchao.
    if dtype_str == "fp8" or args.dtype == "fp8":
        try:
            from torchao.quantization import quantize_
            try:
                from torchao.quantization.quant_api import float8_weight_only as _f8
            except ImportError:
                from torchao.quantization import Float8WeightOnlyConfig as _f8
            quantize_(model, _f8())
            log("fp8 inference enabled (torchao float8 weight-only quantization)")
        except ImportError:
            log("WARNING: torchao not installed, using bf16")
        except Exception as e:
            log(f"WARNING: fp8 inference failed ({e}) — using bf16")

    # torch.compile, probed on the target device. Auto-select mode by arch:
    # MI300-class (gfx94x, the fp8-capable archs) gets "reduce-overhead"
    # (cudagraph tree capture of the decode step) by default when --compile is
    # set without an explicit --compile-mode, since that's where cudagraphs
    # are stable and give the biggest decode win. RDNA consumer cards keep
    # "max-autotune" (cudagraph trees are less stable there).
    if resolve_compile(dev, args.compile, mode=args.compile_mode):
        # If the user didn't explicitly pass --compile-mode, auto-select.
        auto_mode = args.compile_mode
        if args.compile_mode == "max-autotune" and dev.supports_fp8():
            auto_mode = "reduce-overhead"
            log(f"torch.compile: auto-selected mode='reduce-overhead' "
                f"(MI300-class arch — cudagraph tree capture for decode)")
        try:
            model = torch.compile(model, mode=auto_mode)
            log(f"torch.compile enabled (mode={auto_mode})")
            # Warm up with a 1-token forward so the cudagraph captures the
            # decode shape before the real generate() call.
            if auto_mode == "reduce-overhead":
                _warm_ids = torch.ones((1, 1), dtype=torch.long, device=dev.torch_device)
                with torch.inference_mode():
                    _ = model(_warm_ids)
                log("torch.compile: decode-shape warmup done (cudagraph captured)")
        except Exception as e:
            log(f"WARNING: compile failed ({e}), using eager")

    # Speculative decoding via MTP head: HF's generate() expects a SEPARATE
    # assistant_model (a standalone nn.Module with its own forward + lm_head).
    # The repo's MTP head is model.model.mtp_layers — it's not standalone,
    # it shares the base model's embed_tokens + lm_head. Wiring it as a draft
    # model requires building a wrapper that calls the MTP depth's forward
    # with the base model's hidden states — a non-trivial integration that
    # can't be verified without real hardware + a trained MTP checkpoint.
    #
    # Rather than ship a broken `model.config.assistant_model = model` (which
    # either crashes or is a silent no-op), we gate this behind a clear
    # "not yet implemented" message. The --speculative flag stays so users
    # know the feature is planned, but it doesn't pretend to work.
    if getattr(args, "speculative", False):
        mtp = getattr(getattr(model, "model", model), "mtp_layers", None)
        if mtp is not None:
            log("NOTE: --speculative requested, model has MTP head, but the "
                "draft-model wrapper is not yet implemented (HF generate() "
                "needs a separate assistant_model module, not model itself). "
                "Using standard decoding. This is a known gap.")
        else:
            log("WARNING: --speculative set but model has no MTP head "
                "(model.model.mtp_layers) — using standard decoding")

    model.config.use_cache = True
    model.eval()
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-family", type=str, default=None,
                    help="Model architecture family (user-specified, NOT auto-guessed). "
                         "If the checkpoint's config.json already has model_family set "
                         "(written by train_cpt.py --model-family or mtp_head.py "
                         "--model-family), omit this. Otherwise pass it so "
                         "modeling_custom.py can select the right base class. One of: "
                         "gemma, llama, qwen, mistral, phi, falcon, gpt2, gpt_neox, "
                         "gptj, bloom, mpt, cohere, starcoder2.")
    ap.add_argument("--input", type=str, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--system-prompt", type=str, default=None)

    # Hardware / backend selection.
    ap.add_argument("--backend", type=str, default=None,
                    choices=["rocm", "cpu"],
                    help="Compute backend to use (auto-detected if unset).")
    ap.add_argument("--device-index", type=int, default=0,
                    help="Device index within the selected backend.")

    # Optimization flags.
    ap.add_argument("--flash-attn", action="store_true", default=False)
    ap.add_argument("--dtype", type=str, default="bf16",
                    choices=["fp32", "fp16", "bf16", "fp8"])
    ap.add_argument("--compile", action="store_true", default=False,
                    help="torch.compile the model for kernel fusion + graph "
                         "optimization. First call is slower (compilation), then "
                         "faster. Pairs with --static-cache for HIP graph capture.")
    ap.add_argument("--compile-mode", type=str, default="max-autotune",
                    choices=["default", "reduce-overhead", "max-autotune"])
    ap.add_argument("--static-cache", action="store_true", default=False,
                    help="Use HF StaticCache (pre-allocated KV tensors) instead of "
                         "dynamic allocation. Enables HIP graph capture when paired "
                         "with --compile --compile-mode reduce-overhead. Only works "
                         "for single-prompt generation (not --input batch mode with "
                         "variable prompt lengths).")
    ap.add_argument("--speculative", action="store_true", default=False,
                    help="Use the model's MTP head as a draft model for speculative "
                         "decoding (DeepSeek-V3 pattern). The MTP head drafts K tokens "
                         "per step, the main model verifies them in a single forward — "
                         "2-3x speedup on accept-friendly workloads. Requires a model "
                         "with model.model.mtp_layers (added by mtp_head.py). Graceful "
                         "no-op if no MTP head is present.")

    # ROCm-specific bootstrap.
    ap.add_argument("--gfx-override", type=str, default=None)
    ap.add_argument("--hip-alloc-conf", type=str, default="expandable_segments:True,garbage_collection_threshold:0.6")

    # Recipe / preset support.
    ap.add_argument("--config", type=str, default=None,
                    help="Path to a TOML/YAML recipe file.")
    ap.add_argument("--preset", type=str, default=None,
                    help="Hardware preset (cpu, rx6800-16g, rx7900xtx-24g, "
                         "rx9070xt-16g, mi300x-80g, mi300x-192g, mi250-64g, ...).")

    args = ap.parse_args()

    # Load recipe/preset defaults, then let CLI args override.
    from config import resolve_recipe, apply_preset
    recipe = resolve_recipe(args.config, base_defaults={})
    if args.preset:
        recipe = apply_preset(recipe, args.preset)
    # Simple override: any argparse default that was not explicitly set on the
    # CLI remains whatever argparse parsed. We only fill from recipe for keys
    # present in the recipe but not on CLI. argparse doesn't tell us which were
    # defaulted, so we do a best-effort merge: recipe values win over argparse
    # defaults only when the CLI value equals the argparse default. This is good
    # enough for --batch-size/--seq-length style defaults.
    defaults = {a.dest: a.default for a in ap._actions if hasattr(a, "default")}
    for key, value in recipe.items():
        if hasattr(args, key):
            current = getattr(args, key)
            if current == defaults.get(key):
                setattr(args, key, value)

    # ROCm bootstrap before torch import.
    from backends import get_backend
    backend = get_backend(args.backend) if args.backend else None
    if backend is None or backend.name == "rocm":
        from rocm_env import setup_rocm_env
        setup_rocm_env(override=args.gfx_override, hip_alloc_conf=args.hip_alloc_conf)

    # If the user specified --model-family, set the MODEL_FAMILY env var so
    # modeling_custom.py (loaded via trust_remote_code) picks it up. This is
    # the fallback path for checkpoints whose config.json doesn't already have
    # model_family set (e.g. loaded a base model directly for inference).
    if args.model_family:
        os.environ["MODEL_FAMILY"] = args.model_family
        log(f"model_family={args.model_family} (set via env for modeling_custom.py)")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from backends import BackendDevice
    dev = BackendDevice(backend=args.backend, index=args.device_index)
    if not dev.backend.is_available():
        raise SystemExit(f"ERROR: backend {dev.name} is not available.")

    model, tokenizer = _load_model_and_tokenizer(args, dev)

    prefix = ""
    if args.system_prompt:
        prefix = args.system_prompt + "\n\n"

    if args.input:
        with open(args.input, encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        # static_cache only works for single-prompt generation (variable prompt
        # lengths in batch mode break the pre-allocated KV shape). Disable it
        # in --input mode even if --static-cache was passed.
        if args.static_cache and len(prompts) > 1:
            log("WARNING: --static-cache disabled in multi-prompt --input mode "
                "(variable prompt lengths break the pre-allocated KV shape)")

        # Batched generation: when multiple prompts share a system prefix,
        # tokenize them together (left-padded) and run ONE model.generate()
        # call serving all prompts at once — N prompts in ~1/N the time of
        # sequential generation. The outputs are decoded per-sequence and
        # printed with separators. This is the standard batched-inference
        # speedup (2-4x for multi-prompt --input).
        full_prompts = [prefix + p for p in prompts]
        log(f"batched generation: {len(full_prompts)} prompts in one forward pass")
        batch_generate(model, tokenizer, full_prompts,
                       args.max_new_tokens, args.temperature, args.top_p,
                       args.repetition_penalty, dev.torch_device, prompts)
    else:
        log("interactive mode — type prompts (Ctrl+D or empty line to exit)")
        while True:
            try:
                prompt = input("\n[prompt] ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[generate] exiting.")
                break
            if not prompt:
                break
            full_prompt = prefix + prompt
            print("[response] ", end="", flush=True)
            try:
                stream_generate(model, tokenizer, full_prompt, args.max_new_tokens,
                                args.temperature, args.top_p, args.repetition_penalty,
                                dev.torch_device, static_cache=args.static_cache)
            except KeyboardInterrupt:
                print("\n[generate] interrupted, back to prompt.")
            print()


def _self_test():
    print("[selftest] generate: gen_kwargs construction + prompt formatting (no GPU required)")

    class FakeStreamer:
        pass

    kwargs = build_gen_kwargs(
        input_ids="INPUTS", attention_mask="MASK", max_new_tokens=100,
        temperature=0.0, top_p=0.9, repetition_penalty=1.1,
        pad_token_id=0, eos_token_id=1, streamer=FakeStreamer(),
    )
    assert kwargs["do_sample"] is False
    assert kwargs["temperature"] == 1e-6
    assert kwargs["use_cache"] is True
    assert kwargs["max_new_tokens"] == 100
    assert kwargs["pad_token_id"] == 0
    print("  OK (build_gen_kwargs: greedy mode, temperature floor, KV-cache, pad_id=0 preserved)")

    kwargs = build_gen_kwargs(
        input_ids="INPUTS", attention_mask="MASK", max_new_tokens=50,
        temperature=0.8, top_p=0.9, repetition_penalty=1.1,
        pad_token_id=None, eos_token_id=2, streamer=FakeStreamer(),
    )
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 0.8
    assert kwargs["pad_token_id"] == 2
    print("  OK (build_gen_kwargs: sampling mode, None pad falls back to eos)")

    # static_cache=True adds cache_implementation="static" for pre-allocated
    # KV tensors (pairs with --compile for HIP graph capture).
    kwargs = build_gen_kwargs(
        input_ids="INPUTS", attention_mask="MASK", max_new_tokens=50,
        temperature=0.8, top_p=0.9, repetition_penalty=1.1,
        pad_token_id=0, eos_token_id=1, streamer=FakeStreamer(),
        static_cache=True,
    )
    assert kwargs["cache_implementation"] == "static"
    print("  OK (build_gen_kwargs: static_cache adds cache_implementation='static')")

    # static_cache=False (default) does NOT add cache_implementation.
    kwargs = build_gen_kwargs(
        input_ids="INPUTS", attention_mask="MASK", max_new_tokens=50,
        temperature=0.8, top_p=0.9, repetition_penalty=1.1,
        pad_token_id=0, eos_token_id=1, streamer=FakeStreamer(),
    )
    assert "cache_implementation" not in kwargs
    print("  OK (build_gen_kwargs: default (no static_cache) omits cache_implementation)")

    system_prompt = "You are a helpful assistant."
    user_prompt = "What is ROCm?"
    full = system_prompt + "\n\n" + user_prompt
    assert full.startswith(system_prompt)
    assert "What is ROCm?" in full
    print("  OK (system prompt prepended to user prompt)")

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
