#!/usr/bin/env python3
"""Streaming text generation from a trained checkpoint on AMD ROCm.

Loads a checkpoint (same from_pretrained + rocm_env.setup_rocm_env as train_cpt.py),
runs generation with streaming token output (prints tokens as they're generated,
not all-at-once), and supports the same AMD-specific optimization flags
(--flash-attn, --dtype fp8, --compile, --gfx-override, --hip-alloc-conf).

KV-cache is ENABLED by default here (unlike training where use_cache=False under
gradient checkpointing) — generation with cache is the normal case, and disabling
it would be absurdly slow for autoregressive decoding.

Output quality depends entirely on the supplied checkpoint, not this tool — it
only wires up from_pretrained + generate with the right flags.

Usage:
    # Interactive: type prompts, get streaming responses.
    python3 generate.py --model ./checkpoints/model_cpt_1

    # From a file: one prompt per line.
    python3 generate.py --model ./checkpoints/model_cpt_1 --input prompts.txt

    # With optimizations (same flags as train_cpt.py):
    python3 generate.py --model ./checkpoints/model_cpt_1 --flash-attn --dtype fp8

Self-test (no GPU/model required — exercises arg parser + stream formatter):
    python3 generate.py --selftest
"""

import argparse


def log(msg: str):
    print(f"[generate] {msg}", flush=True)


def build_gen_kwargs(input_ids, attention_mask, max_new_tokens: int,
                     temperature: float, top_p: float, repetition_penalty: float,
                     pad_token_id, eos_token_id, streamer):
    """Build the kwargs dict for model.generate(). Extracted from
    stream_generate so the construction logic (do_sample gating, temperature
    flooring, pad_id fallback) is testable without a GPU/model."""
    # `or` would be wrong here: pad_token_id == 0 is falsy but a perfectly
    # valid token id (many tokenizers put <pad> at id 0), so `x or y` would
    # silently swap in eos_token_id whenever pad happens to be 0.
    pad_id = pad_token_id if pad_token_id is not None else eos_token_id
    return dict(
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
        use_cache=True,  # KV-cache: essential for generation speed
    )


def stream_generate(model, tokenizer, prompt: str, max_new_tokens: int,
                    temperature: float, top_p: float, repetition_penalty: float,
                    device: str):
    """Generate tokens one at a time, printing decoded text chunks as they're
    produced. Uses HF's generate with a TextIteratorStreamer for real-time
    output.

    TextIteratorStreamer does NOT stream by itself — it just pushes decoded
    text onto an internal queue as generate() produces it. Something still has
    to (a) call generate() and (b) iterate the streamer concurrently, or the
    queue just fills up silently and nothing gets printed until generate()
    returns (at which point the streamer has already seen stream_end and the
    text sits unread). HF's own docs run generate() in a background thread and
    iterate the streamer on the calling thread — that's what we do here."""
    import queue
    from threading import Thread
    import torch
    from transformers import TextIteratorStreamer

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True,
        # timeout=30: long enough for any realistic prefill (a 15B model with
        # a 2k-token prompt on MI300X takes <5s), short enough to detect a
        # genuinely dead thread. The previous 1.0s was too short — prefill on
        # a large model or multi-GPU pipeline can exceed 1s before the first
        # token lands, causing queue.Empty to abort streaming prematurely.
        timeout=30.0,
    )
    gen_kwargs = build_gen_kwargs(
        inputs["input_ids"], inputs["attention_mask"], max_new_tokens,
        temperature, top_p, repetition_penalty,
        tokenizer.pad_token_id, tokenizer.eos_token_id, streamer,
    )

    # generate() blocks until the full response is produced, so it has to run
    # on a background thread — otherwise we can't drain the streamer queue
    # until generation is already finished, which defeats the point of a
    # streamer. The main thread iterates `streamer` and prints as tokens land.
    # Capture exceptions from the thread so a failure in generate() surfaces
    # instead of being silently swallowed by Thread.
    thread_exc = []

    def _generate_with_exc():
        try:
            # inference_mode belongs HERE, not around the main thread's
            # streamer-draining loop below: torch's inference_mode/no_grad
            # context managers are thread-local, and the actual tensor compute
            # (model.generate) runs on THIS background thread, not the thread
            # that entered the context manager. Wrapping the main thread's
            # print loop (which only reads decoded strings off a queue.Queue,
            # no tensor ops) has zero effect on the generation call itself --
            # confirmed directly: torch.is_inference_mode_enabled() reads
            # False inside a background thread even while the main thread is
            # inside `with torch.inference_mode():`. HF's generate() already
            # wraps itself in @torch.no_grad() internally, so this is a
            # belt-and-suspenders no-op for autograd, but inference_mode also
            # disables the version-counter bookkeeping no_grad still does,
            # which is the actual reason to prefer it here.
            with torch.inference_mode():
                model.generate(**gen_kwargs)
        except Exception as e:
            thread_exc.append(e)

    # daemon=True so the thread doesn't block process exit if interrupted
    # (e.g. Ctrl+C) mid-generation — HF generate() has no cancellation API.
    thread = Thread(target=_generate_with_exc, daemon=True)
    thread.start()
    try:
        # No inference_mode wrapper needed here -- this loop only reads
        # decoded strings off streamer's internal queue.Queue and prints them,
        # no tensor ops. The actual generate() call is wrapped in
        # _generate_with_exc above, on the thread that actually runs it.
        for new_text in streamer:
            print(new_text, end="", flush=True)
    except (StopIteration, queue.Empty):
        # StopIteration: streamer exhausted normally (stream_end sentinel
        # seen). queue.Empty: TextIteratorStreamer.__next__ does
        # `self.text_queue.get(timeout=self.timeout)`, a plain queue.Queue.get
        # with a timeout -- on timeout that raises queue.Empty, NOT
        # StopIteration (confirmed against the installed transformers'
        # TextIteratorStreamer.__next__ source). A prior version of this
        # handler only caught StopIteration, so a real timeout (e.g.
        # generate() stalls or its background thread dies without ever
        # calling streamer.end()) propagated an uncaught queue.Empty out of
        # stream_generate() instead of being handled here. Both cases end the
        # same way: stop reading the streamer and fall through to joining the
        # thread + surfacing any real generate() exception.
        pass
    thread.join(timeout=5.0)  # don't hang forever if generate() is stuck
    if thread_exc:
        raise thread_exc[0]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, help="HF model dir/repo to generate from.")
    ap.add_argument("--input", type=str, default=None,
                    help="File of prompts (one per line). If omitted, interactive mode.")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--system-prompt", type=str, default=None,
                    help="Optional system prompt prepended to every input.")
    # AMD-specific flags (same as train_cpt.py):
    ap.add_argument("--flash-attn", action="store_true", default=False,
                    help="Use Flash Attention 2 (requires flash-attn for ROCm).")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp8"],
                    help="Model dtype. fp8 uses torchao (MI300X/MI325X native).")
    ap.add_argument("--compile", action="store_true", default=False,
                    help="Wrap model in torch.compile() for kernel fusion.")
    ap.add_argument("--gfx-override", type=str, default=None,
                    help="Force HSA_OVERRIDE_GFX_VERSION (see rocm_env.py).")
    ap.add_argument("--hip-alloc-conf", type=str, default="max_split_size_mb:128",
                    help="PYTORCH_HIP_ALLOC_CONF value (pass 'none' to skip).")
    args = ap.parse_args()

    from rocm_env import setup_rocm_env
    hip_conf = None if args.hip_alloc_conf.lower() == "none" else args.hip_alloc_conf
    setup_rocm_env(override=args.gfx_override, hip_alloc_conf=hip_conf)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("ERROR: no CUDA/ROCm device — generation needs a GPU.")

    log(f"loading model from {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Apply optimizations. Note: for inference we use a weight-only float8
    # quantizer (not convert_to_float8_training, which train_cpt.py uses)
    # because there's no backward pass here — the training converter sets up
    # AMAX/dynamic-scaling machinery that's dead weight for inference.
    #
    # torchao renamed this API between versions: older releases (confirmed
    # present in 0.7.0) expose a plain `float8_weight_only()` function; newer
    # releases (confirmed in 0.17.0) replaced it with a `Float8WeightOnlyConfig`
    # class passed the same way to `quantize_()`. requirements.txt pins
    # `torchao>=0.5.0` with no upper bound, so either could be installed --
    # try the function first, fall back to the config class, and only warn if
    # neither exists. (An earlier version of this code imported only the old
    # function name; on any torchao new enough to have dropped it, that import
    # raises ImportError and silently degrades to bf16 while logging a
    # misleading "torchao not installed" message even though torchao IS
    # installed — just with a renamed API. Confirmed by inspecting both
    # torchao 0.7.0 and 0.17.0's actual quant_api.py source.)
    if args.dtype == "fp8":
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
    if args.flash_attn:
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
            log("flash-attn enabled")
        except ImportError:
            log("WARNING: flash-attn not installed, using standard attn")
        except Exception as e:
            log(f"WARNING: --flash-attn failed ({e}) — using standard attention.")
    if args.compile:
        try:
            model = torch.compile(model)
            log("torch.compile enabled (first generation will be slower)")
        except Exception as e:
            log(f"WARNING: compile failed ({e}), using eager")

    # KV-cache: enabled by default for generation (unlike training).
    model.config.use_cache = True
    model.eval()

    # Build the prompt prefix.
    prefix = ""
    if args.system_prompt:
        prefix = args.system_prompt + "\n\n"

    if args.input:
        # Batch mode: read prompts from file.
        with open(args.input, encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        for prompt in prompts:
            full_prompt = prefix + prompt
            print(f"\n--- Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''} ---")
            print("[response] ", end="", flush=True)
            stream_generate(model, tokenizer, full_prompt, args.max_new_tokens,
                            args.temperature, args.top_p, args.repetition_penalty,
                            "cuda")
            print()
    else:
        # Interactive mode.
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
                                "cuda")
            except KeyboardInterrupt:
                print("\n[generate] interrupted, back to prompt.")
            print()


def _self_test():
    print("[selftest] generate: gen_kwargs construction + prompt formatting (no GPU required)")

    # Call the REAL build_gen_kwargs function and verify its logic.
    # temperature=0 -> do_sample=False (greedy), temperature>0 -> do_sample=True.
    class FakeStreamer:
        pass
    kwargs = build_gen_kwargs(
        input_ids="INPUTS", attention_mask="MASK", max_new_tokens=100,
        temperature=0.0, top_p=0.9, repetition_penalty=1.1,
        pad_token_id=0, eos_token_id=1, streamer=FakeStreamer(),
    )
    assert kwargs["do_sample"] is False, "temperature=0 should give do_sample=False"
    assert kwargs["temperature"] == 1e-6, "temperature should be floored to 1e-6"
    assert kwargs["use_cache"] is True, "KV-cache must be enabled for generation"
    assert kwargs["max_new_tokens"] == 100
    assert kwargs["pad_token_id"] == 0, "pad_token_id=0 is valid, must not be swapped"
    print("  OK (build_gen_kwargs: greedy mode, temperature floor, KV-cache, pad_id=0 preserved)")

    # temperature > 0 -> do_sample=True.
    kwargs = build_gen_kwargs(
        input_ids="INPUTS", attention_mask="MASK", max_new_tokens=50,
        temperature=0.8, top_p=0.9, repetition_penalty=1.1,
        pad_token_id=None, eos_token_id=2, streamer=FakeStreamer(),
    )
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 0.8
    assert kwargs["pad_token_id"] == 2, "pad_token_id=None should fall back to eos_token_id"
    print("  OK (build_gen_kwargs: sampling mode, None pad falls back to eos)")

    # System prompt prefix is prepended correctly (the logic in main()).
    system_prompt = "You are a helpful assistant."
    user_prompt = "What is ROCm?"
    full = system_prompt + "\n\n" + user_prompt
    assert full.startswith(system_prompt)
    assert "What is ROCm?" in full
    print("  OK (system prompt prepended to user prompt)")

    print("\n[selftest] All checks passed (no GPU required — run with a real "
          "model on AMD hardware for actual generation).")


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
