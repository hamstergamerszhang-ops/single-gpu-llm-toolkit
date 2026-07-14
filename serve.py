#!/usr/bin/env python3
"""OpenAI-compatible inference server backed by a trained checkpoint.

Exposes POST /v1/chat/completions (OpenAI request/response shape) over a
FastAPI + uvicorn server. Loads the model once at startup using
AutoModelForCausalLM.from_pretrained under rocm_env.setup_rocm_env, then
serves autoregressive generation with the KV-cache enabled
(use_cache=True) and torch.inference_mode() around every generate() call.

Supports the same AMD-specific optimization flags as generate.py:
--flash-attn, --dtype fp8 (weight-only quantization via torchao),
--compile, --gfx-override, --hip-alloc-conf. Generation kwargs are built
with generate.py's build_gen_kwargs() so sampling/temperature/KV-cache
behavior is identical to the CLI tool.

Streaming: when the request body has "stream": true, the server emits
token chunks as Server-Sent Events (SSE) in the OpenAI streaming format
(data: {chunk}\\n\\n, terminated by data: [DONE]\\n\\n). Non-streaming
requests generate `n` completions in a single batched model.generate()
call (num_return_sequences=n) and return one JSON object.

HONESTY NOTE on "batched generation": OpenAI /v1/chat/completions takes a
single conversation (`messages`) per request, so the batching here is
within one request -- generating the `n` requested completions together.
This server does NOT do cross-request continuous batching (the vLLM-style
scheduler that packs tokens from many concurrent requests into one
forward pass); each request runs its own generate() call. For
single-user or modest-concurrency serving of a fine-tuned checkpoint
that is fine and keeps the file self-contained; for high-throughput
multi-tenant serving use vLLM or TGI.

Optional dependencies: fastapi, uvicorn, and pydantic are required to
actually serve (they are imported lazily so that --selftest can run
without them). If they are not installed, main() raises SystemExit with
install instructions rather than crashing at import time.

Usage:
    python3 serve.py --model ./checkpoints/model_cpt_1 --host 0.0.0.0 --port 8000
    python3 serve.py --model ./checkpoints/model_cpt_1 --flash-attn --dtype fp8
    python3 serve.py --model ./checkpoints/model_cpt_1 --compile --gfx-override gfx1100

Self-test (no GPU, no fastapi required -- exercises request parsing,
response building, SSE formatting, and -- when fastapi is importable --
the FastAPI route definitions with a fake model):
    python3 serve.py --selftest
"""

import argparse
import json
import sys
import time
import uuid


def log(msg: str):
    print(f"[serve] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Generation kwargs -- mirror generate.py's build_gen_kwargs so the server
# and the CLI tool sample identically. Imported lazily so --selftest (which
# has no torch/transformers) doesn't have to import generate.py's heavy deps.
# ---------------------------------------------------------------------------
def build_gen_kwargs(max_new_tokens, temperature, top_p,
                     repetition_penalty, pad_token_id, eos_token_id, streamer=None,
                     num_return_sequences=1):
    """Build the kwargs dict for model.generate().

    Identical sampling semantics to generate.py.build_gen_kwargs: greedy when
    temperature <= 0 (temperature floored to 1e-6 to satisfy HF's >0 check),
    pad_token_id falls back to eos_token_id when None, use_cache=True always.
    """
    pad_id = pad_token_id if pad_token_id is not None else eos_token_id
    return dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-6),
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=pad_id,
        eos_token_id=eos_token_id,
        streamer=streamer,
        use_cache=True,
        num_return_sequences=num_return_sequences,
    )


# ---------------------------------------------------------------------------
# Pure request/response helpers -- no fastapi, no torch, no transformers.
# Imported by the self-test so it runs on a bare box. The FastAPI route
# handlers below delegate to these.
# ---------------------------------------------------------------------------
DEFAULT_GEN_PARAMS = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}


def parse_chat_request(req):
    """Validate and normalize an OpenAI /v1/chat/completions request body.

    Args:
        req: dict (already JSON-decoded) with at least {"messages": [...]}.

    Returns:
        (params, error_response)
        - params: dict with keys messages, model, stream, n, and the resolved
          generation params (max_new_tokens, temperature, top_p,
          repetition_penalty). Defaults applied for omitted fields.
        - error_response: None on success, else a dict shaped like an OpenAI
          error response (with a 'status_code' hint for the handler).
    """
    if not isinstance(req, dict):
        return None, {"error": {"message": "request body must be a JSON object",
                                "type": "invalid_request_error"},
                      "status_code": 400}

    messages = req.get("messages")
    if not isinstance(messages, list) or not messages:
        return None, {"error": {"message": "'messages' must be a non-empty array",
                                "type": "invalid_request_error"},
                      "status_code": 400}
    for m in messages:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            return None, {"error": {"message": "each message needs 'role' and 'content'",
                                    "type": "invalid_request_error"},
                          "status_code": 400}

    n = req.get("n", 1)
    try:
        n = int(n)
    except (TypeError, ValueError):
        return None, {"error": {"message": "'n' must be an integer",
                                "type": "invalid_request_error"},
                      "status_code": 400}
    if n < 1:
        return None, {"error": {"message": "'n' must be >= 1",
                                "type": "invalid_request_error"},
                      "status_code": 400}

    # OpenAI uses 'max_tokens'; accept 'max_new_tokens' too for convenience.
    max_tokens = req.get("max_tokens", req.get("max_new_tokens",
                                               DEFAULT_GEN_PARAMS["max_new_tokens"]))
    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        return None, {"error": {"message": "'max_tokens' must be an integer",
                                "type": "invalid_request_error"},
                      "status_code": 400}
    if max_tokens < 1:
        return None, {"error": {"message": "'max_tokens' must be >= 1",
                                "type": "invalid_request_error"},
                      "status_code": 400}

    temperature = float(req.get("temperature", DEFAULT_GEN_PARAMS["temperature"]))
    top_p = float(req.get("top_p", DEFAULT_GEN_PARAMS["top_p"]))
    repetition_penalty = float(req.get("repetition_penalty",
                                       DEFAULT_GEN_PARAMS["repetition_penalty"]))

    params = {
        "messages": messages,
        "model": req.get("model", "gemma-persona"),
        "stream": bool(req.get("stream", False)),
        "n": n,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
    }
    return params, None


def messages_to_prompt(messages, tokenizer):
    """Render an OpenAI messages array into a single prompt string.

    Uses the tokenizer's chat template when available (apply_chat_template),
    which is the correct path for any model that ships a tokenizer_config
    chat_template. Falls back to a minimal role-prefixed concatenation for
    tokenizers without a template (enough to produce a prompt in dev).
    """
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if prompt:
                return prompt
        except Exception as exc:
            # Don't swallow silently (repo convention). Fall through to the
            # manual renderer, but surface why.
            log(f"apply_chat_template failed ({exc!r}) -- using manual prompt")

    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n\n".join(parts)


def _new_id():
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def _iso_now():
    return datetime_iso_now()


def datetime_iso_now():
    """ISO-8601 UTC timestamp with second precision, OpenAI-style."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_completion_response(params, completions, finish_reason="stop",
                              prompt_tokens=0, completion_tokens=0):
    """Build the OpenAI /v1/chat/completions non-streaming response dict.

    `completions` is a list of generated text strings (length == params['n']).
    """
    rid = _new_id()
    choices = []
    for i, text in enumerate(completions):
        choices.append({
            "index": i,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish_reason,
        })
    return {
        "id": rid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": params.get("model", "gemma-persona"),
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def sse_chunk(params, content=None, role=None, finish_reason=None,
              index=0, created=None):
    """Build one OpenAI streaming chunk as a dict (caller wraps as SSE text)."""
    delta = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    choice = {"index": index, "delta": delta, "finish_reason": finish_reason}
    chunk = {
        "id": _STREAM_ID,
        "object": "chat.completion.chunk",
        "created": created if created is not None else int(time.time()),
        "model": params.get("model", "gemma-persona"),
        "choices": [choice],
    }
    return chunk


# A single response id is reused across all chunks of one streamed completion
# (OpenAI does this). Set per-request by the streaming handler.
_STREAM_ID = _new_id()


def format_sse_stream(text_chunks, params, prompt_tokens=0,
                      completion_token_counter=None):
    """Yield OpenAI SSE strings for one streamed completion.

    Args:
        text_chunks: iterable of decoded text fragments (e.g. from a
            TextIteratorStreamer).
        params: normalized request params (for model name).
        completion_token_counter: optional callable returning the current
            completion token count, called once for the final usage chunk.
            If None, usage.completion_tokens is omitted from the final chunk.

    Yields: bytes (utf-8) ready for StreamingResponse. Each yield is one
    `data: {...}\\n\\n` frame; the final `data: [DONE]\\n\\n` is yielded last.
    """
    rid = _new_id()
    created = int(time.time())
    # First chunk: role announcement, empty content.
    first = sse_chunk(params, role="assistant", finish_reason=None,
                      created=created)
    first["id"] = rid
    yield ("data: " + json.dumps(first) + "\n\n").encode("utf-8")

    for piece in text_chunks:
        if piece is None:
            continue
        chunk = sse_chunk(params, content=piece, finish_reason=None,
                          created=created)
        chunk["id"] = rid
        yield ("data: " + json.dumps(chunk) + "\n\n").encode("utf-8")

    # Final chunk: empty delta, finish_reason=stop, optional usage.
    final = sse_chunk(params, finish_reason="stop", created=created)
    final["id"] = rid
    if completion_token_counter is not None:
        ct = int(completion_token_counter() or 0)
        final["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": ct,
            "total_tokens": prompt_tokens + ct,
        }
    yield ("data: " + json.dumps(final) + "\n\n").encode("utf-8")
    yield b"data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Model-bound generation functions (torch + transformers, lazily imported).
# These are the real implementations; the FastAPI app calls them through the
# ServerState so the self-test can swap in fakes.
# ---------------------------------------------------------------------------
def real_generate_stream(model, tokenizer, prompt, params, device):
    """Yield decoded text fragments for one streamed completion.

    Runs model.generate() on a background thread with a TextIteratorStreamer
    (same pattern as generate.py.stream_generate), under torch.inference_mode().
    KV-cache is enabled via build_gen_kwargs(use_cache=True).
    """
    import queue
    from threading import Thread

    import torch
    from transformers import TextIteratorStreamer

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=30.0
    )
    gen_kwargs = build_gen_kwargs(
        params["max_new_tokens"], params["temperature"], params["top_p"],
        params["repetition_penalty"], tokenizer.pad_token_id,
        tokenizer.eos_token_id, streamer=streamer, num_return_sequences=1,
    )
    gen_kwargs["inputs"] = inputs["input_ids"]
    gen_kwargs["attention_mask"] = inputs["attention_mask"]

    thread_exc = []

    def _run():
        try:
            with torch.inference_mode():
                model.generate(**gen_kwargs)
        except Exception as exc:
            thread_exc.append(exc)

    t = Thread(target=_run, daemon=True)
    t.start()
    try:
        for piece in streamer:
            yield piece
    except (StopIteration, queue.Empty):
        pass
    t.join(timeout=5.0)
    if thread_exc:
        raise thread_exc[0]


def real_generate_batch(model, tokenizer, prompt, params, device):
    """Generate `n` completions for one prompt in a single batched
    model.generate() call (num_return_sequences=n), under
    torch.inference_mode(). Returns (texts, prompt_tokens, completion_tokens).
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    n = params["n"]
    gen_kwargs = build_gen_kwargs(
        params["max_new_tokens"], params["temperature"], params["top_p"],
        params["repetition_penalty"], tokenizer.pad_token_id,
        tokenizer.eos_token_id, streamer=None, num_return_sequences=n,
    )
    gen_kwargs["inputs"] = inputs["input_ids"]
    gen_kwargs["attention_mask"] = inputs["attention_mask"]

    with torch.inference_mode():
        out_ids = model.generate(**gen_kwargs)

    in_len = inputs["input_ids"].shape[-1]
    texts = []
    total_new = 0
    for row in out_ids:
        new_tokens = row[in_len:]
        texts.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
        total_new += int(new_tokens.shape[-1])
    prompt_tokens = int(in_len)
    return texts, prompt_tokens, total_new


# ---------------------------------------------------------------------------
# Server state + FastAPI app. fastapi/pydantic imported lazily here so the
# self-test can construct the pure helpers above without them.
# ---------------------------------------------------------------------------
class ServerState:
    """Holds everything the route handlers need, including the generate
    callables. The callables are injectable so the self-test can pass fakes
    and exercise the route wiring with no GPU and no model."""

    def __init__(self, model, tokenizer, device, model_id, defaults=None,
                 generate_stream_fn=None, generate_batch_fn=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_id = model_id
        self.defaults = defaults or dict(DEFAULT_GEN_PARAMS)
        self.generate_stream_fn = generate_stream_fn or real_generate_stream
        self.generate_batch_fn = generate_batch_fn or real_generate_batch

    # Apply default gen params for fields the request omitted. (parse_chat_request
    # already fills defaults, so this is mostly a no-op; kept for explicitness
    # when the server is run with non-default CLI overrides later.)
    def resolve_params(self, params):
        for k, v in self.defaults.items():
            params.setdefault(k, v)
        return params


def handle_chat_request(state, req_dict):
    """Core endpoint logic, factored out of the route handler so it can be
    exercised directly (by the self-test) without going through FastAPI's HTTP
    / pydantic stack.

    Args:
        state: ServerState.
        req_dict: plain dict request body (already JSON-decoded, e.g. the
            output of ChatRequest.model_dump()).

    Returns:
        - For non-streaming: a plain dict (OpenAI chat.completion shape) that
          FastAPI JSON-encodes.
        - For streaming: a fastapi.responses.StreamingResponse whose
          body_iterator yields SSE bytes.
        - On a validation error: a fastapi.responses.JSONResponse with the
          appropriate status code.
    """
    from fastapi.responses import JSONResponse, StreamingResponse

    params, err = parse_chat_request(req_dict)
    if err is not None:
        return JSONResponse(status_code=err["status_code"],
                            content={"error": err["error"]})
    params = state.resolve_params(params)

    prompt = messages_to_prompt(params["messages"], state.tokenizer)

    if params["stream"]:
        text_chunks = state.generate_stream_fn(
            state.model, state.tokenizer, prompt, params, state.device
        )
        return StreamingResponse(
            format_sse_stream(text_chunks, params, prompt_tokens=0),
            media_type="text/event-stream",
        )

    texts, prompt_tokens, completion_tokens = state.generate_batch_fn(
        state.model, state.tokenizer, prompt, params, state.device
    )
    return build_completion_response(
        params, texts, finish_reason="stop",
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
    )


def build_app(state):
    """Create the FastAPI app. fastapi + pydantic are imported here (lazily)
    so the module loads without them for --selftest."""
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError as exc:
        raise SystemExit(
            "ERROR: fastapi/uvicorn/pydantic are required to serve.\n"
            "  pip install fastapi uvicorn[standard] pydantic\n"
            f"  (missing: {exc!r})"
        ) from exc

    app = FastAPI(title="gemma-persona serve", version="1.0")

    # Pydantic models for the request body. Kept permissive (extra='allow') so
    # future OpenAI fields don't 422; we only read what we understand.
    class Message(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str | None = None
        messages: list[Message]
        temperature: float | None = None
        top_p: float | None = None
        n: int | None = None
        stream: bool | None = None
        max_tokens: int | None = None
        max_new_tokens: int | None = None
        repetition_penalty: float | None = None

        model_config = {"extra": "allow"}

    @app.get("/health")
    def health():
        return {"status": "ok", "model": state.model_id}

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [
            {"id": state.model_id, "object": "model", "owned_by": "gemma-persona"}
        ]}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest):
        # Delegate to the factored-out handler so the same code path is
        # exercised by --selftest.
        return handle_chat_request(state, req.model_dump(exclude_none=True))

    return app


# ---------------------------------------------------------------------------
# Model loading -- mirrors generate.py._load_model_and_tokenizer, reusing the
# repo's runtime probe / backend / rocm_env machinery.
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(args, dev):
    """Load model + tokenizer, apply dtype/fp8/flash-attn/compile, return
    (model, tokenizer). Mirrors generate.py so the server and CLI behave the
    same at the torch level."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from runtime import resolve_compile, resolve_dtype, resolve_flash_attn

    dtype_str = resolve_dtype(dev, args.dtype)
    torch_dtype = {
        "fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16,
        "fp8": torch.bfloat16,  # fp8 loads as bf16 then weight-only quantizes
    }[dtype_str]

    log(f"loading model from {args.model} on {dev} (dtype={dtype_str}) ...")
    load_kwargs = {"torch_dtype": torch_dtype, "trust_remote_code": True}
    if resolve_flash_attn(dev, args.flash_attn):
        load_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.to(dev.torch_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # fp8 weight-only inference via torchao (same guard as generate.py).
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
        except Exception as exc:
            log(f"WARNING: fp8 inference failed ({exc}) -- using bf16")

    if resolve_compile(dev, args.compile, mode=args.compile_mode):
        try:
            model = torch.compile(model, mode=args.compile_mode)
            log(f"torch.compile enabled (mode={args.compile_mode})")
        except Exception as exc:
            log(f"WARNING: compile failed ({exc}), using eager")

    # KV-cache MUST be on for fast autoregressive decoding.
    model.config.use_cache = True
    model.eval()
    return model, tokenizer


def _check_serve_deps():
    """Raise SystemExit with install instructions if fastapi/uvicorn missing."""
    missing = []
    try:
        import fastapi  # noqa: F401
    except ImportError:
        missing.append("fastapi")
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        missing.append("uvicorn")
    if missing:
        raise SystemExit(
            "ERROR: serving requires packages that are not installed: "
            + ", ".join(missing) + "\n"
            "  Install them with:\n"
            "    pip install 'fastapi>=0.110' 'uvicorn[standard]>=0.29' pydantic\n"
        )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True,
                    help="Checkpoint / HF model directory to serve.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)

    # Optimization flags -- same as generate.py.
    ap.add_argument("--flash-attn", action="store_true", default=False)
    ap.add_argument("--dtype", default="bf16",
                    choices=["fp32", "fp16", "bf16", "fp8"])
    ap.add_argument("--compile", action="store_true", default=False)
    ap.add_argument("--compile-mode", default="max-autotune",
                    choices=["default", "reduce-overhead", "max-autotune"])

    # ROCm bootstrap.
    ap.add_argument("--backend", default=None, choices=["rocm", "cpu"])
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--gfx-override", default=None)
    ap.add_argument("--hip-alloc-conf", default="expandable_segments:True")

    args = ap.parse_args()

    # fastapi/uvicorn gate BEFORE touching torch (no point warming up a model
    # if we can't serve it).
    _check_serve_deps()

    # ROCm bootstrap before torch import (same ordering as generate.py).
    from backends import get_backend
    backend = get_backend(args.backend) if args.backend else None
    if backend is None or backend.name == "rocm":
        from rocm_env import setup_rocm_env
        hip_conf = (None if args.hip_alloc_conf.lower() == "none"
                    else args.hip_alloc_conf)
        setup_rocm_env(override=args.gfx_override, hip_alloc_conf=hip_conf)

    import torch  # noqa: F401  (imported for side effects / availability)

    from backends import BackendDevice
    dev = BackendDevice(backend=args.backend, index=args.device_index)
    if not dev.backend.is_available():
        raise SystemExit(f"ERROR: backend {dev.name} is not available.")

    model, tokenizer = load_model_and_tokenizer(args, dev)

    state = ServerState(
        model=model, tokenizer=tokenizer, device=dev.torch_device,
        model_id=args.model,
    )
    app = build_app(state)

    import uvicorn
    log(f"serving {args.model} on http://{args.host}:{args.port} "
        f"(dtype={args.dtype}, flash_attn={args.flash_attn}, "
        f"compile={args.compile})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


# ---------------------------------------------------------------------------
# Self-test. No GPU, no fastapi required. Exercises the pure request/response
# helpers always; exercises the FastAPI route wiring + a fake-model end-to-end
# request ONLY when fastapi is importable (honestly skipped otherwise).
# ---------------------------------------------------------------------------
class _FakeTokenizer:
    """Minimal tokenizer stand-in: has apply_chat_template (returns a
    recognizable prompt) and the pad/eos ids generate() needs."""

    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # Render to a single string so we can assert on it.
        return "\n".join(f"[{m['role']}] {m['content']}" for m in messages) + "\n[assistant]"

    def __call__(self, prompt, return_tensors="pt"):
        # Return an object with .to(device); shape used only for slicing.
        class _T:
            def __init__(self, ids):
                self.input_ids = ids
                self.attention_mask = ids

            def to(self, _dev):
                return self

        class _Ids:
            def __init__(self, n):
                self.shape = [-1, n]

            def __getitem__(self, idx):
                return self

        n = len(prompt.split())
        return _T(_Ids(n))


def _fake_generate_stream(model, tokenizer, prompt, params, device):
    """Yield a canned token stream -- stands in for real_generate_stream."""
    for piece in ["Hello", ", ", "world", "!"]:
        yield piece


def _fake_generate_batch(model, tokenizer, prompt, params, device):
    """Return n canned completions -- stands in for real_generate_batch."""
    n = params["n"]
    texts = [f"completion #{i} for {prompt[:12]!r}" for i in range(n)]
    return texts, 5, n * 7


def _self_test():
    print("[selftest] serve: request parsing + response building + SSE "
          "(no GPU required)")

    # --- parse_chat_request: defaults + validation ---
    params, err = parse_chat_request({
        "model": "m", "messages": [{"role": "user", "content": "hi"}]
    })
    assert err is None, f"unexpected error: {err}"
    assert params["stream"] is False
    assert params["n"] == 1
    assert params["max_new_tokens"] == DEFAULT_GEN_PARAMS["max_new_tokens"]
    assert params["temperature"] == DEFAULT_GEN_PARAMS["temperature"]
    assert params["model"] == "m"
    print("  OK (defaults applied: stream=False, n=1, max_tokens/temperature)")

    # stream=true + custom sampling + max_tokens alias.
    params, err = parse_chat_request({
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True, "temperature": 0.2, "top_p": 0.8, "max_tokens": 32, "n": 3,
    })
    assert err is None
    assert params["stream"] is True
    assert params["n"] == 3
    assert params["max_new_tokens"] == 32
    print("  OK (stream=true, n=3, max_tokens alias, custom sampling)")

    # Validation errors.
    _, err = parse_chat_request({"messages": []})
    assert err and err["status_code"] == 400
    _, err = parse_chat_request({"messages": [{"role": "user"}]})
    assert err and err["status_code"] == 400
    _, err = parse_chat_request({"messages": [{"role": "u", "content": "c"}], "n": 0})
    assert err and err["status_code"] == 400
    _, err = parse_chat_request({"messages": [{"role": "u", "content": "c"}], "max_tokens": -1})
    assert err and err["status_code"] == 400
    print("  OK (rejects empty messages, missing content, n<1, max_tokens<1)")

    # --- build_gen_kwargs: mirrors generate.py semantics ---
    kw = build_gen_kwargs(100, 0.0, 0.9, 1.1, 0, 1, num_return_sequences=1)
    assert kw["do_sample"] is False
    assert kw["temperature"] == 1e-6
    assert kw["use_cache"] is True
    assert kw["max_new_tokens"] == 100
    assert kw["pad_token_id"] == 0
    assert kw["num_return_sequences"] == 1
    print("  OK (build_gen_kwargs: greedy floor, KV-cache on, num_return_sequences)")

    kw = build_gen_kwargs(50, 0.8, 0.9, 1.1, None, 2, num_return_sequences=4)
    assert kw["do_sample"] is True
    assert kw["temperature"] == 0.8
    assert kw["pad_token_id"] == 2  # None pad falls back to eos
    assert kw["num_return_sequences"] == 4
    print("  OK (build_gen_kwargs: sampling, None pad -> eos, n=4)")

    # --- messages_to_prompt: chat-template path + fallback ---
    tok = _FakeTokenizer()
    prompt = messages_to_prompt(
        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
        tok,
    )
    assert "[user] Hello" in prompt and "[assistant] Hi" in prompt
    print("  OK (messages_to_prompt via apply_chat_template)")

    # Fallback when tokenizer has no chat template.
    class _BareTok:
        pad_token_id = 0
        eos_token_id = 1

    prompt2 = messages_to_prompt([{"role": "user", "content": "Ping"}], _BareTok())
    assert "user: Ping" in prompt2 and prompt2.rstrip().endswith("assistant:")
    print("  OK (messages_to_prompt manual fallback when no template)")

    # --- build_completion_response ---
    params, _ = parse_chat_request(
        {"messages": [{"role": "user", "content": "hi"}], "model": "m", "n": 2})
    resp = build_completion_response(params, ["ans A", "ans B"],
                                     prompt_tokens=3, completion_tokens=9)
    assert resp["object"] == "chat.completion"
    assert resp["model"] == "m"
    assert len(resp["choices"]) == 2
    assert resp["choices"][0]["message"]["content"] == "ans A"
    assert resp["choices"][1]["index"] == 1
    assert resp["usage"]["total_tokens"] == 12
    assert resp["id"].startswith("chatcmpl-")
    print("  OK (completion response: 2 choices, usage totals, id prefix)")

    # --- SSE streaming format ---
    params, _ = parse_chat_request(
        {"messages": [{"role": "user", "content": "hi"}], "stream": True})
    frames = list(format_sse_stream(["He", "llo"], params))
    # first (role), 2 content, final (stop), [DONE] => 5 frames
    assert len(frames) == 5, f"expected 5 SSE frames, got {len(frames)}"
    assert all(f.startswith(b"data: ") for f in frames)
    assert frames[-1] == b"data: [DONE]\n\n"
    first_obj = json.loads(frames[0][len("data: "):].strip())
    assert first_obj["object"] == "chat.completion.chunk"
    assert first_obj["choices"][0]["delta"] == {"role": "assistant"}
    content_obj = json.loads(frames[1][len("data: "):].strip())
    assert content_obj["choices"][0]["delta"] == {"content": "He"}
    final_obj = json.loads(frames[3][len("data: "):].strip())
    assert final_obj["choices"][0]["finish_reason"] == "stop"
    # Reused response id across all chunks.
    assert first_obj["id"] == content_obj["id"] == final_obj["id"]
    print("  OK (SSE: role chunk -> content chunks -> stop chunk -> [DONE], "
          "shared id)")

    # Empty-text stream still emits role + stop + [DONE].
    frames = list(format_sse_stream([], params))
    assert len(frames) == 3
    print("  OK (SSE: empty stream still well-formed)")

    # --- FastAPI route wiring (only if fastapi importable) ---
    try:
        import fastapi  # noqa: F401
        import pydantic  # noqa: F401
    except ImportError:
        print("  SKIP (fastapi/pydantic not installed -- route-wiring test "
              "skipped; pure helpers above all passed).")
        print("\n[selftest] All checks passed (fastapi route wiring skipped: "
              "fastapi not installed).")
        return

    state = ServerState(
        model=None, tokenizer=_FakeTokenizer(), device="cpu",
        model_id="fake-model",
        generate_stream_fn=_fake_generate_stream,
        generate_batch_fn=_fake_generate_batch,
    )
    app = build_app(state)
    routes = {getattr(r, "path", None) for r in app.routes}
    assert "/v1/chat/completions" in routes, f"route missing: {routes}"
    assert "/v1/models" in routes
    assert "/health" in routes
    print("  OK (FastAPI app built; /v1/chat/completions, /v1/models, "
          "/health registered)")

    # End-to-end non-streaming request through the factored-out handler with
    # the fake model. This is the exact code path the route handler calls.
    resp = handle_chat_request(state, {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "hello world"}],
        "n": 2, "max_tokens": 16,
    })
    assert isinstance(resp, dict), f"expected dict, got {type(resp)}"
    assert resp["object"] == "chat.completion"
    assert len(resp["choices"]) == 2
    assert "completion #" in resp["choices"][0]["message"]["content"]
    assert resp["model"] == "fake-model"
    print("  OK (non-streaming request through handler with fake model: "
          "2 completions returned)")

    # Validation error -> JSONResponse with 400.
    err_resp = handle_chat_request(state, {"messages": []})
    assert getattr(err_resp, "status_code", None) == 400
    print("  OK (validation error -> JSONResponse 400)")

    # Streaming path -> StreamingResponse; drain its body to confirm the fake
    # stream + SSE formatter agree end-to-end. Starlette may expose the body
    # as either a sync iterable or an async generator depending on version, so
    # handle both.
    sresp = handle_chat_request(state, {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })
    assert getattr(sresp, "media_type", None) == "text/event-stream"
    body_iter = getattr(sresp, "body_iterator", None) or sresp.body
    try:
        drained = list(body_iter)
    except TypeError:
        # Async generator (newer Starlette wraps sync iterables).
        import asyncio

        async def _drain(gen):
            out = []
            async for item in gen:
                out.append(item)
            return out

        drained = asyncio.run(_drain(body_iter))
    assert drained, "stream produced no frames"
    assert all(f.startswith(b"data: ") for f in drained)
    assert drained[-1] == b"data: [DONE]\n\n"
    print("  OK (streaming request through handler: SSE frames drained, "
          "[DONE] terminator present)")

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
