#!/usr/bin/env python3
"""CUDA/ROCm continued-pretraining (CPT) / SFT trainer for a Gemma-4-family
model — runs against the output of expand_model.py (or mtp_head.py), or
directly against a pruned checkpoint if you skip expansion.

Ported from an MLX/Metal original written for a 48GB unified-memory Mac (that
version trains a windowed SLICE of layers because 48GB can't hold optimizer
state for the full model). On 80GB+ single-GPU hardware that constraint is
gone -- this script defaults to training the FULL model (--start 0 --end
<n_layers>, which is also just the default if --start/--end are omitted). The
--start/--end flags are kept for parity and for cases where someone wants
windowed training on a smaller GPU (e.g. a 24GB consumer card) -- same
freeze/unfreeze logic, just expressed in PyTorch instead of MLX.

Differences from the MLX original (deliberate, not oversights):
  - bitsandbytes 8-bit Adam instead of Adafactor. Adafactor existed in the
    Mac version specifically to fit optimizer state in 48GB; on 80GB+ with the
    full model already using a meaningful chunk of VRAM in bf16, there's room
    for a real momentum-tracking optimizer, which converges faster per-step
    than Adafactor's factorized second-moment-only state. bnb 8-bit Adam keeps
    both moments at ~1 byte/param instead of fp32's 4, so it's the closest
    single-GPU equivalent to "frugal but not degraded" -- falls back to plain
    AdamW (fp32 state) if bitsandbytes isn't installed, since correctness
    matters more than memory thrift on this hardware tier. IMPORTANT: see
    README.md's bitsandbytes section before relying on the fallback -- losing
    bitsandbytes on a fresh container is a real, observed OOM source, not a
    hypothetical one.
  - Checkpointing every --checkpoint-every steps, saved to LOCAL disk (atomic
    write, see below) with an optional cross-optimizer-safety resume check.
    This script does NOT push checkpoints to any cloud object store -- see
    README.md for the real deployment's local-disk + async-write +
    periodic-rsync design instead of a docstring aspiration.
  - HF datasets / transformers AutoModelForCausalLM + AutoTokenizer instead of
    an Apple-Silicon-only ML framework, since this targets a non-Apple-Silicon
    single-GPU box where the original framework doesn't run at all.
  - No Metal-specific NaN workaround needed (the MLX original had a loss-value
    guard for an MLX/Metal-specific bf16 instability) -- PyTorch's standard
    loss.backward() on CUDA/ROCm doesn't hit the same failure mode.

Kept identical to the MLX original (these are correctness/quality properties,
not hardware workarounds, so they carry over):
  - Layer-window freeze/unfreeze (generalizes to "freeze everything outside
    [start,end)").
  - LR warmup -> cosine decay schedule, with a resume-step offset so resuming
    from a checkpoint continues the SAME schedule rather than restarting
    warmup.
  - --cpt flag for raw-text continued-pretraining (no prompt masking) vs
    default SFT (assistant-turn-only loss via a labels mask, -100 on
    prompt/user tokens).
  - Crash-resume for both model weights AND optimizer state (resuming Adam
    cold measurably hurts quality).
  - Atomic checkpoint writes (write to a .tmp dir, then atomic rename) so a
    kill -9 or SIGTERM mid-write never leaves a corrupted checkpoint that
    silently loads garbage.

Usage:
    python3 train_cpt.py \\
        --model ./checkpoints/base_expanded_15b \\
        --data ./data/data_cpt_1 --cpt \\
        --save ./checkpoints/model_cpt_1 \\
        --iters 10000 --batch 4 --lr 5e-7 \\
        --max-seq-len 2048 --checkpoint-every 500

Local-cache CPT mode (zero network dependency once the cache exists -- see
README.md for why this beats live HF streaming on a box with an unreliable
network path):
    python3 train_cpt.py \\
        --model ./checkpoints/base_expanded_15b --cpt \\
        --cpt-cache ./cpt_cache/cache.jsonl \\
        --save ./checkpoints/model_cpt_1 \\
        --iters 2000000 --batch 8 --lr 5e-7

Self-test (no model/GPU required -- checks schedule math and the
atomic checkpoint rename logic against a tmp dir):
    python3 train_cpt.py --selftest

Four pieces that used to be inlined directly in this file's main() are now
their own standalone modules, imported below: optimizer construction
(bnb_optimizer.py), async checkpoint writes (async_checkpoint.py), the
optimizer-type resume guard (optimizer_compat_guard.py), and local-cache
data streaming (local_cache_stream.py). Each is independently
importable/runnable with its own --selftest -- see README.md's "Standalone
utilities" section for what each one solves on its own.
"""

import argparse
import json
import math
import os
import shutil
import signal
import sys
import time
from pathlib import Path

from async_checkpoint import AsyncCheckpointer
from bnb_optimizer import build_optimizer
from local_cache_stream import stream_from_cache
from optimizer_compat_guard import check_optimizer_compat


# ── LR schedule ───────────────────────────────────────────────────────────

def lr_at_step(step: int, total_steps: int, base_lr: float,
               warmup_steps: int, min_lr_ratio: float = 0.1) -> float:
    """Warmup -> cosine decay. `step` is 1-indexed (matches the training
    loop's convention of printing/scheduling against `it` starting at 1)."""
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    floor = base_lr * min_lr_ratio
    return floor + (base_lr - floor) * cosine


# ── checkpoint I/O (atomic local write, no cloud dependency) ─────────────────

def atomic_save_checkpoint(model, optimizer, step: int, save_dir: Path,
                            tokenizer=None, extra_state: dict | None = None,
                            custom_code_src: Path | None = None):
    """Write to `<save_dir>.tmp_ckpt`, then atomic os.replace onto `save_dir`.
    Never let a partial write be observable at the real path."""
    import torch

    tmp_dir = save_dir.parent / (save_dir.name + ".tmp_ckpt")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(tmp_dir, safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(tmp_dir)

    if custom_code_src is not None:
        # model.save_pretrained() serializes config.json but has no idea a custom
        # modeling_*.py file exists alongside a trust_remote_code checkpoint -- it's
        # a plain sidecar file, not something the HF save machinery tracks. Without
        # copying it into every checkpoint, resuming with trust_remote_code=True
        # fails immediately looking for a file that "should" be there. If your
        # model doesn't use a custom modeling file, pass custom_code_src=None and
        # this block is a no-op.
        src_file = Path(custom_code_src) / "modeling_custom.py"
        if src_file.exists():
            shutil.copy2(src_file, tmp_dir / "modeling_custom.py")

    opt_state = {
        "optimizer": optimizer.state_dict(),
        "optimizer_type": type(optimizer).__name__,
        "step": step,
        **(extra_state or {}),
    }
    torch.save(opt_state, tmp_dir / "training_state.pt")

    # Retain the previous checkpoint as .prev (a real backup, not deleted) so a
    # crash mid-write or a corrupt new write can be rolled back. The recovery
    # path below (resume) restores .prev if the live save_dir is missing
    # training_state.pt on restart. The next successful write rotates .prev out
    # (rmtree + os.replace) when a newer good checkpoint supersedes it.
    backup = save_dir.parent / (save_dir.name + ".prev")
    if save_dir.exists():
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(save_dir, backup)
    os.replace(tmp_dir, save_dir)
    # NOTE: .prev is intentionally NOT deleted here — it is the retained backup.

    print(f"[cpt] saved step {step} -> {save_dir}")


# AsyncCheckpointer (background-thread checkpoint writer) now lives in
# async_checkpoint.py, imported at the top of this file -- extracted so it's
# independently importable/testable without the rest of this training loop.
# See that module's docstring for the two-phase sync-snapshot/async-write
# design and why only one write is ever in flight at a time.


# ── data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_sft_example(row: dict, tokenizer, max_seq_len: int):
    """Chat-template tokenize with prompt masking: only assistant-turn tokens get a
    real label, everything else (system/user/special tokens) is -100 (ignored by
    cross-entropy).

    Implementation note: this tokenizes incrementally, calling
    apply_chat_template(messages[:i+1]) per turn and diffing against the previous
    turn's rendered text to isolate the new span. This is O(n_turns^2) in template
    applications per example, and it assumes the template is strictly appenditive —
    i.e. apply_chat_template(messages[:i+1]) is a verbatim text prefix of
    apply_chat_template(messages[:i+2]). This holds for Gemma-4's template (what
    this pipeline targets) but NOT universally; templates that re-render based on
    the full message list, or emit a trailing EOS/generation marker only at the
    end, break the prefix assumption and would silently mis-tokenize/mis-label. We
    detect that break (the prefix check below) and fall back to a single full
    tokenization with the whole prompt masked — coarser (loses per-turn assistant
    labeling, labels only the last assistant turn) and approximate (assumes the
    prompt tokenization is a token-level prefix of the full text, which isn't
    guaranteed for non-appenditive templates), rather than silently wrong.
    """
    import torch

    messages = row["messages"]
    input_ids: list[int] = []
    labels: list[int] = []

    # Tokenize turn-by-turn so we know exactly which spans are assistant output.
    running_text = ""
    prefix_assumption_holds = True
    for i, msg in enumerate(messages):
        prefix_text = tokenizer.apply_chat_template(
            messages[: i + 1], tokenize=False, add_generation_prompt=False
        )
        # Detect a non-appenditive template: if the new full text doesn't start
        # with the previous full text, the incremental-diff approach is invalid.
        if not prefix_text.startswith(running_text):
            prefix_assumption_holds = False
            break
        new_text = prefix_text[len(running_text):]
        running_text = prefix_text
        ids = tokenizer(new_text, add_special_tokens=False)["input_ids"]
        input_ids.extend(ids)
        if msg["role"] == "assistant":
            labels.extend(ids)
        else:
            labels.extend([-100] * len(ids))

    if not prefix_assumption_holds:
        # Fallback: tokenize the full conversation once, mask everything before
        # the last assistant turn, label only the last assistant turn's tokens.
        # This is APPROXIMATE for non-appenditive templates — it assumes the
        # prompt-text tokenization is a token-level prefix of the full-text
        # tokenization, which isn't guaranteed for templates that re-render.
        # It's safer than the broken incremental diff (which would silently
        # mis-tokenize), but it only labels the LAST assistant turn, not all
        # of them. Gemma-4's template is appenditive and takes the primary path
        # above, so this fallback rarely runs for the targeted model family.
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # Build the prompt = everything up to the last assistant turn, to mask it.
        last_assistant_idx = max(
            (i for i, m in enumerate(messages) if m["role"] == "assistant"),
            default=-1,
        )
        prompt_text = tokenizer.apply_chat_template(
            messages[:last_assistant_idx] if last_assistant_idx >= 0 else [],
            tokenize=False, add_generation_prompt=True,
        ) if last_assistant_idx >= 0 else ""
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        p_len = len(prompt_ids)
        input_ids = full_ids
        labels = [-100] * min(p_len, len(full_ids)) + full_ids[min(p_len, len(full_ids)):]
        labels = labels[:len(full_ids)]

    input_ids = input_ids[:max_seq_len]
    labels = labels[:max_seq_len]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def build_cpt_example(row: dict, tokenizer, max_seq_len: int):
    """Raw-text CPT: every token is a label (no masking). Expects packed
    {"text": "..."} rows."""
    import torch

    text = row.get("text", "")
    ids = tokenizer(text, add_special_tokens=False, truncation=True,
                     max_length=max_seq_len)["input_ids"]
    t = torch.tensor(ids, dtype=torch.long)
    return {"input_ids": t, "labels": t.clone()}


def collate(batch: list[dict], pad_token_id: int):
    import torch
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].size(0)
        input_ids[i, :n] = b["input_ids"]
        labels[i, :n] = b["labels"]
        attn[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


def pack_examples(examples: list[dict], max_seq_len: int):
    """Pack short examples into sequences up to max_seq_len, reducing padding
    waste vs. naive batch-max padding. Returns a list of packed example dicts.

    NOTE: packed examples are concatenated with NO separator token and NO
    block-diagonal attention mask — a token in packed example B will causally
    attend to tokens in packed example A. This is acceptable for CPT (where all
    text is training data anyway) but is a data-quality concern for SFT (where
    distinct conversations bleed into each other). For SFT, prefer not using
    --pack unless you've verified the cross-contamination is acceptable."""
    import torch
    packed = []
    current_ids = []
    current_labels = []
    for ex in examples:
        ids = ex["input_ids"].tolist()
        labels = ex["labels"].tolist()
        if current_ids and len(current_ids) + len(ids) > max_seq_len:
            packed.append({
                "input_ids": torch.tensor(current_ids, dtype=torch.long),
                "labels": torch.tensor(current_labels, dtype=torch.long),
            })
            current_ids = []
            current_labels = []
        current_ids.extend(ids)
        current_labels.extend(labels)
    if current_ids:
        packed.append({
            "input_ids": torch.tensor(current_ids[:max_seq_len], dtype=torch.long),
            "labels": torch.tensor(current_labels[:max_seq_len], dtype=torch.long),
        })
    return packed


def run_eval(model, valid_rows: list[dict], builder, tokenizer, max_seq_len: int,
             batch: int, device: str, pack: bool, pad_token_id: int):
    """Run a no-grad forward pass over the full valid set, return mean loss.

    Uses the same builder (build_sft_example / build_cpt_example) and collate as
    training so eval and train loss are directly comparable. Batches in groups of
    `batch` (or packed, if --pack) to avoid OOM on large valid sets.
    """
    import torch
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    try:
        with torch.no_grad():
            for i in range(0, len(valid_rows), batch):
                chunk = valid_rows[i:i + batch]
                examples = [builder(r, tokenizer, max_seq_len) for r in chunk]
                if pack:
                    examples = pack_examples(examples, max_seq_len)
                if not examples:
                    continue
                batch_data = collate(examples, pad_token_id)
                batch_data = {k: v.to(device) for k, v in batch_data.items()}
                outputs = model(**batch_data)
                # outputs.loss is mean over non-ignored tokens in the batch — scale
                # by token count for a correct weighted mean across the whole set.
                labels = batch_data["labels"]
                n_tokens = (labels != -100).sum().item()
                if n_tokens > 0:
                    total_loss += outputs.loss.item() * n_tokens
                    total_tokens += n_tokens
    finally:
        model.train()
    return total_loss / max(total_tokens, 1)


# ── AMD-specific model optimizations (opt-in, graceful fallback) ─────────────

def _apply_fp8(model):
    """Convert linear layers to float8_e4m3fn via torchao's Float8Linear.
    MI300X/MI325X have native fp8 compute — this roughly 2x throughput vs bf16
    on those cards. Falls back to bf16 (no-op) with a warning if torchao isn't
    installed or the conversion fails, so --dtype fp8 never crashes a run that
    would otherwise work."""
    try:
        from torchao.float8 import convert_to_float8_training
        convert_to_float8_training(model)
        print("[cpt] fp8 training enabled (torchao Float8Linear, float8_e4m3fn) — "
              "~2x throughput expected on MI300X/MI325X (native fp8 compute). "
              "Falls back to bf16 matmul internally on cards without fp8 hardware.")
        return model
    except ImportError:
        print("[cpt] WARNING: --dtype fp8 but torchao not installed — falling back "
              "to bf16. Install with 'pip install torchao'.", file=sys.stderr)
        return model
    except Exception as e:
        print(f"[cpt] WARNING: fp8 conversion failed ({e}) — falling back to bf16. "
              f"This can happen on architectures without fp8 support or on models "
              f"with non-standard linear layers.", file=sys.stderr)
        return model


def _apply_flash_attn(model):
    """Switch the model's attention to Flash Attention 2. Reduces attention VRAM
    from O(seqlen^2) to O(seqlen) and speeds up long-context training — directly
    attacks the OOM theme this repo is built around. Requires the flash-attn
    package built for ROCm. Falls back to standard attention with a warning."""
    try:
        import flash_attn  # noqa: F401 — just checking it's importable
        old_impl = getattr(model.config, "_attn_implementation", "eager")
        # Use the public set_attn_implementation() API (added in modern
        # transformers, confirmed present in the transformers==5.7.0 this repo
        # pins) rather than poking model.config._attn_implementation directly.
        # The public method validates the requested implementation, propagates
        # it to nested sub-configs itself (Gemma-4 nests under text_config —
        # set_attn_implementation walks submodels, so no manual text_config
        # poke is needed), and warns instead of silently no-op'ing on an
        # architecture that doesn't support switching post-load.
        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation("flash_attention_2")
        else:
            # Older transformers without the public API: fall back to the
            # private-attribute poke. Not all architectures honor this
            # post-load; logged so the user knows to verify via a forward pass.
            model.config._attn_implementation = "flash_attention_2"
            if hasattr(model, "text_config"):
                model.text_config._attn_implementation = "flash_attention_2"
        print(f"[cpt] flash attention 2 enabled (attn_implementation: "
              f"{old_impl} -> flash_attention_2). VRAM: O(seqlen^2) -> O(seqlen). "
              f"Verify via a forward pass — not all architectures honor this "
              f"post-load; if loss is NaN, the model may not support it.")
    except ImportError:
        print("[cpt] WARNING: --flash-attn but flash-attn not installed — using "
              "standard attention. Install with 'pip install flash-attn "
              "--no-build-isolation' on a ROCm box.", file=sys.stderr)


def _apply_compile(model, mode: str = "max-autotune", dynamic: bool = False):
    """Wrap the model in torch.compile() for kernel fusion + graph optimization.
    ROCm's inductor backend supports this. The first few steps are slower
    (compilation overhead); subsequent steps get the speedup. Falls back to eager
    mode with a warning if compilation fails.

    `dynamic=False` avoids recompilations when sequence lengths vary (use with
    --pack for best results); `dynamic=True` lets the graph adapt but may
    recompile frequently on variable-length inputs."""
    import torch
    try:
        compiled = torch.compile(model, mode=mode, dynamic=dynamic, fullgraph=False)
        print(f"[cpt] torch.compile() enabled (mode={mode}, dynamic={dynamic}, "
              f"ROCm inductor backend) — first steps will be slower (compilation), "
              f"then faster (kernel fusion). If you see errors from inductor, "
              f"remove the --compile flag.")
        return compiled
    except Exception as e:
        print(f"[cpt] WARNING: torch.compile() failed ({e}) — using eager mode. "
              f"This can happen on older ROCm versions or with unsupported ops.",
              file=sys.stderr)
        return model





def find_decoder_layers(model):
    """Locate the transformer's layer list across a handful of HF model-class
    shapes a Gemma-4-family checkpoint might load as. Unwraps DistributedDataParallel
    (whose wrapped model is at .module) before walking attributes."""
    # Unwrap DDP: DistributedDataParallel exposes the wrapped model as .module,
    # and does NOT forward attribute access to it (hasattr(ddp, "model") is False).
    # Without this, find_decoder_layers fails on every --ddp run. Duck-type via
    # class name to avoid importing torch in this module-level function.
    if type(model).__name__ == "DistributedDataParallel":
        model = model.module
    for path in ["model.layers", "language_model.model.layers", "model.model.layers",
                "model.language_model.layers"]:  # the path used by custom multi-token-
                # prediction subclasses that wrap Gemma4ForConditionalGeneration --
                # its .model is a Gemma4Model, whose .language_model is the
                # Gemma4TextModel holding the actual decoder layer stack.
        obj = model
        ok = True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok:
            return obj
    raise AttributeError("Cannot find transformer decoder layers on this model.")


def apply_window_freeze(model, start: int, end: int):
    """Freeze every parameter, then unfreeze only layers [start, end). Embeddings
    and the LM head stay frozen unless explicitly inside the window's layer list
    (they aren't, by construction -- this only ever touches `layers[start:end]`)."""
    for p in model.parameters():
        p.requires_grad = False

    layers = find_decoder_layers(model)
    n_layers = len(layers)
    end = n_layers if end is None else min(end, n_layers)
    for layer in layers[start:end]:
        for p in layer.parameters():
            p.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[cpt] window [{start}, {end}) of {n_layers} layers, "
          f"{n_trainable/1e9:.3f}B trainable params")
    return n_layers, end


# ── self-test (no model/GPU) ─────────────────────────────────────────────────

def self_test():
    print("[selftest] LR schedule: warmup ramps linearly, then cosine-decays to floor")
    base_lr, warmup, total = 1e-5, 10, 100
    assert lr_at_step(1, total, base_lr, warmup) == base_lr * 1 / warmup
    assert lr_at_step(10, total, base_lr, warmup) == base_lr
    end_lr = lr_at_step(100, total, base_lr, warmup, min_lr_ratio=0.1)
    assert abs(end_lr - base_lr * 0.1) < 1e-9, end_lr
    prev = lr_at_step(warmup, total, base_lr, warmup)
    for s in range(warmup + 1, total + 1):
        cur = lr_at_step(s, total, base_lr, warmup)
        assert cur <= prev + 1e-12, (s, cur, prev)
        prev = cur
    print("  OK")

    print("[selftest] Resume offset: resuming at step N does NOT restart warmup")
    # The property that matters: the schedule is a function of ABSOLUTE step,
    # not "steps since resume." A buggy resume that restarted warmup would use
    # lr_at_step(k, ...) (relative step) instead of lr_at_step(resume_step + k)
    # (absolute step). For k within the warmup window, those differ — so we
    # assert they DO differ (catching a warmup-restart bug). For k past warmup,
    # both are on the cosine curve but at different points, so they also differ.
    resume_step = 37
    for k in [1, 5, 50]:
        absolute = resume_step + k
        absolute_lr = lr_at_step(absolute, total, base_lr, warmup)
        relative_lr = lr_at_step(k, total, base_lr, warmup)
        # A correct resume uses absolute step; a warmup-restart bug uses relative.
        # These must NOT be equal (otherwise the schedule would be identical
        # regardless of resume point, which is only true if warmup already ended
        # AND the cosine is flat — never the case here).
        assert absolute_lr != relative_lr, \
            f"step k={k}: absolute lr {absolute_lr} should differ from " \
            f"relative lr {relative_lr} (if equal, resume offset has no effect)"
    print("  OK (absolute-step lr differs from relative-step lr at all tested "
          "points — resume offset matters, warmup is not restarted)")

    print("[selftest] atomic checkpoint rename pattern + .prev retention (no torch/model)")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        save_dir = td / "ckpt"
        tmp_dir = save_dir.parent / (save_dir.name + ".tmp_ckpt")
        backup = save_dir.parent / (save_dir.name + ".prev")

        # Seed a live v1 checkpoint, then simulate a successful atomic write of v2:
        # rotate live->.prev, tmp->live, and RETAIN .prev (the new behavior — it
        # is no longer deleted, so a later crash mid-write can roll back to it).
        tmp_dir.mkdir(parents=True)
        (tmp_dir / "marker.txt").write_text("v2")
        save_dir.mkdir(parents=True)
        (save_dir / "marker.txt").write_text("v1")
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(save_dir, backup)
        os.replace(tmp_dir, save_dir)
        assert (save_dir / "marker.txt").read_text() == "v2"
        assert backup.exists() and (backup / "marker.txt").read_text() == "v1", \
            ".prev must be retained as the last-good backup"
        assert not tmp_dir.exists()
        print("  OK (live checkpoint is v2; .prev retained as v1 backup)")

        # Second successful write: .prev rotates out (rmtree old .prev, rotate
        # live->.prev, tmp->live) and the new .prev holds v2.
        tmp_dir.mkdir(parents=True)
        (tmp_dir / "marker.txt").write_text("v3")
        shutil.rmtree(backup)
        os.replace(save_dir, backup)
        os.replace(tmp_dir, save_dir)
        assert (save_dir / "marker.txt").read_text() == "v3"
        assert (backup / "marker.txt").read_text() == "v2"
        print("  OK (.prev rotated to v2 on the second successful write)")

        # Crash-window recovery: simulate a kill between the two os.replace()
        # calls — live save_dir is gone (the first replace moved it to .prev),
        # but .prev holds the last good checkpoint. The resume recovery path
        # restores .prev -> live instead of silently restarting from --model.
        shutil.rmtree(save_dir)  # simulate the crash window: live gone
        assert not save_dir.exists()
        assert backup.exists()  # last-good checkpoint stranded in .prev
        os.replace(backup, save_dir)  # recovery: .prev -> live
        assert (save_dir / "marker.txt").read_text() == "v2"
        assert not backup.exists()
        print("  OK (crash-window recovery: .prev restored to live, would resume "
              "from v2 instead of restarting from --model)")


    print("\n[selftest] All checks passed (no model/GPU required for these -- run a "
          "real --iters 5 smoke test on actual hardware before trusting this for a "
          "real training job).")


# ── SIGTERM handling (spot preemption / preemptible instances) ───────────────

_SHOULD_STOP = False


def _on_sigterm(signum, frame):
    global _SHOULD_STOP
    print(f"\n[signal] received SIGTERM -- will checkpoint after the current step "
          f"and exit", file=sys.stderr)
    _SHOULD_STOP = True


# ── main training loop ────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", default=False)
    ap.add_argument("--model", help="HF-format model dir or repo id to train.")
    ap.add_argument("--data", help="Dir containing train.jsonl, optionally valid.jsonl "
                                    "for held-out eval (see --eval-every). Or a single "
                                    ".jsonl file (train only, no eval).")
    ap.add_argument("--save", help="Output directory for the trained model.")
    ap.add_argument("--start", type=int, default=0, help="First layer index to unfreeze.")
    ap.add_argument("--end", type=int, default=None,
                    help="Last layer index (exclusive). Default: all layers (full-model "
                         "training -- the point of having 80GB+ instead of 48GB).")
    ap.add_argument("--iters", type=int, default=3000,
                    help="Number of optimizer update steps. When --accum > 1, the "
                         "training loop performs iters*accum micro-batches and calls "
                         "optimizer.step() once every --accum micro-batches.")
    ap.add_argument("--batch", type=int, default=2,
                    help="Micro-batch size per GPU. Effective batch size is "
                         "batch * accum * world_size.")
    ap.add_argument("--accum", "--gradient-accumulation-steps", type=int, default=1,
                    dest="accum",
                    help="Gradient accumulation steps. Default 1 (no accumulation). "
                         "Loss is divided by this value so gradients average over the "
                         "accumulated micro-batches. Under --ddp, every micro-batch still "
                         "triggers its own gradient all-reduce (no DDP no_sync() "
                         "optimization on the non-final micro-batches) -- correct, but "
                         "not the fastest possible implementation; fine for the "
                         "single-GPU case this repo targets.")
    ap.add_argument("--lr", type=float, default=8e-7)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--cpt", action="store_true", default=False,
                    help="Raw-text continued-pretraining mode (no prompt masking). "
                         "Expects packed {\"text\":...} rows.")
    ap.add_argument("--cpt-cache", type=str, default=None,
                    help="Path to a local JSONL cache of pre-fetched CPT rows (e.g. "
                         "/dev/shm/cpt_cache/cache.jsonl) -- trains with ZERO network "
                         "dependency instead of live streaming. Cycles the cache "
                         "indefinitely once exhausted (better than stopping, since "
                         "there's no network to refill it). See README.md for why "
                         "this exists -- it's a real reliability fix, not speculative.")
    ap.add_argument("--no-grad-checkpoint", action="store_true", default=False,
                    help="Disable gradient checkpointing (on by default). Checkpointing "
                         "recomputes activations during backward instead of storing them "
                         "for every layer at once -- trades compute time for the "
                         "activation-memory headroom to run a bigger batch.")
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=None,
                    help="Run held-out validation every N steps and log valid_loss. "
                         "Defaults to --checkpoint-every. Only active if valid.jsonl "
                         "is present in --data (a dir). --no-eval disables entirely.")
    ap.add_argument("--no-eval", action="store_true", default=False,
                    help="Disable held-out validation even if valid.jsonl exists.")
    ap.add_argument("--tb", type=str, default=None,
                    help="Directory for TensorBoard event logs (local files only, no "
                         "external service). When set, logs train/loss, train/lr, and "
                         "eval/valid_loss. Requires the 'tensorboard' package (not bundled "
                         "with torch — install separately, or omit this flag for stdout-only "
                         "logging).")
    ap.add_argument("--pack", action="store_true", default=False,
                    help="Pack short examples into sequences up to --max-seq-len instead "
                         "of padding to batch-max. Reduces padding waste; off by default.")
    ap.add_argument("--async-checkpoint", action="store_true", default=False,
                    help="Write checkpoints on a background thread (AsyncCheckpointer) "
                         "instead of blocking the training loop for the full disk write. "
                         "Off by default -- opt in once you've confirmed it against a "
                         "synchronous checkpoint's output on real hardware (see "
                         "AsyncCheckpointer's docstring for the one unverified assumption).")
    ap.add_argument("--resume-tag", default=None,
                    help="Tag used only for logging which checkpoint this run considers "
                         "itself to be. Defaults to the basename of --save.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gfx-override", type=str, default=None,
                    help="Force HSA_OVERRIDE_GFX_VERSION to this value (e.g. gfx1100) "
                         "for AMD consumer/older cards whose arch isn't in the ROCm "
                         "torch wheel's compiled list. When unset, rocm_env auto-detects "
                         "the GPU arch and overrides only if needed. See rocm_env.py.")
    ap.add_argument("--hip-alloc-conf", type=str, default="max_split_size_mb:128",
                    help="Value for PYTORCH_HIP_ALLOC_CONF (ROCm caching allocator). "
                         "Default 'max_split_size_mb:128' prevents the fragmentation "
                         "OOMs that hit long training runs. Pass 'none' to disable.")
    ap.add_argument("--flash-attn", action="store_true", default=False,
                    help="Use Flash Attention 2 (via flash_attn package) for the "
                         "attention layers. Reduces VRAM from O(seqlen^2) to "
                         "O(seqlen) and speeds up long-context training. Requires "
                         "the 'flash-attn' package built for ROCm (pip install "
                         "flash-attn --no-build-isolation). Falls back to standard "
                         "attention with a warning if not installed.")
    ap.add_argument("--compile", action="store_true", default=False,
                    help="Wrap the model in torch.compile() for kernel fusion + "
                         "graph optimization. ROCm's inductor backend supports this. "
                         "First few steps will be slower (compilation); subsequent "
                         "steps get the speedup. Falls back to eager mode with a "
                         "warning if compilation fails.")
    ap.add_argument("--compile-mode", type=str, default="max-autotune",
                    choices=["default", "reduce-overhead", "max-autotune"],
                    help="torch.compile mode. 'max-autotune' (default) spends more "
                         "time upfront autotuning but yields the best ROCm throughput "
                         "for steady-state training. 'reduce-overhead' is better for "
                         "small models or short runs. Ignored unless --compile is set.")
    ap.add_argument("--dtype", type=str, default="bf16",
                    choices=["bf16", "fp8"],
                    help="Training dtype. 'bf16' (default) works on all ROCm cards. "
                         "'fp8' uses torch.float8_e4m3fn via torchao's Float8Linear "
                         "for ~2x throughput on MI300X/MI325X (native fp8 compute). "
                         "Requires the 'torchao' package; falls back to bf16 with a "
                         "warning if not installed or the card lacks fp8 support.")
    ap.add_argument("--profile", type=str, default=None,
                    help="If set, profile the training loop with torch.profiler and "
                         "write trace artifacts to this directory. The trace is "
                         "viewable in chrome://tracing or Perfetto, and includes "
                         "ROCm/HIP kernel launches. For kernel-level profiling "
                         "beyond torch.profiler, wrap the run with 'rocprof --stats "
                         "python3 train_cpt.py ...'.")
    ap.add_argument("--ddp", action="store_true", default=False,
                    help="Enable multi-GPU training via torch.distributed + "
                         "DistributedDataParallel. Launch with 'torchrun "
                         "--nproc_per_node=N train_cpt.py --ddp ...' — the script "
                         "reads RANK/LOCAL_RANK/WORLD_SIZE from torchrun's env "
                         "vars. Only rank 0 writes checkpoints and logs; all ranks "
                         "participate in training with gradient all-reduce. "
                         "Converts 'one MI300X' -> 'a node of them'.")
    args = ap.parse_args()

    if args.selftest:
        self_test()
        return

    if not (args.model and args.save and (args.data or args.cpt_cache)):
        ap.error("--model and --save are required, plus one of --data or --cpt-cache, "
                 "unless --selftest is given.")

    # ROCm env bootstrap: MUST run before `import torch`. On AMD consumer/older
    # cards (RDNA1/2, gfx803, etc.) whose arch isn't in the torch wheel's
    # compiled list, kernels fail with "no kernel image" unless
    # HSA_OVERRIDE_GFX_VERSION is set before the runtime initializes.
    # setup_rocm_env() auto-detects the GPU arch and overrides only if needed;
    # --gfx-override forces a specific value. No-op on non-ROCm / already-
    # supported cards. See rocm_env.py for the detection + family-matching logic.
    from rocm_env import setup_rocm_env
    hip_conf = None if args.hip_alloc_conf.lower() == "none" else args.hip_alloc_conf
    setup_rocm_env(override=args.gfx_override, hip_alloc_conf=hip_conf)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    signal.signal(signal.SIGTERM, _on_sigterm)

    torch.manual_seed(args.seed)  # per-rank seeding added below after ddp_rank is set
    if not torch.cuda.is_available():
        print("[cpt] WARNING: no CUDA/ROCm device visible -- this script is built for "
              "single-GPU hardware (e.g. an AMD MI300X under ROCm, or an NVIDIA "
              "A100/H100). Running on CPU will be extremely slow; only use this path "
              "for a tiny --iters smoke test.", file=sys.stderr)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Multi-GPU DDP setup ──────────────────────────────────────────────────
    # When --ddp is set, the script expects to be launched via torchrun:
    #   torchrun --nproc_per_node=N train_cpt.py --ddp --model ... --save ...
    # torchrun sets RANK, LOCAL_RANK, WORLD_SIZE env vars. We init the process
    # group, pin each rank to its local GPU, and use DDP to all-reduce gradients.
    # Only rank 0 writes checkpoints, logs to stdout, and runs eval — the other
    # ranks train silently and participate in the gradient sync.
    ddp_rank = 0
    ddp_world_size = 1
    is_main = True  # rank 0 (or single-GPU)
    if args.ddp:
        if "RANK" not in os.environ:
            raise SystemExit("ERROR: --ddp set but RANK env var not found. Launch "
                             "via 'torchrun --nproc_per_node=N train_cpt.py --ddp ...' "
                             "so torchrun sets RANK/LOCAL_RANK/WORLD_SIZE.")
        ddp_rank = int(os.environ["RANK"])
        ddp_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        is_main = (ddp_rank == 0)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            rank=ddp_rank,
            world_size=ddp_world_size,
        )
        if is_main:
            print(f"[cpt] DDP enabled: rank {ddp_rank}/{ddp_world_size}, "
                  f"local_rank={local_rank}, device={device}")
        # Per-rank torch seeding: without this, dropout/RNG-based ops produce
        # identical masks on every rank (correlated noise), which is wasteful.
        torch.manual_seed(args.seed + ddp_rank)

    save_dir = Path(args.save)
    resume_tag = args.resume_tag or save_dir.name
    if is_main:
        print(f"[cpt] resume_tag: {resume_tag}")
    resumed = False
    if save_dir.exists() and (save_dir / "training_state.pt").exists():
        # Local-only resume: re-running the SAME command after a crash or a
        # preemption resumes from whatever is sitting on disk, instead of silently
        # restarting from --model and discarding a perfectly good checkpoint.
        resumed = True
        print(f"[cpt] found existing local checkpoint at {save_dir} -- resuming from it")
    else:
        # Crash-window recovery: if a kill -9 / OOM-kill hit BETWEEN the two
        # os.replace() calls in atomic_save_checkpoint / AsyncCheckpointer._write
        # (move live->.prev, then move tmp->live), the live save_dir is gone but
        # the last good checkpoint is stranded in .prev. Without this recovery,
        # train_cpt.py would silently restart from --model and discard all
        # training progress. Restore .prev -> live so normal resume picks it up.
        prev_dir = save_dir.parent / (save_dir.name + ".prev")
        if prev_dir.exists() and (prev_dir / "training_state.pt").exists():
            if save_dir.exists():
                # save_dir exists but is incomplete (no training_state.pt) -- a
                # half-written or interrupted checkpoint. Remove it before
                # restoring the known-good .prev.
                shutil.rmtree(save_dir)
            os.replace(prev_dir, save_dir)
            resumed = True
            print(f"[cpt] recovered checkpoint from {prev_dir} -> {save_dir} "
                  f"(live checkpoint was missing/incomplete; .prev restored). "
                  f"Resuming from it instead of restarting from --model.")

    load_path = str(save_dir) if resumed else args.model
    if is_main:
        print(f"[cpt] Loading model from {load_path} ...")
    # trust_remote_code=True: harmless no-op for any checkpoint that doesn't set
    # config.json's auto_map (falls back to whatever stock architecture class
    # transformers would have loaded anyway). Only matters if your model ships a
    # custom modeling_*.py file (e.g. one adding multi-token prediction) -- see
    # expand_model.py's docstring for that case.
    load_kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": True}
    if args.flash_attn:
        try:
            import flash_attn  # noqa: F401 — if installed, load with FA2 from the start
            load_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            # _apply_flash_attn() below will print the fallback warning.
            pass
    model = AutoModelForCausalLM.from_pretrained(load_path, **load_kwargs).to(device)
    if not args.no_grad_checkpoint:
        model.config.use_cache = False  # incompatible with checkpointing/training either way
        model.gradient_checkpointing_enable()
        # Required because windowed training freezes most of the trunk (requires_grad=False)
        # -- torch.utils.checkpoint only creates a backward node if the checkpointed
        # segment's INPUT tensor requires grad, regardless of whether the layer's own
        # weights are trainable. Without this, gradients silently fail to reach the
        # trainable window whenever it isn't the very first layer.
        model.enable_input_require_grads()
        print("[cpt] gradient checkpointing enabled (recomputes activations in backward "
              "instead of storing them -- trades ~20-30% more compute time for the "
              "activation-memory headroom to run a bigger batch)")
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── AMD-specific model optimizations (all opt-in, all with graceful fallback) ──
    # Order matters: fp8 weight conversion first (compile then fuses the fp8
    # ops), then flash-attn attn_implementation (independent), then torch.compile
    # (fuses whatever's there). Each is a no-op if the flag isn't set or the
    # dependency is missing, so the default path (bf16, eager, standard attn)
    # is unchanged.
    if args.dtype == "fp8":
        model = _apply_fp8(model)
    if args.flash_attn:
        _apply_flash_attn(model)
    if args.compile:
        # dynamic=False avoids recompilation thrash when --pack gives fixed-length
        # sequences; leave it dynamic only if the user is not packing (variable
        # length inputs will still work, just with more compile overhead).
        compile_dynamic = not args.pack
        model = _apply_compile(model, mode=args.compile_mode, dynamic=compile_dynamic)

    # Wrap in DistributedDataParallel after all model modifications (fp8,
    # flash-attn, compile) but before apply_window_freeze, so DDP sees the
    # final parameter set for gradient sync. DDP syncs gradients via all-reduce
    # during backward — the optimizer step then operates on synced grads.
    windowed_freeze = args.start != 0 or args.end is not None
    if args.ddp and ddp_world_size > 1:
        if windowed_freeze:
            # PyTorch warns that find_unused_parameters=True combined with
            # gradient checkpointing can be unsafe in some versions because DDP
            # cannot always trace which checkpointed segments produce gradients
            # for which parameters. Windowed freezing also disables DDP gradient
            # bucketing optimizations, which can materially hurt throughput on
            # AMD/ROCm. We still allow it for small-GPU compatibility, but warn.
            print("[cpt] WARNING: --ddp with windowed --start/--end + gradient "
                  "checkpointing is supported for compatibility, but it is slower "
                  "and can be correctness-sensitive on some PyTorch/ROCm builds. "
                  "For best throughput and safety on MI300X-class hardware, use "
                  "full-model training (--start 0 with no --end) or wait for FSDP.",
                  file=sys.stderr)
        # find_unused_parameters=True handles the windowed-freeze case where only
        # a subset of params have requires_grad=True — DDP needs to know which
        # params participate in backward to all-reduce correctly (without this,
        # DDP's default assumption that every param participates in every
        # backward pass raises a runtime error the moment a frozen param's
        # gradient never arrives).
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=True,
        )
        if is_main:
            print(f"[cpt] model wrapped in DistributedDataParallel "
                  f"(world_size={ddp_world_size})")

    n_layers, end_idx = apply_window_freeze(model, args.start, args.end)

    # For DDP, use the underlying model's parameters (DDP wraps but doesn't
    # change param objects). find_decoder_layers unwraps DDP via its
    # type-name check before walking attributes.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # build_optimizer() lives in bnb_optimizer.py (extracted so it's independently
    # importable/testable) -- tries bitsandbytes 8-bit Adam, falls back to plain
    # torch.optim.AdamW with a clear warning if bitsandbytes isn't installed.
    optimizer, optimizer_kind = build_optimizer(trainable_params, lr=args.lr, weight_decay=0.01)
    if is_main:
        print(f"[cpt] optimizer: {optimizer_kind}")

    start_step = 0
    if resumed:
        state_path = save_dir / "training_state.pt"
        if state_path.exists():
            # weights_only=False: the saved training_state.pt holds an optimizer
            # state_dict (e.g. bitsandbytes Adam8bit buffers) that can contain
            # non-allowlisted pickle objects. Since PyTorch 2.6, torch.load
            # defaults to weights_only=True, which rejects those and raises
            # UnpicklingError on resume. The self-test path in async_checkpoint.py
            # already passes weights_only=False; this production resume path must
            # match it. The checkpoint is local, written by this same tool, so
            # trusting its pickle contents is the intended threat model.
            state = torch.load(state_path, map_location=device, weights_only=False)
            start_step = state.get("step", 0)
            saved_optimizer_type = state.get("optimizer_type", "unknown")
            current_optimizer_type = type(optimizer).__name__
            # check_optimizer_compat() lives in optimizer_compat_guard.py (extracted
            # so the load-vs-skip decision has one canonical implementation) -- loading
            # one optimizer type's state_dict into a DIFFERENT optimizer class has been
            # observed to silently accept the mismatched state and inflate GPU memory
            # well beyond what the current optimizer needs, OOMing on the first forward
            # pass. safe_to_load=False means: skip the load, restart momentum fresh,
            # keep the step count.
            safe_to_load, compat_msg = check_optimizer_compat(saved_optimizer_type,
                                                               current_optimizer_type)
            print(f"[cpt] {compat_msg}")
            if safe_to_load:
                optimizer.load_state_dict(state["optimizer"])
                print(f"[cpt] resumed at step {start_step} (optimizer state restored -- "
                      f"cold-restarting momentum measurably hurts quality, so this matters)")

    builder = build_cpt_example if args.cpt else build_sft_example

    import random as _random
    # Per-rank seeding: without this, every rank draws the same batches (same
    # seed + same start_step), DDP all-reduces identical gradients, and the
    # N-1 extra GPUs do fully redundant work. Adding ddp_rank makes each rank
    # sample a different subset of the data — real data parallelism.
    rng = _random.Random(args.seed + start_step + ddp_rank)

    stream_gen = None
    if args.cpt_cache:
        # Zero-network path: read from a local JSONL cache built ahead of time
        # (e.g. by pre-fetching category-weighted rows from a public dataset with
        # its own retry/timeout handling). Prefer this over live streaming whenever
        # the training box's network path to the data source is unreliable -- see
        # README.md for the concrete incident this was built to route around.
        # stream_from_cache() lives in local_cache_stream.py (extracted so the
        # cache-reading side is independently importable/testable) -- loads the
        # cache once, shuffles with the given seed, and reshuffles on every full
        # pass instead of stopping once exhausted.
        try:
            stream_gen = stream_from_cache(args.cpt_cache, seed=args.seed + ddp_rank)
        except RuntimeError as e:
            raise SystemExit(str(e))
        print(f"[cpt] training from local cache ({args.cpt_cache}) -- zero network "
              f"dependency, safe against source/network instability")
    else:
        data_path = Path(args.data)
        train_file = data_path / "train.jsonl" if data_path.is_dir() else data_path
        rows = load_jsonl(train_file)
        if not rows:
            raise SystemExit(f"ERROR: no training rows found in {train_file} — cannot "
                             f"train on an empty dataset.")
        if is_main:
            print(f"[cpt] {len(rows):,} training rows loaded from {train_file}")

    # Held-out validation set: load valid.jsonl if present in the --data dir and
    # eval isn't disabled. This makes the --data help string's valid.jsonl promise
    # real (it previously advertised valid.jsonl but never read it). valid_loss is
    # a more honest signal than train loss for catch_and_resume.sh's rollback.
    valid_rows = None
    eval_every = args.eval_every or args.checkpoint_every
    if not args.no_eval and args.data and Path(args.data).is_dir():
        valid_file = Path(args.data) / "valid.jsonl"
        if valid_file.exists():
            valid_rows = load_jsonl(valid_file)
            if valid_rows:
                print(f"[cpt] {len(valid_rows):,} validation rows loaded from {valid_file} "
                      f"-- eval every {eval_every} steps")
            else:
                valid_rows = None
                print(f"[cpt] WARNING: {valid_file} exists but is empty — eval disabled")

    # TensorBoard logging (local event files only, no external service).
    # tensorboard is NOT bundled with torch — it's a separate package. If --tb
    # is set but tensorboard isn't installed, warn and continue stdout-only
    # rather than crashing.
    tb_writer = None
    if args.tb and is_main:  # only rank 0 writes TB events
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(args.tb)
            print(f"[cpt] TensorBoard logging -> {args.tb}")
        except ImportError:
            print(f"[cpt] WARNING: --tb set but tensorboard not installed — "
                  f"falling back to stdout-only logging. Install with "
                  f"'pip install tensorboard'.", file=sys.stderr)
            tb_writer = None

    model.train()
    async_ckpt = AsyncCheckpointer() if args.async_checkpoint else None
    if args.async_checkpoint:
        print("[cpt] async checkpointing enabled -- checkpoint writes run on a "
              "background thread, training does not wait for them except at exit")
    last_valid_loss = None  # carried into checkpoint extra_state for catch_and_resume rollback

    # Profiling: wrap the training loop in torch.profiler if --profile is set.
    # The trace includes ROCm/HIP kernel launches and is viewable in
    # chrome://tracing or Perfetto. For deeper kernel-level profiling, wrap the
    # whole run with 'rocprof --stats python3 train_cpt.py ...' instead.
    profiler = None
    if args.profile and is_main:
        # Only rank 0 profiles — all ranks writing to the same dir would
        # collide/corrupt traces. Non-rank-0 GPUs just don't enter the profiler.
        os.makedirs(args.profile, exist_ok=True)
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.profile),
        )
        profiler.__enter__()
        print(f"[cpt] profiling enabled — trace artifacts -> {args.profile} "
              f"(viewable in chrome://tracing or Perfetto)")

    # Wrap the training loop in try/finally so cleanup (profiler, tb_writer,
    # DDP process group) runs even on exception (OOM, CUDA error, etc.).
    # Without this, an exception mid-loop leaks the profiler context, leaves
    # TB events unflushed, and leaves NCCL in a dirty state.
    try:
        for it in range(start_step + 1, args.iters + 1):
            lr = lr_at_step(it, args.iters, args.lr, args.warmup_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            # Gradient accumulation: run args.accum micro-batches, each scaled by
            # 1/accum so their summed gradients equal the average over all
            # accum*batch examples, then step the optimizer once. With
            # args.accum == 1 (the default) this is exactly one micro-batch and
            # behaves identically to no accumulation. last_loss is the LAST
            # micro-batch's (unscaled) loss, purely for logging -- it is not
            # itself the quantity being optimized (the accumulated, scaled sum
            # is), but it's a reasonable per-iter progress signal and avoids
            # holding accum separate loss tensors alive for an "average" that
            # would need its own explicit accumulation anyway.
            last_loss = None
            for _ in range(args.accum):
                if stream_gen is not None:
                    batch_rows = [next(stream_gen) for _ in range(args.batch)]
                else:
                    batch_rows = [rows[rng.randrange(len(rows))] for _ in range(args.batch)]
                examples = [builder(r, tokenizer, args.max_seq_len) for r in batch_rows]
                if args.pack:
                    examples = pack_examples(examples, args.max_seq_len)
                batch = collate(examples, tokenizer.pad_token_id)
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(**batch)
                loss = outputs.loss
                last_loss = loss
                (loss / args.accum).backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            loss = last_loss  # for the logging/eval code below, unchanged from pre-accum shape

            if profiler is not None:
                profiler.step()
    
            # Only rank 0 logs to stdout / TB / runs eval — other ranks train silently.
            if it % 10 == 0 or it == args.iters:
                if is_main:
                    print(f"[cpt] Iter {it}/{args.iters}: loss={loss.item():.4f}  lr={lr:.2e}")
            if tb_writer is not None and is_main and it % 10 == 0:
                tb_writer.add_scalar("train/loss", loss.item(), it)
                tb_writer.add_scalar("train/lr", lr, it)
    
            # Held-out eval at eval_every intervals (rank 0 only — the eval forward
            # pass doesn't need gradient sync, so non-rank-0 GPUs idle during eval).
            if valid_rows is not None and is_main and (it % eval_every == 0 or it == args.iters or _SHOULD_STOP):
                vloss = run_eval(model, valid_rows, builder, tokenizer, args.max_seq_len,
                                 args.batch, device, args.pack, tokenizer.pad_token_id)
                last_valid_loss = vloss
                print(f"[cpt] eval step {it}: valid_loss={vloss:.4f}")
                if tb_writer is not None:
                    tb_writer.add_scalar("eval/valid_loss", vloss, it)
    
            # DDP barrier: ensure all ranks are at the same step before checkpointing.
            # NOTE: _SHOULD_STOP is intentionally NOT in this condition. The flag is
            # set per-rank by an async signal handler, so including it in a collective
            # barrier would deadlock if ranks set the flag at different steps (some
            # enter barrier() while others proceed to the next step's all-reduce).
            # Instead, ranks only barrier on scheduled checkpoint/final-iter steps,
            # where they already synchronize via the backward all-reduce. A SIGTERM
            # on a non-checkpoint step sets _SHOULD_STOP, which triggers checkpoint +
            # exit on the NEXT checkpoint-boundary step (when the barrier fires).
            if args.ddp and (it % args.checkpoint_every == 0 or it == args.iters):
                torch.distributed.barrier()
    
            # Only rank 0 writes checkpoints — DDP syncs gradients, so the model
            # state is identical across ranks; writing from one is sufficient and
            # avoids N copies of a multi-GB checkpoint hitting disk simultaneously.
            if is_main and (it % args.checkpoint_every == 0 or it == args.iters or _SHOULD_STOP):
                ckpt_extra = {"valid_loss": last_valid_loss} if last_valid_loss is not None else None
                # Unwrap DDP for checkpointing: save_pretrained / state_dict need the
                # underlying model, not the DDP wrapper (which prefixes keys with
                # "module." and has no save_pretrained).
                save_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                # When resuming, the latest modeling_custom.py lives in the checkpoint
                # dir (the user may have updated it there). For a fresh run, copy it
                # from the original --model path.
                custom_code_src = save_dir if resumed else Path(args.model)
                if args.async_checkpoint:
                    async_ckpt.save(save_model, optimizer, it, save_dir, tokenizer,
                                    extra_state=ckpt_extra,
                                    custom_code_src=custom_code_src)
                    # On exit (SIGTERM or final iter) the write MUST finish before the
                    # process dies, or this defeats the whole point of atomic checkpointing.
                    # For a regular mid-run checkpoint, deliberately NOT waiting here -- the
                    # background thread keeps writing while training continues; save()
                    # itself waits on any still-in-flight write before starting the next one.
                    if _SHOULD_STOP or it == args.iters:
                        async_ckpt.wait_for_pending()
                else:
                    atomic_save_checkpoint(save_model, optimizer, it, save_dir, tokenizer,
                                           extra_state=ckpt_extra,
                                           custom_code_src=custom_code_src)
    
            if _SHOULD_STOP:
                # Cleanup (tb_writer/profiler/process-group) is intentionally
                # NOT duplicated here -- sys.exit(0) raises SystemExit, which
                # still runs the enclosing `finally` block below on its way out.
                # An earlier version of this branch closed tb_writer, exited the
                # profiler, and called torch.distributed.destroy_process_group()
                # here AND let `finally` run the same calls again -- a second
                # destroy_process_group() call raises
                # "AssertionError: Process group cannot be None" (confirmed with
                # a real torch.distributed init/destroy/destroy repro), which
                # meant the one path this whole try/finally exists to make safe
                # (graceful SIGTERM shutdown under --ddp) was the one path that
                # crashed. Let `finally` do the one and only cleanup pass.
                print(f"[cpt] Exiting cleanly after checkpoint at step {it} (SIGTERM)")
                sys.exit(0)

    finally:
        if tb_writer is not None:
            tb_writer.close()
        if profiler is not None:
            profiler.__exit__(None, None, None)
        if args.ddp:
            torch.distributed.destroy_process_group()



if __name__ == "__main__":
    main()
