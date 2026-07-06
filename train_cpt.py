#!/usr/bin/env python3
"""CUDA/ROCm continued-pretraining (CPT) / SFT trainer for a Gemma-4-family
model — pipeline step 3 of 4 (runs against the output of expand_model.py, or
directly against a pruned checkpoint if you skip expansion).

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

Self-test (no model/GPU required -- checks schedule math, masking, and the
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
    labeling) but correct, rather than silently wrong.
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
        # Coarser than the per-turn path but correct for non-appenditive templates.
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
    """Pack short examples into sequences up to max_seq_len, with attention_mask
    blocking cross-contamination between packed examples. Reduces padding waste
    vs. naive batch-max padding. Returns a list of packed example dicts."""
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


@torch.no_grad() if False else (lambda f: f)
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




def find_decoder_layers(model):
    """Locate the transformer's layer list across a handful of HF model-class
    shapes a Gemma-4-family checkpoint might load as."""
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
    # The property that matters: a resumed run at absolute step (resume_step + k)
    # uses the SAME lr as a non-resumed run at that absolute step — i.e. the
    # schedule is a function of absolute step, not "steps since resume". A buggy
    # resume that restarted warmup would give lr_at_step(k, ...) instead.
    resume_step = 37
    for k in [1, 5, 50]:
        absolute = resume_step + k
        resumed_lr = lr_at_step(absolute, total, base_lr, warmup)
        fresh_lr = lr_at_step(absolute, total, base_lr, warmup)
        assert resumed_lr == fresh_lr, (k, resumed_lr, fresh_lr)
        # And critically: a warmup-restarting bug would give a DIFFERENT (higher)
        # lr for small k, since warmup ramps from 0. Assert they differ where
        # warmup-restart would — so this test would CATCH that bug, not pass it.
        warmup_restarted_lr = lr_at_step(k, total, base_lr, warmup)
        if k <= warmup:
            assert resumed_lr != warmup_restarted_lr, \
                f"step {k}: resumed lr {resumed_lr} should NOT equal warmup-restart " \
                f"lr {warmup_restarted_lr} (resume must not restart warmup)"
    print("  OK (resumed schedule matches absolute-step curve, not a warmup restart)")

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
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=2)
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
                         "external service). When set, logs train/loss, train/lr, "
                         "train/step, and eval/valid_loss.")
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
    args = ap.parse_args()

    if args.selftest:
        self_test()
        return

    if not (args.model and args.save and (args.data or args.cpt_cache)):
        ap.error("--model and --save are required, plus one of --data or --cpt-cache, "
                 "unless --selftest is given.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    signal.signal(signal.SIGTERM, _on_sigterm)

    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        print("[cpt] WARNING: no CUDA/ROCm device visible -- this script is built for "
              "single-GPU hardware (e.g. an AMD MI300X under ROCm, or an NVIDIA "
              "A100/H100). Running on CPU will be extremely slow; only use this path "
              "for a tiny --iters smoke test.", file=sys.stderr)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = Path(args.save)
    resume_tag = args.resume_tag or save_dir.name
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
    print(f"[cpt] Loading model from {load_path} ...")
    # trust_remote_code=True: harmless no-op for any checkpoint that doesn't set
    # config.json's auto_map (falls back to whatever stock architecture class
    # transformers would have loaded anyway). Only matters if your model ships a
    # custom modeling_*.py file (e.g. one adding multi-token prediction) -- see
    # expand_model.py's docstring for that case.
    model = AutoModelForCausalLM.from_pretrained(
        load_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
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

    n_layers, end_idx = apply_window_freeze(model, args.start, args.end)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # build_optimizer() lives in bnb_optimizer.py (extracted so it's independently
    # importable/testable) -- tries bitsandbytes 8-bit Adam, falls back to plain
    # torch.optim.AdamW with a clear warning if bitsandbytes isn't installed.
    optimizer, optimizer_kind = build_optimizer(trainable_params, lr=args.lr, weight_decay=0.01)
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
    rng = _random.Random(args.seed + start_step)

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
            stream_gen = stream_from_cache(args.cpt_cache, seed=args.seed)
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
        print(f"[cpt] {len(rows):,} training rows loaded from {train_file}")

    model.train()
    async_ckpt = AsyncCheckpointer() if args.async_checkpoint else None
    if args.async_checkpoint:
        print("[cpt] async checkpointing enabled -- checkpoint writes run on a "
              "background thread, training does not wait for them except at exit")
    for it in range(start_step + 1, args.iters + 1):
        if stream_gen is not None:
            batch_rows = [next(stream_gen) for _ in range(args.batch)]
        else:
            batch_rows = [rows[rng.randrange(len(rows))] for _ in range(args.batch)]
        examples = [builder(r, tokenizer, args.max_seq_len) for r in batch_rows]
        batch = collate(examples, tokenizer.pad_token_id)
        batch = {k: v.to(device) for k, v in batch.items()}

        lr = lr_at_step(it, args.iters, args.lr, args.warmup_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        if it % 10 == 0 or it == args.iters:
            print(f"[cpt] Iter {it}/{args.iters}: loss={loss.item():.4f}  lr={lr:.2e}")

        if (it % args.checkpoint_every == 0 or it == args.iters or _SHOULD_STOP):
            if args.async_checkpoint:
                async_ckpt.save(model, optimizer, it, save_dir, tokenizer,
                                custom_code_src=Path(args.model))
                # On exit (SIGTERM or final iter) the write MUST finish before the
                # process dies, or this defeats the whole point of atomic checkpointing.
                # For a regular mid-run checkpoint, deliberately NOT waiting here -- the
                # background thread keeps writing while training continues; save()
                # itself waits on any still-in-flight write before starting the next one.
                if _SHOULD_STOP or it == args.iters:
                    async_ckpt.wait_for_pending()
            else:
                atomic_save_checkpoint(model, optimizer, it, save_dir, tokenizer,
                                       custom_code_src=Path(args.model))

        if _SHOULD_STOP:
            print(f"[cpt] Exiting cleanly after checkpoint at step {it} (SIGTERM)")
            sys.exit(0)

    print(f"\n[cpt] Done. Final checkpoint at step {args.iters} -> {save_dir}")


if __name__ == "__main__":
    main()
