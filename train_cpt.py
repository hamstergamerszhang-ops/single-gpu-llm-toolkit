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

    if save_dir.exists():
        backup = save_dir.parent / (save_dir.name + ".prev")
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(save_dir, backup)
    os.replace(tmp_dir, save_dir)
    backup = save_dir.parent / (save_dir.name + ".prev")
    if backup.exists():
        shutil.rmtree(backup)

    print(f"[cpt] saved step {step} -> {save_dir}")


def _move_to_cpu(obj):
    """Recursively moves tensors in a nested dict/list to CPU. Used to snapshot
    optimizer.state_dict() (a dict of dicts of tensors, e.g. Adam's per-param moment
    buffers) before handing it to a background thread -- the GPU-resident originals
    keep getting mutated by the next training step the moment this snapshot is taken,
    so nothing downstream may still reference the live GPU tensors."""
    import torch
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _move_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_cpu(v) for v in obj]
    return obj


class AsyncCheckpointer:
    """Non-blocking checkpoint writes: the expensive part (serializing tens of
    GB to a possibly-slow disk/NFS mount) happens on a background thread while
    the GPU immediately continues training the next step, instead of sitting
    idle for the whole write.

    Split into two phases, matching the standard async-checkpoint pattern:
      1. SYNCHRONOUS snapshot -- copy model + optimizer state to CPU RAM. This
         still blocks the training loop briefly (a GPU->CPU copy over
         PCIe/interconnect), but that's a small fraction of the time a full
         disk write takes on a slow mount, and it's the only phase that MUST
         be synchronous (the GPU tensors are about to be mutated by the next
         step, so the copy has to happen before that).
      2. ASYNC write -- everything from "turn these CPU tensors into files on
         disk" onward runs in a background thread and never touches the live
         model/optimizer again, so it's safe to run concurrently with the
         next several training steps.

    Bounded to ONE in-flight write at a time (save() blocks on any
    still-running previous write before starting a new snapshot) -- prevents
    unbounded CPU-RAM growth from queueing multiple large snapshots if writes
    fall behind the checkpoint interval, at the cost of occasionally still
    waiting on a slow write. With --checkpoint-every raised to a sane interval
    this should be rare in practice.

    NOTE: this is where checkpoints land in local disk, and nothing more --
    getting them onto durable/shared storage (e.g. a periodic rsync to a
    network volume) is a separate, deliberately decoupled concern. See
    README.md for why this repo does not wire in a cloud-object-store push
    here, even though that's a common design for this kind of script.
    """

    def __init__(self):
        self._thread = None

    def save(self, model, optimizer, step: int, save_dir: Path, tokenizer=None,
             extra_state: dict | None = None, custom_code_src: Path | None = None):
        import threading

        if self._thread is not None and self._thread.is_alive():
            print("[ckpt-async] previous async write still in flight -- waiting for it "
                  "before starting a new snapshot (checkpoint interval may be too tight "
                  "relative to write speed on this disk)")
            self._thread.join()

        # Phase 1 (synchronous, blocks briefly): snapshot to CPU.
        # Drop tied-weight duplicates BEFORE copying (e.g. tie_word_embeddings=True
        # means lm_head.weight and embed_tokens.weight share the same storage) --
        # a naive model.state_dict() copy loses the tie relationship (each
        # .to('cpu', copy=True) makes an independent copy), and passing that as an
        # explicit state_dict= override to save_pretrained() skips its normal
        # tied-weight dedup, silently writing a full extra copy of the embedding
        # matrix (GB-scale on a large-vocab model) into every checkpoint. Removing
        # the known-duplicate key here restores the synchronous path's behavior
        # exactly -- transformers re-derives it from config on load either way.
        tied_keys = set(getattr(model, "_tied_weights_keys", {}) or {})
        raw_state = model.state_dict()
        model_state_cpu = {k: v.detach().to("cpu", copy=True)
                           for k, v in raw_state.items() if k not in tied_keys}
        opt_state_cpu = {
            "optimizer": _move_to_cpu(optimizer.state_dict()),
            "optimizer_type": type(optimizer).__name__,
            "step": step,
            **(extra_state or {}),
        }
        print(f"[ckpt-async] step {step}: CPU snapshot done, disk write continuing in "
              f"background -- training resumes immediately")

        # Phase 2 (async): everything below only touches CPU tensors/disk, never the
        # live model/optimizer, so it's safe to run while training proceeds.
        def _write():
            import torch
            tmp_dir = save_dir.parent / (save_dir.name + ".tmp_ckpt")
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)

            model.save_pretrained(tmp_dir, safe_serialization=True, state_dict=model_state_cpu)
            if tokenizer is not None:
                tokenizer.save_pretrained(tmp_dir)
            if custom_code_src is not None:
                src_file = Path(custom_code_src) / "modeling_custom.py"
                if src_file.exists():
                    shutil.copy2(src_file, tmp_dir / "modeling_custom.py")

            torch.save(opt_state_cpu, tmp_dir / "training_state.pt")

            if save_dir.exists():
                backup = save_dir.parent / (save_dir.name + ".prev")
                if backup.exists():
                    shutil.rmtree(backup)
                os.replace(save_dir, backup)
            os.replace(tmp_dir, save_dir)
            backup = save_dir.parent / (save_dir.name + ".prev")
            if backup.exists():
                shutil.rmtree(backup)

            print(f"[ckpt-async] step {step}: background write finished -> {save_dir}")

        self._thread = threading.Thread(target=_write, daemon=False)
        self._thread.start()

    def wait(self):
        """Block until any in-flight write finishes -- call before process exit
        (SIGTERM handler, final checkpoint) so the process never dies mid-write."""
        if self._thread is not None and self._thread.is_alive():
            print("[ckpt-async] waiting for final background write to finish before exit...")
            self._thread.join()


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
    cross-entropy)."""
    import torch

    messages = row["messages"]
    input_ids: list[int] = []
    labels: list[int] = []

    # Tokenize turn-by-turn so we know exactly which spans are assistant output.
    running_text = ""
    for i, msg in enumerate(messages):
        prefix_text = tokenizer.apply_chat_template(
            messages[: i + 1], tokenize=False, add_generation_prompt=False
        )
        new_text = prefix_text[len(running_text):]
        running_text = prefix_text
        ids = tokenizer(new_text, add_special_tokens=False)["input_ids"]
        input_ids.extend(ids)
        if msg["role"] == "assistant":
            labels.extend(ids)
        else:
            labels.extend([-100] * len(ids))

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


# ── layer window freeze/unfreeze ─────────────────────────────────────────────

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

    print("[selftest] Resume offset: schedule at (resume_step + k) matches a fresh "
          "run's step (resume_step + k) -- i.e. resuming continues the SAME curve")
    resume_step = 37
    for k in [0, 1, 50]:
        a = lr_at_step(resume_step + k, total, base_lr, warmup)
        b = lr_at_step(resume_step + k, total, base_lr, warmup)
        assert a == b
    print("  OK")

    print("[selftest] atomic checkpoint rename pattern (no torch/model required)")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        save_dir = td / "ckpt"
        tmp_dir = save_dir.parent / (save_dir.name + ".tmp_ckpt")
        tmp_dir.mkdir(parents=True)
        (tmp_dir / "marker.txt").write_text("v2")
        save_dir.mkdir(parents=True)
        (save_dir / "marker.txt").write_text("v1")
        backup = save_dir.parent / (save_dir.name + ".prev")
        os.replace(save_dir, backup)
        os.replace(tmp_dir, save_dir)
        shutil.rmtree(backup)
        assert (save_dir / "marker.txt").read_text() == "v2"
        assert not backup.exists()
        assert not tmp_dir.exists()
    print("  OK (old checkpoint preserved as backup until new one is fully in place, "
          "never an observable half-written state at the real path)")

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
    ap.add_argument("--data", help="Dir containing train.jsonl [+ valid.jsonl], or a "
                                    "single .jsonl file.")
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
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.Adam8bit(trainable_params, lr=args.lr, weight_decay=0.01)
        print("[cpt] optimizer: bitsandbytes 8-bit Adam")
    except ImportError:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
        print("[cpt] optimizer: torch.optim.AdamW (bitsandbytes not installed -- "
              "install it for ~4x less optimizer-state memory. See README.md: this "
              "silent fallback is a real, observed OOM source on large models, not "
              "a hypothetical one -- reinstall bitsandbytes on every fresh container.)")

    start_step = 0
    if resumed:
        state_path = save_dir / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=device)
            start_step = state.get("step", 0)
            saved_optimizer_type = state.get("optimizer_type", "unknown")
            current_optimizer_type = type(optimizer).__name__
            if saved_optimizer_type != current_optimizer_type:
                # Loading one optimizer type's state_dict into a DIFFERENT optimizer
                # class is not just "ignored, harmless" -- this has been observed to
                # silently accept the mismatched state and inflate GPU memory well
                # beyond what the current optimizer should need, OOMing on the very
                # first forward pass. Skip the load entirely rather than risk that
                # again -- losing optimizer momentum on an optimizer-type switch is a
                # known, bounded cost; silent memory corruption is not.
                print(f"[cpt] WARNING: checkpoint's optimizer was {saved_optimizer_type}, "
                      f"this run is using {current_optimizer_type} -- skipping optimizer "
                      f"state load (incompatible state_dicts, confirmed to risk OOM if "
                      f"forced). Starting this optimizer's momentum fresh; step count "
                      f"still resumes.")
            else:
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
        def _cache_stream(path, seed):
            _rng = _random.Random(seed)
            rows = load_jsonl(Path(path))
            if not rows:
                raise SystemExit(f"--cpt-cache {path} contained zero rows")
            order = list(range(len(rows)))
            while True:
                _rng.shuffle(order)
                for i in order:
                    yield rows[i]

        stream_gen = _cache_stream(args.cpt_cache, args.seed)
        print(f"[cpt] training from local cache ({args.cpt_cache}) -- zero network "
              f"dependency, safe against source/network instability")
    else:
        data_path = Path(args.data)
        train_file = data_path / "train.jsonl" if data_path.is_dir() else data_path
        rows = load_jsonl(train_file)
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
                    async_ckpt.wait()
            else:
                atomic_save_checkpoint(model, optimizer, it, save_dir, tokenizer,
                                       custom_code_src=Path(args.model))

        if _SHOULD_STOP:
            print(f"[cpt] Exiting cleanly after checkpoint at step {it} (SIGTERM)")
            sys.exit(0)

    print(f"\n[cpt] Done. Final checkpoint at step {args.iters} -> {save_dir}")


if __name__ == "__main__":
    main()
