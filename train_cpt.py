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
import contextlib
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


# ── throughput / MFU estimation ─────────────────────────────────────────────

def estimate_step_tflops(n_trainable_params: int, n_tokens: int, step_time_s: float) -> float:
    """Rough achieved TFLOPs/s for one optimizer step.

    Uses the standard ~6*N*T heuristic (2 FLOPs/param/token forward + 4 backward)
    for a causal LM. `n_tokens` should be the total tokens processed across all
    micro-batches in the step (including padding — exact for packed sequences).
    Returns achieved TFLOPs/s; divide by your GPU's bf16 peak to get MFU.
    """
    if step_time_s <= 0:
        return 0.0
    # 6 FLOPs per parameter per token; /1e12 -> TFLOPs/s.
    return (6.0 * n_trainable_params * n_tokens) / step_time_s / 1e12


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
        # NOTE: optimizer.__class__.__name__ (attribute access), NOT
        # type(optimizer).__name__ (the builtin). Under FSDP, `optimizer` can
        # be a _StateDictOptimizer shim (see train_cpt.py's checkpoint-write
        # block), which overrides __class__ via a @property specifically so
        # this reports the WRAPPED optimizer's real class name (e.g.
        # "AdamW") instead of "_StateDictOptimizer". But `type(x)` is a
        # builtin that reads the instance's actual C-level type slot and
        # does NOT consult a __class__ property override -- only normal
        # attribute access (`x.__class__`) does. A prior version of this
        # line used type(optimizer).__name__, which always wrote
        # "_StateDictOptimizer" here on every FSDP save, so on resume
        # saved_optimizer_type ("_StateDictOptimizer") could never match
        # current_optimizer_type (the real class, since the shim doesn't
        # exist at resume time) -- check_optimizer_compat() always returned
        # safe_to_load=False, silently discarding Adam momentum on every
        # single FSDP checkpoint resume. Exactly the failure mode
        # _StateDictOptimizer's own docstring says this __class__ override
        # exists to prevent.
        "optimizer_type": optimizer.__class__.__name__,
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
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[cpt] WARNING: skipping malformed JSON line in {path}: {e}",
                          file=sys.stderr)
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
        if last_assistant_idx == -1:
            return {"input_ids": torch.tensor([], dtype=torch.long),
                    "labels": torch.tensor([], dtype=torch.long)}
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
    MI300X/MI300A/MI325X have native fp8 compute — this roughly 2x throughput
    vs bf16 on those cards. Falls back to bf16 (no-op) with a warning if
    torchao isn't installed, the GPU lacks fp8 hardware, or the conversion
    fails."""
    import torch
    # Runtime capability gate: fp8 matmul (torch._scaled_mm) requires gfx942
    # (MI300X/MI300A/MI325X). On other AMD cards (e.g. gfx1100 / RX 7900 XTX),
    # the conversion would succeed but the first forward pass would crash with
    # no kernel. Check up front so the user gets a clear message instead.
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        # On ROCm, get_device_capability returns (gfx_major, gfx_minor).
        # MI300X, MI300A, AND MI325X are all gfx942 -- verified directly
        # against AMD's own current gpu-arch-specs docs (all three list
        # "CDNA3" / "gfx942"; there is no separate gfx940/gfx941 target for
        # MI300A/MI325X as an earlier version of this comment claimed).
        # gfx940/gfx941 appear in some early/internal ROCm references but are
        # not real, currently-shipping distinct architectures -- don't
        # reintroduce a "gfx940=MI300A, gfx941=MI325X" mapping. The `>= 40`
        # bound below is intentionally a little loose (accepts the whole
        # gfx940-94f range) as a defensive margin, not because those other
        # values correspond to real distinct chips.
        if not (cap[0] == 9 and cap[1] >= 40):
            arch = f"gfx{cap[0]}{cap[1]}"
            print(f"[cpt] WARNING: --dtype fp8 but GPU arch {arch} lacks native "
                  f"fp8 compute (needs gfx942: MI300X/MI300A/MI325X). Falling "
                  f"back to bf16. Use --dtype fp8 only on gfx942 CDNA3 hardware.",
                  file=sys.stderr)
            return model
    try:
        from torchao.float8 import convert_to_float8_training
        convert_to_float8_training(model)
        print("[cpt] fp8 training enabled (torchao Float8Linear, float8_e4m3fn) — "
              "native fp8 compute on gfx942 (MI300X/MI300A/MI325X).")
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
    except Exception as e:
        print(f"[cpt] WARNING: --flash-attn failed ({e}) — using standard "
              f"attention. This can happen if the architecture doesn't support "
              f"FA2, or if flash-attn was built for the wrong gfx arch.", file=sys.stderr)


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





def unwrap_ddp(model):
    """Returns the underlying model if `model` is DDP-wrapped, else `model`
    itself unchanged. Duck-typed on the class name (not isinstance) so this
    stays importable/testable without a torch.distributed process group --
    same convention find_decoder_layers below already uses.

    Why this matters (regression test for a real deadlock found in review):
    calling the DDP wrapper's own forward() -- i.e. `model(**batch)` where
    `model` is still the DistributedDataParallel instance -- runs
    _pre_forward()/_post_forward(), which broadcast-syncs the module's
    buffers (RoPE's non-persistent inv_freq, among others) as a COLLECTIVE
    operation whenever require_forward_param_sync is True (true right after
    any training-step forward, independent of grad mode). If only rank 0
    ever makes that forward call -- e.g. a held-out eval loop gated on
    `is_main` -- every other rank never joins that collective, and the
    process group hangs (or, as reproduced with a real 2-process gloo job in
    review, crashes with a protocol desync error instead). Calling
    `unwrap_ddp(model)(**batch)` -- the raw nn.Module, not the DDP wrapper --
    triggers none of DDP's hooks and therefore no collective at all, so a
    rank-0-only forward call (checkpointing, eval) is always safe. Confirmed
    against torch's own DistributedDataParallel source that no_sync() does
    NOT help here: it only flips require_backward_grad_sync (gradient
    all-reduce on backward()), and buffer sync is gated on a separate flag
    checked in the forward pre-hook, unaffected by no_sync().
    """
    if type(model).__name__ == "DistributedDataParallel":
        return model.module
    return model


def _fsdp_unwrap(model):
    """Returns the underlying model if `model` is FSDP-wrapped, else `model`
    itself. Duck-typed on the class name (not isinstance) so this stays
    importable/testable without a torch.distributed process group.

    FSDP exposes the wrapped module as .module (same attribute name as DDP),
    so this also handles single-level FSDP wrapping. For nested FSDP wrapping
    (auto_wrap_policy wraps each transformer layer individually), only the
    top-level is unwrapped here — callers that need the fully unsharded state
    dict use get_full_state_dict() instead."""
    if type(model).__name__ in ("FullyShardedDataParallel", "DistributedDataParallel"):
        return model.module
    return model


def get_full_state_dict(model):
    """Gather the full (unsharded) model state dict from an FSDP-wrapped model.

    FSDP shards parameters across ranks, so model.state_dict() on an FSDP
    wrapper returns only the local shard. To save a loadable checkpoint, we
    need the FULL state dict — gathered from all ranks onto rank 0 (rank 0
    gets the real tensors, other ranks get empty placeholders to free memory).

    For non-FSDP models (single-GPU, DDP), this is just model.state_dict().

    IMPORTANT: This is a COLLECTIVE operation under FSDP. All ranks must call
    it (even though only rank 0 gets the result). Calling it only on rank 0
    will deadlock — the all-gather needs all ranks to participate.
    """
    import torch
    cls_name = type(model).__name__
    if cls_name == "FullyShardedDataParallel":
        # FSDP's recommended pattern for full state dict gathering:
        # set state_dict_type to FULL_STATE_DICT, call get_state_dict (which
        # gathers shards to rank 0), use it, then reset to the original type.
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig
        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
            # All ranks call this; rank0_only=True means only rank 0 gets the
            # real gathered tensors, others get an empty dict.
            full_sd = model.state_dict()
            # Clone on rank 0 so the dict is independent of FSDP internals
            # after the context exits. Other ranks get {} (no copy needed).
            if full_sd:
                full_sd = {k: v.clone() for k, v in full_sd.items()}
        return full_sd
    # Non-FSDP: DDP or single-GPU. state_dict() is already complete.
    return unwrap_ddp(model).state_dict()


def get_full_optim_state_dict(model, optimizer):
    """Gather the full (unsharded) optimizer state dict from an FSDP-wrapped model.

    Like get_full_state_dict, but for optimizer state. The optimizer is built
    over FSDP-wrapped params, so optimizer.state_dict() returns only the local
    shard. This gathers the full state onto rank 0.

    IMPORTANT: Collective under FSDP — all ranks must call it.

    Uses FSDP.state_dict_type with StateDictType.FULL_STATE_DICT, passing BOTH
    a FullStateDictConfig (model config, 3rd arg) and a FullOptimStateDictConfig
    (optim config, 4th arg). PyTorch uses a single StateDictType enum for both
    model and optimizer state — there is no separate OptimStateDictType.
    FSDP.optim_state_dict(model, optimizer) is the recommended gathering call.
    """
    if type(model).__name__ == "FullyShardedDataParallel":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import (
            StateDictType, FullStateDictConfig, FullOptimStateDictConfig)
        # Both configs are required: state_dict_type validates that the
        # state_dict_config (3rd arg) is FullStateDictConfig and the
        # optim_state_dict_config (4th arg) is FullOptimStateDictConfig.
        model_cfg = FullStateDictConfig(rank0_only=True)
        optim_cfg = FullOptimStateDictConfig(rank0_only=True)
        with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, model_cfg, optim_cfg):
            # FSDP.optim_state_dict is the recommended gathering call — it
            # returns the full (unsharded) optimizer state on rank 0 and {}
            # on other ranks (when rank0_only=True).
            full_optim_sd = FSDP.optim_state_dict(model, optimizer)
        return full_optim_sd
    # Non-FSDP: optimizer.state_dict() is already complete.
    return optimizer.state_dict()


def shard_optim_state_dict_for_load(model, optimizer, full_optim_sd):
    """Convert a full (unsharded) optimizer state dict to the sharded format
    FSDP expects for loading on resume.

    The inverse of get_full_optim_state_dict: takes a saved FULL_STATE_DICT
    optimizer state and returns the sharded version that
    optimizer.load_state_dict() can consume under FSDP.

    Collective under FSDP — all ranks must call it.
    """
    if type(model).__name__ == "FullyShardedDataParallel":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import (
            StateDictType, FullStateDictConfig, FullOptimStateDictConfig)
        model_cfg = FullStateDictConfig(rank0_only=True)
        optim_cfg = FullOptimStateDictConfig(rank0_only=True)
        with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, model_cfg, optim_cfg):
            # optim_state_dict_to_load converts the full state dict to the
            # sharded format the local optimizer expects.
            sharded = FSDP.optim_state_dict_to_load(
                model, optimizer, full_optim_sd)
        return sharded
    # Non-FSDP: the state dict is already in the right format.
    return full_optim_sd


class _StateDictModel:
    """Shim that wraps an HF model and overrides save_pretrained() to use a
    pre-gathered full state dict (for FSDP checkpoint saving).

    FSDP shards parameters across ranks, so the model's own state_dict() returns
    only the local shard. get_full_state_dict() gathers the full state dict onto
    rank 0, but the underlying model still has sharded params. This shim lets
    atomic_save_checkpoint / async_ckpt.save call save_pretrained() with the
    gathered state dict without modifying those functions.

    Delegates everything except save_pretrained to the wrapped model (config,
    tokenizer, etc.) via __getattr__.
    """
    def __init__(self, model, full_state_dict):
        self._model = model
        self._full_state_dict = full_state_dict

    def state_dict(self, *args, **kwargs):
        # Return the gathered full state (NOT the wrapped model's sharded state).
        # This is critical for AsyncCheckpointer.save, which calls
        # model.state_dict() to snapshot before handing to the background
        # thread, then strips tied-weight keys from THAT snapshot (see
        # async_checkpoint.py's tied_keys dedup) and passes the deduped dict
        # back in via save_pretrained(..., state_dict=...). See save_pretrained
        # below for why we must NOT clobber that caller-supplied dict.
        return self._full_state_dict

    def save_pretrained(self, save_directory, **kwargs):
        # Use the gathered full state dict instead of the model's own (sharded)
        # state_dict() -- but ONLY if the caller didn't already pass one in.
        # async_checkpoint.py's save() calls model.state_dict() (our override
        # above, returning self._full_state_dict), then strips tied-weight
        # keys from that snapshot into its own `model_state_cpu`, and passes
        # THAT deduped dict here as state_dict=. A prior version of this shim
        # unconditionally did `kwargs["state_dict"] = self._full_state_dict`,
        # which clobbered the caller's deduped dict with the raw (undeduped)
        # one -- silently writing a full extra copy of the tied embedding
        # matrix (GB-scale on a large-vocab model) into every FSDP checkpoint,
        # since the sync save path (atomic_save_checkpoint, which never passes
        # state_dict=) still needs this shim to supply one. Only fall back to
        # self._full_state_dict when the caller hasn't already supplied one.
        kwargs.setdefault("state_dict", self._full_state_dict)
        # safe_serialization defaults to True in modern transformers; pass it
        # explicitly so the behavior matches the non-FSDP path.
        kwargs.setdefault("safe_serialization", True)
        self._model.save_pretrained(save_directory, **kwargs)

    def __getattr__(self, name):
        # Delegate any attribute access (config, etc.) to the wrapped model.
        # __getattr__ is only called when normal attribute lookup fails, so
        # self._model and self._full_state_dict are found normally.
        return getattr(self._model, name)


class _StateDictOptimizer:
    """Shim that wraps an optimizer and overrides state_dict() to return a
    pre-gathered full optimizer state dict (for FSDP checkpoint saving).

    Same pattern as _StateDictModel: FSDP shards optimizer state, so we gather
    the full state via get_full_optim_state_dict() and return it from
    state_dict() instead of the sharded local state.

    Delegates everything else to the wrapped optimizer via __getattr__.
    """
    def __init__(self, optimizer, full_state_dict):
        self._optimizer = optimizer
        self._full_state_dict = full_state_dict

    def state_dict(self):
        return self._full_state_dict

    @property
    def __class__(self):
        # Make type(shim).__name__ report the wrapped optimizer's class name
        # (e.g. "Adam8bit" / "AdamW") so the checkpoint's "optimizer_type"
        # metadata is correct for optimizer_compat_guard's resume check.
        # Without this, type(shim).__name__ would be "_StateDictOptimizer",
        # which never matches the real optimizer on resume -> the guard
        # would always return safe_to_load=False, silently discarding
        # Adam momentum on every FSDP resume.
        return type(self._optimizer)

    def __getattr__(self, name):
        return getattr(self._optimizer, name)


def _wrap_fsdp(model, sharding_strategy: str, local_rank: int):
    """Wrap a model in FullyShardedDataParallel with an auto_wrap_policy that
    shards each transformer decoder layer independently. Returns the wrapped
    model. Called after model modifications (fp8, flash-attn, compile) but
    before apply_window_freeze.

    auto_wrap_policy: wraps modules whose class name matches common transformer
    decoder layer patterns (GemmaDecoderLayer, LlamaDecoderLayer, etc.). This
    gives FSDP the right granularity — too-large a unit (the whole model) means
    one rank holds the full param set momentarily during all-gather; too-small
    (individual Linears) adds communication overhead per layer.
    """
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    strategy_map = {
        "full": ShardingStrategy.FULL_SHARD,
        "shard-grad-op": ShardingStrategy.SHARD_GRAD_OP,
        "no-shard": ShardingStrategy.NO_SHARD,
    }
    strategy = strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)

    # Auto-wrap transformer decoder layers. We match by class name suffix
    # ("DecoderLayer", "Block") to cover Gemma/Llama/Mistral/Qwen naming
    # without importing model-specific classes (which may not be installed).
    def _is_decoder_layer(module):
        cls = type(module).__name__
        return cls.endswith("DecoderLayer") or cls.endswith("Block")

    from functools import partial
    from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

    layer_policy = partial(lambda_auto_wrap_policy, lambda_fn=_is_decoder_layer)

    model = FSDP(
        model,
        sharding_strategy=strategy,
        auto_wrap_policy=layer_policy,
        device_id=local_rank if torch.cuda.is_available() else None,
        # use_orig_params=True: keeps parameter names stable so the optimizer
        # state_dict keys match across save/resume. Without this, FSDP flattens
        # params into "FlatParamHandle" groups and the optimizer state keys
        # become indecipherable, breaking resume.
        use_orig_params=True,
        # limit_all_gathers=True: overlaps all-gather with forward compute to
        # reduce peak memory — critical on MI300X where the all-gathered params
        # for a layer can be large. PyTorch 2.0+ default is True but we set it
        # explicitly for clarity and older versions.
        limit_all_gathers=True,
    )
    return model


def find_decoder_layers(model):
    """Locate the transformer's layer list across a handful of HF model-class
    shapes a Gemma-4-family checkpoint might load as. Unwraps DDP/FSDP
    (whose wrapped model is at .module) before walking attributes."""
    # Unwrap DDP/FSDP: both expose the wrapped model as .module, and neither
    # forwards attribute access to it (hasattr(ddp, "model") is False).
    # Without this, find_decoder_layers fails on every --ddp/--fsdp run.
    # Duck-type via class name to avoid importing torch in this module-level
    # function (same convention unwrap_ddp uses).
    if type(model).__name__ in ("DistributedDataParallel", "FullyShardedDataParallel"):
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
    # REQUIRED: main() re-assigns _SHOULD_STOP further down (the rank-0 ->
    # all-ranks broadcast under --ddp/--fsdp). Without this `global`
    # declaration, that assignment makes Python treat _SHOULD_STOP as a LOCAL
    # name for the entire function body (Python's scoping is static: any
    # assignment anywhere in a function marks the name local for the whole
    # function, regardless of where the assignment sits or whether it's
    # behind a conditional) -- which turns every earlier READ of
    # _SHOULD_STOP in this function (including the one inside the broadcast
    # block itself, which reads it before conditionally reassigning it) into
    # an UnboundLocalError on the very first loop iteration under
    # --ddp/--fsdp. Confirmed with a minimal repro of the same read-before-
    # conditional-assign shape while reviewing this file.
    global _SHOULD_STOP
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
                         "accumulated micro-batches. Under --ddp, the non-final "
                         "micro-batches use DDP no_sync() to skip redundant gradient "
                         "all-reduces -- only the last micro-batch's backward syncs, "
                         "which is the correct and fast implementation.")
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
                         "for ~2x throughput on gfx942 (MI300X/MI300A/MI325X -- all "
                         "three are gfx942, per AMD's own gpu-arch-specs docs; there "
                         "is no separate gfx940/gfx941 architecture for MI300A/MI325X). "
                         "A runtime capability gate checks the GPU arch and gracefully "
                         "falls back to bf16 with a warning on any card outside gfx942, "
                         "so --dtype fp8 is safe to pass on any AMD card. Requires the "
                         "'torchao' package; falls back to bf16 if torchao is missing "
                         "or conversion fails.")
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
    ap.add_argument("--fsdp", action="store_true", default=False,
                    help="Enable multi-GPU training via FullyShardedDataParallel "
                         "(FSDP). Shards params/grads/optimizer state across GPUs, "
                         "so models larger than a single GPU's VRAM can be trained "
                         "(e.g. a 27B model across 4x MI300X). Also avoids the "
                         "find_unused_parameters=True hazard that --ddp hits with "
                         "windowed --start/--end freezing. Launch with 'torchrun "
                         "--nproc_per_node=N train_cpt.py --fsdp ...'. Mutually "
                         "exclusive with --ddp.")
    ap.add_argument("--sharding-strategy", type=str, default="full",
                    choices=["full", "shard-grad-op", "no-shard"],
                    help="FSDP sharding strategy (only with --fsdp). 'full' "
                         "(FULL_SHARD, default) shards params+grads+optimizer "
                         "state -- maximum memory savings, most communication. "
                         "'shard-grad-op' (SHARD_GRAD_OP) shards grads+optimizer "
                         "state only -- params stay replicated, less comm overhead, "
                         "more memory. 'no-shard' (NO_SHARD) is equivalent to DDP.")
    args = ap.parse_args()

    # --checkpoint-every of 0 would hit ZeroDivisionError on `it % args.checkpoint_every`
    # in the training loop; validate up front with a clear message.
    if args.checkpoint_every < 1:
        ap.error("--checkpoint-every must be >= 1")

    # --accum of 0 makes `for micro in range(args.accum):` never execute --
    # `outputs`/`last_loss` are never assigned that step, so the `del outputs`
    # right after the loop raises NameError (and even without that, zero
    # micro-batches means zero backward() calls, so optimizer.step() would
    # apply a stale or nonexistent gradient). Validate up front, same as
    # --checkpoint-every above.
    if args.accum < 1:
        ap.error("--accum must be >= 1")

    if args.selftest:
        self_test()
        return

    if not (args.model and args.save and (args.data or args.cpt_cache)):
        ap.error("--model and --save are required, plus one of --data or --cpt-cache, "
                 "unless --selftest is given.")

    # Validate accumulation: range(accum) is empty for accum <= 0, which would
    # leave `loss` as None and crash at loss.item() in the logging block. Catch
    # it here with a clear message instead.
    if args.accum < 1:
        ap.error("--accum / --gradient-accumulation-steps must be >= 1 "
                 f"(got {args.accum}).")

    # --fsdp and --ddp are mutually exclusive (both set up process groups and
    # wrap the model; using both would double-init distributed and crash).
    if args.fsdp and args.ddp:
        ap.error("--fsdp and --ddp are mutually exclusive. --fsdp shards params "
                 "across GPUs (for models that don't fit on one); --ddp replicates "
                 "params on every GPU (for models that do). Pick one.")

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

    # ── Multi-GPU DDP/FSDP setup ──────────────────────────────────────────────
    # When --ddp or --fsdp is set, the script expects to be launched via torchrun:
    #   torchrun --nproc_per_node=N train_cpt.py --ddp --model ... --save ...
    #   torchrun --nproc_per_node=N train_cpt.py --fsdp --model ... --save ...
    # torchrun sets RANK, LOCAL_RANK, WORLD_SIZE env vars. We init the process
    # group, pin each rank to its local GPU, and use DDP or FSDP to sync gradients.
    # Only rank 0 writes checkpoints, logs to stdout, and runs eval — the other
    # ranks train silently and participate in the gradient sync.
    ddp_rank = 0
    ddp_world_size = 1
    is_main = True  # rank 0 (or single-GPU)
    is_distributed = args.ddp or args.fsdp
    if is_distributed:
        if "RANK" not in os.environ:
            raise SystemExit(f"ERROR: --{'ddp' if args.ddp else 'fsdp'} set but RANK "
                             f"env var not found. Launch via 'torchrun "
                             f"--nproc_per_node=N train_cpt.py "
                             f"--{'ddp' if args.ddp else 'fsdp'} ...' so torchrun "
                             f"sets RANK/LOCAL_RANK/WORLD_SIZE.")
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
            mode = "FSDP" if args.fsdp else "DDP"
            print(f"[cpt] {mode} enabled: rank {ddp_rank}/{ddp_world_size}, "
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
    # For FSDP, load WITHOUT .to(device) — FSDP handles device placement via
    # device_id at wrap time, and pre-moving the whole model to one GPU defeats
    # the memory savings (the full model would momentarily sit on rank 0's VRAM).
    # Load on CPU, let FSDP shard it to the right devices.
    #
    # IMPORTANT: this must match the EXACT condition used below to decide
    # whether the model actually gets FSDP-wrapped (`args.fsdp and
    # ddp_world_size > 1`), not just `args.fsdp` alone. `torchrun
    # --nproc_per_node=1 ... --fsdp` sets RANK=0/WORLD_SIZE=1 (torchrun sets
    # these even for a single process), which passes the `--fsdp` validation
    # above but makes `ddp_world_size == 1` — so `if args.fsdp and
    # ddp_world_size > 1` below is False and _wrap_fsdp() is never called. A
    # prior version of this branch keyed off `args.fsdp` alone: the model was
    # skipped past `.to(device)` AND never wrapped/moved by FSDP either,
    # silently training on CPU with no error (GPU-tensor-vs-CPU-model crashes
    # only surface much later, if at all, deep in the training loop).
    will_wrap_fsdp = args.fsdp and ddp_world_size > 1
    if will_wrap_fsdp:
        model = AutoModelForCausalLM.from_pretrained(load_path, **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(load_path, **load_kwargs).to(device)
    model.config.use_cache = False  # incompatible with checkpointing/training either way
    if not args.no_grad_checkpoint:
        # FSDP requires use_reentrant=False for gradient checkpointing (the
        # reentrant variant is incompatible with FSDP's forward hooks and raises
        # "Calling _checkpoint without use_reentrant=False is incompatible with
        # FSDP"). DDP and single-GPU work with either; use_reentrant=False is
        # the PyTorch-recommended default for new code regardless.
        # HF's gradient_checkpointing_enable takes a SINGLE dict kwarg
        # `gradient_checkpointing_kwargs`, not **kwargs — passing use_reentrant
        # directly raises TypeError.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
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

    # ── Distributed wrapping: DDP or FSDP ─────────────────────────────────────
    # Both wrap the model after all modifications (fp8, flash-attn, compile) but
    # before apply_window_freeze, so the wrapper sees the final parameter set.
    # DDP replicates params on every GPU and all-reduces gradients; FSDP shards
    # params/grads/optimizer state across GPUs (fits bigger models, avoids the
    # find_unused_parameters hazard with windowed freezing).
    windowed_freeze = args.start != 0 or args.end is not None
    if will_wrap_fsdp:
        if windowed_freeze:
            print("[cpt] NOTE: --fsdp with windowed --start/--end is safe (FSDP "
                  "handles frozen params without find_unused_parameters, unlike "
                  "DDP). No performance warning needed for this combination.")
        model = _wrap_fsdp(model, args.sharding_strategy, local_rank)
        if is_main:
            print(f"[cpt] model wrapped in FullyShardedDataParallel "
                  f"(world_size={ddp_world_size}, strategy={args.sharding_strategy})")
    elif args.fsdp:
        # --fsdp was requested but world_size == 1 (e.g. `torchrun
        # --nproc_per_node=1 ... --fsdp`, or WORLD_SIZE unset/1 while RANK is
        # present). FSDP sharding across 1 rank has no effect, so we already
        # skipped _wrap_fsdp() above -- the model was instead loaded with
        # .to(device) (see will_wrap_fsdp at load time). Tell the user
        # explicitly rather than silently behaving like plain single-GPU
        # training under a flag that implies distributed sharding.
        if is_main:
            print(f"[cpt] NOTE: --fsdp requested but world_size={ddp_world_size} "
                  f"(<=1) -- FSDP sharding needs >1 rank to do anything, so "
                  f"training proceeds as plain single-GPU (model on {device}, "
                  f"not FSDP-wrapped). Use 'torchrun --nproc_per_node=N>=2 "
                  f"... --fsdp' to actually shard.")
    elif args.ddp and ddp_world_size > 1:
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
                  "full-model training (--start 0 with no --end) or switch to --fsdp.",
                  file=sys.stderr)
        # find_unused_parameters=True handles the windowed-freeze case where only
        # a subset of params have requires_grad=True — DDP needs to know which
        # params participate in backward to all-reduce correctly (without this,
        # DDP's default assumption that every param participates in every
        # backward pass raises a runtime error the moment a frozen param's
        # gradient never arrives).
        #
        # NOTE for MTP checkpoints (mtp_head.py + modeling_custom.py) under
        # full-model --ddp (windowed_freeze=False here): modeling_custom.py's
        # CustomForCausalLM is an explicitly-unfinished stub whose forward()
        # does NOT add an MTP loss term (see that file's docstring) -- if you
        # use it as-is, the MTP params (enorm/eh_proj/block/lnorm/norm) never
        # receive gradients, and find_unused_parameters=False will make DDP
        # raise. Once you extend forward() to actually include the MTP loss
        # (the whole point of extending it), every param participates and
        # False is correct. If you need to run DDP against the unmodified
        # stub for some reason, pass windowed --start/--end (even a no-op
        # window covering all layers) to force find_unused_parameters=True,
        # or use --fsdp instead (FSDP doesn't have this hazard).
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=windowed_freeze,
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
                # Under FSDP, the saved optimizer state is a FULL (unsharded)
                # state dict (gathered at save time via get_full_optim_state_dict).
                # The optimizer expects SHARDED state, so we must convert the
                # full state to the sharded format before loading. This is a
                # collective — all ranks must call it.
                optim_state = state["optimizer"]
                if args.fsdp:
                    optim_state = shard_optim_state_dict_for_load(
                        model, optimizer, optim_state)
                optimizer.load_state_dict(optim_state)
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

    # Trainable param count for MFU/throughput estimation (computed once; cheap).
    n_trainable = sum(p.numel() for p in trainable_params)
    # Reset peak memory so the first reported VRAM number reflects steady-state
    # allocation, not one-off setup (compile, fp8 conversion, model load).
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

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
        # profiler.__enter__() is deferred into the try block below so that if
        # __enter__ raises, the `finally` doesn't try to __exit__ a profiler
        # that was never entered (which would mask the real error).

    # Wrap the training loop in try/finally so cleanup (profiler, tb_writer,
    # DDP process group) runs even on exception (OOM, CUDA error, etc.).
    # Without this, an exception mid-loop leaks the profiler context, leaves
    # TB events unflushed, and leaves NCCL in a dirty state.
    try:
        if profiler is not None:
            profiler.__enter__()
            print(f"[cpt] profiling enabled — trace artifacts -> {args.profile} "
                  f"(viewable in chrome://tracing or Perfetto)")
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
            step_tokens = 0
            if is_main and torch.cuda.is_available():
                step_start = time.time()
            # Hoisted out of the micro-batch loop: the wrapper type doesn't
            # change between micro-batches, so checking once is correct and
            # avoids a per-micro-batch isinstance() call.
            # Both DDP and FSDP support no_sync() for gradient accumulation —
            # DDP skips the gradient all-reduce, FSDP skips the reduce-scatter
            # and all-gather. Same semantics: only the final micro-batch's
            # backward triggers the sync.
            model_cls_name = type(model).__name__
            supports_no_sync = model_cls_name in (
                "DistributedDataParallel", "FullyShardedDataParallel")
            for micro in range(args.accum):
                if stream_gen is not None:
                    batch_rows = [next(stream_gen) for _ in range(args.batch)]
                else:
                    batch_rows = [rows[rng.randrange(len(rows))] for _ in range(args.batch)]
                examples = [builder(r, tokenizer, args.max_seq_len) for r in batch_rows]
                if args.pack:
                    examples = pack_examples(examples, args.max_seq_len)
                batch = collate(examples, tokenizer.pad_token_id)
                batch = {k: v.to(device) for k, v in batch.items()}
                step_tokens += batch["input_ids"].numel()

                # DDP optimization: skip the gradient all-reduce on every
                # micro-batch except the last. Without this, DDP all-reduces
                # (or FSDP reduce-scatter + all-gather) after EACH backward --
                # with accumulation that's accum-1 redundant syncs per step, a
                # real throughput hit on MI300X multi-GPU. no_sync() defers the
                # sync to the final micro-batch's backward. No-op outside DDP/FSDP.
                no_sync_ctx = (model.no_sync() if supports_no_sync and micro < args.accum - 1
                               else contextlib.nullcontext())
                with no_sync_ctx:
                    outputs = model(**batch)
                    loss = outputs.loss
                    if loss is None:
                        raise RuntimeError(
                            "model returned loss=None -- ensure labels are present "
                            "in the batch (the collator should always pass them)")
                    last_loss = loss
                    # Scale by 1/accum so summed micro-batch grads equal the
                    # average-loss gradient (standard accumulation semantics).
                    (loss / args.accum).backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            del outputs  # free the logits tensor (~1GB) before the next step
            loss = last_loss  # for the logging/eval code below, unchanged from pre-accum shape

            if profiler is not None:
                profiler.step()

            # Throughput / VRAM logging (rank 0). Computed cheaply every step
            # but only printed/written at the logging cadence below.
            step_tps = step_peak_vram_gb = step_tflops = step_ms = None
            if is_main and torch.cuda.is_available():
                step_ms = (time.time() - step_start) * 1000
                step_tps = step_tokens / max(step_ms / 1000, 1e-9)
                step_peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
                step_tflops = estimate_step_tflops(n_trainable, step_tokens, step_ms / 1000)

            # Only rank 0 logs to stdout / TB / runs eval — other ranks train silently.
            if it % 10 == 0 or it == args.iters:
                if is_main:
                    li = loss.item()
                    msg = f"[cpt] Iter {it}/{args.iters}: loss={li:.4f}  lr={lr:.2e}"
                    if step_tps is not None:
                        msg += (f"  {step_tps:,.0f} tok/s  step={step_ms:.0f}ms"
                                f"  vram={step_peak_vram_gb:.1f}GB  {step_tflops:.0f} TFLOPs/s")
                    print(msg)
            if tb_writer is not None and is_main and it % 10 == 0:
                tb_writer.add_scalar("train/loss", li, it)
                tb_writer.add_scalar("train/lr", lr, it)
                if step_tps is not None:
                    tb_writer.add_scalar("train/tokens_per_s", step_tps, it)
                    tb_writer.add_scalar("train/step_ms", step_ms, it)
                    tb_writer.add_scalar("train/peak_vram_gb", step_peak_vram_gb, it)
                    tb_writer.add_scalar("train/achieved_tflops", step_tflops, it)
    
            # Broadcast _SHOULD_STOP from rank 0 to all ranks BEFORE the
            # eval/checkpoint decisions below. Under --fsdp, the checkpoint
            # gather (get_full_state_dict etc.) is a COLLECTIVE — if SIGTERM
            # hits only rank 0 (common: torchrun doesn't forward signals to
            # all ranks), rank 0 would enter the all-gather alone and deadlock
            # while other ranks proceed to the next step's gradient sync.
            # Broadcasting the flag ensures all ranks make the same checkpoint
            # decision on the same step. (For --ddp this is harmless — DDP's
            # checkpoint path is rank-0-only with no collective, so the extra
            # broadcast is a no-op on correctness.)
            if is_distributed:
                stop_tensor = torch.tensor(
                    [1 if _SHOULD_STOP else 0], device=device)
                torch.distributed.broadcast(stop_tensor, src=0)
                if stop_tensor.item():
                    _SHOULD_STOP = True

            # Held-out eval at eval_every intervals (rank 0 only — the eval forward
            # pass doesn't need gradient sync, so non-rank-0 GPUs idle during eval).
            # Under --ddp/--fsdp with eval_every != checkpoint_every, rank 0 can be
            # inside run_eval while other ranks reach the next step's backward sync
            # and block (or time out). Barrier here on eval steps so all ranks wait
            # for rank 0's eval to finish.
            #
            # NOTE: _SHOULD_STOP is intentionally NOT in the barrier condition (same
            # principle as the checkpoint barrier below). The flag is broadcast
            # above so all ranks see the same value, but an asymmetric barrier
            # (some ranks enter, others don't) would still deadlock.
            is_eval_step = (valid_rows is not None
                            and (it % eval_every == 0 or it == args.iters))
            if is_distributed and is_eval_step:
                torch.distributed.barrier()
            do_eval = is_eval_step and is_main
            if is_eval_step:
                # For DDP: unwrap to avoid the buffer-broadcast collective (see
                # unwrap_ddp()'s docstring). For FSDP: summon_full_params gathers
                # sharded params to rank 0 (a COLLECTIVE — all ranks must enter
                # the context, even though only rank 0 runs the eval forward).
                # Without summon_full_params, model.module(**batch) would run on
                # each rank's local shard -> garbage valid_loss.
                #
                # ALL ranks enter this block (is_eval_step, not do_eval) so the
                # collective completes; only rank 0 runs run_eval inside it.
                is_fsdp = type(model).__name__ == "FullyShardedDataParallel"
                if is_fsdp:
                    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                    eval_ctx = FSDP.summon_full_params(
                        model, writeback=False, rank0_only=True, with_grads=False)
                else:
                    eval_ctx = contextlib.nullcontext()
                with eval_ctx:
                    if do_eval:
                        vloss = run_eval(_fsdp_unwrap(unwrap_ddp(model)), valid_rows,
                                         builder, tokenizer, args.max_seq_len,
                                         args.batch, device, args.pack,
                                         tokenizer.pad_token_id)
                        last_valid_loss = vloss
                        print(f"[cpt] eval step {it}: valid_loss={vloss:.4f}")
                        if tb_writer is not None:
                            tb_writer.add_scalar("eval/valid_loss", vloss, it)
    
            # DDP/FSDP barrier: ensure all ranks are at the same step before
            # checkpointing.
            # NOTE: _SHOULD_STOP is intentionally NOT in this condition. The flag is
            # set per-rank by an async signal handler, so including it in a collective
            # barrier would deadlock if ranks set the flag at different steps (some
            # enter barrier() while others proceed to the next step's all-reduce).
            # Instead, ranks only barrier on scheduled checkpoint/final-iter steps,
            # where they already synchronize via the backward all-reduce. A SIGTERM
            # on ANY step sets _SHOULD_STOP, which triggers checkpoint + exit on
            # the CURRENT step (the checkpoint condition below includes _SHOULD_STOP
            # independently of the barrier), so no next-step all-reduce is needed.
            if is_distributed and (it % args.checkpoint_every == 0 or it == args.iters):
                torch.distributed.barrier()

            # Only rank 0 writes checkpoints — DDP syncs gradients (so the model
            # state is identical across ranks); FSDP shards params, so rank 0
            # gathers the full state dict before writing (other ranks get empty
            # dicts, avoiding N copies of a multi-GB checkpoint on disk).
            #
            # IMPORTANT: get_full_state_dict() is a COLLECTIVE operation under
            # FSDP (it all-gathers sharded params to rank 0). Even with
            # rank0_only=True, ALL ranks must enter the context and call
            # state_dict() — non-rank-0 ranks participate in the all-gather
            # but receive {}. So the gather MUST happen OUTSIDE the `if is_main`
            # gate; only the disk write is rank-0-only.
            is_checkpoint_step = (it % args.checkpoint_every == 0
                                  or it == args.iters or _SHOULD_STOP)
            # FSDP: all ranks gather the full model + optimizer state (both are
            # collectives — all ranks must participate even though only rank 0
            # gets the result).
            full_sd = None
            full_optim_sd = None
            is_fsdp = type(model).__name__ == "FullyShardedDataParallel"
            if is_checkpoint_step and is_fsdp:
                full_sd = get_full_state_dict(model)
                full_optim_sd = get_full_optim_state_dict(model, optimizer)
            if is_main and is_checkpoint_step:
                ckpt_extra = {"valid_loss": last_valid_loss} if last_valid_loss is not None else None
                # For FSDP: use the gathered full state dicts via shims that
                # override save_pretrained() / state_dict().
                # For DDP/single-GPU: unwrap_ddp gives the raw model/optimizer.
                if full_sd is not None:
                    save_model = _StateDictModel(_fsdp_unwrap(model), full_sd)
                    save_optimizer = _StateDictOptimizer(optimizer, full_optim_sd)
                else:
                    save_model = unwrap_ddp(model)
                    save_optimizer = optimizer
                # When resuming, the latest modeling_custom.py lives in the checkpoint
                # dir (the user may have updated it there). For a fresh run, copy it
                # from the original --model path.
                custom_code_src = save_dir if resumed else Path(args.model)
                if args.async_checkpoint:
                    async_ckpt.save(save_model, save_optimizer, it, save_dir, tokenizer,
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
                    atomic_save_checkpoint(save_model, save_optimizer, it, save_dir, tokenizer,
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
        if is_distributed and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()



if __name__ == "__main__":
    main()
