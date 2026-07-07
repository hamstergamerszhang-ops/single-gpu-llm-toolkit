#!/usr/bin/env python3
"""Standalone optimizer construction: bitsandbytes 8-bit Adam with a plain
torch.optim.AdamW fallback.

Extracted from train_cpt.py, where this used to be ~9 lines inlined directly
in main() (a try/except bitsandbytes import around the optimizer construction
call). Pulling it out into its own function means:
  - it's independently importable/testable without pulling in the whole
    training script or a real model,
  - the "which optimizer did I actually get" decision is made in ONE place
    instead of being implicitly whatever main() happened to do, so a second
    script (e.g. a future SFT-only entry point) can reuse the exact same
    fallback behavior instead of re-copy-pasting the try/except.

Why this matters on single-GPU hardware (the same reasoning train_cpt.py's
module docstring gives, kept in sync here): bitsandbytes' 8-bit Adam keeps
both first and second moment buffers at ~1 byte/param instead of fp32
AdamW's 4 bytes/param -- roughly a 4x reduction in optimizer-state memory.
On an 80GB+ single GPU that's the difference between comfortably fitting a
big model + optimizer state, and being one fallback away from an OOM dozens
of steps in (not at step 0 -- the fallback optimizer allocates lazily as
each param's state is first touched, so the failure shows up progressively,
which is exactly what makes it confusing to debug if you don't know to
check for a silently-missing bitsandbytes install first). See
README.md's bitsandbytes section for the observed-on-real-hardware version
of this warning.

Usage as a library:
    from bnb_optimizer import build_optimizer
    optimizer, kind = build_optimizer(trainable_params, lr=8e-7, weight_decay=0.01)
    print(f"[cpt] optimizer: {kind}")

Self-test (no GPU/model required -- constructs whichever optimizer is available
in the current environment against a tiny dummy tensor, and checks the fallback
path is reachable without bitsandbytes installed; re-run on a box WITH
bitsandbytes to confirm 'kind' comes back as bnb_adam8bit):
    python3 bnb_optimizer.py --selftest
"""

import argparse


def build_optimizer(trainable_params, lr: float, weight_decay: float = 0.01):
    """Try bitsandbytes 8-bit Adam first, fall back to torch.optim.AdamW.

    Returns (optimizer, optimizer_kind) where optimizer_kind is one of
    "bnb_adam8bit" or "torch_adamw" -- callers should log this (train_cpt.py
    prints it) since which optimizer got built silently determines both
    memory footprint and whether a checkpoint saved under one is safe to
    resume under the other (see optimizer_compat_guard.py in this repo).
    """
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.Adam8bit(trainable_params, lr=lr, weight_decay=weight_decay)
        return optimizer, "bnb_adam8bit"
    except Exception as e:
        # On ROCm, bitsandbytes can import successfully but fail at construction
        # time if the installed wheel lacks HIP kernels. Catch any construction
        # error, not just ImportError, and fall back cleanly.
        import torch
        # foreach=None (the default) lets PyTorch auto-select the vectorized
        # foreach path when the param dtypes support it (bf16 on CUDA/ROCm) and
        # fall back to the for-loop otherwise. Forcing foreach=True would raise
        # at step() on unsupported dtypes (e.g. fp8), with no fallback.
        optimizer = torch.optim.AdamW(
            trainable_params, lr=lr, weight_decay=weight_decay, foreach=None
        )
        print(f"[bnb_optimizer] WARNING: bitsandbytes 8-bit Adam unavailable "
              f"({type(e).__name__}: {e}) -- falling back to torch.optim.AdamW "
              f"(~4x more optimizer-state memory). This is a real, observed OOM "
              f"source on large models. Reinstall bitsandbytes on every fresh "
              f"container (ROCm builds must be HIP-aware). See README.md.")
        return optimizer, "torch_adamw"


def _self_test():
    print("[selftest] build_optimizer() falls back to AdamW when bitsandbytes is "
          "unavailable, and always returns a (optimizer, kind) pair")
    import torch

    dummy = [torch.nn.Parameter(torch.randn(4, 4))]
    optimizer, kind = build_optimizer(dummy, lr=1e-4)
    assert kind in ("bnb_adam8bit", "torch_adamw"), kind
    assert hasattr(optimizer, "step") and hasattr(optimizer, "zero_grad")
    print(f"  OK (built: {kind})")

    print("[selftest] optimizer actually accepts a step on the dummy param")
    loss = (dummy[0] ** 2).sum()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print("  OK")

    print("[selftest] lr/weight_decay are honored in param_groups")
    dummy2 = [torch.nn.Parameter(torch.randn(2, 2))]
    optimizer2, _kind2 = build_optimizer(dummy2, lr=5e-6, weight_decay=0.02)
    assert abs(optimizer2.param_groups[0]["lr"] - 5e-6) < 1e-12
    assert abs(optimizer2.param_groups[0]["weight_decay"] - 0.02) < 1e-12
    print("  OK")

    print("\n[selftest] All checks passed (no bitsandbytes install or GPU required for "
          "these -- on a box WITH bitsandbytes installed, re-run to confirm 'kind' comes "
          "back as bnb_adam8bit, not just the AdamW fallback).")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", default=False)
    args = ap.parse_args()
    if args.selftest:
        _self_test()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
