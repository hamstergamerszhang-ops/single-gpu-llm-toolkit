#!/usr/bin/env python3
"""Standalone guard against loading a checkpoint's optimizer state into a
DIFFERENT optimizer class than the one that saved it.

Extracted from train_cpt.py's resume block, where this used to be ~18 lines
inlined directly in main() right after loading training_state.pt. The
underlying problem is small to describe but expensive to get wrong: loading
one optimizer type's state_dict into a different optimizer class is not just
"ignored, harmless" -- e.g. loading fp32 AdamW's saved state into a
bitsandbytes Adam8bit instance (or the reverse, which is the more likely real
case here: you resumed WITHOUT bitsandbytes installed after training WITH it,
or vice versa, per bnb_optimizer.py's silent-fallback risk) has been observed
to silently accept the mismatched state and inflate GPU memory well beyond
what the current optimizer should need, OOMing on the very first forward
pass of the resumed run. That's a nasty failure mode: it doesn't crash on
load, it crashes a step later, minutes into a run you thought had resumed
cleanly.

The guard's policy: skip the optimizer-state load entirely (restart momentum
fresh, keep the step count) rather than risk it. Losing Adam's momentum
state on an optimizer-type switch is a known, bounded cost (it measurably
hurts quality for a while, but it's a quality regression, not a crash).
Silent memory corruption is not bounded -- it's an OOM you don't see coming.

Usage as a library:
    from optimizer_compat_guard import check_optimizer_compat
    saved_type = state.get("optimizer_type", "unknown")
    current_type = type(optimizer).__name__
    ok, message = check_optimizer_compat(saved_type, current_type)
    print(message)
    if ok:
        optimizer.load_state_dict(state["optimizer"])

Self-test (no GPU/model required -- pure string comparison logic):
    python3 optimizer_compat_guard.py --selftest
"""

import argparse


def check_optimizer_compat(saved_optimizer_type: str, current_optimizer_type: str):
    """Returns (safe_to_load: bool, message: str).

    safe_to_load is True only when the saved checkpoint's optimizer class
    name matches the current run's optimizer class name exactly. Callers
    should treat False as "skip optimizer_state_dict load, keep the step
    count, let momentum restart fresh" -- never as "load anyway."
    """
    if saved_optimizer_type == current_optimizer_type:
        return True, (
            f"[optimizer_compat_guard] optimizer type matches ({current_optimizer_type}) "
            f"-- safe to restore optimizer state (cold-restarting momentum measurably "
            f"hurts quality, so this matters)"
        )
    return False, (
        f"[optimizer_compat_guard] WARNING: checkpoint's optimizer was "
        f"{saved_optimizer_type}, this run is using {current_optimizer_type} -- skipping "
        f"optimizer state load (incompatible state_dicts, confirmed to risk OOM if "
        f"forced). Starting this optimizer's momentum fresh; step count still resumes."
    )


def _self_test():
    print("[selftest] matching optimizer types are declared safe to load")
    ok, msg = check_optimizer_compat("Adam8bit", "Adam8bit")
    assert ok is True
    assert "safe to restore" in msg
    print("  OK")

    print("[selftest] mismatched optimizer types are declared UNSAFE to load")
    ok, msg = check_optimizer_compat("AdamW", "Adam8bit")
    assert ok is False
    assert "skipping optimizer state load" in msg
    print("  OK")

    print("[selftest] the reverse mismatch direction is also caught (bnb -> torch)")
    ok, msg = check_optimizer_compat("Adam8bit", "AdamW")
    assert ok is False
    print("  OK")

    print("[selftest] 'unknown' (checkpoint predates optimizer_type being recorded) "
          "is treated as a mismatch against any real optimizer, not silently trusted")
    ok, msg = check_optimizer_compat("unknown", "AdamW")
    assert ok is False
    print("  OK")

    print("\n[selftest] All checks passed (pure logic, no GPU/model required).")


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
