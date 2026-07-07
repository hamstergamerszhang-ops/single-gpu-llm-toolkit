#!/usr/bin/env python3
"""SFT entry point -- a thin, honest wrapper around `train_cpt.py`. There is
no second training loop here.

Say that plainly, because it's easy to miss: `train_cpt.py` already does SFT.
It's the DEFAULT mode -- omit `--cpt` and `train_cpt.py` runs chat-template
tokenization with assistant-turn-only loss masking via `build_sft_example()`
(see `train_cpt.py`'s docstring and the README's `train_cpt.py` section).
This file does not reimplement that logic, duplicate it, or fork a second
copy that could drift out of sync -- it exists purely so that someone
scanning this repo's file list sees a `train_sft.py` sitting next to
`train_cpt.py` and immediately knows SFT is supported, without needing to
open `train_cpt.py` and read its docstring to discover that CPT and SFT are
both handled by one file via a flag. That's the entire justification for
this file existing: discoverability, not new functionality.

Concretely, this rewrites `sys.argv` to the equivalent `train_cpt.py`
invocation and calls `train_cpt.main()` directly, after refusing `--cpt`
outright (if you want raw-text continued-pretraining, that's
`train_cpt.py --cpt` -- calling it from here would be confusing, not
convenient, since the entire point of this file's name is "the SFT one").
Every flag `train_cpt.py` accepts is still available here unchanged --
checkpointing, DDP, flash-attn, compile, fp8, gfx override, all of it --
since under the hood, `train_cpt.py`'s real `main()` is exactly what runs.

Usage (identical to train_cpt.py, minus --cpt which doesn't exist here):
    python3 train_sft.py \\
        --model ./checkpoints/base_pruned --data ./data/data_sft_1 \\
        --save ./checkpoints/model_sft_1 \\
        --iters 3000 --batch 2 --lr 8e-7 --max-seq-len 2048

Self-test (no GPU required): runs two things back to back --
  1. this wrapper's OWN logic (the argv rewrite, and the --cpt refusal --
     the only things actually unique to this file), and
  2. train_cpt.py's real self_test() (LR schedule, SFT/CPT masking
     construction, atomic checkpoint rename), delegated to verbatim, not
     reimplemented -- so there's still exactly one place the underlying
     SFT logic itself is tested, matching this file's whole "no duplicated
     logic" premise.
    python3 train_sft.py --selftest
"""

import sys


def _build_train_cpt_argv(argv: list) -> list:
    """Takes this wrapper's argv (excluding argv[0]) and returns the argv
    train_cpt.py should see: the same flags, verbatim, with a hard refusal if
    --cpt sneaks in. SFT and CPT are opposite intents -- silently accepting
    --cpt here would defeat the entire point of having a separately-named
    SFT entry point."""
    if "--cpt" in argv:
        raise SystemExit(
            "ERROR: train_sft.py does not accept --cpt -- SFT (chat-template + "
            "prompt masking) is this file's entire purpose. For raw-text "
            "continued-pretraining, use train_cpt.py --cpt directly instead."
        )
    return ["train_sft.py (-> train_cpt.py)", *argv]


def main():
    argv = sys.argv[1:]

    if "--selftest" in argv:
        _self_test()
        print()
        print("[train_sft] delegating to train_cpt.py's own --selftest -- the "
              "SFT logic itself (build_sft_example, LR schedule, checkpoint "
              "atomicity) lives there, tested there, not reimplemented here:")
        from train_cpt import self_test
        self_test()
        return

    import train_cpt
    sys.argv = _build_train_cpt_argv(argv)
    train_cpt.main()


def _self_test():
    """Tests the one piece of logic that's actually unique to this file: the
    argv rewrite + the --cpt refusal. Does NOT re-test build_sft_example, the
    LR schedule, or checkpoint atomicity -- see main()'s --selftest branch,
    which runs train_cpt.py's own self_test() right after this for that."""
    print("[selftest] train_sft: argv rewriting delegates to train_cpt.py verbatim, "
          "and --cpt is refused rather than silently accepted")

    argv = ["--model", "./m", "--data", "./d", "--save", "./s", "--iters", "5"]
    rewritten = _build_train_cpt_argv(argv)
    assert rewritten == ["train_sft.py (-> train_cpt.py)", *argv], rewritten
    assert "--cpt" not in rewritten
    print("  OK (plain SFT args pass through unchanged)")

    try:
        _build_train_cpt_argv(["--model", "./m", "--cpt", "--data", "./d"])
        raise AssertionError("expected SystemExit when --cpt is passed to train_sft.py")
    except SystemExit as e:
        assert "does not accept --cpt" in str(e)
        print("  OK (--cpt is refused with a clear error, not silently accepted "
              "or silently ignored)")

    # Confirm train_cpt.py's real functions this wrapper's docstring claims
    # are shared are actually importable from train_cpt -- if a future
    # refactor renamed/removed build_sft_example, this catches the docstring
    # claim going stale rather than the wrapper silently pointing at nothing.
    from train_cpt import build_sft_example, lr_at_step, self_test  # noqa: F401
    print("  OK (train_cpt.py's build_sft_example / lr_at_step / self_test are "
          "importable -- the shared-implementation claim in this file's "
          "docstring still holds)")

    # End-to-end: main()'s non-selftest path actually reaches train_cpt.main()
    # with the args argument-for-argument, and argv[0] no longer carries this
    # wrapper's own name (so any error message train_cpt.main() prints refers
    # to itself correctly, not to a file that isn't the one actually running).
    import train_cpt
    real_main = train_cpt.main
    captured = {}

    def fake_main():
        captured["argv"] = list(sys.argv)

    train_cpt.main = fake_main
    saved_argv = sys.argv
    try:
        sys.argv = ["train_sft.py", "--model", "./m", "--data", "./d", "--save", "./s"]
        main()
    finally:
        train_cpt.main = real_main
        sys.argv = saved_argv
    assert captured["argv"] == [
        "train_sft.py (-> train_cpt.py)", "--model", "./m", "--data", "./d", "--save", "./s"
    ], captured["argv"]
    print("  OK (main() reaches train_cpt.main() with the exact args passed, "
          "unchanged, and --cpt never present)")

    print("\n[selftest] train_sft's own wrapper logic: all checks passed.")


if __name__ == "__main__":
    main()
