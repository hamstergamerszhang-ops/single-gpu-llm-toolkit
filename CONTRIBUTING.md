# Contributing

This repo holds standalone tools for adapting an LLM checkpoint on a single
AMD GPU under ROCm/PyTorch. Contributions are welcome, especially fixes for
real bugs you hit running these on actual hardware.

## Core principles

1. **No lies.** Docstrings and the README must describe what the code actually
   does, not what it aspires to do. If something is untested against a given
   config, say "configurable is not the same claim as verified" — don't claim
   it works. The worst sin here is a docstring that promises behavior the code
   doesn't implement.

2. **No mocks.** Every `--selftest` uses real torch tensors, real threads, real
   temp files — not fake implementations. If you add a feature, its self-test
   must exercise the real logic, not a stub.

3. **Works on every AMD device.** No device-name checks, no architecture
   branches, no hardcoded VRAM thresholds in logic. Model-family constants
   belong in CLI flags (defaulting to Gemma-4), not in code paths. If a card
   needs an env override (e.g. `HSA_OVERRIDE_GFX_VERSION`), handle it via
   `rocm_env.py`, not a hardcoded check.

## Before submitting

- Run the self-tests for modules that have them (all CPU-only, no GPU needed).
  Note: `prune_vocab.py`, `prune_embeddings_torch.py`, and `expand_model.py`
  are transformation tools that need real checkpoints, so they have no
  `--selftest` — their logic is covered by the pytest suite instead.
  ```
  for f in train_cpt.py async_checkpoint.py bnb_optimizer.py \
           local_cache_stream.py optimizer_compat_guard.py \
           rocm_env.py mtp_head.py lora_train.py train_sft.py; do
    python3 "$f" --selftest || { echo "FAILED: $f"; exit 1; }
  done
  ```
  `lora_train.py`'s self-test needs `peft` installed (see `requirements.txt`);
  `train_sft.py`'s delegates straight to `train_cpt.py`'s self-test after
  checking its own argv-rewrite logic, so it needs whatever `train_cpt.py`
  needs.
- Run the pytest suite: `pytest tests/ -v`
- If you change a docstring, verify the claim against the code it documents.

## Adding a new tool

- One file, independently runnable, with a `--selftest` (unless it's a
  transformation tool that requires a real checkpoint as input — in that case,
  cover its pure-logic functions in `tests/test_all.py` instead).
- Lazy-import torch/transformers inside functions so `--selftest` runs without
  GPU deps where possible.
- Document honestly: what it does, what it doesn't, what's Gemma-4-specific vs
  architecture-agnostic.
