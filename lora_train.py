#!/usr/bin/env python3
"""LoRA fine-tuning, via HuggingFace's real `peft` library -- offered as an
option, NOT the recommended path in this repo.

Read that twice before reaching for this file. Every real run this repo's
own engineering narrative is built on -- the README's tips, the OOM war
stories, the batch/seqlen tradeoffs -- comes from `train_cpt.py` doing
full-parameter fine-tuning. This script exists because some developers
specifically want LoRA for its memory and storage tradeoffs (a much smaller
optimizer footprint, adapter checkpoints in the tens of MB instead of tens
of GB), not because it's a better default. It generally adapts a model less
than full fine-tuning does, for the same data and step count -- that's the
real cost of only updating a small low-rank slice of the weight space
instead of every parameter. If you can afford full-parameter training on
your hardware (this repo's whole point is that a single 80GB+ AMD GPU
usually can), `train_cpt.py` / `train_sft.py` is still the path this repo
actually recommends and has actually run.

Uses `peft.LoraConfig` + `peft.get_peft_model` -- HuggingFace's own,
widely-used LoRA implementation -- rather than hand-rolled low-rank matrix
math. That's a deliberate choice: `peft` is the implementation most
developers already expect and trust, and reimplementing LoRA's forward/
backward math here would just be a second, less-tested copy of something
that already has a large real-world test surface. This script's job is
wiring `peft` into this repo's existing data/tokenization/training-loop
conventions, not reimplementing what `peft` already does correctly.

What's genuinely reused from `train_cpt.py`, not duplicated: `load_jsonl`,
`build_sft_example` / `build_cpt_example` (chat-template tokenization with
assistant-turn masking), `collate`, `pack_examples`, `run_eval`,
`lr_at_step`, and `_apply_flash_attn`. Same "no duplicated logic" rule this
repo has followed since the standalone-utilities extraction pass. What's
NOT reused, because it doesn't apply here: `apply_window_freeze` /
`find_decoder_layers` (LoRA already restricts trainable parameters to the
adapter, so there's no separate freeze-window concept), the atomic full-
checkpoint rename dance in `train_cpt.py` (adapters are small enough that
`peft`'s own `save_pretrained` -- itself writing to a fresh directory --
is a reasonable checkpoint unit on its own), DDP / fp8 / torch.compile
(LoRA's whole premise is a lighter-weight path; if you need multi-GPU
full-parameter training at scale, that's `train_cpt.py --ddp`, not this).

--target-modules defaults to the same submodule key suffixes this repo's
other tools already target (`expand_model.py`, `mtp_head.py`):
q_proj/k_proj/v_proj/o_proj + gate_proj/up_proj/down_proj. Checked the same
way the tensor-suffix generalization work in expand_model.py was checked --
against the installed `transformers` library's own modeling source, not
assumed -- and confirmed real for Llama, Mistral, Qwen2, Qwen3, and every
Gemma generation (see expand_model.py's docstring for the full verified
list, including the genuine exceptions: GPT-2/Phi/Phi-3/Falcon/MPT/BLOOM
use different, often-fused naming and would need a different
--target-modules value, not a code change, since peft's target_modules
match is name-based and doesn't care about architecture beyond that).

Usage:
    python3 lora_train.py \\
        --model ./checkpoints/base_pruned --data ./data/data_sft_1 \\
        --save ./checkpoints/model_lora_1 \\
        --iters 2000 --batch 4 --lr 1e-4 --lora-r 16 --lora-alpha 32

    # Merge the adapter into the base model and save a full standalone
    # checkpoint (bigger on disk, but loadable with plain
    # AutoModelForCausalLM.from_pretrained(), no peft import required):
    python3 lora_train.py --model ./checkpoints/base_pruned --data ./data/data_sft_1 \\
        --save ./checkpoints/model_lora_1 --merge-and-save

Self-test (no GPU required -- builds a tiny real Llama-architecture model
via transformers, wraps it with a real peft.LoraConfig, and runs a real
forward+backward+save+reload+merge cycle against it):
    python3 lora_train.py --selftest
"""

import argparse
import json
import os
import sys
from pathlib import Path

from train_cpt import (
    build_cpt_example,
    build_sft_example,
    collate,
    load_jsonl,
    lr_at_step,
    pack_examples,
    run_eval,
)

DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def log(msg: str):
    print(f"[lora] {msg}", flush=True)


def build_lora_model(base_model, r: int, alpha: int, dropout: float,
                     target_modules: list, bias: str = "none"):
    """Wraps `base_model` with a real peft.LoraConfig / get_peft_model call.
    Returns the wrapped PeftModel. Raises ImportError with an actionable
    message if peft isn't installed -- this is the one hard dependency this
    script adds beyond what train_cpt.py already requires."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise ImportError(
            "peft is required for lora_train.py (pip install peft -- see "
            "requirements.txt). It is NOT required for train_cpt.py / "
            "train_sft.py, which do full-parameter fine-tuning instead."
        )
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias=bias,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(base_model, lora_cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", default=False)
    ap.add_argument("--model", help="HF-format model dir or repo id to fine-tune.")
    ap.add_argument("--data", help="Dir containing train.jsonl, optionally valid.jsonl. "
                                    "Or a single .jsonl file (train only).")
    ap.add_argument("--save", help="Output directory for the LoRA adapter (and the "
                                     "merged model too, if --merge-and-save).")
    ap.add_argument("--cpt", action="store_true", default=False,
                    help="Raw-text mode (no prompt masking), same semantics as "
                         "train_cpt.py's --cpt. Default is SFT (chat-template "
                         "tokenization with assistant-turn-only loss masking).")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="LoRA is typically trained at a higher LR than full "
                         "fine-tuning (1e-4-ish vs train_cpt.py's 8e-7 default) "
                         "since only a small low-rank slice of weights is "
                         "updated -- this default reflects that, not a typo.")
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    ap.add_argument("--lora-alpha", type=int, default=32, help="LoRA scaling factor.")
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-bias", type=str, default="none",
                    choices=["none", "all", "lora_only"],
                    help="Whether to also train bias terms (peft.LoraConfig's own "
                         "'bias' argument, passed through unchanged).")
    ap.add_argument("--target-modules", type=str, nargs="+", default=DEFAULT_TARGET_MODULES,
                    help="Submodule name suffixes peft attaches LoRA adapters to. "
                         "Defaults to this repo's usual q/k/v/o_proj + gate/up/"
                         "down_proj set (verified against the installed transformers "
                         "library's modeling source for Llama/Mistral/Qwen2/Qwen3/"
                         "Gemma -- see expand_model.py's docstring). Override for an "
                         "architecture with different submodule names (peft matches "
                         "by name suffix, so this is a plain string list, not a path).")
    ap.add_argument("--pack", action="store_true", default=False)
    ap.add_argument("--eval-every", type=int, default=None,
                    help="Defaults to checking every 10th of --iters. --no-eval disables.")
    ap.add_argument("--no-eval", action="store_true", default=False)
    ap.add_argument("--checkpoint-every", type=int, default=200,
                    help="Save the adapter (not the full model) every N steps.")
    ap.add_argument("--merge-and-save", action="store_true", default=False,
                    help="After training, also merge the adapter into the base "
                         "model's weights and save a full standalone checkpoint "
                         "under <save>/merged -- loadable with plain "
                         "AutoModelForCausalLM.from_pretrained(), no peft import "
                         "needed. Off by default (adapter-only save is the point "
                         "of using LoRA in the first place -- a merged save is "
                         "opt-in for when you specifically want a drop-in "
                         "standalone checkpoint).")
    ap.add_argument("--flash-attn", action="store_true", default=False,
                    help="Same as train_cpt.py's --flash-attn -- falls back to "
                         "standard attention with a warning if flash-attn isn't "
                         "installed.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gfx-override", type=str, default=None)
    args = ap.parse_args()

    if args.selftest:
        _self_test()
        return

    if not (args.model and args.data and args.save):
        ap.error("--model, --data, and --save are required, unless --selftest is given.")

    from rocm_env import setup_rocm_env
    setup_rocm_env(override=args.gfx_override)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        print("[lora] WARNING: no CUDA/ROCm device visible -- only use this path "
              "for a tiny --iters smoke test.", file=sys.stderr)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    log(f"loading base model from {args.model} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.flash_attn:
        from train_cpt import _apply_flash_attn
        _apply_flash_attn(base_model)

    model = build_lora_model(base_model, r=args.lora_r, alpha=args.lora_alpha,
                             dropout=args.lora_dropout, target_modules=args.target_modules,
                             bias=args.lora_bias)
    model.print_trainable_parameters()

    data_path = Path(args.data)
    if data_path.is_dir():
        train_rows = load_jsonl(data_path / "train.jsonl")
        valid_path = data_path / "valid.jsonl"
        valid_rows = load_jsonl(valid_path) if valid_path.exists() else []
    else:
        train_rows = load_jsonl(data_path)
        valid_rows = []
    log(f"loaded {len(train_rows)} train rows, {len(valid_rows)} valid rows")

    builder = build_cpt_example if args.cpt else build_sft_example
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0
    )
    eval_every = args.eval_every or max(args.iters // 10, 1)

    model.train()
    rng_idx = 0
    for step in range(1, args.iters + 1):
        chunk = []
        for _ in range(args.batch):
            chunk.append(train_rows[rng_idx % len(train_rows)])
            rng_idx += 1
        examples = [builder(r, tokenizer, args.max_seq_len) for r in chunk]
        if args.pack:
            examples = pack_examples(examples, args.max_seq_len)
        if not examples:
            continue
        batch_data = collate(examples, tokenizer.pad_token_id)
        batch_data = {k: v.to(device) for k, v in batch_data.items()}

        lr = lr_at_step(step, args.iters, args.lr, args.warmup_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        outputs = model(**batch_data)
        loss = outputs.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == 1:
            log(f"step {step}/{args.iters}  loss={loss.item():.4f}  lr={lr:.2e}")

        if not args.no_eval and valid_rows and step % eval_every == 0:
            valid_loss = run_eval(model, valid_rows, builder, tokenizer, args.max_seq_len,
                                  args.batch, device, args.pack, tokenizer.pad_token_id)
            log(f"step {step}: valid_loss={valid_loss:.4f}")

        if step % args.checkpoint_every == 0 or step == args.iters:
            adapter_dir = save_dir / "adapter"
            model.save_pretrained(str(adapter_dir))
            log(f"step {step}: adapter saved -> {adapter_dir}")

    if args.merge_and_save:
        log("merging adapter into base model weights ...")
        merged = model.merge_and_unload()
        merged_dir = save_dir / "merged"
        merged.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        log(f"merged model saved -> {merged_dir} (loadable with plain "
            f"AutoModelForCausalLM.from_pretrained(), no peft import needed)")

    log("done. NEXT: load-test the adapter (or merged model) before trusting it -- "
        "same discipline as every other tool in this repo.")


def _self_test():
    import tempfile

    print("[selftest] lora_train: real peft.LoraConfig / get_peft_model wiring "
          "against a tiny real transformers model (no GPU required)")

    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    # A tiny real Llama-architecture model -- Llama is one of the architectures
    # expand_model.py's docstring confirms shares this repo's default
    # q/k/v/o_proj + gate/up/down_proj submodule naming (checked against the
    # installed transformers library's own modeling source), so this is a
    # genuine, representative wiring test, not a synthetic stand-in for a
    # naming convention that doesn't actually match anything real.
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, max_position_embeddings=64)
    base_model = LlamaForCausalLM(cfg)

    model = build_lora_model(base_model, r=4, alpha=8, dropout=0.0,
                             target_modules=DEFAULT_TARGET_MODULES)

    # Every trainable parameter must be a LoRA param, not a base-model weight --
    # this is the whole point of LoRA (only the adapter trains).
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert len(trainable) > 0, "no trainable parameters after wrapping with LoRA"
    assert all("lora_" in n for n in trainable), \
        f"non-LoRA parameter left trainable: {[n for n in trainable if 'lora_' not in n]}"
    print(f"  OK ({len(trainable)} LoRA parameter tensors trainable, "
          f"0 base-model weights trainable)")

    # Real forward + backward pass -- confirms the wrapped model actually
    # computes a loss and gradients land only on the adapter.
    input_ids = torch.randint(0, 64, (2, 8))
    labels = input_ids.clone()
    outputs = model(input_ids=input_ids, labels=labels)
    assert torch.isfinite(outputs.loss), f"non-finite loss: {outputs.loss}"
    outputs.loss.backward()
    grads = [n for n, p in model.named_parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all("lora_" in n for n in grads), \
        f"gradient landed on a non-LoRA parameter: {[n for n in grads if 'lora_' not in n]}"
    print(f"  OK (forward pass produces a finite loss; backward pass gradients "
          f"land only on {len(grads)} LoRA parameter tensors)")

    # Real save -> reload onto a fresh base model -> merge -> save, exactly the
    # path main() exercises with --merge-and-save.
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        adapter_dir = td / "adapter"
        model.save_pretrained(str(adapter_dir))
        assert (adapter_dir / "adapter_config.json").exists()
        assert (adapter_dir / "adapter_model.safetensors").exists()
        print("  OK (adapter saved: adapter_config.json + adapter_model.safetensors)")

        from peft import PeftModel
        fresh_base = LlamaForCausalLM(cfg)
        reloaded = PeftModel.from_pretrained(fresh_base, str(adapter_dir))
        # Reloaded adapter weights must match what was saved (round-trip, not
        # just "a file exists").
        orig_state = {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}
        reload_state = {k: v for k, v in reloaded.state_dict().items() if "lora_" in k}
        assert len(orig_state) == len(reload_state) and len(orig_state) > 0
        for k in orig_state:
            assert torch.equal(orig_state[k], reload_state[k]), f"adapter weight mismatch: {k}"
        print(f"  OK (reloaded adapter's {len(orig_state)} LoRA weight tensors match "
              f"the saved ones exactly)")

        merged = reloaded.merge_and_unload()
        merged_dir = td / "merged"
        merged.save_pretrained(str(merged_dir))
        assert (merged_dir / "config.json").exists()
        assert any(f.name.endswith(".safetensors") for f in merged_dir.iterdir())
        # A merged model must NOT have any lora_ keys left -- merge_and_unload()
        # folds the adapter into the base weights and drops the peft wrapper.
        merged_keys = list(merged.state_dict().keys())
        assert not any("lora_" in k for k in merged_keys), \
            "merged model still has lora_ keys -- merge_and_unload() didn't fold them in"
        print(f"  OK (merge_and_unload produces a plain model with 0 lora_ keys "
              f"remaining, saved successfully to disk)")

    # target_modules actually matches real submodule names on this model --
    # if peft.LoraConfig's target_modules string list didn't match anything,
    # get_peft_model would silently produce a model with ZERO adapters (a real,
    # observed peft footgun -- a typo'd target module name fails silently, not
    # loudly). Confirm every default target module name appears at least once.
    module_names = {n.rsplit(".", 1)[-1] for n, _ in base_model.named_modules()}
    for target in DEFAULT_TARGET_MODULES:
        assert target in module_names, \
            f"--target-modules default {target!r} does not match any real submodule " \
            f"name on this model -- the default would silently produce zero adapters"
    print(f"  OK (all {len(DEFAULT_TARGET_MODULES)} default --target-modules names "
          f"match real submodule names on a Llama-architecture model)")

    print("\n[selftest] All checks passed (no GPU required -- run a real --iters 5 "
          "smoke test against an actual checkpoint on real hardware before trusting "
          "this for a real training job).")


if __name__ == "__main__":
    main()
