#!/usr/bin/env python3
"""Append real Multi-Token-Prediction (MTP) modules to an expanded checkpoint.

Standalone tool — run it AFTER expand_model.py if you want an auxiliary
multi-token-prediction loss head. It is strictly opt-in: expand_model.py no
longer touches MTP at all (it used to write two config fields and claim, in its
docstring, to instantiate MTP modules — that claim was false, and the stub has
been removed). This file is the real replacement: it generates actual MTP
module weights, writes them as a safetensors shard, and merges them into the
checkpoint's index.

Architecture implemented: the DeepSeek-V3 MTP pattern. Per MTP depth i:

    enorm        (RMSNorm, weight init to 1.0)        — normalizes the incoming
                                                         hidden state h.
    eh_proj      (Linear, 2*hidden -> hidden, no bias) — projects the
                                                         concatenation of
                                                         [normed_h, token_emb]
                                                         down to hidden.
    block        (one full transformer block)          — CLONED from the last
                                                         decoder layer of the
                                                         base model (real
                                                         pretrained weights, not
                                                         fresh init), so the MTP
                                                         head starts from a
                                                         already-functional
                                                         transformer block instead
                                                         of random noise.
    lnorm        (RMSNorm, weight init to 1.0)        — normalizes the block's
                                                         output.

A single shared final `norm` (RMSNorm, 1.0) sits after the last depth. The
embedding table and lm_head are NOT duplicated here — DeepSeek-V3 ties them to
the base model's, and the user's modeling code is expected to reference the
existing `embed_tokens` / `lm_head` keys rather than MTP-local copies.

Tensor key naming convention (so your modeling_custom.py can match it):
    model.mtp_layers.{i}.enorm.weight            (hidden,)
    model.mtp_layers.{i}.eh_proj.weight          (hidden, 2*hidden)
    model.mtp_layers.{i}.block.<suffix>          (cloned from last layer)
    model.mtp_layers.{i}.lnorm.weight            (hidden,)
    model.mtp_layers.norm.weight                 (hidden,)   [shared final]

The block <suffix> matches whatever the last decoder layer uses (e.g.
self_attn.q_proj.weight, mlp.gate_proj.weight, ...) — controlled by
--layer-prefix, same default as expand_model.py.

IMPORTANT — what this file does and does NOT provide:
  - DOES: generate correct-shape, orthogonally-initialized MTP weights; clone a
    real transformer block per depth; write a merged safetensors shard + index;
    set config.json's mtp_depths / mtp_loss_weight / auto_map.
  - DOES NOT: supply the modeling Python code. For these weights to actually be
    USED at train/inference time, you need a modeling_custom.py (alongside the
    checkpoint) defining a CustomForCausalLM class whose forward instantiates
    MTP modules consuming the keys above. expand_model.py's docstring used to
    claim it instantiated MTP modules end-to-end — it never did. This file is
    the honest version of that intent: it produces correct weights + config, and
    is explicit that the modeling code is your responsibility.

Usage:
    python3 mtp_head.py --src ./checkpoints/base_expanded --dst ./checkpoints/base_mtp
    python3 mtp_head.py --src ./checkpoints/base_expanded --dst ./checkpoints/base_mtp --dry-run

Self-test (no GPU/model required — uses a tiny fake checkpoint in a tmp dir):
    python3 mtp_head.py --selftest
"""

import argparse
import json
import os
import shutil

# Reuse the orthogonal-pad init, sharded writer, and layer-clone helper from
# expand_model.py so the MTP head's init philosophy matches the rest of the
# pipeline (orthogonal-QR via numpy, same INIT_SCALE, same shard format).
from expand_model import (
    INIT_SCALE,
    clone_layer_tensors,
    log as _expand_model_log,
    orthogonal_pad,
    write_sharded,
)


def log(msg: str):
    """Thin wrapper around expand_model.log() with this file's OWN prefix.
    Fix for a real bug: mtp_head.py used to import expand_model.log()
    directly and call it unmodified, so every mtp_head.py --selftest / CLI
    run printed "[expand_model] ..." lines -- confusing when debugging
    mtp_head.py specifically, since the two are independently-runnable tools.
    Deliberately NOT a second copy of the print/flush logic -- just supplies
    the one thing that needs to differ (the prefix) to the shared helper."""
    _expand_model_log(msg, prefix="mtp_head")


DEFAULT_MTP_DEPTHS = 2
DEFAULT_MTP_LOSS_WEIGHT = 0.3
DEFAULT_MTP_PREFIX = "model.mtp_layers"
DEFAULT_LAYER_PREFIX = "model.language_model.layers"
DEFAULT_MAX_SHARD_BYTES = 5 * 1024**3


def _ones(hidden: int, dtype):
    """RMSNorm weight initialized to 1.0 — the standard 'identity' start for a
    learned scale, so the norm is a no-op until training moves it."""
    import torch
    return torch.ones(hidden, dtype=dtype)


def build_mtp_tensors(tensors: dict, cfg_text: dict, layer_prefix: str,
                      mtp_prefix: str, mtp_depths: int, init_scale: float):
    """Generate the full set of MTP-module tensors. Returns a dict of new keys
    -> tensors. Does NOT mutate `tensors` (the caller merges the result).

    Clones the LAST decoder layer per depth (real pretrained weights) for the
    `block` submodule, and orthogonally initializes eh_proj. enorm/lnorm/norm
    start at 1.0 (RMSNorm identity).
    """
    import torch

    hidden = cfg_text["hidden_size"]
    num_layers = cfg_text["num_hidden_layers"]
    last_layer_prefix = f"{layer_prefix}.{num_layers - 1}"

    # Detect the dtype of the existing weights so the new tensors match (bf16
    # for a typical ROCm training checkpoint, but don't hardcode it — read it
    # from an actual existing tensor).
    probe_key = f"{last_layer_prefix}.self_attn.q_proj.weight"
    if probe_key not in tensors:
        # Fall back to any tensor under the last layer, then any tensor at all.
        candidates = [k for k in tensors if k.startswith(last_layer_prefix + ".")]
        if not candidates:
            raise SystemExit(
                f"ERROR: no tensors found under last-layer prefix "
                f"{last_layer_prefix!r} — is --layer-prefix correct?"
            )
        probe_key = candidates[0]
    dtype = tensors[probe_key].dtype

    new_tensors = {}
    for i in range(mtp_depths):
        block_prefix = f"{mtp_prefix}.{i}.block"
        # Clone the last real decoder layer as this depth's transformer block.
        # zero_output_projections=False: these are standalone MTP modules, not
        # identity-inserted into the main residual stream, so we want the real
        # pretrained weights (not zeroed output projections like expand_model's
        # depth-duplication case, which has a different purpose).
        cloned = clone_layer_tensors(
            tensors, last_layer_prefix, block_prefix, zero_output_projections=False
        )
        new_tensors.update(cloned)

        # enorm (RMSNorm weight = 1.0).
        new_tensors[f"{mtp_prefix}.{i}.enorm.weight"] = _ones(hidden, dtype)

        # eh_proj: Linear(2*hidden -> hidden), weight shape (hidden, 2*hidden).
        # Orthogonally-pad-init the whole matrix (it's a fresh Linear with no
        # existing weights to preserve). CAUGHT BY THIS FILE'S OWN SELF-TEST,
        # twice: transpose_for_rows controls which arg becomes the ROW count
        # of the result -- True gives shape (n_new, n_existing), False gives
        # (n_existing, n_new) (verified against expand_model.py's own two
        # existing, working call sites: gate/up_proj growth uses
        # transpose_for_rows=True to get (n_new, hidden) for a dim=0 cat;
        # down_proj growth uses transpose_for_rows=False to get (hidden,
        # n_new) for a dim=1 cat). We want `hidden` as the ROW count here
        # (out_features of the Linear), so transpose_for_rows=True is the
        # correct flag -- n_new/n_existing stay as originally written
        # (hidden new orthonormal directions in a 2*hidden ambient space).
        # A first attempted fix swapped n_new/n_existing instead of the flag
        # -- that broke the underlying QR math (n_new > n_existing there
        # silently truncates the result, since reduced-mode QR caps returned
        # columns at min(rows, cols) with no error), caught by re-running the
        # self-test rather than trusting the first fix.
        eh = orthogonal_pad(
            n_new=hidden,
            n_existing=2 * hidden,
            scale=init_scale,
            transpose_for_rows=True,
        )
        # orthogonal_pad(n_new=hidden, n_existing=2*hidden, transpose_for_rows=True)
        # returns shape (n_new, n_existing) = (hidden, 2*hidden) -- exactly the
        # Linear weight shape we want.
        new_tensors[f"{mtp_prefix}.{i}.eh_proj.weight"] = eh.to(dtype)

        # lnorm (RMSNorm weight = 1.0).
        new_tensors[f"{mtp_prefix}.{i}.lnorm.weight"] = _ones(hidden, dtype)

    # Shared final norm after the last depth.
    new_tensors[f"{mtp_prefix}.norm.weight"] = _ones(hidden, dtype)

    return new_tensors


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", required=True, help="Expanded checkpoint dir (read-only).")
    ap.add_argument("--dst", required=True, help="Output dir (src copied + MTP shard added).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print planned tensor shapes + param count, write nothing.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mtp-depths", type=int, default=DEFAULT_MTP_DEPTHS,
                    help="Number of MTP modules to append (each clones one real layer).")
    ap.add_argument("--mtp-loss-weight", type=float, default=DEFAULT_MTP_LOSS_WEIGHT)
    ap.add_argument("--mtp-prefix", type=str, default=DEFAULT_MTP_PREFIX,
                    help="Safetensors key prefix for MTP modules "
                         f"(default {DEFAULT_MTP_PREFIX!r}).")
    ap.add_argument("--layer-prefix", type=str, default=DEFAULT_LAYER_PREFIX,
                    help="Decoder-layer key prefix in the base checkpoint, same "
                         "semantics as expand_model.py --layer-prefix.")
    ap.add_argument("--max-shard-bytes", type=int, default=DEFAULT_MAX_SHARD_BYTES)
    args = ap.parse_args()

    import numpy as np
    np.random.seed(args.seed)

    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        raise SystemExit("ERROR: --src and --dst must differ — refusing to overwrite src.")

    src_cfg_path = os.path.join(args.src, "config.json")
    if not os.path.exists(src_cfg_path):
        raise SystemExit(f"ERROR: {src_cfg_path} not found")
    with open(src_cfg_path) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    if "hidden_size" not in tc or "num_hidden_layers" not in tc:
        raise SystemExit("ERROR: config.json missing hidden_size / num_hidden_layers — "
                         "is this an expanded Gemma-family checkpoint?")

    hidden = tc["hidden_size"]
    num_layers = tc["num_hidden_layers"]

    log(f"src: {args.src}")
    log(f"  hidden_size={hidden}, num_hidden_layers={num_layers}")
    log(f"  mtp_depths={args.mtp_depths}, mtp_loss_weight={args.mtp_loss_weight}")
    log(f"  cloning last layer ({args.layer_prefix}.{num_layers - 1}) per MTP depth")

    # Dry run: don't load shards, just project shapes from config.
    if args.dry_run:
        log(f"DRY RUN — planned new tensors (per depth):")
        log(f"  {args.mtp_prefix}.{{i}}.enorm.weight: ({hidden},)")
        log(f"  {args.mtp_prefix}.{{i}}.eh_proj.weight: ({hidden}, {2*hidden})")
        log(f"  {args.mtp_prefix}.{{i}}.block.<suffix>: cloned from {args.layer_prefix}.{num_layers - 1}")
        log(f"  {args.mtp_prefix}.{{i}}.lnorm.weight: ({hidden},)")
        log(f"  {args.mtp_prefix}.norm.weight: ({hidden},)  [shared final]")
        log(f"  eh_proj params/depth: {hidden * 2 * hidden:,} "
            f"({hidden * 2 * hidden * 2 / 1024**3:.2f}GB at bf16)")
        log(f"  (block params depend on layer width — load for real to count exactly)")
        log("Nothing written.")
        return

    # Load all source shards.
    index_path = os.path.join(args.src, "model.safetensors.index.json")
    single_file = "model.safetensors"
    if os.path.exists(index_path):
        with open(index_path) as f:
            src_index = json.load(f)
        shard_files = sorted(set(src_index["weight_map"].values()))
    elif os.path.exists(os.path.join(args.src, single_file)):
        # Synthesize a single-shard index (same fallback as prune_embeddings_torch.py).
        shard_files = [single_file]
        src_index = {
            "metadata": {"total_size": os.path.getsize(os.path.join(args.src, single_file))},
            "weight_map": {},
        }
    else:
        raise SystemExit(f"ERROR: no model.safetensors.index.json and no {single_file} in {args.src}")

    from safetensors.torch import load_file
    log(f"loading {len(shard_files)} source shards ...")
    tensors = {}
    for shard in shard_files:
        tensors.update(load_file(os.path.join(args.src, shard)))
    log(f"  loaded {len(tensors)} tensors")

    new_tensors = build_mtp_tensors(
        tensors, tc, args.layer_prefix, args.mtp_prefix, args.mtp_depths, INIT_SCALE
    )
    merged = {**tensors, **new_tensors}
    del tensors

    added_params = sum(v.numel() for v in new_tensors.values())
    total_params = sum(v.numel() for v in merged.values())
    log(f"MTP added {added_params/1e9:.3f}B params across {len(new_tensors)} new tensors")
    log(f"checkpoint total: {total_params/1e9:.3f}B params")

    # Write output: copy everything from src EXCEPT the weight shards and old
    # index (those get rewritten by write_sharded below with the merged base+MTP
    # set — copying the old shards first would leave orphaned files not
    # referenced by the new index, wasting disk and confusing loaders).
    os.makedirs(args.dst, exist_ok=True)
    for fname in os.listdir(args.src):
        if fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
            continue
        s = os.path.join(args.src, fname)
        d = os.path.join(args.dst, fname)
        if os.path.isfile(s):
            shutil.copy2(s, d)
    log("copied source checkpoint files (weights + index rewritten below)")

    # Overwrite weights + index with the merged (base + MTP) set.
    write_sharded(merged, args.dst, args.max_shard_bytes, log_prefix="mtp_head")

    # Update config.json. Write through `tc` -- the SAME nested-or-flat
    # reference the read side already resolved above (`tc = cfg.get("text_config",
    # cfg)`) -- not cfg["text_config"] unconditionally. Regression fix: the old
    # code always did cfg.setdefault("text_config", {}) here regardless of
    # whether the source config was nested (Gemma-4) or flat (Llama/Mistral/
    # Qwen). On a flat config that meant `tc IS cfg` on the read side, but the
    # write side still created a brand-new, disconnected `text_config` dict
    # containing ONLY mtp_depths/mtp_loss_weight -- reproduced end-to-end in
    # review with a synthetic flat Llama-style checkpoint: hidden_size/
    # num_hidden_layers/etc. stayed at the top level while mtp_depths ended up
    # isolated one level down, a structurally inconsistent config.json that
    # doesn't crash but is silently wrong. Writing through `tc` keeps the
    # output in the same single namespace the rest of the config already
    # lives in, on both nested and flat inputs.
    tc["mtp_depths"] = args.mtp_depths
    tc["mtp_loss_weight"] = args.mtp_loss_weight
    # auto_map wires AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)
    # to a custom modeling class. REQUIRES modeling_custom.py (defining
    # CustomForCausalLM with MTP modules consuming the keys above) to exist
    # alongside this checkpoint — this script does NOT generate that file.
    cfg["auto_map"] = {"AutoModelForCausalLM": "modeling_custom.CustomForCausalLM"}
    with open(os.path.join(args.dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    log(f"wrote config.json (mtp_depths={args.mtp_depths}, "
        f"mtp_loss_weight={args.mtp_loss_weight}, auto_map set)")
    log(f"NOTE: weights + config written. For these to be USED, place a "
        f"modeling_custom.py defining CustomForCausalLM (with MTP modules) "
        f"alongside {args.dst} — this script does not generate that file.")
    log("done.")


def _self_test():
    import tempfile
    from pathlib import Path

    print("[selftest] mtp_head: generate MTP weights against a tiny fake checkpoint")
    import torch
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src"
        dst = td / "dst"
        src.mkdir()

        hidden = 16
        num_layers = 2
        layer_prefix = "model.language_model.layers"
        mtp_prefix = "model.mtp_layers"

        # Minimal config matching the fields build_mtp_tensors reads.
        cfg = {
            "text_config": {
                "hidden_size": hidden,
                "num_hidden_layers": num_layers,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "layer_types": ["full_attention", "full_attention"],
                "vocab_size": 100,
            },
            "model_type": "gemma4",
        }
        with open(src / "config.json", "w") as f:
            json.dump(cfg, f)

        # Two fake decoder layers, each with the suffix keys clone_layer_tensors
        # + the probe key in build_mtp_tensors expect.
        fake_tensors = {}
        for i in range(num_layers):
            p = f"{layer_prefix}.{i}"
            fake_tensors[f"{p}.self_attn.q_proj.weight"] = torch.randn(8, hidden).to(torch.bfloat16)
            fake_tensors[f"{p}.self_attn.k_proj.weight"] = torch.randn(8, hidden).to(torch.bfloat16)
            fake_tensors[f"{p}.self_attn.o_proj.weight"] = torch.randn(hidden, 8).to(torch.bfloat16)
            fake_tensors[f"{p}.mlp.gate_proj.weight"] = torch.randn(32, hidden).to(torch.bfloat16)
            fake_tensors[f"{p}.mlp.up_proj.weight"] = torch.randn(32, hidden).to(torch.bfloat16)
            fake_tensors[f"{p}.mlp.down_proj.weight"] = torch.randn(hidden, 32).to(torch.bfloat16)
        save_file(fake_tensors, src / "model.safetensors")
        with open(src / "model.safetensors.index.json", "w") as f:
            json.dump(
                {"metadata": {"total_size": 0},
                 "weight_map": {k: "model.safetensors" for k in fake_tensors}},
                f,
            )

        # Run build_mtp_tensors directly (avoids argparse).
        from expand_model import INIT_SCALE
        new_tensors = build_mtp_tensors(
            fake_tensors, cfg["text_config"], layer_prefix, mtp_prefix,
            mtp_depths=2, init_scale=INIT_SCALE,
        )

        # Shape assertions.
        assert new_tensors[f"{mtp_prefix}.0.enorm.weight"].shape == (hidden,)
        assert new_tensors[f"{mtp_prefix}.0.eh_proj.weight"].shape == (hidden, 2 * hidden), \
            new_tensors[f"{mtp_prefix}.0.eh_proj.weight"].shape
        assert new_tensors[f"{mtp_prefix}.0.lnorm.weight"].shape == (hidden,)
        assert new_tensors[f"{mtp_prefix}.1.enorm.weight"].shape == (hidden,)
        assert new_tensors[f"{mtp_prefix}.norm.weight"].shape == (hidden,)
        # Block tensors cloned from last layer (index num_layers-1) with the
        # .block. infix.
        assert f"{mtp_prefix}.0.block.self_attn.q_proj.weight" in new_tensors
        assert new_tensors[f"{mtp_prefix}.0.block.self_attn.q_proj.weight"].shape == (8, hidden)
        # RMSNorm weights start at 1.0 (identity).
        assert torch.all(new_tensors[f"{mtp_prefix}.0.enorm.weight"] == 1.0)
        assert torch.all(new_tensors[f"{mtp_prefix}.norm.weight"] == 1.0)
        # Cloned block weights are real (not zeroed) and detached from donor.
        donor = fake_tensors[f"{layer_prefix}.{num_layers - 1}.self_attn.q_proj.weight"]
        cloned = new_tensors[f"{mtp_prefix}.0.block.self_attn.q_proj.weight"]
        assert torch.equal(cloned, donor), "cloned block weight should equal donor"
        print("  OK (shapes, identity-norm init, real cloned block weights)")

        # End-to-end: run main() via argparse to exercise write + index merge.
        import sys
        sys.argv = [
            "mtp_head.py", "--src", str(src), "--dst", str(dst),
            "--mtp-depths", "2",
        ]
        main()

        # Index merged: MTP keys present in the written index.
        with open(dst / "model.safetensors.index.json") as f:
            out_index = json.load(f)
        assert f"{mtp_prefix}.0.eh_proj.weight" in out_index["weight_map"]
        assert f"{mtp_prefix}.norm.weight" in out_index["weight_map"]
        assert f"{mtp_prefix}.1.block.self_attn.q_proj.weight" in out_index["weight_map"]

        # Config updated.
        with open(dst / "config.json") as f:
            out_cfg = json.load(f)
        assert out_cfg["text_config"]["mtp_depths"] == 2
        assert "auto_map" in out_cfg
        assert out_cfg["auto_map"]["AutoModelForCausalLM"] == "modeling_custom.CustomForCausalLM"

        # Shards loadable and the MTP tensor is present on disk.
        from safetensors.torch import load_file as lf
        any_shard = list(set(out_index["weight_map"].values()))[0]
        loaded = lf(dst / any_shard)
        # At least one MTP key should land in whatever shard we picked if it's
        # the shard holding eh_proj; instead check across all shards.
        all_loaded = {}
        for shard in set(out_index["weight_map"].values()):
            all_loaded.update(lf(dst / shard))
        assert f"{mtp_prefix}.0.eh_proj.weight" in all_loaded
        assert all_loaded[f"{mtp_prefix}.0.eh_proj.weight"].shape == (hidden, 2 * hidden)
        print("  OK (index merged, config updated, shards loadable)")

    print("\n[selftest] All checks passed (no GPU required — run against a real "
          "expanded checkpoint before trusting this for a real training job).")


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", default=False)
    # Placeholder args so --selftest doesn't choke on argparse; main() re-parses.
    args, _ = ap.parse_known_args()
    if args.selftest:
        _self_test()
    else:
        main()


if __name__ == "__main__":
    main_cli()
