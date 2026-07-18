#!/usr/bin/env python3
"""PyTorch/ROCm port of an MLX embedding-slicing script — runs on the output
of prune_vocab.py, before expand_model.py. For CUDA/ROCm boxes without MLX
(Apple Silicon only).

Same logic as an MLX original built first for local Apple Silicon experiments,
ported to run on a single AMD MI300X (ROCm) box that can't run MLX. Uses
safetensors.torch instead of mx.load/mx.save_safetensors; PyTorch has native
bfloat16 support so the dtype handling carries over directly.

Usage:
    python3 prune_embeddings_torch.py \\
        --src ./checkpoints/base_12b \\
        --dst ./checkpoints/base_12b_pruned

The embedding tensor's key defaults to the Gemma-4-family layout this was
built and run against, but is a plain CLI flag (--embed-key), not a
hardcoded assumption -- point it at whatever your own checkpoint's
safetensors header actually calls the embedding weight (e.g. plain
`model.embed_tokens.weight` on many non-Gemma architectures) and the tensor
surgery below is otherwise architecture-agnostic: it only ever reads one
named tensor out of the state dict, slices its rows, and writes it back.
Configurable is not the same claim as verified -- this has only actually
been run against the Gemma-4-family key below.
"""

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

# Default matches the Gemma-4-family checkpoints this script has actually been
# run against. Override with --embed-key for a different model family's key
# layout -- see module docstring.
DEFAULT_EMBED_KEY = "model.language_model.embed_tokens.weight"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--embed-key", type=str, default=DEFAULT_EMBED_KEY,
                    help="Safetensors key for the embedding weight tensor to slice. "
                         f"Default ({DEFAULT_EMBED_KEY!r}) matches the Gemma-4-family "
                         "layout this script was built and run against -- for another "
                         "model family, check its safetensors header (e.g. via "
                         "safetensors.safe_open(...).keys()) and pass the real key here.")
    args = ap.parse_args()
    src, dst = args.src, args.dst
    embed_key = args.embed_key

    remap_path = os.path.join(dst, "_old_to_new_ids.json")
    if not os.path.exists(remap_path):
        raise SystemExit(f"ERROR: {remap_path} not found — run prune_vocab.py against "
                          f"this --dst first, it produces the id remap this script needs.")
    with open(remap_path) as f:
        old_to_new = {int(k): int(v) for k, v in json.load(f).items()}

    keep_old_ids = sorted(old_to_new.keys())
    new_vocab_size = len(keep_old_ids)
    expected_new_ids = sorted(old_to_new.values())
    if expected_new_ids != list(range(new_vocab_size)):
        raise SystemExit("ERROR: old_to_new id remap is not contiguous 0..N-1 — "
                          "aborting, slicing logic assumes it is.")

    # A checkpoint this size can come down from HF as EITHER a sharded
    # multi-file layout (index.json + model-NNNNN-of-MMMMM.safetensors) or a
    # single unsharded model.safetensors with no index at all -- some public
    # Gemma-4-family checkpoints download as the latter (confirmed against a
    # real downloaded checkpoint, not assumed). Build a synthetic single-shard
    # index in that case so the rest of this script can rely on one
    # consistent format.
    index_path = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
    else:
        # No index -- synthesize one for the single-file checkpoint using the
        # shared helper (was duplicated inline here + in expand_model.py +
        # mtp_head.py; now one canonical implementation in expand_model.py).
        from expand_model import synthesize_single_shard_index
        index = synthesize_single_shard_index(src)
        print(f"[prune_embed] no index.json found -- synthesized one for the single-file checkpoint")

    embed_shard = index["weight_map"][embed_key]
    all_shards = sorted(set(index["weight_map"].values()))

    print(f"[prune_embed] embed tensor lives in shard: {embed_shard}")
    print(f"[prune_embed] old vocab={len(old_to_new):,} new vocab={new_vocab_size:,}")

    os.makedirs(dst, exist_ok=True)
    new_total_size = 0
    new_param_delta = 0
    keep_idx = torch.tensor(keep_old_ids, dtype=torch.long)

    for shard in all_shards:
        src_path = os.path.join(src, shard)
        dst_path = os.path.join(dst, shard)

        if shard != embed_shard:
            shutil.copy2(src_path, dst_path)
            new_total_size += os.path.getsize(dst_path)
            print(f"[prune_embed] {shard}: copied unchanged")
            continue

        tensors_in = load_file(src_path)
        tensors_out = {}
        for key, val in tensors_in.items():
            if key == embed_key:
                sliced = val[keep_idx, :].contiguous()
                tensors_out[key] = sliced
                removed_rows = val.shape[0] - sliced.shape[0]
                new_param_delta -= removed_rows * val.shape[1]
                print(f"[prune_embed] {embed_key}: {tuple(val.shape)} -> {tuple(sliced.shape)}")
            else:
                tensors_out[key] = val

        save_file(tensors_out, dst_path)
        new_total_size += os.path.getsize(dst_path)
        print(f"[prune_embed] {shard}: rewrote with sliced embedding")

    index["metadata"] = index.get("metadata", {})
    index["metadata"]["total_size"] = new_total_size
    old_total_params = index["metadata"].get("total_parameters")
    if old_total_params is not None:
        index["metadata"]["total_parameters"] = old_total_params + new_param_delta
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"[prune_embed] wrote index.json  total_size={new_total_size/1024**3:.2f}GB  "
          f"param_delta={new_param_delta:,}")
    print("[prune_embed] done. Next: load-test the result before trusting it for CPT.")


if __name__ == "__main__":
    main()
