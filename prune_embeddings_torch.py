#!/usr/bin/env python3
"""PyTorch/ROCm port of an MLX embedding-slicing script, for CUDA/ROCm boxes
without MLX (Apple Silicon only).

Same logic as an MLX original built first for local Apple Silicon experiments,
ported to run on a single AMD MI300X (ROCm) box that can't run MLX. Uses
safetensors.torch instead of mx.load/mx.save_safetensors; PyTorch has native
bfloat16 support so the dtype handling carries over directly.

Usage:
    python3 prune_embeddings_torch.py \\
        --src ./checkpoints/base_12b \\
        --dst ./checkpoints/base_12b_pruned
"""

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

EMBED_KEY = "model.language_model.embed_tokens.weight"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    src, dst = args.src, args.dst

    remap_path = os.path.join(dst, "_old_to_new_ids.json")
    if not os.path.exists(remap_path):
        raise SystemExit(f"ERROR: {remap_path} not found — run prune_vocab.py against "
                          f"this --dst first, it produces the id remap this script needs.")
    with open(remap_path) as f:
        old_to_new = {int(k): v for k, v in json.load(f).items()}

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
        single_file = "model.safetensors"
        if not os.path.exists(os.path.join(src, single_file)):
            raise SystemExit(f"ERROR: no model.safetensors.index.json AND no {single_file} in {src}")
        with open(os.path.join(src, single_file), "rb") as f:
            header_len = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(header_len))
        weight_map = {k: single_file for k in header if k != "__metadata__"}
        index = {"metadata": {"total_size": os.path.getsize(os.path.join(src, single_file))}, "weight_map": weight_map}
        print(f"[prune_embed] no index.json found -- synthesized one for the single-file checkpoint ({single_file})")

    embed_shard = index["weight_map"][EMBED_KEY]
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
            if key == EMBED_KEY:
                sliced = val[keep_idx, :].contiguous()
                tensors_out[key] = sliced
                removed_rows = val.shape[0] - sliced.shape[0]
                new_param_delta -= removed_rows * val.shape[1]
                print(f"[prune_embed] {EMBED_KEY}: {tuple(val.shape)} -> {tuple(sliced.shape)}")
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
