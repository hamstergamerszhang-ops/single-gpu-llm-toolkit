#!/usr/bin/env python3
"""Export a checkpoint to a single consolidated `model.safetensors` file.

Useful when a training run produces sharded checkpoints and you want a single
file for downstream tools that don't read `model.safetensors.index.json`.

Also copies every non-weight file from the source checkpoint (config.json,
tokenizer files, chat_template.jinja, generation_config.json, etc.) into the
output directory, matching mtp_head.py's copy behavior -- without this, the
output directory isn't loadable via `from_pretrained` on its own (it would
have weights but no config/tokenizer).
"""

import argparse
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="Source checkpoint directory.")
    ap.add_argument("--dst", required=True, help="Output directory.")
    ap.add_argument("--max-shard-bytes", type=int, default=None,
                    help="If set, shard output so no file exceeds this many bytes.")
    args = ap.parse_args()

    from safetensors.torch import load_file, save_file

    index_path = os.path.join(args.src, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
    elif os.path.exists(os.path.join(args.src, "model.safetensors")):
        shard_files = ["model.safetensors"]
    else:
        raise SystemExit(f"No safetensors found in {args.src}")

    tensors = {}
    for shard in shard_files:
        tensors.update(load_file(os.path.join(args.src, shard)))

    os.makedirs(args.dst, exist_ok=True)

    # Copy every non-weight file from src into dst (config.json, tokenizer
    # files, chat_template.jinja, generation_config.json, etc.) -- same
    # approach as mtp_head.py: copy everything EXCEPT the weight shards and
    # the old index (those are rewritten below with the consolidated set).
    # Without this, dst has weights but no config/tokenizer and isn't
    # loadable via from_pretrained on its own.
    shard_set = set(shard_files)
    copied = 0
    for fname in os.listdir(args.src):
        if fname in shard_set or fname == "model.safetensors.index.json":
            continue
        s = os.path.join(args.src, fname)
        d = os.path.join(args.dst, fname)
        if os.path.isfile(s):
            shutil.copy2(s, d)
            copied += 1
    print(f"[export_safetensors] copied {copied} non-weight file(s) "
          f"(config/tokenizer/etc.) from {args.src} to {args.dst}")

    if args.max_shard_bytes:
        from expand_model import write_sharded
        write_sharded(tensors, args.dst, args.max_shard_bytes, log_prefix="export_safetensors")
    else:
        save_file(tensors, os.path.join(args.dst, "model.safetensors"))
        print(f"[export_safetensors] wrote {len(tensors)} tensors to {args.dst}/model.safetensors")


if __name__ == "__main__":
    main()
