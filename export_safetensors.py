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
    ap.add_argument("--src", help="Source checkpoint directory.")
    ap.add_argument("--dst", help="Output directory.")
    ap.add_argument("--max-shard-bytes", type=int, default=None,
                    help="If set, shard output so no file exceeds this many bytes.")
    ap.add_argument("--selftest", action="store_true", default=False,
                    help="Run built-in self-test (no GPU required).")
    args = ap.parse_args()

    if args.selftest:
        _self_test()
        return

    if not args.src or not args.dst:
        ap.error("--src and --dst are required (unless --selftest).")

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


def _self_test():
    """Self-test: create a sharded checkpoint, consolidate it, verify the
    output is a single model.safetensors with all tensors + non-weight files
    copied (no GPU required)."""
    import tempfile
    import torch
    from safetensors.torch import save_file
    print("[selftest] export_safetensors: consolidate sharded checkpoint (no GPU)")

    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "src")
        os.makedirs(src)
        # Write a fake sharded checkpoint: 2 shards + index + config.json.
        tensors1 = {"layer.0.weight": torch.zeros(4), "layer.0.bias": torch.ones(4)}
        tensors2 = {"layer.1.weight": torch.zeros(2)}
        save_file(tensors1, os.path.join(src, "model-00001-of-00002.safetensors"))
        save_file(tensors2, os.path.join(src, "model-00002-of-00002.safetensors"))
        index = {
            "metadata": {"total_size": 40},
            "weight_map": {
                "layer.0.weight": "model-00001-of-00002.safetensors",
                "layer.0.bias": "model-00001-of-00002.safetensors",
                "layer.1.weight": "model-00002-of-00002.safetensors",
            },
        }
        with open(os.path.join(src, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f)
        with open(os.path.join(src, "config.json"), "w") as f:
            json.dump({"model_type": "test"}, f)

        # Run the consolidation via main() by simulating argv.
        import sys
        dst = os.path.join(td, "dst")
        old_argv = sys.argv
        sys.argv = ["export_safetensors.py", "--src", src, "--dst", dst]
        try:
            main()
        finally:
            sys.argv = old_argv

        # Verify: single model.safetensors with all 3 tensors.
        assert os.path.exists(os.path.join(dst, "model.safetensors"))
        assert not os.path.exists(os.path.join(dst, "model.safetensors.index.json"))
        # config.json must be copied (non-weight file).
        assert os.path.exists(os.path.join(dst, "config.json"))
        # All tensors present.
        from safetensors.torch import load_file
        loaded = load_file(os.path.join(dst, "model.safetensors"))
        assert set(loaded.keys()) == {"layer.0.weight", "layer.0.bias", "layer.1.weight"}, loaded.keys()
        print("  OK (3 tensors consolidated into single model.safetensors, config.json copied)")

    print("\n[selftest] All checks passed.")


if __name__ == "__main__":
    main()
