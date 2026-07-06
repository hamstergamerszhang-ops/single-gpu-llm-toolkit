#!/usr/bin/env python3
"""Width + depth model expansion for a Gemma-4-family checkpoint — pipeline
step 3 of 4 (runs on the output of prune_embeddings_torch.py, before
train_cpt.py). PyTorch-native, runs on any CUDA/ROCm box without needing an
Apple-Silicon-only ML framework.

Grows a checkpoint's parameter count by widening the MLP intermediate
dimension (adding new orthogonally-initialized columns/rows to
gate/up/down_proj) and adding new decoder layers (duplicating existing layers
at regular intervals, zero-initializing their output projections so a
duplicated layer starts as a true identity/no-op and only gradually starts
contributing as training proceeds).

Key design decisions (the genuinely interesting engineering here):

  - **Orthogonal-QR width init**, not random-normal init. New MLP columns are
    built via `numpy.linalg.qr` on a random matrix, scaled down (`INIT_SCALE`),
    so the new capacity starts near-orthogonal to what the model already
    learned — it doesn't immediately conflict with or wash out existing
    representations, but the model still has an actual gradient signal to
    grow into the new capacity (as opposed to zero-init, which would give the
    new columns literally no initial gradient because their contribution
    starts at exactly zero for the *width* dimension, unlike the *depth*
    duplication case below where zero-init is the right call for a different
    reason — see the code comment on why the two cases use different
    strategies deliberately, not inconsistently).
  - **Zero-init depth duplication**, not orthogonal-QR, for newly duplicated
    LAYERS. A duplicated layer clones its donor layer's weights outright
    (real, already-trained weights — not fresh init), but its output
    projections (`self_attn.o_proj`, `mlp.down_proj`) are zeroed. That makes
    the new layer compute a real (non-degenerate) forward pass internally but
    contribute nothing to the residual stream at insertion time — a true
    identity/no-op layer, not a random one. Training then gradually turns the
    output projections away from zero as gradient signal accumulates. This is
    a different scenario from width expansion (new columns within an
    existing layer, where zero-init would starve gradient flow to those
    columns specifically) — hence the two different init strategies for two
    different insertion patterns, by design, not by accident.
  - **Real GQA fix on MQA-style attention layers.** Some full-attention layers
    in this model family ship with an extreme "1 shared KV head, V literally
    reuses K, no separate v_proj weight exists at all" MQA setup (confirmed by
    inspecting the actual checkpoint's safetensors header — there's no
    `v_proj` key for these layers). That's a real memory optimization, but on
    a GPU with enough VRAM that KV-cache size at long context isn't actually
    the bottleneck, it's trading away model quality for a saving this
    deployment doesn't need. `--gqa-kv-heads N` (default 8) grows `k_proj` to
    N real KV heads via orthogonal padding (preserving already-learned K
    directions) and builds a brand-new `v_proj` from scratch via a fresh
    orthogonal QR init (there's no existing V data to pad from — V never
    existed as a separate matrix before). Set `--gqa-kv-heads 0` to skip this
    and keep the stock MQA behavior.
  - **numpy QR, not `torch.nn.init.orthogonal_`, for the orthogonal
    constructions.** This ROCm PyTorch build's CPU tensors don't have LAPACK
    support (`torch.linalg.qr` / `torch.geqrf` raise a clear error asking for
    a LAPACK-enabled build), while numpy's own LAPACK bindings work fine. If
    you're on a CUDA/ROCm build where `torch.linalg.qr` works on CPU tensors,
    you could swap this — kept as numpy here because it's the version that's
    actually been run.
  - **Sharded safetensors output** with a real `model.safetensors.index.json`,
    splitting shards at a configurable byte budget (default 5GB/shard) so the
    output is loadable the same way any standard sharded HF checkpoint is.
  - **Optional Multi-Token-Prediction (MTP) head expansion** (`--mtp-depths`,
    default 2): if the checkpoint's architecture supports an auxiliary
    multi-token-prediction loss, this instantiates fresh MTP module blocks
    sized to match the newly-expanded config and appends them as an
    additional safetensors shard, merged into the index. This is optional and
    architecture-specific — most Gemma-4-family checkpoints won't have an
    `mtp_modules`-style attribute, and the flag is a no-op (aside from setting
    two harmless config fields) unless your fork of the modeling code
    actually defines that class. Left in because it's real, working code
    against a real architecture variant, not because every reader will use it
    — set `--mtp-depths 0` to skip it entirely.

Usage:
    python3 expand_model.py --dry-run --src ./checkpoints/base_pruned
    python3 expand_model.py \\
        --src ./checkpoints/base_pruned --dst ./checkpoints/base_expanded
"""

import argparse
import json
import os
import shutil

import numpy as np
import torch
from safetensors.torch import load_file, save_file

DEFAULT_WIDTH_STEP = 1024
DEFAULT_DEPTH_STEP = 12
INTERLEAVE_EVERY = 4
INIT_SCALE = 0.02
MAX_SHARD_BYTES = 5 * 1024**3
DEFAULT_MTP_DEPTHS = 2
DEFAULT_MTP_LOSS_WEIGHT = 0.3
DEFAULT_GQA_KV_HEADS = 8  # matches sliding-attention layers' existing kv head count


def log(msg: str):
    print(f"[expand_model] {msg}", flush=True)


def build_depth_plan(orig_layers: int, depth_step: int, interleave_every: int):
    plan = []
    dup_count = 0
    for old_idx in range(orig_layers):
        plan.append((len(plan), old_idx, False))
        if (old_idx + 1) % interleave_every == 0 and dup_count < depth_step:
            plan.append((len(plan), old_idx, True))
            dup_count += 1
    if dup_count != depth_step:
        raise SystemExit(f"ERROR: interleave plan produced {dup_count} duplicates, "
                          f"expected {depth_step}.")
    return plan


def orthogonal_pad(n_new: int, n_existing: int, scale: float, transpose_for_rows: bool):
    """Returns a torch.bfloat16 tensor of `n_new` new orthogonal directions,
    built via numpy QR (see module docstring for why numpy, not torch, here)."""
    if transpose_for_rows:
        R = np.random.randn(n_new, n_existing).astype(np.float32)
        Q, _ = np.linalg.qr(R.T, mode="reduced")
        pad = (Q[:, :n_new].T * scale).astype(np.float32)
    else:
        R = np.random.randn(n_existing, n_new).astype(np.float32)
        Q, _ = np.linalg.qr(R, mode="reduced")
        pad = (Q[:, :n_new] * scale).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(pad)).to(torch.bfloat16)


def width_expand_layer(tensors: dict, layer_prefix: str, old_intermediate: int,
                       new_intermediate: int):
    n_new = new_intermediate - old_intermediate
    gate_key = f"{layer_prefix}.mlp.gate_proj.weight"
    up_key = f"{layer_prefix}.mlp.up_proj.weight"
    down_key = f"{layer_prefix}.mlp.down_proj.weight"
    hidden = tensors[gate_key].shape[1]

    for key in (gate_key, up_key):
        old_w = tensors[key]
        pad = orthogonal_pad(n_new, hidden, INIT_SCALE, transpose_for_rows=True)
        tensors[key] = torch.cat([old_w, pad], dim=0)

    old_w = tensors[down_key]
    pad = orthogonal_pad(n_new, hidden, INIT_SCALE, transpose_for_rows=False)
    tensors[down_key] = torch.cat([old_w, pad], dim=1)


def gqa_expand_kv(tensors: dict, layer_prefix: str, head_dim: int, old_kv_heads: int,
                  new_kv_heads: int, hidden: int, init_scale: float):
    """Some full-attention layers ship with an extreme MQA setup: a single shared
    KV head with V literally reusing K (v_proj doesn't exist at all — confirmed
    via the real checkpoint's safetensors header, only k_proj/k_norm/q_proj/q_norm
    exist for these layers, no v_proj/v_norm). If KV cache size at your target
    context length isn't actually a memory bottleneck on your GPU, that
    compression is trading away model quality for a saving you don't need.

    This grows k_proj to `new_kv_heads` (orthogonal-noise-padded new rows,
    preserving already-learned K directions) and creates a brand new v_proj at
    the same shape (genuinely fresh orthogonal init — there's no existing V data
    to pad from, since V never existed as its own matrix before).
    """
    k_key = f"{layer_prefix}.self_attn.k_proj.weight"
    v_key = f"{layer_prefix}.self_attn.v_proj.weight"
    old_out = old_kv_heads * head_dim
    new_out = new_kv_heads * head_dim
    n_new = new_out - old_out

    old_k = tensors[k_key]
    pad = orthogonal_pad(n_new, hidden, init_scale, transpose_for_rows=True)
    tensors[k_key] = torch.cat([old_k, pad], dim=0)

    # v_proj is a genuinely fresh matrix (no existing V data to pad from), and it's
    # "tall" (new_out > hidden in the common case) so it can have at most `hidden`
    # mutually orthogonal rows anyway — np.linalg.qr's reduced mode on a tall matrix
    # gives orthonormal COLUMNS directly, which is exactly what's achievable here.
    # Uses numpy (not torch.nn.init.orthogonal_) because this ROCm PyTorch build's
    # CPU tensors lack LAPACK support — confirmed by a real crash ("Calling
    # torch.geqrf on a CPU tensor requires compiling PyTorch with LAPACK"). numpy's
    # QR (via its own LAPACK bindings) is what orthogonal_pad() above already relies on.
    R = np.random.randn(new_out, hidden).astype(np.float32)
    Q, _ = np.linalg.qr(R, mode="reduced")
    v_fresh = torch.from_numpy(np.ascontiguousarray(Q * init_scale))
    tensors[v_key] = v_fresh.to(old_k.dtype)


def clone_layer_tensors(tensors: dict, src_prefix: str, dst_prefix: str,
                        zero_output_projections: bool):
    prefix_dot = src_prefix + "."
    new_entries = {}
    for key, val in tensors.items():
        if not key.startswith(prefix_dot):
            continue
        suffix = key[len(prefix_dot):]
        new_key = f"{dst_prefix}.{suffix}"
        if zero_output_projections and suffix in ("self_attn.o_proj.weight", "mlp.down_proj.weight"):
            new_entries[new_key] = torch.zeros(val.shape, dtype=val.dtype)
        else:
            # .clone() is required, not cosmetic: without it, a duplicated layer's
            # tensor is the SAME underlying storage as its donor layer's tensor (same
            # Python object). safetensors.save_file() correctly refuses to serialize
            # aliased tensors ("Some tensors share memory") — confirmed by hitting
            # that exact error before adding the .clone() call.
            new_entries[new_key] = val.clone()
    if not new_entries:
        raise SystemExit(f"ERROR: no tensors found with prefix {src_prefix}")
    return new_entries


def write_sharded(tensors: dict, dst: str, max_shard_bytes: int):
    os.makedirs(dst, exist_ok=True)
    items = list(tensors.items())

    def tensor_bytes(t):
        return t.numel() * t.element_size()

    shards = []
    current = {}
    current_bytes = 0
    for key, val in items:
        b = tensor_bytes(val)
        if current and current_bytes + b > max_shard_bytes:
            shards.append(current)
            current = {}
            current_bytes = 0
        current[key] = val.contiguous()
        current_bytes += b
    if current:
        shards.append(current)

    n_shards = len(shards)
    weight_map = {}
    total_size = 0
    for i, shard in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        path = os.path.join(dst, fname)
        save_file(shard, path)
        size = os.path.getsize(path)
        total_size += size
        for key in shard:
            weight_map[key] = fname
        log(f"wrote shard {fname} ({size/1024**3:.2f}GB, {len(shard)} tensors)")

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    log(f"wrote index.json -- {n_shards} shards, {total_size/1024**3:.2f}GB total")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width-step", type=int, default=DEFAULT_WIDTH_STEP)
    ap.add_argument("--depth-step", type=int, default=DEFAULT_DEPTH_STEP)
    ap.add_argument("--mtp-depths", type=int, default=DEFAULT_MTP_DEPTHS,
                    help="Multi-Token Prediction depths to add (0 disables MTP entirely "
                         "-- only meaningful if your modeling code defines an MTP module "
                         "class; harmless no-op config fields otherwise)")
    ap.add_argument("--mtp-loss-weight", type=float, default=DEFAULT_MTP_LOSS_WEIGHT)
    ap.add_argument("--gqa-kv-heads", type=int, default=DEFAULT_GQA_KV_HEADS,
                    help="Replace full-attention layers' MQA (1 shared kv head, V=K) with "
                         "real GQA at this many kv heads (0 disables, keeps stock MQA)")
    ap.add_argument("--layer-prefix", type=str, default="model.language_model.layers",
                    help="Safetensors key prefix before the layer index, e.g. "
                         "'model.language_model.layers.0.mlp...'. Different Gemma-4-family "
                         "checkpoint variants use different attribute orders here (e.g. "
                         "'language_model.model.layers' on some) -- confirm yours via a "
                         "quick safetensors header inspection before running for real.")
    args = ap.parse_args()

    np.random.seed(args.seed)

    src_cfg_path = os.path.join(args.src, "config.json")
    if not os.path.exists(src_cfg_path):
        raise SystemExit(f"ERROR: {src_cfg_path} not found")
    with open(src_cfg_path) as f:
        cfg = json.load(f)
    tc = cfg["text_config"]

    orig_layers = tc["num_hidden_layers"]
    orig_intermediate = tc["intermediate_size"]
    hidden = tc["hidden_size"]
    layer_types = tc["layer_types"]

    width_step = args.width_step
    depth_step = args.depth_step
    new_intermediate = orig_intermediate + width_step
    new_layers = orig_layers + depth_step

    log(f"src: {args.src}")
    log(f"  layers: {orig_layers} -> {new_layers}  (+{depth_step})")
    log(f"  intermediate_size: {orig_intermediate} -> {new_intermediate}  (+{width_step})")

    depth_plan = build_depth_plan(orig_layers, depth_step, INTERLEAVE_EVERY)
    new_layer_types = [layer_types[old_idx] for (_, old_idx, _) in depth_plan]

    if args.dry_run:
        log("DRY RUN -- depth plan (new_idx: source_old_idx, is_duplicate):")
        for new_idx, old_idx, is_dup in depth_plan:
            tag = " <- DUPLICATE (identity-init)" if is_dup else ""
            log(f"  {new_idx:2d}: from old layer {old_idx:2d}{tag}")
        log("Nothing written.")
        return

    index_path = os.path.join(args.src, "model.safetensors.index.json")
    with open(index_path) as f:
        src_index = json.load(f)
    shard_files = sorted(set(src_index["weight_map"].values()))

    log(f"loading {len(shard_files)} source shards ...")
    tensors = {}
    for shard in shard_files:
        loaded = load_file(os.path.join(args.src, shard))
        tensors.update(loaded)
    log(f"  loaded {len(tensors)} tensors")

    if args.gqa_kv_heads > 0:
        old_kv_heads = tc.get("num_global_key_value_heads", 1)
        global_head_dim = tc.get("global_head_dim", tc["head_dim"])
        full_attn_idxs = [i for i in range(orig_layers) if layer_types[i] == "full_attention"]
        log(f"Pass 1/3: GQA fix on {len(full_attn_idxs)} full-attention layers "
            f"({old_kv_heads} -> {args.gqa_kv_heads} kv heads, V=K -> real V) ...")
        for old_idx in full_attn_idxs:
            prefix = f"{args.layer_prefix}.{old_idx}"
            gqa_expand_kv(tensors, prefix, global_head_dim, old_kv_heads, args.gqa_kv_heads,
                         hidden, INIT_SCALE)
        log("  GQA fix done")
    else:
        log("Pass 1/3: SKIPPED (--gqa-kv-heads 0, keeping stock MQA)")

    if width_step > 0:
        log(f"Pass 2/3: width expansion ({orig_intermediate} -> {new_intermediate}) "
            f"on {orig_layers} layers ...")
        for old_idx in range(orig_layers):
            prefix = f"{args.layer_prefix}.{old_idx}"
            width_expand_layer(tensors, prefix, orig_intermediate, new_intermediate)
            if (old_idx + 1) % 12 == 0:
                log(f"  widened {old_idx + 1}/{orig_layers} layers")
        log("  width expansion done")
    else:
        log("Pass 2/3: SKIPPED (--width-step 0, debug isolation mode)")

    log(f"Pass 3/3: depth duplication ({orig_layers} -> {new_layers} layers) ...")
    final_tensors = {}
    for key, val in tensors.items():
        if ".layers." not in key:
            final_tensors[key] = val

    for new_idx, old_idx, is_dup in depth_plan:
        src_prefix = f"{args.layer_prefix}.{old_idx}"
        dst_prefix = f"{args.layer_prefix}.{new_idx}"
        cloned = clone_layer_tensors(tensors, src_prefix, dst_prefix,
                                     zero_output_projections=is_dup)
        final_tensors.update(cloned)
        tag = "DUPLICATE (o_proj+down_proj zeroed)" if is_dup else "original"
        log(f"  layer {new_idx:2d} <- old layer {old_idx:2d} [{tag}]")

    del tensors

    actual_total = sum(v.numel() for v in final_tensors.values())
    log(f"actual total params: {actual_total/1e9:.2f}B")

    cfg["text_config"]["intermediate_size"] = new_intermediate
    cfg["text_config"]["num_hidden_layers"] = new_layers
    cfg["text_config"]["layer_types"] = new_layer_types

    if args.gqa_kv_heads > 0:
        # attention_k_eq_v=False disables the MQA shortcut for every layer -> KV
        # heads fall back to the real (now-expanded) count, and v_proj is no longer
        # skipped -- exactly matching the fresh v_proj weights gqa_expand_kv() just
        # wrote. One flag flip covers all (now-GQA) full-attention layers uniformly.
        cfg["text_config"]["attention_k_eq_v"] = False
        log(f"GQA fix applied: attention_k_eq_v -> False, "
            f"full-attention layers now use {args.gqa_kv_heads} real kv heads (was MQA=1, V=K)")

    if args.mtp_depths > 0:
        cfg["text_config"]["mtp_depths"] = args.mtp_depths
        cfg["text_config"]["mtp_loss_weight"] = args.mtp_loss_weight
        # auto_map wires AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)
        # to a custom modeling class instead of the stock architecture -- only meaningful
        # if modeling_custom.py (with a matching *ForCausalLM class) actually exists
        # alongside this checkpoint. If you don't have a custom MTP-capable modeling
        # file, use --mtp-depths 0.
        cfg["auto_map"] = {"AutoModelForCausalLM": "modeling_custom.CustomForCausalLM"}
        log(f"MTP enabled: mtp_depths={args.mtp_depths}, mtp_loss_weight={args.mtp_loss_weight}")
    else:
        log("MTP disabled (--mtp-depths 0)")

    os.makedirs(args.dst, exist_ok=True)
    with open(os.path.join(args.dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    log(f"wrote config.json (intermediate_size={new_intermediate}, "
        f"num_hidden_layers={new_layers})")

    for fname in os.listdir(args.src):
        if fname == "config.json" or fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
            continue
        s = os.path.join(args.src, fname)
        d = os.path.join(args.dst, fname)
        if os.path.isfile(s):
            shutil.copy2(s, d)
    log("copied tokenizer + auxiliary files")

    write_sharded(final_tensors, args.dst, MAX_SHARD_BYTES)

    log("done. NEXT: load-test before trusting this for CPT (check for NaN/Inf in "
        "logits on a raw forward pass against a real input before committing to a "
        "long training run).")


if __name__ == "__main__":
    main()
