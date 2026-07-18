#!/usr/bin/env python3
"""Width + depth model expansion for a Gemma-4-family checkpoint — runs on the
output of prune_embeddings_torch.py, before optional mtp_head.py and
train_cpt.py. PyTorch-native, runs on any CUDA/ROCm box without needing an
Apple-Silicon-only ML framework.

Grows a checkpoint's parameter count by widening the MLP intermediate
dimension (adding new orthogonally-initialized columns/rows to
gate/up/down_proj) and adding new decoder layers (duplicating existing layers
at regular intervals, zero-initializing their output projections so a
duplicated layer starts as a true identity/no-op and only gradually starts
contributing as training proceeds).

Key design decisions (the genuinely interesting engineering here):

  - **Orthogonal-QR width init**, not random-normal init. New MLP columns are
    built via `numpy.linalg.qr` on a random matrix, scaled down (`INIT_SCALE`).
    The QR construction makes the new columns orthonormal *among themselves*
    (mutually orthogonal), NOT orthogonal to the model's existing weight columns
    — the QR runs on a fresh random matrix, with no reference to the existing
    weights. What actually limits disruption to existing representations is the
    small `INIT_SCALE` (0.02), not orthogonality to existing weights. The
    orthonormal-among-themselves property still helps: it means the new capacity
    isn't redundant within itself (no two new columns are near-identical), so
    the model has a real, non-degenerate gradient signal to grow into the new
    capacity (as opposed to zero-init, which would give the new columns literally
    no initial gradient because their contribution starts at exactly zero for the
    *width* dimension, unlike the *depth* duplication case below where zero-init
    is the right call for a different reason — see the code comment on why the
    two cases use different strategies deliberately, not inconsistently).
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
  - **Real GQA fix on MQA-style attention layers, with an actual detection
    step in front of it -- not a blind Gemma-4-only assumption anymore.**
    Some full-attention layers in Gemma-4-family checkpoints specifically
    ship with an extreme "1 shared KV head, V literally reuses K, no separate
    v_proj weight exists at all" MQA setup (confirmed by inspecting the
    actual checkpoint's safetensors header — there's no `v_proj` key for
    these layers). That's a real memory optimization, but on a GPU with
    enough VRAM that KV-cache size at long context isn't actually the
    bottleneck, it's trading away model quality for a saving this deployment
    doesn't need. `--gqa-kv-heads N` (default 8) grows `k_proj` to N real KV
    heads via orthogonal padding (preserving already-learned K directions)
    and builds a brand-new `v_proj` from scratch via a fresh orthogonal QR
    init (there's no existing V data to pad from — V never existed as a
    separate matrix before). Before touching any tensor, though,
    `detect_mqa_v_shares_k_layout()` actually checks the loaded checkpoint
    against the assumed layout: does a `v_proj` key exist for these layers
    (it shouldn't, if this optimization applies), and does `k_proj`'s real
    shape agree with what the config claims the kv-head count is? If either
    check fails — meaning this checkpoint doesn't have the MQA-with-no-v_proj
    layout at all, e.g. any standard GQA/MHA architecture that ships a real
    `v_proj` — the fix is skipped with a clear log message instead of running
    anyway and either crashing on a shape mismatch or silently overwriting a
    real, already-trained V matrix. `--force-gqa-fix` overrides the check for
    the rare case you've verified by hand that it's still correct. Set
    `--gqa-kv-heads 0` to skip this pass entirely and keep whatever attention
    layout the checkpoint already has.
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

MTP note: this script does NOT touch multi-token-prediction at all. An earlier
version of this docstring claimed it instantiated MTP modules and appended them
as a safetensors shard — that claim was false (the code only wrote two config
fields). The real MTP weight generation + shard/index merge now lives in its
own standalone tool, `mtp_head.py` — run that AFTER expand_model.py if you want
an MTP head. See mtp_head.py's docstring for the architecture it implements
(DeepSeek-V3 MTP pattern) and what it does vs. does not provide.

Usage:
    python3 expand_model.py --dry-run --src ./checkpoints/base_pruned
    python3 expand_model.py \\
        --src ./checkpoints/base_pruned --dst ./checkpoints/base_expanded

Configurability note: --width-step, --depth-step,
--gqa-kv-heads, and --layer-prefix were already CLI flags
before this pass (their module-level DEFAULT_* constants are just the
argparse defaults). --interleave-every and --max-shard-bytes are new in this
pass -- they used to be hardcoded module constants (INTERLEAVE_EVERY,
MAX_SHARD_BYTES) with no way to override them without editing source; now
they're flags too, same pattern as the others.

On tensor key SUFFIXES (gate/up/down_proj, self_attn.q/k/v/o_proj): these are
NOT parameterized here (only the prefix before the layer index is, via
--layer-prefix), but checked directly against the installed `transformers`
library's own modeling source (not assumed) -- Llama, Mistral, Qwen2, Qwen3,
and every Gemma generation all define these exact attribute names
(`self.gate_proj`/`self.up_proj`/`self.down_proj`, `self.q_proj`/`k_proj`/
`v_proj`/`o_proj`), because HF safetensors keys are just the module's
attribute path, and this naming traces back to the original Llama
implementation that most subsequent decoder-only architectures copied. So
this suffix assumption is honestly broader than "Gemma-4-specific" -- it's
"most Llama-derived decoder architectures," which is most of what's popular
right now. It is NOT universal, though, and here are the real, verified
exceptions: GPT-2 fuses QKV into one `c_attn` (a `Conv1D`, not `Linear`) and
uses `c_fc`/`c_proj` for the MLP; the original Phi (phi-1/phi-2) uses
`fc1`/`fc2` for its MLP instead of gate/up/down; Phi-3 fuses attention into
one `qkv_proj` and the MLP into one `gate_up_proj`; Falcon, MPT, and BLOOM
all fuse attention into a single `query_key_value` (or `Wqkv`) matrix and use
model-specific MLP names (`dense_h_to_4h`/`dense_4h_to_h` or
`up_proj`/`down_proj` with no `gate_proj` at all). Point this at one of those
and it will KeyError cleanly on a missing tensor key rather than silently
doing the wrong thing -- but it won't work without real code changes to
width_expand_layer() / gqa_expand_kv() / clone_layer_tensors() for those
architectures' actual submodule names.

This has only ever actually been run end-to-end against Gemma-4-family
checkpoints on a single MI300X; the flags plus the suffix research above make
it plausible (and, for the common Llama-style suffix case, probably correct)
to point elsewhere, but that's not the same as verified there.
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
DEFAULT_GQA_KV_HEADS = 8  # matches sliding-attention layers' existing kv head count


def log(msg: str, prefix: str = "expand_model"):
    """Prints a `[prefix] msg` line. `prefix` defaults to this module's own
    name so every existing `log(...)` call in this file is unaffected.
    mtp_head.py reuses this function (rather than duplicating a second
    logging helper) but needs its OWN prefix -- it was calling this log()
    unmodified and every mtp_head.py --selftest / CLI run printed
    "[expand_model] ..." lines, which is confusing when debugging mtp_head.py
    specifically. See mtp_head.log() below for the one-line wrapper that
    supplies prefix="mtp_head"."""
    print(f"[{prefix}] {msg}", flush=True)


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
    """Returns a float32 tensor of `n_new` new orthogonal directions, built via
    numpy QR (see module docstring for why numpy, not torch, here). Returned as
    float32 (NOT forced to bfloat16) so callers can cast to whatever dtype the
    checkpoint actually uses -- see width_expand_layer / gqa_expand_kv for the
    `.to(old_w.dtype)` cast applied at each torch.cat call site.

    QR on an (m, n) matrix with m <= n can only produce at most m orthogonal
    columns; requesting more than that silently truncated the pad before. We
    guard against that explicitly.
    """
    if n_new > n_existing:
        raise ValueError(
            f"orthogonal_pad cannot produce {n_new} orthogonal directions from a "
            f"{n_existing}-dimensional space (n_new must be <= n_existing). "
            f"transpose_for_rows={transpose_for_rows}"
        )
    # Use torch.linalg.qr on GPU when available (rocSOLVER-backed on ROCm) —
    # a 15B-width expansion took minutes on CPU (numpy single-threaded). Falls
    # back to numpy if no GPU or if torch.linalg.qr is unavailable.
    try:
        import torch
        if transpose_for_rows:
            R = torch.randn(n_new, n_existing, dtype=torch.float32, device="cuda")
            Q, _ = torch.linalg.qr(R.T, mode="reduced")
            pad = (Q[:, :n_new].T * scale).contiguous().to("cpu")
        else:
            R = torch.randn(n_existing, n_new, dtype=torch.float32, device="cuda")
            Q, _ = torch.linalg.qr(R, mode="reduced")
            pad = (Q[:, :n_new] * scale).contiguous().to("cpu")
        return pad
    except (RuntimeError, Exception):
        pass  # no GPU, or torch.linalg.qr unavailable — fall back to numpy
    if transpose_for_rows:
        R = np.random.randn(n_new, n_existing).astype(np.float32)
        Q, _ = np.linalg.qr(R.T, mode="reduced")
        pad = (Q[:, :n_new].T * scale).astype(np.float32)
    else:
        R = np.random.randn(n_existing, n_new).astype(np.float32)
        Q, _ = np.linalg.qr(R, mode="reduced")
        pad = (Q[:, :n_new] * scale).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(pad))


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
        tensors[key] = torch.cat([old_w, pad.to(old_w.dtype)], dim=0)

    old_w = tensors[down_key]
    pad = orthogonal_pad(n_new, hidden, INIT_SCALE, transpose_for_rows=False)
    tensors[down_key] = torch.cat([old_w, pad.to(old_w.dtype)], dim=1)


def detect_mqa_v_shares_k_layout(tensors: dict, full_attn_idxs: list, layer_prefix: str,
                                 head_dim: int, old_kv_heads: int) -> tuple:
    """Checks whether this checkpoint's full-attention layers actually match the
    specific MQA layout gqa_expand_kv() assumes ("1 shared KV head, V literally
    reuses K, no separate v_proj key exists at all"), instead of assuming every
    input matches it.

    Returns (matches: bool, reason: str). `matches` is True only if EVERY
    full-attention layer checked has:
      - no `{prefix}.self_attn.v_proj.weight` key in the loaded tensors, AND
      - a `{prefix}.self_attn.k_proj.weight` key whose output dim equals
        old_kv_heads * head_dim (i.e. the config's advertised kv-head count
        actually matches the tensor's real shape).

    If ANY checked layer has a real v_proj key, or the k_proj shape doesn't
    match what the config claims, this returns False with a reason string --
    the caller should skip the fix rather than apply gqa_expand_kv() blindly,
    since that function unconditionally overwrites v_proj (would silently
    clobber a real, already-trained V matrix on architectures that have one)
    and assumes old_kv_heads is the true current shape (would misshape the
    k_proj concatenation otherwise).

    This is a real safety check, not a formality: it's the difference between
    "only works on Gemma-4's exact MQA layout" and "auto-detects whether this
    specific optimization applies, and cleanly skips it otherwise."
    """
    if not full_attn_idxs:
        return False, "no full-attention layers found (layer_types has none marked 'full_attention')"

    for old_idx in full_attn_idxs:
        prefix = f"{layer_prefix}.{old_idx}"
        v_key = f"{prefix}.self_attn.v_proj.weight"
        k_key = f"{prefix}.self_attn.k_proj.weight"

        if v_key in tensors:
            return False, (f"layer {old_idx}: found a real {v_key!r} tensor -- this checkpoint "
                            f"already has a separate V projection, it doesn't use the "
                            f"'V literally reuses K, no v_proj at all' MQA layout this fix targets")

        if k_key not in tensors:
            return False, (f"layer {old_idx}: expected {k_key!r} not found in loaded tensors -- "
                            f"can't verify the kv-head layout, --layer-prefix may not match this "
                            f"checkpoint's actual key naming (see --layer-prefix help)")

        actual_k_out = tensors[k_key].shape[0]
        expected_k_out = old_kv_heads * head_dim
        if actual_k_out != expected_k_out:
            return False, (f"layer {old_idx}: {k_key!r} has output dim {actual_k_out}, but the "
                            f"config-derived old_kv_heads*head_dim ({old_kv_heads}*{head_dim}="
                            f"{expected_k_out}) doesn't match -- the config's kv-head count "
                            f"doesn't agree with the tensor's real shape, refusing to guess")

    return True, (f"confirmed: {len(full_attn_idxs)} full-attention layer(s) have no v_proj key "
                  f"and k_proj shape matches the config's kv-head count -- matches the assumed "
                  f"MQA layout")


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
    tensors[k_key] = torch.cat([old_k, pad.to(old_k.dtype)], dim=0)

    # v_proj is a genuinely fresh matrix (no existing V data to pad from).
    # We need shape (new_out, hidden) for Linear(hidden -> new_out).
    # QR of (new_out, hidden) in reduced mode gives Q of shape
    # (new_out, min(new_out, hidden)). When new_out >= hidden this is
    # (new_out, hidden) — correct. When new_out < hidden, Q is (new_out, new_out)
    # and we must pad with random columns to reach (new_out, hidden).
    # Uses numpy (not torch.nn.init.orthogonal_) because this ROCm PyTorch build's
    # CPU tensors lack LAPACK support — confirmed by a real crash ("Calling
    # torch.geqrf on a CPU tensor requires compiling PyTorch with LAPACK"). numpy's
    # QR (via its own LAPACK bindings) is what orthogonal_pad() above already relies on.
    R = np.random.randn(new_out, hidden).astype(np.float32)
    Q, _ = np.linalg.qr(R, mode="reduced")
    if Q.shape[1] < hidden:
        # new_out < hidden: pad with random columns to reach (new_out, hidden).
        pad = np.random.randn(new_out, hidden - Q.shape[1]).astype(np.float32) * init_scale
        Q = np.concatenate([Q * init_scale, pad], axis=1)
    else:
        Q = Q * init_scale
    v_fresh = torch.from_numpy(np.ascontiguousarray(Q))
    tensors[v_key] = v_fresh.to(old_k.dtype)
    assert tensors[v_key].shape == (new_out, hidden), \
        f"v_proj shape {tensors[v_key].shape} != expected ({new_out}, {hidden})"


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


def write_sharded(tensors: dict, dst: str, max_shard_bytes: int, log_prefix: str = None):
    """log_prefix lets a caller from a DIFFERENT tool (mtp_head.py reuses this
    function rather than duplicating a sharded-writer) get its own log lines
    instead of every "wrote shard ..." / "wrote index.json ..." line printing
    "[expand_model]" regardless of who actually called it. Default None means
    "use log()'s own default prefix" -- expand_model.py's own caller (below)
    is unaffected."""
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
        _log_kwargs = {"prefix": log_prefix} if log_prefix else {}
        log(f"wrote shard {fname} ({size/1024**3:.2f}GB, {len(shard)} tensors)", **_log_kwargs)

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    _log_kwargs = {"prefix": log_prefix} if log_prefix else {}
    log(f"wrote index.json -- {n_shards} shards, {total_size/1024**3:.2f}GB total", **_log_kwargs)


def synthesize_single_shard_index(src_dir: str) -> dict:
    """Synthesize a model.safetensors.index.json for a single-file checkpoint
    (one that ships as a lone model.safetensors with no index).

    Reads the 8-byte header length, parses the header JSON, and builds a
    weight_map pointing every tensor key at "model.safetensors". Returns the
    full index dict (metadata + weight_map).

    Extracted as a shared helper (was duplicated in expand_model.py,
    mtp_head.py, and prune_embeddings_torch.py). Callers that already have
    an index.json should use it directly; this is the fallback for the
    single-file case.
    """
    single_file = "model.safetensors"
    single_path = os.path.join(src_dir, single_file)
    if not os.path.exists(single_path):
        raise SystemExit(
            f"ERROR: no model.safetensors.index.json AND no {single_file} in {src_dir}")
    with open(single_path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
    weight_map = {k: single_file for k in header if k != "__metadata__"}
    return {
        "metadata": {"total_size": os.path.getsize(single_path)},
        "weight_map": weight_map,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width-step", type=int, default=DEFAULT_WIDTH_STEP)
    ap.add_argument("--depth-step", type=int, default=DEFAULT_DEPTH_STEP)
    ap.add_argument("--gqa-kv-heads", type=int, default=DEFAULT_GQA_KV_HEADS,
                    help="Replace full-attention layers' MQA (1 shared kv head, V=K) with "
                         "real GQA at this many kv heads (0 disables, keeps stock MQA). "
                         "Before applying, this checks whether the checkpoint's tensors "
                         "actually match the assumed layout (no v_proj key, k_proj shape "
                         "agrees with the config's kv-head count) and skips cleanly with a "
                         "warning if they don't -- see --force-gqa-fix to override that check.")
    ap.add_argument("--force-gqa-fix", action="store_true",
                    help="Apply the GQA fix even if the layout-detection check fails (i.e. the "
                         "checkpoint has a real v_proj already, or its k_proj shape doesn't match "
                         "the config's kv-head count). Only use this if you've independently "
                         "verified via a safetensors header inspection that the fix is still "
                         "correct for your checkpoint -- forcing past a failed check is very "
                         "likely to either crash on a shape mismatch or silently overwrite a real, "
                         "already-trained v_proj tensor.")
    ap.add_argument("--layer-prefix", type=str, default="model.language_model.layers",
                    help="Safetensors key prefix before the layer index, e.g. "
                         "'model.language_model.layers.0.mlp...'. Different Gemma-4-family "
                         "checkpoint variants use different attribute orders here (e.g. "
                         "'language_model.model.layers' on some) -- confirm yours via a "
                         "quick safetensors header inspection before running for real. "
                         "Submodule key SUFFIXES (mlp.gate_proj, self_attn.k_proj, etc.) "
                         "are NOT parameterized here -- only the prefix before the layer "
                         "index is. These suffixes are shared by most Llama-derived "
                         "decoder architectures (Llama, Mistral, Qwen2/3, Gemma) but NOT "
                         "by GPT-2/Phi/Phi-3/Falcon/MPT/BLOOM-style fused-QKV models. "
                         "See module docstring for the verified list.")
    ap.add_argument("--interleave-every", type=int, default=INTERLEAVE_EVERY,
                    help="Insert one duplicated layer after every N original layers "
                         "(subject to --depth-step's total cap). Was a hardcoded module "
                         "constant; now a flag with the same default.")
    ap.add_argument("--max-shard-bytes", type=int, default=MAX_SHARD_BYTES,
                    help="Byte budget per output safetensors shard (default 5GB). Was a "
                         "hardcoded module constant; now a flag with the same default.")
    args = ap.parse_args()

    np.random.seed(args.seed)

    src_cfg_path = os.path.join(args.src, "config.json")
    if not os.path.exists(src_cfg_path):
        raise SystemExit(f"ERROR: {src_cfg_path} not found")
    with open(src_cfg_path) as f:
        cfg = json.load(f)
    # Support both nested (Gemma-4: text_config) and flat (Llama/Mistral/Qwen)
    # config layouts. mtp_head.py uses the same .get() fallback pattern.
    tc = cfg.get("text_config", cfg)

    orig_layers = tc["num_hidden_layers"]
    orig_intermediate = tc["intermediate_size"]
    hidden = tc["hidden_size"]
    # layer_types is Gemma-4-specific (full_attention/sliding_attention). Most
    # architectures don't have it — default to all "full_attention" so the
    # depth-plan and GQA logic treats every layer uniformly.
    layer_types = tc.get("layer_types", ["full_attention"] * orig_layers)

    width_step = args.width_step
    depth_step = args.depth_step
    new_intermediate = orig_intermediate + width_step
    new_layers = orig_layers + depth_step

    log(f"src: {args.src}")
    log(f"  layers: {orig_layers} -> {new_layers}  (+{depth_step})")
    log(f"  intermediate_size: {orig_intermediate} -> {new_intermediate}  (+{width_step})")

    depth_plan = build_depth_plan(orig_layers, depth_step, args.interleave_every)
    new_layer_types = [layer_types[old_idx] for (_, old_idx, _) in depth_plan]

    if args.dry_run:
        log("DRY RUN -- depth plan (new_idx: source_old_idx, is_duplicate):")
        for new_idx, old_idx, is_dup in depth_plan:
            tag = " <- DUPLICATE (identity-init)" if is_dup else ""
            log(f"  {new_idx:2d}: from old layer {old_idx:2d}{tag}")
        log("Nothing written.")
        return

    index_path = os.path.join(args.src, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            src_index = json.load(f)
        shard_files = sorted(set(src_index["weight_map"].values()))
    else:
        # No index -- synthesize one for the single-file checkpoint.
        src_index = synthesize_single_shard_index(args.src)
        shard_files = ["model.safetensors"]
        log(f"no index.json found -- synthesized one for the single-file checkpoint")

    log(f"loading {len(shard_files)} source shards ...")
    tensors = {}
    for shard in shard_files:
        loaded = load_file(os.path.join(args.src, shard))
        tensors.update(loaded)
    log(f"  loaded {len(tensors)} tensors")

    gqa_applied = False
    if args.gqa_kv_heads > 0:
        old_kv_heads = tc.get("num_global_key_value_heads", tc.get("num_key_value_heads", 1))
        # .get(...) all the way down -- NOT tc["head_dim"] -- because "no
        # head_dim key at all" must flow into the same safe-skip path as any
        # other layout mismatch, not raise a raw KeyError before
        # detect_mqa_v_shares_k_layout() even gets a chance to run. Confirmed
        # via a real Qwen2Config().to_dict() that head_dim is genuinely absent
        # on that architecture (it's derived internally, not serialized) --
        # this is a real, reachable case, not a hypothetical one.
        global_head_dim = tc.get("global_head_dim", tc.get("head_dim"))
        full_attn_idxs = [i for i in range(orig_layers) if layer_types[i] == "full_attention"]

        if global_head_dim is None:
            matches, reason = False, (
                "config has neither 'global_head_dim' nor 'head_dim' -- can't "
                "compute the expected k_proj/v_proj shape for the MQA layout "
                "check, so refusing to guess (this is expected on architectures "
                "like Qwen2, whose config doesn't serialize head_dim at all)"
            )
        else:
            matches, reason = detect_mqa_v_shares_k_layout(tensors, full_attn_idxs, args.layer_prefix,
                                                           global_head_dim, old_kv_heads)
        if not matches and not args.force_gqa_fix:
            log(f"Pass 1/3: SKIPPED -- checkpoint doesn't match the assumed MQA layout ({reason}). "
                f"This fix is a narrow, Gemma-4-specific optimization for checkpoints that ship "
                f"with 'V literally reuses K, no v_proj at all' on full-attention layers; your "
                f"checkpoint doesn't look like that, so applying it blindly could clobber a real "
                f"v_proj or misshape k_proj. Pass --force-gqa-fix to override this check (only if "
                f"you've verified the layout yourself), or --gqa-kv-heads 0 to silence this.")
        else:
            if global_head_dim is None:
                # --force-gqa-fix was passed but there's still no usable head_dim
                # to compute shapes from -- gqa_expand_kv() below does
                # old_kv_heads * head_dim arithmetic and would raise a confusing
                # TypeError deep inside a tensor-shape computation. Force-fix
                # means "override the layout check," not "override having a
                # head_dim at all" -- fail clearly here instead.
                raise SystemExit(
                    "ERROR: --force-gqa-fix was set, but the config has neither "
                    "'global_head_dim' nor 'head_dim' -- there's no way to compute "
                    "the expected tensor shape at all, forcing past the layout "
                    "check can't help here. Pass --gqa-kv-heads 0 to skip the GQA "
                    "fix, or add a 'head_dim' key to the checkpoint's config.json "
                    "if you know the real value."
                )
            if not matches:
                log(f"Pass 1/3: WARNING -- --force-gqa-fix set, applying GQA fix despite a failed "
                    f"layout check ({reason}). This is very likely to corrupt weights or crash. "
                    f"Proceeding only because you explicitly forced it.")
            else:
                log(f"Pass 1/3: layout check passed ({reason})")
            log(f"Pass 1/3: GQA fix on {len(full_attn_idxs)} full-attention layers "
                f"({old_kv_heads} -> {args.gqa_kv_heads} kv heads, V=K -> real V) ...")
            for old_idx in full_attn_idxs:
                prefix = f"{args.layer_prefix}.{old_idx}"
                gqa_expand_kv(tensors, prefix, global_head_dim, old_kv_heads, args.gqa_kv_heads,
                             hidden, INIT_SCALE)
            log("  GQA fix done")
            gqa_applied = True
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

    # Write through tc (which is either cfg["text_config"] or cfg itself,
    # depending on whether the config was nested or flat). This avoids creating
    # a spurious text_config dict on flat configs (Llama/Mistral/Qwen).
    tc["intermediate_size"] = new_intermediate
    tc["num_hidden_layers"] = new_layers
    # Only write layer_types back if the original config had it — don't create
    # a spurious Gemma-4-specific field on flat (Llama/Mistral/Qwen) configs.
    if "layer_types" in tc:
        tc["layer_types"] = new_layer_types

    if gqa_applied:
        # attention_k_eq_v=False disables the MQA shortcut (V reusing K) for
        # every layer, so v_proj is materialized independently -- matching the
        # fresh v_proj weights gqa_expand_kv() wrote for the full-attention
        # layers. Gated on gqa_applied (not args.gqa_kv_heads > 0) -- if the
        # layout-detection check skipped the fix, this config flag must NOT
        # flip either, or the written config would claim a GQA layout the
        # tensors don't actually have.
        tc["attention_k_eq_v"] = False
        # num_global_key_value_heads governs the FULL-attention layers, which
        # are the only layers gqa_expand_kv() touched -- so updating it to the
        # new head count is correct.
        if "num_global_key_value_heads" in tc:
            tc["num_global_key_value_heads"] = args.gqa_kv_heads
        # DO NOT overwrite num_key_value_heads: that field governs the
        # SLIDING-attention layers, whose k_proj/v_proj tensors were NOT
        # expanded (gqa_expand_kv only iterates full_attn_idxs). Overwriting
        # it with args.gqa_kv_heads would create a config/tensor shape
        # mismatch for every sliding layer whenever args.gqa_kv_heads differs
        # from the sliding layers' original head count. The default
        # --gqa-kv-heads happens to equal the sliding count, which is why this
        # bug was latent -- but any other value would crash at load.
        log(f"GQA fix applied: attention_k_eq_v -> False, "
            f"full-attention layers now use {args.gqa_kv_heads} real kv heads "
            f"(was MQA=1, V=K). num_key_value_heads (sliding layers) left "
            f"unchanged -- sliding layers were not expanded.")

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

    write_sharded(final_tensors, args.dst, args.max_shard_bytes)

    log("done. NEXT: load-test before trusting this for CPT (check for NaN/Inf in "
        "logits on a raw forward pass against a real input before committing to a "
        "long training run).")


if __name__ == "__main__":
    main()
