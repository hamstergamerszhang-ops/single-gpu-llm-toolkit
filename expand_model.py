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

On tensor key SUFFIXES (gate/up/down_proj, self_attn.q/k/v/o_proj): these used
to be hardcoded here (only the prefix before the layer index was a flag, via
--layer-prefix), checked against the installed `transformers` library's own
modeling source (not assumed) -- Llama, Mistral, Qwen2, Qwen3, and every Gemma
generation all define these exact attribute names
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
`up_proj`/`down_proj` with no `gate_proj` at all).

The suffixes are now sourced from `models.registry.ModelFamily` (auto-detected
from config.json's model_type before any tensor is touched, overridable via
--model-family) rather than hardcoded -- same "detect before you act" pattern
as detect_mqa_v_shares_k_layout(). GPT-2 is the first fused-QKV architecture
with REAL code support here (not just a flag): its `c_attn` is a Conv1D whose
weight is stored TRANSPOSED as (in, out) with Q|K|V as three column-blocks, so
width_expand_layer pads the axis OPPOSITE the nn.Linear case, extends the
Conv1D biases in lockstep with the grown weights (Llama's Linear layers have
bias=False; GPT-2's Conv1D always carries a bias that must grow too -- a bug
caught by the load-test this tool's closing log line asks for), and the GQA
pass is skipped outright (a fused-QKV layout has no separate v_proj for the
fix to target). Verified end-to-end against a real GPT-2 checkpoint built from
GPT2Config: expand (width+depth) -> from_pretrained load (zero missing/
unexpected) -> forward pass (no NaN/Inf) -> the zeroed-output-projection
duplicate layers confirmed numerically no-op. Phi-3/Falcon/MPT/BLOOM remain
KeyError-on-missing-key (clean failure, not silent wrong behavior) until they
get the same per-architecture code GPT-2 just got.

This has been run end-to-end against Gemma-4-family checkpoints on a single
MI300X and against a real GPT-2 checkpoint on CPU; the flags plus the suffix
research above make it plausible (and, for the common Llama-style suffix case,
probably correct) to point elsewhere, but that's not the same as verified
there.
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
                       new_intermediate: int, family=None):
    """Grow one decoder layer's MLP intermediate dimension by `n_new` columns.

    `family` (a models.registry.ModelFamily, or None for the legacy Llama
    default) drives TWO architecture-dependent decisions, both verified against
    the installed transformers modeling source rather than assumed:

      1. The tensor-key segments and leaf suffixes come from the family
         (mlp_path + mlp_suffixes), not hardcoded `mlp.gate_proj` etc. -- so
         GPT-2's `mlp.c_fc`/`mlp.c_proj` are addressed by name.
      2. The axis padded depends on `family.weight_orientation`:
           - "linear" (Llama/Gemma/Qwen, nn.Linear, weight=(out,in)): gate/up
             pad dim=0 (the output dim), down pads dim=1.
           - "conv1d" (GPT-2/GPT-NeoX, HF Conv1D, weight=(in,out) TRANSPOSED):
             the up projection (c_fc) pads dim=1 and the down projection
             (c_proj) pads dim=0 -- the OPPOSITE axes from the linear case,
             because the matrices are stored transposed. Padding the linear
             case's axes here would grow the wrong dimension and silently
             corrupt the checkpoint.

    A gated MLP (Llama gate_proj) and a non-gated MLP (GPT-2, no gate) are both
    handled: gate is only touched when the family declares one. There is no
    GQA/attention work here -- that's gqa_expand_kv() (skipped for fused-QKV
    families; see main()).
    """
    n_new = new_intermediate - old_intermediate
    # Default to the legacy Llama layout when called without a family (the
    # pre-existing call sites and tests that pass family=None get byte-identical
    # behavior to before this function was family-aware).
    mlp_path = getattr(family, "mlp_path", "mlp") if family else "mlp"
    mlp_suffixes = getattr(family, "mlp_suffixes", None) if family else None
    orientation = getattr(family, "weight_orientation", "linear") if family else "linear"
    if mlp_suffixes is None:
        mlp_suffixes = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}

    up_suffix = mlp_suffixes.get("up")
    down_suffix = mlp_suffixes["down"]
    gate_suffix = mlp_suffixes.get("gate")  # None for non-gated MLPs (GPT-2, MPT)

    up_key = f"{layer_prefix}.{mlp_path}.{up_suffix}.weight"
    down_key = f"{layer_prefix}.{mlp_path}.{down_suffix}.weight"

    # `hidden` is the input dim common to both projections. For nn.Linear it is
    # .shape[1] (the in-dim) on gate/up; for Conv1D (transposed) it is .shape[0]
    # (the in-dim) on c_fc. Detect it from whichever key exists.
    hidden = tensors[up_key].shape[1] if orientation == "linear" else tensors[up_key].shape[0]

    # The "expanding" projections (gate+up for Llama, c_fc for GPT-2) grow along
    # the OUTPUT dim. For linear weights (out,in) that's dim=0; for Conv1D
    # (in,out) that's dim=1. The pad shape that fits that dim is orientation-
    # dependent too: orthogonal_pad(transpose_for_rows=True) returns (n_new,
    # n_existing) -- correct for cat on dim=0 (Linear); transpose_for_rows=False
    # returns (n_existing, n_new) -- correct for cat on dim=1 (Conv1D). So the
    # transpose flag is SWAPPED vs the project-back case below.
    expand_dim = 0 if orientation == "linear" else 1
    expand_transpose = True if orientation == "linear" else False
    expand_keys = [f"{layer_prefix}.{mlp_path}.{gate_suffix}.weight"] if gate_suffix else []
    expand_keys.append(up_key)
    for key in expand_keys:
        old_w = tensors[key]
        pad = orthogonal_pad(n_new, hidden, INIT_SCALE, transpose_for_rows=expand_transpose)
        tensors[key] = torch.cat([old_w, pad.to(old_w.dtype)], dim=expand_dim)
        # The matching bias (if present) must grow in lockstep with the weight's
        # OUTPUT dim -- Llama's gate/up_proj have bias=False (no bias key), but
        # GPT-2's Conv1D always carries a bias of shape (out,). A weight grown
        # to a new out-dim with a stale bias fails from_pretrained with a size
        # mismatch (caught by a real load test). Zero-pad the new bias entries:
        # new capacity contributes nothing until training turns it on, matching
        # the orthogonal-pad-then-small-scale principle for the weights.
        bias_key = key[:-len(".weight")] + ".bias"
        if bias_key in tensors:
            old_b = tensors[bias_key]
            new_b = torch.zeros(n_new, dtype=old_b.dtype)
            tensors[bias_key] = torch.cat([old_b, new_b], dim=0)

    # The "projecting-back" projection (down_proj for Llama, c_proj for GPT-2)
    # grows along the INPUT dim -- the opposite axis from the expanding one,
    # so the pad shape (and thus the transpose flag) is the opposite too. Its
    # bias is (hidden,) -- the OUTPUT dim, which is unchanged -- so the bias
    # needs NO growth here (only the input dim grew).
    old_w = tensors[down_key]
    down_transpose = False if orientation == "linear" else True
    pad = orthogonal_pad(n_new, hidden, INIT_SCALE, transpose_for_rows=down_transpose)
    down_dim = 1 if orientation == "linear" else 0
    tensors[down_key] = torch.cat([old_w, pad.to(old_w.dtype)], dim=down_dim)


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
                        zero_output_projections: bool, family=None):
    prefix_dot = src_prefix + "."
    # The output projections whose zeroing turns a duplicated layer into a true
    # no-op (contributes nothing to the residual stream until training turns it
    # on). Family-derived so GPT-2's attn.c_proj.weight + mlp.c_proj.weight are
    # zeroed, not Llama's self_attn.o_proj.weight + mlp.down_proj.weight --
    # GPT-2 reuses the leaf name "c_proj" for BOTH attention-output and
    # MLP-output, so the match must be path-segment-aware (attn_path/mlp_path),
    # not leaf-name-aware, or it would zero the wrong tensors / miss the right
    # ones. Defaults to the legacy Llama paths when called without a family.
    if family is not None:
        attn_o = f"{family.attn_path}.{family.attn_suffixes.get('o', 'o_proj')}.weight"
        mlp_down = f"{family.mlp_path}.{family.mlp_suffixes['down']}.weight"
        zero_suffixes = (attn_o, mlp_down)
    else:
        zero_suffixes = ("self_attn.o_proj.weight", "mlp.down_proj.weight")
    new_entries = {}
    for key, val in tensors.items():
        if not key.startswith(prefix_dot):
            continue
        suffix = key[len(prefix_dot):]
        new_key = f"{dst_prefix}.{suffix}"
        if zero_output_projections and suffix in zero_suffixes:
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
    ap.add_argument("--model-family", type=str, default=None,
                    help="Force a specific model family (from models.registry: llama, gemma, "
                         "gpt2, phi3, falcon, mpt, bloom, ...) instead of auto-detecting from "
                         "config.json's model_type. Auto-detection is run BEFORE any tensor is "
                         "touched and drives the tensor-key segments + pad axes -- a fused-QKV "
                         "architecture like GPT-2 (c_attn Conv1D, attn/mlp paths) needs real "
                         "code changes, not just --layer-prefix, and the family is how those "
                         "changes are routed. If detection fails it falls back to the legacy "
                         "Llama-derived layout with a logged warning.")
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

    # Pre-flight model-family detection BEFORE touching any tensor -- same
    # "detect before you act" discipline as detect_mqa_v_shares_k_layout()
    # below. The family (from models.registry, auto-detected from config.json's
    # model_type, overridable via --model-family) drives the tensor-key segments
    # and pad axes used by width_expand_layer / clone_layer_tensors, so a wrong
    # guess here would corrupt weights silently. If detection fails we fall back
    # to the legacy Llama-derived layout (the only layout this tool handled
    # before GPT-2 support) and log it, rather than aborting -- a Llama-derived
    # checkpoint still expands correctly under that fallback, and a genuinely
    # unrecognized fused-QKV architecture will KeyError cleanly on a missing
    # tensor key (as the module docstring already documents), not silently
    # mis-expand. --layer-prefix remains the explicit override for the prefix
    # before the layer index; the family only governs the suffixes/segments
    # after it.
    family = None
    try:
        from models import resolve_model_family
        family = resolve_model_family(cfg, override=getattr(args, "model_family", None))
        log(f"detected model family: {family.name} "
            f"(attn_path={family.attn_path}, mlp_path={family.mlp_path}, "
            f"orientation={family.weight_orientation}, layout={family.attn_layout})")
    except Exception as e:
        log(f"model-family detection did not match ({e}); falling back to the "
            f"legacy Llama-derived layout (self_attn/mlp, linear, separate q/k/v). "
            f"Pass --model-family to force one if you know it.")
        family = None

    # Family-aware config keys: GPT-2's serialized config.json uses n_embd/
    # n_layer/n_head/n_inner (verified against GPT2Config().to_dict()), not the
    # unified hidden_size/num_hidden_layers names. Falling back to the unified
    # names keeps the legacy Llama/Gemma/Qwen path byte-identical. Returns None
    # if neither the family key nor the fallback is present (e.g. GPT-2 with
    # n_inner absent -- the caller resolves that to 4*hidden below).
    def _cfg(key_attr, fallback_key):
        if family is not None:
            k = getattr(family, key_attr)
            if k is not None and k in tc:
                return tc[k]
        return tc.get(fallback_key)

    orig_layers = _cfg("num_hidden_layers_key", "num_hidden_layers")
    orig_intermediate = _cfg("intermediate_size_key", "intermediate_size")
    hidden = _cfg("hidden_size_key", "hidden_size")
    # hidden_size and num_hidden_layers are REQUIRED (unlike n_inner, which
    # legitimately can be absent on GPT-2). A None here means the config is
    # missing a field the rest of main() can't proceed without -- fail clearly
    # rather than crashing later with a confusing None+int TypeError.
    if hidden is None:
        raise SystemExit(
            "ERROR: config.json has no hidden-size field (looked for "
            f"{getattr(family, 'hidden_size_key', 'hidden_size')!r}"
            + (" and 'hidden_size'" if family else "") + ").")
    if orig_layers is None:
        raise SystemExit(
            "ERROR: config.json has no num-hidden-layers field (looked for "
            f"{getattr(family, 'num_hidden_layers_key', 'num_hidden_layers')!r}"
            + (" and 'num_hidden_layers'" if family else "") + ").")
    # GPT-2's n_inner is None when unset -- the real modeling code then uses
    # 4*n_embd. A None here would crash the width arithmetic below, so resolve
    # it to the actual intermediate size the weights were built with.
    if orig_intermediate is None:
        orig_intermediate = 4 * hidden
        log(f"  intermediate_size key was None (GPT-2 n_inner unset) -- resolved "
            f"to 4*hidden = {orig_intermediate} (the real GPT-2 default)")
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
        # No index -- some checkpoints ship as a single unsharded
        # model.safetensors with no index at all (a real downloaded case, not
        # an assumption -- same fallback as prune_embeddings_torch.py). Synthesize
        # a single-shard index from the safetensors header so the rest of this
        # script can rely on one consistent format.
        single_file = "model.safetensors"
        single_path = os.path.join(args.src, single_file)
        if not os.path.exists(single_path):
            raise SystemExit(f"ERROR: no model.safetensors.index.json AND no "
                             f"{single_file} in {args.src}")
        with open(single_path, "rb") as f:
            header_len = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(header_len))
        weight_map = {k: single_file for k in header if k != "__metadata__"}
        src_index = {"metadata": {"total_size": os.path.getsize(single_path)},
                     "weight_map": weight_map}
        shard_files = [single_file]
        log(f"no index.json found -- synthesized one for the single-file "
            f"checkpoint ({single_file})")

    log(f"loading {len(shard_files)} source shards ...")
    tensors = {}
    for shard in shard_files:
        loaded = load_file(os.path.join(args.src, shard))
        tensors.update(loaded)
    log(f"  loaded {len(tensors)} tensors")

    gqa_applied = False
    if args.gqa_kv_heads > 0 and family is not None and family.attn_layout == "fused_qkv":
        # GQA/MQA expansion assumes SEPARATE k_proj/v_proj weights it can grow
        # independently. A fused-QKV family (GPT-2's c_attn holds Q|K|V as one
        # matrix) has no separate v_proj to clobber and no k_proj to misshape --
        # the whole premise of the fix doesn't apply, and detect_mqa_v_shares_k_layout()
        # below would look for a v_proj key that structurally cannot exist and
        # report "not found" as if it were a layout mismatch. Skip it outright
        # with a specific reason instead of letting it run that misleading path.
        log(f"Pass 1/3: SKIPPED -- family {family.name} uses a fused-QKV attention "
            f"layout ({family.attn_suffixes.get('qkv')}); the GQA/MQA fix targets "
            f"separate k_proj/v_proj weights and does not apply here. Use "
            f"--gqa-kv-heads 0 to silence this explicitly.")
    elif args.gqa_kv_heads > 0:
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
            width_expand_layer(tensors, prefix, orig_intermediate, new_intermediate, family=family)
            if (old_idx + 1) % 12 == 0:
                log(f"  widened {old_idx + 1}/{orig_layers} layers")
        log("  width expansion done")
    else:
        log("Pass 2/3: SKIPPED (--width-step 0, debug isolation mode)")

    log(f"Pass 3/3: depth duplication ({orig_layers} -> {new_layers} layers) ...")
    # Carry through every NON-layer tensor (embeddings, norms, lm_head) before
    # re-emitting layers by their new indices. The filter must match the
    # family's actual layers path, not a hardcoded ".layers." substring -- GPT-2's
    # path is "transformer.h", which contains no ".layers.", so the old filter
    # would have kept ALL original layer tensors AND added re-indexed clones
    # (stale duplicates). Fall back to ".layers." only when no family resolved.
    if family is not None:
        layer_marker = f".{family.decoder_layers_path}."
    else:
        layer_marker = ".layers."
    final_tensors = {}
    for key, val in tensors.items():
        if layer_marker not in key:
            final_tensors[key] = val

    for new_idx, old_idx, is_dup in depth_plan:
        src_prefix = f"{args.layer_prefix}.{old_idx}"
        dst_prefix = f"{args.layer_prefix}.{new_idx}"
        cloned = clone_layer_tensors(tensors, src_prefix, dst_prefix,
                                     zero_output_projections=is_dup, family=family)
        final_tensors.update(cloned)
        tag = "DUPLICATE (output projections zeroed)" if is_dup else "original"
        log(f"  layer {new_idx:2d} <- old layer {old_idx:2d} [{tag}]")

    del tensors

    actual_total = sum(v.numel() for v in final_tensors.values())
    log(f"actual total params: {actual_total/1e9:.2f}B")

    # Write through tc (which is either cfg["text_config"] or cfg itself,
    # depending on whether the config was nested or flat). This avoids creating
    # a spurious text_config dict on flat configs (Llama/Mistral/Qwen).
    # Write back via the SAME config keys the family reads (GPT-2's serialized
    # config uses n_inner/n_layer, not intermediate_size/num_hidden_layers) --
    # writing the Llama names would leave the real keys stale at their old
    # values and the expanded checkpoint would load with the wrong dimensions.
    intermediate_key = getattr(family, "intermediate_size_key", "intermediate_size") if family else "intermediate_size"
    layers_key = getattr(family, "num_hidden_layers_key", "num_hidden_layers") if family else "num_hidden_layers"
    tc[intermediate_key] = new_intermediate
    tc[layers_key] = new_layers
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
