#!/usr/bin/env python3
"""Minimal STUB modeling file for checkpoints produced by mtp_head.py.

WHY THIS FILE EXISTS
--------------------
mtp_head.py writes real Multi-Token-Prediction (MTP) module weights into a
checkpoint and sets config.json's `auto_map` to
`{"AutoModelForCausalLM": "modeling_custom.CustomForCausalLM"}`. For
`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` to load
that checkpoint, a `modeling_custom.py` defining `CustomForCausalLM` must sit
alongside config.json. This file is that stub.

THIS IS A STARTING POINT, NOT A FINISHED IMPLEMENTATION
-------------------------------------------------------
The class below is intentionally minimal. It:
  * subclasses a transformers `*ForCausalLM` base, picked via a try/except
    import chain so it works across model families (Gemma3, Gemma2, Llama,
    ...);
  * builds MTP modules matching the DeepSeek-V3 pattern and the tensor key
    naming mtp_head.py writes (`model.mtp_layers.{i}.{enorm,eh_proj,block,
    lnorm}` plus a shared `model.mtp.norm`);
  * wires a `forward` that runs the base model and then, when
    `config.mtp_depths > 0`, runs the MTP modules over the last hidden state.

It does NOT implement the full DeepSeek-V3 MTP training/loss bookkeeping
(proper multi-step target shifting, the weighted MTP loss, KV-cache handling
for the cloned blocks, etc.). Treat it as scaffolding: load it, confirm the
MTP weights map onto the modules without missing-key errors, then extend
`forward` for your real train/inference path. The module names and tensor
shapes below are chosen to match mtp_head.py's output so `from_pretrained`
populates them; the forward logic is where you do the real work.
"""

import copy
import inspect

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Base class selection. Try the common transformers CausalLM classes in turn;
# the first import that succeeds wins. This is deliberately a coarse, global
# choice -- for a specific checkpoint you will usually want to dispatch on
# `config.model_type` instead, but a try/except chain is enough for a stub and
# keeps this file free of model-family conditionals. Extend the chain if your
# checkpoint uses a family not listed here.
# ---------------------------------------------------------------------------
try:
    from transformers import Gemma3ForCausalLM as _BaseForCausalLM
except ImportError:
    try:
        from transformers import Gemma2ForCausalLM as _BaseForCausalLM
    except ImportError:
        from transformers import LlamaForCausalLM as _BaseForCausalLM


def _find_decoder_layers(model):
    """Locate the decoder-layer ModuleList across common base-model layouts.

    mtp_head.py's default --layer-prefix is `model.language_model.layers`, but
    flat Llama/Mistral/Qwen checkpoints use `model.layers`. Try both (and the
    multimodal `text_model.layers`) so the cloned MTP `block` matches the base
    model's real block type. Returns the ModuleList, or raises if not found.
    """
    for path in ("layers", "language_model.layers", "text_model.layers"):
        obj = model
        found = True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                found = False
                break
        if found and isinstance(obj, nn.ModuleList) and len(obj) > 0:
            return obj
    raise AttributeError(
        "Could not locate a decoder-layer ModuleList on the base model "
        "(tried .layers, .language_model.layers, .text_model.layers). "
        "Set self.mtp_layers manually or override _find_decoder_layers() "
        "for your architecture."
    )


class _MTPRMSNorm(nn.Module):
    """Minimal RMSNorm with a single `weight` parameter (init 1.0).

    Matches the enorm/lnorm/norm tensors mtp_head.py writes: shape (hidden,),
    initialized to 1.0. The base model's own RMSNorm class would also work;
    this avoids depending on a model-family-specific norm class.
    """

    def __init__(self, hidden, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(in_dtype)


class _MTPModule(nn.Module):
    """One MTP depth: enorm -> eh_proj -> block -> lnorm.

    Mirrors the per-depth tensor group mtp_head.py writes:
        model.mtp_layers.{i}.enorm.weight    (RMSNorm, hidden)
        model.mtp_layers.{i}.eh_proj.weight  (Linear 2*hidden -> hidden, no bias)
        model.mtp_layers.{i}.block.<suffix>  (cloned from the base model's last layer)
        model.mtp_layers.{i}.lnorm.weight    (RMSNorm, hidden)
    """

    def __init__(self, hidden, base_block):
        super().__init__()
        self.enorm = _MTPRMSNorm(hidden)
        self.eh_proj = nn.Linear(2 * hidden, hidden, bias=False)
        # Clone the base model's last decoder layer so the MTP block starts
        # from real pretrained weights (matches mtp_head.py's block cloning).
        self.block = copy.deepcopy(base_block)
        self.lnorm = _MTPRMSNorm(hidden)

    def forward(self, h, token_emb, position_embeddings=None):
        # DeepSeek-V3 MTP step: norm the incoming hidden state, concat with the
        # token embedding for this prediction step, project down, run the
        # cloned transformer block, then norm. Structural shape only -- the
        # exact target-shifting / loss wiring is left to the user.
        h_norm = self.enorm(h)
        x = torch.cat([h_norm, token_emb], dim=-1)
        x = self.eh_proj(x)
        # The cloned decoder layer's forward computes attention internally and
        # requires the rotary (cos, sin) position embeddings to be passed in
        # explicitly -- the top-level model forward normally computes these
        # once and threads them through every layer; a bare `self.block(x)`
        # call leaves `position_embeddings=None` and crashes inside attention
        # trying to unpack it. CustomForCausalLM.forward computes this once
        # (via the base model's `rotary_emb`) and passes it down here.
        if position_embeddings is not None:
            x = self.block(x, position_embeddings=position_embeddings)
        else:
            x = self.block(x)
        # Some decoder layer implementations return a tuple (hidden_states, ...).
        if isinstance(x, tuple):
            x = x[0]
        x = self.lnorm(x)
        return x


class CustomForCausalLM(_BaseForCausalLM):
    """Causal LM with an optional Multi-Token-Prediction (MTP) head.

    STUB -- see module docstring. `mtp_depths` (read from config) controls how
    many MTP modules are appended. When 0 (or absent), this behaves exactly
    like the base `*ForCausalLM` and adds no MTP modules.
    """

    def __init__(self, config):
        super().__init__(config)
        mtp_depths = int(getattr(config, "mtp_depths", 0) or 0)
        self.mtp_depths = mtp_depths
        self.mtp_loss_weight = float(getattr(config, "mtp_loss_weight", 0.0) or 0.0)

        if mtp_depths > 0:
            hidden = config.hidden_size
            # Clone the base model's last decoder layer as the MTP block donor.
            # Done after super().__init__ so the base layers exist. If the
            # layer list can't be auto-located, fail loudly rather than build a
            # half-functional module: `_MTPModule(hidden, None)` would silently
            # end up with `self.block = None` (copy.deepcopy(None) is None, and
            # assigning None never registers an nn.Module submodule), which
            # means (a) from_pretrained silently drops every
            # `model.mtp_layers.{i}.block.*` tensor as "unexpected" instead of
            # loading it, and (b) forward() crashes with
            # "'NoneType' object is not callable" the first time it's called.
            # Neither of those is a usable degraded mode, so surface the
            # original error instead of masking it.
            donor_block = _find_decoder_layers(self.model)[-1]

            # Attach to self.model (not self) so the state_dict key prefix
            # matches mtp_head.py's `model.mtp_layers.*`. The shared final norm
            # is attached AS AN ATTRIBUTE OF THE SAME ModuleList (not a sibling
            # `self.model.mtp` container) so its state_dict key comes out as
            # `model.mtp_layers.norm.weight` -- matching what mtp_head.py
            # actually writes to disk (`f"{mtp_prefix}.norm.weight"` with
            # mtp_prefix="model.mtp_layers"). nn.ModuleList is a plain
            # nn.Module under the hood, so attaching a named submodule
            # alongside its indexed items is supported and produces exactly
            # this key shape.
            self.model.mtp_layers = nn.ModuleList(
                [_MTPModule(hidden, donor_block) for _ in range(mtp_depths)]
            )
            self.model.mtp_layers.norm = _MTPRMSNorm(hidden)

    def forward(self, *args, **kwargs):
        # Run the base model, forcing hidden states so the MTP head has input.
        kwargs.pop("output_hidden_states", None)
        outputs = super().forward(*args, output_hidden_states=True, **kwargs)

        if self.mtp_depths > 0 and hasattr(self.model, "mtp_layers"):
            # --- MINIMAL MTP forward (EXTEND THIS for real use) ---
            # DeepSeek-V3 MTP predicts multiple future tokens from the hidden
            # states, shifting targets per depth and summing a weighted MTP
            # loss (weight = self.mtp_loss_weight). This stub only runs the
            # modules over the last hidden state + input embeddings so the
            # weights are exercised; it does NOT compute the MTP loss or do
            # target shifting. See the module docstring.
            h = outputs.hidden_states[-1]  # (B, T, H)

            # Token embeddings for the eh_proj concat. Best-effort: resolve
            # from input_ids via the base model's embed_tokens, falling back to
            # the hidden state itself when input_ids isn't available (e.g. an
            # inputs_embeds call path). A real implementation should pass the
            # correct per-step embeddings explicitly.
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            embed_tokens = getattr(self.model, "embed_tokens", None)
            if embed_tokens is not None and input_ids is not None:
                token_emb = embed_tokens(input_ids)
            else:
                token_emb = h

            # The cloned decoder block inside each _MTPModule needs rotary
            # (cos, sin) position embeddings -- without them the block's
            # internal attention crashes trying to unpack `position_embeddings`
            # (it defaults to None when called directly, unlike a full model
            # forward which always threads this through). Compute it the same
            # way the base model does, via its `rotary_emb` submodule, using
            # position_ids if the caller supplied them (falls back to a plain
            # 0..T-1 range, matching a from-scratch forward pass with no past
            # KV cache).
            position_embeddings = None
            rotary_emb = getattr(self.model, "rotary_emb", None)
            if rotary_emb is not None:
                position_ids = kwargs.get("position_ids")
                if position_ids is None:
                    seq_len = h.shape[1]
                    position_ids = torch.arange(seq_len, device=h.device).unsqueeze(0)
                # Gemma3's rotary_emb additionally requires a `layer_type`
                # kwarg (hybrid sliding/full-attention layers use different
                # inv_freq tables) -- Llama/Gemma2-style rotary_emb takes just
                # (x, position_ids). Detect which shape we have instead of
                # hardcoding a model-family branch. The donor block for the
                # MTP clone is always the LAST decoder layer, so use that
                # layer's type when one exists.
                try:
                    needs_layer_type = "layer_type" in inspect.signature(
                        rotary_emb.forward
                    ).parameters
                except (TypeError, ValueError):
                    needs_layer_type = False

                if needs_layer_type:
                    layer_types = getattr(self.config, "layer_types", None)
                    layer_type = layer_types[-1] if layer_types else "full_attention"
                    position_embeddings = rotary_emb(h, position_ids, layer_type)
                else:
                    position_embeddings = rotary_emb(h, position_ids)

            # NOTE: iterate by explicit index (0..mtp_depths-1), NOT
            # `for mtp in self.model.mtp_layers`. nn.ModuleList.__iter__ walks
            # every registered submodule, including ones attached as plain
            # attributes (like `.norm` below) rather than just the indexed
            # depth entries -- a bare `for` loop would wrongly also call
            # `.norm(h, token_emb, ...)` as if it were an _MTPModule.
            for i in range(self.mtp_depths):
                h = self.model.mtp_layers[i](h, token_emb, position_embeddings=position_embeddings)
            h = self.model.mtp_layers.norm(h)

            # Surface the final MTP hidden state for downstream user code.
            # Guarded: if the base model was called with return_dict=False,
            # `outputs` is a plain tuple and can't take an attribute.
            try:
                outputs.mtp_hidden_states = h
            except (AttributeError, TypeError):
                pass

        return outputs
