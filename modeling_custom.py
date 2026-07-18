#!/usr/bin/env python3
"""Custom CausalLM with Multi-Token-Prediction (MTP) support.

This file is copied alongside a checkpoint by `train_cpt.py` (see
`custom_code_src` handling) so that `AutoModelForCausalLM.from_pretrained(...,
trust_remote_code=True)` can load the extra `model.mtp_layers.*` weights.
`mtp_head.py` does NOT copy this file -- it writes the MTP weights + config
(`auto_map` pointing at `modeling_custom.CustomForCausalLM`) and explicitly
logs that the user must place a `modeling_custom.py` alongside the checkpoint
themselves; it never does so automatically.

It implements:
  * DeepSeek-V3-style MTP modules (one cloned decoder block per depth).
  * Real MTP training loss: each depth predicts input_ids shifted by depth+1,
    summed and weighted by `config.mtp_loss_weight`.
  * Best-effort KV-cache support for inference: each MTP depth maintains its
    own `past_key_value` entry when `use_cache=True`.

BASE CLASS SELECTION: the user specifies the model family explicitly via
`--model-family <name>` at train time (train_cpt.py) or MTP-add time
(mtp_head.py). That family name is written into config.json as
`model_family`. At load time, this file reads `model_family` from the
config.json sitting alongside it (they're in the same checkpoint directory)
and selects the matching CausalLM base class. This is NOT auto-guessing --
the user told us which family, and we only try classes within that family
(in version order, since the exact class name depends on the transformers
version installed).
"""

from __future__ import annotations

import copy
import inspect
import json
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model family -> CausalLM class chain.
#
# Each family maps to a list of candidate class names, most specific first
# (e.g. Gemma4 before Gemma3 before Gemma2). We try them in order BECAUSE the
# exact class name depends on the transformers version installed -- Gemma4ForCausalLM
# only exists in transformers 5.5+, so on an older install we fall back to
# Gemma3ForCausalLM, etc. This is version resolution WITHIN the user-specified
# family, not cross-family guessing.
#
# To add a new model family: add an entry here and pass --model-family <name>.
# ---------------------------------------------------------------------------
_FAMILY_CLASS_CHAINS: dict[str, list[str]] = {
    "gemma":    ["Gemma4ForCausalLM", "Gemma3ForCausalLM", "Gemma2ForCausalLM", "GemmaForCausalLM"],
    "llama":    ["Llama4ForCausalLM", "LlamaForCausalLM"],
    "qwen":     ["Qwen3ForCausalLM", "Qwen2ForCausalLM"],
    "mistral":  ["MistralForCausalLM"],
    "phi3":     ["Phi4ForCausalLM", "Phi3ForCausalLM"],
    "phi":      ["Phi4ForCausalLM", "Phi3ForCausalLM"],  # alias for phi3
    "falcon":   ["FalconForCausalLM"],
    "gpt2":     ["GPT2LMHeadModel"],
    "gpt_neox": ["GPTNeoXForCausalLM"],
    "gptj":     ["GPTJForCausalLM"],
    "bloom":    ["BloomForCausalLM"],
    "mpt":      ["MPTForCausalLM"],
    "cohere":   ["CohereForCausalLM"],
    "starcoder2": ["Starcoder2ForCausalLM"],
}


def _read_model_family_from_config() -> str | None:
    """Read `model_family` from the config.json sitting alongside this file.

    modeling_custom.py is copied alongside config.json by train_cpt.py's
    checkpoint save. When HF loads via trust_remote_code=True, this file and
    config.json are in the same directory, so we can read model_family here
    at import time to select the right base class -- no guessing.

    Also checks nested `text_config.model_family` (Gemma-4 / multimodal layouts
    where the text config is one level down).
    """
    # Try the directory this file lives in.
    this_dir = os.path.dirname(os.path.abspath(__file__))
    for config_path in (
        os.path.join(this_dir, "config.json"),
    ):
        if not os.path.exists(config_path):
            continue
        try:
            with open(config_path) as f:
                cfg = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        # Check top-level first, then nested text_config (Gemma-4 layout).
        family = cfg.get("model_family")
        if family:
            return family
        tc = cfg.get("text_config")
        if isinstance(tc, dict):
            family = tc.get("model_family")
            if family:
                return family
    # Fallback: environment variable (for cases where config.json isn't
    # readable at import time, e.g. HF Hub cached code without the config).
    return os.environ.get("MODEL_FAMILY")


def _resolve_base_class(family: str):
    """Resolve the CausalLM base class for a user-specified model family.

    Tries the family's class chain in order (most specific first) since the
    exact class name depends on the transformers version. This is NOT guessing
    across families -- the user explicitly specified which family, and we only
    try classes within that family.
    """
    import transformers

    if family not in _FAMILY_CLASS_CHAINS:
        raise ValueError(
            f"Unknown model family '{family}'. "
            f"Supported families: {', '.join(sorted(_FAMILY_CLASS_CHAINS))}. "
            f"Pass --model-family <name> at train time."
        )
    chain = _FAMILY_CLASS_CHAINS[family]
    for cls_name in chain:
        cls = getattr(transformers, cls_name, None)
        if cls is not None:
            return cls
    raise ImportError(
        f"No CausalLM class for family '{family}' found in transformers "
        f"{transformers.__version__}. Tried: {chain}. Install a transformers "
        f"version that has one of these classes, or pick a different --model-family."
    )


# ---------------------------------------------------------------------------
# Resolve the base class at import time from the user-specified model_family
# in config.json. If model_family is not set, we CANNOT guess -- error clearly
# telling the user to pass --model-family.
# ---------------------------------------------------------------------------
_model_family = _read_model_family_from_config()
if _model_family is not None:
    _BaseForCausalLM = _resolve_base_class(_model_family)
else:
    # No model_family in config. We deliberately do NOT fall back to a
    # cross-family guess (the old behavior tried Gemma -> Qwen -> Phi -> Llama,
    # which could silently pick the wrong architecture and corrupt training).
    # Instead, raise a clear error. The user must pass --model-family.
    #
    # We can't raise at import time because HF imports this module before
    # the user sees any output. Instead, set _BaseForCausalLM to a sentinel
    # that raises a clear error when CustomForCausalLM is instantiated.
    class _ModelFamilyNotSet:
        """Sentinel base class that raises a clear error at instantiation."""
        def __init__(self, *args, **kwargs):
            raise ValueError(
                "config.model_family is not set. This file (modeling_custom.py) "
                "was loaded via trust_remote_code but the checkpoint's config.json "
                "does not contain a 'model_family' field. Pass --model-family <name> "
                "when training (e.g. --model-family gemma) so the base class is "
                f"known. Supported families: {', '.join(sorted(_FAMILY_CLASS_CHAINS))}."
            )
    _BaseForCausalLM = _ModelFamilyNotSet


# ---------------------------------------------------------------------------
# Pure MTP helpers (duplicated here so this file is self-contained when copied
# to a checkpoint directory outside the repo root).
# ---------------------------------------------------------------------------
def _shift_labels(
    input_ids: Optional[torch.Tensor],
    depth: int,
    ignore_index: int = -100,
    inputs_embeds: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build depth-shifted labels for MTP loss.

    `input_ids` is the normal source, but callers may legitimately pass
    `inputs_embeds` instead of `input_ids` (e.g. soft-prompting, embedding-
    level data pipelines) -- in that case `input_ids` is None and we can't
    read `.shape` off it directly. When that happens we can still recover the
    (batch, seq_len) shape from `inputs_embeds` (batch, seq_len, hidden), but
    we have no token ids to use as shifted targets -- there is nothing
    meaningful to predict, so every position comes back fully masked
    (ignore_index). Callers MUST treat an all-ignore_index label tensor
    specially (see `_compute_mtp_total_loss`): `F.cross_entropy(...,
    reduction="mean")` divides by the unmasked count, which is 0 here, and
    silently returns NaN rather than raising -- that would poison the whole
    loss instead of crashing loudly, which is worse. This function only
    builds the label tensor; it does not itself decide how to average.
    """
    if input_ids is not None:
        b, t = input_ids.shape
        labels = input_ids.new_full((b, t), ignore_index)
        shift = depth + 1
        if t > shift:
            labels[:, : t - shift] = input_ids[:, shift:]
        return labels

    if inputs_embeds is None:
        raise ValueError(
            "_shift_labels requires either input_ids or inputs_embeds to "
            "determine the target shape."
        )
    b, t, _ = inputs_embeds.shape
    return inputs_embeds.new_full((b, t), ignore_index, dtype=torch.long)


def _compute_mtp_total_loss(
    all_mtp_logits: List[torch.Tensor],
    input_ids: Optional[torch.Tensor],
    global_weight: float,
    inputs_embeds: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if global_weight == 0.0 or not all_mtp_logits:
        return all_mtp_logits[0].new_tensor(0.0) if all_mtp_logits else torch.tensor(0.0)
    total = all_mtp_logits[0].new_tensor(0.0)
    for depth, logits in enumerate(all_mtp_logits):
        labels = _shift_labels(input_ids, depth=depth, inputs_embeds=inputs_embeds)
        valid = labels != -100
        if not bool(valid.any()):
            # Every position is masked (this only happens when input_ids is
            # None, i.e. the caller used inputs_embeds -- there are no token
            # ids to predict against for this or any deeper MTP head).
            # F.cross_entropy(reduction="mean") divides by the unmasked count
            # (0 here) and returns NaN rather than raising, which would
            # silently poison `total` for every depth (NaN + anything = NaN).
            # Contribute 0 instead so the rest of the loss stays finite.
            continue
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="mean",
        )
        total = total + loss
    return total * global_weight


# ---------------------------------------------------------------------------
# Layer-path detection (duplicated here so the file is self-contained).
# ---------------------------------------------------------------------------
def _find_decoder_layers(model: nn.Module) -> nn.ModuleList:
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
        "(tried .layers, .language_model.layers, .text_model.layers)."
    )


class _MTPRMSNorm(nn.Module):
    """RMSNorm for MTP layers. Upcasts only the variance reduction to fp32
    (not the full multiply) — matches HF's LlamaRMSNorm pattern and avoids the
    per-layer full-tensor fp32 round-trip the old version did. When Liger
    Kernel is available, the base model's RMSNorm is already fused by
    apply_liger_kernel at load time; this custom norm only runs in the MTP
    head layers (which Liger doesn't touch)."""
    def __init__(self, hidden: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute variance in fp32 (numerical stability) but keep the
        # multiply + scale in the input dtype — avoids materializing a full
        # fp32 copy of x (the old `x = x.to(torch.float32)` did that).
        input_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = x.to(input_dtype)
        # Cast weight to input_dtype BEFORE the multiply — without this,
        # float32 weight * bf16 x promotes to float32, and the output dtype
        # no longer matches the input (a dtype mismatch bug that crashes the
        # MTP forward pass on bf16 runs).
        return self.weight.to(input_dtype) * x


class _MTPModule(nn.Module):
    """One MTP depth: enorm -> eh_proj -> block -> lnorm."""

    def __init__(self, hidden: int, base_block: nn.Module):
        super().__init__()
        self.enorm = _MTPRMSNorm(hidden)
        self.eh_proj = nn.Linear(2 * hidden, hidden, bias=False)
        self.block = copy.deepcopy(base_block)
        self.lnorm = _MTPRMSNorm(hidden)
        # Cache the block's forward signature once — inspect.signature() is
        # expensive to call every forward step (Python reflection in the hot
        # loop). The signature doesn't change after construction.
        self._block_fwd_params = inspect.signature(self.block.forward).parameters

    def forward(
        self,
        h: torch.Tensor,
        token_emb: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[object] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[object]]:
        h_norm = self.enorm(h)
        x = torch.cat([h_norm, token_emb], dim=-1)
        x = self.eh_proj(x)

        block_kwargs = {}
        params = self._block_fwd_params  # cached in __init__
        if "position_embeddings" in params:
            block_kwargs["position_embeddings"] = position_embeddings
        if "past_key_value" in params:
            block_kwargs["past_key_value"] = past_key_value
        if "use_cache" in params:
            block_kwargs["use_cache"] = use_cache

        out = self.block(x, **block_kwargs)
        present_key_value = None
        if isinstance(out, tuple):
            x = out[0]
            if use_cache and len(out) > 1:
                present_key_value = out[1]
        else:
            x = out
        x = self.lnorm(x)
        return x, present_key_value


class CustomForCausalLM(_BaseForCausalLM):
    """Causal LM with optional MTP head.

    When `config.mtp_depths > 0`, the model appends MTP modules and computes a
    weighted MTP training loss. When `mtp_depths` is 0 or absent, it behaves
    exactly like the base CausalLM.
    """

    def __init__(self, config):
        super().__init__(config)
        self.mtp_depths = int(getattr(config, "mtp_depths", 0) or 0)
        self.mtp_loss_weight = float(getattr(config, "mtp_loss_weight", 0.0) or 0.0)

        if self.mtp_depths > 0:
            hidden = config.hidden_size
            donor_block = _find_decoder_layers(self.model)[-1]
            self.model.mtp_layers = nn.ModuleList(
                [_MTPModule(hidden, donor_block) for _ in range(self.mtp_depths)]
            )
            self.model.mtp_layers.norm = _MTPRMSNorm(hidden)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[object]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Always ask the base model for hidden states when MTP is active.
        need_hidden = self.mtp_depths > 0

        base_outputs = super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            labels=labels if self.mtp_depths == 0 else None,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=need_hidden or (output_hidden_states is True),
            return_dict=True,
            **kwargs,
        )

        mtp_loss = None
        mtp_logits: List[torch.Tensor] = []
        mtp_past_key_values: List[Optional[object]] = []

        if self.mtp_depths > 0 and hasattr(self.model, "mtp_layers"):
            h = base_outputs.hidden_states[-1]

            if inputs_embeds is not None:
                token_emb = inputs_embeds
            elif input_ids is not None:
                token_emb = self.model.embed_tokens(input_ids)
            else:
                token_emb = h

            position_embeddings = self._compute_position_embeddings(h, position_ids)

            for i in range(self.mtp_depths):
                past_kv = None
                if past_key_values is not None and i < len(past_key_values):
                    past_kv = past_key_values[i]
                depth_h, present_kv = self.model.mtp_layers[i](
                    h,
                    token_emb,
                    position_embeddings=position_embeddings,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                )
                h = depth_h
                mtp_past_key_values.append(present_kv)
                # Project to vocabulary using the shared lm_head.
                depth_logits = self.lm_head(h)
                mtp_logits.append(depth_logits)

            h = self.model.mtp_layers.norm(h)

            if labels is not None:
                # Base model was NOT given labels above (so we can reuse its
                # hidden states). Compute the base CE loss explicitly, then add
                # MTP losses.
                base_logits = base_outputs.logits
                base_loss = F.cross_entropy(
                    base_logits.view(-1, base_logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction="mean",
                )
                mtp_loss = _compute_mtp_total_loss(
                    mtp_logits, input_ids, self.mtp_loss_weight, inputs_embeds=inputs_embeds
                )
                total_loss = base_loss + mtp_loss
            else:
                total_loss = base_outputs.loss

            # Attach MTP outputs for downstream use.
            base_outputs.mtp_logits = mtp_logits
            base_outputs.mtp_hidden_states = h
            if use_cache:
                base_outputs.mtp_past_key_values = mtp_past_key_values
        else:
            total_loss = base_outputs.loss

        if total_loss is not None:
            base_outputs.loss = total_loss

        if not return_dict:
            # Preserve tuple contract: (loss, logits, ...) if loss present.
            out = (base_outputs.loss,) + base_outputs[1:]
            return out
        return base_outputs

    def _compute_position_embeddings(
        self,
        hidden_state: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        rotary_emb = getattr(self.model, "rotary_emb", None)
        if rotary_emb is None:
            return None

        if position_ids is None:
            seq_len = hidden_state.shape[1]
            position_ids = torch.arange(seq_len, device=hidden_state.device).unsqueeze(0)

        # Cache the rotary_emb forward signature lazily (rotary_emb may not
        # exist at __init__ time). inspect.signature is expensive to call
        # every generation step.
        if not hasattr(self, "_rotary_emb_fwd_params"):
            try:
                self._rotary_emb_fwd_params = inspect.signature(rotary_emb.forward).parameters
            except (TypeError, ValueError):
                self._rotary_emb_fwd_params = {}
        needs_layer_type = "layer_type" in self._rotary_emb_fwd_params

        if needs_layer_type:
            layer_types = getattr(self.config, "layer_types", None)
            layer_type = layer_types[-1] if layer_types else "full_attention"
            return rotary_emb(hidden_state, position_ids, layer_type)
        return rotary_emb(hidden_state, position_ids)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """Thread base + MTP past_key_values through generation."""
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, **kwargs
        )
        # If the caller passed mtp_past_key_values separately, merge them.
        if "mtp_past_key_values" in kwargs:
            model_inputs["past_key_values"] = kwargs["mtp_past_key_values"]
        return model_inputs
