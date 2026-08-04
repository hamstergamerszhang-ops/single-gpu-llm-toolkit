"""Model-family registry.

Each supported decoder architecture declares where its layers live, what its
weight keys look like, and how to expand/prune it. This removes the hardcoded
assumptions scattered through `expand_model.py`, `mtp_head.py`, and friends.
"""

from __future__ import annotations


class ModelFamily:
    """Declarative description of a decoder architecture."""

    def __init__(
        self,
        name: str,
        model_types: tuple[str, ...],
        decoder_layer_class_suffixes: tuple[str, ...],
        decoder_layers_path: str,
        embed_key: str,
        lm_head_key: str,
        norm_key: str,
        mlp_suffixes: dict[str, str],
        attn_suffixes: dict[str, str],
        tie_weights: bool = False,
        hidden_size_key: str = "hidden_size",
        intermediate_size_key: str = "intermediate_size",
        num_hidden_layers_key: str = "num_hidden_layers",
        num_attention_heads_key: str = "num_attention_heads",
        num_key_value_heads_key: str | None = None,
        vocab_size_key: str = "vocab_size",
        # --- expand_model.py architecture metadata (optional, Llama-default) ---
        # The submodule attribute path segment BETWEEN the layer index and the
        # leaf weight: the key is `<layers_path>.<idx>.<attn_path>.<suffix>`.
        # Llama-derived architectures use self_attn/mlp; GPT-2 uses attn/mlp.
        # Defaults to the Llama names so existing entries are unchanged.
        attn_path: str = "self_attn",
        mlp_path: str = "mlp",
        # "linear" (default): weights are nn.Linear, stored (out, in) -- pad the
        # OUTPUT dim when growing width. "conv1d": weights are HF Conv1D
        # (GPT-2/GPT-NeoX), stored TRANSPOSED as (in, out) -- pad the INPUT dim,
        # i.e. the axis opposite of what the linear case would touch. This flag
        # is what lets width_expand_layer grow GPT-2's c_fc/c_proj/c_attn on the
        # correct axis instead of silently corrupting them.
        weight_orientation: str = "linear",
        # "separate" (default): q/k/v/o are distinct weights (Llama/Gemma/Qwen).
        # "fused_qkv": a single weight (e.g. c_attn) holds Q|K|V as N contiguous
        # column-blocks; expand_model splits, grows each block, and re-cats
        # rather than treating it as one opaque matrix. GQA/MQA never applies to
        # a fused layout, so the GQA pass is skipped entirely for these.
        attn_layout: str = "separate",
    ):
        self.name = name
        self.model_types = model_types
        self.decoder_layer_class_suffixes = decoder_layer_class_suffixes
        self.decoder_layers_path = decoder_layers_path
        self.embed_key = embed_key
        self.lm_head_key = lm_head_key
        self.norm_key = norm_key
        self.mlp_suffixes = mlp_suffixes
        self.attn_suffixes = attn_suffixes
        self.tie_weights = tie_weights
        self.hidden_size_key = hidden_size_key
        self.intermediate_size_key = intermediate_size_key
        self.num_hidden_layers_key = num_hidden_layers_key
        self.num_attention_heads_key = num_attention_heads_key
        self.num_key_value_heads_key = num_key_value_heads_key
        self.vocab_size_key = vocab_size_key
        self.attn_path = attn_path
        self.mlp_path = mlp_path
        self.weight_orientation = weight_orientation
        self.attn_layout = attn_layout

    def __repr__(self) -> str:
        return f"ModelFamily({self.name})"


_LLAMA_LIKE_MLP = {
    "gate": "gate_proj",
    "up": "up_proj",
    "down": "down_proj",
}
_LLAMA_LIKE_ATTN = {
    "q": "q_proj",
    "k": "k_proj",
    "v": "v_proj",
    "o": "o_proj",
}

REGISTRY: dict[str, ModelFamily] = {
    "llama": ModelFamily(
        name="llama",
        model_types=("llama", "mistral", "qwen2", "qwen2_5", "qwen3"),
        decoder_layer_class_suffixes=("DecoderLayer", "Qwen2DecoderLayer"),
        decoder_layers_path="model.layers",
        embed_key="model.embed_tokens.weight",
        lm_head_key="lm_head.weight",
        norm_key="model.norm",
        mlp_suffixes=_LLAMA_LIKE_MLP,
        attn_suffixes=_LLAMA_LIKE_ATTN,
        num_key_value_heads_key="num_key_value_heads",
    ),
    "gemma": ModelFamily(
        name="gemma",
        model_types=("gemma", "gemma2", "gemma3", "gemma4"),
        decoder_layer_class_suffixes=("GemmaDecoderLayer", "Gemma2DecoderLayer", "Gemma3DecoderLayer", "Gemma4DecoderLayer"),
        decoder_layers_path="model.layers",
        embed_key="model.embed_tokens.weight",
        lm_head_key="lm_head.weight",
        norm_key="model.norm",
        mlp_suffixes=_LLAMA_LIKE_MLP,
        attn_suffixes=_LLAMA_LIKE_ATTN,
        tie_weights=True,
        num_key_value_heads_key="num_key_value_heads",
    ),
    "phi3": ModelFamily(
        name="phi3",
        model_types=("phi3", "phi4"),
        decoder_layer_class_suffixes=("Phi3DecoderLayer",),
        decoder_layers_path="model.layers",
        embed_key="model.embed_tokens.weight",
        lm_head_key="lm_head.weight",
        norm_key="model.norm",
        mlp_suffixes={
            "gate": "gate_up_proj",  # Phi-3 fuses gate+up
            "up": "gate_up_proj",
            "down": "down_proj",
        },
        attn_suffixes={
            "qkv": "qkv_proj",  # fused QKV
            "o": "o_proj",
        },
        num_key_value_heads_key="num_key_value_heads",
    ),
    "falcon": ModelFamily(
        name="falcon",
        model_types=("falcon", "falcon_mamba"),
        decoder_layer_class_suffixes=("FalconDecoderLayer",),
        decoder_layers_path="transformer.h",
        embed_key="transformer.word_embeddings.weight",
        lm_head_key="lm_head.weight",
        norm_key="transformer.ln_f",
        mlp_suffixes={
            "dense": "dense_h_to_4h",
            "down": "dense_4h_to_h",
        },
        attn_suffixes={
            "query": "query_key_value",  # fused
            "o": "dense",
        },
        num_key_value_heads_key="num_kv_heads",
    ),
    "mpt": ModelFamily(
        name="mpt",
        model_types=("mpt",),
        decoder_layer_class_suffixes=("MPTBlock",),
        decoder_layers_path="transformer.blocks",
        embed_key="transformer.wte.weight",
        lm_head_key="transformer.output.weight",
        norm_key="transformer.norm_f",
        mlp_suffixes={
            "up": "up_proj",
            "down": "down_proj",
        },
        attn_suffixes={
            "qkv": "Wqkv",
            "o": "out_proj",
        },
    ),
    "gpt2": ModelFamily(
        name="gpt2",
        model_types=("gpt2", "gpt2_refined"),
        decoder_layer_class_suffixes=("GPT2Block",),
        decoder_layers_path="transformer.h",
        embed_key="transformer.wte.weight",
        lm_head_key="transformer.wte.weight",  # tied to input embeddings
        norm_key="transformer.ln_f",
        mlp_suffixes={
            "up": "c_fc",
            "down": "c_proj",
        },
        attn_suffixes={
            "qkv": "c_attn",
            "o": "c_proj",
        },
        tie_weights=True,
        # GPT-2's serialized config.json uses the original OpenAI n_* key names
        # (n_embd/n_layer/n_head/n_inner), NOT the unified hidden_size/
        # num_hidden_layers/... names that HF's PretrainedConfig exposes as
        # Python properties. expand_model.py reads the raw config.json dict, so
        # it sees these n_* names directly. Verified against a real
        # GPT2Config().to_dict() on transformers 5.7.0: n_inner is None when
        # unset (real GPT-2 defaults the intermediate to 4*n_embd).
        hidden_size_key="n_embd",
        intermediate_size_key="n_inner",
        num_hidden_layers_key="n_layer",
        num_attention_heads_key="n_head",
        vocab_size_key="vocab_size",
        # GPT-2 architecture metadata (verified against the installed
        # transformers GPT2Block/GPT2Attention/GPT2MLP __init__ source):
        #   - the attention submodule is named `attn` (not self_attn);
        #   - c_attn/c_fc/c_proj are HF Conv1D, whose .weight is stored
        #     TRANSPOSED as (in, out) -- "basically works like a linear layer
        #     but the weights are transposed" (Conv1D docstring). Width padding
        #     must therefore touch the axis opposite the nn.Linear case.
        #   - c_attn = Conv1D(3*hidden, hidden) holds Q|K|V as three equal
        #     column-blocks along dim=1 -- fused_qkv, not separate q/k/v.
        #     GQA/MQA never applies to a fused layout, so expand_model skips the
        #     GQA pass outright for this family.
        attn_path="attn",
        mlp_path="mlp",
        weight_orientation="conv1d",
        attn_layout="fused_qkv",
    ),
    "gpt_neox": ModelFamily(
        name="gpt_neox",
        model_types=("gpt_neox",),
        decoder_layer_class_suffixes=("GPTNeoXLayer",),
        decoder_layers_path="gpt_neox.layers",
        embed_key="gpt_neox.embed_in.weight",
        lm_head_key="embed_out.weight",
        norm_key="gpt_neox.final_layer_norm",
        mlp_suffixes={
            "dense": "dense_h_to_4h",
            "down": "dense_4h_to_h",
        },
        attn_suffixes={
            "qkv": "query_key_value",
            "o": "dense",
        },
    ),
    "gptj": ModelFamily(
        name="gptj",
        model_types=("gptj",),
        decoder_layer_class_suffixes=("GPTJBlock",),
        decoder_layers_path="transformer.h",
        embed_key="transformer.wte.weight",
        lm_head_key="transformer.wte.weight",
        norm_key="transformer.ln_f",
        mlp_suffixes={
            "up": "fc_in",
            "down": "fc_out",
        },
        attn_suffixes={
            "qkv": "q_proj",  # GPT-J uses separate q/k/v but with fused rotary
            "k": "k_proj",
            "v": "v_proj",
            "o": "out_proj",
        },
        tie_weights=True,
    ),
    "bloom": ModelFamily(
        name="bloom",
        model_types=("bloom",),
        decoder_layer_class_suffixes=("BloomBlock",),
        decoder_layers_path="transformer.h",
        embed_key="transformer.word_embeddings.weight",
        lm_head_key="transformer.word_embeddings.weight",  # tied
        norm_key="transformer.ln_f",
        mlp_suffixes={
            "dense": "dense_h_to_4h",
            "down": "dense_4h_to_h",
        },
        attn_suffixes={
            "query": "query_key_value",
            "o": "dense",
        },
        tie_weights=True,
        num_key_value_heads_key=None,
    ),
}


def _resolve_text_config(config: dict) -> dict:
    """If a model nests its text config (Gemma-4, multimodal layouts), return
    the nested dict; otherwise return the top-level dict."""
    if isinstance(config.get("text_config"), dict):
        return config["text_config"]
    return config


def list_model_families() -> list[str]:
    return list(REGISTRY.keys())


def get_model_family(name: str) -> ModelFamily:
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown model family '{name}'. Available: {', '.join(list_model_families())}"
        )
    return REGISTRY[name]


def detect_model_family(config: dict, state_dict_keys: list[str] | None = None) -> ModelFamily:
    """Auto-detect the model family from config.json and optionally safetensors keys.

    Raises ValueError if no family matches.
    """
    cfg = _resolve_text_config(config)
    model_type = cfg.get("model_type", "")

    # Direct model_type match.
    for family in REGISTRY.values():
        if model_type in family.model_types:
            return family

    # Fallback: inspect tensor names in the checkpoint.
    if state_dict_keys:
        keys_str = " ".join(state_dict_keys)
        if "transformer.wte.weight" in keys_str:
            return REGISTRY["gpt2"]
        if "transformer.word_embeddings.weight" in keys_str:
            return REGISTRY["bloom"]
        if "model.embed_tokens.weight" in keys_str and "model.layers" in keys_str:
            return REGISTRY["llama"]
        if "transformer.blocks" in keys_str:
            return REGISTRY["mpt"]
        if "gpt_neox.embed_in.weight" in keys_str:
            return REGISTRY["gpt_neox"]

    raise ValueError(
        f"Could not detect model family for model_type='{model_type}'. "
        f"Use --model-family with one of: {', '.join(list_model_families())}"
    )


def get_config_value(config: dict, family: ModelFamily, key_attr: str) -> any:
    """Read a config value, respecting `text_config` nesting if present."""
    cfg = _resolve_text_config(config)
    key = getattr(family, key_attr)
    return cfg.get(key)


def resolve_model_family(
    config: dict,
    override: str | None = None,
    state_dict_keys: list[str] | None = None,
) -> ModelFamily:
    """Resolve family from explicit override or auto-detect."""
    if override is not None:
        return get_model_family(override)
    return detect_model_family(config, state_dict_keys=state_dict_keys)
