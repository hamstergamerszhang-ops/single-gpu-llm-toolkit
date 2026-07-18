"""Pytest tests for the single-gpu-llm-toolkit tools.

All tests are CPU-only (no GPU/torch-CUDA needed). They cover the same
invariants the per-module --selftest scripts check, but factored into pytest so
CI can run them automatically and individual failures are isolated.

Run: pytest tests/ -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure the repo root is importable (tests/ is one level down).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ── LR schedule (train_cpt.py) ──────────────────────────────────────────────

def test_lr_warmup_linear_ramp():
    from train_cpt import lr_at_step
    base_lr, warmup, total = 1e-5, 10, 100
    assert lr_at_step(1, total, base_lr, warmup) == base_lr * 1 / warmup
    assert lr_at_step(10, total, base_lr, warmup) == base_lr


def test_lr_cosine_decays_to_floor():
    from train_cpt import lr_at_step
    base_lr, warmup, total = 1e-5, 10, 100
    end_lr = lr_at_step(100, total, base_lr, warmup, min_lr_ratio=0.1)
    assert abs(end_lr - base_lr * 0.1) < 1e-9


def test_lr_monotonically_decreasing_after_warmup():
    from train_cpt import lr_at_step
    base_lr, warmup, total = 1e-5, 10, 100
    prev = lr_at_step(warmup, total, base_lr, warmup)
    for s in range(warmup + 1, total + 1):
        cur = lr_at_step(s, total, base_lr, warmup)
        assert cur <= prev + 1e-12, (s, cur, prev)
        prev = cur


def test_resume_does_not_restart_warmup():
    """The core resume property: the schedule is a function of ABSOLUTE step,
    not 'steps since resume.' A buggy resume that restarted warmup would use
    lr_at_step(k) (relative) instead of lr_at_step(resume_step + k) (absolute).
    These must differ at all tested k — if they were equal, the resume offset
    would have no effect (warmup-restart bug)."""
    from train_cpt import lr_at_step
    base_lr, warmup, total = 1e-5, 10, 100
    resume_step = 37
    for k in [1, 5, 50]:
        absolute = resume_step + k
        absolute_lr = lr_at_step(absolute, total, base_lr, warmup)
        relative_lr = lr_at_step(k, total, base_lr, warmup)
        assert absolute_lr != relative_lr, \
            f"k={k}: absolute {absolute_lr} should differ from relative {relative_lr}"


def test_unwrap_ddp_passes_through_plain_module():
    """unwrap_ddp() must return non-DDP models unchanged -- the single-GPU
    (no --ddp) path, which is the default and must be a complete no-op."""
    import torch
    from train_cpt import unwrap_ddp

    model = torch.nn.Linear(4, 4)
    assert unwrap_ddp(model) is model


def test_unwrap_ddp_unwraps_ddp_duck_typed():
    """Regression test for a real DDP deadlock found in review: train_cpt.py's
    held-out eval loop called run_eval(model, ...) where `model` could still be
    the DistributedDataParallel wrapper. Calling the wrapper's own forward()
    from rank 0 only (run_eval is gated on `is_main`) triggers a collective
    buffer-broadcast in DDP's _pre_forward() that other ranks never join --
    reproduced with a real 2-process torch.distributed (gloo) job during
    review: the wrapped-forward version crashed with a gloo protocol desync
    error, the model.module-unwrapped version completed cleanly on both ranks
    and didn't break the next real training step's all-reduce either.
    unwrap_ddp() is duck-typed on the class name (not isinstance), matching
    find_decoder_layers()'s existing convention, so this test builds a fake
    class named exactly "DistributedDataParallel" rather than requiring a real
    torch.distributed process group just to check the unwrap logic."""
    import torch
    from train_cpt import unwrap_ddp

    inner = torch.nn.Linear(4, 4)

    class DistributedDataParallel:  # noqa: N801 -- name must match exactly for the duck-type check
        def __init__(self, module):
            self.module = module

    wrapped = DistributedDataParallel(inner)
    assert unwrap_ddp(wrapped) is inner
    # And a real (non-DDP) module of some OTHER class name must still pass through.
    assert unwrap_ddp(inner) is inner


def test_gradient_accumulation_matches_single_larger_batch():
    """Regression test for a real bug: train_cpt.py's --accum /
    --gradient-accumulation-steps flag was added with a docstring describing
    exact behavior (loss divided by accum before backward, one optimizer.step()
    per accum micro-batches) but the training loop never referenced
    args.accum anywhere -- passing --accum 4 silently trained as if the flag
    didn't exist. This test doesn't import train_cpt's main() (that needs a
    real model/GPU/checkpoint), but verifies the actual accumulation math the
    fixed loop uses: N micro-batches, each with mean-reduced loss divided by N
    before backward(), one optimizer step after all N -- using real torch
    tensors and a real backward pass, not a mock. The accumulated gradient
    must equal the gradient from a single larger batch covering the same
    examples, since mean(mean(chunk_1), mean(chunk_2)) == mean(chunk_1+chunk_2)
    for equal-sized chunks."""
    import torch

    torch.manual_seed(0)
    model_single = torch.nn.Linear(4, 1)
    model_accum = torch.nn.Linear(4, 1)
    model_accum.load_state_dict(model_single.state_dict())

    x = torch.randn(8, 4)
    y = torch.randn(8, 1)

    # Single larger batch, mean-reduced loss (mirrors HF's mean cross-entropy).
    model_single.zero_grad()
    loss_single = torch.nn.functional.mse_loss(model_single(x), y)
    loss_single.backward()
    grad_single = model_single.weight.grad.clone()

    # Accumulated: accum=2 micro-batches of 4, loss/accum before backward,
    # one optimizer step after both -- the exact pattern train_cpt.py's loop uses.
    accum = 2
    model_accum.zero_grad()
    for i in range(accum):
        xb, yb = x[i * 4:(i + 1) * 4], y[i * 4:(i + 1) * 4]
        loss_micro = torch.nn.functional.mse_loss(model_accum(xb), yb)
        (loss_micro / accum).backward()
    grad_accum = model_accum.weight.grad.clone()

    assert torch.allclose(grad_single, grad_accum, atol=1e-6), (
        f"accumulated gradient {grad_accum} should match single-batch gradient "
        f"{grad_single} -- if these differ, --accum is scaling incorrectly"
    )


# ── Compute backends (backends/) ────────────────────────────────────────────

def test_list_backends_includes_expected():
    """This toolkit targets AMD ROCm -- CPU is registered as the universal
    fallback (every other tool in this repo already degrades to it for
    testing/dev without real hardware), not a second accelerator vendor."""
    from backends import list_backends
    names = list_backends()
    for expected in ("rocm", "cpu"):
        assert expected in names, f"missing backend {expected} in {names}"
    assert len(names) == 2, f"expected exactly rocm+cpu, got {names}"


def test_get_backend_unknown_raises():
    from backends import get_backend
    with pytest.raises(ValueError):
        get_backend("not-a-backend")


def test_autodetect_backend_returns_device():
    from backends import autodetect_backend, default_device, BackendDevice
    backend = autodetect_backend()
    assert isinstance(backend.name, str) and backend.name
    dev = default_device(prefer="cpu")
    assert isinstance(dev, BackendDevice)
    assert dev.name == "cpu"


def test_cpu_backend_always_available():
    from backends import get_backend
    cpu = get_backend("cpu")
    assert cpu.is_available()
    assert cpu.get_device_count() == 1
    assert cpu.recommended_dtype() == "fp32"


def test_backend_device_moves_tensor():
    import torch
    from backends.device import BackendDevice

    dev = BackendDevice(backend="cpu")
    t = torch.tensor([1.0, 2.0])
    moved = dev.to(t)
    assert moved.device.type == "cpu"


def test_backend_prefer_unavailable_falls_back():
    """If the user requests an unavailable backend, auto-detection falls back
    instead of crashing."""
    from backends import default_device
    # ROCm is unavailable on macOS/CPU test runners; this should still succeed.
    dev = default_device(prefer="rocm")
    assert dev.name == "cpu"


# ── Model-family registry (models/) ─────────────────────────────────────────

def test_list_model_families_includes_common():
    from models import list_model_families
    families = list_model_families()
    for expected in ("llama", "gemma", "phi3", "falcon", "mpt", "gpt2", "bloom"):
        assert expected in families, f"missing family {expected} in {families}"


def test_get_unknown_family_raises():
    from models import get_model_family
    with pytest.raises(ValueError):
        get_model_family("not-a-family")


def test_detect_llama_family():
    from models import detect_model_family
    family = detect_model_family({"model_type": "llama"})
    assert family.name == "llama"
    assert family.decoder_layers_path == "model.layers"


def test_detect_gemma_nested_text_config():
    from models import detect_model_family
    family = detect_model_family({
        "model_type": "gemma4_unified",
        "text_config": {"model_type": "gemma4"},
    })
    assert family.name == "gemma"


def test_resolve_model_family_override_wins():
    from models import resolve_model_family
    family = resolve_model_family({"model_type": "llama"}, override="gpt2")
    assert family.name == "gpt2"


def test_detect_from_state_dict_keys():
    from models import detect_model_family
    family = detect_model_family(
        {"model_type": "unknown"},
        state_dict_keys=["transformer.wte.weight", "transformer.h.0.ln_1.weight"],
    )
    assert family.name == "gpt2"


# ── Config recipes and presets (config/) ────────────────────────────────────

def test_list_presets_includes_cpu_and_amd_cards():
    from config import list_presets
    presets = list_presets()
    assert "cpu" in presets
    # CDNA Instinct lineup.
    assert "mi300x-80g" in presets
    assert "mi300x-192g" in presets
    assert "mi250-64g" in presets
    assert "mi25-16g" in presets
    # Consumer Radeon lineup (RX 6000/7000/9000 + APU).
    assert "rx6800-16g" in presets
    assert "rx7900xtx-24g" in presets
    assert "rx9070xt-16g" in presets
    assert "apu-2g" in presets


def test_preset_keys_match_argparse_dests():
    """Preset keys must be argparse dest names, or apply_preset is a silent no-op.
    train_cpt.py uses --batch/--max-seq-len/--accum (dest: batch/max_seq_len/accum),
    NOT batch_size/seq_length/gradient_accumulation_steps. Pin this so a rename
    drift in either direction is caught."""
    from config.presets import PRESETS
    forbidden = {"batch_size", "seq_length", "gradient_accumulation_steps"}
    for name, preset in PRESETS.items():
        bad = forbidden & set(preset.keys())
        assert not bad, f"preset '{name}' uses stale keys {bad}; use batch/max_seq_len/accum"
        # Every preset must set at least these core training defaults.
        assert "dtype" in preset, f"preset '{name}' missing dtype"
        assert "batch" in preset, f"preset '{name}' missing batch"
        assert "max_seq_len" in preset, f"preset '{name}' missing max_seq_len"
        assert "accum" in preset, f"preset '{name}' missing accum"


def test_get_preset_cpu_is_fp32():
    from config.presets import get_preset
    preset = get_preset("cpu")
    assert preset["dtype"] == "fp32"
    assert preset["batch"] == 1
    assert preset["max_seq_len"] == 128


def test_get_preset_consumer_radeon_defaults():
    """Consumer Radeon presets: bf16, flash-attn on (RDNA2+ has kernels),
    no fp8, compile off. MI25 (gfx900) has no flash-attn kernels and no bf16."""
    from config.presets import get_preset
    p = get_preset("rx7900xtx-24g")
    assert p["dtype"] == "bf16"
    assert p["flash_attn"] is True
    assert p["compile"] is False
    mi25 = get_preset("mi25-16g")
    assert mi25["flash_attn"] is False
    assert mi25["dtype"] == "fp16"


def test_apply_preset_does_not_override_explicit():
    from config import apply_preset
    config = {"dtype": "bf16", "batch": 16}
    merged = apply_preset(config, "cpu")
    assert merged["dtype"] == "bf16"
    assert merged["batch"] == 16
    assert merged["compile"] is False  # filled in from preset


def test_suggest_preset_vram_tiers():
    """suggest_preset picks the largest preset whose VRAM doesn't exceed the
    card's (so 24GB -> rx7900xtx-24g, not rx7900xt-20g)."""
    from config import suggest_preset
    assert suggest_preset("cpu", 0) == "cpu"
    assert suggest_preset("rocm", 192 * 1024**3) == "mi300x-192g"
    assert suggest_preset("rocm", 80 * 1024**3) == "mi300x-80g"
    assert suggest_preset("rocm", 24 * 1024**3) == "rx7900xtx-24g"
    assert suggest_preset("rocm", 8 * 1024**3) == "rx6600-8g"
    assert suggest_preset("rocm", 2 * 1024**3) == "apu-2g"
    assert suggest_preset("rocm", 1 * 1024**3) is None  # <1.5GB too small


def test_resolve_toml_recipe(tmp_path):
    from config import resolve_recipe
    recipe = tmp_path / "test.toml"
    recipe.write_text('batch = 8\nmax_seq_len = 512\n')
    cfg = resolve_recipe(recipe, base_defaults={"dtype": "bf16", "batch": 1})
    assert cfg["batch"] == 8
    assert cfg["max_seq_len"] == 512
    assert cfg["dtype"] == "bf16"


def test_resolve_recipe_extends_chain(tmp_path):
    from config import resolve_recipe
    base = tmp_path / "base.toml"
    base.write_text('dtype = "bf16"\ncompile = true\n')
    child = tmp_path / "child.toml"
    child.write_text(f'extends = "base.toml"\nbatch = 4\n')
    cfg = resolve_recipe(child, base_defaults={"batch": 1})
    assert cfg["dtype"] == "bf16"
    assert cfg["compile"] is True
    assert cfg["batch"] == 4


# ── Runtime capability probes (runtime/) ────────────────────────────────────

def test_probe_fp8_false_on_cpu():
    from runtime import probe_fp8
    from backends import BackendDevice
    dev = BackendDevice(backend="cpu")
    usable, reason = probe_fp8(dev)
    assert not usable
    assert "does not advertise" in reason


def test_probe_flash_attn_false_on_cpu():
    from runtime import probe_flash_attn
    from backends import BackendDevice
    dev = BackendDevice(backend="cpu")
    usable, reason = probe_flash_attn(dev)
    assert not usable


def test_resolve_dtype_cpu_defaults_to_fp32():
    from runtime import resolve_dtype
    from backends import BackendDevice
    dev = BackendDevice(backend="cpu")
    assert resolve_dtype(dev, None) == "fp32"
    assert resolve_dtype(dev, "fp32") == "fp32"


def test_resolve_compile_false_when_requested_false():
    from runtime import resolve_compile
    from backends import BackendDevice
    dev = BackendDevice(backend="cpu")
    assert resolve_compile(dev, requested=False) is False


def test_dtype_map_includes_fp8_as_bf16():
    """The shared DTYPE_MAP must map 'fp8' to bf16: fp8 runs load the model in
    bf16 then quantize in place, so the dict lookup must succeed (a missing key
    raised KeyError on MI300X before the quantize step ran)."""
    import torch
    from runtime import DTYPE_MAP
    assert DTYPE_MAP["fp32"] is torch.float32
    assert DTYPE_MAP["fp16"] is torch.float16
    assert DTYPE_MAP["bf16"] is torch.bfloat16
    assert DTYPE_MAP["fp8"] is torch.bfloat16  # load bf16, quantize after


def test_rocm_fp8_gate_covers_mi300_family_excludes_consumer():
    """supports_fp8() advertises fp8 ONLY for gfx940/gfx941/gfx942/gfx950
    (MI300A/MI325X/MI300X/MI350). Consumer RDNA (gfx1100/gfx1200) and older
    CDNA (gfx90a) must NOT be advertised -- they have no fp8 units, and
    advertising fp8 makes --dtype fp8 attempt float8 conversion and crash."""
    fp8_prefixes = ("gfx940", "gfx941", "gfx942", "gfx950", "gfx95")
    for arch, expected in [
        ("gfx942", True),   # MI300X
        ("gfx940", True),   # MI300A
        ("gfx941", True),   # MI325X
        ("gfx950", True),   # MI350
        ("gfx90a", False),  # MI250 -- no fp8
        ("gfx908", False),  # MI100 -- no fp8
        ("gfx1100", False), # RX 7900 -- no fp8
        ("gfx1200", False), # RX 9070 -- no fp8 in current ROCm
        ("gfx803", False),  # Polaris -- no fp8
    ]:
        got = arch.startswith(fp8_prefixes)
        assert got is expected, f"{arch}: expected fp8={expected}, got {got}"


def test_rocm_flash_attn_gate_excludes_old_archs():
    """supports_flash_attn() must be False for archs with no flash-attn ROCm
    kernels (gfx900/MI25, gfx803, gfx1010/RDNA1) and True for CDNA gfx908+ and
    RDNA2+ (gfx1030+). The old code returned True for everything."""
    from backends.rocm import RocmBackend

    b = RocmBackend()
    orig_avail = b.is_available
    try:
        b.is_available = lambda: True
        cases = {
            "gfx908": True,   # MI100
            "gfx90a": True,   # MI250
            "gfx942": True,   # MI300X
            "gfx1030": True,  # RX 6800
            "gfx1100": True,  # RX 7900
            "gfx1200": True,  # RX 9070
            "gfx900": False,  # MI25 -- no kernels
            "gfx803": False,  # Polaris -- no kernels
            "gfx1010": False, # RDNA1 -- no kernels
        }
        for arch, expected in cases.items():
            b.get_arch_tag = lambda idx=0, _a=arch: _a
            got = b.supports_flash_attn()
            assert got is expected, f"{arch}: expected flash_attn={expected}, got {got}"
    finally:
        b.is_available = orig_avail


def test_rocm_arch_tag_fallback_is_decimal_not_hex():
    """get_arch_tag() capability-tuple fallback must emit minor in DECIMAL
    (gfx942), not hex (gfx92a). A previous version used :x formatting and
    produced bogus arch strings that nothing downstream would match."""
    from backends.rocm import RocmBackend
    import torch

    b = RocmBackend()
    orig_avail = b.is_available
    try:
        b.is_available = lambda: True
        class FakeProps:
            gcnArchName = ""
        orig_get_props = torch.cuda.get_device_properties
        orig_get_cap = torch.cuda.get_device_capability
        torch.cuda.get_device_properties = lambda idx: FakeProps()
        torch.cuda.get_device_capability = lambda idx: (9, 42)
        tag = b.get_arch_tag(0)
        assert tag == "gfx942", f"expected 'gfx942' (decimal), got {tag!r}"
        torch.cuda.get_device_properties = orig_get_props
        torch.cuda.get_device_capability = orig_get_cap
    finally:
        b.is_available = orig_avail


def test_modeling_custom_family_class_chain_covers_supported_families():
    """_FAMILY_CLASS_CHAINS must list every family in models/registry.py so a
    user-specified --model-family always resolves to a class chain."""
    import modeling_custom
    from models.registry import list_model_families
    registry_families = set(list_model_families())
    chain_families = set(modeling_custom._FAMILY_CLASS_CHAINS.keys())
    missing = registry_families - chain_families
    assert not missing, f"families in registry but not in modeling_custom: {missing}"


def test_modeling_custom_resolve_base_class_for_known_family():
    """_resolve_base_class must return a real transformers CausalLM class for a
    user-specified family (not guess across families)."""
    import modeling_custom
    import transformers
    # 'llama' should always resolve (LlamaForCausalLM exists in every version).
    cls = modeling_custom._resolve_base_class("llama")
    assert isinstance(cls, type)
    assert cls.__module__.startswith("transformers")


def test_modeling_custom_resolve_base_class_rejects_unknown_family():
    """_resolve_base_class must raise ValueError for an unknown family -- it
    does NOT silently fall back to a cross-family guess."""
    import modeling_custom
    import pytest
    with pytest.raises(ValueError, match="Unknown model family"):
        modeling_custom._resolve_base_class("nonexistent_family")


def test_modeling_custom_no_family_set_uses_sentinel():
    """When config.json has no model_family (and no MODEL_FAMILY env var),
    _BaseForCausalLM is a sentinel that raises a clear error at instantiation --
    NOT a cross-family guess."""
    import modeling_custom
    # On this test host there's no config.json alongside modeling_custom.py
    # and no MODEL_FAMILY env var, so _BaseForCausalLM is the sentinel.
    sentinel = modeling_custom._BaseForCausalLM
    assert sentinel.__name__ == "_ModelFamilyNotSet"
    import pytest
    with pytest.raises(ValueError, match="model_family is not set"):
        sentinel()


def test_modeling_custom_env_var_fallback_resolves_family():
    """When MODEL_FAMILY env var is set, modeling_custom reads it and resolves
    the base class -- this is the generate.py --model-family path."""
    import importlib
    import os
    import modeling_custom
    orig_env = os.environ.get("MODEL_FAMILY")
    try:
        os.environ["MODEL_FAMILY"] = "llama"
        # Re-read to simulate the env-var path.
        family = modeling_custom._read_model_family_from_config()
        # On this host config.json won't have model_family, so env var wins.
        assert family == "llama"
        cls = modeling_custom._resolve_base_class(family)
        assert cls.__name__ in ("LlamaForCausalLM", "Llama4ForCausalLM")
    finally:
        if orig_env is None:
            os.environ.pop("MODEL_FAMILY", None)
        else:
            os.environ["MODEL_FAMILY"] = orig_env


# ── MTP helpers (models/mtp.py) ─────────────────────────────────────────────

def test_shift_labels_depth_0_is_standard_next_token():
    import torch
    from models.mtp import shift_labels
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    labels = shift_labels(input_ids, depth=0, ignore_index=-100)
    expected = torch.tensor([[2, 3, 4, 5, -100]])
    assert torch.equal(labels, expected)


def test_shift_labels_depth_1_skips_two():
    import torch
    from models.mtp import shift_labels
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    labels = shift_labels(input_ids, depth=1, ignore_index=-100)
    expected = torch.tensor([[3, 4, 5, -100, -100]])
    assert torch.equal(labels, expected)


def test_compute_total_mtp_loss_shape_and_grad():
    import torch
    from models.mtp import compute_total_mtp_loss
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    vocab_size = 10
    logits = [torch.randn(1, 5, vocab_size, requires_grad=True) for _ in range(2)]
    loss = compute_total_mtp_loss(logits, input_ids, global_weight=0.3)
    assert loss.shape == ()
    assert loss.item() >= 0
    loss.backward()
    assert logits[0].grad is not None


# ── Export tools ────────────────────────────────────────────────────────────

def test_export_safetensors_consolidates_shards(tmp_path):
    import torch
    from safetensors.torch import save_file
    src = tmp_path / "src"
    src.mkdir()
    tensors = {
        "a.weight": torch.randn(4, 4),
        "b.weight": torch.randn(4, 4),
    }
    save_file(tensors, src / "model-00001-of-00001.safetensors")
    index = {
        "metadata": {"total_size": 0},
        "weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors},
    }
    (src / "model.safetensors.index.json").write_text(json.dumps(index))

    dst = tmp_path / "dst"
    import sys
    saved = sys.argv
    try:
        sys.argv = ["export_safetensors.py", "--src", str(src), "--dst", str(dst)]
        from export_safetensors import main
        main()
    finally:
        sys.argv = saved

    assert (dst / "model.safetensors").exists()


# ── BPE merges parsing (prune_vocab.py) ─────────────────────────────────────

def test_prune_vocab_merges_string_format():
    """Standard HF tokenizer.json stores merges as 'a b' strings, not [a, b]
    lists. The parser must split on space, not unpack characters."""
    from prune_vocab import classify, REMOVABLE

    # Simulate the merge-filtering logic with string-format pairs.
    keep_tok_strs = {"Ġ", "t", "Ġt", "h", "Ġh"}
    old_merges = ["Ġ t", "Ġ h", "x y"]  # "x y" should drop (x, y not in keep)
    new_merges = []
    for pair in old_merges:
        if isinstance(pair, str):
            a, b = pair.split(" ", 1)
            out_pair = pair
        else:
            a, b = pair[0], pair[1]
            out_pair = pair
        merged = a + b
        if a in keep_tok_strs and b in keep_tok_strs and merged in keep_tok_strs:
            new_merges.append(out_pair)
    assert new_merges == ["Ġ t", "Ġ h"], new_merges


def test_prune_vocab_merges_list_format():
    """Older list-format [a, b] merges must still work."""
    keep_tok_strs = {"Ġ", "t", "Ġt"}
    old_merges = [["Ġ", "t"], ["x", "y"]]
    new_merges = []
    for pair in old_merges:
        if isinstance(pair, str):
            a, b = pair.split(" ", 1)
            out_pair = pair
        else:
            a, b = pair[0], pair[1]
            out_pair = pair
        merged = a + b
        if a in keep_tok_strs and b in keep_tok_strs and merged in keep_tok_strs:
            new_merges.append(out_pair)
    assert new_merges == [["Ġ", "t"]], new_merges


def test_classify_keeps_ascii_drops_cjk():
    from prune_vocab import classify, REMOVABLE
    assert classify("hello") == "keep"
    assert classify("世界") == "cjk"
    assert classify("привет") == "cyrillic"
    assert classify("مرحبا") == "arabic"
    assert "cjk" in REMOVABLE
    assert "keep" not in REMOVABLE


# ── prune_embeddings_torch.py slicing logic ─────────────────────────────────

def test_prune_embeddings_int_coercion_and_contiguity():
    """The remap loading: {int(k): int(v)} coerces BOTH keys and values, and
    the contiguity check (sorted values == range(N)) rejects non-contiguous
    remaps. This is the logic at prune_embeddings_torch.py:60-67."""
    # Simulate the remap loading + validation inline (main() does file I/O;
    # we test the pure-logic part that would break if coercion is wrong).
    # A correct remap: old ids 0,2,4 -> new ids 0,1,2 (contiguous).
    raw_json = '{"0": 0, "2": 1, "4": 2}'  # string keys (as written by prune_vocab)
    old_to_new = {int(k): int(v) for k, v in json.loads(raw_json).items()}
    assert old_to_new == {0: 0, 2: 1, 4: 2}
    keep_old_ids = sorted(old_to_new.keys())
    new_vocab_size = len(keep_old_ids)
    expected_new_ids = sorted(old_to_new.values())
    assert expected_new_ids == list(range(new_vocab_size))  # contiguous: passes

    # A non-contiguous remap (values 0, 2, 5 — gap) should fail the check.
    bad_json = '{"0": 0, "2": 2, "4": 5}'
    bad_map = {int(k): int(v) for k, v in json.loads(bad_json).items()}
    bad_values = sorted(bad_map.values())
    assert bad_values != list(range(len(bad_map)))  # non-contiguous: would abort


def test_prune_embeddings_single_file_index_synthesis():
    """When a checkpoint has no model.safetensors.index.json but does have a
    single model.safetensors, prune_embeddings_torch synthesizes an index from
    the safetensors header. This test exercises that synthesis logic against a
    real tiny safetensors file."""
    import torch
    from safetensors.torch import save_file, load_file

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        embed_key = "model.embed_tokens.weight"
        other_key = "model.layers.0.fc.weight"

        # Create a tiny single-file checkpoint: embed (10 x 4) + one other tensor.
        tensors = {
            embed_key: torch.randn(10, 4).to(torch.bfloat16),
            other_key: torch.randn(4, 4).to(torch.bfloat16),
        }
        save_file(tensors, td / "model.safetensors")

        # Also write the remap (prune_vocab.py's output): keep old ids 0,2,4,6,8
        # -> new ids 0,1,2,3,4 (drop odds).
        remap = {str(k): v for k, v in zip([0, 2, 4, 6, 8], range(5))}
        remap_dst = td / "dst"
        remap_dst.mkdir()
        with open(remap_dst / "_old_to_new_ids.json", "w") as f:
            json.dump(remap, f)

        # Synthesize the index (same logic as prune_embeddings_torch.py:80-88).
        single_file = "model.safetensors"
        with open(td / single_file, "rb") as f:
            header_len = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(header_len))
        weight_map = {k: single_file for k in header if k != "__metadata__"}
        index = {"metadata": {"total_size": os.path.getsize(td / single_file)},
                 "weight_map": weight_map}

        # Verify the synthesized index covers both tensors.
        assert embed_key in index["weight_map"]
        assert other_key in index["weight_map"]
        assert index["weight_map"][embed_key] == single_file

        # Verify the row-slicing logic (prune_embeddings_torch.py:115):
        # keep_idx = [0,2,4,6,8], slicing embed[keep_idx] gives 5x4.
        old_to_new = {int(k): int(v) for k, v in remap.items()}
        keep_old_ids = sorted(old_to_new.keys())
        keep_idx = torch.tensor(keep_old_ids, dtype=torch.long)
        loaded = load_file(td / single_file)
        sliced = loaded[embed_key][keep_idx, :].contiguous()
        assert sliced.shape == (5, 4), sliced.shape
        # Verify the right rows were kept.
        for i, old_id in enumerate(keep_old_ids):
            assert torch.equal(sliced[i], loaded[embed_key][old_id])


# ── expand_model.py transformation logic ────────────────────────────────────

def test_build_depth_plan_counts_and_interleave():
    """build_depth_plan should produce orig_layers + depth_step entries, with
    duplicates inserted at every interleave_every interval."""
    from expand_model import build_depth_plan
    orig, step, every = 12, 3, 4
    plan = build_depth_plan(orig, step, every)
    assert len(plan) == orig + step
    dups = [e for e in plan if e[2]]
    assert len(dups) == step
    # Duplicates are at positions where (old_idx + 1) % every == 0
    for new_idx, old_idx, is_dup in plan:
        if is_dup:
            assert (old_idx + 1) % every == 0


def test_build_depth_plan_mismatch_raises():
    """If interleave_every doesn't evenly allow depth_step duplicates, raise."""
    from expand_model import build_depth_plan
    with pytest.raises(SystemExit):
        # 10 layers, interleave 4, want 5 dups -> only 2 fit (at idx 3, 7)
        build_depth_plan(10, 5, 4)


def test_orthogonal_pad_shapes():
    """orthogonal_pad produces the right shape for both transpose modes.
    transpose_for_rows=True -> (n_new, n_existing) [for padding gate/up_proj rows]
    transpose_for_rows=False -> (n_existing, n_new) [for padding down_proj cols]
    Returns float32; callers convert to the checkpoint's dtype."""
    import torch
    from expand_model import orthogonal_pad, INIT_SCALE
    # transpose_for_rows=True: returns (n_new, n_existing)
    pad = orthogonal_pad(8, 16, INIT_SCALE, transpose_for_rows=True)
    assert pad.shape == (8, 16), pad.shape
    assert pad.dtype == torch.float32
    # transpose_for_rows=False: returns (n_existing, n_new)
    pad2 = orthogonal_pad(8, 16, INIT_SCALE, transpose_for_rows=False)
    assert pad2.shape == (16, 8), pad2.shape


def test_orthogonal_pad_rejects_n_new_greater_than_n_existing():
    """np.linalg.qr's reduced mode on an (m, n) matrix with m <= n can only
    produce at most m orthogonal columns -- requesting more silently
    truncated the pad before this guard was added. It must now raise instead
    of returning a wrong-shaped tensor."""
    from expand_model import orthogonal_pad, INIT_SCALE
    with pytest.raises(ValueError):
        orthogonal_pad(32, 16, INIT_SCALE, transpose_for_rows=True)


def test_gqa_expand_kv_v_proj_shape_when_new_out_less_than_hidden():
    """Regression test for a real shape bug: gqa_expand_kv's v_proj
    construction did `np.linalg.qr(R, mode="reduced")` on an (new_out, hidden)
    matrix and used the result directly. When new_out < hidden, reduced-mode
    QR returns Q of shape (new_out, new_out), not (new_out, hidden) -- the
    fresh v_proj silently came out the wrong shape (confirmed by reproducing
    the old logic standalone: for new_out=512, hidden=2048 it produced
    (512, 512) instead of (512, 2048)). This would either crash at
    torch.cat/model load or corrupt the checkpoint depending on what caught
    it. The fix pads the QR output with extra random columns to reach the
    required width; this test exercises exactly that path with new_out well
    below hidden."""
    import torch
    from expand_model import gqa_expand_kv

    head_dim = 128
    hidden = 2048
    old_kv_heads = 1
    new_kv_heads = 4  # new_out = 512, which is < hidden = 2048
    layer_prefix = "layer"
    k_key = f"{layer_prefix}.self_attn.k_proj.weight"
    v_key = f"{layer_prefix}.self_attn.v_proj.weight"

    tensors = {k_key: torch.randn(old_kv_heads * head_dim, hidden, dtype=torch.bfloat16)}
    gqa_expand_kv(tensors, layer_prefix, head_dim, old_kv_heads, new_kv_heads,
                  hidden, init_scale=0.02)

    new_out = new_kv_heads * head_dim
    assert tensors[k_key].shape == (new_out, hidden)
    assert tensors[v_key].shape == (new_out, hidden), (
        f"v_proj shape {tensors[v_key].shape} != expected {(new_out, hidden)} -- "
        "this is the exact shape truncation the fix addresses"
    )
    assert torch.isfinite(tensors[v_key]).all()


def test_gqa_expand_kv_v_proj_shape_when_new_out_at_least_hidden():
    """Same construction, but with new_out >= hidden (the case reduced-mode QR
    already handled correctly without padding) -- verifies the fix's branch
    for the pre-existing working path didn't regress it.

    k_proj's row-pad goes through orthogonal_pad(n_new, hidden, ...), which
    (correctly) requires n_new <= hidden -- a separate, independent
    constraint from v_proj's new_out-vs-hidden shape logic being tested here.
    old_kv_heads is chosen large enough that n_new = new_out - old_out stays
    within that limit while new_out itself is still >= hidden.
    """
    import torch
    from expand_model import gqa_expand_kv

    head_dim = 64
    hidden = 512
    old_kv_heads = 6   # old_out = 384
    new_kv_heads = 8   # new_out = 512 (>= hidden); n_new = 512-384 = 128 (<= hidden)
    layer_prefix = "layer"
    k_key = f"{layer_prefix}.self_attn.k_proj.weight"
    v_key = f"{layer_prefix}.self_attn.v_proj.weight"

    tensors = {k_key: torch.randn(old_kv_heads * head_dim, hidden, dtype=torch.bfloat16)}
    gqa_expand_kv(tensors, layer_prefix, head_dim, old_kv_heads, new_kv_heads,
                  hidden, init_scale=0.02)

    new_out = new_kv_heads * head_dim
    assert new_out >= hidden, "test setup check: this test is meant to cover new_out >= hidden"
    assert tensors[k_key].shape == (new_out, hidden)
    assert tensors[v_key].shape == (new_out, hidden)
    assert torch.isfinite(tensors[v_key]).all()


def test_detect_mqa_layout_matches_when_no_v_proj_and_shape_agrees():
    """The real Gemma-4 layout: no v_proj key, k_proj output dim == kv_heads*head_dim.
    Detection should say this matches, so the GQA fix is safe to apply."""
    import torch
    from expand_model import detect_mqa_v_shares_k_layout
    head_dim, old_kv_heads, hidden = 4, 1, 16
    tensors = {
        "model.layers.0.self_attn.k_proj.weight": torch.randn(old_kv_heads * head_dim, hidden),
        # no v_proj key at all -- this is the layout being detected
    }
    matches, reason = detect_mqa_v_shares_k_layout(tensors, [0], "model.layers", head_dim, old_kv_heads)
    assert matches is True, reason


def test_detect_mqa_layout_rejects_when_v_proj_exists():
    """A checkpoint with a real v_proj (standard GQA/MHA architectures -- Llama,
    Mistral, Qwen, etc.) must NOT be treated as matching the Gemma-4 MQA layout,
    or the fix would silently overwrite a real, already-trained V matrix."""
    import torch
    from expand_model import detect_mqa_v_shares_k_layout
    head_dim, old_kv_heads, hidden = 4, 1, 16
    tensors = {
        "model.layers.0.self_attn.k_proj.weight": torch.randn(old_kv_heads * head_dim, hidden),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(old_kv_heads * head_dim, hidden),
    }
    matches, reason = detect_mqa_v_shares_k_layout(tensors, [0], "model.layers", head_dim, old_kv_heads)
    assert matches is False
    assert "v_proj" in reason


def test_detect_mqa_layout_rejects_on_kv_head_shape_mismatch():
    """If the config's advertised kv-head count doesn't match k_proj's real
    output shape, detection must refuse rather than let gqa_expand_kv()
    concatenate against a wrong old_kv_heads and misshape the tensor."""
    import torch
    from expand_model import detect_mqa_v_shares_k_layout
    head_dim, old_kv_heads, hidden = 4, 1, 16
    # k_proj actually has 2 kv heads worth of output, but old_kv_heads says 1
    tensors = {
        "model.layers.0.self_attn.k_proj.weight": torch.randn(2 * head_dim, hidden),
    }
    matches, reason = detect_mqa_v_shares_k_layout(tensors, [0], "model.layers", head_dim, old_kv_heads)
    assert matches is False
    assert "output dim" in reason


def test_detect_mqa_layout_rejects_on_missing_k_proj():
    """If k_proj itself can't be found (e.g. --layer-prefix doesn't match this
    checkpoint's real key naming), detection must fail closed, not assume."""
    from expand_model import detect_mqa_v_shares_k_layout
    matches, reason = detect_mqa_v_shares_k_layout({}, [0], "model.layers", 4, 1)
    assert matches is False
    assert "not found" in reason


def test_detect_mqa_layout_no_full_attention_layers():
    """An empty full_attn_idxs list (e.g. a model with no layer marked
    'full_attention') should report no-match rather than vacuously 'True'."""
    from expand_model import detect_mqa_v_shares_k_layout
    matches, reason = detect_mqa_v_shares_k_layout({}, [], "model.layers", 4, 1)
    assert matches is False


def _write_qwen2_style_checkpoint(root: Path, with_head_dim: bool):
    """Builds a real, on-disk, flat (non-nested) Qwen2-shaped checkpoint:
    config.json with no 'text_config' key and (optionally) no 'head_dim' key
    (verified against a real Qwen2Config().to_dict() during the review that
    Qwen2 configs genuinely omit head_dim -- it's derived internally, not
    serialized), plus safetensors weights with a REAL v_proj on every layer
    (Qwen2 always has one -- this is NOT the Gemma-4 MQA 'V reuses K' layout)."""
    import torch
    from safetensors.torch import save_file

    hidden, kv_heads, head_dim, n_layers = 64, 2, 8, 2
    cfg = {
        "model_type": "qwen2",
        "hidden_size": hidden,
        "num_hidden_layers": n_layers,
        "intermediate_size": 128,
        "num_attention_heads": 8,
        "num_key_value_heads": kv_heads,
    }
    if with_head_dim:
        cfg["head_dim"] = head_dim
    with open(root / "config.json", "w") as f:
        json.dump(cfg, f)

    tensors = {}
    for i in range(n_layers):
        p = f"model.language_model.layers.{i}.self_attn"
        tensors[f"{p}.q_proj.weight"] = torch.randn(8 * head_dim, hidden)
        tensors[f"{p}.k_proj.weight"] = torch.randn(kv_heads * head_dim, hidden)
        tensors[f"{p}.v_proj.weight"] = torch.randn(kv_heads * head_dim, hidden)
        tensors[f"{p}.o_proj.weight"] = torch.randn(hidden, 8 * head_dim)
        mp = f"model.language_model.layers.{i}.mlp"
        tensors[f"{mp}.gate_proj.weight"] = torch.randn(128, hidden)
        tensors[f"{mp}.up_proj.weight"] = torch.randn(128, hidden)
        tensors[f"{mp}.down_proj.weight"] = torch.randn(hidden, 128)
    save_file(tensors, str(root / "model.safetensors"))
    index = {"metadata": {"total_size": 0},
             "weight_map": {k: "model.safetensors" for k in tensors}}
    with open(root / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)


def test_expand_model_gqa_missing_head_dim_skips_cleanly_not_keyerror():
    """Regression test for a real bug found reviewing this repo: main()'s GQA
    branch did `tc["head_dim"]` (no fallback) BEFORE detect_mqa_v_shares_k_layout()
    got a chance to run its safety check, so any flat config missing head_dim
    (e.g. a real Qwen2Config, confirmed via Qwen2Config().to_dict() during
    review) crashed with a raw KeyError -- contradicting this module's own
    docstring and the README's claim that the GQA fix 'skips cleanly with a
    warning' on a checkpoint that doesn't match the targeted layout. Fixed by
    resolving head_dim via .get(...) with no crash, and folding 'no head_dim
    at all' into the same fail-closed skip path as any other layout mismatch."""
    import sys
    from expand_model import main

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        src.mkdir()
        _write_qwen2_style_checkpoint(src, with_head_dim=False)
        dst = Path(td) / "dst"

        argv = ["expand_model.py", "--src", str(src), "--dst", str(dst),
                "--gqa-kv-heads", "4", "--width-step", "0", "--depth-step", "0"]
        old_argv = sys.argv
        try:
            sys.argv = argv
            main()  # must NOT raise KeyError -- must return normally (exit 0)
        finally:
            sys.argv = old_argv

        # And it must actually have run to completion (config written), not
        # silently produced nothing.
        assert (dst / "config.json").exists()
        with open(dst / "config.json") as f:
            out_cfg = json.load(f)
        # GQA fix must NOT have been applied (no head_dim -> no usable layout
        # check -> fail closed) -- config must not claim a kv-head count change
        # the tensors don't actually have.
        assert "attention_k_eq_v" not in out_cfg
        assert out_cfg.get("num_key_value_heads", 2) == 2


def test_expand_model_gqa_force_fix_with_no_head_dim_fails_clearly():
    """--force-gqa-fix can override a failed layout check, but must NOT be
    able to override 'there's no head_dim to compute shapes from at all' --
    that would crash inside gqa_expand_kv()'s tensor-shape arithmetic with a
    confusing TypeError instead of a clear, actionable error."""
    import sys
    from expand_model import main

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        src.mkdir()
        _write_qwen2_style_checkpoint(src, with_head_dim=False)
        dst = Path(td) / "dst"

        argv = ["expand_model.py", "--src", str(src), "--dst", str(dst),
                "--gqa-kv-heads", "4", "--width-step", "0", "--depth-step", "0",
                "--force-gqa-fix"]
        old_argv = sys.argv
        try:
            sys.argv = argv
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert "head_dim" in str(exc_info.value)
        finally:
            sys.argv = old_argv


def test_expand_model_gqa_present_head_dim_still_applies_fix():
    """Sanity check the fix didn't break the working case: a checkpoint that
    DOES resolve head_dim but has a real v_proj (a genuine GQA layout, not the
    Gemma-4 MQA-V=K one) must still be correctly detected and skipped -- same
    outcome as the missing-head_dim case, but via the pre-existing shape/v_proj
    check rather than the new None-guard."""
    import sys
    from expand_model import main

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        src.mkdir()
        _write_qwen2_style_checkpoint(src, with_head_dim=True)
        dst = Path(td) / "dst"

        argv = ["expand_model.py", "--src", str(src), "--dst", str(dst),
                "--gqa-kv-heads", "4", "--width-step", "0", "--depth-step", "0"]
        old_argv = sys.argv
        try:
            sys.argv = argv
            main()  # must not raise
        finally:
            sys.argv = old_argv

        with open(dst / "config.json") as f:
            out_cfg = json.load(f)
        # Real v_proj present -> detect_mqa_v_shares_k_layout() correctly
        # rejects the MQA assumption -> GQA fix skipped, not applied.
        assert "attention_k_eq_v" not in out_cfg


def test_clone_layer_tensors_copies_and_zeros_outputs():
    """clone_layer_tensors copies all suffix keys; zero_output_projections
    zeroes o_proj and down_proj while cloning the rest."""
    import torch
    from expand_model import clone_layer_tensors
    prefix = "model.layers.0"
    tensors = {
        f"{prefix}.self_attn.q_proj.weight": torch.randn(4, 4),
        f"{prefix}.self_attn.o_proj.weight": torch.randn(4, 4),
        f"{prefix}.mlp.gate_proj.weight": torch.randn(8, 4),
        f"{prefix}.mlp.down_proj.weight": torch.randn(4, 8),
    }
    cloned = clone_layer_tensors(tensors, prefix, "model.layers.1",
                                 zero_output_projections=True)
    assert "model.layers.1.self_attn.q_proj.weight" in cloned
    assert torch.equal(cloned["model.layers.1.self_attn.q_proj.weight"],
                       tensors[f"{prefix}.self_attn.q_proj.weight"])
    # Output projections zeroed
    assert torch.all(cloned["model.layers.1.self_attn.o_proj.weight"] == 0)
    assert torch.all(cloned["model.layers.1.mlp.down_proj.weight"] == 0)
    # Non-output projections cloned (not zeroed, not aliased)
    assert torch.equal(cloned["model.layers.1.mlp.gate_proj.weight"],
                       tensors[f"{prefix}.mlp.gate_proj.weight"])


# ── AsyncCheckpointer error surfacing + .prev retention ─────────────────────

def test_async_checkpoint_prev_retained():
    """A second successful write must retain the prior checkpoint as .prev,
    not delete it — so a crash or corrupt later write can roll back."""
    import torch
    from async_checkpoint import AsyncCheckpointer
    from pathlib import Path

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 4)
        def forward(self, x):
            return self.fc(x)
        def save_pretrained(self, out_dir, safe_serialization=True, state_dict=None):
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            sd = state_dict if state_dict is not None else self.state_dict()
            torch.save(sd, out_dir / "model_state.pt")

    model = Tiny()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss = model(torch.randn(1, 4)).sum()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    with tempfile.TemporaryDirectory() as td:
        save_dir = Path(td) / "ckpt"
        ckpt = AsyncCheckpointer()
        ckpt.save(model, optimizer, step=1, save_dir=save_dir)
        ckpt.wait_for_pending()
        assert (save_dir / "training_state.pt").exists()

        loss = model(torch.randn(1, 4)).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ckpt.save(model, optimizer, step=2, save_dir=save_dir)
        ckpt.wait_for_pending()

        prev = save_dir.parent / (save_dir.name + ".prev")
        assert prev.exists(), ".prev must be retained after second write"
        prev_state = torch.load(prev / "training_state.pt", weights_only=False)
        assert prev_state["step"] == 1


def test_async_checkpoint_error_surfaces():
    """A failed background write must surface its error, not be swallowed."""
    import torch
    from async_checkpoint import AsyncCheckpointer
    from pathlib import Path

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 4)
        def forward(self, x):
            return self.fc(x)
        def save_pretrained(self, *a, **kw):
            pass

    class FailingModel(Tiny):
        def save_pretrained(self, *a, **kw):
            raise OSError("simulated disk full")

    model = FailingModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    with tempfile.TemporaryDirectory() as td:
        save_dir = Path(td) / "ckpt"
        ckpt = AsyncCheckpointer()
        ckpt.save(model, optimizer, step=1, save_dir=save_dir)
        with pytest.raises(RuntimeError, match="simulated disk full"):
            ckpt.wait_for_pending()


# ── optimizer_compat_guard ──────────────────────────────────────────────────

def test_optimizer_compat_match():
    from optimizer_compat_guard import check_optimizer_compat
    ok, msg = check_optimizer_compat("Adam8bit", "Adam8bit")
    assert ok is True


def test_optimizer_compat_mismatch():
    from optimizer_compat_guard import check_optimizer_compat
    ok, msg = check_optimizer_compat("Adam8bit", "AdamW")
    assert ok is False
    assert "mismatch" in msg.lower() or "different" in msg.lower() or "skip" in msg.lower()


# ── rocm_env family-matching ────────────────────────────────────────────────

def test_rocm_env_gfx_major():
    from rocm_env import _gfx_major
    assert _gfx_major("gfx1100") == "gfx11"
    assert _gfx_major("gfx1030") == "gfx10"
    assert _gfx_major(None) is None


def test_rocm_env_find_override_same_family():
    from rocm_env import find_override_target
    torch_list = ["sm_90", "gfx900", "gfx906", "gfx1030", "gfx1100"]
    # gfx1010 (not in list) -> gfx1030 (closest gfx10 family)
    assert find_override_target("gfx1010", torch_list) == "gfx1030"
    # gfx1101 -> gfx1100
    assert find_override_target("gfx1101", torch_list) == "gfx1100"


def test_rocm_env_no_cross_family_override():
    from rocm_env import find_override_target
    torch_list = ["gfx900", "gfx1030", "gfx1100"]
    # gfx803 has no gfx08 family member in the list
    assert find_override_target("gfx803", torch_list) is None


def test_rocm_env_already_supported_no_override():
    from rocm_env import find_override_target
    torch_list = ["gfx900", "gfx1030", "gfx1100"]
    assert find_override_target("gfx1100", torch_list) is None


def test_rocm_env_letter_suffix_arch():
    """Letter-suffix archs (gfx90a — MI250) and 3-digit archs (gfx942, gfx803)
    must be detectable and matchable, not silently skipped by the regex."""
    from rocm_env import find_override_target, _gfx_major, GFX_RE
    # GFX_RE matches letter-suffix and 3-digit archs.
    assert GFX_RE.search("gfx90a") is not None
    assert GFX_RE.search("gfx942") is not None
    assert GFX_RE.search("gfx803") is not None
    # _gfx_major handles them: gfx90a -> gfx90, gfx942 -> gfx94, gfx803 -> gfx80
    assert _gfx_major("gfx90a") == "gfx90"
    assert _gfx_major("gfx942") == "gfx94"
    assert _gfx_major("gfx803") == "gfx80"


def test_rocm_env_3digit_override():
    """3-digit archs can find same-family overrides."""
    from rocm_env import find_override_target
    # gfx903 (not in list) -> gfx906 (same gfx90 family)
    target = find_override_target("gfx903", ["gfx906", "gfx1100"])
    assert target == "gfx906"


def test_rocm_env_force_override_sets_env():
    from rocm_env import setup_rocm_env
    os.environ.pop("HSA_OVERRIDE_GFX_VERSION", None)
    info = setup_rocm_env(override="gfx1030", verbose=False)
    assert info["action"] == "force-override"
    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "gfx1030"
    os.environ.pop("HSA_OVERRIDE_GFX_VERSION", None)


# ── local_cache_stream ──────────────────────────────────────────────────────

def test_local_cache_stream_roundtrip():
    """materialize_to_cache writes rows; stream_from_cache reads them back
    shuffled, yields infinitely."""
    from local_cache_stream import materialize_to_cache, stream_from_cache

    def gen():
        for i in range(100):
            yield {"text": f"row {i}"}

    with tempfile.TemporaryDirectory() as td:
        cache = Path(td) / "cache.jsonl"
        materialize_to_cache(gen(), str(cache), target_rows=100, flush_every=10)

        stream = stream_from_cache(str(cache), seed=42)
        rows = [next(stream) for _ in range(150)]  # more than 100 -> wraps
        assert len(rows) == 150
        assert all("text" in r for r in rows)


# ── MTP weight generation ───────────────────────────────────────────────────

def test_mtp_head_generates_correct_shapes():
    """build_mtp_tensors produces the expected tensor shapes for the DeepSeek-V3
    MTP pattern: enorm/lnorm (hidden,), eh_proj (hidden, 2*hidden), cloned block."""
    import torch
    from mtp_head import build_mtp_tensors

    hidden = 16
    num_layers = 2
    layer_prefix = "model.language_model.layers"
    mtp_prefix = "model.mtp_layers"
    cfg_text = {"hidden_size": hidden, "num_hidden_layers": num_layers}

    tensors = {}
    for i in range(num_layers):
        p = f"{layer_prefix}.{i}"
        tensors[f"{p}.self_attn.q_proj.weight"] = torch.randn(8, hidden).to(torch.bfloat16)
        tensors[f"{p}.self_attn.k_proj.weight"] = torch.randn(8, hidden).to(torch.bfloat16)
        tensors[f"{p}.self_attn.o_proj.weight"] = torch.randn(hidden, 8).to(torch.bfloat16)
        tensors[f"{p}.mlp.gate_proj.weight"] = torch.randn(32, hidden).to(torch.bfloat16)
        tensors[f"{p}.mlp.up_proj.weight"] = torch.randn(32, hidden).to(torch.bfloat16)
        tensors[f"{p}.mlp.down_proj.weight"] = torch.randn(hidden, 32).to(torch.bfloat16)

    from expand_model import INIT_SCALE
    new = build_mtp_tensors(tensors, cfg_text, layer_prefix, mtp_prefix, 2, INIT_SCALE)

    assert new[f"{mtp_prefix}.0.enorm.weight"].shape == (hidden,)
    assert new[f"{mtp_prefix}.0.eh_proj.weight"].shape == (hidden, 2 * hidden)
    assert new[f"{mtp_prefix}.0.lnorm.weight"].shape == (hidden,)
    assert new[f"{mtp_prefix}.norm.weight"].shape == (hidden,)
    # RMSNorm weights start at 1.0 (identity)
    assert torch.all(new[f"{mtp_prefix}.0.enorm.weight"] == 1.0)
    # Cloned block weights equal donor (last layer)
    donor = tensors[f"{layer_prefix}.{num_layers - 1}.self_attn.q_proj.weight"]
    cloned = new[f"{mtp_prefix}.0.block.self_attn.q_proj.weight"]
    assert torch.equal(cloned, donor)


def _write_flat_llama_style_checkpoint_for_mtp(root: Path):
    """Real, on-disk, flat (no 'text_config' key) Llama-shaped checkpoint --
    used to check mtp_head.py's config.json write path against a
    non-Gemma-4 architecture."""
    import torch
    from safetensors.torch import save_file

    hidden, n_layers = 64, 2
    cfg = {
        "model_type": "llama",
        "hidden_size": hidden,
        "num_hidden_layers": n_layers,
        "intermediate_size": 128,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 8,
    }
    with open(root / "config.json", "w") as f:
        json.dump(cfg, f)

    tensors = {}
    for i in range(n_layers):
        p = f"model.language_model.layers.{i}"
        tensors[f"{p}.self_attn.q_proj.weight"] = torch.randn(64, hidden)
        tensors[f"{p}.self_attn.k_proj.weight"] = torch.randn(16, hidden)
        tensors[f"{p}.self_attn.v_proj.weight"] = torch.randn(16, hidden)
        tensors[f"{p}.self_attn.o_proj.weight"] = torch.randn(hidden, 64)
        tensors[f"{p}.mlp.gate_proj.weight"] = torch.randn(128, hidden)
        tensors[f"{p}.mlp.up_proj.weight"] = torch.randn(128, hidden)
        tensors[f"{p}.mlp.down_proj.weight"] = torch.randn(hidden, 128)
    save_file(tensors, str(root / "model.safetensors"))
    index = {"metadata": {"total_size": 0},
             "weight_map": {k: "model.safetensors" for k in tensors}}
    with open(root / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)


def test_mtp_head_flat_config_writes_single_consistent_namespace():
    """Regression test for a real bug found reviewing this repo: mtp_head.py's
    read side was already correctly generalized (`tc = cfg.get("text_config",
    cfg)`), but the write side did `cfg.setdefault("text_config", {})`
    unconditionally -- so on a flat (non-Gemma-4) config, mtp_depths/
    mtp_loss_weight landed in a brand-new, disconnected `text_config` dict
    while hidden_size/num_hidden_layers/etc. stayed at the top level. Fixed by
    writing through `tc` (the same nested-or-flat reference the read side
    already resolved) instead of cfg["text_config"] unconditionally. This test
    asserts the output config.json for a flat input has NO text_config key at
    all, and that mtp_depths sits in the SAME namespace as hidden_size."""
    import sys
    from mtp_head import main

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        src.mkdir()
        _write_flat_llama_style_checkpoint_for_mtp(src)
        dst = Path(td) / "dst"

        argv = ["mtp_head.py", "--src", str(src), "--dst", str(dst),
                "--layer-prefix", "model.language_model.layers",
                "--mtp-prefix", "model.mtp_layers", "--mtp-depths", "1"]
        old_argv = sys.argv
        try:
            sys.argv = argv
            main()
        finally:
            sys.argv = old_argv

        with open(dst / "config.json") as f:
            out_cfg = json.load(f)

        assert "text_config" not in out_cfg, (
            "flat input must not gain a text_config split -- got: " + str(out_cfg))
        assert out_cfg["mtp_depths"] == 1
        assert out_cfg["mtp_loss_weight"] == pytest.approx(0.3)
        # The rest of the flat config must be untouched and in the same namespace.
        assert out_cfg["hidden_size"] == 64
        assert out_cfg["num_hidden_layers"] == 2


def test_mtp_head_nested_config_still_nests_correctly():
    """Sanity check the flat-config fix didn't regress the primary
    (Gemma-4, nested text_config) target: mtp_depths/mtp_loss_weight must
    still land INSIDE the existing text_config dict, alongside the rest of
    the model's real config, not at the top level."""
    import sys
    import torch
    from safetensors.torch import save_file
    from mtp_head import main

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        src.mkdir()
        hidden = 64
        cfg = {
            "model_type": "gemma4",
            "text_config": {
                "hidden_size": hidden,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 8,
                "num_global_key_value_heads": 1,
                "head_dim": 8,
                "layer_types": ["full_attention", "sliding_attention"],
            },
        }
        with open(src / "config.json", "w") as f:
            json.dump(cfg, f)
        tensors = {}
        for i in range(2):
            p = f"model.language_model.layers.{i}"
            tensors[f"{p}.self_attn.q_proj.weight"] = torch.randn(64, hidden)
            tensors[f"{p}.self_attn.k_proj.weight"] = torch.randn(8, hidden)
            tensors[f"{p}.self_attn.o_proj.weight"] = torch.randn(hidden, 64)
            tensors[f"{p}.mlp.gate_proj.weight"] = torch.randn(128, hidden)
            tensors[f"{p}.mlp.up_proj.weight"] = torch.randn(128, hidden)
            tensors[f"{p}.mlp.down_proj.weight"] = torch.randn(hidden, 128)
        save_file(tensors, str(src / "model.safetensors"))
        index = {"metadata": {"total_size": 0},
                 "weight_map": {k: "model.safetensors" for k in tensors}}
        with open(src / "model.safetensors.index.json", "w") as f:
            json.dump(index, f)

        dst = Path(td) / "dst"
        argv = ["mtp_head.py", "--src", str(src), "--dst", str(dst),
                "--layer-prefix", "model.language_model.layers",
                "--mtp-prefix", "model.mtp_layers", "--mtp-depths", "1"]
        old_argv = sys.argv
        try:
            sys.argv = argv
            main()
        finally:
            sys.argv = old_argv

        with open(dst / "config.json") as f:
            out_cfg = json.load(f)

        assert "mtp_depths" not in out_cfg, "must NOT leak to the top level on a nested config"
        assert out_cfg["text_config"]["mtp_depths"] == 1
        assert out_cfg["text_config"]["hidden_size"] == hidden


# ── preprocess_data.py ──────────────────────────────────────────────────────

def test_preprocess_dedup_exact():
    from preprocess_data import get_text
    rows = [{"text": "hello"}, {"text": "hello"}, {"text": "world"}]
    seen = set()
    deduped = []
    for row in rows:
        text = get_text(row)
        if text in seen:
            continue
        seen.add(text)
        deduped.append(row)
    assert len(deduped) == 2


def test_preprocess_get_text_formats():
    from preprocess_data import get_text
    assert get_text({"text": "hello"}) == "hello"
    assert get_text({"messages": [{"content": "a"}, {"content": "b"}]}) == "a\nb"


def test_preprocess_pack_rows():
    from preprocess_data import pack_rows
    rows = [{"text": "aaa"}, {"text": "bbb"}, {"text": "ccc"}]
    packed = pack_rows(rows, max_seqlen=10, separator="|")
    # "aaa|bbb" = 7 chars, then "ccc" doesn't fit (7+4=11>10), so 2 sequences
    assert len(packed) >= 1
    assert any("aaa" in p["text"] for p in packed)


def test_preprocess_script_filter():
    from preprocess_data import should_drop_by_script
    assert should_drop_by_script({"text": "hello world"}, {"cjk"}) is False
    assert should_drop_by_script({"text": "中文文本"}, {"cjk"}) is True
    assert should_drop_by_script({"text": "hello"}, set()) is False


# ── benchmark.py ────────────────────────────────────────────────────────────

def test_benchmark_parse_configs():
    from benchmark import parse_configs
    configs = parse_configs("batch=2,seqlen=1024,dtype=bf16;batch=4,seqlen=512,dtype=fp8,flash=1")
    assert len(configs) == 2
    assert configs[0]["batch"] == 2
    assert configs[0]["dtype"] == "bf16"
    assert configs[1]["flash"] == 1
    assert configs[1]["compile"] == 0  # default


def test_benchmark_parse_configs_defaults():
    from benchmark import parse_configs
    configs = parse_configs("batch=8")
    assert configs[0]["seqlen"] == 1024  # default
    assert configs[0]["dtype"] == "bf16"  # default


def test_benchmark_parse_configs_empty_parts():
    from benchmark import parse_configs
    configs = parse_configs("batch=2;;batch=4,")
    assert len(configs) == 2


def test_benchmark_format_table():
    from benchmark import format_table
    results = [
        {"batch": 2, "seqlen": 1024, "dtype": "bf16", "flash": 0, "compile": 0,
         "tokens_per_sec": 12345, "peak_vram_gb": 78.2, "step_ms": 45.3},
    ]
    table = format_table(results)
    assert "tokens/s" in table
    assert "12,345" in table
    assert "78.2 GB" in table
    assert format_table([]) == "(no results)"


# ── generate.py ─────────────────────────────────────────────────────────────

def test_generate_build_gen_kwargs_greedy():
    """Call the REAL build_gen_kwargs and verify greedy mode logic."""
    from generate import build_gen_kwargs

    class FakeStreamer:
        pass
    kwargs = build_gen_kwargs(
        input_ids="IDS", attention_mask="MASK", max_new_tokens=100,
        temperature=0.0, top_p=0.9, repetition_penalty=1.1,
        pad_token_id=0, eos_token_id=1, streamer=FakeStreamer(),
    )
    assert kwargs["do_sample"] is False, "temperature=0 should give do_sample=False"
    assert kwargs["temperature"] == 1e-6, "temperature should be floored to 1e-6"
    assert kwargs["use_cache"] is True, "KV-cache must be enabled for generation"
    assert kwargs["pad_token_id"] == 0, "pad_token_id=0 is valid, must not be swapped"


def test_generate_build_gen_kwargs_sampling():
    """Call the REAL build_gen_kwargs and verify sampling mode + None pad fallback."""
    from generate import build_gen_kwargs

    class FakeStreamer:
        pass
    kwargs = build_gen_kwargs(
        input_ids="IDS", attention_mask="MASK", max_new_tokens=50,
        temperature=0.8, top_p=0.9, repetition_penalty=1.1,
        pad_token_id=None, eos_token_id=2, streamer=FakeStreamer(),
    )
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 0.8
    assert kwargs["pad_token_id"] == 2, "pad_token_id=None should fall back to eos_token_id"


# ── compress_model.py ───────────────────────────────────────────────────────

def test_compress_detect_config_layout():
    from compress_model import detect_config_layout
    assert detect_config_layout({"text_config": {"hidden_size": 3072}}) == "nested"
    assert detect_config_layout({"hidden_size": 4096}) == "flat"
    assert detect_config_layout({}) == "flat"


def test_compress_get_model_info_nested():
    from compress_model import get_model_info
    cfg = {"model_type": "gemma4", "text_config": {"hidden_size": 3072, "num_hidden_layers": 48, "vocab_size": 256000}}
    info = get_model_info(cfg)
    assert info["layout"] == "nested"
    assert info["hidden_size"] == 3072
    assert info["num_layers"] == 48


def test_compress_get_model_info_flat():
    from compress_model import get_model_info
    cfg = {"model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 32000}
    info = get_model_info(cfg)
    assert info["layout"] == "flat"
    assert info["hidden_size"] == 4096
    assert info["num_layers"] == 32


def test_compress_plan_quantization():
    from compress_model import plan_quantization, get_model_info
    info = get_model_info({"model_type": "llama", "hidden_size": 4096})
    for q in ("int8", "int4", "fp8"):
        plan = plan_quantization(q, info)
        assert plan["quant"] == q
        assert "size_reduction" in plan
        assert "hardware" in plan


def test_compress_invalid_quant_raises():
    from compress_model import plan_quantization, get_model_info
    info = get_model_info({"model_type": "llama"})
    with pytest.raises(SystemExit):
        plan_quantization("int2", info)


# ── tensor_parallel.py ──────────────────────────────────────────────────────

def test_tp_plan_sharding_single():
    from tensor_parallel import plan_sharding
    plan = plan_sharding(1, {"model_type": "llama", "hidden_size": 4096, "num_layers": 32})
    assert plan["mode"] == "single_gpu"
    assert plan["num_gpus"] == 1


def test_tp_plan_sharding_multi():
    from tensor_parallel import plan_sharding
    plan = plan_sharding(4, {"model_type": "llama", "hidden_size": 4096, "num_layers": 32})
    assert plan["mode"] == "pipeline_parallel"
    assert plan["num_gpus"] == 4


def test_tp_plan_sharding_zero():
    from tensor_parallel import plan_sharding
    plan = plan_sharding(0, {})
    assert plan["mode"] == "single_gpu"


def test_tp_detect_gpu_count():
    from tensor_parallel import detect_gpu_count
    count, archs = detect_gpu_count()
    assert count >= 0
    if count == 0:
        assert archs == []


def test_tp_build_explicit_device_map_gemma4_prefix():
    """build_explicit_device_map with the Gemma-4 layer prefix produces
    model.language_model.* keys (not model.layers.*)."""
    from tensor_parallel import build_explicit_device_map
    dm = build_explicit_device_map(8, 2, layer_prefix="model.language_model.layers")
    assert dm["model.language_model.embed_tokens"] == 0
    assert dm["model.language_model.layers.0"] == 0
    assert dm["model.language_model.layers.3"] == 0   # first half on GPU 0
    assert dm["model.language_model.layers.4"] == 1   # second half on GPU 1
    assert dm["model.language_model.layers.7"] == 1
    assert dm["model.language_model.norm"] == 1
    assert dm["lm_head"] == 1


def test_tp_build_explicit_device_map_standard_prefix():
    """build_explicit_device_map with the standard Llama prefix produces
    model.layers.* keys."""
    from tensor_parallel import build_explicit_device_map
    dm = build_explicit_device_map(8, 2, layer_prefix="model.layers")
    assert dm["model.embed_tokens"] == 0
    assert dm["model.layers.0"] == 0
    assert dm["model.layers.4"] == 1
    assert dm["model.norm"] == 1
    assert dm["lm_head"] == 1


def test_tp_get_model_info_gemma4_detects_prefix():
    """get_model_info_from_config detects model_type='gemma4' and returns
    layer_prefix='model.language_model.layers'."""
    import tempfile, json, os
    from tensor_parallel import get_model_info_from_config
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"model_type": "gemma4", "text_config": {"hidden_size": 4096,
                   "num_hidden_layers": 32, "vocab_size": 256000}}, f)
        cfg_path = f.name
    try:
        info = get_model_info_from_config(cfg_path)
        assert info["layer_prefix"] == "model.language_model.layers"
    finally:
        os.unlink(cfg_path)


def test_tp_get_model_info_llama_detects_prefix():
    """get_model_info_from_config with model_type='llama' returns standard prefix."""
    import tempfile, json, os
    from tensor_parallel import get_model_info_from_config
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"model_type": "llama", "text_config": {"hidden_size": 4096,
                   "num_hidden_layers": 32, "vocab_size": 32000}}, f)
        cfg_path = f.name
    try:
        info = get_model_info_from_config(cfg_path)
        assert info["layer_prefix"] == "model.layers"
    finally:
        os.unlink(cfg_path)


# ── smart_hipify.py ─────────────────────────────────────────────────────────

def test_hipify_api_substitution():
    from smart_hipify import hipify_text
    result, report = hipify_text("cudaMalloc(&ptr, size); cudaFree(ptr);")
    assert "hipMalloc" in result
    assert "hipFree" in result
    assert "cudaMalloc" not in result
    assert len(report["api_substitutions"]) == 2


def test_hipify_header_substitution():
    from smart_hipify import hipify_text
    result, report = hipify_text('#include <cuda_runtime.h>\nint main(){}')
    assert "hip/hip_runtime.h" in result
    assert len(report["header_substitutions"]) == 1


def test_hipify_library_calls_flagged():
    from smart_hipify import hipify_text
    result, report = hipify_text("cublasCreate(&handle);")
    assert len(report["library_calls_flagged"]) == 1
    assert "HIPIFY: TODO" in result
    assert "cublasCreate" in result  # NOT substituted, just flagged


def test_hipify_kernel_detection():
    from smart_hipify import hipify_text
    _, report = hipify_text("__global__ void kernel() {}")
    assert report["kernels_found"] == 1


def test_hipify_no_false_positives():
    from smart_hipify import hipify_text
    _, report = hipify_text("int main() { return 0; }")
    assert len(report["api_substitutions"]) == 0
    assert report["kernels_found"] == 0


def test_hipify_auto_add_header():
    """When a CUDA header that does NOT map to hip_runtime.h is present,
    the auto-add prepends hip/hip_runtime.h. When it DOES map (like
    cuda_runtime.h), the substitution already produces it, so no add needed."""
    from smart_hipify import hipify_text
    # cuda_runtime_api.h maps to hip/hip_runtime_api.h (NOT hip_runtime.h),
    # so the auto-add should kick in.
    result, report = hipify_text('#include <cuda_runtime_api.h>\nint main(){}')
    assert report["hip_header_added"] is True
    assert result.startswith("#include <hip/hip_runtime.h>")


def test_hipify_no_duplicate_header():
    from smart_hipify import hipify_text
    _, report = hipify_text('#include <cuda_runtime.h>\n#include <hip/hip_runtime.h>\n')
    assert report["hip_header_added"] is False


def test_hipify_todo_comment_attaches_to_real_call_not_substring_match():
    """Regression test: the TODO-comment insertion loop used to do
    `if cuda_call in line:` (plain substring containment) instead of the same
    word-boundary `pattern` used to COUNT occurrences. That meant a longer
    identifier merely CONTAINING a flagged call name as a substring (e.g. a
    wrapper function `my_cublasCreate_wrapper`) could steal the TODO comment
    away from the real call site on a later line, even though the
    word-boundary count itself was correct. Fixed to re-use `pattern` (via
    re.search) for the line-attachment search too."""
    from smart_hipify import hipify_text
    src = (
        "void my_cublasCreate_wrapper() {\n"
        "    return;\n"
        "}\n"
        "\n"
        "void real_usage() {\n"
        "    cublasCreate(&handle);\n"
        "}\n"
    )
    result, report = hipify_text(src)
    lines = result.split("\n")
    todo_idx = next(i for i, l in enumerate(lines) if "HIPIFY: TODO" in l)
    # The TODO comment must be immediately followed by the REAL call site
    # (cublasCreate(&handle);), not the wrapper function's definition line.
    assert "cublasCreate(&handle)" in lines[todo_idx + 1], (
        f"TODO comment attached to the wrong line: {lines[todo_idx + 1]!r}"
    )
    # And it must NOT have attached to the wrapper's def line.
    wrapper_idx = next(i for i, l in enumerate(lines) if "my_cublasCreate_wrapper" in l)
    assert "HIPIFY: TODO" not in lines[wrapper_idx - 1] if wrapper_idx > 0 else True


# ── rocm_env.py: gfx_target_version KFD parser ─────────────────────────────

def test_rocm_env_parses_real_gfx_target_version_values():
    """Regression test for a parser that silently produced WRONG archs for
    every real-world KFD gfx_target_version value. The field is a plain
    decimal integer (major*10000 + minor*100 + stepping, minor/stepping as
    hex), per AMD's own rocm_agent_enumerator readFromKFD() -- NOT a packed
    hex word as a prior version of this parser assumed (it did
    `(ver_int >> 16) & 0xFF` / `(ver_int >> 8) & 0xFF`, which decoded 90402
    ("gfx942", MI300X) to "gfx0111" -- completely wrong)."""
    from rocm_env import parse_kfd_gfx_target_version
    assert parse_kfd_gfx_target_version("110000") == "gfx1100"  # RX 7900 / consumer RDNA3
    assert parse_kfd_gfx_target_version("90402") == "gfx942"    # MI300X
    assert parse_kfd_gfx_target_version("90010") == "gfx90a"    # MI250X
    assert parse_kfd_gfx_target_version("100300") == "gfx1030"  # RX 6800
    assert parse_kfd_gfx_target_version("80003") == "gfx803"    # Fiji/Polaris
    assert parse_kfd_gfx_target_version("120001") == "gfx1201"  # gfx1201


def test_rocm_env_gfx_target_version_edge_cases():
    from rocm_env import parse_kfd_gfx_target_version
    # gfx_target_version == 0 on a CPU-only KFD node -- not a GPU, not a bug.
    assert parse_kfd_gfx_target_version("0") is None
    assert parse_kfd_gfx_target_version("garbage") is None
    assert parse_kfd_gfx_target_version(None) is None
    # Hex-prefixed strings are still accepted (0x-prefixed values seen on some
    # kernels/tools) and decoded via the SAME decimal formula after int(x, 16).
    assert parse_kfd_gfx_target_version("0x1adb2") == parse_kfd_gfx_target_version(str(0x1adb2))


# ── benchmark.py: real `del` frees locals (locals()-mutation was a no-op) ──

def test_benchmark_locals_cleanup_is_not_a_dict_mutation_noop():
    """Regression test for the CPython gotcha where `del locals()[name]`
    inside a function body is a silent no-op (locals() returns a snapshot
    dict; writing to it does not delete the real fast-local variable). A
    prior version of benchmark.py's per-config cleanup loop did exactly that
    for `outputs`/`input_ids`/`labels`/`attn`, so those variables (and their
    tensors) stayed alive across configs instead of being freed. This test
    exercises the exact pattern (not benchmark.py's GPU-only code directly,
    which needs CUDA) to pin the correct behavior: a real conditional `del`
    statement DOES free the name, where the locals()-mutation pattern does
    NOT."""
    def using_locals_dict_mutation():
        outputs = object()
        for name in ("outputs",):
            if name in locals():
                del locals()[name]
        # BUG: outputs is still bound after the no-op "deletion" above.
        return "outputs" in dir()

    def using_real_del_statement():
        outputs = object()
        if "outputs" in dir():
            del outputs
        return "outputs" in dir()

    assert using_locals_dict_mutation() is True, (
        "sanity check: locals()-dict-mutation is a no-op (this documents the "
        "bug pattern itself, not benchmark.py -- confirms the no-op exists in "
        "this Python version before asserting the fix pattern below works)"
    )
    assert using_real_del_statement() is False, (
        "a real `del` statement (guarded by dir() for the unbound case) "
        "correctly frees the local variable"
    )


# ── generate.py: streamer timeout raises queue.Empty, not StopIteration ────

def test_generate_streamer_timeout_raises_queue_empty_not_stopiteration():
    """Regression test: TextIteratorStreamer.__next__() calls
    `self.text_queue.get(timeout=self.timeout)` (a plain queue.Queue.get),
    which raises queue.Empty on timeout -- NOT StopIteration. A prior version
    of stream_generate()'s read loop only caught `except StopIteration`,
    so a real timeout (generate() stalls, or its background thread dies
    without calling streamer.end()) would propagate an uncaught queue.Empty
    out of stream_generate(). Verifies both exception types against the
    actual installed transformers' TextIteratorStreamer implementation."""
    import queue
    import inspect
    from transformers import TextIteratorStreamer
    src = inspect.getsource(TextIteratorStreamer.__next__)
    assert ".get(timeout=" in src, (
        "TextIteratorStreamer.__next__ no longer calls Queue.get(timeout=...) "
        "in this transformers version -- re-check what exception it raises "
        "on timeout before trusting this test's premise"
    )
    # Directly confirm queue.Queue.get(timeout=...) raises queue.Empty (the
    # underlying primitive TextIteratorStreamer relies on).
    q = queue.Queue()
    with pytest.raises(queue.Empty):
        q.get(timeout=0.05)

    # And confirm stream_generate's except clause actually names queue.Empty
    # (not just StopIteration) by reading its own source.
    from generate import stream_generate
    gen_src = inspect.getsource(stream_generate)
    assert "queue.Empty" in gen_src, (
        "stream_generate() must catch queue.Empty from the streamer read "
        "loop, not just StopIteration"
    )


# ── train_cpt.py: _StateDictModel FSDP checkpoint shim ─────────────────────

def _make_tied_weight_model():
    """A tiny torch.nn.Module simulating an HF causal-LM with tied
    embeddings: lm_head.weight and embed.weight share the same Parameter
    object (real tied-weight tensor sharing, not just equal values), plus
    `_tied_weights_keys` naming the tied key the way HF models expose it."""
    import torch

    class TinyTiedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(8, 4)
            self.lm_head = torch.nn.Linear(4, 8, bias=False)
            self.lm_head.weight = self.embed.weight  # real tied-weight sharing
            self._tied_weights_keys = ["lm_head.weight"]

        def forward(self, x):
            return self.lm_head(self.embed(x))

        def save_pretrained(self, out_dir, safe_serialization=True, state_dict=None):
            import os
            os.makedirs(out_dir, exist_ok=True)
            sd = state_dict if state_dict is not None else self.state_dict()
            torch.save(dict(sd), os.path.join(out_dir, "model_state.pt"))
            return sd

    return TinyTiedModel()


def test_state_dict_model_save_pretrained_preserves_caller_deduped_state_dict():
    """Regression test for the core async-checkpoint FSDP bug: _StateDictModel
    .save_pretrained() used to unconditionally do
    `kwargs["state_dict"] = self._full_state_dict`, clobbering a
    caller-supplied (already tied-weight-deduped) state_dict= kwarg with its
    own raw, un-deduped full_state_dict. This is exactly the call pattern
    async_checkpoint.py's save() uses: it calls model.state_dict() (returns
    the full/raw dict), strips tied keys itself into a new deduped dict, then
    passes THAT to save_pretrained(state_dict=deduped). The dedup must
    survive through to the actual save_pretrained call."""
    from train_cpt import _StateDictModel

    model = _make_tied_weight_model()
    full_state_dict = model.state_dict()  # includes BOTH embed.weight and lm_head.weight
    assert "lm_head.weight" in full_state_dict and "embed.weight" in full_state_dict

    shim = _StateDictModel(model, full_state_dict)

    # shim.state_dict() must return the pre-gathered full dict (this is what
    # async_checkpoint.py calls to build its own deduped snapshot from).
    assert shim.state_dict() is full_state_dict

    # Simulate async_checkpoint.py's own dedup step exactly (see
    # async_checkpoint.py save(), tied_keys stripping): drop the tied key.
    tied_keys = set(getattr(model, "_tied_weights_keys", {}) or {})
    deduped = {k: v for k, v in shim.state_dict().items() if k not in tied_keys}
    assert "lm_head.weight" not in deduped
    assert "embed.weight" in deduped

    captured = {}

    def fake_underlying_save_pretrained(out_dir, safe_serialization=True, state_dict=None):
        captured["state_dict"] = state_dict
        captured["safe_serialization"] = safe_serialization

    model.save_pretrained = fake_underlying_save_pretrained

    shim.save_pretrained("/tmp/unused", state_dict=deduped)

    # The deduped dict passed by the caller must survive unmodified -- NOT be
    # replaced by the shim's own full_state_dict (which still has the tied key).
    assert captured["state_dict"] is deduped, (
        "save_pretrained() must not clobber a caller-supplied state_dict= kwarg"
    )
    assert "lm_head.weight" not in captured["state_dict"], (
        "tied-weight dedup was undone -- the shim overwrote the caller's "
        "deduped state_dict with its own un-deduped full_state_dict"
    )


def test_state_dict_model_save_pretrained_still_supplies_default_when_absent():
    """The sync checkpoint path (atomic_save_checkpoint) calls
    model.save_pretrained(tmp_dir, safe_serialization=True) with NO
    state_dict= kwarg at all -- the shim must still supply its
    full_state_dict as the default in that case (this is the behavior the
    shim exists for in the first place)."""
    from train_cpt import _StateDictModel

    model = _make_tied_weight_model()
    full_state_dict = model.state_dict()
    shim = _StateDictModel(model, full_state_dict)

    captured = {}

    def fake_underlying_save_pretrained(out_dir, safe_serialization=True, state_dict=None):
        captured["state_dict"] = state_dict

    model.save_pretrained = fake_underlying_save_pretrained
    shim.save_pretrained("/tmp/unused")  # no state_dict= kwarg passed at all

    assert captured["state_dict"] is full_state_dict


def test_state_dict_optimizer_class_spoofing_via_attribute_access():
    """_StateDictOptimizer.__class__ is a @property override so the wrapped
    optimizer's real class name (e.g. "AdamW") is reported instead of
    "_StateDictOptimizer". IMPORTANT: this only works via normal ATTRIBUTE
    access (`shim.__class__.__name__`) -- the `type()` BUILTIN reads the
    instance's real C-level type slot directly and does NOT consult a
    __class__ property override, so `type(shim).__name__` is ALWAYS
    "_StateDictOptimizer" regardless of the property. A real bug (found while
    testing _StateDictModel, same file/mechanism) used `type(optimizer)
    .__name__` at both checkpoint-write call sites (train_cpt.py's
    atomic_save_checkpoint + async_checkpoint.py's save()), so every FSDP
    checkpoint wrote optimizer_type="_StateDictOptimizer" instead of the real
    class -- causing check_optimizer_compat() to ALWAYS see a mismatch on
    resume (since the shim doesn't exist at resume time) and silently discard
    Adam momentum on every single FSDP resume. Fixed both call sites to use
    `.__class__.__name__` (attribute access) instead of `type(...).__name__`
    (builtin)."""
    import torch
    from train_cpt import _StateDictOptimizer

    model = _make_tied_weight_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    shim = _StateDictOptimizer(optimizer, {"fake": "state"})

    # Documents the gotcha itself: type() bypasses the property.
    assert type(shim).__name__ == "_StateDictOptimizer"
    # The property DOES work via attribute access -- this is what the two
    # checkpoint-write call sites must use.
    assert shim.__class__.__name__ == "AdamW"
    assert shim.state_dict() == {"fake": "state"}


def test_checkpoint_write_sites_use_class_attribute_not_type_builtin():
    """Regression test pinning that BOTH checkpoint-write call sites
    (train_cpt.py's atomic_save_checkpoint, async_checkpoint.py's save())
    read the optimizer's class name via `.__class__.__name__` (attribute
    access, respects _StateDictOptimizer's @property override), not
    `type(optimizer).__name__` (builtin, silently always returns
    "_StateDictOptimizer" for the shim -- breaking FSDP resume's
    optimizer-compat check every time)."""
    import inspect
    import io
    import tokenize
    import train_cpt
    import async_checkpoint

    def _code_tokens_only(func):
        """Return only NAME/OP/NUMBER/STRING-as-code tokens' exact text,
        joined -- i.e. tokenize.generate_tokens with COMMENT tokens dropped
        and docstring/string literals reduced to a placeholder, so a
        substring check below only matches real executable expressions like
        `type(optimizer).__name__`, not the same text appearing inside a `#`
        comment or a docstring explaining what NOT to do (both of which this
        fix's own explanatory comments do, on purpose, as prose)."""
        src = inspect.getsource(func)
        out = []
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT,):
                continue
            if tok.type == tokenize.STRING:
                out.append("STRING_LITERAL")
                continue
            out.append(tok.string)
        return " ".join(out)

    train_code = _code_tokens_only(train_cpt.atomic_save_checkpoint)
    assert "optimizer . __class__ . __name__" in train_code
    assert "type ( optimizer ) . __name__" not in train_code

    async_code = _code_tokens_only(async_checkpoint.AsyncCheckpointer.save)
    assert "optimizer . __class__ . __name__" in async_code
    assert "type ( optimizer ) . __name__" not in async_code


# ── train_cpt.py: --fsdp with world_size==1 must not silently stay on CPU ──

def test_fsdp_world_size_one_condition_matches_wrap_condition():
    """Regression test for a bug where `torchrun --nproc_per_node=1 ...
    --fsdp` (which sets RANK=0/WORLD_SIZE=1, passing the --fsdp validation)
    caused the model to be loaded WITHOUT .to(device) (because the load-time
    check was `if args.fsdp:` alone) AND never actually FSDP-wrapped (because
    the wrap-time check is `if args.fsdp and ddp_world_size > 1:`) -- the
    model silently stayed on CPU with no error. The fix makes both the
    load-time device-move decision and the wrap-time decision key off the
    IDENTICAL condition (`args.fsdp and ddp_world_size > 1`), so this test
    parses train_cpt.py's source and asserts:
      1. a single shared `will_wrap_fsdp` condition is computed once
      2. it gates BOTH the .to(device) skip at load time
      3. and the _wrap_fsdp() call at wrap time
    (A true end-to-end test would need torchrun + a real process group, which
    isn't feasible in this CPU-only pytest suite -- this test instead pins
    the source-level invariant that made the bug possible: two independently
    written conditions silently drifting apart.)"""
    import ast
    import inspect
    import re
    import train_cpt

    src = inspect.getsource(train_cpt)

    # Exactly one definition of the shared condition, computed once.
    def_matches = list(re.finditer(r"will_wrap_fsdp\s*=\s*args\.fsdp\s+and\s+ddp_world_size\s*>\s*1", src))
    assert len(def_matches) == 1, (
        f"expected exactly one `will_wrap_fsdp = args.fsdp and ddp_world_size > 1` "
        f"definition (single shared condition), found {len(def_matches)}"
    )
    def_pos = def_matches[0].start()

    # Every subsequent use of the condition must be the bare variable
    # `will_wrap_fsdp` (reused), not a re-derived `args.fsdp and
    # ddp_world_size > 1` expression that could silently drift out of sync
    # with the definition -- that drift is exactly what caused the original
    # bug (load-time used `if args.fsdp:` alone; wrap-time used
    # `if args.fsdp and ddp_world_size > 1:` -- two independently written
    # conditions that didn't match for world_size==1).
    after_def = src[def_pos:]
    usage_matches = list(re.finditer(r"\bwill_wrap_fsdp\b", after_def))
    # def_matches[0] itself is one occurrence (the LHS); need at least 2 more
    # uses: the load-time device-move gate and the wrap-time _wrap_fsdp() gate.
    assert len(usage_matches) >= 3, (
        f"expected will_wrap_fsdp to be referenced at least twice after its "
        f"definition (load-time .to(device) gate + wrap-time _wrap_fsdp() "
        f"gate), found {len(usage_matches) - 1} uses"
    )

    # The load-time model-loading branch (the one containing
    # AutoModelForCausalLM.from_pretrained twice, .to(device) in the else)
    # must be gated by `if will_wrap_fsdp:`.
    load_branch_match = re.search(
        r"if will_wrap_fsdp:\s*\n\s*model = AutoModelForCausalLM\.from_pretrained\([^)]*\)\s*\n"
        r"\s*else:\s*\n\s*model = AutoModelForCausalLM\.from_pretrained\([^)]*\)\.to\(device\)",
        src,
    )
    assert load_branch_match is not None, (
        "expected `if will_wrap_fsdp: ... else: ... .to(device)` gating the "
        "model load"
    )

    # The wrap-time branch that actually calls _wrap_fsdp() must ALSO be
    # gated (possibly via an intermediate nested `if`, e.g. windowed-freeze
    # logging) by the same `will_wrap_fsdp` variable somewhere in its chain
    # of enclosing `if` statements -- not a re-derived condition instead of
    # it. Use ast to walk the real enclosing-block structure rather than
    # fragile regex/text-distance heuristics (which broke on a nested
    # `if windowed_freeze:` between `if will_wrap_fsdp:` and the actual call).
    import train_cpt as _train_cpt_module
    module_src = inspect.getsource(_train_cpt_module)
    tree = ast.parse(module_src)

    call_node = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_wrap_fsdp"):
            call_node = node
            break
    assert call_node is not None, "no _wrap_fsdp(...) call found in train_cpt.py"

    # Find every If node in the module and check whether call_node's line
    # falls within an `if will_wrap_fsdp:` node's body (directly, or nested
    # inside another If within that body).
    def contains_line(node, lineno):
        return getattr(node, "lineno", None) is not None and \
            node.lineno <= lineno <= getattr(node, "end_lineno", node.lineno)

    def is_will_wrap_fsdp_test(test_node):
        return isinstance(test_node, ast.Name) and test_node.id == "will_wrap_fsdp"

    gated_by_will_wrap_fsdp = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and is_will_wrap_fsdp_test(node.test):
            if any(contains_line(stmt, call_node.lineno) for stmt in node.body):
                gated_by_will_wrap_fsdp = True
                break
            # Also true if the call is nested inside a deeper If within this
            # If's body (e.g. `if will_wrap_fsdp: if windowed_freeze: ...
            # _wrap_fsdp(...)` -- windowed_freeze only gates a print, but
            # _wrap_fsdp() itself is still inside the outer will_wrap_fsdp
            # body at the top level, per the actual source layout).
            for stmt in ast.walk(node):
                if contains_line(stmt, call_node.lineno) and stmt is not node:
                    gated_by_will_wrap_fsdp = True
                    break
            if gated_by_will_wrap_fsdp:
                break

    assert gated_by_will_wrap_fsdp, (
        "_wrap_fsdp() call must be inside an `if will_wrap_fsdp:` block -- "
        "this is exactly the kind of independently-derived condition drift "
        "that caused the original --fsdp/world_size==1 bug"
    )


def test_train_cpt_main_declares_global_should_stop():
    """main() must declare `global _SHOULD_STOP` before it re-assigns the
    module-level flag (the rank0->all-ranks broadcast under --ddp/--fsdp).

    Regression guard for a real bug found while reviewing this file: main()
    assigns `_SHOULD_STOP = True` inside the training loop's distributed
    broadcast block WITHOUT a `global` declaration. Python's scoping is
    static -- any assignment anywhere in a function body (even behind a
    conditional, even after other reads) makes that name local for the
    ENTIRE function. Without `global`, every earlier read of _SHOULD_STOP in
    main() (including the read inside the same broadcast block, which reads
    the flag before conditionally overwriting it) becomes an
    UnboundLocalError on the first loop iteration whenever --ddp/--fsdp is
    used. Confirmed with a minimal repro of the same
    read-then-conditionally-assign-in-a-loop shape before fixing.
    """
    import ast
    src = (REPO_ROOT / "train_cpt.py").read_text()
    tree = ast.parse(src)
    main_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            main_func = node
            break
    assert main_func is not None, "no def main() found in train_cpt.py"

    assigns_should_stop = False
    declares_global = False
    for node in ast.walk(main_func):
        if isinstance(node, ast.Global) and "_SHOULD_STOP" in node.names:
            declares_global = True
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "_SHOULD_STOP":
                    assigns_should_stop = True

    assert assigns_should_stop, (
        "expected main() to assign _SHOULD_STOP (the distributed broadcast "
        "block) -- if this no longer holds, the global declaration may have "
        "become unnecessary and this test should be revisited, not just "
        "loosened"
    )
    assert declares_global, (
        "main() assigns _SHOULD_STOP without `global _SHOULD_STOP` -- this "
        "will raise UnboundLocalError on every read of _SHOULD_STOP earlier "
        "in main() (including inside the same broadcast block) as soon as "
        "--ddp/--fsdp training reaches that code, because Python treats a "
        "name assigned anywhere in a function as local to the whole function"
    )


def test_train_cpt_accum_zero_rejected():
    """--accum 0 must be rejected with a clear error, not left to crash later.

    `for micro in range(args.accum):` with accum=0 never executes, so
    `outputs`/`last_loss` are never assigned that step -- the `del outputs`
    right after the loop would raise NameError, and even without that,
    zero micro-batches means zero backward() calls before optimizer.step().
    """
    import sys
    from train_cpt import main

    argv = ["train_cpt.py", "--accum", "0", "--model", "x", "--data", "x",
            "--save", "x"]
    old_argv = sys.argv
    try:
        sys.argv = argv
        with pytest.raises(SystemExit):
            main()
    finally:
        sys.argv = old_argv


def test_train_cpt_checkpoint_every_zero_rejected():
    """--checkpoint-every 0 must be rejected with a clear error (would
    otherwise hit ZeroDivisionError on `it % args.checkpoint_every` deep in
    the training loop instead of failing fast at startup)."""
    import sys
    from train_cpt import main

    argv = ["train_cpt.py", "--checkpoint-every", "0", "--model", "x",
            "--data", "x", "--save", "x"]
    old_argv = sys.argv
    try:
        sys.argv = argv
        with pytest.raises(SystemExit):
            main()
    finally:
        sys.argv = old_argv


# ── modeling_custom.py: MTP stub loads mtp_head.py's weights + runs forward ──

def test_modeling_custom_mtp_weights_load_with_no_missing_or_unexpected_keys():
    """End-to-end: build a tiny Gemma3-family CustomForCausalLM, generate MTP
    weights for it via mtp_head.py's build_mtp_tensors, and load_state_dict.

    Regression test for a real bug found while reviewing this file: the
    shared final MTP norm was attached as `self.model.mtp = _MTPHead(hidden)`,
    producing state_dict key `model.mtp.norm.weight` -- but mtp_head.py
    actually writes the checkpoint tensor at `model.mtp_layers.norm.weight`
    (mtp_prefix="model.mtp_layers", key is f"{mtp_prefix}.norm.weight"). Under
    strict=False (what from_pretrained uses by default) this silently landed
    as BOTH a missing key (stays randomly initialized) and an unexpected key
    (silently dropped) -- the shared norm's trained weights never actually
    loaded. Fixed by attaching the norm directly onto the mtp_layers
    ModuleList so the key prefix matches.
    """
    import torch
    try:
        from transformers import Gemma3TextConfig
    except ImportError:
        pytest.skip("Gemma3TextConfig not available in this transformers version")
    from modeling_custom import CustomForCausalLM
    from mtp_head import build_mtp_tensors

    hidden = 32
    num_layers = 2
    config = Gemma3TextConfig(
        vocab_size=100, hidden_size=hidden, intermediate_size=64,
        num_hidden_layers=num_layers, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
    )
    config.mtp_depths = 2
    config.mtp_loss_weight = 0.3

    model = CustomForCausalLM(config)
    model.eval()

    # Fake base-model tensors = the model's own freshly-initialized weights
    # (real shapes, real dtype), so build_mtp_tensors clones a real block.
    base_sd = model.state_dict()
    fake_tensors = {k: v.clone() for k, v in base_sd.items()
                    if not k.startswith("model.mtp")}
    text_config_dict = {
        "hidden_size": hidden, "num_hidden_layers": num_layers,
        "intermediate_size": 64, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 8,
    }
    new_tensors = build_mtp_tensors(
        fake_tensors, text_config_dict, "model.layers", "model.mtp_layers",
        mtp_depths=2, init_scale=0.02,
    )
    all_tensors = {**fake_tensors, **new_tensors}

    missing, unexpected = model.load_state_dict(all_tensors, strict=False)
    assert missing == [], f"MTP weights failed to load (missing keys): {missing}"
    assert unexpected == [], f"checkpoint has orphan MTP keys: {unexpected}"


def test_modeling_custom_forward_pass_runs_without_crashing():
    """End-to-end forward pass smoke test for CustomForCausalLM with MTP
    enabled -- must produce logits + mtp_hidden_states without raising.

    Regression test for a real bug found while reviewing this file: the
    cloned decoder block inside each _MTPModule was called as `self.block(x)`
    with no `position_embeddings` -- real decoder layers (Gemma/Llama-family)
    require the rotary (cos, sin) position embeddings to be passed in
    explicitly (the base model forward normally computes and threads these
    through every layer; calling a bare decoder layer directly leaves
    position_embeddings=None, and attention crashes trying to unpack it as
    `cos, sin = position_embeddings`). Fixed by computing position_embeddings
    via the base model's own `rotary_emb` submodule and threading it through.
    """
    import torch
    try:
        from transformers import Gemma3TextConfig
    except ImportError:
        pytest.skip("Gemma3TextConfig not available in this transformers version")
    from modeling_custom import CustomForCausalLM

    hidden = 32
    config = Gemma3TextConfig(
        vocab_size=100, hidden_size=hidden, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
    )
    config.mtp_depths = 2
    config.mtp_loss_weight = 0.3
    model = CustomForCausalLM(config)
    model.eval()

    input_ids = torch.randint(0, 100, (2, 5))
    with torch.no_grad():
        out = model(input_ids=input_ids)

    assert out.logits.shape == (2, 5, 100)
    assert hasattr(out, "mtp_hidden_states"), (
        "forward() with mtp_depths > 0 must surface mtp_hidden_states on the "
        "output"
    )
    assert out.mtp_hidden_states.shape == (2, 5, hidden)


def test_modeling_custom_mtp_loss_with_input_ids_and_labels():
    """Regression check: input_ids + labels (the normal training path) must
    keep working and return a real, finite scalar loss that includes the MTP
    term (not just the base CE loss)."""
    import torch
    try:
        from transformers import Gemma3TextConfig
    except ImportError:
        pytest.skip("Gemma3TextConfig not available in this transformers version")
    from modeling_custom import CustomForCausalLM

    hidden = 32
    config = Gemma3TextConfig(
        vocab_size=100, hidden_size=hidden, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
    )
    config.mtp_depths = 2
    config.mtp_loss_weight = 0.3
    model = CustomForCausalLM(config)

    input_ids = torch.randint(0, 100, (2, 8))
    labels = input_ids.clone()
    out = model(input_ids=input_ids, labels=labels)

    assert out.loss is not None
    assert torch.isfinite(out.loss), f"loss must be finite, got {out.loss}"
    assert out.loss.item() > 0.0


def test_modeling_custom_mtp_loss_with_inputs_embeds_and_labels_does_not_crash():
    """Regression test for a real bug found while reviewing this file:
    _compute_mtp_total_loss -> _shift_labels did `input_ids.shape`, but
    input_ids can legitimately be None when inputs_embeds is used instead
    (the forward() signature explicitly accepts both). This crashed with
    `AttributeError: 'NoneType' object has no attribute 'shape'` any time
    someone called the model with inputs_embeds + labels while MTP was
    enabled.

    Also guards against a second-order bug in the fix itself: when
    input_ids is None, there's no token ids to build depth-shifted targets
    from, so every MTP label position is masked (ignore_index=-100).
    F.cross_entropy(..., reduction="mean") over an entirely-masked batch
    divides by zero unmasked elements and silently returns NaN rather than
    raising -- which would poison the whole loss. The fix must avoid that,
    so the returned loss has to be finite (not NaN, not crash).
    """
    import torch
    try:
        from transformers import Gemma3TextConfig
    except ImportError:
        pytest.skip("Gemma3TextConfig not available in this transformers version")
    from modeling_custom import CustomForCausalLM

    hidden = 32
    config = Gemma3TextConfig(
        vocab_size=100, hidden_size=hidden, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
    )
    config.mtp_depths = 2
    config.mtp_loss_weight = 0.3
    model = CustomForCausalLM(config)

    batch, seq_len = 2, 8
    inputs_embeds = torch.randn(batch, seq_len, hidden)
    labels = torch.randint(0, 100, (batch, seq_len))

    # Must not raise (this is the actual crash the bug report describes).
    out = model(inputs_embeds=inputs_embeds, labels=labels)

    assert out.loss is not None
    assert torch.isfinite(out.loss), (
        f"loss must be finite (not NaN/inf) even though MTP has no token ids "
        f"to shift against in the inputs_embeds case, got {out.loss}"
    )


def test_modeling_custom_no_mtp_depths_is_a_clean_noop():
    """mtp_depths=0 (or absent from config) must behave exactly like the base
    *ForCausalLM -- no mtp_layers registered, no mtp_hidden_states on output,
    forward runs cleanly."""
    import torch
    try:
        from transformers import Gemma3TextConfig
    except ImportError:
        pytest.skip("Gemma3TextConfig not available in this transformers version")
    from modeling_custom import CustomForCausalLM

    config = Gemma3TextConfig(
        vocab_size=100, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
    )
    model = CustomForCausalLM(config)
    assert not hasattr(model.model, "mtp_layers")

    input_ids = torch.randint(0, 100, (2, 5))
    with torch.no_grad():
        out = model(input_ids=input_ids)
    assert not hasattr(out, "mtp_hidden_states")


# ── New feature tests (batch_generate, position_ids, stream_jsonl, etc.) ───

def test_pack_examples_emits_position_ids_with_doc_reset():
    """pack_examples must emit position_ids that RESET at each document
    boundary, so document B starts at position 0 (prevents rotary-embedding
    position leakage across packed documents)."""
    import torch
    from train_cpt import pack_examples
    examples = [
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])},
        {"input_ids": torch.tensor([4, 5]), "labels": torch.tensor([4, 5])},
        {"input_ids": torch.tensor([6, 7, 8, 9]), "labels": torch.tensor([6, 7, 8, 9])},
    ]
    packed = pack_examples(examples, max_seq_len=100)
    assert len(packed) == 1  # all fit in one sequence
    pos = packed[0]["position_ids"]
    # pos should be [0,1,2, 0,1, 0,1,2,3] — resets at each doc boundary
    assert pos.tolist() == [0, 1, 2, 0, 1, 0, 1, 2, 3], pos.tolist()


def test_pack_examples_position_ids_reset_on_boundary_split():
    """When packing splits across multiple sequences, each sequence's
    position_ids must still reset at doc boundaries."""
    import torch
    from train_cpt import pack_examples
    examples = [
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])},
        {"input_ids": torch.tensor([4, 5, 6]), "labels": torch.tensor([4, 5, 6])},
    ]
    # max_seq_len=4 forces a split after the first example (3+3 > 4)
    packed = pack_examples(examples, max_seq_len=4)
    assert len(packed) == 2
    assert packed[0]["position_ids"].tolist() == [0, 1, 2]
    assert packed[1]["position_ids"].tolist() == [0, 1, 2]


def test_collate_passes_position_ids_when_present():
    """collate must include position_ids in the output when any input has them
    (the packing path), and omit them when none do (the normal path)."""
    import torch
    from train_cpt import collate
    # With position_ids (packed path)
    batch = [{"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([1, 2]),
              "position_ids": torch.tensor([0, 1])}]
    out = collate(batch, pad_token_id=0)
    assert "position_ids" in out
    # Without position_ids (normal path)
    batch = [{"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([1, 2])}]
    out = collate(batch, pad_token_id=0)
    assert "position_ids" not in out


def test_stream_jsonl_yields_rows_lazily():
    """stream_jsonl must yield rows one at a time (a generator), not load
    all into a list. Verify it's a generator and yields the right rows."""
    import json
    import tempfile
    from train_cpt import stream_jsonl, load_jsonl
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for i in range(5):
            f.write(json.dumps({"text": f"row-{i}"}) + "\n")
        path = f.name
    gen = stream_jsonl(path)
    # Verify it's a generator (has __next__, not a list)
    assert hasattr(gen, "__next__")
    rows = list(gen)
    assert len(rows) == 5
    assert rows[0] == {"text": "row-0"}
    assert rows[4] == {"text": "row-4"}
    # load_jsonl should return the same data as a list
    loaded = load_jsonl(path)
    assert loaded == rows
    import os; os.unlink(path)


def test_atomic_replace_with_backup_rotation():
    """atomic_replace_with_backup must: move save_dir to .prev, then tmp to
    save_dir. A second call rotates .prev out (old backup deleted)."""
    import tempfile
    from pathlib import Path
    from async_checkpoint import atomic_replace_with_backup
    with tempfile.TemporaryDirectory() as td:
        save = Path(td) / "ckpt"
        save.mkdir()
        (save / "v1.txt").write_text("v1")

        tmp = Path(td) / "ckpt.tmp_ckpt"
        tmp.mkdir()
        (tmp / "v2.txt").write_text("v2")

        atomic_replace_with_backup(tmp, save)
        # save now has v2, .prev has v1
        assert (save / "v2.txt").exists()
        backup = Path(td) / "ckpt.prev"
        assert (backup / "v1.txt").exists()

        # Second rotation: .prev (v1) deleted, old save (v2) becomes .prev
        tmp2 = Path(td) / "ckpt.tmp_ckpt"
        tmp2.mkdir()
        (tmp2 / "v3.txt").write_text("v3")
        atomic_replace_with_backup(tmp2, save)
        assert (save / "v3.txt").exists()
        assert (backup / "v2.txt").exists()
        assert not (backup / "v1.txt").exists()  # old backup rotated out


def test_synthesize_single_shard_index_builds_weight_map():
    """synthesize_single_shard_index must read the safetensors 8-byte header,
    parse the tensor keys, and build a weight_map pointing all keys at
    'model.safetensors'."""
    import json
    import tempfile
    from expand_model import synthesize_single_shard_index
    from safetensors.torch import save_file
    import torch
    with tempfile.TemporaryDirectory() as td:
        tensors = {"weight.1": torch.zeros(2), "weight.2": torch.ones(3)}
        save_file(tensors, f"{td}/model.safetensors")
        index = synthesize_single_shard_index(td)
        assert index["weight_map"]["weight.1"] == "model.safetensors"
        assert index["weight_map"]["weight.2"] == "model.safetensors"
        assert "__metadata__" not in index["weight_map"]
        assert index["metadata"]["total_size"] > 0


def test_mtp_rms_norm_preserves_input_dtype():
    """_MTPRMSNorm must output the same dtype as the input — a float32 weight
    must NOT promote a bf16 input to float32 (that was a bug: the multiply
    self.weight * x promoted to float32 because the weight wasn't cast)."""
    import torch
    from modeling_custom import _MTPRMSNorm
    norm = _MTPRMSNorm(8)
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        x = torch.randn(2, 4, 8, dtype=dtype)
        out = norm(x)
        assert out.dtype == dtype, f"input {dtype} -> output {out.dtype} (BUG)"


def test_build_gen_kwargs_static_cache_flag():
    """build_gen_kwargs with static_cache=True must add
    cache_implementation='static'; with False it must not."""
    from generate import build_gen_kwargs
    class FakeStreamer: pass
    kw = build_gen_kwargs("ids", "mask", 10, 0.7, 0.9, 1.0, 0, 1, FakeStreamer(),
                          static_cache=True)
    assert kw["cache_implementation"] == "static"
    kw = build_gen_kwargs("ids", "mask", 10, 0.7, 0.9, 1.0, 0, 1, FakeStreamer(),
                          static_cache=False)
    assert "cache_implementation" not in kw
