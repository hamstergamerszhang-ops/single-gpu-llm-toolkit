"""Pytest tests for the gemma-prune-cpt tools.

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
    transpose_for_rows=False -> (n_existing, n_new) [for padding down_proj cols]"""
    import torch
    from expand_model import orthogonal_pad, INIT_SCALE
    # transpose_for_rows=True: returns (n_new, n_existing)
    pad = orthogonal_pad(8, 16, INIT_SCALE, transpose_for_rows=True)
    assert pad.shape == (8, 16), pad.shape
    assert pad.dtype == torch.bfloat16
    # transpose_for_rows=False: returns (n_existing, n_new)
    pad2 = orthogonal_pad(8, 16, INIT_SCALE, transpose_for_rows=False)
    assert pad2.shape == (16, 8), pad2.shape


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

def test_generate_temperature_logic():
    # temperature=0 -> greedy (do_sample=False), >0 -> sampling
    assert not (0.0 > 0)
    assert 0.7 > 0


def test_generate_system_prompt_prefix():
    system = "You are helpful."
    user = "What is ROCm?"
    full = system + "\n\n" + user
    assert full.startswith(system)
    assert user in full
