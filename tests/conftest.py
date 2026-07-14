"""Shared pytest fixtures for the gemma_persona test suite.

Provides reusable AMD-specific test fixtures so individual test files don't
each rebuild their own tiny models / configs / tmp dirs inline. Import these
via pytest's fixture injection (no explicit import needed)."""

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_dir():
    """A temporary directory that auto-cleans after the test."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def tiny_gemma4_config():
    """A minimal Gemma-4 config dict for testing config parsing,
    device_map building, and expand_model logic. Matches the layout
    real Gemma-4 checkpoints use (model_type='gemma4', nested text_config)."""
    return {
        "model_type": "gemma4",
        "text_config": {
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "vocab_size": 256,
            "head_dim": 16,
        },
    }


@pytest.fixture
def tiny_llama_config():
    """A minimal Llama config dict for testing standard-layout paths
    (model.layers, not model.language_model.layers)."""
    return {
        "model_type": "llama",
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": 256,
    }


@pytest.fixture
def gemma4_config_file(tmp_dir, tiny_gemma4_config):
    """Write a tiny Gemma-4 config.json to a tmp dir and return the path.
    For testing config-reading functions that expect a file path."""
    cfg_path = tmp_dir / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(tiny_gemma4_config, f)
    return cfg_path


@pytest.fixture
def amd_gfx_archs():
    """A dict of known AMD gfx arch -> (major, minor) capability tuples,
    for testing the FP8 capability gate and gfx detection logic."""
    return {
        "gfx900": (9, 0),    # MI25 (Vega)
        "gfx906": (9, 6),    # MI50 / Vega20
        "gfx908": (9, 8),    # MI100
        "gfx90a": (9, 10),   # MI250X (letter suffix)
        "gfx940": (9, 40),   # MI300A
        "gfx942": (9, 42),   # MI300X
        "gfx950": (9, 50),   # MI350
        "gfx1030": (10, 30), # RX 6800/6900 (RDNA2)
        "gfx1100": (11, 0),  # RX 7900 (RDNA3)
        "gfx1150": (11, 50), # RDNA3.5
        "gfx1200": (12, 0),  # RDNA4
    }


@pytest.fixture
def fp8_capable_archs():
    """AMD archs that should pass the FP8 capability gate (gfx94x + gfx950)."""
    return ["gfx940", "gfx942", "gfx950"]


@pytest.fixture
def non_fp8_archs():
    """AMD archs that should fail the FP8 capability gate."""
    return ["gfx900", "gfx906", "gfx908", "gfx90a", "gfx1030", "gfx1100", "gfx1150", "gfx1200"]
