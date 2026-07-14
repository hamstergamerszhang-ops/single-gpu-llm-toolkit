"""Hardware presets.

A preset is a bundle of sensible defaults for a given device class. It is
applied *before* explicit CLI overrides, so `--batch 8` always beats the
preset's batch size.

Keys match train_cpt.py CLI flag `dest` names exactly:
  batch_size -> --batch (dest="batch")
  seq_length -> --max-seq-len (dest="max_seq_len")
  gradient_accumulation_steps -> --accum (dest="accum")

The generate.py recipe-merge code at main() line ~189 maps these via
`hasattr(args, key)`, so the dest names must match what argparse stores.
"""

from __future__ import annotations


PRESETS: dict[str, dict] = {
    "cpu": {
        "dtype": "fp32",
        "batch": 1,
        "max_seq_len": 128,
        "accum": 8,
        "compile": False,
        "flash_attn": False,
        "fsdp": False,
        "ddp": False,
    },
    "rx7900-24g": {
        "dtype": "bf16",
        "batch": 1,
        "max_seq_len": 2048,
        "accum": 4,
        "start": 0,
        "end": -1,
        "compile": False,
        "flash_attn": True,
    },
    "mi300x-80g": {
        "dtype": "bf16",
        "batch": 2,
        "max_seq_len": 4096,
        "accum": 2,
        "start": 0,
        "end": -1,
        "compile": True,
        "flash_attn": True,
    },
    "mi300x-192g": {
        "dtype": "bf16",
        "batch": 4,
        "max_seq_len": 4096,
        "accum": 1,
        "start": 0,
        "end": -1,
        "compile": True,
        "flash_attn": True,
    },
    "mi250-128g": {
        "dtype": "bf16",
        "batch": 2,
        "max_seq_len": 2048,
        "accum": 2,
        "start": 0,
        "end": -1,
        "compile": False,
        "flash_attn": True,
    },
}


def list_presets() -> list[str]:
    return list(PRESETS.keys())


def get_preset(name: str) -> dict:
    if name not in PRESETS:
        raise ValueError(
            f"Unknown preset '{name}'. Available: {', '.join(list_presets())}"
        )
    return PRESETS[name]


def suggest_preset(backend_name: str, total_memory_bytes: int | None = None) -> str | None:
    """Suggest a preset based on detected backend and VRAM, or None if unsure."""
    if backend_name == "cpu":
        return "cpu"
    if backend_name == "rocm" and total_memory_bytes:
        total_gib = total_memory_bytes / (1024 ** 3)
        if total_gib >= 180:
            return "mi300x-192g"
        if total_gib >= 70:
            return "mi300x-80g"
        if total_gib >= 30:
            return "mi250-128g"
        if total_gib >= 20:
            return "rx7900-24g"
    return None
