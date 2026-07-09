"""Hardware presets.

A preset is a bundle of sensible defaults for a given device class. It is
applied *before* explicit CLI overrides, so `--batch-size 8` always beats the
preset's batch size.
"""

from __future__ import annotations


PRESETS: dict[str, dict] = {
    "cpu": {
        "dtype": "fp32",
        "batch_size": 1,
        "seq_length": 128,
        "gradient_accumulation_steps": 8,
        "compile": False,
        "flash_attn": False,
        "fsdp": False,
        "ddp": False,
    },
    "mps": {
        "dtype": "bf16",
        "batch_size": 1,
        "seq_length": 512,
        "gradient_accumulation_steps": 4,
        "compile": False,
        "flash_attn": False,
        "fsdp": False,
        "ddp": False,
    },
    "rx7900-24g": {
        "dtype": "bf16",
        "batch_size": 1,
        "seq_length": 2048,
        "gradient_accumulation_steps": 4,
        "start": 0,
        "end": -1,
        "compile": False,
        "flash_attn": True,
    },
    "mi300x-80g": {
        "dtype": "bf16",
        "batch_size": 2,
        "seq_length": 4096,
        "gradient_accumulation_steps": 2,
        "start": 0,
        "end": -1,
        "compile": True,
        "flash_attn": True,
    },
    "mi300x-192g": {
        "dtype": "bf16",
        "batch_size": 4,
        "seq_length": 4096,
        "gradient_accumulation_steps": 1,
        "start": 0,
        "end": -1,
        "compile": True,
        "flash_attn": True,
    },
    "mi250-128g": {
        "dtype": "bf16",
        "batch_size": 2,
        "seq_length": 2048,
        "gradient_accumulation_steps": 2,
        "start": 0,
        "end": -1,
        "compile": False,
        "flash_attn": True,
    },
    "a100-40g": {
        "dtype": "bf16",
        "batch_size": 1,
        "seq_length": 2048,
        "gradient_accumulation_steps": 4,
        "start": 0,
        "end": -1,
        "compile": True,
        "flash_attn": True,
    },
    "a100-80g": {
        "dtype": "bf16",
        "batch_size": 2,
        "seq_length": 4096,
        "gradient_accumulation_steps": 2,
        "start": 0,
        "end": -1,
        "compile": True,
        "flash_attn": True,
    },
    "h100-80g": {
        "dtype": "bf16",
        "batch_size": 2,
        "seq_length": 4096,
        "gradient_accumulation_steps": 2,
        "start": 0,
        "end": -1,
        "compile": True,
        "flash_attn": True,
    },
    "intel-xpu": {
        "dtype": "bf16",
        "batch_size": 1,
        "seq_length": 512,
        "gradient_accumulation_steps": 4,
        "compile": False,
        "flash_attn": False,
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
    if backend_name == "mps":
        return "mps"
    if backend_name == "xpu":
        return "intel-xpu"
    if backend_name in ("rocm", "cuda") and total_memory_bytes:
        total_gib = total_memory_bytes / (1024 ** 3)
        if total_gib >= 180:
            return "mi300x-192g"
        if total_gib >= 70:
            return "mi300x-80g" if backend_name == "rocm" else "a100-80g"
        if total_gib >= 30:
            return "mi250-128g" if backend_name == "rocm" else "a100-40g"
        if total_gib >= 20:
            return "rx7900-24g"
    return None
