"""Hardware presets.

A preset is a bundle of sensible defaults for a given device class. It is
applied *before* explicit CLI overrides, so `--batch 8` always beats the
preset's batch size.

KEY NAMES MUST MATCH ARGPARSE DEST NAMES. A preset is merged into the parsed
args namespace, so each key is the `dest` of an argparse option, NOT the flag
string. This repo's trainer uses:
    --batch         -> args.batch
    --max-seq-len    -> args.max_seq_len
    --accum          -> args.accum
    --dtype          -> args.dtype
    --compile        -> args.compile
    --flash-attn     -> args.flash_attn
    --fsdp           -> args.fsdp
    --ddp            -> args.ddp
    --start / --end  -> args.start / args.end
An earlier version used batch_size/seq_length/gradient_accumulation_steps,
which matched NOTHING and silently made every preset a no-op. That is fixed
here and pinned by tests.

COVERAGE: the full AMD ROCm lineup this toolkit targets, from the MI25
(gfx900, 2017) through MI300X (gfx942) and RDNA4 (gfx1201, RX 9000). Consumer
Radeon cards (RX 6000/7000/9000, APUs) are first-class presets, not an
afterthought -- the repo's gfx-override auto-detection (rocm_env.py) already
handles the arch-mismatch case that hits consumer cards most; these presets
give them sane batch/seq defaults for the VRAM they actually have.

NOTE on start/end: train_cpt.py treats --end as an EXCLUSIVE upper bound, so
`end: -1` (used in an earlier version) would freeze ALL layers. Presets that
unfreeze layers set `start: 0` and OMIT `end` entirely (letting train_cpt
default to None = through the last layer). Setting end to a negative number is
a bug, not a feature.
"""

from __future__ import annotations


PRESETS: dict[str, dict] = {
    # ------------------------------------------------------------------
    # CPU (the universal fallback for testing/dev without real hardware).
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # CDNA -- Instinct accelerators.
    # ------------------------------------------------------------------
    # MI25 (Vega 20, gfx900, 16GB HBM2). Oldest Instinct with a ROCm 6.x
    # build; no bf16 compute, no fp8, no flash-attn kernels. Small and slow
    # but it does launch.
    "mi25-16g": {
        "dtype": "fp16",
        "batch": 1,
        "max_seq_len": 512,
        "accum": 16,
        "compile": False,
        "flash_attn": False,
        "fsdp": False,
        "ddp": False,
    },
    # MI250 / MI250X (gfx90a, 128GB HBM2e per package -- 2 GCDs reported as
    # 2 devices, so this preset targets one GCD = 64GB; use --ddp for both).
    "mi250-64g": {
        "dtype": "bf16",
        "batch": 2,
        "max_seq_len": 2048,
        "accum": 2,
        "start": 0,
        "compile": False,
        "flash_attn": True,
    },
    # MI300X (gfx942, 192GB HBM3). The headline target -- big batch, long
    # context, compile on, fp8 available via --dtype fp8.
    "mi300x-192g": {
        "dtype": "bf16",
        "batch": 4,
        "max_seq_len": 4096,
        "accum": 1,
        "start": 0,
        "compile": True,
        "flash_attn": True,
    },
    # MI300X 80GB SKU / MI300A (APU, 128GB unified but ~96GB GPU-accessible).
    # Conservative batch for the smaller visible VRAM.
    "mi300x-80g": {
        "dtype": "bf16",
        "batch": 2,
        "max_seq_len": 4096,
        "accum": 2,
        "start": 0,
        "compile": True,
        "flash_attn": True,
    },

    # ------------------------------------------------------------------
    # RDNA3 -- RX 7000 series (gfx1100/1101/1102). 24GB/20GB/12GB/8GB.
    # Native bf16, flash-attn kernels exist, no fp8.
    # ------------------------------------------------------------------
    "rx7900xtx-24g": {
        "dtype": "bf16",
        "batch": 1,
        "max_seq_len": 2048,
        "accum": 4,
        "start": 0,
        "compile": False,
        "flash_attn": True,
    },
    "rx7900xt-20g": {
        "dtype": "bf16",
        "batch": 1,
        "max_seq_len": 2048,
        "accum": 6,
        "start": 0,
        "compile": False,
        "flash_attn": True,
    },
    "rx7700xt-12g": {
        "dtype": "bf16",
        "batch": 1,
        "max_seq_len": 1024,
        "accum": 8,
        "start": 0,
        "compile": False,
        "flash_attn": True,
    },
    "rx7600-8g": {
        "dtype": "bf16",
        "batch": 1,
        "max_seq_len": 512,
        "accum": 16,
        "start": 0,
        "compile": False,
        "flash_attn": True,
    },

    # ------------------------------------------------------------------
    # RDNA2 -- RX 6000 series (gfx1030/1031/1032). 16GB/12GB/8GB.
    # Native bf16, flash-attn kernels exist, no fp8.
    # ------------------------------------------------------------------
    "rx6800-16g": {
        "dtype": "bf16",
        "batch": 1,
        "max_seq_len": 1024,
        "accum": 8,
        "start": 0,
        "compile": False,
        "flash_attn": True,
    },
    "rx6700xt-12g": {
        "dtype": "bf16",
        "batch": 1,
        "max_seq_len": 1024,
        "accum": 10,
        "start": 0,
        "compile": False,
        "flash_attn": True,
    },
    "rx6600-8g": {
        "dtype": "bf16",
        "batch": 1,
        "max_seq_len": 512,
        "accum": 16,
        "start": 0,
        "compile": False,
        "flash_attn": True,
    },

    # ------------------------------------------------------------------
    # RDNA4 -- RX 9000 series (gfx1200/1201). 16GB.
    # Newest consumer arch; bf16 + flash-attn, no fp8 in current ROCm.
    # ------------------------------------------------------------------
    "rx9070xt-16g": {
        "dtype": "bf16",
        "batch": 1,
        "max_seq_len": 2048,
        "accum": 6,
        "start": 0,
        "compile": False,
        "flash_attn": True,
    },

    # ------------------------------------------------------------------
    # APU -- integrated Radeon (Phoenix/Hawk Point, gfx1103 / gfx115x).
    # Shared system RAM, typically 512MB-2GB carveout. Small everything.
    # ------------------------------------------------------------------
    "apu-2g": {
        "dtype": "fp16",
        "batch": 1,
        "max_seq_len": 256,
        "accum": 32,
        "compile": False,
        "flash_attn": False,
        "fsdp": False,
        "ddp": False,
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
    """Suggest a preset based on detected backend and VRAM, or None if unsure.

    VRAM tiers are chosen so a card lands on the preset whose VRAM is the
    largest one NOT exceeding its own (so a 24GB card picks the 24GB preset,
    not the 20GB one). This matches how the presets are named.
    """
    if backend_name == "cpu":
        return "cpu"
    if backend_name == "rocm" and total_memory_bytes:
        total_gib = total_memory_bytes / (1024 ** 3)
        # CDNA Instinct tier (>= 64GB visible -- MI250 GCD / MI300X).
        if total_gib >= 160:
            return "mi300x-192g"
        if total_gib >= 70:
            return "mi300x-80g"
        if total_gib >= 56:
            return "mi250-64g"
        # Consumer Radeon tiers (largest-not-exceeding).
        if total_gib >= 22:
            return "rx7900xtx-24g"
        if total_gib >= 18:
            return "rx7900xt-20g"
        if total_gib >= 14:
            return "rx6800-16g"
        if total_gib >= 11:
            return "rx7700xt-12g"
        if total_gib >= 7:
            return "rx6600-8g"
        if total_gib >= 1.5:
            return "apu-2g"
        # < 1.5GB visible: too small for any sane training preset; let the
        # caller fall back to its own defaults rather than guessing.
        return None
    return None
