"""AMD ROCm (HIP) backend."""

import os

import torch

from backends.base import ComputeBackend


class RocmBackend(ComputeBackend):
    name = "rocm"

    def is_available(self) -> bool:
        # ROCm reports through torch.cuda APIs on AMD.
        return torch.cuda.is_available() and torch.version.hip is not None

    def get_device_count(self) -> int:
        if not self.is_available():
            return 0
        return torch.cuda.device_count()

    def get_device_properties(self, device_index: int):
        if not self.is_available():
            return None
        return torch.cuda.get_device_properties(device_index)

    def get_arch_tag(self, device_index: int) -> str | None:
        if not self.is_available():
            return None
        props = torch.cuda.get_device_properties(device_index)
        # ROCm exposes gcnArchName on the properties object.
        arch = getattr(props, "gcnArchName", "")
        if arch and arch.startswith("gfx"):
            return arch
        # Fallback to capability tuple, which on ROCm is gfx_major/minor.
        # Emit minor in DECIMAL, not hex: a real gfx942 reports cap=(9, 42),
        # so f"gfx{cap[0]}{cap[1]:x}" produced "gfx92a" (42 -> 0x2a) instead of
        # "gfx942" -- a bogus arch string that downstream code (the FP8 gate,
        # supports_flash_attn) would never match. cap[1] on ROCm is a plain
        # decimal minor (gfx940->40, gfx942->42).
        cap = torch.cuda.get_device_capability(device_index)
        if cap is not None:
            return f"gfx{cap[0]}{cap[1]}"
        return None

    def recommended_dtype(self) -> str:
        return "bf16"

    def supports_fp8(self) -> bool:
        if not self.is_available():
            return False
        # The AMD families with native fp8 compute are the CDNA3 MI300 line:
        #   gfx940 (MI300A), gfx941 (MI325X), gfx942 (MI300X)
        # plus the upcoming gfx950 (MI350). All share the gfx94x prefix.
        # NOT included:
        #   - gfx12 / RDNA4 (RX 9000 series): ROCm does not ship working native
        #     fp8 kernels for these in current wheels; advertising fp8 here
        #     makes --dtype fp8 attempt float8 conversion and crash on exactly
        #     the consumer hardware this repo targets.
        #   - gfx90a/gfx908 (MI200/MI100) and all RDNA1/2/3 consumer cards: no
        #     fp8 units.
        props = torch.cuda.get_device_properties(0)
        arch = getattr(props, "gcnArchName", "")
        if arch:
            return arch.startswith(("gfx940", "gfx941", "gfx942", "gfx950", "gfx95"))
        cap = torch.cuda.get_device_capability(0)
        if cap is None:
            return False
        return cap[0] == 9 and cap[1] >= 40

    def supports_flash_attn(self) -> bool:
        if not self.is_available():
            return False
        # flash-attn ships prebuilt ROCm kernels only for a specific set of
        # archs. Advertising it for everything (the old `return True`) made
        # resolve_flash_attn() probe -- and on gfx900 (MI25), gfx803 (Fiji/
        # Polaris), gfx1010 (RDNA1) the probe either crashes or the import
        # fails. Gate to the archs that actually have kernels:
        #   CDNA:  gfx908 (MI100), gfx90a (MI250), gfx94x (MI300), gfx950 (MI350)
        #   RDNA2+: gfx1030 (RX 6800/6900), gfx1031, gfx1100/1101/1102 (RX 7900),
        #           gfx115x (RX 7000 mobile), gfx120x (RX 9000)
        # Explicitly EXCLUDED: gfx803, gfx900, gfx906 (no kernels), gfx1010
        # (RDNA1 -- no flash-attn ROCm build).
        arch = self.get_arch_tag(0) or ""
        if arch.startswith(("gfx908", "gfx90a", "gfx94", "gfx95")):
            return True
        if arch.startswith("gfx10") and arch not in ("gfx1010", "gfx1011", "gfx1012"):
            return True
        if arch.startswith(("gfx11", "gfx12")):
            return True
        return False

    def memory_info(self, device_index: int) -> dict:
        if not self.is_available():
            return {"allocated_bytes": 0, "total_bytes": 0}
        return {
            "allocated_bytes": torch.cuda.memory_allocated(device_index),
            "total_bytes": torch.cuda.get_device_properties(device_index).total_memory,
        }

    def synchronize(self, device_index: int | None = None) -> None:
        if self.is_available():
            torch.cuda.synchronize(device_index)

    def reset_peak_memory_stats(self, device_index: int | None = None) -> None:
        if self.is_available():
            torch.cuda.reset_peak_memory_stats(device_index)

    def max_memory_allocated(self, device_index: int | None = None) -> int:
        if self.is_available():
            return torch.cuda.max_memory_allocated(device_index)
        return 0

    def empty_cache(self) -> None:
        if self.is_available():
            torch.cuda.empty_cache()
