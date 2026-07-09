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
        cap = torch.cuda.get_device_capability(device_index)
        if cap is not None:
            return f"gfx{cap[0]}{cap[1]:x}"
        return None

    def recommended_dtype(self) -> str:
        return "bf16"

    def supports_fp8(self) -> bool:
        if not self.is_available():
            return False
        # gfx942 (MI300X/MI300A/MI325X) and gfx950+ are the AMD families with
        # native fp8 support. Use the gcnArchName when available.
        props = torch.cuda.get_device_properties(0)
        arch = getattr(props, "gcnArchName", "")
        if arch:
            return arch.startswith(("gfx942", "gfx950", "gfx95", "gfx12"))
        cap = torch.cuda.get_device_capability(0)
        if cap is None:
            return False
        return cap[0] == 9 and cap[1] >= 40

    def supports_flash_attn(self) -> bool:
        return self.is_available()

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

    def setup_environment(self, override: str | None = None, hip_alloc_conf: str | None = None) -> dict:
        """Bootstrap ROCm environment before torch initializes.

        This is a thin wrapper around `rocm_env.setup_rocm_env` so callers don't
        have to import the old module directly. On non-ROCm hosts it returns an
        empty info dict without erroring.
        """
        if not self.is_available() and override is None:
            return {}
        try:
            from rocm_env import setup_rocm_env
            return setup_rocm_env(override=override, hip_alloc_conf=hip_alloc_conf)
        except Exception as exc:
            # Don't swallow silently (repo convention: no bare `except: pass`).
            # rocm_env's own environment bootstrap is best-effort by design --
            # a failure here shouldn't crash callers that only want the info
            # dict -- but it must be visible, not silent.
            print(f"[backends.rocm] WARNING: setup_rocm_env failed, continuing "
                  f"without ROCm env bootstrap: {exc!r}")
            return {}
