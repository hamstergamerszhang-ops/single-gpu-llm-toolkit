"""Intel XPU backend."""

import torch

from backends.base import ComputeBackend


class XpuBackend(ComputeBackend):
    name = "xpu"

    def is_available(self) -> bool:
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    def get_device_count(self) -> int:
        if not self.is_available():
            return 0
        return torch.xpu.device_count()

    def get_device_properties(self, device_index: int):
        if not self.is_available():
            return None
        return torch.xpu.get_device_properties(device_index)

    def get_arch_tag(self, device_index: int) -> str | None:
        props = self.get_device_properties(device_index)
        if props is None:
            return None
        # Intel properties do not expose a gfx/sm-style arch string.
        return getattr(props, "name", None)

    def recommended_dtype(self) -> str:
        return "bf16"

    def memory_info(self, device_index: int) -> dict:
        if not self.is_available():
            return {"allocated_bytes": 0, "total_bytes": 0}
        return {
            "allocated_bytes": torch.xpu.memory_allocated(device_index),
            "total_bytes": torch.xpu.get_device_properties(device_index).total_memory,
        }

    def synchronize(self, device_index: int | None = None) -> None:
        if self.is_available():
            torch.xpu.synchronize(device_index)

    def reset_peak_memory_stats(self, device_index: int | None = None) -> None:
        if self.is_available() and hasattr(torch.xpu, "reset_peak_memory_stats"):
            torch.xpu.reset_peak_memory_stats(device_index)

    def max_memory_allocated(self, device_index: int | None = None) -> int:
        if self.is_available() and hasattr(torch.xpu, "max_memory_allocated"):
            return torch.xpu.max_memory_allocated(device_index)
        return 0

    def empty_cache(self) -> None:
        if self.is_available() and hasattr(torch.xpu, "empty_cache"):
            torch.xpu.empty_cache()
