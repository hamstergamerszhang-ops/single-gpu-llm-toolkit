"""NVIDIA CUDA backend."""

import torch

from backends.base import ComputeBackend


class CudaBackend(ComputeBackend):
    name = "cuda"

    def is_available(self) -> bool:
        return torch.cuda.is_available()

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
        cap = torch.cuda.get_device_capability(device_index)
        if cap is None:
            return None
        return f"sm{cap[0]}{cap[1]}"

    def recommended_dtype(self) -> str:
        return "bf16"

    def supports_fp8(self) -> bool:
        if not self.is_available():
            return False
        cap = torch.cuda.get_device_capability()
        # H100 (sm90) and newer.
        return cap is not None and cap[0] >= 9

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
