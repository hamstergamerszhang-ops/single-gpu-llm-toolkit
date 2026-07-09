"""Apple Metal Performance Shaders (MPS) backend."""

import torch

from backends.base import ComputeBackend


class MpsBackend(ComputeBackend):
    name = "mps"

    def is_available(self) -> bool:
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    def get_device_count(self) -> int:
        return 1 if self.is_available() else 0

    def get_device_properties(self, device_index: int):
        class _Props:
            name = "mps"
            total_memory = 0
        return _Props()

    def recommended_dtype(self) -> str:
        # MPS supports bf16 in recent PyTorch, but fp16 has wider kernel coverage.
        return "bf16"

    def memory_info(self, device_index: int) -> dict:
        if not self.is_available():
            return {"allocated_bytes": 0, "total_bytes": 0}
        allocated = torch.mps.current_allocated_memory() if hasattr(torch.mps, "current_allocated_memory") else 0
        return {"allocated_bytes": allocated, "total_bytes": 0}

    def synchronize(self, device_index: int | None = None) -> None:
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()

    def empty_cache(self) -> None:
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
