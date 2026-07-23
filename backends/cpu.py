"""CPU backend — always available, always the safe fallback."""

from backends.base import ComputeBackend


class CpuBackend(ComputeBackend):
    name = "cpu"

    def is_available(self) -> bool:
        return True

    def get_device_count(self) -> int:
        return 1

    def get_device_properties(self, device_index: int):
        class _Props:
            name = "cpu"
            total_memory = 0
        return _Props()

    def recommended_dtype(self) -> str:
        # CPU PyTorch often lacks efficient bf16 matmuls; fp32 is the safest default.
        return "fp32"

    def memory_info(self, device_index: int) -> dict:
        return {"allocated_bytes": 0, "total_bytes": 0}
