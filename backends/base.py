"""Abstract base class for compute backends.

The toolkit originally assumed ROCm/`torch.cuda` everywhere. This abstraction
lets the rest of the code ask "what can this device do?" instead of hard-coding
CUDA API calls. Each backend implements a small capability surface: availability,
properties, memory helpers, dtype advice, and feature probes.
"""

from abc import ABC, abstractmethod


class ComputeBackend(ABC):
    """Capability surface for a single compute backend (ROCm, CUDA, XPU, MPS, CPU).

    Instances are stateless and cheap to create. They answer questions about the
    backend as a whole; per-device placement is handled by `backends.device`.
    """

    name: str = ""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend has at least one usable device."""
        ...

    @abstractmethod
    def get_device_count(self) -> int:
        """Number of devices visible to this backend."""
        ...

    @abstractmethod
    def get_device_properties(self, device_index: int):
        """Return backend-specific properties object, or a minimal fallback."""
        ...

    def get_arch_tag(self, device_index: int) -> str | None:
        """Return a short architecture tag (e.g. 'gfx942', 'sm90'), or None."""
        return None

    @abstractmethod
    def recommended_dtype(self) -> str:
        """Default dtype string for this backend ('bf16', 'fp16', 'fp32')."""
        ...

    def supports_fp8(self) -> bool:
        """Whether the backend can run fp8 training/inference reliably."""
        return False

    def supports_flash_attn(self) -> bool:
        """Whether flash-attention is expected to work out of the box."""
        return False

    def memory_info(self, device_index: int) -> dict:
        """Return a dict with at least 'allocated_bytes' and 'total_bytes'."""
        return {"allocated_bytes": 0, "total_bytes": 0}

    def synchronize(self, device_index: int | None = None) -> None:
        """Synchronize the given device, if the backend supports it."""
        return

    def reset_peak_memory_stats(self, device_index: int | None = None) -> None:
        """Reset peak memory counters, if supported."""
        return

    def max_memory_allocated(self, device_index: int | None = None) -> int:
        """Return peak allocated memory in bytes, or 0 if unsupported."""
        return 0

    def empty_cache(self) -> None:
        """Release cached allocator memory, if supported."""
        return
