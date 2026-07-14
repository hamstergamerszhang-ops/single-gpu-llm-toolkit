"""Per-device handle bound to a compute backend.

`BackendDevice` is what the rest of the toolkit should hold: it knows which
backend it belongs to, which device index, and how to do memory/sync operations
on that device without calling `torch.cuda` directly.
"""

from __future__ import annotations

import torch

from backends.base import ComputeBackend
from backends.registry import autodetect_backend, get_backend


class BackendDevice:
    """A concrete device attached to a backend."""

    def __init__(self, backend: ComputeBackend | str | None = None, index: int = 0):
        if backend is None:
            backend = autodetect_backend()
        elif isinstance(backend, str):
            backend = get_backend(backend)
        self.backend: ComputeBackend = backend
        self.index: int = index

    @property
    def name(self) -> str:
        return self.backend.name

    @property
    def torch_device(self) -> torch.device:
        # ROCm builds of PyTorch report themselves through the "cuda" device
        # type/namespace -- torch.cuda.is_available(), torch.cuda.set_device(),
        # etc. all work on ROCm, but PyTorch has NO "rocm" device type string.
        # `torch.device("rocm", 0)` raises RuntimeError immediately ("Expected
        # one of cpu, cuda, ipu, ... device type at start of device string:
        # rocm") -- confirmed directly against an installed torch build. This
        # was breaking every caller that touched .torch_device on the rocm
        # backend (evaluate.py, generate.py, compress_model.py, and
        # runtime/probe.py's probe_flash_attn/probe_compile) on exactly the
        # hardware this toolkit targets. experimental/ddp.py and
        # experimental/fsdp.py already do this same "rocm" -> "cuda" mapping
        # correctly when calling torch.cuda.set_device(); this makes the
        # device-string construction consistent with that.
        device_type = "cuda" if self.backend.name == "rocm" else self.backend.name
        return torch.device(device_type, self.index)

    def to(self, module: torch.nn.Module | torch.Tensor) -> torch.nn.Module | torch.Tensor:
        """Move a module or tensor to this device."""
        return module.to(self.torch_device)

    def synchronize(self) -> None:
        self.backend.synchronize(self.index)

    def reset_peak_memory_stats(self) -> None:
        self.backend.reset_peak_memory_stats(self.index)

    def max_memory_allocated(self) -> int:
        return self.backend.max_memory_allocated(self.index)

    def memory_info(self) -> dict:
        return self.backend.memory_info(self.index)

    def empty_cache(self) -> None:
        self.backend.empty_cache()

    def supports_fp8(self) -> bool:
        return self.backend.supports_fp8()

    def supports_flash_attn(self) -> bool:
        return self.backend.supports_flash_attn()

    def recommended_dtype(self) -> str:
        return self.backend.recommended_dtype()

    def arch_tag(self) -> str | None:
        return self.backend.get_arch_tag(self.index)

    def __repr__(self) -> str:
        return f"BackendDevice({self.backend.name}:{self.index})"


def default_device(prefer: str | None = None) -> BackendDevice:
    """Return the default device for the current process.

    In distributed runs callers should pass `LOCAL_RANK` as the index.
    """
    backend = autodetect_backend(prefer=prefer)
    return BackendDevice(backend=backend, index=0)
