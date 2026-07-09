"""Backend registry and auto-detection.

Order matters for auto-detection: we prefer ROCm and CUDA accelerators first,
then Intel XPU, then Apple MPS, and fall back to CPU. Users can always pin a
backend with `--backend rocm`, `--backend cpu`, etc.
"""

from __future__ import annotations

from typing import Type

from backends.base import ComputeBackend
from backends.cpu import CpuBackend
from backends.cuda import CudaBackend
from backends.mps import MpsBackend
from backends.rocm import RocmBackend
from backends.xpu import XpuBackend

_BACKEND_CLASSES: tuple[Type[ComputeBackend], ...] = (
    RocmBackend,
    CudaBackend,
    XpuBackend,
    MpsBackend,
    CpuBackend,
)

_BACKENDS: dict[str, ComputeBackend] = {cls().name: cls() for cls in _BACKEND_CLASSES}


def list_backends() -> list[str]:
    """Return all registered backend names."""
    return list(_BACKENDS.keys())


def get_backend(name: str) -> ComputeBackend:
    """Fetch a backend by name. Raises ValueError for unknown names."""
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown backend '{name}'. Available: {', '.join(list_backends())}"
        )
    return _BACKENDS[name]


def autodetect_backend(prefer: str | None = None) -> ComputeBackend:
    """Return the first available backend, optionally preferring a named one.

    If `prefer` is supplied and available, it wins. Otherwise we walk the
    registry in priority order and return the first backend reporting itself
    available. CPU is always available, so this never fails.
    """
    if prefer is not None:
        backend = get_backend(prefer)
        if backend.is_available():
            return backend
        # If a user explicitly requested a backend that isn't available, warn
        # but still fall back rather than crash during detection.
        import warnings
        warnings.warn(
            f"Requested backend '{prefer}' is not available; auto-detecting.",
            stacklevel=2,
        )

    for cls in _BACKEND_CLASSES:
        backend = cls()
        if backend.is_available():
            return backend

    # CPU is guaranteed available because CpuBackend.is_available() returns True.
    return CpuBackend()
