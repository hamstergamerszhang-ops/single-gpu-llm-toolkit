"""Runtime capability probing.

Rather than assuming flash-attention / fp8 / torch.compile work because an
import succeeded, we run a tiny operation and verify it actually executes
without NaNs or crashes on the target device.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from backends.device import BackendDevice


def _quietly(probe_fn):
    """Run a probe and swallow exceptions, returning (False, reason)."""
    try:
        return probe_fn()
    except Exception as exc:
        return False, str(exc)


def probe_fp8(device: BackendDevice) -> tuple[bool, str]:
    """Check whether fp8 training can run on this device.

    Requires torchao and a backend that advertises fp8 support. We also run a
    tiny scaled matmul when possible so unsupported arches don't get enabled by
    accident.
    """
    if not device.supports_fp8():
        return False, f"backend {device.name} does not advertise fp8 support"

    try:
        import torchao
    except Exception as exc:
        return False, f"torchao not available: {exc}"

    # Try a minimal scaled matmul if the op exists.
    if hasattr(torch, "_scaled_mm"):
        return _quietly(lambda: _run_scaled_mm(device))

    return True, "torchao available; scaled matmul op not present, trusting backend"


def _run_scaled_mm(device: BackendDevice) -> tuple[bool, str]:
    dev = device.torch_device
    a = torch.randint(-128, 127, (16, 16), dtype=torch.float8_e4m3fn, device=dev)
    b = torch.randint(-128, 127, (16, 16), dtype=torch.float8_e4m3fn, device=dev).t()
    scale_a = torch.tensor(1.0, device=dev)
    scale_b = torch.tensor(1.0, device=dev)
    out = torch._scaled_mm(a, b, scale_a, scale_b)
    if torch.isnan(out).any():
        return False, "scaled matmul produced NaN"
    return True, "scaled matmul succeeded"


def probe_flash_attn(device: BackendDevice) -> tuple[bool, str]:
    """Check whether flash-attention can run on this device."""
    if not device.supports_flash_attn():
        return False, f"backend {device.name} does not advertise flash-attention support"

    try:
        from flash_attn import flash_attn_func  # type: ignore
    except Exception as exc:
        return False, f"flash-attn not importable: {exc}"

    def _run() -> tuple[bool, str]:
        dev = device.torch_device
        q = torch.randn(2, 8, 4, 64, dtype=torch.bfloat16, device=dev)
        k = torch.randn(2, 8, 4, 64, dtype=torch.bfloat16, device=dev)
        v = torch.randn(2, 8, 4, 64, dtype=torch.bfloat16, device=dev)
        out = flash_attn_func(q, k, v)
        if torch.isnan(out).any():
            return False, "flash-attention produced NaN"
        return True, "flash-attention forward succeeded"

    return _quietly(_run)


def probe_compile(device: BackendDevice, mode: str = "default") -> tuple[bool, str]:
    """Check whether torch.compile works on this device."""
    def _run() -> tuple[bool, str]:
        dev = device.torch_device
        mod = nn.Linear(8, 8).to(dev)
        compiled = torch.compile(mod, mode=mode, fullgraph=False)
        x = torch.randn(4, 8, device=dev)
        out = compiled(x)
        loss = out.sum()
        loss.backward()
        return True, f"torch.compile(mode={mode}) succeeded"

    return _quietly(_run)


def resolve_dtype(device: BackendDevice, requested: str | None) -> str:
    """Return a dtype string that is safe for this device.

    Args:
        requested: User-requested dtype ('bf16', 'fp16', 'fp8', 'fp32') or None.
    """
    if requested == "fp8":
        usable, reason = probe_fp8(device)
        if usable:
            return "fp8"
        warnings.warn(f"fp8 requested but not usable: {reason}; falling back to bf16")
        requested = "bf16"

    if requested in ("bf16", "fp16"):
        if requested == "bf16" and not torch.cuda.is_bf16_supported(device=device.index) and device.name == "rocm":
            warnings.warn("bf16 not supported on this device; falling back to fp16")
            return "fp16"
        return requested

    if requested == "fp32":
        return "fp32"

    # Default: use backend recommendation.
    return device.recommended_dtype()


def resolve_compile(device: BackendDevice, requested: bool, mode: str = "default") -> bool:
    """Return whether torch.compile should be enabled."""
    if not requested:
        return False
    usable, reason = probe_compile(device, mode=mode)
    if not usable:
        warnings.warn(f"torch.compile requested but not usable: {reason}; using eager.")
    return usable


def resolve_flash_attn(device: BackendDevice, requested: bool) -> bool:
    """Return whether flash-attention should be enabled."""
    if not requested:
        return False
    usable, reason = probe_flash_attn(device)
    if not usable:
        warnings.warn(f"flash-attention requested but not usable: {reason}; using eager.")
    return usable
