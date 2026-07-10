"""Compute backend abstraction for the toolkit.

Public API:
    from backends import autodetect_backend, get_backend, default_device, BackendDevice

Backends are registered by name: rocm (the target), cpu (universal fallback).
"""

from backends.base import ComputeBackend
from backends.device import BackendDevice, default_device
from backends.registry import autodetect_backend, get_backend, list_backends

__all__ = [
    "ComputeBackend",
    "BackendDevice",
    "default_device",
    "autodetect_backend",
    "get_backend",
    "list_backends",
]
