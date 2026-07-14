"""Abstract distributed strategy."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from backends.base import ComputeBackend
from backends.device import BackendDevice


class DistributedStrategy(ABC):
    """Pluggable wrapper around single-device, DDP, or FSDP execution.

    Each strategy owns the lifecycle: environment discovery, process-group setup,
    model wrapping, checkpoint state-dict conversion, and teardown.
    """

    name: str = ""

    def __init__(self, backend: ComputeBackend, device: BackendDevice):
        self.backend = backend
        self.device = device
        self._world_size: int = 1
        self._rank: int = 0
        self._local_rank: int = 0
        self._is_main: bool = True
        self._initialized: bool = False

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def local_rank(self) -> int:
        return self._local_rank

    @property
    def is_main(self) -> bool:
        return self._is_main

    @abstractmethod
    def setup(self) -> None:
        """Initialize process group and related state."""
        ...

    @abstractmethod
    def wrap_model(self, model: nn.Module, **kwargs) -> nn.Module:
        """Wrap the model for distributed training, if needed."""
        ...

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        """Return the inner model for rank-0-only operations."""
        return model

    def model_state_dict(self, model: nn.Module) -> dict:
        """Return a full, saveable state dict from the (possibly wrapped) model."""
        return model.state_dict()

    def load_model_state_dict(self, model: nn.Module, state_dict: dict) -> None:
        """Load a state dict into the (possibly wrapped) model."""
        model.load_state_dict(state_dict)

    def optimizer_state_dict(
        self, model: nn.Module, optimizer: torch.optim.Optimizer
    ) -> dict:
        """Return a full optimizer state dict."""
        return optimizer.state_dict()

    def load_optimizer_state_dict(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, state_dict: dict
    ) -> None:
        """Load a full optimizer state dict."""
        optimizer.load_state_dict(state_dict)

    def barrier(self) -> None:
        """Synchronize all ranks, if running distributed."""
        return

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> None:
        """Broadcast a tensor from `src` to all ranks, if distributed."""
        return

    def all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> None:
        """In-place all-reduce, if distributed."""
        return

    def no_sync(self, model: nn.Module):
        """Context manager to disable gradient sync for gradient accumulation."""
        return _NullNoSync()

    @abstractmethod
    def destroy(self) -> None:
        """Tear down process group."""
        ...


class _NullNoSync:
    """No-op context manager used by single-device strategy."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
