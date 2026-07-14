"""Single-device (non-distributed) strategy."""

import torch.nn as nn

from backends.base import ComputeBackend
from backends.device import BackendDevice
from experimental.base import DistributedStrategy


class SingleDeviceStrategy(DistributedStrategy):
    """No-op strategy for one device, one process."""

    name = "single"

    def __init__(self, backend: ComputeBackend, device: BackendDevice):
        super().__init__(backend, device)
        self._world_size = 1
        self._rank = 0
        self._local_rank = 0
        self._is_main = True
        self._initialized = True

    def setup(self) -> None:
        pass

    def wrap_model(self, model: nn.Module, **kwargs) -> nn.Module:
        return model

    def destroy(self) -> None:
        pass
