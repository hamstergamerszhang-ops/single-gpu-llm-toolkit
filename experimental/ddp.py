"""DistributedDataParallel strategy."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from backends.base import ComputeBackend
from backends.device import BackendDevice
from experimental.base import DistributedStrategy
from experimental.env import detect_process_group_env


class DDPStrategy(DistributedStrategy):
    """PyTorch DDP wrapper with process-group setup and teardown."""

    name = "ddp"

    def __init__(self, backend: ComputeBackend, device: BackendDevice):
        super().__init__(backend, device)
        self._env = detect_process_group_env()
        self._pg_backend: str = "gloo"

    def setup(self) -> None:
        if self._initialized:
            return
        if self._env.world_size <= 1:
            raise RuntimeError(
                "DDP requested but world_size <= 1; use --distributed single instead."
            )

        # Prefer nccl on ROCm, gloo otherwise.
        if self.backend.name == "rocm" and dist.is_nccl_available():
            self._pg_backend = "nccl"
        else:
            self._pg_backend = "gloo"

        os.environ.setdefault("MASTER_ADDR", self._env.master_addr)
        os.environ.setdefault("MASTER_PORT", self._env.master_port)

        dist.init_process_group(
            backend=self._pg_backend,
            rank=self._env.rank,
            world_size=self._env.world_size,
        )

        self._world_size = self._env.world_size
        self._rank = self._env.rank
        self._local_rank = self._env.local_rank
        self._is_main = self._rank == 0
        self._initialized = True

        if self.backend.name == "rocm":
            torch.cuda.set_device(self._local_rank)

    def wrap_model(self, model: nn.Module, **kwargs) -> nn.Module:
        find_unused = kwargs.get("find_unused_parameters", False)
        return DistributedDataParallel(
            model,
            device_ids=[self._local_rank] if self.backend.name == "rocm" else None,
            output_device=self._local_rank if self.backend.name == "rocm" else None,
            find_unused_parameters=find_unused,
        )

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        if type(model).__name__ == "DistributedDataParallel":
            return model.module
        return model

    def barrier(self) -> None:
        if self._initialized and dist.is_initialized():
            dist.barrier()

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> None:
        if self._initialized and dist.is_initialized():
            dist.broadcast(tensor, src=src)

    def all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> None:
        if not (self._initialized and dist.is_initialized()):
            return
        dist_op = dist.ReduceOp.SUM if op == "sum" else dist.ReduceOp.AVG
        dist.all_reduce(tensor, op=dist_op)

    def no_sync(self, model: nn.Module):
        if type(model).__name__ == "DistributedDataParallel":
            return model.no_sync()
        return super().no_sync(model)

    def destroy(self) -> None:
        if self._initialized and dist.is_initialized():
            dist.destroy_process_group()
            self._initialized = False
