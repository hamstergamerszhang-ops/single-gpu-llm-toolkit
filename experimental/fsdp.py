"""FullyShardedDataParallel strategy."""

from __future__ import annotations

import os
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn

from backends.base import ComputeBackend
from backends.device import BackendDevice
from experimental.base import DistributedStrategy
from experimental.env import detect_process_group_env

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
    from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
    _FSDP_AVAILABLE = True
except Exception:
    _FSDP_AVAILABLE = False


class FSDPStrategy(DistributedStrategy):
    """PyTorch FSDP wrapper with full-state-dict checkpoint helpers."""

    name = "fsdp"

    def __init__(self, backend: ComputeBackend, device: BackendDevice):
        super().__init__(backend, device)
        self._env = detect_process_group_env()
        self._pg_backend: str = "gloo"
        self._sharding_strategy_name: str = "full"

    def setup(self) -> None:
        if self._initialized:
            return
        if not _FSDP_AVAILABLE:
            raise RuntimeError("FSDP is not available in this PyTorch build.")
        if self._env.world_size <= 1:
            raise RuntimeError(
                "FSDP requested but world_size <= 1; use --distributed single instead."
            )

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

    def configure(self, sharding_strategy: str = "full") -> None:
        """Set FSDP sharding strategy: full, shard-grad-op, or no-shard."""
        self._sharding_strategy_name = sharding_strategy

    def wrap_model(self, model: nn.Module, **kwargs) -> nn.Module:
        if not _FSDP_AVAILABLE:
            raise RuntimeError("FSDP is not available in this PyTorch build.")

        decoder_layer_classes = kwargs.get("decoder_layer_classes", ())
        if not decoder_layer_classes:
            decoder_layer_classes = ("DecoderLayer", "Block")

        def _wrap_fn(module):
            return any(module.__class__.__name__.endswith(suffix) for suffix in decoder_layer_classes)

        from torch.distributed.fsdp import ShardingStrategy

        sharding_strategy_map = {
            "full": ShardingStrategy.FULL_SHARD,
            "shard-grad-op": ShardingStrategy.SHARD_GRAD_OP,
            "no-shard": ShardingStrategy.NO_SHARD,
        }
        strategy = sharding_strategy_map.get(
            self._sharding_strategy_name, ShardingStrategy.FULL_SHARD
        )

        device_id = self._local_rank if self.backend.name == "rocm" else None

        wrapped = FSDP(
            model,
            auto_wrap_policy=lambda_auto_wrap_policy(_wrap_fn),
            use_orig_params=True,
            limit_all_gathers=True,
            sharding_strategy=strategy,
            device_id=device_id,
        )

        # Configure state-dict type to full state dict on rank 0.
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        FSDP.set_state_dict_type(
            wrapped,
            StateDictType.FULL_STATE_DICT,
            state_dict_config=cfg,
        )
        return wrapped

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        if not _FSDP_AVAILABLE:
            return model
        if type(model).__name__ == "FullyShardedDataParallel":
            return model.module
        return model

    def model_state_dict(self, model: nn.Module) -> dict:
        if not _FSDP_AVAILABLE:
            return model.state_dict()
        return FSDP.state_dict(model)

    def load_model_state_dict(self, model: nn.Module, state_dict: dict) -> None:
        if not _FSDP_AVAILABLE:
            model.load_state_dict(state_dict)
            return
        FSDP.load_state_dict(model, state_dict)

    def optimizer_state_dict(
        self, model: nn.Module, optimizer: torch.optim.Optimizer
    ) -> dict:
        if not _FSDP_AVAILABLE:
            return optimizer.state_dict()
        return FSDP.optim_state_dict(model, optimizer)

    def load_optimizer_state_dict(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, state_dict: dict
    ) -> None:
        if not _FSDP_AVAILABLE:
            optimizer.load_state_dict(state_dict)
            return
        # FSDP sharded optimizer state loading is model-specific; callers should
        # use `shard_full_optim_state_dict` before reaching here. This method
        # exists only for interface parity.
        optimizer.load_state_dict(state_dict)

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
        if _FSDP_AVAILABLE and type(model).__name__ == "FullyShardedDataParallel":
            return model.no_sync()
        return super().no_sync(model)

    def destroy(self) -> None:
        if self._initialized and dist.is_initialized():
            dist.destroy_process_group()
            self._initialized = False
