"""Process-group environment detection.

Supports torchrun-style env vars (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`, ...) and
SLURM (`SLURM_PROCID`, `SLURM_NTASKS`, `SLURM_LOCALID`, ...). On a single GPU
with none of these set, we report world_size=1 and rank=0.
"""

from __future__ import annotations

import os


def _read_int(keys: tuple[str, ...], default: int) -> int:
    for key in keys:
        val = os.environ.get(key)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                continue
    return default


def _read_str(keys: tuple[str, ...], default: str) -> str:
    for key in keys:
        val = os.environ.get(key)
        if val is not None:
            return val
    return default


class ProcessGroupEnv:
    """Normalized view of distributed environment variables."""

    def __init__(self):
        # Rank identification.
        self.rank: int = _read_int(("RANK", "SLURM_PROCID"), 0)
        self.world_size: int = _read_int(("WORLD_SIZE", "SLURM_NTASKS"), 1)
        self.local_rank: int = _read_int(("LOCAL_RANK", "SLURM_LOCALID"), 0)

        # Network rendezvous.
        self.master_addr: str = _read_str(("MASTER_ADDR", "SLURM_LAUNCH_NODE_IPADDR"), "localhost")
        self.master_port: str = _read_str(("MASTER_PORT",), "29500")

        # Is anything distributed at all?
        self.is_distributed: bool = self.world_size > 1

    def __repr__(self) -> str:
        return (
            f"ProcessGroupEnv(rank={self.rank}, world_size={self.world_size}, "
            f"local_rank={self.local_rank}, master={self.master_addr}:{self.master_port})"
        )


def detect_process_group_env() -> ProcessGroupEnv:
    return ProcessGroupEnv()
