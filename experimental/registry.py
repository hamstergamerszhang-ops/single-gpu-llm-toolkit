"""Strategy factory.

Given a backend, device, and user-requested strategy name, create the right
`DistributedStrategy` implementation. The factory also decides whether DDP/FSDP
make sense given the detected world size.

NOT CURRENTLY WIRED IN: `train_cpt.py` (the real training entrypoint) does not
call `create_strategy` -- it has its own inline DDP/FSDP wrapping. See the
`distributed/__init__.py` module docstring for the full status note and what
wiring this in would require.
"""

from __future__ import annotations

from backends.base import ComputeBackend
from backends.device import BackendDevice
from experimental.base import DistributedStrategy
from experimental.ddp import DDPStrategy
from experimental.env import detect_process_group_env
from experimental.fsdp import FSDPStrategy
from experimental.single import SingleDeviceStrategy


def create_strategy(
    name: str,
    backend: ComputeBackend,
    device: BackendDevice,
    **strategy_kwargs,
) -> DistributedStrategy:
    """Create a distributed strategy by name.

    Args:
        name: "auto", "single", "ddp", or "fsdp".
        backend: Compute backend in use.
        device: Device handle for this process.
        **strategy_kwargs: Extra args passed to the strategy (e.g. sharding_strategy).

    "auto" picks ddp/fsdp if world_size > 1, otherwise single. If a distributed
    strategy is requested but world_size is 1, we fall back to single with a
    warning so tests and single-GPU runs don't crash.
    """
    env = detect_process_group_env()

    if name == "auto":
        name = "ddp" if env.is_distributed else "single"

    if name in ("ddp", "fsdp") and not env.is_distributed:
        import warnings
        warnings.warn(
            f"--distributed {name} requested but only one process detected; "
            "falling back to single-device strategy.",
            stacklevel=2,
        )
        name = "single"

    if name == "single":
        strategy: DistributedStrategy = SingleDeviceStrategy(backend, device)
    elif name == "ddp":
        strategy = DDPStrategy(backend, device)
    elif name == "fsdp":
        strategy = FSDPStrategy(backend, device)
        sharding = strategy_kwargs.get("sharding_strategy", "full")
        strategy.configure(sharding)
    else:
        raise ValueError(
            f"Unknown distributed strategy '{name}'. Choose from: auto, single, ddp, fsdp."
        )

    return strategy
