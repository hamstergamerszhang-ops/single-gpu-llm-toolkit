"""Distributed training strategies.

Public API:
    from experimental import create_strategy, DistributedStrategy

STATUS (2026-07, verified during code review): this package is NOT wired into
the actual training entrypoint. `train_cpt.py` -- the real training script --
has its own inline DDP wrapping (`torch.nn.parallel.DistributedDataParallel`
called directly around line 1397) and its own inline FSDP wrapping
(`_wrap_fsdp()`, around line 757). Neither of those call sites imports or uses
`create_strategy` / `DDPStrategy` / `FSDPStrategy` from this package. Only
`tests/test_all.py` currently imports from here.

This package is a faithful, non-conflicting reimplementation of the same
DDP/FSDP wrapping logic behind a cleaner `DistributedStrategy` interface
(setup/wrap_model/unwrap_model/barrier/broadcast/all_reduce/no_sync/destroy).
It is NOT byte-for-byte identical to `train_cpt.py`'s inline versions --
notable differences spotted on inspection: `DDPStrategy.wrap_model` also sets
`output_device`, which `train_cpt.py`'s inline DDP call does not;
`FSDPStrategy.wrap_model` additionally configures `FullStateDictConfig` /
`StateDictType.FULL_STATE_DICT` for checkpointing, which `train_cpt.py`'s
inline `_wrap_fsdp()` does not do at wrap time (it handles state-dict
extraction separately elsewhere). Swapping `train_cpt.py` over to use this
package instead of its inline logic is a real, non-trivial refactor -- not a
safe mechanical one -- and hasn't been done. If you want to wire this in:
  1. Replace the inline DDP block in train_cpt.py with
     `create_strategy("ddp", backend, device).wrap_model(model,
     find_unused_parameters=windowed_freeze)`, and verify the
     `output_device` change doesn't alter behavior on this codebase's target
     hardware (AMD ROCm/MI300X).
  2. Replace `_wrap_fsdp()` similarly, and manually verify the
     FullStateDictConfig addition doesn't change how checkpoint save/resume
     behaves (train_cpt.py's own save/resume path may already assume its
     current, simpler state-dict handling).
  3. Re-run the full test suite AND manually trace both the `--ddp` and
     `--fsdp` flag paths end-to-end (a unit-test pass alone won't catch a
     subtle distributed-training regression).
Until that's done, treat this package as a parallel design sketch, not live
infrastructure.
"""

from experimental.base import DistributedStrategy
from experimental.registry import create_strategy

__all__ = ["DistributedStrategy", "create_strategy"]
