"""Pure helper functions for Multi-Token-Prediction (MTP) training.

These functions are detached from any specific modeling file so they can be
unit-tested (and used) without importing transformers.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def shift_labels(input_ids: torch.Tensor, depth: int, ignore_index: int = -100) -> torch.Tensor:
    """Create MTP depth-`depth` labels by shifting `input_ids` left by `depth+1`.

    For depth 0 this is the standard next-token prediction target. For depth 1 it
    predicts the token two positions ahead, etc. Positions without a target are
    filled with `ignore_index`.

    Args:
        input_ids: (B, T) token IDs.
        depth: MTP depth index (0-based).
        ignore_index: Value to use where no target exists.

    Returns:
        (B, T) label tensor aligned with logits at each position.
    """
    b, t = input_ids.shape
    labels = input_ids.new_full((b, t), ignore_index)
    shift = depth + 1
    if t > shift:
        labels[:, : t - shift] = input_ids[:, shift:]
    return labels


def compute_mtp_depth_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """Cross-entropy loss for one MTP depth.

    Args:
        logits: (B, T, V) MTP logits for this depth.
        labels: (B, T) label tensor (with ignore_index for ignored positions).
        weight: Scalar multiplier for this depth's loss.

    Returns:
        Scalar weighted loss (mean over non-ignored positions).
    """
    if weight == 0.0:
        return logits.new_tensor(0.0)
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)
    loss = F.cross_entropy(logits, labels, ignore_index=-100, reduction="mean")
    return loss * weight


def compute_total_mtp_loss(
    all_mtp_logits: list[torch.Tensor],
    input_ids: torch.Tensor,
    global_weight: float,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Sum MTP losses across depths.

    Args:
        all_mtp_logits: List of (B, T, V) logits, one per MTP depth.
        input_ids: (B, T) token IDs used to derive targets.
        global_weight: `mtp_loss_weight` applied to the summed depth losses.
        ignore_index: Label value for ignored positions.

    Returns:
        Scalar total MTP loss.
    """
    if global_weight == 0.0 or not all_mtp_logits:
        return all_mtp_logits[0].new_tensor(0.0) if all_mtp_logits else torch.tensor(0.0)

    total = all_mtp_logits[0].new_tensor(0.0)
    for depth, logits in enumerate(all_mtp_logits):
        labels = shift_labels(input_ids, depth=depth, ignore_index=ignore_index)
        total = total + compute_mtp_depth_loss(logits, labels, weight=1.0)
    return total * global_weight
