"""Model-family registry.

Public API:
    from models import resolve_model_family, get_model_family, list_model_families, ModelFamily
"""

from models.mtp import (
    compute_mtp_depth_loss,
    compute_total_mtp_loss,
    shift_labels,
)
from models.registry import (
    ModelFamily,
    detect_model_family,
    get_model_family,
    list_model_families,
    resolve_model_family,
)

__all__ = [
    "ModelFamily",
    "compute_mtp_depth_loss",
    "compute_total_mtp_loss",
    "detect_model_family",
    "get_model_family",
    "list_model_families",
    "resolve_model_family",
    "shift_labels",
]
