"""Model-family registry and MTP helpers.

Public API:
    from models import resolve_model_family, get_model_family, list_model_families, ModelFamily

STATUS: models/registry.py is a STRUCTURAL registry (layer paths, weight key
names, embed/lm_head/norm locations per architecture). It is NOT wired into
expand_model.py or mtp_head.py — those tools hardcode --layer-prefix instead.
It IS used by tests (which verify the registry's family detection logic).
modeling_custom.py has its own RUNTIME registry (_FAMILY_CLASS_CHAINS, for
CausalLM base-class selection). The two registries serve different purposes
(structural vs runtime) and are kept in sync by a bidirectional test
(test_modeling_custom_family_class_chain_covers_supported_families).
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
