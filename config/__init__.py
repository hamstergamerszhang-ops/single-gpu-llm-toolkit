"""Configuration recipes and hardware presets.

Public API:
    from config import resolve_recipe, apply_preset, list_presets, get_preset
"""

from config.loader import apply_preset, load_recipe, resolve_recipe
from config.presets import get_preset, list_presets, suggest_preset

__all__ = [
    "apply_preset",
    "get_preset",
    "load_recipe",
    "list_presets",
    "resolve_recipe",
    "suggest_preset",
]
