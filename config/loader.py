"""Recipe/preset loader.

Recipes are declarative TOML (or YAML, if PyYAML is installed) files that set
defaults for a training run. CLI flags always override recipe values, and
recipes can include other recipes via `extends`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from config.presets import get_preset


def _load_toml(path: str | os.PathLike) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:
        raise RuntimeError(
            f"TOML recipes require Python 3.11+ (tomllib is stdlib since 3.11). "
            f"Your Python is {__import__('sys').version_info[0]}.{__import__('sys').version_info[1]}. "
            f"Either upgrade to 3.11+ or use a YAML recipe (.yaml/.yml) instead."
        )
    with open(path, "rb") as f:
        return tomllib.load(f)


def _load_yaml(path: str | os.PathLike) -> dict:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load YAML recipe {path}; is PyYAML installed?"
        ) from exc


def load_recipe(path: str | os.PathLike) -> dict:
    """Load a TOML or YAML recipe file."""
    p = Path(path)
    if p.suffix in (".toml",):
        return _load_toml(p)
    if p.suffix in (".yaml", ".yml"):
        return _load_yaml(p)
    # Try TOML first, then YAML, for extensionless paths.
    try:
        return _load_toml(p)
    except Exception as exc:
        # Don't swallow silently (repo convention: no bare `except: pass`).
        # This is a deliberate format-probing fallback -- a TOML parse
        # failure on an extensionless path is expected if the file is
        # actually YAML -- but the original TOML error must still be visible
        # for debugging a genuinely malformed file (both parsers failing back
        # to back is confusing without a trace of the first attempt).
        print(f"[config.loader] NOTE: {p} did not parse as TOML "
              f"({exc!r}); trying YAML instead.")
        return _load_yaml(p)


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge `override` into `base` recursively."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_recipe(
    path: str | os.PathLike | None,
    base_defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a recipe and any `extends` chain, then return the merged config.

    Args:
        path: Path to the recipe file, or None for an empty recipe.
        base_defaults: Baseline defaults (e.g., argparse defaults) to merge under.

    Returns:
        Merged configuration dict. Recipe values override base defaults.
    """
    if base_defaults is None:
        base_defaults = {}

    if path is None:
        recipe = {}
    else:
        recipe = load_recipe(path)

    # Handle `extends = "other.toml"` inheritance.
    merged = dict(base_defaults)
    extends = recipe.get("extends")
    if extends:
        parent_path = Path(path).parent / extends
        parent = resolve_recipe(parent_path, base_defaults={})
        merged = _deep_merge(merged, parent)

    merged = _deep_merge(merged, recipe)
    # `extends` itself is not a training arg; remove it.
    merged.pop("extends", None)
    return merged


def apply_preset(config: dict[str, Any], preset_name: str) -> dict[str, Any]:
    """Merge a hardware preset into `config`. CLI values already present win."""
    preset = get_preset(preset_name)
    # Preset is applied *under* explicit config values.
    result = _deep_merge(preset, config)
    return result
