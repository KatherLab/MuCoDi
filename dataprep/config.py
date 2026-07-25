"""Load dataset profiles and resolve their shard paths against ``DATA_ROOT``."""

from __future__ import annotations

from pathlib import Path

import yaml

from dataprep._common.cluster import resolve_data_path

_YAML_PATH = Path(__file__).parent / "datasets.yaml"


def load_profile(name: str) -> dict:
    """Load a named profile from ``datasets.yaml``."""
    with _YAML_PATH.open(encoding="utf-8") as file:
        profiles = yaml.safe_load(file)

    if not isinstance(profiles, dict) or name not in profiles:
        available = list(profiles) if isinstance(profiles, dict) else []
        raise ValueError(f"Unknown dataset profile '{name}'. Available: {available}")

    profile = dict(profiles[name])
    if "teacher_dims" not in profile or not isinstance(profile["teacher_dims"], dict):
        raise ValueError(f"Dataset profile '{name}' must define a teacher_dims mapping.")
    if "shard_dir" not in profile:
        raise ValueError(f"Dataset profile '{name}' must define shard_dir.")

    profile["shard_dir"] = str(resolve_data_path(profile["shard_dir"]))
    return profile


def get_teacher_dims_str(profile: dict) -> str:
    """Convert a profile mapping to the CLI form ``name:dim,name:dim``."""
    return ",".join(f"{name}:{dimension}" for name, dimension in profile["teacher_dims"].items())
