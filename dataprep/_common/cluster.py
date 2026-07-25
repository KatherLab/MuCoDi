"""Resolve staged-shard paths against $DATA_ROOT (data lives outside the repo)."""

from __future__ import annotations

import os
from pathlib import Path


def data_root() -> Path:
    """Data root: $DATA_ROOT if set, else ./data."""
    env = os.environ.get("DATA_ROOT")
    return Path(env) if env else Path("./data")


def resolve_data_path(path: str | os.PathLike) -> Path:
    """Resolve a relative path against data_root (absolute paths pass through)."""
    p = Path(path)
    return p if p.is_absolute() else data_root() / p
