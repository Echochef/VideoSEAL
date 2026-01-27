from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str) -> Path | None:
    val = os.getenv(name)
    if not val:
        return None
    try:
        return Path(val).expanduser().resolve()
    except Exception:
        return Path(val).expanduser()


def get_data_root() -> Path:
    """Base data root.

    Priority:
      1) DEFAULT_DATA_ROOT env
      2) <repo>/data
    """
    env_root = _env_path("DEFAULT_DATA_ROOT")
    if env_root:
        return env_root
    return (Path(__file__).resolve().parents[2] / "data").resolve()


def get_cache_root() -> Path:
    """Cache root.

    Priority:
      1) CACHE_DIR / CACHE_ROOT env
      2) <data>/cache
    """
    env_root = _env_path("CACHE_DIR") or _env_path("CACHE_ROOT")
    if env_root:
        return env_root
    return get_data_root() / "cache"


def get_indexes_root() -> Path:
    """Indexes root.

    Priority:
      1) INDEXES_ROOT env
      2) <data>/indexes
    """
    env_root = _env_path("INDEXES_ROOT")
    if env_root:
        return env_root
    return get_data_root() / "indexes"


def get_summaries_root() -> Path:
    """Summaries root.

    Priority:
      1) SUMMARIES_ROOT env
      2) <data>/summaries
    """
    env_root = _env_path("SUMMARIES_ROOT")
    if env_root:
        return env_root
    return get_data_root() / "summaries"


def get_frames_root() -> Path:
    """Frames root.

    Priority:
      1) FRAMES_ROOT env
      2) <data>/frames
    """
    env_root = _env_path("FRAMES_ROOT")
    if env_root:
        return env_root
    return get_data_root() / "frames"

