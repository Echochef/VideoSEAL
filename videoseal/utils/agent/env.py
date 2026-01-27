from __future__ import annotations

import os

from videoseal.utils.api.mllm import MLLMClient


def require_env(name: str) -> str:
    v = os.getenv(name)
    if v is None or not str(v).strip():
        raise RuntimeError(f"Missing required env var: {name}")
    return str(v).strip()


def env_flag(name: str, default: bool = False) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def env_bool_strict(name: str, default: bool) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise RuntimeError(f"Invalid {name}={v!r}; use 1/0.")


def require_int_env(name: str) -> int:
    v = require_env(name)
    try:
        return int(v)
    except Exception as e:
        raise RuntimeError(f"Invalid int for env var {name}: {v!r}") from e


def require_float_env(name: str) -> float:
    v = require_env(name)
    try:
        return float(v)
    except Exception as e:
        raise RuntimeError(f"Invalid float for env var {name}: {v!r}") from e


def env_int(name: str, default: int) -> int:
    v = (os.getenv(name) or "").strip()
    if not v:
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)


def env_float(name: str, default: float) -> float:
    v = (os.getenv(name) or "").strip()
    if not v:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def build_mllm_client_from_env_prefix(prefix: str) -> MLLMClient:
    p = (prefix or "").strip().upper()
    base_url = require_env(f"{p}_API_BASE")
    api_key = require_env(f"{p}_API_KEY")
    model = require_env(f"{p}_MODEL")
    backend = require_env(f"{p}_BACKEND")
    return MLLMClient(base_url=base_url, api_key=api_key, model=model, backend=backend)


def env_int_strict(name: str, default: int) -> int:
    v = (os.getenv(name) or "").strip()
    if not v:
        return int(default)
    try:
        return int(v)
    except Exception as e:
        raise RuntimeError(f"Invalid int for env var {name}: {v!r}") from e


def env_float_strict(name: str, default: float) -> float:
    v = (os.getenv(name) or "").strip()
    if not v:
        return float(default)
    try:
        return float(v)
    except Exception as e:
        raise RuntimeError(f"Invalid float for env var {name}: {v!r}") from e


def env_int_first(names: list[str] | tuple[str, ...], default: int) -> int:
    for name in names:
        v = (os.getenv(name) or "").strip()
        if v:
            return require_int_env(name)
    return int(default)


def env_float_first(names: list[str] | tuple[str, ...], default: float) -> float:
    for name in names:
        v = (os.getenv(name) or "").strip()
        if v:
            return require_float_env(name)
    return float(default)
