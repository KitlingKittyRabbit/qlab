from __future__ import annotations

from pathlib import Path

from .data_roots import (
    LEGACY_REPO_DATA_ROOT,
    WORKSPACE_ROOT,
    resolve_data_root,
    resolve_trade_env_path,
)


TRADE_ENV_PATH = resolve_trade_env_path()


DATA_ROOT = resolve_data_root()
CACHE_DIR = DATA_ROOT / "caches"
MANIFEST_DIR = DATA_ROOT / "manifests"
RAW_HISTORY_ROOT = DATA_ROOT / "raw_history"


def ensure_data_dirs() -> None:
    for directory in [DATA_ROOT, CACHE_DIR, MANIFEST_DIR, RAW_HISTORY_ROOT]:
        directory.mkdir(parents=True, exist_ok=True)


def cache_path(filename: str) -> Path:
    ensure_data_dirs()
    return CACHE_DIR / filename


def manifest_path(filename: str) -> Path:
    ensure_data_dirs()
    return MANIFEST_DIR / filename
