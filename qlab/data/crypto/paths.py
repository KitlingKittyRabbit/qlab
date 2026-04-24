from __future__ import annotations

import os
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
QLAB_REPO_ROOT = WORKSPACE_ROOT / "qlab"
DEFAULT_DATA_ROOT = QLAB_REPO_ROOT / "data" / "crypto"
TRADE_ENV_PATH = WORKSPACE_ROOT / "trade" / "crypto_signal" / ".env"


def _resolve_data_root() -> Path:
    raw_value = os.environ.get("QLAB_CRYPTO_DATA_DIR", "").strip() or os.environ.get(
        "COINGLASS_DATA_DIR", ""
    ).strip()
    if not raw_value:
        return DEFAULT_DATA_ROOT
    candidate = Path(raw_value).expanduser()
    return candidate if candidate.is_absolute() else (WORKSPACE_ROOT / candidate)


DATA_ROOT = _resolve_data_root()
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


ensure_data_dirs()
