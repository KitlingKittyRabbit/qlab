"""Canonical crypto data-root resolution without import-time validation."""

from __future__ import annotations

import os
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
QLAB_REPO_ROOT = WORKSPACE_ROOT / "qlab"
LEGACY_REPO_DATA_ROOT = QLAB_REPO_ROOT / "data" / "crypto"
LEGACY_TRADE_ENV_PATH = WORKSPACE_ROOT / "trade" / "crypto_signal" / ".env"


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def resolve_trade_env_path() -> Path:
    raw_value = os.environ.get("QLAB_TRADE_ENV_PATH", "").strip() or os.environ.get(
        "QLAB_CRYPTO_ENV_PATH", ""
    ).strip()
    if raw_value:
        candidate = Path(raw_value).expanduser()
        return candidate if candidate.is_absolute() else WORKSPACE_ROOT / candidate
    return LEGACY_TRADE_ENV_PATH


def _anchor_workspace(raw_value: str) -> Path:
    candidate = Path(raw_value).expanduser()
    return candidate if candidate.is_absolute() else WORKSPACE_ROOT / candidate


def resolve_data_root(data_root: Path | str | None = None) -> Path:
    """Resolve one data root using the project-wide precedence contract.

    Explicit arguments win over process environment, which wins over the
    compatible trade environment file. No sibling-directory fallback exists.
    """
    if data_root is not None:
        raw_value = str(data_root).strip()
        if not raw_value:
            raise ValueError("data_root must not be empty")
    else:
        raw_value = os.environ.get("QLAB_CRYPTO_DATA_DIR", "").strip() or os.environ.get(
            "COINGLASS_DATA_DIR", ""
        ).strip()
        if not raw_value:
            env_file_values = _load_env_file(resolve_trade_env_path())
            raw_value = env_file_values.get("QLAB_CRYPTO_DATA_DIR", "").strip() or env_file_values.get(
                "COINGLASS_DATA_DIR", ""
            ).strip()
    if not raw_value:
        raise RuntimeError(
            "Crypto data root is not configured. Set QLAB_CRYPTO_DATA_DIR "
            "(preferred) or COINGLASS_DATA_DIR. For temporary compatibility, "
            f"you may point that env var at the legacy repo path: {LEGACY_REPO_DATA_ROOT}"
        )
    return _anchor_workspace(raw_value)
