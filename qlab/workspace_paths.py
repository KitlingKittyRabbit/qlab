"""Resolve optional sibling workspaces without duplicating their contents."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_blueprint_root(workspace_root: Path | None = None) -> Path:
    """Return the configured or discoverable sibling blueprint repository."""
    configured = os.environ.get("QLAB_BLUEPRINT_DIR")
    if configured:
        root = Path(configured).expanduser().resolve()
        if root.is_dir():
            return root
        raise RuntimeError(f"QLAB_BLUEPRINT_DIR is not a directory: {root}")

    root = (workspace_root or Path(__file__).resolve().parents[2]).resolve()
    for name in ("quant-research-blueprints-private", "蓝图"):
        candidate = root / name
        if candidate.is_dir():
            return candidate
    raise RuntimeError(
        "Blueprint repository is not available. Set QLAB_BLUEPRINT_DIR or place "
        "a sibling repository at 蓝图 or quant-research-blueprints-private."
    )
