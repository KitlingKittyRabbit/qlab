from __future__ import annotations

from pathlib import Path

import pytest

from qlab.workspace_paths import resolve_blueprint_root


def test_formal_blueprint_sibling_precedes_legacy_directory(tmp_path: Path, monkeypatch):
    formal = tmp_path / "quant-research-blueprints-private"
    legacy = tmp_path / "蓝图"
    formal.mkdir()
    legacy.mkdir()

    monkeypatch.delenv("QLAB_BLUEPRINT_DIR", raising=False)

    assert resolve_blueprint_root(tmp_path) == formal


def test_missing_blueprint_directory_fails_closed(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("QLAB_BLUEPRINT_DIR", raising=False)

    with pytest.raises(RuntimeError, match="Blueprint repository is not available"):
        resolve_blueprint_root(tmp_path)
