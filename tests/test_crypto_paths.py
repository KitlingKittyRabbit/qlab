from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from qlab.data.crypto import data_roots


def load_paths_module():
    sys.modules.pop("qlab.data.crypto.paths", None)
    return importlib.import_module("qlab.data.crypto.paths")


def test_data_root_requires_explicit_config_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("QLAB_CRYPTO_DATA_DIR", raising=False)
    monkeypatch.delenv("COINGLASS_DATA_DIR", raising=False)
    monkeypatch.setenv("QLAB_TRADE_ENV_PATH", str(tmp_path / "missing.env"))

    with pytest.raises(RuntimeError, match="QLAB_CRYPTO_DATA_DIR"):
        load_paths_module()


def test_data_root_can_switch_to_external_directory_and_stop_falling_back_on_rollback(tmp_path, monkeypatch):
    external_root = tmp_path / "external-crypto-data"
    monkeypatch.setenv("QLAB_CRYPTO_DATA_DIR", str(external_root))
    monkeypatch.delenv("COINGLASS_DATA_DIR", raising=False)
    monkeypatch.setenv("QLAB_TRADE_ENV_PATH", str(tmp_path / "missing.env"))

    paths = load_paths_module()

    assert paths.DATA_ROOT == external_root
    assert paths.CACHE_DIR == external_root / "caches"
    assert paths.MANIFEST_DIR == external_root / "manifests"
    assert paths.RAW_HISTORY_ROOT == external_root / "raw_history"
    assert not paths.CACHE_DIR.exists()
    assert not paths.MANIFEST_DIR.exists()
    assert not paths.RAW_HISTORY_ROOT.exists()

    paths.ensure_data_dirs()

    assert paths.CACHE_DIR.exists()
    assert paths.MANIFEST_DIR.exists()
    assert paths.RAW_HISTORY_ROOT.exists()

    monkeypatch.delenv("QLAB_CRYPTO_DATA_DIR", raising=False)
    sys.modules.pop("qlab.data.crypto.paths", None)

    with pytest.raises(RuntimeError, match="QLAB_CRYPTO_DATA_DIR"):
        importlib.import_module("qlab.data.crypto.paths")


def test_qlab_data_env_takes_priority_over_legacy_name(tmp_path, monkeypatch):
    primary_root = tmp_path / "primary-root"
    legacy_root = tmp_path / "legacy-root"
    monkeypatch.setenv("QLAB_CRYPTO_DATA_DIR", str(primary_root))
    monkeypatch.setenv("COINGLASS_DATA_DIR", str(legacy_root))

    paths = load_paths_module()

    assert paths.DATA_ROOT == primary_root


def test_data_root_can_be_read_from_trade_env_file(tmp_path, monkeypatch):
    external_root = tmp_path / "external-from-env-file"
    env_file = tmp_path / "trade.env"
    env_file.write_text(f"QLAB_CRYPTO_DATA_DIR={external_root}\n", encoding="utf-8")

    monkeypatch.delenv("QLAB_CRYPTO_DATA_DIR", raising=False)
    monkeypatch.delenv("COINGLASS_DATA_DIR", raising=False)
    monkeypatch.setenv("QLAB_TRADE_ENV_PATH", str(env_file))

    paths = load_paths_module()

    assert paths.DATA_ROOT == external_root


def test_trade_env_override_supports_relative_workspace_paths(tmp_path, monkeypatch):
    relative_env = Path("tmp") / "trade" / ".env"
    monkeypatch.setenv("QLAB_CRYPTO_DATA_DIR", str(tmp_path / "external-crypto-data"))
    monkeypatch.setenv("QLAB_TRADE_ENV_PATH", str(relative_env))
    paths = load_paths_module()

    assert paths.TRADE_ENV_PATH == paths.WORKSPACE_ROOT / relative_env


def test_canonical_resolver_explicit_argument_wins_without_environment(tmp_path, monkeypatch):
    env_file = tmp_path / "trade.env"
    env_file.write_text("QLAB_CRYPTO_DATA_DIR=from-file\n", encoding="utf-8")
    monkeypatch.setenv("QLAB_CRYPTO_DATA_DIR", str(tmp_path / "from-process"))
    monkeypatch.setenv("COINGLASS_DATA_DIR", str(tmp_path / "from-legacy-process"))
    monkeypatch.setenv("QLAB_TRADE_ENV_PATH", str(env_file))

    explicit = tmp_path / "explicit-root"

    assert data_roots.resolve_data_root(explicit) == explicit


def test_canonical_resolver_uses_same_relative_anchor_for_process_and_env_file(tmp_path, monkeypatch):
    relative = Path("relative") / "crypto-data"
    env_file = tmp_path / "trade.env"
    env_file.write_text("QLAB_CRYPTO_DATA_DIR=relative/crypto-data\n", encoding="utf-8")
    monkeypatch.setenv("QLAB_TRADE_ENV_PATH", str(env_file))
    monkeypatch.delenv("QLAB_CRYPTO_DATA_DIR", raising=False)
    monkeypatch.delenv("COINGLASS_DATA_DIR", raising=False)

    from_file = data_roots.resolve_data_root()

    monkeypatch.setenv("QLAB_CRYPTO_DATA_DIR", str(relative))
    from_process = data_roots.resolve_data_root()

    assert from_file == from_process == data_roots.WORKSPACE_ROOT / relative


def test_canonical_resolver_accepts_coinglass_process_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("QLAB_CRYPTO_DATA_DIR", raising=False)
    monkeypatch.setenv("COINGLASS_DATA_DIR", str(tmp_path / "coinglass-root"))
    monkeypatch.setenv("QLAB_TRADE_ENV_PATH", str(tmp_path / "missing.env"))

    assert data_roots.resolve_data_root() == tmp_path / "coinglass-root"


def test_canonical_resolver_accepts_coinglass_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / "trade.env"
    expected = tmp_path / "coinglass-from-file"
    env_file.write_text(f"COINGLASS_DATA_DIR={expected}\n", encoding="utf-8")
    monkeypatch.delenv("QLAB_CRYPTO_DATA_DIR", raising=False)
    monkeypatch.delenv("COINGLASS_DATA_DIR", raising=False)
    monkeypatch.setenv("QLAB_TRADE_ENV_PATH", str(env_file))

    assert data_roots.resolve_data_root() == expected


def test_canonical_resolver_does_not_use_existing_sibling_as_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(data_roots, "WORKSPACE_ROOT", tmp_path)
    (tmp_path / "qlab_crypto_data").mkdir()
    monkeypatch.delenv("QLAB_CRYPTO_DATA_DIR", raising=False)
    monkeypatch.delenv("COINGLASS_DATA_DIR", raising=False)
    monkeypatch.setenv("QLAB_TRADE_ENV_PATH", str(tmp_path / "missing.env"))

    with pytest.raises(RuntimeError, match="Crypto data root is not configured"):
        data_roots.resolve_data_root()
