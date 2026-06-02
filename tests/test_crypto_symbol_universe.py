from __future__ import annotations

from pathlib import Path

import pytest

from qlab.data.crypto.symbol_universe import (
    load_symbol_list_file,
    normalize_symbol_list,
    parse_symbol_csv,
    resolve_target_symbols,
)


def test_normalize_symbol_list_dedupes_and_uppercases():
    assert normalize_symbol_list([" btc ", "ETH", "btc", "", "sol"]) == [
        "BTC", "ETH", "SOL"]


def test_parse_symbol_csv_handles_commas_and_newlines():
    assert parse_symbol_csv("btc, eth\nsol") == ["BTC", "ETH", "SOL"]


def test_load_symbol_list_file_supports_comments_and_whitespace(tmp_path: Path):
    path = tmp_path / "symbols.txt"
    path.write_text("btc, eth\n# comment\nsol ada\n", encoding="utf-8")

    assert load_symbol_list_file(path) == ["BTC", "ETH", "SOL", "ADA"]


def test_resolve_target_symbols_prefers_csv_over_file_and_default(tmp_path: Path):
    path = tmp_path / "symbols.txt"
    path.write_text("ada\nsol\n", encoding="utf-8")

    assert resolve_target_symbols("btc,eth", file_value=path, default=[
                                  "DOGE"]) == ["BTC", "ETH"]
    assert resolve_target_symbols(None, file_value=path, default=[
                                  "DOGE"]) == ["ADA", "SOL"]
    assert resolve_target_symbols(None, default=["DOGE"]) == ["DOGE"]


def test_load_symbol_list_file_raises_for_missing_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_symbol_list_file(tmp_path / "missing.txt")
