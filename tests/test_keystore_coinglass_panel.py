from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from qlab.data.crypto import keystore_coinglass_factors as registry
from qlab.data.crypto import keystore_coinglass_panel as panel_builder


def sample_payloads() -> dict[str, dict[str, pd.DataFrame]]:
    idx_1h = pd.date_range("2026-01-01", periods=36, freq="1h", tz="UTC", name="ts")
    idx_12h = pd.date_range("2026-01-01", periods=3, freq="12h", tz="UTC", name="ts")
    payloads = {"ksv4_1h": {}, "ksv4_12h": {}}
    for symbol, offset in {"AAA": 0.0, "BBB": 10.0}.items():
        payloads["ksv4_1h"][f"{symbol}_ob_pair"] = pd.DataFrame(
            {
                "bids_usd": [10.0 + offset + value for value in range(len(idx_1h))],
                "asks_usd": [50.0 + offset - value for value in range(len(idx_1h))],
            },
            index=idx_1h,
        )
        payloads["ksv4_12h"][f"{symbol}_futures_whale_index"] = pd.DataFrame(
            {"whale_index_value": [1.0 + offset, 2.0 + offset, 3.0 + offset]},
            index=idx_12h,
        )
    return payloads


def selected_registry() -> pd.DataFrame:
    return registry.feature_registry_for_panel(
        ("ob_pair_imbalance__1h", "whale_index_raw__12h"),
        frequency="1h",
    )


def test_extract_feature_series_applies_ob_imbalance_column_order():
    row = registry.feature_registry_for_panel(("ob_pair_imbalance__1h",)).iloc[0]
    frame = pd.DataFrame(
        {"bids_usd": [30.0], "asks_usd": [10.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-01-01T00:00:00Z")], name="ts"),
    )

    series = panel_builder.extract_feature_series(row, frame)

    assert series.iloc[0] == pytest.approx(0.5)


def test_ksv4_panel_builder_uses_1h_grid_without_carry_forward():
    artifacts = panel_builder.build_panel_from_payloads(
        admitted_symbols=["AAA", "BBB"],
        cache_payloads=sample_payloads(),
        registry_frame=selected_registry(),
        min_common_rows=20,
    )

    panel = artifacts.panel
    assert panel.index.names == ["decision_ts", "symbol"]
    assert list(panel.columns) == [
        "label_ts",
        "signal_bar_end_ts",
        "ob_pair_imbalance__1h",
        "whale_index_raw__12h",
    ]
    assert artifacts.summary["rows"].tolist() == [25, 25]

    non_null_whale = panel["whale_index_raw__12h"].dropna().index.get_level_values("decision_ts").unique()
    assert list(non_null_whale) == [
        pd.Timestamp("2026-01-01 12:00:00+00:00"),
        pd.Timestamp("2026-01-02 00:00:00+00:00"),
        pd.Timestamp("2026-01-02 12:00:00+00:00"),
    ]
    assert panel["signal_bar_end_ts"].equals(
        pd.Series(
            panel.index.get_level_values("decision_ts"),
            index=panel.index,
            name="signal_bar_end_ts",
        )
    )
    assert panel["whale_index_raw__12h"].isna().sum() > 0


def test_ksv4_panel_builder_fails_closed_when_cache_key_missing():
    payloads = sample_payloads()
    del payloads["ksv4_1h"]["BBB_ob_pair"]

    with pytest.raises(ValueError, match="missing ksv4 cache entry"):
        panel_builder.build_panel_from_payloads(
            admitted_symbols=["AAA", "BBB"],
            cache_payloads=payloads,
            registry_frame=selected_registry(),
            min_common_rows=1,
        )


def test_build_panel_requires_explicit_universe_input():
    with pytest.raises(ValueError, match="admitted_symbols or universe_audit_path"):
        panel_builder.build_panel(feature_names=("ob_pair_imbalance__1h",), min_common_rows=1)


def test_load_admitted_symbols_from_audit(tmp_path: Path):
    audit = tmp_path / "universe.csv"
    audit.write_text("symbol,base_admitted\nbtc,yes\neth,no\nsol,yes\n", encoding="utf-8")

    assert panel_builder.load_admitted_symbols_from_audit(audit) == ["BTC", "SOL"]
