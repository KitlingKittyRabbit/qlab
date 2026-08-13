from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from qlab.data.crypto import keystore_coinglass_factors as registry
from qlab.data.crypto import keystore_coinglass_panel as panel_builder


def test_logical_source_scope_resolves_to_native_cache_timeframe() -> None:
    assert panel_builder.signal_timeframe_from_scope("ksv4_12h") == "12h"
    assert (
        panel_builder.CACHE_FILENAME_BY_SCOPE[
            panel_builder.signal_timeframe_from_scope("ksv4_1d")
        ]
        == "keystore_coinglass_v4_1d_cache.pkl"
    )


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


def test_observed_overlay_is_copy_on_write_and_latest_observation_wins():
    original = {
        "ksv4_1h": {
            "BTC_fr": pd.DataFrame(
                {"fr_close": [1.0, 2.0]},
                index=pd.date_range(
                    "2026-01-01", periods=2, freq="1h", tz="UTC", name="ts"
                ),
            )
        }
    }
    observed = pd.DataFrame(
        {"fr_close": [20.0, 3.0]},
        index=pd.date_range(
            "2026-01-01 01:00", periods=2, freq="1h", tz="UTC", name="ts"
        ),
    )

    overlaid = panel_builder.overlay_observed_cache_frames(
        original, {("ksv4_1h", "BTC_fr"): observed}
    )

    assert original["ksv4_1h"]["BTC_fr"].iloc[-1, 0] == 2.0
    assert overlaid["ksv4_1h"]["BTC_fr"]["fr_close"].tolist() == [1.0, 20.0, 3.0]


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
        pd.Timestamp("2026-01-01 00:00:00+00:00"),
        pd.Timestamp("2026-01-01 12:00:00+00:00"),
        pd.Timestamp("2026-01-02 00:00:00+00:00"),
    ]
    assert panel["signal_bar_end_ts"].equals(
        pd.Series(
            panel.index.get_level_values("decision_ts"),
            index=panel.index,
            name="signal_bar_end_ts",
        )
    )
    assert panel["whale_index_raw__12h"].isna().sum() > 0


def test_ksv4_panel_builder_normalizes_mixed_datetime_precision_end_to_end():
    payloads = sample_payloads()
    for cache_key, frame in payloads["ksv4_1h"].items():
        payloads["ksv4_1h"][cache_key] = frame.set_axis(
            pd.DatetimeIndex(frame.index).as_unit("ns")
        )
    for cache_key, frame in payloads["ksv4_12h"].items():
        payloads["ksv4_12h"][cache_key] = frame.set_axis(
            pd.DatetimeIndex(frame.index).as_unit("us")
        )

    artifacts = panel_builder.build_panel_from_payloads(
        admitted_symbols=["AAA", "BBB"],
        cache_payloads=payloads,
        registry_frame=selected_registry(),
        min_common_rows=20,
    )

    decision_ts = artifacts.panel.index.get_level_values("decision_ts")
    assert decision_ts.dtype == pd.DatetimeTZDtype(unit="ns", tz="UTC")
    assert artifacts.summary["rows"].tolist() == [25, 25]
    assert artifacts.panel["whale_index_raw__12h"].notna().sum() == 6


def test_start_and_end_labels_meet_at_same_native_bar_end() -> None:
    start_index = pd.DatetimeIndex(
        [pd.Timestamp("2026-01-01 00:00:00Z")], name="ts"
    )
    end_index = pd.DatetimeIndex(
        [pd.Timestamp("2026-01-01 12:00:00Z")], name="ts"
    )

    start_bar_end = panel_builder.source_index_to_native_bar_end(
        start_index, signal_timeframe="12h", timestamp_kind="bar_start"
    )
    end_bar_end = panel_builder.source_index_to_native_bar_end(
        end_index, signal_timeframe="12h", timestamp_kind="bar_end"
    )

    assert start_bar_end.equals(end_bar_end)
    assert start_bar_end[0] == pd.Timestamp("2026-01-01 12:00:00Z")


def test_unknown_timestamp_kind_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported timestamp_kind"):
        panel_builder.source_index_to_native_bar_end(
            pd.DatetimeIndex([pd.Timestamp("2026-01-01T00:00:00Z")]),
            signal_timeframe="1h",
            timestamp_kind="unknown",
        )


def test_frozen_l2_registry_has_exact_audited_identity() -> None:
    frame = registry.base_panel_registry()

    assert len(frame) == 68
    assert frame["endpoint"].nunique() == 19
    assert set(frame["timestamp_kind"]) == {"bar_start", "bar_end"}
    assert set(frame.loc[frame["endpoint"] == "futures_whale_index", "timestamp_kind"]) == {
        "bar_end"
    }
    assert set(frame.loc[frame["endpoint"] == "basis", "timestamp_kind"]) == {
        "bar_start"
    }


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
