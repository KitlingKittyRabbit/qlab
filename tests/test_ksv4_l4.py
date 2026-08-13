from __future__ import annotations

import pandas as pd
import pytest

from qlab.data.crypto.ksv4_l4 import (
    attach_historical_execution_ledger,
    apply_orderbook_venue_availability,
    build_realtime_source_plan,
    evaluate_realtime_source_comparisons,
    evaluate_serialized_route_runtime,
    frozen_execution_delay_minutes,
    summarize_realtime_equivalence_evidence,
    validate_serialized_route_runtime_summary,
    validate_realtime_equivalence_contract,
)
from qlab.data.crypto.ksv4_realtime import build_shadow_source_contract


DELTAS = {
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "8h": pd.Timedelta(hours=8),
    "12h": pd.Timedelta(hours=12),
    "1d": pd.Timedelta(days=1),
}
ENDPOINTS = [
    "fr", "fr_oi_weight", "fr_vol_weight", "oi", "futures_net_pos_v2",
    "top_pos", "ob_pair", "ob_agg",
]


def _dependencies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "endpoint": [*ENDPOINTS, "top_pos"],
            "signal_timeframe": ["1h", "1h", "1h", "12h", "1d", "1h", "1h", "1d", "12h"],
        }
    )


def _registry() -> pd.DataFrame:
    dependencies = _dependencies().drop_duplicates()
    return dependencies.assign(
        required_columns="value",
        panel_transform="raw_column",
        timestamp_kind="bar_start",
    )


def test_repaired_source_plan_uses_26_keystore_requests_and_no_basis() -> None:
    symbols = ("BTC", "ETH", "FET", *(f"C{index:02d}" for index in range(1, 18)))
    plan = build_realtime_source_plan(_dependencies(), symbols=symbols)
    keystore = plan.loc[plan["source"].eq("keystore")]
    assert len(keystore) == 26
    assert set(keystore["route"]) == {
        "coins-markets", "funding-rate/exchange-list", "futures/v2/net-position/history",
        "orderbook/ask-bids-history", "orderbook/aggregated-ask-bids-history",
    }
    assert not plan["route"].str.contains("pairs-markets").any()
    assert len(plan.loc[plan["source"].eq("binance_public") & plan["route"].eq("top-position-ratio")]) == 40
    assert len(plan.loc[plan["route"].eq("orderbook")]) == 54
    keystore_books = keystore.loc[keystore["route"].str.startswith("orderbook/")]
    assert set(keystore_books["symbol"]) == {"BTC", "ETH"}
    assert len(keystore_books) == 4
    assert frozen_execution_delay_minutes(210) == 4


def test_orderbook_plan_removes_only_proven_unavailable_route() -> None:
    symbols = ("BTC", "ETH", "FET", *(f"C{index:02d}" for index in range(1, 18)))
    plan = build_realtime_source_plan(_dependencies(), symbols=symbols)
    rows = []
    for source in ("binance_public", "okx_public", "bybit_public"):
        for symbol in symbols:
            available = not (
                symbol == "FET" and source in {"okx_public", "bybit_public"}
            )
            rows.append({
                "source": source, "symbol": symbol, "available": available,
                "evidence_code": (
                    "0" if available else "51001" if source == "okx_public" else "Closed"
                ),
            })
    filtered = apply_orderbook_venue_availability(plan, pd.DataFrame(rows))
    assert len(filtered.loc[filtered["route"].eq("orderbook")]) == 52
    fet_sources = set(
        filtered.loc[
            filtered["symbol"].eq("FET") & filtered["route"].eq("orderbook"),
            "source",
        ]
    )
    assert fet_sources == {"binance_public"}

    contract = build_shadow_source_contract(filtered)
    assert len(contract) == 118
    assert contract.groupby("source").size().to_dict() == {
        "binance_public": 58,
        "bybit_public": 17,
        "keystore": 26,
        "okx_public": 17,
    }
    assert contract["request_id"].nunique() == 118
    top = contract.loc[
        contract["request_id"].eq("binance_public|top-position-ratio|BTC|12h")
    ].iloc[0]
    assert top["request_path"] == "/futures/data/topLongShortPositionRatio"
    assert '"period":"12h"' in top["request_params_json"]
    keystore_books = contract.loc[
        contract["source"].eq("keystore")
        & contract["route"].str.startswith("orderbook/")
    ]
    assert len(keystore_books) == 4
    assert keystore_books["request_params_json"].str.contains(
        '"limit":10', regex=False
    ).all()


def test_equivalence_contract_requires_every_check_and_all_symbols() -> None:
    symbols = ("AAA", "BBB")
    coverage = pd.DataFrame(
        {
            "endpoint": ENDPOINTS,
            "schema_ok": True,
            "direction_ok": True,
            "unit_ok": True,
            "native_window_ok": True,
            "formula_test_ok": True,
            "evidence_status": "verified",
            "mapping_id": [f"mapping:{endpoint}" for endpoint in ENDPOINTS],
            "raw_evidence_sha256": "a" * 64,
            "comparison_row_count": 20,
            "covered_symbols": "AAA,BBB",
            "source_observed_ts": "2026-08-02T00:03:00Z",
        }
    )
    result = validate_realtime_equivalence_contract(
        _dependencies(), _registry(), coverage, expected_symbols=symbols
    )
    assert result["endpoint"].tolist() == sorted(ENDPOINTS)
    broken = coverage.copy()
    broken.loc[broken["endpoint"].eq("oi"), "unit_ok"] = False
    with pytest.raises(ValueError, match="preflight failed"):
        validate_realtime_equivalence_contract(
            _dependencies(), _registry(), broken, expected_symbols=symbols
        )
    incomplete = coverage.copy()
    incomplete.loc[incomplete["endpoint"].eq("fr"), "covered_symbols"] = "AAA"
    with pytest.raises(ValueError, match="symbol set"):
        validate_realtime_equivalence_contract(
            _dependencies(), _registry(), incomplete, expected_symbols=symbols
        )
    unverified = coverage.copy()
    unverified.loc[unverified["endpoint"].eq("ob_agg"), "evidence_status"] = "unverified"
    with pytest.raises(ValueError, match="evidence is not verified"):
        validate_realtime_equivalence_contract(
            _dependencies(), _registry(), unverified, expected_symbols=symbols
        )


def test_equivalence_summary_does_not_turn_coverage_into_equivalence() -> None:
    rows = []
    for endpoint in ENDPOINTS:
        for symbol in ("AAA", "BBB"):
            rows.append(
                {
                    "endpoint": endpoint,
                    "symbol": symbol,
                    "normalized_value": 1.0,
                    "comparison_ok": not (endpoint == "top_pos" and symbol == "AAA"),
                    "mapping_id": f"mapping:{endpoint}",
                    "full_depth_band_covered": True,
                }
            )
    result = summarize_realtime_equivalence_evidence(
        pd.DataFrame(rows),
        expected_symbols=["AAA", "BBB"],
        formula_test_ok=True,
        raw_evidence_sha256={endpoint: "a" * 64 for endpoint in ENDPOINTS},
        source_observed_ts="2026-08-02T00:03:00Z",
    )
    top = result.loc[result["endpoint"].eq("top_pos")].iloc[0]
    assert bool(top["schema_ok"])
    assert not bool(top["direction_ok"])
    assert top["evidence_status"] == "unverified"


def test_source_comparison_rejects_bad_btc_pair_despite_full_coverage() -> None:
    frame = pd.DataFrame(
        [
            {
                "endpoint": "top_pos", "symbol": "BTC", "normalized_value": 1.2,
                "market_data_ts": "2026-08-02T00:00:00Z",
                "mapping_id": "top_pos:binance_top_position_long_short_ratio:raw_or_delta1",
                "reference_value": 1.3,
                "reference_market_ts": "2026-08-02T00:00:00Z",
                "reference_source": "CoinGlass",
            },
            {
                "endpoint": "ob_pair", "symbol": "ADA", "normalized_value": 0.1,
                "market_data_ts": "2026-08-02T00:05:00Z",
                "mapping_id": "ob_pair:frozen_source_depth_pm1pct:imbalance",
                "reference_value": 0.1,
                "reference_market_ts": "2026-08-02T00:00:00Z",
                "reference_source": "CoinGlass", "full_depth_band_covered": True,
                "binance_market_ts": "2026-08-02T00:05:00Z",
                "required_venues": "binance_public",
            },
        ]
    )
    result = evaluate_realtime_source_comparisons(frame)
    assert not result["comparison_ok"].any()


def test_public_aggregate_orderbook_requires_all_three_venue_timestamps() -> None:
    base = {
        "endpoint": "ob_agg", "symbol": "ADA", "normalized_value": 0.1,
        "market_data_ts": "2026-08-02T00:01:00Z",
        "mapping_id": "ob_agg:frozen_source_depth_pm1pct:imbalance",
        "reference_value": 0.2,
        "reference_market_ts": "2026-08-02T00:00:00Z",
        "reference_source": "CoinGlass", "full_depth_band_covered": True,
        "binance_market_ts": "2026-08-02T00:01:00Z",
        "okx_market_ts": "2026-08-02T00:01:10Z",
        "bybit_market_ts": "2026-08-02T00:01:20Z",
        "required_venues": "binance_public,okx_public,bybit_public",
    }
    assert bool(evaluate_realtime_source_comparisons(pd.DataFrame([base])).loc[0, "comparison_ok"])
    broken = dict(base, bybit_market_ts="")
    assert not bool(
        evaluate_realtime_source_comparisons(pd.DataFrame([broken])).loc[0, "comparison_ok"]
    )


def test_btc_keystore_orderbook_path_requires_keystore_source() -> None:
    row = {
        "endpoint": "ob_pair", "symbol": "BTC", "normalized_value": 0.1,
        "market_data_ts": "2026-08-02T00:00:00Z",
        "mapping_id": "ob_pair:frozen_source_depth_pm1pct:imbalance",
        "reference_value": 0.1,
        "reference_market_ts": "2026-08-02T00:00:00Z",
        "reference_source": "same KeyStore/CoinGlass field",
        "full_depth_band_covered": True,
        "keystore_market_ts": "2026-08-02T00:00:00Z",
        "required_venues": "keystore",
    }
    assert bool(evaluate_realtime_source_comparisons(pd.DataFrame([row])).loc[0, "comparison_ok"])
    wrong_source = dict(
        row,
        required_venues="binance_public",
        binance_market_ts="2026-08-02T00:00:00Z",
    )
    assert not bool(
        evaluate_realtime_source_comparisons(pd.DataFrame([wrong_source])).loc[0, "comparison_ok"]
    )


def test_top_position_comparison_uses_coinglass_two_decimal_precision() -> None:
    row = {
        "endpoint": "top_pos", "symbol": "BTC", "normalized_value": 1.592,
        "market_data_ts": "2026-08-02T00:00:00Z",
        "mapping_id": "top_pos:binance_top_position_long_short_ratio:raw_or_delta1",
        "reference_value": 1.59,
        "reference_market_ts": "2026-08-02T00:00:00Z",
        "reference_source": "CoinGlass",
    }
    assert bool(evaluate_realtime_source_comparisons(pd.DataFrame([row])).loc[0, "comparison_ok"])
    assert not bool(
        evaluate_realtime_source_comparisons(
            pd.DataFrame([dict(row, normalized_value=1.606)])
        ).loc[0, "comparison_ok"]
    )


def test_serialized_route_runtime_is_hand_calculable_and_exact() -> None:
    receipts = pd.DataFrame(
        {
            "name": ["first", "second"],
            "request_ts": ["2026-08-02T00:00:00Z", "2026-08-02T00:00:06Z"],
            "response_ts": ["2026-08-02T00:00:01Z", "2026-08-02T00:00:07Z"],
        }
    )
    result = evaluate_serialized_route_runtime(
        receipts, expected_names=["first", "second"], timeout_seconds=7
    )
    assert result.loc[0, "elapsed_seconds"] == 7.0
    assert bool(result.loc[0, "within_timeout"])
    validated = validate_serialized_route_runtime_summary(
        result, expected_request_count=2, timeout_seconds=7
    )
    pd.testing.assert_frame_equal(validated, result)
    assert not bool(
        evaluate_serialized_route_runtime(
            receipts, expected_names=["first", "second"], timeout_seconds=6
        ).loc[0, "within_timeout"]
    )
    with pytest.raises(ValueError, match="request order"):
        evaluate_serialized_route_runtime(
            receipts, expected_names=["second", "first"], timeout_seconds=7
        )
    with pytest.raises(ValueError, match="request count"):
        validate_serialized_route_runtime_summary(
            result, expected_request_count=3, timeout_seconds=7
        )
    inconsistent = result.copy()
    inconsistent.loc[0, "elapsed_seconds"] = 6.0
    with pytest.raises(ValueError, match="inconsistent"):
        validate_serialized_route_runtime_summary(
            inconsistent, expected_request_count=2, timeout_seconds=7
        )


def test_fet_binance_only_orderbook_path_verifies_and_fails_if_required_venue_is_missing() -> None:
    plan = build_realtime_source_plan(
        pd.DataFrame({"endpoint": ["ob_agg"], "signal_timeframe": ["1h"]}),
        symbols=["FET"],
    )
    availability = pd.DataFrame(
        [
            {"source": "binance_public", "symbol": "FET", "available": True, "evidence_code": "0"},
            {"source": "okx_public", "symbol": "FET", "available": False, "evidence_code": "51001"},
            {"source": "bybit_public", "symbol": "FET", "available": False, "evidence_code": "Closed"},
        ]
    )
    filtered = apply_orderbook_venue_availability(plan, availability)
    assert set(filtered["source"]) == {"binance_public"}

    row = {
        "endpoint": "ob_agg", "symbol": "FET", "normalized_value": 0.1,
        "market_data_ts": "2026-08-02T00:01:00Z",
        "mapping_id": "ob_agg:frozen_source_depth_pm1pct:imbalance",
        "reference_value": 0.2,
        "reference_market_ts": "2026-08-02T00:00:00Z",
        "reference_source": "CoinGlass FET aggregate",
        "full_depth_band_covered": True,
        "binance_market_ts": "2026-08-02T00:01:00Z",
        "okx_market_ts": "",
        "bybit_market_ts": "",
        "required_venues": "binance_public",
    }
    evaluated = evaluate_realtime_source_comparisons(pd.DataFrame([row]))
    assert bool(evaluated.loc[0, "comparison_ok"])
    coverage = summarize_realtime_equivalence_evidence(
        evaluated,
        expected_symbols=["FET"],
        formula_test_ok=True,
        raw_evidence_sha256={"ob_agg": "a" * 64},
        source_observed_ts="2026-08-02T00:03:00Z",
    )
    ob_agg = coverage.loc[coverage["endpoint"].eq("ob_agg")].iloc[0]
    assert ob_agg["evidence_status"] == "verified"

    missing_binance = dict(row, binance_market_ts="")
    rejected = evaluate_realtime_source_comparisons(pd.DataFrame([missing_binance]))
    assert not bool(rejected.loc[0, "comparison_ok"])
    undeclared_bybit = dict(
        row,
        bybit_market_ts="2026-08-02T00:01:20Z",
        required_venues="binance_public,bybit_public",
    )
    rejected_extra = evaluate_realtime_source_comparisons(pd.DataFrame([undeclared_bybit]))
    assert not bool(rejected_extra.loc[0, "comparison_ok"])


def test_attach_historical_execution_ledger_is_hand_calculable() -> None:
    decisions = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 04:00"], utc=True)
    targets = pd.DataFrame(
        {
            "combo_id": "combo",
            "track": "all",
            "weight_scheme": "equal",
            "panel_frequency": "4h",
            "return_horizon": "4h",
            "component_features": "top_pos_ls_ratio__1h",
            "fold_idx": 0,
            "decision_ts": decisions,
            "symbol": "AAA",
            "signal_value": [1.0, 2.0],
            "bucket": 5,
            "leg": "long",
            "target_weight": 0.5,
        }
    )
    opens = pd.Series(
        [100.0, 110.0, 121.0],
        index=pd.to_datetime(
            ["2026-01-01 00:04", "2026-01-01 04:04", "2026-01-01 08:04"], utc=True
        ),
    )
    result = attach_historical_execution_ledger(
        targets, {"AAA": opens}, horizon_deltas=DELTAS, execution_delay_minutes=4
    )
    assert result["execution_ts"].tolist() == list(decisions + pd.Timedelta(minutes=4))
    assert result["executable_return"].tolist() == pytest.approx([0.10, 0.10])
    assert result["return_horizon"].tolist() == ["4h", "4h"]
    assert not any(column.endswith(("_x", "_y", "_ledger")) for column in result.columns)
