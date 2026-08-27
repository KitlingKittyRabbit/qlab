from __future__ import annotations

import pytest

from qlab.data.crypto.factor_equivalence import (
    apply_factor_registry_transform,
    aggregate_factor_equivalence_records,
    build_factor_equivalence_records,
    build_source_equivalence_identity,
)


def _item(
    *,
    symbol: str,
    registry_row: dict[str, object],
    realtime: dict[str, object],
    historical: dict[str, object],
    realtime_previous: dict[str, object] | None = None,
    historical_previous: dict[str, object] | None = None,
    realtime_identity: dict[str, object] | None = None,
    historical_identity: dict[str, object] | None = None,
    same_source_identity: bool = False,
) -> dict[str, object]:
    default_realtime_identity = build_source_equivalence_identity(
        str(registry_row["endpoint"]), symbol,
        str(registry_row["signal_timeframe"]),
        timestamp_kind=str(registry_row["timestamp_kind"]), side="realtime",
    )
    default_historical_identity = build_source_equivalence_identity(
        str(registry_row["endpoint"]), symbol,
        str(registry_row["signal_timeframe"]),
        timestamp_kind=str(registry_row["timestamp_kind"]), side="historical",
    )
    if realtime_identity:
        default_realtime_identity.update(realtime_identity)
    if historical_identity:
        default_historical_identity.update(historical_identity)
    if same_source_identity:
        default_historical_identity = dict(default_realtime_identity)
    timeframe = str(registry_row["signal_timeframe"])
    native_end = (
        "2026-08-27T00:00:00Z"
        if str(registry_row["timestamp_kind"]) == "bar_end"
        else {
            "1h": "2026-08-27T01:00:00Z",
            "12h": "2026-08-27T12:00:00Z",
            "1d": "2026-08-28T00:00:00Z",
        }[timeframe]
    )
    return {
        "collector_id": "ksv4_source_consistency_test",
        "capture_ts": "2026-08-27T00:00:00+00:00",
        "source_scope": "ksv4_1h",
        "signal_timeframe": str(registry_row["signal_timeframe"]),
        "endpoint": str(registry_row["endpoint"]),
        "symbol": symbol,
        "target_label_ts": "2026-08-27T00:00:00+00:00",
        "realtime_receipt_id": f"rt-{symbol}",
        "reference_receipt_id": f"hist-{symbol}",
        "reference_role": "initial",
        "observed_ts": "2026-08-27T00:15:00+00:00",
        "realtime_native_bar_end_ts": native_end,
        "reference_native_bar_end_ts": native_end,
        "realtime_values": realtime,
        "reference_values": historical,
        "realtime_previous_values": realtime_previous,
        "reference_previous_values": historical_previous,
        "registry_row": registry_row,
        "realtime_source_identity": default_realtime_identity,
        "historical_source_identity": default_historical_identity,
    }


def test_close_only_factor_ignores_extra_historical_ohlc_and_keeps_raw_diagnostic() -> None:
    row = {
        "feature_name": "funding_oi_weight_close__1h",
        "endpoint": "fr_oi_weight",
        "signal_timeframe": "1h",
        "required_columns": "close",
        "panel_transform": "raw_column",
        "cross_section_standardization": "none",
        "timestamp_kind": "bar_start",
    }
    records = build_factor_equivalence_records(
        [
            _item(
                symbol="BTC",
                registry_row=row,
                realtime={"close": 0.1},
                historical={"open": 0.08, "high": 0.12, "low": 0.07, "close": 0.1},
            ),
            _item(
                symbol="ETH",
                registry_row=row,
                realtime={"close": 0.2},
                historical={"open": 0.18, "high": 0.22, "low": 0.17, "close": 0.2},
            ),
        ]
    )
    assert [record["status"] for record in records] == ["exact_match", "exact_match"]
    assert all(not record["raw_structure_diagnostic"]["raw_structure_equal"] for record in records)
    assert all(record["final_strategy_input_equal"] for record in records)


def test_weighted_funding_close_only_contract_covers_20_coins_three_timeframes_two_endpoints() -> None:
    items = []
    symbols = ["BTC", "ETH", *(f"C{index:02d}" for index in range(1, 19))]
    for endpoint in ("fr_oi_weight", "fr_vol_weight"):
        for timeframe in ("1h", "12h", "1d"):
            row = {
                "feature_name": f"{endpoint}_close__{timeframe}",
                "endpoint": endpoint,
                "signal_timeframe": timeframe,
                "required_columns": "close",
                "panel_transform": "raw_column",
                "cross_section_standardization": "none",
                "timestamp_kind": "bar_start",
            }
            for index, symbol in enumerate(symbols, start=1):
                close = index / 1000.0
                items.append(_item(
                    symbol=symbol, registry_row=row,
                    realtime={"close": close},
                    historical={
                        "open": close - 0.01, "high": close + 0.02,
                        "low": close - 0.02, "close": close,
                    },
                ))
    records = build_factor_equivalence_records(items, expected_symbols=symbols)
    assert len(records) == 120
    assert all(record["status"] == "exact_match" for record in records)
    assert all(record["factor_direction_reversed"] is False for record in records)
    assert all(
        not record["raw_structure_diagnostic"]["raw_structure_equal"]
        and record["final_strategy_input_equal"]
        for record in records
    )


def test_two_coin_two_time_delta_and_rank_are_hand_calculable() -> None:
    row = {
        "feature_name": "oi_close_delta1__1h",
        "endpoint": "oi",
        "signal_timeframe": "1h",
        "required_columns": "oi_close",
        "panel_transform": "delta1_raw_column",
        "cross_section_standardization": "rank_to_minus1_1",
        "timestamp_kind": "bar_start",
    }
    items = [
        _item(
            symbol="BTC", registry_row=row,
            realtime={"oi_close": 15.0}, historical={"oi_close": 15.0},
            realtime_previous={"oi_close": 10.0}, historical_previous={"oi_close": 10.0},
            realtime_identity={"exchange_scope": "same"},
            historical_identity={"exchange_scope": "same"},
        ),
        _item(
            symbol="ETH", registry_row=row,
            realtime={"oi_close": 13.0}, historical={"oi_close": 13.0},
            realtime_previous={"oi_close": 10.0}, historical_previous={"oi_close": 10.0},
            realtime_identity={"exchange_scope": "same"},
            historical_identity={"exchange_scope": "same"},
        ),
    ]
    records = build_factor_equivalence_records(items)
    assert [record["realtime_factor_value"] for record in records] == [5.0, 3.0]
    assert [record["realtime_standardized_value"] for record in records] == [1.0, -1.0]
    assert all(record["status"] == "exact_match" for record in records)


def test_two_coin_two_native_times_cover_close_delta_imbalance_and_final_input() -> None:
    """Hand calculation: 10->15/10->13 and (120,80)/(80,120) give ranks +1/-1."""
    close_row = {
        "feature_name": "weighted_close__1h",
        "endpoint": "fr_oi_weight",
        "signal_timeframe": "1h",
        "required_columns": "close",
        "panel_transform": "raw_column",
        "cross_section_standardization": "none",
        "timestamp_kind": "bar_start",
    }
    delta_row = {
        "feature_name": "oi_delta1__1h",
        "endpoint": "oi",
        "signal_timeframe": "1h",
        "required_columns": "oi_close",
        "panel_transform": "delta1_raw_column",
        "cross_section_standardization": "rank_to_minus1_1",
        "timestamp_kind": "bar_start",
    }
    imbalance_row = {
        "feature_name": "ob_agg_imbalance__1h",
        "endpoint": "ob_agg",
        "signal_timeframe": "1h",
        "required_columns": "aggregated_bids_usd,aggregated_asks_usd",
        "panel_transform": "buy_sell_imbalance",
        "cross_section_standardization": "rank_to_minus1_1",
        "timestamp_kind": "bar_end",
    }
    items = [
        _item(
            symbol="BTC", registry_row=close_row,
            realtime={"close": 0.10}, historical={"close": 0.10, "open": 0.09},
        ),
        _item(
            symbol="ETH", registry_row=close_row,
            realtime={"close": 0.20}, historical={"close": 0.20, "open": 0.19},
        ),
            _item(
                symbol="BTC", registry_row=delta_row,
                realtime={"oi_close": 15.0}, historical={"oi_close": 15.0},
                realtime_previous={"oi_close": 10.0}, historical_previous={"oi_close": 10.0},
                realtime_identity={"exchange_scope": "same"},
                historical_identity={"exchange_scope": "same"},
            ),
            _item(
                symbol="ETH", registry_row=delta_row,
                realtime={"oi_close": 13.0}, historical={"oi_close": 13.0},
                realtime_previous={"oi_close": 10.0}, historical_previous={"oi_close": 10.0},
                realtime_identity={"exchange_scope": "same"},
                historical_identity={"exchange_scope": "same"},
            ),
            _item(
                symbol="BTC", registry_row=imbalance_row,
                realtime={"aggregated_bids_usd": 120.0, "aggregated_asks_usd": 80.0},
                historical={"aggregated_bids_usd": 120.0, "aggregated_asks_usd": 80.0},
                same_source_identity=True,
            ),
        _item(
                symbol="ETH", registry_row=imbalance_row,
                realtime={"aggregated_bids_usd": 80.0, "aggregated_asks_usd": 120.0},
                historical={"aggregated_bids_usd": 80.0, "aggregated_asks_usd": 120.0},
                same_source_identity=True,
        ),
    ]
    records = build_factor_equivalence_records(items)
    assert all(record["status"] == "exact_match" for record in records)
    by_feature_symbol = {
        (record["feature_name"], record["symbol"]): record for record in records
    }
    assert by_feature_symbol[("oi_delta1__1h", "BTC")]["realtime_factor_value"] == 5.0
    assert by_feature_symbol[("oi_delta1__1h", "ETH")]["realtime_factor_value"] == 3.0
    assert by_feature_symbol[("ob_agg_imbalance__1h", "BTC")]["realtime_factor_value"] == 0.2
    assert by_feature_symbol[("ob_agg_imbalance__1h", "ETH")]["realtime_factor_value"] == -0.2
    assert by_feature_symbol[("ob_agg_imbalance__1h", "BTC")]["realtime_standardized_value"] == 1.0
    assert by_feature_symbol[("ob_agg_imbalance__1h", "ETH")]["realtime_standardized_value"] == -1.0


def test_delta_or_rank_difference_is_not_hidden_by_equal_raw_level() -> None:
    row = {
        "feature_name": "net_delta1__1h",
        "endpoint": "futures_net_pos_v2",
        "signal_timeframe": "1h",
        "required_columns": "net_position_change_cum",
        "panel_transform": "delta1_raw_column",
        "cross_section_standardization": "rank_to_minus1_1",
        "timestamp_kind": "bar_start",
    }
    record = build_factor_equivalence_records(
        [
            _item(
                symbol="BTC", registry_row=row,
                realtime={"net_position_change_cum": 15.0},
                historical={"net_position_change_cum": 15.0},
                realtime_previous={"net_position_change_cum": 10.0},
                historical_previous={"net_position_change_cum": 12.0},
            )
        ]
    )[0]
    assert record["factor_value_equal"] is False
    assert record["status"] == "value_mismatch_decision_equivalent"


def test_raw_factor_difference_can_preserve_the_cross_section_rank() -> None:
    row = {
        "feature_name": "top_pos_level__1h",
        "endpoint": "top_pos",
        "signal_timeframe": "1h",
        "required_columns": "top_pos_ls_ratio",
        "panel_transform": "raw_column",
        "cross_section_standardization": "rank_to_minus1_1",
        "timestamp_kind": "bar_end",
    }
    records = build_factor_equivalence_records(
        [
            _item(
                symbol="BTC", registry_row=row,
                realtime={"top_pos_ls_ratio": 2.0},
                historical={"top_pos_ls_ratio": 3.0},
            ),
            _item(
                symbol="ETH", registry_row=row,
                realtime={"top_pos_ls_ratio": 1.0},
                historical={"top_pos_ls_ratio": 2.0},
            ),
        ]
    )
    assert all(record["factor_value_equal"] is False for record in records)
    assert all(record["cross_section_equal"] is True for record in records)
    assert all(record["final_strategy_input_equal"] is True for record in records)
    assert all(
        record["status"] == "value_mismatch_decision_equivalent"
        for record in records
    )


def test_top_position_native_timestamp_mapping_is_explicit_and_comparable() -> None:
    realtime = build_source_equivalence_identity(
        "top_pos", "BTC", "1h", timestamp_kind="bar_end", side="realtime"
    )
    historical = build_source_equivalence_identity(
        "top_pos", "BTC", "1h", timestamp_kind="bar_end", side="historical"
    )
    assert realtime["source_native_timestamp_kind"] == "bar_start"
    assert historical["source_native_timestamp_kind"] == "bar_end"
    assert realtime["strategy_timestamp_kind"] == historical["strategy_timestamp_kind"] == "bar_end"
    assert realtime["field_precision"] == historical["field_precision"]
    assert realtime["rounding"] == historical["rounding"] == "none_before_comparison"
    row = {
        "feature_name": "top_pos_level__1h",
        "endpoint": "top_pos",
        "signal_timeframe": "1h",
        "required_columns": "top_pos_ls_ratio",
        "panel_transform": "raw_column",
        "cross_section_standardization": "none",
        "timestamp_kind": "bar_end",
    }
    record = build_factor_equivalence_records(
        [_item(symbol="BTC", registry_row=row, realtime={"top_pos_ls_ratio": 1.2}, historical={"top_pos_ls_ratio": 1.2})]
    )[0]
    assert record["status"] == "exact_match"


def test_scope_and_timestamp_identity_fail_closed() -> None:
    row = {
        "feature_name": "oi_delta1__12h",
        "endpoint": "oi",
        "signal_timeframe": "12h",
        "required_columns": "oi_close",
        "panel_transform": "delta1_raw_column",
        "cross_section_standardization": "rank_to_minus1_1",
        "timestamp_kind": "bar_start",
    }
    item = _item(
        symbol="BTC", registry_row=row,
        realtime={"oi_close": 15.0}, historical={"oi_close": 15.0},
        realtime_previous={"oi_close": 10.0}, historical_previous={"oi_close": 10.0},
        realtime_identity={"exchange_scope": "Binance,OKX,Bybit"},
        historical_identity={"exchange_scope": "historical_unfiltered"},
    )
    assert build_factor_equivalence_records([item])[0]["status"] == "scope_not_comparable"

    item["realtime_source_identity"] = {"scope": "same"}
    item["historical_source_identity"] = {"scope": "same"}
    item["reference_native_bar_end_ts"] = "2026-08-27T00:00:00Z"
    assert build_factor_equivalence_records([item])[0]["status"] == "native_identity_mismatch"


def test_native_timestamp_contract_is_checked_for_start_and_end_labels() -> None:
    for kind, native_end in (
        ("bar_start", "2026-08-27T01:00:00Z"),
        ("bar_end", "2026-08-27T00:00:00Z"),
    ):
        row = {
            "feature_name": f"funding__{kind}",
            "endpoint": "fr",
            "signal_timeframe": "1h",
            "required_columns": "fr_close",
            "panel_transform": "raw_column",
            "cross_section_standardization": "none",
            "timestamp_kind": kind,
        }
        item = _item(
            symbol="BTC", registry_row=row,
            realtime={"fr_close": 0.1}, historical={"fr_close": 0.1},
        )
        item["realtime_native_bar_end_ts"] = native_end
        item["reference_native_bar_end_ts"] = native_end
        assert build_factor_equivalence_records([item])[0]["status"] == "exact_match"
        item["reference_native_bar_end_ts"] = "2026-08-27T02:00:00Z"
        assert build_factor_equivalence_records([item])[0]["status"] == "native_identity_mismatch"


def test_net_position_native_interval_is_part_of_identity() -> None:
    one_hour = build_source_equivalence_identity(
        "futures_net_pos_v2", "BTC", "1h",
        timestamp_kind="bar_start", side="realtime",
    )
    one_day = build_source_equivalence_identity(
        "futures_net_pos_v2", "BTC", "1d",
        timestamp_kind="bar_start", side="realtime",
    )
    assert one_hour["native_interval"] == "1h"
    assert one_day["native_interval"] == "1d"
    assert one_hour != one_day


def test_one_hour_net_position_cannot_be_projected_as_one_day() -> None:
    row = {
        "feature_name": "net_delta1__1d",
        "endpoint": "futures_net_pos_v2",
        "signal_timeframe": "1d",
        "required_columns": "net_position_change_cum",
        "panel_transform": "delta1_raw_column",
        "cross_section_standardization": "rank_to_minus1_1",
        "timestamp_kind": "bar_start",
    }
    item = _item(
        symbol="BTC", registry_row=row,
        realtime={"net_position_change_cum": 15.0},
        historical={"net_position_change_cum": 15.0},
        realtime_previous={"net_position_change_cum": 10.0},
        historical_previous={"net_position_change_cum": 10.0},
        realtime_identity={"native_interval": "1h"},
        historical_identity={"native_interval": "1h"},
    )
    record = build_factor_equivalence_records([item])[0]
    assert record["status"] == "native_identity_mismatch"
    assert record["final_strategy_input_equal"] is False


def test_final_rank_difference_is_decision_material() -> None:
    row = {
        "feature_name": "top_pos__1h",
        "endpoint": "top_pos",
        "signal_timeframe": "1h",
        "required_columns": "top_pos_ls_ratio",
        "panel_transform": "raw_column",
        "cross_section_standardization": "rank_to_minus1_1",
        "timestamp_kind": "bar_end",
    }
    records = build_factor_equivalence_records(
        [
            _item(
                symbol="BTC", registry_row=row,
                realtime={"top_pos_ls_ratio": 2.0},
                historical={"top_pos_ls_ratio": 1.0},
            ),
            _item(
                symbol="ETH", registry_row=row,
                realtime={"top_pos_ls_ratio": 1.0},
                historical={"top_pos_ls_ratio": 2.0},
            ),
        ]
    )
    assert [record["status"] for record in records] == [
        "decision_material_mismatch",
        "decision_material_mismatch",
    ]
    assert all(not record["final_strategy_input_equal"] for record in records)


def test_delta_requires_a_real_previous_native_observation() -> None:
    row = {
        "feature_name": "net_delta1__1d",
        "endpoint": "futures_net_pos_v2",
        "signal_timeframe": "1d",
        "required_columns": "net_position_change_cum",
        "panel_transform": "delta1_raw_column",
        "cross_section_standardization": "rank_to_minus1_1",
        "timestamp_kind": "bar_start",
    }
    record = build_factor_equivalence_records(
        [_item(
            symbol="BTC", registry_row=row,
            realtime={"net_position_change_cum": 2.0},
            historical={"net_position_change_cum": 2.0},
        )]
    )[0]
    assert record["status"] == "missing_prior_observation"
    assert record["final_strategy_input_equal"] is False


def test_orderbook_contract_is_depth_not_pseudo_usd_and_opposite_direction_changes_rank() -> None:
    identity = build_source_equivalence_identity(
        "ob_pair", "BTC", "1h", timestamp_kind="bar_end", side="realtime"
    )
    assert identity["unit"] == "unitless_imbalance"
    assert identity["raw_input_unit"] == "USD_depth"
    assert identity["depth_formula"].startswith("sum(price*quantity")
    assert identity["snapshot_time_semantics"] == "exact_target_label_ts"
    row = {
        "feature_name": "ob_pair_imbalance__1h",
        "endpoint": "ob_pair",
        "signal_timeframe": "1h",
        "required_columns": "bids_usd,asks_usd",
        "panel_transform": "buy_sell_imbalance",
        "cross_section_standardization": "rank_to_minus1_1",
        "timestamp_kind": "bar_end",
    }
    records = build_factor_equivalence_records(
        [
            _item(
                symbol="BTC", registry_row=row,
                realtime={"bids_usd": 125.0, "asks_usd": 75.0},
                historical={"bids_usd": 125.0, "asks_usd": 75.0},
                same_source_identity=True,
            ),
            _item(
                symbol="ETH", registry_row=row,
                realtime={"bids_usd": 75.0, "asks_usd": 125.0},
                historical={"bids_usd": 75.0, "asks_usd": 125.0},
                same_source_identity=True,
            ),
        ]
    )
    assert [record["realtime_factor_value"] for record in records] == [0.25, -0.25]
    assert all(record["status"] == "exact_match" for record in records)
    with pytest.raises(KeyError):
        apply_factor_registry_transform(row, {"imbalance": 0.25})
    opposite = build_factor_equivalence_records(
        [
            _item(
                symbol="BTC", registry_row=row,
                realtime={"bids_usd": 125.0, "asks_usd": 75.0},
                historical={"bids_usd": 75.0, "asks_usd": 125.0},
                same_source_identity=True,
            ),
            _item(
                symbol="ETH", registry_row=row,
                realtime={"bids_usd": 75.0, "asks_usd": 125.0},
                historical={"bids_usd": 125.0, "asks_usd": 75.0},
                same_source_identity=True,
            ),
        ]
    )
    assert all(record["factor_direction_reversed"] is True for record in opposite)
    assert all(record["status"] == "decision_material_mismatch" for record in opposite)


def test_dimensionless_orderbook_value_cannot_claim_usd_depth_identity() -> None:
    row = {
        "feature_name": "ob_pair_imbalance__1h",
        "endpoint": "ob_pair",
        "signal_timeframe": "1h",
        "required_columns": "bids_usd,asks_usd",
        "panel_transform": "buy_sell_imbalance",
        "cross_section_standardization": "none",
        "timestamp_kind": "bar_end",
    }
    item = _item(
        symbol="BTC", registry_row=row,
        realtime={"bids_usd": 1.25, "asks_usd": 0.75},
        historical={"bids_usd": 1.25, "asks_usd": 0.75},
        realtime_identity={"raw_input_unit": "unitless_imbalance"},
        historical_identity={"raw_input_unit": "unitless_imbalance"},
    )
    record = build_factor_equivalence_records([item])[0]
    assert record["status"] == "native_identity_mismatch"
    assert record["final_strategy_input_equal"] is False


def test_expected_cross_section_is_fail_closed_before_rank_standardization() -> None:
    row = {
        "feature_name": "top_pos__1h",
        "endpoint": "top_pos",
        "signal_timeframe": "1h",
        "required_columns": "top_pos_ls_ratio",
        "panel_transform": "raw_column",
        "cross_section_standardization": "rank_to_minus1_1",
        "timestamp_kind": "bar_end",
    }
    record = build_factor_equivalence_records(
        [_item(
            symbol="BTC", registry_row=row,
            realtime={"top_pos_ls_ratio": 1.2},
            historical={"top_pos_ls_ratio": 1.2},
        )],
        expected_symbols=("BTC", "ETH"),
    )[0]
    assert record["status"] == "cross_section_incomplete"
    assert record["cross_section_complete"] is False
    assert record["realtime_standardized_value"] is None


def test_event_aggregation_is_the_qlab_only_status_reduction() -> None:
    row = {
        "feature_name": "funding__1h",
        "endpoint": "fr",
        "signal_timeframe": "1h",
        "required_columns": "fr_close",
        "panel_transform": "raw_column",
        "cross_section_standardization": "none",
        "timestamp_kind": "bar_start",
    }
    records = build_factor_equivalence_records(
        [_item(
            symbol="BTC", registry_row=row,
            realtime={"fr_close": 0.1}, historical={"fr_close": 0.1},
        )]
    )
    records[0]["realtime_receipt_id"] = "same-receipt"
    records[0]["reference_role"] = "initial"
    aggregate = aggregate_factor_equivalence_records(records)
    assert len(aggregate) == 1
    assert aggregate[0]["status"] == "exact_match"
    assert aggregate[0]["final_strategy_input_equal"] is True
    assert aggregate[0]["factor_equivalence_count"] == 1


def test_registry_source_identity_version_rejects_stale_contract() -> None:
    row = {
        "feature_name": "funding__1h",
        "endpoint": "fr",
        "signal_timeframe": "1h",
        "required_columns": "fr_close",
        "panel_transform": "raw_column",
        "cross_section_standardization": "none",
        "timestamp_kind": "bar_start",
        "source_identity_contract_version": "old_source_contract",
    }
    record = build_factor_equivalence_records(
        [_item(
            symbol="BTC", registry_row=row,
            realtime={"fr_close": 0.1}, historical={"fr_close": 0.1},
        )]
    )[0]
    assert record["status"] == "native_identity_mismatch"
    assert record["final_strategy_input_equal"] is False


def test_observed_identity_version_and_symbol_are_bound_to_the_comparison() -> None:
    row = {
        "feature_name": "funding__1h",
        "endpoint": "fr",
        "signal_timeframe": "1h",
        "required_columns": "fr_close",
        "panel_transform": "raw_column",
        "cross_section_standardization": "none",
        "timestamp_kind": "bar_start",
    }
    stale = _item(
        symbol="BTC", registry_row=row,
        realtime={"fr_close": 0.1}, historical={"fr_close": 0.1},
    )
    stale["realtime_source_identity"]["identity_version"] = "old_source_contract"
    record = build_factor_equivalence_records([stale])[0]
    assert record["status"] == "native_identity_mismatch"
    assert record["final_strategy_input_equal"] is False

    wrong_symbol = _item(
        symbol="BTC", registry_row=row,
        realtime={"fr_close": 0.1}, historical={"fr_close": 0.1},
    )
    wrong_symbol["realtime_source_identity"]["symbol"] = "ETH"
    record = build_factor_equivalence_records([wrong_symbol])[0]
    assert record["status"] == "native_identity_mismatch"
    assert record["final_strategy_input_equal"] is False
