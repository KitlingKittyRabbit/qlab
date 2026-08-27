from __future__ import annotations

import pytest
import pandas as pd

from qlab.data.crypto import keystore_coinglass_panel
from qlab.data.crypto.factor_equivalence import (
    apply_factor_registry_transform,
    aggregate_factor_equivalence_records,
    build_factor_equivalence_records,
    build_source_equivalence_identity,
    source_semantic_contract_from_request,
)


def _identity(
    endpoint: str,
    symbol: str,
    timeframe: str,
    *,
    timestamp_kind: str,
    side: str,
    exchange: str = "same",
) -> dict[str, object]:
    return build_source_equivalence_identity(
        endpoint,
        symbol,
        timeframe,
        timestamp_kind=timestamp_kind,
        side=side,
        request_contract={
            "source": "synthetic",
            "route": f"{side}/{endpoint}",
            "request_path": f"/synthetic/{side}/{endpoint}",
            "request_params": {
                "symbol": symbol,
                "interval": timeframe,
                "exchange": exchange,
            },
            "source_contract_version": "test-contract",
        },
        receipt_lineage={
            "receipt_id": f"{side}-{symbol}-receipt",
            "payload_sha256": f"{side}-{symbol}-payload",
        },
        semantic_contract={
            "metric": endpoint,
            "unit": "test",
            "native_interval": timeframe,
            "scope_status": "declared",
            "scope_key": exchange,
        },
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
    target_label: str | pd.Timestamp = "2026-08-27T00:00:00+00:00",
) -> dict[str, object]:
    timeframe = str(registry_row["signal_timeframe"])
    target = pd.Timestamp(target_label)
    target = target.tz_localize("UTC") if target.tz is None else target.tz_convert("UTC")
    endpoint = str(registry_row["endpoint"])
    def request_contract(side: str) -> dict[str, object]:
        return {
            "source": "synthetic",
            "route": f"{side}/{endpoint}",
            "request_path": f"/synthetic/{endpoint}",
            "request_params": {
                "symbol": symbol,
                "interval": timeframe,
                "exchange": "same",
            },
            "source_contract_version": "test-contract",
        }

    def receipt_lineage(side: str) -> dict[str, object]:
        return {
            "receipt_id": f"{side}-{symbol}-receipt",
            "payload_sha256": f"{side}-{symbol}-payload",
        }

    def configured_request(side: str, override: dict[str, object] | None) -> dict[str, object]:
        request = request_contract(side)
        params = dict(request["request_params"])
        if override and "exchange_scope" in override:
            params["exchange"] = str(override["exchange_scope"])
        if override and "native_interval" in override:
            params["interval"] = str(override["native_interval"])
        request["request_params"] = params
        return request

    realtime_override = realtime_identity or {}
    historical_override = historical_identity or {}
    realtime_request = configured_request("realtime", realtime_override)
    historical_request = configured_request("historical", historical_override)
    realtime_semantic = source_semantic_contract_from_request(
        realtime_request,
        registry_row,
        symbol=symbol,
        signal_timeframe=timeframe,
        side="realtime",
    )
    historical_semantic = source_semantic_contract_from_request(
        historical_request,
        registry_row,
        symbol=symbol,
        signal_timeframe=timeframe,
        side="historical",
    )
    default_realtime_identity = build_source_equivalence_identity(
        endpoint,
        symbol,
        timeframe,
        timestamp_kind=str(registry_row["timestamp_kind"]),
        side="realtime",
        request_contract=realtime_request,
        receipt_lineage=receipt_lineage("realtime"),
        semantic_contract=realtime_semantic,
    )
    default_historical_identity = build_source_equivalence_identity(
        endpoint,
        symbol,
        timeframe,
        timestamp_kind=str(registry_row["timestamp_kind"]),
        side="historical",
        request_contract=historical_request,
        receipt_lineage=receipt_lineage("historical"),
        semantic_contract=historical_semantic,
    )
    def apply_identity_override(
        identity: dict[str, object], override: dict[str, object]
    ) -> None:
        semantic_override = override.get("semantic_contract")
        if isinstance(semantic_override, dict):
            identity["semantic_contract"] = dict(semantic_override)
        for key in ("metric", "unit", "raw_input_unit"):
            if key in override:
                identity[key] = override[key]
                identity["semantic_contract"][key] = override[key]
        for key in ("identity_version", "symbol", "source_side"):
            if key in override:
                identity[key] = override[key]

    if realtime_identity:
        apply_identity_override(default_realtime_identity, realtime_identity)
    if historical_identity:
        apply_identity_override(default_historical_identity, historical_identity)
    if same_source_identity:
        default_historical_identity = dict(default_realtime_identity)
        default_historical_identity["source_side"] = "historical"
        default_historical_identity["request_contract"] = historical_request
        default_historical_identity["receipt_lineage"] = receipt_lineage("historical")
    timeframe = str(registry_row["signal_timeframe"])
    native_end = (
        target
        if str(registry_row["timestamp_kind"]) == "bar_end"
        else target + pd.Timedelta(timeframe)
    ).isoformat()
    return {
        "collector_id": "ksv4_source_consistency_test",
        "capture_ts": "2026-08-27T00:00:00+00:00",
        "source_scope": "ksv4_1h",
        "signal_timeframe": str(registry_row["signal_timeframe"]),
        "endpoint": str(registry_row["endpoint"]),
        "symbol": symbol,
        "target_label_ts": target.isoformat(),
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


def test_factor_equivalence_reuses_formal_panel_transform_and_ranking_with_ties() -> None:
    """Both paths agree for a complete two-symbol panel at two native times."""
    targets = [
        pd.Timestamp("2026-08-27T00:00:00Z"),
        pd.Timestamp("2026-08-27T01:00:00Z"),
    ]
    cases = [
        (
            {
                "feature_name": "close__1h",
                "endpoint": "fr_oi_weight",
                "signal_timeframe": "1h",
                "required_columns": "close",
                "panel_transform": "raw_column",
                "cross_section_standardization": "rank_to_minus1_1",
                "timestamp_kind": "bar_start",
            },
            {
                targets[0]: {"BTC": {"close": 2.0}, "ETH": {"close": 2.0}},
                targets[1]: {"BTC": {"close": 3.0}, "ETH": {"close": 1.0}},
            },
            None,
        ),
        (
            {
                "feature_name": "delta__1h",
                "endpoint": "oi",
                "signal_timeframe": "1h",
                "required_columns": "oi_close",
                "panel_transform": "delta1_raw_column",
                "cross_section_standardization": "rank_to_minus1_1",
                "timestamp_kind": "bar_start",
            },
            {
                targets[0]: {"BTC": {"oi_close": 15.0}, "ETH": {"oi_close": 13.0}},
                targets[1]: {"BTC": {"oi_close": 14.0}, "ETH": {"oi_close": 17.0}},
            },
            {
                targets[0]: {"BTC": {"oi_close": 10.0}, "ETH": {"oi_close": 10.0}},
                targets[1]: {"BTC": {"oi_close": 15.0}, "ETH": {"oi_close": 13.0}},
            },
        ),
        (
            {
                "feature_name": "imbalance__1h",
                "endpoint": "ob_agg",
                "signal_timeframe": "1h",
                "required_columns": "aggregated_bids_usd,aggregated_asks_usd",
                "panel_transform": "buy_sell_imbalance",
                "cross_section_standardization": "rank_to_minus1_1",
                "timestamp_kind": "bar_end",
            },
            {
                targets[0]: {
                    "BTC": {"aggregated_bids_usd": 120.0, "aggregated_asks_usd": 80.0},
                    "ETH": {"aggregated_bids_usd": 80.0, "aggregated_asks_usd": 120.0},
                },
                targets[1]: {
                    "BTC": {"aggregated_bids_usd": 100.0, "aggregated_asks_usd": 100.0},
                    "ETH": {"aggregated_bids_usd": 50.0, "aggregated_asks_usd": 150.0},
                },
            },
            None,
        ),
    ]
    for row, current_by_target, previous_by_target in cases:
        panel_values: dict[tuple[pd.Timestamp, str], float] = {}
        items = []
        for target in targets:
            for symbol, current in current_by_target[target].items():
                previous = (
                    None
                    if previous_by_target is None
                    else previous_by_target[target][symbol]
                )
                frame_rows = [current] if previous is None else [previous, current]
                frame_index = [target] if previous is None else [
                    target - pd.Timedelta(hours=1), target
                ]
                frame = pd.DataFrame(
                    frame_rows,
                    index=pd.DatetimeIndex(frame_index, name="ts"),
                )
                transformed = keystore_coinglass_panel.extract_feature_series(
                    pd.Series(row), frame
                )
                panel_values[(target, symbol)] = float(transformed.loc[target])
                items.append(
                    _item(
                        symbol=symbol,
                        registry_row=row,
                        realtime=current,
                        historical=current,
                        realtime_previous=previous,
                        historical_previous=previous,
                        same_source_identity=True,
                        target_label=target,
                    )
                )
        panel_index = pd.MultiIndex.from_tuples(
            [
                (target, symbol)
                for target in targets
                for symbol in current_by_target[target]
            ],
            names=["decision_ts", "symbol"],
        )
        panel = pd.DataFrame(
            {
                str(row["feature_name"]): [
                    panel_values[(target, symbol)]
                    for target, symbol in panel_index
                ]
            },
            index=panel_index,
        )
        panel_standardized = keystore_coinglass_panel.standardize_panel_cross_section(
            panel,
            pd.DataFrame([row]),
        )[str(row["feature_name"])].to_dict()
        records = build_factor_equivalence_records(items)
        by_key = {
            (pd.Timestamp(record["target_label_ts"]), record["symbol"]): record
            for record in records
        }
        for target in targets:
            for symbol in current_by_target[target]:
                record = by_key[(target, symbol)]
                assert record["realtime_factor_value"] == pytest.approx(
                    panel_values[(target, symbol)]
                )
                assert record["realtime_standardized_value"] == pytest.approx(
                    panel_standardized[(target, symbol)]
                )
                assert record["historical_factor_value"] == pytest.approx(
                    panel_values[(target, symbol)]
                )
                assert record["status"] == "exact_match"


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
    realtime = _identity(
        "top_pos", "BTC", "1h", timestamp_kind="bar_end", side="realtime"
    )
    historical = _identity(
        "top_pos", "BTC", "1h", timestamp_kind="bar_end", side="historical"
    )
    assert realtime["semantic_contract"] == historical["semantic_contract"]
    assert realtime["request_contract"]["request_path"] != historical["request_contract"]["request_path"]
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


def test_actual_request_scope_mismatch_cannot_claim_source_identity() -> None:
    row = {
        "feature_name": "funding__1h",
        "endpoint": "fr",
        "signal_timeframe": "1h",
        "required_columns": "fr_close",
        "panel_transform": "raw_column",
        "cross_section_standardization": "none",
        "timestamp_kind": "bar_start",
        "source_identity_contract_version": "ksv4_source_semantics_v1",
        "source_identity_contract": (
            '{"realtime":{"metric":"funding_rate","unit":"rate"},'
            '"historical":{"metric":"funding_rate","unit":"rate"}}'
        ),
    }
    realtime_request = {
        "source": "keystore",
        "route": "funding-rate",
        "request_path": "/funding-rate",
        "request_params": {"exchange": "Binance", "interval": "1h", "symbol": "BTCUSDT"},
        "source_contract_version": "test-contract",
    }
    historical_request = {**realtime_request, "request_params": {
        "exchange": "OKX", "interval": "1h", "symbol": "BTCUSDT"
    }}
    realtime_semantic = source_semantic_contract_from_request(
        realtime_request, row, symbol="BTC", signal_timeframe="1h", side="realtime"
    )
    historical_semantic = source_semantic_contract_from_request(
        historical_request, row, symbol="BTC", signal_timeframe="1h", side="historical"
    )
    item = _item(
        symbol="BTC", registry_row=row,
        realtime={"fr_close": 0.1}, historical={"fr_close": 0.1},
        realtime_identity={"semantic_contract": realtime_semantic},
        historical_identity={"semantic_contract": historical_semantic},
    )
    item["realtime_source_identity"] = build_source_equivalence_identity(
        "fr", "BTC", "1h", timestamp_kind="bar_start", side="realtime",
        request_contract=realtime_request,
        receipt_lineage={"receipt_id": "rt", "payload_sha256": "rt-sha"},
        semantic_contract=realtime_semantic,
    )
    item["historical_source_identity"] = build_source_equivalence_identity(
        "fr", "BTC", "1h", timestamp_kind="bar_start", side="historical",
        request_contract=historical_request,
        receipt_lineage={"receipt_id": "hist", "payload_sha256": "hist-sha"},
        semantic_contract=historical_semantic,
    )
    assert build_factor_equivalence_records([item])[0]["status"] == "scope_not_comparable"


def test_unverified_request_scope_fails_closed_even_when_endpoint_matches() -> None:
    row = {
        "feature_name": "funding__1h",
        "endpoint": "fr",
        "signal_timeframe": "1h",
        "required_columns": "fr_close",
        "panel_transform": "raw_column",
        "cross_section_standardization": "none",
        "timestamp_kind": "bar_start",
    }
    item = _item(
        symbol="BTC",
        registry_row=row,
        realtime={"fr_close": 0.1},
        historical={"fr_close": 0.1},
        realtime_identity={"exchange_scope": ""},
        historical_identity={"exchange_scope": ""},
    )
    assert item["realtime_source_identity"]["semantic_contract"]["scope_status"] != "declared"
    assert build_factor_equivalence_records([item])[0]["status"] == "scope_not_comparable"


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
    one_hour = _identity(
        "futures_net_pos_v2", "BTC", "1h",
        timestamp_kind="bar_start", side="realtime",
    )
    one_day = _identity(
        "futures_net_pos_v2", "BTC", "1d",
        timestamp_kind="bar_start", side="realtime",
    )
    assert one_hour["semantic_contract"]["native_interval"] == "1h"
    assert one_day["semantic_contract"]["native_interval"] == "1d"
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
        "source_identity_contract_version": "ksv4_source_semantics_v1",
        "source_identity_contract": (
            '{"realtime":{"native_interval":"1d"},'
            '"historical":{"native_interval":"1d"}}'
        ),
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


def test_native_interval_bound_endpoint_cannot_relabel_an_hour_as_a_day() -> None:
    row = {
        "feature_name": "funding__1d",
        "endpoint": "fr",
        "signal_timeframe": "1d",
        "required_columns": "fr_close",
        "panel_transform": "raw_column",
        "cross_section_standardization": "none",
        "timestamp_kind": "bar_start",
    }
    item = _item(
        symbol="BTC",
        registry_row=row,
        realtime={"fr_close": 0.1},
        historical={"fr_close": 0.1},
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
    identity = _identity(
        "ob_pair", "BTC", "1h", timestamp_kind="bar_end", side="realtime"
    )
    assert identity["semantic_contract"]["unit"] == "test"
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
        "source_identity_contract_version": "ksv4_source_semantics_v1",
        "source_identity_contract": '{"realtime":{"raw_input_unit":"USD_depth"},"historical":{"raw_input_unit":"USD_depth"}}',
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


def test_request_symbol_is_bound_to_the_observed_identity() -> None:
    row = {
        "feature_name": "funding__1h",
        "endpoint": "fr",
        "signal_timeframe": "1h",
        "required_columns": "fr_close",
        "panel_transform": "raw_column",
        "cross_section_standardization": "none",
        "timestamp_kind": "bar_start",
    }
    item = _item(
        symbol="BTC",
        registry_row=row,
        realtime={"fr_close": 0.1},
        historical={"fr_close": 0.1},
    )
    item["realtime_source_identity"]["request_contract"]["request_params"][
        "symbol"
    ] = "ETHUSDT"
    record = build_factor_equivalence_records([item])[0]
    assert record["status"] == "native_identity_mismatch"
    assert record["final_strategy_input_equal"] is False


def test_missing_invalid_and_transform_errors_remain_distinct() -> None:
    missing_row = {
        "feature_name": "missing__1h",
        "endpoint": "fr",
        "signal_timeframe": "1h",
        "required_columns": "fr_close",
        "panel_transform": "raw_column",
        "cross_section_standardization": "none",
        "timestamp_kind": "bar_start",
    }
    missing = build_factor_equivalence_records([
        _item(
            symbol="BTC", registry_row=missing_row,
            realtime={}, historical={"fr_close": 0.1},
        )
    ])[0]
    assert missing["status"] == "required_field_missing"

    invalid_row = {
        **missing_row,
        "feature_name": "invalid__1h",
    }
    invalid = build_factor_equivalence_records([
        _item(
            symbol="BTC", registry_row=invalid_row,
            realtime={"fr_close": float("nan")}, historical={"fr_close": 0.1},
        )
    ])[0]
    assert invalid["status"] == "invalid_numeric_value"

    transform_row = {
        "feature_name": "imbalance__1h",
        "endpoint": "ob_agg",
        "signal_timeframe": "1h",
        "required_columns": "bids_usd,asks_usd",
        "panel_transform": "buy_sell_imbalance",
        "cross_section_standardization": "none",
        "timestamp_kind": "bar_end",
    }
    failed = build_factor_equivalence_records([
        _item(
            symbol="BTC", registry_row=transform_row,
            realtime={"bids_usd": 0.0, "asks_usd": 0.0},
            historical={"bids_usd": 1.0, "asks_usd": 1.0},
        )
    ])[0]
    assert failed["status"] == "transform_failed"


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), True])
def test_target_numeric_validation_is_strict_without_changing_public_panel(
    bad_value: object,
) -> None:
    row = {
        "feature_name": "strict_target__1h",
        "endpoint": "fr",
        "signal_timeframe": "1h",
        "required_columns": "fr_close",
        "panel_transform": "raw_column",
        "cross_section_standardization": "none",
        "timestamp_kind": "bar_start",
    }

    record = build_factor_equivalence_records([
        _item(
            symbol="BTC",
            registry_row=row,
            realtime={"fr_close": bad_value},
            historical={"fr_close": 0.1},
        )
    ])[0]

    assert record["status"] == "invalid_numeric_value"
    assert record["final_strategy_input_equal"] is False


@pytest.mark.parametrize(
    ("row", "btc_values", "eth_values", "bad_status"),
    [
        (
            {
                "feature_name": "top_pos_invalid__1h",
                "endpoint": "top_pos",
                "signal_timeframe": "1h",
                "required_columns": "top_pos_ls_ratio",
                "panel_transform": "raw_column",
                "cross_section_standardization": "rank_to_minus1_1",
                "timestamp_kind": "bar_end",
            },
            {
                "realtime": {"top_pos_ls_ratio": 1.0},
                "historical": {"top_pos_ls_ratio": 1.0},
            },
            {
                "realtime": {"top_pos_ls_ratio": float("nan")},
                "historical": {"top_pos_ls_ratio": 2.0},
            },
            "invalid_numeric_value",
        ),
        (
            {
                "feature_name": "top_pos_missing__1h",
                "endpoint": "top_pos",
                "signal_timeframe": "1h",
                "required_columns": "top_pos_ls_ratio",
                "panel_transform": "raw_column",
                "cross_section_standardization": "rank_to_minus1_1",
                "timestamp_kind": "bar_end",
            },
            {
                "realtime": {"top_pos_ls_ratio": 1.0},
                "historical": {"top_pos_ls_ratio": 1.0},
            },
            {
                "realtime": {},
                "historical": {"top_pos_ls_ratio": 2.0},
            },
            "required_field_missing",
        ),
        (
            {
                "feature_name": "net_delta_missing_prior__1h",
                "endpoint": "futures_net_pos_v2",
                "signal_timeframe": "1h",
                "required_columns": "net_position_change_cum",
                "panel_transform": "delta1_raw_column",
                "cross_section_standardization": "rank_to_minus1_1",
                "timestamp_kind": "bar_start",
            },
            {
                "realtime": {"net_position_change_cum": 2.0},
                "historical": {"net_position_change_cum": 2.0},
                "realtime_previous": {"net_position_change_cum": 1.0},
                "historical_previous": {"net_position_change_cum": 1.0},
            },
            {
                "realtime": {"net_position_change_cum": 3.0},
                "historical": {"net_position_change_cum": 3.0},
            },
            "missing_prior_observation",
        ),
        (
            {
                "feature_name": "ob_transform_failed__1h",
                "endpoint": "ob_agg",
                "signal_timeframe": "1h",
                "required_columns": "bids_usd,asks_usd",
                "panel_transform": "buy_sell_imbalance",
                "cross_section_standardization": "rank_to_minus1_1",
                "timestamp_kind": "bar_end",
            },
            {
                "realtime": {"bids_usd": 3.0, "asks_usd": 1.0},
                "historical": {"bids_usd": 3.0, "asks_usd": 1.0},
            },
            {
                "realtime": {"bids_usd": 0.0, "asks_usd": 0.0},
                "historical": {"bids_usd": 1.0, "asks_usd": 1.0},
            },
            "transform_failed",
        ),
    ],
    ids=["invalid-numeric", "required-missing", "prior-missing", "transform-failed"],
)
def test_bad_symbol_does_not_turn_healthy_peer_into_strategy_mismatch(
    row: dict[str, object],
    btc_values: dict[str, dict[str, object]],
    eth_values: dict[str, dict[str, object]],
    bad_status: str,
) -> None:
    items = []
    for symbol, values in (("BTC", btc_values), ("ETH", eth_values)):
        items.append(
            _item(
                symbol=symbol,
                registry_row=row,
                realtime=values["realtime"],
                historical=values["historical"],
                realtime_previous=values.get("realtime_previous"),
                historical_previous=values.get("historical_previous"),
            )
        )

    records = build_factor_equivalence_records(
        items,
        expected_symbols=("BTC", "ETH"),
    )
    by_symbol = {record["symbol"]: record for record in records}

    assert by_symbol["ETH"]["status"] == bad_status
    assert by_symbol["BTC"]["status"] == "cross_section_incomplete"
    assert all(record["cross_section_membership_complete"] for record in records)
    assert all(not record["cross_section_complete"] for record in records)
    assert all(record["realtime_standardized_value"] is None for record in records)
    assert all(record["historical_standardized_value"] is None for record in records)
    assert all(record["realtime_raw_rank"] is None for record in records)
    assert all(record["historical_raw_rank"] is None for record in records)
    assert all(not record["final_strategy_input_equal"] for record in records)

    for record in records:
        record["realtime_receipt_id"] = "same-cross-section-event"
    aggregate = aggregate_factor_equivalence_records(records)
    assert len(aggregate) == 1
    assert aggregate[0]["status"] == bad_status
    assert aggregate[0]["final_strategy_input_equal"] is False
    assert aggregate[0]["factor_equivalence_count"] == 2
