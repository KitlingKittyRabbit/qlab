from __future__ import annotations

"""Lifecycle: candidate.

Tests for the candidate KeyStore/CoinGlass v4 data-source infrastructure. Keep
until the replacement route is either promoted to active or archived.
"""

from qlab.data.crypto.keystore_coinglass_client import (
    extract_row_timestamp_ms,
    find_data_rows,
    parse_timestamp_ms,
)
from qlab.data.crypto.keystore_coinglass_endpoints import (
    DEFAULT_EXCHANGE_LIST,
    ENDPOINTS_BY_NAME,
    KEYSTORE_NATIVE_INTERVALS,
    build_history_params,
)
from qlab.data.crypto.keystore_coinglass_parsers import parse_history_frame


def test_keystore_native_intervals_include_reviewed_subdaily_grids():
    assert {"1h", "2h", "4h", "6h", "8h", "12h", "1d"}.issubset(
        set(KEYSTORE_NATIVE_INTERVALS)
    )


def test_build_history_params_uses_pair_for_taker_and_coin_for_weighted_funding():
    taker = ENDPOINTS_BY_NAME["taker_pair"]
    taker_params = build_history_params(taker, symbol="btc", interval="8h", limit=100)
    assert taker_params["exchange"] == "Binance"
    assert taker_params["symbol"] == "BTCUSDT"
    assert taker_params["interval"] == "8h"

    weighted = ENDPOINTS_BY_NAME["fr_oi_weight"]
    weighted_params = build_history_params(weighted, symbol="BTCUSDT", interval="2h", limit=100)
    assert weighted_params["symbol"] == "BTC"
    assert "exchange" not in weighted_params


def test_aggregated_endpoints_default_to_multi_exchange_list():
    assert DEFAULT_EXCHANGE_LIST == "Binance,OKX,Bybit"
    taker_agg = ENDPOINTS_BY_NAME["taker_agg"]
    params = build_history_params(taker_agg, symbol="btc", interval="1h")
    assert params["exchange_list"] == "Binance,OKX,Bybit"
    assert params["limit"] == "4500"


def test_candidate_registry_includes_reviewed_endpoint_families():
    expected = {
        "ob_pair",
        "ob_agg",
        "spot_cvd_agg",
        "futures_net_pos",
        "futures_net_pos_v2",
        "futures_ma",
        "futures_ema",
        "futures_boll",
    }
    assert expected.issubset(set(ENDPOINTS_BY_NAME))


def test_timestamp_and_row_helpers_handle_common_payload_shapes():
    assert parse_timestamp_ms("2026-05-27T00:00:00+00:00") == 1779840000000
    assert parse_timestamp_ms(1779840000) == 1779840000000
    assert extract_row_timestamp_ms({"time": "1779840000000"}) == 1779840000000
    assert find_data_rows({"code": "0", "data": {"list": [{"x": 1}]}}) == [{"x": 1}]


def test_parse_taker_pair_frame_normalizes_buy_sell_columns():
    rows = [
        {
            "time": 1780012800000,
            "taker_buy_volume_usd": "10.5",
            "taker_sell_volume_usd": "7.25",
        }
    ]
    frame = parse_history_frame("taker_pair", rows)
    assert list(frame.columns) == ["buy", "sell"]
    assert frame.iloc[0]["buy"] == 10.5
    assert frame.iloc[0]["sell"] == 7.25


def test_parse_liquidation_frame_uses_aggregated_field_names():
    frame = parse_history_frame(
        "liquidation",
        [
            {
                "time": 1780758000000,
                "aggregated_long_liquidation_usd": "10",
                "aggregated_short_liquidation_usd": "4",
            }
        ],
    )

    assert list(frame.columns) == ["long_liq", "short_liq", "net_liq", "total_liq"]
    assert frame.iloc[0]["net_liq"] == 6
    assert frame.iloc[0]["total_liq"] == 14


def test_parse_oi_and_funding_frames_match_replacement_schema():
    rows = [
        {
            "time": 1780012800000,
            "open": "1",
            "high": "2",
            "low": "0.5",
            "close": "1.5",
        }
    ]
    oi = parse_history_frame("oi_ohlc", rows)
    assert list(oi.columns) == ["oi_open", "oi_high", "oi_low", "oi_close"]
    assert oi.iloc[0]["oi_close"] == 1.5

    fr = parse_history_frame("fr_ohlc", rows)
    assert list(fr.columns) == ["fr_close"]
    assert fr.iloc[0]["fr_close"] == 1.5


def test_parse_long_short_ratio_uses_keystore_v4_field_names():
    frame = parse_history_frame(
        "global_ls",
        [
            {
                "time": 1780754400000,
                "global_account_long_percent": 65.02,
                "global_account_short_percent": 34.98,
                "global_account_long_short_ratio": 1.86,
            }
        ],
    )

    assert list(frame.columns) == ["global_ls_ratio", "global_long_pct", "global_short_pct"]
    assert frame.iloc[0]["global_ls_ratio"] == 1.86
