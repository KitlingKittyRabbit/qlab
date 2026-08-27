from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from qlab.data.crypto import ksv4_realtime
from qlab.data.crypto.ksv4_realtime import (
    PublicGetClient,
    aggregate_depth_usd,
    aggregate_available_depth_usd,
    binance_depth_usd,
    build_realtime_cache_payloads,
    bybit_depth_usd,
    close_basis_percent,
    depth_covers_band,
    index_coins_markets,
    index_funding_exchange_list,
    latest_net_position_value,
    latest_net_position_observation,
    latest_completed_native_identity,
    net_position_history_frame,
    net_position_observation_at,
    latest_orderbook_history_imbalance,
    orderbook_history_imbalance_at,
    orderbook_history_depth_at,
    ratio_observation_at,
    ratio_history_frame,
    realtime_projection_identity,
    latest_ratio_observation,
    latest_ratio_value,
    okx_depth_usd,
    pairs_markets_params,
    repaired_realtime_raw_values,
    realtime_raw_values,
    select_binance_usdt_pair,
    shadow_response_native_identity,
)


def test_public_exports_are_defined_and_use_only_shadow_contract_v5() -> None:
    assert set(ksv4_realtime.__all__).issubset(vars(ksv4_realtime))
    assert "REALTIME_SOURCE_CONTRACT_VERSION" not in ksv4_realtime.__all__
    assert ksv4_realtime.SHADOW_SOURCE_CONTRACT_VERSION == "ksv4_shadow_sources_v5"


def test_pairs_markets_params_requires_base_asset_not_pair() -> None:
    assert pairs_markets_params(" ada ") == {"symbol": "ADA"}
    with pytest.raises(ValueError, match="base-asset symbol"):
        pairs_markets_params("ADAUSDT")
    with pytest.raises(ValueError, match="base-asset symbol"):
        pairs_markets_params("ADA/USDT")
    with pytest.raises(ValueError, match="base-asset symbol"):
        pairs_markets_params("ADA-USDT")


class _Response:
    status_code = 200
    text = ""

    def __init__(self, payload):
        self.content = json.dumps(payload).encode()


class _Session:
    def __init__(self):
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _Response({"code": "0", "data": []})


def test_public_get_client_is_host_locked_and_get_only() -> None:
    with pytest.raises(ValueError, match="okx or bybit"):
        PublicGetClient("binance")
    session = _Session()
    client = PublicGetClient("okx", session=session)
    result = client.request_raw("/api/v5/market/books", params={"instId": "BTC-USDT-SWAP"})
    assert result.venue == "okx"
    assert session.calls == [
        (
            "https://www.okx.com/api/v5/market/books",
            {
                "params": {"instId": "BTC-USDT-SWAP"},
                "headers": {
                    "User-Agent": "qlab-true-oos/1.0",
                    "Accept": "application/json",
                },
                "timeout": 15.0,
            },
        )
    ]
    assert not hasattr(client, "post")


def test_keystore_realtime_payload_selection_and_basis() -> None:
    coins = index_coins_markets(
        {"code": "0", "data": [{"symbol": "ADA", "open_interest_usd": 10}]}
    )
    assert coins["ADA"]["open_interest_usd"] == 10
    funding = index_funding_exchange_list(
        {
            "code": "0",
            "data": [
                {
                    "symbol": "ADA",
                    "stablecoin_margin_list": [
                        {"exchange": "OKX", "funding_rate": 0.01},
                        {"exchange": "Binance", "funding_rate": -0.02},
                    ],
                }
            ],
        }
    )
    assert funding == {"ADA": -0.02}
    pair = select_binance_usdt_pair(
        {
            "code": "0",
            "data": [
                {
                    "exchange_name": "Binance",
                    "instrument_id": "ADAUSDT",
                    "current_price": 99,
                    "index_price": 100,
                }
            ],
        },
        "ADA",
    )
    assert close_basis_percent(pair) == pytest.approx(1.0)


def test_latest_net_position_and_ratio_are_selected_by_timestamp() -> None:
    assert latest_net_position_value(
        {
            "code": "0",
            "data": [
                {"time": 2, "net_position_change_cum": 20},
                {"time": 1, "net_position_change_cum": 10},
            ],
        }
    ) == 20
    assert latest_ratio_value(
        [
            {"timestamp": 1, "longShortRatio": "1.2"},
            {"timestamp": 2, "longShortRatio": "1.3"},
        ]
    ) == 1.3
    net_value, net_ts = latest_net_position_observation(
        {"code": "0", "data": [{"time": 1_722_470_400_000, "net_position_change_cum": 20}]}
    )
    ratio_value, ratio_ts = latest_ratio_observation(
        [{"timestamp": 1_722_470_400_000, "longShortRatio": "1.3"}]
    )
    assert net_value == 20
    assert ratio_value == 1.3
    assert net_ts == pd.Timestamp("2024-08-01T00:00:00Z")
    assert ratio_ts == pd.Timestamp("2024-08-01T00:00:00Z")


def test_orderbook_history_only_allows_prior_positive_row_for_diagnostic_mode() -> None:
    payload = {
        "code": "0",
        "data": [
            {"time": 1_722_470_400_000, "bid": 60.0, "ask": 40.0},
            {"time": 1_722_470_460_000, "bid": 0.0, "ask": 0.0},
        ],
    }
    with pytest.raises(ValueError, match="no eligible positive"):
        latest_orderbook_history_imbalance(
            payload, bid_key="bid", ask_key="ask"
        )
    value, timestamp = latest_orderbook_history_imbalance(
        payload,
        bid_key="bid",
        ask_key="ask",
        allow_incomplete_latest=True,
    )
    assert value == pytest.approx(0.2)
    assert timestamp == pd.Timestamp("2024-08-01T00:00:00Z")


def test_three_venue_one_percent_depth_is_hand_calculable() -> None:
    # Midpoint 100: 99 and 101 are included, 98 and 102 are outside the band.
    binance = binance_depth_usd(
        {"bids": [[99, 2], [98, 10]], "asks": [[101, 3], [102, 10]]}
    )
    bybit = bybit_depth_usd(
        {"result": {"b": [[99, 1], [98, 10]], "a": [[101, 1], [102, 10]]}}
    )
    okx = okx_depth_usd(
        {
            "code": "0",
            "data": [{"bids": [[99, 4]], "asks": [[101, 5]]}],
        },
        {
            "code": "0",
            "data": [{"ctVal": "0.5", "ctValCcy": "BTC", "settleCcy": "USDT"}],
        },
    )
    assert binance == (198.0, 303.0)
    assert bybit == (99.0, 101.0)
    assert okx == (198.0, 252.5)
    assert aggregate_depth_usd(binance, okx, bybit) == (495.0, 656.5)
    assert aggregate_available_depth_usd(
        {"binance": binance, "bybit": bybit}
    ) == (297.0, 404.0)
    assert aggregate_available_depth_usd({"binance": binance}) == binance
    with pytest.raises(ValueError, match="at least one"):
        aggregate_available_depth_usd({})


def test_finite_orderbook_must_span_the_full_one_percent_band() -> None:
    assert depth_covers_band(
        [[100, 1], [99, 1]],
        [[100, 1], [101, 1]],
        midpoint=100,
    )
    assert not depth_covers_band(
        [[100, 1], [99.5, 1]],
        [[100, 1], [101, 1]],
        midpoint=100,
    )


def test_realtime_values_and_delta_seed_build_existing_panel_contract() -> None:
    raw = realtime_raw_values(
        symbol="ADA",
        coins_row={
            "avg_funding_rate_by_oi": 0.1,
            "avg_funding_rate_by_vol": 0.2,
            "open_interest_usd": 110,
            "long_liquidation_usd_24h": 30,
            "short_liquidation_usd_24h": 10,
        },
        funding_rate=0.3,
        pair_row={"current_price": 99, "index_price": 100},
        net_position_value=50,
        global_ratio=1.1,
        top_account_ratio=1.2,
        top_position_ratio=1.3,
        pair_depth=(20, 10),
        aggregate_depth=(30, 15),
    )
    assert raw["basis"]["close_basis"] == 1.0
    assert raw["liq"] == {"long_liq": 30.0, "short_liq": 10.0}

    registry = pd.DataFrame(
        [
                {
                    "feature_name": "oi_close_delta1__12h",
                    "source_scope": "ksv4_12h",
                    "signal_timeframe": "12h",
                    "endpoint": "oi",
                    "timestamp_kind": "bar_start",
                    "panel_transform": "delta1_raw_column",
                },
            {
                "feature_name": "funding_close__12h",
                "source_scope": "ksv4_12h",
                "signal_timeframe": "12h",
                "endpoint": "fr",
                "timestamp_kind": "bar_start",
            },
            {
                "feature_name": "ob_agg_imbalance__12h",
                "source_scope": "ksv4_12h",
                "signal_timeframe": "12h",
                "endpoint": "ob_agg",
                "timestamp_kind": "bar_end",
            },
        ]
    )
    prior_index = pd.DatetimeIndex(["2026-07-30T00:00:00Z"], name="ts")
    prior_realtime = {
        "ksv4_12h": {
            "ADA_oi": pd.DataFrame({"oi_close": [100.0]}, index=prior_index),
            "ADA_fr": pd.DataFrame({"fr_close": [0.2]}, index=prior_index),
            "ADA_ob_agg": pd.DataFrame(
                {"aggregated_bids_usd": [20.0], "aggregated_asks_usd": [10.0]},
                index=prior_index,
            ),
        }
    }
    built = build_realtime_cache_payloads(
        symbols=["ADA"],
        registry_frame=registry,
        decision_ts="2026-07-31T00:00:00Z",
        values_by_symbol={"ADA": raw},
        previous_realtime_payloads=prior_realtime,
    )
    assert list(built["ksv4_12h"]["ADA_oi"].index) == [
        pd.Timestamp("2026-07-30T00:00:00Z"),
        pd.Timestamp("2026-07-30T12:00:00Z"),
    ]
    assert built["ksv4_12h"]["ADA_oi"].iloc[-1]["oi_close"] == 110.0
    # Non-delta endpoints must not inherit a historical/prior row merely
    # because a caller supplied one; the current realtime observation is the
    # only row needed for this projection.
    assert list(built["ksv4_12h"]["ADA_ob_agg"].index) == [
        pd.Timestamp("2026-07-31T00:00:00Z"),
    ]


def test_repaired_overlay_keeps_1h_and_12h_top_position_distinct() -> None:
    registry = pd.DataFrame(
        [
            {
                "feature_name": f"top_pos_ls_ratio__{timeframe}",
                "source_scope": f"ksv4_{timeframe}",
                "signal_timeframe": timeframe,
                "endpoint": "top_pos",
                "timestamp_kind": "bar_end",
            }
            for timeframe in ("1h", "12h")
        ]
    )
    values = repaired_realtime_raw_values(
        coins_row={
            "avg_funding_rate_by_oi": 0.1,
            "avg_funding_rate_by_vol": 0.2,
            "open_interest_usd": 100.0,
        },
        funding_rate=0.3,
        net_position_values={"1h": 10.0, "1d": 20.0},
        top_position_ratio_1h=1.1,
        top_position_ratio_12h=1.2,
        pair_depth=(2.0, 1.0),
        aggregate_depth=(3.0, 1.5),
    )
    prior = pd.DatetimeIndex(["2026-07-30T00:00:00Z"], name="ts")
    historical = {
        scope: {
            "ADA_top_pos": pd.DataFrame({"top_pos_ls_ratio": [1.0]}, index=prior)
        }
        for scope in ("ksv4_1h", "ksv4_12h")
    }
    built = build_realtime_cache_payloads(
        symbols=["ADA"], registry_frame=registry,
        decision_ts="2026-07-31T00:00:00Z",
        values_by_symbol={"ADA": values}, previous_realtime_payloads=historical,
    )
    assert built["ksv4_1h"]["ADA_top_pos"].iloc[-1, 0] == 1.1
    assert built["ksv4_12h"]["ADA_top_pos"].iloc[-1, 0] == 1.2


def test_repaired_values_require_both_native_net_position_identities() -> None:
    kwargs = {
        "coins_row": {
            "avg_funding_rate_by_oi": 0.1,
            "avg_funding_rate_by_vol": 0.2,
            "open_interest_usd": 100.0,
        },
        "funding_rate": 0.3,
        "top_position_ratio_1h": 1.1,
        "top_position_ratio_12h": 1.2,
        "pair_depth": (2.0, 1.0),
        "aggregate_depth": (3.0, 1.5),
    }
    with pytest.raises(ValueError, match="native 1h and 1d"):
        repaired_realtime_raw_values(net_position_values={"1h": 10.0}, **kwargs)
    values = repaired_realtime_raw_values(
        net_position_values={"1h": 10.0, "1d": 20.0}, **kwargs
    )
    assert values["ksv4_1h:futures_net_pos_v2"]["net_position_change_cum"] == 10.0
    assert values["ksv4_1d:futures_net_pos_v2"]["net_position_change_cum"] == 20.0


def test_exact_native_observation_selectors_reject_adjacent_rows() -> None:
    decision = pd.Timestamp("2026-08-06T00:00:00Z")
    one_hour = int(pd.Timestamp("2026-08-05T23:00:00Z").timestamp() * 1000)
    two_hours = int(pd.Timestamp("2026-08-05T22:00:00Z").timestamp() * 1000)
    net_payload = [
        {"time": two_hours, "net_position_change_cum": 1.0},
        {"time": one_hour, "net_position_change_cum": 2.0},
    ]
    ratio_payload = [
        {"timestamp": two_hours, "longShortRatio": "1.1"},
        {"timestamp": one_hour, "longShortRatio": "1.2"},
    ]
    book_payload = [
        {"time": one_hour, "bids_usd": 3.0, "asks_usd": 1.0},
        {
            "time": int(decision.timestamp() * 1000),
            "bids_usd": 4.0,
            "asks_usd": 2.0,
        },
        {
            "time": int((decision + pd.Timedelta(minutes=1)).timestamp() * 1000),
            "bids_usd": 1.0,
            "asks_usd": 9.0,
        },
        {
            "time": int((decision + pd.Timedelta(minutes=2)).timestamp() * 1000),
            "bids_usd": 9.0,
            "asks_usd": 1.0,
        },
    ]
    assert net_position_observation_at(net_payload, decision - pd.Timedelta(hours=1))[0] == 2.0
    assert ratio_observation_at(ratio_payload, decision - pd.Timedelta(hours=1))[0] == 1.2
    assert orderbook_history_imbalance_at(
        book_payload,
        target_label_ts=decision,
        bid_key="bids_usd",
        ask_key="asks_usd",
    )[0] == pytest.approx(1.0 / 3.0)
    with pytest.raises(ValueError, match="got 0"):
        net_position_observation_at(net_payload, decision)


def test_realtime_history_frames_keep_all_exact_native_observations() -> None:
    payload = {
        "code": "0",
        "data": [
            {
                "time": int(pd.Timestamp("2026-08-06T18:00:00Z").timestamp() * 1000),
                "net_position_change_cum": 10.0,
            },
            {
                "time": int(pd.Timestamp("2026-08-06T19:00:00Z").timestamp() * 1000),
                "net_position_change_cum": 12.0,
            },
        ],
    }
    frame = net_position_history_frame(payload)
    assert frame["net_position_change_cum"].tolist() == [10.0, 12.0]
    ratio = ratio_history_frame(
        {
            "code": "0",
            "data": [
                {
                    "timestamp": int(
                        pd.Timestamp("2026-08-06T18:00:00Z").timestamp() * 1000
                    ),
                    "longShortRatio": "1.1",
                },
                {
                    "timestamp": int(
                        pd.Timestamp("2026-08-06T19:00:00Z").timestamp() * 1000
                    ),
                    "longShortRatio": "1.2",
                },
            ],
        }
    )
    assert ratio["top_pos_ls_ratio"].tolist() == [1.1, 1.2]
    with pytest.raises(ValueError, match="duplicate native labels"):
        net_position_history_frame(
            {
                "code": "0",
                "data": [
                    {
                        "time": int(
                            pd.Timestamp("2026-08-06T19:00:00Z").timestamp() * 1000
                        ),
                        "net_position_change_cum": 12.0,
                    },
                    {
                        "time": int(
                            pd.Timestamp("2026-08-06T19:00:00Z").timestamp() * 1000
                        ),
                        "net_position_change_cum": 13.0,
                    },
                ],
            }
        )


def test_delta_projection_fails_without_exact_realtime_prior() -> None:
    registry = pd.DataFrame(
        [
            {
                "source_scope": "ksv4_1h",
                "signal_timeframe": "1h",
                "endpoint": "oi",
                "timestamp_kind": "bar_start",
                "panel_transform": "delta1_raw_column",
            }
        ]
    )
    with pytest.raises(ValueError, match="missing exact previous realtime observation"):
        build_realtime_cache_payloads(
            symbols=["BTC"],
            registry_frame=registry,
            decision_ts="2026-08-06T20:00:00Z",
            values_by_symbol={"BTC": {"oi": {"oi_close": 120.0}}},
        )


def test_net_response_identity_uses_the_requested_native_interval() -> None:
    decision = pd.Timestamp("2026-08-06T00:00:00Z")
    for timeframe, label in (
        ("1h", decision - pd.Timedelta(hours=1)),
        ("1d", decision - pd.Timedelta(days=1)),
    ):
        response = ksv4_realtime.ShadowSourceResponse(
            request_id=f"net-{timeframe}", request_order=1,
            source="keystore", route="futures/v2/net-position/history",
            symbol="BTC", signal_timeframe=timeframe,
            request_path="/api/futures/v2/net-position/history",
            request_params={"interval": timeframe}, raw_payload=b"{}",
            request_ts=decision.isoformat(), response_ts=decision.isoformat(),
        )
        payload = {
            "code": "0",
            "data": [{"time": int(label.timestamp() * 1000), "net_position_change_cum": 1.0}],
        }
        observed, native_end = shadow_response_native_identity(
            response, payload, decision_ts=decision
        )
        assert observed == label
        assert native_end == decision


def test_orderbook_depth_selector_returns_real_fields_without_pseudo_depth() -> None:
    target = pd.Timestamp("2026-08-06T00:00:00Z")
    depth = orderbook_history_depth_at(
        {
            "code": "0",
            "data": [{
                "time": int(target.timestamp() * 1000),
                "bids_usd": 125.0,
                "asks_usd": 75.0,
            }],
        },
        target_label_ts=target,
        bid_key="bids_usd",
        ask_key="asks_usd",
    )
    assert depth == (125.0, 75.0, target)


def test_realtime_projection_identity_distinguishes_start_and_end_labels() -> None:
    registry = pd.DataFrame(
        [
            {
                "source_scope": "ksv4_12h",
                "endpoint": "oi",
                "signal_timeframe": "12h",
                "timestamp_kind": "bar_start",
            },
            {
                "source_scope": "ksv4_12h",
                "endpoint": "ob_agg",
                "signal_timeframe": "12h",
                "timestamp_kind": "bar_end",
            },
        ]
    )
    decision = pd.Timestamp("2026-08-06T00:00:00Z")
    assert realtime_projection_identity(
        registry,
        source_scope="ksv4_12h",
        endpoint="oi",
        decision_ts=decision,
    ) == (decision - pd.Timedelta(hours=12), decision)
    assert realtime_projection_identity(
        registry,
        source_scope="ksv4_12h",
        endpoint="ob_agg",
        decision_ts=decision,
    ) == (decision, decision)


def test_latest_completed_native_identity_handles_off_phase_acquisition() -> None:
    decision = pd.Timestamp("2026-08-06T20:00:00Z")
    assert latest_completed_native_identity(
        signal_timeframe="12h",
        timestamp_kind="bar_start",
        as_of_ts=decision,
    ) == (
        pd.Timestamp("2026-08-06T00:00:00Z"),
        pd.Timestamp("2026-08-06T12:00:00Z"),
    )
    assert latest_completed_native_identity(
        signal_timeframe="1d",
        timestamp_kind="bar_end",
        as_of_ts=decision,
    ) == (
        pd.Timestamp("2026-08-06T00:00:00Z"),
        pd.Timestamp("2026-08-06T00:00:00Z"),
    )


def test_projection_rejects_a_decision_off_the_signal_phase() -> None:
    registry = pd.DataFrame(
        [
            {
                "source_scope": "ksv4_12h",
                "endpoint": "top_pos",
                "signal_timeframe": "12h",
                "timestamp_kind": "bar_start",
            }
        ]
    )
    with pytest.raises(ValueError, match="not aligned"):
        realtime_projection_identity(
            registry,
            source_scope="ksv4_12h",
            endpoint="top_pos",
            decision_ts="2026-08-06T20:00:00Z",
        )


def test_nondue_12h_ratio_is_bound_to_latest_completed_period() -> None:
    response = SimpleNamespace(
        route="top-position-ratio",
        signal_timeframe="12h",
        response_ts="2026-08-06T20:02:00Z",
    )
    payload = [
        {
            "timestamp": int(pd.Timestamp("2026-08-06T00:00:00Z").timestamp() * 1000),
            "longShortRatio": "1.1",
        },
        {
            "timestamp": int(pd.Timestamp("2026-08-06T12:00:00Z").timestamp() * 1000),
            "longShortRatio": "1.2",
        },
    ]
    assert shadow_response_native_identity(
        response,
        payload,
        decision_ts="2026-08-06T20:00:00Z",
    ) == (
        pd.Timestamp("2026-08-06T00:00:00Z"),
        pd.Timestamp("2026-08-06T12:00:00Z"),
    )


def test_off_phase_source_can_be_acquired_but_not_projected() -> None:
    registry = pd.DataFrame(
        [
            {
                "source_scope": "ksv4_1h",
                "endpoint": "oi",
                "signal_timeframe": "1h",
                "timestamp_kind": "bar_start",
                "panel_transform": "delta1_raw_column",
            },
            {
                "source_scope": "ksv4_12h",
                "endpoint": "top_pos",
                "signal_timeframe": "12h",
                "timestamp_kind": "bar_start",
            },
        ]
    )
    decision = pd.Timestamp("2026-08-06T20:00:00Z")
    due_registry = registry.loc[registry["signal_timeframe"].eq("1h")]
    payloads = build_realtime_cache_payloads(
        symbols=["BTC"],
        registry_frame=due_registry,
        decision_ts=decision,
        values_by_symbol={"BTC": {"oi": {"oi_close": 120.0}}},
        previous_realtime_payloads={
            "ksv4_1h": {
                "BTC_oi": pd.DataFrame(
                    {"oi_close": [100.0]},
                    index=pd.DatetimeIndex(["2026-08-06T18:00:00Z"], name="ts"),
                )
            }
        },
    )
    assert set(payloads) == {"ksv4_1h"}
    assert list(payloads["ksv4_1h"]["BTC_oi"].index) == [
        pd.Timestamp("2026-08-06T18:00:00Z"),
        pd.Timestamp("2026-08-06T19:00:00Z"),
    ]
    with pytest.raises(ValueError, match="not aligned"):
        build_realtime_cache_payloads(
            symbols=["BTC"],
            registry_frame=registry,
            decision_ts=decision,
            values_by_symbol={
                "BTC": {
                    "oi": {"oi_close": 120.0},
                    "ksv4_12h:top_pos": {"top_pos_ls_ratio": 1.2},
                }
            },
            previous_realtime_payloads={
                "ksv4_1h": {
                    "BTC_oi": pd.DataFrame(
                        {"oi_close": [100.0]},
                        index=pd.DatetimeIndex(
                            ["2026-08-06T18:00:00Z"], name="ts"
                        ),
                    )
                },
                "ksv4_12h": {
                    "BTC_top_pos": pd.DataFrame(
                        {"top_pos_ls_ratio": [1.0]},
                        index=pd.DatetimeIndex(
                            ["2026-08-05T12:00:00Z"], name="ts"
                        ),
                    )
                },
            },
        )
