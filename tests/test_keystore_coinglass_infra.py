from __future__ import annotations

"""Lifecycle: candidate.

Tests for the candidate KeyStore/CoinGlass v4 data-source infrastructure. Keep
until the replacement route is either promoted to active or archived.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from qlab.data.crypto.keystore_coinglass_client import (
    KeystoreCoinglassClient,
    extract_row_timestamp_ms,
    find_data_rows,
    parse_timestamp_ms,
    serialized_request_wait_seconds,
)
from qlab.data.crypto.keystore_coinglass_endpoints import (
    DEFAULT_EXCHANGE_LIST,
    ENDPOINTS_BY_NAME,
    KEYSTORE_NATIVE_INTERVALS,
    build_history_params,
)
from qlab.data.crypto.keystore_coinglass_parsers import parse_history_frame
from qlab.data.crypto import refresh_keystore_coinglass_v4_caches as refresh_module
from qlab.data.crypto.raw_history_store import write_timeseries_history
from qlab.data.crypto.refresh_keystore_coinglass_v4_caches import (
    CACHE_TARGET_START,
    active_cache_intervals,
    cache_refresh_run_lock,
    file_sha256,
    rebuild_caches_from_raw,
    write_new_summary_manifest,
)


def test_keystore_native_intervals_include_reviewed_subdaily_grids():
    assert {"1h", "2h", "4h", "6h", "8h", "12h", "1d"}.issubset(
        set(KEYSTORE_NATIVE_INTERVALS)
    )


def test_serialized_request_wait_uses_request_start_to_start_interval():
    previous = "2026-08-02T00:00:00Z"
    assert serialized_request_wait_seconds(
        previous, "2026-08-02T00:00:01Z", min_start_interval_seconds=6.2
    ) == pytest.approx(5.2)
    assert serialized_request_wait_seconds(
        previous, "2026-08-02T00:00:07Z", min_start_interval_seconds=6.2
    ) == 0.0
    assert serialized_request_wait_seconds(
        None, "2026-08-02T00:00:00Z", min_start_interval_seconds=6.2
    ) == 0.0
    with pytest.raises(ValueError, match="precedes"):
        serialized_request_wait_seconds(
            previous, "2026-08-01T23:59:59Z", min_start_interval_seconds=6.2
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


def test_keystore_client_preserves_successful_raw_response(monkeypatch):
    class Response:
        status_code = 200
        content = b'{"code":"0","data":[{"time":1780012800000,"close":"1.5"}]}'

    monkeypatch.setattr(
        "qlab.data.crypto.keystore_coinglass_client.requests.get",
        lambda *args, **kwargs: Response(),
    )
    client = KeystoreCoinglassClient(api_key="fixture", rate_limit_sleep=0.0)
    observed = client.request_raw(
        "/api/futures/funding-rate/history",
        {"symbol": "BTCUSDT", "interval": "1h", "limit": "2"},
    )
    assert observed.raw_payload == Response.content
    assert observed.request_params["symbol"] == "BTCUSDT"
    assert observed.request_ts <= observed.response_ts
    assert find_data_rows(observed.json_payload())[0]["close"] == "1.5"


def test_keystore_client_retries_transient_business_500(monkeypatch):
    class Response:
        status_code = 200

        def __init__(self, content: bytes) -> None:
            self.content = content

    responses = iter(
        [
            Response(b'{"code":"500","msg":"Server Error"}'),
            Response(b'{"code":"0","data":[{"time":1780012800000,"close":"1.5"}]}'),
        ]
    )
    monkeypatch.setattr(
        "qlab.data.crypto.keystore_coinglass_client.requests.get",
        lambda *args, **kwargs: next(responses),
    )
    monkeypatch.setattr(
        "qlab.data.crypto.keystore_coinglass_client.time.sleep", lambda _: None
    )
    client = KeystoreCoinglassClient(api_key="fixture", rate_limit_sleep=0.0)
    observed = client.request_raw("/api/futures/funding-rate/history", retries=2)
    assert observed.json_payload()["code"] == "0"


def test_keystore_client_spaces_every_request_start(monkeypatch):
    monotonic = iter([0.0, 0.0, 1.0, 6.2])
    sleeps = []
    monkeypatch.setattr(
        "qlab.data.crypto.keystore_coinglass_client.time.monotonic",
        lambda: next(monotonic),
    )
    monkeypatch.setattr(
        "qlab.data.crypto.keystore_coinglass_client.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )
    client = KeystoreCoinglassClient(api_key="fixture", rate_limit_sleep=6.2)
    client._wait_for_request_start_slot()
    client._wait_for_request_start_slot()
    assert sleeps == [pytest.approx(5.2)]


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


def test_refresh_summary_manifest_is_immutable(tmp_path):
    destination = tmp_path / "refresh.csv"
    summary = pd.DataFrame({"status": ["ok"]})
    write_new_summary_manifest(summary, destination)
    with pytest.raises(FileExistsError, match="already exists"):
        write_new_summary_manifest(summary, destination)
    assert pd.read_csv(destination).to_dict("records") == [{"status": "ok"}]


def test_refresh_summary_manifest_allows_only_one_concurrent_writer(tmp_path):
    destination = tmp_path / "refresh.csv"

    def attempt(value):
        try:
            write_new_summary_manifest(pd.DataFrame({"value": [value]}), destination)
            return "written"
        except FileExistsError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, [1, 2]))
    assert sorted(results) == ["rejected", "written"]
    assert pd.read_csv(destination)["value"].iloc[0] in {1, 2}


def test_cache_refresh_run_lock_rejects_overlapping_runs(tmp_path):
    lock_path = tmp_path / "refresh.lock"
    with cache_refresh_run_lock(lock_path):
        with pytest.raises(RuntimeError, match="already running"):
            with cache_refresh_run_lock(lock_path):
                pass


def test_active_cache_intervals_include_prior_active_caches(tmp_path):
    (tmp_path / "keystore_coinglass_v4_1h_cache.pkl").touch()
    (tmp_path / "keystore_coinglass_v4_12h_cache.pkl").touch()
    assert active_cache_intervals(
        ("12h",),
        cache_directory=tmp_path,
    ) == ("12h", "1h")


def test_rebuild_caches_from_raw_publishes_complete_interval_payloads(
    tmp_path,
    monkeypatch,
):
    raw_root = tmp_path / "raw"
    cache_root = tmp_path / "cache"
    manifest_root = tmp_path / "manifest"
    index = pd.to_datetime(["2024-05-31", "2024-06-01", "2024-06-02"], utc=True)
    metadata = {
        "api_version": "v4",
        "source": "fixture",
        "interval": "1h",
        "symbol": "BTC",
        "endpoint": "basis",
        "path": "/basis",
        "parser": "basis",
        "migration_type": "",
    }
    for interval in ("1h", "12h"):
        interval_metadata = {**metadata, "interval": interval}
        for symbol in ("BTC", "ETH"):
            write_timeseries_history(
                pd.DataFrame({"close_basis": [0.0, 1.0, 2.0]}, index=index),
                raw_root / "keystore_v4" / interval / f"{symbol}_basis.csv",
                metadata={**interval_metadata, "symbol": symbol},
            )

    monkeypatch.setattr(refresh_module, "RAW_HISTORY_ROOT", raw_root)
    monkeypatch.setattr(
        refresh_module,
        "cache_path",
        lambda name: cache_root / name,
    )
    monkeypatch.setattr(
        refresh_module,
        "manifest_path",
        lambda name: manifest_root / name,
    )
    rebuild_caches_from_raw(
        intervals=("1h", "12h"),
        summary_output="summary.csv",
        target_start=pd.Timestamp("2024-06-01T00:00:00Z"),
    )

    for interval in ("1h", "12h"):
        payload = pd.read_pickle(
            cache_root / f"keystore_coinglass_v4_{interval}_cache.pkl"
        )
        assert sorted(payload) == ["BTC_basis", "ETH_basis"]
        assert all(
            frame.index.min() == pd.Timestamp("2024-06-01T00:00:00Z")
            for frame in payload.values()
        )
    summary = pd.read_csv(manifest_root / "summary.csv")
    assert summary.groupby("interval").size().to_dict() == {"12h": 2, "1h": 2}
    assert set(summary["cache_target_start"]) == {str(CACHE_TARGET_START)}
    assert all(
        Path(row.raw_source).parent.name == row.interval
        for row in summary.itertuples(index=False)
    )
    assert all(
        file_sha256(Path(row.raw_source)) == row.raw_source_sha256
        for row in summary.itertuples(index=False)
    )
    assert all(
        file_sha256(Path(row.cache_path)) == row.cache_sha256
        for row in summary.itertuples(index=False)
    )


def test_manifest_cleanup_failure_does_not_invalidate_published_manifest(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "summary.csv"
    original_unlink = Path.unlink

    def fail_disposable_cleanup(path, *args, **kwargs):
        if path.name.endswith(".lock") or path.name.endswith(".staged"):
            raise OSError("simulated cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_disposable_cleanup)
    write_new_summary_manifest(pd.DataFrame({"status": ["ok"]}), destination)
    assert pd.read_csv(destination).to_dict("records") == [{"status": "ok"}]
