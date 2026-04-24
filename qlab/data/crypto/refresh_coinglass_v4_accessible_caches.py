from __future__ import annotations

import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

if __package__ in (None, ""):
    PACKAGE_ROOT = Path(__file__).resolve().parents[3]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))

from qlab.data.crypto.paths import TRADE_ENV_PATH, cache_path, ensure_data_dirs, manifest_path
from qlab.data.crypto.raw_history_store import (
    RAW_HISTORY_ROOT,
    append_snapshot_history,
    write_timeseries_history,
)
from qlab.data.crypto.symbol_universe import RESEARCH_SYMBOLS_12

BASE = "https://open-api-v4.coinglass.com/api"
ENV_PATH = TRADE_ENV_PATH
SYMBOLS = RESEARCH_SYMBOLS_12
EXCHANGE_SYMBOLS = {symbol: f"{symbol}USDT" for symbol in SYMBOLS}
INTERVALS = ["4h", "6h", "12h", "1d"]
RATE_LIMIT_SLEEP = 2.2


def get_rate_limit_sleep() -> float:
    raw_value = os.environ.get("COINGLASS_RATE_LIMIT_SLEEP", "").strip()
    if not raw_value:
        return RATE_LIMIT_SLEEP
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        return RATE_LIMIT_SLEEP


@dataclass(frozen=True)
class SymbolEndpoint:
    name: str
    path: str
    params_kind: str
    parser: str
    cache_prefix: str
    supported_symbols: tuple[str, ...] | None = None


@dataclass(frozen=True)
class GlobalEndpoint:
    name: str
    path: str
    parser: str
    supports_interval: bool
    static_params: dict[str, str] | None = None


SYMBOL_ENDPOINTS = [
    SymbolEndpoint(
        name="taker_pair",
        path="futures/v2/taker-buy-sell-volume/history",
        params_kind="pair_exchange_interval",
        parser="taker_pair",
        cache_prefix="taker_pair",
    ),
    SymbolEndpoint(
        name="taker_agg",
        path="futures/aggregated-taker-buy-sell-volume/history",
        params_kind="agg_exchange_interval",
        parser="taker_agg",
        cache_prefix="taker_agg",
    ),
    SymbolEndpoint(
        name="basis",
        path="futures/basis/history",
        params_kind="pair_exchange_interval",
        parser="basis",
        cache_prefix="basis",
    ),
    SymbolEndpoint(
        name="coinbase_premium",
        path="coinbase-premium-index",
        params_kind="coin_interval",
        parser="coinbase_premium",
        cache_prefix="coinbase_premium",
        supported_symbols=("BTC",),
    ),
    SymbolEndpoint(
        name="oi_stablecoin",
        path="futures/open-interest/aggregated-stablecoin-history",
        params_kind="agg_exchange_interval",
        parser="ohlc",
        cache_prefix="oi_stablecoin",
    ),
    SymbolEndpoint(
        name="oi_coin_margin",
        path="futures/open-interest/aggregated-coin-margin-history",
        params_kind="agg_exchange_interval",
        parser="ohlc",
        cache_prefix="oi_coin_margin",
    ),
    SymbolEndpoint(
        name="fr_oi_weight",
        path="futures/funding-rate/oi-weight-history",
        params_kind="coin_interval",
        parser="ohlc",
        cache_prefix="fr_oi_weight",
    ),
    SymbolEndpoint(
        name="fr_vol_weight",
        path="futures/funding-rate/vol-weight-history",
        params_kind="coin_interval",
        parser="ohlc",
        cache_prefix="fr_vol_weight",
    ),
    SymbolEndpoint(
        name="bitfinex_margin",
        path="bitfinex-margin-long-short",
        params_kind="coin_interval",
        parser="bitfinex_margin",
        cache_prefix="bitfinex_margin",
        supported_symbols=("BTC", "ETH"),
    ),
]

GLOBAL_ENDPOINTS = [
    GlobalEndpoint(
        name="etf_btc",
        path="etf/bitcoin/flow-history",
        parser="etf_btc",
        supports_interval=False,
        static_params={"limit": "5000"},
    ),
    GlobalEndpoint(
        name="etf_eth",
        path="etf/ethereum/flow-history",
        parser="etf_eth",
        supports_interval=False,
        static_params={"limit": "5000"},
    ),
    GlobalEndpoint(
        name="fear_greed",
        path="index/fear-greed-history",
        parser="series_list",
        supports_interval=False,
        static_params={"limit": "5000"},
    ),
    GlobalEndpoint(
        name="cgdi",
        path="futures/cgdi-index/history",
        parser="cgdi",
        supports_interval=False,
        static_params={"limit": "5000"},
    ),
    GlobalEndpoint(
        name="stablecoin_mcap",
        path="index/stableCoin-marketCap-history",
        parser="series_list",
        supports_interval=False,
        static_params={"limit": "5000"},
    ),
]

SNAPSHOT_ENDPOINTS = [
    GlobalEndpoint(
        name="oi_exchange",
        path="futures/open-interest/exchange-list",
        parser="snapshot_list",
        supports_interval=False,
        static_params=None,
    ),
    GlobalEndpoint(
        name="option_maxpain",
        path="option/max-pain",
        parser="snapshot_list",
        supports_interval=False,
        static_params={"exchange": "Deribit"},
    ),
]


def log(message: str) -> None:
    print(message, flush=True)


def load_api_key() -> str:
    if not ENV_PATH.exists():
        raise FileNotFoundError(f"Missing env file: {ENV_PATH}")

    for line in ENV_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("COINGLASS_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(
        "COINGLASS_API_KEY not found in trade/crypto_signal/.env")


def to_datetime_index(values: list | pd.Series) -> pd.DatetimeIndex:
    series = pd.Series(values).dropna().astype(float)
    if series.empty:
        return pd.DatetimeIndex([])
    unit = "ms" if float(series.abs().max()) >= 1e12 else "s"
    return pd.to_datetime(series, unit=unit, utc=True)


def fetch_json(path: str, headers: dict[str, str], params: dict[str, str], retries: int = 4):
    url = f"{BASE}/{path}"
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url, headers=headers, params=params, timeout=30)
            payload = response.json()
        except Exception as exc:
            last_error = exc
            time.sleep(3 * (attempt + 1))
            continue

        code = str(payload.get("code", payload.get("status", "")))
        if response.status_code == 429 or code in {"429", "40003"}:
            time.sleep(10)
            continue
        if response.status_code == 200 and code == "0":
            return payload.get("data", [])

        last_error = RuntimeError(
            f"HTTP={response.status_code}, code={code}, msg={payload.get('msg', payload.get('error', ''))}"
        )
        break

    raise RuntimeError(
        f"CoinGlass v4 request failed for {path} params={params}: {last_error}")


def build_symbol_params(kind: str, symbol: str, interval: str) -> dict[str, str]:
    if kind == "pair_exchange_interval":
        return {
            "exchange": "Binance",
            "symbol": EXCHANGE_SYMBOLS[symbol],
            "interval": interval,
            "limit": "1000",
        }
    if kind == "agg_exchange_interval":
        return {
            "exchange_list": "Binance",
            "symbol": symbol,
            "interval": interval,
            "limit": "1000",
        }
    if kind == "coin_interval":
        return {
            "symbol": symbol,
            "interval": interval,
            "limit": "1000",
        }
    raise ValueError(f"Unknown params kind: {kind}")


def parse_symbol_frame(parser: str, rows) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)

    if parser == "taker_pair":
        frame["ts"] = to_datetime_index(frame["time"])
        frame["buy"] = pd.to_numeric(
            frame["taker_buy_volume_usd"], errors="coerce")
        frame["sell"] = pd.to_numeric(
            frame["taker_sell_volume_usd"], errors="coerce")
        return frame.set_index("ts")[["buy", "sell"]].sort_index()

    if parser == "taker_agg":
        frame["ts"] = to_datetime_index(frame["time"])
        frame["buy"] = pd.to_numeric(
            frame["aggregated_buy_volume_usd"], errors="coerce")
        frame["sell"] = pd.to_numeric(
            frame["aggregated_sell_volume_usd"], errors="coerce")
        return frame.set_index("ts")[["buy", "sell"]].sort_index()

    if parser == "basis":
        frame["ts"] = to_datetime_index(frame["time"])
        for column in ["open_basis", "close_basis", "open_change", "close_change"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        columns = [column for column in ["open_basis", "close_basis",
                                         "open_change", "close_change"] if column in frame.columns]
        return frame.set_index("ts")[columns].sort_index()

    if parser == "coinbase_premium":
        frame["ts"] = to_datetime_index(frame["time"])
        for column in ["premium", "premium_rate", "coinbase_price"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        columns = [column for column in ["premium", "premium_rate",
                                         "coinbase_price"] if column in frame.columns]
        return frame.set_index("ts")[columns].sort_index()

    if parser == "ohlc":
        frame["ts"] = to_datetime_index(frame["time"])
        for column in ["open", "high", "low", "close"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.set_index("ts")[["open", "high", "low", "close"]].sort_index()

    if parser == "bitfinex_margin":
        frame["ts"] = to_datetime_index(frame["time"])
        for column in ["long_quantity", "short_quantity"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        columns = [column for column in ["long_quantity",
                                         "short_quantity"] if column in frame.columns]
        return frame.set_index("ts")[columns].sort_index()

    raise ValueError(f"Unknown parser: {parser}")


def parse_global_frame(parser: str, rows) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    if parser == "etf_btc":
        frame = pd.DataFrame(rows)
        frame["ts"] = to_datetime_index(frame["timestamp"])
        frame["etf_btc_flow"] = pd.to_numeric(
            frame["flow_usd"], errors="coerce")
        frame["etf_btc_price"] = pd.to_numeric(
            frame["price_usd"], errors="coerce")
        return frame.set_index("ts")[["etf_btc_flow", "etf_btc_price"]].sort_index()

    if parser == "etf_eth":
        frame = pd.DataFrame(rows)
        frame["ts"] = to_datetime_index(frame["timestamp"])
        frame["etf_eth_flow"] = pd.to_numeric(
            frame["flow_usd"], errors="coerce")
        frame["etf_eth_price"] = pd.to_numeric(
            frame["price_usd"], errors="coerce")
        return frame.set_index("ts")[["etf_eth_flow", "etf_eth_price"]].sort_index()

    if parser == "cgdi":
        frame = pd.DataFrame(rows)
        frame["ts"] = to_datetime_index(frame["time"])
        frame["cgdi"] = pd.to_numeric(
            frame["cgdi_index_value"], errors="coerce")
        return frame.set_index("ts")[["cgdi"]].sort_index()

    if parser == "series_list":
        if not isinstance(rows, dict) or "time_list" not in rows or "data_list" not in rows:
            return pd.DataFrame()
        ts = to_datetime_index(rows["time_list"])
        frame = pd.DataFrame(
            {"ts": ts, "value": pd.to_numeric(rows["data_list"], errors="coerce")})
        return frame.set_index("ts")[["value"]].sort_index()

    if parser == "snapshot_list":
        if isinstance(rows, dict):
            rows = [rows]
        return pd.DataFrame(rows)

    raise ValueError(f"Unknown global parser: {parser}")


def summarize_frame(scope: str, name: str, frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {"scope": scope, "name": name, "rows": 0, "start": pd.NaT, "end": pd.NaT}
    if isinstance(frame.index, pd.DatetimeIndex):
        return {
            "scope": scope,
            "name": name,
            "rows": len(frame),
            "start": frame.index.min(),
            "end": frame.index.max(),
        }
    return {"scope": scope, "name": name, "rows": len(frame), "start": pd.NaT, "end": pd.NaT}


def freeze_symbol_interval_endpoints(headers: dict[str, str]) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    rate_limit_sleep = get_rate_limit_sleep()

    for interval in INTERVALS:
        cache_payload: dict[str, pd.DataFrame] = {}
        log(f"\n=== Freeze v4 interval {interval} ===")
        for endpoint in SYMBOL_ENDPOINTS:
            symbols = endpoint.supported_symbols or tuple(SYMBOLS)
            for symbol in symbols:
                params = build_symbol_params(
                    endpoint.params_kind, symbol, interval)
                try:
                    rows = fetch_json(endpoint.path, headers, params)
                    frame = parse_symbol_frame(endpoint.parser, rows)
                except Exception as exc:
                    log(f"  {endpoint.name:<18} {symbol:<4} FAIL {exc}")
                    frame = pd.DataFrame()
                cache_key = f"{symbol}_{endpoint.cache_prefix}"
                cache_payload[cache_key] = frame

                write_timeseries_history(
                    frame=frame,
                    destination=RAW_HISTORY_ROOT /
                    "v4" / interval / f"{symbol}_{endpoint.cache_prefix}.csv",
                    metadata={
                        "api_version": "v4",
                        "scope": "symbol_interval",
                        "interval": interval,
                        "symbol": symbol,
                        "endpoint": endpoint.name,
                        "path": endpoint.path,
                    },
                )

                summary_rows.append(summarize_frame(
                    f"v4_{interval}", cache_key, frame))
                if frame.empty:
                    log(f"  {endpoint.name:<18} {symbol:<4} 0 rows")
                else:
                    log(
                        f"  {endpoint.name:<18} {symbol:<4} {len(frame):>4} rows  "
                        f"{frame.index.min().strftime('%Y-%m-%d %H:%M')} -> {frame.index.max().strftime('%Y-%m-%d %H:%M')}"
                    )
                time.sleep(rate_limit_sleep)

        interval_cache_path = cache_path(f"coinglass_v4_{interval}_cache.pkl")
        with open(interval_cache_path, "wb") as file_handle:
            pickle.dump(cache_payload, file_handle)
        log(f"Saved {interval_cache_path}")

        if interval == "4h":
            taker_payload = {key: value for key, value in cache_payload.items(
            ) if key.endswith("taker_pair") or key.endswith("taker_agg")}
            legacy_taker = {}
            for key, value in taker_payload.items():
                symbol, suffix = key.split("_", 1)
                legacy_key = f"{symbol}_{'pair' if suffix == 'taker_pair' else 'agg'}"
                legacy_taker[legacy_key] = value
            taker_4h_path = cache_path("taker_4h_cache.pkl")
            with open(taker_4h_path, "wb") as file_handle:
                pickle.dump(legacy_taker, file_handle)
            log(f"Saved {taker_4h_path}")

        if interval == "1d":
            taker_payload = {key: value for key, value in cache_payload.items(
            ) if key.endswith("taker_pair") or key.endswith("taker_agg")}
            legacy_taker = {}
            for key, value in taker_payload.items():
                symbol, suffix = key.split("_", 1)
                legacy_key = f"{symbol}_{'pair' if suffix == 'taker_pair' else 'agg'}"
                legacy_taker[legacy_key] = value
            taker_1d_path = cache_path("taker_1d_cache.pkl")
            with open(taker_1d_path, "wb") as file_handle:
                pickle.dump(legacy_taker, file_handle)
            log(f"Saved {taker_1d_path}")

            legacy_new = {}
            rename_map = {
                "basis": "basis",
                "coinbase_premium": "cb_premium",
                "oi_stablecoin": "oi_stable",
                "oi_coin_margin": "oi_coin",
                "bitfinex_margin": "bfx_margin",
                "fr_oi_weight": "fr_oiw",
                "fr_vol_weight": "fr_vw",
            }
            for key, value in cache_payload.items():
                symbol, suffix = key.split("_", 1)
                if suffix in rename_map:
                    legacy_new[f"{symbol}_{rename_map[suffix]}"] = value
            new_endpoints_1d_path = cache_path("new_endpoints_1d_cache.pkl")
            with open(new_endpoints_1d_path, "wb") as file_handle:
                pickle.dump(legacy_new, file_handle)
            log(f"Saved {new_endpoints_1d_path}")

    return pd.DataFrame(summary_rows)


def freeze_global_endpoints(headers: dict[str, str]) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    cache_payload: dict[str, pd.DataFrame] = {}
    rate_limit_sleep = get_rate_limit_sleep()

    log("\n=== Freeze v4 global historical endpoints ===")
    for endpoint in GLOBAL_ENDPOINTS:
        params = dict(endpoint.static_params or {})
        rows = fetch_json(endpoint.path, headers, params)
        frame = parse_global_frame(endpoint.parser, rows)
        cache_payload[endpoint.name] = frame

        write_timeseries_history(
            frame=frame,
            destination=RAW_HISTORY_ROOT /
            "v4_global" / f"{endpoint.name}.csv",
            metadata={
                "api_version": "v4",
                "scope": "global",
                "endpoint": endpoint.name,
                "path": endpoint.path,
            },
        )

        summary_rows.append(summarize_frame("v4_global", endpoint.name, frame))
        if frame.empty:
            log(f"  {endpoint.name:<18} 0 rows")
        else:
            log(
                f"  {endpoint.name:<18} {len(frame):>4} rows  "
                f"{frame.index.min().strftime('%Y-%m-%d %H:%M')} -> {frame.index.max().strftime('%Y-%m-%d %H:%M')}"
            )
        time.sleep(rate_limit_sleep)

    global_cache_path = cache_path("coinglass_v4_global_cache.pkl")
    with open(global_cache_path, "wb") as file_handle:
        pickle.dump(cache_payload, file_handle)
    log(f"Saved {global_cache_path}")

    legacy_global = {
        "global_etf_btc": cache_payload.get("etf_btc", pd.DataFrame()),
        "global_etf_eth": cache_payload.get("etf_eth", pd.DataFrame()),
        "global_fg": cache_payload.get("fear_greed", pd.DataFrame()).rename(columns={"value": "fear_greed"}),
        "global_cgdi": cache_payload.get("cgdi", pd.DataFrame()),
        "global_stable_mcap": cache_payload.get("stablecoin_mcap", pd.DataFrame()).rename(columns={"value": "stable_mcap"}),
    }
    new_endpoints_global_path = cache_path("new_endpoints_global_cache.pkl")
    with open(new_endpoints_global_path, "wb") as file_handle:
        pickle.dump(legacy_global, file_handle)
    log(f"Saved {new_endpoints_global_path}")

    return pd.DataFrame(summary_rows)


def freeze_snapshots(headers: dict[str, str]) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    snapshot_payload: dict[str, pd.DataFrame] = {}
    rate_limit_sleep = get_rate_limit_sleep()

    log("\n=== Freeze v4 snapshot endpoints ===")
    for endpoint in SNAPSHOT_ENDPOINTS:
        if endpoint.name == "oi_exchange":
            for symbol in SYMBOLS:
                params = {"symbol": symbol}
                rows = fetch_json(endpoint.path, headers, params)
                frame = parse_global_frame(endpoint.parser, rows)
                name = f"{symbol}_oi_exchange"
                snapshot_payload[name] = frame

                append_snapshot_history(
                    frame=frame,
                    destination=RAW_HISTORY_ROOT /
                    "v4_snapshots" / f"{name}.csv",
                    metadata={
                        "api_version": "v4",
                        "scope": "snapshot",
                        "symbol": symbol,
                        "endpoint": endpoint.name,
                        "path": endpoint.path,
                    },
                )

                summary_rows.append(summarize_frame(
                    "v4_snapshot", name, frame))
                log(f"  {name:<18} {len(frame):>4} rows")
                time.sleep(rate_limit_sleep)
            continue

        if endpoint.name == "option_maxpain":
            for symbol in ("BTC", "ETH"):
                params = dict(endpoint.static_params or {})
                params["symbol"] = symbol
                rows = fetch_json(endpoint.path, headers, params)
                frame = parse_global_frame(endpoint.parser, rows)
                name = f"{symbol}_option_maxpain"
                snapshot_payload[name] = frame

                append_snapshot_history(
                    frame=frame,
                    destination=RAW_HISTORY_ROOT /
                    "v4_snapshots" / f"{name}.csv",
                    metadata={
                        "api_version": "v4",
                        "scope": "snapshot",
                        "symbol": symbol,
                        "endpoint": endpoint.name,
                        "path": endpoint.path,
                    },
                )

                summary_rows.append(summarize_frame(
                    "v4_snapshot", name, frame))
                log(f"  {name:<18} {len(frame):>4} rows")
                time.sleep(rate_limit_sleep)

    snapshot_cache_path = cache_path("coinglass_v4_snapshots.pkl")
    with open(snapshot_cache_path, "wb") as file_handle:
        pickle.dump(snapshot_payload, file_handle)
    log(f"Saved {snapshot_cache_path}")

    return pd.DataFrame(summary_rows)


def main() -> None:
    ensure_data_dirs()
    RAW_HISTORY_ROOT.mkdir(parents=True, exist_ok=True)
    headers = {"accept": "application/json", "CG-API-KEY": load_api_key()}

    summary_frames = [
        freeze_symbol_interval_endpoints(headers),
        freeze_global_endpoints(headers),
        freeze_snapshots(headers),
    ]
    summary = pd.concat(summary_frames, ignore_index=True)
    summary_path = manifest_path("coinglass_v4_cache_summary.csv")
    summary.to_csv(summary_path, index=False)
    log(f"\nSaved {summary_path}")

    available = summary[summary["rows"] > 0].copy()
    log("\n=== Available Manifest ===")
    print(available.to_string(index=False))


if __name__ == "__main__":
    main()
