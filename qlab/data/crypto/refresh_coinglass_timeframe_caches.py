from __future__ import annotations

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
from qlab.data.crypto.raw_history_store import RAW_HISTORY_ROOT, write_timeseries_history
from qlab.data.crypto.symbol_universe import RESEARCH_SYMBOLS_12

BASE = "https://open-api-v3.coinglass.com/api"
ENV_PATH = TRADE_ENV_PATH
SYMBOLS = RESEARCH_SYMBOLS_12
EXCHANGE_SYMBOLS = {symbol: f"{symbol}USDT" for symbol in SYMBOLS}
RATE_LIMIT_SLEEP = 2.3


@dataclass(frozen=True)
class EndpointSpec:
    name: str
    path: str
    requires_exchange: bool
    is_ohlc: bool
    symbol_uses_exchange_pair: bool


@dataclass(frozen=True)
class IntervalSpec:
    name: str
    limit: int
    cache_name: str
    alias_cache_name: str | None = None


ENDPOINTS = [
    EndpointSpec(
        name="oi",
        path="/futures/openInterest/ohlc-aggregated-history",
        requires_exchange=False,
        is_ohlc=True,
        symbol_uses_exchange_pair=False,
    ),
    EndpointSpec(
        name="liq",
        path="/futures/liquidation/aggregated-history",
        requires_exchange=False,
        is_ohlc=False,
        symbol_uses_exchange_pair=False,
    ),
    EndpointSpec(
        name="global_ls",
        path="/futures/globalLongShortAccountRatio/history",
        requires_exchange=True,
        is_ohlc=False,
        symbol_uses_exchange_pair=True,
    ),
    EndpointSpec(
        name="top_acct",
        path="/futures/topLongShortAccountRatio/history",
        requires_exchange=True,
        is_ohlc=False,
        symbol_uses_exchange_pair=True,
    ),
    EndpointSpec(
        name="top_pos",
        path="/futures/topLongShortPositionRatio/history",
        requires_exchange=True,
        is_ohlc=False,
        symbol_uses_exchange_pair=True,
    ),
    EndpointSpec(
        name="fr",
        path="/futures/fundingRate/ohlc-history",
        requires_exchange=True,
        is_ohlc=True,
        symbol_uses_exchange_pair=True,
    ),
]

INTERVALS = [
    IntervalSpec(name="4h", limit=1100, cache_name="coinglass_4h_cache.pkl",
                 alias_cache_name="coinglass_180d_cache.pkl"),
    IntervalSpec(name="6h", limit=1100, cache_name="coinglass_6h_cache.pkl"),
    IntervalSpec(name="12h", limit=1100, cache_name="coinglass_12h_cache.pkl"),
    IntervalSpec(name="1d", limit=1000,
                 cache_name="coinglass_daily_cache.pkl"),
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


def fetch_json(url: str, headers: dict[str, str], retries: int = 4) -> list[dict]:
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            payload = response.json()
        except Exception as exc:
            last_error = exc
            time.sleep(3 * (attempt + 1))
            continue

        if response.status_code == 429 or payload.get("code") == "429":
            time.sleep(10)
            continue

        if response.status_code == 200 and payload.get("code", "0") == "0":
            return payload.get("data", [])

        last_error = RuntimeError(
            f"HTTP={response.status_code}, code={payload.get('code')}, msg={payload.get('msg')}")
        time.sleep(3 * (attempt + 1))

    raise RuntimeError(f"CoinGlass request failed for {url}: {last_error}")


def build_url(interval: IntervalSpec, endpoint: EndpointSpec, symbol: str) -> str:
    query_symbol = EXCHANGE_SYMBOLS[symbol] if endpoint.symbol_uses_exchange_pair else symbol
    params = [f"symbol={query_symbol}",
              f"interval={interval.name}", f"limit={interval.limit}"]
    if endpoint.requires_exchange:
        params.insert(0, "exchange=Binance")
    return f"{BASE}{endpoint.path}?{'&'.join(params)}"


def parse_frame(endpoint: EndpointSpec, rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    if endpoint.name == "oi":
        frame["ts"] = pd.to_datetime(frame["t"], unit="s", utc=True)
        for column in ["o", "h", "l", "c"]:
            frame[column] = frame[column].astype(float)
        frame = frame.rename(
            columns={"o": "oi_open", "h": "oi_high", "l": "oi_low", "c": "oi_close"})
        return frame.set_index("ts")[["oi_open", "oi_high", "oi_low", "oi_close"]].sort_index()

    if endpoint.name == "liq":
        frame["ts"] = pd.to_datetime(frame["t"], unit="s", utc=True)
        frame["long_liq"] = frame["longLiquidationUsd"].astype(float)
        frame["short_liq"] = frame["shortLiquidationUsd"].astype(float)
        frame["net_liq"] = frame["long_liq"] - frame["short_liq"]
        frame["total_liq"] = frame["long_liq"] + frame["short_liq"]
        return frame.set_index("ts")[["long_liq", "short_liq", "net_liq", "total_liq"]].sort_index()

    if endpoint.name == "fr":
        frame["ts"] = pd.to_datetime(frame["t"], unit="s", utc=True)
        frame["fr_close"] = frame["c"].astype(float)
        return frame.set_index("ts")[["fr_close"]].sort_index()

    frame["ts"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    ratio_column = {
        "global_ls": "global_ls_ratio",
        "top_acct": "top_acct_ls_ratio",
        "top_pos": "top_pos_ls_ratio",
    }[endpoint.name]
    long_column = {
        "global_ls": "global_long_pct",
        "top_acct": "top_acct_long_pct",
        "top_pos": "top_pos_long_pct",
    }[endpoint.name]
    frame[ratio_column] = frame["longShortRatio"].astype(float)
    frame[long_column] = frame["longAccount"].astype(float)
    return frame.set_index("ts")[[ratio_column, long_column]].sort_index()


def refresh_interval(interval: IntervalSpec, headers: dict[str, str]) -> pd.DataFrame:
    log(f"\n=== Refresh {interval.name} ===")
    cache_payload: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []

    for symbol in SYMBOLS:
        log(f"--- {symbol} @ {interval.name} ---")
        for endpoint in ENDPOINTS:
            url = build_url(interval, endpoint, symbol)
            rows = fetch_json(url, headers)
            frame = parse_frame(endpoint, rows)
            cache_key = f"{symbol}_{endpoint.name}"
            cache_payload[cache_key] = frame

            write_timeseries_history(
                frame=frame,
                destination=RAW_HISTORY_ROOT /
                "v3" / interval.name / f"{symbol}_{endpoint.name}.csv",
                metadata={
                    "api_version": "v3",
                    "interval": interval.name,
                    "symbol": symbol,
                    "endpoint": endpoint.name,
                    "path": endpoint.path,
                },
            )

            if frame.empty:
                summary_rows.append({
                    "interval": interval.name,
                    "symbol": symbol,
                    "endpoint": endpoint.name,
                    "rows": 0,
                    "start": pd.NaT,
                    "end": pd.NaT,
                })
                log(f"  {endpoint.name:<10} 0 rows")
            else:
                summary_rows.append({
                    "interval": interval.name,
                    "symbol": symbol,
                    "endpoint": endpoint.name,
                    "rows": len(frame),
                    "start": frame.index.min(),
                    "end": frame.index.max(),
                })
                log(
                    f"  {endpoint.name:<10} {len(frame):>4} rows  "
                    f"{frame.index.min().strftime('%Y-%m-%d %H:%M')} -> {frame.index.max().strftime('%Y-%m-%d %H:%M')}"
                )

            time.sleep(RATE_LIMIT_SLEEP)

    interval_cache_path = cache_path(interval.cache_name)
    with open(interval_cache_path, "wb") as file_handle:
        pickle.dump(cache_payload, file_handle)
    log(f"Saved {interval_cache_path}")

    if interval.alias_cache_name:
        alias_path = cache_path(interval.alias_cache_name)
        with open(alias_path, "wb") as file_handle:
            pickle.dump(cache_payload, file_handle)
        log(f"Saved {alias_path}")

    return pd.DataFrame(summary_rows)


def main() -> None:
    ensure_data_dirs()
    RAW_HISTORY_ROOT.mkdir(parents=True, exist_ok=True)
    headers = {"accept": "application/json", "CG-API-KEY": load_api_key()}

    summary_frames = [refresh_interval(interval, headers)
                      for interval in INTERVALS]
    summary = pd.concat(summary_frames, ignore_index=True)
    summary_path = manifest_path("coinglass_cache_summary.csv")
    summary.to_csv(summary_path, index=False)
    log(f"\nSaved {summary_path}")

    manifest = (
        summary.groupby(["interval", "endpoint"], dropna=False)
        .agg(rows=("rows", "sum"), earliest_start=("start", "min"), latest_end=("end", "max"))
        .reset_index()
        .sort_values(["interval", "endpoint"])
    )
    log("\n=== Manifest ===")
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
