from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from qlab.data.crypto.binance_um_klines import (
    BinanceKlinePartition,
    audit_klines,
    download_day_partition,
    download_month_partition,
    partition_path,
    read_partitions,
    write_source_manifest,
)
from qlab.data.crypto.paths import CACHE_DIR, MANIFEST_DIR, ensure_data_dirs
from qlab.data.crypto.symbol_universe import normalize_symbol_list


CANONICAL_SYMBOLS_20 = [
    "ADA", "APT", "AVAX", "BCH", "BNB", "BTC", "DOGE", "DOT", "ETC", "ETH",
    "FET", "FIL", "LINK", "LTC", "NEAR", "SOL", "SUI", "TRX", "UNI", "XRP",
]


def build_session() -> requests.Session:
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "qlab-binance-um-1m/1.0"})
    return session


def completed_periods(start: pd.Timestamp, end: pd.Timestamp) -> tuple[list[str], list[str]]:
    start = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo else pd.Timestamp(start).tz_localize("UTC")
    end = pd.Timestamp(end).tz_convert("UTC") if pd.Timestamp(end).tzinfo else pd.Timestamp(end).tz_localize("UTC")
    today = pd.Timestamp.now(tz="UTC").normalize()
    last_complete_day = min(end.normalize(), today - pd.Timedelta(days=1))
    if last_complete_day < start.normalize():
        return [], []
    current_month = today.replace(day=1)
    month_end = min(last_complete_day, current_month - pd.Timedelta(days=1))
    months = []
    if month_end >= start.normalize():
        months = [period.strftime("%Y-%m") for period in pd.date_range(start.normalize().replace(day=1), month_end, freq="MS")]
    daily_start = max(start.normalize(), current_month)
    days = [] if daily_start > last_complete_day else [
        period.strftime("%Y-%m-%d")
        for period in pd.date_range(daily_start, last_complete_day, freq="D")
    ]
    return months, days


def refresh(
    *,
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    root: Path,
    sleep_seconds: float = 0.1,
    refresh_existing: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    session = build_session()
    months, days = completed_periods(start, end)
    downloaded: list[BinanceKlinePartition] = []
    coverage: list[dict[str, object]] = []
    for symbol in normalize_symbol_list(symbols):
        for period in months:
            path = partition_path(symbol, "1m", period, root=root)
            if refresh_existing or not path.exists():
                downloaded.append(download_month_partition(session, symbol=symbol, interval="1m", period=period, root=root))
                time.sleep(sleep_seconds)
        for period in days:
            path = partition_path(symbol, "1m", period, root=root)
            if refresh_existing or not path.exists():
                downloaded.append(download_day_partition(session, symbol=symbol, interval="1m", period=period, root=root))
                time.sleep(sleep_seconds)
        paths = sorted((root / "interval=1m" / f"symbol={symbol}USDT").glob("period=*.parquet"))
        frame = read_partitions(paths, interval="1m")
        row = audit_klines(frame, symbol=symbol, interval="1m")
        requested_start = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo else pd.Timestamp(start).tz_localize("UTC")
        requested_end = pd.Timestamp(end).tz_convert("UTC") if pd.Timestamp(end).tzinfo else pd.Timestamp(end).tz_localize("UTC")
        row["requested_start"] = requested_start
        row["requested_end"] = requested_end
        row["covers_requested_start"] = bool(row["rows"] and row["start_open_time"] <= requested_start)
        row["covers_requested_end"] = bool(row["rows"] and row["end_open_time"] >= requested_end.floor("D"))
        coverage.append(row)
    source_frame = pd.DataFrame([partition.__dict__ for partition in downloaded])
    return source_frame, pd.DataFrame(coverage)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh partitioned Binance USD-M 1m klines")
    parser.add_argument("--symbols", default=",".join(CANONICAL_SYMBOLS_20))
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", default=str(pd.Timestamp.now(tz="UTC").date()))
    parser.add_argument("--root", type=Path, default=CACHE_DIR / "binance_um_klines")
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--refresh-existing", action="store_true")
    args = parser.parse_args()
    ensure_data_dirs()
    source, coverage = refresh(
        symbols=args.symbols.split(","),
        start=pd.Timestamp(args.start),
        end=pd.Timestamp(args.end),
        root=args.root,
        sleep_seconds=args.sleep_seconds,
        refresh_existing=args.refresh_existing,
    )
    source_path = MANIFEST_DIR / "binance_um_1m_sources_latest.csv"
    coverage_path = MANIFEST_DIR / "binance_um_1m_coverage_latest.csv"
    if source_path.exists():
        existing_source = pd.read_csv(source_path)
        source = pd.concat([existing_source, source], ignore_index=True)
        source = source.drop_duplicates(["symbol", "interval", "period"], keep="last")
        source = source.sort_values(["symbol", "interval", "period"]).reset_index(drop=True)
    source.to_csv(source_path, index=False)
    if coverage_path.exists():
        existing_coverage = pd.read_csv(coverage_path)
        coverage = pd.concat([existing_coverage, coverage], ignore_index=True)
        coverage = coverage.drop_duplicates(["symbol", "interval"], keep="last")
        coverage = coverage.sort_values(["symbol", "interval"]).reset_index(drop=True)
    coverage.to_csv(coverage_path, index=False)
    print(source_path)
    print(coverage_path)
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
