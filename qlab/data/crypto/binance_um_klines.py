from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from qlab.data.crypto.paths import CACHE_DIR, MANIFEST_DIR, ensure_data_dirs


DATA_VISION_BASE = "https://data.binance.vision/data/futures/um"
USD_M_REST_BASE = "https://fapi.binance.com"
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "source",
]
NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class BinanceKlinePartition:
    symbol: str
    interval: str
    period: str
    path: Path
    source_url: str
    source_sha256: str
    rows: int
    start_open_time: pd.Timestamp | None
    end_open_time: pd.Timestamp | None


def interval_delta(interval: str) -> pd.Timedelta:
    try:
        value = int(interval[:-1])
        unit = interval[-1]
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"Unsupported Binance interval: {interval!r}") from exc
    units = {"m": "minutes", "h": "hours", "d": "days"}
    if value <= 0 or unit not in units:
        raise ValueError(f"Unsupported Binance interval: {interval!r}")
    return pd.Timedelta(**{units[unit]: value})


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=KLINE_COLUMNS)


def normalize_klines(frame: pd.DataFrame, interval: str) -> pd.DataFrame:
    if frame.empty:
        return _empty_frame()
    missing = set(KLINE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Kline frame missing columns: {sorted(missing)}")
    normalized = frame.loc[:, KLINE_COLUMNS].copy()
    for column in ["open_time", "close_time"]:
        normalized[column] = pd.to_datetime(normalized[column], utc=True, errors="coerce")
    for column in NUMERIC_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if normalized[["open_time", "close_time", *NUMERIC_COLUMNS]].isna().any().any():
        raise ValueError("Kline frame contains null or non-numeric required values")
    normalized["source"] = normalized["source"].astype(str)
    normalized = normalized.sort_values("open_time", kind="stable").reset_index(drop=True)
    if normalized["open_time"].duplicated().any():
        duplicates = normalized.loc[normalized["open_time"].duplicated(), "open_time"]
        raise ValueError(f"Duplicate Binance open_time values: {duplicates.iloc[0]}")
    delta = interval_delta(interval)
    expected_close = normalized["open_time"] + delta - pd.Timedelta(milliseconds=1)
    if (normalized["close_time"] != expected_close).any():
        raise ValueError("Binance close_time is inconsistent with open_time and interval")
    invalid_ohlc = (
        (normalized["high"] < normalized[["open", "close", "low"]].max(axis=1))
        | (normalized["low"] > normalized[["open", "close", "high"]].min(axis=1))
        | (normalized["volume"] < 0)
    )
    if invalid_ohlc.any():
        raise ValueError("Binance kline contains invalid OHLCV values")
    return normalized


def parse_data_vision_zip(content: bytes, *, interval: str, source: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if len(names) != 1:
            raise ValueError(f"Expected one file in Binance archive, got {len(names)}")
        raw = pd.read_csv(archive.open(names[0]), header=None)
    if raw.empty:
        return _empty_frame()
    if str(raw.iloc[0, 0]).strip().lower() in {"open_time", "timestamp"}:
        raw = raw.iloc[1:].reset_index(drop=True)
    if raw.shape[1] < 7:
        raise ValueError(f"Binance archive has only {raw.shape[1]} columns")
    parsed = pd.DataFrame(
        {
            "open_time": pd.to_datetime(pd.to_numeric(raw.iloc[:, 0]), unit="ms", utc=True),
            "open": raw.iloc[:, 1],
            "high": raw.iloc[:, 2],
            "low": raw.iloc[:, 3],
            "close": raw.iloc[:, 4],
            "volume": raw.iloc[:, 5],
            "close_time": pd.to_datetime(pd.to_numeric(raw.iloc[:, 6]), unit="ms", utc=True),
            "source": source,
        }
    )
    return normalize_klines(parsed, interval)


def parse_rest_klines(
    payload: object,
    *,
    interval: str,
    source: str,
) -> pd.DataFrame:
    if not isinstance(payload, list):
        raise ValueError("Binance REST kline response must be a list")
    if not payload:
        return _empty_frame()
    if any(not isinstance(row, list) or len(row) < 7 for row in payload):
        raise ValueError("Binance REST kline response contains an invalid row")
    parsed = pd.DataFrame(
        {
            "open_time": pd.to_datetime(
                [int(row[0]) for row in payload], unit="ms", utc=True
            ),
            "open": [row[1] for row in payload],
            "high": [row[2] for row in payload],
            "low": [row[3] for row in payload],
            "close": [row[4] for row in payload],
            "volume": [row[5] for row in payload],
            "close_time": pd.to_datetime(
                [int(row[6]) for row in payload], unit="ms", utc=True
            ),
            "source": source,
        }
    )
    return normalize_klines(parsed, interval)


def partition_path(
    symbol: str,
    interval: str,
    period: str,
    *,
    root: Path | None = None,
) -> Path:
    base = root if root is not None else CACHE_DIR / "binance_um_klines"
    return base / f"interval={interval}" / f"symbol={symbol.upper()}USDT" / f"period={period}.parquet"


def canonical_partition_paths(paths: Iterable[Path]) -> list[Path]:
    """Prefer a complete monthly archive over daily tail files for that month."""
    candidates = sorted(Path(path) for path in paths)
    monthly_periods = {
        path.stem.removeprefix("period=")
        for path in candidates
        if len(path.stem.removeprefix("period=")) == 7
    }
    return [
        path
        for path in candidates
        if len(path.stem.removeprefix("period=")) == 7
        or path.stem.removeprefix("period=")[:7] not in monthly_periods
    ]


def write_partition(frame: pd.DataFrame, path: Path, *, interval: str) -> None:
    normalized = normalize_klines(frame, interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    normalized.to_parquet(temporary, index=False)
    temporary.replace(path)


def read_partitions(paths: Iterable[Path], *, interval: str) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in canonical_partition_paths(paths)]
    if not frames:
        return _empty_frame()
    return normalize_klines(pd.concat(frames, ignore_index=True), interval)


def aggregate_complete_klines(
    frame: pd.DataFrame,
    *,
    source_interval: str,
    target_interval: str,
) -> pd.DataFrame:
    """Aggregate only complete, UTC-aligned source-bar groups."""
    source = interval_delta(source_interval)
    target = interval_delta(target_interval)
    if target <= source or target % source != pd.Timedelta(0):
        raise ValueError("target_interval must be an integer multiple above source_interval")
    normalized = normalize_klines(frame, source_interval)
    if normalized.empty:
        return _empty_frame()
    expected_rows = int(target / source)
    working = normalized.assign(
        __bucket=pd.DatetimeIndex(normalized["open_time"]).floor(target)
    )
    grouped = working.groupby("__bucket", sort=True)
    aggregated = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        source_rows=("open_time", "size"),
        first_open_time=("open_time", "first"),
        last_open_time=("open_time", "last"),
    )
    complete = (
        aggregated["source_rows"].eq(expected_rows)
        & aggregated["first_open_time"].eq(aggregated.index)
        & aggregated["last_open_time"].eq(aggregated.index + target - source)
    )
    aggregated = aggregated.loc[complete].drop(
        columns=["source_rows", "first_open_time", "last_open_time"]
    )
    if aggregated.empty:
        return _empty_frame()
    aggregated = aggregated.reset_index(names="open_time")
    aggregated["close_time"] = (
        aggregated["open_time"] + target - pd.Timedelta(milliseconds=1)
    )
    aggregated["source"] = "aggregated_complete_" + source_interval
    return normalize_klines(aggregated, target_interval)


def price_volume_payload_from_klines(
    frame: pd.DataFrame,
    *,
    source_interval: str = "1m",
    target_interval: str = "15m",
) -> pd.DataFrame:
    """Return an open-indexed close/volume payload for price-volume features."""
    aggregated = aggregate_complete_klines(
        frame,
        source_interval=source_interval,
        target_interval=target_interval,
    )
    if aggregated.empty:
        raise ValueError("no complete target klines available")
    payload = aggregated.set_index("open_time")[["close", "volume"]].rename(
        columns={"close": "c", "volume": "v"}
    )
    payload.index.name = "ts"
    return payload


def audit_klines(frame: pd.DataFrame, *, symbol: str, interval: str) -> dict[str, object]:
    normalized = normalize_klines(frame, interval)
    if normalized.empty:
        return {
            "symbol": symbol.upper(),
            "interval": interval,
            "rows": 0,
            "start_open_time": pd.NaT,
            "end_open_time": pd.NaT,
            "missing_bars": 0,
            "duplicate_open_times": 0,
            "schema_ok": True,
            "ohlcv_ok": True,
        }
    expected = pd.date_range(
        normalized["open_time"].iloc[0],
        normalized["open_time"].iloc[-1],
        freq=interval_delta(interval),
    )
    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "rows": int(len(normalized)),
        "start_open_time": normalized["open_time"].iloc[0],
        "end_open_time": normalized["open_time"].iloc[-1],
        "missing_bars": int(len(expected.difference(pd.DatetimeIndex(normalized["open_time"])))),
        "duplicate_open_times": 0,
        "schema_ok": True,
        "ohlcv_ok": True,
    }


def execution_opens(frame: pd.DataFrame | pd.Series, timestamps: Iterable[pd.Timestamp]) -> pd.Series:
    requested = pd.DatetimeIndex(pd.to_datetime(list(timestamps), utc=True), name="execution_ts")
    if requested.has_duplicates:
        raise ValueError("Requested execution timestamps contain duplicates")
    if isinstance(frame, pd.Series):
        indexed = frame.copy()
        indexed.index = pd.to_datetime(indexed.index, utc=True)
        indexed = pd.to_numeric(indexed, errors="coerce")
    elif {"open_time", "open"}.issubset(frame.columns) and not set(KLINE_COLUMNS).issubset(frame.columns):
        indexed = pd.Series(
            pd.to_numeric(frame["open"], errors="coerce").to_numpy(),
            index=pd.DatetimeIndex(pd.to_datetime(frame["open_time"], utc=True)),
        )
    else:
        normalized = normalize_klines(frame, "1m")
        indexed = normalized.set_index("open_time")["open"]
    if indexed.index.has_duplicates:
        raise ValueError("Binance execution-open source contains duplicate open_time values")
    if indexed.isna().any():
        raise ValueError("Binance execution-open source contains non-numeric values")
    missing = requested.difference(indexed.index)
    if len(missing):
        raise KeyError(f"Missing Binance 1m open for execution timestamp {missing[0]}")
    result = indexed.reindex(requested)
    result.index.name = "execution_ts"
    result.name = "execution_open"
    return result


def read_execution_open_partitions(paths: Iterable[Path]) -> pd.Series:
    frames = [
        pd.read_parquet(path, columns=["open_time", "open"])
        for path in canonical_partition_paths(paths)
    ]
    if not frames:
        return pd.Series(dtype=float, name="open")
    combined = pd.concat(frames, ignore_index=True)
    opens = pd.Series(
        pd.to_numeric(combined["open"], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(pd.to_datetime(combined["open_time"], utc=True)),
        name="open",
    ).sort_index()
    if opens.index.has_duplicates:
        raise ValueError("Binance execution-open partitions contain duplicate open_time values")
    if opens.isna().any():
        raise ValueError("Binance execution-open partitions contain non-numeric open values")
    return opens


def monthly_url(symbol: str, interval: str, period: str) -> str:
    pair = f"{symbol.upper()}USDT"
    return f"{DATA_VISION_BASE}/monthly/klines/{pair}/{interval}/{pair}-{interval}-{period}.zip"


def daily_url(symbol: str, interval: str, period: str) -> str:
    pair = f"{symbol.upper()}USDT"
    return f"{DATA_VISION_BASE}/daily/klines/{pair}/{interval}/{pair}-{interval}-{period}.zip"


def download_month_partition(
    session: requests.Session,
    *,
    symbol: str,
    interval: str,
    period: str,
    root: Path | None = None,
    timeout: float = 60.0,
) -> BinanceKlinePartition:
    ensure_data_dirs()
    url = monthly_url(symbol, interval, period)
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    checksum = hashlib.sha256(response.content).hexdigest()
    frame = parse_data_vision_zip(response.content, interval=interval, source=url)
    path = partition_path(symbol, interval, period, root=root)
    write_partition(frame, path, interval=interval)
    return BinanceKlinePartition(
        symbol=symbol.upper(),
        interval=interval,
        period=period,
        path=path,
        source_url=url,
        source_sha256=checksum,
        rows=int(len(frame)),
        start_open_time=None if frame.empty else frame["open_time"].iloc[0],
        end_open_time=None if frame.empty else frame["open_time"].iloc[-1],
    )


def download_day_partition(
    session: requests.Session,
    *,
    symbol: str,
    interval: str,
    period: str,
    root: Path | None = None,
    timeout: float = 60.0,
) -> BinanceKlinePartition:
    ensure_data_dirs()
    url = daily_url(symbol, interval, period)
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    checksum = hashlib.sha256(response.content).hexdigest()
    frame = parse_data_vision_zip(response.content, interval=interval, source=url)
    path = partition_path(symbol, interval, period, root=root)
    write_partition(frame, path, interval=interval)
    return BinanceKlinePartition(
        symbol=symbol.upper(),
        interval=interval,
        period=period,
        path=path,
        source_url=url,
        source_sha256=checksum,
        rows=int(len(frame)),
        start_open_time=None if frame.empty else frame["open_time"].iloc[0],
        end_open_time=None if frame.empty else frame["open_time"].iloc[-1],
    )


def download_rest_day_partition(
    session: requests.Session,
    *,
    symbol: str,
    interval: str,
    period: str,
    root: Path | None = None,
    base_url: str = USD_M_REST_BASE,
    timeout: float = 60.0,
) -> BinanceKlinePartition:
    day_start = pd.Timestamp(period, tz="UTC")
    day_end = day_start + pd.Timedelta(days=1)
    delta = interval_delta(interval)
    expected_rows = int(pd.Timedelta(days=1) / delta)
    if expected_rows > 1500:
        raise ValueError(
            f"One-day Binance REST refresh exceeds the 1500-row limit for {interval}"
        )
    endpoint = f"{base_url.rstrip('/')}/fapi/v1/klines"
    params = {
        "symbol": f"{symbol.upper()}USDT",
        "interval": interval,
        "startTime": int(day_start.timestamp() * 1000),
        "endTime": int((day_end - pd.Timedelta(milliseconds=1)).timestamp() * 1000),
        "limit": expected_rows,
    }
    response = session.get(endpoint, params=params, timeout=timeout)
    response.raise_for_status()
    try:
        payload = response.json()
    except requests.JSONDecodeError as exc:
        raise ValueError("Binance REST kline response is not valid JSON") from exc
    source = f"{endpoint}?{requests.compat.urlencode(params)}"
    frame = parse_rest_klines(payload, interval=interval, source=source)
    expected_opens = pd.date_range(
        day_start,
        day_end - delta,
        freq=delta,
    )
    actual_opens = pd.DatetimeIndex(frame["open_time"])
    if len(frame) != expected_rows or not actual_opens.equals(expected_opens):
        raise ValueError(
            f"Binance REST day is incomplete for {symbol.upper()} {period}: "
            f"expected {expected_rows} rows, got {len(frame)}"
        )
    raw_payload = response.content or json.dumps(
        payload, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    checksum = hashlib.sha256(raw_payload).hexdigest()
    path = partition_path(symbol, interval, period, root=root)
    write_partition(frame, path, interval=interval)
    return BinanceKlinePartition(
        symbol=symbol.upper(),
        interval=interval,
        period=period,
        path=path,
        source_url=source,
        source_sha256=checksum,
        rows=int(len(frame)),
        start_open_time=frame["open_time"].iloc[0],
        end_open_time=frame["open_time"].iloc[-1],
    )


def write_source_manifest(partitions: Iterable[BinanceKlinePartition], path: Path | None = None) -> Path:
    destination = path if path is not None else MANIFEST_DIR / "binance_um_1m_sources.csv"
    rows = [partition.__dict__ for partition in partitions]
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
    return destination
