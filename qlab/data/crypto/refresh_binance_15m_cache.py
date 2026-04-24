from __future__ import annotations

import os
import io
import pickle
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

if __package__ in (None, ""):
    PACKAGE_ROOT = Path(__file__).resolve().parents[3]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))

from qlab.data.crypto.paths import TRADE_ENV_PATH, cache_path, ensure_data_dirs, manifest_path
from qlab.data.crypto.symbol_universe import RESEARCH_SYMBOLS_12

DEFAULT_BINANCE_FAPI_BASE_URLS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
]
DATA_VISION_BASE = "https://data.binance.vision/data/futures/um"
CACHE_PATH = cache_path("crypto_15m_cache.pkl")
SUMMARY_PATH = manifest_path("crypto_15m_cache_summary.csv")
ENV_PATH = TRADE_ENV_PATH
SYMBOLS = RESEARCH_SYMBOLS_12
INTERVAL = "15m"
PANDAS_INTERVAL = "15min"
BAR_MS = 15 * 60 * 1000
REQUEST_LIMIT = 1500
REQUEST_SLEEP = 0.2
START_TS = pd.Timestamp("2023-07-01 00:00:00", tz="UTC")
OHLCV_COLUMNS = ["o", "h", "l", "c", "v"]
HTTP_RETRY_ATTEMPTS = 4


def log(message: str) -> None:
    print(message, flush=True)


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def get_env_value(env: dict[str, str], key: str) -> str:
    return os.environ.get(key, env.get(key, "")).strip()


def get_csv_env(env: dict[str, str], key: str, default: list[str]) -> list[str]:
    raw_value = get_env_value(env, key)
    if not raw_value:
        return list(default)
    values = [item.strip().rstrip("/")
              for item in raw_value.split(",") if item.strip()]
    return values or list(default)


def get_int_env(env: dict[str, str], key: str, default: int) -> int:
    raw_value = get_env_value(env, key)
    if not raw_value:
        return default
    try:
        return max(1, int(raw_value))
    except ValueError:
        return default


def get_bool_env(env: dict[str, str], key: str, default: bool) -> bool:
    raw_value = get_env_value(env, key)
    if not raw_value:
        return default
    return raw_value.lower() in {"1", "true", "yes", "y", "on"}


def get_symbols(env: dict[str, str]) -> list[str]:
    raw_value = get_env_value(env, "BINANCE_SYMBOLS")
    if not raw_value:
        return list(SYMBOLS)
    values = [item.strip().upper()
              for item in raw_value.split(",") if item.strip()]
    return values or list(SYMBOLS)


def build_http_session(proxy_url: str = "") -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=HTTP_RETRY_ATTEMPTS,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    if proxy_url:
        session.proxies = {"http": proxy_url, "https": proxy_url}
    return session


def utc_floor_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor(PANDAS_INTERVAL)


def to_utc_naive(index_like: pd.Index) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(index_like)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    return index


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=OHLCV_COLUMNS, index=pd.DatetimeIndex([], name="ts"))


def normalize_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return empty_frame()

    normalized = frame.copy()
    normalized.index = to_utc_naive(normalized.index)
    normalized.index.name = "ts"
    for column in OHLCV_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized[OHLCV_COLUMNS].sort_index()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    return normalized


def load_existing_cache() -> dict[str, pd.DataFrame]:
    if not CACHE_PATH.exists():
        return {}
    with open(CACHE_PATH, "rb") as file_handle:
        raw = pickle.load(file_handle)
    return {str(symbol): normalize_frame(frame) for symbol, frame in raw.items()}


def timestamp_ms(timestamp: pd.Timestamp) -> int:
    utc_ts = timestamp.tz_convert(
        "UTC") if timestamp.tzinfo else timestamp.tz_localize("UTC")
    return int(utc_ts.timestamp() * 1000)


def period_frame_from_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        name = archive.namelist()[0]
        frame = pd.read_csv(archive.open(name), header=None)

    if frame.empty:
        return empty_frame()

    first_cell = str(frame.iloc[0, 0]).strip().lower()
    if first_cell in {"open_time", "timestamp"}:
        frame = frame.iloc[1:].reset_index(drop=True)

    if frame.empty:
        return empty_frame()

    frame = frame.iloc[:, :12]
    frame.columns = [
        "ts",
        "o",
        "h",
        "l",
        "c",
        "v",
        "close_time",
        "quote_volume",
        "trade_count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    frame["ts"] = pd.to_datetime(pd.to_numeric(
        frame["ts"], errors="coerce"), unit="ms", utc=True).dt.tz_localize(None)
    for column in OHLCV_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    parsed = frame.dropna(subset=["ts"]).set_index("ts")[
        OHLCV_COLUMNS].sort_index()
    parsed.index.name = "ts"
    parsed = parsed[~parsed.index.duplicated(keep="last")]
    return parsed


def fetch_period_zip(session: requests.Session, url: str) -> pd.DataFrame:
    response = session.get(url, timeout=60)
    if response.status_code == 404:
        return empty_frame()
    response.raise_for_status()
    return period_frame_from_zip(response.content)


def month_periods(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    periods: list[pd.Timestamp] = []
    cursor = start.replace(day=1)
    while cursor <= end:
        periods.append(cursor)
        cursor = cursor + pd.offsets.MonthBegin(1)
    return periods


def day_periods(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start.normalize(), end.normalize(), freq="D"))


def fetch_symbol_history(
    session: requests.Session,
    symbol: str,
    existing: pd.DataFrame,
) -> pd.DataFrame:
    today_utc = pd.Timestamp.now(tz="UTC").normalize()
    last_available_day = today_utc - pd.Timedelta(days=1)
    if last_available_day < START_TS.normalize():
        return existing

    current_month_start = last_available_day.replace(day=1)
    monthly_end = current_month_start - pd.Timedelta(days=1)

    parts = [existing]

    first_existing = pd.Timestamp(existing.index.min()).tz_localize(
        "UTC") if not existing.empty else None
    last_existing = pd.Timestamp(existing.index.max()).tz_localize(
        "UTC") if not existing.empty else None

    def needs_range(start: pd.Timestamp, end: pd.Timestamp) -> bool:
        if existing.empty:
            return True
        start_naive = start.tz_localize(None)
        end_naive = end.tz_localize(None)
        covered = existing.loc[(existing.index >= start_naive) & (
            existing.index <= end_naive)]
        expected = int(((end - start) / pd.Timedelta(minutes=15))) + 1
        return len(covered) < expected

    if monthly_end >= START_TS:
        for month_start in month_periods(START_TS, monthly_end):
            month_end = (month_start + pd.offsets.MonthEnd(0)).replace(
                hour=23, minute=45
            )
            if not needs_range(month_start, month_end):
                continue
            url = (
                f"{DATA_VISION_BASE}/monthly/klines/{symbol}USDT/{INTERVAL}/"
                f"{symbol}USDT-{INTERVAL}-{month_start.strftime('%Y-%m')}.zip"
            )
            log(f"--- {symbol}: monthly {month_start.strftime('%Y-%m')} ---")
            frame = fetch_period_zip(session, url)
            if not frame.empty:
                parts.append(frame)
            time.sleep(REQUEST_SLEEP)

    for day in day_periods(current_month_start, last_available_day):
        day_start = day
        day_end = day + pd.Timedelta(hours=23, minutes=45)
        if not needs_range(day_start, day_end):
            continue
        url = (
            f"{DATA_VISION_BASE}/daily/klines/{symbol}USDT/{INTERVAL}/"
            f"{symbol}USDT-{INTERVAL}-{day.strftime('%Y-%m-%d')}.zip"
        )
        log(f"--- {symbol}: daily {day.strftime('%Y-%m-%d')} ---")
        frame = fetch_period_zip(session, url)
        if not frame.empty:
            parts.append(frame)
        time.sleep(REQUEST_SLEEP)

    merged = pd.concat(parts).sort_index() if parts else empty_frame()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged


def fetch_recent_tail(
    session: requests.Session,
    base_urls: list[str],
    symbol: str,
    existing: pd.DataFrame,
) -> pd.DataFrame:
    today_floor = utc_floor_now().tz_localize(None)
    if existing.empty:
        start_ms = timestamp_ms(
            (today_floor - pd.Timedelta(days=2)).tz_localize("UTC"))
    else:
        start_ms = timestamp_ms((pd.Timestamp(existing.index.max()).tz_localize(
            "UTC") + pd.Timedelta(minutes=15)))
    end_ms = timestamp_ms(today_floor.tz_localize("UTC"))
    if start_ms > end_ms:
        return existing

    rows_accum: list[list[object]] = []
    cursor = start_ms
    while cursor <= end_ms:
        rows = None
        last_error: Exception | None = None
        for base_url in base_urls:
            endpoint = f"{base_url.rstrip('/')}/fapi/v1/klines"
            try:
                response = session.get(
                    endpoint,
                    params={
                        "symbol": f"{symbol}USDT",
                        "interval": INTERVAL,
                        "startTime": cursor,
                        "endTime": end_ms,
                        "limit": REQUEST_LIMIT,
                    },
                    timeout=20,
                )
                response.raise_for_status()
                rows = response.json()
                break
            except Exception as exc:
                last_error = exc
                continue

        if rows is None or not rows:
            return existing

        rows_accum.extend(rows)
        last_open_ms = int(rows[-1][0])
        if last_open_ms < cursor:
            break
        cursor = last_open_ms + BAR_MS
        if len(rows) < REQUEST_LIMIT or cursor > end_ms:
            break

    if not rows_accum:
        return existing

    frame = pd.DataFrame(
        rows_accum,
        columns=[
            "ts",
            "o",
            "h",
            "l",
            "c",
            "v",
            "close_time",
            "quote_volume",
            "trade_count",
            "taker_buy_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )
    frame["ts"] = pd.to_datetime(
        frame["ts"], unit="ms", utc=True).tz_localize(None)
    for column in OHLCV_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    parsed = frame.set_index("ts")[OHLCV_COLUMNS].sort_index()
    parsed.index.name = "ts"
    parsed = parsed[~parsed.index.duplicated(keep="last")]
    merged = pd.concat([existing, parsed]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged


def summarize(symbol: str, frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "symbol": symbol,
            "rows": 0,
            "start": pd.NaT,
            "end": pd.NaT,
        }

    return {
        "symbol": symbol,
        "rows": int(len(frame)),
        "start": frame.index.min(),
        "end": frame.index.max(),
    }


def save_store(store: dict[str, pd.DataFrame]) -> pd.DataFrame:
    with open(CACHE_PATH, "wb") as file_handle:
        pickle.dump(store, file_handle)

    summary = pd.DataFrame(
        [summarize(symbol, normalize_frame(frame))
         for symbol, frame in sorted(store.items())]
    ).sort_values("symbol").reset_index(drop=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    return summary


def refresh_symbol(
    env: dict[str, str],
    base_urls: list[str],
    symbol: str,
    existing: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    session = build_http_session(get_env_value(env, "HTTP_PROXY"))
    updated = fetch_symbol_history(session, symbol, existing)
    if get_bool_env(env, "BINANCE_FETCH_RECENT_TAIL", False):
        updated = fetch_recent_tail(session, base_urls, symbol, updated)
    return symbol, updated


def main() -> None:
    ensure_data_dirs()
    env = load_env_file(ENV_PATH)
    base_urls = get_csv_env(env, "BINANCE_FAPI_BASE_URLS",
                            DEFAULT_BINANCE_FAPI_BASE_URLS)
    store = load_existing_cache()
    symbols = get_symbols(env)
    max_workers = min(len(symbols), get_int_env(
        env, "BINANCE_SYMBOL_WORKERS", 4)) if symbols else 1

    if max_workers == 1:
        for symbol in symbols:
            _, updated = refresh_symbol(
                env, base_urls, symbol, normalize_frame(store.get(symbol)))
            store[symbol] = updated
            summary = summarize(symbol, updated)
            save_store(store)

            if summary["rows"]:
                log(
                    f"Saved {symbol}: {summary['rows']} bars  {summary['start']} -> {summary['end']}"
                )
            else:
                log(f"Saved {symbol}: 0 bars")

            time.sleep(REQUEST_SLEEP)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(
                    refresh_symbol,
                    env,
                    base_urls,
                    symbol,
                    normalize_frame(store.get(symbol)),
                ): symbol
                for symbol in symbols
            }

            for future in as_completed(future_to_symbol):
                symbol, updated = future.result()
                store[symbol] = updated
                summary = summarize(symbol, updated)
                save_store(store)

                if summary["rows"]:
                    log(
                        f"Saved {symbol}: {summary['rows']} bars  {summary['start']} -> {summary['end']}"
                    )
                else:
                    log(f"Saved {symbol}: 0 bars")

                time.sleep(REQUEST_SLEEP)

    summary = save_store(store)
    log(f"Saved {CACHE_PATH}")
    log(f"Saved {SUMMARY_PATH}")
    log("\n=== 15m Cache Summary ===")
    if summary.empty:
        log("No rows fetched")
    else:
        log(summary.to_string(index=False))


if __name__ == "__main__":
    main()
