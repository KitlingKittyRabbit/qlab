from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path
from typing import Any

import requests

if __package__ in (None, ""):
    PACKAGE_ROOT = Path(__file__).resolve().parents[3]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))


DEFAULT_BASE_URL = "https://fapi.binance.com"
DEFAULT_SYMBOLS = ["BTCUSDT"]
DEFAULT_TRADE_LIMIT = 1000
DEFAULT_INITIAL_TRADE_LIMIT = 200
DEFAULT_POLL_SECONDS = 1.0
DEFAULT_ERROR_BACKOFF_SECONDS = 5.0
REQUEST_TIMEOUT_SECONDS = 10
USER_AGENT = "qlab-binance-collector/0.1"
GAP_LOG_FIELDNAMES = [
    "event_time",
    "symbol",
    "event",
    "cycle",
    "stage",
    "failure_count",
    "gap_started_at",
    "gap_duration_seconds",
    "error_type",
    "error_message",
    "last_agg_trade_id",
]


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Binance UM futures agg trades and best bid/ask into qlab raw history."
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated Binance UM symbols, for example BTCUSDT,ETHUSDT.",
    )
    parser.add_argument(
        "--data-root",
        default="",
        help="Optional explicit qlab crypto data root. Overrides env-file lookup.",
    )
    parser.add_argument(
        "--env-file",
        default="",
        help="Optional env file to bootstrap QLAB_TRADE_ENV_PATH and related variables before importing qlab paths.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=DEFAULT_POLL_SECONDS,
        help="Sleep between polling cycles. Default: 1.0 second.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Number of polling cycles to run. Use 0 for continuous collection.",
    )
    parser.add_argument(
        "--initial-trade-limit",
        type=int,
        default=DEFAULT_INITIAL_TRADE_LIMIT,
        help="Recent agg trades to seed on the first run when no state exists.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Binance UM REST base URL.",
    )
    parser.add_argument(
        "--error-backoff-seconds",
        type=float,
        default=DEFAULT_ERROR_BACKOFF_SECONDS,
        help="Minimum sleep after a failed polling cycle. Default: 5.0 seconds.",
    )
    return parser.parse_args()


def _bootstrap_env(env_file_arg: str, data_root_arg: str) -> None:
    if data_root_arg.strip():
        os.environ["QLAB_CRYPTO_DATA_DIR"] = data_root_arg.strip()

    env_file = Path(env_file_arg).expanduser() if env_file_arg.strip() else None
    if env_file is not None:
        if not env_file.is_absolute():
            env_file = Path.cwd() / env_file
        os.environ.setdefault("QLAB_TRADE_ENV_PATH", str(env_file))
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def build_http_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "accept": "application/json"})
    return session


def utc_now() -> datetime:
    return datetime.now(UTC)


def datetime_from_ms(value: int | str | float) -> datetime:
    return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)


def date_label_from_ms(value: int | str | float) -> str:
    return datetime_from_ms(value).strftime("%Y-%m-%d")


def append_csv_rows(destination: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_header = not destination.exists()
    with destination.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def save_json(destination: Path, payload: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def load_json(destination: Path) -> dict[str, Any]:
    if not destination.exists():
        return {}
    try:
        return json.loads(destination.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def request_json(session: requests.Session, url: str, params: dict[str, Any]) -> Any:
    response = session.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def short_error_message(error: Exception) -> str:
    message = str(error).strip() or repr(error)
    return message.replace("\n", " ")[:500]


def fetch_exchange_info(session: requests.Session, base_url: str, symbol: str) -> dict[str, Any]:
    payload = request_json(
        session,
        f"{base_url.rstrip('/')}/fapi/v1/exchangeInfo",
        {"symbol": symbol},
    )
    symbols = payload.get("symbols", [])
    if not symbols:
        raise RuntimeError(f"Binance exchangeInfo returned no symbol payload for {symbol}")
    return symbols[0]


def fetch_book_ticker(session: requests.Session, base_url: str, symbol: str) -> dict[str, Any]:
    payload = request_json(
        session,
        f"{base_url.rstrip('/')}/fapi/v1/ticker/bookTicker",
        {"symbol": symbol},
    )
    local_time_ms = int(utc_now().timestamp() * 1000)
    exchange_time_ms = int(payload.get("time", local_time_ms))
    return {
        "symbol": symbol,
        "exchange_time_ms": exchange_time_ms,
        "local_time_ms": local_time_ms,
        "bid_price": payload.get("bidPrice"),
        "bid_quantity": payload.get("bidQty"),
        "ask_price": payload.get("askPrice"),
        "ask_quantity": payload.get("askQty"),
    }


def fetch_recent_agg_trades(
    session: requests.Session,
    base_url: str,
    symbol: str,
    limit: int,
) -> list[dict[str, Any]]:
    payload = request_json(
        session,
        f"{base_url.rstrip('/')}/fapi/v1/aggTrades",
        {"symbol": symbol, "limit": max(1, min(limit, DEFAULT_TRADE_LIMIT))},
    )
    return payload if isinstance(payload, list) else []


def fetch_incremental_agg_trades(
    session: requests.Session,
    base_url: str,
    symbol: str,
    from_id: int,
) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    cursor = from_id
    while True:
        payload = request_json(
            session,
            f"{base_url.rstrip('/')}/fapi/v1/aggTrades",
            {"symbol": symbol, "fromId": cursor, "limit": DEFAULT_TRADE_LIMIT},
        )
        rows = payload if isinstance(payload, list) else []
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < DEFAULT_TRADE_LIMIT:
            break
        cursor = int(rows[-1]["a"]) + 1
    return all_rows


def normalize_trade_rows(symbol: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    local_time_ms = int(utc_now().timestamp() * 1000)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                "symbol": symbol,
                "exchange_time_ms": int(row["T"]),
                "local_time_ms": local_time_ms,
                "aggregate_trade_id": int(row["a"]),
                "first_trade_id": int(row["f"]),
                "last_trade_id": int(row["l"]),
                "price": row["p"],
                "quantity": row["q"],
                "is_buyer_maker": bool(row["m"]),
            }
        )
    return normalized


def write_partitioned_rows(
    root: Path,
    stream_name: str,
    symbol: str,
    time_field: str,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> int:
    if not rows:
        return 0

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = date_label_from_ms(row[time_field])
        grouped.setdefault(label, []).append(row)

    written = 0
    for label, chunk in grouped.items():
        destination = root / stream_name / symbol / f"{label}.csv"
        append_csv_rows(destination, fieldnames, chunk)
        written += len(chunk)
    return written


def parse_symbols(raw_value: str) -> list[str]:
    values = [item.strip().upper() for item in raw_value.split(",") if item.strip()]
    return values or list(DEFAULT_SYMBOLS)


def runtime_dir(raw_history_root: Path) -> Path:
    return raw_history_root / "binance_um" / "runtime"


def runtime_state_path(raw_history_root: Path) -> Path:
    return runtime_dir(raw_history_root) / "collector_state.json"


def runtime_gap_log_path(raw_history_root: Path) -> Path:
    return runtime_dir(raw_history_root) / "gap_log.csv"


def metadata_path(raw_history_root: Path, symbol: str) -> Path:
    return raw_history_root / "binance_um" / "metadata" / f"{symbol.lower()}_exchange_info.json"


def ensure_symbol_metadata(
    session: requests.Session,
    base_url: str,
    raw_history_root: Path,
    symbol: str,
) -> None:
    destination = metadata_path(raw_history_root, symbol)
    if destination.exists():
        return
    info = fetch_exchange_info(session, base_url, symbol)
    save_json(
        destination,
        {
            "fetched_at": utc_now().isoformat(),
            "base_url": base_url,
            "symbol": symbol,
            "exchange_info": info,
        },
    )


def open_gap(
    raw_history_root: Path,
    symbol: str,
    symbol_state: dict[str, Any],
    cycle: int,
    stage: str,
    error: Exception,
) -> None:
    now = utc_now()
    error_type = type(error).__name__
    error_message = short_error_message(error)

    failure_count = int(symbol_state.get("consecutive_failures", 0)) + 1
    symbol_state["consecutive_failures"] = failure_count
    symbol_state["last_error_at"] = now.isoformat()
    symbol_state["last_error_type"] = error_type
    symbol_state["last_error_message"] = error_message

    gap = symbol_state.get("active_gap")
    if gap is None:
        started_at = now.isoformat()
        gap = {
            "started_at": started_at,
            "stage": stage,
            "failure_count": 1,
            "last_error_at": now.isoformat(),
            "last_error_type": error_type,
            "last_error_message": error_message,
        }
        symbol_state["active_gap"] = gap
        append_csv_rows(
            runtime_gap_log_path(raw_history_root),
            GAP_LOG_FIELDNAMES,
            [
                {
                    "event_time": now.isoformat(),
                    "symbol": symbol,
                    "event": "gap_start",
                    "cycle": cycle,
                    "stage": stage,
                    "failure_count": 1,
                    "gap_started_at": started_at,
                    "gap_duration_seconds": "",
                    "error_type": error_type,
                    "error_message": error_message,
                    "last_agg_trade_id": symbol_state.get("last_agg_trade_id", ""),
                }
            ],
        )
        log(f"Cycle {cycle} {symbol}: gap_start at {stage}: {error_type}: {error_message}")
        return

    gap["stage"] = stage
    gap["failure_count"] = int(gap.get("failure_count", 0)) + 1
    gap["last_error_at"] = now.isoformat()
    gap["last_error_type"] = error_type
    gap["last_error_message"] = error_message
    if int(gap["failure_count"]) % 10 == 0:
        log(
            f"Cycle {cycle} {symbol}: gap still open after {gap['failure_count']} failures at {stage}: {error_type}: {error_message}"
        )


def close_gap(raw_history_root: Path, symbol: str, symbol_state: dict[str, Any], cycle: int) -> None:
    now = utc_now()
    symbol_state["last_success_at"] = now.isoformat()
    symbol_state["consecutive_failures"] = 0

    gap = symbol_state.pop("active_gap", None)
    if gap is None:
        return

    started_at = str(gap.get("started_at", now.isoformat()))
    try:
        started_dt = datetime.fromisoformat(started_at)
        duration_seconds = max(0.0, round((now - started_dt).total_seconds(), 3))
    except ValueError:
        duration_seconds = 0.0

    append_csv_rows(
        runtime_gap_log_path(raw_history_root),
        GAP_LOG_FIELDNAMES,
        [
            {
                "event_time": now.isoformat(),
                "symbol": symbol,
                "event": "gap_end",
                "cycle": cycle,
                "stage": gap.get("stage", ""),
                "failure_count": gap.get("failure_count", 0),
                "gap_started_at": started_at,
                "gap_duration_seconds": duration_seconds,
                "error_type": gap.get("last_error_type", ""),
                "error_message": gap.get("last_error_message", ""),
                "last_agg_trade_id": symbol_state.get("last_agg_trade_id", ""),
            }
        ],
    )
    log(f"Cycle {cycle} {symbol}: gap_end after {duration_seconds} seconds")


def main() -> None:
    args = parse_args()
    _bootstrap_env(args.env_file, args.data_root)

    try:
        from qlab.data.crypto.paths import RAW_HISTORY_ROOT, ensure_data_dirs  # noqa: WPS433,E402
    except RuntimeError as exc:
        if "Crypto data root is not configured" not in str(exc):
            raise

        hint_lines = [
            "Crypto data root is not configured for the collector.",
            "",
            "Use one of the following:",
            "  1. Pass --data-root /path/to/crypto_data",
            "  2. Export QLAB_CRYPTO_DATA_DIR=/path/to/crypto_data",
        ]
        raise SystemExit("\n".join(hint_lines)) from exc

    ensure_data_dirs()
    symbols = parse_symbols(args.symbols)
    state_path = runtime_state_path(RAW_HISTORY_ROOT)
    state = load_json(state_path)
    state.setdefault("symbols", {})
    session = build_http_session()
    trade_fieldnames = [
        "symbol",
        "exchange_time_ms",
        "local_time_ms",
        "aggregate_trade_id",
        "first_trade_id",
        "last_trade_id",
        "price",
        "quantity",
        "is_buyer_maker",
    ]
    book_fieldnames = [
        "symbol",
        "exchange_time_ms",
        "local_time_ms",
        "bid_price",
        "bid_quantity",
        "ask_price",
        "ask_quantity",
    ]

    log(f"Raw history root: {RAW_HISTORY_ROOT}")
    log(f"Symbols: {', '.join(symbols)}")

    cycle = 0
    while True:
        cycle += 1
        cycle_trade_rows = 0
        cycle_book_rows = 0
        cycle_failed = False

        for symbol in symbols:
            symbol_state = state["symbols"].setdefault(symbol, {})
            if not metadata_path(RAW_HISTORY_ROOT, symbol).exists():
                try:
                    ensure_symbol_metadata(session, args.base_url, RAW_HISTORY_ROOT, symbol)
                except Exception as exc:
                    log(f"Cycle {cycle} {symbol}: metadata fetch skipped: {type(exc).__name__}: {short_error_message(exc)}")

            stage = "agg_trades"
            try:
                last_trade_id = symbol_state.get("last_agg_trade_id")
                if last_trade_id is None:
                    raw_trades = fetch_recent_agg_trades(
                        session,
                        args.base_url,
                        symbol,
                        args.initial_trade_limit,
                    )
                else:
                    raw_trades = fetch_incremental_agg_trades(
                        session,
                        args.base_url,
                        symbol,
                        int(last_trade_id) + 1,
                    )

                trade_rows = normalize_trade_rows(symbol, raw_trades)
                if trade_rows:
                    written_trade_rows = write_partitioned_rows(
                        RAW_HISTORY_ROOT / "binance_um",
                        "agg_trades",
                        symbol,
                        "exchange_time_ms",
                        trade_fieldnames,
                        trade_rows,
                    )
                    if written_trade_rows:
                        symbol_state["last_agg_trade_id"] = trade_rows[-1]["aggregate_trade_id"]
                        cycle_trade_rows += written_trade_rows

                stage = "book_ticker"
                book_row = fetch_book_ticker(session, args.base_url, symbol)
                cycle_book_rows += write_partitioned_rows(
                    RAW_HISTORY_ROOT / "binance_um",
                    "book_ticker",
                    symbol,
                    "exchange_time_ms",
                    book_fieldnames,
                    [book_row],
                )
                close_gap(RAW_HISTORY_ROOT, symbol, symbol_state, cycle)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                cycle_failed = True
                open_gap(RAW_HISTORY_ROOT, symbol, symbol_state, cycle, stage, exc)
                session.close()
                session = build_http_session()
                continue

        state["updated_at"] = utc_now().isoformat()
        save_json(state_path, state)
        log(
            f"Cycle {cycle}: wrote {cycle_trade_rows} trade rows and {cycle_book_rows} book rows"
        )

        if args.cycles > 0 and cycle >= args.cycles:
            break
        sleep_seconds = max(0.0, args.poll_seconds)
        if cycle_failed:
            sleep_seconds = max(sleep_seconds, max(0.0, args.error_backoff_seconds))
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
