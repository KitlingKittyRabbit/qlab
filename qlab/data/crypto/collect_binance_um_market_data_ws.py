from __future__ import annotations

import argparse
import asyncio
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
import websockets

if __package__ in (None, ""):
    PACKAGE_ROOT = Path(__file__).resolve().parents[3]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))


DEFAULT_BASE_URL = "https://fapi.binance.com"
DEFAULT_STREAM_BASE_URL = "wss://fstream.binance.com/stream"
DEFAULT_SYMBOLS = ["BTCUSDT"]
DEFAULT_ERROR_BACKOFF_SECONDS = 5.0
DEFAULT_IDLE_TIMEOUT_SECONDS = 30.0
DEFAULT_PING_INTERVAL_SECONDS = 20.0
DEFAULT_PING_TIMEOUT_SECONDS = 20.0
REQUEST_TIMEOUT_SECONDS = 10
USER_AGENT = "qlab-binance-ws-collector/0.1"
GAP_LOG_FIELDNAMES = [
    "event_time",
    "symbol",
    "event",
    "session_id",
    "stage",
    "failure_count",
    "gap_started_at",
    "gap_duration_seconds",
    "error_type",
    "error_message",
    "last_trade_id",
]


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Binance UM trade and bookTicker websocket streams into qlab raw history."
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
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Binance UM REST base URL used for metadata snapshots.",
    )
    parser.add_argument(
        "--stream-base-url",
        default=DEFAULT_STREAM_BASE_URL,
        help="Binance websocket base URL. Default: wss://fstream.binance.com/stream.",
    )
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=0.0,
        help="Run duration in seconds. Use 0 for continuous collection.",
    )
    parser.add_argument(
        "--error-backoff-seconds",
        type=float,
        default=DEFAULT_ERROR_BACKOFF_SECONDS,
        help="Sleep after websocket session failure before reconnecting. Default: 5.0 seconds.",
    )
    parser.add_argument(
        "--idle-timeout-seconds",
        type=float,
        default=DEFAULT_IDLE_TIMEOUT_SECONDS,
        help="Reconnect if no websocket payload arrives within this timeout. Default: 30 seconds.",
    )
    parser.add_argument(
        "--ping-interval-seconds",
        type=float,
        default=DEFAULT_PING_INTERVAL_SECONDS,
        help="Websocket ping interval. Default: 20 seconds.",
    )
    parser.add_argument(
        "--ping-timeout-seconds",
        type=float,
        default=DEFAULT_PING_TIMEOUT_SECONDS,
        help="Websocket ping timeout. Default: 20 seconds.",
    )
    parser.add_argument(
        "--state-flush-seconds",
        type=float,
        default=1.0,
        help="Minimum interval between state flushes. Default: 1 second.",
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


def build_http_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "accept": "application/json"})
    return session


def request_json(session: requests.Session, url: str, params: dict[str, Any]) -> Any:
    response = session.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


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


def short_error_message(error: Exception) -> str:
    message = str(error).strip() or repr(error)
    return message.replace("\n", " ")[:500]


def parse_symbols(raw_value: str) -> list[str]:
    values = [item.strip().upper() for item in raw_value.split(",") if item.strip()]
    return values or list(DEFAULT_SYMBOLS)


def collector_root(raw_history_root: Path) -> Path:
    return raw_history_root / "binance_um_websocket"


def runtime_dir(raw_history_root: Path) -> Path:
    return collector_root(raw_history_root) / "runtime"


def runtime_state_path(raw_history_root: Path) -> Path:
    return runtime_dir(raw_history_root) / "collector_state.json"


def runtime_gap_log_path(raw_history_root: Path) -> Path:
    return runtime_dir(raw_history_root) / "gap_log.csv"


def metadata_path(raw_history_root: Path, symbol: str) -> Path:
    return collector_root(raw_history_root) / "metadata" / f"{symbol.lower()}_exchange_info.json"


def stream_path(raw_history_root: Path, stream_name: str, symbol: str, day_label: str) -> Path:
    return collector_root(raw_history_root) / stream_name / symbol / f"{day_label}.csv"


def write_partitioned_rows(
    raw_history_root: Path,
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
        append_csv_rows(stream_path(raw_history_root, stream_name, symbol, label), fieldnames, chunk)
        written += len(chunk)
    return written


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
    session_id: int,
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
                    "session_id": session_id,
                    "stage": stage,
                    "failure_count": 1,
                    "gap_started_at": started_at,
                    "gap_duration_seconds": "",
                    "error_type": error_type,
                    "error_message": error_message,
                    "last_trade_id": symbol_state.get("last_trade_id", ""),
                }
            ],
        )
        log(f"Session {session_id} {symbol}: gap_start at {stage}: {error_type}: {error_message}")
        return

    gap["stage"] = stage
    gap["failure_count"] = int(gap.get("failure_count", 0)) + 1
    gap["last_error_at"] = now.isoformat()
    gap["last_error_type"] = error_type
    gap["last_error_message"] = error_message


def close_gap(raw_history_root: Path, symbol: str, symbol_state: dict[str, Any], session_id: int) -> None:
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
                "session_id": session_id,
                "stage": gap.get("stage", ""),
                "failure_count": gap.get("failure_count", 0),
                "gap_started_at": started_at,
                "gap_duration_seconds": duration_seconds,
                "error_type": gap.get("last_error_type", ""),
                "error_message": gap.get("last_error_message", ""),
                "last_trade_id": symbol_state.get("last_trade_id", ""),
            }
        ],
    )
    log(f"Session {session_id} {symbol}: gap_end after {duration_seconds} seconds")


def normalize_trade_row(data: dict[str, Any]) -> dict[str, Any]:
    local_time_ms = int(utc_now().timestamp() * 1000)
    return {
        "symbol": data["s"],
        "event_time_ms": int(data["E"]),
        "exchange_time_ms": int(data["T"]),
        "local_time_ms": local_time_ms,
        "trade_id": int(data["t"]),
        "price": data["p"],
        "quantity": data["q"],
        "execution_type": data.get("X", ""),
        "is_buyer_maker": bool(data["m"]),
    }


def normalize_book_ticker_row(data: dict[str, Any]) -> dict[str, Any]:
    local_time_ms = int(utc_now().timestamp() * 1000)
    event_time_ms = int(data.get("E", local_time_ms))
    exchange_time_ms = int(data.get("T", event_time_ms))
    return {
        "symbol": data["s"],
        "event_time_ms": event_time_ms,
        "exchange_time_ms": exchange_time_ms,
        "local_time_ms": local_time_ms,
        "update_id": int(data.get("u", 0)),
        "bid_price": data["b"],
        "bid_quantity": data["B"],
        "ask_price": data["a"],
        "ask_quantity": data["A"],
    }


def build_stream_url(stream_base_url: str, symbols: list[str]) -> str:
    streams: list[str] = []
    for symbol in symbols:
        lower_symbol = symbol.lower()
        streams.append(f"{lower_symbol}@trade")
        streams.append(f"{lower_symbol}@bookTicker")
    return f"{stream_base_url.rstrip('/')}?streams={'/'.join(streams)}"


def maybe_flush_state(
    raw_history_root: Path,
    state: dict[str, Any],
    last_flush_monotonic: float,
    flush_interval_seconds: float,
) -> float:
    now_monotonic = time.monotonic()
    if now_monotonic - last_flush_monotonic < flush_interval_seconds:
        return last_flush_monotonic
    state["updated_at"] = utc_now().isoformat()
    save_json(runtime_state_path(raw_history_root), state)
    return now_monotonic


async def collect(args: argparse.Namespace, raw_history_root: Path) -> None:
    symbols = parse_symbols(args.symbols)
    state_path = runtime_state_path(raw_history_root)
    state = load_json(state_path)
    state.setdefault("symbols", {})
    state["stream_url"] = build_stream_url(args.stream_base_url, symbols)
    state["collector_type"] = "websocket"
    session_id = 0
    last_flush_monotonic = time.monotonic()

    rest_session = build_http_session()
    for symbol in symbols:
        symbol_state = state["symbols"].setdefault(symbol, {})
        symbol_state.setdefault("trade_messages", 0)
        symbol_state.setdefault("book_messages", 0)
        try:
            ensure_symbol_metadata(rest_session, args.base_url, raw_history_root, symbol)
        except Exception as exc:
            log(f"Metadata snapshot skipped for {symbol}: {type(exc).__name__}: {short_error_message(exc)}")

    trade_fieldnames = [
        "symbol",
        "event_time_ms",
        "exchange_time_ms",
        "local_time_ms",
        "trade_id",
        "price",
        "quantity",
        "execution_type",
        "is_buyer_maker",
    ]
    book_fieldnames = [
        "symbol",
        "event_time_ms",
        "exchange_time_ms",
        "local_time_ms",
        "update_id",
        "bid_price",
        "bid_quantity",
        "ask_price",
        "ask_quantity",
    ]

    deadline = time.monotonic() + args.run_seconds if args.run_seconds > 0 else 0.0
    log(f"Raw history root: {raw_history_root}")
    log(f"Symbols: {', '.join(symbols)}")

    while True:
        if deadline and time.monotonic() >= deadline:
            break

        session_id += 1
        state["last_session_id"] = session_id
        state["last_session_started_at"] = utc_now().isoformat()
        state["stream_url"] = build_stream_url(args.stream_base_url, symbols)
        save_json(state_path, state)
        log(f"Session {session_id}: connect {state['stream_url']}")

        try:
            async with websockets.connect(
                state["stream_url"],
                ping_interval=args.ping_interval_seconds,
                ping_timeout=args.ping_timeout_seconds,
                max_size=None,
            ) as websocket:
                while True:
                    if deadline and time.monotonic() >= deadline:
                        state["updated_at"] = utc_now().isoformat()
                        save_json(state_path, state)
                        return

                    raw_message = await asyncio.wait_for(
                        websocket.recv(), timeout=args.idle_timeout_seconds
                    )
                    payload = json.loads(raw_message)
                    data = payload.get("data", payload)
                    symbol = str(data.get("s", "")).upper()
                    if symbol not in symbols:
                        continue

                    symbol_state = state["symbols"].setdefault(symbol, {})
                    event_type = data.get("e")
                    if event_type == "trade":
                        row = normalize_trade_row(data)
                        write_partitioned_rows(
                            raw_history_root,
                            "trades",
                            symbol,
                            "exchange_time_ms",
                            trade_fieldnames,
                            [row],
                        )
                        symbol_state["last_trade_id"] = row["trade_id"]
                        symbol_state["last_trade_time_ms"] = row["exchange_time_ms"]
                        symbol_state["trade_messages"] = int(symbol_state.get("trade_messages", 0)) + 1
                    elif event_type == "bookTicker":
                        row = normalize_book_ticker_row(data)
                        write_partitioned_rows(
                            raw_history_root,
                            "book_ticker",
                            symbol,
                            "exchange_time_ms",
                            book_fieldnames,
                            [row],
                        )
                        symbol_state["last_book_update_id"] = row["update_id"]
                        symbol_state["last_book_time_ms"] = row["exchange_time_ms"]
                        symbol_state["book_messages"] = int(symbol_state.get("book_messages", 0)) + 1
                    else:
                        continue

                    symbol_state["last_message_at"] = utc_now().isoformat()
                    close_gap(raw_history_root, symbol, symbol_state, session_id)
                    last_flush_monotonic = maybe_flush_state(
                        raw_history_root,
                        state,
                        last_flush_monotonic,
                        max(0.0, args.state_flush_seconds),
                    )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            for symbol in symbols:
                symbol_state = state["symbols"].setdefault(symbol, {})
                open_gap(raw_history_root, symbol, symbol_state, session_id, "ws_session", exc)
            state["updated_at"] = utc_now().isoformat()
            state["last_session_error"] = short_error_message(exc)
            save_json(state_path, state)
            await asyncio.sleep(max(0.0, args.error_backoff_seconds))


def main() -> None:
    args = parse_args()
    _bootstrap_env(args.env_file, args.data_root)

    try:
        from qlab.data.crypto.paths import RAW_HISTORY_ROOT, ensure_data_dirs  # noqa: WPS433,E402
    except RuntimeError as exc:
        if "Crypto data root is not configured" not in str(exc):
            raise

        workspace_data_root = Path(__file__).resolve().parents[4] / "qlab_crypto_data"
        hint_lines = [
            "Crypto data root is not configured for the websocket collector.",
            "",
            "Use one of the following:",
            "  1. Pass --data-root /path/to/crypto_data",
            "  2. Export QLAB_CRYPTO_DATA_DIR=/path/to/crypto_data",
        ]
        if workspace_data_root.exists():
            hint_lines.extend(
                [
                    "",
                    "For this workspace, the likely command is:",
                    (
                        "  /root/workspace/.venv311/bin/python "
                        "/root/workspace/qlab/qlab/data/crypto/collect_binance_um_market_data_ws.py "
                        f"--data-root {workspace_data_root} --symbols BTCUSDT"
                    ),
                ]
            )
        raise SystemExit("\n".join(hint_lines)) from exc

    ensure_data_dirs()
    asyncio.run(collect(args, RAW_HISTORY_ROOT))


if __name__ == "__main__":
    main()
