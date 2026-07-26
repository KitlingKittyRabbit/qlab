from __future__ import annotations

"""Lifecycle: candidate.

Build candidate KeyStore/CoinGlass v4 raw history and cache payloads. This is
data-source infrastructure only; it does not run research gates or authorize
live use. Promote to active after pagination, coverage, and overlap validation.
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in (None, ""):
    PACKAGE_ROOT = Path(__file__).resolve().parents[3]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))

from qlab.data.crypto.keystore_coinglass_client import KeystoreCoinglassClient, parse_timestamp_ms  # noqa: E402
from qlab.data.crypto.keystore_coinglass_endpoints import (  # noqa: E402
    KEYSTORE_NATIVE_INTERVALS,
    build_history_params,
    select_endpoints,
)
from qlab.data.crypto.keystore_coinglass_parsers import parse_history_frame  # noqa: E402
from qlab.data.crypto.paths import RAW_HISTORY_ROOT, cache_path, ensure_data_dirs, manifest_path  # noqa: E402
from qlab.data.crypto.raw_history_store import write_timeseries_history  # noqa: E402
from qlab.data.crypto.symbol_universe import RESEARCH_SYMBOLS_12, resolve_target_symbols  # noqa: E402


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def summarize_frame(scope: str, name: str, frame: pd.DataFrame, extra: dict[str, Any]) -> dict[str, Any]:
    row = {"scope": scope, "name": name, **extra}
    if frame.empty:
        row.update({"rows": 0, "start": pd.NaT, "end": pd.NaT})
        return row
    row.update({"rows": len(frame), "start": frame.index.min(), "end": frame.index.max()})
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh candidate KeyStore CoinGlass v4 caches.")
    parser.add_argument("--execute", action="store_true", help="Actually call the API and write raw history/cache files.")
    parser.add_argument("--endpoints", default="", help="Comma-separated endpoint names. Default: base_replacement role.")
    parser.add_argument("--roles", default="base_replacement", help="Comma-separated roles when --endpoints is empty.")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols. Default: configured symbols or RESEARCH_SYMBOLS_12.")
    parser.add_argument("--intervals", default=",".join(KEYSTORE_NATIVE_INTERVALS))
    parser.add_argument("--limit", type=int, default=None, help="Override endpoint default limit. Omit to use endpoint registry.")
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--end", default="")
    parser.add_argument("--target-start", default="")
    parser.add_argument("--exchange", default="Binance")
    parser.add_argument("--exchange-list", default="")
    parser.add_argument("--rate-limit-sleep", type=float, default=None)
    parser.add_argument("--summary-output", default="keystore_coinglass_v4_cache_summary.csv")
    parser.add_argument("--replace-cache", action="store_true", help="Overwrite interval cache instead of merging updated keys.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_data_dirs()
    symbols = resolve_target_symbols(args.symbols, default=RESEARCH_SYMBOLS_12)
    intervals = tuple(parse_csv(args.intervals) or KEYSTORE_NATIVE_INTERVALS)
    endpoint_names = parse_csv(args.endpoints)
    roles = tuple(parse_csv(args.roles) or ["base_replacement"])
    endpoints = select_endpoints(names=endpoint_names or None, roles=roles)
    end_time_ms = parse_timestamp_ms(args.end) if args.end else None
    target_start_ms = parse_timestamp_ms(args.target_start) if args.target_start else None

    plan_rows: list[dict[str, Any]] = []
    for endpoint in endpoints:
        for interval in intervals:
            if not endpoint.supports_interval(interval):
                continue
            for symbol in symbols:
                if endpoint.supported_symbols and symbol not in endpoint.supported_symbols:
                    continue
                params = build_history_params(
                    endpoint,
                    symbol=symbol,
                    interval=interval,
                    limit=args.limit,
                    exchange=args.exchange,
                    exchange_list=args.exchange_list or "Binance,OKX,Bybit",
                )
                plan_rows.append(
                    {
                        "endpoint": endpoint.name,
                        "path": endpoint.path,
                        "symbol": symbol,
                        "interval": interval,
                        "cache_key": f"{symbol}_{endpoint.cache_prefix}",
                        "params": params,
                    }
                )

    if not args.execute:
        print("Dry run. Re-run with --execute to call KeyStore and write files.")
        print(pd.DataFrame(plan_rows).to_string(index=False))
        return

    client = KeystoreCoinglassClient(rate_limit_sleep=args.rate_limit_sleep)
    summary_rows: list[dict[str, Any]] = []
    payload_by_interval: dict[str, dict[str, pd.DataFrame]] = {interval: {} for interval in intervals}

    for item in plan_rows:
        endpoint = next(endpoint for endpoint in endpoints if endpoint.name == item["endpoint"])
        error = ""
        try:
            pages = client.iter_history_pages(
                endpoint.path,
                item["params"],
                target_start_ms=target_start_ms,
                initial_end_time_ms=end_time_ms,
                max_pages=args.max_pages,
            )
            raw_rows = [row for page in pages for row in page.rows]
            frame = parse_history_frame(endpoint.parser, raw_rows)
        except Exception as exc:
            pages = []
            raw_rows = []
            frame = pd.DataFrame()
            error = str(exc)
        payload_by_interval[item["interval"]][item["cache_key"]] = frame

        write_timeseries_history(
            frame=frame,
            destination=RAW_HISTORY_ROOT / "keystore_v4" / item["interval"] / f"{item['cache_key']}.csv",
            metadata={
                "api_version": "keystore_v4",
                "source": "keystore_coinglass_v4",
                "interval": item["interval"],
                "symbol": item["symbol"],
                "endpoint": endpoint.name,
                "path": endpoint.path,
                "parser": endpoint.parser,
                "migration_type": endpoint.migration_type,
            },
        )

        summary_rows.append(
            summarize_frame(
                f"ksv4_{item['interval']}",
                item["cache_key"],
                frame,
                {
                    "endpoint": endpoint.name,
                    "path": endpoint.path,
                    "symbol": item["symbol"],
                    "interval": item["interval"],
                    "page_count": len(pages),
                    "raw_rows": len(raw_rows),
                    "pagination_kind": endpoint.pagination_kind,
                    "error": error,
                },
            )
        )

    for interval, payload in payload_by_interval.items():
        if not payload:
            continue
        output_path = cache_path(f"keystore_coinglass_v4_{interval}_cache.pkl")
        if output_path.exists() and not args.replace_cache:
            with output_path.open("rb") as handle:
                existing_payload = pickle.load(handle)
            if not isinstance(existing_payload, dict):
                raise TypeError(f"Existing cache is not a dict: {output_path}")
            existing_payload.update(payload)
            payload = existing_payload
        with output_path.open("wb") as handle:
            pickle.dump(payload, handle)
        print(f"Saved {output_path}")

    summary = pd.DataFrame(summary_rows)
    summary_path = manifest_path(args.summary_output)
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
