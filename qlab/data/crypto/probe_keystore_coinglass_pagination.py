from __future__ import annotations

"""Lifecycle: candidate.

Small live probe for KeyStore/CoinGlass v4 history pagination. This script is a
technical validation aid, not a research result. Archive after pagination rules
are frozen in the data-source implementation.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in (None, ""):
    PACKAGE_ROOT = Path(__file__).resolve().parents[3]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))

from qlab.data.crypto.keystore_coinglass_client import (  # noqa: E402
    KeystoreCoinglassClient,
    format_timestamp_ms,
    parse_timestamp_ms,
)
from qlab.data.crypto.keystore_coinglass_endpoints import (  # noqa: E402
    KEYSTORE_NATIVE_INTERVALS,
    build_history_params,
    select_endpoints,
)
from qlab.data.crypto.paths import manifest_path  # noqa: E402
from qlab.data.crypto.symbol_universe import RESEARCH_SYMBOLS_12, resolve_target_symbols  # noqa: E402


DEFAULT_PROBE_ENDPOINTS = ["taker_pair", "oi", "fr_oi_weight"]


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe KeyStore pagination with small endpoint samples.")
    parser.add_argument("--endpoints", default=",".join(DEFAULT_PROBE_ENDPOINTS))
    parser.add_argument("--symbols", default="BTC")
    parser.add_argument("--intervals", default="1h,2h,8h,12h")
    parser.add_argument("--limit", type=int, default=None, help="Override endpoint default limit. Omit to use endpoint registry.")
    parser.add_argument("--max-pages", type=int, default=2)
    parser.add_argument("--end", default="")
    parser.add_argument("--target-start", default="")
    parser.add_argument("--exchange", default="Binance")
    parser.add_argument("--exchange-list", default="")
    parser.add_argument("--output", default="keystore_coinglass_v4_pagination_probe.csv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rate-limit-sleep", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    symbols = resolve_target_symbols(args.symbols, default=RESEARCH_SYMBOLS_12)
    intervals = tuple(parse_csv(args.intervals) or KEYSTORE_NATIVE_INTERVALS)
    endpoints = select_endpoints(names=parse_csv(args.endpoints))
    end_time_ms = parse_timestamp_ms(args.end) if args.end else None
    target_start_ms = parse_timestamp_ms(args.target_start) if args.target_start else None

    planned: list[dict[str, Any]] = []
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
                planned.append(
                    {
                        "endpoint": endpoint.name,
                        "path": endpoint.path,
                        "symbol": symbol,
                        "interval": interval,
                        "params": params,
                    }
                )

    if args.dry_run:
        print(pd.DataFrame(planned).to_string(index=False))
        return

    output_path = manifest_path(args.output)
    completed: set[tuple[str, str, str]] = set()
    if args.resume and output_path.exists():
        existing = pd.read_csv(output_path)
        for _, row in existing.iterrows():
            completed.add((str(row["endpoint"]), str(row["symbol"]), str(row["interval"])))

    client = KeystoreCoinglassClient(rate_limit_sleep=args.rate_limit_sleep)
    rows: list[dict[str, Any]] = []
    if args.resume and output_path.exists():
        rows = pd.read_csv(output_path).to_dict("records")

    for item in planned:
        row_key = (item["endpoint"], item["symbol"], item["interval"])
        if row_key in completed:
            continue
        error = ""
        try:
            pages = client.iter_history_pages(
                item["path"],
                item["params"],
                target_start_ms=target_start_ms,
                initial_end_time_ms=end_time_ms,
                max_pages=args.max_pages,
            )
        except Exception as exc:
            pages = []
            error = str(exc)
        first = pages[0] if pages else None
        second = pages[1] if len(pages) > 1 else None
        rows.append(
            {
                "endpoint": item["endpoint"],
                "path": item["path"],
                "symbol": item["symbol"],
                "interval": item["interval"],
                "limit": item["params"].get("limit", ""),
                "page_count": len(pages),
                "first_rows": first.row_count if first else 0,
                "first_request_end_time": first.request_params.get("end_time", "") if first else "",
                "first_start": first.earliest if first else "",
                "first_end": first.latest if first else "",
                "page2_rows": second.row_count if second else 0,
                "page2_request_end_time": second.request_params.get("end_time", "") if second else "",
                "page2_start": second.earliest if second else "",
                "page2_end": second.latest if second else "",
                "overlap_rows": "",
                "gap_rows": "",
                "target_start": format_timestamp_ms(target_start_ms),
                "can_backfill_to_target_start": bool(pages and target_start_ms is not None and pages[-1].earliest_ms is not None and pages[-1].earliest_ms <= target_start_ms),
                "error": error,
            }
        )
        pd.DataFrame(rows).to_csv(output_path, index=False)

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
