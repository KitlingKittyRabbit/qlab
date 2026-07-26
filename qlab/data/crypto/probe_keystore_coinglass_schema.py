from __future__ import annotations

"""Lifecycle: candidate.

Probe KeyStore/CoinGlass v4 candidate endpoint schemas and shallow coverage.
This is a technical validation aid before factor-registry entries are written.
Archive after endpoint schemas are frozen in the KeyStore data-source route.
"""

import argparse
import json
import sys
import time
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
    build_history_params,
    select_endpoints,
)
from qlab.data.crypto.keystore_coinglass_parsers import parse_history_frame  # noqa: E402
from qlab.data.crypto.paths import manifest_path  # noqa: E402


DEFAULT_ENDPOINTS = (
    "futures_cvd",
    "futures_cvd_agg",
    "spot_cvd",
    "spot_cvd_agg",
    "spot_taker_pair",
    "ob_pair",
    "ob_agg",
    "futures_net_pos_v2",
    "futures_net_pos",
    "futures_whale_index",
)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def numeric_columns_from_rows(rows: list[Any]) -> list[str]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    excluded = {"time", "timestamp", "t", "date"}
    columns: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            columns.append(str(column))
    return columns


def summarize_numeric_frame(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "parsed_columns": "",
            "null_rate_max": "",
            "zero_rate_max": "",
            "negative_rate_max": "",
            "duplicate_ts_count": 0,
            "monotonic_ts": True,
        }
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    return {
        "parsed_columns": ",".join(str(column) for column in numeric.columns),
        "null_rate_max": float(numeric.isna().mean().max()) if len(numeric.columns) else "",
        "zero_rate_max": float((numeric == 0).mean().max()) if len(numeric.columns) else "",
        "negative_rate_max": float((numeric < 0).mean().max()) if len(numeric.columns) else "",
        "duplicate_ts_count": int(frame.index.duplicated().sum()),
        "monotonic_ts": bool(frame.index.is_monotonic_increasing),
    }


def sample_text(rows: list[Any], limit: int = 1) -> str:
    return json.dumps(rows[:limit], ensure_ascii=False, default=str)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe KeyStore candidate endpoint schemas.")
    parser.add_argument("--endpoints", default=",".join(DEFAULT_ENDPOINTS))
    parser.add_argument("--symbols", default="BTC,ETH,SUI")
    parser.add_argument("--intervals", default="1h,12h,1d")
    parser.add_argument("--limit", type=int, default=None, help="Override endpoint default limit. Omit to use endpoint registry.")
    parser.add_argument("--max-pages", type=int, default=2)
    parser.add_argument("--end", default="")
    parser.add_argument("--target-start", default="")
    parser.add_argument("--exchange", default="Binance")
    parser.add_argument("--exchange-list", default="")
    parser.add_argument("--output", default="keystore_coinglass_v4_candidate_schema_probe.csv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rate-limit-sleep", type=float, default=6.2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    endpoints = select_endpoints(names=parse_csv(args.endpoints))
    symbols = parse_csv(args.symbols)
    intervals = parse_csv(args.intervals)
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
                        "parser": endpoint.parser,
                        "symbol": symbol,
                        "interval": interval,
                        "params": params,
                    }
                )

    if args.dry_run:
        print(pd.DataFrame(planned).to_string(index=False))
        return

    client = KeystoreCoinglassClient(rate_limit_sleep=args.rate_limit_sleep)
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(planned):
        if idx > 0 and args.rate_limit_sleep:
            time.sleep(args.rate_limit_sleep)
        error = ""
        try:
            pages = client.iter_history_pages(
                item["path"],
                item["params"],
                target_start_ms=target_start_ms,
                initial_end_time_ms=end_time_ms,
                max_pages=args.max_pages,
            )
            raw_rows = [row for page in pages for row in page.rows]
            parsed = parse_history_frame(item["parser"], raw_rows)
        except Exception as exc:
            pages = []
            raw_rows = []
            parsed = pd.DataFrame()
            error = str(exc)

        first = pages[0] if pages else None
        last = pages[-1] if pages else None
        summary = summarize_numeric_frame(parsed)
        rows.append(
            {
                "endpoint": item["endpoint"],
                "path": item["path"],
                "parser": item["parser"],
                "symbol": item["symbol"],
                "interval": item["interval"],
                "limit": item["params"].get("limit", ""),
                "page_count": len(pages),
                "raw_rows": len(raw_rows),
                "parsed_rows": len(parsed),
                "raw_columns": ",".join(pd.DataFrame(raw_rows).columns.astype(str)) if raw_rows else "",
                "raw_numeric_columns": ",".join(numeric_columns_from_rows(raw_rows)),
                **summary,
                "first_start": first.earliest if first else "",
                "first_end": first.latest if first else "",
                "last_start": last.earliest if last else "",
                "last_end": last.latest if last else "",
                "target_start": format_timestamp_ms(target_start_ms),
                "can_backfill_to_target_start": bool(
                    pages
                    and target_start_ms is not None
                    and last is not None
                    and last.earliest_ms is not None
                    and last.earliest_ms <= target_start_ms
                ),
                "sample_row": sample_text(raw_rows),
                "error": error,
            }
        )
        pd.DataFrame(rows).to_csv(manifest_path(args.output), index=False)

    output_path = manifest_path(args.output)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
