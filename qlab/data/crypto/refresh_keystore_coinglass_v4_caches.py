from __future__ import annotations

"""Lifecycle: candidate.

Build candidate KeyStore/CoinGlass v4 raw history and cache payloads. This is
data-source infrastructure only; it does not run research gates or authorize
live use. Promote to active after pagination, coverage, and overlap validation.
"""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import os
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

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
from qlab.data.crypto.paths import (  # noqa: E402
    CACHE_DIR,
    RAW_HISTORY_ROOT,
    cache_path,
    ensure_data_dirs,
    manifest_path,
)
from qlab.data.crypto.raw_history_store import (  # noqa: E402
    build_timeseries_cache_payload,
    read_timeseries_history,
    write_timeseries_cache_payload_batch,
    write_timeseries_history,
)
from qlab.data.crypto.symbol_universe import RESEARCH_SYMBOLS_12, resolve_target_symbols  # noqa: E402


CACHE_TARGET_START = pd.Timestamp("2024-06-01T00:00:00Z")


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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Call the API, merge raw history, and update cache files.",
    )
    mode.add_argument(
        "--rebuild-cache-from-raw",
        action="store_true",
        help="Rebuild complete cache files from existing raw-history CSVs without API calls.",
    )
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
    parser.add_argument(
        "--replace-cache",
        action="store_true",
        help="Deprecated compatibility flag; touched intervals are always rebuilt from complete raw history.",
    )
    return parser


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new_summary_manifest(summary: pd.DataFrame, destination: Path) -> None:
    """Write a new immutable run manifest without replacing prior evidence."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_suffix(destination.suffix + ".lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refresh summary is locked by another writer: {destination}"
        ) from exc
    os.close(lock_fd)
    temporary = destination.with_name(
        f".{destination.name}.{uuid4().hex}.staged"
    )
    try:
        if destination.exists():
            raise FileExistsError(f"refresh summary already exists: {destination}")
        summary.to_csv(temporary, index=False)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refresh summary already exists: {destination}"
            ) from exc
    finally:
        for disposable in (temporary, lock_path):
            try:
                disposable.unlink(missing_ok=True)
            except OSError:
                pass


@contextmanager
def cache_refresh_run_lock(lock_path: Path):
    """Reject concurrent raw/cache refresh runs regardless of manifest name."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another KeyStore cache refresh is already running: {lock_path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def active_cache_intervals(
    touched_intervals: tuple[str, ...],
    *,
    cache_directory: Path = CACHE_DIR,
) -> tuple[str, ...]:
    """Include every existing active cache so prior failed raw writes are synced."""
    prefix = "keystore_coinglass_v4_"
    suffix = "_cache.pkl"
    existing = {
        path.name[len(prefix) : -len(suffix)]
        for path in cache_directory.glob(f"{prefix}*{suffix}")
        if path.name.startswith(prefix) and path.name.endswith(suffix)
    }
    return tuple(sorted(existing.union(touched_intervals)))


def rebuild_caches_from_raw(
    *,
    intervals: tuple[str, ...],
    summary_output: str,
    target_start: pd.Timestamp,
) -> None:
    summary_rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, pd.DataFrame]] = {}
    for interval in intervals:
        raw_directory = RAW_HISTORY_ROOT / "keystore_v4" / interval
        payload = build_timeseries_cache_payload(
            raw_directory,
            target_start=target_start,
        )
        payloads[interval] = payload

    destinations: dict[Path, dict[str, pd.DataFrame]] = {
        cache_path(f"keystore_coinglass_v4_{interval}_cache.pkl"): payload
        for interval, payload in payloads.items()
    }
    summary_path = manifest_path(summary_output)

    def publish_manifest() -> None:
        for interval, payload in payloads.items():
            raw_directory = RAW_HISTORY_ROOT / "keystore_v4" / interval
            output_path = cache_path(f"keystore_coinglass_v4_{interval}_cache.pkl")
            cache_sha256 = file_sha256(output_path)
            for cache_key, frame in payload.items():
                source = raw_directory / f"{cache_key}.csv"
                summary_rows.append(
                    {
                        "lifecycle": "active cache rebuild evidence",
                        "authority": "raw-history-to-cache reconstruction only",
                        "interval": interval,
                        "cache_key": cache_key,
                        "cache_target_start": target_start,
                        "rows": len(frame),
                        "start": frame.index.min(),
                        "end": frame.index.max(),
                        "data_columns": "|".join(
                            str(column) for column in frame.columns
                        ),
                        "raw_source": str(source),
                        "raw_source_sha256": file_sha256(source),
                        "cache_path": str(output_path),
                        "cache_sha256": cache_sha256,
                        "error": "",
                    }
                )
        write_new_summary_manifest(pd.DataFrame(summary_rows), summary_path)

    write_timeseries_cache_payload_batch(
        destinations,
        after_publish=publish_manifest,
    )
    print(f"Saved {summary_path}")


def run(args: argparse.Namespace) -> None:
    ensure_data_dirs()
    symbols = resolve_target_symbols(args.symbols, default=RESEARCH_SYMBOLS_12)
    intervals = tuple(parse_csv(args.intervals) or KEYSTORE_NATIVE_INTERVALS)
    endpoint_names = parse_csv(args.endpoints)
    roles = tuple(parse_csv(args.roles) or ["base_replacement"])
    endpoints = select_endpoints(names=endpoint_names or None, roles=roles)
    end_time_ms = parse_timestamp_ms(args.end) if args.end else None
    target_start_ms = parse_timestamp_ms(args.target_start) if args.target_start else None
    if (args.execute or args.rebuild_cache_from_raw) and target_start_ms is None:
        raise ValueError(
            "--target-start is required for cache writes so the active cache "
            "cannot inherit unbounded upstream padding"
        )
    if (args.execute or args.rebuild_cache_from_raw) and manifest_path(
        args.summary_output
    ).exists():
        raise FileExistsError(
            "refresh summary already exists; choose a new --summary-output: "
            + str(manifest_path(args.summary_output))
        )

    if args.rebuild_cache_from_raw:
        requested_start = pd.to_datetime(target_start_ms, unit="ms", utc=True)
        if requested_start != CACHE_TARGET_START:
            raise ValueError(
                "raw cache rebuild must use the canonical cache target start: "
                + CACHE_TARGET_START.isoformat()
            )
        rebuild_caches_from_raw(
            intervals=intervals,
            summary_output=args.summary_output,
            target_start=CACHE_TARGET_START,
        )
        return

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
        raw_destination = (
            RAW_HISTORY_ROOT
            / "keystore_v4"
            / item["interval"]
            / f"{item['cache_key']}.csv"
        )
        write_timeseries_history(
            frame=frame,
            destination=raw_destination,
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
        if not frame.empty:
            payload_by_interval[item["interval"]][item["cache_key"]] = (
                read_timeseries_history(
                    raw_destination,
                    target_start=pd.to_datetime(target_start_ms, unit="ms", utc=True),
                )
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

    summary = pd.DataFrame(summary_rows)
    summary_path = manifest_path(args.summary_output)
    failure_count = int(summary["error"].fillna("").astype(str).str.strip().ne("").sum())
    empty_count = int(summary["rows"].fillna(0).astype(int).eq(0).sum())
    if failure_count or empty_count:
        summary["cache_update_status"] = "blocked_by_fetch_failure"
        summary["cache_update_error"] = ""
        write_new_summary_manifest(summary, summary_path)
        print(f"Saved {summary_path}")
        raise RuntimeError(
            "KeyStore refresh failed closed before cache update: "
            f"failures={failure_count}, empty_results={empty_count}, summary={summary_path}"
        )

    try:
        touched_intervals = tuple(
            interval for interval, payload in payload_by_interval.items() if payload
        )
        rebuild_intervals = active_cache_intervals(touched_intervals)
        cache_payloads = {
            cache_path(f"keystore_coinglass_v4_{interval}_cache.pkl"):
            build_timeseries_cache_payload(
                RAW_HISTORY_ROOT / "keystore_v4" / interval,
                target_start=CACHE_TARGET_START,
            )
            for interval in rebuild_intervals
        }

        def publish_success_manifest() -> None:
            summary["cache_update_status"] = "completed"
            summary["cache_update_error"] = ""
            write_new_summary_manifest(summary, summary_path)

        write_timeseries_cache_payload_batch(
            cache_payloads,
            after_publish=publish_success_manifest,
        )
        for output_path in cache_payloads:
            print(f"Saved {output_path}")
    except Exception as exc:
        summary["cache_update_status"] = "failed"
        summary["cache_update_error"] = str(exc)
        write_new_summary_manifest(summary, summary_path)
        print(f"Saved {summary_path}")
        raise

    print(f"Saved {summary_path}")


def main() -> None:
    args = build_parser().parse_args()
    if args.execute or args.rebuild_cache_from_raw:
        with cache_refresh_run_lock(
            manifest_path(".keystore_coinglass_v4_cache_refresh.lock")
        ):
            run(args)
        return
    run(args)


if __name__ == "__main__":
    main()
