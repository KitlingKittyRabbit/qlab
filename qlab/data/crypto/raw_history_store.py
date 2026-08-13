from __future__ import annotations

from datetime import datetime, timezone

UTC = timezone.utc
import os
import pickle
from pathlib import Path
from typing import Callable, Mapping
from uuid import uuid4

import pandas as pd

from qlab.data.crypto.paths import RAW_HISTORY_ROOT


RAW_METADATA_COLUMNS = {
    "api_version",
    "source",
    "interval",
    "symbol",
    "endpoint",
    "path",
    "parser",
    "migration_type",
    "fetched_at",
}


def _normalize_table(frame: pd.DataFrame, timestamp_col: str = "ts") -> pd.DataFrame:
    table = frame.copy()
    if isinstance(table.index, pd.DatetimeIndex):
        table = table.reset_index()
        index_col = table.columns[0]
        table = table.rename(columns={index_col: timestamp_col})
    else:
        table = table.reset_index(drop=True)

    if timestamp_col in table.columns:
        table[timestamp_col] = pd.to_datetime(
            table[timestamp_col], utc=True, errors="coerce")
    return table


def write_timeseries_history(
    frame: pd.DataFrame,
    destination: Path,
    metadata: dict[str, object],
    timestamp_col: str = "ts",
    dedupe_subset: list[str] | None = None,
) -> None:
    if frame.empty:
        return

    table = _normalize_table(frame, timestamp_col=timestamp_col)
    for key, value in metadata.items():
        table[key] = value
    table["fetched_at"] = datetime.now(UTC).isoformat()

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = pd.read_csv(destination)
        if timestamp_col in existing.columns:
            existing[timestamp_col] = pd.to_datetime(
                existing[timestamp_col], utc=True, errors="coerce")
        combined = pd.concat([existing, table], ignore_index=True)
    else:
        combined = table

    subset = dedupe_subset or (
        [timestamp_col] if timestamp_col in combined.columns else None)
    if subset:
        combined = combined.drop_duplicates(subset=subset, keep="last")
    if timestamp_col in combined.columns:
        combined = combined.sort_values(timestamp_col, kind="stable")

    combined.to_csv(destination, index=False)


def append_snapshot_history(
    frame: pd.DataFrame,
    destination: Path,
    metadata: dict[str, object],
    timestamp_col: str = "ts",
) -> None:
    if frame.empty:
        return

    table = _normalize_table(frame, timestamp_col=timestamp_col)
    for key, value in metadata.items():
        table[key] = value
    table["fetched_at"] = datetime.now(UTC).isoformat()

    destination.parent.mkdir(parents=True, exist_ok=True)
    header = not destination.exists()
    table.to_csv(destination, mode="a", index=False, header=header)


def read_timeseries_history(
    source: Path,
    *,
    timestamp_col: str = "ts",
    target_start: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Restore a cache-compatible frame from an append-only raw history CSV."""
    if not source.exists():
        raise FileNotFoundError(source)
    table = pd.read_csv(source)
    if timestamp_col not in table.columns:
        raise ValueError(f"raw history missing timestamp column: {timestamp_col}")
    timestamps = pd.to_datetime(table.pop(timestamp_col), utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"raw history contains invalid timestamps: {source}")
    data_columns = [
        column for column in table.columns if column not in RAW_METADATA_COLUMNS
    ]
    if not data_columns:
        raise ValueError(f"raw history has no data columns: {source}")
    restored = table[data_columns].copy()
    restored.index = pd.DatetimeIndex(timestamps, name="ts")
    restored = restored[~restored.index.duplicated(keep="last")].sort_index()
    if target_start is not None:
        start = pd.Timestamp(target_start)
        start = start.tz_localize("UTC") if start.tz is None else start.tz_convert("UTC")
        restored = restored.loc[restored.index >= start]
    return restored


def build_timeseries_cache_payload(
    raw_directory: Path,
    *,
    pattern: str = "*.csv",
    target_start: str | pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    """Build a cache payload from complete raw histories, excluding metadata."""
    sources = sorted(raw_directory.glob(pattern))
    if not sources:
        raise FileNotFoundError(
            f"no raw history files matching {pattern!r}: {raw_directory}"
        )

    payload: dict[str, pd.DataFrame] = {}
    for source in sources:
        cache_key = source.stem
        if cache_key in payload:
            raise ValueError(f"duplicate raw history cache key: {cache_key}")
        frame = read_timeseries_history(source, target_start=target_start)
        if frame.empty:
            raise ValueError(f"raw history restored to an empty frame: {source}")
        payload[cache_key] = frame
    return payload


def write_timeseries_cache_payload(
    payload: Mapping[str, pd.DataFrame],
    destination: Path,
) -> None:
    """Atomically persist one validated cache payload."""
    write_timeseries_cache_payload_batch({destination: payload})


def _validated_cache_payload(
    payload: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    if not payload:
        raise ValueError("cache payload must not be empty")

    normalized: dict[str, pd.DataFrame] = {}
    for cache_key, frame in payload.items():
        if not cache_key:
            raise ValueError("cache key must not be empty")
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"cache frame must be a non-empty DataFrame: {cache_key}")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError(f"cache frame must use a DatetimeIndex: {cache_key}")
        if frame.index.has_duplicates:
            raise ValueError(f"cache frame contains duplicate timestamps: {cache_key}")
        if not frame.index.is_monotonic_increasing:
            raise ValueError(f"cache frame timestamps are not monotonic: {cache_key}")
        metadata_columns = sorted(RAW_METADATA_COLUMNS.intersection(frame.columns))
        if metadata_columns:
            raise ValueError(
                f"cache frame contains raw metadata columns for {cache_key}: "
                + ", ".join(metadata_columns)
            )
        normalized[str(cache_key)] = frame
    return normalized


def write_timeseries_cache_payload_batch(
    payloads: Mapping[Path, Mapping[str, pd.DataFrame]],
    *,
    after_publish: Callable[[], None] | None = None,
) -> None:
    """Stage and publish a cache batch, rolling back on publish exceptions."""
    if not payloads:
        raise ValueError("cache payload batch must not be empty")

    transaction_id = uuid4().hex
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for destination, payload in payloads.items():
            normalized = _validated_cache_payload(payload)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(
                f".{destination.name}.{transaction_id}.staged"
            )
            with temporary.open("xb") as handle:
                pickle.dump(normalized, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            staged[destination] = temporary

        for destination in staged:
            if destination.exists():
                backup = destination.with_name(
                    f".{destination.name}.{transaction_id}.backup"
                )
                os.link(destination, backup)
                backups[destination] = backup

        for destination, temporary in staged.items():
            os.replace(temporary, destination)
            published.append(destination)
        if after_publish is not None:
            after_publish()
    except Exception:
        for destination in reversed(published):
            backup = backups.get(destination)
            if backup is not None and backup.exists():
                os.replace(backup, destination)
            else:
                destination.unlink(missing_ok=True)
        raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)
