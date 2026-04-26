from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from qlab.data.crypto.paths import RAW_HISTORY_ROOT


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
