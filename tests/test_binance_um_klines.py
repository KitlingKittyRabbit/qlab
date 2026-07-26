from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from qlab.data.crypto.binance_um_klines import (
    audit_klines,
    execution_opens,
    parse_data_vision_zip,
    read_partitions,
    read_execution_open_partitions,
    write_partition,
)
from qlab.data.crypto.refresh_binance_um_1m import completed_periods


def _archive(rows: list[list[object]]) -> bytes:
    payload = "\n".join(",".join(map(str, row)) for row in rows).encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("BTCUSDT-1m.csv", payload)
    return buffer.getvalue()


def _row(open_time: str, price: float) -> list[object]:
    start = pd.Timestamp(open_time, tz="UTC")
    return [
        int(start.timestamp() * 1000),
        price,
        price + 1,
        price - 1,
        price + 0.5,
        10,
        int((start + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1)).timestamp() * 1000),
        0,
        0,
        0,
        0,
        0,
    ]


def test_parse_preserves_open_and_close_time_semantics() -> None:
    frame = parse_data_vision_zip(
        _archive([_row("2026-01-01 00:00", 100), _row("2026-01-01 00:01", 101)]),
        interval="1m",
        source="fixture",
    )
    assert frame.loc[0, "open_time"] == pd.Timestamp("2026-01-01 00:00", tz="UTC")
    assert frame.loc[0, "close_time"] == pd.Timestamp("2026-01-01 00:00:59.999", tz="UTC")
    assert frame.loc[1, "open"] == 101


def test_partition_round_trip_and_gap_audit(tmp_path) -> None:
    frame = parse_data_vision_zip(
        _archive([_row("2026-01-01 00:00", 100), _row("2026-01-01 00:02", 102)]),
        interval="1m",
        source="fixture",
    )
    path = tmp_path / "period=2026-01.parquet"
    write_partition(frame, path, interval="1m")
    restored = read_partitions([path], interval="1m")
    audit = audit_klines(restored, symbol="BTC", interval="1m")
    assert audit["rows"] == 2
    assert audit["missing_bars"] == 1
    opens = read_execution_open_partitions([path])
    assert opens.loc[pd.Timestamp("2026-01-01 00:02", tz="UTC")] == 102


def test_execution_open_fails_closed_without_exact_minute() -> None:
    frame = parse_data_vision_zip(
        _archive([_row("2026-01-01 00:00", 100), _row("2026-01-01 00:01", 101)]),
        interval="1m",
        source="fixture",
    )
    result = execution_opens(frame, [pd.Timestamp("2026-01-01 00:01", tz="UTC")])
    assert result.iloc[0] == 101
    with pytest.raises(KeyError, match="00:02:00"):
        execution_opens(frame, [pd.Timestamp("2026-01-01 00:02", tz="UTC")])


def test_close_time_mismatch_fails_closed() -> None:
    row = _row("2026-01-01 00:00", 100)
    row[6] += 1
    with pytest.raises(ValueError, match="close_time"):
        parse_data_vision_zip(_archive([row]), interval="1m", source="fixture")


def test_completed_periods_use_monthly_archives_then_daily_tail(monkeypatch) -> None:
    class FixedTimestamp(pd.Timestamp):
        @classmethod
        def now(cls, tz=None):
            return cls("2026-07-13", tz=tz)

    monkeypatch.setattr(
        "qlab.data.crypto.refresh_binance_um_1m.pd.Timestamp",
        FixedTimestamp,
    )
    months, days = completed_periods(pd.Timestamp("2026-05-15", tz="UTC"), pd.Timestamp("2026-07-12", tz="UTC"))
    assert months == ["2026-05", "2026-06"]
    assert days == [f"2026-07-{day:02d}" for day in range(1, 13)]

    months, days = completed_periods(pd.Timestamp("2026-06-01", tz="UTC"), pd.Timestamp("2026-06-30", tz="UTC"))
    assert months == ["2026-06"]
    assert days == []
