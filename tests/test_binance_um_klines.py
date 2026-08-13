from __future__ import annotations

import io
import json
import zipfile

import pandas as pd
import pytest
import requests

from qlab.data.crypto.binance_um_klines import (
    aggregate_complete_klines,
    audit_klines,
    canonical_partition_paths,
    download_rest_day_partition,
    execution_opens,
    parse_data_vision_zip,
    parse_rest_klines,
    price_volume_payload_from_klines,
    read_partitions,
    read_execution_open_partitions,
    write_partition,
)
from qlab.data.crypto.refresh_binance_um_1m import (
    completed_periods,
    is_ignorable_latest_day_404,
)


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


def test_parse_rest_preserves_open_and_close_time_semantics() -> None:
    frame = parse_rest_klines(
        [_row("2026-01-01 00:00", 100), _row("2026-01-01 00:01", 101)],
        interval="1m",
        source="fixture",
    )
    assert frame.loc[0, "open_time"] == pd.Timestamp("2026-01-01 00:00", tz="UTC")
    assert frame.loc[0, "close_time"] == pd.Timestamp(
        "2026-01-01 00:00:59.999", tz="UTC"
    )
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


def test_month_partition_supersedes_daily_partitions_for_same_month(tmp_path) -> None:
    monthly = tmp_path / "period=2026-07.parquet"
    daily = tmp_path / "period=2026-07-01.parquet"
    august = tmp_path / "period=2026-08-01.parquet"
    for path in [monthly, daily, august]:
        path.touch()

    assert canonical_partition_paths([daily, august, monthly]) == [monthly, august]


def test_partition_readers_do_not_double_count_superseded_daily_files(tmp_path) -> None:
    monthly = tmp_path / "period=2026-07.parquet"
    daily = tmp_path / "period=2026-07-01.parquet"
    august = tmp_path / "period=2026-08-01.parquet"
    write_partition(
        parse_data_vision_zip(
            _archive([_row("2026-07-01 00:00", 100)]),
            interval="1m",
            source="monthly",
        ),
        monthly,
        interval="1m",
    )
    write_partition(
        parse_data_vision_zip(
            _archive([_row("2026-07-01 00:00", 999)]),
            interval="1m",
            source="daily",
        ),
        daily,
        interval="1m",
    )
    write_partition(
        parse_data_vision_zip(
            _archive([_row("2026-08-01 00:00", 101)]),
            interval="1m",
            source="daily",
        ),
        august,
        interval="1m",
    )

    paths = [daily, august, monthly]
    frame = read_partitions(paths, interval="1m")
    opens = read_execution_open_partitions(paths)
    assert frame["open"].tolist() == [100, 101]
    assert opens.tolist() == [100, 101]


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


def test_complete_minute_aggregation_is_hand_calculable_and_drops_gaps() -> None:
    rows = [_row(f"2026-01-01 00:{minute:02d}", 100 + minute) for minute in range(30)]
    rows.pop(20)
    frame = parse_data_vision_zip(_archive(rows), interval="1m", source="fixture")
    aggregated = aggregate_complete_klines(
        frame, source_interval="1m", target_interval="15m"
    )
    assert len(aggregated) == 1
    row = aggregated.iloc[0]
    assert row["open_time"] == pd.Timestamp("2026-01-01 00:00", tz="UTC")
    assert row["open"] == 100
    assert row["high"] == 115
    assert row["low"] == 99
    assert row["close"] == 114.5
    assert row["volume"] == 150
    assert row["close_time"] == pd.Timestamp("2026-01-01 00:14:59.999", tz="UTC")
    payload = price_volume_payload_from_klines(frame)
    assert payload.loc[pd.Timestamp("2026-01-01 00:00", tz="UTC"), "c"] == 114.5
    assert payload.iloc[0]["v"] == 150


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


def test_only_latest_unpublished_daily_archive_is_ignorable() -> None:
    response = requests.Response()
    response.status_code = 404
    error = requests.HTTPError(response=response)
    days = ["2026-07-24", "2026-07-25", "2026-07-26"]
    assert is_ignorable_latest_day_404(
        error, period="2026-07-26", requested_days=days
    )
    assert not is_ignorable_latest_day_404(
        error, period="2026-07-25", requested_days=days
    )


def test_rest_day_download_requires_exact_complete_day(tmp_path) -> None:
    rows = [
        _row(str(ts), 100 + index)
        for index, ts in enumerate(
            pd.date_range("2026-07-28", periods=1440, freq="1min", tz="UTC")
        )
    ]

    class Response:
        content = json.dumps(rows).encode("utf-8")

        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[list[object]]:
            return rows

    class Session:
        def get(self, url, *, params, timeout):
            assert params["startTime"] == int(
                pd.Timestamp("2026-07-28", tz="UTC").timestamp() * 1000
            )
            assert params["limit"] == 1440
            return Response()

    partition = download_rest_day_partition(
        Session(),
        symbol="BTC",
        interval="1m",
        period="2026-07-28",
        root=tmp_path,
    )
    assert partition.rows == 1440
    assert partition.path.is_file()

    rows.pop()
    with pytest.raises(ValueError, match="incomplete"):
        download_rest_day_partition(
            Session(),
            symbol="ETH",
            interval="1m",
            period="2026-07-28",
            root=tmp_path,
        )
