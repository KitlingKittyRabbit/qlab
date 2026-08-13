from __future__ import annotations

import os

import pandas as pd
import pytest

from qlab.data.crypto.raw_history_store import (
    RAW_METADATA_COLUMNS,
    build_timeseries_cache_payload,
    read_timeseries_history,
    write_timeseries_cache_payload,
    write_timeseries_cache_payload_batch,
    write_timeseries_history,
)


def test_raw_history_round_trip_restores_cache_compatible_frame(tmp_path) -> None:
    index = pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True)
    source = pd.DataFrame({"close": [1.0, 2.0]}, index=index)
    path = tmp_path / "history.csv"
    write_timeseries_history(
        source,
        path,
        metadata={
            "api_version": "v4",
            "source": "fixture",
            "interval": "1d",
            "symbol": "BTC",
            "endpoint": "x",
            "path": "/x",
            "parser": "x",
            "migration_type": "",
        },
    )
    restored = read_timeseries_history(path)
    assert restored.columns.tolist() == ["close"]
    assert restored.index.tolist() == index.tolist()
    assert restored["close"].tolist() == [1.0, 2.0]


def test_raw_history_restore_fails_without_data_columns(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame(
        {"ts": ["2026-01-01"], "source": ["fixture"], "symbol": ["BTC"]}
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no data columns"):
        read_timeseries_history(path)


def test_cache_rebuild_uses_complete_merged_raw_history_without_metadata(
    tmp_path,
) -> None:
    raw_directory = tmp_path / "raw"
    path = raw_directory / "BTC_basis.csv"
    first_index = pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True)
    second_index = pd.to_datetime(["2026-01-02", "2026-01-03"], utc=True)
    metadata = {
        "api_version": "v4",
        "source": "fixture",
        "interval": "1d",
        "symbol": "BTC",
        "endpoint": "basis",
        "path": "/basis",
        "parser": "basis",
        "migration_type": "",
    }
    write_timeseries_history(
        pd.DataFrame({"close_basis": [1.0, 2.0]}, index=first_index),
        path,
        metadata=metadata,
    )
    write_timeseries_history(
        pd.DataFrame({"close_basis": [20.0, 3.0]}, index=second_index),
        path,
        metadata=metadata,
    )

    payload = build_timeseries_cache_payload(raw_directory)
    rebuilt = payload["BTC_basis"]
    assert rebuilt.index.tolist() == pd.to_datetime(
        ["2026-01-01", "2026-01-02", "2026-01-03"], utc=True
    ).tolist()
    assert rebuilt["close_basis"].tolist() == [1.0, 20.0, 3.0]
    assert RAW_METADATA_COLUMNS.isdisjoint(rebuilt.columns)

    cache_path = tmp_path / "cache.pkl"
    write_timeseries_cache_payload(payload, cache_path)
    restored_payload = pd.read_pickle(cache_path)
    pd.testing.assert_frame_equal(restored_payload["BTC_basis"], rebuilt)


def test_cache_rebuild_fails_closed_without_raw_histories(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="no raw history files"):
        build_timeseries_cache_payload(tmp_path)


def test_cache_rebuild_applies_declared_target_start(tmp_path) -> None:
    raw_directory = tmp_path / "raw"
    path = raw_directory / "BTC_liq.csv"
    index = pd.to_datetime(
        ["1964-12-20", "2024-05-31", "2024-06-01", "2024-06-02"],
        utc=True,
    )
    write_timeseries_history(
        pd.DataFrame({"total_liq": [0.0, 0.0, 1.0, 2.0]}, index=index),
        path,
        metadata={
            "api_version": "v4",
            "source": "fixture",
            "interval": "1d",
            "symbol": "BTC",
            "endpoint": "liq",
            "path": "/liq",
            "parser": "liquidation",
            "migration_type": "",
        },
    )

    payload = build_timeseries_cache_payload(
        raw_directory,
        target_start="2024-06-01T00:00:00Z",
    )
    assert payload["BTC_liq"].index.tolist() == pd.to_datetime(
        ["2024-06-01", "2024-06-02"], utc=True
    ).tolist()
    restored = read_timeseries_history(
        path,
        target_start="2024-06-01T00:00:00Z",
    )
    pd.testing.assert_frame_equal(restored, payload["BTC_liq"])


def test_cache_writer_rejects_metadata_columns(tmp_path) -> None:
    index = pd.to_datetime(["2026-01-01"], utc=True)
    payload = {
        "BTC_basis": pd.DataFrame(
            {"close_basis": [1.0], "source": ["fixture"]},
            index=index,
        )
    }
    with pytest.raises(ValueError, match="raw metadata columns"):
        write_timeseries_cache_payload(payload, tmp_path / "cache.pkl")


def test_cache_batch_rolls_back_when_later_publish_fails(
    tmp_path,
    monkeypatch,
) -> None:
    index = pd.to_datetime(["2026-01-01"], utc=True)
    first = tmp_path / "first.pkl"
    second = tmp_path / "second.pkl"
    write_timeseries_cache_payload(
        {"BTC_basis": pd.DataFrame({"value": [1.0]}, index=index)},
        first,
    )
    write_timeseries_cache_payload(
        {"BTC_basis": pd.DataFrame({"value": [2.0]}, index=index)},
        second,
    )
    original_replace = os.replace
    staged_publish_count = 0

    def fail_second_staged_publish(source, destination):
        nonlocal staged_publish_count
        if str(source).endswith(".staged"):
            staged_publish_count += 1
            if staged_publish_count == 2:
                raise OSError("simulated second publish failure")
        return original_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_second_staged_publish)
    with pytest.raises(OSError, match="simulated second publish failure"):
        write_timeseries_cache_payload_batch(
            {
                first: {
                    "BTC_basis": pd.DataFrame({"value": [10.0]}, index=index)
                },
                second: {
                    "BTC_basis": pd.DataFrame({"value": [20.0]}, index=index)
                },
            }
        )

    assert pd.read_pickle(first)["BTC_basis"]["value"].tolist() == [1.0]
    assert pd.read_pickle(second)["BTC_basis"]["value"].tolist() == [2.0]


def test_cache_batch_rolls_back_when_commit_callback_fails(tmp_path) -> None:
    index = pd.to_datetime(["2026-01-01"], utc=True)
    destination = tmp_path / "cache.pkl"
    write_timeseries_cache_payload(
        {"BTC_basis": pd.DataFrame({"value": [1.0]}, index=index)},
        destination,
    )

    def fail_commit() -> None:
        raise OSError("simulated manifest publish failure")

    with pytest.raises(OSError, match="simulated manifest publish failure"):
        write_timeseries_cache_payload_batch(
            {
                destination: {
                    "BTC_basis": pd.DataFrame({"value": [2.0]}, index=index)
                }
            },
            after_publish=fail_commit,
        )
    assert pd.read_pickle(destination)["BTC_basis"]["value"].tolist() == [1.0]
