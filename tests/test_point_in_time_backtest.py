from __future__ import annotations

import pandas as pd
import pytest

from qlab.backtest import run_point_in_time_event_backtest
from qlab.point_in_time import PointInTimeSemantics


def test_point_in_time_event_backtest_waits_for_finalized_bar_before_entry():
    signals = pd.Series(
        [1.0],
        index=pd.DatetimeIndex(["2026-05-23T12:00:00Z"]),
    )
    prices = pd.Series(
        [90.0, 99.0, 100.0, 109.0, 110.0],
        index=pd.DatetimeIndex(
            [
                "2026-05-23T12:15:00Z",
                "2026-05-24T00:15:00Z",
                "2026-05-24T00:30:00Z",
                "2026-05-24T12:30:00Z",
                "2026-05-24T12:45:00Z",
            ]
        ),
    )

    result = run_point_in_time_event_backtest(
        signals=signals,
        prices=prices,
        semantics=PointInTimeSemantics(
            timestamp_kind="bar_start", value_status="final"),
        signal_bar_duration=pd.Timedelta(hours=12),
        holding_period=pd.Timedelta(hours=12),
        decision_delay=pd.Timedelta(minutes=15),
        cost_bps=0.0,
    )

    assert result["n_trades"] == 1
    trade = result["trades"].iloc[0]
    assert trade["requested_entry_time"] == pd.Timestamp(
        "2026-05-24T00:15:00Z")
    assert trade["entry_time"] == pd.Timestamp("2026-05-24T00:30:00Z")
    assert trade["requested_exit_time"] == pd.Timestamp("2026-05-24T12:30:00Z")
    assert trade["exit_time"] == pd.Timestamp("2026-05-24T12:45:00Z")
    assert trade["gross_return"] == pytest.approx(0.1)


def test_point_in_time_event_backtest_skips_overlapping_signals_by_default():
    signals = pd.Series(
        [1.0, 1.0],
        index=pd.DatetimeIndex(
            ["2026-05-23T00:00:00Z", "2026-05-23T12:00:00Z"]),
    )
    prices = pd.Series(
        [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
        index=pd.DatetimeIndex(
            [
                "2026-05-23T00:15:00Z",
                "2026-05-23T00:30:00Z",
                "2026-05-23T12:15:00Z",
                "2026-05-23T12:30:00Z",
                "2026-05-24T00:15:00Z",
                "2026-05-24T00:30:00Z",
                "2026-05-24T00:45:00Z",
            ]
        ),
    )

    result = run_point_in_time_event_backtest(
        signals=signals,
        prices=prices,
        semantics=PointInTimeSemantics(
            timestamp_kind="bar_start",
            value_status="partial",
            publication_lag=pd.Timedelta(0),
        ),
        signal_bar_duration=pd.Timedelta(hours=12),
        holding_period=pd.Timedelta(hours=24),
        decision_delay=pd.Timedelta(minutes=15),
        cost_bps=0.0,
    )

    assert result["n_trades"] == 1
    assert result["skipped_overlap"] == 1


def test_point_in_time_event_backtest_allows_same_timestamp_fill_only_when_execution_lag_is_zero():
    signals = pd.Series(
        [1.0],
        index=pd.DatetimeIndex(["2026-05-23T12:00:00Z"]),
    )
    prices = pd.Series(
        [99.0, 110.0],
        index=pd.DatetimeIndex(
            [
                "2026-05-24T00:15:00Z",
                "2026-05-24T12:15:00Z",
            ]
        ),
    )

    result = run_point_in_time_event_backtest(
        signals=signals,
        prices=prices,
        semantics=PointInTimeSemantics(
            timestamp_kind="bar_start", value_status="final"),
        signal_bar_duration=pd.Timedelta(hours=12),
        holding_period=pd.Timedelta(hours=12),
        decision_delay=pd.Timedelta(minutes=15),
        execution_lag=pd.Timedelta(0),
        cost_bps=0.0,
    )

    trade = result["trades"].iloc[0]
    assert trade["entry_time"] == pd.Timestamp("2026-05-24T00:15:00Z")
    assert trade["exit_time"] == pd.Timestamp("2026-05-24T12:15:00Z")


def test_point_in_time_event_backtest_rejects_partial_semantics_without_explicit_lag():
    signals = pd.Series(
        [1.0],
        index=pd.DatetimeIndex(["2026-05-23T12:00:00Z"]),
    )
    prices = pd.Series(
        [100.0, 110.0],
        index=pd.DatetimeIndex(
            [
                "2026-05-23T12:15:00Z",
                "2026-05-24T00:15:00Z",
            ]
        ),
    )

    with pytest.raises(ValueError, match="explicit availability"):
        run_point_in_time_event_backtest(
            signals=signals,
            prices=prices,
            semantics=PointInTimeSemantics(
                timestamp_kind="bar_start", value_status="partial"),
            signal_bar_duration=pd.Timedelta(hours=12),
            holding_period=pd.Timedelta(hours=12),
            decision_delay=pd.Timedelta(minutes=15),
            cost_bps=0.0,
        )


def test_point_in_time_event_backtest_bar_start_delay_matches_relabel_to_bar_end():
    start_labeled_signals = pd.Series(
        [1.0, -1.0],
        index=pd.DatetimeIndex(
            [
                "2026-05-23T00:00:00Z",
                "2026-05-23T12:00:00Z",
            ]
        ),
    )
    end_labeled_signals = pd.Series(
        start_labeled_signals.values,
        index=start_labeled_signals.index + pd.Timedelta(hours=12),
    )
    prices = pd.Series(
        [100.0, 110.0, 105.0, 95.0],
        index=pd.DatetimeIndex(
            [
                "2026-05-23T12:15:00Z",
                "2026-05-24T00:15:00Z",
                "2026-05-24T12:15:00Z",
                "2026-05-25T00:15:00Z",
            ]
        ),
    )

    start_labeled_result = run_point_in_time_event_backtest(
        signals=start_labeled_signals,
        prices=prices,
        semantics=PointInTimeSemantics(
            timestamp_kind="bar_start", value_status="final"),
        signal_bar_duration=pd.Timedelta(hours=12),
        holding_period=pd.Timedelta(hours=12),
        decision_delay=pd.Timedelta(minutes=15),
        execution_lag=pd.Timedelta(0),
        cost_bps=0.0,
    )
    end_labeled_result = run_point_in_time_event_backtest(
        signals=end_labeled_signals,
        prices=prices,
        semantics=PointInTimeSemantics(
            timestamp_kind="bar_end", value_status="final"),
        signal_bar_duration=pd.Timedelta(hours=12),
        holding_period=pd.Timedelta(hours=12),
        decision_delay=pd.Timedelta(minutes=15),
        execution_lag=pd.Timedelta(0),
        cost_bps=0.0,
    )

    assert start_labeled_result["n_trades"] == 2
    assert end_labeled_result["n_trades"] == 2
    pd.testing.assert_frame_equal(
        start_labeled_result["trades"][
            [
                "position",
                "requested_entry_time",
                "entry_time",
                "requested_exit_time",
                "exit_time",
                "gross_return",
            ]
        ].reset_index(drop=True),
        end_labeled_result["trades"][
            [
                "position",
                "requested_entry_time",
                "entry_time",
                "requested_exit_time",
                "exit_time",
                "gross_return",
            ]
        ].reset_index(drop=True),
    )
