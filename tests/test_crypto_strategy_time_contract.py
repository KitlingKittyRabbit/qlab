from __future__ import annotations

import pandas as pd
import pytest

from qlab.data.crypto.panel import (
    boundary_reference_prices_from_execution_opens,
    executable_returns_for_symbol,
)
from qlab.data.crypto.strategy_time_contract import (
    ContinuousHoldingTimeContract,
    factor_eligibility_manifest,
    validate_continuous_holding_contract,
    validate_decision_phase,
)
from qlab.factor_research import validated_executable_return_adapter


DELTAS = {
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "8h": pd.Timedelta(hours=8),
    "12h": pd.Timedelta(hours=12),
    "1d": pd.Timedelta(days=1),
}


def _contract(horizon: str = "12h", signals: tuple[str, ...] = ("1h", "12h")) -> ContinuousHoldingTimeContract:
    return ContinuousHoldingTimeContract(
        return_horizon=horizon,
        decision_interval=horizon,
        holding_interval=horizon,
        strategy_return_interval=horizon,
        signal_timeframes=signals,
    )


def _minute_frame() -> pd.DataFrame:
    open_times = pd.to_datetime(
        ["2026-01-01 00:01", "2026-01-01 12:01", "2026-01-02 00:01"],
        utc=True,
    )
    return pd.DataFrame(
        {
            "open_time": open_times,
            "open": [100.0, 110.0, 121.0],
            "high": [101.0, 111.0, 122.0],
            "low": [99.0, 109.0, 120.0],
            "close": [100.5, 110.5, 121.5],
            "volume": [10.0, 10.0, 10.0],
            "close_time": open_times + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1),
            "source": ["fixture"] * 3,
        }
    )


def test_mismatched_horizon_and_decision_interval_fail_closed() -> None:
    contract = ContinuousHoldingTimeContract(
        return_horizon="12h",
        decision_interval="1d",
        holding_interval="1d",
        strategy_return_interval="12h",
        signal_timeframes=("12h",),
    )
    with pytest.raises(ValueError, match="return_horizon"):
        validate_continuous_holding_contract(contract, DELTAS)


@pytest.mark.parametrize("signal", ["1d", "8h"])
def test_stale_or_non_divisor_signal_fails_closed(signal: str) -> None:
    with pytest.raises(ValueError, match="not an exact divisor"):
        validate_continuous_holding_contract(_contract("12h", (signal,)), DELTAS)


def test_wrong_utc_phase_fails_closed() -> None:
    with pytest.raises(ValueError, match="UTC phase"):
        validate_decision_phase([pd.Timestamp("2026-01-01 00:15", tz="UTC")], _contract(), DELTAS)


def test_exact_next_minute_open_to_open_return() -> None:
    result = executable_returns_for_symbol(
        pd.DatetimeIndex([pd.Timestamp("2026-01-01 00:00", tz="UTC")]),
        _minute_frame(),
        _contract(),
        DELTAS,
    )
    assert result.loc[0, "execution_ts"] == pd.Timestamp("2026-01-01 00:01", tz="UTC")
    assert result.loc[0, "next_execution_ts"] == pd.Timestamp("2026-01-01 12:01", tz="UTC")
    assert result.loc[0, "entry_price"] == 100.0
    assert result.loc[0, "exit_price"] == 110.0
    assert result.loc[0, "executable_return"] == pytest.approx(0.10)
    assert result.loc[0, "exit_ts"] == result.loc[0, "next_execution_ts"]


def test_frozen_four_minute_open_to_open_return() -> None:
    contract = ContinuousHoldingTimeContract(
        return_horizon="12h",
        decision_interval="12h",
        holding_interval="12h",
        strategy_return_interval="12h",
        signal_timeframes=("1h", "12h"),
        execution_delay_minutes=4,
    )
    open_times = pd.to_datetime(
        ["2026-01-01 00:04", "2026-01-01 12:04"], utc=True
    )
    minute_frame = pd.DataFrame(
        {
            "open_time": open_times,
            "open": [100.0, 120.0],
            "high": [101.0, 121.0],
            "low": [99.0, 119.0],
            "close": [100.0, 120.0],
            "volume": [1.0, 1.0],
            "close_time": open_times + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1),
            "source": ["fixture", "fixture"],
        }
    )
    result = executable_returns_for_symbol(
        pd.DatetimeIndex([pd.Timestamp("2026-01-01 00:00", tz="UTC")]),
        minute_frame,
        contract,
        DELTAS,
    )
    assert result.loc[0, "execution_ts"] == pd.Timestamp("2026-01-01 00:04", tz="UTC")
    assert result.loc[0, "next_execution_ts"] == pd.Timestamp("2026-01-01 12:04", tz="UTC")
    assert result.loc[0, "executable_return"] == pytest.approx(0.20)


def test_reference_prices_keep_only_exact_boundaries() -> None:
    series = pd.Series(
        [100.0, 101.0, 102.0],
        index=pd.to_datetime(
            ["2026-01-01 00:00", "2026-01-01 00:01", "2026-01-01 00:15"],
            utc=True,
        ),
    )

    result = boundary_reference_prices_from_execution_opens(series)

    assert result.index.tolist() == [
        pd.Timestamp("2026-01-01 00:00", tz="UTC"),
        pd.Timestamp("2026-01-01 00:15", tz="UTC"),
    ]
    assert result["c"].tolist() == [100.0, 102.0]


def test_missing_exit_minute_fails_closed() -> None:
    with pytest.raises(KeyError, match="12:01:00"):
        executable_returns_for_symbol(
            pd.DatetimeIndex([pd.Timestamp("2026-01-01 00:00", tz="UTC")]),
            _minute_frame().iloc[[0]],
            _contract(),
            DELTAS,
        )


def test_factor_eligibility_is_horizon_specific() -> None:
    registry = pd.DataFrame(
        {"feature_name": ["fast", "eight", "daily"], "signal_timeframe": ["1h", "8h", "1d"]}
    )
    result = factor_eligibility_manifest(registry, horizon="12h", horizon_deltas=DELTAS)
    assert result.set_index("feature_name")["admitted"].to_dict() == {
        "fast": True,
        "eight": False,
        "daily": False,
    }
    assert "availability_delay_minutes" not in result.columns
    assert "common_release_phase" not in result.columns


def test_strategy_adapter_rejects_incomplete_or_forged_execution_ledger() -> None:
    ledger = executable_returns_for_symbol(
        pd.DatetimeIndex([pd.Timestamp("2026-01-01 00:00", tz="UTC")]),
        _minute_frame(),
        _contract(),
        DELTAS,
    )
    adapted = validated_executable_return_adapter(
        ledger,
        return_horizon="12h",
        decision_frequency="12h",
        horizon_deltas=DELTAS,
        execution_delay_minutes=1,
    )
    assert adapted.loc[0, "strategy_forward_return"] == pytest.approx(0.10)

    forged = ledger.copy()
    forged["executable_return"] = 0.25
    with pytest.raises(ValueError, match="does not match"):
        validated_executable_return_adapter(
            forged,
            return_horizon="12h",
            decision_frequency="12h",
            horizon_deltas=DELTAS,
            execution_delay_minutes=1,
        )
