"""Point-in-time safe event backtests.

This engine is designed for timestamped factor events where the signal label is
not automatically the same thing as the first tradable timestamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrics import max_drawdown, profit_factor, sharpe, win_rate
from ..point_in_time import PointInTimeSemantics
from ..signal import threshold_signal


def _empty_result() -> dict:
    return {
        "trades": pd.DataFrame(),
        "returns": np.array([]),
        "sharpe": np.nan,
        "max_drawdown": 0.0,
        "win_rate": np.nan,
        "profit_factor": np.nan,
        "n_trades": 0,
        "avg_return_pct": 0.0,
        "skipped_untradeable": 0,
        "skipped_overlap": 0,
    }


def _next_tradeable_time(index: pd.DatetimeIndex, target: pd.Timestamp) -> pd.Timestamp | None:
    if index.empty:
        return None
    position = index.searchsorted(target, side="left")
    if position >= len(index):
        return None
    return pd.Timestamp(index[position])


def run_point_in_time_event_backtest(
    signals: pd.Series,
    prices: pd.Series,
    semantics: PointInTimeSemantics,
    signal_bar_duration: pd.Timedelta,
    holding_period: pd.Timedelta,
    decision_delay: pd.Timedelta = pd.Timedelta(0),
    execution_lag: pd.Timedelta = pd.Timedelta(microseconds=1),
    signal_threshold: float = 0.0,
    cost_bps: float = 5.0,
    trading_days_per_year: int = 365,
    allow_overlap: bool = False,
) -> dict:
    """Run a point-in-time safe event backtest on timestamped signals.

    Signals are converted to discrete positions with ``threshold_signal``. The
    engine derives the earliest safe decision time from ``semantics`` and then
    snaps to the first tradeable timestamp strictly after that instant unless
    the caller explicitly overrides ``execution_lag``.
    """

    if not semantics.availability_contract_is_explicit():
        raise ValueError(
            "point-in-time semantics must define explicit availability before backtesting"
        )
    execution_lag = pd.Timedelta(execution_lag)
    if execution_lag < pd.Timedelta(0):
        raise ValueError("execution_lag must be >= 0")

    signals = signals.dropna().sort_index().astype(float)
    prices = prices.dropna().sort_index().astype(float)
    if signals.empty or prices.empty:
        return _empty_result()

    positions = threshold_signal(signals.values, threshold=signal_threshold)
    trades = []
    skipped_untradeable = 0
    skipped_overlap = 0
    active_until: pd.Timestamp | None = None

    for signal_time, signal_value, position in zip(signals.index, signals.values, positions):
        if position == 0:
            continue

        requested_entry = semantics.earliest_safe_decision_time(
            signal_time,
            signal_bar_duration,
            decision_delay,
        )
        if requested_entry is None:
            raise ValueError("semantics did not yield an entry time")

        actual_entry = _next_tradeable_time(
            prices.index, requested_entry + execution_lag)
        if actual_entry is None:
            skipped_untradeable += 1
            continue
        if not allow_overlap and active_until is not None and actual_entry < active_until:
            skipped_overlap += 1
            continue

        requested_exit = actual_entry + pd.Timedelta(holding_period)
        actual_exit = _next_tradeable_time(
            prices.index, requested_exit + execution_lag)
        if actual_exit is None:
            skipped_untradeable += 1
            continue

        entry_price = float(prices.loc[actual_entry])
        exit_price = float(prices.loc[actual_exit])
        gross_ret = int(position) * (exit_price / entry_price - 1)
        net_ret = gross_ret - 2 * cost_bps / 10_000
        trades.append(
            {
                "signal_time": pd.Timestamp(signal_time),
                "signal_value": float(signal_value),
                "position": int(position),
                "requested_entry_time": requested_entry,
                "entry_time": actual_entry,
                "requested_exit_time": requested_exit,
                "exit_time": actual_exit,
                "execution_lag": execution_lag,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_ret,
                "net_return": net_ret,
            }
        )
        active_until = actual_exit

    if not trades:
        result = _empty_result()
        result["skipped_untradeable"] = skipped_untradeable
        result["skipped_overlap"] = skipped_overlap
        return result

    trade_frame = pd.DataFrame(trades)
    returns = trade_frame["net_return"].to_numpy(dtype=float)
    holding_days = max(
        float(pd.Timedelta(holding_period) / pd.Timedelta(days=1)), 1e-12)

    return {
        "trades": trade_frame,
        "returns": returns,
        "sharpe": sharpe(
            returns,
            holding_days=holding_days,
            trading_days_per_year=trading_days_per_year,
        ),
        "max_drawdown": max_drawdown(np.cumprod(1 + returns)),
        "win_rate": win_rate(returns),
        "profit_factor": profit_factor(returns),
        "n_trades": len(trade_frame),
        "avg_return_pct": float(np.mean(returns) * 100),
        "skipped_untradeable": skipped_untradeable,
        "skipped_overlap": skipped_overlap,
    }
