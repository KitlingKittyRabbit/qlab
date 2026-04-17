"""
Fixed-period signal backtest engine.

Evaluates signal every `holding_days`, enters position if non-zero,
holds for exactly `holding_days`, then exits. This design eliminates
the overlapping-return Sharpe inflation bug.
"""

import numpy as np
import pandas as pd

from ..metrics import sharpe, max_drawdown, win_rate, profit_factor


def run_signal_backtest(
    signals: pd.Series,
    prices: pd.Series,
    holding_days: int = 14,
    cost_bps: float = 5.0,
    trading_days_per_year: int = 365,
) -> dict:
    """
    Non-overlapping fixed-period backtest.

    Parameters
    ----------
    signals : pd.Series
        Position signal (+1/0/-1) with DatetimeIndex.
    prices : pd.Series
        Close prices with DatetimeIndex.
    holding_days : int
        Fixed holding period and evaluation interval.
    cost_bps : float
        One-way transaction cost in basis points.
    trading_days_per_year : int
        365 for crypto, 252 for FX/equities.

    Returns
    -------
    dict
        trades, returns, sharpe, max_drawdown, win_rate, profit_factor,
        n_trades, avg_return_pct.
    """
    common = signals.dropna().index.intersection(
        prices.dropna().index
    ).sort_values()

    if len(common) < holding_days + 1:
        return _empty_result()

    signals = signals.loc[common]
    prices = prices.loc[common]

    trades = []
    i = 0

    while i + holding_days < len(common):
        entry_date = common[i]
        exit_idx = i + holding_days
        exit_date = common[exit_idx]

        pos = int(signals.iloc[i])
        entry_price = float(prices.iloc[i])
        exit_price = float(prices.iloc[exit_idx])

        if pos != 0 and entry_price > 0:
            gross_ret = pos * (exit_price / entry_price - 1)
            cost = 2 * cost_bps / 10_000
            net_ret = gross_ret - cost

            trades.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "position": pos,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_ret,
                "net_return": net_ret,
            })

        i += holding_days  # always advance, even if flat

    if not trades:
        return _empty_result()

    df = pd.DataFrame(trades)
    returns = df["net_return"].values

    return {
        "trades": df,
        "returns": returns,
        "sharpe": sharpe(
            returns,
            holding_days=holding_days,
            trading_days_per_year=trading_days_per_year,
        ),
        "max_drawdown": max_drawdown(np.cumprod(1 + returns)),
        "win_rate": win_rate(returns),
        "profit_factor": profit_factor(returns),
        "n_trades": len(trades),
        "avg_return_pct": float(np.mean(returns) * 100),
    }


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
    }
