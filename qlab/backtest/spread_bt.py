"""
Spread mean-reversion backtest engine (FX pair trade style).

Bar-by-bar simulation with z-score entry/exit, stop loss,
and optional maximum holding period.
"""

import numpy as np
import pandas as pd

from ..metrics import sharpe


def run_spread_backtest(
    spread: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
    stop_z: float = 4.0,
    lookback: int = 60,
    cost_per_trade: float = 0.0,
    max_holding_bars: int = None,
    trading_days_per_year: int = 252,
) -> dict:
    """
    Spread mean-reversion backtest with z-score triggers.

    Parameters
    ----------
    spread : pd.Series
        Spread time series (e.g. price_A - hedge_ratio * price_B).
    entry_z : float
        Absolute z-score threshold for entry.
    exit_z : float
        Z-score threshold for mean-reversion exit. 0 = exit at mean.
    stop_z : float
        Z-score threshold for stop loss. Should be > entry_z.
    lookback : int
        Rolling window for z-score computation.
    cost_per_trade : float
        Round-trip cost per trade in spread units.
    max_holding_bars : int, optional
        Maximum bars to hold before forced exit.
    trading_days_per_year : int
        For Sharpe annualization (252 for FX, 365 for crypto).

    Returns
    -------
    dict
        trades, daily_pnl, equity_curve, sharpe, max_drawdown,
        win_rate, profit_factor, n_trades, avg_holding, avg_pnl.
    """
    spread = spread.dropna()
    if len(spread) < lookback + 10:
        return _empty_result()

    # Rolling z-score
    mu = spread.rolling(lookback).mean()
    sd = spread.rolling(lookback).std()
    z = (spread - mu) / (sd + 1e-10)

    position = 0
    entry_price = 0.0
    entry_bar = 0
    trades = []
    daily_pnl = pd.Series(0.0, index=spread.index)

    for i in range(lookback, len(spread)):
        zi = float(z.iloc[i])
        si = float(spread.iloc[i])

        if not np.isfinite(zi):
            continue

        # Mark-to-market daily PnL
        if position != 0 and i > 0:
            prev_s = float(spread.iloc[i - 1])
            daily_pnl.iloc[i] = position * (si - prev_s)

        if position == 0:
            # ── Entry ──
            if zi > entry_z:
                position = -1  # short spread (expect reversion down)
                entry_price = si
                entry_bar = i
            elif zi < -entry_z:
                position = 1   # long spread (expect reversion up)
                entry_price = si
                entry_bar = i
        else:
            # ── Exit check ──
            should_exit = False
            exit_reason = ""

            # Mean reversion
            if position == 1 and zi >= -exit_z:
                should_exit = True
                exit_reason = "reversion"
            elif position == -1 and zi <= exit_z:
                should_exit = True
                exit_reason = "reversion"

            # Stop loss
            if position == 1 and zi < -stop_z:
                should_exit = True
                exit_reason = "stop"
            elif position == -1 and zi > stop_z:
                should_exit = True
                exit_reason = "stop"

            # Max holding
            if max_holding_bars is not None and (i - entry_bar) >= max_holding_bars:
                should_exit = True
                exit_reason = "max_hold"

            if should_exit:
                pnl = position * (si - entry_price) - cost_per_trade
                trades.append({
                    "entry_date": spread.index[entry_bar],
                    "exit_date": spread.index[i],
                    "position": position,
                    "entry_price": entry_price,
                    "exit_price": si,
                    "entry_z": float(z.iloc[entry_bar]),
                    "exit_z": zi,
                    "pnl": pnl,
                    "holding_bars": i - entry_bar,
                    "exit_reason": exit_reason,
                })
                position = 0

    if not trades:
        return _empty_result()

    df = pd.DataFrame(trades)
    dpnl = daily_pnl.iloc[lookback:]

    # Sharpe on daily PnL (holding_days=1 since truly daily)
    dpnl_arr = dpnl.values
    if len(dpnl_arr) > 1 and np.std(dpnl_arr) > 0:
        sr = sharpe(dpnl_arr, holding_days=1,
                    trading_days_per_year=trading_days_per_year)
    else:
        sr = np.nan

    # Max peak-to-trough drawdown in PnL units
    cum = dpnl.cumsum()
    peak = cum.cummax()
    mdd = float((cum - peak).min())

    # Profit factor
    pos_pnl = df.loc[df["pnl"] > 0, "pnl"].sum()
    neg_pnl = abs(df.loc[df["pnl"] < 0, "pnl"].sum())
    if neg_pnl > 0:
        pf = float(pos_pnl / neg_pnl)
    elif pos_pnl > 0:
        pf = np.inf
    else:
        pf = np.nan

    return {
        "trades": df,
        "daily_pnl": dpnl,
        "equity_curve": cum,
        "sharpe": sr,
        "max_drawdown": mdd,
        "win_rate": float((df["pnl"] > 0).mean()),
        "profit_factor": pf,
        "n_trades": len(trades),
        "avg_holding": float(df["holding_bars"].mean()),
        "avg_pnl": float(df["pnl"].mean()),
    }


def _empty_result() -> dict:
    return {
        "trades": pd.DataFrame(),
        "daily_pnl": pd.Series(dtype=float),
        "equity_curve": pd.Series(dtype=float),
        "sharpe": np.nan,
        "max_drawdown": 0.0,
        "win_rate": np.nan,
        "profit_factor": np.nan,
        "n_trades": 0,
        "avg_holding": 0.0,
        "avg_pnl": 0.0,
    }
