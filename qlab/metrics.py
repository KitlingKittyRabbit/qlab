"""
Performance metrics with correct annualization.

Key design: Sharpe ratio requires knowing the holding period
to correctly annualize. There is no safe default — callers must specify.

Common mistake this module prevents:
    Using sqrt(365) for 14-day returns sampled daily inflates Sharpe by
    sqrt(14) ≈ 3.74x due to overlapping returns.

Conventions:
    - Sortino uses downside deviation averaged across all periods.
    - Calmar uses CAGR over the same compounded equity curve used for MaxDD.
    - max_drawdown expects a strictly positive equity / NAV curve.
"""

import warnings
from typing import Union

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.Series, list]


def sharpe(
    returns: ArrayLike,
    holding_days: int = 1,
    trading_days_per_year: int = 365,
    risk_free: float = 0.0,
) -> float:
    """
    Annualized Sharpe ratio.

    Parameters
    ----------
    returns : array-like
        Strategy returns. Must be sampled at `holding_days` intervals
        (non-overlapping). If you have daily-sampled overlapping N-day
        returns, subsample first: ``returns[::N]``.
    holding_days : int
        Return horizon in calendar/trading days.
    trading_days_per_year : int
        365 for crypto (24/7), 252 for equities/FX.
    risk_free : float
        Annualized risk-free rate (default 0).

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return np.nan

    _check_sample_size(r)
    _check_autocorrelation(r, holding_days)

    periods_per_year = trading_days_per_year / holding_days
    rf_per_period = (1 + risk_free) ** (holding_days /
                                        trading_days_per_year) - 1

    excess = r - rf_per_period
    std = np.std(excess, ddof=1)
    if std < 1e-14:
        return np.nan

    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def sortino(
    returns: ArrayLike,
    holding_days: int = 1,
    trading_days_per_year: int = 365,
    risk_free: float = 0.0,
) -> float:
    """
    Annualized Sortino ratio.

    Downside deviation is computed across all periods, with positive
    excess returns contributing zero downside variance.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return np.nan

    _check_sample_size(r)

    periods_per_year = trading_days_per_year / holding_days
    rf_per_period = (1 + risk_free) ** (holding_days /
                                        trading_days_per_year) - 1

    excess = r - rf_per_period
    downside_squared = np.minimum(0.0, excess) ** 2
    if np.all(downside_squared == 0):
        return np.inf
    dd = np.sqrt(np.mean(downside_squared))
    if dd < 1e-14:
        return np.nan

    return float(np.mean(excess) / dd * np.sqrt(periods_per_year))


def max_drawdown(equity: ArrayLike) -> float:
    """
    Maximum drawdown from a strictly positive equity / NAV curve.

    This function is defined for wealth curves such as cumulative NAV,
    not for absolute PnL series that may start at zero or go negative.

    Returns a negative float (e.g. -0.25 for 25% drawdown).
    """
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 2:
        return 0.0
    if np.any(~np.isfinite(eq)):
        raise ValueError("equity must contain only finite values")
    if np.any(eq <= 0):
        raise ValueError(
            "max_drawdown expects a strictly positive equity / NAV curve; "
            "use absolute drawdown for PnL series that can be zero or negative"
        )
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return float(np.min(dd))


def calmar(
    returns: ArrayLike,
    holding_days: int = 1,
    trading_days_per_year: int = 365,
) -> float:
    """Calmar ratio: CAGR / |max drawdown|."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return np.nan

    _check_sample_size(r)

    equity = np.cumprod(1 + r)
    mdd = max_drawdown(equity)
    if mdd == 0:
        return np.nan

    periods_per_year = trading_days_per_year / holding_days
    total_return = equity[-1]
    ann_return = total_return ** (periods_per_year / len(r)) - 1

    return float(ann_return / abs(mdd))


def win_rate(returns: ArrayLike) -> float:
    """Fraction of positive returns (zeros excluded)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r) & (r != 0)]
    if len(r) == 0:
        return np.nan
    return float(np.mean(r > 0))


def profit_factor(returns: ArrayLike) -> float:
    """Sum of gains / sum of losses."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / losses)


# ── Internal ──────────────────────────────────────────────────


def _check_autocorrelation(
    returns: np.ndarray,
    holding_days: int,
    threshold: float = 0.3,
) -> None:
    """Warn if returns show high lag-1 autocorrelation."""
    if len(returns) < 20:
        return
    r = returns[np.isfinite(returns)]
    if len(r) < 20:
        return
    ac1 = np.corrcoef(r[:-1], r[1:])[0, 1]
    if np.isfinite(ac1) and abs(ac1) > threshold:
        warnings.warn(
            f"Returns have lag-1 autocorrelation = {ac1:.3f} (>{threshold}). "
            f"This often indicates overlapping {holding_days}-day returns "
            f"sampled more frequently than every {holding_days} days. "
            f"Sharpe may be inflated. Consider subsampling: returns[::holding_days].",
            UserWarning,
            stacklevel=3,
        )


def _check_sample_size(
    returns: np.ndarray,
    min_periods: int = 20,
) -> None:
    """Warn when annualized metrics are estimated from very small samples."""
    if len(returns) >= min_periods:
        return
    warnings.warn(
        f"Metric estimated from only {len(returns)} observations. "
        f"Annualized risk metrics are unstable below ~{min_periods} samples.",
        UserWarning,
        stacklevel=3,
    )
