"""
Spread construction and analysis utilities for pair trading.

Provides hedge ratio estimation, half-life calculation, and
cointegration testing.
"""

from typing import Tuple

import numpy as np
import pandas as pd


def build_spread(
    prices_a: pd.Series,
    prices_b: pd.Series,
    hedge_ratio: float = 1.0,
) -> pd.Series:
    """
    Build price spread: A - hedge_ratio * B.

    Aligns on common dates automatically.
    """
    common = prices_a.index.intersection(prices_b.index)
    return prices_a.loc[common] - hedge_ratio * prices_b.loc[common]


def rolling_hedge_ratio(
    prices_a: pd.Series,
    prices_b: pd.Series,
    window: int = 60,
) -> pd.Series:
    """
    Rolling OLS hedge ratio: regress A on B over rolling window.

    Returns beta such that spread = A - beta * B is approximately stationary.
    """
    common = prices_a.index.intersection(prices_b.index)
    a = prices_a.loc[common]
    b = prices_b.loc[common]

    ratios = pd.Series(np.nan, index=common)
    for i in range(window, len(common)):
        y = a.iloc[i - window : i].values
        x = b.iloc[i - window : i].values
        x_dm = x - x.mean()
        denom = np.dot(x_dm, x_dm)
        if denom < 1e-10:
            continue
        beta = np.dot(x_dm, y - y.mean()) / denom
        ratios.iloc[i] = beta

    return ratios


def half_life(spread: pd.Series) -> float:
    """
    Estimate mean-reversion half-life via Ornstein-Uhlenbeck regression.

        ΔS_t = α + β · S_{t-1} + ε
        half_life = -ln(2) / β

    Returns np.inf if spread is not mean-reverting (β ≥ 0).
    """
    s = spread.dropna()
    if len(s) < 20:
        return np.inf

    lag = s.shift(1).dropna()
    delta = s.diff().dropna()
    common = lag.index.intersection(delta.index)

    y = delta.loc[common].values
    x = lag.loc[common].values
    x_with_const = np.column_stack([np.ones(len(x)), x])

    try:
        beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.inf

    b = beta[1]
    if b >= 0:
        return np.inf

    return float(-np.log(2) / b)


def coint_test(
    prices_a: pd.Series,
    prices_b: pd.Series,
) -> Tuple[float, float, np.ndarray]:
    """
    Engle-Granger cointegration test.

    Returns
    -------
    t_stat : float
    p_value : float
    crit_values : ndarray
        Critical values at 1%, 5%, 10%.

    Raises
    ------
    ImportError
        If statsmodels is not installed.
    """
    try:
        from statsmodels.tsa.stattools import coint
    except ImportError:
        raise ImportError(
            "statsmodels is required for coint_test. "
            "Install with: pip install statsmodels"
        )

    common = prices_a.index.intersection(prices_b.index)
    a = prices_a.loc[common].values
    b = prices_b.loc[common].values

    t_stat, p_value, crit = coint(a, b)
    return float(t_stat), float(p_value), np.asarray(crit)
