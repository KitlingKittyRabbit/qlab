"""
Signal generation utilities.

Provides z-score normalization, IC computation, and threshold-based
signal generation with a clear distinction between rolling (live) and
fixed (walk-forward) z-score methods.
"""

from typing import Union

import numpy as np
import pandas as pd
from scipy import stats

ArrayLike = Union[np.ndarray, pd.Series, list]


def zscore(
    series: pd.Series,
    method: str = "rolling",
    window: int = 90,
) -> pd.Series:
    """
    Compute z-score of a time series.

    Parameters
    ----------
    series : pd.Series
        Input time series.
    method : 'rolling' or 'expanding'
        - rolling: uses a fixed rolling window.
        - expanding: uses all history up to each point (min_periods=window).
    window : int
        Window size (periods, not calendar days).
    """
    if method == "rolling":
        mu = series.rolling(window).mean()
        sd = series.rolling(window).std()
    elif method == "expanding":
        mu = series.expanding(min_periods=window).mean()
        sd = series.expanding(min_periods=window).std()
    else:
        raise ValueError(
            f"method must be 'rolling' or 'expanding', got {method!r}")

    return (series - mu) / (sd + 1e-10)


def zscore_fixed(
    series: pd.Series,
    mu: float,
    sd: float,
) -> pd.Series:
    """
    Z-score with externally provided statistics (e.g. from a training set).

    This is what walk-forward backtests use: compute mu/sd on training data,
    apply to test data.
    """
    if sd < 1e-10:
        return pd.Series(np.nan, index=series.index)
    return (series - mu) / sd


def ic(
    factor: ArrayLike,
    forward_ret: ArrayLike,
) -> float:
    """
    Spearman rank IC between factor values and forward returns.

    Returns NaN if fewer than 10 valid paired observations.
    """
    f = np.asarray(factor, dtype=float)
    r = np.asarray(forward_ret, dtype=float)
    mask = np.isfinite(f) & np.isfinite(r)
    if mask.sum() < 10:
        return np.nan
    corr, _ = stats.spearmanr(f[mask], r[mask])
    return float(corr)


def ic_direction(
    factor: ArrayLike,
    forward_ret: ArrayLike,
    min_abs_ic: float = 0.005,
) -> int:
    """
    Determine IC sign from a TRAINING slice.

    Returns +1 if IC > min_abs_ic, -1 if IC < -min_abs_ic, else 0.

    IMPORTANT: only call with training data. Using full-sample data
    introduces look-ahead bias.
    """
    ic_val = ic(factor, forward_ret)
    if np.isnan(ic_val):
        return 0
    if ic_val > min_abs_ic:
        return 1
    elif ic_val < -min_abs_ic:
        return -1
    return 0


def threshold_signal(
    composite: ArrayLike,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Convert composite score to discrete position signal.

    Returns array of +1 (long), -1 (short), or 0 (flat).
    Values exactly at ±threshold are classified as flat (0).
    """
    c = np.asarray(composite, dtype=float)
    return np.where(c > threshold, 1, np.where(c < -threshold, -1, 0))
