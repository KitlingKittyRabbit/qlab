"""
Signal generation utilities.

Provides z-score normalization, IC computation, and threshold-based
signal generation with a clear distinction between rolling (live) and
fixed (walk-forward) z-score methods.
"""

from typing import Sequence, Union

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


def rank_standardize_cross_section(
    series: pd.Series,
    *,
    output_range: tuple[float, float] = (-1.0, 1.0),
) -> pd.Series:
    """Rank-standardize one cross-section into a bounded numeric range."""
    values = series.dropna()
    result = pd.Series(np.nan, index=series.index,
                       dtype=float, name=series.name)
    if values.empty:
        return result
    lower, upper = output_range
    if lower >= upper:
        raise ValueError(
            "output_range lower bound must be smaller than upper bound")
    ranks = values.rank(method="average")
    if len(values) == 1:
        result.loc[values.index] = 0.0
        return result
    scaled = lower + (ranks - 1.0) * ((upper - lower) / (len(values) - 1.0))
    result.loc[values.index] = scaled.astype(float)
    return result


def rank_standardize_panel_cross_section(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    decision_level: str | int = "decision_ts",
) -> pd.DataFrame:
    """Apply rank standardization column-wise within each decision cross-section."""
    result = frame.copy()
    for column in columns:
        result[column] = result.groupby(level=decision_level)[
            column].transform(rank_standardize_cross_section)
    return result


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
