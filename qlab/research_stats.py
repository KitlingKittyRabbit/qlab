"""Small research-statistics primitives used by private research routes."""

from __future__ import annotations

from math import floor, sqrt

import numpy as np
import pandas as pd


def newey_west_max_lags(observation_count: int) -> int:
    """Conservative Newey-West lag count for a univariate test series."""
    if observation_count <= 1:
        return 0
    return max(0, min(observation_count - 1, int(floor(4 * (observation_count / 100) ** (2 / 9)))))


def hac_t_stat(values: pd.Series | np.ndarray | list[float], max_lags: int | None = None) -> float:
    """Newey-West HAC t-statistic for whether a series mean differs from zero."""
    series = pd.Series(values).dropna().astype(float)
    observation_count = len(series)
    if observation_count < 3:
        return float("nan")
    lags = newey_west_max_lags(
        observation_count) if max_lags is None else int(max_lags)
    lags = max(0, min(lags, observation_count - 1))
    centered = series.to_numpy() - float(series.mean())
    gamma_zero = float(np.dot(centered, centered) / observation_count)
    long_run_variance = gamma_zero
    for lag in range(1, lags + 1):
        gamma = float(
            np.dot(centered[lag:], centered[:-lag]) / observation_count)
        weight = 1.0 - lag / (lags + 1.0)
        long_run_variance += 2.0 * weight * gamma
    if not np.isfinite(long_run_variance) or long_run_variance <= 0.0:
        return float("nan")
    standard_error = sqrt(long_run_variance / observation_count)
    if not np.isfinite(standard_error) or standard_error <= 0.0:
        return float("nan")
    return float(series.mean() / standard_error)


def simple_t_stat(values: pd.Series | np.ndarray | list[float]) -> float:
    """Plain t-statistic for whether a series mean differs from zero."""
    series = pd.Series(values).dropna().astype(float)
    observation_count = len(series)
    if observation_count < 2:
        return float("nan")
    std = float(series.std(ddof=1))
    if not np.isfinite(std) or std <= 0.0:
        return float("nan")
    return float(series.mean() / std * sqrt(observation_count))


def annualized_sharpe_from_periods(
    values: pd.Series | np.ndarray | list[float],
    periods_per_year: int | float,
) -> float:
    """Annualized Sharpe when the caller already knows periods per year."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    series = pd.Series(values).dropna().astype(float)
    if len(series) < 2:
        return float("nan")
    std = float(series.std(ddof=1))
    if not np.isfinite(std) or std <= 0.0:
        return float("nan")
    return float(series.mean() / std * sqrt(float(periods_per_year)))
