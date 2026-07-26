"""Small research-statistics primitives used by private research routes."""

from __future__ import annotations

from dataclasses import dataclass
from math import erfc, floor, sqrt

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StepdownMaxTBootstrapArtifacts:
    summary: pd.DataFrame
    bootstrap_t_values: pd.DataFrame
    block_starts: pd.DataFrame


def newey_west_max_lags(observation_count: int, overlap_lags: int = 0) -> int:
    """Conservative Newey-West lag count for a univariate test series.

    ``overlap_lags`` is the minimum lag count implied by overlapping returns
    or another known serial-dependence horizon. The returned value is always
    clipped to ``observation_count - 1``.
    """
    if observation_count <= 1:
        return 0
    overlap_lags = int(overlap_lags)
    if overlap_lags < 0:
        raise ValueError("overlap_lags must be non-negative")
    sample_rule_lags = int(floor(4 * (observation_count / 100) ** (2 / 9)))
    return max(0, min(observation_count - 1, max(sample_rule_lags, overlap_lags)))


def hac_t_stat(
    values: pd.Series | np.ndarray | list[float],
    max_lags: int | None = None,
    *,
    overlap_lags: int = 0,
) -> float:
    """Newey-West HAC t-statistic for whether a series mean differs from zero."""
    series = pd.Series(values).dropna().astype(float)
    observation_count = len(series)
    if observation_count < 3:
        return float("nan")
    lags = newey_west_max_lags(
        observation_count, overlap_lags=overlap_lags) if max_lags is None else int(max_lags)
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


def normal_one_sided_p_value(t_stat: float) -> float:
    """Upper-tail one-sided normal p-value for a positive-alpha t-statistic."""
    value = float(t_stat)
    if not np.isfinite(value):
        return float("nan")
    return float(0.5 * erfc(value / sqrt(2.0)))


def normal_two_sided_p_value(t_stat: float) -> float:
    """Two-sided normal p-value for a finite t-statistic."""
    value = float(t_stat)
    if not np.isfinite(value):
        return float("nan")
    return float(erfc(abs(value) / sqrt(2.0)))


def benjamini_hochberg_q_values(values: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values, aligned to the input order.

    NaN p-values remain NaN. Finite values must lie in ``[0, 1]``.
    """
    raw = np.asarray(values, dtype=float)
    result = np.full(raw.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(raw)
    if not finite_mask.any():
        return result
    finite_values = raw[finite_mask]
    if ((finite_values < 0.0) | (finite_values > 1.0)).any():
        raise ValueError("p-values must be between 0 and 1")
    order = np.argsort(finite_values, kind="mergesort")
    sorted_values = finite_values[order]
    count = len(sorted_values)
    ranks = np.arange(1, count + 1, dtype=float)
    adjusted = sorted_values * count / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    finite_result = np.empty_like(finite_values)
    finite_result[order] = adjusted
    result[finite_mask] = finite_result
    return result


def holm_adjusted_p_values(values: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    """Holm FWER-adjusted p-values, aligned to the input order.

    NaN p-values remain NaN and are not counted in the family. Callers that
    require a fixed family including invalid hypotheses must explicitly pass
    1.0 for those hypotheses before restoring their invalid status.
    """
    raw = np.asarray(values, dtype=float)
    result = np.full(raw.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(raw)
    if not finite_mask.any():
        return result
    finite_values = raw[finite_mask]
    if ((finite_values < 0.0) | (finite_values > 1.0)).any():
        raise ValueError("p-values must be between 0 and 1")
    order = np.argsort(finite_values, kind="mergesort")
    sorted_values = finite_values[order]
    count = len(sorted_values)
    multipliers = np.arange(count, 0, -1, dtype=float)
    adjusted = np.maximum.accumulate(sorted_values * multipliers)
    adjusted = np.clip(adjusted, 0.0, 1.0)
    finite_result = np.empty_like(finite_values)
    finite_result[order] = adjusted
    result[finite_mask] = finite_result
    return result


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


def circular_block_bootstrap_stepdown_max_t(
    daily_centered_sums: pd.DataFrame,
    daily_counts: pd.DataFrame,
    observed_effects: pd.Series,
    *,
    block_length: int,
    n_bootstrap: int,
    seed: int,
    batch_size: int = 128,
    monte_carlo_batch_count: int = 20,
) -> StepdownMaxTBootstrapArtifacts:
    """One-sided synchronized circular-block step-down maxT inference.

    The same sampled day indices are used for every hypothesis. Each hypothesis
    is standardized by its own fixed bootstrap standard deviation.
    """
    if int(n_bootstrap) < 10_000:
        raise ValueError("n_bootstrap must be at least 10000")
    if int(block_length) <= 0:
        raise ValueError("block_length must be positive")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(monte_carlo_batch_count) < 2:
        raise ValueError("monte_carlo_batch_count must be at least 2")
    if int(n_bootstrap) % int(monte_carlo_batch_count) != 0:
        raise ValueError(
            "n_bootstrap must be divisible by monte_carlo_batch_count"
        )
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    if not isinstance(daily_centered_sums, pd.DataFrame) or not isinstance(
        daily_counts, pd.DataFrame
    ):
        raise TypeError("daily_centered_sums and daily_counts must be DataFrames")
    if daily_centered_sums.empty:
        raise ValueError("daily centered sums must not be empty")
    if not daily_centered_sums.index.equals(daily_counts.index):
        raise ValueError("daily centered sums and counts must share the same index")
    if not daily_centered_sums.columns.equals(daily_counts.columns):
        raise ValueError("daily centered sums and counts must share ordered hypotheses")
    if daily_centered_sums.index.has_duplicates:
        raise ValueError("daily inputs contain duplicate day keys")
    if daily_centered_sums.columns.has_duplicates:
        raise ValueError("daily inputs contain duplicate hypothesis ids")
    hypothesis_ids = daily_centered_sums.columns.astype(str)
    if len(set(hypothesis_ids)) != len(hypothesis_ids):
        raise ValueError("hypothesis ids are not unique after string normalization")
    if int(block_length) > len(daily_centered_sums):
        raise ValueError("block_length exceeds the available day count")

    sums = daily_centered_sums.to_numpy(dtype=float)
    counts = daily_counts.to_numpy(dtype=float)
    if not np.isfinite(sums).all() or not np.isfinite(counts).all():
        raise ValueError("daily bootstrap inputs must be finite")
    if (counts <= 0.0).any() or not np.allclose(counts, np.round(counts)):
        raise ValueError("daily counts must be positive integers")
    centered_total = sums.sum(axis=0)
    if not np.allclose(centered_total, 0.0, atol=1e-10, rtol=0.0):
        raise ValueError("daily centered sums are not centered by hypothesis")

    effects = pd.Series(observed_effects).copy()
    effects.index = effects.index.astype(str)
    effects = effects.reindex(hypothesis_ids)
    if effects.isna().any() or not np.isfinite(effects.to_numpy(dtype=float)).all():
        raise ValueError("observed effects must be finite and cover every hypothesis")
    observed = effects.to_numpy(dtype=float)

    day_count = len(daily_centered_sums)
    blocks_per_sample = int(np.ceil(day_count / int(block_length)))
    rng = np.random.default_rng(int(seed))
    block_starts_array = rng.integers(
        0,
        day_count,
        size=(int(n_bootstrap), blocks_per_sample),
        endpoint=False,
    )
    bootstrap_effects = np.empty(
        (int(n_bootstrap), len(hypothesis_ids)), dtype=float
    )
    offsets = np.arange(int(block_length), dtype=int)
    for start in range(0, int(n_bootstrap), int(batch_size)):
        stop = min(start + int(batch_size), int(n_bootstrap))
        starts = block_starts_array[start:stop]
        sampled_days = (
            starts[:, :, None] + offsets[None, None, :]
        ) % day_count
        sampled_days = sampled_days.reshape(stop - start, -1)[:, :day_count]
        sampled_sums = sums[sampled_days].sum(axis=1)
        sampled_counts = counts[sampled_days].sum(axis=1)
        if (sampled_counts <= 0.0).any():
            raise ValueError("bootstrap sample produced a non-positive denominator")
        bootstrap_effects[start:stop] = sampled_sums / sampled_counts

    bootstrap_se = bootstrap_effects.std(axis=0, ddof=1)
    if not np.isfinite(bootstrap_se).all() or (bootstrap_se <= 0.0).any():
        raise ValueError("bootstrap standard errors must be finite and positive")
    observed_t = observed / bootstrap_se
    bootstrap_t = bootstrap_effects / bootstrap_se
    raw_counts = (bootstrap_t >= observed_t[None, :]).sum(axis=0)
    raw_p = (1.0 + raw_counts) / (int(n_bootstrap) + 1.0)
    raw_mcse = np.sqrt(raw_p * (1.0 - raw_p) / (int(n_bootstrap) + 1.0))

    order = np.argsort(-observed_t, kind="mergesort")
    ordered_observed = observed_t[order]
    ordered_bootstrap = bootstrap_t[:, order]
    suffix_max = np.maximum.accumulate(ordered_bootstrap[:, ::-1], axis=1)[:, ::-1]
    step_counts = (suffix_max >= ordered_observed[None, :]).sum(axis=0)
    step_raw = (1.0 + step_counts) / (int(n_bootstrap) + 1.0)
    ordered_raw_p = raw_p[order]
    step_adjusted = np.maximum.accumulate(np.maximum(step_raw, ordered_raw_p))
    step_adjusted = np.clip(step_adjusted, 0.0, 1.0)
    adjusted = np.empty_like(step_adjusted)
    adjusted[order] = step_adjusted
    if (adjusted + 1e-15 < raw_p).any():
        raise RuntimeError("step-down adjusted p-value fell below marginal p-value")
    monte_carlo_batch_size = int(n_bootstrap) // int(monte_carlo_batch_count)
    batch_adjusted_values = np.empty(
        (int(monte_carlo_batch_count), len(hypothesis_ids)), dtype=float
    )
    for batch_index in range(int(monte_carlo_batch_count)):
        start = batch_index * monte_carlo_batch_size
        stop = start + monte_carlo_batch_size
        batch_bootstrap = bootstrap_t[start:stop]
        batch_raw = (
            1.0 + (batch_bootstrap >= observed_t[None, :]).sum(axis=0)
        ) / (monte_carlo_batch_size + 1.0)
        batch_ordered = batch_bootstrap[:, order]
        batch_suffix_max = np.maximum.accumulate(
            batch_ordered[:, ::-1], axis=1
        )[:, ::-1]
        batch_step_raw = (
            1.0
            + (batch_suffix_max >= ordered_observed[None, :]).sum(axis=0)
        ) / (monte_carlo_batch_size + 1.0)
        batch_adjusted_ordered = np.maximum.accumulate(
            np.maximum(batch_step_raw, batch_raw[order])
        )
        batch_adjusted_values[batch_index, order] = batch_adjusted_ordered
    adjusted_batch_mcse = (
        batch_adjusted_values.std(axis=0, ddof=1)
        / sqrt(float(monte_carlo_batch_count))
    )

    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    summary = pd.DataFrame(
        {
            "hypothesis_id": hypothesis_ids,
            "observed_effect": observed,
            "bootstrap_se": bootstrap_se,
            "observed_t": observed_t,
            "raw_one_sided_p_value": raw_p,
            "raw_p_mcse": raw_mcse,
            "stepdown_max_t_adjusted_p_value": adjusted,
            "stepdown_adjusted_p_batch_mcse": adjusted_batch_mcse,
            "monte_carlo_batch_count": int(monte_carlo_batch_count),
            "monte_carlo_batch_size": monte_carlo_batch_size,
            "observed_t_descending_rank": ranks,
            "block_length_days": int(block_length),
            "n_bootstrap": int(n_bootstrap),
            "seed": int(seed),
        }
    )
    bootstrap_frame = pd.DataFrame(bootstrap_t, columns=hypothesis_ids)
    bootstrap_frame.index.name = "bootstrap_idx"
    starts_frame = pd.DataFrame(
        block_starts_array,
        columns=[f"block_{index:03d}_start_day_offset" for index in range(blocks_per_sample)],
    )
    starts_frame.index.name = "bootstrap_idx"
    return StepdownMaxTBootstrapArtifacts(
        summary=summary,
        bootstrap_t_values=bootstrap_frame,
        block_starts=starts_frame,
    )
