"""Small research-statistics primitives used by private research routes."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, erfc, floor, log, log10, sqrt

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov


@dataclass(frozen=True)
class StepdownMaxTBootstrapArtifacts:
    summary: pd.DataFrame
    bootstrap_t_values: pd.DataFrame
    block_starts: pd.DataFrame


@dataclass(frozen=True)
class AdaptiveFlatTopRatioSEArtifacts:
    standard_error: np.ndarray
    bandwidth: np.ndarray
    raw_long_run_variance: np.ndarray
    long_run_variance: np.ndarray
    bartlett_fallback_applied: np.ndarray


@dataclass(frozen=True)
class SelfNormalizedRatioSEArtifacts:
    standard_error: np.ndarray
    self_normalizer: np.ndarray


@dataclass(frozen=True)
class AutoRegressiveSpectralHolmArtifacts:
    summary: pd.DataFrame
    selected_coefficients: pd.DataFrame


@dataclass(frozen=True)
class AutoRegressiveSpectralBhArtifacts:
    summary: pd.DataFrame
    selected_coefficients: pd.DataFrame
    family_summary: pd.DataFrame


@dataclass(frozen=True)
class RandomizationStepdownMaxTArtifacts:
    summary: pd.DataFrame
    null_t_values: pd.DataFrame


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


def _stable_ar_finite_sample_variance(
    coefficients: np.ndarray,
    innovation_variance: float,
    *,
    observation_count: int,
) -> tuple[float, float]:
    """Finite-sample mean variance and zero-frequency LRV of a stable AR model."""
    phi = np.asarray(coefficients, dtype=float)
    if phi.ndim != 1 or not np.isfinite(phi).all():
        raise ValueError("AR coefficients must be one-dimensional and finite")
    sigma2 = float(innovation_variance)
    count = int(observation_count)
    if count < 3 or not np.isfinite(sigma2) or sigma2 <= 0.0:
        raise ValueError("AR variance inputs must be finite and positive")
    if len(phi) == 0:
        return float(sigma2 / count), float(sigma2)

    companion = np.zeros((len(phi), len(phi)), dtype=float)
    companion[0] = phi
    if len(phi) > 1:
        companion[1:, :-1] = np.eye(len(phi) - 1)
    if float(np.max(np.abs(np.linalg.eigvals(companion)))) >= 1.0 - 1e-10:
        raise ValueError("fitted AR model is not stationary")
    innovation_covariance = np.zeros_like(companion)
    innovation_covariance[0, 0] = sigma2
    state_covariance = solve_discrete_lyapunov(
        companion, innovation_covariance
    )
    if not np.isfinite(state_covariance).all() or state_covariance[0, 0] <= 0.0:
        raise ValueError("fitted AR covariance is not finite and positive")
    autocovariances = np.empty(count, dtype=float)
    transition = np.eye(len(phi), dtype=float)
    for lag in range(count):
        autocovariances[lag] = float((transition @ state_covariance)[0, 0])
        transition = transition @ companion
    pair_counts = count - np.arange(1, count, dtype=float)
    mean_variance = (
        count * autocovariances[0]
        + 2.0 * np.dot(pair_counts, autocovariances[1:])
    ) / float(count * count)
    denominator = 1.0 - float(phi.sum())
    if abs(denominator) <= 1e-10:
        raise ValueError("fitted AR zero-frequency denominator is singular")
    long_run_variance = sigma2 / (denominator * denominator)
    if (
        not np.isfinite(mean_variance)
        or mean_variance <= 0.0
        or not np.isfinite(long_run_variance)
        or long_run_variance <= 0.0
    ):
        raise ValueError("fitted AR variance is not finite and positive")
    return float(mean_variance), float(long_run_variance)


def _is_stable_ar(coefficients: np.ndarray) -> bool:
    """Return whether an AR polynomial has all companion roots inside the unit circle."""
    phi = np.asarray(coefficients, dtype=float)
    if phi.ndim != 1 or not np.isfinite(phi).all():
        return False
    if len(phi) == 0:
        return True
    companion = np.zeros((len(phi), len(phi)), dtype=float)
    companion[0] = phi
    if len(phi) > 1:
        companion[1:, :-1] = np.eye(len(phi) - 1)
    return float(np.max(np.abs(np.linalg.eigvals(companion)))) < 1.0 - 1e-10


def autoregressive_spectral_holm_test(
    daily_centered_sums: pd.DataFrame,
    daily_counts: pd.DataFrame,
    observed_effects: pd.Series,
    *,
    order_criterion: str,
    standard_error_multiplier: float = 1.0,
    alternative: str = "greater",
    expected_hypothesis_count: int = 47,
) -> AutoRegressiveSpectralHolmArtifacts:
    """AR spectral marginal tests with a fixed complete-family Holm adjustment.

    Every candidate order is fitted on the same trailing observations. The
    temporal-dependence label and truth-known variance are deliberately absent
    from this production-capable entry.
    """
    hypothesis_ids, sums, counts, effects = _validated_daily_family_arrays(
        daily_centered_sums,
        daily_counts,
        observed_effects,
        dependence_length=1,
    )
    family_size = int(expected_hypothesis_count)
    if family_size <= 0 or len(hypothesis_ids) != family_size:
        raise ValueError("AR spectral Holm family has the wrong hypothesis count")
    criterion = str(order_criterion).upper()
    if criterion not in {"AIC", "BIC"}:
        raise ValueError("order_criterion must be AIC or BIC")
    if alternative not in {"greater", "two-sided"}:
        raise ValueError("alternative must be greater or two-sided")
    multiplier = float(standard_error_multiplier)
    if not np.isfinite(multiplier) or multiplier < 1.0:
        raise ValueError("standard_error_multiplier must be finite and at least one")
    day_count = len(sums)
    maximum_order = min(28, int(floor(sqrt(day_count))))
    fit_count = day_count - maximum_order
    if maximum_order < 1 or fit_count <= maximum_order + 1:
        raise ValueError("too few observations for the frozen AR order search")
    mean_counts = counts.mean(axis=0)
    influence = sums / mean_counts[None, :]
    if not np.allclose(influence.sum(axis=0), 0.0, atol=1e-12, rtol=1e-12):
        raise ValueError("scaled ratio influence values are not centered")
    if (np.var(influence, axis=0) <= 0.0).any():
        raise ValueError("AR spectral input contains a constant hypothesis")

    rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    for column_index, hypothesis_id in enumerate(hypothesis_ids):
        series = influence[:, column_index]
        target = series[maximum_order:]
        best: tuple[float, int, float, np.ndarray] | None = None
        for order in range(maximum_order + 1):
            if order == 0:
                residual = target
                coefficients = np.empty(0, dtype=float)
            else:
                design = np.column_stack(
                    [
                        series[maximum_order - lag:day_count - lag]
                        for lag in range(1, order + 1)
                    ]
                )
                coefficients, _, rank, _ = np.linalg.lstsq(
                    design, target, rcond=None
                )
                if int(rank) != order:
                    continue
                residual = target - design @ coefficients
            innovation_variance = float(np.dot(residual, residual) / fit_count)
            if not np.isfinite(innovation_variance) or innovation_variance <= 0.0:
                continue
            penalty = 2.0 * order if criterion == "AIC" else log(fit_count) * order
            information_criterion = fit_count * np.log(innovation_variance) + penalty
            if not _is_stable_ar(coefficients):
                continue
            candidate = (
                float(information_criterion),
                int(order),
                innovation_variance,
                np.asarray(coefficients, dtype=float),
            )
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None:
            raise ValueError(f"no valid stable AR fit for {hypothesis_id}")
        information_criterion, order, innovation_variance, coefficients = best
        mean_variance, long_run_variance = _stable_ar_finite_sample_variance(
            coefficients,
            innovation_variance,
            observation_count=day_count,
        )
        uncalibrated_standard_error = sqrt(mean_variance)
        standard_error = uncalibrated_standard_error * multiplier
        statistic = float(effects[column_index] / standard_error)
        if alternative == "greater":
            raw_p = 0.5 * erfc(statistic / sqrt(2.0))
        else:
            raw_p = erfc(abs(statistic) / sqrt(2.0))
        rows.append(
            {
                "hypothesis_id": str(hypothesis_id),
                "observed_effect": float(effects[column_index]),
                "selected_ar_order": int(order),
                "maximum_ar_order": int(maximum_order),
                "common_fit_observation_count": int(fit_count),
                "order_criterion": criterion,
                "information_criterion": float(information_criterion),
                "innovation_variance": float(innovation_variance),
                "zero_frequency_long_run_variance": float(long_run_variance),
                "finite_sample_mean_variance": float(mean_variance),
                "uncalibrated_standard_error": float(uncalibrated_standard_error),
                "standard_error_multiplier": multiplier,
                "standard_error": float(standard_error),
                "observed_t": statistic,
                "raw_one_sided_p_value": raw_p if alternative == "greater" else np.nan,
                "raw_two_sided_p_value": raw_p if alternative == "two-sided" else np.nan,
                "alternative": alternative,
                "inference_engine": "autoregressive_spectral_normal_holm",
            }
        )
        for lag, coefficient in enumerate(coefficients, start=1):
            coefficient_rows.append(
                {
                    "hypothesis_id": str(hypothesis_id),
                    "lag": int(lag),
                    "coefficient": float(coefficient),
                }
            )
    summary = pd.DataFrame(rows)
    raw_column = (
        "raw_one_sided_p_value"
        if alternative == "greater"
        else "raw_two_sided_p_value"
    )
    summary["holm_adjusted_p_value"] = holm_adjusted_p_values(summary[raw_column])
    summary["family_adjusted_p_value"] = summary["holm_adjusted_p_value"]
    coefficients = pd.DataFrame(
        coefficient_rows,
        columns=["hypothesis_id", "lag", "coefficient"],
    )
    return AutoRegressiveSpectralHolmArtifacts(summary, coefficients)


def autoregressive_spectral_bh_test(
    daily_centered_sums: pd.DataFrame,
    daily_counts: pd.DataFrame,
    observed_effects: pd.Series,
    *,
    order_criterion: str,
    standard_error_multiplier: float = 1.0,
    alternative: str = "greater",
    expected_hypothesis_count: int = 47,
    alpha: float = 0.05,
) -> AutoRegressiveSpectralBhArtifacts:
    """AR spectral marginal tests with task-level Benjamini-Hochberg FDR control.

    The marginal estimates are exactly those of the tested AR spectral entry.
    Only the complete-family multiplicity contract changes from Holm to BH.
    """
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    marginal = autoregressive_spectral_holm_test(
        daily_centered_sums,
        daily_counts,
        observed_effects,
        order_criterion=order_criterion,
        standard_error_multiplier=standard_error_multiplier,
        alternative=alternative,
        expected_hypothesis_count=expected_hypothesis_count,
    )
    summary = marginal.summary.drop(
        columns=["holm_adjusted_p_value", "family_adjusted_p_value"]
    ).copy()
    raw_column = (
        "raw_one_sided_p_value"
        if alternative == "greater"
        else "raw_two_sided_p_value"
    )
    summary["bh_adjusted_q_value"] = benjamini_hochberg_q_values(
        summary[raw_column]
    )
    summary["family_adjusted_p_value"] = summary["bh_adjusted_q_value"]
    summary["inference_engine"] = "autoregressive_spectral_normal_bh"
    summary["alpha"] = float(alpha)
    summary["discovered"] = summary["bh_adjusted_q_value"].le(float(alpha))
    family_summary = pd.DataFrame(
        [
            {
                "hypothesis_count": int(len(summary)),
                "discovery_count": int(summary["discovered"].sum()),
                "alpha": float(alpha),
                "alternative": str(alternative),
                "order_criterion": str(order_criterion).upper(),
                "standard_error_multiplier": float(standard_error_multiplier),
                "inference_engine": "autoregressive_spectral_normal_bh",
            }
        ]
    )
    return AutoRegressiveSpectralBhArtifacts(
        summary=summary,
        selected_coefficients=marginal.selected_coefficients.copy(),
        family_summary=family_summary,
    )


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


def _circular_block_bootstrap_stepdown_max_t(
    daily_centered_sums: pd.DataFrame,
    daily_counts: pd.DataFrame,
    observed_effects: pd.Series,
    *,
    block_length: int,
    n_bootstrap: int,
    seed: int,
    batch_size: int = 128,
    monte_carlo_batch_count: int = 20,
    alternative: str = "greater",
    minimum_bootstrap_repetitions: int,
) -> StepdownMaxTBootstrapArtifacts:
    """Shared implementation behind production and simulation-only contracts.

    The same sampled day indices are used for every hypothesis. Each hypothesis
    is standardized by its own fixed bootstrap standard deviation. ``greater``
    preserves the registered one-sided contract; ``two-sided`` compares the
    absolute observed and bootstrap t statistics.
    """
    if int(n_bootstrap) < int(minimum_bootstrap_repetitions):
        raise ValueError(
            "n_bootstrap must be at least "
            f"{int(minimum_bootstrap_repetitions)}"
        )
    if int(block_length) <= 0:
        raise ValueError("block_length must be positive")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(monte_carlo_batch_count) < 1:
        raise ValueError("monte_carlo_batch_count must be positive")
    if int(n_bootstrap) % int(monte_carlo_batch_count) != 0:
        raise ValueError(
            "n_bootstrap must be divisible by monte_carlo_batch_count"
        )
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    if alternative not in {"greater", "two-sided"}:
        raise ValueError("alternative must be greater or two-sided")
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
    _validate_daily_index(daily_centered_sums.index)
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
    observed_test_stat = (
        np.abs(observed_t) if alternative == "two-sided" else observed_t
    )
    bootstrap_test_stat = (
        np.abs(bootstrap_t) if alternative == "two-sided" else bootstrap_t
    )
    raw_counts = (
        bootstrap_test_stat >= observed_test_stat[None, :]
    ).sum(axis=0)
    raw_p = (1.0 + raw_counts) / (int(n_bootstrap) + 1.0)
    raw_mcse = np.sqrt(raw_p * (1.0 - raw_p) / (int(n_bootstrap) + 1.0))

    order = np.argsort(-observed_test_stat, kind="mergesort")
    ordered_observed = observed_test_stat[order]
    ordered_bootstrap = bootstrap_test_stat[:, order]
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
    adjusted_batch_mcse = np.full(len(hypothesis_ids), np.nan, dtype=float)
    if int(monte_carlo_batch_count) >= 2:
        batch_adjusted_values = np.empty(
            (int(monte_carlo_batch_count), len(hypothesis_ids)), dtype=float
        )
        for batch_index in range(int(monte_carlo_batch_count)):
            start = batch_index * monte_carlo_batch_size
            stop = start + monte_carlo_batch_size
            batch_bootstrap = bootstrap_test_stat[start:stop]
            batch_raw = (
                1.0 + (batch_bootstrap >= observed_test_stat[None, :]).sum(axis=0)
            ) / (monte_carlo_batch_size + 1.0)
            batch_ordered = bootstrap_test_stat[start:stop][:, order]
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
            "raw_one_sided_p_value": (
                raw_p if alternative == "greater" else np.nan
            ),
            "raw_two_sided_p_value": (
                raw_p if alternative == "two-sided" else np.nan
            ),
            "raw_p_mcse": raw_mcse,
            "stepdown_max_t_adjusted_p_value": adjusted,
            "stepdown_adjusted_p_batch_mcse": adjusted_batch_mcse,
            "monte_carlo_batch_count": int(monte_carlo_batch_count),
            "monte_carlo_batch_size": monte_carlo_batch_size,
            "observed_t_descending_rank": ranks,
            "block_length_days": int(block_length),
            "n_bootstrap": int(n_bootstrap),
            "seed": int(seed),
            "alternative": alternative,
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
    alternative: str = "greater",
) -> StepdownMaxTBootstrapArtifacts:
    """Production synchronized circular-block step-down maxT inference.

    Production evidence keeps the registered minimum of 10,000 repetitions.
    Lower repetition counts must use the explicitly calibration-only entry.
    """
    return _circular_block_bootstrap_stepdown_max_t(
        daily_centered_sums,
        daily_counts,
        observed_effects,
        block_length=block_length,
        n_bootstrap=n_bootstrap,
        seed=seed,
        batch_size=batch_size,
        monte_carlo_batch_count=monte_carlo_batch_count,
        alternative=alternative,
        minimum_bootstrap_repetitions=10_000,
    )


def simulation_calibration_circular_block_stepdown_max_t(
    daily_centered_sums: pd.DataFrame,
    daily_counts: pd.DataFrame,
    observed_effects: pd.Series,
    *,
    block_length: int,
    n_bootstrap: int,
    seed: int,
    batch_size: int = 128,
    alternative: str = "greater",
) -> StepdownMaxTBootstrapArtifacts:
    """Calibration-only step-down maxT entry with the frozen 499 minimum.

    This entry exists solely for truth-known method simulation. It must not be
    used for empirical candidate evidence, reports, shadow, or live decisions.
    """
    return _circular_block_bootstrap_stepdown_max_t(
        daily_centered_sums,
        daily_counts,
        observed_effects,
        block_length=block_length,
        n_bootstrap=n_bootstrap,
        seed=seed,
        batch_size=batch_size,
        monte_carlo_batch_count=1,
        alternative=alternative,
        minimum_bootstrap_repetitions=499,
    )


def _validate_daily_index(index: pd.Index) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("daily inputs must use a DatetimeIndex")
    if index.tz is None or str(index.tz) != "UTC":
        raise ValueError("daily input index must use UTC")
    if not index.is_monotonic_increasing:
        raise ValueError("daily input index must be strictly increasing")
    if not index.equals(index.normalize()):
        raise ValueError("daily input index must contain UTC midnight timestamps")
    expected_days = pd.date_range(index[0], periods=len(index), freq="D", tz="UTC")
    if not index.equals(expected_days):
        raise ValueError("daily input index must contain consecutive calendar days")


def _validated_daily_family_arrays(
    daily_centered_sums: pd.DataFrame,
    daily_counts: pd.DataFrame,
    observed_effects: pd.Series,
    *,
    dependence_length: int,
) -> tuple[pd.Index, np.ndarray, np.ndarray, np.ndarray]:
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
    _validate_daily_index(daily_centered_sums.index)
    if daily_centered_sums.columns.has_duplicates:
        raise ValueError("daily inputs contain duplicate hypothesis ids")
    if int(dependence_length) <= 0 or int(dependence_length) > len(daily_centered_sums):
        raise ValueError("dependence_length must be between one and the day count")
    hypothesis_ids = daily_centered_sums.columns.astype(str)
    if len(set(hypothesis_ids)) != len(hypothesis_ids):
        raise ValueError("hypothesis ids are not unique after string normalization")
    sums = daily_centered_sums.to_numpy(dtype=float)
    counts = daily_counts.to_numpy(dtype=float)
    if not np.isfinite(sums).all() or not np.isfinite(counts).all():
        raise ValueError("daily resampling inputs must be finite")
    if (counts <= 0.0).any() or not np.allclose(counts, np.round(counts)):
        raise ValueError("daily counts must be positive integers")
    if not np.allclose(sums.sum(axis=0), 0.0, atol=1e-10, rtol=0.0):
        raise ValueError("daily centered sums are not centered by hypothesis")
    effects = pd.Series(observed_effects).copy()
    effects.index = effects.index.astype(str)
    effects = effects.reindex(hypothesis_ids)
    if effects.isna().any() or not np.isfinite(effects.to_numpy(dtype=float)).all():
        raise ValueError("observed effects must be finite and cover every hypothesis")
    return hypothesis_ids, sums, counts, effects.to_numpy(dtype=float)


def _bartlett_ratio_standard_error(
    influence: np.ndarray,
    counts: np.ndarray,
    *,
    max_lag: int,
) -> np.ndarray:
    """Bartlett-HAC standard error for one or many ratio estimators.

    Arrays may be ``day x hypothesis`` or ``draw x day x hypothesis``. The
    influence array must already equal ``daily_sum - ratio * daily_count``.
    """
    u = np.asarray(influence, dtype=float)
    c = np.asarray(counts, dtype=float)
    if u.shape != c.shape or u.ndim not in {2, 3}:
        raise ValueError("influence and counts must share a 2D or 3D shape")
    if not np.isfinite(u).all() or not np.isfinite(c).all() or (c <= 0.0).any():
        raise ValueError("ratio standard-error inputs must be finite with positive counts")
    day_axis = u.ndim - 2
    day_count = u.shape[day_axis]
    lag_count = int(max_lag)
    if lag_count < 0 or lag_count >= day_count:
        raise ValueError("max_lag must be between zero and day_count - 1")
    long_run = np.mean(np.square(u), axis=day_axis)
    for lag in range(1, lag_count + 1):
        left = np.take(u, np.arange(lag, day_count), axis=day_axis)
        right = np.take(u, np.arange(0, day_count - lag), axis=day_axis)
        gamma = np.sum(left * right, axis=day_axis) / float(day_count)
        long_run += 2.0 * (1.0 - lag / (lag_count + 1.0)) * gamma
    tolerance = np.finfo(float).eps * np.maximum(1.0, np.mean(np.square(u), axis=day_axis))
    if (long_run < -tolerance).any():
        raise ValueError("Bartlett long-run variance is materially negative")
    long_run = np.maximum(long_run, 0.0)
    mean_count = np.mean(c, axis=day_axis)
    standard_error = np.sqrt(long_run / float(day_count)) / mean_count
    if not np.isfinite(standard_error).all() or (standard_error <= 0.0).any():
        raise ValueError("ratio standard errors must be finite and positive")
    return standard_error


def adaptive_flat_top_ratio_standard_error(
    influence: np.ndarray,
    counts: np.ndarray,
) -> AdaptiveFlatTopRatioSEArtifacts:
    """Adaptive trapezoidal flat-top standard errors for ratio estimators.

    Arrays may be ``day x hypothesis`` or ``draw x day x hypothesis``. Each
    series receives its own Politis-style correlogram bandwidth. A nonpositive
    flat-top estimate falls back to Bartlett HAC at the same bandwidth; if
    that estimate is also nonpositive, the calculation fails closed.
    """
    u = np.asarray(influence, dtype=float)
    c = np.asarray(counts, dtype=float)
    if u.shape != c.shape or u.ndim not in {2, 3}:
        raise ValueError("influence and counts must share a 2D or 3D shape")
    if not np.isfinite(u).all() or not np.isfinite(c).all() or (c <= 0.0).any():
        raise ValueError("ratio standard-error inputs must be finite with positive counts")
    day_axis = u.ndim - 2
    day_count = int(u.shape[day_axis])
    k_t = max(5, int(ceil(sqrt(log10(day_count))))) if day_count > 1 else 5
    if day_count <= k_t + 2:
        raise ValueError("too few days for adaptive flat-top bandwidth selection")

    series = np.moveaxis(u, day_axis, -1)
    fft_length = 1 << (2 * day_count - 1).bit_length()
    transform = np.fft.rfft(series, n=fft_length, axis=-1)
    autocovariance = np.fft.irfft(
        transform * np.conjugate(transform), n=fft_length, axis=-1
    )[..., :day_count] / float(day_count)
    gamma_zero = autocovariance[..., 0]
    if not np.isfinite(gamma_zero).all() or (gamma_zero <= 0.0).any():
        raise ValueError("adaptive flat-top input must have positive variance")

    threshold = 2.0 * sqrt(log10(day_count) / float(day_count))
    correlations = autocovariance / gamma_zero[..., None]
    windows = np.lib.stride_tricks.sliding_window_view(
        np.abs(correlations[..., 1:]), window_shape=k_t + 1, axis=-1
    )
    acceptable = np.all(windows < threshold, axis=-1)
    has_bandwidth = np.any(acceptable, axis=-1)
    if not bool(np.all(has_bandwidth)):
        raise ValueError("adaptive flat-top bandwidth search found no cutoff")
    q_hat = np.argmax(acceptable, axis=-1) + 1
    bandwidth = np.minimum(2 * q_hat, day_count - 1).astype(int)

    lags = np.arange(1, day_count, dtype=float)
    scaled_lags = lags / bandwidth[..., None]
    flat_top_weights = np.where(
        scaled_lags <= 0.5,
        1.0,
        np.where(scaled_lags <= 1.0, 2.0 * (1.0 - scaled_lags), 0.0),
    )
    raw_long_run = gamma_zero + 2.0 * np.sum(
        autocovariance[..., 1:] * flat_top_weights, axis=-1
    )
    bartlett_weights = np.maximum(
        1.0 - lags / (bandwidth[..., None] + 1.0), 0.0
    )
    bartlett_long_run = gamma_zero + 2.0 * np.sum(
        autocovariance[..., 1:] * bartlett_weights, axis=-1
    )
    fallback = raw_long_run <= 0.0
    long_run = np.where(fallback, bartlett_long_run, raw_long_run)
    if not np.isfinite(long_run).all() or (long_run <= 0.0).any():
        raise ValueError("adaptive flat-top and Bartlett long-run variances are nonpositive")

    mean_count = np.mean(c, axis=day_axis)
    standard_error = np.sqrt(long_run / float(day_count)) / mean_count
    if not np.isfinite(standard_error).all() or (standard_error <= 0.0).any():
        raise ValueError("ratio standard errors must be finite and positive")
    return AdaptiveFlatTopRatioSEArtifacts(
        standard_error=standard_error,
        bandwidth=bandwidth,
        raw_long_run_variance=raw_long_run,
        long_run_variance=long_run,
        bartlett_fallback_applied=fallback,
    )


def self_normalized_ratio_standard_error(
    influence: np.ndarray,
    counts: np.ndarray,
) -> SelfNormalizedRatioSEArtifacts:
    """Recursive self-normalized standard errors for ratio estimators.

    Arrays may be ``day x hypothesis`` or ``draw x day x hypothesis``. The
    supplied influence values must equal ``daily_sum - ratio * daily_count``
    and therefore sum to zero over days for every series. This estimator does
    not estimate a long-run variance or select a smoothing bandwidth.
    """
    u = np.asarray(influence, dtype=float)
    c = np.asarray(counts, dtype=float)
    if u.shape != c.shape or u.ndim not in {2, 3}:
        raise ValueError("influence and counts must share a 2D or 3D shape")
    if not np.isfinite(u).all() or not np.isfinite(c).all() or (c <= 0.0).any():
        raise ValueError(
            "ratio standard-error inputs must be finite with positive counts"
        )
    day_axis = u.ndim - 2
    day_count = int(u.shape[day_axis])
    if day_count < 3:
        raise ValueError("self-normalization requires at least three days")
    total = np.sum(u, axis=day_axis)
    scale = np.maximum(1.0, np.sum(np.abs(u), axis=day_axis))
    if not np.all(np.abs(total) <= 1e-12 * scale):
        raise ValueError("ratio influence values must sum to zero")

    cumulative = np.cumsum(u, axis=day_axis)
    self_normalizer = np.sum(np.square(cumulative), axis=day_axis) / float(
        day_count**2
    )
    if not np.isfinite(self_normalizer).all() or (self_normalizer <= 0.0).any():
        raise ValueError("self-normalizers must be finite and positive")
    mean_count = np.mean(c, axis=day_axis)
    standard_error = (
        np.sqrt(self_normalizer / float(day_count)) / mean_count
    )
    if not np.isfinite(standard_error).all() or (standard_error <= 0.0).any():
        raise ValueError("self-normalized ratio standard errors must be finite and positive")
    return SelfNormalizedRatioSEArtifacts(
        standard_error=standard_error,
        self_normalizer=self_normalizer,
    )


def _stepdown_from_t_values(
    hypothesis_ids: pd.Index,
    observed: np.ndarray,
    observed_se: np.ndarray,
    bootstrap_t: np.ndarray,
    *,
    alternative: str,
    n_bootstrap: int,
    seed: int,
    engine: str,
    dependence_length: int,
) -> pd.DataFrame:
    if alternative not in {"greater", "two-sided"}:
        raise ValueError("alternative must be greater or two-sided")
    observed_t = observed / observed_se
    observed_test = np.abs(observed_t) if alternative == "two-sided" else observed_t
    bootstrap_test = np.abs(bootstrap_t) if alternative == "two-sided" else bootstrap_t
    raw_p = (
        1.0 + (bootstrap_test >= observed_test[None, :]).sum(axis=0)
    ) / (int(n_bootstrap) + 1.0)
    order = np.argsort(-observed_test, kind="mergesort")
    ordered_bootstrap = bootstrap_test[:, order]
    suffix_max = np.maximum.accumulate(ordered_bootstrap[:, ::-1], axis=1)[:, ::-1]
    step_raw = (
        1.0 + (suffix_max >= observed_test[order][None, :]).sum(axis=0)
    ) / (int(n_bootstrap) + 1.0)
    ordered_adjusted = np.maximum.accumulate(np.maximum(step_raw, raw_p[order]))
    adjusted = np.empty_like(ordered_adjusted)
    adjusted[order] = np.clip(ordered_adjusted, 0.0, 1.0)
    if (adjusted + 1e-15 < raw_p).any():
        raise RuntimeError("step-down adjusted p-value fell below marginal p-value")
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return pd.DataFrame(
        {
            "hypothesis_id": hypothesis_ids,
            "observed_effect": observed,
            "bootstrap_se": observed_se,
            "observed_t": observed_t,
            "raw_one_sided_p_value": raw_p if alternative == "greater" else np.nan,
            "raw_two_sided_p_value": raw_p if alternative == "two-sided" else np.nan,
            "raw_p_mcse": np.sqrt(
                raw_p * (1.0 - raw_p) / (int(n_bootstrap) + 1.0)
            ),
            "stepdown_max_t_adjusted_p_value": adjusted,
            "stepdown_adjusted_p_batch_mcse": np.nan,
            "monte_carlo_batch_count": 1,
            "monte_carlo_batch_size": int(n_bootstrap),
            "observed_t_descending_rank": ranks,
            "dependence_length_days": int(dependence_length),
            "n_bootstrap": int(n_bootstrap),
            "seed": int(seed),
            "alternative": alternative,
            "resampling_engine": str(engine),
        }
    )


def _restudentized_circular_block_stepdown_max_t(
    daily_centered_sums: pd.DataFrame,
    daily_counts: pd.DataFrame,
    observed_effects: pd.Series,
    *,
    block_length: int,
    n_bootstrap: int,
    seed: int,
    batch_size: int,
    alternative: str,
    minimum_bootstrap_repetitions: int,
    studentizer: str = "bartlett",
) -> StepdownMaxTBootstrapArtifacts:
    if int(n_bootstrap) < int(minimum_bootstrap_repetitions):
        raise ValueError(
            f"n_bootstrap must be at least {int(minimum_bootstrap_repetitions)}"
        )
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    hypothesis_ids, sums, counts, observed = _validated_daily_family_arrays(
        daily_centered_sums,
        daily_counts,
        observed_effects,
        dependence_length=int(block_length),
    )
    if studentizer not in {"bartlett", "adaptive_flat_top", "self_normalized"}:
        raise ValueError("unknown re-studentization engine")
    max_lag = int(block_length) - 1
    observed_flat_top = None
    if studentizer == "bartlett":
        observed_se = _bartlett_ratio_standard_error(sums, counts, max_lag=max_lag)
    elif studentizer == "adaptive_flat_top":
        observed_flat_top = adaptive_flat_top_ratio_standard_error(sums, counts)
        observed_se = observed_flat_top.standard_error
    else:
        observed_self_normalized = self_normalized_ratio_standard_error(sums, counts)
        observed_se = observed_self_normalized.standard_error
    day_count = len(sums)
    blocks_per_sample = int(np.ceil(day_count / int(block_length)))
    rng = np.random.default_rng(int(seed))
    starts_array = rng.integers(
        0, day_count, size=(int(n_bootstrap), blocks_per_sample), endpoint=False
    )
    bootstrap_t = np.empty((int(n_bootstrap), len(hypothesis_ids)), dtype=float)
    bootstrap_bandwidth = (
        np.empty_like(bootstrap_t, dtype=int)
        if studentizer == "adaptive_flat_top"
        else None
    )
    bootstrap_fallback = (
        np.empty_like(bootstrap_t, dtype=bool)
        if studentizer == "adaptive_flat_top"
        else None
    )
    offsets = np.arange(int(block_length), dtype=int)
    for start in range(0, int(n_bootstrap), int(batch_size)):
        stop = min(start + int(batch_size), int(n_bootstrap))
        sampled_days = (
            starts_array[start:stop, :, None] + offsets[None, None, :]
        ) % day_count
        sampled_days = sampled_days.reshape(stop - start, -1)[:, :day_count]
        sampled_sums = sums[sampled_days]
        sampled_counts = counts[sampled_days]
        sample_effect = sampled_sums.sum(axis=1) / sampled_counts.sum(axis=1)
        sample_influence = sampled_sums - sampled_counts * sample_effect[:, None, :]
        if studentizer == "bartlett":
            sample_se = _bartlett_ratio_standard_error(
                sample_influence, sampled_counts, max_lag=max_lag
            )
        elif studentizer == "adaptive_flat_top":
            sample_flat_top = adaptive_flat_top_ratio_standard_error(
                sample_influence, sampled_counts
            )
            sample_se = sample_flat_top.standard_error
            bootstrap_bandwidth[start:stop] = sample_flat_top.bandwidth
            bootstrap_fallback[start:stop] = sample_flat_top.bartlett_fallback_applied
        else:
            sample_se = self_normalized_ratio_standard_error(
                sample_influence, sampled_counts
            ).standard_error
        bootstrap_t[start:stop] = sample_effect / sample_se
    summary = _stepdown_from_t_values(
        hypothesis_ids,
        observed,
        observed_se,
        bootstrap_t,
        alternative=alternative,
        n_bootstrap=int(n_bootstrap),
        seed=int(seed),
        engine=(
            "adaptive_flat_top_restudentized_circular_block_bootstrap_t"
            if studentizer == "adaptive_flat_top"
            else (
                "self_normalized_circular_block_bootstrap_t"
                if studentizer == "self_normalized"
                else "restudentized_circular_block_bootstrap_t"
            )
        ),
        dependence_length=int(block_length),
    )
    if observed_flat_top is not None:
        summary["adaptive_flat_top_bandwidth_days"] = observed_flat_top.bandwidth
        summary["adaptive_flat_top_raw_long_run_variance"] = (
            observed_flat_top.raw_long_run_variance
        )
        summary["adaptive_flat_top_long_run_variance"] = (
            observed_flat_top.long_run_variance
        )
        summary["adaptive_flat_top_bartlett_fallback_applied"] = (
            observed_flat_top.bartlett_fallback_applied
        )
        summary["bootstrap_median_adaptive_bandwidth_days"] = np.median(
            bootstrap_bandwidth, axis=0
        )
        summary["bootstrap_bartlett_fallback_rate"] = np.mean(
            bootstrap_fallback, axis=0
        )
    if studentizer == "self_normalized":
        summary["self_normalizer"] = observed_self_normalized.self_normalizer
    bootstrap_frame = pd.DataFrame(bootstrap_t, columns=hypothesis_ids)
    bootstrap_frame.index.name = "bootstrap_idx"
    starts_frame = pd.DataFrame(
        starts_array,
        columns=[
            f"block_{index:03d}_start_day_offset"
            for index in range(blocks_per_sample)
        ],
    )
    starts_frame.index.name = "bootstrap_idx"
    return StepdownMaxTBootstrapArtifacts(summary, bootstrap_frame, starts_frame)


def restudentized_circular_block_bootstrap_stepdown_max_t(
    daily_centered_sums: pd.DataFrame,
    daily_counts: pd.DataFrame,
    observed_effects: pd.Series,
    *,
    block_length: int,
    n_bootstrap: int,
    seed: int,
    batch_size: int = 128,
    alternative: str = "greater",
) -> StepdownMaxTBootstrapArtifacts:
    """Production re-studentized synchronized circular-block step-down maxT."""
    return _restudentized_circular_block_stepdown_max_t(
        daily_centered_sums,
        daily_counts,
        observed_effects,
        block_length=block_length,
        n_bootstrap=n_bootstrap,
        seed=seed,
        batch_size=batch_size,
        alternative=alternative,
        minimum_bootstrap_repetitions=10_000,
    )


def simulation_calibration_restudentized_circular_block_stepdown_max_t(
    daily_centered_sums: pd.DataFrame,
    daily_counts: pd.DataFrame,
    observed_effects: pd.Series,
    *,
    block_length: int,
    n_bootstrap: int,
    seed: int,
    batch_size: int = 128,
    alternative: str = "greater",
) -> StepdownMaxTBootstrapArtifacts:
    """Truth-known simulation entry for the E1 inference candidate."""
    return _restudentized_circular_block_stepdown_max_t(
        daily_centered_sums,
        daily_counts,
        observed_effects,
        block_length=block_length,
        n_bootstrap=n_bootstrap,
        seed=seed,
        batch_size=batch_size,
        alternative=alternative,
        minimum_bootstrap_repetitions=999,
    )


def adaptive_flat_top_restudentized_circular_block_bootstrap_stepdown_max_t(
    daily_centered_sums: pd.DataFrame,
    daily_counts: pd.DataFrame,
    observed_effects: pd.Series,
    *,
    block_length: int,
    n_bootstrap: int,
    seed: int,
    batch_size: int = 128,
    alternative: str = "greater",
) -> StepdownMaxTBootstrapArtifacts:
    """Production E1F synchronized block step-down maxT entry."""
    return _restudentized_circular_block_stepdown_max_t(
        daily_centered_sums,
        daily_counts,
        observed_effects,
        block_length=block_length,
        n_bootstrap=n_bootstrap,
        seed=seed,
        batch_size=batch_size,
        alternative=alternative,
        minimum_bootstrap_repetitions=10_000,
        studentizer="adaptive_flat_top",
    )


def simulation_calibration_adaptive_flat_top_restudentized_stepdown_max_t(
    daily_centered_sums: pd.DataFrame,
    daily_counts: pd.DataFrame,
    observed_effects: pd.Series,
    *,
    block_length: int,
    n_bootstrap: int,
    seed: int,
    batch_size: int = 128,
    alternative: str = "greater",
) -> StepdownMaxTBootstrapArtifacts:
    """Truth-known simulation entry for the E1F inference candidate."""
    return _restudentized_circular_block_stepdown_max_t(
        daily_centered_sums,
        daily_counts,
        observed_effects,
        block_length=block_length,
        n_bootstrap=n_bootstrap,
        seed=seed,
        batch_size=batch_size,
        alternative=alternative,
        minimum_bootstrap_repetitions=999,
        studentizer="adaptive_flat_top",
    )


def self_normalized_circular_block_bootstrap_stepdown_max_t(
    daily_centered_sums: pd.DataFrame,
    daily_counts: pd.DataFrame,
    observed_effects: pd.Series,
    *,
    block_length: int,
    n_bootstrap: int,
    seed: int,
    batch_size: int = 128,
    alternative: str = "greater",
) -> StepdownMaxTBootstrapArtifacts:
    """Production E1S synchronized self-normalized step-down maxT entry."""
    return _restudentized_circular_block_stepdown_max_t(
        daily_centered_sums,
        daily_counts,
        observed_effects,
        block_length=block_length,
        n_bootstrap=n_bootstrap,
        seed=seed,
        batch_size=batch_size,
        alternative=alternative,
        minimum_bootstrap_repetitions=10_000,
        studentizer="self_normalized",
    )


def simulation_calibration_self_normalized_stepdown_max_t(
    daily_centered_sums: pd.DataFrame,
    daily_counts: pd.DataFrame,
    observed_effects: pd.Series,
    *,
    block_length: int,
    n_bootstrap: int,
    seed: int,
    batch_size: int = 128,
    alternative: str = "greater",
) -> StepdownMaxTBootstrapArtifacts:
    """Truth-known simulation entry for the E1S inference candidate."""
    return _restudentized_circular_block_stepdown_max_t(
        daily_centered_sums,
        daily_counts,
        observed_effects,
        block_length=block_length,
        n_bootstrap=n_bootstrap,
        seed=seed,
        batch_size=batch_size,
        alternative=alternative,
        minimum_bootstrap_repetitions=999,
        studentizer="self_normalized",
    )


def _dependent_multiplier_stepdown_max_t(
    daily_centered_sums: pd.DataFrame,
    daily_counts: pd.DataFrame,
    observed_effects: pd.Series,
    *,
    bandwidth: int,
    n_bootstrap: int,
    seed: int,
    batch_size: int,
    alternative: str,
    minimum_bootstrap_repetitions: int,
) -> StepdownMaxTBootstrapArtifacts:
    if int(n_bootstrap) < int(minimum_bootstrap_repetitions):
        raise ValueError(
            f"n_bootstrap must be at least {int(minimum_bootstrap_repetitions)}"
        )
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    hypothesis_ids, sums, counts, observed = _validated_daily_family_arrays(
        daily_centered_sums,
        daily_counts,
        observed_effects,
        dependence_length=int(bandwidth),
    )
    max_lag = int(bandwidth) - 1
    observed_se = _bartlett_ratio_standard_error(sums, counts, max_lag=max_lag)
    day_count = len(sums)
    denominator = counts.sum(axis=0)
    rng = np.random.default_rng(int(seed))
    bootstrap_t = np.empty((int(n_bootstrap), len(hypothesis_ids)), dtype=float)
    multipliers = np.empty((int(n_bootstrap), day_count), dtype=float)
    scale = sqrt(float(bandwidth))
    for start in range(0, int(n_bootstrap), int(batch_size)):
        stop = min(start + int(batch_size), int(n_bootstrap))
        innovations = rng.standard_normal((stop - start, day_count))
        weights = sum(
            np.roll(innovations, shift=lag, axis=1)
            for lag in range(int(bandwidth))
        ) / scale
        multipliers[start:stop] = weights
        bootstrap_effect = np.einsum("bd,dh->bh", weights, sums) / denominator
        bootstrap_t[start:stop] = bootstrap_effect / observed_se
    summary = _stepdown_from_t_values(
        hypothesis_ids,
        observed,
        observed_se,
        bootstrap_t,
        alternative=alternative,
        n_bootstrap=int(n_bootstrap),
        seed=int(seed),
        engine="dependent_gaussian_multiplier_bootstrap_t",
        dependence_length=int(bandwidth),
    )
    bootstrap_frame = pd.DataFrame(bootstrap_t, columns=hypothesis_ids)
    bootstrap_frame.index.name = "bootstrap_idx"
    multiplier_frame = pd.DataFrame(
        multipliers,
        columns=[f"day_{index:03d}_multiplier" for index in range(day_count)],
    )
    multiplier_frame.index.name = "bootstrap_idx"
    return StepdownMaxTBootstrapArtifacts(
        summary, bootstrap_frame, multiplier_frame
    )


def dependent_multiplier_bootstrap_stepdown_max_t(
    daily_centered_sums: pd.DataFrame,
    daily_counts: pd.DataFrame,
    observed_effects: pd.Series,
    *,
    bandwidth: int,
    n_bootstrap: int,
    seed: int,
    batch_size: int = 128,
    alternative: str = "greater",
) -> StepdownMaxTBootstrapArtifacts:
    """Production synchronized dependent-multiplier step-down maxT."""
    return _dependent_multiplier_stepdown_max_t(
        daily_centered_sums,
        daily_counts,
        observed_effects,
        bandwidth=bandwidth,
        n_bootstrap=n_bootstrap,
        seed=seed,
        batch_size=batch_size,
        alternative=alternative,
        minimum_bootstrap_repetitions=10_000,
    )


def simulation_calibration_dependent_multiplier_bootstrap_stepdown_max_t(
    daily_centered_sums: pd.DataFrame,
    daily_counts: pd.DataFrame,
    observed_effects: pd.Series,
    *,
    bandwidth: int,
    n_bootstrap: int,
    seed: int,
    batch_size: int = 128,
    alternative: str = "greater",
) -> StepdownMaxTBootstrapArtifacts:
    """Truth-known simulation entry for the E2 inference candidate."""
    return _dependent_multiplier_stepdown_max_t(
        daily_centered_sums,
        daily_counts,
        observed_effects,
        bandwidth=bandwidth,
        n_bootstrap=n_bootstrap,
        seed=seed,
        batch_size=batch_size,
        alternative=alternative,
        minimum_bootstrap_repetitions=999,
    )


def randomization_stepdown_max_t(
    null_effects: pd.DataFrame,
    observed_effects: pd.Series,
    *,
    min_randomizations: int = 9_999,
) -> RandomizationStepdownMaxTArtifacts:
    """One-sided candidate-wise randomization inference with step-down maxT.

    Each hypothesis is standardized by the mean and standard deviation of its
    own joint randomization null. Rows of ``null_effects`` must be synchronized
    randomization realizations across the complete hypothesis family.
    """
    if not isinstance(null_effects, pd.DataFrame):
        raise TypeError("null_effects must be a DataFrame")
    if null_effects.empty:
        raise ValueError("null_effects must not be empty")
    if len(null_effects) < int(min_randomizations):
        raise ValueError(
            f"null_effects must contain at least {int(min_randomizations)} randomizations"
        )
    if null_effects.index.has_duplicates:
        raise ValueError("null_effects contains duplicate randomization ids")
    if null_effects.columns.has_duplicates:
        raise ValueError("null_effects contains duplicate hypothesis ids")
    hypothesis_ids = null_effects.columns.astype(str)
    if len(set(hypothesis_ids)) != len(hypothesis_ids):
        raise ValueError("hypothesis ids are not unique after string normalization")
    null = null_effects.to_numpy(dtype=float)
    if not np.isfinite(null).all():
        raise ValueError("null_effects must be finite")

    observed = pd.Series(observed_effects).copy()
    observed.index = observed.index.astype(str)
    observed = observed.reindex(hypothesis_ids)
    if observed.isna().any() or not np.isfinite(observed.to_numpy(dtype=float)).all():
        raise ValueError("observed_effects must be finite and cover every hypothesis")
    observed_values = observed.to_numpy(dtype=float)

    null_mean = null.mean(axis=0)
    null_sd = null.std(axis=0, ddof=1)
    if not np.isfinite(null_sd).all() or (null_sd <= 0.0).any():
        raise ValueError("randomization null standard deviations must be finite and positive")
    observed_t = (observed_values - null_mean) / null_sd
    null_t = (null - null_mean[None, :]) / null_sd[None, :]

    randomization_count = len(null_effects)
    raw_counts = (null >= observed_values[None, :]).sum(axis=0)
    raw_p = (1.0 + raw_counts) / (randomization_count + 1.0)
    raw_mcse = np.sqrt(raw_p * (1.0 - raw_p) / (randomization_count + 1.0))

    order = np.argsort(-observed_t, kind="mergesort")
    ordered_observed = observed_t[order]
    ordered_null = null_t[:, order]
    suffix_max = np.maximum.accumulate(ordered_null[:, ::-1], axis=1)[:, ::-1]
    step_counts = (suffix_max >= ordered_observed[None, :]).sum(axis=0)
    step_raw = (1.0 + step_counts) / (randomization_count + 1.0)
    ordered_raw = raw_p[order]
    ordered_adjusted = np.maximum.accumulate(np.maximum(step_raw, ordered_raw))
    ordered_adjusted = np.clip(ordered_adjusted, 0.0, 1.0)
    adjusted = np.empty_like(ordered_adjusted)
    adjusted[order] = ordered_adjusted
    if (adjusted + 1e-15 < raw_p).any():
        raise RuntimeError("step-down adjusted p-value fell below marginal p-value")

    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    summary = pd.DataFrame(
        {
            "hypothesis_id": hypothesis_ids,
            "observed_effect": observed_values,
            "null_mean": null_mean,
            "null_std": null_sd,
            "null_median": np.quantile(null, 0.50, axis=0),
            "null_q95": np.quantile(null, 0.95, axis=0),
            "null_q99": np.quantile(null, 0.99, axis=0),
            "observed_null_percentile": (null <= observed_values[None, :]).mean(axis=0),
            "observed_t": observed_t,
            "raw_one_sided_p_value": raw_p,
            "raw_p_mcse": raw_mcse,
            "stepdown_max_t_adjusted_p_value": adjusted,
            "observed_t_descending_rank": ranks,
            "n_randomizations": randomization_count,
        }
    )
    null_t_frame = pd.DataFrame(
        null_t,
        index=null_effects.index.copy(),
        columns=hypothesis_ids,
    )
    null_t_frame.index.name = null_effects.index.name or "randomization_idx"
    return RandomizationStepdownMaxTArtifacts(
        summary=summary,
        null_t_values=null_t_frame,
    )
