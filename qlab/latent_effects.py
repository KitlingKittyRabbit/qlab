"""Formal latent-effect distribution and truth-known simulation contracts.

This module estimates spike-and-positive-slab distributions from noisy effect
estimates.  It is deliberately independent of any one research route: callers
must supply frozen daily effects, parameter domains, and random seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, roots_legendre
from scipy.stats import gamma as gamma_distribution
from scipy.stats import lognorm as lognormal_distribution
from scipy.stats import norm as normal_distribution
from scipy.stats import truncnorm as truncated_normal_distribution

from qlab import method_simulation, research_stats


SUPPORTED_SLAB_FAMILIES = ("truncated_normal", "gamma", "lognormal")


@dataclass(frozen=True)
class StandardizedEffectBootstrapArtifacts:
    hypothesis_summary: pd.DataFrame
    bootstrap_effects: pd.DataFrame
    bootstrap_covariance: pd.DataFrame
    bootstrap_correlation: pd.DataFrame


@dataclass(frozen=True)
class SpikeSlabFit:
    family: str
    pi0: float
    parameter_1: float
    parameter_2: float
    log_likelihood: float
    converged: bool
    boundary_hit: bool
    start_results: pd.DataFrame

    def parameter_record(self) -> dict[str, object]:
        names = slab_parameter_names(self.family)
        return {
            "family": self.family,
            "pi0": self.pi0,
            names[0]: self.parameter_1,
            names[1]: self.parameter_2,
            "log_likelihood": self.log_likelihood,
            "converged": self.converged,
            "boundary_hit": self.boundary_hit,
        }


@dataclass(frozen=True)
class IdentificationDecision:
    switch_to_partial_identification: bool
    reasons: tuple[str, ...]
    diagnostics: pd.DataFrame


@dataclass(frozen=True)
class LikelihoodRatioCalibrationArtifacts:
    replicate_statistics: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class LatentDistributionSimulationArtifacts:
    generated_tasks: pd.DataFrame
    hypothesis_results: pd.DataFrame
    task_summary: pd.DataFrame
    scenario_summary: pd.DataFrame
    conditional_discovery_summary: pd.DataFrame


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size < 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    return result


def standardized_effect_bootstrap(
    daily_effects: pd.DataFrame,
    *,
    block_length: int,
    n_draws: int,
    seed: int,
    batch_size: int = 100,
) -> StandardizedEffectBootstrapArtifacts:
    """Estimate standardized effects and synchronized block-bootstrap error."""
    if not isinstance(daily_effects, pd.DataFrame) or daily_effects.empty:
        raise ValueError("daily_effects must be a non-empty DataFrame")
    if daily_effects.columns.has_duplicates or daily_effects.index.has_duplicates:
        raise ValueError("daily_effects must have unique rows and hypotheses")
    values = daily_effects.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("daily_effects must be finite and complete")
    if values.shape[0] < 3 or values.shape[1] < 2:
        raise ValueError("daily_effects requires at least three days and two hypotheses")
    standard_deviation = values.std(axis=0, ddof=1)
    if (standard_deviation <= 0.0).any():
        raise ValueError("daily_effects contains a constant hypothesis")
    draws = int(n_draws)
    batch = int(batch_size)
    if draws < 2 or batch <= 0:
        raise ValueError("n_draws must exceed one and batch_size must be positive")
    effect_draws = np.empty((draws, values.shape[1]), dtype=float)
    completed = 0
    while completed < draws:
        current = min(batch, draws - completed)
        indices = method_simulation.synchronized_circular_block_indices(
            len(values), int(block_length), draw_count=current,
            seed=int(seed) + completed,
        )
        sampled = values[indices]
        sampled_sd = sampled.std(axis=1, ddof=1)
        if (sampled_sd <= 0.0).any() or not np.isfinite(sampled_sd).all():
            raise RuntimeError("bootstrap produced a constant or invalid hypothesis")
        effect_draws[completed:completed + current] = sampled.mean(axis=1) / sampled_sd
        completed += current

    hypothesis_ids = daily_effects.columns.astype(str)
    draw_frame = pd.DataFrame(effect_draws, columns=hypothesis_ids)
    centered = effect_draws - effect_draws.mean(axis=0, keepdims=True)
    draw_sd = effect_draws.std(axis=0, ddof=1)
    skew = (
        np.mean(centered ** 3, axis=0)
        / np.maximum(np.mean(centered ** 2, axis=0) ** 1.5, np.finfo(float).tiny)
    )
    point = values.mean(axis=0) / standard_deviation
    summary = pd.DataFrame(
        {
            "hypothesis_id": hypothesis_ids,
            "day_count": len(values),
            "observed_standardized_effect": point,
            "bootstrap_standard_error": draw_sd,
            "bootstrap_skewness": skew,
            "bootstrap_q025": np.quantile(effect_draws, 0.025, axis=0),
            "bootstrap_median": np.quantile(effect_draws, 0.5, axis=0),
            "bootstrap_q975": np.quantile(effect_draws, 0.975, axis=0),
            "block_length_days": int(block_length),
            "bootstrap_draw_count": draws,
            "seed": int(seed),
        }
    )
    covariance = pd.DataFrame(
        np.cov(effect_draws, rowvar=False, ddof=1),
        index=hypothesis_ids, columns=hypothesis_ids,
    )
    correlation = pd.DataFrame(
        np.corrcoef(effect_draws, rowvar=False),
        index=hypothesis_ids, columns=hypothesis_ids,
    )
    return StandardizedEffectBootstrapArtifacts(
        hypothesis_summary=summary,
        bootstrap_effects=draw_frame,
        bootstrap_covariance=covariance,
        bootstrap_correlation=correlation,
    )


def slab_parameter_names(family: str) -> tuple[str, str]:
    selected = str(family)
    if selected == "truncated_normal":
        return "location", "scale"
    if selected == "gamma":
        return "shape", "scale"
    if selected == "lognormal":
        return "log_scale", "shape"
    raise ValueError(f"unsupported slab family: {selected}")


def validate_parameter_bounds(
    family: str,
    bounds: Mapping[str, Sequence[float]],
) -> dict[str, tuple[float, float]]:
    names = ("pi0", *slab_parameter_names(family))
    if set(bounds) != set(names):
        raise ValueError("parameter bounds do not match the selected slab family")
    normalized: dict[str, tuple[float, float]] = {}
    for name in names:
        pair = tuple(float(value) for value in bounds[name])
        if len(pair) != 2 or not np.isfinite(pair).all() or pair[0] > pair[1]:
            raise ValueError(f"invalid parameter bounds for {name}")
        normalized[name] = pair
    if normalized["pi0"][0] < 0.0 or normalized["pi0"][1] > 1.0:
        raise ValueError("pi0 bounds must lie within [0, 1]")
    if family in {"truncated_normal", "gamma"} and normalized["scale"][0] <= 0.0:
        raise ValueError("scale lower bound must be positive")
    if family == "gamma" and normalized["shape"][0] <= 0.0:
        raise ValueError("gamma shape lower bound must be positive")
    if family == "lognormal" and normalized["shape"][0] <= 0.0:
        raise ValueError("lognormal shape lower bound must be positive")
    return normalized


def _quadrature_rule(node_count: int) -> tuple[np.ndarray, np.ndarray]:
    count = int(node_count)
    if count < 16:
        raise ValueError("quadrature requires at least 16 nodes")
    nodes, weights = roots_legendre(count)
    return (nodes + 1.0) / 2.0, weights / 2.0


def _slab_quantiles(
    family: str,
    parameter_1: float,
    parameter_2: float,
    probabilities: np.ndarray,
) -> np.ndarray:
    if family == "truncated_normal":
        location = float(parameter_1)
        scale = float(parameter_2)
        return truncated_normal_distribution.ppf(
            probabilities, (0.0 - location) / scale,
            np.inf, loc=location, scale=scale,
        )
    if family == "gamma":
        return gamma_distribution.ppf(
            probabilities, a=float(parameter_1), scale=float(parameter_2)
        )
    if family == "lognormal":
        return lognormal_distribution.ppf(
            probabilities, s=float(parameter_2), scale=np.exp(float(parameter_1))
        )
    raise ValueError(f"unsupported slab family: {family}")


def spike_slab_log_likelihood(
    observed_effects: Sequence[float],
    standard_errors: Sequence[float],
    *,
    family: str,
    pi0: float,
    parameter_1: float,
    parameter_2: float,
    quadrature_nodes: int = 96,
) -> float:
    """Heteroskedastic measurement-error spike-and-slab log likelihood."""
    observed = _finite_vector(observed_effects, "observed_effects")
    errors = _finite_vector(standard_errors, "standard_errors")
    if observed.shape != errors.shape or (errors <= 0.0).any():
        raise ValueError("standard errors must be positive and align with effects")
    zero_probability = float(pi0)
    if not 0.0 <= zero_probability <= 1.0:
        raise ValueError("pi0 must lie within [0, 1]")
    probabilities, weights = _quadrature_rule(int(quadrature_nodes))
    latent = _slab_quantiles(
        str(family), float(parameter_1), float(parameter_2), probabilities
    )
    if not np.isfinite(latent).all() or (latent <= 0.0).any():
        return float("-inf")
    slab_log_density = logsumexp(
        normal_distribution.logpdf(
            observed[:, None], loc=latent[None, :], scale=errors[:, None]
        ) + np.log(weights)[None, :],
        axis=1,
    )
    null_log_density = normal_distribution.logpdf(observed, loc=0.0, scale=errors)
    if zero_probability == 0.0:
        mixture_log_density = slab_log_density
    elif zero_probability == 1.0:
        mixture_log_density = null_log_density
    else:
        mixture_log_density = np.logaddexp(
            np.log(zero_probability) + null_log_density,
            np.log1p(-zero_probability) + slab_log_density,
        )
    return float(np.sum(mixture_log_density))


def _default_starts(bounds: Mapping[str, tuple[float, float]]) -> tuple[tuple[float, ...], ...]:
    names = tuple(bounds)
    fractions = (
        (0.00, 0.00, 0.00),
        (1.00, 1.00, 1.00),
        (0.00, 1.00, 0.50),
        (1.00, 0.00, 0.50),
        (0.25, 0.25, 0.75),
        (0.25, 0.75, 0.25),
        (0.50, 0.50, 0.50),
        (0.75, 0.25, 0.25),
        (0.75, 0.75, 0.75),
    )
    return tuple(
        tuple(
            float(np.clip(
                bounds[name][0] + fraction[index] * (bounds[name][1] - bounds[name][0]),
                bounds[name][0],
                bounds[name][1],
            ))
            for index, name in enumerate(names)
        )
        for fraction in fractions
    )


def fit_spike_slab_measurement_model(
    observed_effects: Sequence[float],
    standard_errors: Sequence[float],
    *,
    family: str,
    parameter_bounds: Mapping[str, Sequence[float]],
    starts: Sequence[Sequence[float]] | None = None,
    quadrature_nodes: int = 96,
    optimizer_tolerance: float = 1e-9,
) -> SpikeSlabFit:
    """Fit one registered spike-and-positive-slab measurement model."""
    selected_family = str(family)
    if selected_family not in SUPPORTED_SLAB_FAMILIES:
        raise ValueError(f"unsupported slab family: {selected_family}")
    observed = _finite_vector(observed_effects, "observed_effects")
    errors = _finite_vector(standard_errors, "standard_errors")
    if observed.shape != errors.shape or (errors <= 0.0).any():
        raise ValueError("standard errors must be positive and align with effects")
    normalized = validate_parameter_bounds(selected_family, parameter_bounds)
    names = tuple(normalized)
    scipy_bounds = tuple(normalized[name] for name in names)
    start_values = tuple(tuple(float(value) for value in row) for row in (
        starts if starts is not None else _default_starts(normalized)
    ))
    if not start_values:
        raise ValueError("at least one optimization start is required")
    for row in start_values:
        if len(row) != len(names):
            raise ValueError("optimization starts do not match parameter count")
        if any(value < scipy_bounds[index][0] or value > scipy_bounds[index][1]
               for index, value in enumerate(row)):
            raise ValueError("optimization start lies outside parameter bounds")

    def objective(values: np.ndarray) -> float:
        likelihood = spike_slab_log_likelihood(
            observed, errors, family=selected_family,
            pi0=float(values[0]), parameter_1=float(values[1]),
            parameter_2=float(values[2]), quadrature_nodes=int(quadrature_nodes),
        )
        return float("inf") if not np.isfinite(likelihood) else -likelihood

    rows: list[dict[str, object]] = []
    fitted = []
    for start_index, start in enumerate(start_values):
        result = minimize(
            objective, np.asarray(start, dtype=float), method="L-BFGS-B",
            bounds=scipy_bounds,
            options={"ftol": float(optimizer_tolerance), "gtol": float(optimizer_tolerance)},
        )
        values = np.asarray(result.x, dtype=float)
        likelihood = -float(result.fun) if np.isfinite(result.fun) else float("-inf")
        rows.append(
            {
                "start_index": start_index,
                "start_pi0": start[0],
                f"start_{names[1]}": start[1],
                f"start_{names[2]}": start[2],
                "pi0": values[0],
                names[1]: values[1],
                names[2]: values[2],
                "log_likelihood": likelihood,
                "optimizer_success": bool(result.success),
                "optimizer_status": int(getattr(result, "status", 0)),
                "optimizer_message": str(result.message),
            }
        )
        if np.isfinite(likelihood) and bool(result.success):
            fitted.append((likelihood, bool(result.success), values))
    if not fitted:
        raise RuntimeError("every spike-and-slab optimization start failed")
    best_likelihood, best_success, best_values = max(fitted, key=lambda item: item[0])
    boundary = any(
        np.isclose(best_values[index], scipy_bounds[index][0], atol=1e-7, rtol=0.0)
        or np.isclose(best_values[index], scipy_bounds[index][1], atol=1e-7, rtol=0.0)
        for index in range(len(best_values))
    )
    return SpikeSlabFit(
        family=selected_family,
        pi0=float(best_values[0]),
        parameter_1=float(best_values[1]),
        parameter_2=float(best_values[2]),
        log_likelihood=float(best_likelihood),
        converged=bool(best_success),
        boundary_hit=bool(boundary),
        start_results=pd.DataFrame(rows),
    )


def profile_spike_slab_parameter(
    observed_effects: Sequence[float],
    standard_errors: Sequence[float],
    *,
    family: str,
    parameter_bounds: Mapping[str, Sequence[float]],
    parameter_name: str,
    parameter_grid: Sequence[float],
    quadrature_nodes: int = 96,
) -> pd.DataFrame:
    """Profile one named parameter with multi-start optimization of the others."""
    normalized = validate_parameter_bounds(str(family), parameter_bounds)
    selected = str(parameter_name)
    if selected not in normalized:
        raise ValueError("profile parameter is not registered for this family")
    rows: list[dict[str, object]] = []
    for value in tuple(float(item) for item in parameter_grid):
        if value < normalized[selected][0] or value > normalized[selected][1]:
            raise ValueError("profile point lies outside parameter bounds")
        fixed = dict(normalized)
        fixed[selected] = (value, value)
        fit = fit_spike_slab_measurement_model(
            observed_effects,
            standard_errors,
            family=str(family),
            parameter_bounds=fixed,
            quadrature_nodes=int(quadrature_nodes),
        )
        record = fit.parameter_record()
        record["profile_parameter"] = selected
        record["profile_value"] = value
        rows.append(record)
    frame = pd.DataFrame(rows)
    maximum = float(frame["log_likelihood"].max())
    frame["likelihood_ratio_from_profile_max"] = 2.0 * (
        maximum - frame["log_likelihood"].astype(float)
    )
    return frame


def combine_spike_slab_profile_chunks(profile_chunks: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Combine complete profile shards and restore family-wide LR distances."""
    chunks = tuple(profile_chunks)
    if not chunks:
        raise ValueError("at least one profile chunk is required")
    frame = pd.concat(chunks, ignore_index=True)
    required = {
        "profile_parameter", "profile_value", "log_likelihood", "converged"
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("profile chunks missing columns: " + ", ".join(missing))
    if frame.duplicated(["profile_parameter", "profile_value"]).any():
        raise ValueError("profile chunks contain duplicate parameter points")
    if not frame["converged"].astype(bool).all():
        raise RuntimeError("profile chunks contain unconverged points")
    if not np.isfinite(frame["log_likelihood"].to_numpy(dtype=float)).all():
        raise RuntimeError("profile chunks contain invalid likelihoods")
    maxima = frame.groupby("profile_parameter")["log_likelihood"].transform("max")
    frame["likelihood_ratio_from_profile_max"] = 2.0 * (
        maxima.astype(float) - frame["log_likelihood"].astype(float)
    )
    return frame.sort_values(
        ["profile_parameter", "profile_value"], kind="mergesort"
    ).reset_index(drop=True)


def profile_spike_slab_pi0(
    observed_effects: Sequence[float],
    standard_errors: Sequence[float],
    *,
    family: str,
    parameter_bounds: Mapping[str, Sequence[float]],
    pi0_grid: Sequence[float],
    quadrature_nodes: int = 96,
) -> pd.DataFrame:
    """Profile a registered slab family over an explicit pi0 grid."""
    frame = profile_spike_slab_parameter(
        observed_effects,
        standard_errors,
        family=str(family),
        parameter_bounds=parameter_bounds,
        parameter_name="pi0",
        parameter_grid=pi0_grid,
        quadrature_nodes=int(quadrature_nodes),
    )
    return frame


def evaluate_parameter_acceptance_grid(
    observed_effects: Sequence[float],
    standard_errors: Sequence[float],
    parameter_grid: pd.DataFrame,
    *,
    family: str,
    maximum_log_likelihood: float,
    likelihood_ratio_critical_value: float,
    quadrature_nodes: int = 96,
) -> pd.DataFrame:
    """Evaluate one frozen parameter grid against a calibrated LR acceptance rule."""
    names = ("pi0", *slab_parameter_names(str(family)))
    if not isinstance(parameter_grid, pd.DataFrame) or parameter_grid.empty:
        raise ValueError("parameter_grid must be a non-empty DataFrame")
    missing = sorted(set(names).difference(parameter_grid.columns))
    if missing:
        raise ValueError("parameter_grid missing columns: " + ", ".join(missing))
    if parameter_grid.loc[:, names].isna().any().any():
        raise ValueError("parameter_grid must be complete")
    maximum = float(maximum_log_likelihood)
    critical = float(likelihood_ratio_critical_value)
    if not np.isfinite(maximum) or not np.isfinite(critical) or critical < 0.0:
        raise ValueError("likelihood acceptance inputs are invalid")
    output = parameter_grid.copy()
    likelihoods = []
    for row in output.loc[:, names].itertuples(index=False, name=None):
        likelihoods.append(
            spike_slab_log_likelihood(
                observed_effects, standard_errors, family=str(family),
                pi0=float(row[0]), parameter_1=float(row[1]),
                parameter_2=float(row[2]), quadrature_nodes=int(quadrature_nodes),
            )
        )
    output["log_likelihood"] = likelihoods
    output["likelihood_ratio"] = 2.0 * (
        maximum - output["log_likelihood"].astype(float)
    )
    output["accepted"] = output["likelihood_ratio"].le(critical)
    return output


def assess_identification(
    bootstrap_parameter_fits: pd.DataFrame,
    likelihood_profile: pd.DataFrame,
    *,
    parameter_columns: Sequence[str],
    parameter_bounds: Mapping[str, Sequence[float]],
    equality_tolerance: float,
    profile_likelihood_ratio_critical_value: float,
    point_start_results: pd.DataFrame | None = None,
    equivalent_likelihood_tolerance: float = 1e-6,
) -> IdentificationDecision:
    """Apply the preregistered finite-interval identification failover rules."""
    columns = tuple(str(value) for value in parameter_columns)
    required = {*columns, "converged", "log_likelihood"}
    missing = sorted(required.difference(bootstrap_parameter_fits.columns))
    if missing:
        raise ValueError("bootstrap fits missing columns: " + ", ".join(missing))
    normalized = {name: tuple(float(value) for value in parameter_bounds[name]) for name in columns}
    tolerance = float(equality_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("equality_tolerance must be finite and positive")
    critical = float(profile_likelihood_ratio_critical_value)
    if not np.isfinite(critical) or critical < 0.0:
        raise ValueError("profile likelihood-ratio critical value must be finite and nonnegative")
    successful = bootstrap_parameter_fits.loc[
        bootstrap_parameter_fits["converged"].astype(bool)
    ].copy()
    if successful.empty:
        return IdentificationDecision(
            True, ("no_converged_bootstrap_fits",),
            pd.DataFrame([{"diagnostic": "converged_fraction", "value": 0.0}]),
        )
    reasons: list[str] = []
    diagnostics: list[dict[str, object]] = []
    for name in columns:
        values = successful[name].astype(float).to_numpy()
        lower, upper = normalized[name]
        q025, q975 = np.quantile(values, [0.025, 0.975])
        lower_hit = bool(np.isclose(values, lower, atol=tolerance, rtol=0.0).mean() >= 0.025)
        upper_hit = bool(np.isclose(values, upper, atol=tolerance, rtol=0.0).mean() >= 0.025)
        diagnostics.append(
            {
                "diagnostic": f"{name}_bootstrap_interval",
                "value": float(q975 - q025),
                "q025": float(q025),
                "q975": float(q975),
                "lower_boundary_mass_ge_025": lower_hit,
                "upper_boundary_mass_ge_025": upper_hit,
            }
        )
        if lower_hit or upper_hit:
            reasons.append(f"{name}_bootstrap_interval_hits_boundary")
    profile_required = {
        "profile_parameter", "profile_value", "likelihood_ratio_from_profile_max"
    }
    if not profile_required.issubset(likelihood_profile.columns):
        raise ValueError("likelihood_profile lacks named likelihood-ratio columns")
    for parameter, parameter_profile in likelihood_profile.groupby(
        "profile_parameter", sort=True
    ):
        ordered = parameter_profile.sort_values("profile_value").reset_index(drop=True)
        accepted_mask = ordered["likelihood_ratio_from_profile_max"].le(critical).to_numpy()
        if not accepted_mask.any():
            reasons.append(f"{parameter}_profile_has_no_95pct_points")
            continue
        accepted_positions = np.flatnonzero(accepted_mask)
        if accepted_positions[0] == 0:
            reasons.append(f"{parameter}_profile_interval_open_at_lower_domain")
        if accepted_positions[-1] == len(ordered) - 1:
            reasons.append(f"{parameter}_profile_interval_open_at_upper_domain")
        accepted_region_count = int(accepted_mask[0]) + int(
            np.sum(np.diff(accepted_mask.astype(int)) == 1)
        )
        if accepted_region_count > 1:
            reasons.append(f"{parameter}_profile_has_disconnected_regions")
    if point_start_results is not None:
        required_starts = {"log_likelihood", "optimizer_success", *columns}
        missing_starts = sorted(required_starts.difference(point_start_results.columns))
        if missing_starts:
            raise ValueError("point start results missing columns: " + ", ".join(missing_starts))
        successful_starts = point_start_results.loc[
            point_start_results["optimizer_success"].astype(bool)
        ].copy()
        if successful_starts.empty:
            reasons.append("no_converged_point_fit_starts")
        else:
            maximum = float(successful_starts["log_likelihood"].max())
            equivalent = successful_starts.loc[
                successful_starts["log_likelihood"].ge(
                    maximum - float(equivalent_likelihood_tolerance)
                )
            ]
            for left in range(len(equivalent)):
                for right in range(left + 1, len(equivalent)):
                    left_values = equivalent.iloc[left][list(columns)].to_numpy(dtype=float)
                    right_values = equivalent.iloc[right][list(columns)].to_numpy(dtype=float)
                    if not np.allclose(
                        left_values, right_values, atol=tolerance, rtol=0.0
                    ):
                        reasons.append("equivalent_likelihood_distinct_parameter_solutions")
                        break
                if "equivalent_likelihood_distinct_parameter_solutions" in reasons:
                    break
    diagnostics.append(
        {
            "diagnostic": "converged_fraction",
            "value": float(len(successful) / len(bootstrap_parameter_fits)),
        }
    )
    return IdentificationDecision(
        switch_to_partial_identification=bool(reasons),
        reasons=tuple(sorted(set(reasons))),
        diagnostics=pd.DataFrame(diagnostics),
    )


def calibrate_composite_likelihood_ratio(
    bootstrap_effects: pd.DataFrame,
    standard_errors: pd.Series,
    bootstrap_parameter_fits: pd.DataFrame,
    reference_fit: SpikeSlabFit,
    *,
    quantile: float = 0.95,
    quadrature_nodes: int = 96,
) -> LikelihoodRatioCalibrationArtifacts:
    """Calibrate a composite-likelihood acceptance cutoff from synchronized draws.

    Each row compares that bootstrap draw's fitted optimum with the likelihood
    of the original-data fit evaluated on the same draw.  Bootstrap identities
    must align one-to-one; failed fits are retained in the audit and excluded
    from the finite cutoff only when explicitly marked unconverged.
    """
    if not isinstance(bootstrap_effects, pd.DataFrame) or bootstrap_effects.empty:
        raise ValueError("bootstrap_effects must be a non-empty DataFrame")
    if bootstrap_effects.index.has_duplicates or bootstrap_effects.columns.has_duplicates:
        raise ValueError("bootstrap_effects identities must be unique")
    if not isinstance(standard_errors, pd.Series) or tuple(
        standard_errors.index.astype(str)
    ) != tuple(bootstrap_effects.columns.astype(str)):
        raise ValueError("standard_errors must align with bootstrap hypotheses")
    if not np.isfinite(bootstrap_effects.to_numpy(dtype=float)).all():
        raise ValueError("bootstrap_effects must be finite")
    if not np.isfinite(standard_errors.to_numpy(dtype=float)).all() or (
        standard_errors.to_numpy(dtype=float) <= 0.0
    ).any():
        raise ValueError("standard_errors must be finite and positive")
    required = {"bootstrap_idx", "log_likelihood", "converged"}
    missing = sorted(required.difference(bootstrap_parameter_fits.columns))
    if missing:
        raise ValueError("bootstrap_parameter_fits missing: " + ", ".join(missing))
    if bootstrap_parameter_fits["bootstrap_idx"].duplicated().any():
        raise ValueError("bootstrap_parameter_fits contains duplicate identities")
    expected = set(range(len(bootstrap_effects)))
    observed_ids = set(bootstrap_parameter_fits["bootstrap_idx"].astype(int))
    if observed_ids != expected:
        raise ValueError("bootstrap fit identities do not cover every effect draw")
    probability = float(quantile)
    if not 0.5 < probability < 1.0:
        raise ValueError("quantile must lie strictly between 0.5 and 1")
    fit_table = bootstrap_parameter_fits.set_index("bootstrap_idx").sort_index()
    rows: list[dict[str, object]] = []
    errors = standard_errors.to_numpy(dtype=float)
    for bootstrap_idx, effects in enumerate(
        bootstrap_effects.to_numpy(dtype=float)
    ):
        row = fit_table.loc[bootstrap_idx]
        converged = bool(row["converged"])
        fitted_likelihood = float(row["log_likelihood"])
        reference_likelihood = spike_slab_log_likelihood(
            effects,
            errors,
            family=reference_fit.family,
            pi0=reference_fit.pi0,
            parameter_1=reference_fit.parameter_1,
            parameter_2=reference_fit.parameter_2,
            quadrature_nodes=int(quadrature_nodes),
        )
        statistic = (
            max(0.0, 2.0 * (fitted_likelihood - reference_likelihood))
            if converged and np.isfinite(fitted_likelihood) and np.isfinite(reference_likelihood)
            else np.nan
        )
        rows.append(
            {
                "bootstrap_idx": bootstrap_idx,
                "converged": converged,
                "fitted_log_likelihood": fitted_likelihood,
                "reference_log_likelihood": reference_likelihood,
                "likelihood_ratio_statistic": statistic,
            }
        )
    replicate = pd.DataFrame(rows)
    finite = replicate["likelihood_ratio_statistic"].dropna().to_numpy(dtype=float)
    if len(finite) < max(2, int(np.ceil(0.9 * len(replicate)))):
        raise RuntimeError("fewer than 90% of bootstrap LR calibrations are finite")
    critical = float(np.quantile(finite, probability, method="higher"))
    summary = pd.DataFrame(
        [
            {
                "family": reference_fit.family,
                "bootstrap_draw_count": len(replicate),
                "finite_draw_count": len(finite),
                "finite_fraction": len(finite) / len(replicate),
                "quantile": probability,
                "likelihood_ratio_critical_value": critical,
                "median_likelihood_ratio": float(np.median(finite)),
                "maximum_likelihood_ratio": float(np.max(finite)),
            }
        ]
    )
    return LikelihoodRatioCalibrationArtifacts(replicate, summary)


def refit_spike_slab_bootstrap(
    bootstrap_effects: pd.DataFrame,
    standard_errors: pd.Series,
    *,
    family: str,
    parameter_bounds: Mapping[str, Sequence[float]],
    quadrature_nodes: int = 96,
) -> pd.DataFrame:
    """Refit a registered measurement model to every synchronized draw."""
    if not isinstance(bootstrap_effects, pd.DataFrame) or bootstrap_effects.empty:
        raise ValueError("bootstrap_effects must be a non-empty DataFrame")
    if bootstrap_effects.index.has_duplicates or bootstrap_effects.columns.has_duplicates:
        raise ValueError("bootstrap effect identities must be unique")
    if not isinstance(standard_errors, pd.Series) or tuple(
        standard_errors.index.astype(str)
    ) != tuple(bootstrap_effects.columns.astype(str)):
        raise ValueError("standard_errors must align with bootstrap hypotheses")
    if not np.isfinite(bootstrap_effects.to_numpy(dtype=float)).all():
        raise ValueError("bootstrap_effects must be finite")
    names = slab_parameter_names(str(family))
    rows: list[dict[str, object]] = []
    for bootstrap_idx, values in enumerate(bootstrap_effects.to_numpy(dtype=float)):
        try:
            fit = fit_spike_slab_measurement_model(
                values,
                standard_errors.to_numpy(dtype=float),
                family=str(family),
                parameter_bounds=parameter_bounds,
                quadrature_nodes=int(quadrature_nodes),
            )
            record = fit.parameter_record()
            record.update({"bootstrap_idx": bootstrap_idx, "error": ""})
        except (RuntimeError, ValueError, FloatingPointError) as error:
            record = {
                "bootstrap_idx": bootstrap_idx,
                "family": str(family),
                "pi0": np.nan,
                names[0]: np.nan,
                names[1]: np.nan,
                "log_likelihood": np.nan,
                "converged": False,
                "boundary_hit": False,
                "error": f"{type(error).__name__}: {error}",
            }
        rows.append(record)
    return pd.DataFrame(rows).sort_values("bootstrap_idx", kind="mergesort").reset_index(drop=True)


def sample_spike_slab_effects(
    fit: SpikeSlabFit,
    *,
    size: int,
    seed: int,
) -> np.ndarray:
    """Draw standardized truth effects from one frozen fitted distribution."""
    count = int(size)
    if count <= 0:
        raise ValueError("size must be positive")
    generator = np.random.default_rng(int(seed))
    active = generator.random(count) >= float(fit.pi0)
    result = np.zeros(count, dtype=float)
    active_count = int(active.sum())
    if active_count == 0:
        return result
    if fit.family == "truncated_normal":
        result[active] = truncated_normal_distribution.rvs(
            (0.0 - fit.parameter_1) / fit.parameter_2, np.inf,
            loc=fit.parameter_1, scale=fit.parameter_2,
            size=active_count, random_state=generator,
        )
    elif fit.family == "gamma":
        result[active] = generator.gamma(
            shape=fit.parameter_1, scale=fit.parameter_2, size=active_count
        )
    elif fit.family == "lognormal":
        result[active] = generator.lognormal(
            mean=fit.parameter_1, sigma=fit.parameter_2, size=active_count
        )
    else:
        raise ValueError(f"unsupported slab family: {fit.family}")
    if not np.isfinite(result).all() or (result < 0.0).any():
        raise RuntimeError("latent effect draw is invalid")
    return result


def prepare_latent_null_task_base(
    null_daily_effects: pd.DataFrame,
    reference_standard_errors: pd.Series,
    *,
    replicate: int,
    block_length: int,
    noise_seed: int,
) -> pd.DataFrame:
    """Prepare one reusable empirical-noise task before assigning latent truth."""
    if not isinstance(null_daily_effects, pd.DataFrame) or null_daily_effects.empty:
        raise ValueError("null_daily_effects must be a non-empty DataFrame")
    hypotheses = tuple(null_daily_effects.columns.astype(str))
    expected = tuple(f"H{index:02d}" for index in range(1, len(hypotheses) + 1))
    if hypotheses != expected:
        raise ValueError("latent task requires ordered H01-Hnn hypotheses")
    values = null_daily_effects.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.allclose(
        values.mean(axis=0), 0.0, atol=1e-12, rtol=0.0
    ):
        raise ValueError("null_daily_effects must be finite and zero mean")
    if not isinstance(reference_standard_errors, pd.Series) or tuple(
        reference_standard_errors.index.astype(str)
    ) != hypotheses:
        raise ValueError("reference_standard_errors must match the hypothesis family")
    if not np.isfinite(reference_standard_errors.to_numpy(dtype=float)).all() or (
        reference_standard_errors.to_numpy(dtype=float) <= 0.0
    ).any():
        raise ValueError("reference_standard_errors must be finite and positive")
    indices = method_simulation.synchronized_circular_block_indices(
        len(values), int(block_length), draw_count=1, seed=int(noise_seed)
    )[0]
    sampled = null_daily_effects.iloc[indices].copy()
    sampled.index = pd.date_range(
        "2000-01-01", periods=len(sampled), freq="D", tz="UTC",
        name="simulated_day",
    )
    observed_null = sampled.mean(axis=0)
    centered = sampled.subtract(observed_null, axis="columns")
    counts = pd.DataFrame(1, index=centered.index, columns=centered.columns)
    marginal = research_stats.autoregressive_spectral_bh_test(
        centered, counts, observed_null,
        order_criterion="BIC", expected_hypothesis_count=len(hypotheses),
        alternative="greater",
    ).summary.set_index("hypothesis_id")
    return pd.DataFrame(
        {
            "replicate": int(replicate),
            "hypothesis_id": hypotheses,
            "null_observed_effect": observed_null.to_numpy(dtype=float),
            "uncalibrated_standard_error": marginal.loc[
                list(hypotheses), "uncalibrated_standard_error"
            ].to_numpy(dtype=float),
            "reference_standard_error": reference_standard_errors.to_numpy(dtype=float),
            "block_length_days": int(block_length),
            "noise_seed": int(noise_seed),
        }
    )


def apply_latent_fit_to_null_task_bases(
    task_bases: pd.DataFrame,
    null_daily_standard_deviations: pd.Series,
    fit: SpikeSlabFit,
    *,
    scenario_id: str,
    truth_seed_base: int,
    all_null: bool = False,
) -> pd.DataFrame:
    """Assign reproducible latent truth to complete reusable noise-task bases."""
    required = {
        "replicate", "hypothesis_id", "null_observed_effect",
        "uncalibrated_standard_error", "reference_standard_error",
        "block_length_days", "noise_seed",
    }
    if not isinstance(task_bases, pd.DataFrame) or task_bases.empty:
        raise ValueError("task_bases must be a non-empty DataFrame")
    missing = sorted(required.difference(task_bases.columns))
    if missing:
        raise ValueError("task_bases missing columns: " + ", ".join(missing))
    hypotheses = tuple(null_daily_standard_deviations.index.astype(str))
    if not hypotheses or not np.isfinite(
        null_daily_standard_deviations.to_numpy(dtype=float)
    ).all() or (null_daily_standard_deviations.to_numpy(dtype=float) <= 0.0).any():
        raise ValueError("null_daily_standard_deviations must be finite and positive")
    rows = []
    for replicate, task in task_bases.groupby("replicate", sort=True):
        ordered = task.sort_values("hypothesis_id", kind="mergesort").reset_index(drop=True)
        if tuple(ordered["hypothesis_id"].astype(str)) != hypotheses:
            raise ValueError("each task base must contain the complete ordered family")
        standardized = (
            np.zeros(len(hypotheses), dtype=float)
            if bool(all_null)
            else sample_spike_slab_effects(
                fit, size=len(hypotheses), seed=int(truth_seed_base) + int(replicate)
            )
        )
        true_effect = standardized * null_daily_standard_deviations.to_numpy(dtype=float)
        assigned = ordered.copy()
        assigned["registered_task_idx"] = int(replicate)
        assigned["scenario_id"] = str(scenario_id)
        assigned["analysis_specification"] = (
            f"{scenario_id}__right_tail_primary__latent_effect"
        )
        assigned["observed_effect"] = (
            assigned["null_observed_effect"].to_numpy(dtype=float) + true_effect
        )
        assigned["alternative"] = "greater"
        assigned["true_effect"] = true_effect
        assigned["standardized_true_effect"] = standardized
        assigned["all_null"] = bool(all_null)
        assigned["latent_family"] = fit.family
        assigned["latent_pi0"] = fit.pi0
        assigned["latent_parameter_1"] = fit.parameter_1
        assigned["latent_parameter_2"] = fit.parameter_2
        assigned["truth_seed"] = int(truth_seed_base) + int(replicate)
        rows.append(assigned)
    return pd.concat(rows, ignore_index=True)


def summarize_latent_truth_performance(
    task_summary: pd.DataFrame,
    hypothesis_results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize task-mean and pooled truth-known performance with MC error."""
    task_required = {
        "dataset_id", "method_variant", "scenario_id", "analysis_family",
        "registered_task_idx", "false_discovery_proportion", "true_positive_rate",
        "discovery_count",
    }
    hypothesis_required = {
        "dataset_id", "method_variant", "scenario_id", "analysis_family",
        "registered_task_idx", "discovered", "is_true_alternative",
    }
    for name, frame, required in (
        ("task_summary", task_summary, task_required),
        ("hypothesis_results", hypothesis_results, hypothesis_required),
    ):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"{name} must be a non-empty DataFrame")
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} missing columns: " + ", ".join(missing))
    keys = ["dataset_id", "method_variant", "scenario_id", "analysis_family"]
    task_rows = []
    for identity, group in task_summary.groupby(keys, sort=True):
        fdp = group["false_discovery_proportion"].astype(float).to_numpy()
        tpr = group["true_positive_rate"].astype(float).dropna().to_numpy()
        task_count = len(group)
        fdr_se = float(np.std(fdp, ddof=1) / np.sqrt(task_count)) if task_count > 1 else np.nan
        tpr_se = float(np.std(tpr, ddof=1) / np.sqrt(len(tpr))) if len(tpr) > 1 else np.nan
        row = dict(zip(keys, identity))
        row.update(
            {
                "task_count": task_count,
                "fdr": float(np.mean(fdp)),
                "fdr_monte_carlo_standard_error": fdr_se,
                "fdr_ci95_lower": max(0.0, float(np.mean(fdp)) - 1.96 * fdr_se),
                "fdr_ci95_upper": min(1.0, float(np.mean(fdp)) + 1.96 * fdr_se),
                "task_mean_tpr": float(np.mean(tpr)) if len(tpr) else np.nan,
                "tpr_defined_task_count": len(tpr),
                "task_mean_tpr_monte_carlo_standard_error": tpr_se,
                "task_mean_tpr_ci95_lower": (
                    max(0.0, float(np.mean(tpr)) - 1.96 * tpr_se) if len(tpr) > 1 else np.nan
                ),
                "task_mean_tpr_ci95_upper": (
                    min(1.0, float(np.mean(tpr)) + 1.96 * tpr_se) if len(tpr) > 1 else np.nan
                ),
                "mean_discovery_count": float(group["discovery_count"].mean()),
                "no_discovery_rate": float(group["discovery_count"].eq(0).mean()),
                "discovery_count_q50": float(group["discovery_count"].quantile(0.50)),
                "discovery_count_q90": float(group["discovery_count"].quantile(0.90)),
                "discovery_count_q95": float(group["discovery_count"].quantile(0.95)),
                "discovery_count_q99": float(group["discovery_count"].quantile(0.99)),
            }
        )
        task_rows.append(row)
    summary = pd.DataFrame(task_rows)
    hypothesis = hypothesis_results.copy()
    hypothesis["true_discovery"] = (
        hypothesis["discovered"].astype(bool)
        & hypothesis["is_true_alternative"].astype(bool)
    )
    pooled = hypothesis.groupby(keys, sort=True, as_index=False).agg(
        true_alternative_count=("is_true_alternative", "sum"),
        true_discovery_count=("true_discovery", "sum"),
    )
    pooled["pooled_true_effect_discovery_rate"] = np.where(
        pooled["true_alternative_count"].gt(0),
        pooled["true_discovery_count"] / pooled["true_alternative_count"],
        np.nan,
    )
    return summary.merge(pooled, on=keys, how="left", validate="one_to_one")


def summarize_latent_task_level_metrics(
    task_summary: pd.DataFrame,
    *,
    family_size: int,
) -> pd.DataFrame:
    """Summarize extended Monte Carlo metrics from complete task-level rows."""
    required = {
        "dataset_id", "method_variant", "scenario_id", "analysis_family",
        "registered_task_idx", "discovery_count", "true_discovery_count",
        "true_alternative_count",
    }
    if not isinstance(task_summary, pd.DataFrame) or task_summary.empty:
        raise ValueError("task_summary must be a non-empty DataFrame")
    missing = sorted(required.difference(task_summary.columns))
    if missing:
        raise ValueError("task_summary missing columns: " + ", ".join(missing))
    size = int(family_size)
    if size < 1:
        raise ValueError("family_size must be positive")
    keys = ["dataset_id", "method_variant", "scenario_id", "analysis_family"]
    identity = keys + ["registered_task_idx"]
    if task_summary.duplicated(identity).any():
        raise ValueError("task_summary contains duplicate task identities")

    def proportion_interval(successes: int, count: int) -> tuple[float, float]:
        z = 1.96
        estimate = successes / count
        denominator = 1.0 + z**2 / count
        center = (estimate + z**2 / (2.0 * count)) / denominator
        radius = (
            z
            * np.sqrt(estimate * (1.0 - estimate) / count + z**2 / (4.0 * count**2))
            / denominator
        )
        return max(0.0, center - radius), min(1.0, center + radius)

    rows = []
    for group_identity, group in task_summary.groupby(keys, sort=True):
        discovery = group["discovery_count"].to_numpy(dtype=float)
        true_discovery = group["true_discovery_count"].to_numpy(dtype=float)
        true_alternative = group["true_alternative_count"].to_numpy(dtype=float)
        if (
            not np.isfinite(discovery).all()
            or not np.isfinite(true_discovery).all()
            or not np.isfinite(true_alternative).all()
            or (discovery < 0.0).any()
            or (discovery > size).any()
            or (true_alternative < 0.0).any()
            or (true_alternative > size).any()
            or (true_discovery < 0.0).any()
            or (true_discovery > true_alternative).any()
        ):
            raise ValueError("task_summary contains invalid count values")
        count = len(group)
        if count < 2:
            raise ValueError("each task group must contain at least two tasks")
        mean_discovery = float(np.mean(discovery))
        discovery_se = float(np.std(discovery, ddof=1) / np.sqrt(count))
        no_discovery_count = int(np.count_nonzero(discovery == 0.0))
        all_null_count = int(np.count_nonzero(true_alternative == 0.0))
        no_discovery_rate = no_discovery_count / count
        all_null_rate = all_null_count / count
        no_discovery_se = float(
            np.sqrt(no_discovery_rate * (1.0 - no_discovery_rate) / count)
        )
        all_null_se = float(np.sqrt(all_null_rate * (1.0 - all_null_rate) / count))
        no_discovery_interval = proportion_interval(no_discovery_count, count)
        all_null_interval = proportion_interval(all_null_count, count)

        alternative_total = float(np.sum(true_alternative))
        discovery_total = float(np.sum(true_discovery))
        if alternative_total > 0.0:
            pooled_tpr = discovery_total / alternative_total
            mean_alternative = alternative_total / count
            influence = (
                true_discovery - pooled_tpr * true_alternative
            ) / mean_alternative
            pooled_tpr_se = float(np.std(influence, ddof=1) / np.sqrt(count))
            pooled_tpr_lower = max(0.0, pooled_tpr - 1.96 * pooled_tpr_se)
            pooled_tpr_upper = min(1.0, pooled_tpr + 1.96 * pooled_tpr_se)
        else:
            pooled_tpr = np.nan
            pooled_tpr_se = np.nan
            pooled_tpr_lower = np.nan
            pooled_tpr_upper = np.nan

        row = dict(zip(keys, group_identity))
        row.update(
            {
                "task_count": count,
                "mean_discovery_count": mean_discovery,
                "mean_discovery_count_monte_carlo_standard_error": discovery_se,
                "mean_discovery_count_ci95_lower": max(
                    0.0, mean_discovery - 1.96 * discovery_se
                ),
                "mean_discovery_count_ci95_upper": min(
                    float(size), mean_discovery + 1.96 * discovery_se
                ),
                "no_discovery_rate": no_discovery_rate,
                "no_discovery_rate_monte_carlo_standard_error": no_discovery_se,
                "no_discovery_rate_ci95_lower": no_discovery_interval[0],
                "no_discovery_rate_ci95_upper": no_discovery_interval[1],
                "all_null_task_rate": all_null_rate,
                "all_null_task_rate_monte_carlo_standard_error": all_null_se,
                "all_null_task_rate_ci95_lower": all_null_interval[0],
                "all_null_task_rate_ci95_upper": all_null_interval[1],
                "true_alternative_count": int(alternative_total),
                "true_discovery_count": int(discovery_total),
                "pooled_true_effect_discovery_rate": pooled_tpr,
                "pooled_true_effect_discovery_rate_monte_carlo_standard_error": (
                    pooled_tpr_se
                ),
                "pooled_true_effect_discovery_rate_ci95_lower": pooled_tpr_lower,
                "pooled_true_effect_discovery_rate_ci95_upper": pooled_tpr_upper,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_discovery_by_true_effect_quantile(
    generated_tasks: pd.DataFrame,
    hypothesis_results: pd.DataFrame,
    *,
    quantile_count: int = 4,
) -> pd.DataFrame:
    """Report discovery rates in deterministic equal-count true-effect groups."""
    task_required = {
        "registered_task_idx", "hypothesis_id", "standardized_true_effect",
    }
    result_required = {
        "dataset_id", "method_variant", "scenario_id", "analysis_family",
        "registered_task_idx", "hypothesis_id", "discovered",
    }
    for name, frame, required in (
        ("generated_tasks", generated_tasks, task_required),
        ("hypothesis_results", hypothesis_results, result_required),
    ):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"{name} must be a non-empty DataFrame")
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} missing columns: " + ", ".join(missing))
    groups = int(quantile_count)
    if groups < 2:
        raise ValueError("quantile_count must be at least two")
    truth = generated_tasks.loc[
        generated_tasks["standardized_true_effect"].astype(float).ne(0.0),
        ["registered_task_idx", "hypothesis_id", "standardized_true_effect"],
    ].copy()
    if truth.empty:
        return pd.DataFrame(
            columns=[
                "dataset_id", "method_variant", "scenario_id", "analysis_family",
                "effect_quantile_group", "true_alternative_count",
                "true_discovery_count", "conditional_discovery_rate",
                "effect_lower", "effect_upper",
            ]
        )
    if truth.duplicated(["registered_task_idx", "hypothesis_id"]).any():
        raise ValueError("generated_tasks has duplicate task-hypothesis truth rows")
    truth = truth.sort_values(
        ["standardized_true_effect", "registered_task_idx", "hypothesis_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    truth["effect_quantile_group"] = (
        np.floor(np.arange(len(truth), dtype=float) * groups / len(truth)).astype(int) + 1
    )
    merged = hypothesis_results.merge(
        truth,
        on=["registered_task_idx", "hypothesis_id"],
        how="inner",
        validate="many_to_one",
    )
    keys = [
        "dataset_id", "method_variant", "scenario_id", "analysis_family",
        "effect_quantile_group",
    ]
    summary = merged.groupby(keys, sort=True, as_index=False).agg(
        true_alternative_count=("discovered", "size"),
        true_discovery_count=("discovered", "sum"),
        effect_lower=("standardized_true_effect", "min"),
        effect_upper=("standardized_true_effect", "max"),
    )
    summary["conditional_discovery_rate"] = (
        summary["true_discovery_count"] / summary["true_alternative_count"]
    )
    return summary


def evaluate_latent_fit_on_null_task_bases(
    task_bases: pd.DataFrame,
    null_daily_standard_deviations: pd.Series,
    fit: SpikeSlabFit,
    *,
    dataset_id: str,
    scenario_id: str,
    truth_seed_base: int,
    all_null: bool = False,
    alpha: float = 0.05,
) -> LatentDistributionSimulationArtifacts:
    """Assign one latent distribution to frozen bases and evaluate all BH methods."""
    tasks = apply_latent_fit_to_null_task_bases(
        task_bases,
        null_daily_standard_deviations,
        fit,
        scenario_id=str(scenario_id),
        truth_seed_base=int(truth_seed_base),
        all_null=bool(all_null),
    )
    family_size = task_bases["hypothesis_id"].nunique()
    evaluated = method_simulation.evaluate_bh_fdr_variants(
        tasks,
        dataset_id=str(dataset_id),
        day_count=494,
        family_size=int(family_size),
        alpha=float(alpha),
        include_cross_scenario_summary=False,
    )
    summary = summarize_latent_truth_performance(
        evaluated.task_summary, evaluated.hypothesis_results
    )
    conditional = summarize_discovery_by_true_effect_quantile(
        tasks, evaluated.hypothesis_results
    )
    return LatentDistributionSimulationArtifacts(
        generated_tasks=tasks,
        hypothesis_results=evaluated.hypothesis_results,
        task_summary=evaluated.task_summary,
        scenario_summary=summary,
        conditional_discovery_summary=conditional,
    )


def select_performance_envelope_candidates(
    screening_summary: pd.DataFrame,
    *,
    familywise_alpha: float = 0.001,
) -> pd.DataFrame:
    """Retain every candidate not excluded from a bound by simultaneous MC CIs."""
    required = {
        "parameter_id", "family", "method_variant", "fdr", "task_mean_tpr",
        "fdr_monte_carlo_standard_error",
        "task_mean_tpr_monte_carlo_standard_error",
        "task_count", "tpr_defined_task_count",
    }
    if not isinstance(screening_summary, pd.DataFrame) or screening_summary.empty:
        raise ValueError("screening_summary must be a non-empty DataFrame")
    missing = sorted(required.difference(screening_summary.columns))
    if missing:
        raise ValueError("screening_summary missing columns: " + ", ".join(missing))
    alpha = float(familywise_alpha)
    if not 0.0 < alpha < 0.05:
        raise ValueError("familywise_alpha must lie between zero and 0.05")
    interval_count = 2 * len(screening_summary)
    per_interval_alpha = alpha / interval_count
    log_term = float(np.log(2.0 / per_interval_alpha))
    rows = []
    for method, group in screening_summary.groupby("method_variant", sort=True):
        for metric, standard_error, sample_count in (
            ("fdr", "fdr_monte_carlo_standard_error", "task_count"),
            (
                "task_mean_tpr", "task_mean_tpr_monte_carlo_standard_error",
                "tpr_defined_task_count",
            ),
        ):
            valid = group.loc[
                group[metric].notna() & group[standard_error].notna()
                & group[sample_count].astype(float).gt(1.0)
            ].sort_values(
                [metric, "parameter_id"], kind="mergesort"
            ).copy()
            if valid.empty:
                continue
            count = valid[sample_count].astype(float)
            radius = (
                valid[standard_error].astype(float) * np.sqrt(2.0 * log_term)
                + 7.0 * log_term / (3.0 * (count - 1.0))
            )
            valid["screening_ci_lower"] = np.maximum(0.0, valid[metric] - radius)
            valid["screening_ci_upper"] = np.minimum(1.0, valid[metric] + radius)
            minimum_threshold = float(valid["screening_ci_upper"].min())
            maximum_threshold = float(valid["screening_ci_lower"].max())
            boundary_sets = {
                "minimum": valid.loc[valid["screening_ci_lower"].le(minimum_threshold)],
                "maximum": valid.loc[valid["screening_ci_upper"].ge(maximum_threshold)],
            }
            for bound, boundary in boundary_sets.items():
                boundary = boundary.sort_values(
                    [metric, "parameter_id"],
                    ascending=[bound == "minimum", True],
                    kind="mergesort",
                )
                threshold = minimum_threshold if bound == "minimum" else maximum_threshold
                for rank, (_, row) in enumerate(boundary.iterrows(), start=1):
                    rows.append(
                        {
                            "method_variant": method,
                            "metric": metric,
                            "bound": bound,
                            "boundary_rank": rank,
                            "parameter_id": row["parameter_id"],
                            "family": row["family"],
                            "screening_value": float(row[metric]),
                            "screening_standard_error": float(row[standard_error]),
                            "screening_ci_lower": float(row["screening_ci_lower"]),
                            "screening_ci_upper": float(row["screening_ci_upper"]),
                            "selection_threshold": threshold,
                            "finite_sample_radius": float(radius.loc[row.name]),
                            "per_interval_alpha": per_interval_alpha,
                            "familywise_alpha": alpha,
                        }
                    )
    return pd.DataFrame(rows)


def summarize_performance_envelope(full_summary: pd.DataFrame) -> pd.DataFrame:
    """Extract deterministic observed bounds from fully evaluated retained points."""
    required = {
        "parameter_id", "family", "method_variant", "fdr", "task_mean_tpr",
        "fdr_ci95_lower", "fdr_ci95_upper",
        "task_mean_tpr_ci95_lower", "task_mean_tpr_ci95_upper",
    }
    if not isinstance(full_summary, pd.DataFrame) or full_summary.empty:
        raise ValueError("full_summary must be a non-empty DataFrame")
    missing = sorted(required.difference(full_summary.columns))
    if missing:
        raise ValueError("full_summary missing columns: " + ", ".join(missing))
    rows = []
    for method, group in full_summary.groupby("method_variant", sort=True):
        for metric, lower, upper in (
            ("fdr", "fdr_ci95_lower", "fdr_ci95_upper"),
            (
                "task_mean_tpr", "task_mean_tpr_ci95_lower",
                "task_mean_tpr_ci95_upper",
            ),
        ):
            valid = group.loc[group[metric].notna()].sort_values(
                [metric, "parameter_id"], kind="mergesort"
            )
            if valid.empty:
                continue
            for bound, row in (("minimum", valid.iloc[0]), ("maximum", valid.iloc[-1])):
                rows.append(
                    {
                        "method_variant": method,
                        "metric": metric,
                        "bound": bound,
                        "parameter_id": row["parameter_id"],
                        "family": row["family"],
                        "value": float(row[metric]),
                        "ci95_lower": float(row[lower]),
                        "ci95_upper": float(row[upper]),
                    }
                )
    return pd.DataFrame(rows)


def summarize_extended_performance_envelope(full_summary: pd.DataFrame) -> pd.DataFrame:
    """Extract bounds for all blueprint-required task performance metrics."""
    metric_specs = (
        ("fdr", "fdr_ci95_lower", "fdr_ci95_upper"),
        (
            "task_mean_tpr", "task_mean_tpr_ci95_lower",
            "task_mean_tpr_ci95_upper",
        ),
        (
            "pooled_true_effect_discovery_rate",
            "pooled_true_effect_discovery_rate_ci95_lower",
            "pooled_true_effect_discovery_rate_ci95_upper",
        ),
        (
            "mean_discovery_count", "mean_discovery_count_ci95_lower",
            "mean_discovery_count_ci95_upper",
        ),
        (
            "no_discovery_rate", "no_discovery_rate_ci95_lower",
            "no_discovery_rate_ci95_upper",
        ),
        (
            "all_null_task_rate", "all_null_task_rate_ci95_lower",
            "all_null_task_rate_ci95_upper",
        ),
    )
    required = {"parameter_id", "family", "method_variant"}
    required.update(column for spec in metric_specs for column in spec)
    if not isinstance(full_summary, pd.DataFrame) or full_summary.empty:
        raise ValueError("full_summary must be a non-empty DataFrame")
    missing = sorted(required.difference(full_summary.columns))
    if missing:
        raise ValueError("full_summary missing columns: " + ", ".join(missing))
    rows = []
    for method, group in full_summary.groupby("method_variant", sort=True):
        for metric, lower, upper in metric_specs:
            valid = group.loc[group[metric].notna()].sort_values(
                [metric, "parameter_id"], kind="mergesort"
            )
            if valid.empty:
                continue
            for bound, row in (("minimum", valid.iloc[0]), ("maximum", valid.iloc[-1])):
                rows.append(
                    {
                        "method_variant": method,
                        "metric": metric,
                        "bound": bound,
                        "parameter_id": row["parameter_id"],
                        "family": row["family"],
                        "value": float(row[metric]),
                        "ci95_lower": float(row[lower]),
                        "ci95_upper": float(row[upper]),
                    }
                )
    return pd.DataFrame(rows)


def simulate_latent_effect_block_task(
    null_daily_effects: pd.DataFrame,
    null_daily_standard_deviations: pd.Series,
    reference_standard_errors: pd.Series,
    fit: SpikeSlabFit,
    *,
    replicate: int,
    block_length: int,
    noise_seed: int,
    truth_seed: int,
    scenario_id: str,
    all_null: bool = False,
) -> pd.DataFrame:
    """Generate one complete BH-ready task from a frozen latent distribution."""
    base = prepare_latent_null_task_base(
        null_daily_effects,
        reference_standard_errors,
        replicate=int(replicate),
        block_length=int(block_length),
        noise_seed=int(noise_seed),
    )
    return apply_latent_fit_to_null_task_bases(
        base,
        null_daily_standard_deviations,
        fit,
        scenario_id=str(scenario_id),
        truth_seed_base=int(truth_seed) - int(replicate),
        all_null=bool(all_null),
    )


def evaluate_latent_effect_distribution(
    null_daily_effects: pd.DataFrame,
    null_daily_standard_deviations: pd.Series,
    reference_standard_errors: pd.Series,
    fit: SpikeSlabFit,
    *,
    dataset_id: str,
    scenario_id: str,
    replicate_count: int,
    block_length: int,
    noise_seed_base: int,
    truth_seed_base: int,
    all_null: bool = False,
    alpha: float = 0.05,
) -> LatentDistributionSimulationArtifacts:
    """Generate and evaluate a complete truth-known latent-effect scenario."""
    count = int(replicate_count)
    if count <= 0:
        raise ValueError("replicate_count must be positive")
    method_simulation.validate_disjoint_seed_namespaces(
        {
            "noise": [int(noise_seed_base) + value for value in range(count)],
            "truth": [int(truth_seed_base) + value for value in range(count)],
        }
    )
    tasks = pd.concat(
        [
            simulate_latent_effect_block_task(
                null_daily_effects,
                null_daily_standard_deviations,
                reference_standard_errors,
                fit,
                replicate=replicate,
                block_length=int(block_length),
                noise_seed=int(noise_seed_base) + replicate,
                truth_seed=int(truth_seed_base) + replicate,
                scenario_id=str(scenario_id),
                all_null=bool(all_null),
            )
            for replicate in range(count)
        ],
        ignore_index=True,
    )
    evaluated = method_simulation.evaluate_bh_fdr_variants(
        tasks,
        dataset_id=str(dataset_id),
        day_count=len(null_daily_effects),
        family_size=len(null_daily_effects.columns),
        alpha=float(alpha),
        include_cross_scenario_summary=False,
    )
    summary = summarize_latent_truth_performance(
        evaluated.task_summary, evaluated.hypothesis_results
    )
    conditional = summarize_discovery_by_true_effect_quantile(
        tasks, evaluated.hypothesis_results
    )
    return LatentDistributionSimulationArtifacts(
        generated_tasks=tasks,
        hypothesis_results=evaluated.hypothesis_results,
        task_summary=evaluated.task_summary,
        scenario_summary=summary,
        conditional_discovery_summary=conditional,
    )
