"""Truth-known simulation infrastructure for residual-covariance methods.

The module is research-domain agnostic. It validates a frozen design, creates
the registered synthetic data, and computes Monte Carlo evidence tables. It
does not write files or assign empirical candidate labels.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from math import sqrt
from pathlib import Path
from time import perf_counter, process_time
from typing import Mapping, Sequence
import json

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution
from scipy.stats import norm as normal_distribution

from qlab import coinglass_substitution as substitution
from qlab import research_stats
from qlab.walkforward import WalkForwardFold, walk_forward_splits


HORIZON_HOURS = {"4h": 4, "8h": 8, "12h": 12, "1d": 24}
RESIDUAL_METHODS = tuple(f"R{index}" for index in range(7))
CORRECTION_IDENTITY_SCHEMA_VERSION = "correction_identity_v1"
LEGACY_LAYER_BC_ROUTE = "legacy.layer_bc"
REGISTERED_CORRECTION_IDENTITY_ROUTES = frozenset({LEGACY_LAYER_BC_ROUTE})
CANONICAL_CORRECTION_IDENTITY_FIELDS = (
    "identity_schema_version",
    "namespace",
    "method_id",
    "algorithm",
    "legacy_code",
)
CORRECTION_IDENTITIES: Mapping[str, Mapping[str, str]] = {
    "C0": {
        "namespace": "legacy.family_adjustment",
        "method_id": "legacy.family_adjustment.raw@v1",
        "algorithm": "raw p-values, no family adjustment",
    },
    "C1": {
        "namespace": "legacy.family_adjustment",
        "method_id": "legacy.family_adjustment.holm@v1",
        "algorithm": "Holm adjusted p-values",
    },
    "C2": {
        "namespace": "legacy.family_adjustment",
        "method_id": "legacy.family_adjustment.stepdown_maxT@v1",
        "algorithm": "synchronized step-down maxT",
    },
}
FAMILY_ADJUSTMENTS = tuple(CORRECTION_IDENTITIES)
JOINT_INFERENCE_ENGINES = (
    "E0", "E1", "E1F", "E1S", "E1H_AIC", "E1H_BIC", "E1J_BIC_1125", "E2"
)


def correction_identity_for_code(code: str) -> dict[str, str]:
    """Return the canonical identity fields for a legacy correction code."""
    legacy_code = str(code)
    try:
        identity = CORRECTION_IDENTITIES[legacy_code]
    except KeyError as exc:
        raise ValueError(f"unknown correction identity: {legacy_code}") from exc
    return {
        "identity_schema_version": CORRECTION_IDENTITY_SCHEMA_VERSION,
        "namespace": str(identity["namespace"]),
        "method_id": str(identity["method_id"]),
        "algorithm": str(identity["algorithm"]),
        "legacy_code": legacy_code,
    }


def _validated_correction_identity_route(route: str | None) -> str | None:
    if route is None:
        return None
    normalized = str(route)
    if normalized not in REGISTERED_CORRECTION_IDENTITY_ROUTES:
        raise ValueError(f"unknown correction identity route: {normalized}")
    return normalized


def normalize_correction_identity_frame(
    frame: pd.DataFrame, *, route: str | None = None
) -> pd.DataFrame:
    """Return a validated copy, with explicit compatibility for legacy Layer B/C."""
    route = _validated_correction_identity_route(route)
    canonical = set(CANONICAL_CORRECTION_IDENTITY_FIELDS)
    present = canonical.intersection(frame.columns)
    if present and present != canonical:
        missing = sorted(canonical.difference(frame.columns))
        raise ValueError(
            "correction identity fields are partially present: "
            + ", ".join(missing)
        )
    normalized = frame.copy()
    if not present:
        if route != LEGACY_LAYER_BC_ROUTE:
            raise ValueError(
                "legacy correction identity route is required for pre-schema artifact"
            )
        if "family_adjustment" not in normalized.columns:
            raise ValueError("correction identity fields missing: family_adjustment")
        identities = normalized["family_adjustment"].map(correction_identity_for_code)
        for field in CANONICAL_CORRECTION_IDENTITY_FIELDS:
            normalized[field] = identities.map(lambda identity: identity[field])
    required = {"family_adjustment", *CANONICAL_CORRECTION_IDENTITY_FIELDS}
    missing = sorted(required.difference(normalized.columns))
    if missing:
        raise ValueError("correction identity fields missing: " + ", ".join(missing))
    for row in normalized.itertuples(index=False):
        expected = correction_identity_for_code(str(row.family_adjustment))
        actual = {field: str(getattr(row, field)) for field in expected}
        if actual != expected:
            raise ValueError(
                "correction identity mismatch for "
                f"{row.family_adjustment}: {actual} != {expected}"
            )
    return normalized


def validate_correction_identity_frame(
    frame: pd.DataFrame, *, route: str | None = None
) -> None:
    """Fail closed while allowing only the registered legacy compatibility route."""
    normalize_correction_identity_frame(frame, route=route)


@contextmanager
def _performance_stage(
    rows: list[dict[str, object]] | None,
    stage: str,
    **metadata: object,
):
    if rows is None:
        yield
        return
    cpu_started = process_time()
    wall_started = perf_counter()
    yield
    row = {
        "stage": str(stage),
        "cpu_seconds": process_time() - cpu_started,
        "wall_seconds": perf_counter() - wall_started,
        **metadata,
    }
    row.setdefault("estimator_fit_count", 0)
    rows.append(row)


@dataclass(frozen=True)
class LayerADataset:
    daily_values: pd.DataFrame
    true_effects: pd.Series


@dataclass(frozen=True)
class RealEffectCalibrationArtifacts:
    hypothesis_effects: pd.DataFrame
    distribution_quantiles: pd.DataFrame
    simulation_grid_alignment: pd.DataFrame


@dataclass(frozen=True)
class MonteCarloSummary:
    scenario_summary: pd.DataFrame
    hypothesis_summary: pd.DataFrame


@dataclass(frozen=True)
class LayerAInferenceArtifacts:
    results: pd.DataFrame
    bootstrap_max_statistics: pd.DataFrame


@dataclass(frozen=True)
class DependenceProfileArtifacts:
    temporal_by_hypothesis: pd.DataFrame
    temporal_summary: pd.DataFrame
    cross_pairs: pd.DataFrame
    cross_summary: pd.DataFrame


@dataclass(frozen=True)
class E1FailureDiagnosticTaskArtifacts:
    hypothesis_diagnostics: pd.DataFrame
    replicate_summary: pd.DataFrame
    temporal_profile: pd.DataFrame
    cross_profile: pd.DataFrame


@dataclass(frozen=True)
class E1FailureDiagnosticSummaryArtifacts:
    mechanism_summary: pd.DataFrame
    temporal_profile: pd.DataFrame
    cross_profile: pd.DataFrame
    scenario_distances: pd.DataFrame
    comparison_decision: Mapping[str, object]


@dataclass(frozen=True)
class JointInferenceRevisionTaskArtifacts:
    results: pd.DataFrame
    bootstrap_max_statistics: pd.DataFrame


@dataclass(frozen=True)
class JointInferenceFaultDecompositionArtifacts:
    oracle_hypothesis: pd.DataFrame
    engine_hypothesis: pd.DataFrame
    specification_summary: pd.DataFrame
    standard_error_diagnostics: pd.DataFrame
    decision: Mapping[str, object]


@dataclass(frozen=True)
class BhFdrEvaluationArtifacts:
    hypothesis_results: pd.DataFrame
    task_summary: pd.DataFrame
    scenario_summary: pd.DataFrame
    true_hypothesis_summary: pd.DataFrame
    raw_p_value_calibration_summary: pd.DataFrame
    cross_scenario_summary: pd.DataFrame


@dataclass(frozen=True)
class EmpiricalBlockFamilyArtifacts:
    null_daily_effects: pd.DataFrame
    empirical_standardized_effects: pd.Series
    null_daily_standard_deviations: pd.Series
    hypothesis_manifest: pd.DataFrame


@dataclass(frozen=True)
class EmpiricalMeanSeCalibrationArtifacts:
    hypothesis_standard_errors: pd.DataFrame
    calibration_summary: pd.DataFrame


@dataclass(frozen=True)
class EmpiricalBhDiagnosticArtifacts:
    paired_method_differences: pd.DataFrame
    block_length_sensitivity: pd.DataFrame


@dataclass(frozen=True)
class RealisticEffectPowerArtifacts:
    counterfactual_inputs: pd.DataFrame
    bh_evaluation: BhFdrEvaluationArtifacts
    scenario_summary: pd.DataFrame
    paired_effect_contrasts: pd.DataFrame
    paired_method_contrasts: pd.DataFrame


def prepare_empirical_block_family(
    daily_effects: pd.DataFrame,
    daily_centered_sums: pd.DataFrame,
    daily_counts: pd.DataFrame,
    representative_hypothesis_ids: Sequence[str],
    *,
    expected_day_count: int = 494,
    centering_tolerance: float = 1e-12,
) -> EmpiricalBlockFamilyArtifacts:
    """Prepare a synchronized empirical null family and observed effect pool."""
    frames = {
        "daily_effects": daily_effects,
        "daily_centered_sums": daily_centered_sums,
        "daily_counts": daily_counts,
    }
    for name, frame in frames.items():
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"{name} must be a non-empty DataFrame")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError(f"{name} must use a DatetimeIndex")
        if frame.index.has_duplicates or frame.columns.has_duplicates:
            raise ValueError(f"{name} must have unique dates and hypotheses")
    first_index = daily_effects.index
    first_columns = daily_effects.columns.astype(str)
    for name, frame in frames.items():
        if not frame.index.equals(first_index):
            raise ValueError("empirical family inputs must share one date index")
        if tuple(frame.columns.astype(str)) != tuple(first_columns):
            raise ValueError("empirical family inputs must share ordered hypotheses")
    day_count = int(expected_day_count)
    if len(first_index) != day_count:
        raise ValueError(f"empirical family requires exactly {day_count} days")
    utc_index = pd.DatetimeIndex(pd.to_datetime(first_index, utc=True))
    expected_index = pd.date_range(utc_index[0], periods=day_count, freq="D", tz="UTC")
    if not utc_index.equals(expected_index):
        raise ValueError("empirical family dates must be consecutive UTC days")

    representatives = tuple(str(value) for value in representative_hypothesis_ids)
    if len(representatives) < 2 or len(set(representatives)) != len(representatives):
        raise ValueError("representative hypotheses must be unique and contain at least two ids")
    missing = sorted(set(representatives).difference(first_columns))
    if missing:
        raise ValueError("empirical family is missing representatives: " + ", ".join(missing))
    selected_effects = daily_effects.loc[:, list(representatives)].astype(float)
    selected_sums = daily_centered_sums.loc[:, list(representatives)].astype(float)
    selected_counts = daily_counts.loc[:, list(representatives)].astype(float)
    for name, frame in (
        ("daily_effects", selected_effects),
        ("daily_centered_sums", selected_sums),
        ("daily_counts", selected_counts),
    ):
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} must be finite and complete")
    if (selected_counts.to_numpy(dtype=float) <= 0.0).any():
        raise ValueError("daily_counts must be strictly positive")
    null_values = selected_sums / selected_counts
    maximum_mean = float(np.max(np.abs(null_values.mean(axis=0).to_numpy(dtype=float))))
    tolerance = float(centering_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("centering_tolerance must be finite and positive")
    if maximum_mean > tolerance:
        raise ValueError("empirical centered family is not zero mean")
    null_values = null_values.subtract(null_values.mean(axis=0), axis="columns")
    standard_deviations = selected_effects.std(axis=0, ddof=1)
    if (
        not np.isfinite(standard_deviations.to_numpy(dtype=float)).all()
        or (standard_deviations <= 0.0).any()
    ):
        raise ValueError("empirical family contains a constant hypothesis")
    standardized = selected_effects.mean(axis=0) / standard_deviations
    if not np.isfinite(standardized.to_numpy(dtype=float)).all() or (standardized <= 0.0).any():
        raise ValueError("empirical standardized effect pool must be finite and positive")

    synthetic_ids = tuple(f"H{index:02d}" for index in range(1, len(representatives) + 1))
    rename = dict(zip(representatives, synthetic_ids, strict=True))
    manifest = pd.DataFrame(
        {
            "hypothesis_id": synthetic_ids,
            "source_hypothesis_id": representatives,
            "observed_standardized_effect": standardized.to_numpy(dtype=float),
            "null_daily_standard_deviation": null_values.std(axis=0, ddof=1).to_numpy(dtype=float),
        }
    )
    prepared = null_values.rename(columns=rename)
    prepared.index = utc_index
    prepared.index.name = "utc_day"
    effect_pool = pd.Series(
        standardized.to_numpy(dtype=float), index=synthetic_ids,
        name="observed_standardized_effect",
    )
    null_sd = pd.Series(
        prepared.std(axis=0, ddof=1).to_numpy(dtype=float), index=synthetic_ids,
        name="null_daily_standard_deviation",
    )
    return EmpiricalBlockFamilyArtifacts(prepared, effect_pool, null_sd, manifest)


def circular_block_indices_from_starts(
    day_count: int,
    block_length: int,
    block_starts: Sequence[int],
) -> np.ndarray:
    """Expand explicit circular-block starts into one fixed-length draw."""
    count = int(day_count)
    length = int(block_length)
    starts = np.asarray(tuple(block_starts), dtype=int)
    if count <= 1 or length <= 0 or length > count:
        raise ValueError("invalid circular-block dimensions")
    required = int(np.ceil(count / length))
    if starts.ndim != 1 or len(starts) != required:
        raise ValueError("circular-block draw has the wrong number of starts")
    if ((starts < 0) | (starts >= count)).any():
        raise ValueError("circular-block starts lie outside the sample")
    indices = (starts[:, None] + np.arange(length, dtype=int)[None, :]) % count
    return indices.reshape(-1)[:count]


def synchronized_circular_block_indices(
    day_count: int,
    block_length: int,
    *,
    draw_count: int,
    seed: int,
) -> np.ndarray:
    """Generate reproducible synchronized circular-block index draws."""
    count = int(day_count)
    length = int(block_length)
    draws = int(draw_count)
    if count <= 1 or length <= 0 or length > count or draws <= 0:
        raise ValueError("invalid synchronized circular-block request")
    blocks = int(np.ceil(count / length))
    generator = np.random.default_rng(int(seed))
    starts = generator.integers(0, count, size=(draws, blocks), endpoint=False)
    offsets = np.arange(length, dtype=int)
    return ((starts[..., None] + offsets) % count).reshape(draws, -1)[:, :count]


def validate_disjoint_seed_namespaces(
    seed_namespaces: Mapping[str, Sequence[int]],
) -> None:
    """Fail when independently registered Monte Carlo stages reuse a seed."""
    if len(seed_namespaces) < 2:
        raise ValueError("at least two seed namespaces are required")
    seen: dict[int, str] = {}
    for name, values in seed_namespaces.items():
        label = str(name)
        seeds = tuple(int(value) for value in values)
        if not label or not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("seed namespaces must be named, non-empty, and internally unique")
        for seed in seeds:
            prior = seen.get(seed)
            if prior is not None:
                raise ValueError(f"seed {seed} is shared by {prior} and {label}")
            seen[seed] = label


def calibrate_empirical_block_mean_standard_errors(
    null_daily_effects: pd.DataFrame,
    *,
    block_length: int,
    n_draws: int,
    seed: int,
    batch_size: int = 100,
) -> EmpiricalMeanSeCalibrationArtifacts:
    """Estimate marginal mean standard errors from independent empirical-null draws."""
    if not isinstance(null_daily_effects, pd.DataFrame) or null_daily_effects.empty:
        raise ValueError("null_daily_effects must be a non-empty DataFrame")
    if null_daily_effects.columns.has_duplicates:
        raise ValueError("null_daily_effects hypotheses must be unique")
    values = null_daily_effects.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("null_daily_effects must be finite")
    if not np.allclose(values.mean(axis=0), 0.0, atol=1e-12, rtol=0.0):
        raise ValueError("null_daily_effects must be zero mean")
    draws = int(n_draws)
    batch = int(batch_size)
    if draws < 2 or batch <= 0:
        raise ValueError("n_draws must exceed one and batch_size must be positive")
    mean_sum = np.zeros(values.shape[1], dtype=float)
    mean_square_sum = np.zeros(values.shape[1], dtype=float)
    completed = 0
    while completed < draws:
        current = min(batch, draws - completed)
        indices = synchronized_circular_block_indices(
            len(values), int(block_length), draw_count=current,
            seed=int(seed) + completed,
        )
        means = values[indices].mean(axis=1)
        mean_sum += means.sum(axis=0)
        mean_square_sum += np.square(means).sum(axis=0)
        completed += current
    variance = (mean_square_sum - np.square(mean_sum) / draws) / (draws - 1)
    if not np.isfinite(variance).all() or (variance <= 0.0).any():
        raise RuntimeError("empirical calibration produced invalid mean variances")
    standard_errors = np.sqrt(variance)
    relative_mc_error = float(1.0 / np.sqrt(2.0 * (draws - 1)))
    hypothesis = pd.DataFrame(
        {
            "hypothesis_id": null_daily_effects.columns.astype(str),
            "reference_standard_error": standard_errors,
            "calibration_draw_count": draws,
            "approximate_relative_monte_carlo_error": relative_mc_error,
        }
    )
    summary = pd.DataFrame(
        [{
            "block_length_days": int(block_length),
            "calibration_draw_count": draws,
            "seed": int(seed),
            "batch_size": batch,
            "hypothesis_count": values.shape[1],
            "approximate_relative_monte_carlo_error": relative_mc_error,
            "minimum_reference_standard_error": float(np.min(standard_errors)),
            "maximum_reference_standard_error": float(np.max(standard_errors)),
        }]
    )
    return EmpiricalMeanSeCalibrationArtifacts(hypothesis, summary)


def simulate_empirical_block_task_family(
    null_daily_effects: pd.DataFrame,
    empirical_standardized_effects: pd.Series,
    null_daily_standard_deviations: pd.Series,
    reference_standard_errors: pd.Series,
    scenario_specs: Sequence[Mapping[str, object]],
    *,
    replicate: int,
    block_length: int,
    noise_seed: int,
    truth_seed: int,
) -> pd.DataFrame:
    """Simulate one paired empirical-block task and all registered scenarios."""
    hypotheses = tuple(null_daily_effects.columns.astype(str))
    expected = tuple(f"H{index:02d}" for index in range(1, len(hypotheses) + 1))
    if hypotheses != expected:
        raise ValueError("empirical task requires ordered H01-Hnn hypotheses")
    for name, values in (
        ("empirical_standardized_effects", empirical_standardized_effects),
        ("null_daily_standard_deviations", null_daily_standard_deviations),
        ("reference_standard_errors", reference_standard_errors),
    ):
        if not isinstance(values, pd.Series) or tuple(values.index.astype(str)) != hypotheses:
            raise ValueError(f"{name} must match the empirical hypothesis family")
        if not np.isfinite(values.to_numpy(dtype=float)).all() or (values <= 0.0).any():
            raise ValueError(f"{name} must be finite and positive")
    specs = [dict(value) for value in scenario_specs]
    if not specs:
        raise ValueError("scenario_specs must not be empty")
    required = {"scenario_id", "block_length", "active_count", "shrinkage_multiplier"}
    scenario_ids: set[str] = set()
    for spec in specs:
        missing = sorted(required.difference(spec))
        if missing:
            raise ValueError("empirical scenario spec missing: " + ", ".join(missing))
        scenario_id = str(spec["scenario_id"])
        if not scenario_id or scenario_id in scenario_ids:
            raise ValueError("empirical scenario ids must be unique and non-empty")
        if int(spec["block_length"]) != int(block_length):
            raise ValueError("scenario block length differs from task block length")
        active_count = int(spec["active_count"])
        shrinkage = float(spec["shrinkage_multiplier"])
        if active_count == 0:
            if shrinkage != 0.0:
                raise ValueError("all-null scenarios require zero shrinkage")
        elif not 0 < active_count < len(hypotheses) or not 0.0 < shrinkage <= 1.0:
            raise ValueError("mixed scenario truth settings are invalid")
        scenario_ids.add(scenario_id)

    indices = synchronized_circular_block_indices(
        len(null_daily_effects), int(block_length), draw_count=1, seed=int(noise_seed)
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
        centered,
        counts,
        observed_null,
        order_criterion="BIC",
        expected_hypothesis_count=len(hypotheses),
        alternative="greater",
    ).summary.set_index("hypothesis_id")
    uncalibrated = marginal.loc[list(hypotheses), "uncalibrated_standard_error"]

    truth_cache: dict[int, np.ndarray] = {}
    output: list[pd.DataFrame] = []
    pool = empirical_standardized_effects.to_numpy(dtype=float)
    null_sd = null_daily_standard_deviations.to_numpy(dtype=float)
    reference = reference_standard_errors.to_numpy(dtype=float)
    for spec in specs:
        active_count = int(spec["active_count"])
        shrinkage = float(spec["shrinkage_multiplier"])
        if active_count not in truth_cache:
            assigned = np.zeros(len(hypotheses), dtype=float)
            if active_count:
                generator = np.random.default_rng(int(truth_seed) + active_count * 1_000_003)
                active_positions = generator.choice(
                    len(hypotheses), size=active_count, replace=False
                )
                pool_positions = generator.permutation(len(hypotheses))[:active_count]
                assigned[active_positions] = pool[pool_positions] * null_sd[active_positions]
            truth_cache[active_count] = assigned
        true_effect = shrinkage * truth_cache[active_count]
        scenario_id = str(spec["scenario_id"])
        frame = pd.DataFrame(
            {
                "registered_task_idx": int(replicate),
                "scenario_id": scenario_id,
                "analysis_specification": f"{scenario_id}__right_tail_primary__empirical_block",
                "replicate": int(replicate),
                "hypothesis_id": hypotheses,
                "observed_effect": observed_null.to_numpy(dtype=float) + true_effect,
                "uncalibrated_standard_error": uncalibrated.to_numpy(dtype=float),
                "reference_standard_error": reference,
                "alternative": "greater",
                "true_effect": true_effect,
                "block_length_days": int(block_length),
                "active_count": active_count,
                "shrinkage_multiplier": shrinkage,
                "noise_seed": int(noise_seed),
                "truth_seed": int(truth_seed),
            }
        )
        output.append(frame)
    return pd.concat(output, ignore_index=True)


def retarget_additive_effect_scenarios(
    results: pd.DataFrame,
    scenario_specs: Sequence[Mapping[str, object]],
    *,
    family_size: int = 47,
    expected_tasks_per_base: int | None = None,
) -> pd.DataFrame:
    """Retarget additive Layer-A effects while preserving frozen noise fits.

    The Layer-A DGP adds a constant true effect to each active daily series.
    Subtracting the old true effect and adding a new one therefore changes only
    the sample mean.  Centered observations and every standard-error fit remain
    identical.  This entry validates that contract before returning paired
    counterfactual task families.
    """
    required = {
        "registered_task_idx",
        "scenario_id",
        "analysis_specification",
        "replicate",
        "hypothesis_id",
        "observed_effect",
        "uncalibrated_standard_error",
        "alternative",
        "true_effect",
    }
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError("effect-retargeting input missing columns: " + ", ".join(missing))
    if int(family_size) <= 1:
        raise ValueError("family_size must exceed one")
    if not scenario_specs:
        raise ValueError("scenario_specs must not be empty")

    frame = results.loc[:, sorted(required)].copy()
    if frame.isna().any().any():
        raise ValueError("effect-retargeting input must be complete")
    numeric = frame[
        ["observed_effect", "uncalibrated_standard_error", "true_effect"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("effect-retargeting numeric inputs must be finite")
    if (frame["uncalibrated_standard_error"].astype(float) <= 0.0).any():
        raise ValueError("effect-retargeting standard errors must be positive")
    if set(frame["alternative"].astype(str)) != {"greater"}:
        raise ValueError("effect-retargeting requires one-sided greater alternatives")

    expected_ids = tuple(f"H{index:02d}" for index in range(1, int(family_size) + 1))
    task_keys = ["scenario_id", "registered_task_idx", "replicate"]
    if frame.duplicated(task_keys + ["hypothesis_id"]).any():
        raise ValueError("effect-retargeting input contains duplicate task hypotheses")
    identity = frame.groupby(task_keys, sort=False).agg(
        hypothesis_count=("hypothesis_id", "size"),
        hypothesis_ids=("hypothesis_id", lambda values: tuple(values)),
        analysis_count=("analysis_specification", "nunique"),
    )
    if (
        identity["hypothesis_count"].ne(int(family_size)).any()
        or (~identity["hypothesis_ids"].map(lambda value: value == expected_ids)).any()
        or identity["analysis_count"].ne(1).any()
    ):
        raise ValueError("effect-retargeting requires complete, ordered fixed-size families")

    normalized_specs: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for raw in scenario_specs:
        keys = {"scenario_id", "base_scenario_id", "target_effect", "active_count"}
        missing_spec = sorted(keys.difference(raw))
        if missing_spec:
            raise ValueError("effect-retargeting scenario spec missing: " + ", ".join(missing_spec))
        scenario_id = str(raw["scenario_id"])
        base_id = str(raw["base_scenario_id"])
        target = float(raw["target_effect"])
        active_count = int(raw["active_count"])
        if not scenario_id or scenario_id in seen_ids:
            raise ValueError("effect-retargeting scenario ids must be unique and non-empty")
        if not np.isfinite(target) or target < 0.0:
            raise ValueError("target effects must be finite and non-negative")
        if target == 0.0:
            if active_count != 0:
                raise ValueError("all-null targets require active_count zero")
        elif not 0 < active_count < int(family_size):
            raise ValueError("active_count must lie inside the hypothesis family")
        seen_ids.add(scenario_id)
        normalized_specs.append(
            {
                "scenario_id": scenario_id,
                "base_scenario_id": base_id,
                "target_effect": target,
                "active_count": active_count,
            }
        )

    available = set(frame["scenario_id"].astype(str))
    requested = {str(spec["base_scenario_id"]) for spec in normalized_specs}
    if available != requested:
        raise ValueError("effect-retargeting input base scenarios are incomplete or excessive")

    output: list[pd.DataFrame] = []
    next_task_idx = 0
    for spec in normalized_specs:
        base_id = str(spec["base_scenario_id"])
        base = frame.loc[frame["scenario_id"].astype(str).eq(base_id)].copy()
        base = base.sort_values(
            ["registered_task_idx", "replicate", "hypothesis_id"], kind="stable"
        ).reset_index(drop=True)
        task_count = base[["registered_task_idx", "replicate"]].drop_duplicates().shape[0]
        if expected_tasks_per_base is not None and task_count != int(expected_tasks_per_base):
            raise ValueError("effect-retargeting base task count differs from contract")
        old_positive_values = sorted(
            value for value in base["true_effect"].astype(float).unique() if value > 0.0
        )
        if (base["true_effect"].astype(float) < 0.0).any() or len(old_positive_values) > 1:
            raise ValueError("effect-retargeting requires all-null or one positive old effect")
        old_active = base["true_effect"].astype(float).gt(0.0)
        if old_positive_values:
            active_per_task = old_active.groupby(
                [base["registered_task_idx"], base["replicate"]]
            ).sum()
            if not active_per_task.eq(int(spec["active_count"])).all():
                raise ValueError("effect-retargeting active pattern differs from scenario spec")
            active_ids = base.loc[old_active].groupby(
                ["registered_task_idx", "replicate"], sort=False
            )["hypothesis_id"].agg(tuple)
        else:
            active_ids = pd.Series(
                [expected_ids[: int(spec["active_count"])]], dtype=object
            )
        expected_active_ids = expected_ids[: int(spec["active_count"])]
        if not active_ids.map(lambda value: value == expected_active_ids).all():
            raise ValueError("effect-retargeting active hypotheses are not the registered prefix")

        transformed = base.copy()
        transformed["base_scenario_id"] = base_id
        transformed["base_registered_task_idx"] = transformed["registered_task_idx"].astype(int)
        target = float(spec["target_effect"])
        positions = transformed["hypothesis_id"].str[1:].astype(int)
        target_active = positions.le(int(spec["active_count"]))
        transformed.loc[target_active, "observed_effect"] = (
            transformed.loc[target_active, "observed_effect"].astype(float)
            - transformed.loc[target_active, "true_effect"].astype(float)
            + target
        )
        transformed["true_effect"] = np.where(target_active, target, 0.0)
        transformed["scenario_id"] = str(spec["scenario_id"])
        transformed["analysis_specification"] = (
            str(spec["scenario_id"]) + "__right_tail_primary__realistic_effect"
        )
        task_map = {
            int(value): next_task_idx + index
            for index, value in enumerate(
                transformed["base_registered_task_idx"].drop_duplicates().tolist()
            )
        }
        transformed["registered_task_idx"] = transformed[
            "base_registered_task_idx"
        ].map(task_map).astype(int)
        next_task_idx += len(task_map)
        output.append(transformed)

    combined = pd.concat(output, ignore_index=True)
    if combined.duplicated(
        ["registered_task_idx", "scenario_id", "replicate", "hypothesis_id"]
    ).any():
        raise RuntimeError("effect-retargeting produced duplicate task hypotheses")
    return combined


@dataclass(frozen=True)
class DiscoveryPowerAttributionArtifacts:
    hypothesis_results: pd.DataFrame
    task_summary: pd.DataFrame
    scenario_summary: pd.DataFrame
    attribution_summary: pd.DataFrame


@dataclass(frozen=True)
class FederatedTaskValidationArtifacts:
    task_inventory: pd.DataFrame
    runtime_inventory: pd.DataFrame
    receipt: Mapping[str, object]


@dataclass(frozen=True)
class LayerBCSummary:
    replicate_results: pd.DataFrame
    scenario_summary: pd.DataFrame


@dataclass(frozen=True)
class LayerBDataset:
    scenario_id: str
    replicate: int
    frames: Mapping[str, pd.DataFrame]
    folds: Mapping[str, tuple[WalkForwardFold, ...]]
    candidate_mapping: pd.DataFrame
    alias_mapping: pd.DataFrame


@dataclass(frozen=True)
class LayerCArtifacts:
    decision_moments: pd.DataFrame
    observations: pd.DataFrame
    comparison_grid: pd.DataFrame
    bootstrap_starts: pd.DataFrame


@dataclass(frozen=True)
class LayerCInferenceArtifacts:
    comparison_grid: pd.DataFrame
    bootstrap_starts: pd.DataFrame


@dataclass(frozen=True)
class LayerBCSimulationArtifacts:
    dataset: LayerBDataset
    layer_c: LayerCArtifacts


@dataclass(frozen=True)
class TemporalFalsificationDataset:
    observations: pd.DataFrame
    t05_observations: pd.DataFrame


@dataclass(frozen=True)
class TemporalFalsificationArtifacts:
    dataset: TemporalFalsificationDataset
    daily_effects: pd.DataFrame
    randomization: substitution.DoubleResidualTimeRandomizationArtifacts
    primary_summary: pd.DataFrame
    t05_summary: pd.DataFrame


def load_frozen_design(path: str | Path) -> dict[str, object]:
    """Load and fail closed on the approved KSV4 simulation manifest."""
    design = json.loads(Path(path).read_text(encoding="utf-8"))
    return _validate_frozen_design(design)


def _validate_frozen_design(design: Mapping[str, object]) -> dict[str, object]:
    design = dict(design)
    required = {"schema_version", "lifecycle", "approved", "layer_a", "layer_b"}
    missing = sorted(required.difference(design))
    if missing:
        raise ValueError("simulation design missing keys: " + ", ".join(missing))
    if design["schema_version"] != "ksv4_method_simulation_design_v6":
        raise ValueError("unsupported simulation design schema")
    if design["lifecycle"] != "approved_frozen_design" or design["approved"] is not True:
        raise ValueError("simulation design is not approved and frozen")
    if design.get("must_not_run_before_approval") is not False:
        raise ValueError("simulation design still forbids execution")
    revision = design.get("engineering_revision", {})
    if revision.get("formal_run_blocked_until_revision_implemented") is not False:
        raise ValueError("simulation engineering revision still forbids formal execution")
    layer_a = design["layer_a"]
    if layer_a["day_count"] != 494 or layer_a["hypothesis_count"] != 47:
        raise ValueError("Layer A application scale changed")
    if sum(layer_a["hypothesis_group_sizes"]) != layer_a["hypothesis_count"]:
        raise ValueError("Layer A hypothesis groups do not cover the family")
    layer_b = design["layer_b"]
    if layer_b["unique_signal_count_main"] != 4:
        raise ValueError("Layer B unique-signal count changed")
    workload = layer_b.get("declared_workload", {})
    if workload.get("complete_datasets") != 114:
        raise ValueError("Layer B complete-dataset workload changed")
    if workload.get("cpu_hour_limit_mode") != "reporting_only_not_feasibility_gate":
        raise ValueError("Layer B CPU-hour reporting contract changed")
    revision = design.get("engineering_revision", {})
    if revision.get("scientific_workload_changed") is not False:
        raise ValueError("simulation engineering revision changed scientific workload")
    return design


def grouped_correlation(group_sizes: Sequence[int], within: float, between: float) -> np.ndarray:
    """Build and validate the frozen block correlation structure."""
    sizes = tuple(int(value) for value in group_sizes)
    if not sizes or min(sizes) <= 0:
        raise ValueError("group sizes must be positive")
    if not (-1.0 < float(between) < 1.0 and -1.0 < float(within) < 1.0):
        raise ValueError("correlations must lie strictly between -1 and 1")
    groups = np.concatenate([np.full(size, index) for index, size in enumerate(sizes)])
    matrix = np.where(groups[:, None] == groups[None, :], float(within), float(between))
    np.fill_diagonal(matrix, 1.0)
    if np.linalg.eigvalsh(matrix).min() <= 1e-10:
        raise ValueError("registered hypothesis correlation is not positive definite")
    return matrix


def _parse_dependence(raw: str) -> tuple[float, float]:
    if raw == "independent":
        return 0.0, 0.0
    parts = raw.split("_")
    if len(parts) != 4 or parts[0] != "within" or parts[2] != "between":
        raise ValueError(f"unsupported hypothesis dependence: {raw}")
    return float(parts[1]), float(parts[3])


def layer_a_dependence_parameters(
    scenario: Mapping[str, object],
) -> dict[str, float | str]:
    """Expose the registered Layer-A dependence parameters for diagnostics."""
    if "hypothesis_dependence" not in scenario or "temporal_dependence" not in scenario:
        raise ValueError("Layer-A scenario is missing its dependence contract")
    within, between = _parse_dependence(str(scenario["hypothesis_dependence"]))
    return {
        "temporal_dependence": str(scenario["temporal_dependence"]),
        "within_correlation": within,
        "between_correlation": between,
    }


def _effect_vector(raw: str, count: int) -> np.ndarray:
    result = np.zeros(count, dtype=float)
    if raw == "all_null":
        return result
    parts = raw.split("_")
    if len(parts) == 4 and parts[0] == "first" and parts[2] == "positive":
        active_count = int(parts[1])
        effect = float(parts[3])
        if not 0 < active_count < int(count):
            raise ValueError("positive-effect count must lie inside the family")
        if not np.isfinite(effect) or effect <= 0.0:
            raise ValueError("positive effect must be finite and positive")
        result[:active_count] = effect
        return result
    if raw == "first_6_positive_0.20_next_6_negative_0.20":
        result[:6], result[6:12] = 0.20, -0.20
        return result
    raise ValueError(f"unsupported effect assignment: {raw}")


def generate_layer_a_dataset(
    scenario: Mapping[str, object], *, day_count: int, group_sizes: Sequence[int], seed: int
) -> LayerADataset:
    """Generate one registered Layer A daily hypothesis matrix."""
    count = sum(int(value) for value in group_sizes)
    within, between = _parse_dependence(str(scenario["hypothesis_dependence"]))
    chol = np.linalg.cholesky(grouped_correlation(group_sizes, within, between))
    rng = np.random.Generator(np.random.PCG64DXSM(int(seed)))
    temporal = str(scenario["temporal_dependence"])
    burn = 64 if temporal.startswith("ar1_") else 13 if temporal == "ma_14" else 0
    innovations = rng.standard_normal((day_count + burn + 13, count)) @ chol.T
    if temporal == "iid":
        noise = innovations[:day_count]
    elif temporal.startswith("ar1_"):
        phi = float(temporal.rsplit("_", 1)[1])
        state = innovations[0].copy()
        rows = []
        for innovation in innovations[1: day_count + burn + 1]:
            state = phi * state + sqrt(1.0 - phi * phi) * innovation
            rows.append(state.copy())
        noise = np.asarray(rows[-day_count:])
    elif temporal == "ma_14":
        noise = np.stack(
            [innovations[index:index + 14].sum(axis=0) / sqrt(14.0) for index in range(day_count)]
        )
    else:
        raise ValueError(f"unsupported temporal dependence: {temporal}")
    effects = _effect_vector(str(scenario["effect"]), count)
    columns = [f"H{index + 1:02d}" for index in range(count)]
    index = pd.date_range("2025-01-01", periods=day_count, freq="D", tz="UTC")
    return LayerADataset(
        daily_values=pd.DataFrame(noise + effects, index=index, columns=columns),
        true_effects=pd.Series(effects, index=columns, name="true_effect"),
    )


def exact_gaussian_mean_variance(
    temporal_dependence: str,
    *,
    day_count: int,
) -> dict[str, float]:
    """Return asymptotic LRV and exact finite-sample mean variance.

    The registered Layer-A processes have unit marginal variance. The exact
    variance keeps the finite-sample ``(D-lag)`` pair counts and must not be
    replaced by ``LRV / D``.
    """
    days = int(day_count)
    if days <= 1:
        raise ValueError("day_count must exceed one")
    dependence = str(temporal_dependence)
    lags = np.arange(days, dtype=float)
    if dependence == "iid":
        autocovariance = np.zeros(days, dtype=float)
        autocovariance[0] = 1.0
        long_run_variance = 1.0
    elif dependence.startswith("ar1_"):
        phi = float(dependence.rsplit("_", 1)[1])
        if not -1.0 < phi < 1.0:
            raise ValueError("AR(1) coefficient must lie strictly between -1 and 1")
        autocovariance = np.power(phi, lags)
        long_run_variance = (1.0 + phi) / (1.0 - phi)
    elif dependence.startswith("ma_"):
        window = int(dependence.rsplit("_", 1)[1])
        if window <= 0:
            raise ValueError("MA window must be positive")
        autocovariance = np.zeros(days, dtype=float)
        active = min(days, window)
        autocovariance[:active] = (
            window - np.arange(active, dtype=float)
        ) / float(window)
        long_run_variance = float(window)
    else:
        raise ValueError(f"unsupported temporal dependence: {dependence}")
    pair_counts = days - np.arange(1, days, dtype=float)
    exact_variance = (
        days * autocovariance[0]
        + 2.0 * np.sum(pair_counts * autocovariance[1:])
    ) / float(days * days)
    if not np.isfinite(exact_variance) or exact_variance <= 0.0:
        raise RuntimeError("exact mean variance is not finite and positive")
    return {
        "day_count": days,
        "asymptotic_long_run_variance": float(long_run_variance),
        "asymptotic_mean_variance": float(long_run_variance / days),
        "exact_mean_variance": float(exact_variance),
        "exact_mean_standard_error": float(np.sqrt(exact_variance)),
    }


def exact_gaussian_time_covariance(
    temporal_dependence: str,
    *,
    day_count: int,
) -> np.ndarray:
    """Return the truth-known finite-sample covariance of one daily series."""
    days = int(day_count)
    if days <= 1:
        raise ValueError("day_count must exceed one")
    dependence = str(temporal_dependence)
    lags = np.arange(days, dtype=int)
    if dependence == "iid":
        autocovariance = np.zeros(days, dtype=float)
        autocovariance[0] = 1.0
    elif dependence.startswith("ar1_"):
        phi = float(dependence.rsplit("_", 1)[1])
        if not -1.0 < phi < 1.0:
            raise ValueError("AR(1) coefficient must lie strictly between -1 and 1")
        autocovariance = np.power(phi, lags, dtype=float)
    elif dependence.startswith("ma_"):
        window = int(dependence.rsplit("_", 1)[1])
        if window <= 0:
            raise ValueError("MA window must be positive")
        autocovariance = np.maximum(window - lags, 0).astype(float) / float(window)
    else:
        raise ValueError(f"unsupported temporal dependence: {dependence}")
    covariance = autocovariance[np.abs(np.subtract.outer(lags, lags))]
    if not np.isfinite(covariance).all():
        raise RuntimeError("truth-known time covariance is not finite")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("truth-known time covariance is not positive definite") from exc
    return covariance


@lru_cache(maxsize=16)
def _exact_gaussian_gls_contract(
    temporal_dependence: str,
    day_count: int,
) -> tuple[np.ndarray, float]:
    covariance = exact_gaussian_time_covariance(
        temporal_dependence,
        day_count=day_count,
    )
    ones = np.ones(int(day_count), dtype=float)
    precision_ones = np.linalg.solve(covariance, ones)
    denominator = float(ones @ precision_ones)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError("truth-known GLS precision is not finite and positive")
    weights = precision_ones / denominator
    weights.setflags(write=False)
    return weights, float(1.0 / denominator)


def exact_gaussian_gls_contract(
    temporal_dependence: str,
    *,
    day_count: int,
) -> dict[str, object]:
    """Return a copy of the truth-known GLS weights and estimator variance."""
    weights, variance = _exact_gaussian_gls_contract(
        str(temporal_dependence),
        int(day_count),
    )
    return {
        "weights": weights.copy(),
        "estimator_variance": float(variance),
    }


def oracle_mean_gls_family(
    dataset: LayerADataset,
    *,
    temporal_dependence: str,
    alternative: str = "greater",
) -> pd.DataFrame:
    """Compare equal-weight and truth-known GLS Gaussian tests for one family."""
    if alternative not in {"greater", "two-sided"}:
        raise ValueError("alternative must be greater or two-sided")
    values = dataset.daily_values
    truth = dataset.true_effects
    if values.empty or values.shape[1] <= 1:
        raise ValueError("oracle mean/GLS family requires multiple hypotheses")
    if tuple(values.columns.astype(str)) != tuple(truth.index.astype(str)):
        raise ValueError("daily values and true effects must have identical ordered hypotheses")
    numeric_values = values.to_numpy(dtype=float)
    numeric_truth = truth.to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all() or not np.isfinite(numeric_truth).all():
        raise ValueError("oracle mean/GLS inputs must be finite")
    day_count = len(values)
    weights, gls_variance = _exact_gaussian_gls_contract(
        str(temporal_dependence),
        int(day_count),
    )
    mean_variance = exact_gaussian_mean_variance(
        str(temporal_dependence),
        day_count=int(day_count),
    )["exact_mean_variance"]
    mean_effect = numeric_values.mean(axis=0)
    gls_effect = weights @ numeric_values
    mean_z = mean_effect / sqrt(float(mean_variance))
    gls_z = gls_effect / sqrt(float(gls_variance))
    if alternative == "greater":
        mean_raw = normal_distribution.sf(mean_z)
        gls_raw = normal_distribution.sf(gls_z)
        true_alternative = numeric_truth > 0.0
    else:
        mean_raw = 2.0 * normal_distribution.sf(np.abs(mean_z))
        gls_raw = 2.0 * normal_distribution.sf(np.abs(gls_z))
        true_alternative = numeric_truth != 0.0
    result = pd.DataFrame(
        {
            "hypothesis_id": values.columns.astype(str),
            "true_effect": numeric_truth,
            "is_true_alternative": true_alternative,
            "mean_effect": mean_effect,
            "mean_standard_error": sqrt(float(mean_variance)),
            "mean_raw_p_value": mean_raw,
            "mean_bh_q_value": research_stats.benjamini_hochberg_q_values(mean_raw),
            "gls_effect": gls_effect,
            "gls_standard_error": sqrt(float(gls_variance)),
            "gls_raw_p_value": gls_raw,
            "gls_bh_q_value": research_stats.benjamini_hochberg_q_values(gls_raw),
            "alternative": alternative,
        }
    )
    if not np.isfinite(
        result.drop(columns=["hypothesis_id", "alternative"]).to_numpy(dtype=float)
    ).all():
        raise RuntimeError("oracle mean/GLS output is not finite")
    return result


def summarize_discovery_power_attribution(
    hypothesis_results: pd.DataFrame,
    *,
    alpha: float = 0.05,
    family_size: int = 47,
) -> DiscoveryPowerAttributionArtifacts:
    """Summarize within-scenario losses from multiplicity and mean inefficiency."""
    required = {
        "registered_task_idx", "scenario_id", "analysis_specification", "replicate",
        "hypothesis_id", "true_effect", "is_true_alternative", "alternative",
        "mean_effect", "mean_standard_error", "mean_raw_p_value", "mean_bh_q_value",
        "gls_effect", "gls_standard_error", "gls_raw_p_value", "gls_bh_q_value",
    }
    missing = sorted(required.difference(hypothesis_results.columns))
    if missing:
        raise ValueError("power-attribution input missing columns: " + ", ".join(missing))
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if int(family_size) <= 1:
        raise ValueError("family_size must exceed one")
    frame = hypothesis_results.loc[:, sorted(required)].copy()
    if frame.isna().any().any():
        raise ValueError("power-attribution input must not contain missing values")
    task_keys = ["registered_task_idx", "scenario_id", "analysis_specification", "replicate"]
    if frame.duplicated(task_keys + ["hypothesis_id"]).any():
        raise ValueError("power-attribution input contains duplicate task hypotheses")
    expected_ids = tuple(f"H{index:02d}" for index in range(1, int(family_size) + 1))
    identity = frame.groupby(task_keys, sort=False).agg(
        hypothesis_count=("hypothesis_id", "size"),
        hypothesis_ids=("hypothesis_id", lambda values: tuple(values)),
        alternative_count=("alternative", "nunique"),
    )
    if (
        identity["hypothesis_count"].ne(int(family_size)).any()
        or (~identity["hypothesis_ids"].map(lambda value: value == expected_ids)).any()
        or identity["alternative_count"].ne(1).any()
    ):
        raise ValueError("power attribution requires complete, ordered fixed-size families")
    numeric_columns = [
        "true_effect", "mean_effect", "mean_standard_error", "mean_raw_p_value",
        "mean_bh_q_value", "gls_effect", "gls_standard_error", "gls_raw_p_value",
        "gls_bh_q_value",
    ]
    numeric = frame[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("power-attribution numeric inputs must be finite")
    probability_columns = [
        "mean_raw_p_value", "mean_bh_q_value", "gls_raw_p_value", "gls_bh_q_value"
    ]
    if ((frame[probability_columns] < 0.0) | (frame[probability_columns] > 1.0)).any().any():
        raise ValueError("power-attribution probabilities must lie in [0, 1]")

    variants = {
        "MEAN_RAW": ("mean_raw_p_value", "mean_effect", "mean_standard_error", False),
        "MEAN_BH": ("mean_bh_q_value", "mean_effect", "mean_standard_error", True),
        "GLS_RAW": ("gls_raw_p_value", "gls_effect", "gls_standard_error", False),
        "GLS_BH": ("gls_bh_q_value", "gls_effect", "gls_standard_error", True),
    }
    long_rows: list[pd.DataFrame] = []
    for variant, (probability, effect, standard_error, controls_fdr) in variants.items():
        candidate = frame[
            task_keys + ["hypothesis_id", "true_effect", "is_true_alternative", "alternative"]
        ].copy()
        candidate["method_variant"] = variant
        candidate["observed_effect"] = frame[effect].to_numpy(dtype=float)
        candidate["standard_error"] = frame[standard_error].to_numpy(dtype=float)
        candidate["decision_probability"] = frame[probability].to_numpy(dtype=float)
        candidate["controls_fdr_by_design"] = bool(controls_fdr)
        candidate["rejected"] = candidate["decision_probability"] <= float(alpha)
        long_rows.append(candidate)
    hypothesis = pd.concat(long_rows, ignore_index=True)
    hypothesis["false_discovery"] = hypothesis["rejected"] & ~hypothesis["is_true_alternative"]
    hypothesis["true_discovery"] = hypothesis["rejected"] & hypothesis["is_true_alternative"]

    group_keys = task_keys + ["method_variant", "controls_fdr_by_design"]
    task = hypothesis.groupby(group_keys, as_index=False, sort=False).agg(
        discovery_count=("rejected", "sum"),
        false_discovery_count=("false_discovery", "sum"),
        true_discovery_count=("true_discovery", "sum"),
        true_alternative_count=("is_true_alternative", "sum"),
    )
    task["false_discovery_proportion"] = np.where(
        task["discovery_count"] > 0,
        task["false_discovery_count"] / task["discovery_count"],
        0.0,
    )
    task["true_positive_rate"] = np.where(
        task["true_alternative_count"] > 0,
        task["true_discovery_count"] / task["true_alternative_count"],
        np.nan,
    )
    scenario_keys = [
        "scenario_id", "analysis_specification", "method_variant", "controls_fdr_by_design"
    ]
    scenario = task.groupby(scenario_keys, as_index=False, sort=False).agg(
        monte_carlo_repetitions=("replicate", "nunique"),
        mean_false_discovery_proportion=("false_discovery_proportion", "mean"),
        mean_true_positive_rate=("true_positive_rate", "mean"),
        any_true_discovery_rate=("true_discovery_count", lambda values: float((values > 0).mean())),
        any_false_discovery_rate=("false_discovery_count", lambda values: float((values > 0).mean())),
        mean_discovery_count=("discovery_count", "mean"),
    )
    pivot = scenario.pivot(
        index=["scenario_id", "analysis_specification"],
        columns="method_variant",
        values="mean_true_positive_rate",
    ).reset_index()
    for variant in variants:
        if variant not in pivot:
            raise RuntimeError(f"power attribution is missing method variant {variant}")
    attribution = pivot.rename(
        columns={variant: f"{variant.lower()}_tpr" for variant in variants}
    )
    attribution["bh_loss"] = attribution["mean_raw_tpr"] - attribution["mean_bh_tpr"]
    attribution["gls_raw_gain"] = attribution["gls_raw_tpr"] - attribution["mean_raw_tpr"]
    attribution["gls_bh_gain"] = attribution["gls_bh_tpr"] - attribution["mean_bh_tpr"]
    return DiscoveryPowerAttributionArtifacts(hypothesis, task, scenario, attribution)


def calibrate_real_standardized_effects(
    daily_effects: pd.DataFrame,
    *,
    block_length: int = 14,
    n_bootstrap: int = 10_000,
    seed: int = 20_260_814_01,
    distribution_quantiles: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
    simulation_grid: Sequence[float] = (0.10, 0.20, 0.35),
    batch_size: int = 100,
) -> RealEffectCalibrationArtifacts:
    """Calibrate a simulation effect grid to synchronized real daily statistics."""
    if not isinstance(daily_effects, pd.DataFrame) or daily_effects.empty:
        raise ValueError("daily_effects must be a non-empty DataFrame")
    if daily_effects.index.has_duplicates:
        raise ValueError("daily_effects index must be unique")
    if daily_effects.columns.has_duplicates:
        raise ValueError("daily_effects hypothesis columns must be unique")
    values = daily_effects.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("daily_effects must be finite and complete")
    day_count, hypothesis_count = values.shape
    if day_count < 3 or hypothesis_count < 2:
        raise ValueError("daily_effects require at least three days and two hypotheses")
    length = int(block_length)
    draws = int(n_bootstrap)
    batch = int(batch_size)
    if length <= 0 or length > day_count:
        raise ValueError("block_length must lie between one and day_count")
    if draws <= 0 or batch <= 0:
        raise ValueError("n_bootstrap and batch_size must be positive")
    standard_deviation = values.std(axis=0, ddof=1)
    if not np.isfinite(standard_deviation).all() or (standard_deviation <= 0.0).any():
        raise ValueError("daily_effects contains a constant hypothesis")
    quantile_levels = np.asarray(tuple(distribution_quantiles), dtype=float)
    if (
        quantile_levels.ndim != 1
        or len(quantile_levels) == 0
        or not np.isfinite(quantile_levels).all()
        or (quantile_levels < 0.0).any()
        or (quantile_levels > 1.0).any()
        or np.any(np.diff(quantile_levels) <= 0.0)
    ):
        raise ValueError(
            "distribution_quantiles must be unique and strictly increasing in [0, 1]"
        )
    grid = np.asarray(tuple(simulation_grid), dtype=float)
    if (
        grid.ndim != 1
        or len(grid) == 0
        or not np.isfinite(grid).all()
        or np.any(np.diff(grid) <= 0.0)
    ):
        raise ValueError("simulation_grid must be finite and strictly increasing")

    observed = values.mean(axis=0) / standard_deviation
    bootstrap_effects = np.empty((draws, hypothesis_count), dtype=float)
    blocks_per_draw = int(np.ceil(day_count / length))
    offsets = np.arange(length, dtype=int)
    generator = np.random.default_rng(int(seed))
    for start in range(0, draws, batch):
        stop = min(start + batch, draws)
        starts = generator.integers(
            0, day_count, size=(stop - start, blocks_per_draw), endpoint=False
        )
        indices = (starts[..., None] + offsets) % day_count
        indices = indices.reshape(stop - start, -1)[:, :day_count]
        sampled = values[indices]
        sampled_sd = sampled.std(axis=1, ddof=1)
        if not np.isfinite(sampled_sd).all() or (sampled_sd <= 0.0).any():
            raise RuntimeError("bootstrap produced a constant hypothesis")
        bootstrap_effects[start:stop] = sampled.mean(axis=1) / sampled_sd

    hypothesis = pd.DataFrame(
        {
            "hypothesis_id": daily_effects.columns.astype(str),
            "day_count": day_count,
            "daily_mean": values.mean(axis=0),
            "daily_standard_deviation": standard_deviation,
            "standardized_effect": observed,
            "bootstrap_ci_lower": np.quantile(bootstrap_effects, 0.025, axis=0),
            "bootstrap_median": np.quantile(bootstrap_effects, 0.5, axis=0),
            "bootstrap_ci_upper": np.quantile(bootstrap_effects, 0.975, axis=0),
        }
    )
    bootstrap_distribution = np.quantile(
        bootstrap_effects, quantile_levels, axis=1
    )
    observed_distribution = np.quantile(observed, quantile_levels)
    distribution = pd.DataFrame(
        {
            "quantile": quantile_levels,
            "observed_standardized_effect": observed_distribution,
            "bootstrap_ci_lower": np.quantile(bootstrap_distribution, 0.025, axis=1),
            "bootstrap_median": np.quantile(bootstrap_distribution, 0.5, axis=1),
            "bootstrap_ci_upper": np.quantile(bootstrap_distribution, 0.975, axis=1),
        }
    )
    observed_bounds = {
        level: float(np.quantile(observed, level))
        for level in (0.0, 0.1, 0.25, 0.75, 0.9, 1.0)
    }
    alignment = pd.DataFrame(
        {
            "simulation_effect": grid,
            "empirical_percentile": [float(np.mean(observed <= value)) for value in grid],
            "within_observed_min_max": [
                bool(observed_bounds[0.0] <= value <= observed_bounds[1.0])
                for value in grid
            ],
            "within_observed_p10_p90": [
                bool(observed_bounds[0.1] <= value <= observed_bounds[0.9])
                for value in grid
            ],
            "within_observed_p25_p75": [
                bool(observed_bounds[0.25] <= value <= observed_bounds[0.75])
                for value in grid
            ],
        }
    )
    return RealEffectCalibrationArtifacts(hypothesis, distribution, alignment)


def dependence_profile(
    daily_values: pd.DataFrame,
    group_labels: Sequence[str] | pd.Series,
    *,
    max_lag: int,
    expected_day_count: int | None = None,
    expected_hypothesis_count: int | None = None,
) -> DependenceProfileArtifacts:
    """Describe temporal and contemporaneous dependence on one daily panel."""
    if not isinstance(daily_values, pd.DataFrame) or daily_values.empty:
        raise TypeError("daily_values must be a non-empty DataFrame")
    if daily_values.index.has_duplicates or daily_values.columns.has_duplicates:
        raise ValueError("daily dependence input has duplicate index or columns")
    if not isinstance(daily_values.index, pd.DatetimeIndex):
        raise TypeError("daily dependence input requires a DatetimeIndex")
    if daily_values.index.tz is None or str(daily_values.index.tz) != "UTC":
        raise ValueError("daily dependence index must use UTC")
    expected_index = pd.date_range(
        daily_values.index[0], periods=len(daily_values), freq="D", tz="UTC"
    )
    if not daily_values.index.equals(expected_index):
        raise ValueError("daily dependence index must be strictly consecutive")
    if expected_day_count is not None and len(daily_values) != int(expected_day_count):
        raise ValueError("daily dependence input has the wrong day count")
    if (
        expected_hypothesis_count is not None
        and daily_values.shape[1] != int(expected_hypothesis_count)
    ):
        raise ValueError("daily dependence input has the wrong hypothesis count")
    lag_count = int(max_lag)
    if lag_count <= 0 or lag_count >= len(daily_values):
        raise ValueError("max_lag must be positive and less than the day count")
    values = daily_values.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("daily dependence input must be finite")
    centered = values - values.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.mean(np.square(centered), axis=0))
    if not np.isfinite(scale).all() or (scale <= 0.0).any():
        raise ValueError("daily dependence input contains a constant column")
    standardized = centered / scale[None, :]
    labels = pd.Series(group_labels, index=daily_values.columns, dtype="object")
    if len(labels) != daily_values.shape[1] or labels.isna().any():
        raise ValueError("group_labels must cover every hypothesis")
    labels = labels.astype(str)
    if (labels.str.len() == 0).any():
        raise ValueError("group_labels must be non-empty")

    temporal_rows: list[dict[str, object]] = []
    day_count = len(daily_values)
    for lag in range(1, lag_count + 1):
        correlations = np.sum(
            standardized[lag:] * standardized[:-lag], axis=0
        ) / float(day_count)
        for column, value in zip(daily_values.columns.astype(str), correlations):
            temporal_rows.append(
                {"hypothesis_id": column, "lag_days": lag, "autocorrelation": float(value)}
            )
    temporal = pd.DataFrame(temporal_rows)
    temporal_summary = (
        temporal.groupby("lag_days", as_index=False, sort=True)
        .agg(
            autocorrelation_mean=("autocorrelation", "mean"),
            autocorrelation_median=("autocorrelation", "median"),
            autocorrelation_q25=("autocorrelation", lambda value: value.quantile(0.25)),
            autocorrelation_q75=("autocorrelation", lambda value: value.quantile(0.75)),
        )
    )

    contemporaneous = standardized.T @ standardized / float(day_count)
    cross_rows: list[dict[str, object]] = []
    columns = daily_values.columns.astype(str)
    for left in range(len(columns)):
        for right in range(left + 1, len(columns)):
            relation = "within" if labels.iloc[left] == labels.iloc[right] else "between"
            cross_rows.append(
                {
                    "left_hypothesis_id": columns[left],
                    "right_hypothesis_id": columns[right],
                    "left_group": labels.iloc[left],
                    "right_group": labels.iloc[right],
                    "relation": relation,
                    "correlation": float(contemporaneous[left, right]),
                }
            )
    cross_pairs = pd.DataFrame(cross_rows)
    if cross_pairs.empty or set(cross_pairs["relation"]) != {"within", "between"}:
        raise ValueError("dependence groups must produce within and between pairs")
    cross_summary = (
        cross_pairs.groupby("relation", as_index=False, sort=True)
        .agg(
            pair_count=("correlation", "size"),
            correlation_mean=("correlation", "mean"),
            correlation_median=("correlation", "median"),
            correlation_q25=("correlation", lambda value: value.quantile(0.25)),
            correlation_q75=("correlation", lambda value: value.quantile(0.75)),
        )
    )
    return DependenceProfileArtifacts(
        temporal_by_hypothesis=temporal,
        temporal_summary=temporal_summary,
        cross_pairs=cross_pairs,
        cross_summary=cross_summary,
    )


def theoretical_family_max_quantiles(
    group_sizes: Sequence[int],
    *,
    within: float,
    between: float,
    seed: int | np.random.SeedSequence,
    draw_count: int = 1_000_000,
    batch_size: int = 10_000,
) -> pd.DataFrame:
    """Simulate the frozen Gaussian family-maximum reference distribution."""
    draws = int(draw_count)
    batch = int(batch_size)
    if draws <= 0 or batch <= 0 or draws % batch != 0:
        raise ValueError("draw_count must be positive and divisible by batch_size")
    if isinstance(seed, np.random.SeedSequence):
        seed_sequence = seed
    else:
        if int(seed) < 0:
            raise ValueError("seed must be non-negative")
        seed_sequence = np.random.SeedSequence(int(seed))
    correlation = grouped_correlation(group_sizes, float(within), float(between))
    cholesky = np.linalg.cholesky(correlation)
    rng = np.random.Generator(np.random.PCG64DXSM(seed_sequence))
    maxima = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, batch):
        stop = start + batch
        sample = rng.standard_normal((batch, len(correlation))) @ cholesky.T
        maxima[start:stop] = sample.max(axis=1)
    return pd.DataFrame(
        [{
            "draw_count": draws,
            "batch_size": batch,
            "seed_entropy": str(seed_sequence.entropy),
            "seed_spawn_key": ";".join(str(value) for value in seed_sequence.spawn_key),
            "generator_algorithm": "PCG64DXSM",
            "matrix_factorization": "numpy.linalg.cholesky",
            "float_dtype": "float64",
            "quantile_method": "linear",
            "within_correlation": float(within),
            "between_correlation": float(between),
            "q90": float(np.quantile(maxima, 0.90, method="linear")),
            "q95": float(np.quantile(maxima, 0.95, method="linear")),
            "q99": float(np.quantile(maxima, 0.99, method="linear")),
        }]
    )


def infer_layer_a_dataset(
    dataset: LayerADataset, *, block_length: int, n_bootstrap: int, seed: int,
    alternative: str = "greater", production_equivalent: bool = False,
) -> pd.DataFrame:
    """Apply the registered family inference to one truth-known dataset."""
    return infer_layer_a_dataset_with_engine(
        dataset,
        engine="E0",
        dependence_length=block_length,
        n_bootstrap=n_bootstrap,
        seed=seed,
        alternative=alternative,
        production_equivalent=production_equivalent,
    )


def infer_layer_a_dataset_with_engine(
    dataset: LayerADataset,
    *,
    engine: str,
    dependence_length: int,
    n_bootstrap: int,
    seed: int,
    alternative: str = "greater",
    production_equivalent: bool = False,
) -> pd.DataFrame:
    """Apply one frozen E0-E2 joint-inference engine to a Layer-A dataset."""
    return infer_layer_a_dataset_with_engine_artifacts(
        dataset,
        engine=engine,
        dependence_length=dependence_length,
        n_bootstrap=n_bootstrap,
        seed=seed,
        alternative=alternative,
        production_equivalent=production_equivalent,
    ).results


def infer_layer_a_dataset_with_engine_artifacts(
    dataset: LayerADataset,
    *,
    engine: str,
    dependence_length: int,
    n_bootstrap: int,
    seed: int,
    alternative: str = "greater",
    production_equivalent: bool = False,
) -> LayerAInferenceArtifacts:
    """Apply a registered Layer-A inference engine and retain diagnostics."""
    engine = str(engine)
    if engine not in JOINT_INFERENCE_ENGINES:
        raise ValueError(f"unsupported joint-inference engine: {engine}")
    values = dataset.daily_values
    effects = values.mean(axis=0)
    centered = values.subtract(effects, axis="columns")
    counts = pd.DataFrame(1, index=values.index, columns=values.columns)
    if engine in {"E1H_AIC", "E1H_BIC", "E1J_BIC_1125"}:
        criterion = "BIC" if engine == "E1J_BIC_1125" else engine.rsplit("_", 1)[1]
        multiplier = 1.125 if engine == "E1J_BIC_1125" else 1.0
        artifacts = research_stats.autoregressive_spectral_holm_test(
            centered,
            counts,
            effects,
            order_criterion=criterion,
            standard_error_multiplier=multiplier,
            alternative=alternative,
            expected_hypothesis_count=len(values.columns),
        )
        truth = dataset.true_effects.rename("true_effect").rename_axis(
            "hypothesis_id"
        ).reset_index()
        result = artifacts.summary.merge(
            truth,
            on="hypothesis_id",
            validate="one_to_one",
        )
        result.insert(0, "joint_inference_engine", engine)
        result["dependence_length_days"] = 0
        result["n_bootstrap"] = 0
        result["seed"] = int(seed)
        if alternative == "two-sided":
            result["is_true_positive"] = result["true_effect"] != 0.0
            result["is_true_null"] = result["true_effect"] == 0.0
        else:
            result["is_true_positive"] = result["true_effect"] > 0.0
            result["is_true_null"] = result["true_effect"] <= 0.0
        return LayerAInferenceArtifacts(
            result,
            pd.DataFrame(
                columns=["bootstrap_idx", "bootstrap_max_test_statistic"]
            ),
        )
    if engine == "E0":
        entry = (
            research_stats.circular_block_bootstrap_stepdown_max_t
            if production_equivalent
            else research_stats.simulation_calibration_circular_block_stepdown_max_t
        )
        dependence_keyword = "block_length"
    elif engine in {"E1", "E1F", "E1S"}:
        entry = (
            (
                research_stats.self_normalized_circular_block_bootstrap_stepdown_max_t
                if production_equivalent
                else research_stats.simulation_calibration_self_normalized_stepdown_max_t
            )
            if engine == "E1S"
            else
            (
                research_stats.adaptive_flat_top_restudentized_circular_block_bootstrap_stepdown_max_t
                if production_equivalent
                else research_stats.simulation_calibration_adaptive_flat_top_restudentized_stepdown_max_t
            )
            if engine == "E1F"
            else (
                research_stats.restudentized_circular_block_bootstrap_stepdown_max_t
                if production_equivalent
                else research_stats.simulation_calibration_restudentized_circular_block_stepdown_max_t
            )
        )
        dependence_keyword = "block_length"
    else:
        entry = (
            research_stats.dependent_multiplier_bootstrap_stepdown_max_t
            if production_equivalent
            else research_stats.simulation_calibration_dependent_multiplier_bootstrap_stepdown_max_t
        )
        dependence_keyword = "bandwidth"
    artifacts = entry(
        centered,
        counts,
        effects,
        **{dependence_keyword: int(dependence_length)},
        n_bootstrap=int(n_bootstrap),
        seed=int(seed),
        alternative=alternative,
    )
    truth = dataset.true_effects.rename("true_effect").rename_axis(
        "hypothesis_id"
    ).reset_index()
    result = artifacts.summary.merge(
        truth,
        on="hypothesis_id", validate="one_to_one",
    )
    result.insert(0, "joint_inference_engine", engine)
    if alternative == "two-sided":
        result["is_true_positive"] = result["true_effect"] != 0.0
        result["is_true_null"] = result["true_effect"] == 0.0
    else:
        result["is_true_positive"] = result["true_effect"] > 0.0
        result["is_true_null"] = result["true_effect"] <= 0.0
    bootstrap_values = artifacts.bootstrap_t_values.to_numpy(dtype=float)
    if alternative == "two-sided":
        bootstrap_values = np.abs(bootstrap_values)
    bootstrap_max = pd.DataFrame(
        {
            "bootstrap_idx": np.arange(len(bootstrap_values), dtype=int),
            "bootstrap_max_test_statistic": bootstrap_values.max(axis=1),
        }
    )
    return LayerAInferenceArtifacts(result, bootstrap_max)


def evaluate_e1s_mechanism_preflight(
    results: pd.DataFrame,
    *,
    scenario_ids: Sequence[str] = ("A03", "A04", "A05"),
    replicates_per_scenario: int = 100,
    fail_rejection_count: int = 15,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Evaluate the frozen E1S obvious-failure preflight.

    This is a development stop rule, not a method-promotion criterion. Every
    registered replicate must contain the complete 47-hypothesis all-null
    family and is rejected when any step-down adjusted p-value is at most .05.
    """
    required = {
        "joint_inference_engine", "scenario_id", "replicate", "hypothesis_id",
        "true_effect", "stepdown_max_t_adjusted_p_value",
    }
    if not required.issubset(results.columns):
        raise ValueError("E1S preflight results are missing required columns")
    frame = results.copy()
    scenarios = tuple(str(value) for value in scenario_ids)
    if len(scenarios) != len(set(scenarios)) or not scenarios:
        raise ValueError("E1S preflight scenario ids must be unique")
    if int(replicates_per_scenario) <= 0 or not 0 < int(fail_rejection_count) <= int(
        replicates_per_scenario
    ):
        raise ValueError("invalid E1S preflight repetition or failure count")
    if set(frame["joint_inference_engine"].astype(str)) != {"E1S"}:
        raise ValueError("E1S preflight contains another engine")
    if set(frame["scenario_id"].astype(str)) != set(scenarios):
        raise ValueError("E1S preflight scenario coverage changed")
    if not np.isfinite(frame["true_effect"].to_numpy(dtype=float)).all() or not np.allclose(
        frame["true_effect"].to_numpy(dtype=float), 0.0
    ):
        raise ValueError("E1S mechanism preflight requires all-null scenarios")
    if frame.duplicated(["scenario_id", "replicate", "hypothesis_id"]).any():
        raise ValueError("E1S preflight contains duplicate hypotheses")
    expected_hypotheses = {f"H{index:02d}" for index in range(1, 48)}
    rows = []
    for scenario in scenarios:
        group = frame.loc[frame["scenario_id"].astype(str).eq(scenario)]
        if set(group["replicate"].astype(int)) != set(range(int(replicates_per_scenario))):
            raise ValueError(f"E1S preflight replicate coverage changed for {scenario}")
        coverage = group.groupby("replicate")["hypothesis_id"].agg(
            lambda values: set(values.astype(str))
        )
        if not coverage.map(lambda value: value == expected_hypotheses).all():
            raise ValueError(f"E1S preflight hypothesis coverage changed for {scenario}")
        family_rejected = group.groupby("replicate")[
            "stepdown_max_t_adjusted_p_value"
        ].min().le(0.05)
        rejection_count = int(family_rejected.sum())
        rows.append(
            {
                "scenario_id": scenario,
                "replicate_count": int(replicates_per_scenario),
                "false_family_rejection_count": rejection_count,
                "false_family_rejection_rate": rejection_count
                / float(replicates_per_scenario),
                "fail_rejection_count": int(fail_rejection_count),
                "obvious_failure": rejection_count >= int(fail_rejection_count),
            }
        )
    summary = pd.DataFrame(rows)
    failed = summary.loc[summary["obvious_failure"], "scenario_id"].tolist()
    decision = {
        "engine": "E1S",
        "status": "mechanism_preflight_failed" if failed else "mechanism_preflight_pass",
        "pass": not failed,
        "failed_scenarios": failed,
        "promotes_method": False,
    }
    return summary, decision


def _clopper_pearson_bounds(
    successes: int,
    trials: int,
    *,
    confidence: float,
    sides: str,
) -> tuple[float, float]:
    """Exact binomial bounds used by the frozen simulation gates."""
    successes = int(successes)
    trials = int(trials)
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if sides not in {"two-sided", "upper"}:
        raise ValueError("sides must be two-sided or upper")
    tail = (1.0 - float(confidence)) / 2.0 if sides == "two-sided" else 1.0 - float(confidence)
    lower = 0.0 if successes == 0 else float(
        beta_distribution.ppf(tail, successes, trials - successes + 1)
    )
    upper = 1.0 if successes == trials else float(
        beta_distribution.ppf(1.0 - tail, successes + 1, trials - successes)
    )
    return lower, upper


def evaluate_joint_inference_calibration(
    results: pd.DataFrame,
    *,
    phase: str,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Evaluate the frozen development or confirmation calibration gates.

    The input contains hypothesis-level results for complete Monte Carlo
    replicates. This is the only conclusion-bearing entry for the E1/E2
    calibration and power gates.
    """
    required = {
        "joint_inference_engine",
        "scenario_id",
        "analysis_specification",
        "replicate",
        "hypothesis_id",
        "true_effect",
        "alternative",
    }
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError("joint-inference results missing columns: " + ", ".join(missing))
    if phase not in {"development", "confirmation"}:
        raise ValueError("phase must be development or confirmation")
    frame = results.copy()
    if frame[list(required)].isna().any().any():
        raise ValueError("joint-inference gate inputs must not contain missing values")
    if not set(frame["joint_inference_engine"].astype(str)).issubset(
        set(JOINT_INFERENCE_ENGINES)
    ):
        raise ValueError("joint-inference gate input contains an unknown engine")
    if "family_adjusted_p_value" in frame.columns:
        adjusted = frame["family_adjusted_p_value"].astype(float)
        if "stepdown_max_t_adjusted_p_value" in frame.columns:
            legacy = frame["stepdown_max_t_adjusted_p_value"].astype(float)
            both = adjusted.notna() & legacy.notna()
            if both.any() and not np.allclose(
                adjusted.loc[both], legacy.loc[both], atol=1e-12, rtol=0.0
            ):
                raise ValueError("generic and legacy family-adjusted p-values differ")
    elif "stepdown_max_t_adjusted_p_value" in frame.columns:
        adjusted = frame["stepdown_max_t_adjusted_p_value"].astype(float)
    else:
        raise ValueError("joint-inference results lack family-adjusted p-values")
    if adjusted.isna().any() or ((adjusted < 0.0) | (adjusted > 1.0)).any():
        raise ValueError("family-adjusted p-values must be finite and lie in [0, 1]")
    frame["rejected"] = adjusted <= float(alpha)
    alternative = frame["alternative"].astype(str)
    if not set(alternative).issubset({"greater", "two-sided"}):
        raise ValueError("unsupported alternative in joint-inference gate input")
    frame["true_positive"] = np.where(
        alternative.eq("two-sided"),
        frame["true_effect"].astype(float).ne(0.0),
        frame["true_effect"].astype(float).gt(0.0),
    )
    replicate_keys = [
        "joint_inference_engine",
        "scenario_id",
        "analysis_specification",
        "replicate",
    ]
    replicate = (
        frame.groupby(replicate_keys, sort=True, as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "any_false_rejection": bool(
                        (group["rejected"] & ~group["true_positive"]).any()
                    ),
                    "any_true_rejection": bool(
                        (group["rejected"] & group["true_positive"]).any()
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    rows: list[dict[str, object]] = []
    specification_keys = [
        "joint_inference_engine",
        "scenario_id",
        "analysis_specification",
    ]
    for keys, group in replicate.groupby(specification_keys, sort=True):
        engine, scenario_id, specification = keys
        trials = len(group)
        false_count = int(group["any_false_rejection"].sum())
        lower, two_sided_upper = _clopper_pearson_bounds(
            false_count, trials, confidence=0.95, sides="two-sided"
        )
        _, one_sided_upper = _clopper_pearson_bounds(
            false_count, trials, confidence=0.95, sides="upper"
        )
        rows.append(
            {
                "joint_inference_engine": engine,
                "scenario_id": scenario_id,
                "analysis_specification": specification,
                "replicate_count": trials,
                "false_family_detection_count": false_count,
                "family_wise_error_rate": false_count / trials,
                "clopper_pearson_two_sided_95_low": lower,
                "clopper_pearson_two_sided_95_high": two_sided_upper,
                "clopper_pearson_one_sided_95_upper": one_sided_upper,
                "any_power": float(group["any_true_rejection"].mean()),
            }
        )
    specification_summary = pd.DataFrame(rows)
    true_hypothesis = (
        frame.loc[frame["true_positive"]]
        .groupby(
            specification_keys + ["hypothesis_id"], sort=True, as_index=False
        )
        .agg(true_effect=("true_effect", "first"), rejection_rate=("rejected", "mean"))
    )
    decision: dict[str, object] = {"phase": phase, "alpha": float(alpha), "engines": {}}
    for engine, group in specification_summary.groupby(
        "joint_inference_engine", sort=True
    ):
        if phase == "development":
            calibrated = bool(
                group["family_wise_error_rate"].le(0.060).all()
                and group["clopper_pearson_two_sided_95_low"].le(0.05).all()
            )
            a08_power = true_hypothesis.loc[
                true_hypothesis["joint_inference_engine"].eq(engine)
                & true_hypothesis["scenario_id"].eq("A08"),
                "rejection_rate",
            ]
            power_pass = bool(not a08_power.empty and a08_power.ge(0.80).all())
            decision["engines"][str(engine)] = {
                "calibration_pass": calibrated,
                "a08_each_true_hypothesis_power_pass": power_pass,
                "pass": calibrated and power_pass,
                "worst_specification_fwer": float(group["family_wise_error_rate"].max()),
            }
        else:
            if group["replicate_count"].nunique() != 1:
                raise ValueError("confirmation specifications must use equal replicate counts")
            pooled_successes = int(group["false_family_detection_count"].sum())
            pooled_trials = int(group["replicate_count"].sum())
            _, pooled_upper = _clopper_pearson_bounds(
                pooled_successes, pooled_trials, confidence=0.95, sides="upper"
            )
            pooled_pass = pooled_upper <= 0.060
            per_spec_pass = bool(
                group["family_wise_error_rate"].le(0.065).all()
                and group["clopper_pearson_one_sided_95_upper"].le(0.075).all()
            )
            a08_power = true_hypothesis.loc[
                true_hypothesis["joint_inference_engine"].eq(engine)
                & true_hypothesis["scenario_id"].eq("A08"),
                "rejection_rate",
            ]
            power_pass = bool(not a08_power.empty and a08_power.ge(0.80).all())
            decision["engines"][str(engine)] = {
                "pooled_false_family_detection_count": pooled_successes,
                "pooled_replicate_count": pooled_trials,
                "pooled_fwer": pooled_successes / pooled_trials,
                "pooled_clopper_pearson_one_sided_95_upper": pooled_upper,
                "pooled_pass": pooled_pass,
                "each_specification_pass": per_spec_pass,
                "a08_each_true_hypothesis_power_pass": power_pass,
                "pass": pooled_pass and per_spec_pass and power_pass,
            }
    return specification_summary, true_hypothesis, decision


def select_e1h_development_specification(
    decision: Mapping[str, object],
    *,
    fwer_tolerance: float = 0.01,
) -> str:
    """Select the sole E1H confirmation specification by the frozen rule."""
    engines = decision.get("engines")
    if not isinstance(engines, Mapping):
        raise ValueError("development decision has no engine mapping")
    passing: list[tuple[str, float]] = []
    for engine in ("E1H_AIC", "E1H_BIC"):
        outcome = engines.get(engine)
        if not isinstance(outcome, Mapping):
            raise ValueError(f"development decision is missing {engine}")
        if bool(outcome.get("pass", False)):
            passing.append((engine, float(outcome["worst_specification_fwer"])))
    if not passing:
        raise ValueError("no E1H specification passed development")
    if len(passing) == 1:
        return passing[0][0]
    ordered = sorted(passing, key=lambda item: (item[1], item[0]))
    if ordered[1][1] - ordered[0][1] >= float(fwer_tolerance):
        return ordered[0][0]
    return "E1H_AIC"


def validate_e1j_development_result(
    specification_summary: pd.DataFrame,
    true_hypothesis_summary: pd.DataFrame,
    decision: Mapping[str, object],
) -> str:
    """Validate the frozen E1J development identity before confirmation."""
    engine = "E1J_BIC_1125"
    engines = decision.get("engines")
    if not isinstance(engines, Mapping) or set(engines) != {engine}:
        raise ValueError("E1J development must contain exactly its frozen engine")
    outcome = engines[engine]
    if not isinstance(outcome, Mapping) or not bool(outcome.get("pass", False)):
        raise ValueError("E1J did not pass the generic development gates")
    group = specification_summary.loc[
        specification_summary["joint_inference_engine"].astype(str).eq(engine)
    ]
    if group.empty or not np.isclose(
        float(group["family_wise_error_rate"].max()), 0.047, rtol=0.0, atol=1e-12
    ):
        raise ValueError("E1J worst development FWER does not reproduce 0.047")
    a08 = true_hypothesis_summary.loc[
        true_hypothesis_summary["joint_inference_engine"].astype(str).eq(engine)
        & true_hypothesis_summary["scenario_id"].astype(str).eq("A08"),
        "rejection_rate",
    ]
    if len(a08) != 5 or not np.isclose(
        float(a08.min()), 0.848, rtol=0.0, atol=1e-12
    ):
        raise ValueError("E1J minimum A08 development power does not reproduce 0.848")
    return engine


def calibrate_e1j_standard_error_multiplier(
    e1h_development_results: pd.DataFrame,
    *,
    multiplier_grid: Sequence[float] = (
        1.0, 1.025, 1.05, 1.075, 1.1, 1.125,
        1.15, 1.175, 1.2, 1.225, 1.25,
    ),
) -> tuple[pd.DataFrame, float]:
    """Select the first frozen E1J safety multiplier on E1H-BIC development rows."""
    required = {
        "joint_inference_engine", "registered_task_idx", "observed_t",
        "alternative", "scenario_id", "analysis_specification", "replicate",
        "hypothesis_id", "true_effect",
    }
    if missing := sorted(required.difference(e1h_development_results.columns)):
        raise ValueError("E1J calibration input missing columns: " + ", ".join(missing))
    frame = e1h_development_results.loc[
        e1h_development_results["joint_inference_engine"].astype(str).eq("E1H_BIC")
    ].copy()
    if frame.empty or frame.duplicated(
        ["registered_task_idx", "hypothesis_id"]
    ).any():
        raise ValueError("E1J calibration requires unique E1H_BIC development rows")
    family_sizes = frame.groupby("registered_task_idx")["hypothesis_id"].nunique()
    if not family_sizes.eq(47).all():
        raise ValueError("E1J calibration requires complete 47-hypothesis families")
    grid = tuple(float(value) for value in multiplier_grid)
    if grid != (1.0, 1.025, 1.05, 1.075, 1.1, 1.125, 1.15, 1.175, 1.2, 1.225, 1.25):
        raise ValueError("E1J multiplier grid differs from the frozen grid")
    rows: list[dict[str, object]] = []
    for multiplier in grid:
        candidate = frame.copy()
        statistic = candidate["observed_t"].astype(float) / multiplier
        candidate["family_adjusted_p_value"] = np.where(
            candidate["alternative"].astype(str).eq("two-sided"),
            2.0 * normal_distribution.sf(np.abs(statistic)),
            normal_distribution.sf(statistic),
        )
        candidate["family_adjusted_p_value"] = candidate.groupby(
            "registered_task_idx", sort=False
        )["family_adjusted_p_value"].transform(research_stats.holm_adjusted_p_values)
        candidate["joint_inference_engine"] = "E1J_BIC_1125"
        specification, hypothesis, _ = evaluate_joint_inference_calibration(
            candidate, phase="development"
        )
        worst = float(specification["family_wise_error_rate"].max())
        a08 = hypothesis.loc[
            hypothesis["scenario_id"].astype(str).eq("A08"), "rejection_rate"
        ]
        minimum_power = float(a08.min()) if len(a08) == 5 else float("nan")
        rows.append(
            {
                "standard_error_multiplier": multiplier,
                "worst_specification_fwer": worst,
                "a08_minimum_per_hypothesis_power": minimum_power,
                "eligible": bool(worst <= 0.05 and minimum_power >= 0.80),
            }
        )
    summary = pd.DataFrame(rows)
    eligible = summary.loc[summary["eligible"]]
    if eligible.empty:
        raise ValueError("no frozen E1J multiplier satisfies development gates")
    return summary, float(eligible.iloc[0]["standard_error_multiplier"])


def _oracle_gaussian_raw_p_values(
    observed_effects: np.ndarray,
    *,
    oracle_se: float | np.ndarray,
    alternative: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(observed_effects, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("observed effects must be one-dimensional and finite")
    standard_errors = np.asarray(oracle_se, dtype=float)
    if not np.isfinite(standard_errors).all() or (standard_errors <= 0.0).any():
        raise ValueError("oracle standard error must be finite and positive")
    try:
        z_values = values / standard_errors
    except ValueError as error:
        raise ValueError("oracle standard errors do not match observed effects") from error
    if alternative == "greater":
        raw = normal_distribution.sf(z_values)
    elif alternative == "two-sided":
        raw = 2.0 * normal_distribution.sf(np.abs(z_values))
    else:
        raise ValueError("alternative must be greater or two-sided")
    return z_values, np.asarray(raw, dtype=float)


def oracle_gaussian_holm_family(
    observed_effects: pd.Series,
    *,
    temporal_dependence: str,
    day_count: int,
    alternative: str,
) -> pd.DataFrame:
    """Truth-known Gaussian marginal and Holm-adjusted family inference."""
    if not isinstance(observed_effects, pd.Series) or observed_effects.empty:
        raise ValueError("observed_effects must be a non-empty Series")
    if observed_effects.index.has_duplicates:
        raise ValueError("observed_effects index must be unique")
    oracle_se = sqrt(
        exact_gaussian_mean_variance(
            str(temporal_dependence), day_count=int(day_count)
        )["exact_mean_variance"]
    )
    z_values, raw = _oracle_gaussian_raw_p_values(
        observed_effects.to_numpy(dtype=float),
        oracle_se=oracle_se,
        alternative=str(alternative),
    )
    return pd.DataFrame(
        {
            "hypothesis_id": observed_effects.index.astype(str),
            "observed_effect": observed_effects.to_numpy(dtype=float),
            "oracle_se": oracle_se,
            "oracle_z": z_values,
            "oracle_raw_p_value": raw,
            "oracle_holm_adjusted_p_value": research_stats.holm_adjusted_p_values(raw),
        }
    )


def oracle_gaussian_bh_family(
    observed_effects: pd.Series,
    *,
    temporal_dependence: str,
    day_count: int,
    alternative: str,
) -> pd.DataFrame:
    """Truth-known Gaussian marginal and BH-adjusted family inference."""
    if not isinstance(observed_effects, pd.Series) or observed_effects.empty:
        raise ValueError("observed_effects must be a non-empty Series")
    if observed_effects.index.has_duplicates:
        raise ValueError("observed_effects index must be unique")
    oracle_se = sqrt(
        exact_gaussian_mean_variance(
            str(temporal_dependence), day_count=int(day_count)
        )["exact_mean_variance"]
    )
    z_values, raw = _oracle_gaussian_raw_p_values(
        observed_effects.to_numpy(dtype=float),
        oracle_se=oracle_se,
        alternative=str(alternative),
    )
    if not np.isfinite(raw).all():
        raise ValueError("oracle raw p-values must be finite")
    return pd.DataFrame(
        {
            "hypothesis_id": observed_effects.index.astype(str),
            "observed_effect": observed_effects.to_numpy(dtype=float),
            "oracle_se": oracle_se,
            "oracle_z": z_values,
            "oracle_raw_p_value": raw,
            "oracle_bh_q_value": research_stats.benjamini_hochberg_q_values(raw),
        }
    )


def evaluate_bh_fdr_variants(
    results: pd.DataFrame,
    *,
    dataset_id: str,
    scenario_temporal_dependence: Mapping[str, str] | None = None,
    day_count: int = 494,
    family_size: int = 47,
    alpha: float = 0.05,
    include_cross_scenario_summary: bool = True,
) -> BhFdrEvaluationArtifacts:
    """Evaluate frozen Gaussian variants by task-level BH FDR.

    FDR is the mean task-level false-discovery proportion within each frozen
    scenario and analysis specification. Scenarios are never pooled as if
    they shared one Bernoulli error probability.
    """
    required = {
        "registered_task_idx",
        "scenario_id",
        "analysis_specification",
        "replicate",
        "hypothesis_id",
        "observed_effect",
        "uncalibrated_standard_error",
        "alternative",
        "true_effect",
    }
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError("BH-FDR input missing columns: " + ", ".join(missing))
    if not str(dataset_id):
        raise ValueError("dataset_id must be non-empty")
    if int(day_count) <= 1 or int(family_size) <= 1:
        raise ValueError("day_count and family_size must exceed one")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")

    optional = {
        "expected_1125_raw_p_value", "reference_standard_error",
    }.intersection(results.columns)
    frame = results.loc[:, sorted(required | optional)].copy()
    if frame.isna().any().any():
        raise ValueError("BH-FDR identity and numeric inputs must not be missing")
    frame["scenario_id"] = frame["scenario_id"].astype(str)
    frame["analysis_specification"] = frame["analysis_specification"].astype(str)
    frame["hypothesis_id"] = frame["hypothesis_id"].astype(str)
    frame["alternative"] = frame["alternative"].astype(str)
    if not set(frame["alternative"]).issubset({"greater", "two-sided"}):
        raise ValueError("unsupported alternative in BH-FDR input")
    numeric_columns = ["observed_effect", "uncalibrated_standard_error", "true_effect"]
    if "reference_standard_error" in frame:
        numeric_columns.append("reference_standard_error")
    numeric = frame[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("BH-FDR numeric inputs must be finite")
    if (frame["uncalibrated_standard_error"].astype(float) <= 0.0).any():
        raise ValueError("uncalibrated standard errors must be positive")
    if (
        "reference_standard_error" in frame
        and (frame["reference_standard_error"].astype(float) <= 0.0).any()
    ):
        raise ValueError("reference standard errors must be positive")

    task_keys = [
        "registered_task_idx",
        "scenario_id",
        "analysis_specification",
        "replicate",
    ]
    if frame.duplicated(task_keys + ["hypothesis_id"]).any():
        raise ValueError("BH-FDR input contains duplicate task hypotheses")
    expected_ids = tuple(f"H{index:02d}" for index in range(1, int(family_size) + 1))
    task_identity = frame.groupby(task_keys, sort=False).agg(
        hypothesis_count=("hypothesis_id", "size"),
        hypothesis_ids=("hypothesis_id", lambda values: tuple(values)),
        alternative_count=("alternative", "nunique"),
    )
    invalid = task_identity.loc[
        task_identity["hypothesis_count"].ne(int(family_size))
        | ~task_identity["hypothesis_ids"].map(lambda value: value == expected_ids)
        | task_identity["alternative_count"].ne(1)
    ]
    if not invalid.empty:
        raise ValueError("BH-FDR requires complete, ordered fixed-size families")
    scenarios = set(frame["scenario_id"])
    has_reference = "reference_standard_error" in frame
    temporal_mapping = {
        str(key): str(value)
        for key, value in (scenario_temporal_dependence or {}).items()
    }
    if has_reference:
        if temporal_mapping:
            raise ValueError(
                "empirical reference standard errors cannot be combined with a temporal mapping"
            )
    elif scenarios != set(temporal_mapping):
        raise ValueError("scenario temporal-dependence mapping is incomplete or excessive")

    logical = np.where(
        frame["alternative"].eq("two-sided"),
        "two_sided_supplement",
        "right_tail_primary",
    )
    frame["analysis_family"] = logical
    if not frame.apply(
        lambda row: str(row["analysis_specification"]).startswith(
            f"{row['scenario_id']}__{row['analysis_family']}__"
        ),
        axis=1,
    ).all():
        raise ValueError("analysis specification disagrees with scenario or alternative")
    frame["is_true_alternative"] = np.where(
        frame["alternative"].eq("two-sided"),
        frame["true_effect"].astype(float).ne(0.0),
        frame["true_effect"].astype(float).gt(0.0),
    )

    variant_frames: list[pd.DataFrame] = []
    for variant, multiplier in (
        ("AR_BIC_1000_BH", 1.0),
        ("AR_BIC_1125_BH", 1.125),
    ):
        candidate = frame.copy()
        statistic = candidate["observed_effect"].astype(float) / (
            multiplier * candidate["uncalibrated_standard_error"].astype(float)
        )
        candidate["standard_error"] = (
            multiplier * candidate["uncalibrated_standard_error"].astype(float)
        )
        candidate["test_statistic"] = statistic
        candidate["raw_p_value"] = np.where(
            candidate["alternative"].eq("two-sided"),
            2.0 * normal_distribution.sf(np.abs(statistic)),
            normal_distribution.sf(statistic),
        )
        if multiplier == 1.125 and "expected_1125_raw_p_value" in candidate:
            expected = candidate["expected_1125_raw_p_value"].to_numpy(dtype=float)
            actual = candidate["raw_p_value"].to_numpy(dtype=float)
            if not np.isfinite(expected).all() or not np.allclose(
                actual, expected, atol=1e-12, rtol=0.0
            ):
                raise ValueError("recomputed 1.125 raw p-values differ from frozen values")
        candidate["method_variant"] = variant
        variant_frames.append(candidate)

    reference_frame = frame.copy()
    if has_reference:
        reference_frame["standard_error"] = reference_frame[
            "reference_standard_error"
        ].astype(float)
        reference_method = "MC_REFERENCE_BH"
    else:
        oracle_variance = {
            scenario: exact_gaussian_mean_variance(
                temporal_mapping[scenario], day_count=int(day_count)
            )["exact_mean_variance"]
            for scenario in sorted(scenarios)
        }
        reference_frame["standard_error"] = reference_frame["scenario_id"].map(
            {scenario: sqrt(value) for scenario, value in oracle_variance.items()}
        )
        reference_method = "ORACLE_BH"
    reference_frame["test_statistic"] = (
        reference_frame["observed_effect"].astype(float)
        / reference_frame["standard_error"]
    )
    reference_frame["raw_p_value"] = np.where(
        reference_frame["alternative"].eq("two-sided"),
        2.0 * normal_distribution.sf(np.abs(reference_frame["test_statistic"])),
        normal_distribution.sf(reference_frame["test_statistic"]),
    )
    reference_frame["method_variant"] = reference_method
    variant_frames.append(reference_frame)

    hypothesis = pd.concat(variant_frames, ignore_index=True)
    raw = hypothesis["raw_p_value"].to_numpy(dtype=float)
    if not np.isfinite(raw).all() or ((raw < 0.0) | (raw > 1.0)).any():
        raise ValueError("all fixed-family raw p-values must be finite and in [0, 1]")
    adjustment_keys = ["method_variant", *task_keys]
    hypothesis["bh_q_value"] = hypothesis.groupby(
        adjustment_keys, sort=False
    )["raw_p_value"].transform(research_stats.benjamini_hochberg_q_values)
    if hypothesis["bh_q_value"].isna().any():
        raise ValueError("BH adjustment must preserve the complete finite family")
    hypothesis["discovered"] = hypothesis["bh_q_value"].le(float(alpha))
    hypothesis["false_discovery"] = (
        hypothesis["discovered"] & ~hypothesis["is_true_alternative"]
    )
    hypothesis["true_discovery"] = (
        hypothesis["discovered"] & hypothesis["is_true_alternative"]
    )
    hypothesis["dataset_id"] = str(dataset_id)

    aggregate_keys = [
        "dataset_id",
        "method_variant",
        "registered_task_idx",
        "scenario_id",
        "analysis_family",
        "replicate",
    ]
    task = hypothesis.groupby(aggregate_keys, sort=True, as_index=False).agg(
        discovery_count=("discovered", "sum"),
        false_discovery_count=("false_discovery", "sum"),
        true_discovery_count=("true_discovery", "sum"),
        true_alternative_count=("is_true_alternative", "sum"),
    )
    task["false_discovery_proportion"] = task["false_discovery_count"] / task[
        "discovery_count"
    ].clip(lower=1)
    task["true_positive_rate"] = task["true_discovery_count"] / task[
        "true_alternative_count"
    ].replace(0, np.nan)
    task["any_true_discovery"] = task["true_discovery_count"].gt(0)
    task["any_false_discovery"] = task["false_discovery_count"].gt(0)

    scenario_keys = ["dataset_id", "method_variant", "scenario_id", "analysis_family"]
    scenario = task.groupby(scenario_keys, sort=True, as_index=False).agg(
        task_count=("registered_task_idx", "size"),
        fdr=("false_discovery_proportion", "mean"),
        fdp_standard_deviation=("false_discovery_proportion", "std"),
        mean_true_positive_rate=("true_positive_rate", "mean"),
        true_positive_rate_standard_deviation=("true_positive_rate", "std"),
        any_true_discovery_rate=("any_true_discovery", "mean"),
        any_false_discovery_rate=("any_false_discovery", "mean"),
        mean_discovery_count=("discovery_count", "mean"),
        mean_false_discovery_count=("false_discovery_count", "mean"),
        mean_true_discovery_count=("true_discovery_count", "mean"),
        discovery_count_q25=("discovery_count", lambda values: values.quantile(0.25)),
        discovery_count_median=("discovery_count", "median"),
        discovery_count_q75=("discovery_count", lambda values: values.quantile(0.75)),
        maximum_discovery_count=("discovery_count", "max"),
    )
    scenario["fdr_monte_carlo_standard_error"] = (
        scenario["fdp_standard_deviation"].fillna(0.0)
        / np.sqrt(scenario["task_count"].astype(float))
    )
    margin = 1.96 * scenario["fdr_monte_carlo_standard_error"]
    scenario["fdr_ci95_lower"] = (scenario["fdr"] - margin).clip(0.0, 1.0)
    scenario["fdr_ci95_upper"] = (scenario["fdr"] + margin).clip(0.0, 1.0)
    scenario["true_positive_rate_monte_carlo_standard_error"] = (
        scenario["true_positive_rate_standard_deviation"]
        / np.sqrt(scenario["task_count"].astype(float))
    )
    tpr_margin = 1.96 * scenario[
        "true_positive_rate_monte_carlo_standard_error"
    ]
    scenario["true_positive_rate_ci95_lower"] = (
        scenario["mean_true_positive_rate"] - tpr_margin
    ).clip(0.0, 1.0)
    scenario["true_positive_rate_ci95_upper"] = (
        scenario["mean_true_positive_rate"] + tpr_margin
    ).clip(0.0, 1.0)
    for column in ("any_true_discovery", "any_false_discovery"):
        rate_column = f"{column}_rate"
        standard_error = np.sqrt(
            scenario[rate_column] * (1.0 - scenario[rate_column])
            / scenario["task_count"].astype(float)
        )
        scenario[f"{rate_column}_ci95_lower"] = (
            scenario[rate_column] - 1.96 * standard_error
        ).clip(0.0, 1.0)
        scenario[f"{rate_column}_ci95_upper"] = (
            scenario[rate_column] + 1.96 * standard_error
        ).clip(0.0, 1.0)

    true_hypothesis = (
        hypothesis.loc[hypothesis["is_true_alternative"]]
        .groupby(
            [
                "dataset_id",
                "method_variant",
                "scenario_id",
                "analysis_family",
                "hypothesis_id",
                "true_effect",
            ],
            sort=True,
            as_index=False,
        )
        .agg(discovery_rate=("discovered", "mean"), task_count=("discovered", "size"))
    )

    calibration = hypothesis.copy()
    calibration["truth_class"] = np.where(
        calibration["is_true_alternative"], "true_alternative", "true_null"
    )
    calibration_keys = [
        "dataset_id", "method_variant", "scenario_id", "analysis_family",
        "truth_class",
    ]
    raw_p_value_calibration = calibration.groupby(
        calibration_keys, sort=True, as_index=False
    ).agg(
        observation_count=("raw_p_value", "size"),
        mean_raw_p_value=("raw_p_value", "mean"),
        raw_p_q01=("raw_p_value", lambda values: values.quantile(0.01)),
        raw_p_q05=("raw_p_value", lambda values: values.quantile(0.05)),
        raw_p_q10=("raw_p_value", lambda values: values.quantile(0.10)),
        raw_p_q25=("raw_p_value", lambda values: values.quantile(0.25)),
        raw_p_median=("raw_p_value", "median"),
        raw_p_q75=("raw_p_value", lambda values: values.quantile(0.75)),
        raw_p_q90=("raw_p_value", lambda values: values.quantile(0.90)),
        raw_p_q95=("raw_p_value", lambda values: values.quantile(0.95)),
        raw_p_q99=("raw_p_value", lambda values: values.quantile(0.99)),
        raw_p_le_0_01_rate=("raw_p_value", lambda values: values.le(0.01).mean()),
        raw_p_le_0_05_rate=("raw_p_value", lambda values: values.le(0.05).mean()),
        raw_p_le_0_10_rate=("raw_p_value", lambda values: values.le(0.10).mean()),
    )

    def _uniform_ks_distance(values: pd.Series) -> float:
        ordered = np.sort(values.to_numpy(dtype=float))
        count = ordered.size
        upper = np.arange(1, count + 1, dtype=float) / count - ordered
        lower = ordered - np.arange(0, count, dtype=float) / count
        return float(max(np.max(upper), np.max(lower)))

    null_ks = (
        calibration.loc[calibration["truth_class"].eq("true_null")]
        .groupby(calibration_keys, sort=True)["raw_p_value"]
        .apply(_uniform_ks_distance)
        .rename("null_uniform_ks_distance")
        .reset_index()
    )
    raw_p_value_calibration = raw_p_value_calibration.merge(
        null_ks, on=calibration_keys, how="left", validate="one_to_one"
    )

    if include_cross_scenario_summary:
        right_tail = scenario.loc[scenario["analysis_family"].eq("right_tail_primary")]
        cross = right_tail.groupby(
            ["dataset_id", "method_variant", "analysis_family"],
            sort=True,
            as_index=False,
        ).agg(
            scenario_count=("scenario_id", "nunique"),
            equal_weight_mean_fdr=("fdr", "mean"),
            equal_weight_mean_true_positive_rate=("mean_true_positive_rate", "mean"),
            equal_weight_any_true_discovery_rate=("any_true_discovery_rate", "mean"),
            equal_weight_any_false_discovery_rate=("any_false_discovery_rate", "mean"),
            equal_weight_mean_discovery_count=("mean_discovery_count", "mean"),
        )
        if not cross["scenario_count"].eq(len(scenarios)).all():
            raise ValueError("right-tail cross-scenario summary is incomplete")
    else:
        cross = pd.DataFrame()

    keep = [
        "dataset_id", "method_variant", "registered_task_idx", "scenario_id",
        "analysis_family", "replicate", "hypothesis_id", "alternative",
        "true_effect", "is_true_alternative", "observed_effect", "standard_error",
        "test_statistic", "raw_p_value", "bh_q_value", "discovered",
        "false_discovery", "true_discovery",
    ]
    return BhFdrEvaluationArtifacts(
        hypothesis_results=hypothesis.loc[:, keep].sort_values(
            ["method_variant", "registered_task_idx", "hypothesis_id"]
        ).reset_index(drop=True),
        task_summary=task,
        scenario_summary=scenario,
        true_hypothesis_summary=true_hypothesis,
        raw_p_value_calibration_summary=raw_p_value_calibration,
        cross_scenario_summary=cross,
    )


def summarize_randomized_true_hypotheses(
    hypothesis_results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize conditional discovery rates when truth identities vary by task."""
    required = {
        "dataset_id", "method_variant", "scenario_id", "analysis_family",
        "registered_task_idx", "hypothesis_id", "true_effect", "discovered",
        "is_true_alternative",
    }
    missing = sorted(required.difference(hypothesis_results.columns))
    if missing:
        raise ValueError("randomized-truth summary missing columns: " + ", ".join(missing))
    frame = hypothesis_results.loc[:, sorted(required)].copy()
    if frame.isna().any().any():
        raise ValueError("randomized-truth summary must be complete")
    if frame.duplicated(
        ["dataset_id", "method_variant", "scenario_id", "analysis_family",
         "registered_task_idx", "hypothesis_id"]
    ).any():
        raise ValueError("randomized-truth summary contains duplicate task hypotheses")
    active = frame.loc[frame["is_true_alternative"].astype(bool)].copy()
    if active.empty:
        return pd.DataFrame(
            columns=[
                "dataset_id", "method_variant", "scenario_id", "analysis_family",
                "hypothesis_id", "assigned_task_count", "conditional_discovery_rate",
                "mean_true_effect", "minimum_true_effect", "maximum_true_effect",
            ]
        )
    if (active["true_effect"].astype(float) <= 0.0).any():
        raise ValueError("right-tail true alternatives must have positive effects")
    keys = [
        "dataset_id", "method_variant", "scenario_id", "analysis_family",
        "hypothesis_id",
    ]
    return active.groupby(keys, sort=True, as_index=False).agg(
        assigned_task_count=("registered_task_idx", "size"),
        conditional_discovery_rate=("discovered", "mean"),
        mean_true_effect=("true_effect", "mean"),
        minimum_true_effect=("true_effect", "min"),
        maximum_true_effect=("true_effect", "max"),
    )


def summarize_empirical_bh_diagnostics(
    task_summary: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    scenario_manifest: pd.DataFrame,
) -> EmpiricalBhDiagnosticArtifacts:
    """Summarize paired method differences and registered block sensitivity."""
    task_required = {
        "dataset_id", "method_variant", "registered_task_idx", "scenario_id",
        "analysis_family", "replicate", "false_discovery_proportion",
        "true_positive_rate", "any_true_discovery", "any_false_discovery",
    }
    scenario_required = {
        "dataset_id", "method_variant", "scenario_id", "analysis_family",
        "fdr", "mean_true_positive_rate", "any_true_discovery_rate",
        "any_false_discovery_rate", "mean_discovery_count",
    }
    manifest_required = {
        "scenario_id", "block_length", "active_count", "shrinkage_multiplier", "role",
    }
    for name, frame, required in (
        ("task_summary", task_summary, task_required),
        ("scenario_summary", scenario_summary, scenario_required),
        ("scenario_manifest", scenario_manifest, manifest_required),
    ):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"{name} must be a non-empty DataFrame")
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} missing columns: " + ", ".join(missing))

    methods = tuple(sorted(task_summary["method_variant"].astype(str).unique()))
    if len(methods) < 2:
        raise ValueError("paired method diagnostics require at least two methods")
    identity = [
        "dataset_id", "registered_task_idx", "scenario_id", "analysis_family", "replicate",
    ]
    metrics = (
        "false_discovery_proportion", "true_positive_rate",
        "any_true_discovery", "any_false_discovery",
    )
    paired_rows: list[dict[str, object]] = []
    for scenario_id, scenario_tasks in task_summary.groupby("scenario_id", sort=True):
        for left_index, left_method in enumerate(methods):
            for right_method in methods[left_index + 1:]:
                left = scenario_tasks.loc[
                    scenario_tasks["method_variant"].astype(str).eq(left_method),
                    identity + list(metrics),
                ]
                right = scenario_tasks.loc[
                    scenario_tasks["method_variant"].astype(str).eq(right_method),
                    identity + list(metrics),
                ]
                merged = left.merge(
                    right, on=identity, how="inner", validate="one_to_one",
                    suffixes=("_left", "_right"),
                )
                if len(left) != len(right) or len(merged) != len(left):
                    raise ValueError("paired method comparison is incomplete")
                for metric in metrics:
                    differences = (
                        merged[f"{metric}_left"].astype(float)
                        - merged[f"{metric}_right"].astype(float)
                    ).dropna()
                    count = len(differences)
                    if count == 0:
                        continue
                    mean = float(differences.mean())
                    standard_error = (
                        float(differences.std(ddof=1) / np.sqrt(count)) if count > 1 else 0.0
                    )
                    paired_rows.append({
                        "scenario_id": str(scenario_id),
                        "left_method": left_method,
                        "right_method": right_method,
                        "metric": metric,
                        "paired_task_count": count,
                        "mean_left_minus_right": mean,
                        "ci95_lower": mean - 1.96 * standard_error,
                        "ci95_upper": mean + 1.96 * standard_error,
                    })

    merged_scenario = scenario_summary.merge(
        scenario_manifest.loc[:, sorted(manifest_required)],
        on="scenario_id", how="left", validate="many_to_one",
    )
    if merged_scenario["block_length"].isna().any():
        raise ValueError("scenario manifest does not cover every scenario summary row")
    sensitivity = merged_scenario.loc[
        merged_scenario["active_count"].eq(0)
        | (
            merged_scenario["active_count"].eq(17)
            & merged_scenario["shrinkage_multiplier"].eq(0.75)
        ),
        [
            "dataset_id", "method_variant", "scenario_id", "analysis_family", "role",
            "block_length", "active_count", "shrinkage_multiplier", "fdr",
            "mean_true_positive_rate", "any_true_discovery_rate",
            "any_false_discovery_rate", "mean_discovery_count",
        ],
    ].copy()
    expected_blocks = {7, 14, 28}
    for _, group in sensitivity.groupby(
        ["method_variant", "active_count", "shrinkage_multiplier"], dropna=False
    ):
        if set(group["block_length"].astype(int)) != expected_blocks:
            raise ValueError("block sensitivity does not contain 7/14/28-day rows")
    return EmpiricalBhDiagnosticArtifacts(
        paired_method_differences=pd.DataFrame(paired_rows),
        block_length_sensitivity=sensitivity.sort_values(
            ["active_count", "method_variant", "block_length"], kind="mergesort"
        ).reset_index(drop=True),
    )


def evaluate_realistic_effect_power(
    base_results: pd.DataFrame,
    scenario_specs: Sequence[Mapping[str, object]],
    *,
    day_count: int = 494,
    family_size: int = 47,
    expected_tasks_per_base: int = 2000,
    alpha: float = 0.05,
) -> RealisticEffectPowerArtifacts:
    """Evaluate a frozen paired realistic-effect grid without pooling scenarios."""
    required_spec = {
        "scenario_id", "base_scenario_id", "structure_id", "effect_label",
        "target_effect", "active_count", "temporal_dependence",
    }
    specs = [dict(spec) for spec in scenario_specs]
    if not specs:
        raise ValueError("realistic-effect scenario specs must not be empty")
    for spec in specs:
        missing = sorted(required_spec.difference(spec))
        if missing:
            raise ValueError("realistic-effect scenario spec missing: " + ", ".join(missing))
    metadata = pd.DataFrame(specs)
    if metadata["scenario_id"].astype(str).duplicated().any():
        raise ValueError("realistic-effect scenario ids must be unique")
    structure_counts = metadata.groupby("structure_id").agg(
        scenario_count=("scenario_id", "size"),
        effect_count=("target_effect", "nunique"),
        base_count=("base_scenario_id", "nunique"),
        active_count_count=("active_count", "nunique"),
        temporal_count=("temporal_dependence", "nunique"),
    )
    if (
        structure_counts["scenario_count"].ne(3).any()
        or structure_counts["effect_count"].ne(3).any()
        or structure_counts["base_count"].ne(1).any()
        or structure_counts["active_count_count"].ne(1).any()
        or structure_counts["temporal_count"].ne(1).any()
    ):
        raise ValueError("each realistic-effect structure requires three paired effect levels")

    counterfactual = retarget_additive_effect_scenarios(
        base_results,
        specs,
        family_size=int(family_size),
        expected_tasks_per_base=int(expected_tasks_per_base),
    )
    temporal = {
        str(spec["scenario_id"]): str(spec["temporal_dependence"])
        for spec in specs
    }
    bh = evaluate_bh_fdr_variants(
        counterfactual,
        dataset_id="realistic_effect_grid",
        scenario_temporal_dependence=temporal,
        day_count=int(day_count),
        family_size=int(family_size),
        alpha=float(alpha),
    )
    meta_columns = [
        "scenario_id", "base_scenario_id", "structure_id", "effect_label",
        "target_effect", "active_count", "temporal_dependence",
    ]
    scenario = bh.scenario_summary.merge(
        metadata[meta_columns], on="scenario_id", how="left", validate="many_to_one"
    )
    if scenario["target_effect"].isna().any():
        raise RuntimeError("realistic-effect scenario metadata failed to join")
    task = bh.task_summary.merge(
        metadata[meta_columns], on="scenario_id", how="left", validate="many_to_one"
    )

    def summarize_differences(
        left: pd.DataFrame,
        right: pd.DataFrame,
        keys: Sequence[str],
        *,
        contrast_type: str,
        left_label: str,
        right_label: str,
    ) -> dict[str, object]:
        paired = left[list(keys) + ["true_positive_rate"]].merge(
            right[list(keys) + ["true_positive_rate"]],
            on=list(keys), how="inner", validate="one_to_one", suffixes=("_left", "_right"),
        )
        if len(paired) != int(expected_tasks_per_base):
            raise ValueError("paired realistic-effect contrast is incomplete")
        differences = (
            paired["true_positive_rate_right"] - paired["true_positive_rate_left"]
        ).to_numpy(dtype=float)
        if not np.isfinite(differences).all():
            raise ValueError("paired realistic-effect contrast contains non-finite values")
        standard_error = float(np.std(differences, ddof=1) / np.sqrt(len(differences)))
        mean = float(np.mean(differences))
        return {
            "contrast_type": contrast_type,
            "left_label": left_label,
            "right_label": right_label,
            "task_count": int(len(differences)),
            "mean_true_positive_rate_difference": mean,
            "monte_carlo_standard_error": standard_error,
            "ci95_lower": max(-1.0, mean - 1.96 * standard_error),
            "ci95_upper": min(1.0, mean + 1.96 * standard_error),
        }

    effect_rows: list[dict[str, object]] = []
    for (structure_id, method), group in task.groupby(
        ["structure_id", "method_variant"], sort=True
    ):
        levels = sorted(group["target_effect"].astype(float).unique())
        if len(levels) != 3:
            raise ValueError("paired effect contrast requires exactly three effect levels")
        for lower, upper in zip(levels[:-1], levels[1:]):
            left = group.loc[group["target_effect"].astype(float).eq(lower)]
            right = group.loc[group["target_effect"].astype(float).eq(upper)]
            row = summarize_differences(
                left, right, ["replicate"], contrast_type="adjacent_effect",
                left_label=f"effect_{lower:.2f}", right_label=f"effect_{upper:.2f}",
            )
            row.update(
                {
                    "structure_id": structure_id,
                    "method_variant": method,
                    "lower_effect": lower,
                    "upper_effect": upper,
                }
            )
            effect_rows.append(row)

    method_rows: list[dict[str, object]] = []
    for scenario_id, group in task.groupby("scenario_id", sort=True):
        oracle = group.loc[group["method_variant"].eq("ORACLE_BH")]
        if oracle.empty:
            raise ValueError("paired method contrast requires ORACLE_BH")
        meta = metadata.loc[metadata["scenario_id"].astype(str).eq(str(scenario_id))].iloc[0]
        for method in ("AR_BIC_1000_BH", "AR_BIC_1125_BH"):
            practical = group.loc[group["method_variant"].eq(method)]
            row = summarize_differences(
                practical, oracle, ["replicate"], contrast_type="oracle_minus_practical",
                left_label=method, right_label="ORACLE_BH",
            )
            row.update(
                {
                    "scenario_id": scenario_id,
                    "structure_id": meta["structure_id"],
                    "target_effect": float(meta["target_effect"]),
                    "practical_method": method,
                }
            )
            method_rows.append(row)

    return RealisticEffectPowerArtifacts(
        counterfactual_inputs=counterfactual,
        bh_evaluation=bh,
        scenario_summary=scenario.sort_values(
            ["structure_id", "target_effect", "method_variant"]
        ).reset_index(drop=True),
        paired_effect_contrasts=pd.DataFrame(effect_rows),
        paired_method_contrasts=pd.DataFrame(method_rows),
    )


def evaluate_joint_inference_fault_decomposition(
    e1f_results: pd.DataFrame,
    e1s_results: pd.DataFrame,
    *,
    scenarios: Sequence[Mapping[str, object]],
    day_count: int,
    alpha: float = 0.05,
) -> JointInferenceFaultDecompositionArtifacts:
    """Decompose marginal, multiplicity, and oracle feasibility evidence.

    The oracle branch is diagnostic only: it uses the registered Gaussian
    process variance and Holm adjustment. Engine branches preserve their
    archived marginal and step-down p-values without recomputation.
    """
    if int(day_count) <= 1:
        raise ValueError("day_count must exceed one")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    required = {
        "joint_inference_engine",
        "scenario_id",
        "analysis_specification",
        "replicate",
        "inference_variant",
        "dependence_length",
        "hypothesis_id",
        "observed_effect",
        "bootstrap_se",
        "raw_one_sided_p_value",
        "raw_two_sided_p_value",
        "stepdown_max_t_adjusted_p_value",
        "alternative",
        "true_effect",
    }
    identity_keys = [
        "scenario_id",
        "replicate",
        "hypothesis_id",
        "alternative",
        "dependence_length",
    ]

    def validated(frame: pd.DataFrame, engine: str) -> pd.DataFrame:
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{engine} results missing columns: {', '.join(missing)}")
        result = frame.copy()
        if set(result["joint_inference_engine"].astype(str)) != {engine}:
            raise ValueError(f"{engine} results contain another engine")
        if result.duplicated(identity_keys).any():
            raise ValueError(f"{engine} results contain duplicate simulation identities")
        numeric = [
            "observed_effect",
            "bootstrap_se",
            "stepdown_max_t_adjusted_p_value",
            "true_effect",
        ]
        if not np.isfinite(result[numeric].to_numpy(dtype=float)).all():
            raise ValueError(f"{engine} results contain non-finite values")
        if (result["bootstrap_se"].astype(float) <= 0.0).any():
            raise ValueError(f"{engine} results contain non-positive standard errors")
        alternatives = set(result["alternative"].astype(str))
        if not alternatives.issubset({"greater", "two-sided"}):
            raise ValueError(f"{engine} results contain unsupported alternatives")
        return result.sort_values(identity_keys).reset_index(drop=True)

    e1f = validated(e1f_results, "E1F")
    e1s = validated(e1s_results, "E1S")
    if not e1f[identity_keys].equals(e1s[identity_keys]):
        raise ValueError("E1F and E1S simulation identity sets differ")
    for column in ("true_effect", "observed_effect"):
        if not np.allclose(
            e1f[column].to_numpy(dtype=float),
            e1s[column].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"E1F and E1S {column} values differ")

    scenario_rows = {str(row["id"]): dict(row) for row in scenarios}
    observed_scenarios = set(e1f["scenario_id"].astype(str))
    if observed_scenarios != set(scenario_rows):
        raise ValueError("archived results do not cover the registered scenarios")
    temporal_by_scenario: dict[str, str] = {}
    oracle_se_by_scenario: dict[str, float] = {}
    for scenario_id, scenario in scenario_rows.items():
        temporal = str(scenario["temporal_dependence"])
        temporal_by_scenario[scenario_id] = temporal
        oracle_se_by_scenario[scenario_id] = sqrt(
            exact_gaussian_mean_variance(
                temporal, day_count=int(day_count)
            )["exact_mean_variance"]
        )

    oracle = e1f[identity_keys + ["true_effect", "observed_effect"]].copy()
    oracle["joint_inference_engine"] = "ORACLE_HOLM"
    oracle["oracle_se"] = oracle["scenario_id"].map(oracle_se_by_scenario)
    oracle["oracle_z"] = np.nan
    oracle["oracle_raw_p_value"] = np.nan
    for alternative, row_index in oracle.groupby("alternative", sort=False).groups.items():
        z_values, raw = _oracle_gaussian_raw_p_values(
            oracle.loc[row_index, "observed_effect"].to_numpy(dtype=float),
            oracle_se=oracle.loc[row_index, "oracle_se"].to_numpy(dtype=float),
            alternative=str(alternative),
        )
        oracle.loc[row_index, "oracle_z"] = z_values
        oracle.loc[row_index, "oracle_raw_p_value"] = raw
    family_keys = ["scenario_id", "replicate", "alternative", "dependence_length"]
    oracle["oracle_holm_adjusted_p_value"] = (
        oracle.groupby(family_keys, sort=False)["oracle_raw_p_value"]
        .transform(lambda values: research_stats.holm_adjusted_p_values(values))
        .astype(float)
    )

    def truth_flags(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        two_sided = frame["alternative"].eq("two-sided")
        true_positive = pd.Series(
            np.where(
                two_sided,
                frame["true_effect"].astype(float).ne(0.0),
                frame["true_effect"].astype(float).gt(0.0),
            ),
            index=frame.index,
        )
        return true_positive.astype(bool), ~true_positive.astype(bool)

    def hypothesis_summary(
        frame: pd.DataFrame,
        *,
        engine: str,
        raw_column: str | None,
        adjusted_column: str,
    ) -> pd.DataFrame:
        working = frame.copy()
        true_positive, true_null = truth_flags(working)
        working["true_positive"] = true_positive
        working["true_null"] = true_null
        if raw_column is None:
            selected_raw = np.where(
                working["alternative"].eq("greater"),
                working["raw_one_sided_p_value"].astype(float),
                working["raw_two_sided_p_value"].astype(float),
            )
        else:
            selected_raw = working[raw_column].to_numpy(dtype=float)
        if (
            not np.isfinite(selected_raw).all()
            or (selected_raw < 0.0).any()
            or (selected_raw > 1.0).any()
        ):
            raise ValueError(f"{engine} selected raw p-values must lie in [0, 1]")
        adjusted = working[adjusted_column].to_numpy(dtype=float)
        if (
            not np.isfinite(adjusted).all()
            or (adjusted < 0.0).any()
            or (adjusted > 1.0).any()
        ):
            raise ValueError(f"{engine} adjusted p-values must lie in [0, 1]")
        working["raw_rejected"] = selected_raw <= float(alpha)
        working["adjusted_rejected"] = (
            working[adjusted_column].astype(float) <= float(alpha)
        )
        grouped = (
            working.groupby(
                ["scenario_id", "alternative", "dependence_length", "hypothesis_id"],
                sort=True,
                as_index=False,
            )
            .agg(
                replicate_count=("replicate", "size"),
                true_effect=("true_effect", "first"),
                true_positive=("true_positive", "first"),
                true_null=("true_null", "first"),
                marginal_detection_rate=("raw_rejected", "mean"),
                adjusted_detection_rate=("adjusted_rejected", "mean"),
            )
        )
        grouped.insert(0, "joint_inference_engine", engine)
        grouped["multiplicity_detection_rate_change"] = (
            grouped["adjusted_detection_rate"] - grouped["marginal_detection_rate"]
        )
        return grouped

    oracle_hypothesis = hypothesis_summary(
        oracle,
        engine="ORACLE_HOLM",
        raw_column="oracle_raw_p_value",
        adjusted_column="oracle_holm_adjusted_p_value",
    )
    engine_hypothesis = pd.concat(
        [
            hypothesis_summary(
                e1f,
                engine="E1F",
                raw_column=None,
                adjusted_column="stepdown_max_t_adjusted_p_value",
            ),
            hypothesis_summary(
                e1s,
                engine="E1S",
                raw_column=None,
                adjusted_column="stepdown_max_t_adjusted_p_value",
            ),
        ],
        ignore_index=True,
    )

    def specification_summary(
        frame: pd.DataFrame,
        *,
        engine: str,
        adjusted_column: str,
    ) -> pd.DataFrame:
        working = frame.copy()
        true_positive, true_null = truth_flags(working)
        working["true_positive"] = true_positive
        working["true_null"] = true_null
        working["adjusted_rejected"] = (
            working[adjusted_column].astype(float) <= float(alpha)
        )
        replicate = (
            working.groupby(family_keys, sort=True, as_index=False)
            .apply(
                lambda group: pd.Series(
                    {
                        "any_false_rejection": bool(
                            (group["adjusted_rejected"] & group["true_null"]).any()
                        ),
                        "any_true_rejection": bool(
                            (group["adjusted_rejected"] & group["true_positive"]).any()
                        ),
                    }
                ),
                include_groups=False,
            )
            .reset_index(drop=True)
        )
        rows: list[dict[str, object]] = []
        for keys, group in replicate.groupby(
            ["scenario_id", "alternative", "dependence_length"], sort=True
        ):
            scenario_id, alternative, dependence_length = keys
            trials = len(group)
            false_count = int(group["any_false_rejection"].sum())
            lower, upper = _clopper_pearson_bounds(
                false_count, trials, confidence=0.95, sides="two-sided"
            )
            rows.append(
                {
                    "joint_inference_engine": engine,
                    "scenario_id": scenario_id,
                    "alternative": alternative,
                    "dependence_length": int(dependence_length),
                    "replicate_count": trials,
                    "false_family_detection_count": false_count,
                    "family_wise_error_rate": false_count / trials,
                    "clopper_pearson_two_sided_95_low": lower,
                    "clopper_pearson_two_sided_95_high": upper,
                    "any_power": float(group["any_true_rejection"].mean()),
                }
            )
        return pd.DataFrame(rows)

    specification = pd.concat(
        [
            specification_summary(
                oracle,
                engine="ORACLE_HOLM",
                adjusted_column="oracle_holm_adjusted_p_value",
            ),
            specification_summary(
                e1f,
                engine="E1F",
                adjusted_column="stepdown_max_t_adjusted_p_value",
            ),
            specification_summary(
                e1s,
                engine="E1S",
                adjusted_column="stepdown_max_t_adjusted_p_value",
            ),
        ],
        ignore_index=True,
    )

    standard_error_rows: list[pd.DataFrame] = []
    for engine, frame in (("E1F", e1f), ("E1S", e1s)):
        working = frame.copy()
        working["oracle_se"] = working["scenario_id"].map(oracle_se_by_scenario)
        working["estimated_to_oracle_se_ratio"] = (
            working["bootstrap_se"].astype(float) / working["oracle_se"].astype(float)
        )
        diagnostic = (
            working.groupby(
                ["scenario_id", "alternative", "dependence_length"],
                sort=True,
                as_index=False,
            )
            .agg(
                oracle_se=("oracle_se", "first"),
                ratio_median=("estimated_to_oracle_se_ratio", "median"),
                ratio_q025=(
                    "estimated_to_oracle_se_ratio",
                    lambda values: float(values.quantile(0.025)),
                ),
                ratio_q975=(
                    "estimated_to_oracle_se_ratio",
                    lambda values: float(values.quantile(0.975)),
                ),
            )
        )
        diagnostic.insert(0, "joint_inference_engine", engine)
        standard_error_rows.append(diagnostic)
    standard_error_diagnostics = pd.concat(standard_error_rows, ignore_index=True)

    def engine_gate(engine: str, hypothesis: pd.DataFrame) -> dict[str, object]:
        specs = specification.loc[
            specification["joint_inference_engine"].eq(engine)
        ]
        calibrated = bool(
            specs["family_wise_error_rate"].le(0.060).all()
            and specs["clopper_pearson_two_sided_95_low"].le(0.05).all()
        )
        a08 = hypothesis.loc[
            hypothesis["scenario_id"].eq("A08") & hypothesis["true_positive"]
        ]
        if len(a08) != 5:
            raise ValueError(f"{engine} A08 true-hypothesis coverage changed")
        marginal_power_pass = bool(a08["marginal_detection_rate"].ge(0.80).all())
        adjusted_power_pass = bool(a08["adjusted_detection_rate"].ge(0.80).all())
        return {
            "calibration_pass": calibrated,
            "a08_marginal_each_power_pass": marginal_power_pass,
            "a08_adjusted_each_power_pass": adjusted_power_pass,
            "pass": calibrated and adjusted_power_pass,
            "worst_specification_fwer": float(
                specs["family_wise_error_rate"].max()
            ),
            "a08_marginal_power_min": float(a08["marginal_detection_rate"].min()),
            "a08_adjusted_power_min": float(a08["adjusted_detection_rate"].min()),
        }

    oracle_gate = engine_gate("ORACLE_HOLM", oracle_hypothesis)
    e1f_gate = engine_gate("E1F", engine_hypothesis.loc[
        engine_hypothesis["joint_inference_engine"].eq("E1F")
    ])
    e1s_gate = engine_gate("E1S", engine_hypothesis.loc[
        engine_hypothesis["joint_inference_engine"].eq("E1S")
    ])
    decision = {
        "alpha": float(alpha),
        "oracle_gate": oracle_gate,
        "frozen_gate_feasible": bool(oracle_gate["pass"]),
        "E1F": {
            **e1f_gate,
            "direct_failure_layer": (
                "false_positive_control"
                if e1f_gate["a08_adjusted_each_power_pass"]
                and not e1f_gate["calibration_pass"]
                else "mixed_or_other"
            ),
        },
        "E1S": {
            **e1s_gate,
            "direct_failure_layer": (
                "joint_adjustment_power_loss"
                if e1s_gate["a08_marginal_each_power_pass"]
                and not e1s_gate["a08_adjusted_each_power_pass"]
                else "marginal_or_other"
            ),
        },
        "status": "complete_diagnostic",
        "authorizes_successor_method_design": bool(oracle_gate["pass"]),
        "authorizes_confirmation_or_real_research": False,
    }
    return JointInferenceFaultDecompositionArtifacts(
        oracle_hypothesis=oracle_hypothesis,
        engine_hypothesis=engine_hypothesis,
        specification_summary=specification,
        standard_error_diagnostics=standard_error_diagnostics,
        decision=decision,
    )


def evaluate_confirmation_precision_match(
    main_results: pd.DataFrame,
    precision_results: pd.DataFrame,
    *,
    expected_precision_replicates_per_specification: int,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Check whether 10,000 draws change any frozen confirmation gate.

    The first registered confirmation replicates are replaced by their matched
    high-precision results, while all later replicates remain unchanged. The
    original and hybrid samples then pass through the identical confirmation
    gate. Individual p-values may differ; a specification or pooled pass/fail
    state may not.
    """
    keys = [
        "joint_inference_engine",
        "scenario_id",
        "replicate",
        "hypothesis_id",
        "alternative",
    ]
    required = set(keys) | {
        "analysis_specification",
        "true_effect",
        "stepdown_max_t_adjusted_p_value",
    }
    for name, frame in (("main", main_results), ("precision", precision_results)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} precision-match results missing columns: " + ", ".join(missing))
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} precision-match results contain duplicate keys")
    expected_count = int(expected_precision_replicates_per_specification)
    if expected_count <= 0:
        raise ValueError("expected precision replicate count must be positive")
    precision_support = (
        precision_results.groupby(
            ["joint_inference_engine", "scenario_id", "alternative"], sort=True
        )["replicate"]
        .agg(lambda values: set(values.astype(int)))
    )
    expected_support = set(range(expected_count))
    main_specifications = set(
        main_results[
            ["joint_inference_engine", "scenario_id", "alternative"]
        ].itertuples(index=False, name=None)
    )
    if set(precision_support.index) != main_specifications:
        raise ValueError("precision results do not cover every main specification")
    if precision_support.empty or not all(
        support == expected_support for support in precision_support
    ):
        raise ValueError("precision results do not cover the frozen leading replicates")
    main_keys = main_results[keys]
    precision_keys = precision_results[keys]
    if not precision_keys.merge(main_keys, on=keys, how="left", indicator=True)[
        "_merge"
    ].eq("both").all():
        raise ValueError("precision results are not a subset of main confirmation keys")

    replacement = precision_results[keys + ["stepdown_max_t_adjusted_p_value"]].rename(
        columns={
            "stepdown_max_t_adjusted_p_value": "precision_adjusted_p_value"
        }
    )
    hybrid = main_results.merge(replacement, on=keys, how="left", validate="one_to_one")
    replace_mask = hybrid["precision_adjusted_p_value"].notna()
    hybrid.loc[replace_mask, "stepdown_max_t_adjusted_p_value"] = hybrid.loc[
        replace_mask, "precision_adjusted_p_value"
    ]
    hybrid = hybrid.drop(columns="precision_adjusted_p_value")
    main_spec, _, main_decision = evaluate_joint_inference_calibration(
        main_results, phase="confirmation", alpha=alpha
    )
    hybrid_spec, _, hybrid_decision = evaluate_joint_inference_calibration(
        hybrid, phase="confirmation", alpha=alpha
    )
    compare_keys = [
        "joint_inference_engine",
        "scenario_id",
        "analysis_specification",
    ]
    columns = compare_keys + [
        "family_wise_error_rate",
        "clopper_pearson_one_sided_95_upper",
    ]
    comparison = main_spec[columns].merge(
        hybrid_spec[columns],
        on=compare_keys,
        suffixes=("_main_1999", "_hybrid_10000"),
        validate="one_to_one",
    )
    for suffix in ("main_1999", "hybrid_10000"):
        comparison[f"specification_pass_{suffix}"] = (
            comparison[f"family_wise_error_rate_{suffix}"].le(0.065)
            & comparison[f"clopper_pearson_one_sided_95_upper_{suffix}"].le(0.075)
        )
    comparison["specification_status_changed"] = comparison[
        "specification_pass_main_1999"
    ].ne(comparison["specification_pass_hybrid_10000"])
    engines: dict[str, object] = {}
    for engine in sorted(main_decision["engines"]):
        main_engine = main_decision["engines"][engine]
        hybrid_engine = hybrid_decision["engines"][engine]
        pooled_changed = bool(main_engine["pooled_pass"] != hybrid_engine["pooled_pass"])
        overall_changed = bool(main_engine["pass"] != hybrid_engine["pass"])
        specification_changed = bool(
            comparison.loc[
                comparison["joint_inference_engine"].eq(engine),
                "specification_status_changed",
            ].any()
        )
        engines[engine] = {
            "main_1999": main_engine,
            "hybrid_first_replicates_10000": hybrid_engine,
            "pooled_status_changed": pooled_changed,
            "specification_status_changed": specification_changed,
            "overall_status_changed": overall_changed,
            "pass": not (pooled_changed or specification_changed or overall_changed),
        }
    decision = {
        "expected_precision_replicates_per_specification": expected_count,
        "engines": engines,
        "pass": bool(engines) and all(bool(value["pass"]) for value in engines.values()),
    }
    return comparison, decision


def registered_joint_inference_development_tasks(
    prior_design: Mapping[str, object] | str | Path,
) -> tuple[dict[str, object], ...]:
    """Enumerate the frozen A01-A11 development datasets and specifications.

    Each old dataset is generated once. A10 is deliberately represented twice
    because its right-tailed primary analysis and two-sided supplement are
    separate specifications evaluated by both E1 and E2.
    """
    rows: list[dict[str, object]] = []
    for source in registered_layer_a_tasks(prior_design):
        alternatives = (
            ("greater", "right_tail_primary"),
            ("two-sided", "two_sided_supplement"),
        ) if str(source["scenario_id"]) == "A10" else (
            ("greater", "right_tail_primary"),
        )
        for alternative, suffix in alternatives:
            row = dict(source)
            row["development_task_idx"] = len(rows)
            row["development_task_id"] = (
                f"development_{len(rows):05d}__{source['scenario_id']}__"
                f"r{int(source['replicate']):04d}__{suffix}"
            )
            row["analysis_specification"] = (
                f"{source['scenario_id']}__{suffix}"
            )
            row["alternative"] = alternative
            rows.append(row)
    return tuple(rows)


def registered_e0_diagnostic_tasks(
    prior_design: Mapping[str, object] | str | Path,
) -> tuple[dict[str, object], ...]:
    """Return the paired A01-A05 datasets used by the E0 root diagnosis."""
    rows = []
    for source in registered_layer_a_tasks(prior_design):
        if str(source["scenario_id"]) not in {"A01", "A02", "A03", "A04", "A05"}:
            continue
        row = dict(source)
        row["diagnostic_task_idx"] = len(rows)
        row["diagnostic_task_id"] = (
            f"e0_diagnostic_{len(rows):05d}__{source['scenario_id']}__"
            f"r{int(source['replicate']):04d}"
        )
        row["analysis_specification"] = f"{source['scenario_id']}__right_tail_primary"
        row["alternative"] = "greater"
        rows.append(row)
    return tuple(rows)


def _registered_e1_failure_diagnostic_tasks(
    prior_design: Mapping[str, object] | str | Path,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for source in registered_joint_inference_development_tasks(prior_design):
        if str(source["scenario_id"]) not in {"A03", "A04", "A05"}:
            continue
        if str(source["alternative"]) != "greater":
            continue
        row = dict(source)
        row["e1_diagnostic_task_idx"] = len(rows)
        row["e1_diagnostic_task_id"] = (
            f"e1_failure_{len(rows):04d}__{source['scenario_id']}__"
            f"r{int(source['replicate']):04d}"
        )
        rows.append(row)
    if len(rows) != 3_000:
        raise RuntimeError("E1 failure diagnostic registry must contain 3,000 tasks")
    counts = pd.Series([row["scenario_id"] for row in rows]).value_counts().to_dict()
    if counts != {"A03": 1_000, "A04": 1_000, "A05": 1_000}:
        raise RuntimeError("E1 failure diagnostic scenarios are incomplete")
    return tuple(rows)


@lru_cache(maxsize=4)
def _registered_e1_failure_diagnostic_tasks_from_path(
    path: str,
) -> tuple[dict[str, object], ...]:
    return _registered_e1_failure_diagnostic_tasks(path)


def registered_e1_failure_diagnostic_tasks(
    prior_design: Mapping[str, object] | str | Path,
) -> tuple[dict[str, object], ...]:
    """Return the frozen A03-A05 right-tail development tasks."""
    if isinstance(prior_design, (str, Path)):
        return _registered_e1_failure_diagnostic_tasks_from_path(
            str(Path(prior_design).resolve())
        )
    return _registered_e1_failure_diagnostic_tasks(prior_design)


def _registered_group_labels(group_sizes: Sequence[int]) -> tuple[str, ...]:
    return tuple(
        f"G{group_index + 1:02d}"
        for group_index, size in enumerate(group_sizes)
        for _ in range(int(size))
    )


def decompose_e1_failure(
    inference_results: pd.DataFrame,
    bootstrap_max_statistics: Sequence[float] | np.ndarray | pd.Series,
    *,
    exact_true_mean_variance: float,
    theoretical_q95: float,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply the four registered E1 all-null counterfactuals."""
    required = {
        "hypothesis_id", "true_effect", "observed_effect", "bootstrap_se",
        "observed_t", "stepdown_max_t_adjusted_p_value",
    }
    if missing := sorted(required.difference(inference_results.columns)):
        raise ValueError("E1 failure input missing columns: " + ", ".join(missing))
    frame = inference_results.copy()
    if frame[list(required)].isna().any().any():
        raise ValueError("E1 failure input contains missing values")
    if frame["hypothesis_id"].astype(str).duplicated().any():
        raise ValueError("E1 failure input contains duplicate hypotheses")
    if not frame["true_effect"].astype(float).eq(0.0).all():
        raise ValueError("E1 failure decomposition requires an all-null family")
    exact_variance = float(exact_true_mean_variance)
    theory_q95 = float(theoretical_q95)
    if not np.isfinite(exact_variance) or exact_variance <= 0.0:
        raise ValueError("exact_true_mean_variance must be finite and positive")
    if not np.isfinite(theory_q95):
        raise ValueError("theoretical_q95 must be finite")
    bootstrap_max = np.asarray(bootstrap_max_statistics, dtype=float)
    if bootstrap_max.ndim != 1 or bootstrap_max.size == 0 or not np.isfinite(bootstrap_max).all():
        raise ValueError("bootstrap maximum statistics must be a finite vector")
    true_se = float(np.sqrt(exact_variance))
    frame["estimated_mean_variance"] = np.square(frame["bootstrap_se"].astype(float))
    frame["exact_true_mean_variance"] = exact_variance
    frame["estimated_to_exact_variance_ratio"] = (
        frame["estimated_mean_variance"] / exact_variance
    )
    frame["true_se_observed_t"] = frame["observed_effect"].astype(float) / true_se
    frame["formal_e1_final_rejected"] = (
        frame["stepdown_max_t_adjusted_p_value"].astype(float) <= float(alpha)
    )
    original_max = float(frame["observed_t"].astype(float).max())
    true_se_max = float(frame["true_se_observed_t"].max())
    bootstrap_q95 = float(np.quantile(bootstrap_max, 0.95, method="linear"))
    frame["original_rejected"] = frame["observed_t"].astype(float) > bootstrap_q95
    frame["true_se_only_rejected"] = (
        frame["true_se_observed_t"].astype(float) > bootstrap_q95
    )
    frame["theory_critical_only_rejected"] = (
        frame["observed_t"].astype(float) > theory_q95
    )
    frame["both_oracle_rejected"] = (
        frame["true_se_observed_t"].astype(float) > theory_q95
    )

    def rejected_ids(column: str) -> str:
        return ";".join(frame.loc[frame[column], "hypothesis_id"].astype(str))

    summary = {
        "exact_true_mean_variance": exact_variance,
        "exact_true_mean_standard_error": true_se,
        "median_estimated_to_exact_variance_ratio": float(
            frame["estimated_to_exact_variance_ratio"].median()
        ),
        "original_max_t": original_max,
        "true_se_max_t": true_se_max,
        "bootstrap_max_q90": float(np.quantile(bootstrap_max, 0.90, method="linear")),
        "bootstrap_max_q95": bootstrap_q95,
        "bootstrap_max_q99": float(np.quantile(bootstrap_max, 0.99, method="linear")),
        "theoretical_max_q95": theory_q95,
        "bootstrap_to_theoretical_q95_ratio": float(
            bootstrap_q95 / theory_q95
        ),
        "formal_e1_final_family_rejected": bool(
            frame["formal_e1_final_rejected"].any()
        ),
        "formal_e1_final_rejected_hypothesis_ids": rejected_ids(
            "formal_e1_final_rejected"
        ),
        "original_family_rejected": bool(frame["original_rejected"].any()),
        "true_se_only_family_rejected": bool(frame["true_se_only_rejected"].any()),
        "theory_critical_only_family_rejected": bool(
            frame["theory_critical_only_rejected"].any()
        ),
        "both_oracle_family_rejected": bool(frame["both_oracle_rejected"].any()),
        "original_rejected_hypothesis_ids": rejected_ids("original_rejected"),
        "true_se_only_rejected_hypothesis_ids": rejected_ids("true_se_only_rejected"),
        "theory_critical_only_rejected_hypothesis_ids": rejected_ids(
            "theory_critical_only_rejected"
        ),
        "both_oracle_rejected_hypothesis_ids": rejected_ids("both_oracle_rejected"),
    }
    return frame, summary


def run_e1_failure_diagnostic_task(
    prior_design: Mapping[str, object] | str | Path,
    task: Mapping[str, object],
    *,
    theoretical_q95: float,
) -> E1FailureDiagnosticTaskArtifacts:
    """Run one frozen E1 task and compute the registered failure decomposition."""
    manifest = (
        load_frozen_design(prior_design)
        if isinstance(prior_design, (str, Path))
        else _validate_frozen_design(prior_design)
    )
    tasks = registered_e1_failure_diagnostic_tasks(prior_design)
    task_idx = int(task["e1_diagnostic_task_idx"])
    if task_idx < 0 or task_idx >= len(tasks) or dict(task) != tasks[task_idx]:
        raise ValueError("E1 diagnostic task does not match the frozen registry")
    scenario = next(
        row for row in manifest["layer_a"]["scenarios"]
        if str(row["id"]) == str(task["scenario_id"])
    )
    if str(scenario["effect"]) != "all_null":
        raise ValueError("E1 failure decomposition requires an all-null scenario")
    day_count = int(manifest["layer_a"]["day_count"])
    dataset = generate_layer_a_dataset(
        scenario,
        day_count=day_count,
        group_sizes=manifest["layer_a"]["hypothesis_group_sizes"],
        seed=int(task["dataset_seed"]),
    )
    inference = infer_layer_a_dataset_with_engine_artifacts(
        dataset,
        engine="E1",
        dependence_length=14,
        n_bootstrap=999,
        seed=int(task["main_inference_seed"]),
        alternative="greater",
        production_equivalent=False,
    )
    result = inference.results.copy()
    if len(result) != 47 or not result["true_effect"].eq(0.0).all():
        raise RuntimeError("E1 diagnostic inference has the wrong hypothesis family")
    exact = exact_gaussian_mean_variance(
        str(scenario["temporal_dependence"]), day_count=day_count
    )
    bootstrap_max = inference.bootstrap_max_statistics[
        "bootstrap_max_test_statistic"
    ].to_numpy(dtype=float)
    if len(bootstrap_max) != 999 or not np.isfinite(bootstrap_max).all():
        raise RuntimeError("E1 diagnostic bootstrap maximum family is incomplete")
    result, decomposition = decompose_e1_failure(
        result,
        bootstrap_max,
        exact_true_mean_variance=float(exact["exact_mean_variance"]),
        theoretical_q95=float(theoretical_q95),
    )
    result.insert(0, "replicate", int(task["replicate"]))
    result.insert(0, "scenario_id", str(task["scenario_id"]))
    result.insert(0, "e1_diagnostic_task_idx", task_idx)
    replicate_summary = pd.DataFrame(
        [{
            "e1_diagnostic_task_idx": task_idx,
            "scenario_id": str(task["scenario_id"]),
            "replicate": int(task["replicate"]),
            "dataset_seed": int(task["dataset_seed"]),
            "inference_seed": int(task["main_inference_seed"]),
            "asymptotic_long_run_variance": exact["asymptotic_long_run_variance"],
            "asymptotic_mean_variance": exact["asymptotic_mean_variance"],
            **decomposition,
        }]
    )
    profile = dependence_profile(
        dataset.daily_values,
        _registered_group_labels(manifest["layer_a"]["hypothesis_group_sizes"]),
        max_lag=28,
        expected_day_count=494,
        expected_hypothesis_count=47,
    )
    temporal = profile.temporal_summary.copy()
    temporal.insert(0, "replicate", int(task["replicate"]))
    temporal.insert(0, "scenario_id", str(task["scenario_id"]))
    temporal.insert(0, "e1_diagnostic_task_idx", task_idx)
    cross = profile.cross_summary.copy()
    cross.insert(0, "replicate", int(task["replicate"]))
    cross.insert(0, "scenario_id", str(task["scenario_id"]))
    cross.insert(0, "e1_diagnostic_task_idx", task_idx)
    return E1FailureDiagnosticTaskArtifacts(
        hypothesis_diagnostics=result,
        replicate_summary=replicate_summary,
        temporal_profile=temporal,
        cross_profile=cross,
    )


def summarize_e1_failure_diagnostics(
    replicate_summaries: pd.DataFrame,
    temporal_profiles: pd.DataFrame,
    cross_profiles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate complete A03-A05 diagnostics without pooling hypotheses."""
    scenarios = ("A03", "A04", "A05")
    required_replicate = {
        "scenario_id", "replicate", "median_estimated_to_exact_variance_ratio",
        "bootstrap_to_theoretical_q95_ratio", "original_family_rejected",
        "true_se_only_family_rejected", "theory_critical_only_family_rejected",
        "both_oracle_family_rejected",
    }
    if missing := sorted(required_replicate.difference(replicate_summaries.columns)):
        raise ValueError("E1 replicate summaries missing columns: " + ", ".join(missing))
    replicate = replicate_summaries.copy()
    if replicate[list(required_replicate)].isna().any().any():
        raise ValueError("E1 replicate summaries contain missing values")
    if set(replicate["scenario_id"].astype(str)) != set(scenarios):
        raise ValueError("E1 replicate summaries have the wrong scenarios")
    if replicate.duplicated(["scenario_id", "replicate"]).any():
        raise ValueError("E1 replicate summaries contain duplicates")
    if not replicate.groupby("scenario_id")["replicate"].nunique().eq(1_000).all():
        raise ValueError("each E1 diagnostic scenario must contain 1,000 replicates")
    mechanism_rows: list[dict[str, object]] = []
    rejection_columns = (
        "original_family_rejected", "true_se_only_family_rejected",
        "theory_critical_only_family_rejected", "both_oracle_family_rejected",
    )
    for scenario_id in scenarios:
        group = replicate.loc[replicate["scenario_id"].astype(str) == scenario_id]
        row: dict[str, object] = {
            "scenario_id": scenario_id,
            "replicate_count": len(group),
        }
        for source in (
            "median_estimated_to_exact_variance_ratio",
            "bootstrap_to_theoretical_q95_ratio",
        ):
            values = group[source].astype(float)
            row[f"{source}_median"] = float(values.median())
            row[f"{source}_q025"] = float(values.quantile(0.025))
            row[f"{source}_q975"] = float(values.quantile(0.975))
        for source in rejection_columns:
            successes = int(group[source].astype(bool).sum())
            lower, upper = _clopper_pearson_bounds(
                successes, len(group), confidence=0.95, sides="two-sided"
            )
            row[f"{source}_count"] = successes
            row[f"{source}_rate"] = successes / len(group)
            row[f"{source}_ci_low"] = lower
            row[f"{source}_ci_high"] = upper
        row["conditional_variance_fix_rate_difference"] = (
            row["original_family_rejected_rate"]
            - row["true_se_only_family_rejected_rate"]
        )
        row["conditional_critical_fix_rate_difference"] = (
            row["original_family_rejected_rate"]
            - row["theory_critical_only_family_rejected_rate"]
        )
        mechanism_rows.append(row)

    required_temporal = {"scenario_id", "replicate", "lag_days", "autocorrelation_median"}
    if missing := sorted(required_temporal.difference(temporal_profiles.columns)):
        raise ValueError("E1 temporal profiles missing columns: " + ", ".join(missing))
    temporal = temporal_profiles.copy()
    if temporal.duplicated(["scenario_id", "replicate", "lag_days"]).any():
        raise ValueError("E1 temporal profiles contain duplicate lags")
    expected_temporal_rows = 3 * 1_000 * 28
    if len(temporal) != expected_temporal_rows:
        raise ValueError("E1 temporal profile coverage is incomplete")
    temporal_summary = (
        temporal.groupby(["scenario_id", "lag_days"], as_index=False, sort=True)
        .agg(
            profile_median=("autocorrelation_median", "median"),
            profile_q025=("autocorrelation_median", lambda value: value.quantile(0.025)),
            profile_q975=("autocorrelation_median", lambda value: value.quantile(0.975)),
        )
    )

    required_cross = {"scenario_id", "replicate", "relation", "correlation_mean"}
    if missing := sorted(required_cross.difference(cross_profiles.columns)):
        raise ValueError("E1 cross profiles missing columns: " + ", ".join(missing))
    cross = cross_profiles.copy()
    if cross.duplicated(["scenario_id", "replicate", "relation"]).any():
        raise ValueError("E1 cross profiles contain duplicate relations")
    if len(cross) != 3 * 1_000 * 2:
        raise ValueError("E1 cross profile coverage is incomplete")
    cross_summary = (
        cross.groupby(["scenario_id", "relation"], as_index=False, sort=True)
        .agg(
            profile_median=("correlation_mean", "median"),
            profile_q025=("correlation_mean", lambda value: value.quantile(0.025)),
            profile_q975=("correlation_mean", lambda value: value.quantile(0.975)),
        )
    )
    return pd.DataFrame(mechanism_rows), temporal_summary, cross_summary


def compare_real_and_simulated_dependence(
    real_temporal_summary: pd.DataFrame,
    real_cross_summary: pd.DataFrame,
    simulated_temporal_profile: pd.DataFrame,
    simulated_cross_profile: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare real and simulated dependence without constructing a weighted score."""
    required_real_temporal = {"lag_days", "autocorrelation_median"}
    required_sim_temporal = {"scenario_id", "lag_days", "profile_median"}
    required_real_cross = {"relation", "correlation_mean"}
    required_sim_cross = {"scenario_id", "relation", "profile_median"}
    for frame, required, label in (
        (real_temporal_summary, required_real_temporal, "real temporal"),
        (simulated_temporal_profile, required_sim_temporal, "simulated temporal"),
        (real_cross_summary, required_real_cross, "real cross"),
        (simulated_cross_profile, required_sim_cross, "simulated cross"),
    ):
        if missing := sorted(required.difference(frame.columns)):
            raise ValueError(f"{label} profile missing columns: " + ", ".join(missing))
    if set(real_temporal_summary["lag_days"].astype(int)) != set(range(1, 29)):
        raise ValueError("real temporal profile must cover lags 1 through 28")
    if set(real_cross_summary["relation"].astype(str)) != {"within", "between"}:
        raise ValueError("real cross profile must cover within and between relations")
    rows: list[dict[str, object]] = []
    for scenario_id in ("A03", "A04", "A05"):
        simulated_temporal = simulated_temporal_profile.loc[
            simulated_temporal_profile["scenario_id"].astype(str) == scenario_id,
            ["lag_days", "profile_median"],
        ]
        temporal_join = real_temporal_summary[
            ["lag_days", "autocorrelation_median"]
        ].merge(simulated_temporal, on="lag_days", validate="one_to_one")
        if len(temporal_join) != 28:
            raise ValueError("simulated temporal profile coverage is incomplete")
        temporal_distance = float(np.sqrt(np.mean(np.square(
            temporal_join["autocorrelation_median"].astype(float)
            - temporal_join["profile_median"].astype(float)
        ))))
        simulated_cross = simulated_cross_profile.loc[
            simulated_cross_profile["scenario_id"].astype(str) == scenario_id,
            ["relation", "profile_median"],
        ]
        cross_join = real_cross_summary[
            ["relation", "correlation_mean"]
        ].merge(simulated_cross, on="relation", validate="one_to_one")
        if len(cross_join) != 2:
            raise ValueError("simulated cross profile coverage is incomplete")
        cross_distance = float(np.sqrt(np.mean(np.square(
            cross_join["correlation_mean"].astype(float)
            - cross_join["profile_median"].astype(float)
        ))))
        rows.append({
            "scenario_id": scenario_id,
            "temporal_rmse": temporal_distance,
            "cross_rmse": cross_distance,
        })
    distances = pd.DataFrame(rows)
    temporal_min = float(distances["temporal_rmse"].min())
    cross_min = float(distances["cross_rmse"].min())
    temporal_mask = np.isclose(
        distances["temporal_rmse"].astype(float), temporal_min, rtol=0.0, atol=1e-15
    )
    cross_mask = np.isclose(
        distances["cross_rmse"].astype(float), cross_min, rtol=0.0, atol=1e-15
    )
    temporal_winners = distances.loc[temporal_mask, "scenario_id"].astype(str).tolist()
    cross_winners = distances.loc[cross_mask, "scenario_id"].astype(str).tolist()
    temporal_winner = (
        temporal_winners[0]
        if len(temporal_winners) == 1
        else "tie:" + ";".join(temporal_winners)
    )
    cross_winner = (
        cross_winners[0]
        if len(cross_winners) == 1
        else "tie:" + ";".join(cross_winners)
    )
    distances["is_temporal_winner"] = temporal_mask
    distances["is_cross_winner"] = cross_mask
    overall = (
        temporal_winners[0]
        if len(temporal_winners) == len(cross_winners) == 1
        and temporal_winners[0] == cross_winners[0]
        else "mixed"
    )
    decision = {
        "temporal_closest_scenario": temporal_winner,
        "cross_closest_scenario": cross_winner,
        "overall_closest_scenario": overall,
        "interpretation": (
            f"overall dependence is closest to {overall}"
            if overall != "mixed"
            else "temporal/cross winners differ or contain a tie"
        ),
    }
    return distances, decision


def _revision_scenario_specifications(
    prior_design: Mapping[str, object] | str | Path,
) -> tuple[tuple[dict[str, object], str, str], ...]:
    manifest = (
        load_frozen_design(prior_design)
        if isinstance(prior_design, (str, Path))
        else _validate_frozen_design(prior_design)
    )
    rows: list[tuple[dict[str, object], str, str]] = []
    for scenario in manifest["layer_a"]["scenarios"]:
        scenario_id = str(scenario["id"])
        rows.append((dict(scenario), "greater", "right_tail_primary"))
        if scenario_id == "A10":
            rows.append((dict(scenario), "two-sided", "two_sided_supplement"))
    return tuple(rows)


def registered_joint_inference_confirmation_tasks(
    prior_design: Mapping[str, object] | str | Path,
    *,
    master_seed: int,
    replicates_per_specification: int = 2_000,
) -> tuple[dict[str, object], ...]:
    """Enumerate fresh confirmation datasets in frozen specification order."""
    if int(master_seed) < 0:
        raise ValueError("confirmation master_seed must be non-negative")
    if int(replicates_per_specification) <= 0:
        raise ValueError("replicates_per_specification must be positive")
    specifications = _revision_scenario_specifications(prior_design)
    total = len(specifications) * int(replicates_per_specification)
    children = np.random.SeedSequence(int(master_seed)).spawn(total)
    rows: list[dict[str, object]] = []
    for scenario, alternative, suffix in specifications:
        scenario_id = str(scenario["id"])
        for replicate in range(int(replicates_per_specification)):
            child = children[len(rows)]
            dataset_child, main_child, precision_child = child.spawn(3)
            rows.append(
                {
                    "confirmation_task_idx": len(rows),
                    "confirmation_task_id": (
                        f"confirmation_{len(rows):05d}__{scenario_id}__"
                        f"r{replicate:04d}__{suffix}"
                    ),
                    "scenario_id": scenario_id,
                    "replicate": replicate,
                    "analysis_specification": f"{scenario_id}__{suffix}",
                    "alternative": alternative,
                    "dataset_seed": int(dataset_child.generate_state(1)[0]),
                    "main_inference_seed": int(main_child.generate_state(1)[0]),
                    "precision_inference_seed": int(
                        precision_child.generate_state(1)[0]
                    ),
                }
            )
    return tuple(rows)


def run_joint_inference_revision_task(
    prior_design: Mapping[str, object] | str | Path,
    task: Mapping[str, object],
    *,
    engines: Sequence[str],
    dependence_length: int,
    n_bootstrap: int,
    production_equivalent: bool = False,
) -> pd.DataFrame:
    """Generate one registered A dataset and apply the requested engines."""
    specifications = [
        {
            "inference_variant": f"{engine}_{int(dependence_length)}d_{int(n_bootstrap)}",
            "engine": str(engine),
            "dependence_length": int(dependence_length),
            "n_bootstrap": int(n_bootstrap),
            "inference_seed": int(task["main_inference_seed"]),
            "production_equivalent": bool(production_equivalent),
        }
        for engine in engines
    ]
    return run_joint_inference_revision_specifications(
        prior_design, task, specifications=specifications
    )


def run_joint_inference_revision_specifications(
    prior_design: Mapping[str, object] | str | Path,
    task: Mapping[str, object],
    *,
    specifications: Sequence[Mapping[str, object]],
) -> pd.DataFrame:
    """Generate one A dataset once and apply frozen inference specifications."""
    return run_joint_inference_revision_specification_artifacts(
        prior_design, task, specifications=specifications
    ).results


def run_joint_inference_revision_specification_artifacts(
    prior_design: Mapping[str, object] | str | Path,
    task: Mapping[str, object],
    *,
    specifications: Sequence[Mapping[str, object]],
) -> JointInferenceRevisionTaskArtifacts:
    """Run frozen specifications and retain their bootstrap-max distributions."""
    manifest = (
        load_frozen_design(prior_design)
        if isinstance(prior_design, (str, Path))
        else _validate_frozen_design(prior_design)
    )
    specs = tuple(dict(specification) for specification in specifications)
    if not specs:
        raise ValueError("joint-inference specifications must not be empty")
    required = {
        "inference_variant", "engine", "dependence_length", "n_bootstrap",
        "inference_seed", "production_equivalent",
    }
    if any(required.difference(specification) for specification in specs):
        raise ValueError("joint-inference specification is incomplete")
    variants = [str(specification["inference_variant"]) for specification in specs]
    if len(set(variants)) != len(variants):
        raise ValueError("joint-inference variants must be unique")
    if not {str(specification["engine"]) for specification in specs}.issubset(
        set(JOINT_INFERENCE_ENGINES)
    ):
        raise ValueError("joint-inference specification contains an unknown engine")
    scenario_id = str(task["scenario_id"])
    matches = [
        row for row in manifest["layer_a"]["scenarios"]
        if str(row["id"]) == scenario_id
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate Layer A scenario: {scenario_id}")
    dataset = generate_layer_a_dataset(
        matches[0],
        day_count=int(manifest["layer_a"]["day_count"]),
        group_sizes=manifest["layer_a"]["hypothesis_group_sizes"],
        seed=int(task["dataset_seed"]),
    )
    frames = []
    bootstrap_frames = []
    for specification in specs:
        engine = str(specification["engine"])
        inference = infer_layer_a_dataset_with_engine_artifacts(
            dataset,
            engine=engine,
            dependence_length=int(specification["dependence_length"]),
            n_bootstrap=int(specification["n_bootstrap"]),
            seed=int(specification["inference_seed"]),
            alternative=str(task["alternative"]),
            production_equivalent=bool(specification["production_equivalent"]),
        )
        frame = inference.results
        frame.insert(1, "scenario_id", scenario_id)
        frame.insert(
            2,
            "analysis_specification",
            f"{task['analysis_specification']}__{specification['inference_variant']}",
        )
        frame.insert(3, "replicate", int(task["replicate"]))
        frame.insert(4, "inference_variant", str(specification["inference_variant"]))
        frame.insert(5, "dependence_length", int(specification["dependence_length"]))
        if not frame["n_bootstrap"].astype(int).eq(
            int(specification["n_bootstrap"])
        ).all():
            raise RuntimeError("inference result disagrees with registered bootstrap count")
        frames.append(frame)
        bootstrap = inference.bootstrap_max_statistics.copy()
        bootstrap.insert(0, "scenario_id", scenario_id)
        bootstrap.insert(
            1,
            "analysis_specification",
            f"{task['analysis_specification']}__{specification['inference_variant']}",
        )
        bootstrap.insert(2, "replicate", int(task["replicate"]))
        bootstrap.insert(3, "joint_inference_engine", engine)
        bootstrap.insert(4, "inference_variant", str(specification["inference_variant"]))
        bootstrap.insert(5, "dependence_length", int(specification["dependence_length"]))
        bootstrap_frames.append(bootstrap)
    return JointInferenceRevisionTaskArtifacts(
        results=pd.concat(frames, ignore_index=True),
        bootstrap_max_statistics=pd.concat(bootstrap_frames, ignore_index=True),
    )


def summarize_e0_root_diagnostic(
    results: pd.DataFrame,
    bootstrap_max_statistics: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize E0 marginal, family, and unadjusted max-stat diagnostics."""
    required_results = {
        "scenario_id", "analysis_specification", "replicate", "hypothesis_id",
        "raw_one_sided_p_value", "stepdown_max_t_adjusted_p_value",
    }
    required_bootstrap = {
        "scenario_id", "analysis_specification", "replicate", "bootstrap_idx",
        "bootstrap_max_test_statistic",
    }
    if missing := sorted(required_results.difference(results.columns)):
        raise ValueError("E0 results missing columns: " + ", ".join(missing))
    if missing := sorted(required_bootstrap.difference(bootstrap_max_statistics.columns)):
        raise ValueError("E0 bootstrap maxima missing columns: " + ", ".join(missing))
    frame = results.copy()
    if frame[list(required_results)].isna().any().any():
        raise ValueError("E0 results contain missing diagnostic values")
    if set(frame.get("joint_inference_engine", pd.Series(dtype=str)).astype(str)) != {"E0"}:
        raise ValueError("E0 diagnostic contains a non-E0 engine")
    keys = ["scenario_id", "analysis_specification"]
    marginal = (
        frame.assign(
            marginal_rejected=frame["raw_one_sided_p_value"].astype(float) <= float(alpha)
        )
        .groupby(keys + ["hypothesis_id"], as_index=False, sort=True)
        .agg(
            replicate_count=("replicate", "nunique"),
            marginal_rejection_rate=("marginal_rejected", "mean"),
        )
    )
    family_by_replicate = (
        frame.assign(
            family_rejected=frame["stepdown_max_t_adjusted_p_value"].astype(float)
            <= float(alpha)
        )
        .groupby(keys + ["replicate"], as_index=False, sort=True)
        .agg(any_family_rejection=("family_rejected", "max"))
    )
    family = (
        family_by_replicate.groupby(keys, as_index=False, sort=True)
        .agg(
            replicate_count=("replicate", "nunique"),
            family_rejection_rate=("any_family_rejection", "mean"),
        )
    )
    maxima = bootstrap_max_statistics.copy()
    if maxima[list(required_bootstrap)].isna().any().any():
        raise ValueError("E0 bootstrap maxima contain missing values")
    maximum_summary = (
        maxima.groupby(keys, as_index=False, sort=True)
        .agg(
            bootstrap_draw_count=("bootstrap_max_test_statistic", "size"),
            max_stat_mean=("bootstrap_max_test_statistic", "mean"),
            max_stat_std=("bootstrap_max_test_statistic", "std"),
            max_stat_q50=("bootstrap_max_test_statistic", lambda value: value.quantile(0.50)),
            max_stat_q90=("bootstrap_max_test_statistic", lambda value: value.quantile(0.90)),
            max_stat_q95=("bootstrap_max_test_statistic", lambda value: value.quantile(0.95)),
            max_stat_q99=("bootstrap_max_test_statistic", lambda value: value.quantile(0.99)),
        )
    )
    return marginal, family, maximum_summary


def select_development_joint_inference_engine(
    decision: Mapping[str, object],
) -> str:
    """Apply the frozen E1/E2 development selection rule."""
    engines = decision.get("engines")
    if not isinstance(engines, Mapping):
        raise ValueError("development decision has no engine mapping")
    candidates = []
    for engine in ("E1", "E2"):
        outcome = engines.get(engine)
        if not isinstance(outcome, Mapping):
            raise ValueError(f"development decision is missing {engine}")
        if bool(outcome.get("pass")):
            candidates.append(
                (engine, float(outcome["worst_specification_fwer"]))
            )
    if not candidates:
        raise RuntimeError("neither E1 nor E2 passed the development gate")
    if len(candidates) == 1:
        return candidates[0][0]
    e1 = dict(candidates)["E1"]
    e2 = dict(candidates)["E2"]
    if abs(e1 - e2) < 0.01:
        return "E1"
    return "E1" if e1 < e2 else "E2"


def summarize_monte_carlo(
    results: pd.DataFrame, *, alpha: float = 0.05
) -> MonteCarloSummary:
    """Compute frozen error/power summaries with binomial Monte Carlo error."""
    required = {"scenario_id", "replicate", "hypothesis_id", "true_effect", "observed_effect", "stepdown_max_t_adjusted_p_value"}
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError("Monte Carlo results missing columns: " + ", ".join(missing))
    frame = results.copy()
    frame["rejected"] = frame["stepdown_max_t_adjusted_p_value"] <= float(alpha)
    alternatives = (
        frame["alternative"].astype(str)
        if "alternative" in frame.columns
        else pd.Series("greater", index=frame.index)
    )
    unsupported = sorted(set(alternatives).difference({"greater", "two-sided"}))
    if unsupported:
        raise ValueError(f"unsupported Monte Carlo alternatives: {unsupported}")
    frame["true_positive"] = np.where(
        alternatives.eq("two-sided"),
        frame["true_effect"] != 0.0,
        frame["true_effect"] > 0.0,
    )
    frame["true_null"] = ~frame["true_positive"]
    replicate = frame.groupby(["scenario_id", "replicate"], as_index=False).apply(
        lambda g: pd.Series({
            "any_false_rejection": bool((g["rejected"] & g["true_null"]).any()),
            "any_true_rejection": (
                bool((g["rejected"] & g["true_positive"]).any())
                if g["true_positive"].any() else np.nan
            ),
            "true_positive_rate": float(g.loc[g["true_positive"], "rejected"].mean()) if g["true_positive"].any() else np.nan,
        }), include_groups=False,
    ).reset_index(drop=True)
    rows = []
    for scenario_id, group in replicate.groupby("scenario_id", sort=True):
        n = len(group)
        row = {"scenario_id": scenario_id, "monte_carlo_repetitions": n}
        for source, target in (("any_false_rejection", "family_wise_error_rate"), ("any_true_rejection", "any_power"), ("true_positive_rate", "true_positive_rate")):
            estimate = float(group[source].mean()) if group[source].notna().any() else np.nan
            mcse = sqrt(estimate * (1.0 - estimate) / n) if np.isfinite(estimate) else np.nan
            row[target], row[f"{target}_mcse"] = estimate, mcse
            row[f"{target}_normal95_low"] = max(0.0, estimate - 1.96 * mcse) if np.isfinite(mcse) else np.nan
            row[f"{target}_normal95_high"] = min(1.0, estimate + 1.96 * mcse) if np.isfinite(mcse) else np.nan
        rows.append(row)
    hypothesis = frame.groupby(["scenario_id", "hypothesis_id"], as_index=False).agg(
        true_effect=("true_effect", "first"),
        rejection_rate=("rejected", "mean"),
        mean_estimate=("observed_effect", "mean"),
        estimate_bias=("observed_effect", lambda x: float(x.mean())),
        estimate_mse=("observed_effect", lambda x: float(np.mean(np.square(x)))),
        monte_carlo_repetitions=("replicate", "nunique"),
    )
    hypothesis["estimate_bias"] -= hypothesis["true_effect"]
    hypothesis["estimate_mse"] = frame.assign(
        squared_error=np.square(frame["observed_effect"] - frame["true_effect"])
    ).groupby(["scenario_id", "hypothesis_id"])["squared_error"].mean().to_numpy()
    return MonteCarloSummary(pd.DataFrame(rows), hypothesis)


def _registered_layer_a_tasks_from_manifest(
    manifest: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    layer_a = manifest["layer_a"]
    scenarios = layer_a["scenarios"]
    total = sum(int(row["monte_carlo"]) for row in scenarios)
    if total != int(layer_a["declared_workload"]["datasets"]):
        raise ValueError("Layer A registered task count disagrees with its workload")
    master = int(manifest["random_seeds"]["layer_a_master"])
    dataset_children = np.random.SeedSequence(master).spawn(total)
    production = layer_a["production_equivalent_subset"]
    sensitivity = layer_a["block_sensitivity_subset"]
    production_ids = set(production["scenario_ids"])
    sensitivity_ids = set(sensitivity["scenario_ids"])
    rows: list[dict[str, object]] = []
    child_idx = 0
    for scenario in scenarios:
        scenario_id = str(scenario["id"])
        for replicate in range(int(scenario["monte_carlo"])):
            seeds = dataset_children[child_idx].spawn(4)
            generated = [int(seed.generate_state(1)[0]) for seed in seeds]
            rows.append(
                {
                    "task_idx": child_idx,
                    "task_id": f"layer_a_{child_idx:04d}__{scenario_id}__r{replicate:04d}",
                    "scenario_id": scenario_id,
                    "replicate": replicate,
                    "dataset_seed": generated[0],
                    "main_inference_seed": generated[1],
                    "production_inference_seed": generated[2],
                    "sensitivity_inference_seed": generated[3],
                    "run_production_equivalent": bool(
                        scenario_id in production_ids
                        and replicate < int(production["first_seed_ordered_replicates"])
                    ),
                    "run_block_sensitivity": bool(
                        scenario_id in sensitivity_ids
                        and replicate < int(sensitivity["first_seed_ordered_replicates"])
                    ),
                }
            )
            child_idx += 1
    return tuple(rows)


@lru_cache(maxsize=4)
def _registered_layer_a_tasks_from_path(path: str) -> tuple[dict[str, object], ...]:
    return _registered_layer_a_tasks_from_manifest(load_frozen_design(path))


def registered_layer_a_tasks(
    design: Mapping[str, object] | str | Path,
) -> tuple[dict[str, object], ...]:
    """Enumerate all 8,000 frozen Layer-A datasets and their child seeds."""
    if isinstance(design, (str, Path)):
        return _registered_layer_a_tasks_from_path(str(Path(design).resolve()))
    return _registered_layer_a_tasks_from_manifest(_validate_frozen_design(design))


def run_registered_layer_a_task(
    design: Mapping[str, object] | str | Path,
    task: Mapping[str, object],
) -> pd.DataFrame:
    """Run one frozen Layer-A dataset and all pre-registered variants."""
    variants = ["main_14d_499"]
    if bool(task["run_production_equivalent"]):
        variants.append("production_14d_10000")
    if bool(task["run_block_sensitivity"]):
        variants.extend(("sensitivity_7d_499", "sensitivity_28d_499"))
    return pd.concat(
        [run_registered_layer_a_task_variant(design, task, variant) for variant in variants],
        ignore_index=True,
    )


def run_registered_layer_a_task_variant(
    design: Mapping[str, object] | str | Path,
    task: Mapping[str, object],
    analysis_variant: str,
) -> pd.DataFrame:
    """Run exactly one pre-registered Layer-A analysis variant."""
    manifest = (
        load_frozen_design(design)
        if isinstance(design, (str, Path))
        else _validate_frozen_design(design)
    )
    tasks = (
        registered_layer_a_tasks(design)
        if isinstance(design, (str, Path))
        else registered_layer_a_tasks(manifest)
    )
    task_idx = int(task["task_idx"])
    if task_idx < 0 or task_idx >= len(tasks) or dict(task) != tasks[task_idx]:
        raise ValueError("Layer A task does not match the frozen registry")
    scenario = next(
        row for row in manifest["layer_a"]["scenarios"]
        if str(row["id"]) == str(task["scenario_id"])
    )
    dataset = generate_layer_a_dataset(
        scenario,
        day_count=int(manifest["layer_a"]["day_count"]),
        group_sizes=manifest["layer_a"]["hypothesis_group_sizes"],
        seed=int(task["dataset_seed"]),
    )
    if analysis_variant == "main_14d_499":
        frame = infer_layer_a_dataset(
            dataset,
            block_length=int(manifest["layer_a"]["main_run"]["block_length_days"]),
            n_bootstrap=int(manifest["layer_a"]["main_run"]["bootstrap_repetitions"]),
            seed=int(task["main_inference_seed"]),
        )
    elif analysis_variant == "a10_two_sided_14d_499":
        if str(task["scenario_id"]) != "A10":
            raise ValueError("the two-sided Layer A variant is registered only for A10")
        frame = infer_layer_a_dataset(
            dataset,
            block_length=int(manifest["layer_a"]["main_run"]["block_length_days"]),
            n_bootstrap=int(manifest["layer_a"]["main_run"]["bootstrap_repetitions"]),
            seed=int(task["main_inference_seed"]),
            alternative="two-sided",
        )
    elif analysis_variant == "production_14d_10000":
        if not bool(task["run_production_equivalent"]):
            raise ValueError("Layer A task is not registered for production-equivalent inference")
        contract = manifest["layer_a"]["production_equivalent_subset"]
        frame = infer_layer_a_dataset(
            dataset,
            block_length=int(contract["block_length_days"]),
            n_bootstrap=int(contract["bootstrap_repetitions"]),
            seed=int(task["production_inference_seed"]),
            production_equivalent=True,
        )
    elif analysis_variant in {"sensitivity_7d_499", "sensitivity_28d_499"}:
        if not bool(task["run_block_sensitivity"]):
            raise ValueError("Layer A task is not registered for block sensitivity")
        contract = manifest["layer_a"]["block_sensitivity_subset"]
        block_length = int(analysis_variant.split("_", 2)[1][:-1])
        registered_lengths = tuple(int(value) for value in contract["block_lengths_days"])
        if block_length not in registered_lengths:
            raise ValueError("Layer A sensitivity block is not registered")
        sensitivity_children = np.random.SeedSequence(
            int(task["sensitivity_inference_seed"])
        ).spawn(len(contract["block_lengths_days"]))
        child = sensitivity_children[registered_lengths.index(block_length)]
        frame = infer_layer_a_dataset(
            dataset,
            block_length=block_length,
            n_bootstrap=int(contract["bootstrap_repetitions"]),
            seed=int(child.generate_state(1)[0]),
        )
    else:
        raise ValueError(f"unsupported Layer A analysis variant: {analysis_variant}")
    enriched = frame.copy()
    enriched.insert(0, "analysis_variant", analysis_variant)
    enriched.insert(0, "replicate", int(task["replicate"]))
    enriched.insert(0, "scenario_id", str(task["scenario_id"]))
    enriched.insert(0, "task_idx", task_idx)
    return enriched


def summarize_registered_layer_a_results(results: pd.DataFrame) -> MonteCarloSummary:
    """Summarize each registered Layer-A variant without mixing estimands."""
    required = {"analysis_variant", "scenario_id", "replicate"}
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError("Layer A results missing columns: " + ", ".join(missing))
    frame = results.copy()
    frame["analysis_id"] = frame["scenario_id"].astype(str) + "__" + frame[
        "analysis_variant"
    ].astype(str)
    frame["registered_scenario_id"] = frame["scenario_id"]
    frame["scenario_id"] = frame["analysis_id"]
    summary = summarize_monte_carlo(frame)
    scenario = summary.scenario_summary.rename(columns={"scenario_id": "analysis_id"})
    scenario[["scenario_id", "analysis_variant"]] = scenario["analysis_id"].str.rsplit(
        "__", n=1, expand=True
    )
    hypothesis = summary.hypothesis_summary.rename(
        columns={"scenario_id": "analysis_id"}
    )
    hypothesis[["scenario_id", "analysis_variant"]] = hypothesis[
        "analysis_id"
    ].str.rsplit("__", n=1, expand=True)
    return MonteCarloSummary(scenario, hypothesis)


def summarize_matched_registered_layer_a_results(
    results: pd.DataFrame,
    *,
    scenario_ids: Sequence[str],
    analysis_variants: Sequence[str],
    replicate_count: int,
) -> MonteCarloSummary:
    """Summarize variants only after proving identical replicate support."""
    if replicate_count <= 0:
        raise ValueError("replicate_count must be positive")
    required = {"scenario_id", "analysis_variant", "replicate"}
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError("Layer A results missing columns: " + ", ".join(missing))
    scenarios = tuple(str(value) for value in scenario_ids)
    variants = tuple(str(value) for value in analysis_variants)
    if not scenarios or not variants:
        raise ValueError("matched Layer A comparison requires scenarios and variants")
    selected = results.loc[
        results["scenario_id"].astype(str).isin(scenarios)
        & results["analysis_variant"].astype(str).isin(variants)
        & (results["replicate"].astype(int) < int(replicate_count))
    ].copy()
    expected_replicates = set(range(int(replicate_count)))
    for scenario_id in scenarios:
        supports = []
        for variant in variants:
            cell = selected.loc[
                selected["scenario_id"].astype(str).eq(scenario_id)
                & selected["analysis_variant"].astype(str).eq(variant)
            ]
            support = set(cell["replicate"].astype(int).unique())
            if support != expected_replicates:
                raise ValueError(
                    "matched Layer A comparison has incomplete replicate support: "
                    f"{scenario_id}/{variant}"
                )
            supports.append(support)
        if any(support != supports[0] for support in supports[1:]):
            raise ValueError(
                f"matched Layer A comparison support differs for {scenario_id}"
            )
    return summarize_registered_layer_a_results(selected)


def shared_sparse_low_order_nonlinearity(x: np.ndarray) -> np.ndarray:
    """Frozen B06 shared price-volume function."""
    values = np.asarray(x, dtype=float)
    if values.shape[-1] < 6:
        raise ValueError("B06 requires at least six features")
    return (
        (values[..., 0] ** 2 - 1.0) / sqrt(2.0)
        + 0.8 * (values[..., 1] ** 2 - 1.0) / sqrt(2.0)
        + 0.6 * values[..., 2] * values[..., 3]
        + 0.4 * values[..., 4] * values[..., 5]
    ) / sqrt(2.16)


def _stationary_panel_process(
    rng: np.random.Generator,
    *,
    hour_count: int,
    object_count: int,
    component_count: int,
    half_life_hours: float,
    common_share: float,
) -> np.ndarray:
    """Generate stationary Gaussian AR panels with a cross-sectional common part."""
    phi = 2.0 ** (-1.0 / float(half_life_hours))
    innovation_scale = sqrt(1.0 - phi * phi)
    common = np.empty((hour_count, component_count), dtype=float)
    specific = np.empty((hour_count, object_count, component_count), dtype=float)
    common[0] = rng.standard_normal(component_count)
    specific[0] = rng.standard_normal((object_count, component_count))
    for index in range(1, hour_count):
        common[index] = phi * common[index - 1] + innovation_scale * rng.standard_normal(
            component_count
        )
        specific[index] = phi * specific[index - 1] + innovation_scale * rng.standard_normal(
            (object_count, component_count)
        )
    return sqrt(float(common_share)) * common[:, None, :] + sqrt(
        1.0 - float(common_share)
    ) * specific


def _feature_function(name: str, values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if name == "linear":
        return values @ weights
    if name == "tree":
        return (
            0.7 * np.sign(values[..., 0])
            + 0.5 * np.sign(values[..., 1] * values[..., 2])
            + 0.3 * np.sign(values[..., 3])
        ) / sqrt(0.83)
    if name == "smooth":
        sine_scale = sqrt((1.0 - np.exp(-2.0)) / 2.0)
        return (
            np.sin(values[..., 0]) / sine_scale
            + 0.5 * values[..., 1] * values[..., 2]
            + 0.3 * (values[..., 3] ** 2 - 1.0) / sqrt(2.0)
        ) / sqrt(1.34)
    raise ValueError(f"unsupported feature function: {name}")


def _layer_b_scenario(
    design: Mapping[str, object],
    scenario_id: str,
    *,
    base_state: str | None,
    beta: float | None,
    q_x: float | None,
) -> dict[str, object]:
    scenario = str(scenario_id)
    if scenario == "B07":
        scenario = "B07-M"
    simple = {
        "B01": ("structural_null", "linear", "linear"),
        "B02": ("structural_null", "tree", "tree"),
        "B03": ("structural_null", "smooth", "smooth"),
        "B04": ("structural_null", "linear", "smooth"),
        "B05": ("structural_null", "smooth", "linear"),
        "B06": (
            "structural_null_shared_plausible_nonlinearity",
            "linear",
            "linear",
        ),
        "B07-M": ("positive_extra_information", "linear", "linear"),
    }
    if scenario in simple:
        truth, signal_function, outcome_function = simple[scenario]
    elif scenario in {"B08", "B09", "B10"}:
        inherited = "B01" if base_state is None else str(base_state)
        if inherited not in {"B01", "B07-M", "B07"}:
            raise ValueError(f"{scenario} base_state must be B01 or B07-M")
        truth, signal_function, outcome_function = simple[
            "B07-M" if inherited == "B07" else inherited
        ]
    else:
        raise ValueError(f"unsupported Layer B scenario: {scenario_id}")
    positive = truth == "positive_extra_information"
    resolved_beta = 0.2 if positive and beta is None else 0.0 if not positive else float(beta)
    resolved_q_x = 0.4 if q_x is None else float(q_x)
    allowed_beta = set(float(value) for value in design["dgp"]["positive_extra_information_beta"])
    allowed_q_x = set(float(value) for value in design["dgp"]["price_volume_signal_share"])
    if positive and resolved_beta not in allowed_beta:
        raise ValueError("beta is outside the frozen B07 grid")
    if resolved_q_x not in allowed_q_x:
        raise ValueError("q_x is outside the frozen grid")
    return {
        "scenario_id": scenario,
        "truth": truth,
        "signal_function": signal_function,
        "outcome_function": outcome_function,
        "beta": resolved_beta,
        "q_x": resolved_q_x,
        "missing": scenario == "B09",
        "aliases": scenario == "B10",
        "stress": scenario == "B08",
        "shared_nonlinearity": scenario == "B06",
    }


def _simulation_parameters(
    design: Mapping[str, object], test_overrides: Mapping[str, object] | None
) -> dict[str, object]:
    application = design["application_contract"]
    walk = application["walk_forward"]
    parameters: dict[str, object] = {
        "day_count": int(application["total_days"]),
        "object_count": int(application["object_count_main"]),
        "feature_count": int(application["feature_count"]),
        "train_days": int(walk["train_days"]),
        "embargo_days": int(walk["embargo_days"]),
        "test_days": int(walk["test_days"]),
        "step_days": int(walk["step_days"]),
        "min_cross_section": int(application["minimum_cross_section"]),
        "block_length": int(design["inference"]["primary_block_length_days"]),
        "n_bootstrap": int(design["layer_b"]["main_bootstrap_repetitions"]),
        "alpha": float(design["inference"]["family_alpha"]),
        "alpha_grid": tuple(float(value) for value in design["registered_models"]["ridge_alpha"]),
        "model_specs": None,
        "allow_model_subset": False,
        "fit_workers": 16,
        "mcar_probability": 0.05,
        "missing_gap_objects": 2,
        "missing_gap_days": 90,
    }
    if test_overrides is not None:
        unknown = sorted(set(test_overrides).difference(parameters))
        if unknown:
            raise ValueError("unknown test override(s): " + ", ".join(unknown))
        parameters.update(test_overrides)
    if int(parameters["feature_count"]) < 6:
        raise ValueError("Layer B/C simulation requires at least six features")
    if int(parameters["object_count"]) < int(parameters["min_cross_section"]):
        raise ValueError("object_count is below min_cross_section")
    if int(parameters["n_bootstrap"]) < 499:
        raise ValueError("simulation bootstrap repetitions must be at least 499")
    return parameters


def _generate_layer_b_dataset(
    design: Mapping[str, object],
    scenario: Mapping[str, object],
    *,
    replicate: int,
    parameters: Mapping[str, object],
    master_seed: int,
) -> LayerBDataset:
    day_count = int(parameters["day_count"])
    object_count = int(parameters["object_count"])
    feature_count = int(parameters["feature_count"])
    burn = int(design["dgp"]["burn_in_hours_discarded"])
    hour_count = burn + day_count * 24
    seed_children = np.random.SeedSequence(int(master_seed)).spawn(int(replicate) + 3)
    weight_rng = np.random.Generator(np.random.PCG64DXSM(seed_children[0]))
    rng = np.random.Generator(np.random.PCG64DXSM(seed_children[int(replicate) + 1]))
    weights = weight_rng.standard_normal((5, feature_count))
    weights /= np.linalg.norm(weights, axis=1, keepdims=True)

    dgp = design["dgp"]
    shares = dict(dgp["cross_section_common_share_base"])
    half_lives = dict(dgp["hourly_half_life"])
    if bool(scenario["stress"]):
        shares = dict(dgp["B08"]["cross_section_common_share"])
        multiplier = float(dgp["B08"]["hourly_half_life_multiplier"])
        half_lives = {key: float(value) * multiplier for key, value in half_lives.items()}
    x = _stationary_panel_process(
        rng,
        hour_count=hour_count,
        object_count=object_count,
        component_count=feature_count,
        half_life_hours=float(half_lives["X"]),
        common_share=float(shares["X"]),
    )
    z = _stationary_panel_process(
        rng,
        hour_count=hour_count,
        object_count=object_count,
        component_count=4,
        half_life_hours=float(half_lives["Z"]),
        common_share=float(shares["Z"]),
    )
    signal_noise = _stationary_panel_process(
        rng,
        hour_count=hour_count,
        object_count=object_count,
        component_count=4,
        half_life_hours=float(half_lives["signal_noise"]),
        common_share=float(shares["signal_noise"]),
    )
    if bool(scenario["aliases"]):
        group_noise = _stationary_panel_process(
            rng,
            hour_count=hour_count,
            object_count=object_count,
            component_count=4,
            half_life_hours=float(half_lives["signal_noise"]),
            common_share=float(shares["signal_noise"]),
        )
        signal_noise = (group_noise + signal_noise) / sqrt(2.0)
    outcome_noise = _stationary_panel_process(
        rng,
        hour_count=hour_count,
        object_count=object_count,
        component_count=1,
        half_life_hours=float(half_lives["outcome_noise"]),
        common_share=float(shares["outcome_noise"]),
    )[..., 0]

    q_x = float(scenario["q_x"])
    beta = float(scenario["beta"])
    signal_function = str(scenario["signal_function"])
    outcome_function = str(scenario["outcome_function"])
    signal_price = np.stack(
        [_feature_function(signal_function, x, weights[group]) for group in range(4)],
        axis=-1,
    )
    outcome_price = _feature_function(outcome_function, x, weights[4])
    active_beta = np.asarray([beta, beta, 0.0, 0.0], dtype=float)
    if bool(scenario["shared_nonlinearity"]):
        shared_q = shared_sparse_low_order_nonlinearity(x)
        signal = (
            sqrt(q_x) * signal_price
            + sqrt(0.20) * shared_q[..., None]
            + sqrt(0.80 - q_x) * signal_noise
        )
        outcome = (
            sqrt(0.40) * outcome_price
            + sqrt(0.20) * shared_q
            + sqrt(0.40) * outcome_noise
        )
        oracle_signal_mean = sqrt(q_x) * signal_price + sqrt(0.20) * shared_q[..., None]
        oracle_outcome_mean = sqrt(0.40) * outcome_price + sqrt(0.20) * shared_q
    else:
        signal = (
            sqrt(q_x) * signal_price
            + sqrt(0.20) * z
            + sqrt(0.80 - q_x) * signal_noise
        )
        outcome = (
            sqrt(0.40) * outcome_price
            + np.sum(z * active_beta[None, None, :], axis=-1)
            + sqrt(0.60 - float(np.square(active_beta).sum())) * outcome_noise
        )
        oracle_signal_mean = sqrt(q_x) * signal_price
        oracle_outcome_mean = sqrt(0.40) * outcome_price

    x = x[burn:]
    signal = signal[burn:]
    outcome = outcome[burn:]
    oracle_signal_mean = oracle_signal_mean[burn:]
    oracle_outcome_mean = oracle_outcome_mean[burn:]
    if bool(scenario["missing"]):
        row_missing = rng.random(x.shape[:2]) < float(parameters["mcar_probability"])
        gap_hours = min(int(parameters["missing_gap_days"]) * 24, len(x))
        start_objects = int(parameters["missing_gap_objects"])
        end_objects = int(parameters["missing_gap_objects"])
        if start_objects + end_objects >= object_count:
            raise ValueError("missing-gap objects leave no usable cross-section")
        row_missing[:gap_hours, :start_objects] = True
        row_missing[-gap_hours:, object_count - end_objects:] = True
        x[row_missing] = np.nan

    symbols = [f"O{index + 1:02d}" for index in range(object_count)]
    feature_columns = [f"X{index + 1:02d}" for index in range(feature_count)]
    mapping_rows = []
    frames: dict[str, pd.DataFrame] = {}
    folds: dict[str, tuple[WalkForwardFold, ...]] = {}
    for candidate_index, (signal_id, horizon) in enumerate(
        zip(("S01", "S02", "S03", "S04"), HORIZON_HOURS, strict=True)
    ):
        group_id = 1 + ((candidate_index + int(replicate)) % 4)
        mapping_rows.append(
            {
                "signal_id": signal_id,
                "candidate_id": signal_id,
                "horizon": horizon,
                "group_id": group_id,
                "is_true_positive": bool(active_beta[group_id - 1] > 0.0),
            }
        )
        step = HORIZON_HOURS[horizon]
        hour_indices = np.arange(0, day_count * 24, step, dtype=int)
        timestamps = pd.date_range(
            "2024-01-01", periods=len(hour_indices), freq=f"{step}h", tz="UTC"
        )
        repeated_times = np.repeat(timestamps.to_numpy(), object_count)
        repeated_symbols = np.tile(symbols, len(timestamps))
        sampled_x = x[hour_indices].reshape(-1, feature_count)
        sampled_signal = signal[hour_indices, :, group_id - 1].reshape(-1)
        sampled_outcome = outcome[hour_indices].reshape(-1)
        sampled_signal_mean = oracle_signal_mean[
            hour_indices, :, group_id - 1
        ].reshape(-1)
        sampled_outcome_mean = oracle_outcome_mean[hour_indices].reshape(-1)
        frame = pd.DataFrame(sampled_x, columns=feature_columns)
        frame.insert(0, "symbol", repeated_symbols)
        frame.insert(0, "decision_ts", pd.to_datetime(repeated_times, utc=True))
        frame["combo_signal"] = sampled_signal
        frame["raw_outcome"] = sampled_outcome
        frame["oracle_signal_mean"] = sampled_signal_mean
        frame["oracle_outcome_mean"] = sampled_outcome_mean
        frames[signal_id] = frame
        folds[horizon] = tuple(
            walk_forward_splits(
                timestamps,
                train_days=int(parameters["train_days"]),
                embargo_days=int(parameters["embargo_days"]),
                test_days=int(parameters["test_days"]),
                step_days=int(parameters["step_days"]),
            )
        )
    application = design["application_contract"]
    walk = application["walk_forward"]
    manifest_scale = (
        day_count == int(application["total_days"])
        and object_count == int(application["object_count_main"])
        and feature_count == int(application["feature_count"])
        and int(parameters["train_days"]) == int(walk["train_days"])
        and int(parameters["embargo_days"]) == int(walk["embargo_days"])
        and int(parameters["test_days"]) == int(walk["test_days"])
        and int(parameters["step_days"]) == int(walk["step_days"])
    )
    if manifest_scale:
        expected_folds = int(design["application_contract"]["outer_fold_count"])
        if any(len(value) != expected_folds for value in folds.values()):
            raise ValueError("Layer B outer-fold count differs from the frozen manifest")

    alias_rows: list[dict[str, object]] = []
    if bool(scenario["aliases"]):
        alias_rows = [dict(row) for row in design["layer_b"]["B10_alias_mapping"]]
    dataset = LayerBDataset(
        scenario_id=str(scenario["scenario_id"]),
        replicate=int(replicate),
        frames=frames,
        folds=folds,
        candidate_mapping=pd.DataFrame(mapping_rows),
        alias_mapping=pd.DataFrame(alias_rows, columns=["alias_id", "source_signal_id"]),
    )
    return dataset


def _fold_canonical_frame(
    frame: pd.DataFrame,
    folds: Sequence[WalkForwardFold],
    *,
    feature_columns: Sequence[str],
    min_cross_section: int,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for fold in folds:
        for split, start, end in (
            ("train", fold.train_start, fold.train_end),
            ("test", fold.test_start, fold.test_end),
        ):
            selected = frame.loc[frame["decision_ts"].between(start, end)].copy()
            finite = np.isfinite(
                selected[[*feature_columns, "combo_signal", "raw_outcome"]].to_numpy(dtype=float)
            ).all(axis=1)
            selected = selected.loc[finite]
            counts = selected.groupby("decision_ts")["symbol"].transform("size")
            selected = selected.loc[counts >= int(min_cross_section)].copy()
            if selected.empty:
                raise ValueError(f"fold {fold.fold_idx} {split} has no common support")
            selected["fold_idx"] = int(fold.fold_idx)
            selected["split"] = split
            parts.append(selected)
    canonical = pd.concat(parts, ignore_index=True).set_index(
        ["fold_idx", "split", "decision_ts", "symbol"]
    ).sort_index()
    canonical = substitution.build_cross_sectional_outcome_target(
        canonical,
        source_column="raw_outcome",
        target_column="outcome_target",
        min_cross_section=int(min_cross_section),
    )
    group_levels = ["fold_idx", "split", "decision_ts"]
    oracle_outcome_cs = canonical["oracle_outcome_mean"] - canonical[
        "oracle_outcome_mean"
    ].groupby(level=group_levels).transform("mean")
    canonical["oracle_signal_residual"] = (
        canonical["combo_signal"] - canonical["oracle_signal_mean"]
    )
    canonical["oracle_outcome_residual"] = canonical["outcome_target"] - oracle_outcome_cs
    canonical["forward_return"] = canonical["outcome_target"]
    canonical["strategy_forward_return"] = canonical["outcome_target"]
    return canonical


def _prediction_stream(
    canonical: pd.DataFrame,
    *,
    target_column: str,
    residual_column: str | None,
    source_model_class: str,
) -> pd.DataFrame:
    test = canonical.xs("test", level="split").reset_index()
    target = test[target_column].to_numpy(dtype=float)
    residual = target if residual_column is None else test[residual_column].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "fold_idx": test["fold_idx"].to_numpy(dtype=int),
            "decision_ts": pd.to_datetime(test["decision_ts"], utc=True),
            "symbol": test["symbol"].astype(str),
            "target_signal": target,
            "replica_signal": target - residual,
            "residual_signal": residual,
            "source_model_class": source_model_class,
        }
    )


def _with_model_class(predictions: pd.DataFrame, model_class: str = "linear_ridge") -> pd.DataFrame:
    result = predictions.copy()
    if "source_model_class" not in result:
        result["source_model_class"] = model_class
    return result


def run_layer_b_c_simulation(
    design: Mapping[str, object] | str | Path,
    *,
    scenario_id: str,
    replicate: int,
    base_state: str | None = None,
    beta: float | None = None,
    q_x: float | None = None,
    seed: int | None = None,
    test_overrides: Mapping[str, object] | None = None,
    performance_timings: list[dict[str, object]] | None = None,
) -> LayerBCSimulationArtifacts:
    """Run the sole formal truth-known B/C mechanism and ablation entry.

    ``test_overrides`` is exclusively for reduced end-to-end tests. The default
    path retains every application-scale and inference value from the manifest.
    This entry deliberately contains no temporal-falsification implementation.
    """
    manifest = (
        load_frozen_design(design)
        if isinstance(design, (str, Path))
        else _validate_frozen_design(design)
    )
    parameters = _simulation_parameters(manifest, test_overrides)
    scenario = _layer_b_scenario(
        manifest, scenario_id, base_state=base_state, beta=beta, q_x=q_x
    )
    master_seed = int(
        manifest["random_seeds"]["layer_b_c_master"] if seed is None else seed
    )
    with _performance_stage(
        performance_timings,
        "data_generation",
        scenario_id=str(scenario_id),
        replicate=int(replicate),
    ):
        dataset = _generate_layer_b_dataset(
            manifest,
            scenario,
            replicate=int(replicate),
            parameters=parameters,
            master_seed=master_seed,
        )
    feature_columns = tuple(f"X{index + 1:02d}" for index in range(int(parameters["feature_count"])))
    all_observations: list[pd.DataFrame] = []
    all_moments: list[pd.DataFrame] = []
    methods: dict[str, list[pd.DataFrame]] = {method: [] for method in RESIDUAL_METHODS}
    truth_by_hypothesis: dict[str, float] = {}

    for row in dataset.candidate_mapping.itertuples(index=False):
        with _performance_stage(
            performance_timings,
            "canonical_frame",
            candidate_id=str(row.signal_id),
            horizon=str(row.horizon),
        ):
            canonical = _fold_canonical_frame(
                dataset.frames[row.signal_id],
                dataset.folds[row.horizon],
                feature_columns=feature_columns,
                min_cross_section=int(parameters["min_cross_section"]),
            )
        folds = dataset.folds[row.horizon]
        with _performance_stage(
            performance_timings,
            "ridge_signal",
            candidate_id=str(row.signal_id),
            horizon=str(row.horizon),
        ):
            ridge_signal = substitution.fit_walk_forward_ridge_replicas(
                canonical,
                folds,
                candidate_id=str(row.signal_id),
                model_features={"level2_full": feature_columns},
                target_column="combo_signal",
                alpha_grid=parameters["alpha_grid"],
                inner_gap=pd.Timedelta(hours=HORIZON_HOURS[row.horizon]),
            )
        if performance_timings is not None:
            performance_timings[-1]["estimator_fit_count"] = int(
                len(ridge_signal.inner_scores)
                + ridge_signal.predictions["fold_idx"].nunique()
            )
        with _performance_stage(
            performance_timings,
            "ridge_outcome",
            candidate_id=str(row.signal_id),
            horizon=str(row.horizon),
        ):
            ridge_outcome = substitution.fit_walk_forward_ridge_replicas(
                canonical,
                folds,
                candidate_id=f"{row.signal_id}__outcome",
                model_features={"level2_full": feature_columns},
                target_column="outcome_target",
                alpha_grid=parameters["alpha_grid"],
                inner_gap=pd.Timedelta(hours=HORIZON_HOURS[row.horizon]),
            )
        if performance_timings is not None:
            performance_timings[-1]["estimator_fit_count"] = int(
                len(ridge_outcome.inner_scores)
                + ridge_outcome.predictions["fold_idx"].nunique()
            )
        registered_signal = substitution.fit_walk_forward_registered_replicas(
            canonical,
            folds,
            candidate_id=str(row.signal_id),
            frozen_ridge_predictions=ridge_signal.predictions,
            frozen_ridge_inner_scores=ridge_signal.inner_scores,
            target_column="combo_signal",
            feature_columns=feature_columns,
            inner_gap=pd.Timedelta(hours=HORIZON_HOURS[row.horizon]),
            model_specs=parameters["model_specs"],
            allow_model_subset=bool(parameters["allow_model_subset"]),
            fit_workers=int(parameters["fit_workers"]),
            performance_timings=performance_timings,
            timing_target="signal",
        )
        registered_outcome = substitution.fit_walk_forward_registered_replicas(
            canonical,
            folds,
            candidate_id=f"{row.signal_id}__outcome",
            frozen_ridge_predictions=ridge_outcome.predictions,
            frozen_ridge_inner_scores=ridge_outcome.inner_scores,
            target_column="outcome_target",
            feature_columns=feature_columns,
            inner_gap=pd.Timedelta(hours=HORIZON_HOURS[row.horizon]),
            model_specs=parameters["model_specs"],
            allow_model_subset=bool(parameters["allow_model_subset"]),
            fit_workers=int(parameters["fit_workers"]),
            performance_timings=performance_timings,
            timing_target="outcome",
        )
        with _performance_stage(
            performance_timings,
            "same_sample_R3_signal",
            candidate_id=str(row.signal_id),
            horizon=str(row.horizon),
            estimator_fit_count=len(folds),
        ):
            same_signal = substitution.fit_same_sample_registered_replicas(
                canonical,
                folds,
                registered_signal.fold_selection,
                target_column="combo_signal",
                feature_columns=feature_columns,
            )
        with _performance_stage(
            performance_timings,
            "same_sample_R3_outcome",
            candidate_id=str(row.signal_id),
            horizon=str(row.horizon),
            estimator_fit_count=len(folds),
        ):
            same_outcome = substitution.fit_same_sample_registered_replicas(
                canonical,
                folds,
                registered_outcome.fold_selection,
                target_column="outcome_target",
                feature_columns=feature_columns,
            )
        raw_signal = _prediction_stream(
            canonical, target_column="combo_signal", residual_column=None,
            source_model_class="none",
        )
        raw_outcome = _prediction_stream(
            canonical, target_column="outcome_target", residual_column=None,
            source_model_class="none",
        )
        oracle_signal = _prediction_stream(
            canonical, target_column="combo_signal", residual_column="oracle_signal_residual",
            source_model_class="oracle",
        )
        oracle_outcome = _prediction_stream(
            canonical, target_column="outcome_target", residual_column="oracle_outcome_residual",
            source_model_class="oracle",
        )
        ridge_signal_stream = _with_model_class(ridge_signal.predictions)
        ridge_outcome_stream = _with_model_class(ridge_outcome.predictions)
        pairs = {
            "R0": (raw_signal, raw_outcome),
            "R1": (_with_model_class(registered_signal.selected_predictions), raw_outcome),
            "R2": (raw_signal, _with_model_class(registered_outcome.selected_predictions)),
            "R3": (same_signal, same_outcome),
            "R4": (ridge_signal_stream, ridge_outcome_stream),
            "R5": (
                _with_model_class(registered_signal.selected_predictions),
                _with_model_class(registered_outcome.selected_predictions),
            ),
            "R6": (oracle_signal, oracle_outcome),
        }
        for method, (signal_predictions, outcome_predictions) in pairs.items():
            with _performance_stage(
                performance_timings,
                "residual_moment_construction",
                candidate_id=str(row.signal_id),
                horizon=str(row.horizon),
                residual_method=str(method),
            ):
                evaluated = substitution.evaluate_cross_fitted_double_residuals(
                    signal_predictions,
                    outcome_predictions,
                    hypothesis_id=str(row.signal_id),
                    horizon=str(row.horizon),
                    min_cross_section=int(parameters["min_cross_section"]),
                )
            observations = evaluated.observations.assign(residual_method=method)
            moments = evaluated.decision_moments.assign(residual_method=method)
            all_observations.append(observations)
            all_moments.append(moments)
            methods[method].append(moments)
        test_counts = pairs["R6"][0].groupby("decision_ts")["symbol"].size().to_numpy(dtype=float)
        truth_by_hypothesis[str(row.signal_id)] = (
            sqrt(0.20)
            * float(scenario["beta"])
            * (1.0 - float(
                (manifest["dgp"]["B08"]["cross_section_common_share"] if scenario["stress"] else manifest["dgp"]["cross_section_common_share_base"])["Z"]
            ))
            * float(np.mean(1.0 - 1.0 / test_counts))
            if bool(row.is_true_positive) and not bool(scenario["shared_nonlinearity"])
            else 0.0
        )

    bootstrap_seed = int(
        np.random.SeedSequence([master_seed, int(replicate), 91]).generate_state(1)[0]
    )
    decision_moments = pd.concat(all_moments, ignore_index=True)
    decision_moments["true_effect"] = decision_moments["hypothesis_id"].map(
        truth_by_hypothesis
    )
    if decision_moments["true_effect"].isna().any():
        raise RuntimeError("Layer C truth mapping is incomplete")
    with _performance_stage(
        performance_timings,
        "joint_inference",
        scenario_id=str(scenario_id),
        replicate=int(replicate),
    ):
        inference = infer_layer_c_comparison(
            decision_moments,
            expected_decisions_per_day={
                horizon: 24 // hours for horizon, hours in HORIZON_HOURS.items()
            },
            block_length=int(parameters["block_length"]),
            n_bootstrap=int(parameters["n_bootstrap"]),
            seed=bootstrap_seed,
            alpha=float(parameters["alpha"]),
        )
    return LayerBCSimulationArtifacts(
        dataset=dataset,
        layer_c=LayerCArtifacts(
            decision_moments=decision_moments,
            observations=pd.concat(all_observations, ignore_index=True),
            comparison_grid=inference.comparison_grid,
            bootstrap_starts=inference.bootstrap_starts,
        ),
    )


def infer_layer_c_comparison(
    decision_moments: pd.DataFrame,
    *,
    expected_decisions_per_day: Mapping[str, int],
    block_length: int,
    n_bootstrap: int,
    seed: int,
    alpha: float = 0.05,
    alternative: str = "greater",
    production_equivalent: bool = False,
) -> LayerCInferenceArtifacts:
    """Run the registered 7-by-3 Layer C inference on fixed moments."""
    required = {
        "residual_method",
        "hypothesis_id",
        "horizon",
        "fold_idx",
        "decision_ts",
        "double_residual_moment",
        "true_effect",
    }
    missing = sorted(required.difference(decision_moments.columns))
    if missing:
        raise ValueError("Layer C moments missing columns: " + ", ".join(missing))
    methods = set(decision_moments["residual_method"].astype(str))
    if methods != set(RESIDUAL_METHODS):
        raise ValueError("Layer C moments must cover exactly R0-R6")
    truth_rows = decision_moments[["hypothesis_id", "true_effect"]].drop_duplicates()
    if truth_rows["hypothesis_id"].duplicated().any():
        raise ValueError("Layer C truth must be constant within hypothesis")
    truth_by_hypothesis = truth_rows.set_index("hypothesis_id")["true_effect"]

    grid_rows: list[dict[str, object]] = []
    starts: list[pd.DataFrame] = []
    for method in RESIDUAL_METHODS:
        moments = decision_moments.loc[
            decision_moments["residual_method"].eq(method)
        ].drop(columns=["residual_method", "true_effect"])
        family = substitution.build_double_residual_daily_family(
            moments,
            expected_decisions_per_day=expected_decisions_per_day,
        )
        entry = (
            research_stats.circular_block_bootstrap_stepdown_max_t
            if production_equivalent
            else research_stats.simulation_calibration_circular_block_stepdown_max_t
        )
        inference = entry(
            family.daily_centered_sums,
            family.daily_counts,
            family.observed_effects,
            block_length=int(block_length),
            n_bootstrap=int(n_bootstrap),
            seed=int(seed),
            alternative=alternative,
        )
        summary = inference.summary.set_index("hypothesis_id")
        raw_p = summary["raw_one_sided_p_value"].reindex(family.observed_effects.index)
        adjusted_by_code = {
            "C0": raw_p.to_numpy(dtype=float),
            "C1": research_stats.holm_adjusted_p_values(raw_p),
            "C2": summary["stepdown_max_t_adjusted_p_value"].reindex(
                family.observed_effects.index
            ).to_numpy(dtype=float),
        }
        observed = family.observed_effects
        truth = truth_by_hypothesis.reindex(observed.index)
        if truth.isna().any():
            raise RuntimeError("Layer C inference truth mapping is incomplete")
        for adjustment in FAMILY_ADJUSTMENTS:
            p_values = adjusted_by_code[adjustment]
            identity = correction_identity_for_code(adjustment)
            rejected = p_values <= float(alpha)
            positive = truth.to_numpy(dtype=float) > 0.0
            grid_rows.append(
                {
                    "cell_id": f"{method}-{adjustment}",
                    "residual_method": method,
                    "family_adjustment": adjustment,
                    **identity,
                    "hypothesis_count": len(observed),
                    "true_positive_count": int(positive.sum()),
                    "rejection_count": int(rejected.sum()),
                    "true_positive_rejection_count": int((rejected & positive).sum()),
                    "true_null_rejection_count": int((rejected & ~positive).sum()),
                    "mean_estimate": float(observed.mean()),
                    "mean_true_effect": float(truth.mean()),
                    "mean_bias": float((observed - truth).mean()),
                    "mean_squared_error": float(np.square(observed - truth).mean()),
                    "p_values_json": json.dumps(
                        dict(zip(observed.index.astype(str), p_values, strict=True)),
                        sort_keys=True,
                    ),
                }
            )
        starts.append(inference.block_starts.assign(residual_method=method))
    comparison = pd.DataFrame(grid_rows)
    if comparison["cell_id"].nunique() != 21 or len(comparison) != 21:
        raise RuntimeError("Layer C comparison grid is not exactly 21 cells")
    return LayerCInferenceArtifacts(
        comparison_grid=comparison,
        bootstrap_starts=pd.concat(starts, ignore_index=True),
    )


def _temporal_falsification_contract(
    design: Mapping[str, object],
) -> dict[str, object]:
    raw = dict(design["temporal_falsification"])
    expected = {
        "source_scenario": "B07-M",
        "effect_lifetimes": ["0.5H", "H", "2H", "7d"],
        "wrong_offsets": ["-2H", "-H", "+H", "+2H"],
        "protection_bands_days": [3, 7, 14],
        "randomizations_per_band": 2_000,
        "effect_generation": (
            "decision_Z_t_total_beta_contribution_spread_equally_over_next_L_hourly_returns"
        ),
        "holding_return": "sum_hourly_returns_over_open_t_close_t_plus_H",
        "wrong_offset_operation": "shift_signal_timestamp_only_keep_return_window_fixed",
        "randomization_operation": (
            "within_fold_permute_among_legal_timestamps_at_least_protection_band_away"
        ),
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ValueError(f"temporal falsification contract changed: {key}")
    if raw.get("T05") != {
        "decision_interval": "H/2",
        "included_in_primary_summary": False,
    }:
        raise ValueError("temporal falsification T05 contract changed")
    return raw


def _lifetime_hours(label: str, holding_hours: int) -> int:
    values = {
        "0.5H": holding_hours // 2,
        "H": holding_hours,
        "2H": 2 * holding_hours,
        "7d": 7 * 24,
    }
    if label not in values or values[label] <= 0:
        raise ValueError(f"unsupported temporal effect lifetime: {label}")
    return values[label]


def _offset_hours(label: str, holding_hours: int) -> int:
    if label == "correct":
        return 0
    multipliers = {"-2H": -2, "-H": -1, "+H": 1, "+2H": 2}
    if label not in multipliers:
        raise ValueError(f"unsupported temporal alignment: {label}")
    return multipliers[label] * holding_hours


def _spread_decision_effects(
    z: np.ndarray,
    *,
    event_hours: np.ndarray,
    lifetime_hours: int,
    beta: float,
) -> np.ndarray:
    """Spread each decision's total beta contribution over its next L returns."""
    differences = np.zeros((len(z) + 1, z.shape[1]), dtype=float)
    contribution = float(beta) * z[event_hours] / float(lifetime_hours)
    np.add.at(differences, event_hours, contribution)
    np.add.at(differences, event_hours + int(lifetime_hours), -contribution)
    return np.cumsum(differences[:-1], axis=0)


def _temporal_expected_effect(
    *,
    decision_hours: np.ndarray,
    source_hours: np.ndarray,
    event_hours: np.ndarray,
    holding_hours: int,
    lifetime_hours: int,
    beta: float,
    z_phi: float,
    cross_section_factor: float,
) -> float:
    coefficients: list[float] = []
    for decision, source in zip(decision_hours, source_hours, strict=True):
        relevant = event_hours[
            (event_hours < decision + holding_hours)
            & (event_hours + lifetime_hours > decision)
        ]
        overlap = np.minimum(
            decision + holding_hours, relevant + lifetime_hours
        ) - np.maximum(decision, relevant)
        coefficients.append(
            float(
                np.sum(
                    overlap
                    * np.power(float(z_phi), np.abs(source - relevant))
                )
            )
            / float(lifetime_hours)
        )
    return (
        sqrt(0.20)
        * float(beta)
        * float(cross_section_factor)
        * float(np.mean(coefficients))
    )


def _temporal_observation_block(
    *,
    z: np.ndarray,
    signal_noise: np.ndarray,
    outcome_noise: np.ndarray,
    folds: Sequence[WalkForwardFold],
    symbols: Sequence[str],
    signal_id: str,
    holding_hours: int,
    lifetime_label: str,
    decision_interval_hours: int,
    alignment_labels: Sequence[str],
    beta: float,
    common_share_z: float,
    z_phi: float,
    included_in_primary_summary: bool,
) -> pd.DataFrame:
    lifetime = _lifetime_hours(lifetime_label, holding_hours)
    event_hours = np.arange(0, len(z) - lifetime, decision_interval_hours, dtype=int)
    hourly_effect = _spread_decision_effects(
        z, event_hours=event_hours, lifetime_hours=lifetime, beta=beta
    )
    hourly_outcome = hourly_effect + outcome_noise
    cumulative = np.vstack(
        [np.zeros((1, len(symbols)), dtype=float), np.cumsum(hourly_outcome, axis=0)]
    )
    timestamps = pd.date_range(
        "2024-01-01", periods=len(z), freq="h", tz="UTC"
    )
    parts: list[pd.DataFrame] = []
    for fold in folds:
        decision_hours = event_hours[
            (timestamps[event_hours] >= fold.test_start)
            & (timestamps[event_hours] <= fold.test_end)
            & (event_hours + holding_hours < len(z))
        ]
        for alignment in alignment_labels:
            offset = _offset_hours(alignment, holding_hours)
            source_hours = decision_hours + offset
            valid = (source_hours >= 0) & (source_hours < len(z))
            decisions = decision_hours[valid]
            sources = source_hours[valid]
            signal = sqrt(0.20) * z[sources] + sqrt(0.40) * signal_noise[sources]
            outcome = cumulative[decisions + holding_hours] - cumulative[decisions]
            signal -= signal.mean(axis=1, keepdims=True)
            outcome -= outcome.mean(axis=1, keepdims=True)
            true_effect = _temporal_expected_effect(
                decision_hours=decisions,
                source_hours=sources,
                event_hours=event_hours,
                holding_hours=holding_hours,
                lifetime_hours=lifetime,
                beta=beta,
                z_phi=z_phi,
                cross_section_factor=(1.0 - float(common_share_z))
                * (1.0 - 1.0 / len(symbols)),
            )
            hypothesis_id = (
                f"{signal_id}__L_{lifetime_label}__align_{alignment}"
                + ("__T05" if not included_in_primary_summary else "")
            )
            count = len(decisions) * len(symbols)
            frame = pd.DataFrame(
                {
                    "hypothesis_id": np.repeat(hypothesis_id, count),
                    "horizon": np.repeat(f"{holding_hours}h", count),
                    "fold_idx": np.repeat(int(fold.fold_idx), count),
                    "decision_ts": np.repeat(timestamps[decisions].to_numpy(), len(symbols)),
                    "signal_source_ts": np.repeat(timestamps[sources].to_numpy(), len(symbols)),
                    "symbol": np.tile(np.asarray(symbols), len(decisions)),
                    "signal_residual": signal.reshape(-1),
                    "outcome_residual": outcome.reshape(-1),
                    "effect_lifetime": np.repeat(lifetime_label, count),
                    "alignment_label": np.repeat(alignment, count),
                    "alignment_offset_hours": np.repeat(offset, count),
                    "holding_hours": np.repeat(holding_hours, count),
                    "decision_interval_hours": np.repeat(decision_interval_hours, count),
                    "true_effect": np.repeat(true_effect, count),
                    "included_in_primary_summary": np.repeat(
                        included_in_primary_summary, count
                    ),
                }
            )
            frame["residual_product"] = (
                frame["signal_residual"] * frame["outcome_residual"]
            )
            parts.append(frame)
    if not parts:
        raise ValueError("temporal falsification generated no OOS observations")
    return pd.concat(parts, ignore_index=True)


def generate_temporal_falsification_dataset(
    design: Mapping[str, object] | str | Path,
    *,
    replicate: int = 0,
    seed: int | None = None,
    test_overrides: Mapping[str, int] | None = None,
) -> TemporalFalsificationDataset:
    """Generate the approved B07-M lifetime, alignment, and T05 panels."""
    manifest = (
        load_frozen_design(design)
        if isinstance(design, (str, Path))
        else _validate_frozen_design(design)
    )
    temporal = _temporal_falsification_contract(manifest)
    application = manifest["application_contract"]
    walk = application["walk_forward"]
    parameters = {
        "day_count": int(application["total_days"]),
        "object_count": int(application["object_count_main"]),
        "train_days": int(walk["train_days"]),
        "embargo_days": int(walk["embargo_days"]),
        "test_days": int(walk["test_days"]),
        "step_days": int(walk["step_days"]),
    }
    if test_overrides is not None:
        unknown = sorted(set(test_overrides).difference(parameters))
        if unknown:
            raise ValueError("unknown temporal test override(s): " + ", ".join(unknown))
        parameters.update({key: int(value) for key, value in test_overrides.items()})
    if parameters["object_count"] < 3:
        raise ValueError("temporal falsification requires at least three objects")
    if parameters["test_days"] <= 28:
        raise ValueError("temporal test folds must exceed the 14-day protection diameter")

    max_holding = max(HORIZON_HOURS.values())
    max_lifetime = 7 * 24
    total_hours = parameters["day_count"] * 24 + max_lifetime + 2 * max_holding + 1
    master_seed = int(
        manifest["random_seeds"]["temporal_falsification_master"]
        if seed is None
        else seed
    )
    child = np.random.SeedSequence([master_seed, int(replicate)]).spawn(3)
    dgp = manifest["dgp"]
    common = dgp["cross_section_common_share_base"]
    z = _stationary_panel_process(
        np.random.Generator(np.random.PCG64DXSM(child[0])),
        hour_count=total_hours,
        object_count=parameters["object_count"],
        component_count=1,
        half_life_hours=float(dgp["hourly_half_life"]["Z"]),
        common_share=float(common["Z"]),
    )[..., 0]
    signal_noise = _stationary_panel_process(
        np.random.Generator(np.random.PCG64DXSM(child[1])),
        hour_count=total_hours,
        object_count=parameters["object_count"],
        component_count=1,
        half_life_hours=float(dgp["hourly_half_life"]["signal_noise"]),
        common_share=float(common["signal_noise"]),
    )[..., 0]
    outcome_noise = _stationary_panel_process(
        np.random.Generator(np.random.PCG64DXSM(child[2])),
        hour_count=total_hours,
        object_count=parameters["object_count"],
        component_count=1,
        half_life_hours=float(dgp["hourly_half_life"]["outcome_noise"]),
        common_share=float(common["outcome_noise"]),
    )[..., 0] / sqrt(max_holding)
    symbols = tuple(f"O{index + 1:02d}" for index in range(parameters["object_count"]))
    base_times = pd.date_range(
        "2024-01-01", periods=parameters["day_count"] * 24, freq="h", tz="UTC"
    )
    folds = tuple(
        walk_forward_splits(
            base_times,
            train_days=parameters["train_days"],
            embargo_days=parameters["embargo_days"],
            test_days=parameters["test_days"],
            step_days=parameters["step_days"],
        )
    )
    if not folds:
        raise ValueError("temporal falsification has no outer folds")
    b07_rows = [
        row
        for row in manifest["layer_b"]["confirmatory"]
        if row.get("id") == "B07-M"
    ]
    if len(b07_rows) != 1:
        raise ValueError("temporal falsification requires one B07-M confirmatory row")
    beta = float(b07_rows[0]["beta"])
    z_phi = 2.0 ** (-1.0 / float(dgp["hourly_half_life"]["Z"]))
    primary_parts: list[pd.DataFrame] = []
    t05_parts: list[pd.DataFrame] = []
    alignments = ("correct", *tuple(str(value) for value in temporal["wrong_offsets"]))
    for signal_id, holding_hours in zip(
        ("S01", "S02", "S03", "S04"), HORIZON_HOURS.values(), strict=True
    ):
        for lifetime in temporal["effect_lifetimes"]:
            primary_parts.append(
                _temporal_observation_block(
                    z=z,
                    signal_noise=signal_noise,
                    outcome_noise=outcome_noise,
                    folds=folds,
                    symbols=symbols,
                    signal_id=signal_id,
                    holding_hours=holding_hours,
                    lifetime_label=str(lifetime),
                    decision_interval_hours=holding_hours,
                    alignment_labels=alignments,
                    beta=beta,
                    common_share_z=float(common["Z"]),
                    z_phi=z_phi,
                    included_in_primary_summary=True,
                )
            )
            t05_parts.append(
                _temporal_observation_block(
                    z=z,
                    signal_noise=signal_noise,
                    outcome_noise=outcome_noise,
                    folds=folds,
                    symbols=symbols,
                    signal_id=signal_id,
                    holding_hours=holding_hours,
                    lifetime_label=str(lifetime),
                    decision_interval_hours=holding_hours // 2,
                    alignment_labels=("correct",),
                    beta=beta,
                    common_share_z=float(common["Z"]),
                    z_phi=z_phi,
                    included_in_primary_summary=False,
                )
            )
    return TemporalFalsificationDataset(
        observations=pd.concat(primary_parts, ignore_index=True),
        t05_observations=pd.concat(t05_parts, ignore_index=True),
    )


def _temporal_decision_moments(observations: pd.DataFrame) -> pd.DataFrame:
    keys = ["hypothesis_id", "horizon", "fold_idx", "decision_ts"]
    return (
        observations.groupby(keys, as_index=False, sort=True)
        .agg(double_residual_moment=("residual_product", "mean"))
        .sort_values(keys, kind="mergesort")
        .reset_index(drop=True)
    )


def run_temporal_falsification(
    design: Mapping[str, object] | str | Path,
    *,
    replicate: int = 0,
    seed: int | None = None,
    test_overrides: Mapping[str, int] | None = None,
) -> TemporalFalsificationArtifacts:
    """Run the registered signal-side temporal falsification family."""
    manifest = (
        load_frozen_design(design)
        if isinstance(design, (str, Path))
        else _validate_frozen_design(design)
    )
    temporal = _temporal_falsification_contract(manifest)
    dataset = generate_temporal_falsification_dataset(
        manifest, replicate=replicate, seed=seed, test_overrides=test_overrides
    )
    expected = {f"{hours}h": 24 // hours for hours in HORIZON_HOURS.values()}
    family = substitution.build_double_residual_daily_family(
        _temporal_decision_moments(dataset.observations),
        expected_decisions_per_day=expected,
    )
    randomization_seed = int(
        manifest["random_seeds"]["temporal_falsification_master"]
        if seed is None
        else seed
    )
    randomized = substitution.simulation_signal_time_randomization(
        dataset.observations,
        family.daily_effects,
        guard_days=tuple(int(value) for value in temporal["protection_bands_days"]),
        n_randomizations=2_000,
        seed=randomization_seed,
        alternative="greater",
    )
    primary_metadata = dataset.observations[
        [
            "hypothesis_id",
            "effect_lifetime",
            "alignment_label",
            "alignment_offset_hours",
            "holding_hours",
            "decision_interval_hours",
            "true_effect",
            "included_in_primary_summary",
        ]
    ].drop_duplicates("hypothesis_id")
    primary_summary = randomized.summary.merge(
        primary_metadata, on="hypothesis_id", validate="many_to_one"
    )
    t05_metadata = dataset.t05_observations[
        [
            "hypothesis_id",
            "effect_lifetime",
            "alignment_label",
            "holding_hours",
            "decision_interval_hours",
            "true_effect",
            "included_in_primary_summary",
        ]
    ].drop_duplicates("hypothesis_id")
    t05_effects = (
        _temporal_decision_moments(dataset.t05_observations)
        .groupby("hypothesis_id", as_index=False)
        .agg(observed_effect=("double_residual_moment", "mean"))
    )
    t05_summary = t05_effects.merge(
        t05_metadata, on="hypothesis_id", validate="one_to_one"
    )
    if t05_summary["included_in_primary_summary"].any():
        raise RuntimeError("T05 leaked into the primary temporal summary")
    return TemporalFalsificationArtifacts(
        dataset=dataset,
        daily_effects=family.daily_effects,
        randomization=randomized,
        primary_summary=primary_summary,
        t05_summary=t05_summary,
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_sha256sum_file_audit(
    raw_text: str, *, allowed_file_names: Sequence[str]
) -> pd.DataFrame:
    """Parse a sha256sum transcript into a strict task/file/hash table."""
    allowed = {str(name) for name in allowed_file_names}
    if not allowed:
        raise ValueError("external file audit has no allowed file names")
    rows = []
    for raw_line in str(raw_text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError("external file audit contains an invalid line")
        digest, raw_path = parts
        path = Path(raw_path.lstrip("*"))
        if (
            path.name not in allowed
            or not path.parent.name.startswith("task_")
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("external file audit contains an invalid task file hash")
        rows.append(
            {
                "task_id": path.parent.name,
                "file_name": path.name,
                "sha256": digest,
            }
        )
    result = pd.DataFrame(rows, columns=["task_id", "file_name", "sha256"])
    if result.empty or result.duplicated(["task_id", "file_name"]).any():
        raise ValueError("external file audit coverage is empty or duplicated")
    return result.sort_values(["task_id", "file_name"]).reset_index(drop=True)


def validate_federated_layer_bc_tasks(
    registered_tasks: Sequence[Mapping[str, object]],
    source_roots: Mapping[str, str | Path],
    *,
    design_sha256: str,
    baseline_source_id: str,
    route: str | None = None,
) -> FederatedTaskValidationArtifacts:
    """Validate a complete Layer-B/C package produced by multiple runtimes."""
    tasks = tuple(dict(task) for task in registered_tasks)
    expected_by_id = {str(task["task_id"]): task for task in tasks}
    if len(expected_by_id) != len(tasks):
        raise ValueError("registered Layer B/C task ids are not unique")
    if baseline_source_id not in source_roots:
        raise ValueError("federated baseline source is absent")
    required_files = (
        "candidate_mapping.csv",
        "alias_mapping.csv",
        "fold_manifest.csv",
        "decision_moments.csv.gz",
        "observations.csv.gz",
        "comparison_grid.csv",
        "bootstrap_starts.csv.gz",
    )
    externally_auditable_files = {"observations.csv.gz"}
    expected_task_manifest = pd.DataFrame(tasks).to_csv(index=False)
    runtime_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    seen: set[str] = set()
    baseline_source_hashes: Mapping[str, object] | None = None
    for source_id, raw_root in source_roots.items():
        root = Path(raw_root)
        runtime_path = root / "runtime_manifest.json"
        task_manifest_path = root / "registered_task_manifest.csv"
        if not runtime_path.is_file() or not task_manifest_path.is_file():
            raise RuntimeError(f"federated source {source_id} is missing manifests")
        if task_manifest_path.read_text(encoding="utf-8") != expected_task_manifest:
            raise RuntimeError(f"federated source {source_id} task manifest changed")
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        identity = runtime.get("identity", {})
        if (
            runtime.get("status") != "frozen"
            or identity.get("design_sha256") != str(design_sha256)
            or not isinstance(identity.get("source_sha256"), dict)
        ):
            raise RuntimeError(f"federated source {source_id} runtime identity is invalid")
        if source_id == baseline_source_id:
            baseline_source_hashes = identity["source_sha256"]
        audit_receipt_path = root / "external_file_audit_receipt.json"
        audit_table_path = root / "external_file_audit.csv"
        audit_raw_path = root / "external_observation_sha256sum.txt"
        external_hashes: dict[tuple[str, str], str] = {}
        if (
            audit_receipt_path.exists()
            or audit_table_path.exists()
            or audit_raw_path.exists()
        ):
            if (
                not audit_receipt_path.is_file()
                or not audit_table_path.is_file()
                or not audit_raw_path.is_file()
            ):
                raise RuntimeError(f"federated source {source_id} has incomplete external audit")
            audit_receipt = json.loads(
                audit_receipt_path.read_text(encoding="utf-8")
            )
            audit = pd.read_csv(audit_table_path, dtype=str)
            required_audit_columns = {"task_id", "file_name", "sha256"}
            if not required_audit_columns.issubset(audit.columns):
                raise RuntimeError(f"federated source {source_id} external audit columns are invalid")
            if (
                audit_receipt.get("status") != "complete"
                or audit_receipt.get("source_id") != str(source_id)
                or audit_receipt.get("design_sha256") != str(design_sha256)
                or audit_receipt.get("runtime_manifest_sha256")
                != _file_sha256(runtime_path)
                or audit_receipt.get("registered_task_manifest_sha256")
                != _file_sha256(task_manifest_path)
                or audit_receipt.get("audit_table_sha256")
                != _file_sha256(audit_table_path)
                or audit_receipt.get("raw_sha256sum_sha256")
                != _file_sha256(audit_raw_path)
                or audit_receipt.get("audited_file_name")
                != "observations.csv.gz"
                or int(audit_receipt.get("audited_file_count", -1)) != len(audit)
                or audit.duplicated(["task_id", "file_name"]).any()
                or not set(audit["file_name"]).issubset(externally_auditable_files)
                or not audit["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
            ):
                raise RuntimeError(f"federated source {source_id} external audit is invalid")
            try:
                raw_audit = parse_sha256sum_file_audit(
                    audit_raw_path.read_text(encoding="utf-8"),
                    allowed_file_names=tuple(externally_auditable_files),
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"federated source {source_id} raw external audit is invalid"
                ) from exc
            normalized_audit = audit[
                ["task_id", "file_name", "sha256"]
            ].sort_values(["task_id", "file_name"]).reset_index(drop=True)
            if not raw_audit.equals(normalized_audit):
                raise RuntimeError(
                    f"federated source {source_id} raw external audit disagrees with audit table"
                )
            external_hashes = {
                (str(row.task_id), str(row.file_name)): str(row.sha256)
                for row in audit.itertuples(index=False)
            }
        runtime_rows.append(
            {
                "source_id": str(source_id),
                "source_root": str(root),
                "runtime_manifest_sha256": _file_sha256(runtime_path),
                "platform": identity.get("environment", {}).get("platform"),
                "python": identity.get("environment", {}).get("python"),
                "data_root": identity.get("data_root"),
                "source_sha256_json": json.dumps(
                    identity["source_sha256"], sort_keys=True
                ),
                "external_file_audit_receipt_sha256": (
                    _file_sha256(audit_receipt_path)
                    if audit_receipt_path.is_file() else None
                ),
            }
        )
        temporary = [
            path.name for path in root.iterdir()
            if path.is_dir() and path.name.startswith(".task_")
        ]
        if temporary:
            raise RuntimeError(f"federated source {source_id} has temporary tasks")
        for task_dir in sorted(root.glob("task_*")):
            if not task_dir.is_dir():
                continue
            task_id = task_dir.name
            if task_id not in expected_by_id:
                raise RuntimeError(f"unexpected federated task: {task_id}")
            if task_id in seen:
                raise RuntimeError(f"duplicate federated task: {task_id}")
            receipt_path = task_dir / "receipt.json"
            if not receipt_path.is_file():
                raise RuntimeError(f"missing receipt for federated task: {task_id}")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            runtime_sha = _file_sha256(runtime_path)
            expected = expected_by_id[task_id]
            for key, value in expected.items():
                if receipt.get(key) != value:
                    raise RuntimeError(f"federated task metadata mismatch: {task_id}:{key}")
            if (
                receipt.get("status") != "complete"
                or receipt.get("design_sha256") != str(design_sha256)
                or receipt.get("runtime_manifest_sha256") != runtime_sha
            ):
                raise RuntimeError(f"federated task receipt identity mismatch: {task_id}")
            actual_hashes = {}
            for name in required_files:
                path = task_dir / name
                external_key = (task_id, name)
                if external_key in external_hashes:
                    actual_hashes[name] = external_hashes[external_key]
                elif not path.is_file():
                    raise RuntimeError(f"federated task file missing: {task_id}/{name}")
                else:
                    actual_hashes[name] = _file_sha256(path)
            if receipt.get("file_sha256") != actual_hashes:
                raise RuntimeError(f"federated task file hash mismatch: {task_id}")
            grid = normalize_correction_identity_frame(
                pd.read_csv(task_dir / "comparison_grid.csv"), route=route
            )
            expected_cells = {
                f"{method}-{adjustment}"
                for method in RESIDUAL_METHODS for adjustment in FAMILY_ADJUSTMENTS
            }
            if (
                len(grid) != 21
                or set(grid.get("cell_id", ())) != expected_cells
                or not grid["hypothesis_count"].eq(4).all()
            ):
                raise RuntimeError(f"federated comparison grid contract failed: {task_id}")
            numeric = grid.select_dtypes(include=[np.number]).to_numpy(dtype=float)
            if not np.isfinite(numeric).all():
                raise RuntimeError(f"federated comparison grid is non-finite: {task_id}")
            task_rows.append(
                {
                    "source_id": str(source_id),
                    "task_dir": str(task_dir),
                    "task_receipt_sha256": _file_sha256(receipt_path),
                    "runtime_manifest_sha256": runtime_sha,
                    **expected,
                }
            )
            seen.add(task_id)
        audited_task_ids = {task_id for task_id, _ in external_hashes}
        source_task_ids = {
            path.name for path in root.glob("task_*") if path.is_dir()
        }
        if external_hashes and audited_task_ids != source_task_ids:
            raise RuntimeError(
                f"federated source {source_id} external audit task coverage is incomplete"
            )
    if baseline_source_hashes is None:
        raise RuntimeError("federated baseline source was not inspected")
    for row in runtime_rows:
        if json.loads(str(row["source_sha256_json"])) != baseline_source_hashes:
            raise RuntimeError("federated scientific source hashes differ across runtimes")
    missing = sorted(set(expected_by_id).difference(seen))
    if missing:
        raise RuntimeError("federated task coverage is incomplete: " + ", ".join(missing))
    inventory = pd.DataFrame(task_rows).sort_values("task_idx").reset_index(drop=True)
    runtimes = pd.DataFrame(runtime_rows).sort_values("source_id").reset_index(drop=True)
    receipt = {
        "status": "complete",
        "design_sha256": str(design_sha256),
        "task_count": len(inventory),
        "runtime_count": len(runtimes),
        "task_indices_complete": inventory["task_idx"].tolist() == list(range(len(tasks))),
        "scientific_source_sha256": dict(baseline_source_hashes),
    }
    if not receipt["task_indices_complete"]:
        raise RuntimeError("federated task indices are not complete and ordered")
    return FederatedTaskValidationArtifacts(inventory, runtimes, receipt)


def summarize_layer_bc_results(
    replicate_results: pd.DataFrame, *, route: str | None = None
) -> LayerBCSummary:
    """Summarize registered B/C replicate distributions without a pass label."""
    frame = normalize_correction_identity_frame(replicate_results, route=route)
    required = {
        "case_id", "replicate", "cell_id", "residual_method", "family_adjustment",
        "hypothesis_count", "true_positive_count", "true_positive_rejection_count",
        "true_null_rejection_count", "mean_estimate", "mean_bias", "mean_squared_error",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("Layer B/C results missing columns: " + ", ".join(missing))
    numeric = frame[
        [
            "hypothesis_count", "true_positive_count", "true_positive_rejection_count",
            "true_null_rejection_count", "mean_estimate", "mean_bias", "mean_squared_error",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Layer B/C results contain non-finite conclusion metrics")
    frame["true_null_count"] = frame["hypothesis_count"] - frame["true_positive_count"]
    frame["any_false_rejection"] = frame["true_null_rejection_count"] > 0
    frame["any_true_rejection"] = np.where(
        frame["true_positive_count"] > 0,
        frame["true_positive_rejection_count"] > 0,
        np.nan,
    )
    frame["true_positive_rejection_rate"] = np.where(
        frame["true_positive_count"] > 0,
        frame["true_positive_rejection_count"] / frame["true_positive_count"],
        np.nan,
    )
    frame["true_null_rejection_rate"] = np.where(
        frame["true_null_count"] > 0,
        frame["true_null_rejection_count"] / frame["true_null_count"],
        np.nan,
    )
    keys = [
        "case_id",
        "cell_id",
        "residual_method",
        "family_adjustment",
        "identity_schema_version",
        "namespace",
        "method_id",
        "algorithm",
        "legacy_code",
    ]
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(keys, sort=True):
        row = dict(zip(keys, key, strict=True))
        row["replicate_count"] = int(group["replicate"].nunique())
        for source, target in (
            ("any_false_rejection", "false_family_detection_rate"),
            ("any_true_rejection", "any_true_detection_rate"),
            ("true_positive_rejection_rate", "true_positive_rejection_rate"),
            ("true_null_rejection_rate", "true_null_rejection_rate"),
        ):
            values = group[source].dropna().astype(float)
            estimate = float(values.mean()) if len(values) else np.nan
            mcse = sqrt(estimate * (1.0 - estimate) / len(values)) if len(values) else np.nan
            row[target] = estimate
            row[f"{target}_mcse"] = mcse
        row["mean_estimate"] = float(group["mean_estimate"].mean())
        row["mean_bias"] = float(group["mean_bias"].mean())
        row["mean_squared_error"] = float(group["mean_squared_error"].mean())
        rows.append(row)
    return LayerBCSummary(frame, pd.DataFrame(rows))


def resource_projection(
    *, measured_cpu_seconds: float, measured_output_bytes: int,
    measured_model_fits: int, total_model_fits: int,
    available_disk_bytes: int, cpu_hour_limit: float, disk_fraction_limit: float,
) -> dict[str, object]:
    """Fail-closed linear projection used only by the performance preflight."""
    if min(measured_cpu_seconds, measured_output_bytes, measured_model_fits, total_model_fits, available_disk_bytes) <= 0:
        raise ValueError("resource projection inputs must be positive")
    scale = float(total_model_fits) / float(measured_model_fits)
    cpu_hours = float(measured_cpu_seconds) * scale / 3600.0
    output_bytes = int(np.ceil(float(measured_output_bytes) * scale))
    return {
        "projected_cpu_hours": cpu_hours,
        "projected_output_bytes": output_bytes,
        "cpu_limit_hours": float(cpu_hour_limit),
        "disk_limit_bytes": int(float(available_disk_bytes) * float(disk_fraction_limit)),
        "cpu_pass": cpu_hours <= float(cpu_hour_limit),
        "disk_pass": output_bytes <= float(available_disk_bytes) * float(disk_fraction_limit),
        "preflight_pass": cpu_hours <= float(cpu_hour_limit) and output_bytes <= float(available_disk_bytes) * float(disk_fraction_limit),
    }


def staged_resource_projection(
    measurements: pd.DataFrame,
    *,
    available_disk_bytes: int,
    cpu_hour_limit: float,
    disk_fraction_limit: float,
    cpu_hours_reporting_only: bool = False,
    peak_rss_bytes: int | None = None,
    physical_memory_bytes: int | None = None,
    memory_fraction_limit: float | None = None,
) -> dict[str, object]:
    """Combine measured stage costs using pre-registered workload multipliers."""
    required = {"stage", "measured_cpu_seconds", "measured_output_bytes", "workload_multiplier"}
    missing = sorted(required.difference(measurements.columns))
    if missing:
        raise ValueError("stage measurements missing columns: " + ", ".join(missing))
    numeric = measurements[["measured_cpu_seconds", "measured_output_bytes", "workload_multiplier"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric <= 0.0).any():
        raise ValueError("stage resource measurements must be finite and positive")
    projected_cpu_seconds = float(
        (measurements["measured_cpu_seconds"] * measurements["workload_multiplier"]).sum()
    )
    projected_output_bytes = int(np.ceil(
        (measurements["measured_output_bytes"] * measurements["workload_multiplier"]).sum()
    ))
    disk_limit = int(float(available_disk_bytes) * float(disk_fraction_limit))
    cpu_hours = projected_cpu_seconds / 3600.0
    cpu_within_legacy_limit = cpu_hours <= float(cpu_hour_limit)
    disk_pass = projected_output_bytes <= disk_limit
    memory_values = (peak_rss_bytes, physical_memory_bytes, memory_fraction_limit)
    if any(value is not None for value in memory_values):
        if any(value is None for value in memory_values):
            raise ValueError("memory resource contract must be specified completely")
        if (
            int(peak_rss_bytes) <= 0
            or int(physical_memory_bytes) <= 0
            or not 0.0 < float(memory_fraction_limit) <= 1.0
        ):
            raise ValueError("memory resource contract inputs are invalid")
        memory_limit = int(
            int(physical_memory_bytes) * float(memory_fraction_limit)
        )
        memory_pass = int(peak_rss_bytes) <= memory_limit
    else:
        memory_limit = None
        memory_pass = True
    cpu_gate_pass = True if bool(cpu_hours_reporting_only) else cpu_within_legacy_limit
    return {
        "projected_cpu_hours": cpu_hours,
        "projected_output_bytes": projected_output_bytes,
        "cpu_limit_hours": float(cpu_hour_limit),
        "disk_limit_bytes": disk_limit,
        "cpu_hours_reporting_only": bool(cpu_hours_reporting_only),
        "cpu_within_legacy_limit": cpu_within_legacy_limit,
        "disk_pass": disk_pass,
        "peak_rss_bytes": int(peak_rss_bytes) if peak_rss_bytes is not None else None,
        "physical_memory_bytes": (
            int(physical_memory_bytes) if physical_memory_bytes is not None else None
        ),
        "memory_limit_bytes": memory_limit,
        "memory_pass": memory_pass,
        "preflight_pass": cpu_gate_pass and disk_pass and memory_pass,
    }


def registered_preflight_workload_multipliers(
    design: Mapping[str, object] | str | Path,
) -> dict[str, int]:
    """Derive the frozen performance-preflight run counts from the manifest."""
    manifest = (
        load_frozen_design(design)
        if isinstance(design, (str, Path))
        else _validate_frozen_design(design)
    )
    layer_a = manifest["layer_a"]
    a_main = sum(int(row["monte_carlo"]) for row in layer_a["scenarios"])
    a_production = (
        len(layer_a["production_equivalent_subset"]["scenario_ids"])
        * int(layer_a["production_equivalent_subset"]["first_seed_ordered_replicates"])
    )
    a_sensitivity_each = (
        len(layer_a["block_sensitivity_subset"]["scenario_ids"])
        * int(layer_a["block_sensitivity_subset"]["first_seed_ordered_replicates"])
    )
    if len(layer_a["block_sensitivity_subset"]["block_lengths_days"]) != 2:
        raise ValueError("preflight expects exactly the registered 7/28 sensitivity blocks")

    layer_b = manifest["layer_b"]
    b_main = sum(int(row["monte_carlo"]) for row in layer_b["confirmatory"])
    for row in layer_b["extended"]:
        scenario_id = str(row["id"])
        if "monte_carlo" in row:
            b_main += int(row["monte_carlo"])
        elif scenario_id == "B07-grid-except-M":
            cells = len(row["beta_grid"]) * len(row["q_x_grid"]) - 1
            b_main += cells * int(row["monte_carlo_per_cell"])
        else:
            b_main += len(row["base_states"]) * int(row["monte_carlo_per_state"])
    declared_b = int(layer_b["declared_workload"]["complete_datasets"])
    if b_main != declared_b:
        raise ValueError("derived Layer B workload disagrees with declared workload")
    b_production = (
        len(layer_b["production_equivalent_subset"]["scenario_ids"])
        * int(layer_b["production_equivalent_subset"]["first_seed_ordered_replicates"])
    )
    return {
        "A_main_499": a_main,
        "A_production_10000": a_production,
        "A_sensitivity_7d_499": a_sensitivity_each,
        "A_sensitivity_28d_499": a_sensitivity_each,
        "B_C_complete_499": b_main,
        "B_C_production_inference_10000": b_production,
        "temporal_falsification_complete": 1,
    }


def registered_layer_b_tasks(
    design: Mapping[str, object] | str | Path,
) -> tuple[dict[str, object], ...]:
    """Enumerate the frozen 114 complete B/C datasets in manifest order."""
    manifest = (
        load_frozen_design(design)
        if isinstance(design, (str, Path))
        else _validate_frozen_design(design)
    )
    layer_b = manifest["layer_b"]
    rows: list[dict[str, object]] = []

    def append(
        *,
        case_id: str,
        scenario_id: str,
        replicate: int,
        base_state: str | None = None,
        beta: float | None = None,
        q_x: float | None = None,
    ) -> None:
        task_idx = len(rows)
        rows.append(
            {
                "task_idx": task_idx,
                "task_id": f"task_{task_idx:03d}__{case_id}__r{int(replicate):03d}",
                "case_id": str(case_id),
                "scenario_id": str(scenario_id),
                "replicate": int(replicate),
                "base_state": base_state,
                "beta": beta,
                "q_x": q_x,
            }
        )

    for scenario in layer_b["confirmatory"]:
        for replicate in range(int(scenario["monte_carlo"])):
            append(
                case_id=str(scenario["id"]),
                scenario_id=str(scenario["id"]),
                replicate=replicate,
                beta=scenario.get("beta"),
                q_x=scenario.get("q_x"),
            )

    for scenario in layer_b["extended"]:
        scenario_id = str(scenario["id"])
        if "monte_carlo" in scenario:
            for replicate in range(int(scenario["monte_carlo"])):
                append(
                    case_id=scenario_id,
                    scenario_id=scenario_id,
                    replicate=replicate,
                )
            continue
        if scenario_id == "B07-grid-except-M":
            excluded = (
                float(scenario["exclude"]["beta"]),
                float(scenario["exclude"]["q_x"]),
            )
            for beta in scenario["beta_grid"]:
                for q_x in scenario["q_x_grid"]:
                    cell = (float(beta), float(q_x))
                    if cell == excluded:
                        continue
                    case_id = f"B07-beta{float(beta):g}-qx{float(q_x):g}"
                    for replicate in range(int(scenario["monte_carlo_per_cell"])):
                        append(
                            case_id=case_id,
                            scenario_id="B07-M",
                            replicate=replicate,
                            beta=float(beta),
                            q_x=float(q_x),
                        )
            continue
        for base_state in scenario["base_states"]:
            for replicate in range(int(scenario["monte_carlo_per_state"])):
                append(
                    case_id=f"{scenario_id}-{base_state}",
                    scenario_id=scenario_id,
                    replicate=replicate,
                    base_state=str(base_state),
                )

    declared = int(layer_b["declared_workload"]["complete_datasets"])
    if len(rows) != declared:
        raise ValueError(
            f"registered Layer B task count {len(rows)} disagrees with {declared}"
        )
    return tuple(rows)


def registered_layer_b_production_tasks(
    design: Mapping[str, object] | str | Path,
) -> tuple[dict[str, object], ...]:
    """Select the frozen 30 B/C datasets requiring 10,000-bootstrap inference."""
    manifest = (
        load_frozen_design(design)
        if isinstance(design, (str, Path))
        else _validate_frozen_design(design)
    )
    contract = manifest["layer_b"]["production_equivalent_subset"]
    scenario_ids = set(str(value) for value in contract["scenario_ids"])
    count = int(contract["first_seed_ordered_replicates"])
    master = int(manifest["random_seeds"]["layer_b_c_master"])
    all_tasks = registered_layer_b_tasks(manifest)
    task_children = np.random.SeedSequence(master).spawn(len(all_tasks))
    rows = []
    for task, child in zip(all_tasks, task_children, strict=True):
        if str(task["case_id"]) not in scenario_ids or int(task["replicate"]) >= count:
            continue
        row = dict(task)
        row["production_inference_seed"] = int(child.generate_state(1)[0])
        rows.append(row)
    expected = len(scenario_ids) * count
    if len(rows) != expected:
        raise ValueError("registered B/C production subset is incomplete")
    return tuple(rows)
