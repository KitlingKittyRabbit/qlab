"""Formal Coinglass signal-substitution research infrastructure.

The entries in this module build point-in-time price/volume replacement
features, enforce one canonical cross-section, fit train-only ridge replicas,
score OOS replication, and replay already-frozen OOS signals.  They do not
select live candidates or authorize paper/live trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import itertools
import json
import resource
import tracemalloc
from time import perf_counter, process_time
from typing import Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from . import factor_research
from . import research_stats
from .data.crypto.panel import normalize_price_frame, rank_standardize_grouped_series
from .walkforward import WalkForwardFold


BASE_BAR = pd.Timedelta(minutes=15)
FEATURE_LAG = pd.Timedelta(minutes=15)
WINDOW_DAYS = (1, 2, 4, 8, 16, 32)
ALPHA_GRID = tuple(float(10**power) for power in range(-4, 5))
REPLICATION_R2_THRESHOLD = 0.10
LEVEL0_RAW_COLUMNS = (
    "size_control_raw",
    "momentum_control_raw",
    "volatility_control_raw",
)
LEVEL0_COLUMNS = ("size_control", "momentum_control", "volatility_control")
RETURN_COLUMNS = tuple(f"return_{days}d" for days in WINDOW_DAYS)
VOLATILITY_COLUMNS = tuple(f"realized_vol_{days}d" for days in WINDOW_DAYS)
DOLLAR_VOLUME_COLUMNS = tuple(f"log_dollar_volume_{days}d" for days in WINDOW_DAYS)
VOLUME_SURPRISE_COLUMNS = (
    "volume_surprise_1d_8d",
    "volume_surprise_2d_16d",
    "volume_surprise_4d_32d",
)
PRICE_VOLUME_COLUMNS = (
    *RETURN_COLUMNS,
    *VOLATILITY_COLUMNS,
    *DOLLAR_VOLUME_COLUMNS,
    *VOLUME_SURPRISE_COLUMNS,
)
ALL_RAW_PREDICTOR_COLUMNS = (*LEVEL0_RAW_COLUMNS, *PRICE_VOLUME_COLUMNS)


@dataclass(frozen=True)
class ReplacementFeatureArtifacts:
    raw_features: pd.DataFrame
    availability_audit: pd.DataFrame
    feature_manifest: pd.DataFrame


@dataclass(frozen=True)
class CanonicalSupportArtifacts:
    frame: pd.DataFrame
    audit: pd.DataFrame


@dataclass(frozen=True)
class RidgeReplicationArtifacts:
    predictions: pd.DataFrame
    coefficients: pd.DataFrame
    inner_scores: pd.DataFrame


@dataclass(frozen=True)
class RegisteredModelReplicationArtifacts:
    class_predictions: pd.DataFrame
    inner_scores: pd.DataFrame
    model_diagnostics: pd.DataFrame
    fold_selection: pd.DataFrame
    selected_predictions: pd.DataFrame


@dataclass(frozen=True)
class RegisteredReplicaFoldRidgeArtifacts:
    prediction_frame: pd.DataFrame
    class_candidate: dict[str, object]
    inner_rows: list[dict[str, object]]


@dataclass(frozen=True)
class RegisteredReplicaFoldClassFit:
    model_class: str
    best_parameter_key: str
    class_candidate: dict[str, object]
    inner_rows: list[dict[str, object]]
    test_prediction: np.ndarray


@dataclass(frozen=True)
class RegisteredReplicaFoldArtifacts:
    fold_idx: int
    prediction_parts: list[pd.DataFrame]
    inner_rows: list[dict[str, object]]
    class_candidates: list[dict[str, object]]
    diagnostic_rows: list[dict[str, object]]
    selection_row: dict[str, object]
    winner: dict[str, object]


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
class SignalReplayArtifacts:
    summary: pd.DataFrame
    timeseries: pd.DataFrame
    holdings: pd.DataFrame
    orders: pd.DataFrame
    diagnostics: pd.DataFrame


@dataclass(frozen=True)
class FoldTargetSignalArtifacts:
    signals: pd.DataFrame
    fold_weights: pd.DataFrame


@dataclass(frozen=True)
class ResidualEquivalenceArtifacts:
    mapping: pd.DataFrame
    groups: pd.DataFrame


@dataclass(frozen=True)
class ResidualInformationArtifacts:
    timeseries: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class ResidualFamilyInformationArtifacts:
    track_timeseries: pd.DataFrame
    family_timeseries: pd.DataFrame
    signal_summary: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class ResidualCandidateSensitivityArtifacts:
    detail: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class CandidateResidualDependencyArtifacts:
    daily_centered_sums: pd.DataFrame
    daily_counts: pd.DataFrame
    daily_coverage_audit: pd.DataFrame
    unique_results: pd.DataFrame
    candidate_results: pd.DataFrame
    bootstrap_by_block_length: Mapping[
        int, research_stats.StepdownMaxTBootstrapArtifacts
    ]


@dataclass(frozen=True)
class DoubleResidualArtifacts:
    observations: pd.DataFrame
    decision_moments: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class DoubleResidualDailyFamilyArtifacts:
    daily_effects: pd.DataFrame
    daily_centered_sums: pd.DataFrame
    daily_counts: pd.DataFrame
    coverage_audit: pd.DataFrame
    observed_effects: pd.Series


@dataclass(frozen=True)
class DoubleResidualTimeRandomizationArtifacts:
    schedule: pd.DataFrame
    null_effects: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class DoubleResidualFamilyInferenceArtifacts:
    daily_effects: pd.DataFrame
    daily_centered_sums: pd.DataFrame
    daily_counts: pd.DataFrame
    coverage_audit: pd.DataFrame
    unique_results: pd.DataFrame
    candidate_results: pd.DataFrame
    bootstrap_starts: pd.DataFrame


@dataclass(frozen=True)
class RegisteredResidualLabelArtifacts:
    unique_results: pd.DataFrame
    candidate_results: pd.DataFrame


@dataclass(frozen=True)
class ResidualTimeShiftPreparationArtifacts:
    observed_effects: pd.DataFrame
    offset_fold_effects: pd.DataFrame
    support_audit: pd.DataFrame


@dataclass(frozen=True)
class ResidualTimeRandomizationArtifacts:
    summary: pd.DataFrame
    null_effects: pd.DataFrame
    null_t_values: pd.DataFrame
    schedule: pd.DataFrame


@dataclass(frozen=True)
class ResidualTimeRandomizationSummaryArtifacts:
    candidate_results: pd.DataFrame
    sensitivity_summary: pd.DataFrame
    horizon_summary: pd.DataFrame
    support_summary: pd.DataFrame


def replacement_feature_manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for days in WINDOW_DAYS:
        rows.extend(
            [
                {
                    "feature_name": f"return_{days}d",
                    "family": "return",
                    "window_days": days,
                    "formula": "log(close[cutoff] / close[cutoff-window])",
                },
                {
                    "feature_name": f"realized_vol_{days}d",
                    "family": "volatility",
                    "window_days": days,
                    "formula": "std(15m_log_returns, ddof=1)",
                },
                {
                    "feature_name": f"log_dollar_volume_{days}d",
                    "family": "volume",
                    "window_days": days,
                    "formula": "log(mean(close*base_volume))",
                },
            ]
        )
    for short_days, long_days in ((1, 8), (2, 16), (4, 32)):
        rows.append(
            {
                "feature_name": f"volume_surprise_{short_days}d_{long_days}d",
                "family": "volume",
                "window_days": long_days,
                "formula": f"log_dollar_volume_{short_days}d-log_dollar_volume_{long_days}d",
            }
        )
    rows.extend(
        [
            {
                "feature_name": "size_control_raw",
                "family": "level0_control",
                "window_days": 90,
                "formula": "log1p(median(last_90_complete_daily_dollar_volumes)); min_days=20",
            },
            {
                "feature_name": "momentum_control_raw",
                "family": "level0_control",
                "window_days": 30,
                "formula": "close[cutoff]/close[cutoff-30d]-1",
            },
            {
                "feature_name": "volatility_control_raw",
                "family": "level0_control",
                "window_days": 30,
                "formula": "std(15m_log_returns_over_30d, ddof=1)",
            },
        ]
    )
    manifest = pd.DataFrame(rows)
    manifest["base_bar"] = "15m"
    manifest["feature_lag"] = "15m"
    manifest["requires_complete_window"] = True
    return manifest


def model_feature_sets() -> dict[str, tuple[str, ...]]:
    models: dict[str, tuple[str, ...]] = {
        "level0_controls": LEVEL0_COLUMNS,
    }
    for feature_name in PRICE_VOLUME_COLUMNS:
        models[f"level1__{feature_name}"] = (feature_name,)
    models.update(
        {
            "level2_return": RETURN_COLUMNS,
            "level2_volatility": VOLATILITY_COLUMNS,
            "level2_volume": (*DOLLAR_VOLUME_COLUMNS, *VOLUME_SURPRISE_COLUMNS),
            "level2_full": PRICE_VOLUME_COLUMNS,
        }
    )
    return models


def build_substitution_target_manifest(
    summary: pd.DataFrame,
    *,
    return_horizon: str = "1d",
) -> pd.DataFrame:
    required = {
        "combo_id",
        "track",
        "weight_scheme",
        "panel_frequency",
        "return_horizon",
        "component_features",
        "n_components",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError("family-weight summary missing columns: " + ", ".join(missing))
    horizon = str(return_horizon)
    manifest = summary.loc[summary["return_horizon"].astype(str) == horizon].copy()
    if manifest.empty:
        raise ValueError(f"no {horizon} family-weight target rows")
    if manifest.duplicated(["combo_id", "weight_scheme"]).any():
        raise ValueError("duplicate combo_id/weight_scheme target rows")
    manifest.insert(
        0,
        "candidate_id",
        manifest["combo_id"].astype(str) + "__" + manifest["weight_scheme"].astype(str),
    )
    manifest.insert(
        1,
        "basket_id",
        manifest["return_horizon"].astype(str)
        + "__"
        + manifest["track"].astype(str)
        + "__"
        + manifest["component_features"].astype(str),
    )
    manifest["lifecycle"] = "candidate_diagnostic_target"
    return manifest.reset_index(drop=True)


def audit_target_signal_reproduction(
    rebuilt_target: pd.DataFrame,
    existing_holdings: pd.DataFrame,
    *,
    candidate_id: str,
    combo_id: str,
    weight_scheme: str,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    required_target = {"fold_idx", "decision_ts", "symbol", "combo_signal"}
    required_holdings = {
        "combo_id",
        "weight_scheme",
        "fold_idx",
        "decision_ts",
        "symbol",
        "signal_value",
    }
    missing_target = sorted(required_target.difference(rebuilt_target.columns))
    missing_holdings = sorted(required_holdings.difference(existing_holdings.columns))
    if missing_target:
        raise ValueError("rebuilt target missing columns: " + ", ".join(missing_target))
    if missing_holdings:
        raise ValueError("existing holdings missing columns: " + ", ".join(missing_holdings))
    expected = existing_holdings.loc[
        (existing_holdings["combo_id"].astype(str) == str(combo_id))
        & (existing_holdings["weight_scheme"].astype(str) == str(weight_scheme))
    ].copy()
    if expected.empty:
        raise ValueError("existing holdings contain no rows for target candidate")
    keys = ["fold_idx", "decision_ts", "symbol"]
    rebuilt = rebuilt_target[[*keys, "combo_signal"]].copy()
    expected["decision_ts"] = pd.to_datetime(expected["decision_ts"], utc=True)
    rebuilt["decision_ts"] = pd.to_datetime(rebuilt["decision_ts"], utc=True)
    rebuilt_keys = pd.MultiIndex.from_frame(rebuilt[keys])
    expected_keys = pd.MultiIndex.from_frame(expected[keys])
    missing_keys = expected_keys.difference(rebuilt_keys)
    unexpected_keys = rebuilt_keys.difference(expected_keys)
    missing_count = int(len(missing_keys))
    unexpected_count = int(len(unexpected_keys))
    missing_key_text = "|".join(
        f"{int(values[0])}@{pd.Timestamp(values[1]).isoformat()}@{values[2]}"
        for values in missing_keys.tolist()
    )
    unexpected_key_text = "|".join(
        f"{int(values[0])}@{pd.Timestamp(values[1]).isoformat()}@{values[2]}"
        for values in unexpected_keys.tolist()
    )
    aligned = expected.merge(rebuilt, on=keys, how="inner", validate="one_to_one")
    expected_values = pd.to_numeric(aligned["signal_value"], errors="coerce")
    rebuilt_values = pd.to_numeric(aligned["combo_signal"], errors="coerce")
    nan_pattern_match = bool(expected_values.isna().equals(rebuilt_values.isna()))
    finite = expected_values.notna() & rebuilt_values.notna()
    differences = (expected_values.loc[finite] - rebuilt_values.loc[finite]).abs()
    max_abs_diff = float(differences.max()) if not differences.empty else 0.0
    passed = (
        missing_count == 0
        and unexpected_count == 0
        and len(rebuilt) == len(expected)
        and nan_pattern_match
        and np.isfinite(max_abs_diff)
        and max_abs_diff <= float(tolerance)
    )
    return pd.DataFrame(
        [
            {
                "candidate_id": candidate_id,
                "combo_id": combo_id,
                "weight_scheme": weight_scheme,
                "existing_holding_rows": len(expected),
                "matched_holding_rows": len(aligned),
                "missing_rebuilt_rows": missing_count,
                "missing_rebuilt_keys": missing_key_text,
                "unexpected_rebuilt_rows": unexpected_count,
                "unexpected_rebuilt_keys": unexpected_key_text,
                "nan_pattern_match": nan_pattern_match,
                "max_abs_signal_difference": max_abs_diff,
                "tolerance": float(tolerance),
                "reproduction_pass": bool(passed),
            }
        ]
    )


def reconstruct_folds_from_oos_holdings(
    existing_holdings: pd.DataFrame,
    *,
    combo_id: str,
    weight_scheme: str,
    train_days: int,
    embargo_days: int,
) -> list[WalkForwardFold]:
    """Recover frozen outer-fold boundaries from an existing OOS holdings artifact."""
    required = {"combo_id", "weight_scheme", "fold_idx", "decision_ts"}
    missing = sorted(required.difference(existing_holdings.columns))
    if missing:
        raise ValueError("existing holdings missing fold columns: " + ", ".join(missing))
    train_days = int(train_days)
    embargo_days = int(embargo_days)
    if train_days <= 0 or embargo_days < 0:
        raise ValueError("invalid train_days or embargo_days")
    selected = existing_holdings.loc[
        (existing_holdings["combo_id"].astype(str) == str(combo_id))
        & (existing_holdings["weight_scheme"].astype(str) == str(weight_scheme)),
        ["fold_idx", "decision_ts"],
    ].drop_duplicates()
    if selected.empty:
        raise ValueError("existing holdings contain no rows for fold reconstruction")
    selected["decision_ts"] = pd.to_datetime(selected["decision_ts"], utc=True)
    bounds = selected.groupby("fold_idx", sort=True)["decision_ts"].agg(["min", "max"])
    expected_indices = list(range(len(bounds)))
    actual_indices = [int(value) for value in bounds.index]
    if actual_indices != expected_indices:
        raise ValueError("existing holdings fold_idx values must be contiguous from zero")
    folds: list[WalkForwardFold] = []
    for fold_idx, row in bounds.iterrows():
        test_start = pd.Timestamp(row["min"])
        test_end = pd.Timestamp(row["max"])
        train_end = test_start - pd.Timedelta(days=embargo_days + 1)
        train_start = train_end - pd.Timedelta(days=train_days - 1)
        folds.append(
            WalkForwardFold(
                fold_idx=int(fold_idx),
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
    return folds


def build_fold_target_signals(
    combo_spec: factor_research.ComboSpec,
    signal_frame: pd.DataFrame,
    folds: Sequence[object],
    *,
    weight_scheme: str,
    feature_families: Mapping[str, str],
    min_cross_section: int,
) -> FoldTargetSignalArtifacts:
    """Export train and test target signals using each fold's frozen combo weights."""
    required = {"symbol", "forward_return", *combo_spec.feature_names}
    missing = sorted(required.difference(signal_frame.columns))
    if missing:
        raise ValueError("signal_frame missing target-signal columns: " + ", ".join(missing))
    signal_frames: list[pd.DataFrame] = []
    weight_rows: list[dict[str, object]] = []
    for fold in folds:
        train_slice = factor_research.select_dates(
            signal_frame[["symbol", *combo_spec.feature_names, "forward_return"]], fold, "train"
        )
        train_stats = factor_research.train_feature_stats(
            train_slice, combo_spec.feature_names, int(min_cross_section)
        )
        if train_stats is None:
            raise ValueError(f"no train feature stats for fold {fold.fold_idx}")
        directions = {feature_name: stat.direction for feature_name, stat in train_stats.items()}
        _, weights, diagnostics = factor_research.composite_weight_scores_weights_and_diagnostics(
            train_stats,
            weight_scheme,
            feature_families=feature_families,
        )
        for feature_name in combo_spec.feature_names:
            weight_rows.append(
                {
                    "combo_id": combo_spec.combo_id,
                    "weight_scheme": weight_scheme,
                    "fold_idx": int(fold.fold_idx),
                    "feature_name": feature_name,
                    "direction": int(directions[feature_name]),
                    "feature_weight": float(weights[feature_name]),
                    **diagnostics,
                }
            )
        for split in ("train", "test"):
            split_frame = factor_research.select_dates(
                signal_frame[["symbol", *combo_spec.feature_names, "forward_return"]], fold, split
            )
            composite = factor_research.build_composite_frame(
                split_frame,
                combo_spec.feature_names,
                directions,
                weights,
                extra_columns=("forward_return",),
            ).rename(columns={"composite_signal": "combo_signal"})
            output = composite.reset_index(names="decision_ts")
            output["fold_idx"] = int(fold.fold_idx)
            output["split"] = split
            output["strategy_forward_return"] = output["forward_return"]
            signal_frames.append(output)
    if not signal_frames:
        raise ValueError("fold target signal export is empty")
    signals = pd.concat(signal_frames, ignore_index=True).sort_values(
        ["fold_idx", "split", "decision_ts", "symbol"]
    ).reset_index(drop=True)
    return FoldTargetSignalArtifacts(
        signals=signals,
        fold_weights=pd.DataFrame(weight_rows),
    )


def _normalized_bar_end_frame(price_frame: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_price_frame(price_frame)
    required = {"c", "v"}
    missing = sorted(required.difference(normalized.columns))
    if missing:
        raise ValueError("price cache entry missing columns: " + ", ".join(missing))
    if len(normalized.index) > 1:
        deltas = normalized.index.to_series().diff().dropna()
        if not deltas.empty and (deltas < BASE_BAR).any():
            raise ValueError("price cache contains intervals shorter than 15m")
    bar_end = pd.DatetimeIndex(normalized.index) + BASE_BAR
    full_index = pd.date_range(bar_end.min(), bar_end.max(), freq=BASE_BAR, tz="UTC", name="bar_end_ts")
    frame = normalized[["c", "v"]].copy()
    frame.index = bar_end
    frame.index.name = "bar_end_ts"
    frame = frame.reindex(full_index)
    frame["c"] = pd.to_numeric(frame["c"], errors="coerce")
    frame["v"] = pd.to_numeric(frame["v"], errors="coerce")
    finite = np.isfinite(frame["c"].to_numpy(dtype=float)) & np.isfinite(frame["v"].to_numpy(dtype=float))
    frame.loc[~finite, ["c", "v"]] = np.nan
    frame.loc[(frame["c"] <= 0.0) | (frame["v"] < 0.0), ["c", "v"]] = np.nan
    return frame


def _window_values(frame: pd.DataFrame, end_position: int, periods: int) -> dict[str, float] | None:
    start_position = end_position - periods
    if start_position < 0:
        return None
    closes = frame["c"].iloc[start_position : end_position + 1].to_numpy(dtype=float)
    volumes = frame["v"].iloc[start_position : end_position + 1].to_numpy(dtype=float)
    if len(closes) != periods + 1 or not np.isfinite(closes).all() or not np.isfinite(volumes).all():
        return None
    log_returns = np.diff(np.log(closes))
    dollar_volume = closes[1:] * volumes[1:]
    if len(log_returns) != periods or len(dollar_volume) != periods:
        return None
    return {
        "return": float(np.log(closes[-1] / closes[0])),
        "volatility": float(np.std(log_returns, ddof=1)),
        "log_dollar_volume": float(np.log(np.mean(dollar_volume)))
        if np.mean(dollar_volume) > 0.0
        else float("nan"),
        "daily_dollar_volume": float(np.sum(dollar_volume)),
    }


def _features_for_symbol(
    symbol: str,
    decision_index: pd.DatetimeIndex,
    price_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _normalized_bar_end_frame(price_frame)
    decisions = pd.DatetimeIndex(decision_index).sort_values().unique()
    decisions = decisions.tz_localize("UTC") if decisions.tz is None else decisions.tz_convert("UTC")
    cutoffs = decisions - FEATURE_LAG
    close = frame["c"].astype(float)
    volume = frame["v"].astype(float)
    valid_bar = close.notna() & volume.notna() & np.isfinite(close) & np.isfinite(volume)
    log_close = np.log(close)
    log_return = log_close.diff()
    dollar_volume = close * volume
    feature_grid = pd.DataFrame(index=frame.index)
    for days in WINDOW_DAYS:
        periods = days * 96
        complete = valid_bar.rolling(periods + 1, min_periods=periods + 1).sum().eq(periods + 1)
        feature_grid[f"return_{days}d"] = (log_close - log_close.shift(periods)).where(complete)
        feature_grid[f"realized_vol_{days}d"] = (
            log_return.rolling(periods, min_periods=periods).std(ddof=1).where(complete)
        )
        mean_dollar_volume = dollar_volume.rolling(periods, min_periods=periods).mean()
        feature_grid[f"log_dollar_volume_{days}d"] = np.log(mean_dollar_volume).where(
            complete & mean_dollar_volume.gt(0.0)
        )
    for short_days, long_days in ((1, 8), (2, 16), (4, 32)):
        feature_grid[f"volume_surprise_{short_days}d_{long_days}d"] = (
            feature_grid[f"log_dollar_volume_{short_days}d"]
            - feature_grid[f"log_dollar_volume_{long_days}d"]
        )

    control_periods = 30 * 96
    control_complete = valid_bar.rolling(
        control_periods + 1, min_periods=control_periods + 1
    ).sum().eq(control_periods + 1)
    feature_grid["momentum_control_raw"] = (
        close / close.shift(control_periods) - 1.0
    ).where(control_complete)
    feature_grid["volatility_control_raw"] = (
        log_return.rolling(control_periods, min_periods=control_periods).std(ddof=1).where(
            control_complete
        )
    )
    daily_complete = valid_bar.rolling(97, min_periods=97).sum().eq(97)
    daily_dollar_volume = dollar_volume.rolling(96, min_periods=96).sum().where(daily_complete)
    if len(cutoffs):
        daily_index = pd.date_range(
            cutoffs.min().normalize() - pd.Timedelta(days=120),
            cutoffs.max().normalize(),
            freq="1D",
            tz="UTC",
        )
        daily_samples = daily_dollar_volume.reindex(daily_index)
        size_by_day = np.log1p(daily_samples.rolling(90, min_periods=20).median())
        feature_grid["size_control_raw"] = size_by_day.reindex(
            feature_grid.index, method="ffill"
        )
    else:
        feature_grid["size_control_raw"] = np.nan

    selected = feature_grid.reindex(cutoffs)[list(ALL_RAW_PREDICTOR_COLUMNS)].copy()
    selected.index = decisions
    selected.index.name = "decision_ts"
    selected["symbol"] = symbol
    features = selected.reset_index().set_index(["decision_ts", "symbol"]).sort_index()
    all_available = np.isfinite(features[list(ALL_RAW_PREDICTOR_COLUMNS)].astype(float)).all(axis=1)
    audit = features.reset_index()[["decision_ts", "symbol"]].copy()
    audit["feature_cutoff_ts"] = audit["decision_ts"] - FEATURE_LAG
    audit["max_input_bar_end_ts"] = audit["feature_cutoff_ts"].where(all_available.to_numpy())
    audit["all_predictors_available"] = all_available.to_numpy(dtype=bool)
    return features, audit


def build_price_volume_replacement_features(
    panel_index: pd.MultiIndex,
    price_payloads: Mapping[str, pd.DataFrame],
) -> ReplacementFeatureArtifacts:
    if not isinstance(panel_index, pd.MultiIndex) or panel_index.names[:2] != ["decision_ts", "symbol"]:
        raise ValueError("panel_index must be a decision_ts/symbol MultiIndex")
    pieces: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    symbols = sorted(panel_index.get_level_values("symbol").astype(str).unique())
    for symbol in symbols:
        if symbol not in price_payloads:
            raise ValueError("missing price payload for symbol: " + symbol)
        decision_index = pd.DatetimeIndex(
            panel_index[panel_index.get_level_values("symbol").astype(str) == symbol]
            .get_level_values("decision_ts")
            .unique()
        )
        features, audit = _features_for_symbol(symbol, decision_index, price_payloads[symbol])
        pieces.append(features)
        audits.append(audit)
    raw_features = pd.concat(pieces).sort_index()
    availability_audit = pd.concat(audits, ignore_index=True).sort_values(
        ["decision_ts", "symbol"]
    ).reset_index(drop=True)
    invalid = availability_audit.loc[
        availability_audit["max_input_bar_end_ts"].notna()
        & (
            pd.to_datetime(availability_audit["max_input_bar_end_ts"], utc=True)
            > pd.to_datetime(availability_audit["feature_cutoff_ts"], utc=True)
        )
    ]
    if not invalid.empty:
        raise ValueError("replacement features crossed feature cutoff")
    return ReplacementFeatureArtifacts(
        raw_features=raw_features,
        availability_audit=availability_audit,
        feature_manifest=replacement_feature_manifest(),
    )


def build_canonical_common_support(
    target_frame: pd.DataFrame,
    raw_predictors: pd.DataFrame,
    *,
    target_column: str = "combo_signal",
    forward_return_column: str = "forward_return",
    strategy_return_column: str = "strategy_forward_return",
    min_cross_section: int = 5,
) -> CanonicalSupportArtifacts:
    if not isinstance(target_frame.index, pd.MultiIndex) or target_frame.index.names[:2] != [
        "decision_ts",
        "symbol",
    ]:
        raise ValueError("target_frame must use decision_ts/symbol MultiIndex")
    required_target = {target_column, forward_return_column, strategy_return_column}
    missing_target = sorted(required_target.difference(target_frame.columns))
    if missing_target:
        raise ValueError("target_frame missing columns: " + ", ".join(missing_target))
    missing_predictors = sorted(set(ALL_RAW_PREDICTOR_COLUMNS).difference(raw_predictors.columns))
    if missing_predictors:
        raise ValueError("raw_predictors missing columns: " + ", ".join(missing_predictors))
    joined = target_frame[[target_column, forward_return_column, strategy_return_column]].join(
        raw_predictors[list(ALL_RAW_PREDICTOR_COLUMNS)], how="left"
    )
    finite_columns = [target_column, forward_return_column, strategy_return_column, *ALL_RAW_PREDICTOR_COLUMNS]
    finite_mask = np.isfinite(joined[finite_columns].astype(float)).all(axis=1)
    joined["__common"] = finite_mask
    audit_rows: list[dict[str, object]] = []
    valid_decisions: list[pd.Timestamp] = []
    for decision_ts, group in joined.groupby(level="decision_ts", sort=True):
        target_valid = np.isfinite(
            group[[target_column, forward_return_column, strategy_return_column]].astype(float)
        ).all(axis=1)
        common = group["__common"].astype(bool)
        common_count = int(common.sum())
        status = "ok" if common_count >= int(min_cross_section) else "small_common_cross_section"
        if status == "ok":
            valid_decisions.append(pd.Timestamp(decision_ts))
        audit_rows.append(
            {
                "decision_ts": decision_ts,
                "original_valid_count": int(target_valid.sum()),
                "common_valid_count": common_count,
                "excluded_count": int(target_valid.sum() - common_count),
                "excluded_symbols": "|".join(
                    group.index.get_level_values("symbol")[target_valid & ~common].astype(str)
                ),
                "status": status,
            }
        )
    frame = joined.loc[
        joined["__common"]
        & joined.index.get_level_values("decision_ts").isin(valid_decisions)
    ].drop(columns="__common")
    if frame.empty:
        raise ValueError("canonical common support is empty")
    for raw_column, standardized_column in zip(LEVEL0_RAW_COLUMNS, LEVEL0_COLUMNS, strict=True):
        frame[standardized_column] = rank_standardize_grouped_series(
            frame[raw_column], level="decision_ts"
        )
    for column in PRICE_VOLUME_COLUMNS:
        frame[column] = rank_standardize_grouped_series(frame[column], level="decision_ts")
    return CanonicalSupportArtifacts(
        frame=frame.sort_index(),
        audit=pd.DataFrame(audit_rows).sort_values("decision_ts").reset_index(drop=True),
    )


def build_fold_canonical_common_support(
    fold_target_signals: pd.DataFrame,
    raw_predictors: pd.DataFrame,
    *,
    target_column: str = "combo_signal",
    min_cross_section: int = 5,
) -> CanonicalSupportArtifacts:
    required_target = {
        "fold_idx",
        "split",
        "decision_ts",
        "symbol",
        target_column,
        "forward_return",
        "strategy_forward_return",
    }
    missing_target = sorted(required_target.difference(fold_target_signals.columns))
    missing_predictors = sorted(set(ALL_RAW_PREDICTOR_COLUMNS).difference(raw_predictors.columns))
    if missing_target:
        raise ValueError("fold target signals missing columns: " + ", ".join(missing_target))
    if missing_predictors:
        raise ValueError("raw predictors missing columns: " + ", ".join(missing_predictors))
    target = fold_target_signals.copy()
    target["decision_ts"] = pd.to_datetime(target["decision_ts"], utc=True)
    joined = (
        target.set_index(["decision_ts", "symbol"])
        .join(raw_predictors[list(ALL_RAW_PREDICTOR_COLUMNS)], how="left")
        .reset_index()
    )
    finite_columns = [
        target_column,
        "forward_return",
        "strategy_forward_return",
        *ALL_RAW_PREDICTOR_COLUMNS,
    ]
    joined["__common"] = np.isfinite(joined[finite_columns].astype(float)).all(axis=1)
    audit_rows: list[dict[str, object]] = []
    valid_keys: set[tuple[int, str, pd.Timestamp]] = set()
    group_columns = ["fold_idx", "split", "decision_ts"]
    for keys, group in joined.groupby(group_columns, sort=True):
        target_valid = np.isfinite(
            group[[target_column, "forward_return", "strategy_forward_return"]].astype(float)
        ).all(axis=1)
        common = group["__common"].astype(bool)
        common_count = int(common.sum())
        status = "ok" if common_count >= int(min_cross_section) else "small_common_cross_section"
        if status == "ok":
            valid_keys.add((int(keys[0]), str(keys[1]), pd.Timestamp(keys[2])))
        audit_rows.append(
            {
                "fold_idx": int(keys[0]),
                "split": str(keys[1]),
                "decision_ts": keys[2],
                "original_valid_count": int(target_valid.sum()),
                "common_valid_count": common_count,
                "excluded_count": int(target_valid.sum() - common_count),
                "excluded_symbols": "|".join(
                    group.loc[target_valid & ~common, "symbol"].astype(str)
                ),
                "status": status,
            }
        )
    key_series = list(
        zip(
            joined["fold_idx"].astype(int),
            joined["split"].astype(str),
            pd.to_datetime(joined["decision_ts"], utc=True),
            strict=True,
        )
    )
    keep = joined["__common"].astype(bool).to_numpy() & np.asarray(
        [key in valid_keys for key in key_series], dtype=bool
    )
    frame = joined.loc[keep].drop(columns="__common").set_index(
        ["fold_idx", "split", "decision_ts", "symbol"]
    ).sort_index()
    if frame.empty:
        raise ValueError("fold canonical common support is empty")
    rank_levels = ["fold_idx", "split", "decision_ts"]
    for raw_column, standardized_column in zip(LEVEL0_RAW_COLUMNS, LEVEL0_COLUMNS, strict=True):
        frame[standardized_column] = rank_standardize_grouped_series(
            frame[raw_column], level=rank_levels
        )
    for column in PRICE_VOLUME_COLUMNS:
        frame[column] = rank_standardize_grouped_series(frame[column], level=rank_levels)
    return CanonicalSupportArtifacts(
        frame=frame,
        audit=pd.DataFrame(audit_rows).sort_values(group_columns).reset_index(drop=True),
    )


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[float, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(y) == 0:
        raise ValueError("invalid ridge training arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("ridge training arrays must be finite")
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    centered_x = x - x_mean
    centered_y = y - y_mean
    gram = centered_x.T @ centered_x
    beta = np.linalg.solve(gram + float(alpha) * np.eye(x.shape[1]), centered_x.T @ centered_y)
    intercept = y_mean - float(x_mean @ beta)
    return intercept, beta


def _inner_time_splits(
    decision_values: pd.DatetimeIndex,
    *,
    gap: pd.Timedelta,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    gap = pd.Timedelta(gap)
    if gap <= pd.Timedelta(0):
        raise ValueError("inner gap must be positive")
    unique = pd.DatetimeIndex(decision_values).sort_values().unique()
    if len(unique) < 8:
        raise ValueError("outer train has too few decision timestamps for 3 inner splits")
    blocks = [pd.DatetimeIndex(values) for values in np.array_split(unique, 4)]
    if any(len(block) == 0 for block in blocks):
        raise ValueError("inner time split contains an empty block")
    splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    for validation_number in range(1, 4):
        validation = blocks[validation_number]
        train = pd.DatetimeIndex(np.concatenate([block.to_numpy() for block in blocks[:validation_number]]))
        gap_cutoff = validation.min() - gap
        train = train[train < gap_cutoff]
        if len(train) == 0:
            raise ValueError(f"inner train is empty after {gap} gap")
        splits.append((train, validation))
    return splits


def _date_mask(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    dates = pd.DatetimeIndex(frame.index.get_level_values("decision_ts"))
    return np.asarray((dates >= start) & (dates <= end), dtype=bool)


REGISTERED_MODEL_CLASS_ORDER = (
    "linear_ridge",
    "hist_gbm",
    "random_forest",
    "poly2_ridge",
    "poly2_elastic_net",
)
REGISTERED_MODEL_RANDOM_STATE = 20_260_725
REGISTERED_SKLEARN_VERSION = "1.5.1"
REGISTERED_CAPABILITY_PREFLIGHT_RAW_FEATURE_COUNT = 24


def registered_replica_model_specs() -> Mapping[str, tuple[Mapping[str, object], ...]]:
    """Return the frozen non-lineage model configurations for five-model R5."""
    hist_gbm_legacy = tuple(
        {
            "max_depth": int(max_depth),
            "max_iter": int(max_iter),
            "learning_rate": float(learning_rate),
        }
        for max_depth, max_iter, learning_rate in itertools.product(
            (2, 3), (100, 300), (0.03, 0.1)
        )
    )
    hist_gbm_additional = (
        {
            "max_depth": 5,
            "max_iter": 300,
            "learning_rate": 0.05,
            "l2_regularization": 1.0,
            "min_samples_leaf": 20,
            "early_stopping": False,
        },
        {
            "max_depth": 5,
            "max_iter": 800,
            "learning_rate": 0.03,
            "l2_regularization": 1.0,
            "min_samples_leaf": 20,
            "early_stopping": False,
        },
        {
            "max_depth": 8,
            "max_iter": 500,
            "learning_rate": 0.03,
            "l2_regularization": 5.0,
            "min_samples_leaf": 20,
            "early_stopping": False,
        },
        {
            "max_depth": 8,
            "max_iter": 800,
            "learning_rate": 0.03,
            "l2_regularization": 5.0,
            "min_samples_leaf": 20,
            "early_stopping": False,
        },
    )
    random_forest_legacy = tuple(
        {
            "max_depth": int(max_depth),
            "n_estimators": int(n_estimators),
        }
        for max_depth, n_estimators in itertools.product((3, 5), (200, 500))
    )
    random_forest_additional = (
        {
            "max_depth": 8,
            "n_estimators": 500,
            "min_samples_leaf": 5,
            "max_features": 0.5,
            "bootstrap": True,
        },
        {
            "max_depth": 12,
            "n_estimators": 500,
            "min_samples_leaf": 5,
            "max_features": 0.5,
            "bootstrap": True,
        },
        {
            "max_depth": 12,
            "n_estimators": 800,
            "min_samples_leaf": 2,
            "max_features": 0.5,
            "bootstrap": True,
        },
        {
            "max_depth": None,
            "n_estimators": 800,
            "min_samples_leaf": 10,
            "max_features": 0.5,
            "bootstrap": True,
        },
    )
    poly2_ridge = tuple({"alpha": float(alpha)} for alpha in ALPHA_GRID)
    poly2_elastic_net = tuple(
        {
            "alpha": float(alpha),
            "l1_ratio": float(l1_ratio),
            "max_iter": 10_000,
            "tol": 1e-6,
        }
        for alpha in ALPHA_GRID
        for l1_ratio in (0.1, 0.5, 0.9)
    )
    return {
        "hist_gbm": (*hist_gbm_legacy, *hist_gbm_additional),
        "random_forest": (*random_forest_legacy, *random_forest_additional),
        "poly2_ridge": poly2_ridge,
        "poly2_elastic_net": poly2_elastic_net,
    }


def _poly2_registered_design(features: np.ndarray) -> np.ndarray:
    """Expand raw columns as raw, squares, then lexicographic i<j products."""
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("poly2 features must be a non-empty 2D array")
    interactions = [
        values[:, left] * values[:, right]
        for left in range(values.shape[1])
        for right in range(left + 1, values.shape[1])
    ]
    parts = [values, np.square(values)]
    if interactions:
        parts.append(np.column_stack(interactions))
    return np.column_stack(parts)


def _registered_model_parameter_specs() -> Mapping[str, tuple[Mapping[str, object], ...]]:
    return {
        "linear_ridge": tuple({"alpha": float(alpha)} for alpha in ALPHA_GRID),
        **registered_replica_model_specs(),
    }


def benchmark_registered_replica_estimators(
    features: np.ndarray,
    target: np.ndarray,
    *,
    repetitions: int = 1,
) -> pd.DataFrame:
    """Time frozen nonlinear estimator fits without evaluating predictions.

    This is a performance-only preflight entry. It exposes elapsed CPU time and
    fit counts, not statistical performance, and cannot produce research labels.
    """
    from time import process_time

    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("features and target must be aligned 2D/1D arrays")
    if len(x) < 40 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("benchmark inputs must contain at least 40 finite rows")
    if int(repetitions) < 1 or int(repetitions) > 3:
        raise ValueError("benchmark repetitions must be between 1 and 3")
    rows: list[dict[str, object]] = []
    for alpha in ALPHA_GRID:
        started = process_time()
        for _ in range(int(repetitions)):
            _fit_registered_estimator("linear_ridge", {"alpha": float(alpha)}, x, y)
        rows.append(
            {
                "model_class": "linear_ridge",
                "parameters": _registered_parameter_key({"alpha": float(alpha)}),
                "row_count": len(x),
                "feature_count": x.shape[1],
                "fit_count": int(repetitions),
                "cpu_seconds": process_time() - started,
                "performance_only": True,
            }
        )
    for model_class, configurations in registered_replica_model_specs().items():
        for parameters in configurations:
            started = process_time()
            for _ in range(int(repetitions)):
                _fit_registered_estimator(model_class, parameters, x, y)
            rows.append(
                {
                    "model_class": model_class,
                    "parameters": _registered_parameter_key(parameters),
                    "row_count": len(x),
                    "feature_count": x.shape[1],
                    "fit_count": int(repetitions),
                    "cpu_seconds": process_time() - started,
                    "performance_only": True,
                }
            )
    return pd.DataFrame(rows)


def evaluate_registered_replica_model_capability_performance(
    train_features: np.ndarray,
    train_target: np.ndarray,
    validation_features: np.ndarray,
    validation_target: np.ndarray,
) -> pd.DataFrame:
    """Evaluate every frozen model configuration on one strict holdout.

    This is the formal non-research model-capability preflight.  It uses the
    exact estimators and preprocessing used by R5, reports no selection label,
    and fails closed on non-finite input, prediction, or model non-convergence.
    """
    train_x = np.asarray(train_features, dtype=float)
    train_y = np.asarray(train_target, dtype=float)
    validation_x = np.asarray(validation_features, dtype=float)
    validation_y = np.asarray(validation_target, dtype=float)
    if (
        train_x.ndim != 2
        or validation_x.ndim != 2
        or train_y.ndim != 1
        or validation_y.ndim != 1
        or len(train_x) != len(train_y)
        or len(validation_x) != len(validation_y)
        or train_x.shape[1] != validation_x.shape[1]
    ):
        raise ValueError("capability preflight inputs must be aligned train/validation arrays")
    if train_x.shape[1] != REGISTERED_CAPABILITY_PREFLIGHT_RAW_FEATURE_COUNT:
        raise ValueError(
            "capability preflight requires "
            f"{REGISTERED_CAPABILITY_PREFLIGHT_RAW_FEATURE_COUNT} raw features"
        )
    if len(train_x) < 40 or len(validation_x) < 20:
        raise ValueError("capability preflight train/validation samples are too small")
    if not all(
        np.isfinite(values).all()
        for values in (train_x, train_y, validation_x, validation_y)
    ):
        raise ValueError("capability preflight inputs must be finite")

    tasks = [
        (model_class, dict(parameters), train_x, train_y, validation_x, validation_y)
        for model_class in REGISTERED_MODEL_CLASS_ORDER
        for parameters in _registered_model_parameter_specs()[model_class]
    ]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as executor:
        for task in tasks:
            started = perf_counter()
            row = executor.submit(
                _evaluate_registered_replica_configuration_isolated, task
            ).result()
            row["isolated_wall_seconds"] = perf_counter() - started
            rows.append(row)
    return pd.DataFrame(rows)


def _evaluate_registered_replica_configuration_isolated(
    task: tuple[str, Mapping[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, object]:
    """Fit one preflight configuration in a fresh process and report its peak RSS."""
    model_class, parameters, train_x, train_y, validation_x, validation_y = task
    cpu_started = process_time()
    wall_started = perf_counter()
    tracemalloc.start()
    try:
        estimator = _fit_registered_estimator(
            model_class, parameters, train_x, train_y
        )
        prediction = np.asarray(estimator.predict(validation_x), dtype=float)
        _, peak_python_memory_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    if prediction.shape != validation_y.shape or not np.isfinite(prediction).all():
        raise RuntimeError(
            f"registered {model_class} produced invalid preflight predictions"
        )
    error = validation_y - prediction
    peak_process_rss_bytes = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    return {
        "model_class": model_class,
        "parameters": _registered_parameter_key(parameters),
        "train_rows": len(train_x),
        "validation_rows": len(validation_x),
        "raw_feature_count": train_x.shape[1],
        "model_feature_count": (
            train_x.shape[1] * (train_x.shape[1] + 3) // 2
            if model_class.startswith("poly2_")
            else train_x.shape[1]
        ),
        "fit_count": 1,
        "validation_mse": float(np.mean(np.square(error))),
        "cpu_seconds": process_time() - cpu_started,
        "wall_seconds": perf_counter() - wall_started,
        "peak_memory_bytes": peak_process_rss_bytes,
        "peak_process_rss_bytes": peak_process_rss_bytes,
        "peak_python_memory_bytes": int(peak_python_memory_bytes),
        "memory_measurement_scope": "fresh_process_per_configuration",
        "convergence_status": "converged",
        "selection_performed": False,
        "research_conclusion_allowed": False,
    }


def _registered_parameter_key(parameters: Mapping[str, object]) -> str:
    return json.dumps(dict(parameters), sort_keys=True, separators=(",", ":"))


def _registered_tie_key(
    model_class: str,
    parameters: Mapping[str, object],
) -> tuple[object, ...]:
    if model_class == "hist_gbm":
        return (
            int(parameters["max_depth"]),
            int(parameters["max_iter"]),
            float(parameters["learning_rate"]),
            float(parameters.get("l2_regularization", 0.0)),
            int(parameters.get("min_samples_leaf", 20)),
        )
    if model_class == "random_forest":
        depth = parameters["max_depth"]
        return (
            float("inf") if depth is None else int(depth),
            int(parameters["n_estimators"]),
            int(parameters.get("min_samples_leaf", 1)),
            float(parameters.get("max_features", 1.0)),
        )
    if model_class == "poly2_ridge":
        return (float(parameters["alpha"]),)
    if model_class == "poly2_elastic_net":
        return (float(parameters["alpha"]), float(parameters["l1_ratio"]))
    raise ValueError(f"unregistered replica model class: {model_class}")


def _registered_estimator(
    model_class: str,
    parameters: Mapping[str, object],
) -> object:
    if sklearn.__version__ != REGISTERED_SKLEARN_VERSION:
        raise ValueError(
            "registered replica sklearn version mismatch; "
            f"expected={REGISTERED_SKLEARN_VERSION}, actual={sklearn.__version__}"
        )
    if model_class == "hist_gbm":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            quantile=None,
            learning_rate=float(parameters["learning_rate"]),
            max_iter=int(parameters["max_iter"]),
            max_leaf_nodes=31,
            max_depth=int(parameters["max_depth"]),
            min_samples_leaf=int(parameters.get("min_samples_leaf", 20)),
            l2_regularization=float(parameters.get("l2_regularization", 0.0)),
            max_features=1.0,
            max_bins=255,
            categorical_features=None,
            monotonic_cst=None,
            interaction_cst=None,
            warm_start=False,
            early_stopping=bool(parameters.get("early_stopping", False)),
            scoring="loss",
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=1e-7,
            verbose=0,
            random_state=REGISTERED_MODEL_RANDOM_STATE,
        )
    if model_class == "random_forest":
        depth = parameters["max_depth"]
        return RandomForestRegressor(
            n_estimators=int(parameters["n_estimators"]),
            criterion="squared_error",
            max_depth=None if depth is None else int(depth),
            min_samples_split=2,
            min_samples_leaf=int(parameters.get("min_samples_leaf", 1)),
            min_weight_fraction_leaf=0.0,
            max_features=float(parameters.get("max_features", 1.0)),
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=bool(parameters.get("bootstrap", True)),
            oob_score=False,
            n_jobs=1,
            random_state=REGISTERED_MODEL_RANDOM_STATE,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
            monotonic_cst=None,
        )
    if model_class == "poly2_ridge":
        model = Ridge(alpha=float(parameters["alpha"]), fit_intercept=True)
    elif model_class == "poly2_elastic_net":
        model = ElasticNet(
            alpha=float(parameters["alpha"]),
            l1_ratio=float(parameters["l1_ratio"]),
            fit_intercept=True,
            max_iter=int(parameters["max_iter"]),
            tol=float(parameters["tol"]),
            selection="cyclic",
            random_state=REGISTERED_MODEL_RANDOM_STATE,
        )
    else:
        raise ValueError(f"unregistered replica model class: {model_class}")
    return Pipeline(
        (
            (
                "poly2",
                FunctionTransformer(
                    _poly2_registered_design,
                    validate=False,
                ),
            ),
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            ("model", model),
        )
    )


def _fit_registered_estimator(
    model_class: str,
    parameters: Mapping[str, object],
    features: np.ndarray,
    target: np.ndarray,
) -> object:
    estimator = (
        Ridge(alpha=float(parameters["alpha"]), fit_intercept=True)
        if model_class == "linear_ridge"
        else _registered_estimator(model_class, parameters)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            estimator.fit(np.asarray(features, dtype=float), np.asarray(target, dtype=float))
        except ConvergenceWarning as error:
            raise RuntimeError(
                f"registered {model_class} failed to converge for "
                f"{_registered_parameter_key(parameters)}"
            ) from error
    return estimator


def _fit_registered_inner_split(
    model_class: str,
    parameters: Mapping[str, object],
    payload: Mapping[str, object],
) -> tuple[str, Mapping[str, object], dict[str, object]]:
    parameter_key = _registered_parameter_key(parameters)
    estimator = _fit_registered_estimator(
        model_class, parameters, payload["train_x"], payload["train_y"]
    )
    validation_prediction = estimator.predict(payload["validation_x"])
    residual = payload["validation_y"] - validation_prediction
    validation_sse = float(residual @ residual)
    return (
        parameter_key,
        dict(parameters),
        {
            "inner_split_idx": int(payload["inner_split_idx"]),
            "inner_train_start": payload["inner_train_start"],
            "inner_train_end": payload["inner_train_end"],
            "validation_start": payload["validation_start"],
            "validation_end": payload["validation_end"],
            "validation_rows": int(len(payload["validation_y"])),
            "validation_sse": validation_sse,
        },
    )


def _registered_score_row(
    payload: Mapping[str, object],
    validation_prediction: np.ndarray,
) -> dict[str, object]:
    residual = np.asarray(payload["validation_y"], dtype=float) - np.asarray(
        validation_prediction, dtype=float
    )
    return {
        "inner_split_idx": int(payload["inner_split_idx"]),
        "inner_train_start": payload["inner_train_start"],
        "inner_train_end": payload["inner_train_end"],
        "validation_start": payload["validation_start"],
        "validation_end": payload["validation_end"],
        "validation_rows": int(len(payload["validation_y"])),
        "validation_sse": float(residual @ residual),
    }


def _fit_registered_hist_gbm_prefix_family(
    parameters: Mapping[str, object],
    payload: Mapping[str, object],
) -> list[tuple[str, Mapping[str, object], dict[str, object]]]:
    max_iterations = 300
    configuration = dict(parameters, max_iter=max_iterations)
    estimator = _fit_registered_estimator(
        "hist_gbm",
        configuration,
        payload["train_x"],
        payload["train_y"],
    )
    wanted = {100, max_iterations}
    predictions = {
        iteration: prediction
        for iteration, prediction in enumerate(
            estimator.staged_predict(payload["validation_x"]), start=1
        )
        if iteration in wanted
    }
    if set(predictions) != wanted:
        raise RuntimeError("registered HGBM prefix predictions are incomplete")
    rows = []
    for max_iter in sorted(wanted):
        configuration = dict(parameters, max_iter=max_iter)
        rows.append(
            (
                _registered_parameter_key(configuration),
                configuration,
                _registered_score_row(payload, predictions[max_iter]),
            )
        )
    return rows


def _fit_registered_random_forest_prefix_family(
    parameters: Mapping[str, object],
    payload: Mapping[str, object],
) -> list[tuple[str, Mapping[str, object], dict[str, object]]]:
    max_estimators = 500
    configuration = dict(parameters, n_estimators=max_estimators)
    estimator = _fit_registered_estimator(
        "random_forest",
        configuration,
        payload["train_x"],
        payload["train_y"],
    )
    rows = []
    for n_estimators in (200, max_estimators):
        validation_prediction = np.mean(
            [
                tree.predict(payload["validation_x"])
                for tree in estimator.estimators_[:n_estimators]
            ],
            axis=0,
        )
        configuration = dict(parameters, n_estimators=n_estimators)
        rows.append(
            (
                _registered_parameter_key(configuration),
                configuration,
                _registered_score_row(payload, validation_prediction),
            )
        )
    return rows


def _registered_prefix_partition(
    model_class: str,
    tasks: Sequence[tuple[Mapping[str, object], Mapping[str, object]]],
) -> tuple[
    list[tuple[Mapping[str, object], Mapping[str, object]]],
    list[tuple[Mapping[str, object], Mapping[str, object]]],
]:
    if not tasks:
        return [], []
    if model_class == "hist_gbm":
        compatible = tuple(
            parameters
            for parameters in registered_replica_model_specs()[model_class]
            if set(parameters) == {"max_depth", "max_iter", "learning_rate"}
        )
    elif model_class == "random_forest":
        compatible = tuple(
            parameters
            for parameters in registered_replica_model_specs()[model_class]
            if set(parameters) == {"max_depth", "n_estimators"}
        )
    else:
        return [], list(tasks)
    expected = {_registered_parameter_key(parameters) for parameters in compatible}
    by_payload: dict[str, list[tuple[Mapping[str, object], Mapping[str, object]]]] = {}
    for parameters, payload in tasks:
        by_payload.setdefault(_registered_payload_identity(payload), []).append(
            (parameters, payload)
        )
    prefix_tasks: list[tuple[Mapping[str, object], Mapping[str, object]]] = []
    generic_tasks: list[tuple[Mapping[str, object], Mapping[str, object]]] = []
    for group in by_payload.values():
        payload = group[0][1]
        reusable = [
            (parameters, value)
            for parameters, value in group
            if _registered_parameter_key(parameters) in expected
        ]
        if (
            {_registered_parameter_key(parameters) for parameters, _ in reusable}
            != expected
            or len(reusable) != len(expected)
        ):
            generic_tasks.extend(group)
            continue
        generic_tasks.extend(
            (parameters, value)
            for parameters, value in group
            if _registered_parameter_key(parameters) not in expected
        )
        if model_class == "hist_gbm":
            family_parameters = {
                (int(parameters["max_depth"]), float(parameters["learning_rate"]))
                for parameters, _ in reusable
            }
            prefix_tasks.extend(
                ({"max_depth": depth, "learning_rate": rate}, payload)
                for depth, rate in sorted(family_parameters)
            )
        elif model_class == "random_forest":
            family_parameters = {
                int(parameters["max_depth"]) for parameters, _ in reusable
            }
            prefix_tasks.extend(
                ({"max_depth": depth}, payload) for depth in sorted(family_parameters)
            )
    return prefix_tasks, generic_tasks


def _registered_payload_identity(payload: Mapping[str, object]) -> str:
    """Identify one physical split without relying on Python object identity."""
    digest = hashlib.sha256()
    metadata = {
        key: str(payload.get(key))
        for key in (
            "inner_split_idx",
            "inner_train_start",
            "inner_train_end",
            "validation_start",
            "validation_end",
        )
    }
    digest.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
    for key in ("train_x", "train_y", "validation_x", "validation_y"):
        if key not in payload:
            continue
        values = np.ascontiguousarray(np.asarray(payload[key]))
        digest.update(key.encode("utf-8"))
        digest.update(str(values.dtype).encode("utf-8"))
        digest.update(str(values.shape).encode("utf-8"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def _registered_prefix_tasks(
    model_class: str,
    tasks: Sequence[tuple[Mapping[str, object], Mapping[str, object]]],
) -> list[tuple[Mapping[str, object], Mapping[str, object]]] | None:
    """Compatibility view used by workload accounting."""
    prefix_tasks, generic_tasks = _registered_prefix_partition(model_class, tasks)
    return prefix_tasks if prefix_tasks and not generic_tasks else None


def _evaluate_registered_inner_tasks(
    model_class: str,
    tasks: Sequence[tuple[Mapping[str, object], Mapping[str, object]]],
    *,
    fit_workers: int,
) -> list[tuple[str, Mapping[str, object], dict[str, object]]]:
    """Run the shared registered-model task scheduler."""
    workers = int(fit_workers)
    if workers < 1 or workers > 32:
        raise ValueError("fit_workers must be between 1 and 32")
    prefix_tasks, generic_tasks = _registered_prefix_partition(model_class, tasks)
    results: list[tuple[str, Mapping[str, object], dict[str, object]]] = []
    if prefix_tasks:
        fit_family = (
            _fit_registered_hist_gbm_prefix_family
            if model_class == "hist_gbm"
            else _fit_registered_random_forest_prefix_family
        )
        if workers == 1:
            nested = [
                fit_family(parameters, payload)
                for parameters, payload in prefix_tasks
            ]
        else:
            with ThreadPoolExecutor(
                max_workers=min(workers, len(prefix_tasks))
            ) as executor:
                futures = [
                    executor.submit(fit_family, parameters, payload)
                    for parameters, payload in prefix_tasks
                ]
                nested = [future.result() for future in futures]
        results.extend(result for family in nested for result in family)
    if not generic_tasks:
        return results
    if workers == 1:
        results.extend(
            _fit_registered_inner_split(model_class, parameters, payload)
            for parameters, payload in generic_tasks
        )
        return results
    with ThreadPoolExecutor(max_workers=min(workers, len(generic_tasks))) as executor:
        futures = [
            executor.submit(
                _fit_registered_inner_split,
                model_class,
                parameters,
                payload,
            )
            for parameters, payload in generic_tasks
        ]
        results.extend(future.result() for future in futures)
    return results


def benchmark_registered_replica_parallelism(
    features: np.ndarray,
    target: np.ndarray,
    *,
    fit_workers: int,
    repetitions: int = 3,
) -> pd.DataFrame:
    """Measure the production task scheduler without exposing model quality.

    The returned receipt contains only workload and timing metadata. Predictions,
    validation errors, selected models, and research statistics are discarded.
    """
    from time import perf_counter, process_time

    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("features and target must be aligned 2D/1D arrays")
    if len(x) < 40 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("parallelism benchmark requires at least 40 finite rows")
    if int(repetitions) < 1 or int(repetitions) > 3:
        raise ValueError("parallelism benchmark repetitions must be between 1 and 3")
    split = max(20, int(np.floor(len(x) * 0.75)))
    if split >= len(x):
        raise ValueError("parallelism benchmark validation split is empty")
    payload = {
        "inner_split_idx": 0,
        "inner_train_start": pd.Timestamp("2024-01-01", tz="UTC"),
        "inner_train_end": pd.Timestamp("2024-01-02", tz="UTC"),
        "validation_start": pd.Timestamp("2024-01-03", tz="UTC"),
        "validation_end": pd.Timestamp("2024-01-04", tz="UTC"),
        "train_x": x[:split],
        "train_y": y[:split],
        "validation_x": x[split:],
        "validation_y": y[split:],
    }
    payloads = [
        dict(payload, inner_split_idx=repeat_idx)
        for repeat_idx in range(int(repetitions))
    ]
    rows: list[dict[str, object]] = []
    for model_class in ("hist_gbm", "random_forest"):
        class_tasks = [
            (parameters, repeated_payload)
            for parameters in registered_replica_model_specs()[model_class]
            for repeated_payload in payloads
        ]
        prefix_tasks, generic_tasks = _registered_prefix_partition(
            model_class, class_tasks
        )
        cpu_started = process_time()
        wall_started = perf_counter()
        _evaluate_registered_inner_tasks(
            model_class, class_tasks, fit_workers=int(fit_workers)
        )
        rows.append(
            {
                "model_class": model_class,
                "fit_workers": int(fit_workers),
                "configuration_evaluation_count": len(class_tasks),
                "estimator_fit_count": (
                    len(prefix_tasks) + len(generic_tasks)
                ),
                "row_count": len(x),
                "feature_count": x.shape[1],
                "repetitions": int(repetitions),
                "cpu_seconds": process_time() - cpu_started,
                "wall_seconds": perf_counter() - wall_started,
                "performance_only": True,
            }
        )
    return pd.DataFrame(rows)


def _replication_r2(target: np.ndarray, prediction: np.ndarray, baseline: float) -> float:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    residual_ss = float(np.square(target - prediction).sum())
    baseline_ss = float(np.square(target - float(baseline)).sum())
    return 1.0 - residual_ss / baseline_ss if baseline_ss > 0.0 else float("nan")


def assemble_train_selected_registered_replica(
    class_predictions: pd.DataFrame,
    fold_selection: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble one outer-OOS stream from train-only fold selections."""
    prediction_keys = ["candidate_id", "fold_idx", "decision_ts", "symbol"]
    required_predictions = {
        *prediction_keys,
        "model_class",
        "target_signal",
        "replica_signal",
        "residual_signal",
        "strategy_forward_return",
    }
    missing_predictions = sorted(required_predictions.difference(class_predictions.columns))
    if missing_predictions:
        raise ValueError(
            "registered class predictions missing columns: "
            + ", ".join(missing_predictions)
        )
    required_selection = {
        "candidate_id",
        "fold_idx",
        "selected_model_class",
        "selection_source",
    }
    missing_selection = sorted(required_selection.difference(fold_selection.columns))
    if missing_selection:
        raise ValueError(
            "registered fold selection missing columns: " + ", ".join(missing_selection)
        )
    predictions = class_predictions.copy()
    predictions["candidate_id"] = predictions["candidate_id"].astype(str)
    predictions["model_class"] = predictions["model_class"].astype(str)
    predictions["fold_idx"] = pd.to_numeric(
        predictions["fold_idx"], errors="raise"
    ).astype(int)
    predictions["decision_ts"] = pd.to_datetime(
        predictions["decision_ts"], utc=True, errors="raise"
    )
    predictions["symbol"] = predictions["symbol"].astype(str)
    numeric = [
        "target_signal",
        "replica_signal",
        "residual_signal",
        "strategy_forward_return",
    ]
    predictions[numeric] = predictions[numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(predictions[numeric].to_numpy(dtype=float)).all():
        raise ValueError("registered class predictions contain non-finite values")
    if predictions.duplicated([*prediction_keys, "model_class"]).any():
        raise ValueError("registered class predictions contain duplicate class/key rows")
    recorded_residual = (
        predictions["target_signal"].to_numpy(dtype=float)
        - predictions["replica_signal"].to_numpy(dtype=float)
    )
    if not np.allclose(
        recorded_residual,
        predictions["residual_signal"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("registered class residual differs from target - replica")

    selection = fold_selection.copy()
    selection["candidate_id"] = selection["candidate_id"].astype(str)
    selection["fold_idx"] = pd.to_numeric(
        selection["fold_idx"], errors="raise"
    ).astype(int)
    selection["selected_model_class"] = selection["selected_model_class"].astype(str)
    if selection.duplicated(["candidate_id", "fold_idx"]).any():
        raise ValueError("registered fold selection contains duplicate candidate/fold rows")
    if set(selection["selection_source"].astype(str)) != {
        "outer_train_inner_validation"
    }:
        raise ValueError("registered fold selection was not produced train-only")
    if not set(selection["selected_model_class"]).issubset(
        set(REGISTERED_MODEL_CLASS_ORDER)
    ):
        raise ValueError("registered fold selection contains an unknown model class")

    expected_folds = predictions[["candidate_id", "fold_idx"]].drop_duplicates()
    actual_folds = selection[["candidate_id", "fold_idx"]].drop_duplicates()
    fold_audit = expected_folds.merge(
        actual_folds,
        on=["candidate_id", "fold_idx"],
        how="outer",
        indicator=True,
    )
    if set(fold_audit["_merge"]) != {"both"}:
        raise ValueError("registered predictions and fold selection disagree on folds")

    selected_parts: list[pd.DataFrame] = []
    for row in selection.itertuples(index=False):
        fold_predictions = predictions.loc[
            (predictions["candidate_id"] == row.candidate_id)
            & (predictions["fold_idx"] == row.fold_idx)
        ]
        class_keys = {
            model_class: group[prediction_keys].sort_values(
                prediction_keys, kind="mergesort"
            ).reset_index(drop=True)
            for model_class, group in fold_predictions.groupby("model_class", sort=True)
        }
        if set(class_keys) != set(REGISTERED_MODEL_CLASS_ORDER):
            raise ValueError("registered fold is missing a model class")
        reference = class_keys[REGISTERED_MODEL_CLASS_ORDER[0]]
        if any(not keys.equals(reference) for keys in class_keys.values()):
            raise ValueError("registered model classes disagree on outer-OOS keys")
        chosen = fold_predictions.loc[
            fold_predictions["model_class"] == row.selected_model_class
        ].copy()
        if chosen.empty:
            raise ValueError("selected registered model class has no predictions")
        chosen["source_model_class"] = str(row.selected_model_class)
        chosen["model_id"] = "train_selected_registered_replica"
        selected_parts.append(chosen)
    selected = pd.concat(selected_parts, ignore_index=True)
    if selected.duplicated(prediction_keys).any():
        raise ValueError("train-selected replica contains duplicate outer-OOS keys")
    return selected.sort_values(prediction_keys, kind="mergesort").reset_index(drop=True)


def summarize_registered_model_diagnostics(
    model_diagnostics: pd.DataFrame,
    *,
    expected_fold_count: int | None = None,
) -> pd.DataFrame:
    """Summarize registered-model selection and train/OOS diagnostics."""
    required = {
        "candidate_id",
        "fold_idx",
        "model_class",
        "inner_validation_sse",
        "outer_train_r2",
        "outer_oos_r2",
        "selected_model_class",
        "model_class_selected",
        "selection_source",
    }
    missing = sorted(required.difference(model_diagnostics.columns))
    if missing:
        raise ValueError(
            "registered model diagnostics missing columns: " + ", ".join(missing)
        )
    frame = model_diagnostics.copy()
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["model_class"] = frame["model_class"].astype(str)
    frame["selected_model_class"] = frame["selected_model_class"].astype(str)
    frame["fold_idx"] = pd.to_numeric(frame["fold_idx"], errors="raise").astype(int)
    if frame.duplicated(["candidate_id", "fold_idx", "model_class"]).any():
        raise ValueError("registered model diagnostics contain duplicate class/fold rows")
    if set(frame["selection_source"].astype(str)) != {
        "outer_train_inner_validation"
    }:
        raise ValueError("registered model diagnostics were not selected train-only")
    if not set(frame["model_class"]).issubset(set(REGISTERED_MODEL_CLASS_ORDER)):
        raise ValueError("registered model diagnostics contain an unknown model class")
    for (candidate_id, fold_idx), fold in frame.groupby(
        ["candidate_id", "fold_idx"], sort=True
    ):
        if set(fold["model_class"]) != set(REGISTERED_MODEL_CLASS_ORDER):
            raise ValueError(
                f"{candidate_id}/{fold_idx}: registered diagnostics are missing a class"
            )
        selected = fold["model_class_selected"].astype(bool)
        if int(selected.sum()) != 1:
            raise ValueError(
                f"{candidate_id}/{fold_idx}: registered diagnostics must select one class"
            )
        declared = fold["selected_model_class"].unique()
        actual = fold.loc[selected, "model_class"].iloc[0]
        if len(declared) != 1 or declared[0] != actual:
            raise ValueError(
                f"{candidate_id}/{fold_idx}: selected class declaration disagrees"
            )
    fold_counts = frame.groupby("candidate_id")["fold_idx"].nunique()
    if expected_fold_count is not None and not (
        fold_counts == int(expected_fold_count)
    ).all():
        raise ValueError(
            f"registered diagnostics must contain {int(expected_fold_count)} folds"
        )
    rows: list[dict[str, object]] = []
    for (candidate_id, model_class), group in frame.groupby(
        ["candidate_id", "model_class"], sort=True
    ):
        train_r2 = pd.to_numeric(group["outer_train_r2"], errors="coerce")
        oos_r2 = pd.to_numeric(group["outer_oos_r2"], errors="raise")
        if not np.isfinite(oos_r2.to_numpy(dtype=float)).all():
            raise ValueError("registered model diagnostics contain non-finite OOS R2")
        finite_train = train_r2[np.isfinite(train_r2)]
        mean_train = (
            float(finite_train.mean()) if not finite_train.empty else float("nan")
        )
        mean_oos = float(oos_r2.mean())
        rows.append(
            {
                "candidate_id": candidate_id,
                "model_class": model_class,
                "fold_count": int(group["fold_idx"].nunique()),
                "selected_fold_count": int(
                    group["model_class_selected"].astype(bool).sum()
                ),
                "mean_inner_validation_sse": float(
                    pd.to_numeric(
                        group["inner_validation_sse"], errors="raise"
                    ).mean()
                ),
                "mean_outer_train_r2": mean_train,
                "mean_outer_oos_r2": mean_oos,
                "mean_train_minus_oos_r2": (
                    mean_train - mean_oos
                    if np.isfinite(mean_train)
                    else float("nan")
                ),
                "selection_source": "outer_train_inner_validation",
            }
        )
    return pd.DataFrame(rows)


def fit_walk_forward_registered_replicas(
    canonical_frame: pd.DataFrame,
    folds: Sequence[object],
    *,
    candidate_id: str,
    frozen_ridge_predictions: pd.DataFrame,
    frozen_ridge_inner_scores: pd.DataFrame,
    target_column: str = "combo_signal",
    feature_columns: Sequence[str] = PRICE_VOLUME_COLUMNS,
    inner_gap: pd.Timedelta = pd.Timedelta(days=1),
    model_specs: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
    allow_model_subset: bool = False,
    fit_workers: int = 1,
    performance_timings: list[dict[str, object]] | None = None,
    timing_target: str = "unspecified",
) -> RegisteredModelReplicationArtifacts:
    """Fit registered nonlinear replicas and select a class inside each outer train."""
    if not isinstance(canonical_frame.index, pd.MultiIndex):
        raise ValueError("canonical_frame must use a MultiIndex")
    features = tuple(str(column) for column in feature_columns)
    if not features:
        raise ValueError("registered replica feature columns must not be empty")
    required = {target_column, *features}
    missing = sorted(required.difference(canonical_frame.columns))
    if missing:
        raise ValueError(
            "canonical_frame missing registered replica columns: " + ", ".join(missing)
        )
    specs = {
        str(model_class): tuple(dict(parameters) for parameters in configurations)
        for model_class, configurations in (
            registered_replica_model_specs() if model_specs is None else model_specs
        ).items()
    }
    registered_non_lineage_classes = set(REGISTERED_MODEL_CLASS_ORDER).difference(
        {"linear_ridge"}
    )
    if set(specs) != registered_non_lineage_classes:
        raise ValueError(
            "registered non-lineage model classes must be hist_gbm, random_forest, "
            "poly2_ridge, and poly2_elastic_net"
        )
    if any(not configurations for configurations in specs.values()):
        raise ValueError("registered nonlinear model class has an empty parameter grid")
    if int(fit_workers) < 1 or int(fit_workers) > 32:
        raise ValueError("fit_workers must be between 1 and 32")
    registered_specs = registered_replica_model_specs()
    for model_class, configurations in specs.items():
        actual_keys = {
            _registered_parameter_key(parameters) for parameters in configurations
        }
        registered_keys = {
            _registered_parameter_key(parameters)
            for parameters in registered_specs[model_class]
        }
        if not actual_keys.issubset(registered_keys):
            raise ValueError("registered replica parameter grid contains an unknown configuration")
        if not bool(allow_model_subset) and actual_keys != registered_keys:
            raise ValueError("registered replica parameter grid is incomplete")
    frozen_predictions = frozen_ridge_predictions.loc[
        (frozen_ridge_predictions["candidate_id"].astype(str) == str(candidate_id))
        & (frozen_ridge_predictions["model_id"].astype(str) == "level2_full")
    ].copy()
    frozen_scores = frozen_ridge_inner_scores.loc[
        (frozen_ridge_inner_scores["candidate_id"].astype(str) == str(candidate_id))
        & (frozen_ridge_inner_scores["model_id"].astype(str) == "level2_full")
    ].copy()
    if frozen_predictions.empty or frozen_scores.empty:
        raise ValueError("frozen level2_full ridge artifacts are missing")
    with _performance_stage(
        performance_timings,
        "ridge_lineage_audit",
        candidate_id=str(candidate_id),
        timing_target=str(timing_target),
        estimator_fit_count=len(folds),
    ):
        audit_frozen_level2_full_ridge_reproduction(
            canonical_frame,
            folds,
            candidate_id=str(candidate_id),
            frozen_predictions=frozen_predictions,
            frozen_inner_scores=frozen_scores,
            target_column=target_column,
            feature_columns=features,
            inner_gap=inner_gap,
        )

    prediction_parts: list[pd.DataFrame] = []
    inner_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    class_order = {name: index for index, name in enumerate(REGISTERED_MODEL_CLASS_ORDER)}
    for fold in folds:
        fold_artifacts = fit_walk_forward_registered_replica_fold(
            canonical_frame,
            fold,
            candidate_id=candidate_id,
            frozen_ridge_predictions=frozen_predictions,
            frozen_ridge_inner_scores=frozen_scores,
            target_column=target_column,
            feature_columns=features,
            inner_gap=inner_gap,
            model_specs=specs,
            fit_workers=fit_workers,
            performance_timings=performance_timings,
            timing_target=timing_target,
            class_order=class_order,
        )
        prediction_parts.extend(fold_artifacts.prediction_parts)
        inner_rows.extend(fold_artifacts.inner_rows)
        diagnostic_rows.extend(fold_artifacts.diagnostic_rows)
        selection_rows.append(fold_artifacts.selection_row)
    class_predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["candidate_id", "model_class", "fold_idx", "decision_ts", "symbol"],
        kind="mergesort",
    ).reset_index(drop=True)
    fold_selection = pd.DataFrame(selection_rows).sort_values(
        ["candidate_id", "fold_idx"], kind="mergesort"
    ).reset_index(drop=True)
    selected_predictions = assemble_train_selected_registered_replica(
        class_predictions, fold_selection
    )
    return RegisteredModelReplicationArtifacts(
        class_predictions=class_predictions,
        inner_scores=pd.DataFrame(inner_rows),
        model_diagnostics=pd.DataFrame(diagnostic_rows),
        fold_selection=fold_selection,
        selected_predictions=selected_predictions,
    )


def registered_replica_fold_inner_tasks(
    model_class, model_specs, split_payloads
) -> list[tuple[str, Mapping[str, object], dict[str, object]]]:
    """Return the ordered fine-grained inner-fit unit descriptors for one
    (model_class, fold).  Each unit is ``(kind, parameters, payload)`` where
    ``kind`` is ``"prefix"`` or ``"generic"``.  The order matches the
    sequential fold body exactly: parameter_key order then split order, with
    prefix families first (as produced by ``_registered_prefix_partition``).
    """
    tasks = [
        (parameters, payload)
        for parameters in model_specs
        for payload in split_payloads
    ]
    prefix_tasks, generic_tasks = _registered_prefix_partition(model_class, tasks)
    units: list[tuple[str, Mapping[str, object], dict[str, object]]] = []
    units.extend(
        ("prefix", parameters, payload) for parameters, payload in prefix_tasks
    )
    units.extend(
        ("generic", parameters, payload) for parameters, payload in generic_tasks
    )
    return units


def registered_replica_physical_workload_units(
    model_specs: Mapping[str, Sequence[Mapping[str, object]]],
    split_payloads: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Enumerate the physical inner-fit categories used by formal execution.

    Prefix units represent one estimator fit that emits two configuration rows;
    generic units represent one estimator fit and one configuration row.  The
    returned descriptors retain the exact qlab task partition and are intended for
    performance-only calibration, not scientific scoring.
    """
    expected_classes = set(REGISTERED_MODEL_CLASS_ORDER[1:])
    if set(model_specs) != expected_classes:
        raise ValueError("physical workload model classes are incomplete")
    units: list[dict[str, object]] = []
    for model_class in REGISTERED_MODEL_CLASS_ORDER[1:]:
        for ordinal, (kind, parameters, payload) in enumerate(
            registered_replica_fold_inner_tasks(
                model_class, model_specs[model_class], split_payloads
            )
        ):
            configuration_count = 2 if str(kind) == "prefix" else 1
            units.append(
                {
                    "physical_category": f"{model_class}:{kind}",
                    "model_class": str(model_class),
                    "kind": str(kind),
                    "ordinal": int(ordinal),
                    "parameters": dict(parameters),
                    "payload": payload,
                    "configuration_count": configuration_count,
                }
            )
    return units


def benchmark_registered_replica_workload(
    features: np.ndarray,
    target: np.ndarray,
    *,
    profile,
    repetitions: int = 1,
) -> pd.DataFrame:
    """Measure every registered physical inner-task category performance-only."""
    from qlab.execution.pool import WorkPool

    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("features and target must be aligned 2D/1D arrays")
    if len(x) < 40 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("workload benchmark requires at least 40 finite rows")
    if int(repetitions) < 1 or int(repetitions) > 3:
        raise ValueError("workload benchmark repetitions must be between 1 and 3")
    split = max(20, int(np.floor(len(x) * 0.75)))
    if split >= len(x):
        raise ValueError("workload benchmark validation split is empty")
    base_payload = {
        "inner_split_idx": 0,
        "inner_train_start": pd.Timestamp("2024-01-01", tz="UTC"),
        "inner_train_end": pd.Timestamp("2024-01-02", tz="UTC"),
        "validation_start": pd.Timestamp("2024-01-03", tz="UTC"),
        "validation_end": pd.Timestamp("2024-01-04", tz="UTC"),
        "train_x": x[:split],
        "train_y": y[:split],
        "validation_x": x[split:],
        "validation_y": y[split:],
    }
    payloads = [
        dict(base_payload, inner_split_idx=repeat_idx)
        for repeat_idx in range(int(repetitions))
    ]
    descriptors = registered_replica_physical_workload_units(
        registered_replica_model_specs(), payloads
    )
    grouped: dict[str, list[dict[str, object]]] = {}
    for descriptor in descriptors:
        grouped.setdefault(str(descriptor["physical_category"]), []).append(descriptor)
    expanded_feature_counts = {
        model_class: (
            int(_poly2_registered_design(x[:1]).shape[1])
            if str(model_class).startswith("poly2_")
            else int(x.shape[1])
        )
        for model_class in REGISTERED_MODEL_CLASS_ORDER[1:]
    }
    rows: list[dict[str, object]] = []
    with WorkPool(profile=profile) as pool:
        for category in sorted(grouped):
            category_units = grouped[category]
            started_cpu = process_time()
            started_wall = perf_counter()
            pool.map(
                [
                        (
                            _evaluate_registered_physical_unit,
                            (
                                descriptor["model_class"],
                                descriptor["parameters"],
                                descriptor["payload"],
                                descriptor["kind"],
                            ),
                        )
                    for descriptor in category_units
                ]
            )
            model_class = str(category_units[0]["model_class"])
            rows.append(
                {
                    "physical_category": category,
                    "model_class": model_class,
                    "kind": str(category_units[0]["kind"]),
                    "fit_workers": int(profile.workers),
                    "native_threads": int(profile.native_threads),
                    "physical_task_count": len(category_units),
                    "configuration_evaluation_count": int(
                        sum(int(item["configuration_count"]) for item in category_units)
                    ),
                    "estimator_fit_count": len(category_units),
                    "row_count": len(x),
                    "feature_count": int(x.shape[1]),
                    "expanded_feature_count": expanded_feature_counts[model_class],
                    "repetitions": int(repetitions),
                    "cpu_seconds": process_time() - started_cpu,
                    "wall_seconds": perf_counter() - started_wall,
                    "performance_only": True,
                }
            )
    return pd.DataFrame(rows)


def _evaluate_registered_physical_unit(model_class, parameters, payload, kind):
    return evaluate_registered_replica_inner_task(
        model_class, parameters, payload, kind=kind
    )


def evaluate_registered_replica_inner_task(
    model_class,
    parameters,
    payload,
    *,
    kind,
) -> list[tuple[str, Mapping[str, object], dict[str, object]]]:
    """Evaluate one fine-grained inner-fit unit and return its scored rows.

    ``kind`` must be the value produced by ``registered_replica_fold_inner_tasks``
    (``"prefix"`` or ``"generic"``).  Returns the same
    ``(parameter_key, parameters, score_row)`` tuples that the sequential fold
    body collects, so the caller can feed them directly into
    ``finalize_registered_replica_fold_class``.
    """
    model_class = str(model_class)
    if str(kind) == "prefix":
        family = (
            _fit_registered_hist_gbm_prefix_family
            if model_class == "hist_gbm"
            else _fit_registered_random_forest_prefix_family
        )
        return family(dict(parameters), payload)
    if str(kind) == "generic":
        return [_fit_registered_inner_split(model_class, dict(parameters), payload)]
    raise ValueError(f"unknown registered inner task kind: {kind}")


def finalize_registered_replica_fold_class(
    model_class,
    evaluated_rows,
    *,
    fold_idx,
    candidate_id,
    train_features,
    test_features,
    train_target,
    test_target,
    train_baseline,
) -> RegisteredReplicaFoldClassFit:
    """Aggregate one (model_class, fold) inner scores, select the best
    configuration, and perform the outer final fit.  Mirrors the sequential
    fold body exactly (configuration ordering, tie-break, diagnostics).
    """
    model_class = str(model_class)
    by_configuration: dict[str, dict[str, object]] = {}
    for parameter_key, parameters, score_row in evaluated_rows:
        entry = by_configuration.setdefault(
            parameter_key,
            {
                "parameters": parameters,
                "score_rows": [],
            },
        )
        entry["score_rows"].append(score_row)
    configuration_results: list[tuple[float, str, Mapping[str, object]]] = []
    inner_rows: list[dict[str, object]] = []
    for parameter_key in sorted(by_configuration):
        entry = by_configuration[parameter_key]
        parameters = entry["parameters"]
        score_rows = sorted(
            entry["score_rows"], key=lambda value: value["inner_split_idx"]
        )
        total_sse = float(
            sum(float(value["validation_sse"]) for value in score_rows)
        )
        configuration_results.append((total_sse, parameter_key, parameters))
        for score_row in score_rows:
            score_row.update(
                {
                    "candidate_id": str(candidate_id),
                    "model_class": model_class,
                    "fold_idx": int(fold_idx),
                    "hyperparameters_json": parameter_key,
                }
            )
        inner_rows.extend(score_rows)
    best_sse = min(value[0] for value in configuration_results)
    best_configurations = [
        value
        for value in configuration_results
        if np.isclose(value[0], best_sse, rtol=1e-12, atol=1e-12)
    ]
    _, best_parameter_key, best_parameters = min(
        best_configurations,
        key=lambda value: _registered_tie_key(model_class, value[2]),
    )
    estimator = _fit_registered_estimator(
        model_class,
        best_parameters,
        np.asarray(train_features, dtype=float),
        np.asarray(train_target, dtype=float),
    )
    train_prediction = estimator.predict(np.asarray(train_features, dtype=float))
    test_prediction = estimator.predict(np.asarray(test_features, dtype=float))
    class_candidate = {
        "model_class": model_class,
        "inner_validation_sse": float(best_sse),
        "hyperparameters_json": best_parameter_key,
        "outer_train_r2": _replication_r2(
            np.asarray(train_target, dtype=float),
            train_prediction,
            float(train_baseline),
        ),
        "outer_oos_r2": _replication_r2(
            np.asarray(test_target, dtype=float),
            test_prediction,
            float(train_baseline),
        ),
    }
    return RegisteredReplicaFoldClassFit(
        model_class=model_class,
        best_parameter_key=best_parameter_key,
        class_candidate=class_candidate,
        inner_rows=inner_rows,
        test_prediction=test_prediction,
    )


def registered_replica_fold_ridge_artifacts(
    *,
    candidate_id,
    fold_idx,
    frozen_score_rows,
    frozen_prediction_rows,
    test_frame,
    test_target,
    train_baseline,
) -> RegisteredReplicaFoldRidgeArtifacts:
    """Build the frozen level2_full ridge artifacts for one outer fold.

    Returns the prediction frame, the linear_ridge class candidate, and the
    ridge inner rows exactly as the sequential fold body produces them.
    """
    fold_idx = int(fold_idx)
    ridge_score_rows = frozen_score_rows.copy()
    if ridge_score_rows.empty:
        raise ValueError(f"frozen ridge inner scores missing fold {fold_idx}")
    ridge_totals = (
        ridge_score_rows.groupby("alpha", as_index=False)["validation_sse"]
        .sum()
        .sort_values("alpha")
    )
    ridge_best_sse = float(ridge_totals["validation_sse"].min())
    ridge_best_alpha = float(
        ridge_totals.loc[
            np.isclose(
                ridge_totals["validation_sse"],
                ridge_best_sse,
                rtol=1e-12,
                atol=1e-12,
            ),
            "alpha",
        ].max()
    )
    inner_rows: list[dict[str, object]] = []
    for score_row in ridge_score_rows.itertuples(index=False):
        inner_rows.append(
            {
                "candidate_id": str(candidate_id),
                "model_class": "linear_ridge",
                "fold_idx": fold_idx,
                "inner_split_idx": int(score_row.inner_split_idx),
                "hyperparameters_json": _registered_parameter_key(
                    {"alpha": float(score_row.alpha)}
                ),
                "inner_train_start": score_row.inner_train_start,
                "inner_train_end": score_row.inner_train_end,
                "validation_start": score_row.validation_start,
                "validation_end": score_row.validation_end,
                "validation_rows": int(score_row.validation_rows),
                "validation_sse": float(score_row.validation_sse),
            }
        )
    ridge_fold = frozen_prediction_rows.copy()
    if ridge_fold.empty:
        raise ValueError(f"frozen ridge predictions missing fold {fold_idx}")
    ridge_fold["decision_ts"] = pd.to_datetime(
        ridge_fold["decision_ts"], utc=True, errors="raise"
    )
    ridge_fold["symbol"] = ridge_fold["symbol"].astype(str)
    ridge_fold = ridge_fold.sort_values(
        ["decision_ts", "symbol"], kind="mergesort"
    ).reset_index(drop=True)
    expected_keys = test_frame.reset_index()[["decision_ts", "symbol"]].copy()
    expected_keys["decision_ts"] = pd.to_datetime(
        expected_keys["decision_ts"], utc=True, errors="raise"
    )
    expected_keys["symbol"] = expected_keys["symbol"].astype(str)
    expected_keys = expected_keys.sort_values(
        ["decision_ts", "symbol"], kind="mergesort"
    ).reset_index(drop=True)
    if not ridge_fold[["decision_ts", "symbol"]].equals(expected_keys):
        raise ValueError("frozen ridge and canonical outer-OOS keys disagree")
    if not np.allclose(
        ridge_fold["target_signal"].to_numpy(dtype=float),
        np.asarray(test_target, dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("frozen ridge and canonical outer-OOS targets disagree")
    recorded_alpha = ridge_fold["selected_alpha"].astype(float).unique()
    if len(recorded_alpha) != 1 or not np.isclose(
        recorded_alpha[0], ridge_best_alpha, rtol=1e-12, atol=1e-12
    ):
        raise ValueError("frozen ridge selected alpha disagrees with inner scores")
    ridge_output = ridge_fold.copy()
    ridge_output["model_class"] = "linear_ridge"
    ridge_output["source_model_id"] = "level2_full"
    ridge_output["hyperparameters_json"] = _registered_parameter_key(
        {"alpha": ridge_best_alpha}
    )
    ridge_outer_r2 = _replication_r2(
        ridge_fold["target_signal"].to_numpy(dtype=float),
        ridge_fold["replica_signal"].to_numpy(dtype=float),
        float(train_baseline),
    )
    class_candidate = {
        "model_class": "linear_ridge",
        "inner_validation_sse": ridge_best_sse,
        "hyperparameters_json": _registered_parameter_key(
            {"alpha": ridge_best_alpha}
        ),
        "outer_train_r2": float("nan"),
        "outer_oos_r2": ridge_outer_r2,
    }
    return RegisteredReplicaFoldRidgeArtifacts(
        prediction_frame=ridge_output,
        class_candidate=class_candidate,
        inner_rows=inner_rows,
    )


def registered_replica_fold_class_output_frame(
    *,
    test_frame,
    model_class,
    candidate_id,
    fold_idx,
    test_target,
    train_baseline,
    best_parameter_key,
    test_prediction,
) -> pd.DataFrame:
    """Assemble the class output prediction frame for one (class, fold)."""
    output = test_frame[["forward_return", "strategy_forward_return"]].copy()
    output["candidate_id"] = str(candidate_id)
    output["model_id"] = model_class
    output["model_class"] = model_class
    output["source_model_id"] = model_class
    output["fold_idx"] = int(fold_idx)
    output["target_signal"] = np.asarray(test_target, dtype=float)
    output["replica_signal"] = np.asarray(test_prediction, dtype=float)
    output["residual_signal"] = (
        np.asarray(test_target, dtype=float)
        - np.asarray(test_prediction, dtype=float)
    )
    output["fold_train_target_mean"] = float(train_baseline)
    output["selected_alpha"] = np.nan
    output["hyperparameters_json"] = str(best_parameter_key)
    return output.reset_index()


def registered_replica_fold_winner(
    class_candidates, class_order
) -> dict[str, object]:
    """Select the winning class candidate for one fold (min inner SSE with
    model-class order tie-break)."""
    winner_sse = min(
        float(candidate["inner_validation_sse"])
        for candidate in class_candidates
    )
    winner = min(
        (
            candidate
            for candidate in class_candidates
            if np.isclose(
                float(candidate["inner_validation_sse"]),
                winner_sse,
                rtol=1e-12,
                atol=1e-12,
            )
        ),
        key=lambda candidate: class_order[str(candidate["model_class"])],
    )
    return dict(winner)


def registered_replica_fold_diagnostic_rows(
    class_candidates, winner, *, candidate_id, fold_idx
) -> list[dict[str, object]]:
    """Build the model_diagnostics rows for one fold, in candidate order."""
    rows: list[dict[str, object]] = []
    for candidate in class_candidates:
        rows.append(
            {
                "candidate_id": str(candidate_id),
                "fold_idx": int(fold_idx),
                **candidate,
                "selected_model_class": str(winner["model_class"]),
                "model_class_selected": (
                    str(candidate["model_class"]) == str(winner["model_class"])
                ),
                "selection_source": "outer_train_inner_validation",
            }
        )
    return rows


def registered_replica_fold_selection_row(
    winner, *, candidate_id, fold_idx
) -> dict[str, object]:
    """Build the fold_selection row for one fold."""
    return {
        "candidate_id": str(candidate_id),
        "fold_idx": int(fold_idx),
        "selected_model_class": str(winner["model_class"]),
        "selected_hyperparameters_json": str(winner["hyperparameters_json"]),
        "selected_inner_validation_sse": float(winner["inner_validation_sse"]),
        "selection_source": "outer_train_inner_validation",
    }


def fit_walk_forward_registered_replica_fold(
    canonical_frame,
    fold,
    *,
    candidate_id,
    frozen_ridge_predictions,
    frozen_ridge_inner_scores,
    target_column="combo_signal",
    feature_columns=PRICE_VOLUME_COLUMNS,
    inner_gap="1d",
    model_specs=None,
    fit_workers=1,
    performance_timings=None,
    timing_target="unspecified",
    class_order=None,
) -> RegisteredReplicaFoldArtifacts:
    """Run the registered-replica model grid for one outer fold and return
    the fold artifacts (prediction parts, inner rows, class candidates,
    diagnostics, selection row, winner).

    This is the single per-fold driver used by both the sequential
    ``fit_walk_forward_registered_replicas`` path and the distributed
    execution path.  Inner fit units are evaluated sequentially here; the
    distributed path evaluates the same units via
    ``evaluate_registered_replica_inner_task`` and feeds the rows into
    ``finalize_registered_replica_fold_class``.
    """
    fold_idx = int(fold.fold_idx)
    features = tuple(str(c) for c in feature_columns)
    specs = {
        str(model_class): tuple(dict(parameters) for parameters in configurations)
        for model_class, configurations in (
            registered_replica_model_specs()
            if model_specs is None
            else model_specs
        ).items()
    }
    if class_order is None:
        class_order = {
            name: index for index, name in enumerate(REGISTERED_MODEL_CLASS_ORDER)
        }
    if {"fold_idx", "split"}.issubset(canonical_frame.index.names):
        try:
            train_frame = canonical_frame.xs(
                (fold_idx, "train"), level=("fold_idx", "split")
            ).sort_index()
            test_frame = canonical_frame.xs(
                (fold_idx, "test"), level=("fold_idx", "split")
            ).sort_index()
        except KeyError as error:
            raise ValueError(f"canonical frame is missing fold {fold_idx}") from error
    else:
        train_frame = canonical_frame.loc[
            _date_mask(canonical_frame, fold.train_start, fold.train_end)
        ].sort_index()
        test_frame = canonical_frame.loc[
            _date_mask(canonical_frame, fold.test_start, fold.test_end)
        ].sort_index()
    if train_frame.empty or test_frame.empty:
        raise ValueError(f"registered replica fold {fold_idx} is empty")
    splits = _inner_time_splits(
        pd.DatetimeIndex(train_frame.index.get_level_values("decision_ts")),
        gap=pd.Timedelta(inner_gap),
    )
    train_dates = pd.DatetimeIndex(
        train_frame.index.get_level_values("decision_ts")
    )
    train_target = train_frame[target_column].to_numpy(dtype=float)
    test_target = test_frame[target_column].to_numpy(dtype=float)
    train_baseline = float(train_frame[target_column].mean())
    train_features = train_frame[list(features)].to_numpy(dtype=float)
    test_features = test_frame[list(features)].to_numpy(dtype=float)
    split_payloads: list[dict[str, object]] = []
    for split_idx, (inner_train_dates, validation_dates) in enumerate(splits):
        inner_train = train_frame.loc[train_dates.isin(inner_train_dates)]
        validation = train_frame.loc[train_dates.isin(validation_dates)]
        split_payloads.append(
            {
                "inner_split_idx": split_idx,
                "inner_train_start": inner_train_dates.min(),
                "inner_train_end": inner_train_dates.max(),
                "validation_start": validation_dates.min(),
                "validation_end": validation_dates.max(),
                "train_x": inner_train[list(features)].to_numpy(dtype=float),
                "train_y": inner_train[target_column].to_numpy(dtype=float),
                "validation_x": validation[list(features)].to_numpy(dtype=float),
                "validation_y": validation[target_column].to_numpy(dtype=float),
            }
        )

    prediction_parts: list[pd.DataFrame] = []
    inner_rows: list[dict[str, object]] = []
    class_candidates: list[dict[str, object]] = []

    ridge_artifacts = registered_replica_fold_ridge_artifacts(
        candidate_id=candidate_id,
        fold_idx=fold_idx,
        frozen_score_rows=frozen_ridge_inner_scores.loc[
            frozen_ridge_inner_scores["fold_idx"] == fold_idx
        ].copy(),
        frozen_prediction_rows=frozen_ridge_predictions.loc[
            frozen_ridge_predictions["fold_idx"] == fold_idx
        ].copy(),
        test_frame=test_frame,
        test_target=test_target,
        train_baseline=train_baseline,
    )
    prediction_parts.append(ridge_artifacts.prediction_frame)
    inner_rows.extend(ridge_artifacts.inner_rows)
    class_candidates.append(ridge_artifacts.class_candidate)

    for model_class in REGISTERED_MODEL_CLASS_ORDER[1:]:
        tasks = [
            (parameters, payload)
            for parameters in specs[model_class]
            for payload in split_payloads
        ]
        prefix_tasks, generic_tasks = _registered_prefix_partition(model_class, tasks)
        estimator_fit_count = len(prefix_tasks) + len(generic_tasks)
        with _performance_stage(
            performance_timings,
            f"registered_{model_class}_inner_selection",
            candidate_id=str(candidate_id),
            timing_target=str(timing_target),
            fold_idx=fold_idx,
            estimator_fit_count=estimator_fit_count,
        ):
            evaluated_splits = _evaluate_registered_inner_tasks(
                model_class, tasks, fit_workers=int(fit_workers)
            )
        with _performance_stage(
            performance_timings,
            f"registered_{model_class}_outer_final_fit",
            candidate_id=str(candidate_id),
            timing_target=str(timing_target),
            fold_idx=fold_idx,
            estimator_fit_count=1,
        ):
            class_fit = finalize_registered_replica_fold_class(
                model_class,
                evaluated_splits,
                fold_idx=fold_idx,
                candidate_id=str(candidate_id),
                train_features=train_features,
                test_features=test_features,
                train_target=train_target,
                test_target=test_target,
                train_baseline=train_baseline,
            )
        prediction_parts.append(
            registered_replica_fold_class_output_frame(
                test_frame=test_frame,
                model_class=class_fit.model_class,
                candidate_id=str(candidate_id),
                fold_idx=fold_idx,
                test_target=test_target,
                train_baseline=train_baseline,
                best_parameter_key=class_fit.best_parameter_key,
                test_prediction=class_fit.test_prediction,
            )
        )
        inner_rows.extend(class_fit.inner_rows)
        class_candidates.append(class_fit.class_candidate)

    winner = registered_replica_fold_winner(class_candidates, class_order)
    diagnostic_rows = registered_replica_fold_diagnostic_rows(
        class_candidates, winner, candidate_id=candidate_id, fold_idx=fold_idx
    )
    selection_row = registered_replica_fold_selection_row(
        winner, candidate_id=candidate_id, fold_idx=fold_idx
    )
    return RegisteredReplicaFoldArtifacts(
        fold_idx=fold_idx,
        prediction_parts=prediction_parts,
        inner_rows=inner_rows,
        class_candidates=class_candidates,
        diagnostic_rows=diagnostic_rows,
        selection_row=selection_row,
        winner=winner,
    )


def fit_same_sample_registered_replicas(
    canonical_frame: pd.DataFrame,
    folds: Sequence[object],
    fold_selection: pd.DataFrame,
    *,
    target_column: str = "combo_signal",
    feature_columns: Sequence[str] = PRICE_VOLUME_COLUMNS,
) -> pd.DataFrame:
    """Fit each R5-selected model on its outer test and predict those same rows.

    This is the deliberately same-sample R3 comparator.  Model class and
    hyperparameters are inherited exactly from the train-only R5 selection;
    this entry never performs model or hyperparameter selection.
    """
    if not isinstance(canonical_frame.index, pd.MultiIndex):
        raise ValueError("canonical_frame must use a MultiIndex")
    features = tuple(str(column) for column in feature_columns)
    if not features:
        raise ValueError("same-sample registered replica feature columns must not be empty")
    required_frame = {
        target_column,
        *features,
        "forward_return",
        "strategy_forward_return",
    }
    missing_frame = sorted(required_frame.difference(canonical_frame.columns))
    if missing_frame:
        raise ValueError(
            "canonical_frame missing same-sample registered replica columns: "
            + ", ".join(missing_frame)
        )

    required_selection = {
        "candidate_id",
        "fold_idx",
        "selected_model_class",
        "selected_hyperparameters_json",
        "selection_source",
    }
    missing_selection = sorted(required_selection.difference(fold_selection.columns))
    if missing_selection:
        raise ValueError(
            "registered fold selection missing columns: " + ", ".join(missing_selection)
        )
    selection = fold_selection.copy()
    selection["candidate_id"] = selection["candidate_id"].astype(str)
    selection["fold_idx"] = pd.to_numeric(selection["fold_idx"], errors="raise").astype(int)
    selection["selected_model_class"] = selection["selected_model_class"].astype(str)
    if selection.empty:
        raise ValueError("registered fold selection is empty")
    if selection.duplicated(["candidate_id", "fold_idx"]).any():
        raise ValueError("registered fold selection contains duplicate candidate/fold rows")
    candidate_ids = selection["candidate_id"].unique()
    if len(candidate_ids) != 1:
        raise ValueError("same-sample registered replica requires exactly one candidate")
    if set(selection["selection_source"].astype(str)) != {
        "outer_train_inner_validation"
    }:
        raise ValueError("registered fold selection was not produced train-only")
    if not set(selection["selected_model_class"]).issubset(
        set(REGISTERED_MODEL_CLASS_ORDER)
    ):
        raise ValueError("registered fold selection contains an unknown model class")

    fold_ids = [int(fold.fold_idx) for fold in folds]
    if len(fold_ids) != len(set(fold_ids)):
        raise ValueError("folds contain duplicate fold_idx values")
    if set(selection["fold_idx"]) != set(fold_ids):
        raise ValueError("registered fold selection and folds disagree")

    parsed_parameters: dict[int, Mapping[str, object]] = {}
    for row in selection.itertuples(index=False):
        try:
            parameters = json.loads(str(row.selected_hyperparameters_json))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("registered fold selection parameters are not valid JSON") from error
        if not isinstance(parameters, dict):
            raise ValueError("registered fold selection parameters must be a JSON object")
        parameter_key = _registered_parameter_key(parameters)
        valid_keys = {
            _registered_parameter_key(configuration)
            for configuration in _registered_model_parameter_specs()[
                row.selected_model_class
            ]
        }
        if parameter_key not in valid_keys:
            raise ValueError(
                "registered fold selection contains invalid parameters for "
                f"{row.selected_model_class}"
            )
        parsed_parameters[int(row.fold_idx)] = parameters

    candidate_id = str(candidate_ids[0])
    selection_by_fold = selection.set_index("fold_idx")
    prediction_parts: list[pd.DataFrame] = []
    for fold in folds:
        fold_idx = int(fold.fold_idx)
        if {"fold_idx", "split"}.issubset(canonical_frame.index.names):
            try:
                test_frame = canonical_frame.xs(
                    (fold_idx, "test"), level=("fold_idx", "split")
                ).sort_index()
            except KeyError as error:
                raise ValueError(f"canonical frame is missing test fold {fold_idx}") from error
        else:
            test_frame = canonical_frame.loc[
                _date_mask(canonical_frame, fold.test_start, fold.test_end)
            ].sort_index()
        if test_frame.empty:
            raise ValueError(f"same-sample registered replica test fold {fold_idx} is empty")
        numeric_columns = [target_column, *features, "forward_return", "strategy_forward_return"]
        numeric = test_frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(
                f"same-sample registered replica test fold {fold_idx} contains non-finite values"
            )

        selection_row = selection_by_fold.loc[fold_idx]
        model_class = str(selection_row["selected_model_class"])
        parameters = parsed_parameters[fold_idx]
        x = numeric[list(features)].to_numpy(dtype=float)
        target = numeric[target_column].to_numpy(dtype=float)
        if model_class == "linear_ridge":
            intercept, beta = _fit_ridge(x, target, float(parameters["alpha"]))
            replica = intercept + x @ beta
        else:
            estimator = _fit_registered_estimator(model_class, parameters, x, target)
            replica = estimator.predict(x)
        if not np.isfinite(replica).all():
            raise ValueError("same-sample registered replica produced non-finite predictions")

        output = test_frame[["forward_return", "strategy_forward_return"]].copy()
        output["candidate_id"] = candidate_id
        output["model_id"] = "same_sample_registered_replica"
        output["model_class"] = model_class
        output["source_model_class"] = model_class
        output["source_model_id"] = model_class
        output["fold_idx"] = fold_idx
        output["target_signal"] = target
        output["replica_signal"] = replica
        output["residual_signal"] = target - replica
        output["fold_train_target_mean"] = float("nan")
        output["selected_alpha"] = (
            float(parameters["alpha"]) if model_class == "linear_ridge" else np.nan
        )
        output["hyperparameters_json"] = _registered_parameter_key(parameters)
        output["prediction_scope"] = "outer_test_same_sample"
        prediction_parts.append(output.reset_index())

    predictions = pd.concat(prediction_parts, ignore_index=True)
    prediction_keys = ["candidate_id", "fold_idx", "decision_ts", "symbol"]
    if predictions.duplicated(prediction_keys).any():
        raise ValueError("same-sample registered replica contains duplicate prediction keys")
    return predictions.sort_values(prediction_keys, kind="mergesort").reset_index(drop=True)


def fit_walk_forward_ridge_replicas(
    canonical_frame: pd.DataFrame,
    folds: Sequence[object],
    *,
    candidate_id: str,
    model_features: Mapping[str, Sequence[str]] | None = None,
    target_column: str = "combo_signal",
    alpha_grid: Sequence[float] = ALPHA_GRID,
    inner_gap: pd.Timedelta = pd.Timedelta(days=1),
) -> RidgeReplicationArtifacts:
    if not isinstance(canonical_frame.index, pd.MultiIndex):
        raise ValueError("canonical_frame must use a MultiIndex")
    models = dict(model_feature_sets() if model_features is None else model_features)
    if not models:
        raise ValueError("model_features must not be empty")
    required = {target_column, *(feature for features in models.values() for feature in features)}
    missing = sorted(required.difference(canonical_frame.columns))
    if missing:
        raise ValueError("canonical_frame missing ridge columns: " + ", ".join(missing))
    prediction_frames: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, object]] = []
    inner_rows: list[dict[str, object]] = []
    for model_id, raw_features in models.items():
        features = tuple(raw_features)
        for fold in folds:
            if {"fold_idx", "split"}.issubset(canonical_frame.index.names):
                try:
                    train_frame = canonical_frame.xs(
                        (int(fold.fold_idx), "train"), level=("fold_idx", "split")
                    )
                    test_frame = canonical_frame.xs(
                        (int(fold.fold_idx), "test"), level=("fold_idx", "split")
                    )
                except KeyError:
                    continue
            else:
                train_mask = _date_mask(canonical_frame, fold.train_start, fold.train_end)
                test_mask = _date_mask(canonical_frame, fold.test_start, fold.test_end)
                train_frame = canonical_frame.loc[train_mask]
                test_frame = canonical_frame.loc[test_mask]
            if train_frame.empty or test_frame.empty:
                continue
            splits = _inner_time_splits(
                pd.DatetimeIndex(train_frame.index.get_level_values("decision_ts")),
                gap=pd.Timedelta(inner_gap),
            )
            alpha_scores: list[tuple[float, float]] = []
            train_dates = pd.DatetimeIndex(train_frame.index.get_level_values("decision_ts"))
            for alpha in alpha_grid:
                total_sse = 0.0
                for split_idx, (inner_train_dates, validation_dates) in enumerate(splits):
                    inner_train = train_frame.loc[train_dates.isin(inner_train_dates)]
                    validation = train_frame.loc[train_dates.isin(validation_dates)]
                    intercept, beta = _fit_ridge(
                        inner_train[list(features)].to_numpy(dtype=float),
                        inner_train[target_column].to_numpy(dtype=float),
                        float(alpha),
                    )
                    prediction = intercept + validation[list(features)].to_numpy(dtype=float) @ beta
                    residual = validation[target_column].to_numpy(dtype=float) - prediction
                    sse = float(residual @ residual)
                    total_sse += sse
                    inner_rows.append(
                        {
                            "candidate_id": candidate_id,
                            "model_id": model_id,
                            "fold_idx": int(fold.fold_idx),
                            "inner_split_idx": split_idx,
                            "alpha": float(alpha),
                            "inner_train_start": inner_train_dates.min(),
                            "inner_train_end": inner_train_dates.max(),
                            "validation_start": validation_dates.min(),
                            "validation_end": validation_dates.max(),
                            "validation_rows": len(validation),
                            "validation_sse": sse,
                        }
                    )
                alpha_scores.append((float(alpha), total_sse))
            best_sse = min(score for _, score in alpha_scores)
            best_alpha = max(alpha for alpha, score in alpha_scores if np.isclose(score, best_sse, rtol=1e-12, atol=1e-12))
            intercept, beta = _fit_ridge(
                train_frame[list(features)].to_numpy(dtype=float),
                train_frame[target_column].to_numpy(dtype=float),
                best_alpha,
            )
            replica = intercept + test_frame[list(features)].to_numpy(dtype=float) @ beta
            target = test_frame[target_column].to_numpy(dtype=float)
            output = test_frame[["forward_return", "strategy_forward_return"]].copy()
            output["candidate_id"] = candidate_id
            output["model_id"] = model_id
            output["fold_idx"] = int(fold.fold_idx)
            output["target_signal"] = target
            output["replica_signal"] = replica
            output["residual_signal"] = target - replica
            output["fold_train_target_mean"] = float(train_frame[target_column].mean())
            output["selected_alpha"] = best_alpha
            prediction_frames.append(output.reset_index())
            coefficient_rows.append(
                {
                    "candidate_id": candidate_id,
                    "model_id": model_id,
                    "fold_idx": int(fold.fold_idx),
                    "feature_name": "__intercept__",
                    "coefficient": intercept,
                    "selected_alpha": best_alpha,
                }
            )
            coefficient_rows.extend(
                {
                    "candidate_id": candidate_id,
                    "model_id": model_id,
                    "fold_idx": int(fold.fold_idx),
                    "feature_name": feature_name,
                    "coefficient": float(value),
                    "selected_alpha": best_alpha,
                }
                for feature_name, value in zip(features, beta, strict=True)
            )
    if not prediction_frames:
        raise ValueError("ridge replication produced no OOS predictions")
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["candidate_id", "model_id", "fold_idx", "decision_ts", "symbol"]
    ).reset_index(drop=True)
    return RidgeReplicationArtifacts(
        predictions=predictions,
        coefficients=pd.DataFrame(coefficient_rows),
        inner_scores=pd.DataFrame(inner_rows),
    )


def audit_frozen_level2_full_ridge_reproduction(
    canonical_frame: pd.DataFrame,
    folds: Sequence[object],
    *,
    candidate_id: str,
    frozen_predictions: pd.DataFrame,
    frozen_inner_scores: pd.DataFrame,
    target_column: str = "combo_signal",
    feature_columns: Sequence[str] = PRICE_VOLUME_COLUMNS,
    inner_gap: pd.Timedelta = pd.Timedelta(days=1),
    atol: float = 1e-12,
) -> pd.DataFrame:
    """Recompute the frozen level2_full ridge path and require exact lineage."""
    del inner_gap
    prediction_columns = [
        "candidate_id",
        "model_id",
        "fold_idx",
        "decision_ts",
        "symbol",
        "target_signal",
        "replica_signal",
        "residual_signal",
        "selected_alpha",
    ]
    required_score_columns = [
        "candidate_id",
        "model_id",
        "fold_idx",
        "inner_split_idx",
        "alpha",
        "validation_rows",
        "validation_sse",
    ]
    for name, frame, required in (
        ("frozen ridge predictions", frozen_predictions, prediction_columns),
        ("frozen ridge inner scores", frozen_inner_scores, required_score_columns),
    ):
        missing = sorted(set(required).difference(frame.columns))
        if missing:
            raise ValueError(f"{name} missing columns: " + ", ".join(missing))

    predictions = frozen_predictions.loc[
        (frozen_predictions["candidate_id"].astype(str) == str(candidate_id))
        & (frozen_predictions["model_id"].astype(str) == "level2_full"),
        prediction_columns,
    ].copy()
    scores = frozen_inner_scores.loc[
        (frozen_inner_scores["candidate_id"].astype(str) == str(candidate_id))
        & (frozen_inner_scores["model_id"].astype(str) == "level2_full"),
        required_score_columns,
    ].copy()
    if predictions.empty or scores.empty:
        raise ValueError("frozen level2_full ridge artifacts are missing")
    predictions["fold_idx"] = pd.to_numeric(
        predictions["fold_idx"], errors="raise"
    ).astype(int)
    predictions["decision_ts"] = pd.to_datetime(
        predictions["decision_ts"], utc=True, errors="raise"
    )
    predictions["symbol"] = predictions["symbol"].astype(str)
    scores["fold_idx"] = pd.to_numeric(scores["fold_idx"], errors="raise").astype(int)
    scores["alpha"] = pd.to_numeric(scores["alpha"], errors="raise")
    scores["validation_sse"] = pd.to_numeric(
        scores["validation_sse"], errors="raise"
    )
    max_prediction_error = 0.0
    audited_fold_count = 0
    for fold in folds:
        fold_idx = int(fold.fold_idx)
        if {"fold_idx", "split"}.issubset(canonical_frame.index.names):
            try:
                train_frame = canonical_frame.xs(
                    (fold_idx, "train"), level=("fold_idx", "split")
                ).sort_index()
                test_frame = canonical_frame.xs(
                    (fold_idx, "test"), level=("fold_idx", "split")
                ).sort_index()
            except KeyError as error:
                raise ValueError(f"canonical frame is missing fold {fold_idx}") from error
        else:
            train_frame = canonical_frame.loc[
                _date_mask(canonical_frame, fold.train_start, fold.train_end)
            ].sort_index()
            test_frame = canonical_frame.loc[
                _date_mask(canonical_frame, fold.test_start, fold.test_end)
            ].sort_index()
        fold_predictions = predictions.loc[
            predictions["fold_idx"] == fold_idx
        ].sort_values(["decision_ts", "symbol"], kind="mergesort").reset_index(drop=True)
        fold_scores = scores.loc[scores["fold_idx"] == fold_idx]
        if train_frame.empty or test_frame.empty or fold_predictions.empty or fold_scores.empty:
            raise ValueError(f"frozen ridge reproduction fold {fold_idx} is incomplete")
        expected_keys = test_frame.reset_index()[["decision_ts", "symbol"]].copy()
        expected_keys["decision_ts"] = pd.to_datetime(
            expected_keys["decision_ts"], utc=True, errors="raise"
        )
        expected_keys["symbol"] = expected_keys["symbol"].astype(str)
        expected_keys = expected_keys.sort_values(
            ["decision_ts", "symbol"], kind="mergesort"
        ).reset_index(drop=True)
        if not fold_predictions[["decision_ts", "symbol"]].equals(expected_keys):
            raise ValueError(
                "recomputed level2_full ridge prediction keys differ from frozen"
            )
        target = test_frame[target_column].to_numpy(dtype=float)
        if not np.allclose(
            fold_predictions["target_signal"].to_numpy(dtype=float),
            target,
            rtol=0.0,
            atol=float(atol),
        ):
            raise ValueError(
                "recomputed level2_full ridge targets differ from frozen values"
            )
        totals = fold_scores.groupby("alpha", as_index=False)["validation_sse"].sum()
        best_sse = float(totals["validation_sse"].min())
        selected_alpha = float(
            totals.loc[
                np.isclose(
                    totals["validation_sse"],
                    best_sse,
                    rtol=1e-12,
                    atol=1e-12,
                ),
                "alpha",
            ].max()
        )
        recorded_alpha = fold_predictions["selected_alpha"].astype(float).unique()
        if len(recorded_alpha) != 1 or not np.isclose(
            recorded_alpha[0], selected_alpha, rtol=1e-12, atol=1e-12
        ):
            raise ValueError(
                "recomputed level2_full ridge selected alpha differs from frozen"
            )
        intercept, beta = _fit_ridge(
            train_frame[list(feature_columns)].to_numpy(dtype=float),
            train_frame[target_column].to_numpy(dtype=float),
            selected_alpha,
        )
        recomputed_prediction = (
            intercept
            + test_frame[list(feature_columns)].to_numpy(dtype=float) @ beta
        )
        error = np.abs(
            recomputed_prediction
            - fold_predictions["replica_signal"].to_numpy(dtype=float)
        )
        max_prediction_error = max(
            max_prediction_error, float(error.max()) if error.size else 0.0
        )
        if not np.allclose(
            recomputed_prediction,
            fold_predictions["replica_signal"].to_numpy(dtype=float),
            rtol=0.0,
            atol=float(atol),
        ):
            raise ValueError(
                "recomputed level2_full ridge replica differs from frozen values"
            )
        audited_fold_count += 1
    return pd.DataFrame(
        [
            {
                "candidate_id": str(candidate_id),
                "model_id": "level2_full",
                "fold_count": audited_fold_count,
                "prediction_row_count": len(predictions),
                "inner_score_row_count": len(scores),
                "max_abs_prediction_error": max_prediction_error,
                "prediction_tolerance": float(atol),
                "reproduction_status": "pass",
            }
        ]
    )


def _safe_corr(left: pd.Series, right: pd.Series, method: str) -> float:
    if len(left) < 2 or left.nunique(dropna=True) <= 1 or right.nunique(dropna=True) <= 1:
        return float("nan")
    return float(left.corr(right, method=method))


def _top_bottom_overlap(group: pd.DataFrame, n_buckets: int = 5) -> tuple[float, float, float]:
    count = len(group)
    leg_count = count // int(n_buckets)
    if leg_count < 1:
        return float("nan"), float("nan"), float("nan")
    ordered_target = group.sort_values(["target_signal", "symbol"], kind="mergesort")
    ordered_replica = group.sort_values(["replica_signal", "symbol"], kind="mergesort")
    target_short = set(ordered_target.head(leg_count)["symbol"].astype(str))
    target_long = set(ordered_target.tail(leg_count)["symbol"].astype(str))
    replica_short = set(ordered_replica.head(leg_count)["symbol"].astype(str))
    replica_long = set(ordered_replica.tail(leg_count)["symbol"].astype(str))
    short_overlap = len(target_short.intersection(replica_short)) / leg_count
    long_overlap = len(target_long.intersection(replica_long)) / leg_count
    return float(long_overlap), float(short_overlap), float((long_overlap + short_overlap) / 2.0)


def _residual_target_overlap(group: pd.DataFrame, n_buckets: int = 5) -> float:
    working = group.copy()
    working["replica_signal"] = working["residual_signal"]
    return _top_bottom_overlap(working, n_buckets=n_buckets)[2]


def summarize_replication_metrics(
    predictions: pd.DataFrame,
    *,
    r2_threshold: float = REPLICATION_R2_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "candidate_id",
        "model_id",
        "fold_idx",
        "decision_ts",
        "symbol",
        "target_signal",
        "replica_signal",
        "fold_train_target_mean",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError("predictions missing metric columns: " + ", ".join(missing))
    decision_keys = ["candidate_id", "model_id", "fold_idx", "decision_ts"]
    work = predictions[
        [*decision_keys, "symbol", "target_signal", "replica_signal", "residual_signal", "fold_train_target_mean"]
    ].copy()
    work = work.sort_values([*decision_keys, "symbol"], kind="mergesort").reset_index(drop=True)
    grouped = work.groupby(decision_keys, sort=False, dropna=False)

    def grouped_pearson(left: str, right: str, prefix: str) -> pd.Series:
        values = work[[*decision_keys, left, right]].copy()
        values["__left_sq"] = np.square(values[left].astype(float))
        values["__right_sq"] = np.square(values[right].astype(float))
        values["__product"] = values[left].astype(float) * values[right].astype(float)
        moments = values.groupby(decision_keys, sort=False, dropna=False).agg(
            n=(left, "size"),
            sum_left=(left, "sum"),
            sum_right=(right, "sum"),
            sum_left_sq=("__left_sq", "sum"),
            sum_right_sq=("__right_sq", "sum"),
            product=("__product", "sum"),
        )
        numerator = moments["product"] - moments["sum_left"] * moments["sum_right"] / moments["n"]
        left_ss = moments["sum_left_sq"] - np.square(moments["sum_left"]) / moments["n"]
        right_ss = moments["sum_right_sq"] - np.square(moments["sum_right"]) / moments["n"]
        denominator = np.sqrt(left_ss.clip(lower=0.0) * right_ss.clip(lower=0.0))
        result = numerator / denominator
        result.name = prefix
        return result.where(denominator > 0.0)

    work["target_rank"] = grouped["target_signal"].rank(method="first")
    work["replica_rank"] = grouped["replica_signal"].rank(method="first")
    work["residual_rank"] = grouped["residual_signal"].rank(method="first")
    work["cross_section_size"] = grouped["symbol"].transform("size").astype(int)
    work["leg_count"] = work["cross_section_size"] // 5
    valid_leg = work["leg_count"] >= 1
    work["target_short"] = valid_leg & (work["target_rank"] <= work["leg_count"])
    work["target_long"] = valid_leg & (
        work["target_rank"] > work["cross_section_size"] - work["leg_count"]
    )
    work["replica_short"] = valid_leg & (work["replica_rank"] <= work["leg_count"])
    work["replica_long"] = valid_leg & (
        work["replica_rank"] > work["cross_section_size"] - work["leg_count"]
    )
    work["residual_short"] = valid_leg & (work["residual_rank"] <= work["leg_count"])
    work["residual_long"] = valid_leg & (
        work["residual_rank"] > work["cross_section_size"] - work["leg_count"]
    )
    overlap = grouped.agg(
        cross_section_size=("cross_section_size", "first"),
        leg_count=("leg_count", "first"),
    )
    for name, mask in {
        "long_overlap": work["target_long"] & work["replica_long"],
        "short_overlap": work["target_short"] & work["replica_short"],
        "residual_long_overlap": work["target_long"] & work["residual_long"],
        "residual_short_overlap": work["target_short"] & work["residual_short"],
    }.items():
        overlap[name] = mask.groupby(
            [work[column] for column in decision_keys], sort=False, dropna=False
        ).sum() / overlap["leg_count"]
        overlap.loc[overlap["leg_count"] < 1, name] = np.nan
    overlap["top_bottom_overlap"] = (
        overlap["long_overlap"] + overlap["short_overlap"]
    ) / 2.0
    overlap["residual_target_top_bottom_overlap"] = (
        overlap["residual_long_overlap"] + overlap["residual_short_overlap"]
    ) / 2.0
    decision_metrics = overlap.reset_index()[
        [*decision_keys, "cross_section_size", "long_overlap", "short_overlap", "top_bottom_overlap", "residual_target_top_bottom_overlap"]
    ]
    decision_metrics = decision_metrics.merge(
        grouped_pearson("target_signal", "replica_signal", "pearson").reset_index(),
        on=decision_keys,
        how="left",
        validate="one_to_one",
    )
    rank_frame = work[[*decision_keys, "target_rank", "replica_rank"]].rename(
        columns={"target_rank": "target_signal", "replica_rank": "replica_signal"}
    )
    rank_frame["__left_sq"] = np.square(rank_frame["target_signal"].astype(float))
    rank_frame["__right_sq"] = np.square(rank_frame["replica_signal"].astype(float))
    rank_frame["__product"] = (
        rank_frame["target_signal"].astype(float) * rank_frame["replica_signal"].astype(float)
    )
    rank_grouped = rank_frame.groupby(decision_keys, sort=False, dropna=False)
    rank_moments = rank_grouped.agg(
        n=("target_signal", "size"),
        sum_left=("target_signal", "sum"),
        sum_right=("replica_signal", "sum"),
        sum_left_sq=("__left_sq", "sum"),
        sum_right_sq=("__right_sq", "sum"),
        product=("__product", "sum"),
    )
    rank_numerator = (
        rank_moments["product"]
        - rank_moments["sum_left"] * rank_moments["sum_right"] / rank_moments["n"]
    )
    rank_denominator = np.sqrt(
        (rank_moments["sum_left_sq"] - np.square(rank_moments["sum_left"]) / rank_moments["n"])
        * (rank_moments["sum_right_sq"] - np.square(rank_moments["sum_right"]) / rank_moments["n"])
    )
    spearman = (rank_numerator / rank_denominator).where(rank_denominator > 0.0).rename("spearman")
    decision_metrics = decision_metrics.merge(
        spearman.reset_index(), on=decision_keys, how="left", validate="one_to_one"
    ).merge(
        grouped_pearson("residual_signal", "replica_signal", "residual_replica_pearson").reset_index(),
        on=decision_keys,
        how="left",
        validate="one_to_one",
    )

    work["residual_sq"] = np.square(work["target_signal"] - work["replica_signal"])
    work["baseline_sq"] = np.square(work["target_signal"] - work["fold_train_target_mean"])
    fold_keys = ["candidate_id", "model_id", "fold_idx"]
    fold_metrics = work.groupby(fold_keys, sort=False, as_index=False).agg(
        oos_rows=("symbol", "size"),
        oos_decisions=("decision_ts", "nunique"),
        residual_ss=("residual_sq", "sum"),
        baseline_ss=("baseline_sq", "sum"),
    )
    fold_metrics["oos_r2"] = 1.0 - fold_metrics["residual_ss"] / fold_metrics["baseline_ss"]
    fold_metrics.loc[fold_metrics["baseline_ss"] <= 0.0, "oos_r2"] = np.nan
    fold_decision_means = decision_metrics.groupby(fold_keys, sort=False, as_index=False).agg(
        mean_pearson=("pearson", "mean"),
        mean_spearman=("spearman", "mean"),
        mean_top_bottom_overlap=("top_bottom_overlap", "mean"),
    )
    fold_metrics = fold_metrics.merge(
        fold_decision_means, on=fold_keys, how="left", validate="one_to_one"
    ).drop(columns=["residual_ss", "baseline_ss"])

    model_keys = ["candidate_id", "model_id"]
    summary = work.groupby(model_keys, sort=False, as_index=False).agg(
        oos_rows=("symbol", "size"),
        oos_decisions=("decision_ts", "nunique"),
        n_folds=("fold_idx", "nunique"),
        residual_ss=("residual_sq", "sum"),
        baseline_ss=("baseline_sq", "sum"),
    )
    summary["stitched_oos_r2"] = 1.0 - summary["residual_ss"] / summary["baseline_ss"]
    summary.loc[summary["baseline_ss"] <= 0.0, "stitched_oos_r2"] = np.nan
    summary["replication_gate_pass"] = (
        np.isfinite(summary["stitched_oos_r2"])
        & (summary["stitched_oos_r2"] >= float(r2_threshold))
    )
    fold_summary = fold_metrics.groupby(model_keys, sort=False, as_index=False).agg(
        positive_r2_fold_share=("oos_r2", lambda values: float((values > 0.0).mean())),
        median_fold_r2=("oos_r2", "median"),
    )
    decision_summary = decision_metrics.groupby(model_keys, sort=False, as_index=False).agg(
        mean_pearson=("pearson", "mean"),
        mean_spearman=("spearman", "mean"),
        mean_long_overlap=("long_overlap", "mean"),
        mean_short_overlap=("short_overlap", "mean"),
        mean_top_bottom_overlap=("top_bottom_overlap", "mean"),
        mean_residual_replica_pearson=("residual_replica_pearson", "mean"),
        mean_residual_target_top_bottom_overlap=("residual_target_top_bottom_overlap", "mean"),
    )
    summary = summary.merge(fold_summary, on=model_keys, how="left", validate="one_to_one").merge(
        decision_summary, on=model_keys, how="left", validate="one_to_one"
    ).drop(columns=["residual_ss", "baseline_ss"])
    ordered = [
        "candidate_id", "model_id", "stitched_oos_r2", "replication_gate_pass",
        "oos_rows", "oos_decisions", "n_folds", "positive_r2_fold_share",
        "median_fold_r2", "mean_pearson", "mean_spearman", "mean_long_overlap",
        "mean_short_overlap", "mean_top_bottom_overlap",
        "mean_residual_replica_pearson", "mean_residual_target_top_bottom_overlap",
    ]
    return summary[ordered], fold_metrics, decision_metrics


def classify_replication_difficulty(replication_summary: pd.DataFrame) -> pd.DataFrame:
    required = {"candidate_id", "model_id", "replication_gate_pass", "stitched_oos_r2"}
    missing = sorted(required.difference(replication_summary.columns))
    if missing:
        raise ValueError("replication summary missing columns: " + ", ".join(missing))
    rows: list[dict[str, object]] = []
    for candidate_id, group in replication_summary.groupby("candidate_id", sort=False):
        passed = group.loc[group["replication_gate_pass"].astype(bool)].copy()
        category = "current_dictionary_replication_failed"
        qualifying_models = ""
        simplest_r2 = float("nan")
        if not passed.empty:
            level0 = passed[passed["model_id"] == "level0_controls"]
            level1 = passed[passed["model_id"].str.startswith("level1__")]
            family = passed[passed["model_id"].isin(("level2_return", "level2_volatility", "level2_volume"))]
            full = passed[passed["model_id"] == "level2_full"]
            if not level0.empty:
                chosen = level0
                category = "level0_partial_replication"
            elif not level1.empty:
                chosen = level1
                category = "level1_single_proxy_partial_replication"
            elif not family.empty:
                chosen = family
                category = "level2_single_family_partial_replication"
            elif not full.empty:
                chosen = full
                category = "level2_full_only_partial_replication"
            else:
                chosen = passed
            qualifying_models = "|".join(sorted(chosen["model_id"].astype(str)))
            simplest_r2 = float(chosen["stitched_oos_r2"].max())
        rows.append(
            {
                "candidate_id": candidate_id,
                "replication_difficulty": category,
                "qualifying_models_at_simplest_level": qualifying_models,
                "best_r2_at_simplest_level": simplest_r2,
                "any_replication_gate_pass": not passed.empty,
            }
        )
    return pd.DataFrame(rows)


def build_replay_signal_frame(
    predictions: pd.DataFrame,
    replication_summary: pd.DataFrame,
    candidate_metadata: pd.DataFrame,
) -> pd.DataFrame:
    metadata_columns = [
        "candidate_id",
        "combo_id",
        "track",
        "weight_scheme",
        "panel_frequency",
        "return_horizon",
        "component_features",
    ]
    missing = sorted(set(metadata_columns).difference(candidate_metadata.columns))
    if missing:
        raise ValueError("candidate metadata missing columns: " + ", ".join(missing))
    metadata = candidate_metadata[metadata_columns].drop_duplicates("candidate_id")
    base_columns = [
        "candidate_id",
        "model_id",
        "fold_idx",
        "decision_ts",
        "symbol",
        "forward_return",
        "strategy_forward_return",
        "target_signal",
        "replica_signal",
        "residual_signal",
    ]
    source = predictions[base_columns].copy()
    frames: list[pd.DataFrame] = []
    original = source.sort_values("model_id").drop_duplicates(
        ["candidate_id", "fold_idx", "decision_ts", "symbol"]
    )
    original["model_id"] = "original"
    original["signal_id"] = "original"
    original["signal_type"] = "original"
    original["signal_value"] = original["target_signal"]
    frames.append(original)
    replica = source.copy()
    replica["signal_id"] = "replica__" + replica["model_id"].astype(str)
    replica["signal_type"] = "replica"
    replica["signal_value"] = replica["replica_signal"]
    frames.append(replica)
    pass_keys = replication_summary.loc[
        replication_summary["replication_gate_pass"].astype(bool), ["candidate_id", "model_id"]
    ].drop_duplicates()
    residual = source.merge(pass_keys, on=["candidate_id", "model_id"], how="inner")
    if not residual.empty:
        residual["signal_id"] = "residual__" + residual["model_id"].astype(str)
        residual["signal_type"] = "residual"
        residual["signal_value"] = residual["residual_signal"]
        frames.append(residual)
    combined = pd.concat(frames, ignore_index=True).merge(
        metadata, on="candidate_id", how="left", validate="many_to_one"
    )
    if combined[metadata_columns[1:]].isna().any(axis=None):
        raise ValueError("replay signal frame has missing candidate metadata")
    combined["replay_combo_id"] = combined["candidate_id"] + "__" + combined["signal_id"]
    return combined.sort_values(
        ["candidate_id", "signal_id", "fold_idx", "decision_ts", "symbol"]
    ).reset_index(drop=True)


def evaluate_precomputed_oos_signals(
    replay_signals: pd.DataFrame,
    folds: Sequence[object],
    walk_forward_spec: Mapping[str, int],
    *,
    n_buckets: int,
    cost_multipliers: Sequence[float],
    taker_fee_rate: float,
    frequency_periods_per_year: Mapping[str, int | float],
) -> SignalReplayArtifacts:
    required = {
        "candidate_id",
        "model_id",
        "signal_id",
        "signal_type",
        "replay_combo_id",
        "track",
        "weight_scheme",
        "panel_frequency",
        "return_horizon",
        "component_features",
        "fold_idx",
        "decision_ts",
        "symbol",
        "signal_value",
        "forward_return",
        "strategy_forward_return",
    }
    missing = sorted(required.difference(replay_signals.columns))
    if missing:
        raise ValueError("replay signals missing columns: " + ", ".join(missing))
    fold_by_idx = {int(fold.fold_idx): fold for fold in folds}
    summary_rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []
    holding_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, object]] = []
    group_columns = [
        "candidate_id",
        "model_id",
        "signal_id",
        "signal_type",
        "replay_combo_id",
        "track",
        "weight_scheme",
        "panel_frequency",
        "return_horizon",
        "component_features",
    ]
    for keys, strategy_frame in replay_signals.groupby(group_columns, sort=False, dropna=False):
        meta = dict(zip(group_columns, keys, strict=True))
        spec = factor_research.ComboSpec(
            combo_id=str(meta["replay_combo_id"]),
            track=str(meta["track"]),
            panel_frequency=str(meta["panel_frequency"]),
            return_horizon=str(meta["return_horizon"]),
            feature_names=(str(meta["signal_id"]),),
            weight_scheme=str(meta["weight_scheme"]),
        )
        rows: list[dict[str, object]] = []
        holdings: list[pd.DataFrame] = []
        diagnostics: list[dict[str, object]] = []
        for fold_idx, fold_frame in strategy_frame.groupby("fold_idx", sort=True):
            fold = fold_by_idx.get(int(fold_idx))
            if fold is None:
                raise ValueError(f"missing fold object for fold_idx={fold_idx}")
            composite = (
                fold_frame[["decision_ts", "symbol", "signal_value", "forward_return", "strategy_forward_return"]]
                .rename(columns={"signal_value": "composite_signal"})
                .set_index("decision_ts")
                .sort_index()
            )
            fold_rows, fold_holdings, fold_diagnostics = factor_research.long_short_strategy_frames_for_fold(
                spec,
                fold,
                composite,
                n_buckets=int(n_buckets),
                cost_multipliers=cost_multipliers,
                taker_fee_rate=float(taker_fee_rate),
                component_features=str(meta["signal_id"]),
            )
            rows.extend(fold_rows)
            diagnostics.extend(fold_diagnostics)
            if not fold_holdings.empty:
                holdings.append(fold_holdings)
        detail = pd.DataFrame(rows)
        if detail.empty:
            raise ValueError("precomputed signal replay produced no scored decisions: " + str(meta["replay_combo_id"]))
        summary = factor_research.summarize_long_short_strategy(
            spec,
            detail,
            diagnostics,
            walk_forward_spec,
            str(meta["panel_frequency"]),
            frequency_periods_per_year,
            cost_multipliers,
        )
        summary.update({key: meta[key] for key in ("candidate_id", "model_id", "signal_id", "signal_type")})
        summary_rows.append(summary)
        detail = detail.assign(**{key: meta[key] for key in ("candidate_id", "model_id", "signal_id", "signal_type")})
        detail_frames.append(detail)
        if holdings:
            holding = pd.concat(holdings, ignore_index=True).assign(
                **{key: meta[key] for key in ("candidate_id", "model_id", "signal_id", "signal_type")}
            )
            holding_frames.append(holding)
        for row in diagnostics:
            diagnostic_rows.append(
                {
                    **{key: meta[key] for key in ("candidate_id", "model_id", "signal_id", "signal_type")},
                    **row,
                }
            )
    return SignalReplayArtifacts(
        summary=pd.DataFrame(summary_rows),
        timeseries=pd.concat(detail_frames, ignore_index=True),
        holdings=pd.concat(holding_frames, ignore_index=True) if holding_frames else pd.DataFrame(),
        orders=pd.DataFrame(),
        diagnostics=pd.DataFrame(diagnostic_rows),
    )


def evaluate_executable_precomputed_oos_signals(
    replay_signals: pd.DataFrame,
    folds: Sequence[object],
    walk_forward_spec: Mapping[str, int],
    *,
    n_buckets: int,
    min_cross_section: int,
    cost_multipliers: Sequence[float],
    taker_fee_rate: float,
    frequency_periods_per_year: Mapping[str, int | float],
    horizon_deltas: Mapping[str, pd.Timedelta],
    execution_delay_minutes: int,
) -> SignalReplayArtifacts:
    """Replay frozen OOS signals through the executable continuous ledger.

    Signals are ranked high-to-low without re-estimating their direction. The
    fold identifier is lineage only; positions remain continuous across fold
    boundaries and close once after the final OOS holding interval.
    """
    identity_columns = [
        "candidate_id",
        "model_id",
        "signal_id",
        "signal_type",
        "replay_combo_id",
        "track",
        "weight_scheme",
        "panel_frequency",
        "return_horizon",
        "component_features",
    ]
    ledger_columns = [
        "signal_timeframes",
        "native_bar_end_ts",
        "signal_bar_end_ts",
        "availability_ts",
        "data_observed_ts",
        "decision_interval",
        "order_submit_ts",
        "execution_ts",
        "execution_open_time",
        "next_execution_ts",
        "holding_interval",
        "exit_rule",
        "score_order",
        "entry_price",
        "exit_price",
        "execution_price",
        "next_execution_price",
        "executable_return",
    ]
    required = {
        *identity_columns,
        *ledger_columns,
        "fold_idx",
        "decision_ts",
        "symbol",
        "signal_value",
    }
    missing = sorted(required.difference(replay_signals.columns))
    if missing:
        raise ValueError("executable replay signals missing columns: " + ", ".join(missing))
    if int(n_buckets) < 2 or int(min_cross_section) < int(n_buckets):
        raise ValueError("invalid bucket or cross-section configuration")

    fold_by_idx = {int(fold.fold_idx): fold for fold in folds}
    summary_rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []
    holding_frames: list[pd.DataFrame] = []
    order_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, object]] = []
    for keys, strategy_frame in replay_signals.groupby(
        identity_columns, sort=False, dropna=False
    ):
        metadata = dict(zip(identity_columns, keys, strict=True))
        route_horizon = str(metadata["return_horizon"])
        route_frequency = str(metadata["panel_frequency"])
        if route_horizon != route_frequency:
            raise ValueError("precomputed executable replay requires horizon-aligned decisions")
        adapted = factor_research.validated_executable_return_adapter(
            strategy_frame.drop(columns=["forward_return", "strategy_forward_return"], errors="ignore"),
            return_horizon=route_horizon,
            decision_frequency=route_frequency,
            horizon_deltas=horizon_deltas,
            execution_delay_minutes=execution_delay_minutes,
        )
        adapted["decision_ts"] = pd.to_datetime(adapted["decision_ts"], utc=True)
        if adapted.duplicated(["fold_idx", "decision_ts", "symbol"]).any():
            raise ValueError("precomputed replay contains duplicate fold/decision/symbol rows")
        unknown_folds = sorted(set(adapted["fold_idx"].astype(int)).difference(fold_by_idx))
        if unknown_folds:
            raise ValueError("precomputed replay contains unknown folds: " + ", ".join(map(str, unknown_folds)))

        finite = np.isfinite(
            adapted[["signal_value", "executable_return"]].astype(float)
        ).all(axis=1)
        valid = adapted.loc[finite].copy()
        all_decisions = pd.DatetimeIndex(adapted["decision_ts"].drop_duplicates().sort_values())
        grouped_valid = valid.groupby("decision_ts", sort=False)
        counts = grouped_valid.size().reindex(all_decisions, fill_value=0).astype(int)
        unique_signal = grouped_valid["signal_value"].nunique(dropna=True).reindex(
            all_decisions, fill_value=0
        ).astype(int)
        statuses = pd.Series("ok", index=all_decisions, dtype=object)
        statuses.loc[counts < int(min_cross_section)] = "small_cross_section"
        statuses.loc[(counts >= int(min_cross_section)) & (unique_signal <= 1)] = "constant_feature"
        strategy_diagnostics = [
            {
                **{key: metadata[key] for key in ("candidate_id", "model_id", "signal_id", "signal_type")},
                "decision_ts": decision_ts,
                "cross_section_size": int(counts.loc[decision_ts]),
                "status": str(statuses.loc[decision_ts]),
            }
            for decision_ts in all_decisions
        ]
        ok_decisions = statuses.index[statuses.eq("ok")]
        if len(ok_decisions) == 0:
            raise ValueError("precomputed executable replay produced no target holdings")
        scored = valid.loc[valid["decision_ts"].isin(ok_decisions)].copy()
        scored = scored.sort_values(
            ["decision_ts", "signal_value", "symbol"], kind="mergesort"
        ).reset_index(drop=True)
        group_sizes = scored.groupby("decision_ts", sort=False)["symbol"].transform("size").astype(int)
        positions = scored.groupby("decision_ts", sort=False).cumcount().astype(int)
        base_sizes = group_sizes // int(n_buckets)
        remainders = group_sizes % int(n_buckets)
        first_block_limits = (base_sizes + 1) * remainders
        scored["bucket"] = np.where(
            positions < first_block_limits,
            positions // (base_sizes + 1),
            remainders + (positions - first_block_limits) // base_sizes,
        ).astype(int) + 1
        targets = scored.loc[scored["bucket"].isin((1, int(n_buckets)))].copy()
        targets["leg"] = np.where(
            targets["bucket"] == int(n_buckets), "long", "short"
        )
        leg_counts = targets.groupby(["decision_ts", "leg"], sort=False)["symbol"].transform("size").astype(float)
        targets["weight"] = np.where(
            targets["leg"] == "long", 0.5 / leg_counts, -0.5 / leg_counts
        )
        targets["combo_id"] = str(metadata["replay_combo_id"])
        for column in ("track", "weight_scheme", "panel_frequency", "return_horizon", "component_features"):
            targets[column] = metadata[column]
        long_sets = targets.loc[targets["leg"] == "long"].groupby("decision_ts")["symbol"].agg(
            lambda values: set(values.astype(str))
        )
        short_sets = targets.loc[targets["leg"] == "short"].groupby("decision_ts")["symbol"].agg(
            lambda values: set(values.astype(str))
        )
        previous_long: set[str] = set()
        previous_short: set[str] = set()
        turnover_rows: list[dict[str, object]] = []
        for decision_ts in pd.DatetimeIndex(ok_decisions):
            current_long = long_sets.get(decision_ts, set())
            current_short = short_sets.get(decision_ts, set())
            turnover_rows.append(
                {
                    "decision_ts": decision_ts,
                    "cross_section_size": int(counts.loc[decision_ts]),
                    "long_name_turnover_share": factor_research.name_turnover_share(current_long, previous_long),
                    "short_name_turnover_share": factor_research.name_turnover_share(current_short, previous_short),
                }
            )
            previous_long = current_long
            previous_short = current_short
        decision_index = pd.DatetimeIndex(targets["decision_ts"].drop_duplicates().sort_values())
        expected_delta = pd.Timedelta(horizon_deltas[route_horizon])
        if len(decision_index) > 1 and not ((decision_index[1:] - decision_index[:-1]) == expected_delta).all():
            raise ValueError("common-support replay decisions are not continuous at the route horizon")
        ledger = adapted[
            ["decision_ts", "symbol", "execution_ts", "next_execution_ts", "entry_price", "exit_price", "executable_return"]
        ].drop_duplicates(["decision_ts", "symbol"])
        detail, orders, actual_holdings = factor_research.continuous_membership_quantity_replay(
            targets,
            ledger,
            target_gross_notional=1.0,
            taker_fee_rate=float(taker_fee_rate),
            cost_multipliers=cost_multipliers,
            execution_delay_minutes=execution_delay_minutes,
        )
        actual_holdings["execution_price"] = actual_holdings["entry_price"].astype(float)
        actual_holdings["next_execution_price"] = actual_holdings["exit_price"].astype(float)
        turnover = pd.DataFrame(turnover_rows)
        turnover["name_turnover_share"] = (
            turnover["long_name_turnover_share"] + turnover["short_name_turnover_share"]
        ) / 2.0
        detail = detail.merge(turnover, on="decision_ts", how="left", validate="one_to_one")
        detail["benchmark_return"] = 0.0
        detail["active_return"] = detail["gross_return"]
        for multiplier in cost_multipliers:
            label = factor_research.scenario_label(float(multiplier))
            detail[f"net_active_return_{label}"] = detail[f"net_return_{label}"]
        for column in ("candidate_id", "model_id", "signal_id", "signal_type"):
            detail[column] = metadata[column]
            orders[column] = metadata[column]
            actual_holdings[column] = metadata[column]
        diagnostics = [
            {
                "decision_ts": row["decision_ts"],
                "cross_section_size": int(row["cross_section_size"]),
                "status": row["status"],
            }
            for row in strategy_diagnostics
        ]
        spec = factor_research.ComboSpec(
            combo_id=str(metadata["replay_combo_id"]),
            track=str(metadata["track"]),
            panel_frequency=route_frequency,
            return_horizon=route_horizon,
            feature_names=(str(metadata["signal_id"]),),
            weight_scheme=str(metadata["weight_scheme"]),
        )
        summary = factor_research.summarize_long_short_strategy(
            spec,
            detail,
            diagnostics,
            walk_forward_spec,
            route_frequency,
            frequency_periods_per_year,
            cost_multipliers,
        )
        summary.update(
            {key: metadata[key] for key in ("candidate_id", "model_id", "signal_id", "signal_type")}
        )
        summary_rows.append(summary)
        detail_frames.append(detail)
        holding_frames.append(actual_holdings)
        order_frames.append(orders)
        diagnostic_rows.extend(strategy_diagnostics)
    return SignalReplayArtifacts(
        summary=pd.DataFrame(summary_rows),
        timeseries=pd.concat(detail_frames, ignore_index=True),
        holdings=pd.concat(holding_frames, ignore_index=True),
        orders=pd.concat(order_frames, ignore_index=True),
        diagnostics=pd.DataFrame(diagnostic_rows),
    )


def compare_signal_replays(
    l3_summary: pd.DataFrame,
    l4_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build the formal original/replica/residual replay comparison table."""
    required_l3 = {
        "candidate_id",
        "model_id",
        "signal_type",
        "net_1x_sharpe",
        "net_1x_annualized_return",
        "net_1x_max_drawdown",
        "net_1x_fold_positive_share",
        "mean_charged_turnover",
    }
    missing = sorted(required_l3.difference(l3_summary.columns))
    if missing:
        raise ValueError("L3 replay summary missing columns: " + ", ".join(missing))
    l4_metric_columns = [
        "net_1x_sharpe_on_equity",
        "net_1x_annualized_return_on_equity",
        "net_1x_max_drawdown_on_equity",
        "net_1x_fold_positive_share_on_equity",
        "filtered_order_share",
        "mean_actual_vs_target_gross_ratio",
        "mean_weight_abs_error_sum",
        "max_abs_net_exposure_share",
    ]
    missing_l4 = sorted(
        {"candidate_id", "model_id", "signal_type", *l4_metric_columns}.difference(l4_summary.columns)
    )
    if missing_l4:
        raise ValueError("L4 replay summary missing columns: " + ", ".join(missing_l4))
    rows: list[dict[str, object]] = []
    for candidate_id, candidate_l3 in l3_summary.groupby("candidate_id", sort=False):
        original_l3 = candidate_l3.loc[candidate_l3["signal_type"] == "original"]
        original_l4 = l4_summary.loc[
            (l4_summary["candidate_id"] == candidate_id) & (l4_summary["signal_type"] == "original")
        ]
        if len(original_l3) != 1 or len(original_l4) != 1:
            raise ValueError("each candidate must have exactly one original L3/L4 replay")
        original_l3_row = original_l3.iloc[0]
        original_l4_row = original_l4.iloc[0]
        model_ids = sorted(
            candidate_l3.loc[candidate_l3["signal_type"] == "replica", "model_id"].astype(str).unique()
        )
        for model_id in model_ids:
            replica_l3 = candidate_l3.loc[
                (candidate_l3["model_id"] == model_id) & (candidate_l3["signal_type"] == "replica")
            ]
            residual_l3 = candidate_l3.loc[
                (candidate_l3["model_id"] == model_id) & (candidate_l3["signal_type"] == "residual")
            ]
            replica_l4 = l4_summary.loc[
                (l4_summary["candidate_id"] == candidate_id)
                & (l4_summary["model_id"] == model_id)
                & (l4_summary["signal_type"] == "replica")
            ]
            residual_l4 = l4_summary.loc[
                (l4_summary["candidate_id"] == candidate_id)
                & (l4_summary["model_id"] == model_id)
                & (l4_summary["signal_type"] == "residual")
            ]
            if len(replica_l3) != 1 or len(replica_l4) != 1:
                raise ValueError("each model must have exactly one replica L3/L4 replay")
            replica_l3_row = replica_l3.iloc[0]
            replica_l4_row = replica_l4.iloc[0]
            row: dict[str, object] = {
                "candidate_id": candidate_id,
                "model_id": model_id,
                "original_l3_net_1x_sharpe": float(original_l3_row["net_1x_sharpe"]),
                "replica_l3_net_1x_sharpe": float(replica_l3_row["net_1x_sharpe"]),
                "replica_l3_sharpe_retention": (
                    float(replica_l3_row["net_1x_sharpe"]) / float(original_l3_row["net_1x_sharpe"])
                    if abs(float(original_l3_row["net_1x_sharpe"])) > 1e-12
                    else float("nan")
                ),
                "original_l4_net_1x_sharpe": float(original_l4_row["net_1x_sharpe_on_equity"]),
                "replica_l4_net_1x_sharpe": float(replica_l4_row["net_1x_sharpe_on_equity"]),
                "replica_l4_sharpe_retention": (
                    float(replica_l4_row["net_1x_sharpe_on_equity"])
                    / float(original_l4_row["net_1x_sharpe_on_equity"])
                    if abs(float(original_l4_row["net_1x_sharpe_on_equity"])) > 1e-12
                    else float("nan")
                ),
                "residual_replay_eligible": not residual_l3.empty,
            }
            for column in (
                "net_1x_annualized_return",
                "net_1x_max_drawdown",
                "net_1x_fold_positive_share",
                "mean_charged_turnover",
            ):
                row[f"original_l3_{column}"] = float(original_l3_row[column])
                row[f"replica_l3_{column}"] = float(replica_l3_row[column])
            for column in l4_metric_columns:
                row[f"original_l4_{column}"] = float(original_l4_row[column])
                row[f"replica_l4_{column}"] = float(replica_l4_row[column])
            if not residual_l3.empty:
                if len(residual_l3) != 1 or len(residual_l4) != 1:
                    raise ValueError("eligible residual must have exactly one L3/L4 replay")
                residual_l3_row = residual_l3.iloc[0]
                residual_l4_row = residual_l4.iloc[0]
                row["residual_l3_net_1x_sharpe"] = float(residual_l3_row["net_1x_sharpe"])
                row["residual_l3_sharpe_retention"] = (
                    float(residual_l3_row["net_1x_sharpe"]) / float(original_l3_row["net_1x_sharpe"])
                    if abs(float(original_l3_row["net_1x_sharpe"])) > 1e-12
                    else float("nan")
                )
                row["residual_l4_net_1x_sharpe"] = float(residual_l4_row["net_1x_sharpe_on_equity"])
                row["residual_l4_sharpe_retention"] = (
                    float(residual_l4_row["net_1x_sharpe_on_equity"])
                    / float(original_l4_row["net_1x_sharpe_on_equity"])
                    if abs(float(original_l4_row["net_1x_sharpe_on_equity"])) > 1e-12
                    else float("nan")
                )
            rows.append(row)
    return pd.DataFrame(rows)


def build_shadow_priority_inferences(
    classifications: pd.DataFrame,
    replication_summary: pd.DataFrame,
    replay_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Create evidence-linked, non-executing shadow-priority inferences."""
    required = {"candidate_id", "replication_difficulty", "qualifying_models_at_simplest_level"}
    missing = sorted(required.difference(classifications.columns))
    if missing:
        raise ValueError("classifications missing columns: " + ", ".join(missing))
    rows: list[dict[str, object]] = []
    for classification in classifications.itertuples(index=False):
        candidate_id = str(classification.candidate_id)
        category = str(classification.replication_difficulty)
        candidate_replication = replication_summary.loc[
            replication_summary["candidate_id"] == candidate_id
        ]
        best = candidate_replication.sort_values("stitched_oos_r2", ascending=False).head(1)
        best_model = str(best.iloc[0]["model_id"]) if not best.empty else ""
        best_r2 = float(best.iloc[0]["stitched_oos_r2"]) if not best.empty else float("nan")
        comparison = replay_comparison.loc[
            (replay_comparison["candidate_id"] == candidate_id)
            & (replay_comparison["model_id"] == best_model)
        ]
        best_l4_retention = (
            float(comparison.iloc[0]["replica_l4_sharpe_retention"])
            if not comparison.empty
            else float("nan")
        )
        if category == "current_dictionary_replication_failed":
            suggestion = "no_substitution_based_priority_downgrade"
            rationale = "No permitted replica explained 10% of stitched OOS signal variation."
        elif category in {"level0_partial_replication", "level1_single_proxy_partial_replication"}:
            suggestion = "review_priority_reduction_or_parallel_replica_shadow"
            rationale = "A simple permitted replica passed the partial-replication gate."
        elif category == "level2_single_family_partial_replication":
            suggestion = "retain_with_single_family_substitution_review"
            rationale = "A multi-scale single-family replica passed, but no simpler level passed."
        else:
            suggestion = "retain_with_full_dictionary_substitution_review"
            rationale = "Only the full cross-family dictionary passed; no simple substitute was found."
        rows.append(
            {
                "candidate_id": candidate_id,
                "evidence_type": "Inference",
                "replication_difficulty": category,
                "qualifying_models_at_simplest_level": classification.qualifying_models_at_simplest_level,
                "best_replica_model": best_model,
                "best_stitched_oos_r2": best_r2,
                "best_replica_l4_sharpe_retention": best_l4_retention,
                "shadow_priority_suggestion": suggestion,
                "rationale": rationale,
                "automatic_state_change": False,
            }
        )
    return pd.DataFrame(rows)


_L5_PREDICTION_COLUMNS = (
    "candidate_id",
    "model_id",
    "fold_idx",
    "decision_ts",
    "symbol",
    "target_signal",
    "replica_signal",
    "strategy_forward_return",
)
_L5_KEY_COLUMNS = ("fold_idx", "decision_ts", "symbol")


def prepare_level2_full_residual_predictions(
    predictions: pd.DataFrame,
    *,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Validate Level-2-full OOS predictions and recompute residuals.

    This is the only supported input adapter for the L5 residual-information
    test. It rejects duplicate keys, missing values, stale residuals, and any
    frame that does not contain the frozen ``level2_full`` model.
    """
    missing = sorted(set(_L5_PREDICTION_COLUMNS).difference(predictions.columns))
    if missing:
        raise ValueError("predictions missing L5 columns: " + ", ".join(missing))
    selected = predictions.loc[
        predictions["model_id"].astype(str) == "level2_full",
        [*_L5_PREDICTION_COLUMNS, *( ["residual_signal"] if "residual_signal" in predictions.columns else [])],
    ].copy()
    if selected.empty:
        raise ValueError("predictions contain no level2_full rows")
    selected["candidate_id"] = selected["candidate_id"].astype(str)
    selected["symbol"] = selected["symbol"].astype(str)
    selected["fold_idx"] = pd.to_numeric(selected["fold_idx"], errors="raise").astype(int)
    selected["decision_ts"] = pd.to_datetime(selected["decision_ts"], utc=True, errors="raise")
    numeric_columns = ["target_signal", "replica_signal", "strategy_forward_return"]
    selected[numeric_columns] = selected[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(selected[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("level2_full predictions contain non-finite required values")
    key_columns = ["candidate_id", *_L5_KEY_COLUMNS]
    if selected.duplicated(key_columns).any():
        raise ValueError("level2_full predictions contain duplicate candidate/fold/time/symbol keys")
    recomputed = selected["target_signal"] - selected["replica_signal"]
    if "residual_signal" in selected.columns:
        recorded = pd.to_numeric(selected["residual_signal"], errors="coerce")
        if recorded.isna().any() or not np.allclose(
            recorded.to_numpy(dtype=float),
            recomputed.to_numpy(dtype=float),
            rtol=0.0,
            atol=float(tolerance),
        ):
            raise ValueError("recorded residual_signal differs from target_signal - replica_signal")
    selected["residual_signal"] = recomputed
    return selected.sort_values(key_columns, kind="mergesort").reset_index(drop=True)


def _l5_signal_hash(frame: pd.DataFrame, value_column: str) -> str:
    digest = hashlib.sha256()
    keys = frame.loc[:, _L5_KEY_COLUMNS].copy()
    keys["decision_ts"] = pd.to_datetime(keys["decision_ts"], utc=True).astype("int64")
    digest.update(pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype="uint64").tobytes())
    digest.update(frame[value_column].to_numpy(dtype="float64").tobytes())
    return digest.hexdigest()


def audit_residual_signal_equivalence(
    prepared_predictions: pd.DataFrame,
    *,
    tolerance: float = 1e-12,
) -> ResidualEquivalenceArtifacts:
    """Deduplicate numerically identical OOS residual hypotheses.

    Target equality is assessed first on the complete fold/time/symbol map.
    Candidates with equal targets but unequal replicas or residuals are
    rejected as a lineage failure rather than forced into one hypothesis.
    """
    required = {
        "candidate_id", *_L5_KEY_COLUMNS, "target_signal", "replica_signal", "residual_signal"
    }
    missing = sorted(required.difference(prepared_predictions.columns))
    if missing:
        raise ValueError("prepared predictions missing equivalence columns: " + ", ".join(missing))
    candidate_frames = {
        str(candidate_id): group.sort_values(list(_L5_KEY_COLUMNS), kind="mergesort").reset_index(drop=True)
        for candidate_id, group in prepared_predictions.groupby("candidate_id", sort=True)
    }
    if not candidate_frames:
        raise ValueError("no candidates supplied for equivalence audit")
    representatives: list[str] = []
    aliases_by_representative: dict[str, list[str]] = {}
    for candidate_id, frame in candidate_frames.items():
        matches: list[str] = []
        for representative in representatives:
            reference = candidate_frames[representative]
            if len(frame) != len(reference):
                continue
            if not frame.loc[:, _L5_KEY_COLUMNS].equals(reference.loc[:, _L5_KEY_COLUMNS]):
                continue
            if np.allclose(
                frame["target_signal"].to_numpy(dtype=float),
                reference["target_signal"].to_numpy(dtype=float),
                rtol=0.0,
                atol=float(tolerance),
            ):
                matches.append(representative)
        if len(matches) > 1:
            raise ValueError("tolerance-based target equivalence is ambiguous")
        if not matches:
            representatives.append(candidate_id)
            aliases_by_representative[candidate_id] = [candidate_id]
        else:
            aliases_by_representative[matches[0]].append(candidate_id)

    mapping_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    for group_number, representative in enumerate(representatives, start=1):
        reference = candidate_frames[representative]
        aliases = aliases_by_representative[representative]
        for alias in aliases:
            frame = candidate_frames[alias]
            for column in ("replica_signal", "residual_signal"):
                if not np.allclose(
                    frame[column].to_numpy(dtype=float),
                    reference[column].to_numpy(dtype=float),
                    rtol=0.0,
                    atol=float(tolerance),
                ):
                    raise ValueError(
                        f"target-equivalent candidates differ in {column}: "
                        f"{representative} vs {alias}"
                    )
        equivalence_id = f"signal_{group_number:03d}"
        target_hash = _l5_signal_hash(reference, "target_signal")
        replica_hash = _l5_signal_hash(reference, "replica_signal")
        residual_hash = _l5_signal_hash(reference, "residual_signal")
        group_rows.append(
            {
                "signal_equivalence_id": equivalence_id,
                "canonical_candidate_id": representative,
                "alias_count": len(aliases),
                "candidate_aliases": "|".join(aliases),
                "row_count": len(reference),
                "target_signal_sha256": target_hash,
                "replica_signal_sha256": replica_hash,
                "residual_signal_sha256": residual_hash,
                "equivalence_status": "pass",
            }
        )
        for alias in aliases:
            mapping_rows.append(
                {
                    "candidate_id": alias,
                    "signal_equivalence_id": equivalence_id,
                    "canonical_candidate_id": representative,
                    "alias_count": len(aliases),
                    "target_signal_sha256": target_hash,
                    "replica_signal_sha256": replica_hash,
                    "residual_signal_sha256": residual_hash,
                }
            )
    return ResidualEquivalenceArtifacts(
        mapping=pd.DataFrame(mapping_rows).sort_values("candidate_id").reset_index(drop=True),
        groups=pd.DataFrame(group_rows),
    )


def summarize_residual_incremental_information(
    prepared_candidate_predictions: pd.DataFrame,
    *,
    signal_equivalence_id: str,
    min_cross_section: int,
    overlap_lags: int = 0,
) -> ResidualInformationArtifacts:
    """Compute the contracted residual Rank-IC series and HAC test summary."""
    candidate_ids = prepared_candidate_predictions["candidate_id"].astype(str).unique()
    if len(candidate_ids) != 1:
        raise ValueError("residual information summary requires exactly one canonical candidate")
    if prepared_candidate_predictions.duplicated(["decision_ts", "symbol"]).any():
        raise ValueError("decision_ts/symbol keys overlap across folds")
    frame = prepared_candidate_predictions.set_index(["decision_ts", "symbol"])
    diagnostics = factor_research.rank_ic_diagnostics_for_features(
        frame,
        ("residual_signal",),
        int(min_cross_section),
        return_column="strategy_forward_return",
    )["residual_signal"]
    timeseries = pd.DataFrame(diagnostics)
    fold_by_decision = (
        prepared_candidate_predictions[["decision_ts", "fold_idx"]]
        .drop_duplicates()
        .set_index("decision_ts")["fold_idx"]
    )
    if fold_by_decision.index.duplicated().any():
        raise ValueError("one decision_ts maps to multiple folds")
    timeseries.insert(0, "signal_equivalence_id", str(signal_equivalence_id))
    timeseries.insert(1, "candidate_id", candidate_ids[0])
    timeseries.insert(2, "fold_idx", timeseries["decision_ts"].map(fold_by_decision))
    valid = timeseries.loc[timeseries["status"] == "ok", "raw_rank_ic"].astype(float)
    observation_count = len(valid)
    mean_ic = float(valid.mean()) if observation_count else float("nan")
    std_ic = float(valid.std(ddof=1)) if observation_count > 1 else float("nan")
    icir = (
        float(mean_ic / std_ic)
        if observation_count > 1 and np.isfinite(std_ic) and std_ic > 0.0
        else float("nan")
    )
    hac_lags = research_stats.newey_west_max_lags(
        observation_count, overlap_lags=int(overlap_lags)
    )
    hac_t = research_stats.hac_t_stat(valid, max_lags=hac_lags)
    raw_p = research_stats.normal_two_sided_p_value(hac_t)
    test_status = "valid" if np.isfinite(raw_p) else "invalid"
    test_failure_reason = "" if test_status == "valid" else "insufficient_or_degenerate_rank_ic_series"
    effect_sign = (
        "positive" if mean_ic > 0.0 else "negative" if mean_ic < 0.0 else "zero"
    ) if np.isfinite(mean_ic) else "undefined"
    status_counts = timeseries["status"].value_counts()
    summary = pd.DataFrame(
        [
            {
                "signal_equivalence_id": str(signal_equivalence_id),
                "canonical_candidate_id": candidate_ids[0],
                "ic_observation_count": observation_count,
                "ic_mean": mean_ic,
                "ic_std": std_ic,
                "icir": icir,
                "hac_lags": hac_lags,
                "hac_t_stat": hac_t,
                "raw_two_sided_p_value": raw_p,
                "effect_sign": effect_sign,
                "test_status": test_status,
                "test_failure_reason": test_failure_reason,
                "decision_count": len(timeseries),
                "invalid_decision_count": int((timeseries["status"] != "ok").sum()),
                "small_cross_section_count": int(status_counts.get("small_cross_section", 0)),
                "constant_residual_count": int(status_counts.get("constant_feature", 0)),
                "constant_return_count": int(status_counts.get("constant_return", 0)),
                "nan_rank_ic_count": int(status_counts.get("nan_rank_ic", 0)),
            }
        ]
    )
    return ResidualInformationArtifacts(timeseries=timeseries, summary=summary)


_L5_TIME_SHIFT_DECISIONS_PER_DAY = {"4h": 6, "8h": 3, "12h": 2, "1d": 1}
_L5_TIME_SHIFT_EXPECTED_UNIQUE = {"4h": 14, "8h": 16, "12h": 25, "1d": 17}
_L5_TIME_SHIFT_GUARDS = (3, 7, 14)


def l5_time_randomization_legal_offset_manifest(
    *,
    expected_fold_days: int = 45,
    guard_days: Sequence[int] = _L5_TIME_SHIFT_GUARDS,
) -> pd.DataFrame:
    """Return the complete pre-registered circular day-offset sets."""
    fold_days = int(expected_fold_days)
    guards = tuple(int(value) for value in guard_days)
    if fold_days <= 1:
        raise ValueError("expected_fold_days must be greater than one")
    if guards != _L5_TIME_SHIFT_GUARDS:
        raise ValueError("guard_days must be exactly (3, 7, 14)")
    rows = []
    for guard in guards:
        legal = [
            offset
            for offset in range(1, fold_days)
            if min(offset, fold_days - offset) >= guard
        ]
        if not legal:
            raise ValueError(
                f"guard {guard} leaves no legal circular offsets"
            )
        rows.extend(
            {
                "guard_days": guard,
                "fold_days": fold_days,
                "offset_days": offset,
                "signed_offset_days": (
                    offset if offset <= fold_days // 2 else offset - fold_days
                ),
                "circular_distance_days": min(offset, fold_days - offset),
            }
            for offset in legal
        )
    return pd.DataFrame(rows)


def l5_time_randomization_seed_manifest(
    *,
    master_seed: int = 20_260_727,
    guard_days: Sequence[int] = _L5_TIME_SHIFT_GUARDS,
) -> pd.DataFrame:
    """Derive deterministic named PCG64DXSM substreams for L5 placebo guards."""
    guards = tuple(int(value) for value in guard_days)
    if guards != _L5_TIME_SHIFT_GUARDS:
        raise ValueError("guard_days must be exactly (3, 7, 14)")
    if int(master_seed) != 20_260_727:
        raise ValueError("master_seed must be 20260727")
    children = np.random.SeedSequence(int(master_seed)).spawn(len(guards))
    rows = []
    for child_index, (guard, child) in enumerate(zip(guards, children, strict=True)):
        child_seed = int(child.generate_state(1, dtype=np.uint64)[0])
        rows.append(
            {
                "guard_days": guard,
                "substream_name": f"guard_{guard}d",
                "child_index": child_index,
                "master_seed": int(master_seed),
                "child_seed_uint64": child_seed,
                "bit_generator": "PCG64DXSM",
            }
        )
    return pd.DataFrame(rows)


def _rowwise_rank_ic(
    feature_values: np.ndarray,
    return_values: np.ndarray,
    *,
    min_cross_section: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature = np.asarray(feature_values, dtype=float)
    returns = np.asarray(return_values, dtype=float)
    if feature.ndim != 2 or returns.shape != feature.shape:
        raise ValueError("rank IC arrays must be matching two-dimensional matrices")
    valid = np.isfinite(feature) & np.isfinite(returns)
    counts = valid.sum(axis=1)
    ranked_feature = (
        pd.DataFrame(np.where(valid, feature, np.nan))
        .rank(axis=1, method="average")
        .to_numpy(dtype=float)
    )
    ranked_return = (
        pd.DataFrame(np.where(valid, returns, np.nan))
        .rank(axis=1, method="average")
        .to_numpy(dtype=float)
    )
    feature_mean = np.divide(
        np.nansum(ranked_feature, axis=1),
        counts,
        out=np.full(len(feature), np.nan, dtype=float),
        where=counts > 0,
    )
    return_mean = np.divide(
        np.nansum(ranked_return, axis=1),
        counts,
        out=np.full(len(feature), np.nan, dtype=float),
        where=counts > 0,
    )
    feature_centered = ranked_feature - feature_mean[:, None]
    return_centered = ranked_return - return_mean[:, None]
    numerator = np.nansum(feature_centered * return_centered, axis=1)
    denominator = np.sqrt(
        np.nansum(feature_centered**2, axis=1)
        * np.nansum(return_centered**2, axis=1)
    )
    ok = (
        (counts >= int(min_cross_section))
        & np.isfinite(denominator)
        & (denominator > 0.0)
    )
    values = np.full(len(feature), np.nan, dtype=float)
    values[ok] = numerator[ok] / denominator[ok]
    return values, ok


def _l5_time_shift_key_sha256(
    times: pd.DatetimeIndex,
    symbols: Sequence[str],
    valid_mask: np.ndarray,
) -> str:
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.shape != (len(times), len(symbols)):
        raise ValueError("time-shift support mask has the wrong shape")
    digest = hashlib.sha256()
    digest.update(np.asarray(times.asi8, dtype="<i8").tobytes())
    digest.update("\x1f".join(str(symbol) for symbol in symbols).encode("utf-8"))
    digest.update(np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def _l5_time_shift_value_multiset_sha256(values: np.ndarray) -> str:
    flattened = np.asarray(values, dtype="<f8").reshape(-1)
    if not np.isfinite(flattened).all():
        raise ValueError("time-shift value multiset contains non-finite values")
    return hashlib.sha256(np.sort(flattened).tobytes()).hexdigest()


def prepare_l5_residual_time_shift_effects(
    predictions: pd.DataFrame,
    *,
    observation_keys: pd.DataFrame | None = None,
    expected_unique_signals: Mapping[str, int] | None = None,
    expected_fold_count: int = 11,
    expected_fold_days: int = 45,
    min_cross_section: int = 10,
    expected_observed_effects: pd.Series | None = None,
    observed_tolerance: float = 1e-12,
) -> ResidualTimeShiftPreparationArtifacts:
    """Precompute observed and fold/offset residual Rank-IC effects.

    The source return cross-section moves as one unit within each outer fold.
    Residual values, target observation keys, symbol identities, and fold
    boundaries remain fixed.
    """
    required = {
        "candidate_id",
        "signal_equivalence_id",
        "horizon",
        "fold_idx",
        "decision_ts",
        "symbol",
        "target_signal",
        "replica_signal",
        "residual_signal",
        "strategy_forward_return",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(
            "time-shift predictions missing columns: " + ", ".join(missing)
        )
    unique_contract = dict(expected_unique_signals or _L5_TIME_SHIFT_EXPECTED_UNIQUE)
    if not unique_contract or not set(unique_contract).issubset(
        _L5_TIME_SHIFT_DECISIONS_PER_DAY
    ):
        raise ValueError("unique-signal contract has the wrong horizon set")
    frame = predictions.loc[:, sorted(required)].copy()
    for column in ("candidate_id", "signal_equivalence_id", "horizon", "symbol"):
        frame[column] = frame[column].astype(str)
    frame["fold_idx"] = pd.to_numeric(frame["fold_idx"], errors="raise").astype(int)
    frame["decision_ts"] = pd.to_datetime(
        frame["decision_ts"], utc=True, errors="raise"
    )
    for column in (
        "target_signal",
        "replica_signal",
        "residual_signal",
        "strategy_forward_return",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(
        frame[
            [
                "target_signal",
                "replica_signal",
                "residual_signal",
                "strategy_forward_return",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("time-shift predictions contain non-finite values")
    recomputed_residual = frame["target_signal"] - frame["replica_signal"]
    if not np.allclose(
        frame["residual_signal"].to_numpy(dtype=float),
        recomputed_residual.to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "recorded residual_signal differs from target_signal - replica_signal"
        )
    frame["residual_signal"] = recomputed_residual
    key_columns = [
        "signal_equivalence_id",
        "fold_idx",
        "decision_ts",
        "symbol",
    ]
    if frame.duplicated(key_columns).any():
        raise ValueError("time-shift predictions contain duplicate hypothesis keys")

    identity = frame.groupby("signal_equivalence_id").agg(
        candidate_count=("candidate_id", "nunique"),
        horizon_count=("horizon", "nunique"),
    )
    if (identity != 1).any().any():
        raise ValueError("one signal equivalence id maps to multiple candidates or horizons")
    actual_counts = (
        frame[["signal_equivalence_id", "horizon"]]
        .drop_duplicates()
        .groupby("horizon")["signal_equivalence_id"]
        .nunique()
        .to_dict()
    )
    if actual_counts != unique_contract:
        raise ValueError(
            f"time-shift unique-signal counts changed: {actual_counts}"
        )
    if int(expected_fold_count) <= 0 or int(expected_fold_days) <= 0:
        raise ValueError("expected fold count and days must be positive")
    if int(min_cross_section) < 2:
        raise ValueError("min_cross_section must be at least two")

    selected_decisions: dict[tuple[str, int], set[pd.Timestamp]] | None = None
    if observation_keys is not None:
        key_required = {"signal_equivalence_id", "fold_idx", "decision_ts"}
        key_missing = sorted(key_required.difference(observation_keys.columns))
        if key_missing:
            raise ValueError(
                "observation keys missing columns: " + ", ".join(key_missing)
            )
        keys = observation_keys.loc[:, sorted(key_required)].copy()
        keys["signal_equivalence_id"] = keys["signal_equivalence_id"].astype(str)
        keys["fold_idx"] = pd.to_numeric(
            keys["fold_idx"], errors="raise"
        ).astype(int)
        keys["decision_ts"] = pd.to_datetime(
            keys["decision_ts"], utc=True, errors="raise"
        )
        keys = keys.drop_duplicates()
        if keys.duplicated(
            ["signal_equivalence_id", "decision_ts"]
        ).any():
            raise ValueError("one observation decision maps to multiple folds")
        if set(keys["signal_equivalence_id"]) != set(
            frame["signal_equivalence_id"]
        ):
            raise ValueError(
                "observation keys do not cover the complete hypothesis family"
            )
        selected_decisions = {
            (str(signal_id), int(fold_idx)): set(group["decision_ts"])
            for (signal_id, fold_idx), group in keys.groupby(
                ["signal_equivalence_id", "fold_idx"], sort=False
            )
        }

    return_key_columns = ["horizon", "fold_idx", "decision_ts", "symbol"]
    return_repetitions = frame.groupby(
        return_key_columns,
        sort=False,
    )["strategy_forward_return"].agg(["min", "max", "median"])
    return_repetitions["max_abs_difference"] = (
        return_repetitions["max"] - return_repetitions["min"]
    )
    return_duplicate_tolerance = 1e-15
    if (
        return_repetitions["max_abs_difference"] > return_duplicate_tolerance
    ).any():
        raise ValueError(
            "duplicated market return keys disagree across candidates beyond "
            "the frozen CSV round-trip tolerance"
        )
    canonical_returns = (
        return_repetitions["median"]
        .rename("strategy_forward_return")
        .reset_index()
    )
    return_difference_by_fold = (
        return_repetitions["max_abs_difference"]
        .groupby(level=["horizon", "fold_idx"])
        .max()
    )

    observed_rows: list[dict[str, object]] = []
    offset_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for horizon in _L5_TIME_SHIFT_DECISIONS_PER_DAY:
        if horizon not in unique_contract:
            continue
        horizon_frame = frame.loc[frame["horizon"] == horizon]
        fold_ids = sorted(horizon_frame["fold_idx"].unique().tolist())
        if fold_ids != list(range(int(expected_fold_count))):
            raise ValueError(f"{horizon}: outer fold ids changed")
        decisions_per_day = _L5_TIME_SHIFT_DECISIONS_PER_DAY[horizon]
        expected_decisions = int(expected_fold_days) * decisions_per_day
        expected_delta = pd.Timedelta(days=1) / decisions_per_day
        horizon_ids = sorted(horizon_frame["signal_equivalence_id"].unique())
        candidate_by_id = (
            horizon_frame[
                ["signal_equivalence_id", "candidate_id"]
            ]
            .drop_duplicates()
            .set_index("signal_equivalence_id")["candidate_id"]
        )
        observed_by_id = {
            signal_id: {"sum": 0.0, "count": 0}
            for signal_id in horizon_ids
        }
        for fold_idx in fold_ids:
            return_fold = canonical_returns.loc[
                (canonical_returns["horizon"] == horizon)
                & (canonical_returns["fold_idx"] == fold_idx)
            ]
            times = pd.DatetimeIndex(sorted(return_fold["decision_ts"].unique()))
            symbols = sorted(return_fold["symbol"].unique())
            if len(times) != expected_decisions:
                raise ValueError(
                    f"{horizon}/fold={fold_idx}: expected {expected_decisions} decisions"
                )
            if len(times) > 1 and not (times.to_series().diff().dropna() == expected_delta).all():
                raise ValueError(f"{horizon}/fold={fold_idx}: decision grid changed")
            returns = (
                return_fold.pivot(
                    index="decision_ts",
                    columns="symbol",
                    values="strategy_forward_return",
                )
                .reindex(index=times, columns=symbols)
            )
            if returns.isna().any().any():
                raise ValueError(
                    f"{horizon}/fold={fold_idx}: canonical return panel is incomplete"
                )
            return_values = returns.to_numpy(dtype=float)
            return_multiset_sha256 = _l5_time_shift_value_multiset_sha256(
                return_values
            )
            symbol_order_sha256 = hashlib.sha256(
                "\x1f".join(symbols).encode("utf-8")
            ).hexdigest()
            shifted_return_hashes: dict[int, str] = {}
            for offset_days in range(1, int(expected_fold_days)):
                shifted_hash = _l5_time_shift_value_multiset_sha256(
                    np.roll(
                        return_values,
                        -offset_days * decisions_per_day,
                        axis=0,
                    )
                )
                if shifted_hash != return_multiset_sha256:
                    raise ValueError(
                        f"{horizon}/fold={fold_idx}/offset={offset_days}: "
                        "time shift changed the return value multiset"
                    )
                shifted_return_hashes[offset_days] = shifted_hash
            fold_frame = horizon_frame.loc[horizon_frame["fold_idx"] == fold_idx]
            for signal_id in horizon_ids:
                candidate = fold_frame.loc[
                    fold_frame["signal_equivalence_id"] == signal_id
                ]
                residual = (
                    candidate.pivot(
                        index="decision_ts",
                        columns="symbol",
                        values="residual_signal",
                    )
                    .reindex(index=times, columns=symbols)
                )
                full_residual_values = residual.to_numpy(dtype=float)
                if selected_decisions is None:
                    evaluation_mask = np.ones(len(times), dtype=bool)
                else:
                    selected_times = selected_decisions.get(
                        (signal_id, int(fold_idx))
                    )
                    if not selected_times:
                        raise ValueError(
                            f"{signal_id}/fold={fold_idx}: observation keys are missing"
                        )
                    unknown_times = selected_times.difference(set(times))
                    if unknown_times:
                        raise ValueError(
                            f"{signal_id}/fold={fold_idx}: observation keys leave the fold grid"
                        )
                    evaluation_mask = times.isin(sorted(selected_times))
                residual_values = full_residual_values[evaluation_mask]
                observed_return_values = return_values[evaluation_mask]
                observed_valid_mask = (
                    np.isfinite(residual_values)
                    & np.isfinite(observed_return_values)
                )
                observation_key_sha256 = _l5_time_shift_key_sha256(
                    times[evaluation_mask],
                    symbols,
                    observed_valid_mask,
                )
                support_counts = np.isfinite(residual_values).sum(axis=1)
                if (support_counts < int(min_cross_section)).any():
                    raise ValueError(
                        f"{signal_id}/fold={fold_idx}: residual support is too small"
                    )
                observed_ic, observed_ok = _rowwise_rank_ic(
                    residual_values,
                    observed_return_values,
                    min_cross_section=int(min_cross_section),
                )
                if not observed_ok.all():
                    raise ValueError(
                        f"{signal_id}/fold={fold_idx}: observed Rank IC is invalid"
                    )
                observed_sum = float(observed_ic.sum())
                observed_count = int(observed_ok.sum())
                observed_by_id[signal_id]["sum"] += observed_sum
                observed_by_id[signal_id]["count"] += observed_count
                offset_counts: list[int] = []
                for offset_days in range(1, int(expected_fold_days)):
                    shifted_returns = np.roll(
                        return_values,
                        -offset_days * decisions_per_day,
                        axis=0,
                    )[evaluation_mask]
                    shifted_multiset_sha256 = shifted_return_hashes[offset_days]
                    shifted_key_sha256 = _l5_time_shift_key_sha256(
                        times[evaluation_mask],
                        symbols,
                        np.isfinite(residual_values)
                        & np.isfinite(shifted_returns),
                    )
                    if shifted_key_sha256 != observation_key_sha256:
                        raise ValueError(
                            f"{signal_id}/fold={fold_idx}/offset={offset_days}: "
                            "time shift changed observation keys"
                        )
                    shifted_ic, shifted_ok = _rowwise_rank_ic(
                        residual_values,
                        shifted_returns,
                        min_cross_section=int(min_cross_section),
                    )
                    shifted_count = int(shifted_ok.sum())
                    if shifted_count != observed_count:
                        raise ValueError(
                            f"{signal_id}/fold={fold_idx}/offset={offset_days}: "
                            "time shift changed the valid observation count"
                        )
                    offset_counts.append(shifted_count)
                    offset_rows.append(
                        {
                            "signal_equivalence_id": signal_id,
                            "candidate_id": str(candidate_by_id.loc[signal_id]),
                            "horizon": horizon,
                            "fold_idx": int(fold_idx),
                            "offset_days": int(offset_days),
                            "ic_sum": float(shifted_ic.sum()),
                            "ic_count": shifted_count,
                            "observation_key_sha256": observation_key_sha256,
                            "shifted_observation_key_sha256": shifted_key_sha256,
                            "symbol_order_sha256": symbol_order_sha256,
                            "source_return_multiset_sha256": (
                                return_multiset_sha256
                            ),
                            "shifted_return_multiset_sha256": (
                                shifted_multiset_sha256
                            ),
                            "invariant_status": "pass",
                        }
                    )
                audit_rows.append(
                    {
                        "signal_equivalence_id": signal_id,
                        "candidate_id": str(candidate_by_id.loc[signal_id]),
                        "horizon": horizon,
                        "fold_idx": int(fold_idx),
                        "fold_start": times.min(),
                        "fold_end": times.max(),
                        "fold_grid_decision_count": len(times),
                        "decision_count": int(evaluation_mask.sum()),
                        "symbol_count": len(symbols),
                        "min_residual_support": int(support_counts.min()),
                        "max_residual_support": int(support_counts.max()),
                        "observed_ic_count": observed_count,
                        "offset_count": int(expected_fold_days) - 1,
                        "offset_ic_count_min": min(offset_counts),
                        "offset_ic_count_max": max(offset_counts),
                        "observation_key_sha256": observation_key_sha256,
                        "symbol_order_sha256": symbol_order_sha256,
                        "source_return_multiset_sha256": (
                            return_multiset_sha256
                        ),
                        "duplicate_return_max_abs_difference": float(
                            return_difference_by_fold.loc[
                                (horizon, int(fold_idx))
                            ]
                        ),
                        "duplicate_return_tolerance": (
                            return_duplicate_tolerance
                        ),
                        "support_status": "pass",
                    }
                )
        for signal_id in horizon_ids:
            values = observed_by_id[signal_id]
            observed_rows.append(
                {
                    "signal_equivalence_id": signal_id,
                    "candidate_id": str(candidate_by_id.loc[signal_id]),
                    "horizon": horizon,
                    "observed_effect": values["sum"] / values["count"],
                    "observed_ic_count": int(values["count"]),
                }
            )

    observed = pd.DataFrame(observed_rows).sort_values(
        "signal_equivalence_id", kind="mergesort"
    ).reset_index(drop=True)
    if expected_observed_effects is not None:
        expected = pd.Series(expected_observed_effects).copy()
        expected.index = expected.index.astype(str)
        expected = expected.reindex(observed["signal_equivalence_id"])
        if expected.isna().any() or not np.allclose(
            observed["observed_effect"].to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
            rtol=0.0,
            atol=float(observed_tolerance),
        ):
            difference = np.abs(
                observed["observed_effect"].to_numpy(dtype=float)
                - expected.to_numpy(dtype=float)
            )
            raise ValueError(
                "recomputed observed residual Rank IC differs from frozen evidence; "
                f"max_abs_difference={float(np.nanmax(difference))}"
            )
    return ResidualTimeShiftPreparationArtifacts(
        observed_effects=observed,
        offset_fold_effects=pd.DataFrame(offset_rows).sort_values(
            ["horizon", "fold_idx", "offset_days", "signal_equivalence_id"],
            kind="mergesort",
        ).reset_index(drop=True),
        support_audit=pd.DataFrame(audit_rows).sort_values(
            ["horizon", "fold_idx", "signal_equivalence_id"],
            kind="mergesort",
        ).reset_index(drop=True),
    )


def evaluate_l5_residual_time_randomization(
    prepared: ResidualTimeShiftPreparationArtifacts,
    *,
    guard_days: int,
    child_seed: int,
    n_randomizations: int = 9_999,
    alpha: float = 0.05,
    expected_fold_days: int = 45,
    expected_unique_signals: Mapping[str, int] | None = None,
) -> ResidualTimeRandomizationArtifacts:
    """Evaluate one pre-registered L5 time-randomization guard."""
    guard = int(guard_days)
    if guard not in _L5_TIME_SHIFT_GUARDS:
        raise ValueError("guard_days must be one of 3, 7, or 14")
    if int(n_randomizations) != 9_999:
        raise ValueError("n_randomizations must be 9999")
    if float(alpha) != 0.05:
        raise ValueError("alpha must be 0.05")
    if int(child_seed) < 0:
        raise ValueError("child_seed must be non-negative")
    observed = prepared.observed_effects.copy()
    required_observed = {
        "signal_equivalence_id",
        "candidate_id",
        "horizon",
        "observed_effect",
        "observed_ic_count",
    }
    required_offset = {
        "signal_equivalence_id",
        "candidate_id",
        "horizon",
        "fold_idx",
        "offset_days",
        "ic_sum",
        "ic_count",
    }
    if missing := sorted(required_observed.difference(observed.columns)):
        raise ValueError(
            "prepared observed effects missing columns: " + ", ".join(missing)
        )
    offsets = prepared.offset_fold_effects.copy()
    if missing := sorted(required_offset.difference(offsets.columns)):
        raise ValueError(
            "prepared offset effects missing columns: " + ", ".join(missing)
        )
    unique_contract = dict(expected_unique_signals or _L5_TIME_SHIFT_EXPECTED_UNIQUE)
    if not unique_contract or not set(unique_contract).issubset(
        _L5_TIME_SHIFT_DECISIONS_PER_DAY
    ):
        raise ValueError("unique-signal contract has the wrong horizon set")
    actual_counts = observed.groupby("horizon")["signal_equivalence_id"].nunique().to_dict()
    if actual_counts != unique_contract:
        raise ValueError(
            f"prepared observed effects have the wrong hypothesis family: {actual_counts}"
        )
    if observed["signal_equivalence_id"].duplicated().any():
        raise ValueError("prepared observed effects contain duplicate hypotheses")
    hypothesis_ids = observed["signal_equivalence_id"].astype(str).tolist()
    hypothesis_index = {
        signal_id: index for index, signal_id in enumerate(hypothesis_ids)
    }
    try:
        legal_manifest = l5_time_randomization_legal_offset_manifest(
            expected_fold_days=int(expected_fold_days)
        )
    except ValueError as error:
        raise ValueError("guard leaves no legal circular offsets") from error
    legal_offsets = legal_manifest.loc[
        legal_manifest["guard_days"] == guard, "offset_days"
    ].astype(int).tolist()
    actual_offsets = sorted(pd.to_numeric(offsets["offset_days"]).unique().tolist())
    if actual_offsets != list(range(1, int(expected_fold_days))):
        raise ValueError("prepared offset effects do not cover the complete fold")

    rng = np.random.Generator(np.random.PCG64DXSM(int(child_seed)))
    null_sums = np.zeros((int(n_randomizations), len(hypothesis_ids)), dtype=float)
    null_counts = np.zeros_like(null_sums)
    schedule_rows: list[pd.DataFrame] = []
    audit_required = {
        "horizon",
        "fold_idx",
        "fold_start",
        "fold_end",
        "fold_grid_decision_count",
    }
    if missing := sorted(audit_required.difference(prepared.support_audit.columns)):
        raise ValueError(
            "prepared support audit missing columns: " + ", ".join(missing)
        )
    boundary_counts = prepared.support_audit.groupby(
        ["horizon", "fold_idx"], sort=True
    ).agg(
        fold_start_count=("fold_start", "nunique"),
        fold_end_count=("fold_end", "nunique"),
        decision_count_count=("fold_grid_decision_count", "nunique"),
    )
    if (boundary_counts != 1).any().any():
        raise ValueError("candidate fold calendars disagree within a horizon")
    boundaries = prepared.support_audit.drop_duplicates(
        ["horizon", "fold_idx"]
    )[
        [
            "horizon",
            "fold_idx",
            "fold_start",
            "fold_end",
            "fold_grid_decision_count",
        ]
    ].copy()
    boundaries["fold_start"] = pd.to_datetime(
        boundaries["fold_start"], utc=True, errors="raise"
    )
    boundaries["fold_end"] = pd.to_datetime(
        boundaries["fold_end"], utc=True, errors="raise"
    )
    boundary_by_key = {
        (str(row.horizon), int(row.fold_idx)): (
            int(row.fold_start.value),
            int(row.fold_end.value),
        )
        for row in boundaries.itertuples(index=False)
    }
    expected_boundary_keys = set(
        offsets[["horizon", "fold_idx"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    expected_boundary_keys = {
        (str(horizon), int(fold_idx))
        for horizon, fold_idx in expected_boundary_keys
    }
    if set(boundary_by_key) != expected_boundary_keys:
        raise ValueError("support audit does not cover every horizon/fold calendar")
    signature_counts: dict[tuple[int, tuple[int, int]], int] = {}
    for (horizon, fold_idx), signature in boundary_by_key.items():
        del horizon
        key = (fold_idx, signature)
        signature_counts[key] = signature_counts.get(key, 0) + 1
    chosen_by_calendar: dict[
        tuple[int, tuple[int, int]], np.ndarray
    ] = {}
    for (horizon, fold_idx), fold_offsets in offsets.groupby(
        ["horizon", "fold_idx"], sort=True
    ):
        fold_ids = sorted(fold_offsets["signal_equivalence_id"].astype(str).unique())
        expected_ids = observed.loc[
            observed["horizon"].astype(str) == str(horizon),
            "signal_equivalence_id",
        ].astype(str).tolist()
        if fold_ids != sorted(expected_ids):
            raise ValueError(f"{horizon}/fold={fold_idx}: hypothesis set changed")
        calendar_key = (
            int(fold_idx),
            boundary_by_key[(str(horizon), int(fold_idx))],
        )
        if calendar_key not in chosen_by_calendar:
            chosen_by_calendar[calendar_key] = rng.choice(
                np.asarray(legal_offsets, dtype=int),
                size=int(n_randomizations),
                replace=True,
            )
        chosen = chosen_by_calendar[calendar_key]
        schedule_scope = (
            "shared_exact_fold_calendar"
            if signature_counts[calendar_key] > 1
            else "horizon_fold"
        )
        schedule_rows.append(
            pd.DataFrame(
                {
                    "randomization_idx": np.arange(
                        int(n_randomizations), dtype=int
                    ),
                    "horizon": str(horizon),
                    "fold_idx": int(fold_idx),
                    "offset_days": chosen,
                    "guard_days": guard,
                    "child_seed_uint64": int(child_seed),
                    "schedule_scope": schedule_scope,
                }
            )
        )
        for signal_id in expected_ids:
            candidate_offsets = (
                fold_offsets.loc[
                    fold_offsets["signal_equivalence_id"].astype(str) == signal_id
                ]
                .set_index("offset_days")
                .reindex(legal_offsets)
            )
            if candidate_offsets[["ic_sum", "ic_count"]].isna().any().any():
                raise ValueError(
                    f"{signal_id}/fold={fold_idx}: legal offset effects are incomplete"
                )
            sums_by_offset = candidate_offsets["ic_sum"]
            counts_by_offset = candidate_offsets["ic_count"]
            column_index = hypothesis_index[signal_id]
            null_sums[:, column_index] += sums_by_offset.loc[chosen].to_numpy(
                dtype=float
            )
            null_counts[:, column_index] += counts_by_offset.loc[chosen].to_numpy(
                dtype=float
            )
    if not np.isfinite(null_sums).all() or not np.isfinite(null_counts).all():
        raise ValueError("randomized effect aggregates are non-finite")
    expected_counts = observed.set_index("signal_equivalence_id")[
        "observed_ic_count"
    ].reindex(hypothesis_ids).to_numpy(dtype=float)
    if not np.allclose(
        null_counts,
        expected_counts[None, :],
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("time randomization changed candidate observation counts")
    null_effect_values = null_sums / null_counts
    null_effects = pd.DataFrame(
        null_effect_values,
        index=pd.RangeIndex(int(n_randomizations), name="randomization_idx"),
        columns=hypothesis_ids,
    )
    inference = research_stats.randomization_stepdown_max_t(
        null_effects,
        observed.set_index("signal_equivalence_id")["observed_effect"],
        min_randomizations=9_999,
    )
    summary = inference.summary.rename(
        columns={"hypothesis_id": "signal_equivalence_id"}
    ).merge(
        observed[
            [
                "signal_equivalence_id",
                "candidate_id",
                "horizon",
                "observed_ic_count",
            ]
        ],
        on="signal_equivalence_id",
        how="left",
        validate="one_to_one",
    )
    summary["guard_days"] = guard
    summary["child_seed_uint64"] = int(child_seed)
    summary["familywise_alpha"] = float(alpha)
    summary["effect_sign"] = np.where(
        summary["observed_effect"] > 0.0,
        "positive",
        np.where(summary["observed_effect"] < 0.0, "negative", "zero"),
    )
    summary["time_alignment_label"] = np.where(
        (summary["observed_effect"] > 0.0)
        & (summary["stepdown_max_t_adjusted_p_value"] <= float(alpha)),
        "time_alignment_detected",
        "time_alignment_not_detected",
    )
    return ResidualTimeRandomizationArtifacts(
        summary=summary.sort_values(
            "signal_equivalence_id", kind="mergesort"
        ).reset_index(drop=True),
        null_effects=null_effects,
        null_t_values=inference.null_t_values,
        schedule=pd.concat(schedule_rows, ignore_index=True).sort_values(
            ["randomization_idx", "horizon", "fold_idx"],
            kind="mergesort",
        ).reset_index(drop=True),
    )


def summarize_l5_residual_time_randomization(
    results: Mapping[int, ResidualTimeRandomizationArtifacts],
    *,
    candidate_metadata: pd.DataFrame,
    prior_observed_effects: pd.DataFrame,
    support_audit: pd.DataFrame,
) -> ResidualTimeRandomizationSummaryArtifacts:
    """Assemble the pre-registered candidate and sensitivity summaries."""
    guards = set(_L5_TIME_SHIFT_GUARDS)
    if set(results) != guards:
        raise ValueError("time-randomization results must cover guards 3, 7, and 14")
    required = {
        "signal_equivalence_id",
        "candidate_id",
        "horizon",
        "observed_effect",
        "observed_ic_count",
        "null_mean",
        "null_std",
        "null_median",
        "null_q95",
        "null_q99",
        "observed_null_percentile",
        "observed_t",
        "raw_one_sided_p_value",
        "raw_p_mcse",
        "stepdown_max_t_adjusted_p_value",
        "n_randomizations",
        "guard_days",
        "effect_sign",
        "time_alignment_label",
    }
    summaries: dict[int, pd.DataFrame] = {}
    family_ids: set[str] | None = None
    for guard in _L5_TIME_SHIFT_GUARDS:
        summary = results[guard].summary.copy()
        if missing := sorted(required.difference(summary.columns)):
            raise ValueError(
                f"guard {guard} summary missing columns: " + ", ".join(missing)
            )
        summary["signal_equivalence_id"] = summary[
            "signal_equivalence_id"
        ].astype(str)
        if summary["signal_equivalence_id"].duplicated().any():
            raise ValueError(f"guard {guard} summary contains duplicate hypotheses")
        if set(pd.to_numeric(summary["guard_days"], errors="raise").astype(int)) != {
            guard
        }:
            raise ValueError(f"guard {guard} summary has the wrong guard_days")
        if not pd.to_numeric(
            summary["n_randomizations"], errors="raise"
        ).eq(9_999).all():
            raise ValueError(f"guard {guard} summary is not based on 9999 randomizations")
        current_ids = set(summary["signal_equivalence_id"])
        if family_ids is None:
            family_ids = current_ids
        elif current_ids != family_ids:
            raise ValueError("time-randomization guard families disagree")
        summaries[guard] = summary
    if family_ids is None or len(family_ids) != sum(
        _L5_TIME_SHIFT_EXPECTED_UNIQUE.values()
    ):
        raise ValueError("time-randomization family must contain 72 hypotheses")

    main = summaries[7].rename(
        columns={
            "null_median": "main_7d_null_median",
            "null_q95": "main_7d_null_q95",
            "null_q99": "main_7d_null_q99",
            "observed_null_percentile": "main_7d_observed_percentile",
            "raw_one_sided_p_value": "main_7d_raw_p_value",
            "stepdown_max_t_adjusted_p_value": "main_7d_stepdown_p_value",
            "time_alignment_label": "main_7d_time_alignment_label",
        }
    )
    candidate_results = main[
        [
            "signal_equivalence_id",
            "candidate_id",
            "horizon",
            "observed_effect",
            "observed_ic_count",
            "main_7d_null_median",
            "main_7d_null_q95",
            "main_7d_null_q99",
            "main_7d_observed_percentile",
            "main_7d_raw_p_value",
            "main_7d_stepdown_p_value",
            "main_7d_time_alignment_label",
            "null_mean",
            "null_std",
            "observed_t",
            "raw_p_mcse",
            "n_randomizations",
            "effect_sign",
        ]
    ].copy()

    prior_required = ["signal_equivalence_id", "observed_effect"]
    if missing := sorted(set(prior_required).difference(prior_observed_effects.columns)):
        raise ValueError(
            "prior observed effects missing columns: " + ", ".join(missing)
        )
    prior = prior_observed_effects[prior_required].copy()
    prior["signal_equivalence_id"] = prior["signal_equivalence_id"].astype(str)
    if (
        prior["signal_equivalence_id"].duplicated().any()
        or set(prior["signal_equivalence_id"]) != family_ids
    ):
        raise ValueError("prior observed effects do not uniquely cover the family")
    candidate_results = candidate_results.merge(
        prior.rename(
            columns={"observed_effect": "prior_in_memory_observed_effect"}
        ),
        on="signal_equivalence_id",
        how="left",
        validate="one_to_one",
    )
    candidate_results["serialized_input_effect_difference"] = (
        candidate_results["observed_effect"]
        - candidate_results["prior_in_memory_observed_effect"]
    )
    for guard in (3, 14):
        candidate_results = candidate_results.merge(
            summaries[guard][
                [
                    "signal_equivalence_id",
                    "stepdown_max_t_adjusted_p_value",
                    "time_alignment_label",
                ]
            ].rename(
                columns={
                    "stepdown_max_t_adjusted_p_value": (
                        f"guard_{guard}d_stepdown_p_value"
                    ),
                    "time_alignment_label": (
                        f"guard_{guard}d_time_alignment_label"
                    ),
                }
            ),
            on="signal_equivalence_id",
            how="left",
            validate="one_to_one",
        )
    label_columns = [
        "guard_3d_time_alignment_label",
        "main_7d_time_alignment_label",
        "guard_14d_time_alignment_label",
    ]
    candidate_results["guard_label_stable"] = (
        candidate_results[label_columns].nunique(axis=1) == 1
    )
    candidate_results["test_status"] = "valid"
    candidate_results["test_failure_reason"] = ""

    metadata_required = [
        "signal_equivalence_id",
        "canonical_candidate_id",
        "alias_count",
        "track",
        "weight_scheme",
        "component_features",
    ]
    if missing := sorted(set(metadata_required).difference(candidate_metadata.columns)):
        raise ValueError("candidate metadata missing columns: " + ", ".join(missing))
    metadata = candidate_metadata[metadata_required].copy()
    metadata["signal_equivalence_id"] = metadata["signal_equivalence_id"].astype(str)
    if (
        metadata["signal_equivalence_id"].duplicated().any()
        or set(metadata["signal_equivalence_id"]) != family_ids
    ):
        raise ValueError("candidate metadata do not uniquely cover the family")
    candidate_results = candidate_results.merge(
        metadata,
        on="signal_equivalence_id",
        how="left",
        validate="one_to_one",
    )

    legal = l5_time_randomization_legal_offset_manifest()
    sensitivity_summary = pd.DataFrame(
        [
            {
                "guard_days": guard,
                "detected_count": int(
                    summaries[guard]["time_alignment_label"]
                    .eq("time_alignment_detected")
                    .sum()
                ),
                "candidate_count": len(summaries[guard]),
                "randomization_count": int(
                    summaries[guard]["n_randomizations"].iloc[0]
                ),
                "legal_offset_count": int(legal["guard_days"].eq(guard).sum()),
            }
            for guard in _L5_TIME_SHIFT_GUARDS
        ]
    )
    horizon_summary = (
        candidate_results.assign(
            detected=candidate_results["main_7d_time_alignment_label"].eq(
                "time_alignment_detected"
            )
        )
        .groupby("horizon", as_index=False)
        .agg(
            candidate_count=("signal_equivalence_id", "size"),
            main_7d_detected_count=("detected", "sum"),
            median_observed_effect=("observed_effect", "median"),
            median_main_7d_adjusted_p=("main_7d_stepdown_p_value", "median"),
            guard_label_stable_count=("guard_label_stable", "sum"),
        )
    )
    support_required = {
        "signal_equivalence_id",
        "horizon",
        "fold_grid_decision_count",
        "decision_count",
        "min_residual_support",
        "max_residual_support",
        "duplicate_return_max_abs_difference",
        "duplicate_return_tolerance",
        "support_status",
    }
    if missing := sorted(support_required.difference(support_audit.columns)):
        raise ValueError("support audit missing columns: " + ", ".join(missing))
    support = support_audit.copy()
    support["signal_equivalence_id"] = support["signal_equivalence_id"].astype(str)
    if set(support["signal_equivalence_id"]) != family_ids:
        raise ValueError("support audit does not cover the 72-hypothesis family")
    if set(support["support_status"].astype(str)) != {"pass"}:
        raise ValueError("support audit contains a non-passing row")
    support_summary = (
        support.groupby("horizon", as_index=False)
        .agg(
            candidate_fold_count=("signal_equivalence_id", "size"),
            fold_grid_decision_count_min=("fold_grid_decision_count", "min"),
            fold_grid_decision_count_max=("fold_grid_decision_count", "max"),
            evaluated_decision_count_min=("decision_count", "min"),
            evaluated_decision_count_max=("decision_count", "max"),
            min_residual_support=("min_residual_support", "min"),
            max_residual_support=("max_residual_support", "max"),
            duplicate_return_max_abs_difference=(
                "duplicate_return_max_abs_difference",
                "max",
            ),
            duplicate_return_tolerance=("duplicate_return_tolerance", "max"),
            support_status=(
                "support_status",
                lambda values: "|".join(sorted(set(values))),
            ),
        )
    )
    return ResidualTimeRandomizationSummaryArtifacts(
        candidate_results=candidate_results,
        sensitivity_summary=sensitivity_summary,
        horizon_summary=horizon_summary,
        support_summary=support_summary,
    )


def summarize_horizon_residual_family_information(
    rank_ic_timeseries: pd.DataFrame,
    candidate_mapping: pd.DataFrame,
    *,
    horizon: str,
    expected_unique_signal_count: int,
    overlap_lags: int = 0,
) -> ResidualFamilyInformationArtifacts:
    """Test a horizon's residual Rank IC after equal weighting by candidate track.

    Each decision timestamp contributes exactly one observation. Unique signals
    are first averaged within their frozen construction track, then tracks are
    averaged equally so alpha-grid multiplicity cannot determine track weight.
    """
    required_timeseries = {
        "signal_equivalence_id",
        "fold_idx",
        "decision_ts",
        "status",
        "raw_rank_ic",
    }
    missing_timeseries = sorted(required_timeseries.difference(rank_ic_timeseries.columns))
    if missing_timeseries:
        raise ValueError(
            "rank IC timeseries missing family-test columns: "
            + ", ".join(missing_timeseries)
        )
    required_mapping = {"signal_equivalence_id", "horizon", "track"}
    missing_mapping = sorted(required_mapping.difference(candidate_mapping.columns))
    if missing_mapping:
        raise ValueError(
            "candidate mapping missing family-test columns: " + ", ".join(missing_mapping)
        )
    if int(expected_unique_signal_count) <= 0:
        raise ValueError("expected_unique_signal_count must be positive")

    mapping = candidate_mapping.loc[:, sorted(required_mapping)].copy()
    mapping["signal_equivalence_id"] = mapping["signal_equivalence_id"].astype(str)
    mapping["horizon"] = mapping["horizon"].astype(str)
    mapping["track"] = mapping["track"].astype(str)
    if set(mapping["horizon"]) != {str(horizon)}:
        raise ValueError("candidate mapping must contain exactly the requested horizon")
    identity_counts = mapping.groupby("signal_equivalence_id").agg(
        horizon_count=("horizon", "nunique"),
        track_count=("track", "nunique"),
    )
    if (identity_counts[["horizon_count", "track_count"]] != 1).any().any():
        raise ValueError("one signal equivalence id maps to multiple horizons or tracks")
    signal_metadata = mapping.drop_duplicates(
        ["signal_equivalence_id", "horizon", "track"]
    ).sort_values("signal_equivalence_id")
    if len(signal_metadata) != int(expected_unique_signal_count):
        raise ValueError(
            "unique signal count mismatch; "
            f"expected={int(expected_unique_signal_count)}, actual={len(signal_metadata)}"
        )
    if (signal_metadata["track"].str.len() == 0).any():
        raise ValueError("candidate mapping contains an empty track")

    timeseries = rank_ic_timeseries.loc[:, sorted(required_timeseries)].copy()
    timeseries["signal_equivalence_id"] = timeseries["signal_equivalence_id"].astype(str)
    timeseries["decision_ts"] = pd.to_datetime(
        timeseries["decision_ts"], utc=True, errors="raise"
    )
    if timeseries.duplicated(["signal_equivalence_id", "decision_ts"]).any():
        raise ValueError("rank IC timeseries contains duplicate signal/decision keys")
    expected_ids = set(signal_metadata["signal_equivalence_id"])
    actual_ids = set(timeseries["signal_equivalence_id"])
    if actual_ids != expected_ids:
        raise ValueError(
            "rank IC timeseries and candidate mapping disagree on unique signals; "
            f"missing={sorted(expected_ids - actual_ids)}, extra={sorted(actual_ids - expected_ids)}"
        )
    if set(timeseries["status"].astype(str)) != {"ok"}:
        raise ValueError("rank IC timeseries contains a non-ok observation")
    timeseries["raw_rank_ic"] = pd.to_numeric(timeseries["raw_rank_ic"], errors="coerce")
    if not np.isfinite(timeseries["raw_rank_ic"].to_numpy(dtype=float)).all():
        raise ValueError("rank IC timeseries contains a non-finite value")
    fold_counts = timeseries.groupby("decision_ts")["fold_idx"].nunique()
    if (fold_counts != 1).any():
        raise ValueError("one decision timestamp maps to multiple folds")

    pivot = timeseries.pivot(
        index="decision_ts",
        columns="signal_equivalence_id",
        values="raw_rank_ic",
    ).sort_index()
    if pivot.empty or pivot.isna().any().any():
        raise ValueError("rank IC timeseries is not a complete balanced signal panel")
    if list(pivot.columns) != sorted(expected_ids):
        pivot = pivot.reindex(columns=sorted(expected_ids))
    expected_rows = len(pivot) * len(expected_ids)
    if len(timeseries) != expected_rows:
        raise ValueError("rank IC timeseries row count is inconsistent with a balanced panel")

    fold_by_decision = (
        timeseries[["decision_ts", "fold_idx"]]
        .drop_duplicates()
        .set_index("decision_ts")["fold_idx"]
        .reindex(pivot.index)
    )
    track_parts: list[pd.DataFrame] = []
    track_columns: dict[str, pd.Series] = {}
    for track, ids in signal_metadata.groupby("track", sort=True)["signal_equivalence_id"]:
        members = sorted(ids.astype(str))
        values = pivot.loc[:, members].mean(axis=1)
        track_columns[str(track)] = values
        track_parts.append(
            pd.DataFrame(
                {
                    "horizon": str(horizon),
                    "decision_ts": pivot.index,
                    "fold_idx": fold_by_decision.to_numpy(),
                    "track": str(track),
                    "unique_signal_count": len(members),
                    "track_residual_ic": values.to_numpy(dtype=float),
                }
            )
        )
    if not track_parts:
        raise ValueError("candidate mapping contains no tracks")
    track_timeseries = pd.concat(track_parts, ignore_index=True)
    track_wide = pd.DataFrame(track_columns, index=pivot.index).sort_index(axis=1)
    family_timeseries = pd.DataFrame(
        {
            "horizon": str(horizon),
            "decision_ts": pivot.index,
            "fold_idx": fold_by_decision.to_numpy(),
            "track_count": track_wide.shape[1],
            "unique_signal_count": len(expected_ids),
            "family_residual_ic": track_wide.mean(axis=1).to_numpy(dtype=float),
            "equal_unique_signal_residual_ic": pivot.mean(axis=1).to_numpy(dtype=float),
            "track_median_residual_ic": track_wide.median(axis=1).to_numpy(dtype=float),
        }
    )
    primary = family_timeseries["family_residual_ic"].astype(float)
    observation_count = len(primary)
    mean_ic = float(primary.mean()) if observation_count else float("nan")
    std_ic = float(primary.std(ddof=1)) if observation_count > 1 else float("nan")
    icir = (
        float(mean_ic / std_ic)
        if observation_count > 1 and np.isfinite(std_ic) and std_ic > 0.0
        else float("nan")
    )
    hac_lags = research_stats.newey_west_max_lags(
        observation_count, overlap_lags=int(overlap_lags)
    )
    hac_t = research_stats.hac_t_stat(primary, max_lags=hac_lags)
    raw_p = research_stats.normal_one_sided_p_value(hac_t)
    test_status = "valid" if np.isfinite(raw_p) else "invalid"
    signal_means = pivot.mean(axis=0)
    signal_summary = signal_metadata.copy()
    signal_summary["signal_ic_mean"] = signal_summary["signal_equivalence_id"].map(
        signal_means
    )
    signal_summary = signal_summary[
        ["horizon", "track", "signal_equivalence_id", "signal_ic_mean"]
    ].sort_values(["track", "signal_equivalence_id"]).reset_index(drop=True)
    summary = pd.DataFrame(
        [
            {
                "horizon": str(horizon),
                "unique_signal_count": len(expected_ids),
                "track_count": track_wide.shape[1],
                "ic_observation_count": observation_count,
                "family_ic_mean": mean_ic,
                "family_ic_std": std_ic,
                "family_icir": icir,
                "hac_lags": hac_lags,
                "hac_t_stat": hac_t,
                "raw_one_sided_p_value": raw_p,
                "effect_sign": (
                    "positive" if mean_ic > 0.0 else "negative" if mean_ic < 0.0 else "zero"
                )
                if np.isfinite(mean_ic)
                else "undefined",
                "test_status": test_status,
                "test_failure_reason": ""
                if test_status == "valid"
                else "insufficient_or_degenerate_family_ic_series",
                "equal_unique_signal_ic_mean": float(
                    family_timeseries["equal_unique_signal_residual_ic"].mean()
                ),
                "track_median_ic_mean": float(
                    family_timeseries["track_median_residual_ic"].mean()
                ),
                "positive_unique_signal_count": int((signal_means > 0.0).sum()),
                "positive_unique_signal_proportion": float((signal_means > 0.0).mean()),
                "min_unique_signal_ic_mean": float(signal_means.min()),
                "max_unique_signal_ic_mean": float(signal_means.max()),
            }
        ]
    )
    return ResidualFamilyInformationArtifacts(
        track_timeseries=track_timeseries,
        family_timeseries=family_timeseries,
        signal_summary=signal_summary,
        summary=summary,
    )


def summarize_residual_candidate_sensitivity(
    signal_summary: pd.DataFrame,
    frozen_candidate_summary: pd.DataFrame,
    *,
    horizon: str,
    expected_unique_signal_count: int,
) -> ResidualCandidateSensitivityArtifacts:
    """Attach frozen candidate-level inference as secondary sensitivity evidence."""
    signal_required = {"horizon", "track", "signal_equivalence_id", "signal_ic_mean"}
    frozen_required = {
        "signal_equivalence_id",
        "raw_two_sided_p_value",
        "holm_adjusted_p_value",
        "incremental_information_label",
    }
    signal_missing = sorted(signal_required.difference(signal_summary.columns))
    frozen_missing = sorted(frozen_required.difference(frozen_candidate_summary.columns))
    if signal_missing:
        raise ValueError(
            "signal sensitivity summary missing columns: " + ", ".join(signal_missing)
        )
    if frozen_missing:
        raise ValueError(
            "frozen candidate summary missing sensitivity columns: "
            + ", ".join(frozen_missing)
        )
    signals = signal_summary.loc[:, sorted(signal_required)].copy()
    signals["horizon"] = signals["horizon"].astype(str)
    signals["signal_equivalence_id"] = signals["signal_equivalence_id"].astype(str)
    if set(signals["horizon"]) != {str(horizon)}:
        raise ValueError("signal sensitivity summary has the wrong horizon")
    if signals["signal_equivalence_id"].duplicated().any():
        raise ValueError("signal sensitivity summary contains duplicate signals")
    if len(signals) != int(expected_unique_signal_count):
        raise ValueError("signal sensitivity summary has incomplete signal coverage")
    frozen = frozen_candidate_summary.loc[:, sorted(frozen_required)].copy()
    frozen["signal_equivalence_id"] = frozen["signal_equivalence_id"].astype(str)
    if frozen["signal_equivalence_id"].duplicated().any():
        raise ValueError("frozen candidate summary contains duplicate signals")
    if set(frozen["signal_equivalence_id"]) != set(signals["signal_equivalence_id"]):
        raise ValueError("frozen candidate summary and signal sensitivity summary disagree")
    detail = signals.merge(
        frozen, on="signal_equivalence_id", how="left", validate="one_to_one"
    ).sort_values(["track", "signal_equivalence_id"]).reset_index(drop=True)
    raw_p = pd.to_numeric(detail["raw_two_sided_p_value"], errors="coerce")
    holm_p = pd.to_numeric(detail["holm_adjusted_p_value"], errors="coerce")
    if not np.isfinite(raw_p.to_numpy(dtype=float)).all():
        raise ValueError("frozen candidate sensitivity contains non-finite raw p-values")
    if not np.isfinite(holm_p.to_numpy(dtype=float)).all():
        raise ValueError("frozen candidate sensitivity contains non-finite Holm p-values")
    label_counts = detail["incremental_information_label"].astype(str).value_counts()
    summary = pd.DataFrame(
        [
            {
                "horizon": str(horizon),
                "unique_signal_count": len(detail),
                "signal_ic_mean_mean": float(detail["signal_ic_mean"].mean()),
                "signal_ic_mean_median": float(detail["signal_ic_mean"].median()),
                "signal_ic_mean_min": float(detail["signal_ic_mean"].min()),
                "signal_ic_mean_max": float(detail["signal_ic_mean"].max()),
                "raw_two_sided_p_value_min": float(raw_p.min()),
                "raw_two_sided_p_value_median": float(raw_p.median()),
                "raw_two_sided_p_value_max": float(raw_p.max()),
                "raw_two_sided_p_le_0p05_count": int((raw_p <= 0.05).sum()),
                "prior_holm_detected_count": int(
                    label_counts.get("incremental_information_detected", 0)
                ),
                "prior_holm_not_detected_count": int(
                    label_counts.get("incremental_information_not_detected", 0)
                ),
                "prior_holm_invalid_count": int(
                    label_counts.get("incremental_information_test_invalid", 0)
                ),
            }
        ]
    )
    return ResidualCandidateSensitivityArtifacts(detail=detail, summary=summary)


def apply_horizon_residual_family_holm(
    horizon_summary: pd.DataFrame,
    *,
    expected_horizons: Sequence[str] = ("4h", "8h", "12h", "1d"),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Close the four-horizon family and assign neutral directional labels."""
    required = {"horizon", "raw_one_sided_p_value", "test_status", "effect_sign"}
    missing = sorted(required.difference(horizon_summary.columns))
    if missing:
        raise ValueError(
            "horizon family summary missing correction columns: " + ", ".join(missing)
        )
    expected = [str(value) for value in expected_horizons]
    if len(expected) != 4 or len(set(expected)) != 4:
        raise ValueError("expected_horizons must contain exactly four unique horizons")
    summary = horizon_summary.copy()
    summary["horizon"] = summary["horizon"].astype(str)
    if summary["horizon"].duplicated().any():
        raise ValueError("horizon family summary contains duplicate horizons")
    actual = set(summary["horizon"])
    if actual != set(expected):
        raise ValueError(
            "four-horizon family is incomplete; "
            f"missing={sorted(set(expected) - actual)}, extra={sorted(actual - set(expected))}"
        )
    summary = summary.set_index("horizon").loc[expected].reset_index()
    valid_mask = summary["test_status"].astype(str) == "valid"
    raw = pd.to_numeric(summary["raw_one_sided_p_value"], errors="coerce")
    valid_raw = raw[valid_mask].to_numpy(dtype=float)
    if not np.isfinite(valid_raw).all():
        raise ValueError("valid horizon has no finite one-sided p-value")
    if ((valid_raw < 0.0) | (valid_raw > 1.0)).any():
        raise ValueError("valid horizon one-sided p-value must be between 0 and 1")
    correction_input = raw.where(valid_mask, 1.0).to_numpy(dtype=float)
    adjusted = research_stats.holm_adjusted_p_values(correction_input)
    summary["holm_adjusted_p_value"] = np.where(valid_mask, adjusted, np.nan)
    summary["holm_family_size"] = len(expected)
    summary["horizon_family_label"] = np.where(
        ~valid_mask,
        "horizon_family_test_invalid",
        np.where(
            summary["holm_adjusted_p_value"] <= float(alpha),
            "horizon_family_incremental_information_detected",
            "horizon_family_incremental_information_not_detected",
        ),
    )
    if valid_mask.all():
        global_p = min(1.0, len(expected) * float(raw.min()))
        global_label = (
            "global_any_horizon_incremental_information_detected"
            if global_p <= float(alpha)
            else "global_any_horizon_incremental_information_not_detected"
        )
    else:
        global_p = float("nan")
        global_label = "global_horizon_family_test_invalid"
    summary["global_bonferroni_intersection_p_value"] = global_p
    summary["global_horizon_family_label"] = global_label
    return summary


def apply_train_selected_registered_replica_labels(
    unique_results: pd.DataFrame,
    candidate_results: pd.DataFrame,
) -> RegisteredResidualLabelArtifacts:
    """Translate dependency-adjusted outcomes into the registered-replica contract."""
    source_column = "candidate_incremental_information_label"
    if source_column not in unique_results.columns:
        raise ValueError("unique results are missing the dependency-adjusted label")
    if source_column not in candidate_results.columns:
        raise ValueError("candidate results are missing the dependency-adjusted label")
    label_mapping = {
        "candidate_incremental_information_detected": (
            "residual_information_detected_under_train_selected_registered_replica"
        ),
        "candidate_incremental_information_not_detected": (
            "residual_information_not_detected_under_train_selected_registered_replica"
        ),
        "candidate_incremental_information_test_invalid": "model_class_test_invalid",
    }

    def apply(frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        unknown = sorted(
            set(output[source_column].astype(str)).difference(label_mapping)
        )
        if unknown:
            raise ValueError(
                "dependency-adjusted results contain an unknown label: "
                + ", ".join(unknown)
            )
        output["registered_replica_residual_information_label"] = (
            output[source_column].astype(str).map(label_mapping)
        )
        return output.drop(columns=source_column)

    return RegisteredResidualLabelArtifacts(
        unique_results=apply(unique_results),
        candidate_results=apply(candidate_results),
    )


def evaluate_candidate_residual_dependency_adjusted_information(
    rank_ic_timeseries: pd.DataFrame,
    candidate_mapping: pd.DataFrame,
    equivalence_groups: pd.DataFrame,
    *,
    calendar_start: str | pd.Timestamp = "2024-12-22",
    calendar_end: str | pd.Timestamp = "2026-04-29",
    expected_daily_decisions: Mapping[str, int] | None = None,
    expected_unique_signals: Mapping[str, int] | None = None,
    expected_candidate_count: int = 96,
    block_lengths: Sequence[int] = (7, 14, 28),
    primary_block_length: int = 14,
    n_bootstrap: int = 20_000,
    seed: int = 20_260_724,
    alpha: float = 0.05,
) -> CandidateResidualDependencyArtifacts:
    """Run synchronized candidate-level residual-IC step-down maxT inference."""
    daily_decisions = dict(
        expected_daily_decisions or {"4h": 6, "8h": 3, "12h": 2, "1d": 1}
    )
    unique_counts = dict(
        expected_unique_signals or {"4h": 14, "8h": 16, "12h": 25, "1d": 17}
    )
    expected_horizons = tuple(daily_decisions)
    if set(unique_counts) != set(expected_horizons):
        raise ValueError("daily-decision and unique-signal horizon contracts disagree")
    if tuple(int(value) for value in block_lengths) != (7, 14, 28):
        raise ValueError("block_lengths must be exactly (7, 14, 28)")
    if int(primary_block_length) != 14:
        raise ValueError("primary_block_length must be 14")
    if int(n_bootstrap) != 20_000:
        raise ValueError("n_bootstrap must be 20000")
    if int(seed) != 20_260_724:
        raise ValueError("seed must be 20260724")

    timeseries_required = {
        "horizon",
        "signal_equivalence_id",
        "decision_ts",
        "status",
        "raw_rank_ic",
    }
    mapping_required = {
        "candidate_id",
        "signal_equivalence_id",
        "canonical_candidate_id",
        "alias_count",
        "horizon",
        "track",
        "weight_scheme",
        "component_features",
    }
    groups_required = {
        "signal_equivalence_id",
        "canonical_candidate_id",
        "alias_count",
    }
    missing_timeseries = sorted(timeseries_required.difference(rank_ic_timeseries.columns))
    missing_mapping = sorted(mapping_required.difference(candidate_mapping.columns))
    missing_groups = sorted(groups_required.difference(equivalence_groups.columns))
    if missing_timeseries:
        raise ValueError(
            "candidate dependency timeseries missing columns: "
            + ", ".join(missing_timeseries)
        )
    if missing_mapping:
        raise ValueError(
            "candidate dependency mapping missing columns: " + ", ".join(missing_mapping)
        )
    if missing_groups:
        raise ValueError(
            "candidate dependency groups missing columns: " + ", ".join(missing_groups)
        )

    mapping = candidate_mapping.copy()
    for column in (
        "candidate_id",
        "signal_equivalence_id",
        "canonical_candidate_id",
        "horizon",
        "track",
        "weight_scheme",
        "component_features",
    ):
        mapping[column] = mapping[column].astype(str)
    if len(mapping) != int(expected_candidate_count) or mapping["candidate_id"].duplicated().any():
        raise ValueError(
            "candidate dependency mapping does not cover "
            f"{int(expected_candidate_count)} unique candidates"
        )
    if set(mapping["horizon"]) != set(expected_horizons):
        raise ValueError("candidate dependency mapping horizon set changed")

    groups = equivalence_groups.loc[:, sorted(groups_required)].copy()
    groups["signal_equivalence_id"] = groups["signal_equivalence_id"].astype(str)
    groups["canonical_candidate_id"] = groups["canonical_candidate_id"].astype(str)
    if groups["signal_equivalence_id"].duplicated().any():
        raise ValueError("equivalence groups contain duplicate signal ids")
    if int(pd.to_numeric(groups["alias_count"], errors="raise").sum()) != int(
        expected_candidate_count
    ):
        raise ValueError(
            "equivalence alias counts do not cover "
            f"{int(expected_candidate_count)} candidates"
        )
    if set(mapping["signal_equivalence_id"]) != set(groups["signal_equivalence_id"]):
        raise ValueError("candidate mapping and equivalence groups disagree")
    actual_alias_counts = mapping.groupby("signal_equivalence_id").size()
    declared_alias_counts = groups.set_index("signal_equivalence_id")["alias_count"].astype(int)
    if not actual_alias_counts.sort_index().equals(declared_alias_counts.sort_index()):
        raise ValueError("candidate mapping alias counts disagree with equivalence groups")
    mapping_canonical_counts = mapping.groupby("signal_equivalence_id")[
        "canonical_candidate_id"
    ].nunique()
    if (mapping_canonical_counts != 1).any():
        raise ValueError("candidate mapping declares multiple canonicals for one signal")
    mapping_canonicals = mapping.groupby("signal_equivalence_id")[
        "canonical_candidate_id"
    ].first()
    declared_canonicals = groups.set_index("signal_equivalence_id")[
        "canonical_candidate_id"
    ]
    if not mapping_canonicals.sort_index().equals(declared_canonicals.sort_index()):
        raise ValueError(
            "candidate mapping canonical ids disagree with equivalence groups"
        )
    canonical_self_rows = (
        mapping["candidate_id"].eq(mapping["canonical_candidate_id"])
        .groupby(mapping["signal_equivalence_id"])
        .sum()
    )
    if (canonical_self_rows != 1).any():
        raise ValueError(
            "each equivalence group must contain exactly one canonical candidate row"
        )

    signal_horizon_counts = mapping.groupby("signal_equivalence_id")["horizon"].nunique()
    if (signal_horizon_counts != 1).any():
        raise ValueError("one unique signal maps to multiple horizons")
    actual_unique_by_horizon = (
        mapping.drop_duplicates("signal_equivalence_id")
        .groupby("horizon")["signal_equivalence_id"]
        .nunique()
        .to_dict()
    )
    if actual_unique_by_horizon != unique_counts:
        raise ValueError(
            f"unique-signal horizon counts changed: {actual_unique_by_horizon}"
        )

    timeseries = rank_ic_timeseries.loc[:, sorted(timeseries_required)].copy()
    timeseries["horizon"] = timeseries["horizon"].astype(str)
    timeseries["signal_equivalence_id"] = timeseries["signal_equivalence_id"].astype(str)
    timeseries["decision_ts"] = pd.to_datetime(
        timeseries["decision_ts"], utc=True, errors="raise"
    )
    timeseries["raw_rank_ic"] = pd.to_numeric(
        timeseries["raw_rank_ic"], errors="coerce"
    )
    if set(timeseries["horizon"]) != set(expected_horizons):
        raise ValueError("rank IC timeseries horizon set changed")
    if timeseries.duplicated(["signal_equivalence_id", "decision_ts"]).any():
        raise ValueError("rank IC timeseries contains duplicate signal/decision keys")
    if set(timeseries["status"].astype(str)) != {"ok"}:
        raise ValueError("rank IC timeseries contains a non-ok row")
    if not np.isfinite(timeseries["raw_rank_ic"].to_numpy(dtype=float)).all():
        raise ValueError("rank IC timeseries contains non-finite values")
    if set(timeseries["signal_equivalence_id"]) != set(groups["signal_equivalence_id"]):
        raise ValueError(
            "rank IC timeseries does not cover the frozen unique-signal family"
        )

    start = pd.Timestamp(calendar_start)
    end = pd.Timestamp(calendar_end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    calendar = pd.date_range(start, end, freq="D", tz="UTC")
    if len(calendar) != 494:
        raise ValueError("common calendar must contain exactly 494 complete UTC days")
    timeseries["day"] = timeseries["decision_ts"].dt.floor("D")
    selected = timeseries.loc[timeseries["day"].isin(calendar)].copy()
    if selected.empty:
        raise ValueError("common calendar contains no rank IC observations")

    audit_rows: list[dict[str, object]] = []
    for horizon in expected_horizons:
        horizon_ids = sorted(
            mapping.loc[mapping["horizon"] == horizon, "signal_equivalence_id"].unique()
        )
        horizon_frame = selected.loc[selected["horizon"] == horizon]
        for signal_id in horizon_ids:
            signal_frame = horizon_frame.loc[
                horizon_frame["signal_equivalence_id"] == signal_id
            ]
            counts_by_day = signal_frame.groupby("day").size().reindex(calendar)
            if counts_by_day.isna().any():
                raise ValueError(f"{signal_id}: common calendar has a missing day")
            expected_count = int(daily_decisions[horizon])
            if not (counts_by_day.to_numpy(dtype=int) == expected_count).all():
                raise ValueError(
                    f"{signal_id}: daily decision count differs from {expected_count}"
                )
        audit_rows.append(
            {
                "horizon": horizon,
                "unique_signal_count": len(horizon_ids),
                "calendar_start": calendar.min(),
                "calendar_end": calendar.max(),
                "calendar_day_count": len(calendar),
                "expected_decisions_per_day": int(daily_decisions[horizon]),
                "actual_decisions_per_signal": int(
                    horizon_frame.groupby("signal_equivalence_id").size().min()
                ),
                "excluded_pre_calendar_row_count": int(
                    len(timeseries.loc[
                        (timeseries["horizon"] == horizon)
                        & (timeseries["day"] < calendar.min())
                    ])
                ),
                "coverage_status": "pass",
            }
        )

    signal_ids = sorted(groups["signal_equivalence_id"].astype(str))
    observed_effects = (
        selected.groupby("signal_equivalence_id")["raw_rank_ic"]
        .mean()
        .reindex(signal_ids)
    )
    selected["observed_effect"] = selected["signal_equivalence_id"].map(
        observed_effects
    )
    selected["centered_ic"] = selected["raw_rank_ic"] - selected["observed_effect"]
    daily_centered_sums = (
        selected.groupby(["day", "signal_equivalence_id"])["centered_ic"]
        .sum()
        .unstack("signal_equivalence_id")
        .reindex(index=calendar, columns=signal_ids)
    )
    daily_counts = (
        selected.groupby(["day", "signal_equivalence_id"]).size()
        .unstack("signal_equivalence_id")
        .reindex(index=calendar, columns=signal_ids)
    )
    if daily_centered_sums.isna().any().any() or daily_counts.isna().any().any():
        raise ValueError("daily candidate bootstrap panel is not complete")

    bootstrap_by_block: dict[
        int, research_stats.StepdownMaxTBootstrapArtifacts
    ] = {}
    for block_length in block_lengths:
        bootstrap_by_block[int(block_length)] = (
            research_stats.circular_block_bootstrap_stepdown_max_t(
                daily_centered_sums,
                daily_counts,
                observed_effects,
                block_length=int(block_length),
                n_bootstrap=int(n_bootstrap),
                seed=int(seed),
            )
        )

    primary = bootstrap_by_block[int(primary_block_length)].summary.rename(
        columns={"hypothesis_id": "signal_equivalence_id"}
    )
    unique_results = primary.copy()
    for block_length in block_lengths:
        block_summary = bootstrap_by_block[int(block_length)].summary.set_index(
            "hypothesis_id"
        )
        unique_results[f"block_{int(block_length)}d_stepdown_p"] = (
            unique_results["signal_equivalence_id"].map(
                block_summary["stepdown_max_t_adjusted_p_value"]
            )
        )
    unique_results["candidate_incremental_information_label"] = np.where(
        (unique_results["observed_effect"] > 0.0)
        & (unique_results["stepdown_max_t_adjusted_p_value"] <= float(alpha)),
        "candidate_incremental_information_detected",
        "candidate_incremental_information_not_detected",
    )
    sensitivity_detected = pd.DataFrame(
        {
            f"block_{int(block_length)}d": (
                unique_results["observed_effect"] > 0.0
            )
            & (unique_results[f"block_{int(block_length)}d_stepdown_p"] <= float(alpha))
            for block_length in block_lengths
        }
    )
    unique_results["block_length_label_stable"] = (
        sensitivity_detected.nunique(axis=1) == 1
    )
    unique_results["near_0p05_within_2mcse"] = (
        (
            unique_results["stepdown_max_t_adjusted_p_value"] - float(alpha)
        ).abs()
        <= 2.0 * unique_results["stepdown_adjusted_p_batch_mcse"]
    )

    canonical_metadata = mapping.loc[
        mapping["candidate_id"] == mapping["canonical_candidate_id"]
    ].copy()
    if len(canonical_metadata) != len(groups) or canonical_metadata[
        "signal_equivalence_id"
    ].duplicated().any():
        raise ValueError("canonical candidate metadata is not one row per unique signal")
    alias_metadata = mapping.groupby("signal_equivalence_id", as_index=False).agg(
        alias_track_summary=("track", lambda values: "|".join(sorted(set(values)))),
        alias_weight_scheme_summary=(
            "weight_scheme",
            lambda values: "|".join(sorted(set(values))),
        ),
        alias_component_summary=(
            "component_features",
            lambda values: "|".join(sorted(set(values))),
        ),
    )
    unique_results = (
        unique_results.merge(
            groups,
            on="signal_equivalence_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            canonical_metadata[
                [
                    "signal_equivalence_id",
                    "horizon",
                    "track",
                    "weight_scheme",
                    "component_features",
                ]
            ],
            on="signal_equivalence_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            alias_metadata,
            on="signal_equivalence_id",
            how="left",
            validate="one_to_one",
        )
    )
    candidate_results = mapping.merge(
        unique_results[
            [
                "signal_equivalence_id",
                "observed_effect",
                "bootstrap_se",
                "observed_t",
                "raw_one_sided_p_value",
                "raw_p_mcse",
                "stepdown_max_t_adjusted_p_value",
                "stepdown_adjusted_p_batch_mcse",
                "observed_t_descending_rank",
                "block_7d_stepdown_p",
                "block_14d_stepdown_p",
                "block_28d_stepdown_p",
                "block_length_label_stable",
                "near_0p05_within_2mcse",
                "candidate_incremental_information_label",
            ]
        ],
        on="signal_equivalence_id",
        how="left",
        validate="many_to_one",
    )
    if len(candidate_results) != int(expected_candidate_count):
        raise ValueError(
            "candidate dependency results do not map back to "
            f"{int(expected_candidate_count)} rows"
        )
    alias_check_columns = [
        "observed_effect",
        "bootstrap_se",
        "observed_t",
        "raw_one_sided_p_value",
        "stepdown_max_t_adjusted_p_value",
        "candidate_incremental_information_label",
    ]
    if (
        candidate_results.groupby("signal_equivalence_id")[alias_check_columns]
        .nunique(dropna=False)
        .to_numpy()
        != 1
    ).any():
        raise RuntimeError("candidate aliases received inconsistent dependency results")

    return CandidateResidualDependencyArtifacts(
        daily_centered_sums=daily_centered_sums,
        daily_counts=daily_counts,
        daily_coverage_audit=pd.DataFrame(audit_rows),
        unique_results=unique_results,
        candidate_results=candidate_results,
        bootstrap_by_block_length=bootstrap_by_block,
    )


def summarize_residual_construction_audit(
    prepared_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize the already-validated residual construction by candidate."""
    required = {
        "candidate_id", "fold_idx", "decision_ts", "symbol",
        "target_signal", "replica_signal", "residual_signal",
    }
    missing = sorted(required.difference(prepared_predictions.columns))
    if missing:
        raise ValueError("prepared predictions missing residual audit columns: " + ", ".join(missing))
    rows: list[dict[str, object]] = []
    for candidate_id, frame in prepared_predictions.groupby("candidate_id", sort=True):
        difference = (
            frame["target_signal"].astype(float)
            - frame["replica_signal"].astype(float)
            - frame["residual_signal"].astype(float)
        ).abs()
        rows.append(
            {
                "candidate_id": str(candidate_id),
                "row_count": len(frame),
                "outer_fold_count": int(frame["fold_idx"].nunique()),
                "decision_count": int(frame["decision_ts"].nunique()),
                "symbol_count_min": int(frame.groupby("decision_ts")["symbol"].nunique().min()),
                "symbol_count_max": int(frame.groupby("decision_ts")["symbol"].nunique().max()),
                "max_abs_residual_identity_error": float(difference.max()),
                "construction_status": "pass",
            }
        )
    return pd.DataFrame(rows)


def audit_precomputed_executable_return_lineage(
    prepared_predictions: pd.DataFrame,
    execution_opens_by_symbol: Mapping[str, pd.Series],
    *,
    horizon: str | pd.Timedelta,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Verify precomputed returns against t+1m and t+H+1m execution opens."""
    required = {"decision_ts", "symbol", "strategy_forward_return"}
    missing = sorted(required.difference(prepared_predictions.columns))
    if missing:
        raise ValueError("prepared predictions missing execution-lineage columns: " + ", ".join(missing))
    delta = pd.Timedelta(horizon)
    if delta <= pd.Timedelta(0):
        raise ValueError("horizon must be positive")
    working = prepared_predictions[["decision_ts", "symbol", "strategy_forward_return"]].copy()
    working["decision_ts"] = pd.to_datetime(working["decision_ts"], utc=True, errors="raise")
    working["symbol"] = working["symbol"].astype(str)
    working["strategy_forward_return"] = pd.to_numeric(
        working["strategy_forward_return"], errors="coerce"
    )
    if not np.isfinite(working["strategy_forward_return"].to_numpy(dtype=float)).all():
        raise ValueError("strategy_forward_return contains non-finite values")
    grouped = working.groupby(["decision_ts", "symbol"], sort=True)["strategy_forward_return"]
    spread = grouped.max() - grouped.min()
    if (spread > float(tolerance)).any():
        raise ValueError("candidate aliases disagree on executable future return")
    ledger = grouped.first().reset_index()
    ledger["execution_ts"] = ledger["decision_ts"] + pd.Timedelta(minutes=1)
    ledger["next_execution_ts"] = ledger["execution_ts"] + delta
    entry_parts: list[pd.Series] = []
    exit_parts: list[pd.Series] = []
    for symbol, frame in ledger.groupby("symbol", sort=True):
        if symbol not in execution_opens_by_symbol:
            raise ValueError(f"missing execution opens for {symbol}")
        opens = pd.Series(execution_opens_by_symbol[symbol]).copy()
        opens.index = pd.to_datetime(opens.index, utc=True, errors="raise")
        if opens.index.has_duplicates:
            raise ValueError(f"execution opens contain duplicate timestamps for {symbol}")
        entry = opens.reindex(pd.DatetimeIndex(frame["execution_ts"]))
        exit_prices = opens.reindex(pd.DatetimeIndex(frame["next_execution_ts"]))
        entry.index = frame.index
        exit_prices.index = frame.index
        entry_parts.append(entry)
        exit_parts.append(exit_prices)
    ledger["entry_price"] = pd.concat(entry_parts).sort_index()
    ledger["exit_price"] = pd.concat(exit_parts).sort_index()
    if ledger[["entry_price", "exit_price"]].isna().any().any():
        raise ValueError("execution-open lineage has missing entry or exit prices")
    expected = ledger["exit_price"] / ledger["entry_price"] - 1.0
    error = (expected - ledger["strategy_forward_return"]).abs()
    if (error > float(tolerance)).any():
        raise ValueError("strategy_forward_return differs from executable-open return")
    return pd.DataFrame(
        [
            {
                "horizon": str(horizon),
                "unique_decision_symbol_count": len(ledger),
                "decision_count": int(ledger["decision_ts"].nunique()),
                "symbol_count": int(ledger["symbol"].nunique()),
                "decision_start": ledger["decision_ts"].min(),
                "decision_end": ledger["decision_ts"].max(),
                "execution_start": ledger["execution_ts"].min(),
                "execution_end": ledger["next_execution_ts"].max(),
                "max_abs_return_error": float(error.max()),
                "lineage_status": "pass",
            }
        ]
    )


def build_cross_sectional_outcome_target(
    canonical_frame: pd.DataFrame,
    *,
    source_column: str = "strategy_forward_return",
    target_column: str = "outcome_target",
    min_cross_section: int = 10,
) -> pd.DataFrame:
    """Demean executable returns within each fold/split/decision cross-section."""
    required_levels = {"fold_idx", "split", "decision_ts", "symbol"}
    if not isinstance(canonical_frame.index, pd.MultiIndex):
        raise TypeError("canonical outcome frame must use a MultiIndex")
    missing_levels = sorted(required_levels.difference(canonical_frame.index.names))
    if missing_levels:
        raise ValueError("canonical outcome frame missing index levels: " + ", ".join(missing_levels))
    if source_column not in canonical_frame.columns:
        raise ValueError(f"canonical outcome frame missing {source_column}")
    frame = canonical_frame.copy()
    source = pd.to_numeric(frame[source_column], errors="raise")
    if source.isna().any() or not np.isfinite(source.to_numpy(dtype=float)).all():
        raise ValueError("executable outcome source must be finite")
    group_levels = ["fold_idx", "split", "decision_ts"]
    counts = source.groupby(level=group_levels).size()
    if counts.empty or (counts < int(min_cross_section)).any():
        raise ValueError("executable outcome cross-section is below the minimum size")
    frame[target_column] = source - source.groupby(level=group_levels).transform("mean")
    sums = frame[target_column].groupby(level=group_levels).sum()
    if not np.allclose(sums.to_numpy(dtype=float), 0.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("cross-sectional executable outcome target is not demeaned")
    return frame


def evaluate_cross_fitted_double_residuals(
    signal_predictions: pd.DataFrame,
    outcome_predictions: pd.DataFrame,
    *,
    hypothesis_id: str,
    horizon: str,
    min_cross_section: int = 10,
) -> DoubleResidualArtifacts:
    """Evaluate one train-selected signal/outcome residual pair on shared OOS keys."""
    keys = ["fold_idx", "decision_ts", "symbol"]
    required = {
        *keys,
        "target_signal",
        "replica_signal",
        "residual_signal",
        "source_model_class",
    }
    prepared: list[pd.DataFrame] = []
    for name, raw in (
        ("signal predictions", signal_predictions),
        ("outcome predictions", outcome_predictions),
    ):
        missing = sorted(required.difference(raw.columns))
        if missing:
            raise ValueError(f"{name} missing columns: " + ", ".join(missing))
        frame = raw[list(required)].copy()
        frame["fold_idx"] = pd.to_numeric(frame["fold_idx"], errors="raise").astype(int)
        frame["decision_ts"] = pd.to_datetime(
            frame["decision_ts"], utc=True, errors="raise"
        )
        frame["symbol"] = frame["symbol"].astype(str)
        numeric = ["target_signal", "replica_signal", "residual_signal"]
        frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} contains non-finite values")
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} contains duplicate OOS keys")
        if not np.allclose(
            frame["target_signal"].to_numpy(dtype=float)
            - frame["replica_signal"].to_numpy(dtype=float),
            frame["residual_signal"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"{name} residual differs from target - prediction")
        prepared.append(frame.sort_values(keys, kind="mergesort").reset_index(drop=True))
    signal, outcome = prepared
    if not signal[keys].equals(outcome[keys]):
        raise ValueError("signal and outcome residual streams do not share exact OOS keys")

    observations = signal[keys].copy()
    observations["hypothesis_id"] = str(hypothesis_id)
    observations["horizon"] = str(horizon)
    observations["signal_target"] = signal["target_signal"].to_numpy(dtype=float)
    observations["signal_prediction"] = signal["replica_signal"].to_numpy(dtype=float)
    observations["signal_residual"] = signal["residual_signal"].to_numpy(dtype=float)
    observations["outcome_target"] = outcome["target_signal"].to_numpy(dtype=float)
    observations["outcome_prediction"] = outcome["replica_signal"].to_numpy(dtype=float)
    observations["outcome_residual"] = outcome["residual_signal"].to_numpy(dtype=float)
    observations["signal_model_class"] = signal["source_model_class"].astype(str)
    observations["outcome_model_class"] = outcome["source_model_class"].astype(str)
    observations["residual_product"] = (
        observations["signal_residual"] * observations["outcome_residual"]
    )

    moment_rows: list[dict[str, object]] = []
    for decision_ts, group in observations.groupby("decision_ts", sort=True):
        if group["fold_idx"].nunique() != 1:
            raise ValueError("one decision timestamp spans multiple outer folds")
        count = len(group)
        if count < int(min_cross_section):
            raise ValueError(
                f"{hypothesis_id}: decision {decision_ts} has {count} shared symbols"
            )
        u = group["signal_residual"].to_numpy(dtype=float)
        v = group["outcome_residual"].to_numpy(dtype=float)
        u_ss = float(np.dot(u, u))
        u_std = float(np.std(u, ddof=1)) if count > 1 else float("nan")
        v_std = float(np.std(v, ddof=1)) if count > 1 else float("nan")
        correlation = (
            float(np.corrcoef(u, v)[0, 1])
            if u_std > 0.0 and v_std > 0.0
            else float("nan")
        )
        moment_rows.append(
            {
                "hypothesis_id": str(hypothesis_id),
                "horizon": str(horizon),
                "fold_idx": int(group["fold_idx"].iloc[0]),
                "decision_ts": pd.Timestamp(decision_ts),
                "symbol_count": count,
                "double_residual_moment": float(np.mean(u * v)),
                "residual_correlation": correlation,
                "residual_slope": float(np.dot(u, v) / u_ss) if u_ss > 0.0 else float("nan"),
            }
        )
    moments = pd.DataFrame(moment_rows).sort_values("decision_ts").reset_index(drop=True)
    u_all = observations["signal_residual"].to_numpy(dtype=float)
    v_all = observations["outcome_residual"].to_numpy(dtype=float)
    u_ss_all = float(np.dot(u_all, u_all))
    summary = pd.DataFrame(
        [
            {
                "hypothesis_id": str(hypothesis_id),
                "horizon": str(horizon),
                "fold_count": int(observations["fold_idx"].nunique()),
                "decision_count": len(moments),
                "observation_count": len(observations),
                "mean_double_residual_moment": float(
                    moments["double_residual_moment"].mean()
                ),
                "pooled_residual_correlation": _safe_corr(
                    observations["signal_residual"],
                    observations["outcome_residual"],
                    "pearson",
                ),
                "pooled_residual_slope": (
                    float(np.dot(u_all, v_all) / u_ss_all)
                    if u_ss_all > 0.0
                    else float("nan")
                ),
            }
        ]
    )
    return DoubleResidualArtifacts(
        observations=observations.sort_values(keys, kind="mergesort").reset_index(drop=True),
        decision_moments=moments,
        summary=summary,
    )


def build_double_residual_daily_family(
    decision_moments: pd.DataFrame,
    *,
    expected_decisions_per_day: Mapping[str, int],
) -> DoubleResidualDailyFamilyArtifacts:
    """Build the complete synchronized UTC-day family required by L5.5."""
    required = {
        "hypothesis_id",
        "horizon",
        "fold_idx",
        "decision_ts",
        "double_residual_moment",
    }
    missing = sorted(required.difference(decision_moments.columns))
    if missing:
        raise ValueError("double residual moments missing columns: " + ", ".join(missing))
    frame = decision_moments.copy()
    frame["hypothesis_id"] = frame["hypothesis_id"].astype(str)
    frame["horizon"] = frame["horizon"].astype(str)
    frame["fold_idx"] = pd.to_numeric(frame["fold_idx"], errors="raise").astype(int)
    frame["decision_ts"] = pd.to_datetime(frame["decision_ts"], utc=True, errors="raise")
    frame["double_residual_moment"] = pd.to_numeric(
        frame["double_residual_moment"], errors="raise"
    )
    if frame.empty or not np.isfinite(
        frame["double_residual_moment"].to_numpy(dtype=float)
    ).all():
        raise ValueError("double residual moments must be non-empty and finite")
    if frame.duplicated(["hypothesis_id", "decision_ts"]).any():
        raise ValueError("double residual moments contain duplicate hypothesis/time rows")
    if not set(frame["horizon"]).issubset(set(expected_decisions_per_day)):
        raise ValueError("expected decision counts do not cover every horizon")
    frame["utc_day"] = frame["decision_ts"].dt.floor("D")

    audit_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    for (hypothesis_id, utc_day), group in frame.groupby(
        ["hypothesis_id", "utc_day"], sort=True
    ):
        horizons = group["horizon"].unique()
        if len(horizons) != 1:
            raise ValueError("one hypothesis/day contains multiple horizons")
        expected = int(expected_decisions_per_day[str(horizons[0])])
        fold_ids = sorted(group["fold_idx"].unique().tolist())
        fold_count = len(fold_ids)
        adjacent_folds = fold_count <= 1 or (
            fold_count == 2 and fold_ids[1] - fold_ids[0] == 1
        )
        if not adjacent_folds:
            raise ValueError(
                "one hypothesis/day contains non-adjacent outer folds: "
                f"hypothesis_id={hypothesis_id}, utc_day={utc_day}, fold_ids={fold_ids}"
            )
        complete = len(group) == expected
        audit_rows.append(
            {
                "hypothesis_id": str(hypothesis_id),
                "horizon": str(horizons[0]),
                "utc_day": pd.Timestamp(utc_day),
                "expected_decision_count": expected,
                "actual_decision_count": len(group),
                "fold_count": fold_count,
                "fold_ids": ",".join(str(value) for value in fold_ids),
                "status": "complete" if complete else "excluded_incomplete_day",
            }
        )
        if complete:
            daily_rows.append(
                {
                    "hypothesis_id": str(hypothesis_id),
                    "horizon": str(horizons[0]),
                    "utc_day": pd.Timestamp(utc_day),
                    "daily_effect": float(group["double_residual_moment"].mean()),
                }
            )
    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        raise ValueError("no complete UTC days remain for double residual inference")
    bounds = daily.groupby("hypothesis_id")["utc_day"].agg(["min", "max"])
    common_start = pd.Timestamp(bounds["min"].max())
    common_end = pd.Timestamp(bounds["max"].min())
    if common_start > common_end:
        raise ValueError("double residual hypotheses have no common UTC-day interval")
    expected_days = pd.date_range(common_start, common_end, freq="D", tz="UTC")
    hypotheses = sorted(daily["hypothesis_id"].unique())
    common = daily.loc[daily["utc_day"].between(common_start, common_end)].copy()
    observed_pairs = set(zip(common["utc_day"], common["hypothesis_id"], strict=True))
    missing_pairs = [
        (day, hypothesis_id)
        for day in expected_days
        for hypothesis_id in hypotheses
        if (day, hypothesis_id) not in observed_pairs
    ]
    if missing_pairs:
        first = missing_pairs[0]
        raise ValueError(
            "double residual common calendar has an internal missing day: "
            f"day={first[0]}, hypothesis_id={first[1]}"
        )
    daily_effects = common.pivot(
        index="utc_day", columns="hypothesis_id", values="daily_effect"
    ).reindex(index=expected_days, columns=hypotheses)
    if daily_effects.isna().any().any():
        raise ValueError("double residual common daily effect matrix is incomplete")
    observed_effects = daily_effects.mean(axis=0)
    centered = daily_effects.subtract(observed_effects, axis="columns")
    if not np.allclose(centered.sum(axis=0), 0.0, rtol=0.0, atol=1e-10):
        raise RuntimeError("double residual daily effects failed exact centering")
    counts = pd.DataFrame(1, index=daily_effects.index, columns=daily_effects.columns)
    return DoubleResidualDailyFamilyArtifacts(
        daily_effects=daily_effects,
        daily_centered_sums=centered,
        daily_counts=counts,
        coverage_audit=pd.DataFrame(audit_rows).sort_values(
            ["hypothesis_id", "utc_day"], kind="mergesort"
        ).reset_index(drop=True),
        observed_effects=observed_effects,
    )


def _evaluate_double_residual_time_randomization(
    observations: pd.DataFrame,
    daily_effects: pd.DataFrame,
    *,
    guard_days: Sequence[int] = (3, 7, 14),
    n_randomizations: int = 2_000,
    seed: int = 20_260_804,
    alternative: str = "two-sided",
    required_seed: int | None,
    randomize_side: str,
) -> DoubleResidualTimeRandomizationArtifacts:
    """Shared implementation for registered outcome- or signal-side shifts."""
    required = {
        "hypothesis_id",
        "horizon",
        "fold_idx",
        "decision_ts",
        "symbol",
        "signal_residual",
        "outcome_residual",
    }
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise ValueError("double residual observations missing columns: " + ", ".join(missing))
    guards = tuple(int(value) for value in guard_days)
    if guards != (3, 7, 14):
        raise ValueError("guard_days must be exactly (3, 7, 14)")
    if int(n_randomizations) != 2_000:
        raise ValueError("n_randomizations must be exactly 2000")
    if required_seed is not None and int(seed) != int(required_seed):
        raise ValueError(f"seed must be exactly {int(required_seed)}")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    if randomize_side not in {"outcome", "signal"}:
        raise ValueError("randomize_side must be outcome or signal")
    if alternative not in {"two-sided", "greater"}:
        raise ValueError("alternative must be two-sided or greater")
    if not isinstance(daily_effects, pd.DataFrame) or daily_effects.empty:
        raise ValueError("daily_effects must be a non-empty DataFrame")
    if not isinstance(daily_effects.index, pd.DatetimeIndex):
        raise ValueError("daily_effects must use a DatetimeIndex")
    frame = observations.copy()
    frame["hypothesis_id"] = frame["hypothesis_id"].astype(str)
    frame["horizon"] = frame["horizon"].astype(str)
    frame["fold_idx"] = pd.to_numeric(frame["fold_idx"], errors="raise").astype(int)
    frame["decision_ts"] = pd.to_datetime(frame["decision_ts"], utc=True, errors="raise")
    frame["symbol"] = frame["symbol"].astype(str)
    for column in ("signal_residual", "outcome_residual"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(
        frame[["signal_residual", "outcome_residual"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("double residual observations contain non-finite residuals")
    if frame.duplicated(["hypothesis_id", "decision_ts", "symbol"]).any():
        raise ValueError("double residual observations contain duplicate keys")
    hypotheses = list(daily_effects.columns.astype(str))
    if set(frame["hypothesis_id"]) != set(hypotheses):
        raise ValueError("observations and daily effects disagree on hypotheses")
    common_days = pd.DatetimeIndex(pd.to_datetime(daily_effects.index, utc=True))
    frame["utc_day"] = frame["decision_ts"].dt.floor("D")
    frame = frame.loc[frame["utc_day"].isin(common_days)].copy()
    if frame.empty:
        raise ValueError("no observations remain on the synchronized daily calendar")
    horizon_counts = frame.groupby("hypothesis_id")["horizon"].nunique()
    if (horizon_counts != 1).any():
        raise ValueError("one hypothesis contains multiple horizons")
    horizon_by_hypothesis = frame.groupby("hypothesis_id")["horizon"].first().to_dict()

    precomputed: dict[tuple[str, str, int, int], tuple[float, int]] = {}
    stratum_lengths: dict[tuple[str, int], int] = {}
    stratum_phase_counts: dict[tuple[str, int], int] = {}
    hypothesis_strata: dict[str, list[tuple[str, int]]] = {
        hypothesis_id: [] for hypothesis_id in hypotheses
    }
    observed_sums: dict[str, float] = {hypothesis_id: 0.0 for hypothesis_id in hypotheses}
    observed_counts: dict[str, int] = {hypothesis_id: 0 for hypothesis_id in hypotheses}
    for horizon in sorted(frame["horizon"].unique()):
        horizon_hypotheses = [
            hypothesis_id
            for hypothesis_id in hypotheses
            if horizon_by_hypothesis[hypothesis_id] == horizon
        ]
        horizon_frame = frame.loc[frame["horizon"] == horizon]
        reference_keys: pd.DataFrame | None = None
        for hypothesis_id in horizon_hypotheses:
            keys = (
                horizon_frame.loc[horizon_frame["hypothesis_id"] == hypothesis_id,
                                  ["fold_idx", "decision_ts"]]
                .drop_duplicates()
                .sort_values(["fold_idx", "decision_ts"], kind="mergesort")
                .reset_index(drop=True)
            )
            if reference_keys is None:
                reference_keys = keys
            elif not keys.equals(reference_keys):
                raise ValueError(
                    "time randomization hypotheses disagree on horizon decision/fold keys"
                )
        if reference_keys is None:
            raise ValueError("time randomization horizon has no hypotheses")
        for fold_idx, fold_keys in reference_keys.groupby("fold_idx", sort=True):
            stratum = (str(horizon), int(fold_idx))
            fold_times = pd.DatetimeIndex(fold_keys["decision_ts"].sort_values())
            phases = sorted((fold_times - fold_times.floor("D")).unique())
            phase_times = {
                phase: fold_times[(fold_times - fold_times.floor("D")) == phase]
                for phase in phases
            }
            for phase, times in phase_times.items():
                if len(times) > 1 and not (times[1:] - times[:-1] == pd.Timedelta(days=1)).all():
                    raise ValueError(
                        "time randomization has a non-daily gap within a fold/clock phase: "
                        f"horizon={horizon}, fold={fold_idx}, phase={phase}"
                    )
            min_phase_length = min(len(times) for times in phase_times.values())
            stratum_lengths[stratum] = min_phase_length
            stratum_phase_counts[stratum] = len(phases)
            if min_phase_length <= 2 * max(guards):
                raise ValueError(
                    f"horizon {horizon} fold {fold_idx} has too few phase-aligned days "
                    "for the 14-day guard"
                )
            for hypothesis_id in horizon_hypotheses:
                hypothesis_strata[hypothesis_id].append(stratum)
                group = frame.loc[
                    (frame["hypothesis_id"] == hypothesis_id)
                    & (frame["fold_idx"] == int(fold_idx))
                    & (frame["horizon"] == horizon)
                ].copy()
                symbols = sorted(group["symbol"].unique())
                shifted_parts: dict[int, list[np.ndarray]] = {
                    shift: [] for shift in range(1, min_phase_length)
                }
                for phase in phases:
                    times = phase_times[phase]
                    index = pd.MultiIndex.from_product(
                        [times, symbols], names=["decision_ts", "symbol"]
                    )
                    aligned = group.set_index(["decision_ts", "symbol"]).reindex(index)
                    shape = (len(times), len(symbols))
                    u = aligned["signal_residual"].to_numpy(dtype=float).reshape(shape)
                    v = aligned["outcome_residual"].to_numpy(dtype=float).reshape(shape)
                    observed_valid = np.isfinite(u) & np.isfinite(v)
                    observed_pair_counts = observed_valid.sum(axis=1)
                    if (observed_pair_counts <= 0).any():
                        raise ValueError("time randomization has an empty observed decision")
                    observed_decision_means = np.sum(
                        np.where(observed_valid, u * v, 0.0), axis=1
                    ) / observed_pair_counts
                    observed_sums[hypothesis_id] += float(observed_decision_means.sum())
                    observed_counts[hypothesis_id] += int(observed_decision_means.size)
                    for shift in range(1, min_phase_length):
                        shifted_u = np.roll(u, -shift, axis=0) if randomize_side == "signal" else u
                        shifted_v = np.roll(v, -shift, axis=0) if randomize_side == "outcome" else v
                        valid = np.isfinite(shifted_u) & np.isfinite(shifted_v)
                        pair_counts = valid.sum(axis=1)
                        if (pair_counts <= 0).any():
                            raise ValueError("time randomization shift has an empty decision")
                        shifted_parts[shift].append(
                            np.sum(np.where(valid, shifted_u * shifted_v, 0.0), axis=1)
                            / pair_counts
                        )
                for shift, parts in shifted_parts.items():
                    values = np.concatenate(parts)
                    precomputed[(hypothesis_id, str(horizon), int(fold_idx), shift)] = (
                        float(values.sum()),
                        int(values.size),
                    )

    for hypothesis_id in hypotheses:
        observed = observed_sums[hypothesis_id] / float(observed_counts[hypothesis_id])
        expected = float(daily_effects[hypothesis_id].mean())
        if not np.isclose(observed, expected, rtol=0.0, atol=1e-12):
            raise ValueError(
                "time randomization observations disagree with synchronized daily effect: "
                f"hypothesis_id={hypothesis_id}"
            )

    strata = sorted(stratum_lengths)
    seed_children = np.random.SeedSequence(int(seed)).spawn(len(guards))
    schedule_rows: list[dict[str, object]] = []
    null_by_guard: dict[int, pd.DataFrame] = {}
    for guard, child in zip(guards, seed_children, strict=True):
        rng = np.random.Generator(np.random.PCG64DXSM(child))
        shifts_by_stratum: dict[tuple[str, int], np.ndarray] = {}
        for horizon, fold_idx in strata:
            day_count = stratum_lengths[(horizon, fold_idx)]
            legal = np.asarray(
                [
                    shift
                    for shift in range(1, day_count)
                    if min(shift, day_count - shift) >= int(guard)
                ],
                dtype=int,
            )
            if legal.size == 0:
                raise ValueError(
                    f"guard {guard} leaves no legal shift in horizon {horizon} fold {fold_idx}"
                )
            chosen = rng.choice(legal, size=int(n_randomizations), replace=True)
            shifts_by_stratum[(horizon, fold_idx)] = chosen
            schedule_rows.extend(
                {
                    "guard_days": int(guard),
                    "randomization_idx": index,
                    "horizon": horizon,
                    "fold_idx": fold_idx,
                    "fold_day_count": day_count,
                    "phase_count": stratum_phase_counts[(horizon, fold_idx)],
                    "shift_days": int(shift),
                }
                for index, shift in enumerate(chosen)
            )
        values = np.empty((int(n_randomizations), len(hypotheses)), dtype=float)
        for column_idx, hypothesis_id in enumerate(hypotheses):
            totals = np.zeros(int(n_randomizations), dtype=float)
            denominators = np.zeros(int(n_randomizations), dtype=int)
            for horizon, fold_idx in hypothesis_strata[hypothesis_id]:
                chosen = shifts_by_stratum[(horizon, fold_idx)]
                sum_lookup = {
                    shift: precomputed[(hypothesis_id, horizon, fold_idx, shift)][0]
                    for shift in np.unique(chosen)
                }
                count_lookup = {
                    shift: precomputed[(hypothesis_id, horizon, fold_idx, shift)][1]
                    for shift in np.unique(chosen)
                }
                totals += np.asarray(
                    [sum_lookup[int(shift)] for shift in chosen], dtype=float
                )
                denominators += np.asarray(
                    [count_lookup[int(shift)] for shift in chosen], dtype=int
                )
            if (denominators <= 0).any():
                raise ValueError("time randomization produced an empty shifted support")
            values[:, column_idx] = totals / denominators.astype(float)
        null_by_guard[int(guard)] = pd.DataFrame(values, columns=hypotheses)

    null_effects = pd.concat(
        [
            frame.assign(guard_days=guard, randomization_idx=frame.index)
            for guard, frame in null_by_guard.items()
        ],
        ignore_index=True,
    ).set_index(["guard_days", "randomization_idx"])
    summary_rows: list[dict[str, object]] = []
    for guard, null in null_by_guard.items():
        for hypothesis_id in hypotheses:
            observed = observed_sums[hypothesis_id] / float(observed_counts[hypothesis_id])
            values = null[hypothesis_id].to_numpy(dtype=float)
            exceedance = (
                np.abs(values) >= abs(observed)
                if alternative == "two-sided"
                else values >= observed
            )
            p_value = float((1 + np.count_nonzero(exceedance)) / (len(values) + 1))
            summary_rows.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "guard_days": guard,
                    "observed_effect": observed,
                    "null_mean": float(values.mean()),
                    "null_std": float(values.std(ddof=1)),
                    "raw_one_sided_randomization_p_value": (
                        p_value if alternative == "greater" else np.nan
                    ),
                    "raw_two_sided_randomization_p_value": (
                        p_value if alternative == "two-sided" else np.nan
                    ),
                    "n_randomizations": len(values),
                    "seed": int(seed),
                    "alternative": alternative,
                }
            )
    schedule = pd.DataFrame(schedule_rows).sort_values(
        ["guard_days", "randomization_idx", "horizon", "fold_idx"], kind="mergesort"
    ).reset_index(drop=True)
    return DoubleResidualTimeRandomizationArtifacts(
        schedule=schedule,
        null_effects=null_effects,
        summary=pd.DataFrame(summary_rows).sort_values(
            ["guard_days", "hypothesis_id"], kind="mergesort"
        ).reset_index(drop=True),
    )


def evaluate_double_residual_time_randomization(
    observations: pd.DataFrame,
    daily_effects: pd.DataFrame,
    *,
    guard_days: Sequence[int] = (3, 7, 14),
    n_randomizations: int = 2_000,
    seed: int = 20_260_804,
    alternative: str = "two-sided",
) -> DoubleResidualTimeRandomizationArtifacts:
    """Production L5.5 outcome-side whole-day time randomization."""
    return _evaluate_double_residual_time_randomization(
        observations,
        daily_effects,
        guard_days=guard_days,
        n_randomizations=n_randomizations,
        seed=seed,
        alternative=alternative,
        required_seed=20_260_804,
        randomize_side="outcome",
    )


def simulation_signal_time_randomization(
    observations: pd.DataFrame,
    daily_effects: pd.DataFrame,
    *,
    guard_days: Sequence[int] = (3, 7, 14),
    n_randomizations: int = 2_000,
    seed: int,
    alternative: str = "greater",
) -> DoubleResidualTimeRandomizationArtifacts:
    """Simulation-only signal-side whole-day time randomization.

    The seed is supplied by the frozen simulation manifest. This entry must
    not be used for empirical L5.5 evidence.
    """
    return _evaluate_double_residual_time_randomization(
        observations,
        daily_effects,
        guard_days=guard_days,
        n_randomizations=n_randomizations,
        seed=seed,
        alternative=alternative,
        required_seed=None,
        randomize_side="signal",
    )


def infer_double_residual_family(
    decision_moments: pd.DataFrame,
    candidate_mapping: pd.DataFrame,
    *,
    expected_decisions_per_day: Mapping[str, int],
    block_lengths: Sequence[int] = (7, 14, 28),
    main_block_length: int = 14,
    n_bootstrap: int = 10_000,
    seed: int = 20_260_803,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> DoubleResidualFamilyInferenceArtifacts:
    """Run the complete registered L5.5 family inference."""
    lengths = tuple(int(value) for value in block_lengths)
    if lengths != (7, 14, 28) or int(main_block_length) != 14:
        raise ValueError("L5.5 block lengths must be (7, 14, 28) with 14 as main")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be between zero and one")
    if alternative not in {"two-sided", "greater"}:
        raise ValueError("alternative must be two-sided or greater")
    required_mapping = {"candidate_id", "signal_equivalence_id"}
    missing_mapping = sorted(required_mapping.difference(candidate_mapping.columns))
    if missing_mapping:
        raise ValueError("candidate mapping missing columns: " + ", ".join(missing_mapping))
    mapping = candidate_mapping.copy()
    mapping["candidate_id"] = mapping["candidate_id"].astype(str)
    mapping["signal_equivalence_id"] = mapping["signal_equivalence_id"].astype(str)
    if mapping["candidate_id"].duplicated().any():
        raise ValueError("candidate mapping contains duplicate candidate ids")

    family = build_double_residual_daily_family(
        decision_moments,
        expected_decisions_per_day=expected_decisions_per_day,
    )
    hypothesis_ids = list(family.daily_effects.columns.astype(str))
    if set(mapping["signal_equivalence_id"]) != set(hypothesis_ids):
        raise ValueError("candidate mapping and synchronized family disagree on hypotheses")

    inference: pd.DataFrame | None = None
    start_parts: list[pd.DataFrame] = []
    for block_length in lengths:
        result = research_stats.circular_block_bootstrap_stepdown_max_t(
            family.daily_centered_sums,
            family.daily_counts,
            family.observed_effects,
            block_length=block_length,
            n_bootstrap=int(n_bootstrap),
            seed=int(seed),
            alternative=alternative,
        )
        p_column = f"block_{block_length}d_stepdown_p"
        current = result.summary[["hypothesis_id", "stepdown_max_t_adjusted_p_value"]].rename(
            columns={"stepdown_max_t_adjusted_p_value": p_column}
        )
        inference = current if inference is None else inference.merge(
            current, on="hypothesis_id", validate="one_to_one"
        )
        start_parts.append(result.block_starts.assign(block_length_days=block_length))
    if inference is None:
        raise RuntimeError("L5.5 inference produced no block-length result")

    descriptive = decision_moments.groupby(
        ["hypothesis_id", "horizon"], as_index=False
    ).agg(
        all_decisions_mean_effect=("double_residual_moment", "mean"),
        decision_count=("decision_ts", "nunique"),
        fold_count=("fold_idx", "nunique"),
        mean_residual_correlation=("residual_correlation", "mean"),
        mean_residual_slope=("residual_slope", "mean"),
    )
    hac_rows: list[dict[str, object]] = []
    for hypothesis_id, group in decision_moments.groupby("hypothesis_id", sort=True):
        values = group.sort_values("decision_ts")["double_residual_moment"]
        hac_t = research_stats.hac_t_stat(values, overlap_lags=0)
        hac_rows.append(
            {
                "hypothesis_id": str(hypothesis_id),
                "hac_t_stat": hac_t,
                "hac_raw_one_sided_p": (
                    research_stats.normal_one_sided_p_value(hac_t)
                    if alternative == "greater"
                    else np.nan
                ),
                "hac_raw_two_sided_p": (
                    research_stats.normal_two_sided_p_value(hac_t)
                    if alternative == "two-sided"
                    else np.nan
                ),
            }
        )
    unique_results = descriptive.merge(
        pd.DataFrame(hac_rows), on="hypothesis_id", validate="one_to_one"
    ).merge(inference, on="hypothesis_id", validate="one_to_one")
    unique_results["formal_common_calendar_effect"] = unique_results["hypothesis_id"].map(
        family.observed_effects
    )
    unique_results["main_stepdown_p"] = unique_results[
        f"block_{int(main_block_length)}d_stepdown_p"
    ]
    unique_results["alternative"] = alternative
    if alternative == "greater":
        detected_label = "positive_double_residual_incremental_information_detected"
        not_detected_label = "positive_double_residual_incremental_information_not_detected"
    else:
        detected_label = "double_residual_incremental_association_detected"
        not_detected_label = "double_residual_incremental_association_not_detected"
    unique_results["l5_5_label"] = np.where(
        unique_results["main_stepdown_p"] <= float(alpha),
        detected_label,
        not_detected_label,
    )
    if len(unique_results) != len(hypothesis_ids):
        raise ValueError("L5.5 result does not contain every synchronized hypothesis")
    candidate_results = mapping.merge(
        unique_results,
        left_on="signal_equivalence_id",
        right_on="hypothesis_id",
        validate="many_to_one",
    )
    if len(candidate_results) != len(mapping):
        raise ValueError("L5.5 candidate mapping lost rows")
    return DoubleResidualFamilyInferenceArtifacts(
        daily_effects=family.daily_effects,
        daily_centered_sums=family.daily_centered_sums,
        daily_counts=family.daily_counts,
        coverage_audit=family.coverage_audit,
        unique_results=unique_results.sort_values("hypothesis_id", kind="mergesort").reset_index(drop=True),
        candidate_results=candidate_results.sort_values("candidate_id", kind="mergesort").reset_index(drop=True),
        bootstrap_starts=pd.concat(start_parts, ignore_index=True),
    )


def apply_residual_information_holm(
    unique_hypothesis_summary: pd.DataFrame,
    equivalence_groups: pd.DataFrame,
    candidate_mapping: pd.DataFrame,
    *,
    expected_candidate_count: int,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Apply one fixed-family Holm correction and assign neutral L5 labels."""
    required = {"signal_equivalence_id", "raw_two_sided_p_value", "test_status"}
    missing = sorted(required.difference(unique_hypothesis_summary.columns))
    if missing:
        raise ValueError("hypothesis summary missing Holm columns: " + ", ".join(missing))
    group_required = {
        "signal_equivalence_id", "canonical_candidate_id", "alias_count",
    }
    group_missing = sorted(group_required.difference(equivalence_groups.columns))
    if group_missing:
        raise ValueError("equivalence groups missing Holm lock columns: " + ", ".join(group_missing))
    if int(expected_candidate_count) <= 0:
        raise ValueError("expected_candidate_count must be positive")
    if equivalence_groups["signal_equivalence_id"].astype(str).duplicated().any():
        raise ValueError("equivalence groups contain duplicate ids")
    alias_counts = pd.to_numeric(equivalence_groups["alias_count"], errors="coerce")
    if alias_counts.isna().any() or (alias_counts < 1).any():
        raise ValueError("equivalence groups contain invalid alias counts")
    if int(alias_counts.sum()) != int(expected_candidate_count):
        raise ValueError(
            "Holm family candidate coverage is incomplete; "
            f"expected={int(expected_candidate_count)}, actual={int(alias_counts.sum())}"
        )
    mapping_required = {
        "candidate_id", "signal_equivalence_id", "canonical_candidate_id",
    }
    mapping_missing = sorted(mapping_required.difference(candidate_mapping.columns))
    if mapping_missing:
        raise ValueError(
            "candidate mapping missing Holm lock columns: " + ", ".join(mapping_missing)
        )
    mapping = candidate_mapping.loc[:, sorted(mapping_required)].copy()
    mapping["candidate_id"] = mapping["candidate_id"].astype(str)
    mapping["signal_equivalence_id"] = mapping["signal_equivalence_id"].astype(str)
    mapping["canonical_candidate_id"] = mapping["canonical_candidate_id"].astype(str)
    if mapping["candidate_id"].duplicated().any():
        raise ValueError("candidate mapping contains duplicate candidate ids")
    if len(mapping) != int(expected_candidate_count):
        raise ValueError(
            "Holm candidate mapping is incomplete; "
            f"expected={int(expected_candidate_count)}, actual={len(mapping)}"
        )
    group_ids = set(equivalence_groups["signal_equivalence_id"].astype(str))
    if set(mapping["signal_equivalence_id"]) != group_ids:
        raise ValueError("candidate mapping and equivalence groups disagree on group ids")
    actual_alias_counts = mapping.groupby("signal_equivalence_id").size()
    declared_alias_counts = pd.Series(
        alias_counts.to_numpy(dtype=int),
        index=equivalence_groups["signal_equivalence_id"].astype(str),
    )
    if not actual_alias_counts.sort_index().equals(declared_alias_counts.sort_index()):
        raise ValueError("candidate mapping row counts disagree with declared alias counts")
    canonical_by_group = equivalence_groups.set_index(
        equivalence_groups["signal_equivalence_id"].astype(str)
    )["canonical_candidate_id"].astype(str)
    mapped_canonical = mapping.groupby("signal_equivalence_id")[
        "canonical_candidate_id"
    ].agg(lambda values: set(values))
    if any(
        values != {canonical_by_group.loc[group_id]}
        for group_id, values in mapped_canonical.items()
    ):
        raise ValueError("candidate mapping canonical ids disagree with equivalence groups")
    for group_id, canonical_id in canonical_by_group.items():
        members = set(
            mapping.loc[
                mapping["signal_equivalence_id"] == group_id, "candidate_id"
            ]
        )
        if canonical_id not in members:
            raise ValueError("equivalence-group canonical candidate is absent from mapping")
    expected = equivalence_groups["signal_equivalence_id"].astype(str).tolist()
    if len(expected) != len(set(expected)):
        raise ValueError("expected equivalence ids contain duplicates")
    summary = unique_hypothesis_summary.copy()
    if summary["signal_equivalence_id"].astype(str).duplicated().any():
        raise ValueError("hypothesis summary contains duplicate equivalence ids")
    actual = set(summary["signal_equivalence_id"].astype(str))
    if actual != set(expected):
        missing_ids = sorted(set(expected).difference(actual))
        extra_ids = sorted(actual.difference(expected))
        raise ValueError(f"Holm family is incomplete; missing={missing_ids}, extra={extra_ids}")
    summary = summary.set_index("signal_equivalence_id").loc[expected].reset_index()
    valid_mask = summary["test_status"].astype(str) == "valid"
    raw = pd.to_numeric(summary["raw_two_sided_p_value"], errors="coerce")
    if raw[valid_mask].isna().any():
        raise ValueError("valid hypothesis has no finite raw p-value")
    correction_input = raw.where(valid_mask, 1.0).to_numpy(dtype=float)
    adjusted_all = research_stats.holm_adjusted_p_values(correction_input)
    summary["holm_adjusted_p_value"] = np.where(valid_mask, adjusted_all, np.nan)
    summary["holm_family_size"] = len(expected)
    summary["incremental_information_label"] = np.where(
        ~valid_mask,
        "incremental_information_test_invalid",
        np.where(
            summary["holm_adjusted_p_value"] <= float(alpha),
            "incremental_information_detected",
            "incremental_information_not_detected",
        ),
    )
    return summary


__all__ = [
    "ALPHA_GRID",
    "ALL_RAW_PREDICTOR_COLUMNS",
    "DOLLAR_VOLUME_COLUMNS",
    "LEVEL0_COLUMNS",
    "LEVEL0_RAW_COLUMNS",
    "PRICE_VOLUME_COLUMNS",
    "REPLICATION_R2_THRESHOLD",
    "RETURN_COLUMNS",
    "VOLATILITY_COLUMNS",
    "VOLUME_SURPRISE_COLUMNS",
    "CanonicalSupportArtifacts",
    "CandidateResidualDependencyArtifacts",
    "DoubleResidualArtifacts",
    "DoubleResidualDailyFamilyArtifacts",
    "DoubleResidualFamilyInferenceArtifacts",
    "DoubleResidualTimeRandomizationArtifacts",
    "FoldTargetSignalArtifacts",
    "ResidualEquivalenceArtifacts",
    "ResidualCandidateSensitivityArtifacts",
    "ResidualFamilyInformationArtifacts",
    "ResidualInformationArtifacts",
    "ReplacementFeatureArtifacts",
    "RegisteredModelReplicationArtifacts",
    "RegisteredResidualLabelArtifacts",
    "ResidualTimeRandomizationArtifacts",
    "ResidualTimeRandomizationSummaryArtifacts",
    "ResidualTimeShiftPreparationArtifacts",
    "RidgeReplicationArtifacts",
    "SignalReplayArtifacts",
    "build_canonical_common_support",
    "build_cross_sectional_outcome_target",
    "build_double_residual_daily_family",
    "build_fold_canonical_common_support",
    "build_fold_target_signals",
    "build_price_volume_replacement_features",
    "build_replay_signal_frame",
    "build_shadow_priority_inferences",
    "build_substitution_target_manifest",
    "audit_target_signal_reproduction",
    "audit_residual_signal_equivalence",
    "audit_precomputed_executable_return_lineage",
    "apply_residual_information_holm",
    "apply_train_selected_registered_replica_labels",
    "apply_horizon_residual_family_holm",
    "classify_replication_difficulty",
    "compare_signal_replays",
    "evaluate_precomputed_oos_signals",
    "evaluate_executable_precomputed_oos_signals",
    "evaluate_candidate_residual_dependency_adjusted_information",
    "evaluate_cross_fitted_double_residuals",
    "evaluate_double_residual_time_randomization",
    "evaluate_registered_replica_model_capability_performance",
    "evaluate_l5_residual_time_randomization",
    "assemble_train_selected_registered_replica",
    "fit_same_sample_registered_replicas",
    "fit_walk_forward_registered_replicas",
    "fit_walk_forward_ridge_replicas",
    "infer_double_residual_family",
    "model_feature_sets",
    "l5_time_randomization_legal_offset_manifest",
    "l5_time_randomization_seed_manifest",
    "prepare_l5_residual_time_shift_effects",
    "registered_replica_model_specs",
    "replacement_feature_manifest",
    "prepare_level2_full_residual_predictions",
    "reconstruct_folds_from_oos_holdings",
    "summarize_replication_metrics",
    "summarize_registered_model_diagnostics",
    "summarize_l5_residual_time_randomization",
    "summarize_residual_construction_audit",
    "summarize_residual_candidate_sensitivity",
    "summarize_residual_incremental_information",
    "summarize_horizon_residual_family_information",
]
