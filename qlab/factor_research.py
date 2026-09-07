"""Cross-sectional factor research primitives.

The functions here are intentionally domain-neutral. Domain research modules
provide feature names, horizons, fold specs, and control columns; qlab owns the
common mechanics for cross-sectional IC, bucket diagnostics, Fama-MacBeth
diagnostics, and multi-factor combination weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import ceil
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import research_stats
from .walkforward import select_dates, walk_forward_splits


@dataclass(frozen=True)
class ComboSpec:
    combo_id: str
    track: str
    panel_frequency: str
    return_horizon: str
    feature_names: tuple[str, ...]
    weight_scheme: str = "equal"


@dataclass(frozen=True)
class FeatureTrainStat:
    direction: int
    mean_ic: float
    std_ic: float
    icir: float
    hac_t_stat: float
    observation_count: int


@dataclass(frozen=True)
class MainlineCatalogOverride:
    track: str
    panel_frequency: str
    return_horizon: str
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class SingleFeatureDirection:
    train_mean_ic: float
    direction: int
    observation_count: int
    status: str


@dataclass(frozen=True)
class SingleFeatureTwoGateScanResult:
    """The shared formal single-feature L2 two-gate result.

    This is the one qlab-owned admission surface used by both the historical
    research runner and known-truth discovery.  It returns the existing
    diagnostics rather than recomputing a second summary in a caller.
    """

    summary: dict[str, object]
    ic_detail: pd.DataFrame
    bucket_detail: pd.DataFrame
    direction_frame: pd.DataFrame
    rank_diagnostics: pd.DataFrame
    bucket_diagnostics: pd.DataFrame


def parse_horizon_csv(value: str, horizon_deltas: Mapping[str, pd.Timedelta]) -> list[str]:
    horizons = [token.strip() for token in value.split(",") if token.strip()]
    if not horizons:
        raise ValueError("no forward-return horizons requested")
    invalid = [horizon for horizon in horizons if horizon not in horizon_deltas]
    if invalid:
        raise ValueError("unsupported forward-return horizons: " +
                         ", ".join(sorted(set(invalid))))
    return horizons


def minimum_embargo_days_for_horizons(
    horizons: Sequence[str] | str,
    horizon_deltas: Mapping[str, pd.Timedelta],
) -> int:
    horizon_tokens = [horizons] if isinstance(
        horizons, str) else list(horizons)
    if not horizon_tokens:
        raise ValueError(
            "no forward-return horizons requested for walk-forward validation")
    required_days = 0
    for horizon in horizon_tokens:
        if horizon not in horizon_deltas:
            raise ValueError("unsupported forward-return horizon: " + horizon)
        horizon_days = horizon_deltas[horizon] / pd.Timedelta(days=1)
        required_days = max(required_days, int(ceil(float(horizon_days))))
    return required_days


def walk_forward_spec_for_frequency(
    frequency: str,
    default_specs: Mapping[str, Mapping[str, int]],
    horizon_deltas: Mapping[str, pd.Timedelta],
    override: Mapping[str, int] | None = None,
    *,
    horizons: Sequence[str] | str,
) -> dict[str, int]:
    if frequency not in default_specs:
        raise ValueError(
            "unsupported panel frequency for walk-forward: " + frequency)
    spec = dict(default_specs[frequency])
    if override is not None:
        spec.update(override)
    required = {"train_days", "test_days", "embargo_days", "step_days"}
    missing = required.difference(spec)
    if missing:
        raise ValueError("walk-forward spec missing keys: " +
                         ", ".join(sorted(missing)))
    for key in required:
        value = int(spec[key])
        if value < 0:
            raise ValueError(
                f"walk-forward spec must be non-negative for {key}: {value}")
        if key != "embargo_days" and value == 0:
            raise ValueError(
                f"walk-forward spec must be positive for {key}: {value}")
        spec[key] = value
    required_embargo_days = minimum_embargo_days_for_horizons(
        horizons, horizon_deltas)
    if spec["embargo_days"] < required_embargo_days:
        horizon_label = horizons if isinstance(
            horizons, str) else ",".join(horizons)
        raise ValueError(
            "walk-forward spec embargo_days must be at least "
            f"{required_embargo_days}d for horizon(s): {horizon_label}"
        )
    return spec


def build_walk_forward_folds(
    decision_index: pd.DatetimeIndex,
    frequency: str,
    horizon: str,
    default_specs: Mapping[str, Mapping[str, int]],
    horizon_deltas: Mapping[str, pd.Timedelta],
    walk_forward_spec: Mapping[str, int] | None = None,
) -> tuple[dict[str, int], list]:
    spec = walk_forward_spec_for_frequency(
        frequency,
        default_specs,
        horizon_deltas,
        walk_forward_spec,
        horizons=horizon,
    )
    dates = pd.DatetimeIndex(decision_index).sort_values().unique()
    folds = list(
        walk_forward_splits(
            dates,
            train_days=spec["train_days"],
            test_days=spec["test_days"],
            embargo_days=spec["embargo_days"],
            step_days=spec["step_days"],
        )
    )
    if not folds:
        raise ValueError(
            f"no walk-forward folds generated for {frequency} with train={spec['train_days']}d, "
            f"test={spec['test_days']}d, embargo={spec['embargo_days']}d, step={spec['step_days']}d"
        )
    return spec, folds


def periods_per_year(frequency: str, frequency_periods_per_year: Mapping[str, int | float]) -> int | float:
    if frequency not in frequency_periods_per_year:
        raise ValueError(
            "unsupported panel frequency for annualization: " + frequency)
    return frequency_periods_per_year[frequency]


def annualized_sharpe_for_frequency(
    values: pd.Series | np.ndarray | list[float],
    frequency: str,
    frequency_periods_per_year: Mapping[str, int | float],
) -> float:
    return research_stats.annualized_sharpe_from_periods(
        values,
        periods_per_year(frequency, frequency_periods_per_year),
    )


def annualized_mean_return(
    values: pd.Series | np.ndarray | list[float],
    frequency: str,
    frequency_periods_per_year: Mapping[str, int | float],
) -> float:
    series = pd.Series(values).dropna().astype(float)
    if series.empty:
        return float("nan")
    return float(series.mean() * periods_per_year(frequency, frequency_periods_per_year))


def annualized_volatility(
    values: pd.Series | np.ndarray | list[float],
    frequency: str,
    frequency_periods_per_year: Mapping[str, int | float],
) -> float:
    series = pd.Series(values).dropna().astype(float)
    if len(series) < 2:
        return float("nan")
    return float(series.std(ddof=1) * np.sqrt(periods_per_year(frequency, frequency_periods_per_year)))


def max_drawdown_from_returns(values: pd.Series | np.ndarray | list[float]) -> float:
    series = pd.Series(values).dropna().astype(float)
    if series.empty:
        return float("nan")
    equity_curve = (1.0 + series).cumprod()
    drawdown = equity_curve.div(equity_curve.cummax()) - 1.0
    return float(drawdown.min()) if not drawdown.empty else float("nan")


def fold_positive_share_from_returns(detail_frame: pd.DataFrame, return_column: str) -> float:
    """Share of folds whose mean return is positive for one return column."""
    if detail_frame.empty or return_column not in detail_frame.columns:
        return float("nan")
    return float((detail_frame.groupby("fold_idx")[return_column].mean() > 0.0).mean())


def top_fold_contribution_from_returns(detail_frame: pd.DataFrame, return_column: str) -> float:
    """Largest fold contribution divided by total return for one return column."""
    if detail_frame.empty or return_column not in detail_frame.columns:
        return float("nan")
    total_return = float(detail_frame[return_column].sum())
    if total_return == 0.0 or not np.isfinite(total_return):
        return float("nan")
    return float(detail_frame.groupby("fold_idx")[return_column].sum().max() / total_return)


def decision_index_for_frame(frame: pd.DataFrame) -> pd.Index:
    if isinstance(frame.index, pd.MultiIndex):
        return pd.Index(frame.index.get_level_values(0).unique())
    return pd.Index(frame.index.unique())


def _fold_count(detail_frame: pd.DataFrame) -> int:
    if detail_frame.empty:
        return 0
    if "fold_idx" not in detail_frame.columns:
        return 1
    return int(detail_frame["fold_idx"].nunique())


def filter_feature_decision_frame(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    return_column: str = "forward_return",
    require_all_features: bool = False,
) -> pd.DataFrame:
    feature_names = tuple(feature_names)
    if not feature_names:
        return frame.iloc[0:0].copy()
    missing = [
        feature_name for feature_name in feature_names if feature_name not in frame.columns]
    if missing:
        raise ValueError("missing feature columns: " + ", ".join(missing))
    feature_values = frame[list(feature_names)]
    feature_mask = feature_values.notna().all(
        axis=1) if require_all_features else feature_values.notna().any(axis=1)
    if return_column in frame.columns:
        feature_mask = feature_mask & frame[return_column].notna()
    return frame.loc[feature_mask].copy()


def rank_ic_diagnostics_for_features(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    min_cross_section: int,
    *,
    return_column: str = "forward_return",
) -> dict[str, list[dict[str, object]]]:
    feature_names = tuple(feature_names)
    if not feature_names:
        return {}
    working = frame[[*feature_names, return_column]
                    ].rename(columns={return_column: "forward_return"})
    decision_index = decision_index_for_frame(working)
    if decision_index.empty:
        return {feature_name: [] for feature_name in feature_names}

    empty_rows = [
        {
            "decision_ts": decision_ts,
            "cross_section_size": 0,
            "status": "small_cross_section",
            "raw_rank_ic": float("nan"),
        }
        for decision_ts in decision_index
    ]
    if working["forward_return"].dropna().empty:
        return {feature_name: [dict(row) for row in empty_rows] for feature_name in feature_names}

    feature_values = working[list(feature_names)].where(
        working["forward_return"].notna(), axis=0)
    return_values = pd.DataFrame(
        {
            feature_name: working["forward_return"].where(
                working[feature_name].notna())
            for feature_name in feature_names
        },
        index=working.index,
    )
    grouped_features = feature_values.groupby(level=0, sort=False)
    grouped_returns = return_values.groupby(level=0, sort=False)
    counts = feature_values.notna().groupby(level=0, sort=False).sum().reindex(
        decision_index, fill_value=0).astype(int)
    feature_unique = grouped_features.nunique(dropna=True).reindex(
        decision_index, fill_value=0).astype(int)
    return_unique = grouped_returns.nunique(dropna=True).reindex(
        decision_index, fill_value=0).astype(int)

    feature_ranks = grouped_features.rank(method="average")
    return_ranks = grouped_returns.rank(method="average")
    feature_rank_sq = feature_ranks * feature_ranks
    return_rank_sq = return_ranks * return_ranks
    rank_product = feature_ranks * return_ranks
    feature_sum = feature_ranks.groupby(
        level=0, sort=False).sum().reindex(decision_index)
    return_sum = return_ranks.groupby(
        level=0, sort=False).sum().reindex(decision_index)
    feature_sq_sum = feature_rank_sq.groupby(
        level=0, sort=False).sum().reindex(decision_index)
    return_sq_sum = return_rank_sq.groupby(
        level=0, sort=False).sum().reindex(decision_index)
    product_sum = rank_product.groupby(
        level=0, sort=False).sum().reindex(decision_index)
    n = counts.astype(float)
    numerator = n * product_sum - feature_sum * return_sum
    feature_denominator = n * feature_sq_sum - feature_sum * feature_sum
    return_denominator = n * return_sq_sum - return_sum * return_sum
    denominator = np.sqrt(feature_denominator * return_denominator)
    correlations = numerator / denominator

    diagnostics_by_feature: dict[str, list[dict[str, object]]] = {}
    decision_values = decision_index.to_numpy()
    for feature_name in feature_names:
        cross_sections = counts[feature_name].to_numpy(dtype=int, copy=False)
        feature_unique_values = feature_unique[feature_name].to_numpy(
            dtype=int, copy=False)
        return_unique_values = return_unique[feature_name].to_numpy(
            dtype=int, copy=False)
        raw_values = correlations[feature_name].to_numpy(
            dtype=float, copy=True)
        statuses = np.full(len(decision_values), "ok", dtype=object)
        small_mask = cross_sections < min_cross_section
        constant_feature_mask = (~small_mask) & (feature_unique_values <= 1)
        constant_return_mask = (
            ~small_mask) & (~constant_feature_mask) & (return_unique_values <= 1)
        nan_mask = (
            ~small_mask
            & (~constant_feature_mask)
            & (~constant_return_mask)
            & pd.isna(raw_values)
        )
        statuses[small_mask] = "small_cross_section"
        statuses[constant_feature_mask] = "constant_feature"
        statuses[constant_return_mask] = "constant_return"
        statuses[nan_mask] = "nan_rank_ic"
        raw_values[statuses != "ok"] = np.nan
        rows = [
            {
                "decision_ts": decision_ts,
                "cross_section_size": int(cross_section_size),
                "status": status,
                "raw_rank_ic": float(raw_rank_ic),
            }
            for decision_ts, cross_section_size, status, raw_rank_ic in zip(
                decision_values,
                cross_sections,
                statuses,
                raw_values,
                strict=True,
            )
        ]
        diagnostics_by_feature[feature_name] = rows
    return diagnostics_by_feature


def rank_ic_diagnostics_for_frame(frame: pd.DataFrame, feature_name: str, min_cross_section: int) -> list[dict[str, object]]:
    return rank_ic_diagnostics_for_features(
        frame=frame,
        feature_names=(feature_name,),
        min_cross_section=min_cross_section,
    )[feature_name]


def rank_ic_rows_for_frame(frame: pd.DataFrame, feature_name: str, min_cross_section: int) -> list[dict[str, object]]:
    diagnostics = rank_ic_diagnostics_for_frame(
        frame=frame, feature_name=feature_name, min_cross_section=min_cross_section)
    return [
        {
            "decision_ts": row["decision_ts"],
            "raw_rank_ic": row["raw_rank_ic"],
            "cross_section_size": row["cross_section_size"],
        }
        for row in diagnostics
        if row["status"] == "ok"
    ]


def single_feature_train_direction(
    train_slice: pd.DataFrame,
    feature_name: str,
    min_cross_section: int,
    *,
    epsilon: float = 1e-12,
) -> SingleFeatureDirection:
    rows = rank_ic_rows_for_frame(
        train_slice[["symbol", feature_name, "forward_return"]],
        feature_name,
        min_cross_section,
    )
    if not rows:
        return SingleFeatureDirection(
            train_mean_ic=float("nan"),
            direction=0,
            observation_count=0,
            status="no_train_ic",
        )
    ic_series = pd.DataFrame(rows)["raw_rank_ic"].astype(float)
    train_mean_ic = float(ic_series.mean())
    if not np.isfinite(train_mean_ic) or abs(train_mean_ic) <= float(epsilon):
        return SingleFeatureDirection(
            train_mean_ic=train_mean_ic,
            direction=0,
            observation_count=len(ic_series),
            status="no_train_direction",
        )
    return SingleFeatureDirection(
        train_mean_ic=train_mean_ic,
        direction=1 if train_mean_ic > 0.0 else -1,
        observation_count=len(ic_series),
        status="ok",
    )


def two_gate_support_flags(
    *,
    ic_mean: float,
    ic_hac_t_stat: float,
    bucket_spread_mean_return: float,
    bucket_monotonic_pair_pass_share: float,
    one_sided_t_threshold: float = 1.645,
    bucket_monotonic_threshold: float = 0.75,
) -> dict[str, bool]:
    stage1 = bool(
        np.isfinite(ic_mean)
        and float(ic_mean) > 0.0
        and np.isfinite(ic_hac_t_stat)
        and float(ic_hac_t_stat) >= float(one_sided_t_threshold)
    )
    stage2 = bool(
        np.isfinite(bucket_spread_mean_return)
        and float(bucket_spread_mean_return) > 0.0
        and np.isfinite(bucket_monotonic_pair_pass_share)
        and float(bucket_monotonic_pair_pass_share) >= float(bucket_monotonic_threshold)
    )
    return {
        "stage1_ic_support": stage1,
        "stage2_bucket_support": stage2,
        "two_gate_support": bool(stage1 and stage2),
    }


def three_gate_support_flags(
    *,
    ic_mean: float,
    ic_hac_t_stat: float,
    bucket_spread_mean_return: float,
    bucket_monotonic_pair_pass_share: float,
    fm_mean_gamma: float,
    fm_hac_t_stat: float,
    one_sided_t_threshold: float = 1.645,
    bucket_monotonic_threshold: float = 0.75,
) -> dict[str, bool]:
    result = two_gate_support_flags(
        ic_mean=ic_mean,
        ic_hac_t_stat=ic_hac_t_stat,
        bucket_spread_mean_return=bucket_spread_mean_return,
        bucket_monotonic_pair_pass_share=bucket_monotonic_pair_pass_share,
        one_sided_t_threshold=one_sided_t_threshold,
        bucket_monotonic_threshold=bucket_monotonic_threshold,
    )
    stage3 = bool(
        np.isfinite(fm_mean_gamma)
        and float(fm_mean_gamma) > 0.0
        and np.isfinite(fm_hac_t_stat)
        and float(fm_hac_t_stat) >= float(one_sided_t_threshold)
    )
    return {
        **result,
        "stage3_fm_support": stage3,
        "three_gate_support": bool(result["two_gate_support"] and stage3),
    }


def summarize_ic_series(
    panel_frequency: str,
    horizon: str,
    feature_name: str,
    detail_frame: pd.DataFrame,
    walk_forward_spec: Mapping[str, int],
    test_diagnostics: list[dict[str, object]],
    hac_overlap_lags: int = 0,
) -> dict[str, object]:
    if detail_frame.empty:
        ic_series = pd.Series(dtype=float)
        unique_train_mean = pd.Series(dtype=float)
        cross_section_median = float("nan")
    else:
        ic_series = detail_frame["rank_ic"].astype(float)
        unique_train_mean = detail_frame[["fold_idx", "train_mean_ic"]].drop_duplicates()[
            "train_mean_ic"]
        cross_section_median = detail_frame["cross_section_size"].median()
    observation_count = len(ic_series)
    std = float(ic_series.std(ddof=1)
                ) if observation_count > 1 else float("nan")
    mean_ic = float(ic_series.mean()) if observation_count else float("nan")
    icir = float(
        mean_ic / std) if observation_count > 1 and np.isfinite(std) and std > 0 else float("nan")
    positive_share = float((ic_series > 0).mean()
                           ) if observation_count else float("nan")
    hac_lags = research_stats.newey_west_max_lags(
        observation_count, overlap_lags=hac_overlap_lags)
    diagnostic_frame = pd.DataFrame(test_diagnostics)
    test_decision_count = len(diagnostic_frame)
    status_counts = diagnostic_frame["status"].value_counts(
    ) if test_decision_count else pd.Series(dtype=int)
    scored_decision_count = int(status_counts.get("ok", 0))
    skipped_decision_count = test_decision_count - scored_decision_count
    skipped_decision_share = float(
        skipped_decision_count / test_decision_count) if test_decision_count else float("nan")
    return {
        "panel_frequency": panel_frequency,
        "return_horizon": horizon,
        "feature_name": feature_name,
        "train_days": walk_forward_spec["train_days"],
        "test_days": walk_forward_spec["test_days"],
        "embargo_days": walk_forward_spec["embargo_days"],
        "step_days": walk_forward_spec["step_days"],
        "n_folds": _fold_count(detail_frame),
        "mean_ic": mean_ic,
        "std_ic": std,
        "icir": icir,
        "ic_positive_share": positive_share,
        "hac_t_stat": research_stats.hac_t_stat(ic_series, max_lags=hac_lags),
        "hac_lags": hac_lags,
        "mean_train_ic": float(unique_train_mean.mean()) if not unique_train_mean.empty else float("nan"),
        "ic_observation_count": observation_count,
        "cross_section_size_median": cross_section_median,
        "test_decision_count": test_decision_count,
        "scored_decision_count": scored_decision_count,
        "skipped_decision_count": skipped_decision_count,
        "skipped_decision_share": skipped_decision_share,
        "skipped_small_cross_section_count": int(status_counts.get("small_cross_section", 0)),
        "skipped_constant_feature_count": int(status_counts.get("constant_feature", 0)),
        "skipped_constant_return_count": int(status_counts.get("constant_return", 0)),
        "skipped_nan_rank_ic_count": int(status_counts.get("nan_rank_ic", 0)),
    }


def bucket_diagnostics_for_group(
    group: pd.DataFrame,
    feature_name: str,
    direction: int,
    n_buckets: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    valid = group.reset_index(names="decision_ts")[
        ["decision_ts", "symbol", feature_name, "forward_return"]].dropna()
    cross_section_size = len(valid)
    diagnostic = {
        "decision_ts": valid["decision_ts"].iloc[0] if cross_section_size else group.index[0],
        "cross_section_size": cross_section_size,
        "status": "ok",
    }
    if cross_section_size < n_buckets:
        diagnostic["status"] = "small_cross_section"
        return pd.DataFrame(), diagnostic
    if valid[feature_name].nunique(dropna=True) <= 1:
        diagnostic["status"] = "constant_feature"
        return pd.DataFrame(), diagnostic
    oriented_feature = valid[feature_name].to_numpy(
        dtype=float, copy=False) * direction
    symbol_order = valid["symbol"].astype(
        str).to_numpy(dtype=object, copy=False)
    sort_index = np.lexsort((symbol_order, oriented_feature))
    sorted_valid = valid.iloc[sort_index].reset_index(drop=True)
    sorted_valid["oriented_feature"] = oriented_feature[sort_index]
    sorted_valid["bucket"] = 0
    for bucket_idx, positions in enumerate(np.array_split(np.arange(cross_section_size), n_buckets), start=1):
        sorted_valid.loc[positions, "bucket"] = bucket_idx
    bucket_frame = sorted_valid.groupby("bucket", observed=False)["forward_return"].agg(
        bucket_return="mean", bucket_size="size").reset_index()
    return bucket_frame, diagnostic


def bucket_diagnostics_for_frame(
    frame: pd.DataFrame,
    feature_name: str,
    direction: int,
    n_buckets: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    decision_index = decision_index_for_frame(frame)
    if decision_index.empty:
        return pd.DataFrame(columns=["decision_ts", "bucket", "bucket_return", "bucket_size"]), []

    decision_values = frame.index.get_level_values(0) if isinstance(
        frame.index, pd.MultiIndex) else frame.index
    if "symbol" in frame.columns:
        symbol_values = frame["symbol"].astype(
            str).to_numpy(dtype=object, copy=False)
    elif isinstance(frame.index, pd.MultiIndex) and "symbol" in frame.index.names:
        symbol_values = frame.index.get_level_values(
            "symbol").astype(str).to_numpy(dtype=object, copy=False)
    elif isinstance(frame.index, pd.MultiIndex) and frame.index.nlevels > 1:
        symbol_values = frame.index.get_level_values(
            1).astype(str).to_numpy(dtype=object, copy=False)
    else:
        symbol_values = pd.RangeIndex(len(frame)).astype(
            str).to_numpy(dtype=object, copy=False)
    working = pd.DataFrame(
        {
            "decision_ts": decision_values,
            "symbol": symbol_values,
            feature_name: frame[feature_name].to_numpy(dtype=float, copy=False),
            "forward_return": frame["forward_return"].to_numpy(dtype=float, copy=False),
        }
    )
    valid = working.dropna(subset=[feature_name, "forward_return"])
    if valid.empty:
        diagnostics = [
            {"decision_ts": decision_ts, "cross_section_size": 0,
                "status": "small_cross_section"}
            for decision_ts in decision_index
        ]
        return pd.DataFrame(columns=["decision_ts", "bucket", "bucket_return", "bucket_size"]), diagnostics

    grouped = valid.groupby("decision_ts", sort=False)
    counts = grouped.size().reindex(decision_index, fill_value=0).astype(int)
    feature_unique = grouped[feature_name].nunique(
        dropna=True).reindex(decision_index, fill_value=0).astype(int)
    diagnostics: list[dict[str, object]] = []
    ok_decisions: list[object] = []
    for decision_ts in decision_index:
        cross_section_size = int(counts.loc[decision_ts])
        status = "ok"
        if cross_section_size < n_buckets:
            status = "small_cross_section"
        elif int(feature_unique.loc[decision_ts]) <= 1:
            status = "constant_feature"
        else:
            ok_decisions.append(decision_ts)
        diagnostics.append({"decision_ts": decision_ts,
                           "cross_section_size": cross_section_size, "status": status})
    if not ok_decisions:
        return pd.DataFrame(columns=["decision_ts", "bucket", "bucket_return", "bucket_size"]), diagnostics

    scored = valid[valid["decision_ts"].isin(ok_decisions)].copy()
    scored["oriented_feature"] = scored[feature_name].to_numpy(
        dtype=float, copy=False) * direction
    scored = scored.sort_values(
        ["decision_ts", "oriented_feature", "symbol"], kind="mergesort").reset_index(drop=True)
    group_sizes = scored.groupby("decision_ts", sort=False)[
        "symbol"].transform("size").astype(int)
    positions = scored.groupby(
        "decision_ts", sort=False).cumcount().astype(int)
    base_sizes = group_sizes // n_buckets
    remainders = group_sizes % n_buckets
    first_block_limits = (base_sizes + 1) * remainders
    scored["bucket"] = np.where(
        positions < first_block_limits,
        positions // (base_sizes + 1),
        remainders + (positions - first_block_limits) // base_sizes,
    ).astype(int) + 1
    bucket_frame = (
        scored.groupby(["decision_ts", "bucket"], observed=False)[
            "forward_return"]
        .agg(bucket_return="mean", bucket_size="size")
        .reset_index()
    )
    return bucket_frame, diagnostics


def summarize_bucket_backtest(
    panel_frequency: str,
    horizon: str,
    feature_name: str,
    detail_frame: pd.DataFrame,
    walk_forward_spec: Mapping[str, int],
    decision_diagnostics: list[dict[str, object]],
    n_buckets: int,
    frequency_periods_per_year: Mapping[str, int | float],
    annualization_frequency: str | None = None,
) -> dict[str, object]:
    return_frequency = panel_frequency if annualization_frequency is None else annualization_frequency
    diagnostic_frame = pd.DataFrame(decision_diagnostics)
    test_decision_count = len(diagnostic_frame)
    status_counts = diagnostic_frame["status"].value_counts(
    ) if test_decision_count else pd.Series(dtype=int)
    scored_decision_count = int(status_counts.get("ok", 0))
    skipped_decision_count = test_decision_count - scored_decision_count
    skipped_decision_share = float(
        skipped_decision_count / test_decision_count) if test_decision_count else float("nan")
    if detail_frame.empty:
        bucket_returns = pd.DataFrame(columns=range(1, n_buckets + 1))
        bucket_sizes = pd.DataFrame(columns=range(1, n_buckets + 1))
    else:
        bucket_returns = detail_frame.pivot_table(
            index="decision_ts", columns="bucket", values="bucket_return", aggfunc="first"
        ).reindex(columns=range(1, n_buckets + 1))
        bucket_sizes = detail_frame.pivot_table(
            index="decision_ts", columns="bucket", values="bucket_size", aggfunc="first"
        ).reindex(columns=range(1, n_buckets + 1))
    bucket_mean_returns = bucket_returns.mean(
        axis=0) if not bucket_returns.empty else pd.Series(dtype=float)
    bucket_avg_sizes = bucket_sizes.mean(
        axis=0) if not bucket_sizes.empty else pd.Series(dtype=float)
    spread_series = bucket_returns[n_buckets] - \
        bucket_returns[1] if not bucket_returns.empty else pd.Series(
            dtype=float)
    spread_series = spread_series.dropna().astype(float)
    spread_mean = float(spread_series.mean()
                        ) if not spread_series.empty else float("nan")
    spread_std = float(spread_series.std(ddof=1)) if len(
        spread_series) > 1 else float("nan")
    bucket_mean_series = pd.Series(
        [float(bucket_mean_returns.get(bucket, np.nan))
         for bucket in range(1, n_buckets + 1)],
        index=range(1, n_buckets + 1),
        dtype=float,
    )
    monotonic_pair_pass_count = int(
        sum(bucket_mean_series.iloc[idx] <= bucket_mean_series.iloc[idx + 1]
            for idx in range(n_buckets - 1))
    ) if bucket_mean_series.notna().all() else 0
    monotonic_pair_pass_share = float(
        monotonic_pair_pass_count / (n_buckets - 1))
    monotonic_increasing = bool(monotonic_pair_pass_count == (
        n_buckets - 1) and bucket_mean_series.notna().all())
    monotonic_spearman = float(
        pd.Series(range(1, n_buckets + 1),
                  dtype=float).corr(bucket_mean_series, method="spearman")
    ) if bucket_mean_series.notna().all() else float("nan")
    summary = {
        "panel_frequency": panel_frequency,
        "return_horizon": horizon,
        "feature_name": feature_name,
        "train_days": walk_forward_spec["train_days"],
        "test_days": walk_forward_spec["test_days"],
        "embargo_days": walk_forward_spec["embargo_days"],
        "step_days": walk_forward_spec["step_days"],
        "n_folds": _fold_count(detail_frame),
        "spread_observation_count": int(len(spread_series)),
        "test_decision_count": test_decision_count,
        "scored_decision_count": scored_decision_count,
        "skipped_decision_count": skipped_decision_count,
        "skipped_decision_share": skipped_decision_share,
        "skipped_small_cross_section_count": int(status_counts.get("small_cross_section", 0)),
        "skipped_constant_feature_count": int(status_counts.get("constant_feature", 0)),
        "spread_mean_return": spread_mean,
        "spread_annualized_return": float(spread_mean * periods_per_year(return_frequency, frequency_periods_per_year)) if np.isfinite(spread_mean) else float("nan"),
        "spread_std_return": spread_std,
        "spread_t_stat": research_stats.simple_t_stat(spread_series),
        "spread_sharpe": annualized_sharpe_for_frequency(spread_series, return_frequency, frequency_periods_per_year),
        "spread_positive_share": float((spread_series > 0).mean()) if not spread_series.empty else float("nan"),
        "monotonic_increasing": monotonic_increasing,
        "monotonic_pair_pass_count": monotonic_pair_pass_count,
        "monotonic_pair_pass_share": monotonic_pair_pass_share,
        "monotonic_spearman": monotonic_spearman,
        "long_leg_mean_return": float(bucket_mean_series.iloc[-1]) if not bucket_mean_series.empty else float("nan"),
        "short_leg_mean_return": float(bucket_mean_series.iloc[0]) if not bucket_mean_series.empty else float("nan"),
    }
    for bucket in range(1, n_buckets + 1):
        mean_return = float(bucket_mean_returns.get(bucket, np.nan))
        summary[f"q{bucket}_mean_return"] = mean_return
        summary[f"q{bucket}_annualized_return"] = float(
            mean_return *
            periods_per_year(return_frequency, frequency_periods_per_year)
        ) if np.isfinite(mean_return) else float("nan")
        summary[f"q{bucket}_avg_size"] = float(
            bucket_avg_sizes.get(bucket, np.nan))
    return summary


def scan_single_feature_two_gate_v1(
    base_frame: pd.DataFrame,
    feature_name: str,
    horizon: str,
    *,
    panel_frequency: str,
    folds: Sequence,
    walk_forward_spec: Mapping[str, int],
    source_timeframes: Sequence[str],
    horizon_deltas: Mapping[str, pd.Timedelta],
    min_cross_section: int,
    n_buckets: int,
    frequency_periods_per_year: Mapping[str, int | float],
    one_sided_t_threshold: float = 1.645,
    bucket_monotonic_threshold: float = 0.75,
    train_ic_epsilon: float = 1e-12,
    signal_timeframe: str | None = None,
) -> SingleFeatureTwoGateScanResult:
    """Run the production single-feature L2 two-gate scan for one unit.

    The sequence and output field names are the formal L2 contract used by
    the KSV4 research runner: non-overlap frequency, canonical frequency
    filter, train-only direction, test rank diagnostics, bucket diagnostics,
    existing summaries, and the existing two-gate flag function.  Callers
    supply the fold ledger and may only organize the returned records.
    """
    if not isinstance(feature_name, str) or not feature_name:
        raise ValueError("feature_name must be non-empty")
    if horizon not in horizon_deltas:
        raise ValueError("horizon missing from horizon_deltas: " + str(horizon))
    if not source_timeframes:
        raise ValueError("source_timeframes must not be empty")
    if signal_timeframe is None:
        evaluation_frequency = non_overlapping_decision_frequency(
            (feature_name,),
            panel_frequency,
            horizon,
            horizon_deltas,
            source_timeframes,
        )
    else:
        signal_timeframe = str(signal_timeframe)
        if signal_timeframe not in source_timeframes:
            raise ValueError("signal_timeframe is not registered: " + signal_timeframe)
        signal_delta = pd.Timedelta(horizon_deltas[signal_timeframe])
        horizon_delta = pd.Timedelta(horizon_deltas[horizon])
        if signal_delta > horizon_delta or horizon_delta % signal_delta != pd.Timedelta(0):
            raise ValueError(
                f"signal timeframe {signal_timeframe} for {feature_name} is not an exact divisor "
                f"of return horizon {horizon}"
            )
        evaluation_frequency = horizon
    combo_spec = ComboSpec(
        combo_id=f"{feature_name}__{horizon}",
        track="single_factor_two_gate",
        panel_frequency=evaluation_frequency,
        return_horizon=horizon,
        feature_names=(feature_name,),
        weight_scheme="single",
    )
    validate_no_overlap_design([combo_spec], horizon_deltas, source_timeframes)
    evaluation_frame = filter_frame_to_decision_frequency(
        base_frame,
        evaluation_frequency,
        horizon_deltas,
    )
    required_columns = ["symbol", feature_name, "forward_return"]
    missing = sorted(set(required_columns).difference(evaluation_frame.columns))
    if missing:
        raise ValueError("single-feature L2 frame missing: " + ", ".join(missing))
    working = evaluation_frame[required_columns].copy()

    ic_rows: list[dict[str, object]] = []
    rank_diagnostics: list[dict[str, object]] = []
    bucket_frames: list[pd.DataFrame] = []
    bucket_diagnostics: list[dict[str, object]] = []
    direction_rows: list[dict[str, object]] = []
    for fold in folds:
        train_slice = select_dates(working, fold, "train")
        test_slice = select_dates(working, fold, "test")
        direction = single_feature_train_direction(
            train_slice,
            feature_name,
            min_cross_section,
            epsilon=train_ic_epsilon,
        )
        direction_rows.append(
            {
                "feature_name": feature_name,
                "return_horizon": horizon,
                "evaluation_frequency": evaluation_frequency,
                "fold_idx": fold.fold_idx,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "train_mean_ic": direction.train_mean_ic,
                "direction": direction.direction if direction.status == "ok" else 0,
                "train_ic_observation_count": direction.observation_count,
                "status": "ok" if direction.status == "ok" else "no_train_direction",
            }
        )
        if direction.status != "ok":
            continue
        test_ic_diagnostics = rank_ic_diagnostics_for_frame(
            test_slice[["symbol", feature_name, "forward_return"]],
            feature_name,
            min_cross_section,
        )
        for diagnostic in test_ic_diagnostics:
            row = {
                **dict(diagnostic),
                "feature_name": feature_name,
                "return_horizon": horizon,
                "evaluation_frequency": evaluation_frequency,
                "fold_idx": fold.fold_idx,
                "train_mean_ic": direction.train_mean_ic,
                "direction": direction.direction,
            }
            rank_diagnostics.append(row)
            if row["status"] == "ok":
                ic_rows.append(
                    {
                        "feature_name": feature_name,
                        "return_horizon": horizon,
                        "evaluation_frequency": evaluation_frequency,
                        "fold_idx": fold.fold_idx,
                        "decision_ts": row["decision_ts"],
                        "train_mean_ic": direction.train_mean_ic,
                        "direction": direction.direction,
                        "raw_rank_ic": row["raw_rank_ic"],
                        "rank_ic": float(row["raw_rank_ic"]) * direction.direction,
                        "cross_section_size": row["cross_section_size"],
                    }
                )
        bucket_frame, diagnostics = bucket_diagnostics_for_frame(
            test_slice[["symbol", feature_name, "forward_return"]],
            feature_name,
            direction.direction,
            n_buckets,
        )
        for diagnostic in diagnostics:
            bucket_diagnostics.append(
                {
                    **dict(diagnostic),
                    "feature_name": feature_name,
                    "return_horizon": horizon,
                    "evaluation_frequency": evaluation_frequency,
                    "fold_idx": fold.fold_idx,
                    "train_mean_ic": direction.train_mean_ic,
                    "direction": direction.direction,
                }
            )
        if not bucket_frame.empty:
            bucket_frames.append(
                bucket_frame.assign(
                    feature_name=feature_name,
                    return_horizon=horizon,
                    evaluation_frequency=evaluation_frequency,
                    fold_idx=fold.fold_idx,
                    train_mean_ic=direction.train_mean_ic,
                    direction=direction.direction,
                )
            )

    ic_detail = pd.DataFrame(ic_rows)
    bucket_detail = pd.concat(bucket_frames, ignore_index=True) if bucket_frames else pd.DataFrame()
    direction_frame = pd.DataFrame(direction_rows)
    ic_summary = summarize_ic_series(
        panel_frequency,
        horizon,
        feature_name,
        ic_detail,
        walk_forward_spec,
        rank_diagnostics,
        hac_overlap_lags=0,
    )
    bucket_summary = summarize_bucket_backtest(
        panel_frequency,
        horizon,
        feature_name,
        bucket_detail,
        walk_forward_spec,
        bucket_diagnostics,
        n_buckets=n_buckets,
        frequency_periods_per_year=frequency_periods_per_year,
        annualization_frequency=horizon,
    )
    row: dict[str, object] = {
        "feature_name": feature_name,
        "return_horizon": horizon,
        "panel_frequency": panel_frequency,
        "evaluation_frequency": evaluation_frequency,
        "source_timeframe": signal_timeframe or (
            feature_name.rsplit("__", 1)[-1] if "__" in feature_name else panel_frequency
        ),
        "train_days": walk_forward_spec["train_days"],
        "test_days": walk_forward_spec["test_days"],
        "embargo_days": walk_forward_spec["embargo_days"],
        "step_days": walk_forward_spec["step_days"],
        "n_folds": ic_summary["n_folds"],
        "direction_ok_fold_count": int(
            (direction_frame["status"] == "ok").sum()
        ) if not direction_frame.empty else 0,
        "direction_skipped_fold_count": int(
            (direction_frame["status"] != "ok").sum()
        ) if not direction_frame.empty else 0,
        "ic_mean": ic_summary["mean_ic"],
        "icir": ic_summary["icir"],
        "ic_hac_t_stat": ic_summary["hac_t_stat"],
        "ic_hac_lags": ic_summary["hac_lags"],
        "ic_positive_share": ic_summary["ic_positive_share"],
        "ic_observation_count": ic_summary["ic_observation_count"],
        "ic_scored_decision_count": ic_summary["scored_decision_count"],
        "bucket_spread_mean_return": bucket_summary["spread_mean_return"],
        "bucket_spread_sharpe": bucket_summary["spread_sharpe"],
        "bucket_spread_positive_share": bucket_summary["spread_positive_share"],
        "bucket_monotonic_increasing": bucket_summary["monotonic_increasing"],
        "bucket_monotonic_pair_pass_share": bucket_summary["monotonic_pair_pass_share"],
        "bucket_scored_decision_count": bucket_summary["scored_decision_count"],
    }
    row.update(
        two_gate_support_flags(
            ic_mean=float(row["ic_mean"]),
            ic_hac_t_stat=float(row["ic_hac_t_stat"]),
            bucket_spread_mean_return=float(row["bucket_spread_mean_return"]),
            bucket_monotonic_pair_pass_share=float(row["bucket_monotonic_pair_pass_share"]),
            one_sided_t_threshold=one_sided_t_threshold,
            bucket_monotonic_threshold=bucket_monotonic_threshold,
        )
    )
    return SingleFeatureTwoGateScanResult(
        summary=row,
        ic_detail=ic_detail,
        bucket_detail=bucket_detail,
        direction_frame=direction_frame,
        rank_diagnostics=pd.DataFrame(rank_diagnostics),
        bucket_diagnostics=pd.DataFrame(bucket_diagnostics),
    )


def top_bottom_diagnostics_for_frame(
    frame: pd.DataFrame,
    feature_name: str,
    direction: int,
    leg_count: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    decision_index = decision_index_for_frame(frame)
    columns = [
        "decision_ts", "leg", "leg_return", "leg_size",
        "spread_return", "cross_section_size",
    ]
    if decision_index.empty:
        return pd.DataFrame(columns=columns), []
    decision_values = frame.index.get_level_values(0) if isinstance(
        frame.index, pd.MultiIndex) else frame.index
    if "symbol" in frame.columns:
        symbol_values = frame["symbol"].astype(
            str).to_numpy(dtype=object, copy=False)
    elif isinstance(frame.index, pd.MultiIndex) and "symbol" in frame.index.names:
        symbol_values = frame.index.get_level_values(
            "symbol").astype(str).to_numpy(dtype=object, copy=False)
    elif isinstance(frame.index, pd.MultiIndex) and frame.index.nlevels > 1:
        symbol_values = frame.index.get_level_values(
            1).astype(str).to_numpy(dtype=object, copy=False)
    else:
        symbol_values = pd.RangeIndex(len(frame)).astype(
            str).to_numpy(dtype=object, copy=False)
    working = pd.DataFrame(
        {
            "decision_ts": decision_values,
            "symbol": symbol_values,
            feature_name: frame[feature_name].to_numpy(dtype=float, copy=False),
            "forward_return": frame["forward_return"].to_numpy(dtype=float, copy=False),
        }
    )
    valid = working.dropna(subset=[feature_name, "forward_return"])
    if valid.empty:
        diagnostics = [
            {"decision_ts": decision_ts, "cross_section_size": 0,
                "status": "small_cross_section"}
            for decision_ts in decision_index
        ]
        return pd.DataFrame(columns=columns), diagnostics

    grouped = valid.groupby("decision_ts", sort=False)
    counts = grouped.size().reindex(decision_index, fill_value=0).astype(int)
    feature_unique = grouped[feature_name].nunique(
        dropna=True).reindex(decision_index, fill_value=0).astype(int)
    diagnostics: list[dict[str, object]] = []
    ok_decisions: list[object] = []
    min_cross_section = int(leg_count) * 2
    for decision_ts in decision_index:
        cross_section_size = int(counts.loc[decision_ts])
        status = "ok"
        if cross_section_size < min_cross_section:
            status = "small_cross_section"
        elif int(feature_unique.loc[decision_ts]) <= 1:
            status = "constant_feature"
        else:
            ok_decisions.append(decision_ts)
        diagnostics.append({
            "decision_ts": decision_ts,
            "cross_section_size": cross_section_size,
            "status": status,
        })
    if not ok_decisions:
        return pd.DataFrame(columns=columns), diagnostics

    scored = valid[valid["decision_ts"].isin(ok_decisions)].copy()
    scored["oriented_feature"] = scored[feature_name].to_numpy(
        dtype=float, copy=False) * direction
    scored = scored.sort_values(
        ["decision_ts", "oriented_feature", "symbol"], kind="mergesort").reset_index(drop=True)
    positions = scored.groupby(
        "decision_ts", sort=False).cumcount().astype(int)
    group_sizes = scored.groupby("decision_ts", sort=False)[
        "symbol"].transform("size").astype(int)
    scored["leg"] = ""
    scored.loc[positions < leg_count, "leg"] = "short"
    scored.loc[positions >= (group_sizes - leg_count), "leg"] = "long"
    legs = scored[scored["leg"].isin(("short", "long"))].copy()
    leg_returns = (
        legs.groupby(["decision_ts", "leg"], sort=False)["forward_return"]
        .agg(leg_return="mean", leg_size="size")
        .reset_index()
    )
    wide_returns = leg_returns.pivot(
        index="decision_ts", columns="leg", values="leg_return")
    spread = (wide_returns["long"] - wide_returns["short"]).rename(
        "spread_return")
    leg_returns = leg_returns.merge(
        spread.reset_index(), on="decision_ts", how="left")
    leg_returns["cross_section_size"] = leg_returns["decision_ts"].map(
        counts.astype(int).to_dict()).astype(int)
    return leg_returns[columns], diagnostics


def summarize_top_bottom_backtest(
    panel_frequency: str,
    horizon: str,
    feature_name: str,
    detail_frame: pd.DataFrame,
    walk_forward_spec: Mapping[str, int],
    decision_diagnostics: list[dict[str, object]],
    leg_count: int,
    frequency_periods_per_year: Mapping[str, int | float],
    annualization_frequency: str | None = None,
) -> dict[str, object]:
    return_frequency = panel_frequency if annualization_frequency is None else annualization_frequency
    diagnostic_frame = pd.DataFrame(decision_diagnostics)
    test_decision_count = len(diagnostic_frame)
    status_counts = diagnostic_frame["status"].value_counts(
    ) if test_decision_count else pd.Series(dtype=int)
    scored_decision_count = int(status_counts.get("ok", 0))
    skipped_decision_count = test_decision_count - scored_decision_count
    if detail_frame.empty:
        leg_returns = pd.DataFrame(columns=["short", "long"])
        leg_sizes = pd.DataFrame(columns=["short", "long"])
        spread_series = pd.Series(dtype=float)
    else:
        leg_returns = detail_frame.pivot_table(
            index="decision_ts", columns="leg", values="leg_return", aggfunc="first"
        ).reindex(columns=["short", "long"])
        leg_sizes = detail_frame.pivot_table(
            index="decision_ts", columns="leg", values="leg_size", aggfunc="first"
        ).reindex(columns=["short", "long"])
        spread_series = (leg_returns["long"] - leg_returns["short"]).dropna().astype(float)
    spread_mean = float(spread_series.mean()
                        ) if not spread_series.empty else float("nan")
    spread_std = float(spread_series.std(ddof=1)) if len(
        spread_series) > 1 else float("nan")
    short_mean = float(leg_returns["short"].mean()
                       ) if not leg_returns.empty else float("nan")
    long_mean = float(leg_returns["long"].mean()
                      ) if not leg_returns.empty else float("nan")
    return {
        "panel_frequency": panel_frequency,
        "return_horizon": horizon,
        "feature_name": feature_name,
        "train_days": walk_forward_spec["train_days"],
        "test_days": walk_forward_spec["test_days"],
        "embargo_days": walk_forward_spec["embargo_days"],
        "step_days": walk_forward_spec["step_days"],
        "n_folds": _fold_count(detail_frame),
        "leg_count": int(leg_count),
        "spread_observation_count": int(len(spread_series)),
        "test_decision_count": test_decision_count,
        "scored_decision_count": scored_decision_count,
        "skipped_decision_count": skipped_decision_count,
        "skipped_decision_share": float(skipped_decision_count / test_decision_count) if test_decision_count else float("nan"),
        "skipped_small_cross_section_count": int(status_counts.get("small_cross_section", 0)),
        "skipped_constant_feature_count": int(status_counts.get("constant_feature", 0)),
        "short_leg_mean_return": short_mean,
        "long_leg_mean_return": long_mean,
        "short_leg_avg_size": float(leg_sizes["short"].mean()) if not leg_sizes.empty else float("nan"),
        "long_leg_avg_size": float(leg_sizes["long"].mean()) if not leg_sizes.empty else float("nan"),
        "spread_mean_return": spread_mean,
        "spread_annualized_return": float(spread_mean * periods_per_year(return_frequency, frequency_periods_per_year)) if np.isfinite(spread_mean) else float("nan"),
        "spread_std_return": spread_std,
        "spread_t_stat": research_stats.simple_t_stat(spread_series),
        "spread_sharpe": annualized_sharpe_for_frequency(spread_series, return_frequency, frequency_periods_per_year),
        "spread_positive_share": float((spread_series > 0).mean()) if not spread_series.empty else float("nan"),
    }


def fama_macbeth_diagnostics_for_frame(
    frame: pd.DataFrame,
    feature_name: str,
    direction: int,
    control_columns: Sequence[str],
    min_cross_section: int,
) -> dict[str, object]:
    control_columns = tuple(control_columns)
    valid = frame[[feature_name, "forward_return",
                   *control_columns]].dropna().copy()
    cross_section_size = len(valid)
    decision_ts = valid.index[0] if cross_section_size else frame.index[0]
    row: dict[str, object] = {"decision_ts": decision_ts,
                              "cross_section_size": cross_section_size, "status": "ok"}
    if cross_section_size < min_cross_section:
        row["status"] = "small_cross_section"
        return row
    if valid[feature_name].nunique(dropna=True) <= 1:
        row["status"] = "constant_signal"
        return row
    if valid["forward_return"].nunique(dropna=True) <= 1:
        row["status"] = "constant_return"
        return row
    for control_name in control_columns:
        if valid[control_name].nunique(dropna=True) <= 1:
            row["status"] = "constant_control"
            row["constant_control_name"] = control_name
            return row
    signal = valid[feature_name].astype(float) * direction
    design = np.column_stack(
        [np.ones(cross_section_size), signal.to_numpy(
        ), *[valid[column].astype(float).to_numpy() for column in control_columns]]
    )
    returns = valid["forward_return"].astype(float).to_numpy()
    coefficients, _, rank, _ = np.linalg.lstsq(design, returns, rcond=None)
    if rank < design.shape[1]:
        row["status"] = "singular_design"
        return row
    fitted = design @ coefficients
    total_sum_squares = float(np.square(returns - returns.mean()).sum())
    residual_sum_squares = float(np.square(returns - fitted).sum())
    row["gamma_intercept"] = float(coefficients[0])
    row["gamma_signal"] = float(coefficients[1])
    for column, coefficient in zip(control_columns, coefficients[2:]):
        row[f"gamma_{column}"] = float(coefficient)
    row["r_squared"] = float(1.0 - residual_sum_squares /
                             total_sum_squares) if total_sum_squares > 0.0 else float("nan")
    return row


def fama_macbeth_diagnostics_for_frame_slice(
    frame: pd.DataFrame,
    feature_name: str,
    direction: int,
    control_columns: Sequence[str],
    min_cross_section: int,
) -> list[dict[str, object]]:
    control_columns = tuple(control_columns)
    decision_index = decision_index_for_frame(frame)
    if decision_index.empty:
        return []
    decision_values = frame.index.get_level_values(0) if isinstance(
        frame.index, pd.MultiIndex) else frame.index
    if "symbol" in frame.columns:
        symbol_values = frame["symbol"].astype(str).to_numpy()
    elif isinstance(frame.index, pd.MultiIndex) and "symbol" in frame.index.names:
        symbol_values = frame.index.get_level_values("symbol").astype(str).to_numpy()
    elif isinstance(frame.index, pd.MultiIndex) and frame.index.nlevels > 1:
        symbol_values = frame.index.get_level_values(1).astype(str).to_numpy()
    else:
        symbol_values = pd.RangeIndex(len(frame)).astype(str).to_numpy()
    working = pd.DataFrame(
        {
            "decision_ts": decision_values,
            "symbol": symbol_values,
            feature_name: frame[feature_name].to_numpy(dtype=float, copy=False),
            "forward_return": frame["forward_return"].to_numpy(dtype=float, copy=False),
            **{column: frame[column].to_numpy(dtype=float, copy=False) for column in control_columns},
        }
    )
    valid = working.dropna(
        subset=[feature_name, "forward_return", *control_columns])
    if valid.empty:
        return [{"decision_ts": decision_ts, "cross_section_size": 0, "status": "small_cross_section"} for decision_ts in decision_index]

    symbols = pd.Index(sorted(valid["symbol"].unique()))
    pivoted = {
        column: valid.pivot(index="decision_ts", columns="symbol", values=column).reindex(
            index=decision_index, columns=symbols)
        for column in (feature_name, "forward_return", *control_columns)
    }
    all_valid = pd.DataFrame(True, index=decision_index, columns=symbols)
    for matrix in pivoted.values():
        all_valid &= matrix.notna()
    counts = all_valid.sum(axis=1).astype(int)
    feature_unique = pivoted[feature_name].where(
        all_valid).nunique(axis=1, dropna=True).astype(int)
    return_unique = pivoted["forward_return"].where(
        all_valid).nunique(axis=1, dropna=True).astype(int)
    control_unique = {
        control_name: pivoted[control_name].where(
            all_valid).nunique(axis=1, dropna=True).astype(int)
        for control_name in control_columns
    }

    diagnostics_by_decision: dict[object, dict[str, object]] = {}
    full_ok_decisions: list[object] = []
    for decision_ts in decision_index:
        cross_section_size = int(counts.loc[decision_ts])
        row: dict[str, object] = {"decision_ts": decision_ts,
                                  "cross_section_size": cross_section_size, "status": "ok"}
        if cross_section_size < min_cross_section:
            row["status"] = "small_cross_section"
        elif int(feature_unique.loc[decision_ts]) <= 1:
            row["status"] = "constant_signal"
        elif int(return_unique.loc[decision_ts]) <= 1:
            row["status"] = "constant_return"
        else:
            constant_control_name = next(
                (control_name for control_name in control_columns if int(
                    control_unique[control_name].loc[decision_ts]) <= 1),
                None,
            )
            if constant_control_name is not None:
                row["status"] = "constant_control"
                row["constant_control_name"] = constant_control_name
            else:
                full_ok_decisions.append(decision_ts)
        diagnostics_by_decision[decision_ts] = row

    if full_ok_decisions:
        ok_index = pd.Index(full_ok_decisions)
        complete_mask = all_valid.loc[ok_index].all(axis=1)
        batch_index = ok_index[complete_mask.to_numpy()]
        fallback_index = ok_index[~complete_mask.to_numpy()]

        if len(batch_index) > 0:
            signal_matrix = pivoted[feature_name].loc[batch_index].to_numpy(dtype=float) * direction
            returns_matrix = pivoted["forward_return"].loc[batch_index].to_numpy(dtype=float)
            controls_matrices = [
                pivoted[column].loc[batch_index].to_numpy(dtype=float)
                for column in control_columns
            ]
            ones = np.ones_like(signal_matrix)
            design = np.stack([ones, signal_matrix, *controls_matrices], axis=2)
            xtx = np.einsum("tnp,tnq->tpq", design, design)
            xty = np.einsum("tnp,tn->tp", design, returns_matrix)
            ranks = np.linalg.matrix_rank(xtx)
            nonsingular = ranks == xtx.shape[1]
            coefficients = np.full((len(batch_index), xtx.shape[1]), np.nan)
            if nonsingular.any():
                coefficients[nonsingular] = np.linalg.solve(
                    xtx[nonsingular],
                    xty[nonsingular, :, np.newaxis],
                )[:, :, 0]
            fitted = np.einsum("tnp,tp->tn", design, coefficients)
            total_sum_squares = np.square(
                returns_matrix - returns_matrix.mean(axis=1, keepdims=True)).sum(axis=1)
            residual_sum_squares = np.square(returns_matrix - fitted).sum(axis=1)
            r_squared = np.where(
                total_sum_squares > 0.0,
                1.0 - residual_sum_squares / total_sum_squares,
                np.nan,
            )
            for idx, decision_ts in enumerate(batch_index):
                row = diagnostics_by_decision[decision_ts]
                if not nonsingular[idx]:
                    row["status"] = "singular_design"
                    continue
                row["gamma_intercept"] = float(coefficients[idx, 0])
                row["gamma_signal"] = float(coefficients[idx, 1])
                for column, coefficient in zip(control_columns, coefficients[idx, 2:]):
                    row[f"gamma_{column}"] = float(coefficient)
                row["r_squared"] = float(r_squared[idx])

        if len(fallback_index) > 0:
            scored = valid[valid["decision_ts"].isin(fallback_index)]
            for decision_ts, group in scored.groupby("decision_ts", sort=False):
                row = diagnostics_by_decision[decision_ts]
                group = group.dropna(
                    subset=[feature_name, "forward_return", *control_columns])
                cross_section_size = len(group)
                signal = group[feature_name].astype(float) * direction
                design = np.column_stack(
                    [np.ones(cross_section_size), signal.to_numpy(
                    ), *[group[column].astype(float).to_numpy() for column in control_columns]]
                )
                returns = group["forward_return"].astype(float).to_numpy()
                coefficients, _, rank, _ = np.linalg.lstsq(design, returns, rcond=None)
                if rank < design.shape[1]:
                    row["status"] = "singular_design"
                    continue
                fitted = design @ coefficients
                total_sum_squares = float(np.square(returns - returns.mean()).sum())
                residual_sum_squares = float(np.square(returns - fitted).sum())
                row["gamma_intercept"] = float(coefficients[0])
                row["gamma_signal"] = float(coefficients[1])
                for column, coefficient in zip(control_columns, coefficients[2:]):
                    row[f"gamma_{column}"] = float(coefficient)
                row["r_squared"] = float(1.0 - residual_sum_squares /
                                         total_sum_squares) if total_sum_squares > 0.0 else float("nan")

    return [diagnostics_by_decision[decision_ts] for decision_ts in decision_index]


def summarize_fama_macbeth(
    panel_frequency: str,
    horizon: str,
    feature_name: str,
    detail_frame: pd.DataFrame,
    walk_forward_spec: Mapping[str, int],
    diagnostics: list[dict[str, object]],
    control_columns: Sequence[str],
    hac_overlap_lags: int = 0,
) -> dict[str, object]:
    gamma_series = detail_frame["gamma_signal"].astype(
        float) if not detail_frame.empty else pd.Series(dtype=float)
    observation_count = len(gamma_series)
    diagnostic_frame = pd.DataFrame(diagnostics)
    test_decision_count = len(diagnostic_frame)
    status_counts = diagnostic_frame["status"].value_counts(
    ) if test_decision_count else pd.Series(dtype=int)
    scored_decision_count = int(status_counts.get("ok", 0))
    skipped_decision_count = test_decision_count - scored_decision_count
    skipped_decision_share = float(
        skipped_decision_count / test_decision_count) if test_decision_count else float("nan")
    hac_lags = research_stats.newey_west_max_lags(
        observation_count, overlap_lags=hac_overlap_lags)
    summary = {
        "panel_frequency": panel_frequency,
        "return_horizon": horizon,
        "feature_name": feature_name,
        "train_days": walk_forward_spec["train_days"],
        "test_days": walk_forward_spec["test_days"],
        "embargo_days": walk_forward_spec["embargo_days"],
        "step_days": walk_forward_spec["step_days"],
        "n_folds": _fold_count(detail_frame),
        "gamma_observation_count": observation_count,
        "test_decision_count": test_decision_count,
        "scored_decision_count": scored_decision_count,
        "skipped_decision_count": skipped_decision_count,
        "skipped_decision_share": skipped_decision_share,
        "skipped_small_cross_section_count": int(status_counts.get("small_cross_section", 0)),
        "skipped_constant_signal_count": int(status_counts.get("constant_signal", 0)),
        "skipped_constant_return_count": int(status_counts.get("constant_return", 0)),
        "skipped_constant_control_count": int(status_counts.get("constant_control", 0)),
        "skipped_singular_design_count": int(status_counts.get("singular_design", 0)),
        "mean_gamma": float(gamma_series.mean()) if observation_count else float("nan"),
        "std_gamma": float(gamma_series.std(ddof=1)) if observation_count > 1 else float("nan"),
        "gamma_positive_share": float((gamma_series > 0).mean()) if observation_count else float("nan"),
        "hac_t_stat": research_stats.hac_t_stat(gamma_series, max_lags=hac_lags),
        "hac_lags": hac_lags,
        "mean_r_squared": float(detail_frame["r_squared"].mean()) if not detail_frame.empty else float("nan"),
        "mean_intercept": float(detail_frame["gamma_intercept"].mean()) if not detail_frame.empty else float("nan"),
    }
    for column in control_columns:
        detail_column = f"gamma_{column}"
        summary[f"mean_{column}_gamma"] = float(detail_frame[detail_column].mean(
        )) if not detail_frame.empty and detail_column in detail_frame.columns else float("nan")
    return summary


def weight_schemes_for_features(feature_names: Sequence[str], weight_schemes: Sequence[str]) -> tuple[str, ...]:
    if len(tuple(feature_names)) <= 1:
        return ("equal",)
    return tuple(weight_schemes)


def combo_catalog(
    default_mainline_track: Mapping[str, object],
    side_track_specs: Sequence[tuple[str, str, str, tuple[str, ...]]],
    weight_schemes: Sequence[str],
    mainline_override: MainlineCatalogOverride | None = None,
) -> list[ComboSpec]:
    specs: list[ComboSpec] = []
    mainline_track = default_mainline_track if mainline_override is None else {
        "track": mainline_override.track,
        "panel_frequency": mainline_override.panel_frequency,
        "return_horizon": mainline_override.return_horizon,
        "feature_names": mainline_override.feature_names,
    }
    mainline_features = tuple(mainline_track["feature_names"])
    for size in range(1, len(mainline_features) + 1):
        for feature_subset in combinations(mainline_features, size):
            for weight_scheme in weight_schemes_for_features(feature_subset, weight_schemes):
                specs.append(
                    ComboSpec(
                        combo_id=f"{mainline_track['panel_frequency']}_{mainline_track['return_horizon']}__{'__'.join(feature_subset)}",
                        track=str(mainline_track["track"]),
                        panel_frequency=str(mainline_track["panel_frequency"]),
                        return_horizon=str(mainline_track["return_horizon"]),
                        feature_names=tuple(feature_subset),
                        weight_scheme=weight_scheme,
                    )
                )
    for track, panel_frequency, return_horizon, feature_names in side_track_specs:
        for weight_scheme in weight_schemes_for_features(feature_names, weight_schemes):
            specs.append(
                ComboSpec(
                    combo_id=f"{panel_frequency}_{return_horizon}__{'__'.join(feature_names)}",
                    track=track,
                    panel_frequency=panel_frequency,
                    return_horizon=return_horizon,
                    feature_names=feature_names,
                    weight_scheme=weight_scheme,
                )
            )
    return specs


def _sorted_gate_candidates(
    frame: pd.DataFrame,
    strict_support_column: str | None,
    rank_columns: Sequence[str],
) -> pd.DataFrame:
    sort_columns: list[str] = []
    ascending: list[bool] = []
    if strict_support_column is not None and strict_support_column in frame.columns:
        sort_columns.append(strict_support_column)
        ascending.append(False)
    for column in rank_columns:
        if column not in frame.columns:
            raise ValueError("gate_summary missing rank column: " + column)
        sort_columns.append(column)
        ascending.append(False)
    sort_columns.append("feature_name")
    ascending.append(True)
    sorted_frame = frame.copy()
    for column in sort_columns:
        if column != "feature_name":
            sorted_frame[column] = pd.to_numeric(sorted_frame[column], errors="coerce").fillna(-np.inf)
    return sorted_frame.sort_values(sort_columns, ascending=ascending, kind="mergesort")


def candidate_combo_specs_from_gate_summary(
    gate_summary: pd.DataFrame,
    registry_frame: pd.DataFrame,
    *,
    panel_frequency: str,
    support_column: str = "two_gate_support",
    strict_support_column: str | None = "three_gate_support",
    rank_columns: Sequence[str] = ("ic_hac_t_stat", "bucket_spread_mean_return", "fm_hac_t_stat", "icir"),
    top_k_values: Sequence[int] = (3, 5, 8),
    family_combo_sizes: Sequence[int] = (2, 3),
    weight_schemes: Sequence[str] = ("equal", "icir"),
    horizon_deltas: Mapping[str, pd.Timedelta] | None = None,
    supported_signal_timeframes: Sequence[str] | None = None,
) -> tuple[list[ComboSpec], pd.DataFrame]:
    """Build finite, same-horizon combo specs from single-factor gate output.

    The helper intentionally generates a bounded candidate set: same return
    horizon only, one best feature per family for family combinations, and a
    small top-K ladder. It does not inspect downstream strategy performance.
    """
    required_gate_columns = {"feature_name", "return_horizon", support_column, *rank_columns}
    if strict_support_column is not None:
        required_gate_columns.add(strict_support_column)
    missing_gate = sorted(required_gate_columns.difference(gate_summary.columns))
    if missing_gate:
        raise ValueError("gate_summary missing columns: " + ", ".join(missing_gate))
    missing_registry = sorted({"feature_name", "family"}.difference(registry_frame.columns))
    if missing_registry:
        raise ValueError("registry_frame missing columns: " + ", ".join(missing_registry))
    if not weight_schemes:
        raise ValueError("weight_schemes must not be empty")
    for scheme in weight_schemes:
        if scheme not in {
            "equal",
            "ic_abs",
            "icir",
            "hac_t",
            "family_equal",
            "family_alpha_0",
            "family_alpha_0p5",
            "family_alpha_1",
            "corr_discount_icir",
        }:
            raise ValueError("unsupported weight scheme: " + scheme)
    if (horizon_deltas is None) != (supported_signal_timeframes is None):
        raise ValueError("horizon_deltas and supported_signal_timeframes must be provided together")

    registry = registry_frame[["feature_name", "family"]].drop_duplicates("feature_name")
    working = gate_summary.merge(registry, on="feature_name", how="left", validate="many_to_one")
    if working["family"].isna().any():
        missing_features = sorted(working.loc[working["family"].isna(), "feature_name"].astype(str).unique())
        raise ValueError("registry_frame missing family for features: " + ", ".join(missing_features))
    working = working.loc[working[support_column].astype(bool)].copy()
    if horizon_deltas is not None and supported_signal_timeframes is not None:
        def eligible(feature_name: object, return_horizon: object) -> bool:
            timeframe = feature_signal_timeframe(
                str(feature_name), panel_frequency, supported_signal_timeframes
            )
            signal_delta = pd.Timedelta(horizon_deltas[timeframe])
            return_delta = pd.Timedelta(horizon_deltas[str(return_horizon)])
            return signal_delta <= return_delta and return_delta % signal_delta == pd.Timedelta(0)

        working["time_contract_eligible"] = [
            eligible(feature_name, return_horizon)
            for feature_name, return_horizon in zip(
                working["feature_name"], working["return_horizon"]
            )
        ]
        working = working.loc[working["time_contract_eligible"]].copy()
    if working.empty:
        raise ValueError("no time-contract-eligible supported gate candidates available for combo specs")

    specs: list[ComboSpec] = []
    catalog_rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, tuple[str, ...], str]] = set()

    def add_specs(
        horizon: str,
        track: str,
        label: str,
        feature_names: Sequence[str],
    ) -> None:
        feature_tuple = tuple(dict.fromkeys(str(feature) for feature in feature_names))
        if len(feature_tuple) < 2:
            return
        spec_panel_frequency = (
            non_overlapping_decision_frequency(
                feature_tuple,
                panel_frequency,
                horizon,
                horizon_deltas,
                supported_signal_timeframes,
            )
            if horizon_deltas is not None and supported_signal_timeframes is not None
            else panel_frequency
        )
        for weight_scheme in weight_schemes_for_features(feature_tuple, weight_schemes):
            key = (horizon, track, feature_tuple, weight_scheme)
            if key in seen:
                continue
            seen.add(key)
            combo_id = f"{spec_panel_frequency}_{horizon}__{track}__{label}__{'__'.join(feature_tuple)}"
            specs.append(
                ComboSpec(
                    combo_id=combo_id,
                    track=track,
                    panel_frequency=spec_panel_frequency,
                    return_horizon=horizon,
                    feature_names=feature_tuple,
                    weight_scheme=weight_scheme,
                )
            )
            catalog_rows.append(
                {
                    "combo_id": combo_id,
                    "track": track,
                    "selection_label": label,
                    "base_panel_frequency": panel_frequency,
                    "panel_frequency": spec_panel_frequency,
                    "return_horizon": horizon,
                    "component_features": feature_list_text(feature_tuple),
                    "n_components": len(feature_tuple),
                    "weight_scheme": weight_scheme,
                }
            )

    for horizon, horizon_frame in working.groupby("return_horizon", sort=True):
        sorted_pool = _sorted_gate_candidates(horizon_frame, strict_support_column, rank_columns)
        family_best = (
            sorted_pool.groupby("family", sort=False, as_index=False).head(1)
            .reset_index(drop=True)
        )
        family_features = tuple(family_best["feature_name"].astype(str))
        add_specs(str(horizon), "two_gate_family_best_all", "family_best_all", family_features)
        for size in family_combo_sizes:
            if int(size) < 2:
                raise ValueError("family_combo_sizes must be >= 2")
            for idx, subset in enumerate(combinations(family_features, int(size)), start=1):
                add_specs(str(horizon), f"two_gate_family_{int(size)}", f"family_{int(size)}_{idx:03d}", subset)
        for top_k in top_k_values:
            if int(top_k) < 2:
                raise ValueError("top_k_values must be >= 2")
            top_features = tuple(sorted_pool["feature_name"].astype(str).head(int(top_k)))
            add_specs(str(horizon), f"two_gate_top{int(top_k)}", f"top{int(top_k)}", top_features)

        if strict_support_column is not None:
            strict_pool = sorted_pool.loc[sorted_pool[strict_support_column].astype(bool)].copy()
            if not strict_pool.empty:
                strict_features = tuple(strict_pool["feature_name"].astype(str))
                add_specs(str(horizon), "three_gate_all", "all", strict_features)
                strict_family_best = (
                    strict_pool.groupby("family", sort=False, as_index=False).head(1)
                    .reset_index(drop=True)
                )
                add_specs(
                    str(horizon),
                    "three_gate_family_best_all",
                    "family_best_all",
                    tuple(strict_family_best["feature_name"].astype(str)),
                )

    if not specs:
        raise ValueError("combo spec generation produced no multi-feature specs")
    return specs, pd.DataFrame(catalog_rows)


def candidate_structured_combo_specs_from_gate_summary(
    gate_summary: pd.DataFrame,
    registry_frame: pd.DataFrame,
    *,
    panel_frequency: str,
    support_column: str = "two_gate_support",
    strict_support_column: str | None = "three_gate_support",
    weight_schemes: Sequence[str] = ("equal", "icir", "corr_discount_icir"),
    horizon_deltas: Mapping[str, pd.Timedelta] | None = None,
    supported_signal_timeframes: Sequence[str] | None = None,
) -> tuple[list[ComboSpec], pd.DataFrame]:
    """Build structured same-horizon combo specs from single-factor gate output.

    This entry intentionally avoids top-K and arbitrary 2/3-factor enumeration.
    For each return horizon it builds all-two-gate, strict three-gate, family,
    and leave-one-family-out baskets. Correlation-aware weighting is handled by
    the evaluator, not by changing membership.
    """
    required_gate_columns = {"feature_name", "return_horizon", support_column}
    if strict_support_column is not None:
        required_gate_columns.add(strict_support_column)
    missing_gate = sorted(required_gate_columns.difference(gate_summary.columns))
    if missing_gate:
        raise ValueError("gate_summary missing columns: " + ", ".join(missing_gate))
    missing_registry = sorted({"feature_name", "family"}.difference(registry_frame.columns))
    if missing_registry:
        raise ValueError("registry_frame missing columns: " + ", ".join(missing_registry))
    if not weight_schemes:
        raise ValueError("weight_schemes must not be empty")
    for scheme in weight_schemes:
        if scheme not in {
            "equal",
            "icir",
            "corr_discount_icir",
            "family_alpha_0",
            "family_alpha_0p5",
            "family_alpha_1",
        }:
            raise ValueError("unsupported structured weight scheme: " + scheme)
    if (horizon_deltas is None) != (supported_signal_timeframes is None):
        raise ValueError("horizon_deltas and supported_signal_timeframes must be provided together")

    registry = registry_frame[["feature_name", "family"]].drop_duplicates("feature_name")
    working = gate_summary.merge(registry, on="feature_name", how="left", validate="many_to_one")
    if working["family"].isna().any():
        missing_features = sorted(working.loc[working["family"].isna(), "feature_name"].astype(str).unique())
        raise ValueError("registry_frame missing family for features: " + ", ".join(missing_features))
    working = working.loc[working[support_column].astype(bool)].copy()
    if working.empty:
        raise ValueError("no supported gate candidates available for structured combo specs")
    working["feature_name"] = working["feature_name"].astype(str)
    working["family"] = working["family"].astype(str)

    specs: list[ComboSpec] = []
    catalog_rows: list[dict[str, object]] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()

    def ordered_features(frame: pd.DataFrame) -> tuple[str, ...]:
        ordered = frame[["family", "feature_name"]].drop_duplicates("feature_name")
        ordered = ordered.sort_values(["family", "feature_name"], kind="mergesort")
        return tuple(ordered["feature_name"].astype(str))

    def add_specs(
        horizon: str,
        track: str,
        label: str,
        feature_names: Sequence[str],
    ) -> None:
        feature_tuple = tuple(dict.fromkeys(str(feature) for feature in feature_names))
        if len(feature_tuple) < 2:
            return
        spec_panel_frequency = (
            non_overlapping_decision_frequency(
                feature_tuple,
                panel_frequency,
                horizon,
                horizon_deltas,
                supported_signal_timeframes,
            )
            if horizon_deltas is not None and supported_signal_timeframes is not None
            else panel_frequency
        )
        for weight_scheme in weight_schemes_for_features(feature_tuple, weight_schemes):
            key = (horizon, feature_tuple, weight_scheme)
            if key in seen:
                continue
            seen.add(key)
            combo_id = f"{spec_panel_frequency}_{horizon}__{track}__{label}__{'__'.join(feature_tuple)}"
            specs.append(
                ComboSpec(
                    combo_id=combo_id,
                    track=track,
                    panel_frequency=spec_panel_frequency,
                    return_horizon=horizon,
                    feature_names=feature_tuple,
                    weight_scheme=weight_scheme,
                )
            )
            catalog_rows.append(
                {
                    "combo_id": combo_id,
                    "track": track,
                    "selection_label": label,
                    "base_panel_frequency": panel_frequency,
                    "panel_frequency": spec_panel_frequency,
                    "return_horizon": horizon,
                    "component_features": feature_list_text(feature_tuple),
                    "n_components": len(feature_tuple),
                    "weight_scheme": weight_scheme,
                }
            )

    for horizon, horizon_frame in working.groupby("return_horizon", sort=True):
        horizon = str(horizon)
        horizon_features = ordered_features(horizon_frame)
        add_specs(horizon, "all_two_gate", "all_two_gate", horizon_features)

        if strict_support_column is not None:
            strict_pool = horizon_frame.loc[horizon_frame[strict_support_column].astype(bool)].copy()
            if strict_pool["feature_name"].nunique() >= 2:
                add_specs(horizon, "three_gate_all", "three_gate_all", ordered_features(strict_pool))

        family_order = sorted(horizon_frame["family"].dropna().astype(str).unique())
        for family_name in family_order:
            family_frame = horizon_frame.loc[horizon_frame["family"].astype(str) == family_name]
            if family_frame["feature_name"].nunique() >= 2:
                label = f"family_{family_name}"
                add_specs(horizon, label, label, ordered_features(family_frame))

        for family_name in family_order:
            remaining = horizon_frame.loc[horizon_frame["family"].astype(str) != family_name]
            if remaining["feature_name"].nunique() >= 2:
                label = f"without_{family_name}"
                add_specs(horizon, label, label, ordered_features(remaining))

    if not specs:
        raise ValueError("structured combo spec generation produced no multi-feature specs")
    return specs, pd.DataFrame(catalog_rows)


def feature_list_text(feature_names: Sequence[str]) -> str:
    return " | ".join(feature_names)


def parse_csv_arg(raw: str | None) -> set[str] | None:
    if raw is None or not raw.strip():
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def parse_weight_schemes_arg(raw: str | None, weight_schemes: Sequence[str]) -> set[str] | None:
    schemes = parse_csv_arg(raw)
    if schemes is None:
        return None
    invalid = schemes.difference(weight_schemes)
    if invalid:
        raise ValueError("unsupported weight schemes: " +
                         ", ".join(sorted(invalid)))
    return schemes


def parse_required_csv_arg(raw: str | None, argument_name: str) -> tuple[str, ...] | None:
    if raw is None:
        return None
    tokens = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not tokens:
        raise ValueError(f"{argument_name} must not be empty when provided")
    return tokens


def parse_mainline_override(
    track: str | None,
    panel_frequency: str | None,
    return_horizon: str | None,
    feature_names: tuple[str, ...] | None,
) -> MainlineCatalogOverride | None:
    provided = [
        track is not None and track.strip() != "",
        panel_frequency is not None and panel_frequency.strip() != "",
        return_horizon is not None and return_horizon.strip() != "",
        feature_names is not None,
    ]
    if not any(provided):
        return None
    if not all(provided):
        raise ValueError(
            "mainline override requires track, panel_frequency, return_horizon, and feature_names together")
    return MainlineCatalogOverride(
        track=str(track).strip(),
        panel_frequency=str(panel_frequency).strip(),
        return_horizon=str(return_horizon).strip(),
        feature_names=feature_names,
    )


def selected_specs(
    combo_ids: set[str] | None,
    tracks: set[str] | None,
    weight_schemes: set[str] | None,
    default_mainline_track: Mapping[str, object],
    side_track_specs: Sequence[tuple[str, str, str, tuple[str, ...]]],
    all_weight_schemes: Sequence[str],
    mainline_override: MainlineCatalogOverride | None = None,
) -> list[ComboSpec]:
    specs = combo_catalog(
        default_mainline_track=default_mainline_track,
        side_track_specs=side_track_specs,
        weight_schemes=all_weight_schemes,
        mainline_override=mainline_override,
    )
    if combo_ids is not None:
        specs = [spec for spec in specs if spec.combo_id in combo_ids]
    if tracks is not None:
        specs = [spec for spec in specs if spec.track in tracks]
    if weight_schemes is not None:
        specs = [spec for spec in specs if spec.weight_scheme in weight_schemes]
    if not specs:
        raise ValueError("no combo specs selected")
    return specs


def build_composite_frame(
    frame: pd.DataFrame,
    feature_names: tuple[str, ...],
    directions: Mapping[str, int],
    feature_weights: Mapping[str, float],
    extra_columns: tuple[str, ...],
) -> pd.DataFrame:
    result = frame[["symbol", *extra_columns]].copy()
    aligned = pd.concat(
        [(frame[feature_name].astype(float) * directions[feature_name]
          ).rename(feature_name) for feature_name in feature_names],
        axis=1,
    )
    weight_series = pd.Series(
        feature_weights, dtype=float).reindex(feature_names)
    total_weight = float(weight_series.sum())
    weighted_signal = aligned.mul(weight_series, axis=1).sum(
        axis=1, min_count=len(feature_names))
    complete_rows = aligned.notna().all(axis=1)
    result["composite_signal"] = weighted_signal.div(
        total_weight if total_weight > 0.0 else np.nan).where(complete_rows)
    return result


def train_feature_stats(
    train_slice: pd.DataFrame,
    feature_names: tuple[str, ...],
    min_cross_section: int,
) -> dict[str, FeatureTrainStat] | None:
    stats: dict[str, FeatureTrainStat] = {}
    diagnostics_by_feature = rank_ic_diagnostics_for_features(
        train_slice[[*feature_names, "forward_return"]],
        feature_names,
        min_cross_section,
    )
    for feature_name in feature_names:
        feature_rows = [
            {
                "decision_ts": row["decision_ts"],
                "raw_rank_ic": row["raw_rank_ic"],
                "cross_section_size": row["cross_section_size"],
            }
            for row in diagnostics_by_feature[feature_name]
            if row["status"] == "ok"
        ]
        if not feature_rows:
            return None
        ic_series = pd.DataFrame(feature_rows)["raw_rank_ic"].astype(float)
        observation_count = len(ic_series)
        mean_ic = float(ic_series.mean())
        std_ic = float(ic_series.std(ddof=1)
                       ) if observation_count > 1 else float("nan")
        icir = float(mean_ic / std_ic) if observation_count > 1 and np.isfinite(
            std_ic) and std_ic > 0.0 else float("nan")
        stats[feature_name] = FeatureTrainStat(
            direction=1 if mean_ic >= 0.0 else -1,
            mean_ic=mean_ic,
            std_ic=std_ic,
            icir=icir,
            hac_t_stat=research_stats.hac_t_stat(
                ic_series, max_lags=research_stats.newey_west_max_lags(observation_count)),
            observation_count=observation_count,
        )
    return stats


def weight_score_for_feature(stat: FeatureTrainStat, weight_scheme: str) -> float:
    if weight_scheme == "equal":
        return 1.0
    if weight_scheme == "ic_abs":
        value = abs(stat.mean_ic)
    elif weight_scheme == "icir":
        value = abs(stat.icir)
    elif weight_scheme == "hac_t":
        value = abs(stat.hac_t_stat)
    else:
        raise ValueError("unsupported weight scheme: " + weight_scheme)
    return float(value) if np.isfinite(value) and value > 0.0 else 0.0


def normalized_feature_weights(
    train_stats: Mapping[str, FeatureTrainStat],
    weight_scheme: str,
) -> tuple[dict[str, float], dict[str, float]]:
    scores = {feature_name: weight_score_for_feature(
        stat, weight_scheme) for feature_name, stat in train_stats.items()}
    total_score = float(sum(scores.values()))
    if not np.isfinite(total_score) or total_score <= 0.0:
        equal_weight = 1.0 / len(train_stats)
        return {feature_name: 1.0 for feature_name in train_stats}, {feature_name: equal_weight for feature_name in train_stats}
    return scores, {feature_name: float(score / total_score) for feature_name, score in scores.items()}


def _decision_symbol_feature_frame(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    missing = [feature_name for feature_name in feature_names if feature_name not in frame.columns]
    if missing:
        raise ValueError("missing feature columns: " + ", ".join(missing))
    if isinstance(frame.index, pd.MultiIndex):
        decision_values = frame.index.get_level_values(0)
        if "symbol" in frame.columns:
            symbol_values = frame["symbol"].astype(str).to_numpy(dtype=object, copy=False)
        elif "symbol" in frame.index.names:
            symbol_values = frame.index.get_level_values("symbol").astype(str).to_numpy(dtype=object, copy=False)
        elif frame.index.nlevels > 1:
            symbol_values = frame.index.get_level_values(1).astype(str).to_numpy(dtype=object, copy=False)
        else:
            symbol_values = pd.RangeIndex(len(frame)).astype(str).to_numpy(dtype=object, copy=False)
    else:
        decision_values = frame.index
        symbol_values = (
            frame["symbol"].astype(str).to_numpy(dtype=object, copy=False)
            if "symbol" in frame.columns
            else pd.RangeIndex(len(frame)).astype(str).to_numpy(dtype=object, copy=False)
        )
    working = pd.DataFrame(
        {
            "decision_ts": decision_values,
            "symbol": symbol_values,
            **{feature_name: frame[feature_name].to_numpy(dtype=float, copy=False) for feature_name in feature_names},
        }
    )
    return working


def train_cross_sectional_feature_correlation(
    train_slice: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    min_cross_section: int,
    min_pair_corr_observations: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate train-only feature correlations from cross-sectional ranks.

    For each decision timestamp, the function computes pairwise Spearman
    correlations across symbols and then aggregates each pair by train-period
    median. It fails closed through diagnostics when pair support is too small;
    downstream weight functions reject missing correlations.
    """
    feature_names = tuple(str(feature_name) for feature_name in feature_names)
    if len(feature_names) < 2:
        raise ValueError("at least two feature_names are required")
    if int(min_cross_section) < 2:
        raise ValueError("min_cross_section must be at least 2")
    working = _decision_symbol_feature_frame(train_slice, feature_names)
    decision_index = pd.Index(working["decision_ts"].drop_duplicates())
    train_decision_count = int(len(decision_index))
    required_observations = (
        max(60, int(ceil(0.33 * train_decision_count)))
        if min_pair_corr_observations is None
        else int(min_pair_corr_observations)
    )
    if required_observations <= 0:
        raise ValueError("min_pair_corr_observations must be positive")

    corr_matrix = pd.DataFrame(np.eye(len(feature_names)), index=feature_names, columns=feature_names, dtype=float)
    pair_values: dict[tuple[str, str], list[float]] = {
        (left, right): [] for left, right in combinations(feature_names, 2)
    }
    skipped_small_by_pair = {pair: 0 for pair in pair_values}
    skipped_constant_by_pair = {pair: 0 for pair in pair_values}
    skipped_nan_by_pair = {pair: 0 for pair in pair_values}
    for _, group in working.groupby("decision_ts", sort=False):
        feature_frame = group[list(feature_names)]
        pair_counts = feature_frame.notna().astype(int).T.dot(feature_frame.notna().astype(int))
        nunique = feature_frame.nunique(dropna=True)
        decision_corr = feature_frame.corr(method="spearman", min_periods=int(min_cross_section))
        for left, right in pair_values:
            cross_section_size = int(pair_counts.loc[left, right])
            if cross_section_size < int(min_cross_section):
                skipped_small_by_pair[(left, right)] += 1
                continue
            if int(nunique[left]) <= 1 or int(nunique[right]) <= 1:
                skipped_constant_by_pair[(left, right)] += 1
                continue
            corr_value = decision_corr.loc[left, right]
            if not np.isfinite(corr_value):
                skipped_nan_by_pair[(left, right)] += 1
                continue
            pair_values[(left, right)].append(float(corr_value))

    diagnostics: list[dict[str, object]] = []
    for left, right in combinations(feature_names, 2):
        values = pair_values[(left, right)]
        observation_count = int(len(values))
        status = "ok" if observation_count >= required_observations else "insufficient_pair_observations"
        median_corr = float(np.median(values)) if status == "ok" else float("nan")
        corr_matrix.loc[left, right] = median_corr
        corr_matrix.loc[right, left] = median_corr
        diagnostics.append(
            {
                "feature_left": left,
                "feature_right": right,
                "train_decision_count": train_decision_count,
                "min_pair_corr_observations": required_observations,
                "pair_corr_observation_count": observation_count,
                "skipped_small_cross_section_count": skipped_small_by_pair[(left, right)],
                "skipped_constant_feature_count": skipped_constant_by_pair[(left, right)],
                "skipped_nan_corr_count": skipped_nan_by_pair[(left, right)],
                "median_spearman_corr": median_corr,
                "abs_median_spearman_corr": abs(median_corr) if np.isfinite(median_corr) else float("nan"),
                "status": status,
            }
        )
    return corr_matrix, pd.DataFrame(diagnostics)


def weight_concentration_diagnostics(
    feature_weights: Mapping[str, float],
    *,
    feature_families: Mapping[str, str] | None = None,
    active_weight_threshold: float = 0.05,
) -> dict[str, object]:
    if not feature_weights:
        raise ValueError("feature_weights must not be empty")
    weights = pd.Series(feature_weights, dtype=float)
    if weights.isna().any() or not np.isfinite(weights.to_numpy(dtype=float)).all():
        raise ValueError("feature_weights must be finite")
    if (weights < 0.0).any():
        raise ValueError("feature_weights must be non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("feature_weights must have positive total")
    weights = weights / total
    effective = float(1.0 / (weights * weights).sum())
    active_count = int((weights > float(active_weight_threshold)).sum())
    max_weight = float(weights.max())
    if feature_families is None:
        family_share_max = float("nan")
        effective_family_count = float("nan")
        max_family_weight = float("nan")
        family_count = 0
    else:
        family_names = [feature_families.get(feature_name, "other") for feature_name in weights.index]
        family_weights = weights.groupby(family_names).sum()
        family_share_max = float(family_weights.max())
        max_family_weight = family_share_max
        effective_family_count = float(1.0 / (family_weights * family_weights).sum())
        family_count = int(len(family_weights))
    return {
        "effective_factor_count": effective,
        "active_factor_count": active_count,
        "max_feature_weight": max_weight,
        "family_weight_share_max": family_share_max,
        "effective_family_count": effective_family_count,
        "max_family_weight": max_family_weight,
        "family_count": family_count,
        "near_single_or_few_factor": bool(active_count <= 2 or max_weight >= 0.67),
    }


def correlation_discount_weights(
    train_stats: Mapping[str, FeatureTrainStat],
    feature_correlations: pd.DataFrame,
    *,
    feature_families: Mapping[str, str] | None = None,
    icir_floor: float = 0.0,
    active_weight_threshold: float = 0.05,
) -> tuple[dict[str, float], dict[str, float], dict[str, object]]:
    feature_names = tuple(train_stats)
    if len(feature_names) < 2:
        raise ValueError("corr_discount_icir requires at least two features")
    if float(icir_floor) < 0.0:
        raise ValueError("icir_floor must be non-negative")
    missing = [feature_name for feature_name in feature_names if feature_name not in feature_correlations.index or feature_name not in feature_correlations.columns]
    if missing:
        raise ValueError("feature_correlations missing features: " + ", ".join(missing))
    corr = feature_correlations.loc[feature_names, feature_names].astype(float).abs()
    off_diag_mask = ~np.eye(len(feature_names), dtype=bool)
    off_diag_values = corr.to_numpy(dtype=float)[off_diag_mask]
    if np.isnan(off_diag_values).any():
        raise ValueError("feature_correlations must contain finite off-diagonal values")
    if not np.isfinite(off_diag_values).all():
        raise ValueError("feature_correlations must be finite")
    off_diag_array = corr.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(off_diag_array, 0.0)
    off_diag = pd.DataFrame(off_diag_array, index=feature_names, columns=feature_names)
    bases: dict[str, float] = {}
    discounts: dict[str, float] = {}
    raw_scores: dict[str, float] = {}
    for feature_name in feature_names:
        icir_value = abs(float(train_stats[feature_name].icir)) if np.isfinite(train_stats[feature_name].icir) else 0.0
        base = max(icir_value, float(icir_floor))
        redundancy = 1.0 + float(off_diag.loc[feature_name].sum())
        discount = 1.0 / redundancy
        bases[feature_name] = base
        discounts[feature_name] = discount
        raw_scores[feature_name] = base * discount
    total_score = float(sum(raw_scores.values()))
    if not np.isfinite(total_score) or total_score <= 0.0:
        raise ValueError("corr_discount_icir raw weights are not positive")
    weights = {feature_name: float(score / total_score) for feature_name, score in raw_scores.items()}
    diagnostics = weight_concentration_diagnostics(
        weights,
        feature_families=feature_families,
        active_weight_threshold=active_weight_threshold,
    )
    diagnostics.update(
        {
            "mean_abs_feature_corr": float(off_diag_values.mean()) if len(feature_names) > 1 else float("nan"),
            "max_abs_feature_corr": float(off_diag_values.max()) if len(feature_names) > 1 else float("nan"),
            "min_corr_discount": float(min(discounts.values())),
            "max_corr_discount": float(max(discounts.values())),
        }
    )
    return raw_scores, weights, diagnostics


def family_alpha_from_weight_scheme(weight_scheme: str) -> float | None:
    if weight_scheme == "family_equal" or weight_scheme == "family_alpha_0":
        return 0.0
    if weight_scheme == "family_alpha_0p5":
        return 0.5
    if weight_scheme == "family_alpha_1":
        return 1.0
    return None


def family_count_feature_weights(
    feature_names: Sequence[str],
    feature_families: Mapping[str, str],
    *,
    alpha: float,
) -> dict[str, float]:
    feature_names = tuple(feature_names)
    if not feature_names:
        raise ValueError("feature_names must not be empty")
    if float(alpha) not in {0.0, 0.5, 1.0}:
        raise ValueError("alpha must be one of 0.0, 0.5, 1.0")
    missing = [feature_name for feature_name in feature_names if feature_name not in feature_families]
    if missing:
        raise ValueError("feature_families missing features: " + ", ".join(missing))
    family_order = list(dict.fromkeys(str(feature_families[feature_name]) for feature_name in feature_names))
    family_members: dict[str, list[str]] = {
        family_name: [
            feature_name
            for feature_name in feature_names
            if str(feature_families[feature_name]) == family_name
        ]
        for family_name in family_order
    }
    family_scores = {
        family_name: float(len(members) ** float(alpha))
        for family_name, members in family_members.items()
    }
    total_family_score = float(sum(family_scores.values()))
    if not np.isfinite(total_family_score) or total_family_score <= 0.0:
        raise ValueError("family scores must have positive finite total")
    weights: dict[str, float] = {}
    for family_name in family_order:
        members = family_members[family_name]
        family_weight = family_scores[family_name] / total_family_score
        member_weight = family_weight / len(members)
        for feature_name in members:
            weights[feature_name] = float(member_weight)
    return weights


def family_equal_feature_weights(
    feature_names: Sequence[str],
    feature_families: Mapping[str, str],
) -> dict[str, float]:
    return family_count_feature_weights(
        feature_names,
        feature_families,
        alpha=0.0,
    )


def family_alpha_basket_comparison(
    summary_frame: pd.DataFrame,
    *,
    metric_column: str = "net_1x_sharpe",
    basket_columns: Sequence[str] = (
        "return_horizon",
        "track",
        "panel_frequency",
        "component_features",
        "n_components",
    ),
    weight_schemes: Sequence[str] = (
        "family_alpha_1",
        "family_alpha_0p5",
        "family_alpha_0",
    ),
    baseline_weight_scheme: str = "family_alpha_0",
) -> pd.DataFrame:
    if summary_frame.empty:
        return pd.DataFrame()
    if not weight_schemes:
        raise ValueError("weight_schemes must not be empty")
    if baseline_weight_scheme not in weight_schemes:
        raise ValueError("baseline_weight_scheme must be present in weight_schemes")
    required_columns = set(basket_columns).union({"weight_scheme", metric_column})
    missing = sorted(required_columns.difference(summary_frame.columns))
    if missing:
        raise ValueError("summary_frame missing columns: " + ", ".join(missing))
    selected = summary_frame.loc[
        summary_frame["weight_scheme"].astype(str).isin(tuple(weight_schemes)),
        [*basket_columns, "weight_scheme", metric_column],
    ].copy()
    if selected.empty:
        return pd.DataFrame()
    duplicated = selected.duplicated([*basket_columns, "weight_scheme"], keep=False)
    if duplicated.any():
        duplicate_keys = selected.loc[duplicated, [*basket_columns, "weight_scheme"]].drop_duplicates()
        raise ValueError(
            "summary_frame has duplicate basket/weight rows: "
            + duplicate_keys.head(5).to_dict(orient="records").__repr__()
        )
    pivot = selected.pivot_table(
        index=list(basket_columns),
        columns="weight_scheme",
        values=metric_column,
        aggfunc="first",
    )
    complete = pivot.dropna(subset=list(weight_schemes), how="any")
    if complete.empty:
        return pd.DataFrame(columns=[*basket_columns, *weight_schemes, "winner_weight_scheme", "winner_metric"])
    complete = complete.loc[:, list(weight_schemes)]
    result = complete.reset_index()
    metric_values = result[list(weight_schemes)].astype(float)
    result["winner_weight_scheme"] = metric_values.idxmax(axis=1)
    result["winner_metric"] = metric_values.max(axis=1)
    baseline = result[baseline_weight_scheme].astype(float)
    for scheme in weight_schemes:
        result[f"{scheme}_minus_{baseline_weight_scheme}"] = result[scheme].astype(float) - baseline
    return result


DEFAULT_FAMILY_ALPHA_L4_HORIZON_QUOTAS: dict[str, int] = {
    "12h": 4,
    "1d": 4,
    "8h": 4,
    "4h": 2,
}


def select_family_alpha_l4_targets(
    l3_summary: pd.DataFrame,
    *,
    horizon_quotas: Mapping[str, int] | None = None,
    metric_column: str = "net_1x_sharpe",
    basket_columns: Sequence[str] = (
        "return_horizon",
        "track",
        "panel_frequency",
        "component_features",
        "n_components",
    ),
    weight_schemes: Sequence[str] = (
        "family_alpha_1",
        "family_alpha_0p5",
        "family_alpha_0",
    ),
    baseline_weight_scheme: str = "family_alpha_0",
) -> pd.DataFrame:
    if l3_summary.empty:
        return pd.DataFrame()
    quotas = dict(horizon_quotas or DEFAULT_FAMILY_ALPHA_L4_HORIZON_QUOTAS)
    if not quotas:
        raise ValueError("horizon_quotas must not be empty")
    bad_quotas = {horizon: quota for horizon, quota in quotas.items() if int(quota) < 0}
    if bad_quotas:
        raise ValueError("horizon quotas must be non-negative: " + ", ".join(sorted(bad_quotas)))
    required_summary_columns = set(basket_columns).union({"combo_id", "weight_scheme", metric_column})
    missing_summary = sorted(required_summary_columns.difference(l3_summary.columns))
    if missing_summary:
        raise ValueError("l3_summary missing columns: " + ", ".join(missing_summary))
    comparison = family_alpha_basket_comparison(
        l3_summary,
        metric_column=metric_column,
        basket_columns=basket_columns,
        weight_schemes=weight_schemes,
        baseline_weight_scheme=baseline_weight_scheme,
    )
    if comparison.empty:
        return pd.DataFrame()
    comparison = comparison.copy()
    comparison["alpha_spread"] = comparison[list(weight_schemes)].astype(float).max(axis=1) - comparison[list(weight_schemes)].astype(float).min(axis=1)
    key_columns = list(basket_columns)

    selected_rows: list[pd.Series] = []
    reason_by_key: dict[tuple[object, ...], list[str]] = {}
    order_by_key: dict[tuple[object, ...], int] = {}

    def key_for(row: pd.Series) -> tuple[object, ...]:
        return tuple(row[column] for column in key_columns)

    def add_row(row: pd.Series, reason: str) -> None:
        key = key_for(row)
        reason_by_key.setdefault(key, [])
        if reason not in reason_by_key[key]:
            reason_by_key[key].append(reason)
        if key not in order_by_key:
            order_by_key[key] = len(order_by_key) + 1
            selected_rows.append(row)

    priority_specs = [
        ("winner_metric", "best_winner_metric"),
        (f"family_alpha_1_minus_{baseline_weight_scheme}", "best_alpha1_edge"),
        (f"family_alpha_0p5_minus_{baseline_weight_scheme}", "best_alpha0p5_edge"),
        ("alpha_spread", "best_alpha_spread"),
    ]
    for horizon, quota in quotas.items():
        quota = int(quota)
        if quota == 0:
            continue
        horizon_frame = comparison.loc[comparison["return_horizon"].astype(str) == str(horizon)].copy()
        if horizon_frame.empty:
            continue
        selected_horizon_keys: list[tuple[object, ...]] = []
        for sort_column, reason in priority_specs:
            if sort_column not in horizon_frame.columns:
                raise ValueError("comparison missing column: " + sort_column)
            ranked = horizon_frame.sort_values(
                [sort_column, "winner_metric", *key_columns],
                ascending=[False, False, *([True] * len(key_columns))],
                kind="mergesort",
            )
            if not ranked.empty:
                candidate = ranked.iloc[0]
                candidate_key = key_for(candidate)
                if candidate_key in selected_horizon_keys:
                    add_row(candidate, reason)
                elif len(selected_horizon_keys) < quota:
                    add_row(candidate, reason)
                    selected_horizon_keys.append(candidate_key)
        if len(selected_horizon_keys) < quota:
            ranked = horizon_frame.sort_values(
                ["winner_metric", *key_columns],
                ascending=[False, *([True] * len(key_columns))],
                kind="mergesort",
            )
            for _, row in ranked.iterrows():
                key = key_for(row)
                if key in selected_horizon_keys:
                    continue
                add_row(row, "quota_fill")
                selected_horizon_keys.append(key)
                if len(selected_horizon_keys) >= quota:
                    break

    if not selected_rows:
        return pd.DataFrame()
    selected = pd.DataFrame(selected_rows).copy()
    selected["selection_rank"] = selected.apply(lambda row: order_by_key[key_for(row)], axis=1)
    selected["selection_reason"] = selected.apply(
        lambda row: "|".join(reason_by_key[key_for(row)]),
        axis=1,
    )
    selected = selected.sort_values(["selection_rank"], kind="mergesort").reset_index(drop=True)
    selected["target_basket_id"] = [
        f"family_l4_target_{idx:03d}" for idx in range(1, len(selected) + 1)
    ]
    selected_basket_columns = [
        *key_columns,
        "target_basket_id",
        "selection_rank",
        "selection_reason",
        "winner_weight_scheme",
        "winner_metric",
        "alpha_spread",
        *list(weight_schemes),
        *(f"{scheme}_minus_{baseline_weight_scheme}" for scheme in weight_schemes),
    ]
    selected_baskets = selected[selected_basket_columns]
    manifest = l3_summary.loc[
        l3_summary["weight_scheme"].astype(str).isin(tuple(weight_schemes))
    ].merge(selected_baskets, on=key_columns, how="inner", validate="many_to_one")
    if manifest.empty:
        return manifest
    count_by_basket = manifest.groupby("target_basket_id")["weight_scheme"].nunique()
    incomplete = count_by_basket[count_by_basket != len(tuple(weight_schemes))]
    if not incomplete.empty:
        raise ValueError(
            "selected L4 target baskets do not have all requested alpha schemes: "
            + ", ".join(incomplete.index.astype(str))
        )
    manifest = manifest.sort_values(
        ["selection_rank", "weight_scheme"],
        kind="mergesort",
    ).reset_index(drop=True)
    return manifest


def compare_l3_l4_family_alpha_replay(
    l3_summary: pd.DataFrame,
    l4_summary: pd.DataFrame,
    *,
    l3_metric_column: str = "net_1x_sharpe",
    l4_metric_column: str = "net_1x_sharpe_on_equity",
    basket_columns: Sequence[str] = (
        "return_horizon",
        "track",
        "panel_frequency",
        "component_features",
        "n_components",
    ),
    weight_schemes: Sequence[str] = (
        "family_alpha_1",
        "family_alpha_0p5",
        "family_alpha_0",
    ),
    baseline_weight_scheme: str = "family_alpha_0",
    filtered_order_share_threshold: float = 0.01,
    actual_vs_target_gross_min_threshold: float = 0.95,
    weight_abs_error_sum_threshold: float = 0.10,
    abs_net_exposure_share_threshold: float = 0.10,
) -> pd.DataFrame:
    if l3_summary.empty or l4_summary.empty:
        return pd.DataFrame()
    required_l4 = {
        *basket_columns,
        "weight_scheme",
        l4_metric_column,
        "filtered_order_share",
        "mean_actual_vs_target_gross_ratio",
        "mean_weight_abs_error_sum",
        "max_abs_net_exposure_share",
        "max_margin_utilization",
    }
    missing_l4 = sorted(required_l4.difference(l4_summary.columns))
    if missing_l4:
        raise ValueError("l4_summary missing columns: " + ", ".join(missing_l4))
    l3_comparison = family_alpha_basket_comparison(
        l3_summary,
        metric_column=l3_metric_column,
        basket_columns=basket_columns,
        weight_schemes=weight_schemes,
        baseline_weight_scheme=baseline_weight_scheme,
    )
    l4_comparison = family_alpha_basket_comparison(
        l4_summary,
        metric_column=l4_metric_column,
        basket_columns=basket_columns,
        weight_schemes=weight_schemes,
        baseline_weight_scheme=baseline_weight_scheme,
    )
    if l3_comparison.empty or l4_comparison.empty:
        return pd.DataFrame()
    key_columns = list(basket_columns)
    l3_prefixed = l3_comparison.rename(
        columns={
            **{scheme: f"l3_{scheme}" for scheme in weight_schemes},
            "winner_weight_scheme": "l3_winner_alpha",
            "winner_metric": "l3_winner_metric",
            **{
                f"{scheme}_minus_{baseline_weight_scheme}": f"l3_{scheme}_minus_{baseline_weight_scheme}"
                for scheme in weight_schemes
            },
        }
    )
    l4_prefixed = l4_comparison.rename(
        columns={
            **{scheme: f"l4_{scheme}" for scheme in weight_schemes},
            "winner_weight_scheme": "l4_winner_alpha",
            "winner_metric": "l4_winner_metric",
            **{
                f"{scheme}_minus_{baseline_weight_scheme}": f"l4_{scheme}_minus_{baseline_weight_scheme}"
                for scheme in weight_schemes
            },
        }
    )
    merged = l3_prefixed.merge(l4_prefixed, on=key_columns, how="inner", validate="one_to_one")
    if merged.empty:
        return merged

    def sign_series(series: pd.Series) -> pd.Series:
        return pd.Series(np.sign(series.astype(float).to_numpy()), index=series.index)

    alpha1_column = f"family_alpha_1_minus_{baseline_weight_scheme}"
    alpha0p5_column = f"family_alpha_0p5_minus_{baseline_weight_scheme}"
    merged["winner_replicated"] = merged["l3_winner_alpha"].astype(str) == merged["l4_winner_alpha"].astype(str)
    merged["l3_alpha1_minus_alpha0"] = merged[f"l3_{alpha1_column}"].astype(float)
    merged["l4_alpha1_minus_alpha0"] = merged[f"l4_{alpha1_column}"].astype(float)
    merged["alpha1_edge_sign_replicated"] = sign_series(merged["l3_alpha1_minus_alpha0"]) == sign_series(merged["l4_alpha1_minus_alpha0"])
    merged["l3_alpha0p5_minus_alpha0"] = merged[f"l3_{alpha0p5_column}"].astype(float)
    merged["l4_alpha0p5_minus_alpha0"] = merged[f"l4_{alpha0p5_column}"].astype(float)
    merged["alpha0p5_edge_sign_replicated"] = sign_series(merged["l3_alpha0p5_minus_alpha0"]) == sign_series(merged["l4_alpha0p5_minus_alpha0"])

    l4_rows = l4_summary.loc[
        l4_summary["weight_scheme"].astype(str).isin(tuple(weight_schemes))
    ].copy()
    l4_rows["row_min_notional_material_impact"] = (
        (l4_rows["filtered_order_share"].astype(float) > float(filtered_order_share_threshold))
        | (l4_rows["mean_actual_vs_target_gross_ratio"].astype(float) < float(actual_vs_target_gross_min_threshold))
        | (l4_rows["mean_weight_abs_error_sum"].astype(float) > float(weight_abs_error_sum_threshold))
        | (l4_rows["max_abs_net_exposure_share"].astype(float) > float(abs_net_exposure_share_threshold))
    )
    material = (
        l4_rows.groupby(key_columns)
        .agg(
            l4_min_notional_material_impact=("row_min_notional_material_impact", "max"),
            l4_max_filtered_order_share=("filtered_order_share", "max"),
            l4_min_actual_vs_target_gross_ratio=("mean_actual_vs_target_gross_ratio", "min"),
            l4_max_weight_abs_error_sum=("mean_weight_abs_error_sum", "max"),
            l4_max_abs_net_exposure_share=("max_abs_net_exposure_share", "max"),
            l4_max_margin_utilization=("max_margin_utilization", "max"),
        )
        .reset_index()
    )
    merged = merged.merge(material, on=key_columns, how="left", validate="one_to_one")
    return merged


def composite_weight_scores_and_weights(
    train_stats: Mapping[str, FeatureTrainStat],
    weight_scheme: str,
    *,
    feature_families: Mapping[str, str] | None = None,
    feature_correlations: pd.DataFrame | None = None,
    active_weight_threshold: float = 0.05,
) -> tuple[dict[str, float], dict[str, float]]:
    scores, weights, _ = composite_weight_scores_weights_and_diagnostics(
        train_stats,
        weight_scheme,
        feature_families=feature_families,
        feature_correlations=feature_correlations,
        active_weight_threshold=active_weight_threshold,
    )
    return scores, weights


def composite_weight_scores_weights_and_diagnostics(
    train_stats: Mapping[str, FeatureTrainStat],
    weight_scheme: str,
    *,
    feature_families: Mapping[str, str] | None = None,
    feature_correlations: pd.DataFrame | None = None,
    active_weight_threshold: float = 0.05,
) -> tuple[dict[str, float], dict[str, float], dict[str, object]]:
    family_alpha = family_alpha_from_weight_scheme(weight_scheme)
    if family_alpha is not None:
        if feature_families is None:
            raise ValueError("feature_families is required for " + weight_scheme)
        weights = family_count_feature_weights(
            tuple(train_stats),
            feature_families,
            alpha=family_alpha,
        )
        diagnostics = weight_concentration_diagnostics(
            weights,
            feature_families=feature_families,
            active_weight_threshold=active_weight_threshold,
        )
        return {feature_name: float(weights[feature_name]) for feature_name in train_stats}, weights, diagnostics
    if weight_scheme == "corr_discount_icir":
        if feature_correlations is None:
            raise ValueError("feature_correlations is required for corr_discount_icir")
        return correlation_discount_weights(
            train_stats,
            feature_correlations,
            feature_families=feature_families,
            active_weight_threshold=active_weight_threshold,
        )
    scores, weights = normalized_feature_weights(train_stats, weight_scheme)
    diagnostics = weight_concentration_diagnostics(
        weights,
        feature_families=feature_families,
        active_weight_threshold=active_weight_threshold,
    )
    return scores, weights, diagnostics


def scenario_label(multiplier: float) -> str:
    value = float(multiplier)
    if value.is_integer():
        return f"{int(value)}x"
    return f"{value:g}".replace(".", "p") + "x"


def parse_positive_float_csv(raw: str | None, default: Sequence[float]) -> tuple[float, ...]:
    if raw is None or not raw.strip():
        return tuple(float(value) for value in default)
    values: list[float] = []
    for token in raw.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        value = float(stripped)
        if value <= 0.0:
            raise ValueError("values must be positive")
        values.append(value)
    if not values:
        raise ValueError("no values requested")
    return tuple(values)


def assigned_bucket_membership(
    group: pd.DataFrame,
    feature_name: str,
    n_buckets: int,
    *,
    return_column: str = "forward_return",
    strategy_return_column: str = "strategy_forward_return",
) -> tuple[pd.DataFrame, dict[str, object]]:
    selected_columns = ["decision_ts", "symbol", feature_name, return_column]
    if strategy_return_column in group.columns:
        selected_columns.append(strategy_return_column)
    valid = group.reset_index(names="decision_ts")[
        selected_columns].dropna()
    cross_section_size = len(valid)
    diagnostic = {
        "decision_ts": valid["decision_ts"].iloc[0] if cross_section_size else group.index[0],
        "cross_section_size": cross_section_size,
        "status": "ok",
    }
    if cross_section_size < n_buckets:
        diagnostic["status"] = "small_cross_section"
        return pd.DataFrame(), diagnostic
    if valid[feature_name].nunique(dropna=True) <= 1:
        diagnostic["status"] = "constant_feature"
        return pd.DataFrame(), diagnostic

    assigned = valid.rename(
        columns={feature_name: "signal_value", return_column: "forward_return"}
    ).copy()
    if strategy_return_column not in assigned.columns:
        assigned[strategy_return_column] = assigned["forward_return"]
    assigned = assigned.sort_values(
        ["signal_value", "symbol"], kind="mergesort").reset_index(drop=True)
    assigned["bucket"] = 0
    for bucket_idx, positions in enumerate(np.array_split(np.arange(cross_section_size), n_buckets), start=1):
        assigned.loc[positions, "bucket"] = bucket_idx
    return assigned, diagnostic


def combo_signal_target_membership(
    composite_detail: pd.DataFrame,
    *,
    n_buckets: int,
) -> pd.DataFrame:
    """Convert OOS combo signals into deterministic equal-weight tail targets.

    This entry only assigns cross-sectional buckets and target weights. It does
    not calculate returns, turnover, costs, or execution metrics.
    """
    identity_columns = [
        "combo_id",
        "track",
        "weight_scheme",
        "panel_frequency",
        "return_horizon",
        "component_features",
        "fold_idx",
        "decision_ts",
    ]
    required = {*identity_columns, "symbol", "combo_signal", "forward_return"}
    missing = sorted(required.difference(composite_detail.columns))
    if missing:
        raise ValueError(
            "composite_detail missing target-membership columns: "
            + ", ".join(missing)
        )
    if composite_detail.empty:
        raise ValueError("composite_detail must not be empty")
    if int(n_buckets) < 2:
        raise ValueError("n_buckets must be at least 2")

    working = composite_detail[
        [*identity_columns, "symbol", "combo_signal", "forward_return"]
    ].dropna().copy()
    group_key = identity_columns
    group_size = working.groupby(group_key, sort=False)["symbol"].transform("size")
    unique_signal = working.groupby(group_key, sort=False)["combo_signal"].transform("nunique")
    if group_size.lt(int(n_buckets)).any():
        raise ValueError("combo target membership contains a small cross-section")
    if unique_signal.le(1).any():
        raise ValueError("combo target membership contains a constant signal")

    working = working.sort_values(
        [*group_key, "combo_signal", "symbol"], kind="mergesort"
    ).reset_index(drop=True)
    group_size = working.groupby(group_key, sort=False)["symbol"].transform("size").to_numpy(dtype=int)
    position = working.groupby(group_key, sort=False).cumcount().to_numpy(dtype=int)
    quotient = group_size // int(n_buckets)
    remainder = group_size % int(n_buckets)
    larger_prefix = (quotient + 1) * remainder
    in_larger_prefix = position < larger_prefix
    bucket = np.where(
        in_larger_prefix,
        position // (quotient + 1) + 1,
        remainder + (position - larger_prefix) // quotient + 1,
    )
    working["bucket"] = bucket.astype(int)
    result = working.loc[working["bucket"].isin({1, int(n_buckets)})].rename(
        columns={"combo_signal": "signal_value"}
    ).copy()
    result["leg"] = np.where(result["bucket"].eq(int(n_buckets)), "long", "short")
    side_counts = result.groupby([*group_key, "leg"], sort=False)["symbol"].transform("count").astype(float)
    result["target_weight"] = np.where(
        result["leg"].eq("long"),
        0.5 / side_counts,
        -0.5 / side_counts,
    )
    result = result[
        [
            *identity_columns,
            "symbol",
            "signal_value",
            "bucket",
            "leg",
            "target_weight",
        ]
    ]
    return result.sort_values(
        ["combo_id", "weight_scheme", "fold_idx", "decision_ts", "leg", "symbol"],
        kind="mergesort",
    ).reset_index(drop=True)


def name_turnover_share(current: set[str], previous: set[str]) -> float:
    if not current and not previous:
        return 0.0
    if not current or not previous:
        return 1.0
    denominator = max(len(current), len(previous))
    return float(1.0 - len(current.intersection(previous)) / denominator)


def _empty_holdings_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", "leg", "weight"])


def long_short_strategy_snapshot(
    combo_spec: ComboSpec,
    fold,
    decision_ts: pd.Timestamp,
    assigned: pd.DataFrame,
    previous_holdings: pd.DataFrame,
    *,
    n_buckets: int,
    cost_multipliers: Sequence[float],
    taker_fee_rate: float,
    component_features: str,
    terminal_close_turnover: float = 0.0,
) -> tuple[dict[str, object], pd.DataFrame]:
    long_members = assigned[assigned["bucket"] == n_buckets].copy()
    short_members = assigned[assigned["bucket"] == 1].copy()
    if long_members.empty or short_members.empty:
        raise ValueError(
            "strategy snapshot requires non-empty long and short legs")

    long_weight = 0.5 / len(long_members)
    short_weight = -0.5 / len(short_members)
    long_holdings = long_members.assign(leg="long", weight=long_weight)
    short_holdings = short_members.assign(leg="short", weight=short_weight)
    holdings = pd.concat([long_holdings, short_holdings], ignore_index=True)
    holdings["contribution"] = holdings["weight"] * \
        holdings["strategy_forward_return"]

    long_symbols = set(long_holdings["symbol"].astype(str))
    short_symbols = set(short_holdings["symbol"].astype(str))
    if previous_holdings.empty:
        previous_long_symbols: set[str] = set()
        previous_short_symbols: set[str] = set()
    else:
        previous_long_symbols = set(previous_holdings.loc[previous_holdings["leg"] == "long", "symbol"].astype(str))
        previous_short_symbols = set(previous_holdings.loc[previous_holdings["leg"] == "short", "symbol"].astype(str))

    current_weights = holdings.groupby("symbol", sort=False)["weight"].sum()
    previous_weights = (
        previous_holdings.groupby("symbol", sort=False)["weight"].sum()
        if not previous_holdings.empty
        else pd.Series(dtype=float)
    )
    rebalance_turnover = float(
        current_weights.sub(previous_weights, fill_value=0.0).abs().sum())
    charged_turnover = float(rebalance_turnover + terminal_close_turnover)
    long_name_turnover = name_turnover_share(
        long_symbols, previous_long_symbols)
    short_name_turnover = name_turnover_share(
        short_symbols, previous_short_symbols)

    long_leg_return = float(long_holdings["forward_return"].mean())
    short_leg_return = float(short_holdings["forward_return"].mean())
    spread_return = long_leg_return - short_leg_return
    strategy_long_leg_return = float(
        long_holdings["strategy_forward_return"].mean())
    strategy_short_leg_return = float(
        short_holdings["strategy_forward_return"].mean())
    strategy_spread_return = strategy_long_leg_return - strategy_short_leg_return
    gross_return = 0.5 * strategy_spread_return
    benchmark_return = float(assigned["strategy_forward_return"].mean())
    active_return = gross_return - benchmark_return

    snapshot: dict[str, object] = {
        "combo_id": combo_spec.combo_id,
        "track": combo_spec.track,
        "weight_scheme": combo_spec.weight_scheme,
        "panel_frequency": combo_spec.panel_frequency,
        "return_horizon": combo_spec.return_horizon,
        "component_features": component_features,
        "n_components": len(combo_spec.feature_names),
        "fold_idx": fold.fold_idx,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "decision_ts": decision_ts,
        "cross_section_size": int(assigned["symbol"].nunique()),
        "long_count": int(len(long_holdings)),
        "short_count": int(len(short_holdings)),
        "benchmark_return": benchmark_return,
        "long_leg_return": long_leg_return,
        "short_leg_return": short_leg_return,
        "spread_return": spread_return,
        "strategy_long_leg_return": strategy_long_leg_return,
        "strategy_short_leg_return": strategy_short_leg_return,
        "strategy_spread_return": strategy_spread_return,
        "gross_return": gross_return,
        "active_return": active_return,
        "rebalance_turnover": rebalance_turnover,
        "terminal_close_turnover": float(terminal_close_turnover),
        "charged_turnover": charged_turnover,
        "long_name_turnover_share": long_name_turnover,
        "short_name_turnover_share": short_name_turnover,
        "name_turnover_share": float((long_name_turnover + short_name_turnover) / 2.0),
    }
    for multiplier in cost_multipliers:
        label = scenario_label(float(multiplier))
        cost = charged_turnover * taker_fee_rate * float(multiplier)
        snapshot[f"cost_{label}"] = float(cost)
        snapshot[f"net_return_{label}"] = float(gross_return - cost)
        snapshot[f"net_active_return_{label}"] = float(active_return - cost)

    holdings = holdings.assign(
        combo_id=combo_spec.combo_id,
        track=combo_spec.track,
        weight_scheme=combo_spec.weight_scheme,
        panel_frequency=combo_spec.panel_frequency,
        return_horizon=combo_spec.return_horizon,
        component_features=component_features,
        fold_idx=fold.fold_idx,
        decision_ts=decision_ts,
    )
    return snapshot, holdings[
        [
            "combo_id",
            "track",
            "weight_scheme",
            "panel_frequency",
            "return_horizon",
            "component_features",
            "fold_idx",
            "decision_ts",
            "symbol",
            "bucket",
            "leg",
            "signal_value",
            "forward_return",
            "strategy_forward_return",
            "weight",
            "contribution",
        ]
    ]


def add_terminal_close_cost(
    rows: list[dict[str, object]],
    last_holdings: pd.DataFrame | None,
    cost_multipliers: Sequence[float],
    taker_fee_rate: float,
) -> None:
    if not rows or last_holdings is None or last_holdings.empty:
        return
    terminal_close_turnover = float(last_holdings["weight"].abs().sum())
    rows[-1]["terminal_close_turnover"] = terminal_close_turnover
    rows[-1]["charged_turnover"] = float(
        rows[-1]["charged_turnover"]) + terminal_close_turnover
    for multiplier in cost_multipliers:
        label = scenario_label(float(multiplier))
        extra_cost = terminal_close_turnover * taker_fee_rate * float(multiplier)
        rows[-1][f"cost_{label}"] = float(rows[-1][f"cost_{label}"]) + extra_cost
        rows[-1][f"net_return_{label}"] = float(
            rows[-1][f"net_return_{label}"]) - extra_cost
        rows[-1][f"net_active_return_{label}"] = float(
            rows[-1][f"net_active_return_{label}"]) - extra_cost


def long_short_strategy_frames_for_fold(
    combo_spec: ComboSpec,
    fold,
    test_composite: pd.DataFrame,
    *,
    n_buckets: int,
    cost_multipliers: Sequence[float],
    taker_fee_rate: float,
    component_features: str,
) -> tuple[list[dict[str, object]], pd.DataFrame, list[dict[str, object]]]:
    decision_index = decision_index_for_frame(test_composite)
    if decision_index.empty:
        return [], pd.DataFrame(), []
    working = test_composite.reset_index(names="decision_ts")[
        ["decision_ts", "symbol", "composite_signal", "forward_return", "strategy_forward_return"]
    ]
    valid = working.dropna(
        subset=["composite_signal", "forward_return", "strategy_forward_return"]).copy()
    if valid.empty:
        diagnostics = [
            {"decision_ts": decision_ts, "cross_section_size": 0,
                "status": "small_cross_section"}
            for decision_ts in decision_index
        ]
        return [], pd.DataFrame(), diagnostics

    grouped = valid.groupby("decision_ts", sort=False)
    counts = grouped.size().reindex(decision_index, fill_value=0).astype(int)
    unique_signal = grouped["composite_signal"].nunique(
        dropna=True).reindex(decision_index, fill_value=0).astype(int)
    diagnostics: list[dict[str, object]] = []
    ok_decisions: list[object] = []
    for decision_ts in decision_index:
        cross_section_size = int(counts.loc[decision_ts])
        status = "ok"
        if cross_section_size < n_buckets:
            status = "small_cross_section"
        elif int(unique_signal.loc[decision_ts]) <= 1:
            status = "constant_feature"
        else:
            ok_decisions.append(decision_ts)
        diagnostics.append(
            {
                "decision_ts": decision_ts,
                "cross_section_size": cross_section_size,
                "status": status,
            }
        )
    if not ok_decisions:
        return [], pd.DataFrame(), diagnostics

    scored = valid[valid["decision_ts"].isin(ok_decisions)].copy()
    scored = scored.sort_values(
        ["decision_ts", "composite_signal", "symbol"], kind="mergesort").reset_index(drop=True)
    group_sizes = scored.groupby("decision_ts", sort=False)[
        "symbol"].transform("size").astype(int)
    positions = scored.groupby(
        "decision_ts", sort=False).cumcount().astype(int)
    base_sizes = group_sizes // n_buckets
    remainders = group_sizes % n_buckets
    first_block_limits = (base_sizes + 1) * remainders
    scored["bucket"] = np.where(
        positions < first_block_limits,
        positions // (base_sizes + 1),
        remainders + (positions - first_block_limits) // base_sizes,
    ).astype(int) + 1

    legs = scored[scored["bucket"].isin((1, n_buckets))].copy()
    leg_counts = legs.groupby(["decision_ts", "bucket"], sort=False)[
        "symbol"].transform("size").astype(float)
    legs["leg"] = np.where(legs["bucket"] == n_buckets, "long", "short")
    legs["weight"] = np.where(
        legs["bucket"] == n_buckets,
        0.5 / leg_counts,
        -0.5 / leg_counts,
    )
    legs["contribution"] = legs["weight"] * legs["strategy_forward_return"]

    weights_wide = legs.pivot_table(
        index="decision_ts",
        columns="symbol",
        values="weight",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()
    previous_weights = weights_wide.shift(1).fillna(0.0)
    rebalance_turnover = weights_wide.sub(
        previous_weights, fill_value=0.0).abs().sum(axis=1)
    terminal_close_turnover = pd.Series(0.0, index=weights_wide.index)
    if len(weights_wide) > 0:
        terminal_close_turnover.iloc[-1] = float(
            weights_wide.iloc[-1].abs().sum())
    charged_turnover = rebalance_turnover + terminal_close_turnover

    returns_by_decision = scored.groupby("decision_ts", sort=False).agg(
        benchmark_return=("strategy_forward_return", "mean"),
        cross_section_size=("symbol", "nunique"),
    )
    leg_returns = legs.pivot_table(
        index="decision_ts",
        columns="leg",
        values=["forward_return", "strategy_forward_return"],
        aggfunc="mean",
    )
    long_leg_return = leg_returns[(
        "forward_return", "long")].reindex(weights_wide.index)
    short_leg_return = leg_returns[(
        "forward_return", "short")].reindex(weights_wide.index)
    strategy_long_leg_return = leg_returns[(
        "strategy_forward_return", "long")].reindex(weights_wide.index)
    strategy_short_leg_return = leg_returns[(
        "strategy_forward_return", "short")].reindex(weights_wide.index)
    strategy_spread_return = strategy_long_leg_return - strategy_short_leg_return
    gross_return = 0.5 * strategy_spread_return
    benchmark_return = returns_by_decision["benchmark_return"].reindex(
        weights_wide.index)

    long_sets = legs[legs["leg"] == "long"].groupby("decision_ts")[
        "symbol"].agg(lambda values: set(values.astype(str)))
    short_sets = legs[legs["leg"] == "short"].groupby("decision_ts")[
        "symbol"].agg(lambda values: set(values.astype(str)))
    long_turnover: list[float] = []
    short_turnover: list[float] = []
    previous_long: set[str] = set()
    previous_short: set[str] = set()
    for decision_ts in weights_wide.index:
        current_long = long_sets.get(decision_ts, set())
        current_short = short_sets.get(decision_ts, set())
        long_turnover.append(name_turnover_share(current_long, previous_long))
        short_turnover.append(name_turnover_share(
            current_short, previous_short))
        previous_long = current_long
        previous_short = current_short

    rows_frame = pd.DataFrame(
        {
            "combo_id": combo_spec.combo_id,
            "track": combo_spec.track,
            "weight_scheme": combo_spec.weight_scheme,
            "panel_frequency": combo_spec.panel_frequency,
            "return_horizon": combo_spec.return_horizon,
            "component_features": component_features,
            "n_components": len(combo_spec.feature_names),
            "fold_idx": fold.fold_idx,
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
            "decision_ts": weights_wide.index,
            "cross_section_size": returns_by_decision["cross_section_size"].reindex(weights_wide.index).astype(int).to_numpy(),
            "long_count": legs[legs["leg"] == "long"].groupby("decision_ts").size().reindex(weights_wide.index).astype(int).to_numpy(),
            "short_count": legs[legs["leg"] == "short"].groupby("decision_ts").size().reindex(weights_wide.index).astype(int).to_numpy(),
            "benchmark_return": benchmark_return.to_numpy(dtype=float),
            "long_leg_return": long_leg_return.to_numpy(dtype=float),
            "short_leg_return": short_leg_return.to_numpy(dtype=float),
            "spread_return": (long_leg_return - short_leg_return).to_numpy(dtype=float),
            "strategy_long_leg_return": strategy_long_leg_return.to_numpy(dtype=float),
            "strategy_short_leg_return": strategy_short_leg_return.to_numpy(dtype=float),
            "strategy_spread_return": strategy_spread_return.to_numpy(dtype=float),
            "gross_return": gross_return.to_numpy(dtype=float),
            "active_return": (gross_return - benchmark_return).to_numpy(dtype=float),
            "rebalance_turnover": rebalance_turnover.to_numpy(dtype=float),
            "terminal_close_turnover": terminal_close_turnover.to_numpy(dtype=float),
            "charged_turnover": charged_turnover.to_numpy(dtype=float),
            "long_name_turnover_share": long_turnover,
            "short_name_turnover_share": short_turnover,
            "name_turnover_share": (np.asarray(long_turnover, dtype=float) + np.asarray(short_turnover, dtype=float)) / 2.0,
        }
    )
    for multiplier in cost_multipliers:
        label = scenario_label(float(multiplier))
        cost = rows_frame["charged_turnover"].astype(
            float) * taker_fee_rate * float(multiplier)
        rows_frame[f"cost_{label}"] = cost
        rows_frame[f"net_return_{label}"] = rows_frame["gross_return"] - cost
        rows_frame[f"net_active_return_{label}"] = rows_frame["active_return"] - cost

    holdings = legs.rename(columns={"composite_signal": "signal_value"})[
        [
            "decision_ts",
            "symbol",
            "bucket",
            "leg",
            "signal_value",
            "forward_return",
            "strategy_forward_return",
            "weight",
            "contribution",
        ]
    ].copy()
    holdings = holdings.assign(
        combo_id=combo_spec.combo_id,
        track=combo_spec.track,
        weight_scheme=combo_spec.weight_scheme,
        panel_frequency=combo_spec.panel_frequency,
        return_horizon=combo_spec.return_horizon,
        component_features=component_features,
        fold_idx=fold.fold_idx,
    )
    holdings = holdings[
        [
            "combo_id",
            "track",
            "weight_scheme",
            "panel_frequency",
            "return_horizon",
            "component_features",
            "fold_idx",
            "decision_ts",
            "symbol",
            "bucket",
            "leg",
            "signal_value",
            "forward_return",
            "strategy_forward_return",
            "weight",
            "contribution",
        ]
    ]
    return rows_frame.to_dict("records"), holdings, diagnostics


def long_short_top_bottom_strategy_frames_for_fold(
    combo_spec: ComboSpec,
    fold,
    test_composite: pd.DataFrame,
    *,
    leg_count: int,
    cost_multipliers: Sequence[float],
    taker_fee_rate: float,
    component_features: str,
) -> tuple[list[dict[str, object]], pd.DataFrame, list[dict[str, object]]]:
    decision_index = decision_index_for_frame(test_composite)
    if decision_index.empty:
        return [], pd.DataFrame(), []
    working = test_composite.reset_index(names="decision_ts")[
        ["decision_ts", "symbol", "composite_signal", "forward_return", "strategy_forward_return"]
    ]
    valid = working.dropna(
        subset=["composite_signal", "forward_return", "strategy_forward_return"]).copy()
    if valid.empty:
        diagnostics = [
            {"decision_ts": decision_ts, "cross_section_size": 0,
                "status": "small_cross_section"}
            for decision_ts in decision_index
        ]
        return [], pd.DataFrame(), diagnostics

    grouped = valid.groupby("decision_ts", sort=False)
    counts = grouped.size().reindex(decision_index, fill_value=0).astype(int)
    unique_signal = grouped["composite_signal"].nunique(
        dropna=True).reindex(decision_index, fill_value=0).astype(int)
    min_cross_section = int(leg_count) * 2
    diagnostics: list[dict[str, object]] = []
    ok_decisions: list[object] = []
    for decision_ts in decision_index:
        cross_section_size = int(counts.loc[decision_ts])
        status = "ok"
        if cross_section_size < min_cross_section:
            status = "small_cross_section"
        elif int(unique_signal.loc[decision_ts]) <= 1:
            status = "constant_feature"
        else:
            ok_decisions.append(decision_ts)
        diagnostics.append(
            {
                "decision_ts": decision_ts,
                "cross_section_size": cross_section_size,
                "status": status,
            }
        )
    if not ok_decisions:
        return [], pd.DataFrame(), diagnostics

    scored = valid[valid["decision_ts"].isin(ok_decisions)].copy()
    scored = scored.sort_values(
        ["decision_ts", "composite_signal", "symbol"], kind="mergesort").reset_index(drop=True)
    positions = scored.groupby(
        "decision_ts", sort=False).cumcount().astype(int)
    group_sizes = scored.groupby("decision_ts", sort=False)[
        "symbol"].transform("size").astype(int)
    scored["leg"] = ""
    scored.loc[positions < leg_count, "leg"] = "short"
    scored.loc[positions >= (group_sizes - leg_count), "leg"] = "long"
    scored["bucket"] = np.where(scored["leg"] == "short", 1, np.where(scored["leg"] == "long", 2, 0))

    legs = scored[scored["leg"].isin(("short", "long"))].copy()
    leg_counts = legs.groupby(["decision_ts", "leg"], sort=False)[
        "symbol"].transform("size").astype(float)
    legs["weight"] = np.where(
        legs["leg"] == "long",
        0.5 / leg_counts,
        -0.5 / leg_counts,
    )
    legs["contribution"] = legs["weight"] * legs["strategy_forward_return"]

    weights_wide = legs.pivot_table(
        index="decision_ts",
        columns="symbol",
        values="weight",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()
    previous_weights = weights_wide.shift(1).fillna(0.0)
    rebalance_turnover = weights_wide.sub(
        previous_weights, fill_value=0.0).abs().sum(axis=1)
    terminal_close_turnover = pd.Series(0.0, index=weights_wide.index)
    if len(weights_wide) > 0:
        terminal_close_turnover.iloc[-1] = float(
            weights_wide.iloc[-1].abs().sum())
    charged_turnover = rebalance_turnover + terminal_close_turnover

    returns_by_decision = scored.groupby("decision_ts", sort=False).agg(
        benchmark_return=("strategy_forward_return", "mean"),
        cross_section_size=("symbol", "nunique"),
    )
    leg_returns = legs.pivot_table(
        index="decision_ts",
        columns="leg",
        values=["forward_return", "strategy_forward_return"],
        aggfunc="mean",
    )
    long_leg_return = leg_returns[(
        "forward_return", "long")].reindex(weights_wide.index)
    short_leg_return = leg_returns[(
        "forward_return", "short")].reindex(weights_wide.index)
    strategy_long_leg_return = leg_returns[(
        "strategy_forward_return", "long")].reindex(weights_wide.index)
    strategy_short_leg_return = leg_returns[(
        "strategy_forward_return", "short")].reindex(weights_wide.index)
    strategy_spread_return = strategy_long_leg_return - strategy_short_leg_return
    gross_return = 0.5 * strategy_spread_return
    benchmark_return = returns_by_decision["benchmark_return"].reindex(
        weights_wide.index)

    long_sets = legs[legs["leg"] == "long"].groupby("decision_ts")[
        "symbol"].agg(lambda values: set(values.astype(str)))
    short_sets = legs[legs["leg"] == "short"].groupby("decision_ts")[
        "symbol"].agg(lambda values: set(values.astype(str)))
    long_turnover: list[float] = []
    short_turnover: list[float] = []
    previous_long: set[str] = set()
    previous_short: set[str] = set()
    for decision_ts in weights_wide.index:
        current_long = long_sets.get(decision_ts, set())
        current_short = short_sets.get(decision_ts, set())
        long_turnover.append(name_turnover_share(current_long, previous_long))
        short_turnover.append(name_turnover_share(current_short, previous_short))
        previous_long = current_long
        previous_short = current_short

    rows_frame = pd.DataFrame(
        {
            "combo_id": combo_spec.combo_id,
            "track": combo_spec.track,
            "weight_scheme": combo_spec.weight_scheme,
            "panel_frequency": combo_spec.panel_frequency,
            "return_horizon": combo_spec.return_horizon,
            "component_features": component_features,
            "n_components": len(combo_spec.feature_names),
            "fold_idx": fold.fold_idx,
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
            "decision_ts": weights_wide.index,
            "cross_section_size": returns_by_decision["cross_section_size"].reindex(weights_wide.index).astype(int).to_numpy(),
            "long_count": legs[legs["leg"] == "long"].groupby("decision_ts").size().reindex(weights_wide.index).astype(int).to_numpy(),
            "short_count": legs[legs["leg"] == "short"].groupby("decision_ts").size().reindex(weights_wide.index).astype(int).to_numpy(),
            "benchmark_return": benchmark_return.to_numpy(dtype=float),
            "long_leg_return": long_leg_return.to_numpy(dtype=float),
            "short_leg_return": short_leg_return.to_numpy(dtype=float),
            "spread_return": (long_leg_return - short_leg_return).to_numpy(dtype=float),
            "strategy_long_leg_return": strategy_long_leg_return.to_numpy(dtype=float),
            "strategy_short_leg_return": strategy_short_leg_return.to_numpy(dtype=float),
            "strategy_spread_return": strategy_spread_return.to_numpy(dtype=float),
            "gross_return": gross_return.to_numpy(dtype=float),
            "active_return": (gross_return - benchmark_return).to_numpy(dtype=float),
            "rebalance_turnover": rebalance_turnover.to_numpy(dtype=float),
            "terminal_close_turnover": terminal_close_turnover.to_numpy(dtype=float),
            "charged_turnover": charged_turnover.to_numpy(dtype=float),
            "long_name_turnover_share": long_turnover,
            "short_name_turnover_share": short_turnover,
            "name_turnover_share": (np.asarray(long_turnover, dtype=float) + np.asarray(short_turnover, dtype=float)) / 2.0,
        }
    )
    for multiplier in cost_multipliers:
        label = scenario_label(float(multiplier))
        cost = rows_frame["charged_turnover"].astype(float) * taker_fee_rate * float(multiplier)
        rows_frame[f"cost_{label}"] = cost
        rows_frame[f"net_return_{label}"] = rows_frame["gross_return"] - cost
        rows_frame[f"net_active_return_{label}"] = rows_frame["active_return"] - cost

    holdings = legs.rename(columns={"composite_signal": "signal_value"})[
        [
            "decision_ts",
            "symbol",
            "bucket",
            "leg",
            "signal_value",
            "forward_return",
            "strategy_forward_return",
            "weight",
            "contribution",
        ]
    ].copy()
    holdings = holdings.assign(
        combo_id=combo_spec.combo_id,
        track=combo_spec.track,
        weight_scheme=combo_spec.weight_scheme,
        panel_frequency=combo_spec.panel_frequency,
        return_horizon=combo_spec.return_horizon,
        component_features=component_features,
        fold_idx=fold.fold_idx,
    )
    holdings = holdings[
        [
            "combo_id",
            "track",
            "weight_scheme",
            "panel_frequency",
            "return_horizon",
            "component_features",
            "fold_idx",
            "decision_ts",
            "symbol",
            "bucket",
            "leg",
            "signal_value",
            "forward_return",
            "strategy_forward_return",
            "weight",
            "contribution",
        ]
    ]
    return rows_frame.to_dict("records"), holdings, diagnostics


def summarize_long_short_strategy(
    combo_spec: ComboSpec,
    detail_frame: pd.DataFrame,
    diagnostics: list[dict[str, object]],
    walk_forward_spec: Mapping[str, int],
    decision_frequency: str,
    frequency_periods_per_year: Mapping[str, int | float],
    cost_multipliers: Sequence[float],
) -> dict[str, object]:
    diagnostic_frame = pd.DataFrame(diagnostics)
    test_decision_count = len(diagnostic_frame)
    status_counts = diagnostic_frame["status"].value_counts(
    ) if test_decision_count else pd.Series(dtype=int)
    scored_decision_count = int(status_counts.get("ok", 0))
    skipped_decision_count = test_decision_count - scored_decision_count
    gross_fold_positive_share = fold_positive_share_from_returns(
        detail_frame, "gross_return")
    gross_top_fold_contribution = top_fold_contribution_from_returns(
        detail_frame, "gross_return")
    summary: dict[str, object] = {
        "combo_id": combo_spec.combo_id,
        "track": combo_spec.track,
        "weight_scheme": combo_spec.weight_scheme,
        "panel_frequency": combo_spec.panel_frequency,
        "return_horizon": combo_spec.return_horizon,
        "component_features": feature_list_text(combo_spec.feature_names),
        "n_components": len(combo_spec.feature_names),
        "train_days": walk_forward_spec["train_days"],
        "test_days": walk_forward_spec["test_days"],
        "embargo_days": walk_forward_spec["embargo_days"],
        "step_days": walk_forward_spec["step_days"],
        "n_folds": int(detail_frame["fold_idx"].nunique()) if not detail_frame.empty else 0,
        "test_decision_count": test_decision_count,
        "scored_decision_count": scored_decision_count,
        "skipped_decision_count": skipped_decision_count,
        "skipped_decision_share": float(skipped_decision_count / test_decision_count) if test_decision_count else float("nan"),
        "skipped_small_cross_section_count": int(status_counts.get("small_cross_section", 0)),
        "skipped_constant_feature_count": int(status_counts.get("constant_feature", 0)),
        "mean_cross_section_size": float(detail_frame["cross_section_size"].mean()) if not detail_frame.empty else float("nan"),
        "mean_long_count": float(detail_frame["long_count"].mean()) if not detail_frame.empty else float("nan"),
        "mean_short_count": float(detail_frame["short_count"].mean()) if not detail_frame.empty else float("nan"),
        "mean_name_turnover_share": float(detail_frame["name_turnover_share"].mean()) if not detail_frame.empty else float("nan"),
        "mean_charged_turnover": float(detail_frame["charged_turnover"].mean()) if not detail_frame.empty else float("nan"),
        "annualized_charged_turnover": annualized_mean_return(detail_frame["charged_turnover"], decision_frequency, frequency_periods_per_year) if not detail_frame.empty else float("nan"),
        "oos_mean_return": float(detail_frame["gross_return"].mean()) if not detail_frame.empty else float("nan"),
        "gross_annualized_return": annualized_mean_return(detail_frame["gross_return"], decision_frequency, frequency_periods_per_year) if not detail_frame.empty else float("nan"),
        "gross_annualized_volatility": annualized_volatility(detail_frame["gross_return"], decision_frequency, frequency_periods_per_year) if not detail_frame.empty else float("nan"),
        "gross_sharpe": annualized_sharpe_for_frequency(detail_frame["gross_return"], decision_frequency, frequency_periods_per_year) if not detail_frame.empty else float("nan"),
        "gross_max_drawdown": max_drawdown_from_returns(detail_frame["gross_return"]) if not detail_frame.empty else float("nan"),
        "gross_fold_positive_share": gross_fold_positive_share,
        "gross_top_fold_contribution": gross_top_fold_contribution,
        "fold_positive_share": gross_fold_positive_share,
        "top_fold_contribution": gross_top_fold_contribution,
    }
    for multiplier in cost_multipliers:
        label = scenario_label(float(multiplier))
        net_return_col = f"net_return_{label}"
        cost_col = f"cost_{label}"
        summary[f"cost_{label}_mean_bps"] = float(detail_frame[cost_col].mean(
        ) * 10_000.0) if not detail_frame.empty else float("nan")
        summary[f"net_{label}_annualized_return"] = annualized_mean_return(
            detail_frame[net_return_col], decision_frequency, frequency_periods_per_year
        ) if not detail_frame.empty else float("nan")
        summary[f"net_{label}_sharpe"] = annualized_sharpe_for_frequency(
            detail_frame[net_return_col], decision_frequency, frequency_periods_per_year
        ) if not detail_frame.empty else float("nan")
        summary[f"net_{label}_max_drawdown"] = max_drawdown_from_returns(
            detail_frame[net_return_col]) if not detail_frame.empty else float("nan")
        summary[f"net_{label}_fold_positive_share"] = fold_positive_share_from_returns(
            detail_frame, net_return_col)
        summary[f"net_{label}_top_fold_contribution"] = top_fold_contribution_from_returns(
            detail_frame, net_return_col)
    weight_columns = {
        "effective_factor_count",
        "active_factor_count",
        "max_feature_weight",
        "family_weight_share_max",
        "effective_family_count",
        "max_family_weight",
        "family_count",
        "near_single_or_few_factor",
        "mean_abs_feature_corr",
        "max_abs_feature_corr",
        "min_corr_discount",
        "max_corr_discount",
        "correlation_pair_count",
        "correlation_min_pair_observation_count",
    }
    if weight_columns.intersection(detail_frame.columns):
        fold_weight_diag = detail_frame[
            ["fold_idx", *sorted(weight_columns.intersection(detail_frame.columns))]
        ].drop_duplicates("fold_idx")
        for column in (
            "effective_factor_count",
            "active_factor_count",
            "max_feature_weight",
            "family_weight_share_max",
            "effective_family_count",
            "max_family_weight",
            "mean_abs_feature_corr",
            "max_abs_feature_corr",
            "min_corr_discount",
            "max_corr_discount",
        ):
            if column in fold_weight_diag.columns:
                summary[f"mean_{column}"] = float(fold_weight_diag[column].astype(float).mean())
                summary[f"min_{column}"] = float(fold_weight_diag[column].astype(float).min())
                summary[f"max_{column}"] = float(fold_weight_diag[column].astype(float).max())
        if "near_single_or_few_factor" in fold_weight_diag.columns:
            near = fold_weight_diag["near_single_or_few_factor"].astype(bool)
            summary["near_single_or_few_factor_fold_count"] = int(near.sum())
            summary["near_single_or_few_factor_fold_share"] = float(near.mean())
        if "family_count" in fold_weight_diag.columns:
            summary["family_count"] = int(fold_weight_diag["family_count"].astype(int).max())
        if "correlation_pair_count" in fold_weight_diag.columns:
            summary["correlation_pair_count"] = int(fold_weight_diag["correlation_pair_count"].astype(int).max())
        if "correlation_min_pair_observation_count" in fold_weight_diag.columns:
            summary["correlation_min_pair_observation_count"] = int(
                fold_weight_diag["correlation_min_pair_observation_count"].astype(int).max()
            )
    return summary


def validate_long_short_strategy_inputs(
    combo_spec: ComboSpec,
    signal_frame: pd.DataFrame,
    folds: Sequence,
    decision_frequency: str,
    n_buckets: int,
    min_cross_section: int,
    frequency_periods_per_year: Mapping[str, int | float],
    cost_multipliers: Sequence[float],
    taker_fee_rate: float,
    horizon_deltas: Mapping[str, pd.Timedelta] | None,
    supported_signal_timeframes: Sequence[str] | None,
    no_overlap_validated: bool,
) -> None:
    """Fail closed before producing long-short strategy conclusion fields."""
    if not combo_spec.feature_names:
        raise ValueError("combo_spec.feature_names must not be empty")
    if decision_frequency not in frequency_periods_per_year:
        raise ValueError("decision_frequency missing from periods-per-year map: " + decision_frequency)
    if int(n_buckets) < 2:
        raise ValueError("n_buckets must be at least 2")
    if int(min_cross_section) < int(n_buckets):
        raise ValueError("min_cross_section must be >= n_buckets")
    if len(folds) == 0:
        raise ValueError("at least one walk-forward fold is required")
    if float(taker_fee_rate) < 0.0:
        raise ValueError("taker_fee_rate must be non-negative")
    if not cost_multipliers:
        raise ValueError("cost_multipliers must not be empty")
    for multiplier in cost_multipliers:
        if float(multiplier) <= 0.0:
            raise ValueError("cost_multipliers must be positive")

    required_columns = {
        "symbol",
        "forward_return",
        "strategy_forward_return",
        *combo_spec.feature_names,
    }
    missing = sorted(column for column in required_columns if column not in signal_frame.columns)
    if missing:
        raise ValueError("signal_frame missing required columns: " + ", ".join(missing))

    if no_overlap_validated:
        return
    if horizon_deltas is None or supported_signal_timeframes is None:
        raise ValueError(
            "no-overlap validation is required; pass horizon_deltas and "
            "supported_signal_timeframes or set no_overlap_validated=True after external validation"
        )
    if decision_frequency not in horizon_deltas:
        raise ValueError("decision_frequency missing from horizon_deltas: " + decision_frequency)
    if combo_spec.return_horizon not in horizon_deltas:
        raise ValueError("return_horizon missing from horizon_deltas: " + combo_spec.return_horizon)
    expected_decision_frequency = non_overlapping_decision_frequency(
        combo_spec.feature_names,
        combo_spec.panel_frequency,
        combo_spec.return_horizon,
        horizon_deltas,
        supported_signal_timeframes,
    )
    if expected_decision_frequency != decision_frequency:
        raise ValueError(
            "decision_frequency must match combo_decision_frequency; got "
            f"{decision_frequency} vs {expected_decision_frequency}"
        )
    validate_no_overlap_design(
        [combo_spec], horizon_deltas, supported_signal_timeframes)


def evaluate_long_short_strategy(
    combo_spec: ComboSpec,
    signal_frame: pd.DataFrame,
    folds: Sequence,
    walk_forward_spec: Mapping[str, int],
    *,
    weight_scheme: str,
    feature_families: Mapping[str, str] | None,
    decision_frequency: str,
    n_buckets: int,
    min_cross_section: int,
    frequency_periods_per_year: Mapping[str, int | float],
    cost_multipliers: Sequence[float],
    taker_fee_rate: float,
    horizon_deltas: Mapping[str, pd.Timedelta] | None = None,
    supported_signal_timeframes: Sequence[str] | None = None,
    no_overlap_validated: bool = False,
    min_pair_corr_observations: int | None = None,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    validate_long_short_strategy_inputs(
        combo_spec=combo_spec,
        signal_frame=signal_frame,
        folds=folds,
        decision_frequency=decision_frequency,
        n_buckets=n_buckets,
        min_cross_section=min_cross_section,
        frequency_periods_per_year=frequency_periods_per_year,
        cost_multipliers=cost_multipliers,
        taker_fee_rate=taker_fee_rate,
        horizon_deltas=horizon_deltas,
        supported_signal_timeframes=supported_signal_timeframes,
        no_overlap_validated=no_overlap_validated,
    )
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    holding_frames: list[pd.DataFrame] = []
    component_features = feature_list_text(combo_spec.feature_names)
    train_columns = ["symbol", *combo_spec.feature_names, "forward_return"]
    signal_columns = ["symbol", *combo_spec.feature_names,
                      "forward_return", "strategy_forward_return"]

    for fold in folds:
        train_slice = select_dates(signal_frame[train_columns], fold, "train")
        train_stats = train_feature_stats(
            train_slice, combo_spec.feature_names, min_cross_section)
        if train_stats is None:
            continue
        feature_correlations = None
        correlation_diag = pd.DataFrame()
        if weight_scheme == "corr_discount_icir":
            feature_correlations, correlation_diag = train_cross_sectional_feature_correlation(
                train_slice,
                combo_spec.feature_names,
                min_cross_section=min_cross_section,
                min_pair_corr_observations=min_pair_corr_observations,
            )
            if not correlation_diag.empty and not (correlation_diag["status"] == "ok").all():
                continue
        directions = {feature_name: stat.direction
                      for feature_name, stat in train_stats.items()}
        _, feature_weights, weight_diagnostics = composite_weight_scores_weights_and_diagnostics(
            train_stats,
            weight_scheme,
            feature_families=feature_families,
            feature_correlations=feature_correlations,
        )
        if not correlation_diag.empty:
            weight_diagnostics.update(
                {
                    "correlation_pair_count": int(len(correlation_diag)),
                    "correlation_min_pair_observation_count": int(
                        correlation_diag["min_pair_corr_observations"].max()
                    ),
                    "correlation_min_pair_observed_count": int(
                        correlation_diag["pair_corr_observation_count"].min()
                    ),
                }
            )
        else:
            weight_diagnostics.update(
                {
                    "correlation_pair_count": 0,
                    "correlation_min_pair_observation_count": 0,
                    "correlation_min_pair_observed_count": 0,
                }
            )
        test_slice = select_dates(signal_frame[signal_columns], fold, "test")
        test_composite = build_composite_frame(
            test_slice,
            combo_spec.feature_names,
            directions,
            feature_weights,
            extra_columns=("forward_return", "strategy_forward_return"),
        )
        fold_rows, fold_holdings, fold_diagnostics = long_short_strategy_frames_for_fold(
            combo_spec,
            fold,
            test_composite,
            n_buckets=n_buckets,
            cost_multipliers=cost_multipliers,
            taker_fee_rate=taker_fee_rate,
            component_features=component_features,
        )
        for row in fold_rows:
            row.update(weight_diagnostics)
        rows.extend(fold_rows)
        diagnostics.extend(fold_diagnostics)
        if not fold_holdings.empty:
            holding_frames.append(fold_holdings)
    detail_frame = pd.DataFrame(rows)
    if detail_frame.empty:
        raise ValueError("long-short strategy produced no scored OOS decisions")
    holdings_frame = (
        pd.concat(holding_frames, ignore_index=True)
        if holding_frames
        else pd.DataFrame()
    )
    summary = summarize_long_short_strategy(
        combo_spec,
        detail_frame,
        diagnostics,
        walk_forward_spec,
        decision_frequency,
        frequency_periods_per_year,
        cost_multipliers,
    )
    return summary, detail_frame, holdings_frame


def validate_combo_signal_diagnostics_inputs(
    combo_spec: ComboSpec,
    signal_frame: pd.DataFrame,
    folds: Sequence,
    *,
    weight_scheme: str,
    control_columns: Sequence[str],
    decision_frequency: str,
    n_buckets: int,
    min_cross_section: int,
    frequency_periods_per_year: Mapping[str, int | float],
    horizon_deltas: Mapping[str, pd.Timedelta] | None,
    supported_signal_timeframes: Sequence[str] | None,
    no_overlap_validated: bool,
) -> None:
    """Fail closed before producing combo-signal diagnostic conclusion fields."""
    if combo_spec.weight_scheme != weight_scheme:
        raise ValueError(
            "weight_scheme must match combo_spec.weight_scheme; got "
            f"{weight_scheme} vs {combo_spec.weight_scheme}"
        )
    if not combo_spec.feature_names:
        raise ValueError("combo_spec.feature_names must not be empty")
    if decision_frequency not in frequency_periods_per_year:
        raise ValueError("decision_frequency missing from periods-per-year map: " + decision_frequency)
    if int(n_buckets) < 2:
        raise ValueError("n_buckets must be at least 2")
    if int(min_cross_section) < int(n_buckets):
        raise ValueError("min_cross_section must be >= n_buckets")
    if len(folds) == 0:
        raise ValueError("at least one walk-forward fold is required")
    control_columns = tuple(control_columns)
    if not control_columns:
        raise ValueError("control_columns must not be empty")

    required_columns = {
        "symbol",
        "forward_return",
        *combo_spec.feature_names,
        *control_columns,
    }
    missing = sorted(column for column in required_columns if column not in signal_frame.columns)
    if missing:
        raise ValueError("signal_frame missing required columns: " + ", ".join(missing))

    if no_overlap_validated:
        return
    if horizon_deltas is None or supported_signal_timeframes is None:
        raise ValueError(
            "no-overlap validation is required; pass horizon_deltas and "
            "supported_signal_timeframes or set no_overlap_validated=True after external validation"
        )
    if decision_frequency not in horizon_deltas:
        raise ValueError("decision_frequency missing from horizon_deltas: " + decision_frequency)
    if combo_spec.return_horizon not in horizon_deltas:
        raise ValueError("return_horizon missing from horizon_deltas: " + combo_spec.return_horizon)
    expected_decision_frequency = non_overlapping_decision_frequency(
        combo_spec.feature_names,
        combo_spec.panel_frequency,
        combo_spec.return_horizon,
        horizon_deltas,
        supported_signal_timeframes,
    )
    if expected_decision_frequency != decision_frequency:
        raise ValueError(
            "decision_frequency must match combo_decision_frequency; got "
            f"{decision_frequency} vs {expected_decision_frequency}"
        )
    validate_no_overlap_design(
        [combo_spec], horizon_deltas, supported_signal_timeframes)


def _decision_to_fold_map(frame: pd.DataFrame) -> dict[pd.Timestamp, int]:
    if frame.empty:
        return {}
    decision_fold = frame[["decision_ts", "fold_idx"]].drop_duplicates()
    return {
        pd.Timestamp(row.decision_ts): int(row.fold_idx)
        for row in decision_fold.itertuples(index=False)
    }


def _fold_metadata(fold) -> dict[str, object]:
    return {
        "fold_idx": fold.fold_idx,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
    }


def _select_fold_dates(frame: pd.DataFrame, fold, split: str) -> pd.DataFrame:
    date_values = frame.index.get_level_values(
        0) if isinstance(frame.index, pd.MultiIndex) else frame.index
    if split == "train":
        mask = (date_values >= fold.train_start) & (date_values <= fold.train_end)
    elif split == "test":
        mask = (date_values >= fold.test_start) & (date_values <= fold.test_end)
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    return frame.loc[mask]


def evaluate_combo_signal_diagnostics(
    combo_spec: ComboSpec,
    signal_frame: pd.DataFrame,
    folds: Sequence,
    walk_forward_spec: Mapping[str, int],
    *,
    weight_scheme: str,
    feature_families: Mapping[str, str] | None,
    control_columns: Sequence[str],
    decision_frequency: str,
    n_buckets: int,
    min_cross_section: int,
    frequency_periods_per_year: Mapping[str, int | float],
    horizon_deltas: Mapping[str, pd.Timedelta] | None = None,
    supported_signal_timeframes: Sequence[str] | None = None,
    hac_overlap_lags: int = 0,
    one_sided_t_threshold: float = 1.645,
    bucket_monotonic_threshold: float = 0.75,
    no_overlap_validated: bool = False,
    min_pair_corr_observations: int | None = None,
    _include_fm: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate OOS composite-signal IC, bucket, and Fama-MacBeth diagnostics.

    The composite signal uses the same train-only direction and weighting
    mechanics as ``evaluate_long_short_strategy``. The returned summary is a
    one-row DataFrame so callers can concatenate rows without reimplementing
    any conclusion metrics.
    """
    control_columns = tuple(control_columns)
    validate_combo_signal_diagnostics_inputs(
        combo_spec=combo_spec,
        signal_frame=signal_frame,
        folds=folds,
        weight_scheme=weight_scheme,
        control_columns=control_columns,
        decision_frequency=decision_frequency,
        n_buckets=n_buckets,
        min_cross_section=min_cross_section,
        frequency_periods_per_year=frequency_periods_per_year,
        horizon_deltas=horizon_deltas,
        supported_signal_timeframes=supported_signal_timeframes,
        no_overlap_validated=no_overlap_validated,
    )
    if int(hac_overlap_lags) < 0:
        raise ValueError("hac_overlap_lags must be non-negative")

    component_features = feature_list_text(combo_spec.feature_names)
    composite_frames: list[pd.DataFrame] = []
    fold_weight_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    train_columns = ["symbol", *combo_spec.feature_names, "forward_return"]
    test_columns = ["symbol", *combo_spec.feature_names, "forward_return", *control_columns]

    for fold in folds:
        train_slice = _select_fold_dates(signal_frame[train_columns], fold, "train")
        train_stats = train_feature_stats(
            train_slice, combo_spec.feature_names, min_cross_section)
        if train_stats is None:
            diagnostic_rows.append(
                {
                    "diagnostic_type": "train",
                    **_fold_metadata(fold),
                    "status": "no_train_stats",
                    "combo_id": combo_spec.combo_id,
                    "track": combo_spec.track,
                    "weight_scheme": combo_spec.weight_scheme,
                    "panel_frequency": combo_spec.panel_frequency,
                    "return_horizon": combo_spec.return_horizon,
                    "component_features": component_features,
                }
            )
            continue

        feature_correlations = None
        correlation_diag = pd.DataFrame()
        if weight_scheme == "corr_discount_icir":
            feature_correlations, correlation_diag = train_cross_sectional_feature_correlation(
                train_slice,
                combo_spec.feature_names,
                min_cross_section=min_cross_section,
                min_pair_corr_observations=min_pair_corr_observations,
            )
            if not correlation_diag.empty and not (correlation_diag["status"] == "ok").all():
                diagnostic_rows.append(
                    {
                        "diagnostic_type": "train",
                        **_fold_metadata(fold),
                        "status": "insufficient_pair_correlations",
                        "combo_id": combo_spec.combo_id,
                        "track": combo_spec.track,
                        "weight_scheme": combo_spec.weight_scheme,
                        "panel_frequency": combo_spec.panel_frequency,
                        "return_horizon": combo_spec.return_horizon,
                        "component_features": component_features,
                    }
                )
                continue

        directions = {feature_name: stat.direction
                      for feature_name, stat in train_stats.items()}
        weight_scores, feature_weights, weight_diagnostics = composite_weight_scores_weights_and_diagnostics(
            train_stats,
            weight_scheme,
            feature_families=feature_families,
            feature_correlations=feature_correlations,
        )
        if not correlation_diag.empty:
            weight_diagnostics.update(
                {
                    "correlation_pair_count": int(len(correlation_diag)),
                    "correlation_min_pair_observation_count": int(
                        correlation_diag["min_pair_corr_observations"].max()
                    ),
                    "correlation_min_pair_observed_count": int(
                        correlation_diag["pair_corr_observation_count"].min()
                    ),
                }
            )
        else:
            weight_diagnostics.update(
                {
                    "correlation_pair_count": 0,
                    "correlation_min_pair_observation_count": 0,
                    "correlation_min_pair_observed_count": 0,
                }
            )
        fold_train_mean_ic = float(
            sum(
                float(feature_weights[feature_name]) * float(stat.mean_ic) * float(stat.direction)
                for feature_name, stat in train_stats.items()
            )
        )
        for feature_name, stat in train_stats.items():
            fold_weight_rows.append(
                {
                    "combo_id": combo_spec.combo_id,
                    "track": combo_spec.track,
                    "weight_scheme": combo_spec.weight_scheme,
                    "panel_frequency": combo_spec.panel_frequency,
                    "return_horizon": combo_spec.return_horizon,
                    "component_features": component_features,
                    **_fold_metadata(fold),
                    "feature_name": feature_name,
                    "direction": stat.direction,
                    "train_mean_ic": stat.mean_ic,
                    "train_icir": stat.icir,
                    "train_hac_t_stat": stat.hac_t_stat,
                    "train_ic_observation_count": stat.observation_count,
                    "weight_score": float(weight_scores[feature_name]),
                    "feature_weight": float(feature_weights[feature_name]),
                    **weight_diagnostics,
                    "status": "ok",
                }
            )

        test_slice = _select_fold_dates(signal_frame[test_columns], fold, "test")
        fold_composite = build_composite_frame(
            test_slice,
            combo_spec.feature_names,
            directions,
            feature_weights,
            extra_columns=("forward_return", *control_columns),
        ).rename(columns={"composite_signal": "combo_signal"})
        if fold_composite.empty:
            continue
        decision_values = (
            fold_composite.index.get_level_values(0)
            if isinstance(fold_composite.index, pd.MultiIndex)
            else fold_composite.index
        )
        fold_composite = fold_composite.assign(
            combo_id=combo_spec.combo_id,
            track=combo_spec.track,
            weight_scheme=combo_spec.weight_scheme,
            panel_frequency=combo_spec.panel_frequency,
            return_horizon=combo_spec.return_horizon,
            component_features=component_features,
            fold_idx=fold.fold_idx,
            train_start=fold.train_start,
            train_end=fold.train_end,
            test_start=fold.test_start,
            test_end=fold.test_end,
            decision_ts=decision_values,
        )
        composite_frames.append(fold_composite)

    if not composite_frames:
        raise ValueError("combo-signal diagnostics produced no OOS composite signal rows")

    composite_detail = pd.concat(composite_frames, axis=0)
    decision_fold_map = _decision_to_fold_map(composite_detail)

    ic_diagnostics = rank_ic_diagnostics_for_frame(
        composite_detail[["symbol", "combo_signal", "forward_return"]],
        "combo_signal",
        min_cross_section,
    )
    ic_rows: list[dict[str, object]] = []
    for row in ic_diagnostics:
        row = dict(row)
        row["diagnostic_type"] = "ic"
        row["combo_id"] = combo_spec.combo_id
        row["track"] = combo_spec.track
        row["weight_scheme"] = combo_spec.weight_scheme
        row["panel_frequency"] = combo_spec.panel_frequency
        row["return_horizon"] = combo_spec.return_horizon
        row["component_features"] = component_features
        row["fold_idx"] = decision_fold_map.get(
            pd.Timestamp(row["decision_ts"]), -1)
        diagnostic_rows.append(dict(row))
        if row["status"] == "ok":
            ic_rows.append(
                {
                    "combo_id": combo_spec.combo_id,
                    "track": combo_spec.track,
                    "weight_scheme": combo_spec.weight_scheme,
                    "panel_frequency": combo_spec.panel_frequency,
                    "return_horizon": combo_spec.return_horizon,
                    "component_features": component_features,
                    "fold_idx": row["fold_idx"],
                    "decision_ts": row["decision_ts"],
                    "rank_ic": row["raw_rank_ic"],
                    "raw_rank_ic": row["raw_rank_ic"],
                    "train_mean_ic": fold_train_mean_ic,
                    "cross_section_size": row["cross_section_size"],
                }
            )
    ic_detail = pd.DataFrame(ic_rows)

    bucket_detail, bucket_diagnostics = bucket_diagnostics_for_frame(
        composite_detail[["symbol", "combo_signal", "forward_return"]],
        "combo_signal",
        direction=1,
        n_buckets=n_buckets,
    )
    if not bucket_detail.empty:
        bucket_detail = bucket_detail.assign(
            combo_id=combo_spec.combo_id,
            track=combo_spec.track,
            weight_scheme=combo_spec.weight_scheme,
            panel_frequency=combo_spec.panel_frequency,
            return_horizon=combo_spec.return_horizon,
            component_features=component_features,
            fold_idx=bucket_detail["decision_ts"].map(
                lambda value: decision_fold_map.get(pd.Timestamp(value), -1)
            ).astype(int),
        )
    for row in bucket_diagnostics:
        row = dict(row)
        row["diagnostic_type"] = "bucket"
        row["combo_id"] = combo_spec.combo_id
        row["track"] = combo_spec.track
        row["weight_scheme"] = combo_spec.weight_scheme
        row["panel_frequency"] = combo_spec.panel_frequency
        row["return_horizon"] = combo_spec.return_horizon
        row["component_features"] = component_features
        row["fold_idx"] = decision_fold_map.get(
            pd.Timestamp(row["decision_ts"]), -1)
        diagnostic_rows.append(row)

    fm_diagnostics = (
        fama_macbeth_diagnostics_for_frame_slice(
            composite_detail[["symbol", "combo_signal", "forward_return", *control_columns]],
            "combo_signal",
            direction=1,
            control_columns=control_columns,
            min_cross_section=min_cross_section,
        )
        if _include_fm
        else []
    )
    fm_rows: list[dict[str, object]] = []
    for row in fm_diagnostics:
        row = dict(row)
        row["diagnostic_type"] = "fm"
        row["combo_id"] = combo_spec.combo_id
        row["track"] = combo_spec.track
        row["weight_scheme"] = combo_spec.weight_scheme
        row["panel_frequency"] = combo_spec.panel_frequency
        row["return_horizon"] = combo_spec.return_horizon
        row["component_features"] = component_features
        row["fold_idx"] = decision_fold_map.get(
            pd.Timestamp(row["decision_ts"]), -1)
        diagnostic_rows.append(dict(row))
        if row["status"] == "ok":
            fm_rows.append(row)
    fm_detail = pd.DataFrame(fm_rows)

    ic_summary = summarize_ic_series(
        combo_spec.panel_frequency,
        combo_spec.return_horizon,
        combo_spec.combo_id,
        ic_detail,
        walk_forward_spec,
        ic_diagnostics,
        hac_overlap_lags=hac_overlap_lags,
    )
    bucket_summary = summarize_bucket_backtest(
        combo_spec.panel_frequency,
        combo_spec.return_horizon,
        combo_spec.combo_id,
        bucket_detail,
        walk_forward_spec,
        bucket_diagnostics,
        n_buckets=n_buckets,
        frequency_periods_per_year=frequency_periods_per_year,
        annualization_frequency=combo_spec.return_horizon,
    )
    fm_summary = (
        summarize_fama_macbeth(
            combo_spec.panel_frequency,
            combo_spec.return_horizon,
            combo_spec.combo_id,
            fm_detail,
            walk_forward_spec,
            fm_diagnostics,
            control_columns,
            hac_overlap_lags=hac_overlap_lags,
        )
        if _include_fm
        else None
    )

    summary: dict[str, object] = {
        "combo_id": combo_spec.combo_id,
        "track": combo_spec.track,
        "weight_scheme": combo_spec.weight_scheme,
        "panel_frequency": combo_spec.panel_frequency,
        "return_horizon": combo_spec.return_horizon,
        "component_features": component_features,
        "n_components": len(combo_spec.feature_names),
        "train_days": walk_forward_spec["train_days"],
        "test_days": walk_forward_spec["test_days"],
        "embargo_days": walk_forward_spec["embargo_days"],
        "step_days": walk_forward_spec["step_days"],
        "n_folds": int(composite_detail["fold_idx"].nunique()),
        "scored_decision_count": int(ic_summary["scored_decision_count"]),
        "combo_ic_mean": ic_summary["mean_ic"],
        "combo_icir": ic_summary["icir"],
        "combo_ic_hac_t_stat": ic_summary["hac_t_stat"],
        "combo_ic_hac_lags": ic_summary["hac_lags"],
        "combo_ic_positive_share": ic_summary["ic_positive_share"],
        "combo_ic_observation_count": ic_summary["ic_observation_count"],
        "combo_bucket_spread_mean_return": bucket_summary["spread_mean_return"],
        "combo_bucket_spread_sharpe": bucket_summary["spread_sharpe"],
        "combo_bucket_spread_positive_share": bucket_summary["spread_positive_share"],
        "combo_bucket_monotonic_pair_pass_share": bucket_summary["monotonic_pair_pass_share"],
        "combo_bucket_scored_decision_count": bucket_summary["scored_decision_count"],
    }
    stage1 = bool(
        float(summary["combo_ic_mean"]) > 0.0
        and float(summary["combo_ic_hac_t_stat"]) >= float(one_sided_t_threshold)
    )
    stage2 = bool(
        float(summary["combo_bucket_spread_mean_return"]) > 0.0
        and float(summary["combo_bucket_monotonic_pair_pass_share"])
        >= float(bucket_monotonic_threshold)
    )
    summary.update(
        {
            "combo_stage1_ic_support_raw": stage1,
            "combo_stage2_bucket_support": stage2,
            "combo_two_gate_support_raw": stage1 and stage2,
            "combo_ic_one_sided_p_value": research_stats.normal_one_sided_p_value(
                float(summary["combo_ic_hac_t_stat"])
            ),
        }
    )
    if _include_fm:
        assert fm_summary is not None
        summary.update(
            {
                "combo_fm_mean_gamma": fm_summary["mean_gamma"],
                "combo_fm_hac_t_stat": fm_summary["hac_t_stat"],
                "combo_fm_hac_lags": fm_summary["hac_lags"],
                "combo_fm_gamma_positive_share": fm_summary["gamma_positive_share"],
                "combo_fm_gamma_observation_count": fm_summary["gamma_observation_count"],
                "combo_fm_scored_decision_count": fm_summary["scored_decision_count"],
            }
        )
        for column in control_columns:
            key = f"mean_{column}_gamma"
            summary[f"combo_fm_{key}"] = fm_summary.get(key, float("nan"))
        stage3 = bool(
            float(summary["combo_fm_mean_gamma"]) > 0.0
            and float(summary["combo_fm_hac_t_stat"]) >= float(one_sided_t_threshold)
        )
        summary.update(
            {
                "combo_stage3_fm_support_raw": stage3,
                "combo_three_gate_support_raw": stage1 and stage2 and stage3,
                "combo_fm_one_sided_p_value": research_stats.normal_one_sided_p_value(
                    float(summary["combo_fm_hac_t_stat"])
                ),
            }
        )

    metadata_columns = [
        "combo_id",
        "track",
        "weight_scheme",
        "panel_frequency",
        "return_horizon",
        "component_features",
    ]
    composite_columns = [
        *metadata_columns,
        "fold_idx",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "decision_ts",
        "symbol",
        "combo_signal",
        "forward_return",
        *control_columns,
    ]
    composite_detail = composite_detail[composite_columns].reset_index(drop=True)
    fold_weight_frame = pd.DataFrame(fold_weight_rows)
    diagnostics_frame = pd.DataFrame(diagnostic_rows)
    return (
        pd.DataFrame([summary]),
        composite_detail,
        ic_detail,
        bucket_detail,
        fm_detail,
        fold_weight_frame,
        diagnostics_frame,
    )


def evaluate_combo_signal_two_gate_diagnostics(
    *args,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate combo IC and bucket gates without executing Fama-MacBeth."""
    if "_include_fm" in kwargs:
        raise ValueError("_include_fm is internal and cannot be supplied")
    summary, composite, ic, bucket, fm, weights, diagnostics = evaluate_combo_signal_diagnostics(
        *args,
        **kwargs,
        _include_fm=False,
    )
    if not fm.empty or (not diagnostics.empty and diagnostics["diagnostic_type"].eq("fm").any()):
        raise AssertionError("Two-gate diagnostics unexpectedly produced FM output")
    return summary, composite, ic, bucket, weights, diagnostics


def apply_combo_signal_diagnostic_fdr(
    summary_frame: pd.DataFrame,
    *,
    q_threshold: float = 0.10,
) -> pd.DataFrame:
    """Add BH-FDR sensitivity columns to combo-signal diagnostic summaries."""
    if not 0.0 < float(q_threshold) < 1.0:
        raise ValueError("q_threshold must be between 0 and 1")
    required = {"combo_ic_one_sided_p_value",
                "combo_fm_one_sided_p_value", "combo_stage2_bucket_support"}
    missing = sorted(required.difference(summary_frame.columns))
    if missing:
        raise ValueError("summary_frame missing required columns: " + ", ".join(missing))
    result = summary_frame.copy()
    result["combo_ic_bh_fdr_q"] = research_stats.benjamini_hochberg_q_values(
        result["combo_ic_one_sided_p_value"].to_numpy(dtype=float)
    )
    result["combo_fm_bh_fdr_q"] = research_stats.benjamini_hochberg_q_values(
        result["combo_fm_one_sided_p_value"].to_numpy(dtype=float)
    )
    ic_pass = (
        result["combo_ic_mean"].astype(float).gt(0.0)
        & result["combo_ic_bh_fdr_q"].astype(float).le(float(q_threshold))
    )
    fm_pass = (
        result["combo_fm_mean_gamma"].astype(float).gt(0.0)
        & result["combo_fm_bh_fdr_q"].astype(float).le(float(q_threshold))
    )
    bucket_pass = result["combo_stage2_bucket_support"].astype(bool)
    result["combo_two_gate_support_fdr_10pct"] = ic_pass & bucket_pass
    result["combo_three_gate_support_fdr_10pct"] = ic_pass & bucket_pass & fm_pass
    return result


def _signed_notional_side(value: float) -> str:
    if value > 0.0:
        return "long"
    if value < 0.0:
        return "short"
    return "flat"


def validate_live_like_min_notional_replay_inputs(
    target_holdings: pd.DataFrame,
    min_notional_by_symbol: Mapping[str, float],
    *,
    account_equity: float,
    target_gross_notional: float,
    exchange_leverage: float,
    taker_fee_rate: float,
    cost_multipliers: Sequence[float],
    frequency_periods_per_year: Mapping[str, int | float],
) -> None:
    """Fail closed before producing live-like replay conclusion fields."""
    required = {
        "combo_id",
        "fold_idx",
        "decision_ts",
        "symbol",
        "weight",
        "strategy_forward_return",
        "panel_frequency",
        "return_horizon",
    }
    missing = sorted(required.difference(target_holdings.columns))
    if missing:
        raise ValueError("target_holdings missing required columns: " + ", ".join(missing))
    if target_holdings.empty:
        raise ValueError("target_holdings must not be empty")
    if float(account_equity) <= 0.0:
        raise ValueError("account_equity must be positive")
    if float(target_gross_notional) <= 0.0:
        raise ValueError("target_gross_notional must be positive")
    if float(exchange_leverage) <= 0.0:
        raise ValueError("exchange_leverage must be positive")
    if float(taker_fee_rate) < 0.0:
        raise ValueError("taker_fee_rate must be non-negative")
    if not cost_multipliers:
        raise ValueError("cost_multipliers must not be empty")
    for multiplier in cost_multipliers:
        if float(multiplier) <= 0.0:
            raise ValueError("cost_multipliers must be positive")
    if not min_notional_by_symbol:
        raise ValueError("min_notional_by_symbol must not be empty")
    bad_min_notional = [
        str(symbol)
        for symbol, value in min_notional_by_symbol.items()
        if float(value) <= 0.0
    ]
    if bad_min_notional:
        raise ValueError("min notional must be positive for: " + ", ".join(sorted(bad_min_notional)))
    missing_min_notional = sorted(
        set(target_holdings["symbol"].astype(str)).difference(
            {str(symbol) for symbol in min_notional_by_symbol}
        )
    )
    if missing_min_notional:
        raise ValueError("min_notional_by_symbol missing symbols: " + ", ".join(missing_min_notional))
    missing_frequency = sorted(
        set(target_holdings["panel_frequency"].astype(str)).difference(frequency_periods_per_year)
    )
    if missing_frequency:
        raise ValueError("panel_frequency missing from periods-per-year map: " + ", ".join(missing_frequency))


def _execute_min_notional_transition(
    *,
    symbol: str,
    previous_notional: float,
    target_notional: float,
    min_notional: float,
    epsilon: float,
) -> dict[str, object]:
    requested_delta = float(target_notional - previous_notional)
    if abs(requested_delta) <= epsilon:
        return {
            "symbol": symbol,
            "previous_notional": float(previous_notional),
            "target_notional": float(target_notional),
            "actual_notional": float(previous_notional),
            "requested_delta_notional": requested_delta,
            "executed_order_notional": 0.0,
            "filtered_order_notional": 0.0,
            "min_notional": float(min_notional),
            "status": "unchanged",
        }

    if abs(target_notional) <= epsilon:
        return {
            "symbol": symbol,
            "previous_notional": float(previous_notional),
            "target_notional": 0.0,
            "actual_notional": 0.0,
            "requested_delta_notional": requested_delta,
            "executed_order_notional": abs(float(previous_notional)),
            "filtered_order_notional": 0.0,
            "min_notional": float(min_notional),
            "status": "close",
        }

    previous_side = _signed_notional_side(float(previous_notional))
    target_side = _signed_notional_side(float(target_notional))
    if previous_side == "flat":
        if abs(target_notional) + epsilon >= min_notional:
            actual_notional = float(target_notional)
            executed = abs(actual_notional)
            filtered = 0.0
            status = "open"
        else:
            actual_notional = 0.0
            executed = 0.0
            filtered = abs(float(target_notional))
            status = "filtered_open"
    elif previous_side != target_side:
        close_notional = abs(float(previous_notional))
        if abs(target_notional) + epsilon >= min_notional:
            actual_notional = float(target_notional)
            executed = close_notional + abs(actual_notional)
            filtered = 0.0
            status = "flip"
        else:
            actual_notional = 0.0
            executed = close_notional
            filtered = abs(float(target_notional))
            status = "flip_close_filtered_open"
    else:
        if abs(requested_delta) + epsilon >= min_notional:
            actual_notional = float(target_notional)
            executed = abs(requested_delta)
            filtered = 0.0
            status = "adjust"
        else:
            actual_notional = float(previous_notional)
            executed = 0.0
            filtered = abs(requested_delta)
            status = "filtered_adjust"

    return {
        "symbol": symbol,
        "previous_notional": float(previous_notional),
        "target_notional": float(target_notional),
        "actual_notional": float(actual_notional),
        "requested_delta_notional": requested_delta,
        "executed_order_notional": float(executed),
        "filtered_order_notional": float(filtered),
        "min_notional": float(min_notional),
        "status": status,
    }


def live_like_min_notional_replay(
    target_holdings: pd.DataFrame,
    min_notional_by_symbol: Mapping[str, float],
    *,
    account_equity: float,
    target_gross_notional: float,
    exchange_leverage: float,
    taker_fee_rate: float,
    cost_multipliers: Sequence[float],
    frequency_periods_per_year: Mapping[str, int | float],
    epsilon: float = 1e-9,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replay target long-short holdings with exchange minimum-order constraints.

    The entry assumes target weights are fixed before each OOS decision. Opening
    and same-side adjustment orders below each symbol's minimum notional are
    skipped and the previous live-like position is retained. Full closes are
    allowed even below the minimum notional. Positions are reset at the end of
    each fold via terminal close cost, matching the OOS fold boundary.
    """
    validate_live_like_min_notional_replay_inputs(
        target_holdings,
        min_notional_by_symbol,
        account_equity=account_equity,
        target_gross_notional=target_gross_notional,
        exchange_leverage=exchange_leverage,
        taker_fee_rate=taker_fee_rate,
        cost_multipliers=cost_multipliers,
        frequency_periods_per_year=frequency_periods_per_year,
    )
    working = target_holdings.copy()
    working["decision_ts"] = pd.to_datetime(working["decision_ts"], utc=True)
    working["symbol"] = working["symbol"].astype(str)
    min_notional = {str(symbol): float(value) for symbol, value in min_notional_by_symbol.items()}

    meta_columns = [
        column
        for column in [
            "combo_id",
            "track",
            "weight_scheme",
            "panel_frequency",
            "return_horizon",
            "component_features",
            "fold_idx",
        ]
        if column in working.columns
    ]
    strategy_group_columns = [
        column
        for column in [
            "combo_id",
            "track",
            "weight_scheme",
            "panel_frequency",
            "return_horizon",
            "component_features",
        ]
        if column in working.columns
    ]
    group_columns = [*strategy_group_columns, "fold_idx"]
    decision_rows: list[dict[str, object]] = []
    order_rows: list[dict[str, object]] = []
    holding_rows: list[dict[str, object]] = []

    sorted_working = working.sort_values(
        [*group_columns, "decision_ts", "symbol"],
        kind="mergesort",
    )
    for group_key, group in sorted_working.groupby(group_columns, sort=False, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_meta = dict(zip(group_columns, group_key, strict=True))
        previous_positions: dict[str, float] = {}
        decisions = list(group.groupby("decision_ts", sort=True))
        for decision_number, (decision_ts, decision_frame) in enumerate(decisions):
            target_by_symbol = (
                decision_frame.assign(target_notional=decision_frame["weight"].astype(float) * float(target_gross_notional))
                .groupby("symbol", sort=False)["target_notional"]
                .sum()
                .to_dict()
            )
            returns_by_symbol = (
                decision_frame.groupby("symbol", sort=False)["strategy_forward_return"]
                .first()
                .astype(float)
                .to_dict()
            )
            target_weight_by_symbol = (
                decision_frame.groupby("symbol", sort=False)["weight"]
                .sum()
                .astype(float)
                .to_dict()
            )
            symbols = sorted(set(previous_positions).union(target_by_symbol))
            actual_positions: dict[str, float] = {}
            decision_orders: list[dict[str, object]] = []
            for symbol in symbols:
                transition = _execute_min_notional_transition(
                    symbol=symbol,
                    previous_notional=float(previous_positions.get(symbol, 0.0)),
                    target_notional=float(target_by_symbol.get(symbol, 0.0)),
                    min_notional=float(min_notional[symbol]),
                    epsilon=float(epsilon),
                )
                if abs(float(transition["actual_notional"])) > epsilon:
                    actual_positions[symbol] = float(transition["actual_notional"])
                decision_orders.append(transition)

            missing_returns = sorted(symbol for symbol in actual_positions if symbol not in returns_by_symbol)
            if missing_returns:
                raise ValueError(
                    f"actual positions missing strategy_forward_return at {decision_ts}: "
                    + ", ".join(missing_returns)
                )

            gross_pnl_usd = sum(
                float(notional) * float(returns_by_symbol[symbol])
                for symbol, notional in actual_positions.items()
            )
            target_gross = sum(abs(float(value)) for value in target_by_symbol.values())
            target_net = sum(float(value) for value in target_by_symbol.values())
            actual_gross = sum(abs(float(value)) for value in actual_positions.values())
            actual_net = sum(float(value) for value in actual_positions.values())
            executed_order_notional = sum(float(row["executed_order_notional"]) for row in decision_orders)
            filtered_order_notional = sum(float(row["filtered_order_notional"]) for row in decision_orders)
            filtered_order_count = sum(1 for row in decision_orders if str(row["status"]).startswith("filtered") or str(row["status"]) == "flip_close_filtered_open")
            executed_order_count = sum(1 for row in decision_orders if float(row["executed_order_notional"]) > epsilon)
            order_attempt_count = sum(
                1
                for row in decision_orders
                if abs(float(row["requested_delta_notional"])) > epsilon
            )
            terminal_close_notional = 0.0
            if decision_number == len(decisions) - 1:
                terminal_close_notional = float(actual_gross)
            charged_order_notional = float(executed_order_notional + terminal_close_notional)

            row = {
                **group_meta,
                "decision_ts": decision_ts,
                "target_gross_notional": float(target_gross),
                "actual_gross_notional": float(actual_gross),
                "target_net_notional": float(target_net),
                "actual_net_notional": float(actual_net),
                "target_abs_net_exposure_share": abs(float(target_net)) / float(target_gross_notional),
                "actual_abs_net_exposure_share": abs(float(actual_net)) / float(target_gross_notional),
                "actual_vs_target_gross_ratio": float(actual_gross) / float(target_gross_notional),
                "margin_required": float(actual_gross) / float(exchange_leverage),
                "margin_utilization": (float(actual_gross) / float(exchange_leverage)) / float(account_equity),
                "target_long_count": int(sum(1 for value in target_by_symbol.values() if float(value) > epsilon)),
                "target_short_count": int(sum(1 for value in target_by_symbol.values() if float(value) < -epsilon)),
                "actual_long_count": int(sum(1 for value in actual_positions.values() if float(value) > epsilon)),
                "actual_short_count": int(sum(1 for value in actual_positions.values() if float(value) < -epsilon)),
                "executed_order_count": int(executed_order_count),
                "filtered_order_count": int(filtered_order_count),
                "order_attempt_count": int(order_attempt_count),
                "executed_order_notional": float(executed_order_notional),
                "filtered_order_notional": float(filtered_order_notional),
                "terminal_close_notional": float(terminal_close_notional),
                "charged_order_notional": float(charged_order_notional),
                "charged_turnover_on_target_gross": float(charged_order_notional) / float(target_gross_notional),
                "gross_pnl_usd": float(gross_pnl_usd),
                "gross_return_on_target_gross": float(gross_pnl_usd) / float(target_gross_notional),
                "gross_return_on_equity": float(gross_pnl_usd) / float(account_equity),
                "weight_abs_error_sum": sum(
                    abs(float(actual_positions.get(symbol, 0.0)) - float(target_by_symbol.get(symbol, 0.0)))
                    for symbol in set(actual_positions).union(target_by_symbol)
                ) / float(target_gross_notional),
            }
            for multiplier in cost_multipliers:
                label = scenario_label(float(multiplier))
                cost_usd = float(charged_order_notional) * float(taker_fee_rate) * float(multiplier)
                net_pnl_usd = float(gross_pnl_usd) - cost_usd
                row[f"cost_{label}_usd"] = cost_usd
                row[f"net_pnl_{label}_usd"] = net_pnl_usd
                row[f"net_return_{label}_on_target_gross"] = net_pnl_usd / float(target_gross_notional)
                row[f"net_return_{label}_on_equity"] = net_pnl_usd / float(account_equity)
            decision_rows.append(row)

            for transition in decision_orders:
                order_rows.append(
                    {
                        **group_meta,
                        "decision_ts": decision_ts,
                        **transition,
                    }
                )
            if terminal_close_notional > epsilon:
                for symbol, notional in actual_positions.items():
                    order_rows.append(
                        {
                            **group_meta,
                            "decision_ts": decision_ts,
                            "symbol": symbol,
                            "previous_notional": float(notional),
                            "target_notional": 0.0,
                            "actual_notional": 0.0,
                            "requested_delta_notional": -float(notional),
                            "executed_order_notional": abs(float(notional)),
                            "filtered_order_notional": 0.0,
                            "min_notional": float(min_notional[symbol]),
                            "status": "terminal_close",
                        }
                    )
            for symbol, notional in actual_positions.items():
                target_notional = float(target_by_symbol.get(symbol, 0.0))
                strategy_return = float(returns_by_symbol[symbol])
                holding_rows.append(
                    {
                        **group_meta,
                        "decision_ts": decision_ts,
                        "symbol": symbol,
                        "target_notional": target_notional,
                        "actual_notional": float(notional),
                        "target_weight": float(target_weight_by_symbol.get(symbol, 0.0)),
                        "actual_weight": float(notional) / float(target_gross_notional),
                        "actual_side": _signed_notional_side(float(notional)),
                        "strategy_forward_return": strategy_return,
                        "pnl_usd": float(notional) * strategy_return,
                        "contribution_on_target_gross": float(notional) * strategy_return / float(target_gross_notional),
                        "contribution_on_equity": float(notional) * strategy_return / float(account_equity),
                    }
                )
            previous_positions = {} if terminal_close_notional > epsilon else actual_positions

    detail = pd.DataFrame(decision_rows)
    orders = pd.DataFrame(order_rows)
    actual_holdings = pd.DataFrame(holding_rows)
    summary_rows: list[dict[str, object]] = []
    summary_group_columns = [
        column
        for column in strategy_group_columns
        if column in detail.columns
    ]
    for _, combo_detail in detail.groupby(summary_group_columns, sort=False, dropna=False):
        decision_frequency = str(combo_detail["panel_frequency"].iloc[0])
        summary_row: dict[str, object] = {
            "combo_id": combo_detail["combo_id"].iloc[0],
            "track": combo_detail["track"].iloc[0] if "track" in combo_detail else "",
            "weight_scheme": combo_detail["weight_scheme"].iloc[0] if "weight_scheme" in combo_detail else "",
            "panel_frequency": decision_frequency,
            "return_horizon": combo_detail["return_horizon"].iloc[0] if "return_horizon" in combo_detail else "",
            "component_features": combo_detail["component_features"].iloc[0] if "component_features" in combo_detail else "",
            "account_equity": float(account_equity),
            "target_gross_notional": float(target_gross_notional),
            "exchange_leverage": float(exchange_leverage),
            "n_folds": int(combo_detail["fold_idx"].nunique()) if "fold_idx" in combo_detail else 0,
            "decision_count": int(len(combo_detail)),
            "mean_actual_gross_notional": float(combo_detail["actual_gross_notional"].mean()),
            "min_actual_gross_notional": float(combo_detail["actual_gross_notional"].min()),
            "max_actual_gross_notional": float(combo_detail["actual_gross_notional"].max()),
            "mean_actual_vs_target_gross_ratio": float(combo_detail["actual_vs_target_gross_ratio"].mean()),
            "mean_abs_net_exposure_share": float(combo_detail["actual_abs_net_exposure_share"].mean()),
            "max_abs_net_exposure_share": float(combo_detail["actual_abs_net_exposure_share"].max()),
            "mean_weight_abs_error_sum": float(combo_detail["weight_abs_error_sum"].mean()),
            "max_weight_abs_error_sum": float(combo_detail["weight_abs_error_sum"].max()),
            "mean_actual_long_count": float(combo_detail["actual_long_count"].mean()),
            "mean_actual_short_count": float(combo_detail["actual_short_count"].mean()),
            "filtered_order_count": int(combo_detail["filtered_order_count"].sum()),
            "executed_order_count": int(combo_detail["executed_order_count"].sum()),
            "order_attempt_count": int(combo_detail["order_attempt_count"].sum()),
            "filtered_order_notional": float(combo_detail["filtered_order_notional"].sum()),
            "executed_order_notional": float(combo_detail["executed_order_notional"].sum()),
            "terminal_close_notional": float(combo_detail["terminal_close_notional"].sum()),
            "mean_charged_order_notional": float(combo_detail["charged_order_notional"].mean()),
            "mean_charged_turnover_on_target_gross": float(combo_detail["charged_turnover_on_target_gross"].mean()),
            "max_margin_required": float(combo_detail["margin_required"].max()),
            "max_margin_utilization": float(combo_detail["margin_utilization"].max()),
            "gross_annualized_return_on_equity": annualized_mean_return(
                combo_detail["gross_return_on_equity"], decision_frequency, frequency_periods_per_year
            ),
            "gross_sharpe_on_equity": annualized_sharpe_for_frequency(
                combo_detail["gross_return_on_equity"], decision_frequency, frequency_periods_per_year
            ),
            "gross_max_drawdown_on_equity": max_drawdown_from_returns(combo_detail["gross_return_on_equity"]),
            "gross_fold_positive_share_on_equity": fold_positive_share_from_returns(combo_detail, "gross_return_on_equity"),
        }
        attempts = int(summary_row["order_attempt_count"])
        summary_row["filtered_order_share"] = (
            float(summary_row["filtered_order_count"]) / attempts if attempts else 0.0
        )
        for multiplier in cost_multipliers:
            label = scenario_label(float(multiplier))
            net_col = f"net_return_{label}_on_equity"
            summary_row[f"cost_{label}_mean_usd"] = float(combo_detail[f"cost_{label}_usd"].mean())
            summary_row[f"net_{label}_annualized_return_on_equity"] = annualized_mean_return(
                combo_detail[net_col], decision_frequency, frequency_periods_per_year
            )
            summary_row[f"net_{label}_sharpe_on_equity"] = annualized_sharpe_for_frequency(
                combo_detail[net_col], decision_frequency, frequency_periods_per_year
            )
            summary_row[f"net_{label}_max_drawdown_on_equity"] = max_drawdown_from_returns(combo_detail[net_col])
            summary_row[f"net_{label}_fold_positive_share_on_equity"] = fold_positive_share_from_returns(combo_detail, net_col)
        summary_rows.append(summary_row)
    summary = pd.DataFrame(summary_rows)
    return summary, detail, orders, actual_holdings


def feature_signal_timeframe(
    feature_name: str,
    panel_frequency: str,
    supported_signal_timeframes: Sequence[str],
    suffix_panel_frequencies: set[str] | frozenset[str] | None = None,
) -> str:
    if suffix_panel_frequencies is not None and panel_frequency not in suffix_panel_frequencies:
        return panel_frequency
    for signal_timeframe in sorted(supported_signal_timeframes, key=len, reverse=True):
        if feature_name.endswith(f"__{signal_timeframe}"):
            return signal_timeframe
    return panel_frequency


def features_decision_frequency(
    feature_names: Sequence[str],
    panel_frequency: str,
    horizon_deltas: Mapping[str, pd.Timedelta],
    supported_signal_timeframes: Sequence[str],
) -> str:
    signal_timeframes = [
        panel_frequency,
        *[
        feature_signal_timeframe(
            feature_name, panel_frequency, supported_signal_timeframes)
        for feature_name in feature_names
        ],
    ]
    if not signal_timeframes:
        raise ValueError("feature_names must not be empty")
    return max(signal_timeframes, key=lambda timeframe: horizon_deltas[timeframe])


def combo_decision_frequency(
    combo_spec: ComboSpec,
    horizon_deltas: Mapping[str, pd.Timedelta],
    supported_signal_timeframes: Sequence[str],
) -> str:
    return features_decision_frequency(combo_spec.feature_names, combo_spec.panel_frequency, horizon_deltas, supported_signal_timeframes)


def non_overlapping_decision_frequency(
    feature_names: Sequence[str],
    panel_frequency: str,
    return_horizon: str,
    horizon_deltas: Mapping[str, pd.Timedelta],
    supported_signal_timeframes: Sequence[str],
) -> str:
    """Return the only legal continuous-holding decision frequency.

    Continuous strategies rebalance exactly at the declared return horizon.
    Every native signal timeframe must divide that horizon exactly.
    """
    if return_horizon not in horizon_deltas:
        raise ValueError("return_horizon missing from horizon_deltas: " + return_horizon)
    horizon_delta = pd.Timedelta(horizon_deltas[return_horizon])
    for feature_name in feature_names:
        timeframe = feature_signal_timeframe(
            feature_name,
            panel_frequency,
            supported_signal_timeframes,
        )
        signal_delta = pd.Timedelta(horizon_deltas[timeframe])
        if signal_delta > horizon_delta or horizon_delta % signal_delta != pd.Timedelta(0):
            raise ValueError(
                f"signal timeframe {timeframe} for {feature_name} is not an exact divisor "
                f"of return horizon {return_horizon}"
            )
    return return_horizon


def decision_timestamps_aligned_to_frequency(
    decision_values: Sequence[pd.Timestamp] | pd.DatetimeIndex | pd.Series,
    frequency: str,
    horizon_deltas: Mapping[str, pd.Timedelta],
    *,
    anchor_minute: int = 0,
) -> np.ndarray:
    if frequency not in horizon_deltas:
        raise ValueError("frequency missing from horizon_deltas: " + frequency)
    delta = pd.Timedelta(horizon_deltas[frequency])
    if delta <= pd.Timedelta(0):
        raise ValueError("frequency delta must be positive")
    index = pd.DatetimeIndex(decision_values)
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    anchors = index.normalize() + pd.Timedelta(minutes=int(anchor_minute))
    offsets = index - anchors
    return (offsets % delta) == pd.Timedelta(0)


def filter_frame_to_decision_frequency(
    frame: pd.DataFrame,
    frequency: str,
    horizon_deltas: Mapping[str, pd.Timedelta],
    *,
    anchor_minute: int = 0,
) -> pd.DataFrame:
    """Filter a cross-sectional frame to UTC-aligned decision timestamps."""
    if isinstance(frame.index, pd.MultiIndex):
        level_name = "decision_ts" if "decision_ts" in frame.index.names else frame.index.names[0]
        decision_values = frame.index.get_level_values(level_name)
    elif isinstance(frame.index, pd.DatetimeIndex):
        decision_values = frame.index
    elif "decision_ts" in frame.columns:
        decision_values = pd.DatetimeIndex(frame["decision_ts"])
    else:
        raise ValueError("frame must have a decision_ts index level, DatetimeIndex, or decision_ts column")
    mask = decision_timestamps_aligned_to_frequency(
        decision_values,
        frequency,
        horizon_deltas,
        anchor_minute=anchor_minute,
    )
    return frame.loc[np.asarray(mask)].copy()


def validate_no_overlap_design(
    combo_specs: Sequence[ComboSpec],
    horizon_deltas: Mapping[str, pd.Timedelta],
    supported_signal_timeframes: Sequence[str],
) -> None:
    for combo_spec in combo_specs:
        non_overlapping_decision_frequency(
            combo_spec.feature_names,
            combo_spec.panel_frequency,
            combo_spec.return_horizon,
            horizon_deltas,
            supported_signal_timeframes,
        )


def validated_executable_return_adapter(
    frame: pd.DataFrame,
    *,
    return_horizon: str,
    decision_frequency: str,
    horizon_deltas: Mapping[str, pd.Timedelta],
    execution_delay_minutes: int,
) -> pd.DataFrame:
    """Adapt a validated execution ledger to legacy internal column names.

    Public research callers must provide the explicit execution ledger. The
    legacy names exist only inside the established strategy engine and are not
    accepted from callers through this entry.
    """
    working = frame.copy()
    if "decision_ts" not in working.columns:
        if working.index.name != "decision_ts":
            raise ValueError("executable frame missing required column: decision_ts")
        working["decision_ts"] = pd.DatetimeIndex(working.index)
    required = {
        "signal_timeframes", "native_bar_end_ts", "signal_bar_end_ts",
        "availability_ts", "data_observed_ts", "decision_interval",
        "order_submit_ts", "execution_ts", "execution_open_time",
        "next_execution_ts", "return_horizon", "holding_interval",
        "exit_rule", "score_order", "entry_price", "exit_price",
        "execution_price", "next_execution_price", "executable_return",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("executable frame missing required columns: " + ", ".join(missing))
    if "strategy_forward_return" in frame.columns:
        raise ValueError("Public executable entry does not accept strategy_forward_return")
    if return_horizon != decision_frequency:
        raise ValueError("Continuous holding requires return_horizon == decision_frequency")
    if return_horizon not in horizon_deltas:
        raise ValueError("return_horizon missing from horizon_deltas: " + return_horizon)
    for column in [
        "native_bar_end_ts", "signal_bar_end_ts", "availability_ts",
        "data_observed_ts", "decision_ts", "order_submit_ts", "execution_ts",
        "execution_open_time", "next_execution_ts",
    ]:
        working[column] = pd.to_datetime(working[column], utc=True)
    if int(execution_delay_minutes) <= 0:
        raise ValueError("execution_delay_minutes must be positive")
    expected_delay = pd.Timedelta(minutes=int(execution_delay_minutes))
    if not (working["execution_ts"] - working["decision_ts"] == expected_delay).all():
        raise ValueError("execution_ts does not match the declared execution delay")
    if not (working["signal_bar_end_ts"] == working["decision_ts"]).all():
        raise ValueError("signal_bar_end_ts must equal decision_ts")
    if not (working["native_bar_end_ts"] == working["signal_bar_end_ts"]).all():
        raise ValueError("native_bar_end_ts must equal the released signal bar end")
    if not (working["availability_ts"] <= working["execution_ts"]).all():
        raise ValueError("signal availability is later than execution")
    if not (working["data_observed_ts"] <= working["execution_ts"]).all():
        raise ValueError("data_observed_ts is later than execution")
    if not (working["order_submit_ts"] == working["execution_ts"]).all():
        raise ValueError("order_submit_ts must equal historical execution_ts")
    if not (working["execution_open_time"] == working["execution_ts"]).all():
        raise ValueError("execution_open_time must equal execution_ts")
    if not (working["return_horizon"].astype(str) == return_horizon).all():
        raise ValueError("return_horizon ledger value does not match route")
    if not (working["decision_interval"].astype(str) == decision_frequency).all():
        raise ValueError("decision_interval ledger value does not match route")
    if not (working["holding_interval"].astype(str) == return_horizon).all():
        raise ValueError("holding_interval ledger value does not match route")
    if not (working["score_order"].astype(str) == "high_score_long_low_score_short").all():
        raise ValueError("score_order must be high_score_long_low_score_short")
    expected_holding = pd.Timedelta(horizon_deltas[return_horizon])
    if not (working["next_execution_ts"] - working["execution_ts"] == expected_holding).all():
        raise ValueError("execution ledger does not cover the complete holding interval")
    recomputed = working["exit_price"].astype(float) / working["entry_price"].astype(float) - 1.0
    if not np.allclose(
        recomputed.to_numpy(),
        working["executable_return"].astype(float).to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("executable_return does not match entry and exit prices")
    if not np.allclose(working["execution_price"].astype(float), working["entry_price"].astype(float)):
        raise ValueError("execution_price does not match entry_price")
    if not np.allclose(working["next_execution_price"].astype(float), working["exit_price"].astype(float)):
        raise ValueError("next_execution_price does not match exit_price")
    working["forward_return"] = working["executable_return"].astype(float)
    working["strategy_forward_return"] = working["executable_return"].astype(float)
    return working


def _quantity_toward_zero(quantity: float, step_size: float) -> float:
    if step_size <= 0.0:
        raise ValueError("stepSize must be positive")
    units = np.floor(abs(float(quantity)) / float(step_size) + 1e-12)
    return float(np.copysign(units * float(step_size), float(quantity)))


def continuous_membership_quantity_replay(
    target_holdings: pd.DataFrame,
    execution_ledger: pd.DataFrame,
    *,
    target_gross_notional: float,
    taker_fee_rate: float,
    cost_multipliers: Sequence[float],
    execution_delay_minutes: int,
    account_equity: float | None = None,
    exchange_leverage: float | None = None,
    exchange_rules: pd.DataFrame | None = None,
    epsilon: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replay desired memberships as one continuous signed-quantity path.

    A continuing member keeps its quantity. Orders occur only for entries,
    removals, or side switches. Fold identifiers are lineage only and never
    reset positions. The sole terminal close occurs after the final holding
    interval of each strategy path.
    """
    if target_holdings.empty:
        raise ValueError("target_holdings must not be empty")
    if target_gross_notional <= 0.0:
        raise ValueError("target_gross_notional must be positive")
    if taker_fee_rate < 0.0 or not cost_multipliers:
        raise ValueError("invalid cost configuration")
    if any(float(value) <= 0.0 for value in cost_multipliers):
        raise ValueError("cost_multipliers must be positive")
    if int(execution_delay_minutes) <= 0:
        raise ValueError("execution_delay_minutes must be positive")
    execution_delay = pd.Timedelta(minutes=int(execution_delay_minutes))
    required_target = {"decision_ts", "symbol", "leg"}
    missing_target = sorted(required_target.difference(target_holdings.columns))
    if missing_target:
        raise ValueError("target_holdings missing required columns: " + ", ".join(missing_target))
    required_ledger = {
        "decision_ts", "symbol", "execution_ts", "next_execution_ts",
        "entry_price", "exit_price", "executable_return",
    }
    missing_ledger = sorted(required_ledger.difference(execution_ledger.columns))
    if missing_ledger:
        raise ValueError("execution_ledger missing required columns: " + ", ".join(missing_ledger))

    targets = target_holdings.copy()
    ledger = execution_ledger.copy()
    for frame in (targets, ledger):
        frame["decision_ts"] = pd.to_datetime(frame["decision_ts"], utc=True)
        frame["symbol"] = frame["symbol"].astype(str)
    for column in ("execution_ts", "next_execution_ts"):
        ledger[column] = pd.to_datetime(ledger[column], utc=True)
    identity_candidates = [
        "combo_id", "track", "weight_scheme", "panel_frequency",
        "return_horizon", "component_features",
    ]
    lineage_candidates = [
        "signal_timeframes", "native_bar_end_ts", "signal_bar_end_ts",
        "availability_ts", "data_observed_ts", "decision_interval",
        "order_submit_ts", "execution_ts", "execution_open_time",
        "next_execution_ts", "holding_interval", "exit_rule", "score_order",
    ]
    identity = [column for column in identity_candidates if column in targets.columns]
    if not identity:
        targets["_strategy_id"] = "strategy"
        identity = ["_strategy_id"]
    if targets.duplicated([*identity, "decision_ts", "symbol"]).any():
        raise ValueError("target_holdings has duplicate strategy/decision_ts/symbol rows")
    if ledger.duplicated(["decision_ts", "symbol"]).any():
        raise ValueError("execution_ledger has duplicate decision_ts/symbol rows")
    invalid_legs = sorted(set(targets["leg"].astype(str)).difference({"long", "short"}))
    if invalid_legs:
        raise ValueError("target_holdings contains invalid legs: " + ", ".join(invalid_legs))

    rules: dict[str, dict[str, float]] | None = None
    if exchange_rules is not None:
        required_rules = {"symbol", "market_min_qty", "market_step", "min_notional"}
        missing_rules = sorted(required_rules.difference(exchange_rules.columns))
        if missing_rules:
            raise ValueError("exchange_rules missing columns: " + ", ".join(missing_rules))
        rules = {
            str(row.symbol): {
                "min_qty": float(row.market_min_qty),
                "step_size": float(row.market_step),
                "min_notional": float(row.min_notional),
            }
            for row in exchange_rules[list(required_rules)].itertuples(index=False)
        }
        missing_symbols = sorted(set(ledger["symbol"]).difference(rules))
        if missing_symbols:
            raise ValueError("exchange_rules missing symbols: " + ", ".join(missing_symbols))

    ledger_lookup = ledger.set_index(["decision_ts", "symbol"])
    if not ledger_lookup.index.is_unique:
        raise ValueError("execution_ledger has duplicate decision_ts/symbol rows")
    price_at_execution: dict[tuple[pd.Timestamp, str], float] = {}
    ledger_by_key: dict[tuple[pd.Timestamp, str], object] = {}
    for row in ledger.itertuples(index=False):
        key = (row.decision_ts, str(row.symbol))
        ledger_by_key[key] = row
        price_at_execution[key] = float(row.entry_price)
        next_decision = pd.Timestamp(row.next_execution_ts) - execution_delay
        price_at_execution[(next_decision, str(row.symbol))] = float(row.exit_price)

    decision_rows: list[dict[str, object]] = []
    order_rows: list[dict[str, object]] = []
    holding_rows: list[dict[str, object]] = []
    for group_key, strategy_targets in targets.groupby(identity, sort=False, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        meta = dict(zip(identity, group_key, strict=True))
        positions: dict[str, float] = {}
        available_lineage = [
            column for column in lineage_candidates if column in strategy_targets.columns
        ]
        if available_lineage:
            lineage_counts = strategy_targets.groupby("decision_ts", sort=False)[
                available_lineage
            ].nunique(dropna=False)
            if lineage_counts.gt(1).any(axis=None):
                conflict_column = lineage_counts.gt(1).any(axis=0).loc[lambda values: values].index[0]
                raise ValueError(
                    f"target_holdings has conflicting {conflict_column} within a decision"
                )
            context_by_decision = (
                strategy_targets.sort_values(["decision_ts", "symbol"])
                .drop_duplicates("decision_ts")
                .set_index("decision_ts")[available_lineage]
                .to_dict(orient="index")
            )
        else:
            context_by_decision = {}
        decisions = list(strategy_targets.sort_values(["decision_ts", "symbol"]).groupby("decision_ts", sort=True))
        decision_index = pd.DatetimeIndex([item[0] for item in decisions])
        if len(decision_index) > 1:
            route = str(strategy_targets["return_horizon"].iloc[0]) if "return_horizon" in strategy_targets else ""
            if route:
                unit = "D" if route.endswith("d") else route[-1]
                expected = pd.to_timedelta(int(route[:-1]), unit=unit)
            else:
                expected = decision_index[1] - decision_index[0]
            if not ((decision_index[1:] - decision_index[:-1]) == expected).all():
                raise ValueError("OOS decision timeline is not continuous at the strategy horizon")

        for decision_number, (decision_ts, desired_frame) in enumerate(decisions):
            decision_context = context_by_decision.get(pd.Timestamp(decision_ts), {})
            desired_side = {
                str(row.symbol): (1 if str(row.leg) == "long" else -1)
                for row in desired_frame[["symbol", "leg"]].itertuples(index=False)
            }
            desired_weight = (
                {
                    str(row.symbol): float(row.weight)
                    for row in desired_frame[["symbol", "weight"]].itertuples(index=False)
                }
                if "weight" in desired_frame.columns
                else {}
            )
            long_count = sum(side > 0 for side in desired_side.values())
            short_count = sum(side < 0 for side in desired_side.values())
            if long_count == 0 or short_count == 0:
                raise ValueError(f"both long and short memberships are required at {decision_ts}")
            slot = {
                1: 0.5 * float(target_gross_notional) / long_count,
                -1: 0.5 * float(target_gross_notional) / short_count,
            }
            fold_idx = int(desired_frame["fold_idx"].iloc[0]) if "fold_idx" in desired_frame else 0
            before = dict(positions)
            decision_order_rows: list[dict[str, object]] = []
            all_symbols = sorted(set(before).union(desired_side))
            for symbol in all_symbols:
                price_key = (pd.Timestamp(decision_ts), symbol)
                if price_key not in price_at_execution:
                    raise ValueError(f"missing historical execution price for {symbol} at {decision_ts}")
                price = float(price_at_execution[price_key])
                previous = float(before.get(symbol, 0.0))
                side = int(desired_side.get(symbol, 0))
                previous_side = 0 if abs(previous) <= epsilon else (1 if previous > 0 else -1)
                status = "hold_unchanged"
                desired_quantity = previous
                if side == 0:
                    desired_quantity = 0.0
                    status = "close"
                elif side != previous_side:
                    raw_target = side * slot[side] / price
                    desired_quantity = raw_target
                    status = "open" if previous_side == 0 else "side_switch"
                    if rules is not None:
                        rule = rules[symbol]
                        rounded = _quantity_toward_zero(raw_target, rule["step_size"])
                        valid = (
                            abs(rounded) + epsilon >= rule["min_qty"]
                            and abs(rounded) * price + epsilon >= rule["min_notional"]
                        )
                        if valid:
                            desired_quantity = rounded
                        else:
                            desired_quantity = previous
                            status = "filtered_keep_previous"
                executed_quantity = desired_quantity - previous
                if abs(desired_quantity) <= epsilon:
                    positions.pop(symbol, None)
                else:
                    positions[symbol] = float(desired_quantity)
                decision_order_rows.append(
                    {
                        **meta,
                        **decision_context,
                        "fold_idx": fold_idx,
                        "decision_ts": decision_ts,
                        "order_submit_ts": pd.Timestamp(decision_ts) + execution_delay,
                        "execution_ts": pd.Timestamp(decision_ts) + execution_delay,
                        "execution_open_time": pd.Timestamp(decision_ts) + execution_delay,
                        "symbol": symbol,
                        "execution_price": price,
                        "previous_signed_quantity": previous,
                        "desired_signed_quantity": desired_quantity,
                        "executed_quantity": executed_quantity,
                        "executed_order_notional": abs(executed_quantity) * price,
                        "status": status,
                    }
                )

            gross_pnl = 0.0
            actual_gross = 0.0
            actual_net = 0.0
            for symbol, quantity in sorted(positions.items()):
                key = (pd.Timestamp(decision_ts), symbol)
                if key not in ledger_by_key:
                    raise ValueError(f"held symbol missing complete execution ledger: {symbol} at {decision_ts}")
                market = ledger_by_key[key]
                entry_price = float(market.entry_price)
                exit_price = float(market.exit_price)
                pnl = quantity * (exit_price - entry_price)
                gross_pnl += pnl
                actual_gross += abs(quantity * entry_price)
                actual_net += quantity * entry_price
                holding_rows.append(
                    {
                        **meta,
                        **decision_context,
                        "fold_idx": fold_idx,
                        "decision_ts": decision_ts,
                        "symbol": symbol,
                        "leg": "long" if quantity > 0 else "short",
                        "target_weight": desired_weight.get(symbol, np.nan),
                        "signed_quantity": quantity,
                        "execution_ts": market.execution_ts,
                        "next_execution_ts": market.next_execution_ts,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "executable_return": float(market.executable_return),
                        "actual_notional": quantity * entry_price,
                        "pnl_usd": pnl,
                    }
                )

            terminal_close_notional = 0.0
            if decision_number == len(decisions) - 1:
                for symbol, quantity in sorted(positions.items()):
                    market = ledger_by_key[(pd.Timestamp(decision_ts), symbol)]
                    close_price = float(market.exit_price)
                    terminal_close_notional += abs(quantity) * close_price
                    order_rows.append(
                        {
                            **meta,
                            **decision_context,
                            "fold_idx": fold_idx,
                            "decision_ts": decision_ts,
                            "order_submit_ts": market.next_execution_ts,
                            "execution_ts": market.next_execution_ts,
                            "execution_open_time": market.next_execution_ts,
                            "symbol": symbol,
                            "execution_price": close_price,
                            "previous_signed_quantity": quantity,
                            "desired_signed_quantity": 0.0,
                            "executed_quantity": -quantity,
                            "executed_order_notional": abs(quantity) * close_price,
                            "status": "terminal_close",
                        }
                    )
            order_rows.extend(decision_order_rows)
            rebalance_notional = sum(float(row["executed_order_notional"]) for row in decision_order_rows)
            charged_notional = rebalance_notional + terminal_close_notional
            detail_row: dict[str, object] = {
                **meta,
                **decision_context,
                "fold_idx": fold_idx,
                "decision_ts": decision_ts,
                "long_count": sum(quantity > epsilon for quantity in positions.values()),
                "short_count": sum(quantity < -epsilon for quantity in positions.values()),
                "actual_gross_notional": actual_gross,
                "actual_net_notional": actual_net,
                "actual_vs_target_gross_ratio": actual_gross / float(target_gross_notional),
                "actual_abs_net_exposure_share": abs(actual_net) / float(target_gross_notional),
                "executed_order_notional": rebalance_notional,
                "terminal_close_notional": terminal_close_notional,
                "charged_order_notional": charged_notional,
                "rebalance_turnover": rebalance_notional / float(target_gross_notional),
                "terminal_close_turnover": terminal_close_notional / float(target_gross_notional),
                "charged_turnover": charged_notional / float(target_gross_notional),
                "gross_pnl_usd": gross_pnl,
                "gross_return": gross_pnl / float(target_gross_notional),
            }
            if account_equity is not None:
                detail_row["gross_return_on_equity"] = gross_pnl / float(account_equity)
            if exchange_leverage is not None and account_equity is not None:
                detail_row["margin_required"] = actual_gross / float(exchange_leverage)
                detail_row["margin_utilization"] = detail_row["margin_required"] / float(account_equity)
            for multiplier in cost_multipliers:
                label = scenario_label(float(multiplier))
                cost_usd = charged_notional * float(taker_fee_rate) * float(multiplier)
                detail_row[f"cost_{label}_usd"] = cost_usd
                detail_row[f"cost_{label}"] = cost_usd / float(target_gross_notional)
                detail_row[f"net_pnl_{label}_usd"] = gross_pnl - cost_usd
                detail_row[f"net_return_{label}"] = (gross_pnl - cost_usd) / float(target_gross_notional)
                if account_equity is not None:
                    detail_row[f"net_return_{label}_on_equity"] = (gross_pnl - cost_usd) / float(account_equity)
            decision_rows.append(detail_row)
        positions = {}
    return pd.DataFrame(decision_rows), pd.DataFrame(order_rows), pd.DataFrame(holding_rows)


def evaluate_executable_long_short_strategy_with_orders(
    combo_spec: ComboSpec,
    executable_signal_frame: pd.DataFrame,
    folds: Sequence,
    walk_forward_spec: Mapping[str, int],
    *,
    weight_scheme: str,
    feature_families: Mapping[str, str] | None,
    decision_frequency: str,
    n_buckets: int,
    min_cross_section: int,
    frequency_periods_per_year: Mapping[str, int | float],
    cost_multipliers: Sequence[float],
    taker_fee_rate: float,
    horizon_deltas: Mapping[str, pd.Timedelta],
    supported_signal_timeframes: Sequence[str],
    execution_delay_minutes: int,
    min_pair_corr_observations: int | None = None,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build train-only targets, then replay one continuous quantity path."""
    adapted = validated_executable_return_adapter(
        executable_signal_frame,
        return_horizon=combo_spec.return_horizon,
        decision_frequency=decision_frequency,
        horizon_deltas=horizon_deltas,
        execution_delay_minutes=execution_delay_minutes,
    )
    _, legacy_detail, target_holdings = evaluate_long_short_strategy(
        combo_spec,
        adapted,
        folds,
        walk_forward_spec,
        weight_scheme=weight_scheme,
        feature_families=feature_families,
        decision_frequency=decision_frequency,
        n_buckets=n_buckets,
        min_cross_section=min_cross_section,
        frequency_periods_per_year=frequency_periods_per_year,
        cost_multipliers=cost_multipliers,
        taker_fee_rate=taker_fee_rate,
        horizon_deltas=horizon_deltas,
        supported_signal_timeframes=supported_signal_timeframes,
        min_pair_corr_observations=min_pair_corr_observations,
    )
    ledger_columns = [
        "decision_ts", "symbol", "signal_timeframes", "native_bar_end_ts",
        "signal_bar_end_ts", "availability_ts", "data_observed_ts",
        "decision_interval", "order_submit_ts", "execution_ts",
        "execution_open_time", "next_execution_ts", "return_horizon",
        "holding_interval", "exit_rule", "score_order", "entry_price",
        "exit_price", "execution_price", "next_execution_price",
        "executable_return",
    ]
    ledger = adapted[ledger_columns].reset_index(drop=True).drop_duplicates(
        ["decision_ts", "symbol"]
    )
    target_holdings = target_holdings.drop(
        columns=["strategy_forward_return", *ledger_columns[2:]], errors="ignore"
    ).merge(
        ledger,
        on=["decision_ts", "symbol"],
        how="left",
        validate="many_to_one",
    )
    if target_holdings[["execution_ts", "next_execution_ts", "executable_return"]].isna().any().any():
        raise ValueError("Strategy targets lost execution-ledger lineage")
    detail, orders, actual_holdings = continuous_membership_quantity_replay(
        target_holdings,
        ledger,
        target_gross_notional=1.0,
        taker_fee_rate=taker_fee_rate,
        cost_multipliers=cost_multipliers,
        execution_delay_minutes=execution_delay_minutes,
    )
    financial_columns = {
        "gross_return", "active_return", "rebalance_turnover",
        "terminal_close_turnover", "charged_turnover",
        *[f"cost_{scenario_label(float(value))}" for value in cost_multipliers],
        *[f"net_return_{scenario_label(float(value))}" for value in cost_multipliers],
        *[f"net_active_return_{scenario_label(float(value))}" for value in cost_multipliers],
    }
    lineage_columns = [
        column for column in legacy_detail.columns
        if column not in financial_columns and column not in detail.columns
    ]
    detail = detail.merge(
        legacy_detail[["decision_ts", *lineage_columns]],
        on="decision_ts",
        how="left",
        validate="one_to_one",
    )
    if "benchmark_return" not in detail:
        detail["benchmark_return"] = 0.0
    detail["active_return"] = detail["gross_return"] - detail["benchmark_return"].astype(float)
    for multiplier in cost_multipliers:
        label = scenario_label(float(multiplier))
        detail[f"net_active_return_{label}"] = detail[f"active_return"] - detail[f"cost_{label}"]
    if "cross_section_size" not in detail:
        detail["cross_section_size"] = np.nan
    if "name_turnover_share" not in detail:
        detail["name_turnover_share"] = np.nan
    diagnostics = [
        {
            "decision_ts": row.decision_ts,
            "cross_section_size": int(row.cross_section_size),
            "status": "ok",
        }
        for row in detail[["decision_ts", "cross_section_size"]].itertuples(index=False)
    ]
    summary = summarize_long_short_strategy(
        combo_spec,
        detail,
        diagnostics,
        walk_forward_spec,
        decision_frequency,
        frequency_periods_per_year,
        cost_multipliers,
    )
    merge_columns = [
        column for column in [
            "combo_id", "track", "weight_scheme", "panel_frequency",
            "return_horizon", "component_features", "fold_idx",
            "decision_ts", "symbol", "leg",
        ] if column in actual_holdings.columns and column in target_holdings.columns
    ]
    overlapping_payload = [
        column for column in target_holdings.columns
        if column in actual_holdings.columns and column not in merge_columns
    ]
    target_metadata = target_holdings.drop(columns=overlapping_payload)
    actual_holdings = actual_holdings.merge(
        target_metadata,
        on=merge_columns,
        how="left",
        validate="one_to_one",
        suffixes=("", "_target"),
    )
    return summary, detail, orders, actual_holdings


def evaluate_executable_long_short_strategy(
    combo_spec: ComboSpec,
    executable_signal_frame: pd.DataFrame,
    folds: Sequence,
    walk_forward_spec: Mapping[str, int],
    **kwargs,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Compatibility facade for the formal executable strategy entry."""
    summary, detail, _, holdings = evaluate_executable_long_short_strategy_with_orders(
        combo_spec,
        executable_signal_frame,
        folds,
        walk_forward_spec,
        **kwargs,
    )
    return summary, detail, holdings


def live_like_executable_min_notional_replay(
    target_holdings: pd.DataFrame,
    exchange_rules: pd.DataFrame,
    *,
    account_equity: float,
    target_gross_notional: float,
    exchange_leverage: float,
    taker_fee_rate: float,
    cost_multipliers: Sequence[float],
    frequency_periods_per_year: Mapping[str, int | float],
    horizon_deltas: Mapping[str, pd.Timedelta],
    execution_delay_minutes: int,
    epsilon: float = 1e-9,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replay L4 with historical prices and a frozen exchange-rule snapshot."""
    if target_holdings.empty:
        raise ValueError("target_holdings must not be empty")
    route_pairs = target_holdings[["panel_frequency", "return_horizon"]].drop_duplicates()
    if len(route_pairs) != 1:
        raise ValueError("Executable L4 replay requires one time-contract route per call")
    route = route_pairs.iloc[0]
    adapted = validated_executable_return_adapter(
        target_holdings,
        return_horizon=str(route["return_horizon"]),
        decision_frequency=str(route["panel_frequency"]),
        horizon_deltas=horizon_deltas,
        execution_delay_minutes=execution_delay_minutes,
    )
    ledger_columns = [
        "decision_ts", "symbol", "execution_ts", "next_execution_ts",
        "entry_price", "exit_price", "executable_return",
    ]
    market_value_columns = ledger_columns[2:]
    conflicts = adapted.groupby(["decision_ts", "symbol"], sort=False)[
        market_value_columns
    ].nunique(dropna=False).gt(1).any(axis=1)
    if conflicts.any():
        raise ValueError("L4 execution ledger has conflicting market rows")
    market_ledger = adapted[ledger_columns].drop_duplicates(["decision_ts", "symbol"])
    detail, orders, actual_holdings = continuous_membership_quantity_replay(
        target_holdings,
        market_ledger,
        target_gross_notional=target_gross_notional,
        taker_fee_rate=taker_fee_rate,
        cost_multipliers=cost_multipliers,
        execution_delay_minutes=execution_delay_minutes,
        account_equity=account_equity,
        exchange_leverage=exchange_leverage,
        exchange_rules=exchange_rules,
        epsilon=epsilon,
    )
    strategy_keys = [
        column for column in [
            "combo_id", "track", "weight_scheme", "panel_frequency",
            "return_horizon", "component_features",
        ] if column in detail.columns and column in orders.columns
    ]
    order_stats = (
        orders.assign(
            filtered=orders["status"].eq("filtered_keep_previous"),
            executed=orders["executed_order_notional"].astype(float) > epsilon,
            attempted=orders["status"].ne("hold_unchanged"),
        )
        .groupby([*strategy_keys, "decision_ts"], as_index=False, dropna=False)
        .agg(
            filtered_order_count=("filtered", "sum"),
            executed_order_count=("executed", "sum"),
            order_attempt_count=("attempted", "sum"),
        )
    )
    detail = detail.merge(
        order_stats,
        on=[*strategy_keys, "decision_ts"],
        how="left",
        validate="one_to_one",
    )
    detail[["filtered_order_count", "executed_order_count", "order_attempt_count"]] = detail[
        ["filtered_order_count", "executed_order_count", "order_attempt_count"]
    ].fillna(0).astype(int)
    detail["charged_turnover_on_target_gross"] = detail["charged_turnover"]
    detail["gross_return_on_target_gross"] = detail["gross_return"]
    detail["filtered_order_notional"] = 0.0
    detail["target_gross_notional"] = float(target_gross_notional)
    detail["target_net_notional"] = 0.0
    detail["target_abs_net_exposure_share"] = 0.0
    detail["weight_abs_error_sum"] = (
        (detail["actual_gross_notional"] - float(target_gross_notional)).abs()
        / float(target_gross_notional)
    )
    summary_rows: list[dict[str, object]] = []
    group_columns = [
        column for column in [
            "combo_id", "track", "weight_scheme", "panel_frequency",
            "return_horizon", "component_features",
        ] if column in detail.columns
    ]
    for _, combo_detail in detail.groupby(group_columns, sort=False, dropna=False):
        decision_frequency = str(combo_detail["panel_frequency"].iloc[0])
        row: dict[str, object] = {
            column: combo_detail[column].iloc[0] for column in group_columns
        }
        row.update(
            {
                "account_equity": float(account_equity),
                "target_gross_notional": float(target_gross_notional),
                "exchange_leverage": float(exchange_leverage),
                "n_folds": int(combo_detail["fold_idx"].nunique()),
                "decision_count": int(len(combo_detail)),
                "mean_actual_gross_notional": float(combo_detail["actual_gross_notional"].mean()),
                "min_actual_gross_notional": float(combo_detail["actual_gross_notional"].min()),
                "max_actual_gross_notional": float(combo_detail["actual_gross_notional"].max()),
                "mean_actual_vs_target_gross_ratio": float(combo_detail["actual_vs_target_gross_ratio"].mean()),
                "mean_abs_net_exposure_share": float(combo_detail["actual_abs_net_exposure_share"].mean()),
                "max_abs_net_exposure_share": float(combo_detail["actual_abs_net_exposure_share"].max()),
                "mean_weight_abs_error_sum": float(combo_detail["weight_abs_error_sum"].mean()),
                "max_weight_abs_error_sum": float(combo_detail["weight_abs_error_sum"].max()),
                "filtered_order_count": int(combo_detail["filtered_order_count"].sum()),
                "executed_order_count": int(combo_detail["executed_order_count"].sum()),
                "order_attempt_count": int(combo_detail["order_attempt_count"].sum()),
                "filtered_order_notional": 0.0,
                "executed_order_notional": float(combo_detail["executed_order_notional"].sum()),
                "terminal_close_notional": float(combo_detail["terminal_close_notional"].sum()),
                "mean_charged_order_notional": float(combo_detail["charged_order_notional"].mean()),
                "mean_charged_turnover_on_target_gross": float(combo_detail["charged_turnover"].mean()),
                "max_margin_required": float(combo_detail["margin_required"].max()),
                "max_margin_utilization": float(combo_detail["margin_utilization"].max()),
                "gross_annualized_return_on_equity": annualized_mean_return(
                    combo_detail["gross_return_on_equity"], decision_frequency, frequency_periods_per_year
                ),
                "gross_sharpe_on_equity": annualized_sharpe_for_frequency(
                    combo_detail["gross_return_on_equity"], decision_frequency, frequency_periods_per_year
                ),
                "gross_max_drawdown_on_equity": max_drawdown_from_returns(combo_detail["gross_return_on_equity"]),
                "gross_fold_positive_share_on_equity": fold_positive_share_from_returns(
                    combo_detail, "gross_return_on_equity"
                ),
            }
        )
        attempts = int(row["order_attempt_count"])
        row["filtered_order_share"] = float(row["filtered_order_count"]) / attempts if attempts else 0.0
        for multiplier in cost_multipliers:
            label = scenario_label(float(multiplier))
            net_col = f"net_return_{label}_on_equity"
            row[f"cost_{label}_mean_usd"] = float(combo_detail[f"cost_{label}_usd"].mean())
            row[f"net_{label}_annualized_return_on_equity"] = annualized_mean_return(
                combo_detail[net_col], decision_frequency, frequency_periods_per_year
            )
            row[f"net_{label}_sharpe_on_equity"] = annualized_sharpe_for_frequency(
                combo_detail[net_col], decision_frequency, frequency_periods_per_year
            )
            row[f"net_{label}_max_drawdown_on_equity"] = max_drawdown_from_returns(combo_detail[net_col])
            row[f"net_{label}_fold_positive_share_on_equity"] = fold_positive_share_from_returns(
                combo_detail, net_col
            )
        summary_rows.append(row)
    return pd.DataFrame(summary_rows), detail, orders, actual_holdings
