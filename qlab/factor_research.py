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
from .walkforward import walk_forward_splits


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
    for feature_name in feature_names:
        rows: list[dict[str, object]] = []
        for decision_ts in decision_index:
            cross_section_size = int(counts.at[decision_ts, feature_name])
            status = "ok"
            raw_rank_ic = correlations.at[decision_ts, feature_name]
            if cross_section_size < min_cross_section:
                status = "small_cross_section"
                raw_rank_ic = float("nan")
            elif int(feature_unique.at[decision_ts, feature_name]) <= 1:
                status = "constant_feature"
                raw_rank_ic = float("nan")
            elif int(return_unique.at[decision_ts, feature_name]) <= 1:
                status = "constant_return"
                raw_rank_ic = float("nan")
            elif pd.isna(raw_rank_ic):
                status = "nan_rank_ic"
                raw_rank_ic = float("nan")
            rows.append(
                {
                    "decision_ts": decision_ts,
                    "cross_section_size": cross_section_size,
                    "status": status,
                    "raw_rank_ic": float(raw_rank_ic),
                }
            )
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


def summarize_ic_series(
    panel_frequency: str,
    horizon: str,
    feature_name: str,
    detail_frame: pd.DataFrame,
    walk_forward_spec: Mapping[str, int],
    test_diagnostics: list[dict[str, object]],
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
    hac_lags = research_stats.newey_west_max_lags(observation_count)
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
    working = pd.DataFrame(
        {
            "decision_ts": decision_values,
            feature_name: frame[feature_name].to_numpy(dtype=float, copy=False),
            "forward_return": frame["forward_return"].to_numpy(dtype=float, copy=False),
            **{column: frame[column].to_numpy(dtype=float, copy=False) for column in control_columns},
        }
    )
    valid = working.dropna(
        subset=[feature_name, "forward_return", *control_columns])
    if valid.empty:
        return [{"decision_ts": decision_ts, "cross_section_size": 0, "status": "small_cross_section"} for decision_ts in decision_index]

    grouped = valid.groupby("decision_ts", sort=False)
    counts = grouped.size().reindex(decision_index, fill_value=0).astype(int)
    feature_unique = grouped[feature_name].nunique(
        dropna=True).reindex(decision_index, fill_value=0).astype(int)
    return_unique = grouped["forward_return"].nunique(
        dropna=True).reindex(decision_index, fill_value=0).astype(int)
    control_unique = {
        control_name: grouped[control_name].nunique(dropna=True).reindex(
            decision_index, fill_value=0).astype(int)
        for control_name in control_columns
    }

    diagnostics: list[dict[str, object]] = []
    rows_by_decision: dict[object, dict[str, object]] = {}
    ok_decisions: list[object] = []
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
                ok_decisions.append(decision_ts)
        diagnostics.append(row)
        rows_by_decision[decision_ts] = row

    if not ok_decisions:
        return diagnostics

    scored = valid[valid["decision_ts"].isin(ok_decisions)]
    for decision_ts, group in scored.groupby("decision_ts", sort=False):
        row = rows_by_decision[decision_ts]
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
    return diagnostics


def summarize_fama_macbeth(
    panel_frequency: str,
    horizon: str,
    feature_name: str,
    detail_frame: pd.DataFrame,
    walk_forward_spec: Mapping[str, int],
    diagnostics: list[dict[str, object]],
    control_columns: Sequence[str],
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
    hac_lags = research_stats.newey_west_max_lags(observation_count)
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
    for feature_name in feature_names:
        feature_rows = rank_ic_rows_for_frame(
            train_slice[["symbol", feature_name, "forward_return"]],
            feature_name,
            min_cross_section,
        )
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


def feature_signal_timeframe(
    feature_name: str,
    panel_frequency: str,
    supported_signal_timeframes: Sequence[str],
    suffix_panel_frequencies: set[str] | frozenset[str] = frozenset({
                                                                    "2h", "4h"}),
) -> str:
    if panel_frequency not in suffix_panel_frequencies:
        return panel_frequency
    for signal_timeframe in supported_signal_timeframes:
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
        feature_signal_timeframe(
            feature_name, panel_frequency, supported_signal_timeframes)
        for feature_name in feature_names
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


def validate_no_overlap_design(
    combo_specs: Sequence[ComboSpec],
    horizon_deltas: Mapping[str, pd.Timedelta],
    supported_signal_timeframes: Sequence[str],
) -> None:
    for combo_spec in combo_specs:
        decision_frequency = combo_decision_frequency(
            combo_spec, horizon_deltas, supported_signal_timeframes)
        panel_delta = horizon_deltas[decision_frequency]
        horizon_delta = horizon_deltas[combo_spec.return_horizon]
        if horizon_delta > panel_delta:
            raise ValueError(
                "strategy evaluation requires non-overlapping signal horizons, so it only supports "
                f"return_horizon <= decision_frequency; got {combo_spec.combo_id} "
                f"with {combo_spec.return_horizon} vs {decision_frequency}"
            )
