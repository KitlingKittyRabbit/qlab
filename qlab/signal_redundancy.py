from __future__ import annotations

"""Deterministic redundancy classes for high-dimensional signal paths."""

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


SIGNAL_REDUNDANCY_KEY_COLUMNS = (
    "fold_idx",
    "decision_ts",
    "symbol",
    "hypothesis_id",
    "horizon",
)
SIGNAL_REDUNDANCY_VIEWS = (
    "signal_target",
    "signal_residual",
)
SIGNAL_REDUNDANCY_COLUMNS = SIGNAL_REDUNDANCY_KEY_COLUMNS + SIGNAL_REDUNDANCY_VIEWS


@dataclass(frozen=True)
class SignalRedundancyArtifacts:
    support_audit: pd.DataFrame
    support_exclusions: pd.DataFrame
    fold_pairwise_correlations: pd.DataFrame
    pairwise_conservative_similarity: pd.DataFrame
    class_membership: pd.DataFrame
    representative_manifest: pd.DataFrame
    threshold_sensitivity: pd.DataFrame


def _key_digest(keys: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype="uint64")
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _validate_thresholds(
    thresholds: Iterable[float], primary_threshold: float
) -> tuple[float, ...]:
    values = tuple(sorted({float(value) for value in thresholds}))
    if not values or any(not np.isfinite(value) or value <= 0.0 or value > 1.0 for value in values):
        raise ValueError("thresholds must be finite values in (0, 1]")
    if float(primary_threshold) not in values:
        raise ValueError("primary_threshold must be included in thresholds")
    return values


def _deterministic_classes(
    ids: tuple[str, ...], similarity: np.ndarray, threshold: float
) -> list[tuple[str, ...]]:
    if len(ids) == 1:
        return [(ids[0],)]
    distance = 1.0 - np.asarray(similarity, dtype=float)
    np.fill_diagonal(distance, 0.0)
    if np.any(distance < -1e-12) or np.any(distance > 2.0 + 1e-12):
        raise ValueError("pairwise similarity must be in [-1, 1]")
    condensed = squareform(np.clip(distance, 0.0, 2.0), checks=True)
    raw_labels = fcluster(
        linkage(condensed, method="complete"),
        t=1.0 - float(threshold),
        criterion="distance",
    )
    groups: dict[int, list[str]] = {}
    for hypothesis_id, label in zip(ids, raw_labels):
        groups.setdefault(int(label), []).append(hypothesis_id)
    classes = [tuple(sorted(members)) for members in groups.values()]
    classes.sort(key=lambda members: members[0])
    index = {hypothesis_id: offset for offset, hypothesis_id in enumerate(ids)}
    for members in classes:
        offsets = [index[hypothesis_id] for hypothesis_id in members]
        within = similarity[np.ix_(offsets, offsets)]
        if float(np.min(within)) < float(threshold) - 1e-12:
            raise RuntimeError("complete-linkage class violates its similarity threshold")
    return classes


def _class_medoid(
    members: tuple[str, ...], ids: tuple[str, ...], similarity: np.ndarray
) -> tuple[str, float]:
    if len(members) == 1:
        return members[0], 1.0
    index = {hypothesis_id: offset for offset, hypothesis_id in enumerate(ids)}
    scores = {
        hypothesis_id: float(
            np.mean(
                [
                    similarity[index[hypothesis_id], index[other]]
                    for other in members
                    if other != hypothesis_id
                ]
            )
        )
        for hypothesis_id in members
    }
    best = max(scores.values())
    representative = min(
        hypothesis_id
        for hypothesis_id, score in scores.items()
        if np.isclose(score, best, rtol=1e-12, atol=1e-12)
    )
    return representative, scores[representative]


def audit_signal_redundancy_classes(
    observations: pd.DataFrame,
    *,
    thresholds: Iterable[float] = (0.90, 0.95, 0.975, 0.99),
    primary_threshold: float = 0.95,
    minimum_common_support_ratio: float = 0.99,
    similarity_views: Iterable[str] = SIGNAL_REDUNDANCY_VIEWS,
) -> SignalRedundancyArtifacts:
    """Classify redundant signal hypotheses without reading outcome information.

    Similarity is the minimum Pearson correlation across every outer fold and
    every explicitly selected signal view. Clustering is complete-linkage
    within each horizon, so every pair in a class satisfies the threshold.
    """

    if not isinstance(observations, pd.DataFrame):
        raise TypeError("observations must be a pandas DataFrame")
    views = tuple(str(view) for view in similarity_views)
    if not views or len(set(views)) != len(views):
        raise ValueError("similarity_views must be non-empty and unique")
    invalid_views = sorted(set(views).difference(SIGNAL_REDUNDANCY_VIEWS))
    if invalid_views:
        raise ValueError("unsupported similarity views: " + ", ".join(invalid_views))
    required_columns = SIGNAL_REDUNDANCY_KEY_COLUMNS + views
    missing = [column for column in required_columns if column not in observations]
    if missing:
        raise ValueError("observations missing columns: " + ", ".join(missing))
    forbidden = {
        "outcome_target",
        "outcome_prediction",
        "outcome_residual",
        "residual_product",
        "p_value",
        "adjusted_p_value",
    }.intersection(observations.columns)
    if forbidden:
        raise ValueError("outcome or inference columns are forbidden: " + ", ".join(sorted(forbidden)))

    frame = observations.loc[:, required_columns].copy()
    if frame.empty:
        raise ValueError("observations must not be empty")
    for column in ("hypothesis_id", "horizon", "symbol"):
        frame[column] = frame[column].astype(str)
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"{column} must not contain empty values")
    frame["decision_ts"] = pd.to_datetime(frame["decision_ts"], utc=True, errors="raise")
    frame["fold_idx"] = pd.to_numeric(frame["fold_idx"], errors="raise").astype(int)
    for column in views:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        if not np.isfinite(frame[column].to_numpy()).all():
            raise ValueError(f"{column} must contain only finite values")

    key_columns = ["horizon", "hypothesis_id", "fold_idx", "decision_ts", "symbol"]
    if frame.duplicated(key_columns).any():
        raise ValueError("duplicate signal observation keys")
    horizon_counts = frame.groupby("hypothesis_id", sort=False)["horizon"].nunique()
    if not horizon_counts.eq(1).all():
        raise ValueError("each hypothesis_id must belong to exactly one horizon")

    threshold_values = _validate_thresholds(thresholds, primary_threshold)
    minimum_common_support_ratio = float(minimum_common_support_ratio)
    if (
        not np.isfinite(minimum_common_support_ratio)
        or minimum_common_support_ratio <= 0.0
        or minimum_common_support_ratio > 1.0
    ):
        raise ValueError("minimum_common_support_ratio must be in (0, 1]")
    support_rows: list[dict[str, object]] = []
    exclusion_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    membership_rows: list[dict[str, object]] = []
    representative_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []

    for horizon in sorted(frame["horizon"].unique()):
        horizon_frame = frame.loc[frame["horizon"].eq(horizon)].copy()
        ids = tuple(sorted(horizon_frame["hypothesis_id"].unique()))
        keys_by_hypothesis: dict[str, pd.DataFrame] = {}
        common_index: pd.MultiIndex | None = None
        for hypothesis_id in ids:
            candidate = horizon_frame.loc[horizon_frame["hypothesis_id"].eq(hypothesis_id)]
            keys = candidate[["fold_idx", "decision_ts", "symbol"]].sort_values(
                ["fold_idx", "decision_ts", "symbol"], kind="mergesort"
            ).reset_index(drop=True)
            keys_by_hypothesis[hypothesis_id] = keys
            candidate_index = pd.MultiIndex.from_frame(keys)
            common_index = (
                candidate_index
                if common_index is None
                else common_index.intersection(candidate_index, sort=False)
            )
        if common_index is None or len(common_index) == 0:
            raise ValueError(f"empty common support within horizon {horizon}")
        common_keys = common_index.to_frame(index=False).sort_values(
            ["fold_idx", "decision_ts", "symbol"], kind="mergesort"
        ).reset_index(drop=True)
        common_digest = _key_digest(common_keys)
        for hypothesis_id in ids:
            keys = keys_by_hypothesis[hypothesis_id]
            candidate_index = pd.MultiIndex.from_frame(keys)
            excluded = candidate_index.difference(common_index, sort=False)
            coverage_ratio = len(common_index) / len(candidate_index)
            if coverage_ratio < minimum_common_support_ratio - 1e-12:
                raise ValueError(
                    f"common support coverage below {minimum_common_support_ratio:.3f} "
                    f"for {horizon} {hypothesis_id}: {coverage_ratio:.6f}"
                )
            support_rows.append(
                {
                    "horizon": horizon,
                    "hypothesis_id": hypothesis_id,
                    "original_row_count": len(keys),
                    "common_row_count": len(common_index),
                    "excluded_row_count": len(excluded),
                    "common_support_ratio": coverage_ratio,
                    "common_decision_count": common_keys["decision_ts"].nunique(),
                    "common_symbol_count": common_keys["symbol"].nunique(),
                    "common_fold_count": common_keys["fold_idx"].nunique(),
                    "common_key_sha256": common_digest,
                }
            )
            for fold_idx, decision_ts, symbol in excluded.tolist():
                exclusion_rows.append(
                    {
                        "horizon": horizon,
                        "hypothesis_id": hypothesis_id,
                        "fold_idx": int(fold_idx),
                        "decision_ts": decision_ts,
                        "symbol": str(symbol),
                        "reason": "not_in_horizon_wide_common_support",
                    }
                )
        common_marker = pd.MultiIndex.from_frame(
            horizon_frame[["fold_idx", "decision_ts", "symbol"]]
        ).isin(common_index)
        horizon_frame = horizon_frame.loc[common_marker].copy()

        index = {hypothesis_id: offset for offset, hypothesis_id in enumerate(ids)}
        similarity = np.eye(len(ids), dtype=float)
        grouped_min: dict[tuple[str, str], list[float]] = {}
        for fold_idx in sorted(horizon_frame["fold_idx"].unique()):
            fold = horizon_frame.loc[horizon_frame["fold_idx"].eq(fold_idx)]
            index_columns = ["decision_ts", "symbol"]
            pivots = {
                view: fold.pivot(index=index_columns, columns="hypothesis_id", values=view)
                .sort_index()
                .loc[:, list(ids)]
                for view in views
            }
            for view, pivot in pivots.items():
                if pivot.isna().any().any():
                    raise ValueError(f"incomplete aligned support for {horizon} fold {fold_idx}")
                standard_deviation = pivot.std(axis=0, ddof=1)
                if (standard_deviation <= 0.0).any() or not np.isfinite(standard_deviation).all():
                    raise ValueError(f"zero-variance signal for {horizon} fold {fold_idx} {view}")
                correlations = pivot.corr(method="pearson")
                for left_offset, left in enumerate(ids):
                    for right in ids[left_offset + 1 :]:
                        value = float(correlations.loc[left, right])
                        if not np.isfinite(value):
                            raise ValueError("pairwise correlation must be finite")
                        fold_rows.append(
                            {
                                "horizon": horizon,
                                "fold_idx": int(fold_idx),
                                "view": view,
                                "left_hypothesis_id": left,
                                "right_hypothesis_id": right,
                                "correlation": value,
                                "observation_count": len(pivot),
                            }
                        )
                        grouped_min.setdefault((left, right), []).append(value)

        for left_offset, left in enumerate(ids):
            for right in ids[left_offset + 1 :]:
                values = grouped_min[(left, right)]
                expected = len(views) * horizon_frame["fold_idx"].nunique()
                if len(values) != expected:
                    raise RuntimeError("pair does not cover every fold and signal view")
                conservative = float(min(values))
                similarity[index[left], index[right]] = conservative
                similarity[index[right], index[left]] = conservative
                pair_rows.append(
                    {
                        "horizon": horizon,
                        "left_hypothesis_id": left,
                        "right_hypothesis_id": right,
                        "conservative_similarity": conservative,
                        "minimum_source_count": len(values),
                    }
                )

        for threshold in threshold_values:
            classes = _deterministic_classes(ids, similarity, threshold)
            sensitivity_rows.append(
                {
                    "threshold": threshold,
                    "scope": horizon,
                    "input_hypothesis_count": len(ids),
                    "representative_count": len(classes),
                    "reduction_count": len(ids) - len(classes),
                }
            )
            for class_offset, members in enumerate(classes, start=1):
                class_id = f"{horizon}_r{int(round(threshold * 1000)):04d}_c{class_offset:03d}"
                representative, medoid_score = _class_medoid(members, ids, similarity)
                offsets = [index[hypothesis_id] for hypothesis_id in members]
                within = similarity[np.ix_(offsets, offsets)]
                minimum_within = float(np.min(within))
                representative_rows.append(
                    {
                        "threshold": threshold,
                        "is_primary_threshold": threshold == float(primary_threshold),
                        "horizon": horizon,
                        "class_id": class_id,
                        "representative_hypothesis_id": representative,
                        "member_count": len(members),
                        "representative_mean_similarity": medoid_score,
                        "minimum_within_class_similarity": minimum_within,
                    }
                )
                for hypothesis_id in members:
                    membership_rows.append(
                        {
                            "threshold": threshold,
                            "is_primary_threshold": threshold == float(primary_threshold),
                            "horizon": horizon,
                            "class_id": class_id,
                            "hypothesis_id": hypothesis_id,
                            "representative_hypothesis_id": representative,
                            "is_representative": hypothesis_id == representative,
                            "member_count": len(members),
                            "minimum_within_class_similarity": minimum_within,
                        }
                    )

    for threshold in threshold_values:
        rows = [row for row in sensitivity_rows if row["threshold"] == threshold]
        sensitivity_rows.append(
            {
                "threshold": threshold,
                "scope": "ALL",
                "input_hypothesis_count": sum(int(row["input_hypothesis_count"]) for row in rows),
                "representative_count": sum(int(row["representative_count"]) for row in rows),
                "reduction_count": sum(int(row["reduction_count"]) for row in rows),
            }
        )

    return SignalRedundancyArtifacts(
        support_audit=pd.DataFrame(support_rows).sort_values(
            ["horizon", "hypothesis_id"], kind="mergesort"
        ).reset_index(drop=True),
        support_exclusions=pd.DataFrame(
            exclusion_rows,
            columns=[
                "horizon",
                "hypothesis_id",
                "fold_idx",
                "decision_ts",
                "symbol",
                "reason",
            ],
        ).sort_values(
            ["horizon", "hypothesis_id", "fold_idx", "decision_ts", "symbol"],
            kind="mergesort",
        ).reset_index(drop=True),
        fold_pairwise_correlations=pd.DataFrame(
            fold_rows,
            columns=[
                "horizon",
                "fold_idx",
                "view",
                "left_hypothesis_id",
                "right_hypothesis_id",
                "correlation",
                "observation_count",
            ],
        ).sort_values(
            ["horizon", "fold_idx", "view", "left_hypothesis_id", "right_hypothesis_id"],
            kind="mergesort",
        ).reset_index(drop=True),
        pairwise_conservative_similarity=pd.DataFrame(
            pair_rows,
            columns=[
                "horizon",
                "left_hypothesis_id",
                "right_hypothesis_id",
                "conservative_similarity",
                "minimum_source_count",
            ],
        ).sort_values(
            ["horizon", "left_hypothesis_id", "right_hypothesis_id"], kind="mergesort"
        ).reset_index(drop=True),
        class_membership=pd.DataFrame(membership_rows).sort_values(
            ["threshold", "horizon", "class_id", "hypothesis_id"], kind="mergesort"
        ).reset_index(drop=True),
        representative_manifest=pd.DataFrame(representative_rows).sort_values(
            ["threshold", "horizon", "class_id"], kind="mergesort"
        ).reset_index(drop=True),
        threshold_sensitivity=pd.DataFrame(sensitivity_rows).sort_values(
            ["threshold", "scope"], kind="mergesort"
        ).reset_index(drop=True),
    )
