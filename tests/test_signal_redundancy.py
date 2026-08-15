from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.signal_redundancy import audit_signal_redundancy_classes


def _observations() -> pd.DataFrame:
    rows = []
    target = {
        "a": np.array([-3, -2, -1, 1, 2, 3], dtype=float),
        "b": np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.01]),
        "c": np.array([-3, -2, -1, 1, 2, 3], dtype=float),
        "d": np.array([3, 2, 1, -1, -2, -3], dtype=float),
    }
    residual = {
        "a": np.array([-2, -1, 0, 0, 1, 2], dtype=float),
        "b": np.array([-2.0, -1.0, 0.0, 0.0, 1.0, 2.01]),
        "c": np.array([-2, 0, 2, -2, 0, 2], dtype=float),
        "d": np.array([2, 1, 0, 0, -1, -2], dtype=float),
    }
    for fold_idx in (0, 1):
        for hypothesis_id in ("a", "b", "c", "d"):
            for offset in range(6):
                rows.append(
                    {
                        "fold_idx": fold_idx,
                        "decision_ts": pd.Timestamp("2025-01-01", tz="UTC")
                        + pd.Timedelta(hours=fold_idx * 12 + offset),
                        "symbol": f"S{offset}",
                        "hypothesis_id": hypothesis_id,
                        "horizon": "12h",
                        "signal_target": target[hypothesis_id][offset] + fold_idx * 0.1,
                        "signal_residual": residual[hypothesis_id][offset] + fold_idx * 0.1,
                    }
                )
    return pd.DataFrame(rows)


def test_redundancy_requires_target_and_residual_similarity_and_is_order_stable():
    frame = _observations()
    result = audit_signal_redundancy_classes(
        frame, thresholds=(0.95,), primary_threshold=0.95
    )
    primary = result.class_membership
    classes = {
        tuple(group["hypothesis_id"].tolist())
        for _, group in primary.groupby("class_id", sort=False)
    }
    assert classes == {("a", "b"), ("c",), ("d",)}
    assert primary.loc[primary["hypothesis_id"].eq("d"), "member_count"].iloc[0] == 1
    merged = primary.loc[primary["hypothesis_id"].isin(["a", "b"])]
    assert merged["representative_hypothesis_id"].unique().tolist() == ["a"]

    shuffled = audit_signal_redundancy_classes(
        frame.sample(frac=1.0, random_state=7),
        thresholds=(0.95,),
        primary_threshold=0.95,
    )
    pd.testing.assert_frame_equal(result.class_membership, shuffled.class_membership)


def test_residual_only_view_merges_without_target_column_or_target_similarity():
    frame = _observations().loc[lambda value: value["hypothesis_id"].isin(["a", "b"])].copy()
    frame.loc[frame["hypothesis_id"].eq("b"), "signal_target"] *= -1.0

    two_view = audit_signal_redundancy_classes(
        frame,
        thresholds=(0.95,),
        primary_threshold=0.95,
    )
    assert len(two_view.representative_manifest) == 2

    residual_only = audit_signal_redundancy_classes(
        frame.drop(columns="signal_target"),
        thresholds=(0.95,),
        primary_threshold=0.95,
        similarity_views=("signal_residual",),
    )
    assert len(residual_only.representative_manifest) == 1
    assert set(residual_only.fold_pairwise_correlations["view"]) == {"signal_residual"}


def test_complete_linkage_does_not_chain_a_b_c():
    base = np.arange(30, dtype=float)
    base = (base - base.mean()) / base.std(ddof=0)
    rng = np.random.default_rng(4)
    orthogonal = rng.normal(size=len(base))
    orthogonal -= orthogonal.mean()
    orthogonal -= np.dot(orthogonal, base) / np.dot(base, base) * base
    orthogonal /= orthogonal.std(ddof=0)
    vectors = {
        "a": base,
        "b": 0.95 * base + np.sqrt(1.0 - 0.95**2) * orthogonal,
        "c": 0.80 * base + 0.60 * orthogonal,
    }
    correlations = pd.DataFrame(vectors).corr()
    assert correlations.loc["a", "b"] >= 0.90
    assert correlations.loc["b", "c"] >= 0.90
    assert correlations.loc["a", "c"] < 0.90
    rows = []
    for hypothesis_id, values in vectors.items():
        for offset, value in enumerate(values):
            rows.append(
                {
                    "fold_idx": 0,
                    "decision_ts": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(hours=offset),
                    "symbol": "BTC",
                    "hypothesis_id": hypothesis_id,
                    "horizon": "1d",
                    "signal_target": value,
                    "signal_residual": value,
                }
            )
    result = audit_signal_redundancy_classes(
        pd.DataFrame(rows), thresholds=(0.90,), primary_threshold=0.90
    )
    sizes = sorted(result.representative_manifest["member_count"].tolist())
    assert sizes == [1, 2]


def test_non_tied_medoid_is_selected():
    base = np.linspace(-1.0, 1.0, 80)
    wave = np.sin(np.linspace(0.0, 4.0 * np.pi, 80))
    vectors = {"a": base + 0.06 * wave, "b": base, "c": base - 0.12 * wave}
    rows = []
    for hypothesis_id, values in vectors.items():
        for offset, value in enumerate(values):
            rows.append(
                {
                    "fold_idx": 0,
                    "decision_ts": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(hours=offset),
                    "symbol": "BTC",
                    "hypothesis_id": hypothesis_id,
                    "horizon": "1d",
                    "signal_target": value,
                    "signal_residual": value,
                }
            )
    result = audit_signal_redundancy_classes(
        pd.DataFrame(rows), thresholds=(0.95,), primary_threshold=0.95
    )
    assert len(result.representative_manifest) == 1
    assert result.representative_manifest["representative_hypothesis_id"].iloc[0] == "b"


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True), "duplicate"),
        (lambda frame: frame.drop(index=frame.index[-2:]), "common support coverage"),
        (lambda frame: frame.assign(outcome_residual=0.0), "forbidden"),
        (lambda frame: frame.assign(signal_residual=np.nan), "finite"),
        (
            lambda frame: frame.assign(
                signal_residual=lambda value: value["signal_residual"].where(
                    ~value["hypothesis_id"].eq("a"), 1.0
                )
            ),
            "zero-variance",
        ),
    ],
)
def test_redundancy_fails_closed(mutator, match):
    with pytest.raises(ValueError, match=match):
        audit_signal_redundancy_classes(
            mutator(_observations()), thresholds=(0.95,), primary_threshold=0.95
        )


def test_different_horizons_never_merge():
    left = _observations().loc[lambda frame: frame["hypothesis_id"].eq("a")].copy()
    right = left.copy()
    right["hypothesis_id"] = "same_values_other_horizon"
    right["horizon"] = "1d"
    result = audit_signal_redundancy_classes(
        pd.concat([left, right], ignore_index=True),
        thresholds=(0.95,),
        primary_threshold=0.95,
    )
    assert len(result.representative_manifest) == 2


def test_small_support_difference_uses_audited_common_intersection():
    frame = _observations()
    extra = frame.loc[frame["hypothesis_id"].eq("a")].iloc[[0]].copy()
    extra["decision_ts"] = pd.Timestamp("2026-01-01", tz="UTC")
    augmented = pd.concat([frame, extra], ignore_index=True)
    result = audit_signal_redundancy_classes(
        augmented,
        thresholds=(0.95,),
        primary_threshold=0.95,
        minimum_common_support_ratio=0.90,
    )
    row = result.support_audit.loc[result.support_audit["hypothesis_id"].eq("a")].iloc[0]
    assert row["excluded_row_count"] == 1
    assert len(result.support_exclusions) == 1
