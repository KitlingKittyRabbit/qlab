"""Side-effect-free cross-sectional panel statistics."""

from __future__ import annotations

import pandas as pd


def rank_grouped_series(
    series: pd.Series,
    *,
    level: str | int = "decision_ts",
) -> pd.Series:
    """Return average ranks within each decision cross-section."""
    grouped = series.groupby(level=level, sort=False)
    return grouped.rank(method="average").astype(float).rename(series.name)


def rank_standardize_grouped_series(
    series: pd.Series,
    *,
    level: str | int = "decision_ts",
) -> pd.Series:
    """Vectorized rank standardization within each decision cross-section."""
    ranks = rank_grouped_series(series, level=level)
    counts = series.groupby(level=level, sort=False).transform("count")
    scaled = -1.0 + (ranks - 1.0) * (2.0 / (counts - 1.0))
    scaled = scaled.where(counts > 1, 0.0)
    return scaled.where(series.notna()).astype(float).rename(series.name)


__all__ = ["rank_grouped_series", "rank_standardize_grouped_series"]
