"""Side-effect-free cross-sectional panel statistics."""

from __future__ import annotations

import pandas as pd


def rank_standardize_grouped_series(
    series: pd.Series,
    *,
    level: str | int = "decision_ts",
) -> pd.Series:
    """Vectorized rank standardization within each decision cross-section."""
    grouped = series.groupby(level=level, sort=False)
    ranks = grouped.rank(method="average")
    counts = grouped.transform("count")
    scaled = -1.0 + (ranks - 1.0) * (2.0 / (counts - 1.0))
    scaled = scaled.where(counts > 1, 0.0)
    return scaled.where(series.notna()).astype(float).rename(series.name)
