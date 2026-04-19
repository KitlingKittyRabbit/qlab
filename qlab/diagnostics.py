"""
Research diagnostics utilities.

This module provides small, reusable building blocks for factor analysis:
- forward return construction for multiple horizons
- IC decay computation across horizons
- quantile forward-return summaries
"""

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .signal import ic


def forward_returns(
    prices: pd.Series,
    horizons: Iterable[int],
) -> pd.DataFrame:
    """
    Build forward simple returns for multiple horizons.

    Parameters
    ----------
    prices : pd.Series
        Price series with a sortable index.
    horizons : iterable of int
        Forward horizons in bars. Each value must be positive.

    Returns
    -------
    pd.DataFrame
        Columns named ``fwd_{h}``, where each column contains
        ``prices.shift(-h) / prices - 1``.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series")

    clean_prices = prices.sort_index().astype(float)
    horizon_list = _normalize_horizons(horizons)

    data = {
        f"fwd_{h}": clean_prices.shift(-h) / clean_prices - 1.0
        for h in horizon_list
    }
    return pd.DataFrame(data, index=clean_prices.index)


def ic_decay(
    factor: pd.Series,
    horizons: Optional[Iterable[int]] = None,
    prices: Optional[pd.Series] = None,
    forward_ret: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute IC across multiple forward-return horizons.

    Provide either ``prices`` + ``horizons`` or a precomputed
    ``forward_ret`` DataFrame whose columns are different horizons.

    Parameters
    ----------
    factor : pd.Series
        Factor values indexed by date/time.
    horizons : iterable of int, optional
        Horizons in bars. Required when ``prices`` is provided.
    prices : pd.Series, optional
        Price series used to build forward returns.
    forward_ret : pd.DataFrame, optional
        Precomputed forward returns. Each column represents one horizon.

    Returns
    -------
    pd.DataFrame
        Columns: ``horizon``, ``column``, ``ic``, ``n_obs``.
    """
    if not isinstance(factor, pd.Series):
        raise TypeError("factor must be a pandas Series")
    if (prices is None) == (forward_ret is None):
        raise ValueError("Provide exactly one of prices or forward_ret")

    factor = factor.sort_index().astype(float)

    if prices is not None:
        horizon_list = _normalize_horizons(horizons)
        ret_df = forward_returns(prices, horizon_list)
        meta = [(h, f"fwd_{h}") for h in horizon_list]
    else:
        if not isinstance(forward_ret, pd.DataFrame):
            raise TypeError("forward_ret must be a pandas DataFrame")
        ret_df = forward_ret.sort_index().astype(float)
        if horizons is not None:
            horizon_list = _normalize_horizons(horizons)
            expected_columns = [f"fwd_{h}" for h in horizon_list]
            missing = [col for col in expected_columns if col not in ret_df.columns]
            if missing:
                raise ValueError(
                    f"forward_ret is missing requested columns: {missing}"
                )
            ret_df = ret_df.loc[:, expected_columns]
            meta = list(zip(horizon_list, expected_columns))
        else:
            meta = [(_extract_horizon(col), str(col)) for col in ret_df.columns]

    rows = []
    for horizon, column in meta:
        aligned = pd.concat([factor.rename("factor"), ret_df[column]], axis=1).dropna()
        rows.append(
            {
                "horizon": horizon,
                "column": column,
                "ic": ic(aligned["factor"].values, aligned[column].values),
                "n_obs": int(len(aligned)),
            }
        )

    return pd.DataFrame(rows)


def quantile_returns(
    factor: pd.Series,
    n_quantiles: int = 5,
    horizon: Optional[int] = None,
    prices: Optional[pd.Series] = None,
    forward_ret: Optional[pd.Series | pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute mean forward return by factor quantile for a single horizon.

    Provide either ``prices`` + ``horizon`` or a single forward-return series.

    Parameters
    ----------
    factor : pd.Series
        Factor values indexed by date/time.
    n_quantiles : int
        Number of quantile buckets.
    horizon : int, optional
        Forward horizon in bars. Required when ``prices`` is provided.
    prices : pd.Series, optional
        Price series used to build forward returns.
    forward_ret : pd.Series or single-column pd.DataFrame, optional
        Precomputed forward returns for one horizon.

    Returns
    -------
    pd.DataFrame
        Columns: ``quantile``, ``mean_return``, ``count``.
    """
    if not isinstance(factor, pd.Series):
        raise TypeError("factor must be a pandas Series")
    if (prices is None) == (forward_ret is None):
        raise ValueError("Provide exactly one of prices or forward_ret")
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be at least 2")

    factor = factor.sort_index().astype(float)

    if prices is not None:
        if horizon is None:
            raise ValueError("horizon must be provided when prices is used")
        forward_series = forward_returns(prices, [horizon]).iloc[:, 0]
    else:
        if isinstance(forward_ret, pd.DataFrame):
            if forward_ret.shape[1] != 1:
                raise ValueError("forward_ret DataFrame must have exactly one column")
            forward_series = forward_ret.iloc[:, 0]
        elif isinstance(forward_ret, pd.Series):
            forward_series = forward_ret
        else:
            raise TypeError("forward_ret must be a pandas Series or single-column DataFrame")

        if horizon is None:
            horizon = _extract_horizon(getattr(forward_series, "name", None))

    aligned = pd.concat(
        [factor.rename("factor"), forward_series.sort_index().astype(float).rename("forward_return")],
        axis=1,
    ).dropna()

    if len(aligned) < n_quantiles:
        raise ValueError("not enough aligned observations for the requested quantiles")

    ranked = aligned["factor"].rank(method="first")
    aligned["quantile"] = pd.qcut(ranked, q=n_quantiles, labels=False) + 1

    result = (
        aligned.groupby("quantile", observed=False)["forward_return"]
        .agg(mean_return="mean", count="size")
        .reset_index()
    )
    result["horizon"] = horizon
    return result.loc[:, ["horizon", "quantile", "mean_return", "count"]]


def _normalize_horizons(horizons: Optional[Iterable[int]]) -> list[int]:
    if horizons is None:
        raise ValueError("horizons must be provided")
    horizon_list = [int(h) for h in horizons]
    if not horizon_list:
        raise ValueError("horizons must not be empty")
    if any(h <= 0 for h in horizon_list):
        raise ValueError("all horizons must be positive integers")
    if len(set(horizon_list)) != len(horizon_list):
        raise ValueError("horizons must be unique")
    return horizon_list


def _extract_horizon(column: object) -> Optional[int]:
    if isinstance(column, str) and column.startswith("fwd_"):
        try:
            return int(column.split("_", 1)[1])
        except ValueError:
            return None
    return None