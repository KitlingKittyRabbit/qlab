"""Crypto panel utilities shared by private research routes."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ...signal import rank_standardize_cross_section


def load_pickle_payload(path: str | Path) -> dict[str, pd.DataFrame]:
    payload_path = Path(path).expanduser()
    if not payload_path.exists():
        raise FileNotFoundError(f"pickle payload missing: {payload_path}")
    with payload_path.open("rb") as file_handle:
        payload = pickle.load(file_handle)
    if not isinstance(payload, dict):
        raise ValueError("pickle payload must be a dict")
    return payload


def normalize_price_frame(frame: pd.DataFrame, *, close_column: str = "c") -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("price cache entry must be a pandas DataFrame")
    normalized = frame.copy().sort_index()
    index = pd.DatetimeIndex(normalized.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    normalized.index = index
    normalized.index.name = "ts"
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    if close_column not in normalized.columns:
        raise ValueError(
            f"price cache entry missing close column {close_column!r}")
    return normalized


def forward_returns_for_symbol(
    decision_index: pd.DatetimeIndex,
    close_series: pd.Series,
    horizon_delta: pd.Timedelta,
    *,
    name: str = "forward_return",
) -> pd.Series:
    horizon_delta = pd.Timedelta(horizon_delta)
    future_index = decision_index + horizon_delta
    entry = close_series.reindex(decision_index)
    exit_prices = close_series.reindex(future_index)
    returns = pd.Series(
        (exit_prices.to_numpy() / entry.to_numpy()) - 1.0,
        index=decision_index,
        name=name,
    )
    return returns.replace([float("inf"), float("-inf")], pd.NA).dropna()


def panel_forward_returns(
    panel: pd.DataFrame,
    price_payloads: Mapping[str, pd.DataFrame],
    horizon_delta: pd.Timedelta,
    *,
    close_column: str = "c",
) -> pd.Series:
    pieces: list[pd.Series] = []
    for symbol in sorted(panel.index.get_level_values("symbol").unique()):
        if symbol not in price_payloads:
            raise ValueError(
                "missing price cache entry for admitted symbol: " + str(symbol))
        symbol_panel = panel.xs(symbol, level="symbol")
        close_series = normalize_price_frame(price_payloads[str(
            symbol)], close_column=close_column)[close_column].astype(float)
        returns = forward_returns_for_symbol(pd.DatetimeIndex(
            symbol_panel.index), close_series, horizon_delta)
        if returns.empty:
            continue
        pieces.append(
            returns.to_frame("forward_return")
            .assign(symbol=symbol)
            .reset_index(names="decision_ts")
            .set_index(["decision_ts", "symbol"])["forward_return"]
        )
    if not pieces:
        raise ValueError("no forward returns available for requested horizon")
    return pd.concat(pieces).sort_index()


def panel_with_forward_return(
    panel: pd.DataFrame,
    price_payloads: Mapping[str, pd.DataFrame],
    horizon_delta: pd.Timedelta,
    *,
    close_column: str = "c",
) -> pd.DataFrame:
    forward_return = panel_forward_returns(
        panel, price_payloads, horizon_delta, close_column=close_column)
    joined = panel.join(forward_return.rename("forward_return"), how="inner")
    if joined.empty:
        raise ValueError(
            "panel-forward-return join is empty for requested horizon")
    return joined.reset_index().sort_values(["decision_ts", "symbol"]).set_index("decision_ts")


def rank_standardize_with_nans(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    result = pd.Series(np.nan, index=series.index,
                       dtype=float, name=series.name)
    if valid.empty:
        return result
    result.loc[valid.index] = rank_standardize_cross_section(valid)
    return result


def price_controls_for_symbol(
    decision_index: pd.DatetimeIndex,
    price_frame: pd.DataFrame,
    *,
    close_column: str = "c",
    volume_column: str = "v",
    size_lookback_days: int = 90,
    size_min_days: int = 20,
    momentum_lookback_days: int = 30,
    volatility_lookback_days: int = 30,
    volatility_min_obs: int = 20,
) -> pd.DataFrame:
    normalized = normalize_price_frame(price_frame, close_column=close_column)
    close = normalized[close_column].astype(float)
    volume = normalized[volume_column].astype(
        float) if volume_column in normalized.columns else pd.Series(0.0, index=normalized.index)
    decision_index = pd.DatetimeIndex(decision_index).sort_values().unique()
    current_close = close.reindex(decision_index)
    lagged_close = close.reindex(
        decision_index - pd.Timedelta(days=momentum_lookback_days))
    momentum = pd.Series((current_close.to_numpy() /
                         lagged_close.to_numpy()) - 1.0, index=decision_index)
    log_close = np.log(close.replace(0.0, np.nan))
    log_returns = log_close.diff()
    volatility = log_returns.rolling(
        f"{volatility_lookback_days}D",
        min_periods=volatility_min_obs,
    ).std().reindex(decision_index)
    daily_dollar_volume = (close * volume).resample("1D").sum().dropna()
    trailing_size = daily_dollar_volume.shift(1).rolling(
        window=size_lookback_days,
        min_periods=size_min_days,
    ).median()
    size = pd.Series(
        np.log1p(trailing_size.reindex(
            decision_index.normalize(), method="ffill").to_numpy()),
        index=decision_index,
    )
    return pd.DataFrame(
        {
            "size_control_raw": size,
            "momentum_control_raw": momentum,
            "volatility_control_raw": volatility,
        },
        index=decision_index,
    ).replace([float("inf"), float("-inf")], np.nan)


def build_control_panel(
    panel: pd.DataFrame,
    price_payloads: Mapping[str, pd.DataFrame],
    *,
    control_columns: Sequence[str] = (
        "size_control", "momentum_control", "volatility_control"),
    size_lookback_days: int = 90,
    size_min_days: int = 20,
    momentum_lookback_days: int = 30,
    volatility_lookback_days: int = 30,
    volatility_min_obs: int = 20,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for symbol in sorted(panel.index.get_level_values("symbol").unique()):
        if symbol not in price_payloads:
            raise ValueError(
                "missing price cache entry for admitted symbol: " + str(symbol))
        decision_index = pd.DatetimeIndex(
            panel.xs(symbol, level="symbol").index)
        symbol_controls = price_controls_for_symbol(
            decision_index,
            price_payloads[str(symbol)],
            size_lookback_days=size_lookback_days,
            size_min_days=size_min_days,
            momentum_lookback_days=momentum_lookback_days,
            volatility_lookback_days=volatility_lookback_days,
            volatility_min_obs=volatility_min_obs,
        )
        pieces.append(symbol_controls.assign(symbol=symbol).reset_index(
            names="decision_ts").set_index(["decision_ts", "symbol"]))
    if not pieces:
        return pd.DataFrame(columns=list(control_columns))
    raw = pd.concat(pieces).sort_index()
    standardized = pd.DataFrame(index=raw.index)
    for raw_column, target_column in (
        ("size_control_raw", "size_control"),
        ("momentum_control_raw", "momentum_control"),
        ("volatility_control_raw", "volatility_control"),
    ):
        if target_column not in control_columns:
            continue
        standardized[target_column] = raw.groupby(level="decision_ts")[
            raw_column].transform(rank_standardize_with_nans)
    return standardized[list(control_columns)]
