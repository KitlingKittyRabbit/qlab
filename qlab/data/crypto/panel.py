"""Crypto panel utilities shared by private research routes."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ...signal import rank_standardize_cross_section
from .binance_um_klines import execution_opens
from .panel_statistics import rank_standardize_grouped_series
from .strategy_time_contract import (
    ContinuousHoldingTimeContract,
    execution_timestamps,
)


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


def relabel_open_indexed_bars_to_bar_end(
    frame: pd.DataFrame,
    interval: pd.Timedelta,
) -> pd.DataFrame:
    """Relabel open-indexed bars to their first observable bar-end timestamp."""
    normalized = frame.copy().sort_index()
    index = pd.DatetimeIndex(normalized.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    interval = pd.Timedelta(interval)
    if interval <= pd.Timedelta(0):
        raise ValueError("interval must be positive")
    normalized.index = index + interval
    normalized.index.name = "ts"
    if normalized.index.has_duplicates:
        raise ValueError("bar-end relabeling produced duplicate timestamps")
    return normalized


def boundary_reference_prices_from_execution_opens(
    open_series: pd.Series,
    interval: pd.Timedelta = pd.Timedelta(minutes=15),
    *,
    column_name: str = "c",
) -> pd.DataFrame:
    """Select exact UTC boundary prices for non-executable research labels.

    The input is a Binance minute-open series. Only observations whose timestamp
    lies exactly on ``interval`` are retained; missing boundaries remain missing
    rather than being replaced by a later minute.
    """
    interval = pd.Timedelta(interval)
    if interval <= pd.Timedelta(0):
        raise ValueError("interval must be positive")
    series = pd.Series(open_series, copy=True).astype(float).sort_index()
    index = pd.DatetimeIndex(pd.to_datetime(series.index, utc=True))
    series.index = index
    series = series[~series.index.duplicated(keep="last")]
    midnight = series.index.normalize()
    exact_boundary = ((series.index - midnight) % interval) == pd.Timedelta(0)
    sampled = series.loc[exact_boundary].rename(column_name)
    if sampled.empty:
        raise ValueError("execution-open series has no exact reference boundaries")
    sampled.index.name = "ts"
    return sampled.to_frame()


def forward_returns_for_symbol(
    decision_index: pd.DatetimeIndex,
    close_series: pd.Series,
    horizon_delta: pd.Timedelta,
    *,
    name: str = "forward_return",
) -> pd.Series:
    """Build exact timestamp close-to-close forward returns for one symbol.

    This is a label/diagnostic helper. It assumes the value at
    ``decision_index`` and ``decision_index + horizon_delta`` are the entry and
    exit close timestamps. It does not prove that a signal was observable at
    the decision timestamp, and it does not model a tradeable entry/exit lag.
    Strategy PnL that needs live-like semantics must use a point-in-time
    replay/backtest entry instead of this label helper.
    """
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
    """Attach exact close-to-close forward-return labels to a crypto panel.

    The result is appropriate for IC diagnostics and label construction. It is
    not a live-like replay and must not be treated as executable strategy PnL.
    """
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
    """Return panel rows joined with exact close-to-close forward-return labels.

    This preserves the label semantics of ``panel_forward_returns``: useful for
    IC/research labels, not sufficient for point-in-time executable strategy
    results.
    """
    forward_return = panel_forward_returns(
        panel, price_payloads, horizon_delta, close_column=close_column)
    joined = panel.join(forward_return.rename("forward_return"), how="inner")
    if joined.empty:
        raise ValueError(
            "panel-forward-return join is empty for requested horizon")
    return joined.reset_index().sort_values(["decision_ts", "symbol"]).set_index("decision_ts")


def executable_returns_for_symbol(
    signal_bar_end_index: pd.DatetimeIndex,
    minute_klines: pd.DataFrame,
    contract: ContinuousHoldingTimeContract,
    horizon_deltas: Mapping[str, pd.Timedelta],
) -> pd.DataFrame:
    """Build exact open-to-open returns under a validated time contract."""
    ledger = execution_timestamps(signal_bar_end_index, contract, horizon_deltas)
    entry = execution_opens(minute_klines, ledger["execution_ts"])
    exit_prices = execution_opens(minute_klines, ledger["next_execution_ts"])
    ledger["entry_price"] = entry.to_numpy()
    ledger["exit_price"] = exit_prices.to_numpy()
    ledger["exit_ts"] = ledger["next_execution_ts"]
    ledger["execution_price"] = ledger["entry_price"]
    ledger["next_execution_price"] = ledger["exit_price"]
    ledger["executable_return"] = ledger["exit_price"] / ledger["entry_price"] - 1.0
    if not (ledger["next_execution_ts"] - ledger["execution_ts"] == pd.Timedelta(horizon_deltas[contract.return_horizon])).all():
        raise ValueError("Executable holding interval does not equal declared return horizon")
    return ledger


def panel_with_executable_return(
    panel: pd.DataFrame,
    minute_klines_by_symbol: Mapping[str, pd.DataFrame | pd.Series],
    contract: ContinuousHoldingTimeContract,
    horizon_deltas: Mapping[str, pd.Timedelta],
) -> pd.DataFrame:
    if not isinstance(panel.index, pd.MultiIndex) or set(panel.index.names) < {"decision_ts", "symbol"}:
        raise ValueError("panel must use a decision_ts/symbol MultiIndex")
    pieces: list[pd.DataFrame] = []
    for symbol in sorted(panel.index.get_level_values("symbol").unique()):
        if symbol not in minute_klines_by_symbol:
            raise ValueError(f"Missing Binance 1m klines for admitted symbol: {symbol}")
        decisions = pd.DatetimeIndex(panel.xs(symbol, level="symbol").index)
        ledger = executable_returns_for_symbol(
            decisions,
            minute_klines_by_symbol[str(symbol)],
            contract,
            horizon_deltas,
        )
        ledger["symbol"] = symbol
        pieces.append(ledger.set_index(["decision_ts", "symbol"]))
    executable = pd.concat(pieces).sort_index()
    base_panel = panel.copy()
    if "signal_bar_end_ts" in base_panel.columns:
        declared = pd.to_datetime(base_panel["signal_bar_end_ts"], utc=True)
        index_values = pd.DatetimeIndex(
            base_panel.index.get_level_values("decision_ts")
        )
        if not (declared.to_numpy() == index_values.to_numpy()).all():
            raise ValueError("panel signal_bar_end_ts does not match decision_ts")
        base_panel = base_panel.drop(columns="signal_bar_end_ts")
    joined = base_panel.join(executable, how="inner")
    if joined.empty:
        raise ValueError("panel-executable-return join is empty")
    return (
        joined.reset_index()
        .sort_values(["decision_ts", "symbol"])
        .set_index("decision_ts")
    )


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
    decision_days = pd.DatetimeIndex(decision_index.normalize())
    unique_decision_days = pd.DatetimeIndex(decision_days.unique())
    size_by_day = np.log1p(trailing_size.reindex(unique_decision_days, method="ffill"))
    size = pd.Series(
        size_by_day.reindex(decision_days).to_numpy(),
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
        standardized[target_column] = rank_standardize_grouped_series(
            raw[raw_column],
            level="decision_ts",
        )
    return standardized[list(control_columns)]
