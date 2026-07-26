from __future__ import annotations

"""Lifecycle: candidate.

Schema parsers for KeyStore/CoinGlass v4 cache construction. Promote to active
with the data-source route after coverage and overlap validation pass.
"""

from typing import Any

import pandas as pd


def to_datetime_index(values: Any) -> pd.DatetimeIndex:
    series = pd.Series(values).dropna()
    if series.empty:
        return pd.DatetimeIndex([], tz="UTC")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        unit = "ms" if float(numeric.abs().max()) >= 10_000_000_000 else "s"
        return pd.to_datetime(numeric, unit=unit, utc=True)
    return pd.to_datetime(series, utc=True, errors="coerce")


def normalize_timeseries_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    normalized = frame.sort_index()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    return normalized


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        values = pd.Series([float("nan")] * len(frame), dtype="float64")
    else:
        values = pd.to_numeric(frame[column], errors="coerce")
    index = frame["ts"] if "ts" in frame.columns else frame.index
    return pd.Series(values.to_numpy(), index=index)


def _first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    return next((column for column in candidates if column in frame.columns), "")


def _time_column(frame: pd.DataFrame) -> str:
    for column in ("time", "timestamp", "t", "date"):
        if column in frame.columns:
            return column
    raise ValueError(f"No timestamp column found in columns={list(frame.columns)}")


def _frame_from_rows(rows: list[dict[str, Any]] | list[Any]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame()
    ts_col = _time_column(frame)
    frame["ts"] = to_datetime_index(frame[ts_col])
    return frame


def _parse_ohlc(rows: list[Any], *, prefix: str | None = None) -> pd.DataFrame:
    frame = _frame_from_rows(rows)
    if frame.empty:
        return pd.DataFrame()
    source_columns = ["open", "high", "low", "close"]
    if not all(column in frame.columns for column in source_columns):
        source_columns = ["o", "h", "l", "c"]
    output = pd.DataFrame(index=frame["ts"])
    names = ["open", "high", "low", "close"]
    for source, name in zip(source_columns, names):
        column_name = f"{prefix}_{name}" if prefix else name
        output[column_name] = pd.to_numeric(frame[source], errors="coerce").to_numpy()
    return normalize_timeseries_frame(output)


def _parse_liquidation(rows: list[Any]) -> pd.DataFrame:
    frame = _frame_from_rows(rows)
    if frame.empty:
        return pd.DataFrame()
    long_candidates = ["aggregated_long_liquidation_usd", "longLiquidationUsd", "long_liquidation_usd", "long_liq", "long"]
    short_candidates = [
        "aggregated_short_liquidation_usd",
        "shortLiquidationUsd",
        "short_liquidation_usd",
        "short_liq",
        "short",
    ]
    long_col = next((column for column in long_candidates if column in frame.columns), "")
    short_col = next((column for column in short_candidates if column in frame.columns), "")
    if not long_col or not short_col:
        numeric_columns = [column for column in frame.columns if column not in {"ts", "time", "timestamp", "t", "date"}]
        if len(numeric_columns) < 2:
            return pd.DataFrame(index=frame["ts"])
        long_col, short_col = numeric_columns[:2]
    output = pd.DataFrame(index=frame["ts"])
    output["long_liq"] = pd.to_numeric(frame[long_col], errors="coerce").to_numpy()
    output["short_liq"] = pd.to_numeric(frame[short_col], errors="coerce").to_numpy()
    output["net_liq"] = output["long_liq"] - output["short_liq"]
    output["total_liq"] = output["long_liq"] + output["short_liq"]
    return normalize_timeseries_frame(output)


def _parse_ls_ratio(rows: list[Any], prefix: str) -> pd.DataFrame:
    frame = _frame_from_rows(rows)
    if frame.empty:
        return pd.DataFrame()
    candidate_map = {
        "global": {
            "ratio": ["global_account_long_short_ratio", "longShortRatio", "long_short_ratio"],
            "long": ["global_account_long_percent", "longAccount", "long_account"],
            "short": ["global_account_short_percent", "shortAccount", "short_account"],
        },
        "top_acct": {
            "ratio": ["top_account_long_short_ratio", "longShortRatio", "long_short_ratio"],
            "long": ["top_account_long_percent", "longAccount", "long_account"],
            "short": ["top_account_short_percent", "shortAccount", "short_account"],
        },
        "top_pos": {
            "ratio": ["top_position_long_short_ratio", "longShortRatio", "long_short_ratio"],
            "long": ["top_position_long_percent", "longAccount", "long_account"],
            "short": ["top_position_short_percent", "shortAccount", "short_account"],
        },
    }
    candidates = candidate_map[prefix]
    ratio_col = _first_existing_column(frame, candidates["ratio"])
    long_col = _first_existing_column(frame, candidates["long"])
    short_col = _first_existing_column(frame, candidates["short"])
    output = pd.DataFrame(index=frame["ts"])
    output[f"{prefix}_ls_ratio"] = _numeric(frame, ratio_col)
    if long_col in frame.columns:
        output[f"{prefix}_long_pct"] = _numeric(frame, long_col)
    if short_col in frame.columns:
        output[f"{prefix}_short_pct"] = _numeric(frame, short_col)
    return normalize_timeseries_frame(output)


def _parse_taker(rows: list[Any], *, aggregated: bool) -> pd.DataFrame:
    frame = _frame_from_rows(rows)
    if frame.empty:
        return pd.DataFrame()
    buy_col = "aggregated_buy_volume_usd" if aggregated else "taker_buy_volume_usd"
    sell_col = "aggregated_sell_volume_usd" if aggregated else "taker_sell_volume_usd"
    output = pd.DataFrame(index=frame["ts"])
    output["buy"] = _numeric(frame, buy_col)
    output["sell"] = _numeric(frame, sell_col)
    return normalize_timeseries_frame(output)


def _parse_named_numeric(rows: list[Any], columns: list[str]) -> pd.DataFrame:
    frame = _frame_from_rows(rows)
    if frame.empty:
        return pd.DataFrame()
    output = pd.DataFrame(index=frame["ts"])
    for column in columns:
        if column in frame.columns:
            output[column] = _numeric(frame, column)
    return normalize_timeseries_frame(output)


def _parse_generic_numeric(rows: list[Any]) -> pd.DataFrame:
    frame = _frame_from_rows(rows)
    if frame.empty:
        return pd.DataFrame()
    excluded = {"ts", "time", "timestamp", "t", "date"}
    output = pd.DataFrame(index=frame["ts"])
    for column in frame.columns:
        if column in excluded:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            output[column] = values.to_numpy()
    return normalize_timeseries_frame(output)


def parse_history_frame(parser: str, rows: list[Any]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    if parser == "ohlc":
        return _parse_ohlc(rows)
    if parser == "oi_ohlc":
        return _parse_ohlc(rows, prefix="oi")
    if parser == "fr_ohlc":
        frame = _parse_ohlc(rows)
        if frame.empty:
            return frame
        output = pd.DataFrame(index=frame.index)
        output["fr_close"] = frame["close"]
        return normalize_timeseries_frame(output)
    if parser == "liquidation":
        return _parse_liquidation(rows)
    if parser == "global_ls":
        return _parse_ls_ratio(rows, "global")
    if parser == "top_acct":
        return _parse_ls_ratio(rows, "top_acct")
    if parser == "top_pos":
        return _parse_ls_ratio(rows, "top_pos")
    if parser == "taker_pair":
        return _parse_taker(rows, aggregated=False)
    if parser == "taker_agg":
        return _parse_taker(rows, aggregated=True)
    if parser == "basis":
        return _parse_named_numeric(rows, ["open_basis", "close_basis", "open_change", "close_change"])
    if parser == "coinbase_premium":
        return _parse_named_numeric(rows, ["premium", "premium_rate", "coinbase_price"])
    if parser == "bitfinex_margin":
        return _parse_named_numeric(rows, ["long_quantity", "short_quantity"])
    if parser == "generic_numeric":
        return _parse_generic_numeric(rows)
    raise ValueError(f"Unknown KeyStore parser: {parser}")
