from __future__ import annotations

"""KeyStore/CoinGlass v4 1h panel builder.

This module builds factor panels from already refreshed KeyStore/CoinGlass v4
cache payloads. It is data infrastructure only. It does not construct forward
return labels, run research gates, or authorize strategy conclusions.
"""

from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import keystore_coinglass_factors as factor_registry
from .data_roots import resolve_data_root as _resolve_canonical_data_root
from .panel_statistics import rank_standardize_grouped_series
from .symbol_universe import normalize_symbol_list


RAW_SOURCE_BAR_DURATION = {
    "1h": pd.Timedelta(hours=1),
    "2h": pd.Timedelta(hours=2),
    "4h": pd.Timedelta(hours=4),
    "6h": pd.Timedelta(hours=6),
    "8h": pd.Timedelta(hours=8),
    "12h": pd.Timedelta(hours=12),
    "1d": pd.Timedelta(days=1),
}
CACHE_FILENAME_BY_SCOPE = {
    signal_timeframe: f"keystore_coinglass_v4_{signal_timeframe}_cache.pkl"
    for signal_timeframe in RAW_SOURCE_BAR_DURATION
}
MIN_COMMON_PANEL_ROWS = 720


@dataclass(frozen=True)
class PanelArtifacts:
    panel: pd.DataFrame
    summary: pd.DataFrame


def resolve_data_root(data_root: Path | str | None = None) -> Path:
    return _resolve_canonical_data_root(data_root)


def load_admitted_symbols_from_audit(
    path: Path | str,
    *,
    symbol_column: str = "symbol",
    admitted_column: str = "base_admitted",
    admitted_value: str = "yes",
) -> list[str]:
    audit_path = Path(path)
    if not audit_path.exists():
        raise FileNotFoundError(f"universe audit missing: {audit_path}")
    frame = pd.read_csv(audit_path)
    required = {symbol_column, admitted_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("universe audit missing required columns: " + ", ".join(sorted(missing)))
    symbols = frame.loc[
        frame[admitted_column].astype(str).str.lower() == admitted_value.lower(),
        symbol_column,
    ].astype(str)
    return normalize_symbol_list(symbols)


def resolve_admitted_symbols(
    admitted_symbols: Sequence[str] | None = None,
    *,
    universe_audit_path: Path | str | None = None,
) -> list[str]:
    if admitted_symbols is not None:
        symbols = normalize_symbol_list(admitted_symbols)
    elif universe_audit_path is not None:
        symbols = load_admitted_symbols_from_audit(universe_audit_path)
    else:
        raise ValueError("admitted_symbols or universe_audit_path is required")
    if not symbols:
        raise ValueError("admitted symbol list is empty")
    return symbols


def signal_timeframe_from_scope(source_scope: str) -> str:
    for signal_timeframe in RAW_SOURCE_BAR_DURATION:
        if str(source_scope).endswith(f"_{signal_timeframe}"):
            return signal_timeframe
    raise ValueError("unable to infer ksv4 signal timeframe from source_scope: " + str(source_scope))


def normalize_cache_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("cache entry must be a pandas DataFrame")
    normalized = frame.copy()
    normalized.index = pd.to_datetime(normalized.index, utc=True)
    normalized.index.name = "ts"
    normalized = normalized.sort_index()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    return normalized


def required_columns(spec_row: pd.Series) -> list[str]:
    return [column.strip() for column in str(spec_row["required_columns"]).split(",") if column.strip()]


def extract_feature_series(spec_row: pd.Series, frame: pd.DataFrame) -> pd.Series:
    normalized = normalize_cache_frame(frame)
    needed = required_columns(spec_row)
    missing = [column for column in needed if column not in normalized.columns]
    if missing:
        raise ValueError(f"feature {spec_row['feature_name']} missing required columns: {', '.join(missing)}")

    transform = str(spec_row["panel_transform"])
    feature_name = str(spec_row["feature_name"])
    if transform == "raw_column":
        series = normalized[needed[0]].astype(float)
    elif transform == "delta1_raw_column":
        series = normalized[needed[0]].astype(float).diff()
    elif transform == "log_ratio":
        numerator = normalized[needed[0]].astype(float)
        denominator = normalized[needed[1]].astype(float)
        invalid = (numerator <= 0.0) | (denominator <= 0.0)
        if invalid.any():
            raise ValueError(f"feature {feature_name} requires strictly positive inputs for log_ratio")
        series = np.log(numerator) - np.log(denominator)
    elif transform == "log1p_ratio":
        numerator = normalized[needed[0]].astype(float)
        denominator = normalized[needed[1]].astype(float)
        invalid = (numerator < 0.0) | (denominator < 0.0)
        if invalid.any():
            raise ValueError(f"feature {feature_name} requires non-negative inputs for log1p_ratio")
        series = np.log1p(numerator) - np.log1p(denominator)
    elif transform == "buy_minus_sell":
        series = normalized[needed[0]].astype(float) - normalized[needed[1]].astype(float)
    elif transform == "buy_sell_imbalance":
        buy = normalized[needed[0]].astype(float)
        sell = normalized[needed[1]].astype(float)
        denominator = (buy + sell).replace(0.0, pd.NA)
        series = (buy - sell) / denominator
    else:
        raise ValueError(f"unsupported ksv4 panel_transform for {feature_name}: {transform}")

    series = series.rename(feature_name).dropna()
    if series.empty:
        raise ValueError(f"feature {feature_name} is empty after transform")
    return series


def standardize_panel_cross_section(panel: pd.DataFrame, registry_frame: pd.DataFrame) -> pd.DataFrame:
    standardized = panel.copy()
    policy_by_feature = registry_frame.set_index("feature_name")["cross_section_standardization"].to_dict()
    for feature_name in [
        column
        for column in standardized.columns
        if column not in {"label_ts", "signal_bar_end_ts"}
    ]:
        policy = policy_by_feature.get(feature_name, "none")
        if policy == "none":
            continue
        if policy != "rank_to_minus1_1":
            raise ValueError(f"unsupported ksv4 standardization policy for {feature_name}: {policy}")
        standardized[feature_name] = rank_standardize_grouped_series(
            standardized[feature_name],
            level="decision_ts",
        )
    return standardized


def load_cache_payloads(
    scopes: Sequence[str],
    data_root: Path | str | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    root = resolve_data_root(data_root)
    payloads: dict[str, dict[str, pd.DataFrame]] = {}
    for scope in scopes:
        signal_timeframe = signal_timeframe_from_scope(scope)
        filename = CACHE_FILENAME_BY_SCOPE[signal_timeframe]
        path = root / "caches" / filename
        if not path.exists():
            raise FileNotFoundError(f"required ksv4 cache missing for {scope}: {path}")
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"ksv4 cache payload must be a dict: {path}")
        payloads[scope] = payload
    return payloads


def overlay_observed_cache_frames(
    cache_payloads: dict[str, dict[str, pd.DataFrame]],
    observed_frames: dict[tuple[str, str], pd.DataFrame],
) -> dict[str, dict[str, pd.DataFrame]]:
    """Overlay verified observations without mutating active cache payloads."""
    result = {
        str(scope): {
            str(cache_key): normalize_cache_frame(frame)
            for cache_key, frame in payload.items()
        }
        for scope, payload in cache_payloads.items()
    }
    for (scope, cache_key), observed in observed_frames.items():
        if scope not in result:
            raise ValueError(f"observed frame references unknown cache scope: {scope}")
        if cache_key not in result[scope]:
            raise ValueError(
                f"observed frame references unknown cache entry: {scope}/{cache_key}"
            )
        incoming = normalize_cache_frame(observed)
        if incoming.empty:
            raise ValueError(f"observed frame is empty: {scope}/{cache_key}")
        combined = pd.concat([result[scope][cache_key], incoming]).sort_index()
        result[scope][cache_key] = combined[
            ~combined.index.duplicated(keep="last")
        ]
    return result


def extract_symbol_feature_series(
    symbol: str,
    spec_row: pd.Series,
    cache_payloads: dict[str, dict[str, pd.DataFrame]],
) -> pd.Series:
    scope = str(spec_row["source_scope"])
    cache_key = f"{symbol}_{spec_row['endpoint']}"
    if scope not in cache_payloads:
        raise ValueError(f"cache scope not loaded for {spec_row['feature_name']}: {scope}")
    if cache_key not in cache_payloads[scope]:
        raise ValueError(f"missing ksv4 cache entry for admitted symbol {symbol}: {cache_key}")
    return extract_feature_series(spec_row, cache_payloads[scope][cache_key])


def source_index_to_native_bar_end(
    index: pd.DatetimeIndex,
    *,
    signal_timeframe: str,
    timestamp_kind: str,
) -> pd.DatetimeIndex:
    source_index = pd.DatetimeIndex(pd.to_datetime(index, utc=True)).as_unit("ns")
    if timestamp_kind == "bar_start":
        return source_index + RAW_SOURCE_BAR_DURATION[signal_timeframe]
    if timestamp_kind == "bar_end":
        return source_index
    raise ValueError(f"unsupported timestamp_kind: {timestamp_kind}")


def build_decision_grid_index(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DatetimeIndex:
    anchor = start_ts.normalize()
    if start_ts > anchor:
        steps = int(np.ceil((start_ts - anchor) / pd.Timedelta(hours=1)))
        grid_start = anchor + steps * pd.Timedelta(hours=1)
    else:
        grid_start = anchor
    end_anchor = end_ts.normalize()
    steps = int(np.floor((end_ts - end_anchor) / pd.Timedelta(hours=1)))
    grid_end = end_anchor + steps * pd.Timedelta(hours=1)
    if grid_end < grid_start:
        return pd.DatetimeIndex([], tz="UTC", name="decision_ts")
    return pd.date_range(
        grid_start, grid_end, freq="1h", tz="UTC", name="decision_ts"
    ).as_unit("ns")


def route_common_decision_index(
    admitted_symbols: Sequence[str],
    registry_frame: pd.DataFrame,
    cache_payloads: dict[str, dict[str, pd.DataFrame]],
) -> pd.DatetimeIndex:
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for symbol in admitted_symbols:
        symbol_starts: list[pd.Timestamp] = []
        symbol_ends: list[pd.Timestamp] = []
        for _, spec_row in registry_frame.iterrows():
            series = extract_symbol_feature_series(symbol, spec_row, cache_payloads)
            signal_timeframe = signal_timeframe_from_scope(spec_row["source_scope"])
            native_bar_ends = source_index_to_native_bar_end(
                pd.DatetimeIndex(series.index),
                signal_timeframe=signal_timeframe,
                timestamp_kind=str(spec_row["timestamp_kind"]),
            )
            if native_bar_ends.empty:
                raise ValueError(f"feature {spec_row['feature_name']} has no available ksv4 decisions for {symbol}")
            symbol_starts.append(native_bar_ends.min())
            symbol_ends.append(native_bar_ends.max())
        starts.append(max(symbol_starts))
        ends.append(min(symbol_ends))
    return build_decision_grid_index(max(starts), min(ends))


def align_series_to_decision_grid(
    series: pd.Series,
    decision_index: pd.DatetimeIndex,
    signal_timeframe: str,
    timestamp_kind: str,
) -> pd.Series:
    normalized_decision_index = pd.DatetimeIndex(
        pd.to_datetime(decision_index, utc=True), name=decision_index.name
    ).as_unit("ns")
    native_bar_ends = pd.DataFrame(
        {
            "native_bar_end_ts": source_index_to_native_bar_end(
                pd.DatetimeIndex(series.index),
                signal_timeframe=signal_timeframe,
                timestamp_kind=timestamp_kind,
            ),
            series.name: series.to_numpy(),
        }
    ).sort_values("native_bar_end_ts")
    aligned = pd.merge_asof(
        pd.DataFrame({"decision_ts": normalized_decision_index}),
        native_bar_ends,
        left_on="decision_ts",
        right_on="native_bar_end_ts",
        direction="backward",
    )
    aligned.loc[
        ~aligned["decision_ts"].isin(native_bar_ends["native_bar_end_ts"]),
        series.name,
    ] = np.nan
    return pd.Series(
        aligned[series.name].to_numpy(),
        index=normalized_decision_index,
        name=series.name,
    )


def build_symbol_frame_on_grid(
    symbol: str,
    registry_frame: pd.DataFrame,
    cache_payloads: dict[str, dict[str, pd.DataFrame]],
    decision_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    pieces: list[pd.Series] = []
    for _, spec_row in registry_frame.iterrows():
        raw_series = extract_symbol_feature_series(symbol, spec_row, cache_payloads)
        signal_timeframe = signal_timeframe_from_scope(spec_row["source_scope"])
        pieces.append(
            align_series_to_decision_grid(
                raw_series.rename(spec_row["feature_name"]),
                decision_index,
                signal_timeframe,
                str(spec_row["timestamp_kind"]),
            )
        )
    frame = pd.concat(pieces, axis=1)
    frame.index = decision_index
    frame.index.name = "decision_ts"
    return frame.sort_index()


def build_panel_from_payloads(
    admitted_symbols: Sequence[str],
    cache_payloads: dict[str, dict[str, pd.DataFrame]],
    registry_frame: pd.DataFrame,
    min_common_rows: int = MIN_COMMON_PANEL_ROWS,
) -> PanelArtifacts:
    symbols = normalize_symbol_list(admitted_symbols)
    if not symbols:
        raise ValueError("admitted symbol list is empty")
    requested_order = {
        str(feature_name): idx
        for idx, feature_name in enumerate(registry_frame["feature_name"].tolist())
    }
    active_registry = factor_registry.validate_factor_eligibility_registry(registry_frame)
    active_registry = active_registry.loc[active_registry["base_panel_allowed"] == "yes"].reset_index(drop=True)
    active_registry["__feature_order"] = active_registry["feature_name"].map(requested_order)
    active_registry = active_registry.sort_values("__feature_order").drop(columns="__feature_order").reset_index(drop=True)
    if active_registry.empty:
        raise ValueError("no ksv4 base-panel factors available")

    decision_index = route_common_decision_index(symbols, active_registry, cache_payloads)
    if decision_index.empty:
        raise ValueError("ksv4 1h decision grid is empty")

    feature_names = active_registry["feature_name"].tolist()
    panel_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    source_frequency_label = "+".join(
        sorted({signal_timeframe_from_scope(scope) for scope in active_registry["source_scope"]})
    )
    for symbol in symbols:
        aligned = build_symbol_frame_on_grid(symbol, active_registry, cache_payloads, decision_index)
        aligned["label_ts"] = aligned.index
        aligned["signal_bar_end_ts"] = aligned.index
        aligned["symbol"] = symbol
        aligned = aligned.reset_index(names="decision_ts").set_index(["decision_ts", "symbol"]).sort_index()
        panel_frames.append(aligned)
        summary_rows.append(
            {
                "panel_frequency": "1h",
                "source_frequency": source_frequency_label,
                "symbol": symbol,
                "rows": len(aligned),
                "feature_count": len(feature_names),
                "label_start": aligned["label_ts"].min(),
                "label_end": aligned["label_ts"].max(),
                "decision_start": aligned.index.get_level_values("decision_ts").min(),
                "decision_end": aligned.index.get_level_values("decision_ts").max(),
            }
        )

    panel = pd.concat(panel_frames).sort_index()
    panel = panel[["label_ts", "signal_bar_end_ts", *feature_names]]
    panel = standardize_panel_cross_section(panel, active_registry)
    summary = pd.DataFrame(summary_rows).sort_values("symbol").reset_index(drop=True)
    shared_rows = int(summary["rows"].min()) if not summary.empty else 0
    if shared_rows < min_common_rows:
        raise ValueError(f"ksv4 common panel overlap below threshold: got {shared_rows}, need at least {min_common_rows}")
    return PanelArtifacts(panel=panel, summary=summary)


def build_panel(
    *,
    feature_names: Sequence[str] | None = None,
    admitted_symbols: Sequence[str] | None = None,
    universe_audit_path: Path | str | None = None,
    min_common_rows: int = MIN_COMMON_PANEL_ROWS,
    data_root: Path | str | None = None,
) -> PanelArtifacts:
    symbols = resolve_admitted_symbols(admitted_symbols, universe_audit_path=universe_audit_path)
    active_registry = (
        factor_registry.base_panel_registry("1h")
        if feature_names is None
        else factor_registry.feature_registry_for_panel(tuple(feature_names), frequency="1h")
    )
    payloads = load_cache_payloads(sorted(active_registry["source_scope"].unique()), data_root=data_root)
    return build_panel_from_payloads(
        admitted_symbols=symbols,
        cache_payloads=payloads,
        registry_frame=active_registry,
        min_common_rows=min_common_rows,
    )


__all__ = [
    "CACHE_FILENAME_BY_SCOPE",
    "MIN_COMMON_PANEL_ROWS",
    "RAW_SOURCE_BAR_DURATION",
    "PanelArtifacts",
    "align_series_to_decision_grid",
    "build_decision_grid_index",
    "build_panel",
    "build_panel_from_payloads",
    "build_symbol_frame_on_grid",
    "extract_feature_series",
    "load_admitted_symbols_from_audit",
    "load_cache_payloads",
    "normalize_cache_frame",
    "overlay_observed_cache_frames",
    "resolve_admitted_symbols",
    "resolve_data_root",
    "route_common_decision_index",
    "signal_timeframe_from_scope",
    "source_index_to_native_bar_end",
    "standardize_panel_cross_section",
]
