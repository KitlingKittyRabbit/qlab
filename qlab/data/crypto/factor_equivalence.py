"""Factor-registry equivalence for the KSV4 source-consistency audit.

This module is the only formal entry point that turns a realtime/historical
raw pair into a frozen-factor comparison.  Raw source-shape diagnostics are
kept in the result, but extra parser fields do not decide factor equivalence.
The research service may collect inputs and persist the returned records; it
must not duplicate these transforms or ranking rules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any

import pandas as pd


FACTOR_EQUIVALENCE_CONTRACT_VERSION = "ksv4_factor_equivalence_v1"
FACTOR_EQUIVALENCE_STATUSES = frozenset(
    {
        "exact_match",
        "value_mismatch_decision_equivalent",
        "decision_material_mismatch",
        "cross_section_incomplete",
        "scope_not_comparable",
        "native_identity_mismatch",
        "required_field_mismatch",
        "missing_prior_observation",
    }
)


def _stable_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def factor_required_columns(registry_row: Mapping[str, object]) -> tuple[str, ...]:
    raw = registry_row.get("required_columns", "")
    if isinstance(raw, str):
        columns = tuple(item.strip() for item in raw.split(",") if item.strip())
    else:
        columns = tuple(str(item).strip() for item in raw if str(item).strip())
    if not columns:
        raise ValueError("factor registry required_columns must not be empty")
    return columns


def _duration(signal_timeframe: str) -> float:
    values = {"1h": 3600.0, "12h": 12.0 * 3600.0, "1d": 24.0 * 3600.0}
    try:
        return values[str(signal_timeframe)]
    except KeyError as exc:
        raise ValueError(f"unsupported KSV4 signal timeframe: {signal_timeframe}") from exc


def previous_native_label(
    target_label_ts: str,
    *,
    signal_timeframe: str,
) -> str:
    """Return the immediately preceding native label for a delta1 factor."""
    target = pd.Timestamp(target_label_ts)
    target = target.tz_localize("UTC") if target.tz is None else target.tz_convert("UTC")
    return (target - pd.Timedelta(seconds=_duration(signal_timeframe))).isoformat()


def _utc_timestamp(value: object):
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tz is None
        else timestamp.tz_convert("UTC")
    )


def _expected_native_bar_end(
    target_label_ts: object,
    *,
    signal_timeframe: str,
    timestamp_kind: str,
):
    target = _utc_timestamp(target_label_ts)
    if str(timestamp_kind) == "bar_start":
        return target + pd.Timedelta(seconds=_duration(signal_timeframe))
    if str(timestamp_kind) == "bar_end":
        return target
    raise ValueError(f"unsupported registry timestamp_kind: {timestamp_kind}")


def apply_factor_registry_transform(
    registry_row: Mapping[str, object],
    values: Mapping[str, object],
    *,
    previous_values: Mapping[str, object] | None = None,
) -> float:
    """Apply the registry transform to one native observation."""
    row = _as_mapping(registry_row, name="registry_row")
    current = _as_mapping(values, name="values")
    needed = factor_required_columns(row)
    missing = [column for column in needed if column not in current]
    if missing:
        raise KeyError("missing required factor columns: " + ", ".join(missing))

    def number(column: str, source: Mapping[str, object] = current) -> float:
        value = source.get(column)
        if value is None or isinstance(value, bool):
            raise ValueError(f"factor column is not numeric: {column}")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"factor column is not finite: {column}")
        return result

    transform = str(row.get("panel_transform", "")).strip()
    if transform == "raw_column":
        return number(needed[0])
    if transform == "delta1_raw_column":
        if previous_values is None:
            raise KeyError("delta1 factor requires the immediately preceding observation")
        previous = _as_mapping(previous_values, name="previous_values")
        missing_previous = [column for column in needed if column not in previous]
        if missing_previous:
            raise KeyError(
                "previous observation missing required factor columns: "
                + ", ".join(missing_previous)
            )
        return number(needed[0]) - number(needed[0], previous)
    if transform == "log_ratio":
        numerator, denominator = number(needed[0]), number(needed[1])
        if numerator <= 0.0 or denominator <= 0.0:
            raise ValueError("log_ratio requires positive inputs")
        return math.log(numerator) - math.log(denominator)
    if transform == "log1p_ratio":
        numerator, denominator = number(needed[0]), number(needed[1])
        if numerator < 0.0 or denominator < 0.0:
            raise ValueError("log1p_ratio requires non-negative inputs")
        return math.log1p(numerator) - math.log1p(denominator)
    if transform == "buy_minus_sell":
        return number(needed[0]) - number(needed[1])
    if transform == "buy_sell_imbalance":
        buy, sell = number(needed[0]), number(needed[1])
        denominator = buy + sell
        if denominator <= 0.0:
            raise ValueError("buy_sell_imbalance requires positive total depth")
        return (buy - sell) / denominator
    raise ValueError(f"unsupported factor panel_transform: {transform}")


def rank_standardize_cross_section(values: Mapping[str, float]) -> dict[str, float]:
    """Apply the qlab rank-to-[-1, 1] rule with average ties."""
    ordered = sorted((str(symbol), float(value)) for symbol, value in values.items())
    if not ordered:
        return {}
    result: dict[str, float] = {}
    if len(ordered) == 1:
        return {ordered[0][0]: 0.0}
    for symbol, value in ordered:
        lower = sum(1 for _, other in ordered if other < value)
        upper = sum(1 for _, other in ordered if other <= value)
        rank = (lower + 1 + upper) / 2.0
        result[symbol] = -1.0 + (rank - 1.0) * 2.0 / (len(ordered) - 1.0)
    return result


def build_source_equivalence_identity(
    endpoint: str,
    symbol: str,
    signal_timeframe: str,
    *,
    timestamp_kind: str,
    side: str,
) -> dict[str, object]:
    """Return the versioned semantic scope used by the factor comparison.

    Provider names are diagnostic metadata.  The identity contains the parts
    that determine whether two values represent the same frozen factor input.
    """
    endpoint_name = str(endpoint)
    symbol_name = str(symbol).upper()
    timeframe = str(signal_timeframe)
    if side not in {"realtime", "historical"}:
        raise ValueError("source identity side must be realtime or historical")
    identity: dict[str, object] = {
        "identity_version": "ksv4_source_semantics_v1",
        "endpoint": endpoint_name,
        "symbol": symbol_name,
        "signal_timeframe": timeframe,
        "timestamp_kind": str(timestamp_kind),
    }
    if endpoint_name == "fr":
        identity.update(metric="funding_rate", unit="rate", aggregation="close")
    elif endpoint_name in {"fr_oi_weight", "fr_vol_weight"}:
        identity.update(
            metric=("oi_weighted_funding" if endpoint_name == "fr_oi_weight" else "volume_weighted_funding"),
            unit="rate",
            aggregation="close",
        )
    elif endpoint_name == "futures_net_pos_v2":
        identity.update(
            metric="net_position_change_cum",
            unit="USD",
            native_interval=timeframe,
            observations="current_and_previous_native_interval",
            exchange_scope="Binance",
        )
    elif endpoint_name == "oi":
        identity.update(
            metric="open_interest",
            unit="USD",
            contract_scope="USD-margined perpetuals",
            exchange_scope=(
                "Binance,OKX,Bybit" if side == "realtime" else "historical_unfiltered"
            ),
            native_interval=timeframe,
        )
    elif endpoint_name == "top_pos":
        identity.update(
            metric="top_position_long_short_ratio",
            unit="ratio",
            source_side=side,
            source_native_timestamp_kind=(
                "bar_start" if side == "realtime" else "bar_end"
            ),
            strategy_timestamp_kind="bar_end",
            market_scope="Binance USDT perpetual",
            extra_fields_ignored=("top_pos_long_pct", "top_pos_short_pct"),
        )
    elif endpoint_name in {"ob_pair", "ob_agg"}:
        if side == "realtime":
            if symbol_name in {"BTC", "ETH"}:
                granularity = "1m"
                venue_scope = "KeyStore:Binance"
            else:
                granularity = "snapshot"
                venue_scope = "Binance" if endpoint_name == "ob_pair" else (
                    "Binance" if symbol_name == "FET" else "Binance,OKX,Bybit"
                )
        else:
            granularity = timeframe
            venue_scope = "Binance" if endpoint_name == "ob_pair" else "Binance,OKX,Bybit"
        identity.update(
            metric="orderbook_depth_imbalance",
            raw_input_unit="USD_depth",
            unit="unitless_imbalance",
            venue_scope=venue_scope,
            contract_type="USD-margined perpetual",
            depth_band="+/-1%",
            depth_formula="sum(price*quantity*contract_multiplier), then (bid-ask)/(bid+ask)",
            snapshot_granularity=granularity,
            snapshot_time_semantics="exact_target_label_ts",
            aggregation=("single venue" if endpoint_name == "ob_pair" else "venue sum"),
        )
    else:
        identity.update(metric=endpoint_name)
    return identity


def _identity_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    def comparable(identity: Mapping[str, object]) -> dict[str, object]:
        result = dict(identity)
        if str(result.get("endpoint", "")) == "top_pos":
            # Binance supplies bar_start while the historical KeyStore row is
            # bar_end; both are mapped to the same strategy bar_end.  Keep
            # the side-specific mapping in diagnostics, but compare the
            # canonical strategy identity here.
            result.pop("source_side", None)
            result.pop("source_native_timestamp_kind", None)
        return result

    return _stable_hash(comparable(left)) == _stable_hash(comparable(right))


def _identity_matches_registry(
    identity: Mapping[str, object],
    registry_row: Mapping[str, object],
) -> bool:
    """Check that an observed source identity really belongs to this factor row."""
    endpoint = str(registry_row.get("endpoint", ""))
    timeframe = str(registry_row.get("signal_timeframe", ""))
    timestamp_kind = str(registry_row.get("timestamp_kind", ""))
    if any(
        str(identity.get(key, "")) != expected
        for key, expected in (
            ("endpoint", endpoint),
            ("signal_timeframe", timeframe),
            ("timestamp_kind", timestamp_kind),
        )
    ):
        return False
    if endpoint == "futures_net_pos_v2" and str(
        identity.get("native_interval", "")
    ) != timeframe:
        return False
    if endpoint == "top_pos":
        expected_source_kind = (
            "bar_start"
            if str(identity.get("source_side", "")) == "realtime"
            else "bar_end"
        )
        if str(identity.get("source_native_timestamp_kind", "")) != expected_source_kind:
            return False
        if str(identity.get("strategy_timestamp_kind", "")) != "bar_end":
            return False
    if endpoint in {"ob_pair", "ob_agg"}:
        if str(identity.get("raw_input_unit", "")) != "USD_depth":
            return False
        if str(identity.get("unit", "")) != "unitless_imbalance":
            return False
        if not str(identity.get("depth_formula", "")).startswith(
            "sum(price*quantity"
        ):
            return False
    if endpoint in {"oi", "futures_net_pos_v2"} and str(
        identity.get("unit", "")
    ) != "USD":
        return False
    return True


def _numeric_equal(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _raw_diagnostic(
    realtime: Mapping[str, object],
    historical: Mapping[str, object],
) -> dict[str, object]:
    realtime_fields = sorted(str(key) for key in realtime)
    historical_fields = sorted(str(key) for key in historical)
    return {
        "realtime_fields": realtime_fields,
        "historical_fields": historical_fields,
        "missing_from_realtime": sorted(set(historical_fields) - set(realtime_fields)),
        "missing_from_historical": sorted(set(realtime_fields) - set(historical_fields)),
        "common_fields": sorted(set(realtime_fields).intersection(historical_fields)),
        "raw_structure_equal": realtime_fields == historical_fields,
    }


def _registry_required_diagnostic(
    required: Sequence[str],
    realtime: Mapping[str, object],
    historical: Mapping[str, object],
) -> dict[str, object]:
    required_set = set(required)
    return {
        "required_columns": list(required),
        "missing_from_realtime": sorted(required_set.difference(realtime)),
        "missing_from_historical": sorted(required_set.difference(historical)),
        "realtime_required_present": required_set.issubset(realtime),
        "historical_required_present": required_set.issubset(historical),
    }


def _factor_record_for_pair(
    item: Mapping[str, object],
    *,
    transformed_realtime: Mapping[str, float],
    transformed_historical: Mapping[str, float],
    standardized_realtime: Mapping[str, float],
    standardized_historical: Mapping[str, float],
    raw_diagnostic: Mapping[str, object],
    required_diagnostic: Mapping[str, object],
    source_identity_equal: bool,
    source_identity_contract_valid: bool,
    native_identity_equal: bool,
    error_status: str | None,
) -> dict[str, object]:
    realtime_factor = transformed_realtime.get(str(item["symbol"]).upper())
    historical_factor = transformed_historical.get(str(item["symbol"]).upper())
    realtime_rank = standardized_realtime.get(str(item["symbol"]).upper())
    historical_rank = standardized_historical.get(str(item["symbol"]).upper())
    factor_equal = (
        realtime_factor is not None
        and historical_factor is not None
        and _numeric_equal(realtime_factor, historical_factor)
    )
    rank_equal = (
        realtime_rank is not None
        and historical_rank is not None
        and _numeric_equal(realtime_rank, historical_rank)
    )
    factor_direction_reversed = bool(
        str(item["endpoint"]) in {"ob_pair", "ob_agg"}
        and realtime_factor is not None
        and historical_factor is not None
        and float(realtime_factor) * float(historical_factor) < 0.0
    )
    required_equal = bool(
        required_diagnostic["realtime_required_present"]
        and required_diagnostic["historical_required_present"]
    )
    if error_status:
        status = error_status
    elif not source_identity_contract_valid or not native_identity_equal:
        status = "native_identity_mismatch"
    elif not source_identity_equal:
        status = "scope_not_comparable"
    elif not required_equal:
        status = "required_field_mismatch"
    elif factor_equal and rank_equal:
        status = "exact_match"
    elif rank_equal:
        status = "value_mismatch_decision_equivalent"
    else:
        status = "decision_material_mismatch"
    return {
        "factor_equivalence_contract_version": FACTOR_EQUIVALENCE_CONTRACT_VERSION,
        "collector_id": str(item["collector_id"]),
        "capture_ts": str(item["capture_ts"]),
        "source_scope": str(item["source_scope"]),
        "signal_timeframe": str(item["signal_timeframe"]),
        "endpoint": str(item["endpoint"]),
        "symbol": str(item["symbol"]).upper(),
        "feature_name": str(item["registry_row"].get("feature_name", "")),
        "target_label_ts": _utc_timestamp(item["target_label_ts"]).isoformat(),
        "realtime_receipt_id": str(item["realtime_receipt_id"]),
        "reference_receipt_id": str(item["reference_receipt_id"]),
        "reference_role": str(item["reference_role"]),
        "observed_ts": str(item["observed_ts"]),
        "realtime_native_bar_end_ts": str(item["realtime_native_bar_end_ts"]),
        "reference_native_bar_end_ts": str(item["reference_native_bar_end_ts"]),
        "native_identity_equal": native_identity_equal,
        "source_identity_equal": source_identity_equal,
        "source_identity_contract_valid": source_identity_contract_valid,
        "realtime_source_identity": dict(item["realtime_source_identity"]),
        "historical_source_identity": dict(item["historical_source_identity"]),
        "registry_spec": dict(item["registry_row"]),
        "raw_structure_diagnostic": dict(raw_diagnostic),
        "registry_required_field_diagnostic": dict(required_diagnostic),
        "realtime_values": dict(item["realtime_values"]),
        "historical_values": dict(item["reference_values"]),
        "realtime_previous_values": (
            None if item.get("realtime_previous_values") is None else dict(item["realtime_previous_values"])
        ),
        "historical_previous_values": (
            None if item.get("reference_previous_values") is None else dict(item["reference_previous_values"])
        ),
        "realtime_factor_value": realtime_factor,
        "historical_factor_value": historical_factor,
        "realtime_standardized_value": realtime_rank,
        "historical_standardized_value": historical_rank,
        "factor_direction_reversed": factor_direction_reversed,
        "factor_value_equal": factor_equal,
        "cross_section_equal": rank_equal,
        "final_strategy_input_equal": bool(
            source_identity_equal
            and source_identity_contract_valid
            and native_identity_equal
            and required_equal
            and factor_equal
            and rank_equal
        ),
        "status": status,
        "raw_values_sha256": {
            "realtime": _stable_hash(dict(item["realtime_values"])),
            "historical": _stable_hash(dict(item["reference_values"])),
        },
    }


def build_factor_equivalence_records(
    items: Sequence[Mapping[str, object]],
    *,
    expected_symbols: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Compare a cross-section of registry-bound source pairs.

    Each item must contain one symbol's raw pair and the registry row.  All
    items in one source identity are ranked together, so a research caller
    cannot accidentally compare ranks one symbol at a time.
    """
    if not items:
        return []
    normalized = [dict(item) for item in items]
    expected = None
    if expected_symbols is not None:
        expected = tuple(dict.fromkeys(str(symbol).upper() for symbol in expected_symbols))
        if not expected or any(not symbol for symbol in expected):
            raise ValueError("expected_symbols must contain non-empty identities")
    required_item_keys = {
        "collector_id", "capture_ts", "source_scope", "signal_timeframe", "endpoint",
        "symbol", "target_label_ts", "realtime_receipt_id", "reference_receipt_id",
        "reference_role", "observed_ts", "realtime_native_bar_end_ts",
        "reference_native_bar_end_ts", "realtime_values", "reference_values",
        "registry_row", "realtime_source_identity", "historical_source_identity",
    }
    for item in normalized:
        missing = sorted(required_item_keys.difference(item))
        if missing:
            raise ValueError("factor equivalence input missing: " + ", ".join(missing))

    groups: dict[tuple[str, str, str, str, str, str, str], list[dict[str, object]]] = {}
    for item in normalized:
        key = (
            str(item["collector_id"]),
            str(item["source_scope"]),
            str(item["endpoint"]),
            str(item["signal_timeframe"]),
            str(item["target_label_ts"]),
            str(item["reference_role"]),
            str(_as_mapping(item["registry_row"], name="registry_row").get("feature_name", "")),
        )
        groups.setdefault(key, []).append(item)

    output: list[dict[str, object]] = []
    for group_items in groups.values():
        symbols = [str(item["symbol"]).upper() for item in group_items]
        if len(symbols) != len(set(symbols)):
            raise ValueError("factor equivalence cross-section contains duplicate symbols")
        cross_section_complete = expected is None or set(symbols) == set(expected)
        row = _as_mapping(group_items[0]["registry_row"], name="registry_row")
        invariant = ("required_columns", "panel_transform", "cross_section_standardization", "timestamp_kind", "signal_timeframe")
        if any(_as_mapping(item["registry_row"], name="registry_row").get(key) != row.get(key) for item in group_items for key in invariant):
            raise ValueError("factor equivalence group has inconsistent registry semantics")
        if str(row.get("endpoint")) != str(group_items[0]["endpoint"]):
            raise ValueError("factor equivalence registry endpoint does not match input")
        required = factor_required_columns(row)
        realtime_factors: dict[str, float] = {}
        historical_factors: dict[str, float] = {}
        errors: dict[str, str] = {}
        raw_diagnostics: dict[str, dict[str, object]] = {}
        required_diagnostics: dict[str, dict[str, object]] = {}
        for item, symbol in zip(group_items, symbols):
            realtime = _as_mapping(item["realtime_values"], name="realtime_values")
            historical = _as_mapping(item["reference_values"], name="reference_values")
            raw_diagnostics[symbol] = _raw_diagnostic(realtime, historical)
            required_diagnostics[symbol] = _registry_required_diagnostic(required, realtime, historical)
            if not required_diagnostics[symbol]["realtime_required_present"] or not required_diagnostics[symbol]["historical_required_present"]:
                errors[symbol] = "required_field_mismatch"
                continue
            try:
                realtime_factors[symbol] = apply_factor_registry_transform(
                    row, realtime, previous_values=item.get("realtime_previous_values")
                )
                historical_factors[symbol] = apply_factor_registry_transform(
                    row, historical, previous_values=item.get("reference_previous_values")
                )
            except KeyError:
                errors[symbol] = "missing_prior_observation"
            except (TypeError, ValueError):
                errors[symbol] = "required_field_mismatch"
        policy = str(row.get("cross_section_standardization", "none"))
        if policy == "none":
            realtime_standardized = dict(realtime_factors)
            historical_standardized = dict(historical_factors)
        elif policy == "rank_to_minus1_1":
            realtime_standardized = rank_standardize_cross_section(realtime_factors)
            historical_standardized = rank_standardize_cross_section(historical_factors)
        else:
            raise ValueError("unsupported cross_section_standardization: " + policy)
        if not cross_section_complete:
            realtime_standardized = {}
            historical_standardized = {}
            for symbol in symbols:
                errors.setdefault(symbol, "cross_section_incomplete")
        for item, symbol in zip(group_items, symbols):
            realtime_native = _utc_timestamp(item["realtime_native_bar_end_ts"])
            reference_native = _utc_timestamp(item["reference_native_bar_end_ts"])
            expected_native = _expected_native_bar_end(
                item["target_label_ts"],
                signal_timeframe=str(row["signal_timeframe"]),
                timestamp_kind=str(row["timestamp_kind"]),
            )
            native_equal = (
                realtime_native == reference_native == expected_native
            )
            identity_equal = _identity_equal(
                _as_mapping(item["realtime_source_identity"], name="realtime_source_identity"),
                _as_mapping(item["historical_source_identity"], name="historical_source_identity"),
            )
            realtime_identity_valid = _identity_matches_registry(
                _as_mapping(item["realtime_source_identity"], name="realtime_source_identity"),
                row,
            )
            historical_identity_valid = _identity_matches_registry(
                _as_mapping(item["historical_source_identity"], name="historical_source_identity"),
                row,
            )
            identity_contract_valid = realtime_identity_valid and historical_identity_valid
            record = _factor_record_for_pair(
                item,
                transformed_realtime=realtime_factors,
                transformed_historical=historical_factors,
                standardized_realtime=realtime_standardized,
                standardized_historical=historical_standardized,
                raw_diagnostic=raw_diagnostics[symbol],
                required_diagnostic=required_diagnostics[symbol],
                source_identity_equal=identity_equal,
                source_identity_contract_valid=identity_contract_valid,
                native_identity_equal=native_equal,
                error_status=errors.get(symbol),
            )
            record["panel_transform"] = str(row["panel_transform"])
            record["cross_section_standardization"] = policy
            record["cross_section_complete"] = cross_section_complete
            record["expected_symbols"] = None if expected is None else list(expected)
            record["observed_symbols"] = sorted(symbols)
            record["expected_native_bar_end_ts"] = expected_native.isoformat()
            record["factor_value_difference"] = (
                None
                if record["realtime_factor_value"] is None
                or record["historical_factor_value"] is None
                else float(record["historical_factor_value"])
                - float(record["realtime_factor_value"])
            )
            record["cross_section_difference"] = (
                None
                if record["realtime_standardized_value"] is None
                or record["historical_standardized_value"] is None
                else float(record["historical_standardized_value"])
                - float(record["realtime_standardized_value"])
            )
            record["record_sha256"] = _stable_hash(record)
            output.append(record)
    return sorted(
        output,
        key=lambda record: (
            str(record["source_scope"]), str(record["endpoint"]),
            str(record["signal_timeframe"]), str(record["target_label_ts"]),
            str(record["symbol"]),
        ),
    )


def build_factor_equivalence_record(**kwargs: object) -> dict[str, object]:
    """Convenience wrapper for a one-symbol comparison."""
    records = build_factor_equivalence_records([kwargs])
    if len(records) != 1:
        raise RuntimeError("one-symbol factor equivalence did not return one record")
    return records[0]


__all__ = [
    "FACTOR_EQUIVALENCE_CONTRACT_VERSION",
    "FACTOR_EQUIVALENCE_STATUSES",
    "apply_factor_registry_transform",
    "build_factor_equivalence_record",
    "build_factor_equivalence_records",
    "build_source_equivalence_identity",
    "factor_required_columns",
    "previous_native_label",
    "rank_standardize_cross_section",
]
