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

from .keystore_coinglass_panel import (
    FactorTransformError,
    InvalidNumericValueError,
    RequiredColumnMissingError,
    extract_feature_series,
)
from .panel_statistics import rank_grouped_series, rank_standardize_grouped_series


FACTOR_EQUIVALENCE_CONTRACT_VERSION = "ksv4_factor_equivalence_v1"
SOURCE_IDENTITY_CONTRACT_VERSION = "ksv4_source_semantics_v1"
FACTOR_EQUIVALENCE_STATUSES = frozenset(
    {
        "exact_match",
        "value_mismatch_decision_equivalent",
        "decision_material_mismatch",
        "cross_section_incomplete",
        "scope_not_comparable",
        "native_identity_mismatch",
        "required_field_missing",
        "invalid_numeric_value",
        "transform_failed",
        "required_field_mismatch",
        "missing_prior_observation",
    }
)

# These endpoints expose a native interval in their frozen request.  The
# interval is part of the event identity: a response requested at 1h cannot
# be relabelled as the 1d observation merely because the registry row says
# ``signal_timeframe=1d``.  Orderbook inputs are excluded because their
# contract explicitly uses snapshot/1m source observations before projection.
_NATIVE_INTERVAL_BOUND_ENDPOINTS = frozenset(
    {
        "fr",
        "fr_oi_weight",
        "fr_vol_weight",
        "oi",
        "futures_net_pos_v2",
        "top_pos",
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


def _registry_identity_contract(
    registry_row: Mapping[str, object], *, side: str
) -> dict[str, object]:
    """Read optional versioned source-identity rules from a registry row."""
    raw = registry_row.get("source_identity_contract")
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("source_identity_contract is not valid JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("source_identity_contract must be a mapping")
    selected = raw.get(side, raw)
    if not isinstance(selected, Mapping):
        raise ValueError(f"source_identity_contract has no {side} mapping")
    return {str(key): value for key, value in selected.items()}


def _identity_value_matches(actual: object, expected: object) -> bool:
    if isinstance(expected, (list, tuple, set, frozenset)):
        return any(_identity_value_matches(actual, item) for item in expected)
    return str(actual) == str(expected)


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
    """Apply the same transform used by the frozen panel builder.

    This is a compatibility wrapper around ``extract_feature_series``.  The
    equivalence path deliberately has no second copy of the registry formulas.
    """
    row = _as_mapping(registry_row, name="registry_row")
    current = _as_mapping(values, name="values")
    transform = str(row.get("panel_transform", "")).strip()
    if transform == "delta1_raw_column" and previous_values is None:
        raise KeyError("delta1 factor requires the immediately preceding observation")
    rows = [current] if previous_values is None else [
        _as_mapping(previous_values, name="previous_values"), current
    ]
    index = pd.date_range(
        "2000-01-01T00:00:00Z", periods=len(rows), freq="1s", name="ts"
    )
    frame = pd.DataFrame(rows, index=index)
    series = extract_feature_series(pd.Series(row), frame)
    if series.empty:
        raise FactorTransformError("factor transform returned no value")
    return float(series.iloc[-1])


def rank_standardize_cross_section(values: Mapping[str, float]) -> dict[str, float]:
    """Compatibility adapter to the shared qlab rank standardization primitive."""
    if not values:
        return {}
    index = pd.MultiIndex.from_tuples(
        [("event", str(symbol)) for symbol in values],
        names=["decision_ts", "symbol"],
    )
    series = pd.Series([float(value) for value in values.values()], index=index)
    standardized = rank_standardize_grouped_series(series)
    return {
        str(symbol): float(value)
        for (_, symbol), value in standardized.items()
    }


def _request_scope_from_contract(
    request_contract: Mapping[str, object],
    *,
    symbol: str,
    signal_timeframe: str,
) -> dict[str, object]:
    """Extract identity-bearing scope from the actual frozen request params."""
    params = _as_mapping(request_contract.get("request_params"), name="request_params")

    def normalized_scope(value: object) -> str:
        if isinstance(value, (list, tuple, set, frozenset)):
            return ",".join(sorted(str(item) for item in value))
        return ",".join(sorted(item.strip() for item in str(value).split(",") if item.strip()))

    exchange_value = next(
        (
            params[key]
            for key in ("exchange_list", "exchanges", "exchange")
            if key in params and str(params[key]).strip()
        ),
        None,
    )
    if exchange_value is None:
        observed_scope = request_contract.get("response_scope")
        if isinstance(observed_scope, Mapping):
            exchange_value = observed_scope.get("exchange_scope")
    if exchange_value is None:
        source_name = str(request_contract.get("source", ""))
        exchange_value = {
            "binance_public": "Binance",
            "okx_public": "OKX",
            "bybit_public": "Bybit",
        }.get(source_name)
    exchange_scope = (
        normalized_scope(exchange_value) if exchange_value is not None else "unbounded"
    )
    interval = request_contract.get(
        "native_interval",
        params.get("interval", params.get("period", signal_timeframe)),
    )
    request_symbol = params.get("symbol", params.get("base", symbol))
    return {
        "exchange_scope": exchange_scope,
        "native_interval": str(interval),
        "request_symbol": str(request_symbol).upper(),
        "request_unit": str(params.get("unit", "")),
        "request_range": str(params.get("range", "")),
        "scope_status": "declared" if exchange_scope != "unbounded" else "unverified",
    }


def source_semantic_contract_from_request(
    request_contract: Mapping[str, object],
    registry_row: Mapping[str, object],
    *,
    symbol: str,
    signal_timeframe: str,
    side: str,
) -> dict[str, object]:
    """Build semantic identity from actual request params plus registry rules.

    The registry supplies the factor's meaning.  Scope and native interval are
    read from the concrete request, so an endpoint name cannot claim an identity
    that its request did not establish.
    """
    if side not in {"realtime", "historical"}:
        raise ValueError("source identity side must be realtime or historical")
    actual_scope = _request_scope_from_contract(
        request_contract, symbol=symbol, signal_timeframe=signal_timeframe
    )
    expected = _registry_identity_contract(registry_row, side=side)
    semantic = dict(expected)
    semantic.update(actual_scope)
    semantic["request_contract_version"] = str(
        request_contract.get("source_contract_version", "")
    )
    semantic["scope_key"] = _stable_hash(
        {
            key: actual_scope[key]
            for key in (
                "exchange_scope", "native_interval", "request_symbol",
                "request_unit", "request_range",
            )
        }
    )
    return semantic


def _request_symbol_matches_identity(
    request_symbol: object,
    identity_symbol: object,
) -> bool:
    """Accept the base symbol and the frozen USD-margined symbol forms only."""
    requested = str(request_symbol).strip().upper().replace("-", "").replace("_", "")
    identity = str(identity_symbol).strip().upper()
    if not requested or not identity:
        return False
    if requested == identity:
        return True
    if requested.endswith("SWAP"):
        requested = requested[:-4]
    return any(requested == identity + quote for quote in ("USDT", "USDC", "BUSD", "USD"))


def build_source_equivalence_identity(
    endpoint: str,
    symbol: str,
    signal_timeframe: str,
    *,
    timestamp_kind: str,
    side: str,
    request_contract: Mapping[str, object],
    receipt_lineage: Mapping[str, object],
    semantic_contract: Mapping[str, object],
) -> dict[str, object]:
    """Return an identity bound to a concrete request and immutable receipt."""
    if side not in {"realtime", "historical"}:
        raise ValueError("source identity side must be realtime or historical")
    request = _as_mapping(request_contract, name="request_contract")
    required_request = {"source", "route", "request_path", "request_params"}
    missing_request = sorted(required_request.difference(request))
    if missing_request:
        raise ValueError(
            "request contract missing identity fields: " + ", ".join(missing_request)
        )
    params = request.get("request_params")
    if not isinstance(params, Mapping):
        raise ValueError("request contract request_params must be a mapping")
    lineage = _as_mapping(receipt_lineage, name="receipt_lineage")
    required_lineage = {"receipt_id", "payload_sha256"}
    missing_lineage = sorted(required_lineage.difference(lineage))
    if missing_lineage:
        raise ValueError(
            "receipt lineage missing identity fields: " + ", ".join(missing_lineage)
        )
    semantic = _as_mapping(semantic_contract, name="semantic_contract")
    return {
        "identity_version": SOURCE_IDENTITY_CONTRACT_VERSION,
        "endpoint": str(endpoint),
        "symbol": str(symbol).upper(),
        "signal_timeframe": str(signal_timeframe),
        "timestamp_kind": str(timestamp_kind),
        "source_side": side,
        "request_contract": request,
        "receipt_lineage": lineage,
        "semantic_contract": semantic,
        **semantic,
    }


def _identity_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    """Compare only canonical event semantics, not side-specific receipt IDs."""
    left_semantic = left.get("semantic_contract")
    right_semantic = right.get("semantic_contract")
    # An endpoint name is not evidence that the two sides cover the same
    # market.  An unverified/unbounded scope must fail closed until the
    # request contract or immutable receipt metadata proves the scope.
    if not isinstance(left_semantic, Mapping) or not isinstance(right_semantic, Mapping):
        return False
    if any(
        str(semantic.get("scope_status", "")).strip().lower() != "declared"
        for semantic in (left_semantic, right_semantic)
    ):
        return False
    comparable_keys = (
        "endpoint", "symbol", "signal_timeframe", "timestamp_kind",
        "semantic_contract",
    )
    return _stable_hash(
        {key: left.get(key) for key in comparable_keys}
    ) == _stable_hash(
        {key: right.get(key) for key in comparable_keys}
    )


def _identity_matches_registry(
    identity: Mapping[str, object],
    registry_row: Mapping[str, object],
    *,
    expected_symbol: str | None = None,
) -> bool:
    """Check a concrete request/receipt identity against one registry row."""
    if str(identity.get("identity_version", "")).strip() != SOURCE_IDENTITY_CONTRACT_VERSION:
        return False
    if not isinstance(identity.get("request_contract"), Mapping):
        return False
    if not isinstance(identity.get("receipt_lineage"), Mapping):
        return False
    if not isinstance(identity.get("semantic_contract"), Mapping):
        return False
    semantic = identity["semantic_contract"]
    request = identity["request_contract"]
    if not {"source", "route", "request_path", "request_params"}.issubset(request):
        return False
    if not isinstance(request["request_params"], Mapping):
        return False
    lineage = identity["receipt_lineage"]
    if not {"receipt_id", "payload_sha256"}.issubset(lineage):
        return False
    if expected_symbol is not None and str(identity.get("symbol", "")).upper() != str(expected_symbol).upper():
        return False
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
    if endpoint in _NATIVE_INTERVAL_BOUND_ENDPOINTS:
        # This check is deliberately made against the recomputed request
        # semantic, not the endpoint name or registry label alone.
        if str(semantic.get("native_interval", "")) != timeframe:
            return False
    if "source_scope" in registry_row:
        if (
            "source_scope" not in identity
            or str(identity.get("source_scope", ""))
            != str(registry_row["source_scope"])
        ):
            return False

    registry_identity_version = str(
        registry_row.get("source_identity_contract_version", "")
    ).strip()
    if registry_identity_version and registry_identity_version != SOURCE_IDENTITY_CONTRACT_VERSION:
        return False

    try:
        contract = _registry_identity_contract(
            registry_row,
            side=str(identity.get("source_side", "")),
        )
    except ValueError:
        return False
    source_side = str(identity.get("source_side", ""))
    try:
        recomputed_semantic = source_semantic_contract_from_request(
            request,
            registry_row,
            symbol=str(identity.get("symbol", "")),
            signal_timeframe=timeframe,
            side=source_side,
        )
    except (TypeError, ValueError):
        return False
    if _stable_hash(dict(semantic)) != _stable_hash(recomputed_semantic):
        return False
    if any(
        _stable_hash(identity.get(key)) != _stable_hash(value)
        for key, value in semantic.items()
    ):
        return False
    if not _request_symbol_matches_identity(
        semantic.get("request_symbol"), identity.get("symbol")
    ):
        return False
    for key, expected in contract.items():
        if key == "source_side":
            continue
        actual = identity.get(key, identity["semantic_contract"].get(key))
        if actual is None or not _identity_value_matches(actual, expected):
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
    raw_rank_realtime: Mapping[str, float],
    raw_rank_historical: Mapping[str, float],
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
    realtime_raw_rank = raw_rank_realtime.get(str(item["symbol"]).upper())
    historical_raw_rank = raw_rank_historical.get(str(item["symbol"]).upper())
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
    direction_semantics = str(
        item["registry_row"].get("direction_semantics", "signed_factor")
    ).strip() or "signed_factor"
    rank_difference_semantics = str(
        item["registry_row"].get(
            "rank_difference_semantics",
            "historical_minus_realtime_raw_rank_and_standardized_value",
        )
    ).strip()
    direction_same = None
    direction_reversed = None
    if direction_semantics not in {"none", "not_applicable"}:
        if realtime_factor is not None and historical_factor is not None:
            realtime_sign = (float(realtime_factor) > 0.0) - (
                float(realtime_factor) < 0.0
            )
            historical_sign = (float(historical_factor) > 0.0) - (
                float(historical_factor) < 0.0
            )
            direction_same = realtime_sign == historical_sign
            direction_reversed = bool(
                realtime_sign != 0
                and historical_sign != 0
                and realtime_sign == -historical_sign
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
        "direction_semantics": direction_semantics,
        "rank_difference_semantics": rank_difference_semantics,
        "direction_same": direction_same,
        "direction_reversed": direction_reversed,
        "factor_direction_reversed": bool(direction_reversed),
        "realtime_raw_rank": realtime_raw_rank,
        "historical_raw_rank": historical_raw_rank,
        "raw_rank_difference": (
            None
            if realtime_raw_rank is None or historical_raw_rank is None
            else float(historical_raw_rank) - float(realtime_raw_rank)
        ),
        "factor_value_equal": factor_equal,
        "cross_section_equal": rank_equal,
        "final_strategy_input_equal": bool(
            source_identity_equal
            and source_identity_contract_valid
            and native_identity_equal
            and required_equal
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
                errors[symbol] = "required_field_missing"
        if cross_section_complete:
            for item, symbol in zip(group_items, symbols):
                if symbol in errors:
                    continue
                realtime = _as_mapping(item["realtime_values"], name="realtime_values")
                historical = _as_mapping(item["reference_values"], name="reference_values")
                try:
                    realtime_factors[symbol] = apply_factor_registry_transform(
                        row, realtime, previous_values=item.get("realtime_previous_values")
                    )
                    historical_factors[symbol] = apply_factor_registry_transform(
                        row, historical, previous_values=item.get("reference_previous_values")
                    )
                except RequiredColumnMissingError:
                    errors[symbol] = "required_field_missing"
                except InvalidNumericValueError:
                    errors[symbol] = "invalid_numeric_value"
                except FactorTransformError:
                    errors[symbol] = "transform_failed"
                except KeyError:
                    errors[symbol] = "missing_prior_observation"
                except (TypeError, ValueError):
                    errors[symbol] = "transform_failed"
        if not cross_section_complete:
            for symbol in symbols:
                errors.setdefault(symbol, "cross_section_incomplete")
        policy = str(row.get("cross_section_standardization", "none"))
        if policy == "none":
            realtime_standardized = dict(realtime_factors) if not errors else {}
            historical_standardized = dict(historical_factors) if not errors else {}
        elif policy == "rank_to_minus1_1":
            realtime_standardized = (
                rank_standardize_cross_section(realtime_factors) if not errors else {}
            )
            historical_standardized = (
                rank_standardize_cross_section(historical_factors) if not errors else {}
            )
        else:
            raise ValueError("unsupported cross_section_standardization: " + policy)
        if not cross_section_complete:
            realtime_standardized = {}
            historical_standardized = {}
        raw_rank_realtime: dict[str, float] = {}
        raw_rank_historical: dict[str, float] = {}
        if realtime_factors:
            realtime_raw_rank_series = rank_grouped_series(
                pd.Series(
                    list(realtime_factors.values()),
                    index=pd.MultiIndex.from_tuples(
                        [("event", symbol) for symbol in realtime_factors],
                        names=["decision_ts", "symbol"],
                    ),
                )
            )
            raw_rank_realtime = {
                str(symbol): float(value)
                for (_, symbol), value in realtime_raw_rank_series.items()
            }
        if historical_factors:
            historical_raw_rank_series = rank_grouped_series(
                pd.Series(
                    list(historical_factors.values()),
                    index=pd.MultiIndex.from_tuples(
                        [("event", symbol) for symbol in historical_factors],
                        names=["decision_ts", "symbol"],
                    ),
                )
            )
            raw_rank_historical = {
                str(symbol): float(value)
                for (_, symbol), value in historical_raw_rank_series.items()
            }
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
                expected_symbol=symbol,
            )
            historical_identity_valid = _identity_matches_registry(
                _as_mapping(item["historical_source_identity"], name="historical_source_identity"),
                row,
                expected_symbol=symbol,
            )
            identity_contract_valid = realtime_identity_valid and historical_identity_valid
            record = _factor_record_for_pair(
                item,
                transformed_realtime=realtime_factors,
                transformed_historical=historical_factors,
                standardized_realtime=realtime_standardized,
                standardized_historical=historical_standardized,
                raw_rank_realtime=raw_rank_realtime,
                raw_rank_historical=raw_rank_historical,
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
            record["standardized_value_difference"] = record["cross_section_difference"]
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


def aggregate_factor_equivalence_records(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Aggregate feature rows into one source-event comparison.

    This event-level reduction is part of the qlab contract.  A research
    caller may persist its result, but may not independently decide the
    aggregate status or final strategy-input equality.
    """
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in records:
        item = dict(record)
        grouped.setdefault(
            (str(item["realtime_receipt_id"]), str(item["reference_role"])),
            [],
        ).append(item)

    status_priority = {
        "scope_not_comparable": 0,
        "native_identity_mismatch": 1,
        "required_field_missing": 2,
        "invalid_numeric_value": 3,
        "transform_failed": 4,
        "required_field_mismatch": 5,
        "missing_prior_observation": 6,
        "cross_section_incomplete": 7,
        "decision_material_mismatch": 8,
        "value_mismatch_decision_equivalent": 9,
        "exact_match": 10,
    }
    output: list[dict[str, object]] = []
    for factor_records in grouped.values():
        ordered = sorted(
            factor_records,
            key=lambda record: str(record.get("feature_name", "")),
        )
        statuses = [str(record["status"]) for record in ordered]
        aggregate = dict(ordered[0])
        aggregate.pop("registry_spec", None)
        aggregate.pop("raw_values_sha256", None)
        aggregate["factor_equivalence_records"] = ordered
        aggregate["factor_statuses"] = {
            str(record.get("feature_name", "")): str(record["status"])
            for record in ordered
        }
        aggregate["status"] = min(
            statuses,
            key=lambda status: status_priority.get(status, -1),
        )
        aggregate["final_strategy_input_equal"] = all(
            bool(record["final_strategy_input_equal"]) for record in ordered
        )
        aggregate["raw_structure_diagnostics"] = {
            str(record.get("feature_name", "")): record["raw_structure_diagnostic"]
            for record in ordered
        }
        aggregate["registry_required_field_diagnostics"] = {
            str(record.get("feature_name", "")): record[
                "registry_required_field_diagnostic"
            ]
            for record in ordered
        }
        aggregate["factor_equivalence_count"] = len(ordered)
        aggregate["record_sha256"] = _stable_hash(
            {key: value for key, value in aggregate.items() if key != "record_sha256"}
        )
        output.append(aggregate)
    return sorted(
        output,
        key=lambda record: (
            str(record.get("source_scope", "")),
            str(record.get("endpoint", "")),
            str(record.get("signal_timeframe", "")),
            str(record.get("target_label_ts", "")),
            str(record.get("realtime_receipt_id", "")),
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
    "SOURCE_IDENTITY_CONTRACT_VERSION",
    "apply_factor_registry_transform",
    "aggregate_factor_equivalence_records",
    "build_factor_equivalence_record",
    "build_factor_equivalence_records",
    "build_source_equivalence_identity",
    "source_semantic_contract_from_request",
    "factor_required_columns",
    "previous_native_label",
    "rank_standardize_cross_section",
]
