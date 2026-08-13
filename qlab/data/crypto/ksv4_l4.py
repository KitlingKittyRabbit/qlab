"""Formal realtime-source and delayed execution contracts for KSV4 L4."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import pandas as pd

from qlab.data.crypto.panel import executable_returns_for_symbol
from qlab.data.crypto.strategy_time_contract import ContinuousHoldingTimeContract


FROZEN_READINESS_TIMEOUT_SECONDS = 210
FROZEN_UNAVAILABLE_ORDERBOOK_ROUTES = frozenset(
    {("okx_public", "FET"), ("bybit_public", "FET")}
)
FROZEN_KEYSTORE_ORDERBOOK_SYMBOLS = frozenset({"BTC", "ETH"})
SUPPORTED_REALTIME_ENDPOINTS = frozenset(
    {
        "fr", "fr_oi_weight", "fr_vol_weight", "futures_net_pos_v2",
        "ob_agg", "ob_pair", "oi", "top_pos",
    }
)
REALTIME_EQUIVALENCE_CONTRACT = {
    "fr": ("funding-rate/exchange-list", "funding_close", "raw", "rate"),
    "fr_oi_weight": ("coins-markets", "funding_oi_weight_close", "raw", "rate"),
    "fr_vol_weight": ("coins-markets", "funding_vol_weight_close", "raw", "rate"),
    "oi": ("coins-markets", "oi_close", "delta1", "usd"),
    "futures_net_pos_v2": ("futures/v2/net-position/history", "net_pos_delta1", "delta1", "usd"),
    "top_pos": ("binance_top_position_ratio", "top_pos_ls_ratio", "raw_or_delta1", "ratio"),
    "ob_pair": ("frozen_mixed_orderbook", "ob_pair_imbalance", "raw", "unitless_imbalance"),
    "ob_agg": ("frozen_mixed_orderbook", "ob_agg_imbalance", "raw", "unitless_imbalance"),
}
REALTIME_SOURCE_DEFINITIONS = {
    "fr": {
        "mapping_id": "fr:binance_funding_rate:raw_rate",
        "direction": "unchanged", "unit": "rate",
        "native_window": "current Binance funding observation",
    },
    "fr_oi_weight": {
        "mapping_id": "fr_oi_weight:coins_markets_oi_weighted:raw_rate",
        "direction": "unchanged", "unit": "rate",
        "native_window": "current CoinGlass OI-weighted funding observation",
    },
    "fr_vol_weight": {
        "mapping_id": "fr_vol_weight:coins_markets_volume_weighted:raw_rate",
        "direction": "unchanged", "unit": "rate",
        "native_window": "current CoinGlass volume-weighted funding observation",
    },
    "oi": {
        "mapping_id": "oi:coins_markets_open_interest_usd:delta1",
        "direction": "current minus previous native observation", "unit": "USD",
        "native_window": "current OI plus persisted prior boundary",
    },
    "futures_net_pos_v2": {
        "mapping_id": "futures_net_pos_v2:net_position_change_cum:delta1",
        "direction": "current minus previous native observation", "unit": "USD",
        "native_window": "requested native interval plus prior row",
    },
    "top_pos": {
        "mapping_id": "top_pos:binance_top_position_long_short_ratio:raw_or_delta1",
        "direction": "long divided by short; delta is current minus previous", "unit": "ratio",
        "native_window": "Binance period 1h or 12h",
    },
    "ob_pair": {
        "mapping_id": "ob_pair:frozen_source_depth_pm1pct:imbalance",
        "direction": "(bid_usd-ask_usd)/(bid_usd+ask_usd)",
        "unit": "unitless imbalance from USD depth",
        "native_window": (
            "KeyStore fixed +/-1% history for BTC/ETH; as-received Binance "
            "USD-M snapshot spanning +/-1% for the other symbols"
        ),
    },
    "ob_agg": {
        "mapping_id": "ob_agg:frozen_source_depth_pm1pct:imbalance",
        "direction": "sum venue bid/ask USD depth, then (bid-ask)/(bid+ask)",
        "unit": "unitless imbalance from USD depth",
        "native_window": (
            "KeyStore fixed +/-1% aggregate history for BTC/ETH; as-received snapshots "
            "spanning +/-1% from all frozen available venues for the other symbols; FET "
            "uses Binance only because OKX/FET is unavailable and Bybit/FET is closed"
        ),
    },
}


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: " + ", ".join(missing))


def expected_orderbook_venues(endpoint: str, symbol: str) -> tuple[str, ...]:
    """Return the exact frozen source set for one orderbook identity."""
    if str(symbol).upper() in FROZEN_KEYSTORE_ORDERBOOK_SYMBOLS:
        return ("keystore",)
    if endpoint == "ob_pair":
        return ("binance_public",)
    if endpoint != "ob_agg":
        raise ValueError(f"unsupported orderbook endpoint={endpoint}")
    if str(symbol).upper() == "FET":
        return ("binance_public",)
    return ("binance_public", "bybit_public", "okx_public")


def frozen_execution_delay_minutes(
    readiness_timeout_seconds: int = FROZEN_READINESS_TIMEOUT_SECONDS,
) -> int:
    """Map a readiness upper bound to the first usable whole-minute open."""
    if not isinstance(readiness_timeout_seconds, int) or readiness_timeout_seconds <= 0:
        raise ValueError("readiness_timeout_seconds must be a positive integer")
    return math.ceil(readiness_timeout_seconds / 60)


def build_realtime_source_plan(
    dependencies: pd.DataFrame,
    *,
    symbols: Sequence[str],
) -> pd.DataFrame:
    """Build the deduplicated shared request plan for the repaired L3 set."""
    _require_columns(dependencies, {"endpoint", "signal_timeframe"}, "dependencies")
    endpoints = frozenset(dependencies["endpoint"].astype(str).unique())
    unsupported = sorted(endpoints.difference(SUPPORTED_REALTIME_ENDPOINTS))
    if unsupported:
        raise ValueError("unsupported realtime endpoints: " + ", ".join(unsupported))
    clean_symbols = tuple(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols))
    if not clean_symbols or any(not symbol for symbol in clean_symbols):
        raise ValueError("symbols must contain non-empty unique identities")
    rows: list[dict[str, object]] = []

    def add(source: str, route: str, symbol: str = "ALL", timeframe: str = "snapshot", serialized: bool = False) -> None:
        rows.append(
            {
                "request_id": f"{source}|{route}|{symbol}|{timeframe}",
                "source": source,
                "route": route,
                "symbol": symbol,
                "signal_timeframe": timeframe,
                "serialized": serialized,
            }
        )

    if endpoints.intersection({"oi", "fr_oi_weight", "fr_vol_weight"}):
        add("keystore", "coins-markets", serialized=True)
    if "fr" in endpoints:
        add("keystore", "funding-rate/exchange-list", serialized=True)
    if "futures_net_pos_v2" in endpoints:
        for symbol in clean_symbols:
            add("keystore", "futures/v2/net-position/history", symbol, serialized=True)
    if "top_pos" in endpoints:
        periods = sorted(
            dependencies.loc[dependencies["endpoint"].eq("top_pos"), "signal_timeframe"].astype(str).unique()
        )
        for timeframe in periods:
            for symbol in clean_symbols:
                add("binance_public", "top-position-ratio", symbol, timeframe)
    if endpoints.intersection({"ob_pair", "ob_agg"}):
        for symbol in clean_symbols:
            if symbol in FROZEN_KEYSTORE_ORDERBOOK_SYMBOLS:
                if "ob_pair" in endpoints:
                    add(
                        "keystore", "orderbook/ask-bids-history", symbol,
                        "1m_pm1pct", serialized=True,
                    )
                if "ob_agg" in endpoints:
                    add(
                        "keystore", "orderbook/aggregated-ask-bids-history", symbol,
                        "1m_pm1pct", serialized=True,
                    )
                continue
            add("binance_public", "orderbook", symbol)
            if "ob_agg" in endpoints:
                add("okx_public", "orderbook", symbol)
                add("bybit_public", "orderbook", symbol)
    result = pd.DataFrame(rows).drop_duplicates("request_id").sort_values(
        ["serialized", "source", "route", "signal_timeframe", "symbol"],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    result["request_order"] = range(1, len(result) + 1)
    return result


def apply_orderbook_venue_availability(
    source_plan: pd.DataFrame,
    availability: pd.DataFrame,
) -> pd.DataFrame:
    """Remove only orderbook routes proven unavailable by exchange evidence."""
    _require_columns(source_plan, {"source", "route", "symbol"}, "source_plan")
    _require_columns(
        availability, {"source", "symbol", "available", "evidence_code"}, "availability"
    )
    if availability.duplicated(["source", "symbol"]).any():
        raise ValueError("orderbook availability contains duplicate venue/symbol rows")
    orderbook = source_plan.loc[source_plan["route"].eq("orderbook")]
    audited = orderbook.merge(
        availability, on=["source", "symbol"], how="left", validate="one_to_one"
    )
    if audited["available"].isna().any():
        raise ValueError("orderbook availability does not cover every planned route")
    unavailable = audited["available"].map(
        lambda value: not (value is True or str(value).lower() == "true")
    )
    if audited.loc[unavailable, "evidence_code"].astype(str).str.strip().eq("").any():
        raise ValueError("unavailable orderbook route lacks exchange evidence code")
    unavailable_routes = frozenset(
        zip(audited.loc[unavailable, "source"], audited.loc[unavailable, "symbol"])
    )
    if unavailable_routes != FROZEN_UNAVAILABLE_ORDERBOOK_ROUTES:
        raise ValueError("orderbook unavailable routes differ from the frozen contract")
    keep = set(
        zip(
            audited.loc[~unavailable, "source"],
            audited.loc[~unavailable, "symbol"],
        )
    )
    result = source_plan.loc[
        ~source_plan["route"].eq("orderbook")
        | source_plan.apply(lambda row: (row["source"], row["symbol"]) in keep, axis=1)
    ].copy()
    result = result.sort_values("request_order").reset_index(drop=True)
    result["request_order"] = range(1, len(result) + 1)
    return result


def validate_realtime_equivalence_contract(
    dependencies: pd.DataFrame,
    registry: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    expected_symbols: Sequence[str],
) -> pd.DataFrame:
    """Fail closed unless every realtime endpoint satisfies its frozen contract.

    ``coverage`` is produced by the as-received source preflight. Each endpoint
    row must report schema, direction, unit, native-window, and symbol coverage.
    """
    _require_columns(dependencies, {"endpoint", "signal_timeframe"}, "dependencies")
    _require_columns(
        registry,
        {"endpoint", "signal_timeframe", "required_columns", "panel_transform", "timestamp_kind"},
        "registry",
    )
    _require_columns(
        coverage,
        {
            "endpoint", "schema_ok", "direction_ok", "unit_ok", "native_window_ok",
            "formula_test_ok", "evidence_status", "mapping_id", "raw_evidence_sha256",
            "comparison_row_count", "covered_symbols", "source_observed_ts",
        },
        "coverage",
    )
    endpoints = sorted(dependencies["endpoint"].astype(str).unique())
    if set(endpoints) != set(REALTIME_EQUIVALENCE_CONTRACT):
        raise ValueError("dependency endpoints do not match frozen realtime equivalence contract")
    if coverage["endpoint"].duplicated().any() or set(coverage["endpoint"]) != set(endpoints):
        raise ValueError("coverage must contain exactly one row per realtime endpoint")
    expected = sorted(dict.fromkeys(str(value).upper() for value in expected_symbols))
    audited = coverage.copy()
    audited["covered_symbols"] = audited["covered_symbols"].map(
        lambda value: ",".join(sorted(item.strip().upper() for item in str(value).split(",") if item.strip()))
    )
    expected_label = ",".join(expected)
    booleans = ["schema_ok", "direction_ok", "unit_ok", "native_window_ok", "formula_test_ok"]
    if not audited[booleans].apply(lambda column: column.map(lambda value: value is True or str(value).lower() == "true")).all().all():
        raise ValueError("realtime source equivalence preflight failed")
    if not audited["evidence_status"].eq("verified").all():
        raise ValueError("realtime source equivalence evidence is not verified")
    if audited["mapping_id"].astype(str).str.strip().eq("").any():
        raise ValueError("realtime source equivalence mapping id is missing")
    if not audited["raw_evidence_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("realtime source raw evidence hash is invalid")
    if pd.to_numeric(audited["comparison_row_count"], errors="coerce").fillna(0).le(0).any():
        raise ValueError("realtime source comparison evidence is empty")
    if not audited["covered_symbols"].eq(expected_label).all():
        raise ValueError("realtime source preflight does not cover the frozen symbol set")
    if pd.to_datetime(audited["source_observed_ts"], utc=True, errors="coerce").isna().any():
        raise ValueError("realtime source preflight lacks observed timestamps")
    registry_identities = registry[["endpoint", "signal_timeframe"]].drop_duplicates()
    missing = dependencies[["endpoint", "signal_timeframe"]].drop_duplicates().merge(
        registry_identities,
        on=["endpoint", "signal_timeframe"],
        how="left",
        indicator=True,
    )
    if missing["_merge"].ne("both").any():
        raise ValueError("registry does not cover every realtime dependency identity")
    return audited.sort_values("endpoint").reset_index(drop=True)


def evaluate_serialized_route_runtime(
    receipts: pd.DataFrame,
    *,
    expected_names: Sequence[str],
    timeout_seconds: int = FROZEN_READINESS_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Validate an exact serialized request route and its end-to-end runtime."""
    _require_columns(receipts, {"name", "request_ts", "response_ts"}, "receipts")
    expected = tuple(str(value) for value in expected_names)
    if tuple(receipts["name"].astype(str)) != expected:
        raise ValueError("serialized route receipts do not match the frozen request order")
    if not expected:
        raise ValueError("serialized route must contain at least one request")
    request_ts = pd.to_datetime(receipts["request_ts"], utc=True, errors="coerce")
    response_ts = pd.to_datetime(receipts["response_ts"], utc=True, errors="coerce")
    if request_ts.isna().any() or response_ts.isna().any():
        raise ValueError("serialized route receipts contain invalid timestamps")
    if (response_ts < request_ts).any():
        raise ValueError("serialized route response precedes request")
    if not request_ts.is_monotonic_increasing:
        raise ValueError("serialized route requests are not ordered")
    elapsed = float((response_ts.iloc[-1] - request_ts.iloc[0]).total_seconds())
    return pd.DataFrame(
        [
            {
                "request_count": len(expected),
                "route_start_ts": request_ts.iloc[0].isoformat(),
                "route_ready_ts": response_ts.iloc[-1].isoformat(),
                "elapsed_seconds": elapsed,
                "timeout_seconds": int(timeout_seconds),
                "within_timeout": elapsed <= float(timeout_seconds),
            }
        ]
    )


def validate_serialized_route_runtime_summary(
    runtime: pd.DataFrame,
    *,
    expected_request_count: int,
    timeout_seconds: int = FROZEN_READINESS_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Fail closed unless a serialized-route runtime receipt matches its contract."""
    _require_columns(
        runtime,
        {
            "request_count", "route_start_ts", "route_ready_ts", "elapsed_seconds",
            "timeout_seconds", "within_timeout",
        },
        "serialized route runtime",
    )
    if len(runtime) != 1:
        raise ValueError("serialized route runtime must contain exactly one row")
    row = runtime.iloc[0]
    if int(row["request_count"]) != int(expected_request_count):
        raise ValueError("serialized route runtime request count differs from the contract")
    if int(row["timeout_seconds"]) != int(timeout_seconds):
        raise ValueError("serialized route runtime timeout differs from the contract")
    start = pd.to_datetime(row["route_start_ts"], utc=True, errors="coerce")
    ready = pd.to_datetime(row["route_ready_ts"], utc=True, errors="coerce")
    elapsed = float(row["elapsed_seconds"])
    within = row["within_timeout"] is True or str(row["within_timeout"]).lower() == "true"
    if pd.isna(start) or pd.isna(ready) or ready < start:
        raise ValueError("serialized route runtime contains invalid timestamps")
    observed_elapsed = float((ready - start).total_seconds())
    if abs(observed_elapsed - elapsed) > 1e-6:
        raise ValueError("serialized route runtime elapsed value is inconsistent")
    if elapsed > float(timeout_seconds) or not within:
        raise ValueError("serialized route exceeds the frozen readiness timeout")
    return runtime.copy().reset_index(drop=True)


def summarize_realtime_equivalence_evidence(
    comparison: pd.DataFrame,
    *,
    expected_symbols: Sequence[str],
    formula_test_ok: bool,
    raw_evidence_sha256: Mapping[str, str],
    source_observed_ts: str,
) -> pd.DataFrame:
    """Summarize row-level source evidence without converting coverage into equivalence."""
    _require_columns(
        comparison,
        {
            "endpoint", "symbol", "normalized_value", "comparison_ok", "mapping_id",
        },
        "comparison",
    )
    expected = sorted(dict.fromkeys(str(value).upper() for value in expected_symbols))
    if not expected:
        raise ValueError("expected_symbols must not be empty")
    rows: list[dict[str, object]] = []
    for endpoint, contract in REALTIME_EQUIVALENCE_CONTRACT.items():
        endpoint_rows = comparison.loc[comparison["endpoint"].eq(endpoint)].copy()
        finite = pd.to_numeric(endpoint_rows["normalized_value"], errors="coerce").notna()
        covered = sorted(set(endpoint_rows.loc[finite, "symbol"].astype(str).str.upper()))
        schema_ok = covered == expected
        mapping_ok = (
            not endpoint_rows.empty
            and endpoint_rows["mapping_id"].astype(str).str.strip().ne("").all()
            and endpoint_rows["mapping_id"].nunique() == 1
        )
        comparison_ok = (
            not endpoint_rows.empty
            and endpoint_rows["comparison_ok"].map(
                lambda value: value is True or str(value).lower() == "true"
            ).all()
        )
        depth_ok = True
        if endpoint in {"ob_pair", "ob_agg"}:
            if "full_depth_band_covered" not in endpoint_rows.columns:
                depth_ok = False
            else:
                depth_ok = endpoint_rows["full_depth_band_covered"].fillna(False).map(
                    lambda value: value is True or str(value).lower() == "true"
                ).all()
        direction_ok = bool(formula_test_ok and mapping_ok and comparison_ok)
        unit_ok = bool(formula_test_ok and mapping_ok and comparison_ok and depth_ok)
        native_window_ok = bool(schema_ok and comparison_ok and depth_ok)
        digest = str(raw_evidence_sha256.get(endpoint, ""))
        evidence_status = (
            "verified"
            if schema_ok and direction_ok and unit_ok and native_window_ok
            and bool(pd.Series([digest]).str.fullmatch(r"[0-9a-f]{64}").iloc[0])
            else "unverified"
        )
        rows.append(
            {
                "endpoint": endpoint,
                "schema_ok": schema_ok,
                "direction_ok": direction_ok,
                "unit_ok": unit_ok,
                "native_window_ok": native_window_ok,
                "formula_test_ok": bool(formula_test_ok),
                "evidence_status": evidence_status,
                "mapping_id": str(endpoint_rows["mapping_id"].iloc[0]) if mapping_ok else "",
                "raw_evidence_sha256": digest,
                "comparison_row_count": len(endpoint_rows),
                "covered_symbols": ",".join(covered),
                "source_observed_ts": source_observed_ts,
            }
        )
    return pd.DataFrame(rows).sort_values("endpoint").reset_index(drop=True)


def evaluate_realtime_source_comparisons(comparison: pd.DataFrame) -> pd.DataFrame:
    """Evaluate row-level source pairs under the frozen endpoint contract."""
    _require_columns(
        comparison,
        {
            "endpoint", "symbol", "normalized_value", "market_data_ts", "mapping_id",
            "reference_value", "reference_market_ts", "reference_source",
        },
        "comparison",
    )
    result = comparison.copy()
    result["comparison_ok"] = pd.to_numeric(
        result["normalized_value"], errors="coerce"
    ).notna()
    result["comparison_note"] = "finite value and frozen same-provider mapping"
    for endpoint, definition in REALTIME_SOURCE_DEFINITIONS.items():
        endpoint_mask = result["endpoint"].eq(endpoint)
        result.loc[endpoint_mask, "comparison_ok"] &= result.loc[
            endpoint_mask, "mapping_id"
        ].eq(definition["mapping_id"])

    top_btc = result["endpoint"].eq("top_pos") & result["symbol"].eq("BTC")
    top_public = pd.to_numeric(result.loc[top_btc, "normalized_value"], errors="coerce")
    top_reference = pd.to_numeric(result.loc[top_btc, "reference_value"], errors="coerce")
    top_time = pd.to_datetime(result.loc[top_btc, "market_data_ts"], utc=True, errors="coerce")
    top_reference_time = pd.to_datetime(
        result.loc[top_btc, "reference_market_ts"], utc=True, errors="coerce"
    )
    top_ok = (
        top_public.notna()
        & top_reference.notna()
        & top_time.notna()
        & top_reference_time.notna()
        & top_time.eq(top_reference_time)
        & (top_public.round(2) - top_reference).abs().le(1e-12)
    )
    result.loc[top_btc, "comparison_ok"] = top_ok.to_numpy()
    result.loc[top_btc, "comparison_note"] = (
        "BTC requires identical native label and agreement at CoinGlass two-decimal precision"
    )

    orderbook = result["endpoint"].isin({"ob_pair", "ob_agg"})
    if orderbook.any():
        _require_columns(result, {"required_venues"}, "orderbook comparison")
    if "full_depth_band_covered" not in result.columns:
        result.loc[orderbook, "comparison_ok"] = False
    else:
        band_ok = result.loc[orderbook, "full_depth_band_covered"].fillna(False).map(
            lambda value: value is True or str(value).lower() == "true"
        )
        result.loc[orderbook, "comparison_ok"] &= band_ok.to_numpy()
    for row_index in result.index[orderbook]:
        endpoint = str(result.at[row_index, "endpoint"])
        declared_venues = tuple(
            value.strip()
            for value in str(result.at[row_index, "required_venues"]).split(",")
            if value.strip()
        )
        expected_venues = expected_orderbook_venues(
            endpoint, str(result.at[row_index, "symbol"])
        )
        if tuple(sorted(declared_venues)) != tuple(sorted(expected_venues)):
            result.at[row_index, "comparison_ok"] = False
    reference_orderbook = orderbook & result["reference_source"].astype(str).ne("definition-only")
    reference_time = pd.to_datetime(
        result.loc[reference_orderbook, "reference_market_ts"], utc=True, errors="coerce"
    )
    reference_value = pd.to_numeric(
        result.loc[reference_orderbook, "reference_value"], errors="coerce"
    )
    time_ok = reference_time.notna() & reference_value.notna()
    venue_columns = {
        "binance_public": "binance_market_ts",
        "okx_public": "okx_market_ts",
        "bybit_public": "bybit_market_ts",
        "keystore": "keystore_market_ts",
    }
    for row_index in result.index[reference_orderbook]:
        endpoint = str(result.at[row_index, "endpoint"])
        reference = pd.to_datetime(
            result.at[row_index, "reference_market_ts"], utc=True, errors="coerce"
        )
        required_venues = tuple(
            value.strip()
            for value in str(result.at[row_index, "required_venues"]).split(",")
            if value.strip()
        )
        expected_venues = expected_orderbook_venues(endpoint, str(result.at[row_index, "symbol"]))
        venue_times = [
            pd.to_datetime(
                result.at[row_index, venue_columns[venue]], utc=True, errors="coerce"
            )
            if venue in venue_columns and venue_columns[venue] in result.columns else pd.NaT
            for venue in required_venues
        ]
        row_time_ok = (
            pd.notna(reference)
            and tuple(sorted(required_venues)) == tuple(sorted(expected_venues))
            and len(venue_times) == len(expected_venues)
            and all(pd.notna(value) for value in venue_times)
            and all(abs(value - reference) <= pd.Timedelta(minutes=3) for value in venue_times)
            and max(venue_times) - min(venue_times) <= pd.Timedelta(minutes=1)
        )
        time_ok.loc[row_index] = bool(time_ok.loc[row_index] and row_time_ok)
    result.loc[reference_orderbook, "comparison_ok"] &= time_ok.to_numpy()
    result.loc[orderbook, "comparison_note"] = (
        "frozen available venues and full +/-1% band required; paired CoinGlass "
        "references must be within 3 minutes"
    )
    return result


def attach_historical_execution_ledger(
    target_membership: pd.DataFrame,
    execution_opens: Mapping[str, pd.Series],
    *,
    horizon_deltas: Mapping[str, pd.Timedelta],
    execution_delay_minutes: int,
) -> pd.DataFrame:
    """Attach exact delayed open-to-open returns to L3 target membership."""
    required = {"decision_ts", "symbol", "panel_frequency", "return_horizon", "component_features"}
    _require_columns(target_membership, required, "target_membership")
    if target_membership.empty:
        raise ValueError("target_membership must not be empty")
    working = target_membership.copy()
    working["decision_ts"] = pd.to_datetime(working["decision_ts"], utc=True)
    route_pairs = working[["panel_frequency", "return_horizon"]].drop_duplicates()
    if len(route_pairs) != 1:
        raise ValueError("one execution-ledger call must contain exactly one horizon route")
    route = route_pairs.iloc[0]
    if str(route.panel_frequency) != str(route.return_horizon):
        raise ValueError("panel_frequency must equal return_horizon")
    signal_timeframes = tuple(
        sorted(
            {
                token.rsplit("__", 1)[-1]
                for value in working["component_features"].astype(str).unique()
                for token in value.split(" | ")
                if "__" in token
            },
            key=lambda value: pd.Timedelta(horizon_deltas[value]),
        )
    )
    contract = ContinuousHoldingTimeContract(
        return_horizon=str(route.return_horizon),
        decision_interval=str(route.panel_frequency),
        holding_interval=str(route.return_horizon),
        strategy_return_interval=str(route.return_horizon),
        signal_timeframes=signal_timeframes,
        execution_delay_minutes=int(execution_delay_minutes),
        data_observed_rule=f"frozen_readiness_timeout_t_plus_{int(execution_delay_minutes)}m_open",
    )
    ledgers: list[pd.DataFrame] = []
    for symbol, symbol_rows in working.groupby("symbol", sort=True):
        if str(symbol) not in execution_opens:
            raise ValueError(f"execution opens missing symbol={symbol}")
        decisions = pd.DatetimeIndex(symbol_rows["decision_ts"].drop_duplicates().sort_values())
        ledger = executable_returns_for_symbol(decisions, execution_opens[str(symbol)], contract, horizon_deltas)
        ledger["symbol"] = str(symbol)
        ledgers.append(ledger)
    ledger = pd.concat(ledgers, ignore_index=True)
    merge_keys = ["decision_ts", "symbol"]
    overlap = sorted(set(working.columns).intersection(ledger.columns).difference(merge_keys))
    result = working.merge(
        ledger,
        on=merge_keys,
        how="left",
        validate="many_to_one",
        suffixes=("", "_ledger"),
    )
    for column in overlap:
        ledger_column = f"{column}_ledger"
        agrees = result[column].eq(result[ledger_column]) | (
            result[column].isna() & result[ledger_column].isna()
        )
        if not agrees.all():
            raise ValueError(f"execution ledger conflicts with target column: {column}")
        result = result.drop(columns=ledger_column)
    check = ["execution_ts", "next_execution_ts", "entry_price", "exit_price", "executable_return"]
    if result[check].isna().any().any():
        raise ValueError("execution ledger is incomplete")
    if any(column.endswith(("_x", "_y", "_ledger")) for column in result.columns):
        raise ValueError("execution ledger contains unresolved merge-suffix columns")
    return result


__all__ = [
    "FROZEN_READINESS_TIMEOUT_SECONDS",
    "FROZEN_KEYSTORE_ORDERBOOK_SYMBOLS",
    "FROZEN_UNAVAILABLE_ORDERBOOK_ROUTES",
    "REALTIME_EQUIVALENCE_CONTRACT",
    "REALTIME_SOURCE_DEFINITIONS",
    "apply_orderbook_venue_availability",
    "attach_historical_execution_ledger",
    "build_realtime_source_plan",
    "evaluate_realtime_source_comparisons",
    "evaluate_serialized_route_runtime",
    "validate_serialized_route_runtime_summary",
    "expected_orderbook_venues",
    "frozen_execution_delay_minutes",
    "summarize_realtime_equivalence_evidence",
    "validate_realtime_equivalence_contract",
]
