"""Formal diagnostic entry for KSV4 orderbook timeframe lineage audits.

This module does not decide whether a source is suitable for research.  It
compares independently labelled orderbook histories and tests a small set of
explicit 1h aggregation hypotheses without silently filling missing symbols.
The factor transform and rank standardisation are delegated to the same qlab
primitives used by the frozen panel path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .keystore_coinglass_client import (
    RawKeystoreDiagnosticResponse,
    find_data_rows,
)
from .keystore_coinglass_panel import extract_feature_series
from .panel_statistics import rank_standardize_grouped_series


ORDERBOOK_RAW_FIELDS = {
    "ob_pair": ("bids_usd", "bids_quantity", "asks_usd", "asks_quantity"),
    "ob_agg": (
        "aggregated_bids_usd", "aggregated_bids_quantity",
        "aggregated_asks_usd", "aggregated_asks_quantity",
    ),
}
ORDERBOOK_IMBALANCE_FIELDS = {
    "ob_pair": ("bids_usd", "asks_usd"),
    "ob_agg": ("aggregated_bids_usd", "aggregated_asks_usd"),
}
AGGREGATION_METHODS = ("same_label", "first", "last", "mean", "median", "min", "max")
MINIMAL_PAIR_PATH = "/api/futures/orderbook/ask-bids-history"
MINIMAL_PAIR_TIMEFRAMES = ("1h", "12h", "1d")


def build_minimal_pair_probe_contract(
    *,
    target_label_ms: int,
    start_time_ms: int,
    end_time_ms: int,
) -> list[dict[str, object]]:
    """Build exactly three BTCUSDT pair requests differing only by interval."""
    if not int(start_time_ms) <= int(target_label_ms) < int(end_time_ms):
        raise ValueError("target label must lie inside the fixed request window")
    common = {
        "exchange": "Binance",
        "symbol": "BTCUSDT",
        "range": "1",
        "limit": 1000,
        "start_time": int(start_time_ms),
        "end_time": int(end_time_ms),
    }
    return [
        {
            "request_id": f"btc_pair_{timeframe}",
            "endpoint": "ob_pair",
            "timeframe": timeframe,
            "target_label_ms": int(target_label_ms),
            "path": MINIMAL_PAIR_PATH,
            "params": {**common, "interval": timeframe},
        }
        for timeframe in MINIMAL_PAIR_TIMEFRAMES
    ]


def persist_minimal_pair_probe_response(
    root: Path,
    request_record: Mapping[str, object],
    response: RawKeystoreDiagnosticResponse,
) -> dict[str, object]:
    """Immutably preserve one raw response and its credential-free receipt."""
    expected_params = dict(request_record["params"])
    if response.path != request_record["path"] or response.request_params != expected_params:
        raise ValueError("response identity differs from the fixed minimal contract")
    payload_sha = hashlib.sha256(response.raw_payload).hexdigest()
    object_path = root / "objects" / payload_sha[:2] / f"{payload_sha}.bin"
    object_path.parent.mkdir(parents=True, exist_ok=True)
    if object_path.exists():
        if object_path.read_bytes() != response.raw_payload:
            raise ValueError("existing minimal pair object differs from response bytes")
    else:
        with object_path.open("xb") as handle:
            handle.write(response.raw_payload)
    try:
        payload = response.json_payload()
        payload_json_parseable = True
    except ValueError:
        payload = None
        payload_json_parseable = False
    rows = find_data_rows(payload)
    target_label_ms = int(request_record["target_label_ms"])
    exact_rows = [
        row for row in rows
        if isinstance(row, Mapping) and int(row.get("time", -1)) == target_label_ms
    ]
    receipt = {
        "Lifecycle": "candidate diagnostic evidence",
        "Authority": "unaltered KeyStore/CoinGlass proxy response bytes bound by SHA-256",
        "Inputs": "fixed Issue #34 minimal BTCUSDT pair request identity; authentication excluded",
        "May be used for": "Issue #34 stage-1 three-timeframe raw-response comparison",
        "Must not be used for": "aggregated/other-symbol inference, rank, L0-L4, simulations, confirmation tests, deployment, or trading",
        "Archive condition": "archive after reviewed Issue #34 stage-1 evidence is superseded",
        "request_id": request_record["request_id"],
        "endpoint": request_record["endpoint"],
        "timeframe": request_record["timeframe"],
        "target_label_ms": target_label_ms,
        "path": response.path,
        "request_params": response.request_params,
        "request_ts": response.request_ts,
        "response_ts": response.response_ts,
        "http_status": response.http_status,
        "business_code": response.business_code,
        "business_message": response.business_message,
        "payload_json_parseable": payload_json_parseable,
        "payload_sha256": payload_sha,
        "payload_bytes": len(response.raw_payload),
        "row_count": len(rows),
        "exact_target_row_count": len(exact_rows),
        "authentication_recorded": False,
    }
    receipt_path = root / "receipts" / f"{request_record['request_id']}.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("xb") as handle:
        handle.write(
            (json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
        )
    return receipt


def compare_minimal_pair_probe(
    root: Path,
    contract: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Compare only the exact target row across the three fixed intervals."""
    if [str(row["timeframe"]) for row in contract] != list(MINIMAL_PAIR_TIMEFRAMES):
        raise ValueError("minimal pair contract must contain exactly 1h, 12h, 1d")
    per_interval: list[dict[str, object]] = []
    comparable_frames: dict[str, pd.DataFrame] = {}
    for record in contract:
        timeframe = str(record["timeframe"])
        receipt_path = root / "receipts" / f"{record['request_id']}.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        payload_sha = str(receipt["payload_sha256"])
        payload_path = root / "objects" / payload_sha[:2] / f"{payload_sha}.bin"
        if file_sha256(payload_path) != payload_sha:
            raise ValueError("minimal pair payload SHA mismatch")
        try:
            payload = json.loads(payload_path.read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        rows = find_data_rows(payload)
        exact_rows = [
            row for row in rows
            if isinstance(row, Mapping)
            and int(row.get("time", -1)) == int(record["target_label_ms"])
        ]
        business_ok = str(receipt["business_code"]) in {"", "0", "success"}
        response_ok = int(receipt["http_status"]) == 200 and business_ok
        result: dict[str, object] = {
            "timeframe": timeframe,
            "http_status": int(receipt["http_status"]),
            "business_code": str(receipt["business_code"]),
            "request_succeeded": response_ok,
            "exact_target_row_count": len(exact_rows),
            "exact_target_present": len(exact_rows) == 1,
            "payload_sha256": payload_sha,
        }
        if response_ok and len(exact_rows) == 1:
            frame = pd.DataFrame(exact_rows)
            frame["ts"] = pd.to_datetime(frame.pop("time"), unit="ms", utc=True)
            frame = _normalized(frame.set_index("ts"), ORDERBOOK_RAW_FIELDS["ob_pair"])
            imbalance = orderbook_imbalance(frame, endpoint="ob_pair")
            comparable_frames[timeframe] = frame
            for field in ORDERBOOK_RAW_FIELDS["ob_pair"]:
                result[field] = float(frame.iloc[0][field])
            result["imbalance"] = float(imbalance.iloc[0])
        per_interval.append(result)
    comparable = len(comparable_frames) == 3
    fields = (*ORDERBOOK_RAW_FIELDS["ob_pair"], "imbalance")
    equality = {
        field: (
            len({float(row[field]) for row in per_interval}) == 1
            if comparable else None
        )
        for field in fields
    }
    return {
        "target_label_ms": int(contract[0]["target_label_ms"]),
        "same_request_identity_except_interval": all(
            {
                key: value for key, value in dict(record["params"]).items()
                if key != "interval"
            }
            == {
                key: value for key, value in dict(contract[0]["params"]).items()
                if key != "interval"
            }
            for record in contract
        ),
        "all_three_comparable": comparable,
        "per_interval": per_interval,
        "field_equal_across_all_three": equality,
        "all_raw_and_imbalance_equal": (
            all(bool(value) for value in equality.values()) if comparable else None
        ),
    }


def _normalized(frame: pd.DataFrame, fields: Sequence[str]) -> pd.DataFrame:
    missing = [field for field in fields if field not in frame.columns]
    if missing:
        raise ValueError("orderbook frame missing fields: " + ", ".join(missing))
    result = frame.loc[:, list(fields)].copy()
    result.index = pd.to_datetime(result.index, utc=True)
    result.index.name = "ts"
    result = result.sort_index()
    if result.index.has_duplicates:
        raise ValueError("orderbook frame contains duplicate labels")
    for field in fields:
        result[field] = pd.to_numeric(result[field], errors="raise")
    return result


def orderbook_imbalance(frame: pd.DataFrame, *, endpoint: str) -> pd.Series:
    """Apply the frozen panel's orderbook imbalance transform."""
    try:
        buy, sell = ORDERBOOK_IMBALANCE_FIELDS[str(endpoint)]
    except KeyError as exc:
        raise ValueError(f"unsupported orderbook endpoint: {endpoint}") from exc
    spec = pd.Series(
        {
            "feature_name": f"{endpoint}_imbalance_diagnostic",
            "required_columns": f"{buy},{sell}",
            "panel_transform": "buy_sell_imbalance",
        }
    )
    return extract_feature_series(spec, _normalized(frame, (buy, sell)))


def one_hour_hypotheses(
    one_hour: pd.DataFrame,
    labels: Sequence[object],
    *,
    endpoint: str,
    coarse_timeframe: str,
) -> dict[str, pd.DataFrame]:
    """Build explicit candidate values from 1h observations.

    Window candidates use ``(label - duration, label]``.  This convention is
    diagnostic only; it is not asserted to be the vendor's undocumented bar
    convention.  ``same_label`` selects only the row exactly at ``label``.
    """
    try:
        duration = {"12h": pd.Timedelta(hours=12), "1d": pd.Timedelta(days=1)}[
            str(coarse_timeframe)
        ]
    except KeyError as exc:
        raise ValueError("coarse_timeframe must be 12h or 1d") from exc
    fields = ORDERBOOK_RAW_FIELDS[str(endpoint)]
    source = _normalized(one_hour, fields)
    normalized_labels = pd.DatetimeIndex(pd.to_datetime(list(labels), utc=True), name="ts")
    exact_positions = source.index.get_indexer(normalized_labels)
    keep = exact_positions >= 0
    index = normalized_labels[keep]
    exact_positions = exact_positions[keep]
    left_positions = source.index.searchsorted(index - duration, side="right")
    same = source.iloc[exact_positions].copy()
    same.index = index
    first = source.iloc[left_positions].copy()
    first.index = index
    rolling = source.rolling(duration, closed="right")
    return {
        "same_label": same,
        "first": first,
        "last": same.copy(),
        "mean": rolling.mean().reindex(index),
        "median": rolling.median().reindex(index),
        "min": rolling.min().reindex(index),
        "max": rolling.max().reindex(index),
    }


def audit_orderbook_timeframe_relationship(
    one_hour_by_symbol: Mapping[str, pd.DataFrame],
    coarse_by_symbol: Mapping[str, pd.DataFrame],
    *,
    endpoint: str,
    coarse_timeframe: str,
    expected_symbols: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare raw fields, transformed imbalance, and complete-section ranks.

    Returns ``(value_records, rank_records)``.  Rank rows are emitted only for
    labels where every expected symbol exists on both sides and has a valid
    imbalance.  The function never reranks a reduced cross-section.
    """
    symbols = tuple(str(symbol).upper() for symbol in expected_symbols)
    if set(one_hour_by_symbol) != set(symbols) or set(coarse_by_symbol) != set(symbols):
        raise ValueError("inputs must contain exactly the expected symbol set")
    fields = ORDERBOOK_RAW_FIELDS[str(endpoint)]
    value_rows: list[dict[str, object]] = []
    candidates: dict[str, dict[str, pd.DataFrame]] = {}
    candidate_imbalances: dict[str, dict[str, pd.Series]] = {}
    coarse_frames: dict[str, pd.DataFrame] = {}
    coarse_imbalances: dict[str, pd.Series] = {}
    for symbol in symbols:
        coarse = _normalized(coarse_by_symbol[symbol], fields)
        coarse_frames[symbol] = coarse
        hypotheses = one_hour_hypotheses(
            one_hour_by_symbol[symbol], coarse.index, endpoint=endpoint,
            coarse_timeframe=coarse_timeframe,
        )
        candidates[symbol] = hypotheses
        coarse_imbalance = orderbook_imbalance(coarse, endpoint=endpoint)
        coarse_imbalances[symbol] = coarse_imbalance
        candidate_imbalances[symbol] = {}
        for method, candidate in hypotheses.items():
            candidate_imbalance = orderbook_imbalance(candidate, endpoint=endpoint)
            candidate_imbalances[symbol][method] = candidate_imbalance

    complete_labels_by_method: dict[str, pd.DatetimeIndex] = {}
    for method in AGGREGATION_METHODS:
        common: pd.DatetimeIndex | None = None
        for symbol in symbols:
            labels = (
                coarse_frames[symbol].index
                .intersection(candidates[symbol][method].index)
                .intersection(coarse_imbalances[symbol].index)
                .intersection(candidate_imbalances[symbol][method].index)
            )
            common = labels if common is None else common.intersection(labels)
        complete_labels_by_method[method] = (
            common if common is not None else pd.DatetimeIndex([], tz="UTC", name="ts")
        )

    for symbol in symbols:
        coarse = coarse_frames[symbol]
        coarse_imbalance = coarse_imbalances[symbol]
        for method, candidate in candidates[symbol].items():
            candidate_imbalance = candidate_imbalances[symbol][method]
            common = complete_labels_by_method[method]
            observed_frame = coarse.reindex(common)
            proposed_frame = candidate.reindex(common)
            for field in fields:
                block = pd.DataFrame(
                    {
                        "label": common,
                        "coarse_value": observed_frame[field].to_numpy(dtype=float),
                        "one_hour_candidate": proposed_frame[field].to_numpy(dtype=float),
                    }
                )
                block["endpoint"] = endpoint
                block["coarse_timeframe"] = coarse_timeframe
                block["symbol"] = symbol
                block["method"] = method
                block["metric"] = field
                block["exact_equal"] = block.coarse_value.eq(block.one_hour_candidate)
                block["absolute_difference"] = block.one_hour_candidate - block.coarse_value
                value_rows.extend(block.to_dict("records"))
            if not common.empty:
                block = pd.DataFrame(
                    {
                        "label": common,
                        "coarse_value": coarse_imbalance.reindex(common).to_numpy(dtype=float),
                        "one_hour_candidate": candidate_imbalance.reindex(common).to_numpy(dtype=float),
                    }
                )
                block["endpoint"] = endpoint
                block["coarse_timeframe"] = coarse_timeframe
                block["symbol"] = symbol
                block["method"] = method
                block["metric"] = "imbalance"
                block["exact_equal"] = block.coarse_value.eq(block.one_hour_candidate)
                block["absolute_difference"] = block.one_hour_candidate - block.coarse_value
                value_rows.extend(block.to_dict("records"))

    rank_rows: list[dict[str, object]] = []
    for method in AGGREGATION_METHODS:
        common_labels = complete_labels_by_method[method]
        if common_labels.empty:
            continue
        coarse_matrix = pd.DataFrame(
            {symbol: coarse_imbalances[symbol].reindex(common_labels) for symbol in symbols},
            index=common_labels,
        )
        candidate_matrix = pd.DataFrame(
            {symbol: candidate_imbalances[symbol][method].reindex(common_labels) for symbol in symbols},
            index=common_labels,
        )
        complete_mask = coarse_matrix.notna().all(axis=1) & candidate_matrix.notna().all(axis=1)
        coarse_matrix = coarse_matrix.loc[complete_mask]
        candidate_matrix = candidate_matrix.loc[complete_mask]
        if coarse_matrix.empty:
            continue

        def standardized(matrix: pd.DataFrame) -> pd.Series:
            stacked = matrix.rename_axis(index="decision_ts", columns="symbol").stack()
            return rank_standardize_grouped_series(stacked, level="decision_ts")

        coarse_rank = standardized(coarse_matrix)
        candidate_rank = standardized(candidate_matrix)
        comparison = pd.DataFrame(
            {"coarse_rank": coarse_rank, "one_hour_candidate_rank": candidate_rank}
        ).reset_index()
        comparison["endpoint"] = endpoint
        comparison["coarse_timeframe"] = coarse_timeframe
        comparison["method"] = method
        comparison["rank_exact_equal"] = comparison.coarse_rank.eq(
            comparison.one_hour_candidate_rank
        )
        comparison["standardized_rank_difference"] = (
            comparison.one_hour_candidate_rank - comparison.coarse_rank
        )
        comparison["cross_section_size"] = len(symbols)
        comparison = comparison.rename(columns={"decision_ts": "label"})
        rank_rows.extend(comparison.to_dict("records"))
    return pd.DataFrame(value_rows), pd.DataFrame(rank_rows)


def compare_cache_to_raw_history(
    raw_by_identity: Mapping[tuple[str, str, str], pd.DataFrame],
    cache_by_identity: Mapping[tuple[str, str, str], pd.DataFrame],
) -> pd.DataFrame:
    """Verify that cache frames preserve their parsed per-interval histories."""
    if set(raw_by_identity) != set(cache_by_identity):
        raise ValueError("raw history and cache identity sets differ")
    records: list[dict[str, object]] = []
    for identity in sorted(raw_by_identity):
        endpoint, timeframe, symbol = identity
        fields = ORDERBOOK_RAW_FIELDS[endpoint]
        raw = _normalized(raw_by_identity[identity], fields)
        cache = _normalized(cache_by_identity[identity], fields)
        same_index = raw.index.equals(cache.index)
        cache_labels_present = bool(cache.index.isin(raw.index).all())
        aligned_raw = raw.reindex(cache.index) if cache_labels_present else None
        same_values = bool(
            cache_labels_present
            and np.array_equal(
                aligned_raw.to_numpy(dtype=float), cache.to_numpy(dtype=float), equal_nan=True
            )
        )
        records.append(
            {
                "endpoint": endpoint, "timeframe": timeframe, "symbol": symbol,
                "raw_rows": len(raw), "cache_rows": len(cache),
                "raw_extra_rows": len(raw.index.difference(cache.index)),
                "index_equal": same_index, "cache_labels_present": cache_labels_present,
                "values_equal_on_cache_labels": same_values,
            }
        )
    return pd.DataFrame(records)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_orderbook_history_csv(path: Path, *, endpoint: str) -> pd.DataFrame:
    """Load one parsed history without changing its labels or numeric fields."""
    frame = pd.read_csv(path, parse_dates=["ts"]).set_index("ts")
    return _normalized(frame, ORDERBOOK_RAW_FIELDS[str(endpoint)])


def load_immutable_orderbook_frames(
    receipt_root: Path,
    *,
    endpoint: str,
    timeframe: str,
    expected_symbols: Sequence[str],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]]]:
    """Load and SHA-verify archived receipts and unparsed supplier bytes."""
    frames: dict[str, pd.DataFrame] = {}
    lineage: list[dict[str, object]] = []
    fields = ORDERBOOK_RAW_FIELDS[str(endpoint)]
    for symbol_value in expected_symbols:
        symbol = str(symbol_value).upper()
        directory = receipt_root / "receipts" / f"keystore_v4_{endpoint}_{timeframe}_{symbol}"
        receipts = sorted(directory.glob("*.json"))
        if len(receipts) != 1:
            raise ValueError(f"expected one immutable receipt for {endpoint}/{timeframe}/{symbol}")
        receipt_path = receipts[0]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        payload_sha = str(receipt["payload_sha256"])
        payload = receipt_root / "objects" / payload_sha[:2] / f"{payload_sha}.bin"
        if file_sha256(payload) != payload_sha:
            raise ValueError(f"payload SHA mismatch: {payload}")
        response = json.loads(payload.read_bytes())
        rows = response["data"]
        frame = pd.DataFrame(rows)
        frame["ts"] = pd.to_datetime(frame.pop("time"), unit="ms", utc=True)
        frames[symbol] = _normalized(frame.set_index("ts"), fields)
        lineage.append(
            {
                "endpoint": endpoint, "timeframe": timeframe, "symbol": symbol,
                "receipt": str(receipt_path), "receipt_sha256": file_sha256(receipt_path),
                "payload": str(payload), "payload_sha256": payload_sha,
                "rows": len(frames[symbol]), "min_label": frames[symbol].index.min(),
                "max_label": frames[symbol].index.max(),
                "request_params_present": False,
                "receipt_direct_exchange_scope": "not recorded",
                "receipt_direct_price_range_percent": "not recorded",
                "request_identity_boundary": (
                    "receipt alone does not prove query parameters; see the separately "
                    "graded historical request provenance"
                ),
            }
        )
    return frames, lineage


def summarize_archived_request_provenance(
    receipt_root: Path,
    *,
    expected_symbols: Sequence[str],
) -> list[dict[str, object]]:
    """Summarize immutable 1h/12h request timing and common-label values.

    This entry deliberately uses receipt timestamps and SHA-verified response
    bytes.  It does not infer undocumented supplier aggregation semantics.
    """
    symbols = tuple(str(symbol).upper() for symbol in expected_symbols)
    records: list[dict[str, object]] = []
    for endpoint in ("ob_pair", "ob_agg"):
        frames: dict[str, dict[str, pd.DataFrame]] = {}
        receipts: dict[tuple[str, str], dict[str, object]] = {}
        for timeframe in ("1h", "12h"):
            frames[timeframe], _ = load_immutable_orderbook_frames(
                receipt_root,
                endpoint=endpoint,
                timeframe=timeframe,
                expected_symbols=symbols,
            )
            for symbol in symbols:
                directory = (
                    receipt_root / "receipts"
                    / f"keystore_v4_{endpoint}_{timeframe}_{symbol}"
                )
                paths = sorted(directory.glob("*.json"))
                if len(paths) != 1:
                    raise ValueError(
                        f"expected one receipt for {endpoint}/{timeframe}/{symbol}"
                    )
                receipts[(timeframe, symbol)] = json.loads(
                    paths[0].read_text(encoding="utf-8")
                )

        lags: list[float] = []
        common_labels: pd.DatetimeIndex | None = None
        differing_symbols = 0
        fields = ORDERBOOK_RAW_FIELDS[endpoint]
        for symbol in symbols:
            one = frames["1h"][symbol]
            twelve = frames["12h"][symbol]
            labels = one.index.intersection(twelve.index)
            if labels.empty:
                raise ValueError(
                    f"no common label for {endpoint}/1h/12h/{symbol}"
                )
            common_labels = labels if common_labels is None else common_labels.intersection(labels)
            label = labels.max()
            if not np.array_equal(
                one.loc[label, list(fields)].to_numpy(dtype=float),
                twelve.loc[label, list(fields)].to_numpy(dtype=float),
                equal_nan=True,
            ):
                differing_symbols += 1
            one_ts = pd.Timestamp(receipts[("1h", symbol)]["source_request_ts"])
            twelve_ts = pd.Timestamp(receipts[("12h", symbol)]["source_request_ts"])
            lags.append(float((one_ts - twelve_ts).total_seconds()))

        if common_labels is None or common_labels.empty:
            raise ValueError(f"no complete common label for {endpoint}")
        windows = {}
        for timeframe in ("1h", "12h"):
            timestamps = sorted(
                pd.Timestamp(receipts[(timeframe, symbol)]["source_request_ts"])
                for symbol in symbols
            )
            windows[timeframe] = {
                "first_request_ts": timestamps[0].isoformat(),
                "last_request_ts": timestamps[-1].isoformat(),
            }
        records.append(
            {
                "endpoint": endpoint,
                "symbols": len(symbols),
                "request_windows": windows,
                "one_hour_minus_twelve_hour_request_lag_seconds_min": min(lags),
                "one_hour_minus_twelve_hour_request_lag_seconds_median": float(
                    np.median(lags)
                ),
                "one_hour_minus_twelve_hour_request_lag_seconds_max": max(lags),
                "complete_common_labels": [label.isoformat() for label in common_labels],
                "latest_complete_common_label": common_labels.max().isoformat(),
                "symbols_with_different_raw_fields_at_latest_common_label": differing_symbols,
            }
        )
    return records


def audit_archived_orderbook_request_provenance(
    *,
    r6_smoke_root: Path,
    r12_smoke_root: Path,
    r6_source_manifest: Path,
    r12_source_manifest: Path,
    session_transcript: Path,
    expected_symbols: Sequence[str],
) -> dict[str, object]:
    """Build the formal offline provenance record for the r6→r12 receipts.

    Receipt facts are grade A.  Parameters that require the unavailable full
    runner source are grade C even when the retained execution transcript and
    frozen endpoint registry strongly support one reconstruction.
    """
    symbols = tuple(str(symbol).upper() for symbol in expected_symbols)
    r6_manifest = json.loads(r6_source_manifest.read_text(encoding="utf-8"))
    r12_manifest = json.loads(r12_source_manifest.read_text(encoding="utf-8"))
    r6_sources = {row["path"]: row["sha256"] for row in r6_manifest["sources"]}
    r12_sources = {row["path"]: row["sha256"] for row in r12_manifest["sources"]}
    endpoint_path = Path(__file__).with_name("keystore_coinglass_endpoints.py")
    endpoint_sha = file_sha256(endpoint_path)
    expected_endpoint_sha = r6_sources[
        "qlab/qlab/data/crypto/keystore_coinglass_endpoints.py"
    ]
    if endpoint_sha != expected_endpoint_sha or r12_sources.get(
        "qlab/qlab/data/crypto/keystore_coinglass_endpoints.py"
    ) != expected_endpoint_sha:
        raise ValueError("local endpoint registry is not the frozen r6/r12 object")

    strings: list[str] = []

    def collect_strings(value: object) -> None:
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, Mapping):
            for item in value.values():
                collect_strings(item)
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            for item in value:
                collect_strings(item)

    for line_number, line in enumerate(
        session_transcript.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            collect_strings(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid historical JSONL at line {line_number}") from exc

    command_pattern = re.compile(
        r"(?:ALL_PROXY=\S+\s+)?(?:HTTPS_PROXY=\S+\s+)?(?:HTTP_PROXY=\S+\s+)?"
        r"python qlab_research_private/research/crypto/live/ksv4_true_oos_shadow\.py "
        r"real-signal-smoke --freeze-version (v2_20260729_r(?:6|7|8|9|10|11|12)) "
        r"--decision-ts 2026-07-29T00:00:00Z[^\n\"]*"
    )
    commands: dict[str, str] = {}
    for value in strings:
        for match in command_pattern.finditer(value):
            command = match.group(0).strip()
            previous = commands.get(match.group(1))
            if previous is None or len(command) > len(previous):
                commands[match.group(1)] = command
    required_freezes = tuple(f"v2_20260729_r{index}" for index in range(6, 13))
    missing = [freeze for freeze in required_freezes if freeze not in commands]
    if missing:
        raise ValueError("historical invocation chain is incomplete: " + ", ".join(missing))
    if "--resume-root" in commands["v2_20260729_r6"]:
        raise ValueError("r6 original invocation unexpectedly resumes prior evidence")
    for freeze in required_freezes[1:]:
        if "--resume-root" not in commands[freeze]:
            raise ValueError(f"{freeze} invocation is missing resume-root")

    runner_call_pattern = re.compile(
        r"params = build_history_params\(\s*endpoint,\s*symbol=symbol,\s*"
        r"interval=timeframe,\s*limit=args\.limit,\s*\)\s*"
        r"observed = client\.request_raw\(endpoint\.path, params=params\)",
        re.MULTILINE,
    )
    runner_default_pattern = re.compile(
        r"real_signal_smoke\.add_argument\(\"--limit\", type=int, default=3\)"
    )
    call_matches = [match.group(0) for value in strings for match in runner_call_pattern.finditer(value)]
    default_matches = [match.group(0) for value in strings for match in runner_default_pattern.finditer(value)]
    if not call_matches or not default_matches:
        raise ValueError("historical runner excerpt is incomplete")

    receipt_fields = (
        "source_id", "source_request_ts", "source_response_ts",
        "source_bar_label_ts", "native_bar_end_ts", "data_observed_ts",
        "payload_sha256",
    )
    verified_objects = 0
    verified_receipt_pairs = 0
    selected_labels: dict[str, set[str]] = {"1h": set(), "12h": set()}
    native_ends: set[str] = set()
    for endpoint in ("ob_pair", "ob_agg"):
        for timeframe in ("1h", "12h"):
            for symbol in symbols:
                source_id = f"keystore_v4_{endpoint}_{timeframe}_{symbol}"
                pairs = []
                for smoke_root in (r6_smoke_root, r12_smoke_root):
                    paths = sorted(
                        (smoke_root / "as_received" / "receipts" / source_id).glob("*.json")
                    )
                    if len(paths) != 1:
                        raise ValueError(f"expected one receipt in each freeze: {source_id}")
                    receipt = json.loads(paths[0].read_text(encoding="utf-8"))
                    payload_sha = str(receipt["payload_sha256"])
                    payload = (
                        smoke_root / "as_received" / "objects" / payload_sha[:2]
                        / f"{payload_sha}.bin"
                    )
                    if file_sha256(payload) != payload_sha:
                        raise ValueError(f"payload SHA mismatch: {payload}")
                    pairs.append(receipt)
                    verified_objects += 1
                for field in receipt_fields:
                    if pairs[0][field] != pairs[1][field]:
                        raise ValueError(f"r6/r12 receipt mismatch: {source_id}/{field}")
                verified_receipt_pairs += 1
                selected_labels[timeframe].add(str(pairs[1]["source_bar_label_ts"]))
                native_ends.add(str(pairs[1]["native_bar_end_ts"]))
    if any(len(labels) != 1 for labels in selected_labels.values()) or len(native_ends) != 1:
        raise ValueError("formal receipt labels are not unique by timeframe")

    timing = summarize_archived_request_provenance(
        r12_smoke_root / "as_received", expected_symbols=symbols
    )
    reconstructed = [
        {"parameter": "path", "ob_pair": "/api/futures/orderbook/ask-bids-history", "ob_agg": "/api/futures/orderbook/aggregated-ask-bids-history"},
        {"parameter": "exchange", "ob_pair": "Binance", "ob_agg": "omitted"},
        {"parameter": "exchange_list", "ob_pair": "omitted", "ob_agg": "Binance,OKX,Bybit"},
        {"parameter": "symbol", "ob_pair": "<COIN>USDT", "ob_agg": "<COIN>"},
        {"parameter": "range", "ob_pair": "1", "ob_agg": "1"},
        {"parameter": "interval", "ob_pair": "1h or 12h", "ob_agg": "1h or 12h"},
        {"parameter": "limit", "ob_pair": "3", "ob_agg": "3"},
        {"parameter": "start_time", "ob_pair": "omitted", "ob_agg": "omitted"},
        {"parameter": "end_time", "ob_pair": "omitted", "ob_agg": "omitted"},
    ]
    reconstructed_by_parameter = {row["parameter"]: row for row in reconstructed}
    request_identity_records: list[dict[str, object]] = []
    for endpoint in ("ob_pair", "ob_agg"):
        for timeframe in ("1h", "12h", "1d"):
            covered = timeframe in {"1h", "12h"}
            request_identity_records.append(
                {
                    "endpoint": endpoint,
                    "timeframe": timeframe,
                    "code_registry_path": reconstructed_by_parameter["path"][endpoint],
                    "code_default_exchange_scope": (
                        reconstructed_by_parameter["exchange"][endpoint]
                        if endpoint == "ob_pair"
                        else reconstructed_by_parameter["exchange_list"][endpoint]
                    ),
                    "code_default_range_percent": reconstructed_by_parameter["range"][endpoint],
                    "parsed_history_proves": ["path", "symbol", "interval"],
                    "parsed_history_does_not_prove": [
                        "exchange or exchange_list", "range", "exact request timestamps"
                    ],
                    "immutable_receipt_available": covered,
                    "immutable_receipt_proves": (
                        [
                            "source_id", "request/response timestamps", "payload SHA",
                            "response bytes",
                        ]
                        if covered else []
                    ),
                    "immutable_receipt_does_not_prove_directly": [
                        "exchange or exchange_list", "range", "full request parameters"
                    ],
                    "historical_code_and_invocation_recovery": (
                        "C: retained transcript and frozen endpoint registry support the "
                        "listed query identity, but the full runner matching its frozen "
                        "SHA is absent"
                        if covered else "not covered by the r6 immutable receipt set"
                    ),
                    "supported_historical_exchange_scope": (
                        reconstructed_by_parameter["exchange"][endpoint]
                        if endpoint == "ob_pair" and covered
                        else reconstructed_by_parameter["exchange_list"][endpoint]
                        if covered else "unknown"
                    ),
                    "supported_historical_range_percent": (
                        reconstructed_by_parameter["range"][endpoint]
                        if covered else "unknown"
                    ),
                    "identity_comparability": (
                        "supported reconstruction at C evidence level; receipt facts are A "
                        "and actual request times differ"
                        if covered else "unknown for immutable 1d"
                    ),
                }
            )
    return {
        "evidence_grades": {
            "A": "directly recorded in immutable receipt",
            "B": "uniquely fixed by a complete frozen implementation and inputs",
            "C": "supported reconstruction but not fully proven",
            "D": "not recoverable from retained evidence",
        },
        "original_request_freeze": "v2_20260729_r6",
        "final_replay_freeze": "v2_20260729_r12",
        "runtime_contract_version": r6_manifest["runtime_contract_version"],
        "original_runner_content_sha256": r6_sources[
            "qlab_research_private/research/crypto/live/ksv4_true_oos_shadow.py"
        ],
        "exact_runner_git_commit": None,
        "runner_source_status": "D: full source matching the frozen SHA is not retained in Git",
        "endpoint_registry_sha256": endpoint_sha,
        "endpoint_registry_git_commit": "c40a8084e29433f78e397ae614f67714037d51d5",
        "historical_runner_relevant_excerpt": call_matches[0],
        "historical_runner_limit_default_excerpt": default_matches[0],
        "original_command": commands["v2_20260729_r6"],
        "replay_chain_commands": [commands[freeze] for freeze in required_freezes],
        "parameter_evidence": [
            *({**row, "grade": "C"} for row in reconstructed),
            {
                "parameter": "request/response time, selected source label, payload SHA",
                "ob_pair": "receipt fields", "ob_agg": "receipt fields", "grade": "A",
            },
            {
                "parameter": "exact runner Git commit", "ob_pair": "unrecoverable",
                "ob_agg": "unrecoverable", "grade": "D",
            },
        ],
        "request_identity_records": request_identity_records,
        "r6_r12_receipts_verified": verified_receipt_pairs,
        "r6_r12_payload_objects_sha_verified": verified_objects,
        "request_timing_and_common_label": timing,
        "formal_selected_labels_from_receipts": {
            timeframe: next(iter(labels)) for timeframe, labels in selected_labels.items()
        },
        "native_bar_end_from_receipts": next(iter(native_ends)),
        "important_boundary": (
            "the retained evidence supports, but does not fully prove, that 1h and 12h "
            "query parameters differed only by interval; requests were about 22m44s apart "
            "and formally selected labels differ"
        ),
        "inputs": [
            {"path": str(r6_source_manifest), "sha256": file_sha256(r6_source_manifest)},
            {"path": str(r12_source_manifest), "sha256": file_sha256(r12_source_manifest)},
            {"path": str(session_transcript), "sha256": file_sha256(session_transcript)},
        ],
    }


def summarize_orderbook_values(records: pd.DataFrame) -> list[dict[str, object]]:
    if records.empty:
        return []
    grouped = records.groupby(
        ["endpoint", "coarse_timeframe", "method", "metric"], sort=True
    )
    return [
        {
            "endpoint": key[0], "coarse_timeframe": key[1], "method": key[2],
            "metric": key[3], "rows": len(group),
            "symbols": int(group.symbol.nunique()), "labels": int(group.label.nunique()),
            "exact": int(group.exact_equal.sum()), "exact_rate": float(group.exact_equal.mean()),
            "median_absolute_difference": float(group.absolute_difference.abs().median()),
            "max_absolute_difference": float(group.absolute_difference.abs().max()),
        }
        for key, group in grouped
    ]


def summarize_orderbook_ranks(records: pd.DataFrame) -> list[dict[str, object]]:
    if records.empty:
        return []
    grouped = records.groupby(["endpoint", "coarse_timeframe", "method"], sort=True)
    return [
        {
            "endpoint": key[0], "coarse_timeframe": key[1], "method": key[2],
            "rows": len(group), "complete_labels": int(group.label.nunique()),
            "cross_section_size": int(group.cross_section_size.min()),
            "rank_exact": int(group.rank_exact_equal.sum()),
            "rank_exact_rate": float(group.rank_exact_equal.mean()),
            "max_absolute_standardized_rank_difference": float(
                group.standardized_rank_difference.abs().max()
            ),
        }
        for key, group in grouped
    ]


def summarize_orderbook_values_by_symbol(
    records: pd.DataFrame,
) -> list[dict[str, object]]:
    selected = records[records.method.eq("same_label")]
    grouped = selected.groupby(
        ["endpoint", "coarse_timeframe", "symbol", "metric"], sort=True
    )
    return [
        {
            "endpoint": key[0], "coarse_timeframe": key[1], "symbol": key[2],
            "metric": key[3], "rows": len(group),
            "exact": int(group.exact_equal.sum()),
            "exact_rate": float(group.exact_equal.mean()),
            "max_absolute_difference": float(group.absolute_difference.abs().max()),
        }
        for key, group in grouped
    ]


__all__ = [
    "AGGREGATION_METHODS", "ORDERBOOK_RAW_FIELDS", "ORDERBOOK_IMBALANCE_FIELDS",
    "audit_orderbook_timeframe_relationship", "one_hour_hypotheses",
    "orderbook_imbalance", "compare_cache_to_raw_history", "file_sha256",
    "build_minimal_pair_probe_contract", "compare_minimal_pair_probe",
    "persist_minimal_pair_probe_response",
    "load_immutable_orderbook_frames", "load_orderbook_history_csv",
    "summarize_archived_request_provenance",
    "audit_archived_orderbook_request_provenance",
    "summarize_orderbook_ranks", "summarize_orderbook_values",
    "summarize_orderbook_values_by_symbol",
]
