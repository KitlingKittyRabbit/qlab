from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from qlab.data.crypto.keystore_coinglass_client import RawKeystoreDiagnosticResponse
from qlab.data.crypto.orderbook_timeframe_audit import (
    audit_archived_orderbook_request_provenance,
    audit_orderbook_timeframe_relationship,
    build_minimal_pair_probe_contract,
    compare_minimal_pair_probe,
    compare_cache_to_raw_history,
    one_hour_hypotheses,
    persist_minimal_pair_probe_response,
    summarize_archived_request_provenance,
)


def _frame(values):
    index = pd.date_range("2026-01-01T01:00:00Z", periods=len(values), freq="1h", name="ts")
    rows = [(bid, bid / 10, ask, ask / 10) for bid, ask in values]
    return pd.DataFrame(
        rows, index=index,
        columns=["bids_usd", "bids_quantity", "asks_usd", "asks_quantity"],
    )


def test_hypotheses_are_hand_calculable_and_window_is_explicit():
    source = _frame([(1, 3), (2, 4), (3, 5), (4, 6)])
    label = source.index[-1]
    result = one_hour_hypotheses(
        source, [label], endpoint="ob_pair", coarse_timeframe="12h"
    )
    assert result["same_label"].loc[label].tolist() == [4.0, 0.4, 6.0, 0.6]
    assert result["first"].loc[label].tolist() == [1.0, 0.1, 3.0, 0.3]
    assert result["last"].loc[label].tolist() == [4.0, 0.4, 6.0, 0.6]
    assert result["mean"].loc[label].tolist() == pytest.approx([2.5, 0.25, 4.5, 0.45])
    assert result["median"].loc[label].tolist() == pytest.approx([2.5, 0.25, 4.5, 0.45])
    assert result["min"].loc[label].tolist() == [1.0, 0.1, 3.0, 0.3]
    assert result["max"].loc[label].tolist() == [4.0, 0.4, 6.0, 0.6]


def test_two_coin_raw_imbalance_and_frozen_rank_are_exact():
    btc_1h = _frame([(1, 3), (2, 2)])
    eth_1h = _frame([(3, 1), (3, 1)])
    label = btc_1h.index[-1]
    btc_coarse = _frame([(2, 2)]).set_axis(pd.DatetimeIndex([label], name="ts"))
    eth_coarse = _frame([(3, 1)]).set_axis(pd.DatetimeIndex([label], name="ts"))
    values, ranks = audit_orderbook_timeframe_relationship(
        {"BTC": btc_1h, "ETH": eth_1h}, {"BTC": btc_coarse, "ETH": eth_coarse},
        endpoint="ob_pair", coarse_timeframe="12h", expected_symbols=("BTC", "ETH"),
    )
    same = values[values.method.eq("same_label")]
    assert same.exact_equal.all()
    assert set(same.metric) == {"bids_usd", "bids_quantity", "asks_usd", "asks_quantity", "imbalance"}
    same_ranks = ranks[ranks.method.eq("same_label")].set_index("symbol")
    assert same_ranks.loc["BTC", "coarse_rank"] == -1.0
    assert same_ranks.loc["ETH", "coarse_rank"] == 1.0
    assert same_ranks.rank_exact_equal.all()


def test_missing_symbol_fails_instead_of_reranking_remainder():
    source = _frame([(1, 2)])
    with pytest.raises(ValueError, match="exactly the expected symbol set"):
        audit_orderbook_timeframe_relationship(
            {"BTC": source}, {"BTC": source}, endpoint="ob_pair",
            coarse_timeframe="12h", expected_symbols=("BTC", "ETH"),
        )


def test_cache_comparison_requires_exact_index_and_values():
    frame = _frame([(1, 2), (3, 4)])
    identity = ("ob_pair", "1h", "BTC")
    result = compare_cache_to_raw_history({identity: frame}, {identity: frame.copy()})
    assert result.iloc[0][["index_equal", "values_equal_on_cache_labels"]].tolist() == [True, True]
    changed = frame.copy()
    changed.iloc[0, 0] = 99
    result = compare_cache_to_raw_history({identity: frame}, {identity: changed})
    assert bool(result.iloc[0].values_equal_on_cache_labels) is False


def test_raw_factor_and_rank_use_one_complete_symbol_label_support():
    btc_1h = _frame([(1, 3), (2, 2)])
    eth_1h = _frame([(3, 1), (3, 1)])
    labels = btc_1h.index
    btc_coarse = btc_1h.copy()
    eth_coarse = eth_1h.iloc[[1]].copy()
    values, ranks = audit_orderbook_timeframe_relationship(
        {"BTC": btc_1h, "ETH": eth_1h}, {"BTC": btc_coarse, "ETH": eth_coarse},
        endpoint="ob_pair", coarse_timeframe="12h", expected_symbols=("BTC", "ETH"),
    )
    same_values = values[values.method.eq("same_label")]
    same_ranks = ranks[ranks.method.eq("same_label")]
    assert set(same_values.label) == {labels[1]}
    assert set(same_ranks.label) == {labels[1]}
    assert same_values.groupby("label").symbol.nunique().eq(2).all()
    assert same_ranks.groupby("label").symbol.nunique().eq(2).all()


def _diagnostic_response(record, rows, *, status=200, code="0"):
    payload = json.dumps({"code": code, "data": rows}, separators=(",", ":")).encode()
    return RawKeystoreDiagnosticResponse(
        path=record["path"], request_params=record["params"],
        request_ts="2026-08-29T00:00:00+00:00",
        response_ts="2026-08-29T00:00:01+00:00",
        http_status=status, business_code=code,
        business_message="" if status == 200 else "rejected", raw_payload=payload,
    )


def test_minimal_pair_contract_and_exact_target_comparison(tmp_path):
    target = 1787788800000
    contract = build_minimal_pair_probe_contract(
        target_label_ms=target,
        start_time_ms=target - 86_400_000,
        end_time_ms=target + 86_400_000,
    )
    assert [row["timeframe"] for row in contract] == ["1h", "12h", "1d"]
    common = [
        {key: value for key, value in row["params"].items() if key != "interval"}
        for row in contract
    ]
    assert common[0] == common[1] == common[2]
    assert common[0] == {
        "exchange": "Binance", "symbol": "BTCUSDT", "range": "1",
        "limit": 1000, "start_time": target - 86_400_000,
        "end_time": target + 86_400_000,
    }
    row = {
        "time": target, "bids_usd": 30, "bids_quantity": 3,
        "asks_usd": 10, "asks_quantity": 1,
    }
    for record in contract:
        receipt = persist_minimal_pair_probe_response(
            tmp_path, record, _diagnostic_response(record, [row])
        )
        assert receipt["authentication_recorded"] is False
    result = compare_minimal_pair_probe(tmp_path, contract)
    assert result["same_request_identity_except_interval"] is True
    assert result["all_three_comparable"] is True
    assert result["all_raw_and_imbalance_equal"] is True
    assert result["field_equal_across_all_three"] == {
        "bids_usd": True, "bids_quantity": True,
        "asks_usd": True, "asks_quantity": True, "imbalance": True,
    }
    assert "fake-secret" not in "".join(
        path.read_text(encoding="utf-8") for path in (tmp_path / "receipts").glob("*.json")
    )


def test_non_json_http_rejection_still_has_complete_receipt(tmp_path):
    target = 1787788800000
    record = build_minimal_pair_probe_contract(
        target_label_ms=target,
        start_time_ms=target - 86_400_000,
        end_time_ms=target + 86_400_000,
    )[0]
    response = RawKeystoreDiagnosticResponse(
        path=record["path"], request_params=record["params"],
        request_ts="2026-08-29T00:00:00+00:00",
        response_ts="2026-08-29T00:00:01+00:00",
        http_status=502, business_code="", business_message="",
        raw_payload=b"upstream unavailable",
    )
    receipt = persist_minimal_pair_probe_response(tmp_path, record, response)
    assert receipt["http_status"] == 502
    assert receipt["payload_json_parseable"] is False
    assert receipt["exact_target_row_count"] == 0
    object_path = (
        tmp_path / "objects" / receipt["payload_sha256"][:2]
        / f"{receipt['payload_sha256']}.bin"
    )
    assert object_path.read_bytes() == b"upstream unavailable"


@pytest.mark.parametrize("mode", ["rejected", "missing_target"])
def test_minimal_pair_probe_fails_closed_without_exact_comparison(tmp_path, mode):
    target = 1787788800000
    contract = build_minimal_pair_probe_contract(
        target_label_ms=target,
        start_time_ms=target - 86_400_000,
        end_time_ms=target + 86_400_000,
    )
    row = {
        "time": target, "bids_usd": 30, "bids_quantity": 3,
        "asks_usd": 10, "asks_quantity": 1,
    }
    for record in contract:
        rows = [row]
        status, code = 200, "0"
        if record["timeframe"] == "12h" and mode == "rejected":
            status, code = 403, "40001"
        if record["timeframe"] == "1d" and mode == "missing_target":
            rows = [{**row, "time": target - 3_600_000}]
        persist_minimal_pair_probe_response(
            tmp_path, record,
            _diagnostic_response(record, rows, status=status, code=code),
        )
    result = compare_minimal_pair_probe(tmp_path, contract)
    assert result["all_three_comparable"] is False
    assert result["all_raw_and_imbalance_equal"] is None


def test_archived_request_provenance_uses_receipt_times_and_verified_bytes(tmp_path):
    symbols = ("BTC", "ETH")
    for endpoint in ("ob_pair", "ob_agg"):
        fields = (
            ("bids_usd", "bids_quantity", "asks_usd", "asks_quantity")
            if endpoint == "ob_pair"
            else (
                "aggregated_bids_usd", "aggregated_bids_quantity",
                "aggregated_asks_usd", "aggregated_asks_quantity",
            )
        )
        for timeframe, request_minute in (("12h", 0), ("1h", 22)):
            for offset, symbol in enumerate(symbols):
                values = [10.0 + offset, 1.0, 20.0, 2.0]
                if timeframe == "1h":
                    values[0] += 1.0
                row = {"time": 1785283200000, **dict(zip(fields, values))}
                raw = json.dumps({"code": "0", "data": [row]}).encode()
                sha = __import__("hashlib").sha256(raw).hexdigest()
                object_path = tmp_path / "objects" / sha[:2] / f"{sha}.bin"
                object_path.parent.mkdir(parents=True, exist_ok=True)
                object_path.write_bytes(raw)
                receipt_dir = (
                    tmp_path / "receipts"
                    / f"keystore_v4_{endpoint}_{timeframe}_{symbol}"
                )
                receipt_dir.mkdir(parents=True, exist_ok=True)
                receipt = {
                    "payload_sha256": sha,
                    "source_request_ts": (
                        pd.Timestamp("2026-07-29T01:00:00Z")
                        + pd.Timedelta(minutes=request_minute, seconds=offset)
                    ).isoformat(),
                }
                (receipt_dir / "receipt.json").write_text(
                    json.dumps(receipt), encoding="utf-8"
                )
    result = summarize_archived_request_provenance(
        tmp_path, expected_symbols=symbols
    )
    assert len(result) == 2
    assert {row["endpoint"] for row in result} == {"ob_pair", "ob_agg"}
    assert all(row["symbols"] == 2 for row in result)
    assert all(
        row["one_hour_minus_twelve_hour_request_lag_seconds_median"] == 1320.0
        for row in result
    )
    assert all(
        row["symbols_with_different_raw_fields_at_latest_common_label"] == 2
        for row in result
    )


def _write_archived_provenance_fixture(tmp_path):
    symbols = ("BTC", "ETH")
    endpoint_registry = Path(__import__(
        "qlab.data.crypto.keystore_coinglass_endpoints", fromlist=["x"]
    ).__file__)
    endpoint_sha = __import__("hashlib").sha256(endpoint_registry.read_bytes()).hexdigest()
    source_manifest = {
        "runtime_contract_version": "test-contract",
        "sources": [
            {
                "path": "qlab/qlab/data/crypto/keystore_coinglass_endpoints.py",
                "sha256": endpoint_sha,
            },
            {
                "path": "qlab_research_private/research/crypto/live/ksv4_true_oos_shadow.py",
                "sha256": "frozen-runner-content-sha",
            },
        ],
    }
    manifests = []
    for freeze in ("r6", "r12"):
        path = tmp_path / f"{freeze}_source_manifest.json"
        path.write_text(json.dumps(source_manifest), encoding="utf-8")
        manifests.append(path)

    smoke_roots = []
    for freeze in ("r6", "r12"):
        root = tmp_path / freeze / "preflight" / "smoke"
        smoke_roots.append(root)
        for endpoint in ("ob_pair", "ob_agg"):
            fields = (
                ("bids_usd", "bids_quantity", "asks_usd", "asks_quantity")
                if endpoint == "ob_pair"
                else (
                    "aggregated_bids_usd", "aggregated_bids_quantity",
                    "aggregated_asks_usd", "aggregated_asks_quantity",
                )
            )
            for timeframe, request_minute, selected_label in (
                ("12h", 0, "2026-07-28T12:00:00+00:00"),
                ("1h", 22, "2026-07-28T23:00:00+00:00"),
            ):
                for offset, symbol in enumerate(symbols):
                    value = 10.0 + offset + (1.0 if timeframe == "1h" else 0.0)
                    row = {
                        "time": 1785283200000,
                        **dict(zip(fields, (value, 1.0, 20.0, 2.0))),
                    }
                    raw = json.dumps({"code": "0", "data": [row]}).encode()
                    sha = __import__("hashlib").sha256(raw).hexdigest()
                    object_path = root / "as_received" / "objects" / sha[:2] / f"{sha}.bin"
                    object_path.parent.mkdir(parents=True, exist_ok=True)
                    object_path.write_bytes(raw)
                    source_id = f"keystore_v4_{endpoint}_{timeframe}_{symbol}"
                    receipt_dir = root / "as_received" / "receipts" / source_id
                    receipt_dir.mkdir(parents=True, exist_ok=True)
                    request_ts = (
                        pd.Timestamp("2026-07-29T01:00:00Z")
                        + pd.Timedelta(minutes=request_minute, seconds=offset)
                    ).isoformat()
                    receipt = {
                        "source_id": source_id,
                        "source_request_ts": request_ts,
                        "source_response_ts": request_ts,
                        "source_bar_label_ts": selected_label,
                        "native_bar_end_ts": "2026-07-29T00:00:00+00:00",
                        "data_observed_ts": request_ts,
                        "payload_sha256": sha,
                    }
                    (receipt_dir / "receipt.json").write_text(
                        json.dumps(receipt), encoding="utf-8"
                    )

    commands = []
    for index in range(6, 13):
        resume = "" if index == 6 else " --resume-root /evidence/v2_20260729_r6/preflight/real_signal_smoke_20260729T024640Z"
        commands.append(
            "python qlab_research_private/research/crypto/live/ksv4_true_oos_shadow.py "
            f"real-signal-smoke --freeze-version v2_20260729_r{index} "
            f"--decision-ts 2026-07-29T00:00:00Z{resume} --output /evidence/r{index}"
        )
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "commands": commands,
                "source": (
                    "params = build_history_params(\n"
                    "    endpoint, symbol=symbol, interval=timeframe, limit=args.limit,\n"
                    ")\n"
                    "observed = client.request_raw(endpoint.path, params=params)\n"
                    "real_signal_smoke.add_argument(\"--limit\", type=int, default=3)"
                ),
            }
        ) + "\n",
        encoding="utf-8",
    )
    return symbols, smoke_roots, manifests, transcript


def test_formal_archived_provenance_preserves_commands_and_grades_query_as_c(tmp_path):
    symbols, roots, manifests, transcript = _write_archived_provenance_fixture(tmp_path)
    result = audit_archived_orderbook_request_provenance(
        r6_smoke_root=roots[0], r12_smoke_root=roots[1],
        r6_source_manifest=manifests[0], r12_source_manifest=manifests[1],
        session_transcript=transcript, expected_symbols=symbols,
    )
    assert result["original_command"].endswith("--output /evidence/r6")
    assert result["replay_chain_commands"][-1].endswith("--output /evidence/r12")
    assert "/preflight/real_signal_smoke_20260729T024640Z" in result["replay_chain_commands"][1]
    assert {row["grade"] for row in result["parameter_evidence"] if row["parameter"] == "limit"} == {"C"}
    assert result["exact_runner_git_commit"] is None
    assert result["r6_r12_receipts_verified"] == 8
    assert result["r6_r12_payload_objects_sha_verified"] == 16
    identity = {
        (row["endpoint"], row["timeframe"]): row
        for row in result["request_identity_records"]
    }
    assert identity[("ob_pair", "1h")]["supported_historical_exchange_scope"] == "Binance"
    assert identity[("ob_agg", "12h")]["supported_historical_exchange_scope"] == "Binance,OKX,Bybit"
    assert identity[("ob_pair", "1d")]["supported_historical_exchange_scope"] == "unknown"
    assert all(
        row["historical_code_and_invocation_recovery"].startswith("C:")
        for row in result["request_identity_records"]
        if row["timeframe"] in {"1h", "12h"}
    )


def test_formal_archived_provenance_fails_when_either_freeze_object_is_tampered(tmp_path):
    symbols, roots, manifests, transcript = _write_archived_provenance_fixture(tmp_path)
    object_path = next((roots[0] / "as_received" / "objects").glob("*/*.bin"))
    object_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="payload SHA mismatch"):
        audit_archived_orderbook_request_provenance(
            r6_smoke_root=roots[0], r12_smoke_root=roots[1],
            r6_source_manifest=manifests[0], r12_source_manifest=manifests[1],
            session_transcript=transcript, expected_symbols=symbols,
        )
