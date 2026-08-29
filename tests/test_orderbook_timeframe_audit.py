from __future__ import annotations

import json

import pandas as pd
import pytest

from qlab.data.crypto.keystore_coinglass_client import RawKeystoreDiagnosticResponse
from qlab.data.crypto.orderbook_timeframe_audit import (
    audit_orderbook_timeframe_relationship,
    build_minimal_pair_probe_contract,
    compare_minimal_pair_probe,
    compare_cache_to_raw_history,
    one_hour_hypotheses,
    persist_minimal_pair_probe_response,
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
