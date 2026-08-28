from __future__ import annotations

import pandas as pd
import pytest

from qlab.data.crypto.orderbook_timeframe_audit import (
    audit_orderbook_timeframe_relationship,
    compare_cache_to_raw_history,
    one_hour_hypotheses,
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
