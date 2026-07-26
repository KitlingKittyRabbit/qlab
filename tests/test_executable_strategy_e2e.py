from __future__ import annotations

import pandas as pd
import pytest

from qlab.factor_research import (
    ComboSpec,
    continuous_membership_quantity_replay,
    evaluate_executable_long_short_strategy,
    live_like_executable_min_notional_replay,
)
from qlab.walkforward import WalkForwardFold


def test_four_symbol_one_fold_executable_strategy_matches_hand_ledger() -> None:
    decisions = pd.date_range("2026-01-01", periods=5, freq="1D", tz="UTC")
    symbols = ["A", "B", "C", "D"]
    scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
    test_returns = {"A": -0.02, "B": -0.01, "C": 0.01, "D": 0.02}
    rows = []
    for decision in decisions:
        for symbol in symbols:
            executable_return = (
                scores[symbol] / 100.0
                if decision <= decisions[1]
                else test_returns[symbol]
            )
            entry = 100.0
            rows.append(
                {
                    "decision_ts": decision,
                    "signal_timeframes": "1d",
                    "native_bar_end_ts": decision,
                    "signal_bar_end_ts": decision,
                    "availability_ts": decision + pd.Timedelta(minutes=1),
                    "data_observed_ts": decision + pd.Timedelta(minutes=1),
                    "decision_interval": "1d",
                    "order_submit_ts": decision + pd.Timedelta(minutes=1),
                    "execution_ts": decision + pd.Timedelta(minutes=1),
                    "execution_open_time": decision + pd.Timedelta(minutes=1),
                    "next_execution_ts": decision + pd.Timedelta(days=1, minutes=1),
                    "return_horizon": "1d",
                    "holding_interval": "1d",
                    "exit_rule": "rebalance_at_next_decision",
                    "score_order": "high_score_long_low_score_short",
                    "symbol": symbol,
                    "signal__1d": scores[symbol],
                    "entry_price": entry,
                    "exit_price": entry * (1.0 + executable_return),
                    "execution_price": entry,
                    "next_execution_price": entry * (1.0 + executable_return),
                    "executable_return": executable_return,
                }
            )
    frame = pd.DataFrame(rows).set_index("decision_ts", drop=False).sort_index()
    fold = WalkForwardFold(
        fold_idx=0,
        train_start=decisions[0],
        train_end=decisions[1],
        test_start=decisions[2],
        test_end=decisions[4],
    )
    spec = ComboSpec(
        combo_id="e2e_1d",
        track="e2e",
        panel_frequency="1d",
        return_horizon="1d",
        feature_names=("signal__1d",),
        weight_scheme="equal",
    )
    summary, detail, holdings = evaluate_executable_long_short_strategy(
        spec,
        frame,
        [fold],
        {"train_days": 2, "test_days": 3, "embargo_days": 0, "step_days": 3},
        weight_scheme="equal",
        feature_families=None,
        decision_frequency="1d",
        n_buckets=2,
        min_cross_section=4,
        frequency_periods_per_year={"1d": 365},
        cost_multipliers=(1.0,),
        taker_fee_rate=0.001,
        horizon_deltas={"1d": pd.Timedelta(days=1)},
        supported_signal_timeframes=("1d",),
    )

    # Gross exposure is 1x: long C,D and short A,B at +/-0.25 each.
    assert detail["gross_return"].tolist() == pytest.approx([0.015, 0.015, 0.015])
    assert detail["charged_turnover"].tolist() == pytest.approx([1.0, 0.0, 1.0])
    assert detail["net_return_1x"].tolist() == pytest.approx([0.014, 0.015, 0.014])
    assert summary["oos_mean_return"] == pytest.approx(0.015)
    assert (1.0 + detail["gross_return"]).prod() - 1.0 == pytest.approx((1.015**3) - 1.0)
    assert holdings["execution_ts"].notna().all()
    assert holdings["return_horizon"].eq("1d").all()
    assert "return_horizon_x" not in holdings
    assert "return_horizon_y" not in holdings
    assert holdings["next_execution_ts"].notna().all()
    assert (
        holdings["next_execution_ts"] - holdings["execution_ts"]
        == pd.Timedelta(days=1)
    ).all()


def test_cross_fold_unchanged_members_keep_quantities_without_reopen() -> None:
    decisions = pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True)
    targets = pd.DataFrame(
        [
            {"combo_id": "demo", "fold_idx": fold, "decision_ts": decision, "symbol": symbol, "leg": leg}
            for fold, decision in enumerate(decisions)
            for symbol, leg in (("A", "long"), ("B", "short"))
        ]
    )
    prices = {
        (decisions[0], "A"): (100.0, 110.0),
        (decisions[0], "B"): (100.0, 90.0),
        (decisions[1], "A"): (110.0, 121.0),
        (decisions[1], "B"): (90.0, 81.0),
    }
    ledger = pd.DataFrame(
        [
            {
                "decision_ts": decision,
                "symbol": symbol,
                "execution_ts": decision + pd.Timedelta(minutes=1),
                "next_execution_ts": decision + pd.Timedelta(days=1, minutes=1),
                "entry_price": entry,
                "exit_price": exit_price,
                "executable_return": exit_price / entry - 1.0,
            }
            for (decision, symbol), (entry, exit_price) in prices.items()
        ]
    )

    detail, orders, holdings = continuous_membership_quantity_replay(
        targets,
        ledger,
        target_gross_notional=100.0,
        taker_fee_rate=0.001,
        cost_multipliers=(1.0,),
    )

    middle = orders.loc[
        (orders["decision_ts"] == decisions[1]) & (orders["status"] != "terminal_close")
    ]
    assert middle["status"].tolist() == ["hold_unchanged", "hold_unchanged"]
    assert middle["executed_quantity"].tolist() == pytest.approx([0.0, 0.0])
    by_symbol = holdings.pivot(index="decision_ts", columns="symbol", values="signed_quantity")
    assert by_symbol.loc[decisions[0]].tolist() == pytest.approx([0.5, -0.5])
    assert by_symbol.loc[decisions[1]].tolist() == pytest.approx([0.5, -0.5])
    assert detail["actual_gross_notional"].tolist() == pytest.approx([100.0, 100.0])
    assert detail["actual_net_notional"].tolist() == pytest.approx([0.0, 10.0])
    assert detail["rebalance_turnover"].tolist() == pytest.approx([1.0, 0.0])
    assert detail["terminal_close_turnover"].tolist() == pytest.approx([0.0, 1.01])


def test_exchange_rules_use_historical_price_and_round_quantity_toward_zero() -> None:
    decision = pd.Timestamp("2026-01-01", tz="UTC")
    targets = pd.DataFrame(
        [
            {"combo_id": "demo", "fold_idx": 0, "decision_ts": decision, "symbol": "A", "leg": "long"},
            {"combo_id": "demo", "fold_idx": 0, "decision_ts": decision, "symbol": "B", "leg": "short"},
        ]
    )
    ledger = pd.DataFrame(
        [
            {
                "decision_ts": decision,
                "symbol": symbol,
                "execution_ts": decision + pd.Timedelta(minutes=1),
                "next_execution_ts": decision + pd.Timedelta(days=1, minutes=1),
                "entry_price": price,
                "exit_price": price,
                "executable_return": 0.0,
            }
            for symbol, price in (("A", 30.0), ("B", 20.0))
        ]
    )
    rules = pd.DataFrame(
        [
            {"symbol": "A", "market_min_qty": 0.1, "market_step": 0.1, "min_notional": 5.0},
            {"symbol": "B", "market_min_qty": 0.1, "market_step": 0.1, "min_notional": 5.0},
        ]
    )

    _, orders, holdings = continuous_membership_quantity_replay(
        targets,
        ledger,
        target_gross_notional=100.0,
        taker_fee_rate=0.0,
        cost_multipliers=(1.0,),
        exchange_rules=rules,
    )

    opened = orders.loc[orders["status"] == "open"].set_index("symbol")
    assert opened.loc["A", "executed_quantity"] == pytest.approx(1.6)
    assert opened.loc["B", "executed_quantity"] == pytest.approx(-2.5)
    quantities = holdings.set_index("symbol")["signed_quantity"].to_dict()
    assert quantities == pytest.approx({"A": 1.6, "B": -2.5})


def test_formal_l4_entry_keeps_one_path_across_folds() -> None:
    decisions = pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True)
    rows = []
    for fold, decision in enumerate(decisions):
        for symbol, leg, price in (("A", "long", 30.0), ("B", "short", 20.0)):
            rows.append(
                {
                    "combo_id": "demo",
                    "track": "all_two_gate",
                    "weight_scheme": "family_alpha_0",
                    "panel_frequency": "1d",
                    "return_horizon": "1d",
                    "component_features": "x|y",
                    "fold_idx": fold,
                    "decision_ts": decision,
                    "symbol": symbol,
                    "leg": leg,
                    "weight": 0.5 if leg == "long" else -0.5,
                    "signal_timeframes": "1d",
                    "native_bar_end_ts": decision,
                    "signal_bar_end_ts": decision,
                    "availability_ts": decision + pd.Timedelta(minutes=1),
                    "data_observed_ts": decision + pd.Timedelta(minutes=1),
                    "decision_interval": "1d",
                    "order_submit_ts": decision + pd.Timedelta(minutes=1),
                    "execution_ts": decision + pd.Timedelta(minutes=1),
                    "execution_open_time": decision + pd.Timedelta(minutes=1),
                    "next_execution_ts": decision + pd.Timedelta(days=1, minutes=1),
                    "holding_interval": "1d",
                    "exit_rule": "rebalance_at_next_decision",
                    "score_order": "high_score_long_low_score_short",
                    "entry_price": price,
                    "exit_price": price,
                    "execution_price": price,
                    "next_execution_price": price,
                    "executable_return": 0.0,
                }
            )
    rules = pd.DataFrame(
        [
            {"symbol": symbol, "market_min_qty": 0.1, "market_step": 0.1, "min_notional": 5.0}
            for symbol in ("A", "B")
        ]
    )

    first_strategy = pd.DataFrame(rows)
    second_strategy = first_strategy.assign(weight_scheme="family_alpha_1")
    summary, detail, orders, holdings = live_like_executable_min_notional_replay(
        pd.concat([first_strategy, second_strategy], ignore_index=True),
        rules,
        account_equity=100.0,
        target_gross_notional=100.0,
        exchange_leverage=5.0,
        taker_fee_rate=0.001,
        cost_multipliers=(1.0,),
        frequency_periods_per_year={"1d": 365},
        horizon_deltas={"1d": pd.Timedelta(days=1)},
    )

    assert len(summary) == 2
    lineage_columns = {
        "signal_timeframes", "native_bar_end_ts", "signal_bar_end_ts",
        "availability_ts", "data_observed_ts", "decision_interval",
        "order_submit_ts", "execution_ts", "execution_open_time",
        "next_execution_ts", "holding_interval", "exit_rule", "score_order",
    }
    for output in (detail, orders, holdings):
        assert lineage_columns.issubset(output.columns)
        assert output[list(lineage_columns)].notna().all().all()
    terminal_orders = orders.loc[orders["status"].eq("terminal_close")]
    assert (
        pd.to_datetime(terminal_orders["execution_open_time"], utc=True)
        == pd.to_datetime(terminal_orders["execution_ts"], utc=True)
    ).all()
    demo_detail = detail.loc[
        detail["combo_id"].eq("demo")
        & detail["weight_scheme"].eq("family_alpha_0")
    ]
    assert demo_detail["rebalance_turnover"].tolist() == pytest.approx([0.98, 0.0])
    assert demo_detail["terminal_close_turnover"].tolist() == pytest.approx([0.0, 0.98])
    assert orders["status"].eq("terminal_close").sum() == 4
    demo_holdings = holdings.loc[
        holdings["combo_id"].eq("demo")
        & holdings["weight_scheme"].eq("family_alpha_0")
    ]
    assert demo_holdings.groupby("decision_ts")["signed_quantity"].apply(list).tolist() == [
        pytest.approx([1.6, -2.5]),
        pytest.approx([1.6, -2.5]),
    ]
