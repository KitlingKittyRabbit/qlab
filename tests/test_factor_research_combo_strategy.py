import pandas as pd
import pytest

from qlab import factor_research
from qlab.walkforward import WalkForwardFold


def test_family_equal_feature_weights_split_by_family():
    weights = factor_research.family_equal_feature_weights(
        ("a", "b", "c"),
        {"a": "one", "b": "one", "c": "two"},
    )

    assert weights == pytest.approx({"a": 0.25, "b": 0.25, "c": 0.5})


def test_family_count_feature_weights_alpha_ladder_hand_computed():
    feature_names = ("a1", "a2", "a3", "a4", "b", "c")
    feature_families = {
        "a1": "A",
        "a2": "A",
        "a3": "A",
        "a4": "A",
        "b": "B",
        "c": "C",
    }

    alpha_1 = factor_research.family_count_feature_weights(
        feature_names,
        feature_families,
        alpha=1.0,
    )
    alpha_0p5 = factor_research.family_count_feature_weights(
        feature_names,
        feature_families,
        alpha=0.5,
    )
    alpha_0 = factor_research.family_count_feature_weights(
        feature_names,
        feature_families,
        alpha=0.0,
    )

    assert sum(alpha_1.values()) == pytest.approx(1.0)
    assert sum(alpha_0p5.values()) == pytest.approx(1.0)
    assert sum(alpha_0.values()) == pytest.approx(1.0)
    assert alpha_1 == pytest.approx(
        {feature_name: 1.0 / 6.0 for feature_name in feature_names}
    )
    assert sum(alpha_0p5[feature_name] for feature_name in ("a1", "a2", "a3", "a4")) == pytest.approx(
        2.0 / (2.0 + 1.0 + 1.0)
    )
    assert alpha_0p5["a1"] == pytest.approx(0.5 / 4.0)
    assert alpha_0p5["b"] == pytest.approx(0.25)
    assert alpha_0["a1"] == pytest.approx((1.0 / 3.0) / 4.0)
    assert alpha_0["b"] == pytest.approx(1.0 / 3.0)


def test_family_alpha_basket_comparison_uses_same_basket_rows():
    summary = pd.DataFrame(
        [
            {
                "return_horizon": "1d",
                "track": "all_two_gate",
                "panel_frequency": "1d",
                "component_features": "a|b|c",
                "n_components": 3,
                "weight_scheme": "family_alpha_1",
                "net_1x_sharpe": 1.2,
            },
            {
                "return_horizon": "1d",
                "track": "all_two_gate",
                "panel_frequency": "1d",
                "component_features": "a|b|c",
                "n_components": 3,
                "weight_scheme": "family_alpha_0p5",
                "net_1x_sharpe": 1.4,
            },
            {
                "return_horizon": "1d",
                "track": "all_two_gate",
                "panel_frequency": "1d",
                "component_features": "a|b|c",
                "n_components": 3,
                "weight_scheme": "family_alpha_0",
                "net_1x_sharpe": 1.0,
            },
            {
                "return_horizon": "12h",
                "track": "without_funding",
                "panel_frequency": "12h",
                "component_features": "d|e",
                "n_components": 2,
                "weight_scheme": "family_alpha_1",
                "net_1x_sharpe": -0.1,
            },
            {
                "return_horizon": "12h",
                "track": "without_funding",
                "panel_frequency": "12h",
                "component_features": "d|e",
                "n_components": 2,
                "weight_scheme": "family_alpha_0p5",
                "net_1x_sharpe": 0.1,
            },
            {
                "return_horizon": "12h",
                "track": "without_funding",
                "panel_frequency": "12h",
                "component_features": "d|e",
                "n_components": 2,
                "weight_scheme": "family_alpha_0",
                "net_1x_sharpe": 0.2,
            },
        ]
    )

    comparison = factor_research.family_alpha_basket_comparison(summary)

    assert len(comparison) == 2
    first = comparison.loc[comparison["return_horizon"] == "1d"].iloc[0]
    second = comparison.loc[comparison["return_horizon"] == "12h"].iloc[0]
    assert first["winner_weight_scheme"] == "family_alpha_0p5"
    assert first["family_alpha_1_minus_family_alpha_0"] == pytest.approx(0.2)
    assert first["family_alpha_0p5_minus_family_alpha_0"] == pytest.approx(0.4)
    assert second["winner_weight_scheme"] == "family_alpha_0"
    assert second["family_alpha_1_minus_family_alpha_0"] == pytest.approx(-0.3)


def test_select_family_alpha_l4_targets_keeps_all_alpha_rows_and_reasons():
    rows = []
    specs = {
        ("1d", "basket_a", "a|b"): {"family_alpha_1": 3.0, "family_alpha_0p5": 2.0, "family_alpha_0": 1.0},
        ("1d", "basket_b", "c|d"): {"family_alpha_1": 2.0, "family_alpha_0p5": 5.0, "family_alpha_0": 1.0},
        ("1d", "basket_c", "e|f"): {"family_alpha_1": 4.0, "family_alpha_0p5": -10.0, "family_alpha_0": 3.0},
        ("12h", "basket_d", "g|h"): {"family_alpha_1": 1.0, "family_alpha_0p5": 0.0, "family_alpha_0": -1.0},
        ("12h", "basket_e", "i|j"): {"family_alpha_1": 0.5, "family_alpha_0p5": 0.4, "family_alpha_0": 0.3},
    }
    for (horizon, track, features), metrics in specs.items():
        for weight_scheme, metric in metrics.items():
            rows.append(
                {
                    "combo_id": f"{horizon}_{track}_{weight_scheme}",
                    "return_horizon": horizon,
                    "track": track,
                    "panel_frequency": horizon,
                    "component_features": features,
                    "n_components": 2,
                    "weight_scheme": weight_scheme,
                    "net_1x_sharpe": metric,
                }
            )
    summary = pd.DataFrame(rows)

    manifest = factor_research.select_family_alpha_l4_targets(
        summary,
        horizon_quotas={"1d": 2, "12h": 1},
    )

    assert manifest["target_basket_id"].nunique() == 3
    assert manifest.groupby("target_basket_id")["weight_scheme"].nunique().tolist() == [3, 3, 3]
    assert set(manifest.loc[manifest["return_horizon"] == "1d", "track"]) == {"basket_a", "basket_b"}
    assert set(manifest.loc[manifest["return_horizon"] == "12h", "track"]) == {"basket_d"}
    reasons = manifest.drop_duplicates("target_basket_id").set_index("track")["selection_reason"].to_dict()
    assert "best_winner_metric" in reasons["basket_b"]
    assert "best_alpha0p5_edge" in reasons["basket_b"]
    assert reasons["basket_a"] == "best_alpha1_edge"
    assert "basket_c" not in reasons


def test_compare_l3_l4_family_alpha_replay_flags_replication_and_material_impact():
    l3_rows = []
    l4_rows = []
    l3_metrics = {
        ("1d", "replicated", "a|b"): {"family_alpha_1": 3.0, "family_alpha_0p5": 2.0, "family_alpha_0": 1.0},
        ("12h", "reversed", "c|d"): {"family_alpha_1": 3.0, "family_alpha_0p5": 1.0, "family_alpha_0": 2.0},
    }
    l4_metrics = {
        ("1d", "replicated", "a|b"): {"family_alpha_1": 4.0, "family_alpha_0p5": 3.0, "family_alpha_0": 2.0},
        ("12h", "reversed", "c|d"): {"family_alpha_1": 1.0, "family_alpha_0p5": 4.0, "family_alpha_0": 2.0},
    }
    for (horizon, track, features), metrics in l3_metrics.items():
        for weight_scheme, metric in metrics.items():
            l3_rows.append(
                {
                    "combo_id": f"l3_{horizon}_{track}_{weight_scheme}",
                    "return_horizon": horizon,
                    "track": track,
                    "panel_frequency": horizon,
                    "component_features": features,
                    "n_components": 2,
                    "weight_scheme": weight_scheme,
                    "net_1x_sharpe": metric,
                }
            )
    for (horizon, track, features), metrics in l4_metrics.items():
        for weight_scheme, metric in metrics.items():
            l4_rows.append(
                {
                    "combo_id": f"l4_{horizon}_{track}_{weight_scheme}",
                    "return_horizon": horizon,
                    "track": track,
                    "panel_frequency": horizon,
                    "component_features": features,
                    "n_components": 2,
                    "weight_scheme": weight_scheme,
                    "net_1x_sharpe_on_equity": metric,
                    "filtered_order_share": 0.02 if track == "reversed" and weight_scheme == "family_alpha_1" else 0.0,
                    "mean_actual_vs_target_gross_ratio": 1.0,
                    "mean_weight_abs_error_sum": 0.0,
                    "max_abs_net_exposure_share": 0.0,
                    "max_margin_utilization": 0.4,
                }
            )

    comparison = factor_research.compare_l3_l4_family_alpha_replay(
        pd.DataFrame(l3_rows),
        pd.DataFrame(l4_rows),
    )

    replicated = comparison.loc[comparison["track"] == "replicated"].iloc[0]
    reversed_row = comparison.loc[comparison["track"] == "reversed"].iloc[0]
    assert bool(replicated["winner_replicated"]) is True
    assert bool(replicated["alpha1_edge_sign_replicated"]) is True
    assert bool(replicated["alpha0p5_edge_sign_replicated"]) is True
    assert bool(replicated["l4_min_notional_material_impact"]) is False
    assert bool(reversed_row["winner_replicated"]) is False
    assert bool(reversed_row["alpha1_edge_sign_replicated"]) is False
    assert bool(reversed_row["alpha0p5_edge_sign_replicated"]) is False
    assert bool(reversed_row["l4_min_notional_material_impact"]) is True


def test_assigned_bucket_membership_sorts_by_signal_and_symbol():
    frame = pd.DataFrame(
        {
            "symbol": ["B", "A", "C", "D"],
            "signal": [1.0, 1.0, -1.0, 0.0],
            "forward_return": [0.01, 0.02, -0.01, 0.0],
        },
        index=pd.DatetimeIndex(["2026-01-01"] * 4, name="decision_ts"),
    )

    assigned, diagnostic = factor_research.assigned_bucket_membership(
        frame,
        "signal",
        2,
    )

    assert diagnostic["status"] == "ok"
    assert assigned.loc[assigned["bucket"] == 1, "symbol"].tolist() == ["C", "D"]
    assert assigned.loc[assigned["bucket"] == 2, "symbol"].tolist() == ["A", "B"]


def test_top_bottom_diagnostics_uses_extreme_names():
    symbols = list("ABCDEFGH")
    frame = pd.DataFrame(
        {
            "symbol": symbols,
            "signal": list(range(8)),
            "forward_return": [-0.04, -0.03, -0.02, 0.0, 0.0, 0.02, 0.03, 0.04],
        },
        index=pd.DatetimeIndex(["2026-01-01"] * 8, name="decision_ts"),
    )

    detail, diagnostics = factor_research.top_bottom_diagnostics_for_frame(
        frame,
        "signal",
        direction=1,
        leg_count=3,
    )

    assert diagnostics == [
        {
            "decision_ts": pd.Timestamp("2026-01-01"),
            "cross_section_size": 8,
            "status": "ok",
        }
    ]
    by_leg = detail.set_index("leg")
    assert by_leg.loc["short", "leg_size"] == 3
    assert by_leg.loc["long", "leg_size"] == 3
    assert by_leg.loc["short", "leg_return"] == pytest.approx(-0.03)
    assert by_leg.loc["long", "leg_return"] == pytest.approx(0.03)
    assert by_leg.loc["long", "spread_return"] == pytest.approx(0.06)


def test_long_short_strategy_snapshot_charges_turnover_cost():
    spec = factor_research.ComboSpec(
        combo_id="demo",
        track="main",
        panel_frequency="1h",
        return_horizon="4h",
        feature_names=("x",),
        weight_scheme="equal",
    )
    fold = type(
        "Fold",
        (),
        {
            "fold_idx": 0,
            "train_start": pd.Timestamp("2026-01-01", tz="UTC"),
            "train_end": pd.Timestamp("2026-01-10", tz="UTC"),
            "test_start": pd.Timestamp("2026-01-11", tz="UTC"),
            "test_end": pd.Timestamp("2026-01-20", tz="UTC"),
        },
    )()
    assigned = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "bucket": [1, 1, 2, 2],
            "signal_value": [-1.0, -0.5, 0.5, 1.0],
            "forward_return": [-0.01, -0.02, 0.03, 0.04],
            "strategy_forward_return": [-0.01, -0.02, 0.03, 0.04],
        }
    )

    snapshot, holdings = factor_research.long_short_strategy_snapshot(
        combo_spec=spec,
        fold=fold,
        decision_ts=pd.Timestamp("2026-01-11", tz="UTC"),
        assigned=assigned,
        previous_holdings=pd.DataFrame(columns=["symbol", "leg", "weight"]),
        n_buckets=2,
        cost_multipliers=(1.0, 2.0),
        taker_fee_rate=0.0005,
        component_features="x",
    )

    assert snapshot["gross_return"] == pytest.approx(0.025)
    assert snapshot["charged_turnover"] == pytest.approx(1.0)
    assert snapshot["cost_1x"] == pytest.approx(0.0005)
    assert snapshot["net_return_2x"] == pytest.approx(0.024)
    assert set(holdings["leg"]) == {"long", "short"}


def _minimal_strategy_frame() -> pd.DataFrame:
    dates = [
        pd.Timestamp("2026-01-01", tz="UTC"),
        pd.Timestamp("2026-01-02", tz="UTC"),
        pd.Timestamp("2026-01-03", tz="UTC"),
    ]
    symbols = ["A", "B", "C", "D"]
    rows = []
    for decision_ts in dates:
        strategy_returns = (
            [-0.04, -0.02, 0.02, 0.06]
            if decision_ts == dates[1]
            else [-0.02, 0.00, 0.02, 0.04]
        )
        if decision_ts == dates[0]:
            strategy_returns = [-0.04, -0.02, 0.02, 0.04]
        for symbol, signal, strategy_return in zip(
            symbols,
            [-1.0, -0.5, 0.5, 1.0],
            strategy_returns,
            strict=True,
        ):
            rows.append(
                {
                    "decision_ts": decision_ts,
                    "symbol": symbol,
                    "signal": signal,
                    "forward_return": strategy_return,
                    "strategy_forward_return": strategy_return,
                }
            )
    return pd.DataFrame(rows).set_index("decision_ts")


def _minimal_fold() -> WalkForwardFold:
    return WalkForwardFold(
        fold_idx=0,
        train_start=pd.Timestamp("2026-01-01", tz="UTC"),
        train_end=pd.Timestamp("2026-01-01", tz="UTC"),
        test_start=pd.Timestamp("2026-01-02", tz="UTC"),
        test_end=pd.Timestamp("2026-01-03", tz="UTC"),
    )


def test_evaluate_long_short_strategy_requires_no_overlap_validation():
    spec = factor_research.ComboSpec(
        combo_id="demo",
        track="main",
        panel_frequency="1d",
        return_horizon="1d",
        feature_names=("signal",),
        weight_scheme="equal",
    )

    with pytest.raises(ValueError, match="no-overlap validation"):
        factor_research.evaluate_long_short_strategy(
            combo_spec=spec,
            signal_frame=_minimal_strategy_frame(),
            folds=[_minimal_fold()],
            walk_forward_spec={
                "train_days": 1,
                "test_days": 2,
                "embargo_days": 1,
                "step_days": 2,
            },
            weight_scheme="equal",
            feature_families=None,
            decision_frequency="1d",
            n_buckets=2,
            min_cross_section=4,
            frequency_periods_per_year={"1d": 365},
            cost_multipliers=(1.0,),
            taker_fee_rate=0.001,
        )


def test_evaluate_long_short_strategy_end_to_end_hand_computed():
    spec = factor_research.ComboSpec(
        combo_id="demo",
        track="main",
        panel_frequency="1d",
        return_horizon="1d",
        feature_names=("signal",),
        weight_scheme="equal",
    )

    summary, detail, holdings = factor_research.evaluate_long_short_strategy(
        combo_spec=spec,
        signal_frame=_minimal_strategy_frame(),
        folds=[_minimal_fold()],
        walk_forward_spec={
            "train_days": 1,
            "test_days": 2,
            "embargo_days": 1,
            "step_days": 2,
        },
        weight_scheme="equal",
        feature_families=None,
        decision_frequency="1d",
        n_buckets=2,
        min_cross_section=4,
        frequency_periods_per_year={"1d": 365},
        cost_multipliers=(1.0, 1.5, 2.0),
        taker_fee_rate=0.001,
        horizon_deltas={"1d": pd.Timedelta(days=1)},
        supported_signal_timeframes=("1d",),
    )

    assert detail["gross_return"].tolist() == pytest.approx([0.035, 0.02])
    assert detail["charged_turnover"].tolist() == pytest.approx([1.0, 1.0])
    assert detail["terminal_close_turnover"].tolist() == pytest.approx([0.0, 1.0])
    assert detail["cost_1x"].tolist() == pytest.approx([0.001, 0.001])
    assert detail["net_return_1x"].tolist() == pytest.approx([0.034, 0.019])
    assert detail["net_return_1p5x"].tolist() == pytest.approx([0.0335, 0.0185])
    assert detail["net_return_2x"].tolist() == pytest.approx([0.033, 0.018])
    assert detail["long_count"].tolist() == [2, 2]
    assert detail["short_count"].tolist() == [2, 2]

    assert summary["gross_fold_positive_share"] == pytest.approx(1.0)
    assert summary["net_1x_fold_positive_share"] == pytest.approx(1.0)
    assert summary["net_1p5x_fold_positive_share"] == pytest.approx(1.0)
    assert summary["net_2x_fold_positive_share"] == pytest.approx(1.0)
    assert summary["fold_positive_share"] == summary["gross_fold_positive_share"]
    assert summary["top_fold_contribution"] == summary["gross_top_fold_contribution"]
    assert summary["gross_top_fold_contribution"] == pytest.approx(1.0)
    assert summary["net_1x_top_fold_contribution"] == pytest.approx(1.0)
    assert summary["gross_max_drawdown"] == pytest.approx(0.0)
    assert summary["net_1x_max_drawdown"] == pytest.approx(0.0)

    assert set(holdings["leg"]) == {"long", "short"}
    assert holdings.groupby("decision_ts")["weight"].sum().tolist() == pytest.approx([0.0, 0.0])
    assert holdings.groupby("decision_ts")["weight"].apply(lambda values: values.abs().sum()).tolist() == pytest.approx([1.0, 1.0])


def test_evaluate_long_short_strategy_supports_corr_discount_icir_weights():
    dates = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
    symbols = ["A", "B", "C", "D"]
    train_return_patterns = [
        [-0.04, -0.02, 0.02, 0.04],
        [-0.04, -0.01, 0.04, 0.02],
        [-0.02, -0.01, 0.04, 0.03],
        [-0.03, -0.02, 0.01, 0.04],
    ]
    rows = []
    for day_idx, decision_ts in enumerate(dates):
        returns = (
            train_return_patterns[day_idx]
            if day_idx < 4
            else [-0.03, -0.01, 0.02, 0.05]
        )
        for symbol, f1, f2, f3, forward_return in zip(
            symbols,
            [-1.0, -0.5, 0.5, 1.0],
            [-2.0, -1.0, 1.0, 2.0],
            [-1.0, 1.0, -0.5, 0.5],
            returns,
            strict=True,
        ):
            rows.append(
                {
                    "decision_ts": decision_ts,
                    "symbol": symbol,
                    "f1": f1,
                    "f2": f2,
                    "f3": f3,
                    "forward_return": forward_return,
                    "strategy_forward_return": forward_return,
                }
            )
    frame = pd.DataFrame(rows).set_index("decision_ts")
    spec = factor_research.ComboSpec(
        combo_id="corr_demo",
        track="all_two_gate",
        panel_frequency="1d",
        return_horizon="1d",
        feature_names=("f1", "f2", "f3"),
        weight_scheme="corr_discount_icir",
    )
    fold = WalkForwardFold(
        fold_idx=0,
        train_start=dates[0],
        train_end=dates[3],
        test_start=dates[4],
        test_end=dates[5],
    )

    summary, detail, holdings = factor_research.evaluate_long_short_strategy(
        combo_spec=spec,
        signal_frame=frame,
        folds=[fold],
        walk_forward_spec={
            "train_days": 4,
            "test_days": 2,
            "embargo_days": 1,
            "step_days": 2,
        },
        weight_scheme="corr_discount_icir",
        feature_families={"f1": "x", "f2": "x", "f3": "y"},
        decision_frequency="1d",
        n_buckets=2,
        min_cross_section=4,
        frequency_periods_per_year={"1d": 365},
        cost_multipliers=(1.0,),
        taker_fee_rate=0.001,
        horizon_deltas={"1d": pd.Timedelta(days=1)},
        supported_signal_timeframes=("1d",),
        min_pair_corr_observations=4,
    )

    assert not holdings.empty
    assert detail["correlation_pair_count"].tolist() == [3, 3]
    assert detail["correlation_min_pair_observation_count"].tolist() == [4, 4]
    assert detail["effective_factor_count"].notna().all()
    assert detail["max_feature_weight"].notna().all()
    assert summary["mean_effective_factor_count"] == pytest.approx(detail["effective_factor_count"].mean())
    assert summary["correlation_pair_count"] == 3


def test_evaluate_long_short_strategy_supports_family_alpha_weights():
    dates = [
        pd.Timestamp("2026-01-01", tz="UTC"),
        pd.Timestamp("2026-01-02", tz="UTC"),
        pd.Timestamp("2026-01-03", tz="UTC"),
    ]
    symbols = ["A", "B", "C", "D"]
    rows = []
    for decision_ts in dates:
        is_train = decision_ts == dates[0]
        f3_values = [-1.0, -0.5, 0.5, 1.0] if is_train else [-1.0, 1.0, -0.5, 0.5]
        for symbol, f1, f2, f3, forward_return in zip(
            symbols,
            [-1.0, -0.5, 0.5, 1.0],
            [-1.0, -0.5, 0.5, 1.0],
            f3_values,
            [-0.04, -0.02, 0.02, 0.04],
            strict=True,
        ):
            rows.append(
                {
                    "decision_ts": decision_ts,
                    "symbol": symbol,
                    "f1": f1,
                    "f2": f2,
                    "f3": f3,
                    "forward_return": forward_return,
                    "strategy_forward_return": forward_return,
                }
            )
    frame = pd.DataFrame(rows).set_index("decision_ts")
    spec = factor_research.ComboSpec(
        combo_id="family_alpha_demo",
        track="all_two_gate",
        panel_frequency="1d",
        return_horizon="1d",
        feature_names=("f1", "f2", "f3"),
        weight_scheme="family_alpha_0",
    )
    fold = WalkForwardFold(
        fold_idx=0,
        train_start=dates[0],
        train_end=dates[0],
        test_start=dates[1],
        test_end=dates[2],
    )

    summary, detail, holdings = factor_research.evaluate_long_short_strategy(
        combo_spec=spec,
        signal_frame=frame,
        folds=[fold],
        walk_forward_spec={
            "train_days": 1,
            "test_days": 2,
            "embargo_days": 1,
            "step_days": 2,
        },
        weight_scheme="family_alpha_0",
        feature_families={"f1": "A", "f2": "A", "f3": "B"},
        decision_frequency="1d",
        n_buckets=2,
        min_cross_section=4,
        frequency_periods_per_year={"1d": 365},
        cost_multipliers=(1.0,),
        taker_fee_rate=0.001,
        horizon_deltas={"1d": pd.Timedelta(days=1)},
        supported_signal_timeframes=("1d",),
    )

    assert detail["gross_return"].tolist() == pytest.approx([0.01, 0.01])
    assert detail["net_return_1x"].tolist() == pytest.approx([0.009, 0.009])
    assert detail["effective_factor_count"].tolist() == pytest.approx([2.6666666667, 2.6666666667])
    assert detail["effective_family_count"].tolist() == pytest.approx([2.0, 2.0])
    assert detail["max_family_weight"].tolist() == pytest.approx([0.5, 0.5])
    assert detail["max_feature_weight"].tolist() == pytest.approx([0.5, 0.5])
    assert summary["mean_effective_family_count"] == pytest.approx(2.0)
    assert summary["family_count"] == 2
    assert not holdings.empty


def test_ic_abs_feature_weights_are_hand_computed():
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "f1": [-1.0, -0.5, 0.5, 1.0],
            "f2": [-1.0, 0.5, -0.5, 1.0],
            "forward_return": [-0.04, -0.02, 0.02, 0.04],
        },
        index=pd.DatetimeIndex(["2026-01-01"] * 4, name="decision_ts"),
    )

    stats = factor_research.train_feature_stats(
        frame,
        ("f1", "f2"),
        min_cross_section=4,
    )
    assert stats is not None
    _, weights = factor_research.composite_weight_scores_and_weights(
        stats, "ic_abs")

    assert stats["f1"].mean_ic == pytest.approx(1.0)
    assert stats["f2"].mean_ic == pytest.approx(0.8)
    assert weights["f1"] == pytest.approx(1.0 / 1.8)
    assert weights["f2"] == pytest.approx(0.8 / 1.8)


def test_name_turnover_share_edge_cases():
    assert factor_research.name_turnover_share(set(), set()) == pytest.approx(0.0)
    assert factor_research.name_turnover_share({"A"}, set()) == pytest.approx(1.0)
    assert factor_research.name_turnover_share(set(), {"A"}) == pytest.approx(1.0)
    assert factor_research.name_turnover_share({"A", "B"}, {"A", "B"}) == pytest.approx(0.0)
    assert factor_research.name_turnover_share({"A", "B"}, {"B", "C"}) == pytest.approx(0.5)


def test_live_like_min_notional_replay_hand_computed():
    target_holdings = pd.DataFrame(
        [
            # Decision 1: open A long 50 and B short 50. PnL = 5 + 5.
            {
                "combo_id": "demo",
                "track": "main",
                "weight_scheme": "equal",
                "panel_frequency": "1d",
                "return_horizon": "1d",
                "component_features": "x",
                "fold_idx": 0,
                "decision_ts": pd.Timestamp("2026-01-01", tz="UTC"),
                "symbol": "A",
                "weight": 0.5,
                "strategy_forward_return": 0.10,
            },
            {
                "combo_id": "demo",
                "track": "main",
                "weight_scheme": "equal",
                "panel_frequency": "1d",
                "return_horizon": "1d",
                "component_features": "x",
                "fold_idx": 0,
                "decision_ts": pd.Timestamp("2026-01-01", tz="UTC"),
                "symbol": "B",
                "weight": -0.5,
                "strategy_forward_return": -0.10,
            },
            # Decision 2: A adjustment from 50 to 30 is below A's 30 min
            # notional, and C open 20 is below C's 30 min notional. Both are
            # skipped, so A/B are retained and no order is charged.
            {
                "combo_id": "demo",
                "track": "main",
                "weight_scheme": "equal",
                "panel_frequency": "1d",
                "return_horizon": "1d",
                "component_features": "x",
                "fold_idx": 0,
                "decision_ts": pd.Timestamp("2026-01-02", tz="UTC"),
                "symbol": "A",
                "weight": 0.3,
                "strategy_forward_return": 0.0,
            },
            {
                "combo_id": "demo",
                "track": "main",
                "weight_scheme": "equal",
                "panel_frequency": "1d",
                "return_horizon": "1d",
                "component_features": "x",
                "fold_idx": 0,
                "decision_ts": pd.Timestamp("2026-01-02", tz="UTC"),
                "symbol": "B",
                "weight": -0.5,
                "strategy_forward_return": 0.0,
            },
            {
                "combo_id": "demo",
                "track": "main",
                "weight_scheme": "equal",
                "panel_frequency": "1d",
                "return_horizon": "1d",
                "component_features": "x",
                "fold_idx": 0,
                "decision_ts": pd.Timestamp("2026-01-02", tz="UTC"),
                "symbol": "C",
                "weight": 0.2,
                "strategy_forward_return": 0.10,
            },
            # Decision 3: close A/B, open C/D, then terminal-close C/D at fold
            # end. PnL = 50*0.04 + (-50)*(-0.02) = 3.
            {
                "combo_id": "demo",
                "track": "main",
                "weight_scheme": "equal",
                "panel_frequency": "1d",
                "return_horizon": "1d",
                "component_features": "x",
                "fold_idx": 0,
                "decision_ts": pd.Timestamp("2026-01-03", tz="UTC"),
                "symbol": "C",
                "weight": 0.5,
                "strategy_forward_return": 0.04,
            },
            {
                "combo_id": "demo",
                "track": "main",
                "weight_scheme": "equal",
                "panel_frequency": "1d",
                "return_horizon": "1d",
                "component_features": "x",
                "fold_idx": 0,
                "decision_ts": pd.Timestamp("2026-01-03", tz="UTC"),
                "symbol": "D",
                "weight": -0.5,
                "strategy_forward_return": -0.02,
            },
        ]
    )

    summary, detail, orders, actual_holdings = factor_research.live_like_min_notional_replay(
        target_holdings,
        {"A": 30.0, "B": 10.0, "C": 30.0, "D": 10.0},
        account_equity=50.0,
        target_gross_notional=100.0,
        exchange_leverage=5.0,
        taker_fee_rate=0.001,
        cost_multipliers=(1.0, 2.0),
        frequency_periods_per_year={"1d": 365},
    )

    assert detail["gross_pnl_usd"].tolist() == pytest.approx([10.0, 0.0, 3.0])
    assert detail["charged_order_notional"].tolist() == pytest.approx([100.0, 0.0, 300.0])
    assert detail["filtered_order_count"].tolist() == [0, 2, 0]
    assert detail["actual_gross_notional"].tolist() == pytest.approx([100.0, 100.0, 100.0])
    assert detail["terminal_close_notional"].tolist() == pytest.approx([0.0, 0.0, 100.0])
    assert detail["net_return_1x_on_equity"].tolist() == pytest.approx([0.198, 0.0, 0.054])
    assert detail["net_return_2x_on_equity"].tolist() == pytest.approx([0.196, 0.0, 0.048])

    second_day_orders = orders.loc[orders["decision_ts"] == pd.Timestamp("2026-01-02", tz="UTC")]
    assert set(second_day_orders["status"]) == {"filtered_adjust", "filtered_open", "unchanged"}
    assert int((orders["status"] == "terminal_close").sum()) == 2
    assert set(actual_holdings.loc[actual_holdings["decision_ts"] == pd.Timestamp("2026-01-02", tz="UTC"), "symbol"]) == {"A", "B"}

    row = summary.iloc[0]
    assert row["filtered_order_count"] == 2
    assert row["order_attempt_count"] == 8
    assert row["filtered_order_share"] == pytest.approx(2 / 8)
    assert row["max_margin_required"] == pytest.approx(20.0)
    assert row["max_margin_utilization"] == pytest.approx(0.4)
    assert row["net_1x_fold_positive_share_on_equity"] == pytest.approx(1.0)


def test_live_like_min_notional_replay_keeps_weight_schemes_separate():
    base_rows = [
        {
            "combo_id": "same_combo_id",
            "track": "main",
            "weight_scheme": "equal",
            "panel_frequency": "1d",
            "return_horizon": "1d",
            "component_features": "x",
            "fold_idx": 0,
            "decision_ts": pd.Timestamp("2026-01-01", tz="UTC"),
            "symbol": "A",
            "weight": 0.5,
            "strategy_forward_return": 0.01,
        },
        {
            "combo_id": "same_combo_id",
            "track": "main",
            "weight_scheme": "equal",
            "panel_frequency": "1d",
            "return_horizon": "1d",
            "component_features": "x",
            "fold_idx": 0,
            "decision_ts": pd.Timestamp("2026-01-01", tz="UTC"),
            "symbol": "B",
            "weight": -0.5,
            "strategy_forward_return": -0.01,
        },
    ]
    icir_rows = [dict(row, weight_scheme="icir") for row in base_rows]
    target_holdings = pd.DataFrame([*base_rows, *icir_rows])

    summary, _, _, _ = factor_research.live_like_min_notional_replay(
        target_holdings,
        {"A": 1.0, "B": 1.0},
        account_equity=50.0,
        target_gross_notional=100.0,
        exchange_leverage=5.0,
        taker_fee_rate=0.001,
        cost_multipliers=(1.0,),
        frequency_periods_per_year={"1d": 365},
    )

    assert len(summary) == 2
    assert set(summary["weight_scheme"]) == {"equal", "icir"}
