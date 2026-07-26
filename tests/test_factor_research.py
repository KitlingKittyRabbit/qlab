import numpy as np
import pandas as pd
import pytest

from qlab.factor_research import (
    ComboSpec,
    FeatureTrainStat,
    annualized_mean_return,
    annualized_volatility,
    apply_combo_signal_diagnostic_fdr,
    bucket_diagnostics_for_frame,
    build_walk_forward_folds,
    candidate_combo_specs_from_gate_summary,
    candidate_structured_combo_specs_from_gate_summary,
    combo_decision_frequency,
    correlation_discount_weights,
    decision_timestamps_aligned_to_frequency,
    evaluate_combo_signal_diagnostics,
    evaluate_combo_signal_two_gate_diagnostics,
    fama_macbeth_diagnostics_for_frame_slice,
    features_decision_frequency,
    filter_frame_to_decision_frequency,
    max_drawdown_from_returns,
    non_overlapping_decision_frequency,
    normalized_feature_weights,
    rank_ic_diagnostics_for_frame,
    single_feature_train_direction,
    summarize_bucket_backtest,
    summarize_ic_series,
    summarize_fama_macbeth,
    summarize_top_bottom_backtest,
    three_gate_support_flags,
    train_cross_sectional_feature_correlation,
    walk_forward_spec_for_frequency,
    train_feature_stats,
    validate_no_overlap_design,
)
from qlab.walkforward import WalkForwardFold


def _panel_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    symbols = ["A", "B", "C", "D", "E", "F", "G", "H"]
    index = pd.MultiIndex.from_product(
        [dates, symbols], names=["decision_ts", "symbol"])
    values = np.tile(np.arange(1, 9, dtype=float), len(dates))
    size_control = np.tile(
        [0.125, -0.8, 0.33, 1.7, -1.4, 0.2, 0.95, -0.55], len(dates))
    momentum_control = np.tile(
        [-1.2, 0.4, 1.1, -0.3, 0.75, -0.9, 1.6, 0.05], len(dates))
    volatility_control = np.tile(
        [0.8, -1.1, -0.6, 0.9, 1.4, -0.2, -1.5, 0.35], len(dates))
    return pd.DataFrame(
        {
            "symbol": index.get_level_values("symbol"),
            "factor": values,
            "forward_return": values / 100.0 + size_control / 1000.0,
            "size_control": size_control,
            "momentum_control": momentum_control,
            "volatility_control": volatility_control,
        },
        index=index,
    )


def test_rank_ic_diagnostics_scores_each_decision():
    diagnostics = rank_ic_diagnostics_for_frame(
        _panel_frame(), "factor", min_cross_section=5)

    assert len(diagnostics) == 3
    assert {row["status"] for row in diagnostics} == {"ok"}
    assert all(row["raw_rank_ic"] == pytest.approx(1.0) for row in diagnostics)


def test_bucket_diagnostics_orients_spread():
    detail, diagnostics = bucket_diagnostics_for_frame(
        _panel_frame(), "factor", direction=1, n_buckets=3)

    assert {row["status"] for row in diagnostics} == {"ok"}
    pivot = detail.pivot_table(
        index="decision_ts", columns="bucket", values="bucket_return")
    assert (pivot[3] > pivot[1]).all()


def test_fama_macbeth_slice_and_summary_use_configurable_controls():
    diagnostics = fama_macbeth_diagnostics_for_frame_slice(
        _panel_frame(),
        "factor",
        direction=1,
        control_columns=("size_control", "momentum_control",
                         "volatility_control"),
        min_cross_section=6,
    )

    detail = pd.DataFrame(
        [row for row in diagnostics if row["status"] == "ok"])
    summary = summarize_fama_macbeth(
        "1d",
        "1d",
        "factor",
        detail,
        {"train_days": 30, "test_days": 10, "embargo_days": 1, "step_days": 10},
        diagnostics,
        ("size_control", "momentum_control", "volatility_control"),
    )

    assert summary["gamma_observation_count"] == 3
    assert summary["scored_decision_count"] == 3
    assert "mean_size_control_gamma" in summary


def test_train_stats_and_weights_are_generic():
    stats = train_feature_stats(
        _panel_frame(), ("factor",), min_cross_section=5)

    assert stats is not None
    scores, weights = normalized_feature_weights(stats, "ic_abs")
    assert scores["factor"] > 0
    assert weights["factor"] == pytest.approx(1.0)


def test_candidate_combo_specs_are_same_horizon_and_family_bounded():
    gate_summary = pd.DataFrame(
        {
            "feature_name": [
                "funding_a__1h",
                "funding_b__12h",
                "oi_a__1h",
                "taker_a__1h",
                "funding_c__1h",
            ],
            "return_horizon": ["4h", "4h", "4h", "4h", "8h"],
            "two_gate_support": [True, True, True, False, True],
            "three_gate_support": [False, True, False, False, True],
            "ic_hac_t_stat": [2.0, 3.0, 2.5, 5.0, 4.0],
            "bucket_spread_mean_return": [0.01, 0.02, 0.03, 0.04, 0.05],
            "fm_hac_t_stat": [0.5, 1.5, 0.7, 2.0, 2.5],
            "icir": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    registry = pd.DataFrame(
        {
            "feature_name": [
                "funding_a__1h",
                "funding_b__12h",
                "oi_a__1h",
                "taker_a__1h",
                "funding_c__1h",
            ],
            "family": ["funding", "funding", "oi", "taker", "funding"],
        }
    )

    specs, catalog = candidate_combo_specs_from_gate_summary(
        gate_summary,
        registry,
        panel_frequency="1h",
        top_k_values=(2,),
        family_combo_sizes=(2,),
        weight_schemes=("equal", "icir"),
        horizon_deltas={
            "1h": pd.Timedelta(hours=1),
            "4h": pd.Timedelta(hours=4),
            "8h": pd.Timedelta(hours=8),
            "12h": pd.Timedelta(hours=12),
        },
        supported_signal_timeframes=("1h", "12h"),
    )

    assert specs
    assert not catalog.empty
    assert {spec.return_horizon for spec in specs} == {"4h"}
    assert all(len(spec.feature_names) >= 2 for spec in specs)
    assert all("taker_a__1h" not in spec.feature_names for spec in specs)
    family_all = [
        spec for spec in specs
        if spec.track == "two_gate_family_best_all" and spec.weight_scheme == "equal"
    ][0]
    assert family_all.feature_names == ("oi_a__1h", "funding_a__1h")
    assert family_all.panel_frequency == "4h"
    assert {spec.weight_scheme for spec in specs} == {"equal", "icir"}


def test_candidate_combo_specs_fail_when_registry_family_missing():
    gate_summary = pd.DataFrame(
        {
            "feature_name": ["factor_a", "factor_b"],
            "return_horizon": ["4h", "4h"],
            "two_gate_support": [True, True],
            "three_gate_support": [False, False],
            "ic_hac_t_stat": [2.0, 2.0],
            "bucket_spread_mean_return": [0.01, 0.02],
            "fm_hac_t_stat": [0.5, 0.6],
            "icir": [0.1, 0.2],
        }
    )
    registry = pd.DataFrame({"feature_name": ["factor_a"], "family": ["x"]})

    with pytest.raises(ValueError, match="missing family"):
        candidate_combo_specs_from_gate_summary(
            gate_summary,
            registry,
            panel_frequency="1h",
        )


def test_candidate_structured_combo_specs_are_fixed_baskets():
    gate_summary = pd.DataFrame(
        {
            "feature_name": [
                "funding_a__1h",
                "funding_b__12h",
                "oi_a__1h",
                "oi_b__12h",
                "taker_a__1h",
                "ignored__1h",
            ],
            "return_horizon": ["1d", "1d", "1d", "1d", "1d", "1d"],
            "two_gate_support": [True, True, True, True, True, False],
            "three_gate_support": [True, False, True, False, False, True],
        }
    )
    registry = pd.DataFrame(
        {
            "feature_name": [
                "funding_a__1h",
                "funding_b__12h",
                "oi_a__1h",
                "oi_b__12h",
                "taker_a__1h",
                "ignored__1h",
            ],
            "family": ["funding", "funding", "oi", "oi", "taker", "ignored"],
        }
    )

    specs, catalog = candidate_structured_combo_specs_from_gate_summary(
        gate_summary,
        registry,
        panel_frequency="1h",
        weight_schemes=("equal", "icir", "corr_discount_icir"),
        horizon_deltas={
            "1h": pd.Timedelta(hours=1),
            "12h": pd.Timedelta(hours=12),
            "1d": pd.Timedelta(days=1),
        },
        supported_signal_timeframes=("1h", "12h", "1d"),
    )

    tracks = set(catalog["track"])
    assert {
        "all_two_gate",
        "three_gate_all",
        "family_funding",
        "family_oi",
        "without_funding",
        "without_oi",
        "without_taker",
    }.issubset(tracks)
    assert not any(str(track).startswith("two_gate_top") for track in tracks)
    assert set(catalog["weight_scheme"]) == {"equal", "icir", "corr_discount_icir"}
    assert set(catalog["return_horizon"]) == {"1d"}
    assert all("ignored__1h" not in spec.feature_names for spec in specs)

    track_members = {
        row.track: tuple(str(row.component_features).split(" | "))
        for row in catalog.loc[catalog["weight_scheme"] == "equal"].itertuples(index=False)
    }
    assert track_members["three_gate_all"] == ("funding_a__1h", "oi_a__1h")
    assert track_members["family_funding"] == ("funding_a__1h", "funding_b__12h")
    assert track_members["without_taker"] == (
        "funding_a__1h",
        "funding_b__12h",
        "oi_a__1h",
        "oi_b__12h",
    )


def test_train_cross_sectional_feature_correlation_is_train_only_and_spearman():
    dates = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
    symbols = ["A", "B", "C", "D"]
    rows = []
    for decision_ts in dates:
        for symbol, f1, f2, f3 in zip(
            symbols,
            [-1.0, -0.5, 0.5, 1.0],
            [-2.0, -1.0, 1.0, 2.0],
            [1.0, 0.5, -0.5, -1.0],
            strict=True,
        ):
            rows.append({"decision_ts": decision_ts, "symbol": symbol, "f1": f1, "f2": f2, "f3": f3})
    frame = pd.DataFrame(rows).set_index("decision_ts")

    corr, diagnostics = train_cross_sectional_feature_correlation(
        frame,
        ("f1", "f2", "f3"),
        min_cross_section=4,
        min_pair_corr_observations=4,
    )

    assert set(diagnostics["status"]) == {"ok"}
    assert corr.loc["f1", "f2"] == pytest.approx(1.0)
    assert corr.loc["f1", "f3"] == pytest.approx(-1.0)
    assert corr.loc["f2", "f3"] == pytest.approx(-1.0)


def test_correlation_discount_weights_are_hand_computed():
    train_stats = {
        feature_name: FeatureTrainStat(
            direction=1,
            mean_ic=0.10,
            std_ic=0.05,
            icir=2.0,
            hac_t_stat=1.0,
            observation_count=100,
        )
        for feature_name in ("f1", "f2", "f3")
    }
    corr = pd.DataFrame(
        [
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ],
        index=["f1", "f2", "f3"],
        columns=["f1", "f2", "f3"],
    )

    scores, weights, diagnostics = correlation_discount_weights(
        train_stats,
        corr,
        feature_families={"f1": "crowded", "f2": "crowded", "f3": "independent"},
    )

    assert scores["f1"] == pytest.approx(2.0 / (1.0 + 0.9 + 0.1))
    assert scores["f2"] == pytest.approx(2.0 / (1.0 + 0.9 + 0.1))
    assert scores["f3"] == pytest.approx(2.0 / (1.0 + 0.1 + 0.1))
    total = scores["f1"] + scores["f2"] + scores["f3"]
    assert weights["f1"] == pytest.approx(scores["f1"] / total)
    assert weights["f2"] == pytest.approx(scores["f2"] / total)
    assert weights["f3"] == pytest.approx(scores["f3"] / total)
    assert weights["f3"] > weights["f1"]
    assert diagnostics["effective_factor_count"] == pytest.approx(
        1.0 / sum(value * value for value in weights.values())
    )
    assert diagnostics["family_weight_share_max"] == pytest.approx(weights["f1"] + weights["f2"])
    assert diagnostics["mean_abs_feature_corr"] == pytest.approx((0.9 + 0.1 + 0.9 + 0.1 + 0.1 + 0.1) / 6.0)


def test_single_feature_train_direction_requires_nonzero_mean_ic():
    frame = _panel_frame().copy()
    frame["flat"] = np.tile([1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0], 3)

    direction = single_feature_train_direction(frame, "flat", min_cross_section=5, epsilon=10.0)

    assert direction.direction == 0
    assert direction.status == "no_train_direction"
    assert direction.observation_count == 3


def test_three_gate_support_flags_are_fail_closed():
    passing = three_gate_support_flags(
        ic_mean=0.01,
        ic_hac_t_stat=2.0,
        bucket_spread_mean_return=0.001,
        bucket_monotonic_pair_pass_share=0.75,
        fm_mean_gamma=0.001,
        fm_hac_t_stat=2.0,
    )
    failing = three_gate_support_flags(
        ic_mean=0.01,
        ic_hac_t_stat=2.0,
        bucket_spread_mean_return=0.001,
        bucket_monotonic_pair_pass_share=0.50,
        fm_mean_gamma=0.001,
        fm_hac_t_stat=2.0,
    )

    assert passing == {
        "stage1_ic_support": True,
        "stage2_bucket_support": True,
        "stage3_fm_support": True,
        "two_gate_support": True,
        "three_gate_support": True,
    }
    assert failing["stage1_ic_support"] is True
    assert failing["stage2_bucket_support"] is False
    assert failing["two_gate_support"] is False
    assert failing["three_gate_support"] is False


def test_continuous_holding_rejects_signal_slower_than_horizon():
    horizon_deltas = {
        "2h": pd.Timedelta(hours=2),
        "4h": pd.Timedelta(hours=4),
        "12h": pd.Timedelta(hours=12),
        "1d": pd.Timedelta(days=1),
    }
    supported = ("4h", "12h", "1d")
    spec = ComboSpec("x", "track", "2h", "12h", ("a__4h", "b__1d"))

    assert features_decision_frequency(
        spec.feature_names, spec.panel_frequency, horizon_deltas, supported) == "1d"
    assert combo_decision_frequency(spec, horizon_deltas, supported) == "1d"

    with pytest.raises(ValueError, match="not an exact divisor"):
        validate_no_overlap_design([spec], horizon_deltas, supported)
    validate_no_overlap_design(
        [ComboSpec("good", "track", "2h", "1d", ("a__4h",))], horizon_deltas, supported)


def test_one_hour_panel_native_suffix_drives_decision_frequency():
    horizon_deltas = {
        "1h": pd.Timedelta(hours=1),
        "12h": pd.Timedelta(hours=12),
        "1d": pd.Timedelta(days=1),
    }
    supported = ("1h", "12h", "1d")

    assert features_decision_frequency(
        ("funding_close__12h",), "1h", horizon_deltas, supported
    ) == "12h"
    assert combo_decision_frequency(
        ComboSpec("x", "track", "1h", "12h", ("funding_close__1d",)),
        horizon_deltas,
        supported,
    ) == "1d"


def test_non_overlapping_frequency_respects_return_horizon():
    horizon_deltas = {
        "1h": pd.Timedelta(hours=1),
        "12h": pd.Timedelta(hours=12),
        "1d": pd.Timedelta(days=1),
    }
    supported = ("1h", "12h", "1d")

    assert non_overlapping_decision_frequency(
        ("funding_close__1h",), "1h", "1d", horizon_deltas, supported
    ) == "1d"
    with pytest.raises(ValueError, match="not an exact divisor"):
        non_overlapping_decision_frequency(
            ("funding_close__12h",), "1h", "1h", horizon_deltas, supported
        )


def test_filter_frame_to_decision_frequency_uses_utc_anchor():
    horizon_deltas = {
        "1h": pd.Timedelta(hours=1),
        "12h": pd.Timedelta(hours=12),
        "1d": pd.Timedelta(days=1),
    }
    index = pd.date_range("2026-01-01 00:00:00Z", periods=49, freq="1h", name="decision_ts")
    frame = pd.DataFrame({"value": range(len(index))}, index=index)

    daily = filter_frame_to_decision_frequency(frame, "1d", horizon_deltas)
    twelve_hour = filter_frame_to_decision_frequency(frame, "12h", horizon_deltas)

    assert daily.index.tolist() == [
        pd.Timestamp("2026-01-01 00:00:00Z"),
        pd.Timestamp("2026-01-02 00:00:00Z"),
        pd.Timestamp("2026-01-03 00:00:00Z"),
    ]
    assert twelve_hour.index.tolist() == [
        pd.Timestamp("2026-01-01 00:00:00Z"),
        pd.Timestamp("2026-01-01 12:00:00Z"),
        pd.Timestamp("2026-01-02 00:00:00Z"),
        pd.Timestamp("2026-01-02 12:00:00Z"),
        pd.Timestamp("2026-01-03 00:00:00Z"),
    ]
    assert decision_timestamps_aligned_to_frequency(
        pd.DatetimeIndex(["2026-01-01 00:00:00Z", "2026-01-01 00:15:00Z"]),
        "1h",
        horizon_deltas,
    ).tolist() == [True, False]


def test_walk_forward_spec_requires_embargo_for_subdaily_horizon():
    default_specs = {
        "1h": {
            "train_days": 30,
            "test_days": 10,
            "embargo_days": 0,
            "step_days": 10,
        }
    }
    horizon_deltas = {"12h": pd.Timedelta(hours=12)}

    with pytest.raises(ValueError, match="at least 1d"):
        walk_forward_spec_for_frequency(
            "1h",
            default_specs,
            horizon_deltas,
            horizons="12h",
        )


def test_build_walk_forward_folds_rejects_invalid_zero_step():
    default_specs = {
        "1d": {
            "train_days": 30,
            "test_days": 10,
            "embargo_days": 1,
            "step_days": 0,
        }
    }
    horizon_deltas = {"1d": pd.Timedelta(days=1)}

    with pytest.raises(ValueError, match="step_days"):
        build_walk_forward_folds(
            pd.date_range("2026-01-01", periods=100, freq="D"),
            "1d",
            "1d",
            default_specs,
            horizon_deltas,
        )


def test_annualized_summary_primitives_match_manual_values():
    values = pd.Series([0.10, -0.05, 0.02], dtype=float)
    periods = {"1d": 365}

    assert annualized_mean_return(values, "1d", periods) == pytest.approx(values.mean() * 365)
    assert annualized_volatility(values, "1d", periods) == pytest.approx(
        values.std(ddof=1) * np.sqrt(365)
    )
    expected_equity = (1.0 + values).cumprod()
    expected_drawdown = (expected_equity / expected_equity.cummax() - 1.0).min()
    assert max_drawdown_from_returns(values) == pytest.approx(expected_drawdown)


def test_summarize_ic_series_matches_hand_values():
    detail = pd.DataFrame(
        {
            "fold_idx": [0, 0, 1],
            "rank_ic": [0.10, -0.05, 0.20],
            "train_mean_ic": [0.01, 0.01, 0.03],
            "cross_section_size": [20, 30, 40],
        }
    )
    diagnostics = [
        {"decision_ts": pd.Timestamp("2026-01-01"), "status": "ok"},
        {"decision_ts": pd.Timestamp("2026-01-02"), "status": "small_cross_section"},
        {"decision_ts": pd.Timestamp("2026-01-03"), "status": "ok"},
    ]

    summary = summarize_ic_series(
        "1d",
        "1d",
        "factor",
        detail,
        {"train_days": 30, "test_days": 10, "embargo_days": 1, "step_days": 10},
        diagnostics,
        hac_overlap_lags=2,
    )

    assert summary["n_folds"] == 2
    assert summary["mean_ic"] == pytest.approx((0.10 - 0.05 + 0.20) / 3.0)
    assert summary["ic_positive_share"] == pytest.approx(2.0 / 3.0)
    assert summary["mean_train_ic"] == pytest.approx((0.01 + 0.03) / 2.0)
    assert summary["cross_section_size_median"] == pytest.approx(30.0)
    assert summary["test_decision_count"] == 3
    assert summary["scored_decision_count"] == 2
    assert summary["skipped_small_cross_section_count"] == 1
    assert summary["hac_lags"] >= 2


def test_summarize_bucket_backtest_matches_hand_values():
    dates = pd.to_datetime(["2026-01-01", "2026-01-02"])
    detail = pd.DataFrame(
        {
            "fold_idx": [0, 0, 0, 1, 1, 1],
            "decision_ts": [dates[0], dates[0], dates[0], dates[1], dates[1], dates[1]],
            "bucket": [1, 2, 3, 1, 2, 3],
            "bucket_return": [-0.03, 0.00, 0.06, -0.01, 0.01, 0.05],
            "bucket_size": [2, 2, 2, 2, 2, 2],
        }
    )
    diagnostics = [
        {"decision_ts": dates[0], "status": "ok"},
        {"decision_ts": dates[1], "status": "ok"},
        {"decision_ts": pd.Timestamp("2026-01-03"), "status": "constant_feature"},
    ]

    summary = summarize_bucket_backtest(
        "1d",
        "1d",
        "factor",
        detail,
        {"train_days": 30, "test_days": 10, "embargo_days": 1, "step_days": 10},
        diagnostics,
        n_buckets=3,
        frequency_periods_per_year={"1d": 365},
    )

    assert summary["n_folds"] == 2
    assert summary["spread_observation_count"] == 2
    assert summary["spread_mean_return"] == pytest.approx((0.09 + 0.06) / 2.0)
    assert summary["spread_annualized_return"] == pytest.approx(0.075 * 365)
    assert summary["spread_positive_share"] == pytest.approx(1.0)
    assert summary["monotonic_increasing"] is True
    assert summary["monotonic_pair_pass_share"] == pytest.approx(1.0)
    assert summary["q1_mean_return"] == pytest.approx(-0.02)
    assert summary["q2_mean_return"] == pytest.approx(0.005)
    assert summary["q3_mean_return"] == pytest.approx(0.055)
    assert summary["skipped_constant_feature_count"] == 1


def test_summarize_top_bottom_backtest_matches_hand_values():
    dates = pd.to_datetime(["2026-01-01", "2026-01-02"])
    detail = pd.DataFrame(
        {
            "fold_idx": [0, 0, 1, 1],
            "decision_ts": [dates[0], dates[0], dates[1], dates[1]],
            "leg": ["short", "long", "short", "long"],
            "leg_return": [-0.03, 0.05, -0.01, 0.03],
            "leg_size": [2, 2, 2, 2],
            "spread_return": [0.08, 0.08, 0.04, 0.04],
            "cross_section_size": [4, 4, 4, 4],
        }
    )
    diagnostics = [
        {"decision_ts": dates[0], "status": "ok"},
        {"decision_ts": dates[1], "status": "ok"},
        {"decision_ts": pd.Timestamp("2026-01-03"), "status": "small_cross_section"},
    ]

    summary = summarize_top_bottom_backtest(
        "1d",
        "1d",
        "factor",
        detail,
        {"train_days": 30, "test_days": 10, "embargo_days": 1, "step_days": 10},
        diagnostics,
        leg_count=2,
        frequency_periods_per_year={"1d": 365},
    )

    assert summary["n_folds"] == 2
    assert summary["leg_count"] == 2
    assert summary["short_leg_mean_return"] == pytest.approx(-0.02)
    assert summary["long_leg_mean_return"] == pytest.approx(0.04)
    assert summary["short_leg_avg_size"] == pytest.approx(2.0)
    assert summary["long_leg_avg_size"] == pytest.approx(2.0)
    assert summary["spread_mean_return"] == pytest.approx(0.06)
    assert summary["spread_annualized_return"] == pytest.approx(0.06 * 365)
    assert summary["spread_positive_share"] == pytest.approx(1.0)
    assert summary["skipped_small_cross_section_count"] == 1


def test_summarize_fama_macbeth_matches_hand_values():
    detail = pd.DataFrame(
        {
            "fold_idx": [0, 1, 1],
            "gamma_signal": [0.10, -0.05, 0.20],
            "gamma_intercept": [0.01, 0.02, 0.03],
            "gamma_size": [0.50, 0.60, 0.70],
            "r_squared": [0.80, 0.70, 0.90],
        }
    )
    diagnostics = [
        {"decision_ts": pd.Timestamp("2026-01-01"), "status": "ok"},
        {"decision_ts": pd.Timestamp("2026-01-02"), "status": "singular_design"},
        {"decision_ts": pd.Timestamp("2026-01-03"), "status": "ok"},
    ]

    summary = summarize_fama_macbeth(
        "1d",
        "1d",
        "factor",
        detail,
        {"train_days": 30, "test_days": 10, "embargo_days": 1, "step_days": 10},
        diagnostics,
        ("size",),
        hac_overlap_lags=2,
    )

    assert summary["n_folds"] == 2
    assert summary["gamma_observation_count"] == 3
    assert summary["mean_gamma"] == pytest.approx((0.10 - 0.05 + 0.20) / 3.0)
    assert summary["gamma_positive_share"] == pytest.approx(2.0 / 3.0)
    assert summary["mean_intercept"] == pytest.approx(0.02)
    assert summary["mean_size_gamma"] == pytest.approx(0.60)
    assert summary["mean_r_squared"] == pytest.approx(0.80)
    assert summary["skipped_singular_design_count"] == 1
    assert summary["hac_lags"] >= 2


def test_combo_signal_diagnostics_use_train_only_weights_and_controls(monkeypatch):
    symbols = ["A", "B", "C", "D"]
    train_dates = pd.date_range("2026-01-01", periods=4, freq="D")
    test_dates = pd.date_range("2026-01-06", periods=4, freq="D")
    rows = []
    base_by_symbol = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
    control_by_symbol = {"A": 1.0, "B": -1.0, "C": -1.0, "D": 1.0}
    test_gammas = {
        test_dates[0]: 0.010,
        test_dates[1]: 0.020,
        test_dates[2]: 0.015,
        test_dates[3]: 0.025,
    }
    for decision_ts in [*train_dates, *test_dates]:
        for symbol in symbols:
            base = base_by_symbol[symbol]
            control = control_by_symbol[symbol]
            f1 = base
            f2 = 5.0 - base
            if decision_ts in test_gammas:
                combo_signal = 0.5 * f1 + 0.5 * (-f2)
                forward_return = test_gammas[decision_ts] * combo_signal + 0.001 * control
            else:
                forward_return = base / 100.0
            rows.append(
                {
                    "decision_ts": decision_ts,
                    "symbol": symbol,
                    "f1__1d": f1,
                    "f2__1d": f2,
                    "forward_return": forward_return,
                    "size_control": control,
                }
            )
    frame = pd.DataFrame(rows).set_index(["decision_ts", "symbol"]).sort_index()
    frame["symbol"] = frame.index.get_level_values("symbol")
    fold = WalkForwardFold(
        fold_idx=0,
        train_start=train_dates[0],
        train_end=train_dates[-1],
        test_start=test_dates[0],
        test_end=test_dates[-1],
    )
    spec = ComboSpec(
        combo_id="same_combo_id",
        track="unit",
        panel_frequency="1d",
        return_horizon="1d",
        feature_names=("f1__1d", "f2__1d"),
        weight_scheme="equal",
    )

    summary, composite, ic_detail, bucket_detail, fm_detail, weights, diagnostics = evaluate_combo_signal_diagnostics(
        spec,
        frame,
        [fold],
        {"train_days": 4, "test_days": 4, "embargo_days": 1, "step_days": 4},
        weight_scheme="equal",
        feature_families=None,
        control_columns=("size_control",),
        decision_frequency="1d",
        n_buckets=2,
        min_cross_section=4,
        frequency_periods_per_year={"1d": 365},
        horizon_deltas={"1d": pd.Timedelta(days=1)},
        supported_signal_timeframes=("1d",),
    )

    first_test = composite.loc[composite["decision_ts"] == test_dates[0]].sort_values("symbol")
    assert first_test["combo_signal"].to_list() == pytest.approx([-1.5, -0.5, 0.5, 1.5])
    assert weights.set_index("feature_name").loc["f1__1d", "direction"] == 1
    assert weights.set_index("feature_name").loc["f2__1d", "direction"] == -1
    assert weights.set_index("feature_name").loc["f1__1d", "feature_weight"] == pytest.approx(0.5)
    assert weights.set_index("feature_name").loc["f2__1d", "feature_weight"] == pytest.approx(0.5)
    assert summary.loc[0, "combo_ic_mean"] == pytest.approx(1.0)
    assert summary.loc[0, "combo_bucket_spread_mean_return"] == pytest.approx(
        2.0 * np.mean(list(test_gammas.values()))
    )
    assert summary.loc[0, "combo_fm_mean_gamma"] == pytest.approx(np.mean(list(test_gammas.values())))
    assert summary.loc[0, "combo_fm_mean_size_control_gamma"] == pytest.approx(0.001)
    assert len(ic_detail) == 4
    assert len(bucket_detail["decision_ts"].unique()) == 4
    assert len(fm_detail) == 4
    assert {"ic", "bucket", "fm"}.issubset(set(diagnostics["diagnostic_type"]))

    def forbidden_fm(*args, **kwargs):
        raise AssertionError("FM must not execute in the two-gate entry")

    with monkeypatch.context() as context:
        context.setattr("qlab.factor_research.fama_macbeth_diagnostics_for_frame_slice", forbidden_fm)
        two_gate_summary, _, _, _, _, two_gate_diagnostics = evaluate_combo_signal_two_gate_diagnostics(
            spec,
            frame,
            [fold],
            {"train_days": 4, "test_days": 4, "embargo_days": 1, "step_days": 4},
            weight_scheme="equal",
            feature_families=None,
            control_columns=("size_control",),
            decision_frequency="1d",
            n_buckets=2,
            min_cross_section=4,
            frequency_periods_per_year={"1d": 365},
            horizon_deltas={"1d": pd.Timedelta(days=1)},
            supported_signal_timeframes=("1d",),
        )
    assert two_gate_summary.loc[0, "combo_ic_mean"] == pytest.approx(1.0)
    assert "combo_fm_mean_gamma" not in two_gate_summary.columns
    assert not two_gate_diagnostics["diagnostic_type"].eq("fm").any()

    with pytest.raises(ValueError, match="no-overlap validation"):
        evaluate_combo_signal_diagnostics(
            spec,
            frame,
            [fold],
            {"train_days": 4, "test_days": 4, "embargo_days": 1, "step_days": 4},
            weight_scheme="equal",
            feature_families=None,
            control_columns=("size_control",),
            decision_frequency="1d",
            n_buckets=2,
            min_cross_section=4,
            frequency_periods_per_year={"1d": 365},
        )

    icir_spec = ComboSpec(
        combo_id="same_combo_id",
        track="unit",
        panel_frequency="1d",
        return_horizon="1d",
        feature_names=("f1__1d", "f2__1d"),
        weight_scheme="icir",
    )
    icir_summary, *_ = evaluate_combo_signal_diagnostics(
        icir_spec,
        frame,
        [fold],
        {"train_days": 4, "test_days": 4, "embargo_days": 1, "step_days": 4},
        weight_scheme="icir",
        feature_families=None,
        control_columns=("size_control",),
        decision_frequency="1d",
        n_buckets=2,
        min_cross_section=4,
        frequency_periods_per_year={"1d": 365},
        horizon_deltas={"1d": pd.Timedelta(days=1)},
        supported_signal_timeframes=("1d",),
    )
    combined = apply_combo_signal_diagnostic_fdr(pd.concat([summary, icir_summary], ignore_index=True))

    assert len(combined) == 2
    assert combined["combo_id"].to_list() == ["same_combo_id", "same_combo_id"]
    assert combined["weight_scheme"].to_list() == ["equal", "icir"]
    assert {"combo_ic_bh_fdr_q", "combo_fm_bh_fdr_q", "combo_three_gate_support_fdr_10pct"}.issubset(
        combined.columns
    )
