import numpy as np
import pandas as pd
import pytest

from qlab import coinglass_substitution as substitution
from qlab import factor_research
from qlab.walkforward import WalkForwardFold


def _price_frame(days: int = 100, *, growth: float = 0.0001, volume: float = 2.0) -> pd.DataFrame:
    periods = days * 96 + 8
    index = pd.date_range("2025-01-01", periods=periods, freq="15min", tz="UTC")
    step = np.arange(periods, dtype=float)
    close = np.exp(step * growth)
    return pd.DataFrame(
        {"o": close, "h": close, "l": close, "c": close, "v": volume},
        index=index,
    )


def _all_predictors(value: float = 1.0) -> dict[str, float]:
    return {column: value for column in substitution.ALL_RAW_PREDICTOR_COLUMNS}


def test_price_volume_features_use_cutoff_and_complete_windows():
    decision_ts = pd.Timestamp("2025-04-01 00:15:00", tz="UTC")
    panel_index = pd.MultiIndex.from_tuples(
        [(decision_ts, "BTC")], names=["decision_ts", "symbol"]
    )
    price = _price_frame()

    artifacts = substitution.build_price_volume_replacement_features(
        panel_index, {"BTC": price}
    )
    row = artifacts.raw_features.iloc[0]

    assert row["return_1d"] == pytest.approx(96 * 0.0001)
    assert row["realized_vol_1d"] == pytest.approx(0.0, abs=1e-12)
    audit = artifacts.availability_audit.iloc[0]
    assert audit["feature_cutoff_ts"] == decision_ts - pd.Timedelta(minutes=15)
    assert audit["max_input_bar_end_ts"] == audit["feature_cutoff_ts"]

    future_mutated = price.copy()
    future_open = decision_ts - pd.Timedelta(minutes=15)
    future_mutated.loc[future_open:, "c"] *= 1000.0
    future_artifacts = substitution.build_price_volume_replacement_features(
        panel_index, {"BTC": future_mutated}
    )
    pd.testing.assert_series_equal(
        artifacts.raw_features.iloc[0],
        future_artifacts.raw_features.iloc[0],
        check_names=False,
    )

    missing = price.drop(index=decision_ts - pd.Timedelta(days=1, minutes=15))
    missing_artifacts = substitution.build_price_volume_replacement_features(
        panel_index, {"BTC": missing}
    )
    assert np.isnan(missing_artifacts.raw_features.iloc[0]["return_1d"])
    assert not bool(missing_artifacts.availability_audit.iloc[0]["all_predictors_available"])


def test_intraday_size_control_uses_latest_completed_day_without_future_data():
    decision_ts = pd.Timestamp("2025-04-01 12:15:00", tz="UTC")
    panel_index = pd.MultiIndex.from_tuples(
        [(decision_ts, "BTC")], names=["decision_ts", "symbol"]
    )
    price = _price_frame()

    first = substitution.build_price_volume_replacement_features(
        panel_index, {"BTC": price}
    )
    assert np.isfinite(first.raw_features.iloc[0]["size_control_raw"])
    assert bool(first.availability_audit.iloc[0]["all_predictors_available"])

    mutated = price.copy()
    first_unavailable_open = decision_ts - pd.Timedelta(minutes=15)
    mutated.loc[first_unavailable_open:, ["c", "v"]] *= 1000.0
    second = substitution.build_price_volume_replacement_features(
        panel_index, {"BTC": mutated}
    )
    pd.testing.assert_series_equal(
        first.raw_features.iloc[0], second.raw_features.iloc[0], check_names=False
    )


def test_canonical_support_ranks_all_levels_on_same_rows():
    decision = pd.Timestamp("2025-01-01", tz="UTC")
    index = pd.MultiIndex.from_product(
        [[decision], ["A", "B", "C"]], names=["decision_ts", "symbol"]
    )
    target = pd.DataFrame(
        {
            "combo_signal": [-1.0, 0.0, 1.0],
            "forward_return": [0.1, 0.0, -0.1],
            "strategy_forward_return": [0.1, 0.0, -0.1],
        },
        index=index,
    )
    predictors = pd.DataFrame(
        [{**_all_predictors(float(i + 1))} for i in range(3)], index=index
    )
    predictors.loc[(decision, "C"), "return_32d"] = np.nan

    artifacts = substitution.build_canonical_common_support(
        target, predictors, min_cross_section=2
    )

    assert list(artifacts.frame.index.get_level_values("symbol")) == ["A", "B"]
    assert artifacts.frame["return_1d"].tolist() == pytest.approx([-1.0, 1.0])
    assert artifacts.frame["size_control"].tolist() == pytest.approx([-1.0, 1.0])
    audit = artifacts.audit.iloc[0]
    assert audit["original_valid_count"] == 3
    assert audit["common_valid_count"] == 2
    assert audit["excluded_symbols"] == "C"


def _ridge_frame() -> tuple[pd.DataFrame, WalkForwardFold]:
    dates = pd.date_range("2025-01-01", periods=20, freq="1D", tz="UTC")
    symbols = [f"S{index:02d}" for index in range(10)]
    index = pd.MultiIndex.from_product([dates, symbols], names=["decision_ts", "symbol"])
    x = np.tile(np.linspace(-1.0, 1.0, len(symbols)), len(dates))
    frame = pd.DataFrame(
        {
            "x": x,
            "combo_signal": 0.5 + 2.0 * x,
            "forward_return": x * 0.01,
            "strategy_forward_return": x * 0.01,
        },
        index=index,
    )
    fold = WalkForwardFold(
        fold_idx=0,
        train_start=dates[0],
        train_end=dates[13],
        test_start=dates[15],
        test_end=dates[19],
    )
    return frame, fold


def test_ridge_replica_is_train_only_and_records_inner_splits():
    frame, fold = _ridge_frame()
    first = substitution.fit_walk_forward_ridge_replicas(
        frame,
        [fold],
        candidate_id="candidate",
        model_features={"model": ("x",)},
    )
    mutated = frame.copy()
    test_mask = mutated.index.get_level_values("decision_ts") >= fold.test_start
    mutated.loc[test_mask, "combo_signal"] += 100.0
    second = substitution.fit_walk_forward_ridge_replicas(
        mutated,
        [fold],
        candidate_id="candidate",
        model_features={"model": ("x",)},
    )

    assert len(first.inner_scores) == len(substitution.ALPHA_GRID) * 3
    assert first.coefficients["selected_alpha"].nunique() == 1
    assert set(first.inner_scores["inner_split_idx"]) == {0, 1, 2}
    assert (
        pd.to_datetime(first.inner_scores["validation_start"], utc=True)
        - pd.to_datetime(first.inner_scores["inner_train_end"], utc=True)
    ).min() >= pd.Timedelta(days=2)
    np.testing.assert_allclose(
        first.predictions["replica_signal"], second.predictions["replica_signal"]
    )
    pd.testing.assert_frame_equal(first.coefficients, second.coefficients)


def test_fold_target_exports_full_train_and_test_before_replica_fit():
    frame, fold = _ridge_frame()
    signal_frame = frame.reset_index(level="symbol")
    signal_frame["feature__1d"] = signal_frame["x"]
    spec = factor_research.ComboSpec(
        combo_id="combo",
        track="track",
        panel_frequency="1d",
        return_horizon="1d",
        feature_names=("feature__1d",),
        weight_scheme="family_alpha_1",
    )
    targets = substitution.build_fold_target_signals(
        spec,
        signal_frame,
        [fold],
        weight_scheme="family_alpha_1",
        feature_families={"feature__1d": "family"},
        min_cross_section=5,
    )
    raw = pd.DataFrame(
        [{**_all_predictors(float(index % 10))} for index in range(len(frame))],
        index=frame.index,
    )
    common = substitution.build_fold_canonical_common_support(
        targets.signals,
        raw,
        min_cross_section=5,
    )
    ridge = substitution.fit_walk_forward_ridge_replicas(
        common.frame,
        [fold],
        candidate_id="candidate",
        model_features={"model": ("return_1d",)},
    )

    split_counts = targets.signals.groupby("split")["decision_ts"].nunique().to_dict()
    assert split_counts == {"test": 5, "train": 14}
    assert len(ridge.predictions) == 5 * 10


def _registered_replica_frame() -> tuple[pd.DataFrame, WalkForwardFold]:
    dates = pd.date_range("2025-01-01", periods=30, freq="1D", tz="UTC")
    symbols = [f"S{index:02d}" for index in range(6)]
    index = pd.MultiIndex.from_product(
        [dates, symbols], names=["decision_ts", "symbol"]
    )
    date_number = np.repeat(np.arange(len(dates), dtype=float), len(symbols))
    symbol_number = np.tile(
        np.linspace(-1.0, 1.0, len(symbols)), len(dates)
    )
    frame = pd.DataFrame(index=index)
    for column_number, column in enumerate(substitution.PRICE_VOLUME_COLUMNS):
        frame[column] = np.sin(
            0.11 * date_number + 0.37 * symbol_number + column_number * 0.07
        )
    frame["combo_signal"] = (
        0.7 * frame["return_1d"]
        + 0.4 * np.square(frame["realized_vol_2d"])
        - 0.3 * frame["log_dollar_volume_4d"]
    )
    frame["forward_return"] = frame["combo_signal"] * 0.01
    frame["strategy_forward_return"] = frame["forward_return"]
    fold = WalkForwardFold(
        fold_idx=0,
        train_start=dates[0],
        train_end=dates[23],
        test_start=dates[25],
        test_end=dates[29],
    )
    return frame, fold


def test_train_selected_registered_replica_uses_exact_selected_class_rows():
    decisions = pd.date_range("2025-01-01", periods=5, freq="1D", tz="UTC")
    rows = []
    for model_number, model_class in enumerate(
        substitution.REGISTERED_MODEL_CLASS_ORDER
    ):
        for decision_ts in decisions:
            for symbol_number, symbol in enumerate(("A", "B")):
                target = float(symbol_number * 2 - 1)
                replica = target * (0.1 + 0.2 * model_number)
                rows.append(
                    {
                        "candidate_id": "candidate",
                        "fold_idx": 0,
                        "decision_ts": decision_ts,
                        "symbol": symbol,
                        "model_class": model_class,
                        "target_signal": target,
                        "replica_signal": replica,
                        "residual_signal": target - replica,
                        "strategy_forward_return": target * 0.01,
                    }
                )
    predictions = pd.DataFrame(rows)
    selection = pd.DataFrame(
        [
            {
                "candidate_id": "candidate",
                "fold_idx": 0,
                "selected_model_class": "random_forest",
                "selection_source": "outer_train_inner_validation",
            }
        ]
    )

    selected = substitution.assemble_train_selected_registered_replica(
        predictions, selection
    )

    assert len(selected) == 10
    assert set(selected["source_model_class"]) == {"random_forest"}
    assert set(selected["model_id"]) == {"train_selected_registered_replica"}
    np.testing.assert_allclose(
        selected["replica_signal"], selected["target_signal"] * 0.5
    )
    residual_information = substitution.summarize_residual_incremental_information(
        selected,
        signal_equivalence_id="signal",
        min_cross_section=2,
    )
    assert residual_information.timeseries["raw_rank_ic"].tolist() == pytest.approx(
        [1.0] * 5
    )

    leaked = selection.assign(selection_source="stitched_outer_oos_r2")
    with pytest.raises(ValueError, match="not produced train-only"):
        substitution.assemble_train_selected_registered_replica(
            predictions, leaked
        )
    missing_class = predictions.loc[
        predictions["model_class"] != "hist_gbm"
    ]
    with pytest.raises(ValueError, match="missing a model class"):
        substitution.assemble_train_selected_registered_replica(
            missing_class, selection
        )


def test_registered_replicas_are_deterministic_and_outer_test_target_blind():
    frame, fold = _registered_replica_frame()
    ridge = substitution.fit_walk_forward_ridge_replicas(
        frame,
        [fold],
        candidate_id="candidate",
        model_features={"level2_full": substitution.PRICE_VOLUME_COLUMNS},
    )
    small_specs = {
        "hist_gbm": (
            {
                "max_depth": 2,
                "max_iter": 100,
                "learning_rate": 0.03,
            },
        ),
        "random_forest": (
            {
                "max_depth": 3,
                "n_estimators": 200,
            },
        ),
    }
    first = substitution.fit_walk_forward_registered_replicas(
        frame,
        [fold],
        candidate_id="candidate",
        frozen_ridge_predictions=ridge.predictions,
        frozen_ridge_inner_scores=ridge.inner_scores,
        allow_model_subset=True,
        model_specs=small_specs,
    )
    second = substitution.fit_walk_forward_registered_replicas(
        frame,
        [fold],
        candidate_id="candidate",
        frozen_ridge_predictions=ridge.predictions,
        frozen_ridge_inner_scores=ridge.inner_scores,
        allow_model_subset=True,
        model_specs=small_specs,
        fit_workers=2,
    )
    pd.testing.assert_frame_equal(first.fold_selection, second.fold_selection)
    pd.testing.assert_frame_equal(
        first.class_predictions, second.class_predictions
    )
    assert set(first.class_predictions["model_class"]) == {
        "linear_ridge",
        "hist_gbm",
        "random_forest",
    }
    assert set(first.fold_selection["selection_source"]) == {
        "outer_train_inner_validation"
    }
    diagnostic_summary = substitution.summarize_registered_model_diagnostics(
        first.model_diagnostics,
        expected_fold_count=1,
    )
    assert len(diagnostic_summary) == 3
    assert diagnostic_summary["selected_fold_count"].sum() == 1
    assert set(diagnostic_summary["fold_count"]) == {1}
    assert diagnostic_summary.loc[
        diagnostic_summary["model_class"] != "linear_ridge",
        "mean_train_minus_oos_r2",
    ].notna().all()
    with pytest.raises(ValueError, match="missing a class"):
        substitution.summarize_registered_model_diagnostics(
            first.model_diagnostics.loc[
                first.model_diagnostics["model_class"] != "hist_gbm"
            ],
            expected_fold_count=1,
        )

    mutated = frame.copy()
    test_mask = (
        mutated.index.get_level_values("decision_ts") >= fold.test_start
    )
    mutated.loc[test_mask, "combo_signal"] += 10.0
    mutated_ridge = ridge.predictions.copy()
    mutated_ridge["target_signal"] += 10.0
    mutated_ridge["residual_signal"] = (
        mutated_ridge["target_signal"] - mutated_ridge["replica_signal"]
    )
    outer_mutated = substitution.fit_walk_forward_registered_replicas(
        mutated,
        [fold],
        candidate_id="candidate",
        frozen_ridge_predictions=mutated_ridge,
        frozen_ridge_inner_scores=ridge.inner_scores,
        allow_model_subset=True,
        model_specs=small_specs,
    )
    pd.testing.assert_frame_equal(
        first.fold_selection, outer_mutated.fold_selection
    )
    for model_class in substitution.REGISTERED_MODEL_CLASS_ORDER:
        original_replica = first.class_predictions.loc[
            first.class_predictions["model_class"] == model_class,
            "replica_signal",
        ].to_numpy()
        mutated_replica = outer_mutated.class_predictions.loc[
            outer_mutated.class_predictions["model_class"] == model_class,
            "replica_signal",
        ].to_numpy()
        np.testing.assert_allclose(original_replica, mutated_replica)

    outside_registry = {
        **small_specs,
        "random_forest": ({"max_depth": 9, "n_estimators": 200},),
    }
    with pytest.raises(ValueError, match="unknown configuration"):
        substitution.fit_walk_forward_registered_replicas(
            frame,
            [fold],
            candidate_id="candidate",
            frozen_ridge_predictions=ridge.predictions,
            frozen_ridge_inner_scores=ridge.inner_scores,
            allow_model_subset=True,
            model_specs=outside_registry,
        )

    frozen_mismatch = ridge.predictions.copy()
    frozen_mismatch["replica_signal"] += 1e-4
    frozen_mismatch["residual_signal"] = (
        frozen_mismatch["target_signal"] - frozen_mismatch["replica_signal"]
    )
    with pytest.raises(
        ValueError, match="recomputed level2_full ridge replica differs"
    ):
        substitution.fit_walk_forward_registered_replicas(
            frame,
            [fold],
            candidate_id="candidate",
            frozen_ridge_predictions=frozen_mismatch,
            frozen_ridge_inner_scores=ridge.inner_scores,
            allow_model_subset=True,
            model_specs=small_specs,
        )


def test_frozen_level2_full_ridge_reproduction_audit_and_exact_replication():
    frame, fold = _ridge_frame()
    ridge = substitution.fit_walk_forward_ridge_replicas(
        frame,
        [fold],
        candidate_id="candidate",
        model_features={"level2_full": ("x",)},
        alpha_grid=(0.0,),
    )
    audit = substitution.audit_frozen_level2_full_ridge_reproduction(
        frame,
        [fold],
        candidate_id="candidate",
        frozen_predictions=ridge.predictions,
        frozen_inner_scores=ridge.inner_scores,
        feature_columns=("x",),
    )

    assert audit.iloc[0]["reproduction_status"] == "pass"
    assert audit.iloc[0]["max_abs_prediction_error"] == pytest.approx(0.0)
    assert ridge.predictions["residual_signal"].abs().max() < 1e-12

    broken_predictions = ridge.predictions.copy()
    broken_predictions["selected_alpha"] += 1.0
    with pytest.raises(ValueError, match="selected alpha differs"):
        substitution.audit_frozen_level2_full_ridge_reproduction(
            frame,
            [fold],
            candidate_id="candidate",
            frozen_predictions=broken_predictions,
            frozen_inner_scores=ridge.inner_scores,
            feature_columns=("x",),
        )


def test_registered_hyperparameter_ties_follow_blueprint_tuple_order():
    low_depth_high_rate = {
        "max_depth": 2,
        "max_iter": 100,
        "learning_rate": 0.1,
    }
    high_depth_low_rate = {
        "max_depth": 3,
        "max_iter": 100,
        "learning_rate": 0.03,
    }

    assert substitution._registered_tie_key(
        "hist_gbm", low_depth_high_rate
    ) < substitution._registered_tie_key("hist_gbm", high_depth_low_rate)


def test_train_selected_composite_preserves_orthogonal_persistent_residual_information():
    decisions = pd.date_range("2025-01-01", periods=45, freq="1D", tz="UTC")
    symbols = [f"S{index:02d}" for index in range(10)]
    x = np.linspace(-1.0, 1.0, len(symbols))
    z = np.asarray([-1.0, 1.0] * 5)
    z = z - np.dot(z, x) / np.dot(x, x) * x
    rows = []
    for model_class in substitution.REGISTERED_MODEL_CLASS_ORDER:
        for decision_number, decision_ts in enumerate(decisions):
            for symbol_number, (symbol, feature_value, orthogonal_value) in enumerate(
                zip(symbols, x, z, strict=True)
            ):
                target = feature_value + orthogonal_value
                rows.append(
                    {
                        "candidate_id": "candidate",
                        "fold_idx": 0,
                        "decision_ts": decision_ts,
                        "symbol": symbol,
                        "model_class": model_class,
                        "target_signal": target,
                        "replica_signal": feature_value,
                        "residual_signal": orthogonal_value,
                        "strategy_forward_return": (
                            orthogonal_value * 0.01
                            + 0.002
                            * np.sin(
                                0.31 * decision_number + 0.73 * symbol_number
                            )
                        ),
                    }
                )
    selection = pd.DataFrame(
        [
            {
                "candidate_id": "candidate",
                "fold_idx": 0,
                "selected_model_class": "hist_gbm",
                "selection_source": "outer_train_inner_validation",
            }
        ]
    )
    selected = substitution.assemble_train_selected_registered_replica(
        pd.DataFrame(rows), selection
    )
    result = substitution.summarize_residual_incremental_information(
        selected,
        signal_equivalence_id="signal",
        min_cross_section=5,
        overlap_lags=0,
    )

    assert result.summary.iloc[0]["ic_mean"] > 0.8
    assert result.summary.iloc[0]["hac_t_stat"] > 0.0


def test_registered_replica_exposes_tree_overfit_on_deterministic_noise(monkeypatch):
    frame, fold = _registered_replica_frame()
    rng = np.random.default_rng(20260725)
    frame["combo_signal"] = rng.normal(size=len(frame))
    ridge = substitution.fit_walk_forward_ridge_replicas(
        frame,
        [fold],
        candidate_id="noise",
        model_features={"level2_full": substitution.PRICE_VOLUME_COLUMNS},
    )
    registered = substitution.registered_replica_model_specs()
    result = substitution.fit_walk_forward_registered_replicas(
        frame,
        [fold],
        candidate_id="noise",
        frozen_ridge_predictions=ridge.predictions,
        frozen_ridge_inner_scores=ridge.inner_scores,
        allow_model_subset=True,
        model_specs={
            "hist_gbm": (registered["hist_gbm"][0],),
            "random_forest": (registered["random_forest"][-1],),
        },
    )
    tree = result.model_diagnostics.loc[
        result.model_diagnostics["model_class"] == "random_forest"
    ].iloc[0]
    assert tree["outer_train_r2"] > tree["outer_oos_r2"]

    monkeypatch.setattr(substitution.sklearn, "__version__", "unexpected")
    with pytest.raises(ValueError, match="version mismatch"):
        substitution.fit_walk_forward_registered_replicas(
            frame,
            [fold],
            candidate_id="noise",
            frozen_ridge_predictions=ridge.predictions,
            frozen_ridge_inner_scores=ridge.inner_scores,
            allow_model_subset=True,
            model_specs={
                "hist_gbm": (registered["hist_gbm"][0],),
                "random_forest": (registered["random_forest"][0],),
            },
        )


def _residual_prediction_fixture(
    candidate_id: str,
    *,
    residual_scale: float = 1.0,
) -> pd.DataFrame:
    decisions = pd.date_range("2025-01-01", periods=5, freq="4h", tz="UTC")
    return_signs = (1.0, 1.0, -1.0, 1.0, 1.0)
    rows = []
    for decision_number, decision_ts in enumerate(decisions):
        for symbol_number, symbol in enumerate(("A", "B")):
            target = float(symbol_number * 2 - 1)
            replica = target * (1.0 - residual_scale)
            future_return = target * return_signs[decision_number] * 0.01
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "model_id": "level2_full",
                    "fold_idx": 0,
                    "decision_ts": decision_ts,
                    "symbol": symbol,
                    "target_signal": target,
                    "replica_signal": replica,
                    "residual_signal": target - replica,
                    "strategy_forward_return": future_return,
                }
            )
    return pd.DataFrame(rows)


def test_residual_information_minimal_end_to_end_and_negative_effect():
    positive = substitution.prepare_level2_full_residual_predictions(
        _residual_prediction_fixture("positive")
    )
    positive_result = substitution.summarize_residual_incremental_information(
        positive,
        signal_equivalence_id="signal_001",
        min_cross_section=2,
    )

    assert positive_result.timeseries["raw_rank_ic"].tolist() == pytest.approx(
        [1.0, 1.0, -1.0, 1.0, 1.0]
    )
    assert positive_result.summary.iloc[0]["effect_sign"] == "positive"
    assert positive_result.summary.iloc[0]["test_status"] == "valid"
    assert positive_result.summary.iloc[0]["raw_two_sided_p_value"] == pytest.approx(
        0.005613231495104954
    )

    negative_input = positive.copy()
    negative_input["strategy_forward_return"] *= -1.0
    negative_result = substitution.summarize_residual_incremental_information(
        negative_input,
        signal_equivalence_id="signal_002",
        min_cross_section=2,
    )
    assert negative_result.timeseries["raw_rank_ic"].tolist() == pytest.approx(
        [-1.0, -1.0, 1.0, -1.0, -1.0]
    )
    assert negative_result.summary.iloc[0]["effect_sign"] == "negative"


def test_residual_information_tracks_small_and_constant_cross_sections():
    frame = _residual_prediction_fixture("candidate")
    frame.loc[frame["decision_ts"] == frame["decision_ts"].min(), "residual_signal"] = 0.0
    frame.loc[frame["decision_ts"] == frame["decision_ts"].min(), "replica_signal"] = frame.loc[
        frame["decision_ts"] == frame["decision_ts"].min(), "target_signal"
    ]
    prepared = substitution.prepare_level2_full_residual_predictions(frame)

    result = substitution.summarize_residual_incremental_information(
        prepared,
        signal_equivalence_id="signal_001",
        min_cross_section=3,
    )

    assert set(result.timeseries["status"]) == {"small_cross_section"}
    assert result.summary.iloc[0]["test_status"] == "invalid"


def test_residual_equivalence_deduplicates_aliases_and_checks_replica_lineage():
    first = substitution.prepare_level2_full_residual_predictions(
        _residual_prediction_fixture("candidate_a", residual_scale=0.5)
    )
    alias = first.copy()
    alias["candidate_id"] = "candidate_b"
    audit = substitution.audit_residual_signal_equivalence(
        pd.concat([first, alias], ignore_index=True)
    )

    assert len(audit.groups) == 1
    assert audit.groups.iloc[0]["alias_count"] == 2
    assert set(audit.mapping["candidate_id"]) == {"candidate_a", "candidate_b"}

    broken = alias.copy()
    broken["replica_signal"] += 0.1
    broken["residual_signal"] = broken["target_signal"] - broken["replica_signal"]
    with pytest.raises(ValueError, match="differ in replica_signal"):
        substitution.audit_residual_signal_equivalence(
            pd.concat([first, broken], ignore_index=True)
        )


def test_residual_preparation_rejects_stale_residual_and_duplicate_keys():
    frame = _residual_prediction_fixture("candidate")
    frame.loc[0, "residual_signal"] += 0.1
    with pytest.raises(ValueError, match="recorded residual_signal"):
        substitution.prepare_level2_full_residual_predictions(frame)

    duplicated = pd.concat([_residual_prediction_fixture("candidate")] * 2, ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        substitution.prepare_level2_full_residual_predictions(duplicated)


def test_residual_holm_requires_complete_family_and_keeps_negative_detection_neutral():
    summary = pd.DataFrame(
        [
            {
                "signal_equivalence_id": "signal_001",
                "raw_two_sided_p_value": 0.001,
                "test_status": "valid",
                "effect_sign": "negative",
            },
            {
                "signal_equivalence_id": "signal_002",
                "raw_two_sided_p_value": np.nan,
                "test_status": "invalid",
                "effect_sign": "undefined",
            },
        ]
    )
    groups = pd.DataFrame(
        [
            {"signal_equivalence_id": "signal_001", "canonical_candidate_id": "a", "alias_count": 1},
            {"signal_equivalence_id": "signal_002", "canonical_candidate_id": "b", "alias_count": 1},
        ]
    )
    result = substitution.apply_residual_information_holm(
        summary,
        groups,
        pd.DataFrame(
            [
                {"candidate_id": "a", "signal_equivalence_id": "signal_001", "canonical_candidate_id": "a"},
                {"candidate_id": "b", "signal_equivalence_id": "signal_002", "canonical_candidate_id": "b"},
            ]
        ),
        expected_candidate_count=2,
    )

    assert result.loc[0, "incremental_information_label"] == "incremental_information_detected"
    assert result.loc[0, "effect_sign"] == "negative"
    assert result.loc[1, "incremental_information_label"] == "incremental_information_test_invalid"
    assert result["holm_family_size"].tolist() == [2, 2]
    with pytest.raises(ValueError, match="incomplete"):
        substitution.apply_residual_information_holm(
            summary.iloc[:1],
            groups,
            pd.DataFrame(
                [
                    {"candidate_id": "a", "signal_equivalence_id": "signal_001", "canonical_candidate_id": "a"},
                    {"candidate_id": "b", "signal_equivalence_id": "signal_002", "canonical_candidate_id": "b"},
                ]
            ),
            expected_candidate_count=2,
        )
    with pytest.raises(ValueError, match="candidate coverage is incomplete"):
        substitution.apply_residual_information_holm(
            summary.iloc[:1],
            groups.iloc[:1],
            pd.DataFrame(
                [
                    {"candidate_id": "a", "signal_equivalence_id": "signal_001", "canonical_candidate_id": "a"},
                ]
            ),
            expected_candidate_count=2,
        )
    fake_group = groups.iloc[:1].copy()
    fake_group["alias_count"] = 96
    fake_mapping = pd.DataFrame(
        [
            {
                "candidate_id": "a",
                "signal_equivalence_id": "signal_001",
                "canonical_candidate_id": "a",
            }
        ]
    )
    with pytest.raises(ValueError, match="candidate mapping is incomplete"):
        substitution.apply_residual_information_holm(
            summary.iloc[:1], fake_group, fake_mapping, expected_candidate_count=96
        )


def _residual_family_fixture(
    *,
    horizon: str = "8h",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions = pd.date_range("2025-01-01", periods=5, freq="8h", tz="UTC")
    values = {
        "signal_a1": (1.0, 1.0, -1.0, 1.0, 1.0),
        "signal_a2": (1.0, -1.0, -1.0, 1.0, 1.0),
        "signal_b1": (-1.0, 1.0, 1.0, 1.0, 1.0),
    }
    rows = []
    for signal_id, series in values.items():
        for decision_ts, value in zip(decisions, series):
            rows.append(
                {
                    "signal_equivalence_id": signal_id,
                    "fold_idx": 0,
                    "decision_ts": decision_ts,
                    "status": "ok",
                    "raw_rank_ic": value,
                }
            )
    mapping = pd.DataFrame(
        [
            {
                "candidate_id": "candidate_a1",
                "signal_equivalence_id": "signal_a1",
                "horizon": horizon,
                "track": "track_a",
            },
            {
                "candidate_id": "candidate_a1_alias",
                "signal_equivalence_id": "signal_a1",
                "horizon": horizon,
                "track": "track_a",
            },
            {
                "candidate_id": "candidate_a2",
                "signal_equivalence_id": "signal_a2",
                "horizon": horizon,
                "track": "track_a",
            },
            {
                "candidate_id": "candidate_b1",
                "signal_equivalence_id": "signal_b1",
                "horizon": horizon,
                "track": "track_b",
            },
        ]
    )
    return pd.DataFrame(rows), mapping


def test_residual_family_information_equal_weights_tracks_end_to_end():
    timeseries, mapping = _residual_family_fixture()

    result = substitution.summarize_horizon_residual_family_information(
        timeseries,
        mapping,
        horizon="8h",
        expected_unique_signal_count=3,
    )

    track = result.track_timeseries.pivot(
        index="decision_ts", columns="track", values="track_residual_ic"
    )
    assert track["track_a"].tolist() == pytest.approx([1.0, 0.0, -1.0, 1.0, 1.0])
    assert track["track_b"].tolist() == pytest.approx([-1.0, 1.0, 1.0, 1.0, 1.0])
    assert result.family_timeseries["family_residual_ic"].tolist() == pytest.approx(
        [0.0, 0.5, 0.0, 1.0, 1.0]
    )
    assert result.family_timeseries["equal_unique_signal_residual_ic"].tolist() == pytest.approx(
        [1.0 / 3.0, 1.0 / 3.0, -1.0 / 3.0, 1.0, 1.0]
    )
    summary = result.summary.iloc[0]
    assert summary["family_ic_mean"] == pytest.approx(0.5)
    assert summary["equal_unique_signal_ic_mean"] == pytest.approx(7.0 / 15.0)
    assert summary["unique_signal_count"] == 3
    assert summary["track_count"] == 2
    assert summary["ic_observation_count"] == 5
    assert summary["raw_one_sided_p_value"] < 0.5


def test_residual_family_information_alias_rows_do_not_change_track_weight():
    timeseries, mapping = _residual_family_fixture()
    without_alias = mapping.loc[mapping["candidate_id"] != "candidate_a1_alias"]

    first = substitution.summarize_horizon_residual_family_information(
        timeseries, mapping, horizon="8h", expected_unique_signal_count=3
    )
    second = substitution.summarize_horizon_residual_family_information(
        timeseries, without_alias, horizon="8h", expected_unique_signal_count=3
    )

    pd.testing.assert_frame_equal(first.track_timeseries, second.track_timeseries)
    pd.testing.assert_frame_equal(first.family_timeseries, second.family_timeseries)
    pd.testing.assert_frame_equal(first.summary, second.summary)


def test_residual_family_information_negative_direction_is_not_positive_evidence():
    timeseries, mapping = _residual_family_fixture()
    timeseries["raw_rank_ic"] *= -1.0

    result = substitution.summarize_horizon_residual_family_information(
        timeseries, mapping, horizon="8h", expected_unique_signal_count=3
    )

    summary = result.summary.iloc[0]
    assert summary["effect_sign"] == "negative"
    assert summary["raw_one_sided_p_value"] > 0.5


def test_residual_family_information_keeps_conflicting_weight_sensitivities():
    decisions = pd.date_range("2025-01-01", periods=5, freq="8h", tz="UTC")
    values = {
        "signal_a1": (1.0, 0.8, 0.6, 0.8, 1.0),
        "signal_b1": (-0.6, -0.4, -0.2, -0.4, -0.6),
        "signal_b2": (-0.6, -0.4, -0.2, -0.4, -0.6),
        "signal_b3": (-0.6, -0.4, -0.2, -0.4, -0.6),
    }
    rows = [
        {
            "signal_equivalence_id": signal_id,
            "fold_idx": 0,
            "decision_ts": decision_ts,
            "status": "ok",
            "raw_rank_ic": value,
        }
        for signal_id, series in values.items()
        for decision_ts, value in zip(decisions, series)
    ]
    mapping = pd.DataFrame(
        [
            {
                "signal_equivalence_id": signal_id,
                "horizon": "8h",
                "track": "track_a" if signal_id == "signal_a1" else "track_b",
            }
            for signal_id in values
        ]
    )

    result = substitution.summarize_horizon_residual_family_information(
        pd.DataFrame(rows),
        mapping,
        horizon="8h",
        expected_unique_signal_count=4,
    )

    assert result.summary.iloc[0]["family_ic_mean"] > 0.0
    assert result.summary.iloc[0]["equal_unique_signal_ic_mean"] < 0.0
    assert (result.family_timeseries["family_residual_ic"] > 0.0).all()
    assert (result.family_timeseries["equal_unique_signal_residual_ic"] < 0.0).all()


def test_residual_family_information_all_positive_ic_has_correct_one_sided_direction():
    timeseries, mapping = _residual_family_fixture()
    timeseries["raw_rank_ic"] = timeseries["raw_rank_ic"].map(
        lambda value: 0.2 if value < 0.0 else 1.0
    )

    result = substitution.summarize_horizon_residual_family_information(
        timeseries, mapping, horizon="8h", expected_unique_signal_count=3
    )

    assert (result.family_timeseries["family_residual_ic"] > 0.0).all()
    assert result.summary.iloc[0]["hac_t_stat"] > 0.0
    assert result.summary.iloc[0]["raw_one_sided_p_value"] < 0.5


@pytest.mark.parametrize("failure", ["missing_decision", "duplicate_key", "non_ok", "non_finite"])
def test_residual_family_information_fails_closed_on_unbalanced_or_invalid_input(failure):
    timeseries, mapping = _residual_family_fixture()
    if failure == "missing_decision":
        timeseries = timeseries.drop(timeseries.index[0])
        match = "balanced"
    elif failure == "duplicate_key":
        timeseries = pd.concat([timeseries, timeseries.iloc[[0]]], ignore_index=True)
        match = "duplicate"
    elif failure == "non_ok":
        timeseries.loc[0, "status"] = "small_cross_section"
        match = "non-ok"
    else:
        timeseries.loc[0, "raw_rank_ic"] = np.nan
        match = "non-finite"

    with pytest.raises(ValueError, match=match):
        substitution.summarize_horizon_residual_family_information(
            timeseries, mapping, horizon="8h", expected_unique_signal_count=3
        )


def test_residual_family_information_fails_closed_on_bad_track_mapping():
    timeseries, mapping = _residual_family_fixture()
    conflict = mapping.iloc[[0]].copy()
    conflict["track"] = "other_track"

    with pytest.raises(ValueError, match="multiple horizons or tracks"):
        substitution.summarize_horizon_residual_family_information(
            timeseries,
            pd.concat([mapping, conflict], ignore_index=True),
            horizon="8h",
            expected_unique_signal_count=3,
        )

    with pytest.raises(ValueError, match="unique signal count mismatch"):
        substitution.summarize_horizon_residual_family_information(
            timeseries,
            mapping.loc[mapping["signal_equivalence_id"] != "signal_b1"],
            horizon="8h",
            expected_unique_signal_count=3,
        )


def test_horizon_residual_family_holm_closes_exactly_four_horizons():
    summary = pd.DataFrame(
        [
            {
                "horizon": horizon,
                "raw_one_sided_p_value": p_value,
                "test_status": "valid",
                "effect_sign": "positive",
            }
            for horizon, p_value in zip(
                ("4h", "8h", "12h", "1d"),
                (0.01, 0.02, 0.03, 0.04),
            )
        ]
    )

    result = substitution.apply_horizon_residual_family_holm(summary)

    assert result["holm_family_size"].tolist() == [4, 4, 4, 4]
    assert result["global_bonferroni_intersection_p_value"].tolist() == pytest.approx(
        [0.04] * 4
    )
    assert set(result["global_horizon_family_label"]) == {
        "global_any_horizon_incremental_information_detected"
    }
    assert result.loc[result["horizon"] == "4h", "holm_adjusted_p_value"].iloc[0] == pytest.approx(
        0.04
    )

    with pytest.raises(ValueError, match="incomplete"):
        substitution.apply_horizon_residual_family_holm(summary.iloc[:3])
    with pytest.raises(ValueError, match="duplicate"):
        substitution.apply_horizon_residual_family_holm(
            pd.concat([summary.iloc[:3], summary.iloc[[0]]], ignore_index=True)
        )
    with pytest.raises(ValueError, match="exactly four"):
        substitution.apply_horizon_residual_family_holm(
            summary,
            expected_horizons=("4h", "8h", "12h", "1d", "2d"),
        )


def test_horizon_residual_family_holm_handles_boundary_and_invalid_global_test():
    boundary = pd.DataFrame(
        [
            {
                "horizon": horizon,
                "raw_one_sided_p_value": p_value,
                "test_status": "valid",
                "effect_sign": "positive",
            }
            for horizon, p_value in zip(
                ("4h", "8h", "12h", "1d"),
                (0.0125, 0.5, 0.5, 0.5),
            )
        ]
    )
    boundary_result = substitution.apply_horizon_residual_family_holm(boundary)
    assert boundary_result["global_bonferroni_intersection_p_value"].iloc[0] == pytest.approx(
        0.05
    )
    assert set(boundary_result["global_horizon_family_label"]) == {
        "global_any_horizon_incremental_information_detected"
    }

    invalid = boundary.copy()
    invalid.loc[invalid["horizon"] == "1d", "raw_one_sided_p_value"] = np.nan
    invalid.loc[invalid["horizon"] == "1d", "test_status"] = "invalid"
    invalid.loc[invalid["horizon"] == "1d", "effect_sign"] = "undefined"
    invalid_result = substitution.apply_horizon_residual_family_holm(invalid)
    assert np.isnan(invalid_result["global_bonferroni_intersection_p_value"].iloc[0])
    assert set(invalid_result["global_horizon_family_label"]) == {
        "global_horizon_family_test_invalid"
    }
    assert (
        invalid_result.loc[
            invalid_result["horizon"] == "1d", "horizon_family_label"
        ].iloc[0]
        == "horizon_family_test_invalid"
    )
    for non_finite in (np.inf, -np.inf):
        broken = boundary.copy()
        broken.loc[broken["horizon"] == "1d", "raw_one_sided_p_value"] = non_finite
        with pytest.raises(ValueError, match="no finite"):
            substitution.apply_horizon_residual_family_holm(broken)
    for outside in (-0.01, 1.01):
        broken = boundary.copy()
        broken.loc[broken["horizon"] == "1d", "raw_one_sided_p_value"] = outside
        with pytest.raises(ValueError, match="between 0 and 1"):
            substitution.apply_horizon_residual_family_holm(broken)


def test_residual_candidate_sensitivity_preserves_frozen_candidate_evidence():
    timeseries, mapping = _residual_family_fixture()
    family = substitution.summarize_horizon_residual_family_information(
        timeseries, mapping, horizon="8h", expected_unique_signal_count=3
    )
    frozen = pd.DataFrame(
        [
            {
                "signal_equivalence_id": signal_id,
                "raw_two_sided_p_value": raw_p,
                "holm_adjusted_p_value": holm_p,
                "incremental_information_label": label,
            }
            for signal_id, raw_p, holm_p, label in (
                ("signal_a1", 0.01, 0.03, "incremental_information_detected"),
                ("signal_a2", 0.02, 0.06, "incremental_information_not_detected"),
                ("signal_b1", 0.03, 0.09, "incremental_information_not_detected"),
            )
        ]
    )

    result = substitution.summarize_residual_candidate_sensitivity(
        family.signal_summary,
        frozen,
        horizon="8h",
        expected_unique_signal_count=3,
    )

    assert len(result.detail) == 3
    assert result.summary.iloc[0]["raw_two_sided_p_le_0p05_count"] == 3
    assert result.summary.iloc[0]["prior_holm_detected_count"] == 1
    assert result.summary.iloc[0]["prior_holm_not_detected_count"] == 2

    with pytest.raises(ValueError, match="disagree"):
        substitution.summarize_residual_candidate_sensitivity(
            family.signal_summary,
            frozen.iloc[:2],
            horizon="8h",
            expected_unique_signal_count=3,
        )


def test_residual_path_ignores_r2_gate_and_rejects_old_return_only_input():
    low = _residual_prediction_fixture("low_r2").assign(stitched_oos_r2=0.09)
    high = _residual_prediction_fixture("high_r2").assign(stitched_oos_r2=0.11)
    prepared = substitution.prepare_level2_full_residual_predictions(
        pd.concat([low, high], ignore_index=True)
    )

    assert set(prepared["candidate_id"]) == {"low_r2", "high_r2"}
    assert prepared.groupby("candidate_id").size().to_dict() == {"high_r2": 10, "low_r2": 10}

    old_only = low.drop(columns="strategy_forward_return").rename(
        columns={"stitched_oos_r2": "forward_return"}
    )
    with pytest.raises(ValueError, match="strategy_forward_return"):
        substitution.prepare_level2_full_residual_predictions(old_only)


def test_executable_return_lineage_uses_t_plus_one_minute_opens():
    prepared = substitution.prepare_level2_full_residual_predictions(
        _residual_prediction_fixture("candidate")
    )
    decisions = pd.DatetimeIndex(prepared["decision_ts"].unique())
    opens_by_symbol = {}
    for symbol_number, symbol in enumerate(("A", "B"), start=1):
        timestamps = sorted(
            set(decisions + pd.Timedelta(minutes=1)).union(
                decisions + pd.Timedelta(hours=4, minutes=1)
            )
        )
        values = pd.Series(100.0 * symbol_number, index=pd.DatetimeIndex(timestamps))
        for decision_number, decision_ts in enumerate(decisions):
            entry_ts = decision_ts + pd.Timedelta(minutes=1)
            exit_ts = decision_ts + pd.Timedelta(hours=4, minutes=1)
            expected_return = prepared.loc[
                (prepared["decision_ts"] == decision_ts) & (prepared["symbol"] == symbol),
                "strategy_forward_return",
            ].iloc[0]
            values.loc[exit_ts] = values.loc[entry_ts] * (1.0 + expected_return)
        opens_by_symbol[symbol] = values

    audit = substitution.audit_precomputed_executable_return_lineage(
        prepared,
        opens_by_symbol,
        horizon="4h",
    )

    assert audit.iloc[0]["lineage_status"] == "pass"
    assert audit.iloc[0]["max_abs_return_error"] == pytest.approx(0.0, abs=1e-12)
    broken = prepared.copy()
    broken.loc[0, "strategy_forward_return"] += 0.01
    with pytest.raises(ValueError, match="executable-open return"):
        substitution.audit_precomputed_executable_return_lineage(
            broken,
            opens_by_symbol,
            horizon="4h",
        )


def test_zero_mechanical_overlap_preserves_positive_sample_rule_hac_lag():
    frame = _residual_prediction_fixture("candidate")
    symbols = [f"S{index:02d}" for index in range(10)]
    expanded = []
    for decision_number, decision_ts in enumerate(
        pd.date_range("2025-01-01", periods=45, freq="4h", tz="UTC")
    ):
        for symbol_number, symbol in enumerate(symbols):
            signal = float(symbol_number)
            expanded.append(
                {
                    "candidate_id": "candidate",
                    "model_id": "level2_full",
                    "fold_idx": 0,
                    "decision_ts": decision_ts,
                    "symbol": symbol,
                    "target_signal": signal,
                    "replica_signal": 0.0,
                    "residual_signal": signal,
                    "strategy_forward_return": signal + (decision_number % 3) * symbol_number * 0.01,
                }
            )
    prepared = substitution.prepare_level2_full_residual_predictions(pd.DataFrame(expanded))
    result = substitution.summarize_residual_incremental_information(
        prepared,
        signal_equivalence_id="signal_001",
        min_cross_section=5,
        overlap_lags=0,
    )

    assert result.summary.iloc[0]["hac_lags"] > 0


def test_reconstruct_folds_preserves_existing_mixed_frequency_oos_boundaries():
    holdings = pd.DataFrame(
        [
            {
                "combo_id": "combo",
                "weight_scheme": "family_alpha_1",
                "fold_idx": fold_idx,
                "decision_ts": decision,
                "symbol": symbol,
            }
            for fold_idx, decisions in enumerate(
                (
                    pd.date_range("2025-07-01 12:15", periods=4, freq="12h", tz="UTC"),
                    pd.date_range("2025-07-03 12:15", periods=2, freq="1D", tz="UTC"),
                )
            )
            for decision in decisions
            for symbol in ("A", "B")
        ]
    )

    folds = substitution.reconstruct_folds_from_oos_holdings(
        holdings,
        combo_id="combo",
        weight_scheme="family_alpha_1",
        train_days=180,
        embargo_days=1,
    )

    assert len(folds) == 2
    assert folds[0].test_start == pd.Timestamp("2025-07-01 12:15", tz="UTC")
    assert folds[0].test_end == pd.Timestamp("2025-07-03 00:15", tz="UTC")
    assert folds[0].train_end == pd.Timestamp("2025-06-29 12:15", tz="UTC")
    assert folds[0].train_start == pd.Timestamp("2025-01-01 12:15", tz="UTC")
    assert folds[1].test_start == pd.Timestamp("2025-07-03 12:15", tz="UTC")


def test_replication_r2_gate_and_difficulty_are_formal_outputs():
    frame, fold = _ridge_frame()
    artifacts = substitution.fit_walk_forward_ridge_replicas(
        frame,
        [fold],
        candidate_id="candidate",
        model_features={"level1__x": ("x",)},
    )
    summary, fold_metrics, decision_metrics = substitution.summarize_replication_metrics(
        artifacts.predictions
    )
    classification = substitution.classify_replication_difficulty(summary)

    assert summary.iloc[0]["stitched_oos_r2"] > 0.99
    assert bool(summary.iloc[0]["replication_gate_pass"])
    assert fold_metrics.iloc[0]["oos_r2"] > 0.99
    assert decision_metrics["top_bottom_overlap"].eq(1.0).all()
    assert classification.iloc[0]["replication_difficulty"] == "level1_single_proxy_partial_replication"


def test_precomputed_signal_replay_matches_hand_computed_two_bucket_returns():
    dates = pd.date_range("2025-01-01", periods=12, freq="1D", tz="UTC")
    symbols = ["A", "B", "C", "D"]
    fold = WalkForwardFold(
        fold_idx=0,
        train_start=dates[0],
        train_end=dates[5],
        test_start=dates[7],
        test_end=dates[11],
    )
    rows = []
    signals = {"A": -2.0, "B": -1.0, "C": 1.0, "D": 2.0}
    returns = {"A": -0.02, "B": -0.01, "C": 0.01, "D": 0.02}
    for decision in dates[7:]:
        for symbol in symbols:
            rows.append(
                {
                    "candidate_id": "candidate",
                    "model_id": "original",
                    "signal_id": "original",
                    "signal_type": "original",
                    "replay_combo_id": "candidate__original",
                    "track": "track",
                    "weight_scheme": "weight",
                    "panel_frequency": "1d",
                    "return_horizon": "1d",
                    "component_features": "a|b",
                    "fold_idx": 0,
                    "decision_ts": decision,
                    "symbol": symbol,
                    "signal_value": signals[symbol],
                    "forward_return": returns[symbol],
                    "strategy_forward_return": returns[symbol],
                }
            )
    replay = substitution.evaluate_precomputed_oos_signals(
        pd.DataFrame(rows),
        [fold],
        {"train_days": 6, "test_days": 5, "embargo_days": 1, "step_days": 5},
        n_buckets=2,
        cost_multipliers=(1.0,),
        taker_fee_rate=0.0,
        frequency_periods_per_year={"1d": 365},
    )

    assert replay.timeseries["gross_return"].tolist() == pytest.approx([0.015] * 5)
    assert replay.summary.iloc[0]["scored_decision_count"] == 5
    assert set(replay.holdings.groupby("decision_ts").size()) == {4}


def test_precomputed_twelve_hour_signal_replay_matches_hand_computation():
    dates = pd.date_range("2025-01-01", periods=24, freq="12h", tz="UTC")
    symbols = ["A", "B", "C", "D"]
    fold = WalkForwardFold(
        fold_idx=0,
        train_start=dates[0],
        train_end=dates[11],
        test_start=dates[14],
        test_end=dates[23],
    )
    signals = {"A": -2.0, "B": -1.0, "C": 1.0, "D": 2.0}
    returns = {"A": -0.02, "B": -0.01, "C": 0.01, "D": 0.02}
    rows = [
        {
            "candidate_id": "candidate_12h",
            "model_id": "original",
            "signal_id": "original",
            "signal_type": "original",
            "replay_combo_id": "candidate_12h__original",
            "track": "track",
            "weight_scheme": "weight",
            "panel_frequency": "12h",
            "return_horizon": "12h",
            "component_features": "a|b",
            "fold_idx": 0,
            "decision_ts": decision,
            "symbol": symbol,
            "signal_value": signals[symbol],
            "forward_return": returns[symbol],
            "strategy_forward_return": returns[symbol],
        }
        for decision in dates[14:]
        for symbol in symbols
    ]

    replay = substitution.evaluate_precomputed_oos_signals(
        pd.DataFrame(rows),
        [fold],
        {"train_days": 6, "test_days": 5, "embargo_days": 1, "step_days": 5},
        n_buckets=2,
        cost_multipliers=(1.0,),
        taker_fee_rate=0.0,
        frequency_periods_per_year={"12h": 730},
    )

    assert replay.timeseries["gross_return"].tolist() == pytest.approx([0.015] * 10)
    assert replay.summary.iloc[0]["scored_decision_count"] == 10
    assert replay.summary.iloc[0]["panel_frequency"] == "12h"
    assert set(replay.holdings.groupby("decision_ts").size()) == {4}


def test_executable_precomputed_replay_matches_five_decision_quantity_ledger():
    decisions = pd.date_range("2025-01-10", periods=5, freq="1D", tz="UTC")
    fold = WalkForwardFold(
        fold_idx=0,
        train_start=pd.Timestamp("2024-07-01", tz="UTC"),
        train_end=pd.Timestamp("2025-01-08", tz="UTC"),
        test_start=decisions[0],
        test_end=decisions[-1],
    )
    entry_prices = {
        "A": [10.0, 11.0, 12.0, 13.0, 14.0],
        "B": [20.0, 19.0, 18.0, 17.0, 16.0],
    }
    exit_prices = {
        "A": [11.0, 12.0, 13.0, 14.0, 15.0],
        "B": [19.0, 18.0, 17.0, 16.0, 15.0],
    }
    rows = []
    for decision_number, decision_ts in enumerate(decisions):
        for symbol in ("A", "B"):
            entry = entry_prices[symbol][decision_number]
            exit_price = exit_prices[symbol][decision_number]
            initial_side = -1.0 if symbol == "A" else 1.0
            signal = initial_side if decision_number >= 2 else -initial_side
            execution_ts = decision_ts + pd.Timedelta(minutes=1)
            next_execution_ts = execution_ts + pd.Timedelta(days=1)
            rows.append(
                {
                    "candidate_id": "candidate",
                    "model_id": "model",
                    "signal_id": "replica__model",
                    "signal_type": "replica",
                    "replay_combo_id": "candidate__replica__model",
                    "track": "track",
                    "weight_scheme": "weight",
                    "panel_frequency": "1d",
                    "return_horizon": "1d",
                    "component_features": "feature",
                    "fold_idx": 0,
                    "decision_ts": decision_ts,
                    "symbol": symbol,
                    "signal_value": signal,
                    "signal_timeframes": "1d",
                    "native_bar_end_ts": decision_ts,
                    "signal_bar_end_ts": decision_ts,
                    "availability_ts": decision_ts,
                    "data_observed_ts": decision_ts,
                    "decision_interval": "1d",
                    "order_submit_ts": execution_ts,
                    "execution_ts": execution_ts,
                    "execution_open_time": execution_ts,
                    "next_execution_ts": next_execution_ts,
                    "holding_interval": "1d",
                    "exit_rule": "next_horizon_execution_open",
                    "score_order": "high_score_long_low_score_short",
                    "entry_price": entry,
                    "exit_price": exit_price,
                    "execution_price": entry,
                    "next_execution_price": exit_price,
                    "executable_return": exit_price / entry - 1.0,
                }
            )

    replay = substitution.evaluate_executable_precomputed_oos_signals(
        pd.DataFrame(rows),
        [fold],
        {"train_days": 180, "test_days": 5, "embargo_days": 1, "step_days": 5},
        n_buckets=2,
        min_cross_section=2,
        cost_multipliers=(1.0,),
        taker_fee_rate=0.0,
        frequency_periods_per_year={"1d": 365},
        horizon_deltas={"1d": pd.Timedelta(days=1)},
    )

    assert replay.timeseries["gross_return"].tolist() == pytest.approx(
        [0.075, 0.075, -(1.0 / 24.0 + 1.0 / 36.0), -(1.0 / 24.0 + 1.0 / 36.0), -(1.0 / 24.0 + 1.0 / 36.0)]
    )
    assert replay.timeseries["charged_turnover"].tolist() == pytest.approx(
        [1.0, 0.0, 2.05, 0.0, 0.625 + 15.0 / 36.0]
    )
    assert replay.orders.groupby("status").size().to_dict() == {
        "hold_unchanged": 6,
        "open": 2,
        "side_switch": 2,
        "terminal_close": 2,
    }
    quantities = replay.holdings.pivot_table(
        index="decision_ts", columns="symbol", values="signed_quantity", aggfunc="first"
    )
    assert quantities.loc[decisions[0], "A"] == pytest.approx(0.05)
    assert quantities.loc[decisions[1], "A"] == pytest.approx(0.05)
    assert quantities.loc[decisions[2], "A"] == pytest.approx(-1.0 / 24.0)
    assert quantities.loc[decisions[3], "A"] == pytest.approx(-1.0 / 24.0)
    assert replay.summary.iloc[0]["scored_decision_count"] == 5
    rules = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "market_min_qty": [1e-12, 1e-12],
            "market_step": [1e-12, 1e-12],
            "min_notional": [0.0, 0.0],
        }
    )
    l4_summary, l4_detail, _, l4_holdings = factor_research.live_like_executable_min_notional_replay(
        replay.holdings,
        rules,
        account_equity=1.0,
        target_gross_notional=1.0,
        exchange_leverage=1.0,
        taker_fee_rate=0.0,
        cost_multipliers=(1.0,),
        frequency_periods_per_year={"1d": 365},
        horizon_deltas={"1d": pd.Timedelta(days=1)},
    )
    assert l4_detail["gross_return"].tolist() == pytest.approx(replay.timeseries["gross_return"])
    assert l4_detail["charged_turnover"].tolist() == pytest.approx(
        replay.timeseries["charged_turnover"]
    )
    assert len(l4_detail) == 5
    assert not l4_holdings.empty


def test_target_manifest_keeps_requested_horizon_rows_without_deduplication():
    summary = pd.DataFrame(
        [
            {
                "combo_id": "combo",
                "track": "track",
                "weight_scheme": scheme,
                "panel_frequency": "1d",
                "return_horizon": "1d",
                "component_features": "a|b",
                "n_components": 2,
            }
            for scheme in ("family_alpha_1", "family_alpha_0p5", "family_alpha_0")
        ]
        + [
            {
                "combo_id": "other",
                "track": "track",
                "weight_scheme": "family_alpha_1",
                "panel_frequency": "12h",
                "return_horizon": "12h",
                "component_features": "c|d",
                "n_components": 2,
            }
        ]
    )

    manifest = substitution.build_substitution_target_manifest(summary, return_horizon="1d")

    assert len(manifest) == 3
    assert manifest["candidate_id"].nunique() == 3

    twelve_hour = substitution.build_substitution_target_manifest(
        summary, return_horizon="12h"
    )
    assert len(twelve_hour) == 1
    assert twelve_hour.iloc[0]["panel_frequency"] == "12h"


def test_ridge_inner_gap_is_parameterized_for_twelve_hour_horizon():
    dates = pd.date_range("2025-01-01", periods=40, freq="12h", tz="UTC")
    symbols = ["A", "B", "C", "D"]
    index = pd.MultiIndex.from_product(
        [dates, symbols], names=["decision_ts", "symbol"]
    )
    x = np.tile(np.linspace(-1.0, 1.0, len(symbols)), len(dates))
    frame = pd.DataFrame(
        {
            "x": x,
            "combo_signal": 0.2 + x,
            "forward_return": x * 0.01,
            "strategy_forward_return": x * 0.01,
        },
        index=index,
    )
    fold = WalkForwardFold(
        fold_idx=0,
        train_start=dates[0],
        train_end=dates[27],
        test_start=dates[30],
        test_end=dates[-1],
    )

    artifacts = substitution.fit_walk_forward_ridge_replicas(
        frame,
        [fold],
        candidate_id="candidate_12h",
        model_features={"model": ("x",)},
        inner_gap=pd.Timedelta(hours=12),
    )

    validation_start = pd.to_datetime(
        artifacts.inner_scores["validation_start"], utc=True
    )
    inner_train_end = pd.to_datetime(
        artifacts.inner_scores["inner_train_end"], utc=True
    )
    assert (validation_start - inner_train_end).min() >= pd.Timedelta(days=1)
    assert artifacts.coefficients["selected_alpha"].nunique() == 1


def test_replay_comparison_and_shadow_inference_use_formal_evidence():
    l3 = pd.DataFrame(
        [
            {
                "candidate_id": "candidate",
                "model_id": model_id,
                "signal_type": signal_type,
                "net_1x_sharpe": sharpe,
                "net_1x_annualized_return": 0.1,
                "net_1x_max_drawdown": -0.2,
                "net_1x_fold_positive_share": 0.6,
                "mean_charged_turnover": 0.5,
            }
            for model_id, signal_type, sharpe in (
                ("original", "original", 2.0),
                ("level1__x", "replica", 1.0),
                ("level1__x", "residual", 1.5),
            )
        ]
    )
    l4_rows = []
    for model_id, signal_type, sharpe in (
        ("original", "original", 1.5),
        ("level1__x", "replica", 0.75),
        ("level1__x", "residual", 1.0),
    ):
        l4_rows.append(
            {
                "candidate_id": "candidate",
                "model_id": model_id,
                "signal_type": signal_type,
                "net_1x_sharpe_on_equity": sharpe,
                "net_1x_annualized_return_on_equity": 0.1,
                "net_1x_max_drawdown_on_equity": -0.2,
                "net_1x_fold_positive_share_on_equity": 0.6,
                "filtered_order_share": 0.0,
                "mean_actual_vs_target_gross_ratio": 1.0,
                "mean_weight_abs_error_sum": 0.0,
                "max_abs_net_exposure_share": 0.0,
            }
        )
    comparison = substitution.compare_signal_replays(l3, pd.DataFrame(l4_rows))
    classifications = pd.DataFrame(
        [
            {
                "candidate_id": "candidate",
                "replication_difficulty": "level1_single_proxy_partial_replication",
                "qualifying_models_at_simplest_level": "level1__x",
            }
        ]
    )
    replication = pd.DataFrame(
        [
            {
                "candidate_id": "candidate",
                "model_id": "level1__x",
                "stitched_oos_r2": 0.25,
            }
        ]
    )
    inference = substitution.build_shadow_priority_inferences(
        classifications, replication, comparison
    )

    assert comparison.iloc[0]["replica_l3_sharpe_retention"] == pytest.approx(0.5)
    assert comparison.iloc[0]["residual_l4_sharpe_retention"] == pytest.approx(2.0 / 3.0)
    assert inference.iloc[0]["evidence_type"] == "Inference"
    assert not bool(inference.iloc[0]["automatic_state_change"])


def _candidate_dependency_fixture(
    *,
    negative_signal: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calendar = pd.date_range(
        "2024-12-22", "2026-04-29", freq="D", tz="UTC"
    )
    rows = []
    signal_specs = {
        "4h": ("signal_4h", 6),
        "8h": ("signal_8h", 3),
        "12h": ("signal_12h", 2),
        "1d": ("signal_1d", 1),
    }
    for horizon, (signal_id, decisions_per_day) in signal_specs.items():
        for day_number, day in enumerate(calendar):
            for decision_number in range(decisions_per_day):
                value = (
                    0.25
                    + 0.04 * np.sin(day_number / 11.0)
                    + 0.01 * decision_number
                )
                if negative_signal and signal_id == "signal_1d":
                    value *= -1.0
                rows.append(
                    {
                        "horizon": horizon,
                        "signal_equivalence_id": signal_id,
                        "decision_ts": day
                        + pd.Timedelta(
                            hours=24 * decision_number / decisions_per_day
                        ),
                        "status": "ok",
                        "raw_rank_ic": value,
                    }
                )
    mapping_rows = []
    groups_rows = []
    for horizon, (signal_id, _) in signal_specs.items():
        aliases = [f"{signal_id}_canonical"]
        if horizon == "8h":
            aliases.append(f"{signal_id}_alias")
        canonical = aliases[0]
        for candidate_id in aliases:
            mapping_rows.append(
                {
                    "candidate_id": candidate_id,
                    "signal_equivalence_id": signal_id,
                    "canonical_candidate_id": canonical,
                    "alias_count": len(aliases),
                    "horizon": horizon,
                    "track": f"track_{horizon}",
                    "weight_scheme": "alpha_1",
                    "component_features": f"features_{horizon}",
                }
            )
        groups_rows.append(
            {
                "signal_equivalence_id": signal_id,
                "canonical_candidate_id": canonical,
                "alias_count": len(aliases),
            }
        )
    return (
        pd.DataFrame(rows),
        pd.DataFrame(mapping_rows),
        pd.DataFrame(groups_rows),
    )


def _evaluate_candidate_dependency_fixture(
    timeseries: pd.DataFrame,
    mapping: pd.DataFrame,
    groups: pd.DataFrame,
) -> substitution.CandidateResidualDependencyArtifacts:
    return substitution.evaluate_candidate_residual_dependency_adjusted_information(
        timeseries,
        mapping,
        groups,
        expected_daily_decisions={"4h": 6, "8h": 3, "12h": 2, "1d": 1},
        expected_unique_signals={"4h": 1, "8h": 1, "12h": 1, "1d": 1},
        expected_candidate_count=5,
    )


def test_candidate_dependency_adapter_maps_aliases_and_uses_primary_block_only():
    timeseries, mapping, groups = _candidate_dependency_fixture()
    result = _evaluate_candidate_dependency_fixture(timeseries, mapping, groups)

    assert len(result.unique_results) == 4
    assert len(result.candidate_results) == 5
    assert set(result.bootstrap_by_block_length) == {7, 14, 28}
    assert set(result.daily_coverage_audit["coverage_status"]) == {"pass"}
    assert result.daily_centered_sums.shape == (494, 4)
    aliases = result.candidate_results.loc[
        result.candidate_results["signal_equivalence_id"] == "signal_8h"
    ]
    for column in (
        "observed_effect",
        "bootstrap_se",
        "observed_t",
        "raw_one_sided_p_value",
        "stepdown_max_t_adjusted_p_value",
        "candidate_incremental_information_label",
    ):
        assert aliases[column].nunique(dropna=False) == 1
    assert (
        result.unique_results["candidate_incremental_information_label"]
        == "candidate_incremental_information_detected"
    ).all()
    assert (
        result.unique_results["block_length_label_stable"].astype(bool)
    ).all()


def test_registered_replica_labels_are_neutral_and_map_to_candidates():
    labels = [
        "candidate_incremental_information_detected",
        "candidate_incremental_information_not_detected",
        "candidate_incremental_information_test_invalid",
    ]
    unique = pd.DataFrame(
        {
            "signal_equivalence_id": ["s1", "s2", "s3"],
            "candidate_incremental_information_label": labels,
        }
    )
    candidates = pd.DataFrame(
        {
            "candidate_id": ["c1", "c2", "c3"],
            "candidate_incremental_information_label": labels,
        }
    )

    result = substitution.apply_train_selected_registered_replica_labels(
        unique, candidates
    )

    expected = [
        "residual_information_detected_under_train_selected_registered_replica",
        "residual_information_not_detected_under_train_selected_registered_replica",
        "model_class_test_invalid",
    ]
    assert result.unique_results[
        "registered_replica_residual_information_label"
    ].tolist() == expected
    assert result.candidate_results[
        "registered_replica_residual_information_label"
    ].tolist() == expected
    assert "candidate_incremental_information_label" not in result.unique_results
    assert "candidate_incremental_information_label" not in result.candidate_results

    broken = unique.assign(candidate_incremental_information_label="promote")
    with pytest.raises(ValueError, match="unknown label"):
        substitution.apply_train_selected_registered_replica_labels(
            broken, candidates
        )


def test_candidate_dependency_adapter_rejects_missing_day_and_bad_daily_count():
    timeseries, mapping, groups = _candidate_dependency_fixture()
    missing_day = timeseries.loc[
        ~(
            (timeseries["signal_equivalence_id"] == "signal_1d")
            & (
                pd.to_datetime(timeseries["decision_ts"], utc=True)
                == pd.Timestamp("2025-01-10", tz="UTC")
            )
        )
    ]
    with pytest.raises(ValueError, match="missing day"):
        _evaluate_candidate_dependency_fixture(missing_day, mapping, groups)

    duplicate = pd.concat(
        [
            timeseries,
            timeseries.loc[
                (timeseries["signal_equivalence_id"] == "signal_1d")
            ].head(1),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="duplicate signal/decision"):
        _evaluate_candidate_dependency_fixture(duplicate, mapping, groups)


def test_candidate_dependency_adapter_rejects_non_ok_and_non_finite_inputs():
    timeseries, mapping, groups = _candidate_dependency_fixture()
    non_ok = timeseries.copy()
    non_ok.loc[0, "status"] = "small_cross_section"
    with pytest.raises(ValueError, match="non-ok"):
        _evaluate_candidate_dependency_fixture(non_ok, mapping, groups)

    non_finite = timeseries.copy()
    non_finite.loc[0, "raw_rank_ic"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        _evaluate_candidate_dependency_fixture(non_finite, mapping, groups)


def test_candidate_dependency_adapter_rejects_canonical_mapping_disagreement():
    timeseries, mapping, groups = _candidate_dependency_fixture()
    inconsistent = mapping.copy()
    inconsistent.loc[
        inconsistent["signal_equivalence_id"] == "signal_4h",
        "canonical_candidate_id",
    ] = "not_a_group_member"
    with pytest.raises(ValueError, match="canonical ids disagree"):
        _evaluate_candidate_dependency_fixture(timeseries, inconsistent, groups)

    no_self_row = mapping.copy()
    no_self_row.loc[
        no_self_row["signal_equivalence_id"] == "signal_4h", "candidate_id"
    ] = "renamed_candidate"
    with pytest.raises(ValueError, match="exactly one canonical candidate row"):
        _evaluate_candidate_dependency_fixture(timeseries, no_self_row, groups)


def test_candidate_dependency_adapter_does_not_detect_negative_effect():
    timeseries, mapping, groups = _candidate_dependency_fixture(
        negative_signal=True
    )
    result = _evaluate_candidate_dependency_fixture(timeseries, mapping, groups)
    row = result.unique_results.set_index("signal_equivalence_id").loc["signal_1d"]
    assert row["observed_effect"] < 0.0
    assert (
        row["candidate_incremental_information_label"]
        == "candidate_incremental_information_not_detected"
    )
