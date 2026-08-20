import numpy as np
import pandas as pd
import pytest

from qlab import coinglass_substitution as substitution
from qlab.coinglass_substitution import REGISTERED_MODEL_CLASS_ORDER
from qlab.execution.equivalence import canonical_frame_sha256
from qlab.walkforward import WalkForwardFold

from tests.test_coinglass_substitution import (
    _registered_replica_frame,
)

SMALL_SPECS = {
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
    "poly2_ridge": ({"alpha": 1.0},),
    "poly2_elastic_net": (
        {"alpha": 1.0, "l1_ratio": 0.5, "max_iter": 10_000, "tol": 1e-6},
    ),
}


def _frozen_ridge():
    frame, fold = _registered_replica_frame()
    ridge = substitution.fit_walk_forward_ridge_replicas(
        frame,
        [fold],
        candidate_id="candidate",
        model_features={"level2_full": substitution.PRICE_VOLUME_COLUMNS},
    )
    return frame, fold, ridge


def _monolithic(frame, fold, ridge):
    return substitution.fit_walk_forward_registered_replicas(
        frame,
        [fold],
        candidate_id="candidate",
        frozen_ridge_predictions=ridge.predictions,
        frozen_ridge_inner_scores=ridge.inner_scores,
        allow_model_subset=True,
        model_specs=SMALL_SPECS,
    )


def _assemble_from_fold_artifacts(fold_artifacts):
    prediction_parts = []
    inner_rows = []
    diagnostic_rows = []
    selection_rows = []
    for artifacts in fold_artifacts:
        prediction_parts.extend(artifacts.prediction_parts)
        inner_rows.extend(artifacts.inner_rows)
        diagnostic_rows.extend(artifacts.diagnostic_rows)
        selection_rows.append(artifacts.selection_row)
    class_predictions = pd.concat(
        prediction_parts, ignore_index=True
    ).sort_values(
        ["candidate_id", "model_class", "fold_idx", "decision_ts", "symbol"],
        kind="mergesort",
    ).reset_index(drop=True)
    fold_selection = pd.DataFrame(selection_rows).sort_values(
        ["candidate_id", "fold_idx"], kind="mergesort"
    ).reset_index(drop=True)
    selected_predictions = substitution.assemble_train_selected_registered_replica(
        class_predictions, fold_selection
    )
    return substitution.RegisteredModelReplicationArtifacts(
        class_predictions=class_predictions,
        inner_scores=pd.DataFrame(inner_rows),
        model_diagnostics=pd.DataFrame(diagnostic_rows),
        fold_selection=fold_selection,
        selected_predictions=selected_predictions,
    )


def test_staged_fold_driver_equals_monolithic():
    frame, fold, ridge = _frozen_ridge()
    expected = _monolithic(frame, fold, ridge)

    fold_artifacts = [
        substitution.fit_walk_forward_registered_replica_fold(
            frame,
            fold,
            candidate_id="candidate",
            frozen_ridge_predictions=ridge.predictions,
            frozen_ridge_inner_scores=ridge.inner_scores,
            model_specs=SMALL_SPECS,
        )
    ]
    assembled = _assemble_from_fold_artifacts(fold_artifacts)

    pd.testing.assert_frame_equal(assembled.class_predictions, expected.class_predictions)
    pd.testing.assert_frame_equal(assembled.inner_scores, expected.inner_scores)
    pd.testing.assert_frame_equal(assembled.model_diagnostics, expected.model_diagnostics)
    pd.testing.assert_frame_equal(assembled.fold_selection, expected.fold_selection)
    pd.testing.assert_frame_equal(
        assembled.selected_predictions, expected.selected_predictions
    )
    assert len(fold_artifacts[0].prediction_parts) == len(REGISTERED_MODEL_CLASS_ORDER)
    assert len(fold_artifacts[0].class_candidates) == len(REGISTERED_MODEL_CLASS_ORDER)


def test_distributed_unit_path_equals_monolithic():
    frame, fold, ridge = _frozen_ridge()
    expected = _monolithic(frame, fold, ridge)

    features = tuple(substitution.PRICE_VOLUME_COLUMNS)
    if {"fold_idx", "split"}.issubset(frame.index.names):
        train_frame = frame.xs((0, "train"), level=("fold_idx", "split")).sort_index()
        test_frame = frame.xs((0, "test"), level=("fold_idx", "split")).sort_index()
    else:
        train_frame = frame.loc[
            substitution._date_mask(frame, fold.train_start, fold.train_end)
        ].sort_index()
        test_frame = frame.loc[
            substitution._date_mask(frame, fold.test_start, fold.test_end)
        ].sort_index()
    train_target = train_frame["combo_signal"].to_numpy(dtype=float)
    test_target = test_frame["combo_signal"].to_numpy(dtype=float)
    train_baseline = float(train_frame["combo_signal"].mean())
    train_features = train_frame[list(features)].to_numpy(dtype=float)
    test_features = test_frame[list(features)].to_numpy(dtype=float)
    splits = substitution._inner_time_splits(
        pd.DatetimeIndex(train_frame.index.get_level_values("decision_ts")),
        gap=pd.Timedelta("1d"),
    )
    train_dates = pd.DatetimeIndex(train_frame.index.get_level_values("decision_ts"))
    split_payloads = []
    for split_idx, (inner_train_dates, validation_dates) in enumerate(splits):
        inner_train = train_frame.loc[train_dates.isin(inner_train_dates)]
        validation = train_frame.loc[train_dates.isin(validation_dates)]
        split_payloads.append(
            {
                "inner_split_idx": split_idx,
                "inner_train_start": inner_train_dates.min(),
                "inner_train_end": inner_train_dates.max(),
                "validation_start": validation_dates.min(),
                "validation_end": validation_dates.max(),
                "train_x": inner_train[list(features)].to_numpy(dtype=float),
                "train_y": inner_train["combo_signal"].to_numpy(dtype=float),
                "validation_x": validation[list(features)].to_numpy(dtype=float),
                "validation_y": validation["combo_signal"].to_numpy(dtype=float),
            }
        )

    class_order = {
        name: index for index, name in enumerate(REGISTERED_MODEL_CLASS_ORDER)
    }
    class_candidates = []
    inner_rows = []
    prediction_parts = []
    ridge_artifacts = substitution.registered_replica_fold_ridge_artifacts(
        candidate_id="candidate",
        fold_idx=0,
        frozen_score_rows=ridge.inner_scores.loc[ridge.inner_scores["fold_idx"] == 0].copy(),
        frozen_prediction_rows=ridge.predictions.loc[ridge.predictions["fold_idx"] == 0].copy(),
        test_frame=test_frame,
        test_target=test_target,
        train_baseline=train_baseline,
    )
    prediction_parts.append(ridge_artifacts.prediction_frame)
    inner_rows.extend(ridge_artifacts.inner_rows)
    class_candidates.append(ridge_artifacts.class_candidate)

    for model_class in REGISTERED_MODEL_CLASS_ORDER[1:]:
        units = substitution.registered_replica_fold_inner_tasks(
            model_class, SMALL_SPECS[model_class], split_payloads
        )
        evaluated = []
        for kind, parameters, payload in units:
            evaluated.extend(
                substitution.evaluate_registered_replica_inner_task(
                    model_class, parameters, payload, kind=kind
                )
            )
        class_fit = substitution.finalize_registered_replica_fold_class(
            model_class,
            evaluated,
            fold_idx=0,
            candidate_id="candidate",
            train_features=train_features,
            test_features=test_features,
            train_target=train_target,
            test_target=test_target,
            train_baseline=train_baseline,
        )
        prediction_parts.append(
            substitution.registered_replica_fold_class_output_frame(
                test_frame=test_frame,
                model_class=class_fit.model_class,
                candidate_id="candidate",
                fold_idx=0,
                test_target=test_target,
                train_baseline=train_baseline,
                best_parameter_key=class_fit.best_parameter_key,
                test_prediction=class_fit.test_prediction,
            )
        )
        inner_rows.extend(class_fit.inner_rows)
        class_candidates.append(class_fit.class_candidate)

    winner = substitution.registered_replica_fold_winner(class_candidates, class_order)
    diagnostic_rows = substitution.registered_replica_fold_diagnostic_rows(
        class_candidates, winner, candidate_id="candidate", fold_idx=0
    )
    selection_row = substitution.registered_replica_fold_selection_row(
        winner, candidate_id="candidate", fold_idx=0
    )
    assembled = _assemble_from_fold_artifacts(
        [
            substitution.RegisteredReplicaFoldArtifacts(
                fold_idx=0,
                prediction_parts=prediction_parts,
                inner_rows=inner_rows,
                class_candidates=class_candidates,
                diagnostic_rows=diagnostic_rows,
                selection_row=selection_row,
                winner=winner,
            )
        ]
    )

    pd.testing.assert_frame_equal(assembled.class_predictions, expected.class_predictions)
    pd.testing.assert_frame_equal(assembled.inner_scores, expected.inner_scores)
    pd.testing.assert_frame_equal(assembled.model_diagnostics, expected.model_diagnostics)
    pd.testing.assert_frame_equal(assembled.fold_selection, expected.fold_selection)
    pd.testing.assert_frame_equal(
        assembled.selected_predictions, expected.selected_predictions
    )
    assert winner["model_class"] == expected.fold_selection.iloc[0]["selected_model_class"]
    assert selection_row["selected_inner_validation_sse"] == pytest.approx(
        expected.fold_selection.iloc[0]["selected_inner_validation_sse"],
        rel=0.0,
        abs=1e-12,
    )


def test_inner_task_unit_order_matches_sequential_batch():
    frame, fold, ridge = _frozen_ridge()
    train_frame = frame.loc[
        substitution._date_mask(frame, fold.train_start, fold.train_end)
    ].sort_index()
    splits = substitution._inner_time_splits(
        pd.DatetimeIndex(train_frame.index.get_level_values("decision_ts")),
        gap=pd.Timedelta("1d"),
    )
    train_dates = pd.DatetimeIndex(train_frame.index.get_level_values("decision_ts"))
    features = tuple(substitution.PRICE_VOLUME_COLUMNS)
    split_payloads = []
    for split_idx, (inner_train_dates, validation_dates) in enumerate(splits):
        inner_train = train_frame.loc[train_dates.isin(inner_train_dates)]
        validation = train_frame.loc[train_dates.isin(validation_dates)]
        split_payloads.append(
            {
                "inner_split_idx": split_idx,
                "inner_train_start": inner_train_dates.min(),
                "inner_train_end": inner_train_dates.max(),
                "validation_start": validation_dates.min(),
                "validation_end": validation_dates.max(),
                "train_x": inner_train[list(features)].to_numpy(dtype=float),
                "train_y": inner_train["combo_signal"].to_numpy(dtype=float),
                "validation_x": validation[list(features)].to_numpy(dtype=float),
                "validation_y": validation["combo_signal"].to_numpy(dtype=float),
            }
        )

    model_class = "hist_gbm"
    units = substitution.registered_replica_fold_inner_tasks(
        model_class, SMALL_SPECS[model_class], split_payloads
    )
    sequential = []
    for parameters in SMALL_SPECS[model_class]:
        for payload in split_payloads:
            sequential.append((parameters, payload))
    prefix_tasks, generic_tasks = substitution._registered_prefix_partition(
        model_class, sequential
    )
    assert [unit[0] for unit in units] == (
        ["prefix"] * len(prefix_tasks) + ["generic"] * len(generic_tasks)
    )
    expected_keys = [substitution._registered_parameter_key(p) for p, _ in prefix_tasks]
    expected_keys.extend(
        substitution._registered_parameter_key(p) for p, _ in generic_tasks
    )
    actual_keys = [
        substitution._registered_parameter_key(dict(unit[1])) for unit in units
    ]
    assert actual_keys == expected_keys
    assert [unit[2]["inner_split_idx"] for unit in units] == [
        payload["inner_split_idx"] for _, payload in (
            prefix_tasks + generic_tasks
        )
    ]


def test_unit_identity_keys_match_evaluated_rows():
    frame, fold, ridge = _frozen_ridge()
    train_frame = frame.loc[
        substitution._date_mask(frame, fold.train_start, fold.train_end)
    ].sort_index()
    splits = substitution._inner_time_splits(
        pd.DatetimeIndex(train_frame.index.get_level_values("decision_ts")),
        gap=pd.Timedelta("1d"),
    )
    train_dates = pd.DatetimeIndex(train_frame.index.get_level_values("decision_ts"))
    features = tuple(substitution.PRICE_VOLUME_COLUMNS)
    split_payloads = []
    for split_idx, (inner_train_dates, validation_dates) in enumerate(splits):
        inner_train = train_frame.loc[train_dates.isin(inner_train_dates)]
        validation = train_frame.loc[train_dates.isin(validation_dates)]
        split_payloads.append(
            {
                "inner_split_idx": split_idx,
                "inner_train_start": inner_train_dates.min(),
                "inner_train_end": inner_train_dates.max(),
                "validation_start": validation_dates.min(),
                "validation_end": validation_dates.max(),
                "train_x": inner_train[list(features)].to_numpy(dtype=float),
                "train_y": inner_train["combo_signal"].to_numpy(dtype=float),
                "validation_x": validation[list(features)].to_numpy(dtype=float),
                "validation_y": validation["combo_signal"].to_numpy(dtype=float),
            }
        )

    for model_class in ("hist_gbm", "random_forest", "poly2_ridge", "poly2_elastic_net"):
        units = substitution.registered_replica_fold_inner_tasks(
            model_class, SMALL_SPECS[model_class], split_payloads
        )
        first_units = substitution.registered_replica_fold_inner_tasks(
            model_class, SMALL_SPECS[model_class], split_payloads
        )
        assert [u[1] for u in units] == [u[1] for u in first_units]
        for kind, parameters, payload in units:
            rows = substitution.evaluate_registered_replica_inner_task(
                model_class, parameters, payload, kind=kind
            )
            assert all(row[2]["inner_split_idx"] == payload["inner_split_idx"] for row in rows)
            assert len(rows) == (
                2 if kind == "prefix" else 1
            )


def test_physical_workload_categories_match_full_registered_grid():
    rng = np.random.default_rng(20260821)
    payload = {
        "inner_split_idx": 0,
        "inner_train_start": pd.Timestamp("2024-01-01", tz="UTC"),
        "inner_train_end": pd.Timestamp("2024-01-02", tz="UTC"),
        "validation_start": pd.Timestamp("2024-01-03", tz="UTC"),
        "validation_end": pd.Timestamp("2024-01-04", tz="UTC"),
        "train_x": rng.normal(size=(40, 24)),
        "train_y": rng.normal(size=40),
        "validation_x": rng.normal(size=(20, 24)),
        "validation_y": rng.normal(size=20),
    }
    units = substitution.registered_replica_physical_workload_units(
        substitution.registered_replica_model_specs(), [payload]
    )
    counts = pd.Series([unit["physical_category"] for unit in units]).value_counts().to_dict()
    assert counts == {
        "hist_gbm:prefix": 4,
        "hist_gbm:generic": 4,
        "random_forest:prefix": 2,
        "random_forest:generic": 4,
        "poly2_ridge:generic": 9,
        "poly2_elastic_net:generic": 27,
    }
    rf_generic = [
        unit for unit in units
        if unit["physical_category"] == "random_forest:generic"
    ]
    assert all("min_samples_leaf" in unit["parameters"] for unit in rf_generic)
    assert sum(int(unit["configuration_count"]) for unit in units) == 56


def test_full_registered_grid_matches_frozen_pre_refactor_reference():
    """Evidence fixture generated from the pre-issue-14 qlab implementation."""
    rng = np.random.default_rng(20260821)
    dates = pd.date_range("2025-01-01", periods=150, freq="1D", tz="UTC")
    symbols = [f"S{index:02d}" for index in range(12)]
    index = pd.MultiIndex.from_product(
        [dates, symbols], names=["decision_ts", "symbol"]
    )
    frame = pd.DataFrame(index=index)
    for column in substitution.PRICE_VOLUME_COLUMNS:
        frame[column] = rng.normal(0.0, 0.2, len(index))
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
        train_end=dates[119],
        test_start=dates[122],
        test_end=dates[149],
    )
    ridge = substitution.fit_walk_forward_ridge_replicas(
        frame,
        [fold],
        candidate_id="candidate",
        model_features={"level2_full": substitution.PRICE_VOLUME_COLUMNS},
    )
    artifacts = substitution.fit_walk_forward_registered_replicas(
        frame,
        [fold],
        candidate_id="candidate",
        frozen_ridge_predictions=ridge.predictions,
        frozen_ridge_inner_scores=ridge.inner_scores,
        model_specs=substitution.registered_replica_model_specs(),
    )
    expected_digests = {
        "class_predictions": "a0dfd892b6f001f1675d40e8b2380270ddf67e20b16307c3112d005439d9090b",
        "inner_scores": "771adb6160af321c5a7088a8d00cc9eddd50f62e76dba9b3acee2aad361d8144",
        "model_diagnostics": "0f9b22799d5db09f6e9e061b7e258e4873555a33b7e302d702188ce3e2f108c5",
        "fold_selection": "6d209497c82454d231db07fe9e83c8b782e3d0f74d7690fa6b91e272360b2b20",
        "selected_predictions": "c5f7aceb31490826f70c01a3b31740575a6f6f2e8e2a3f0506869e57edd0ed25",
    }
    for artifact_name, expected_digest in expected_digests.items():
        actual = getattr(artifacts, artifact_name)
        assert canonical_frame_sha256(actual) == expected_digest
