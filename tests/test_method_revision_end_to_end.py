import json
from pathlib import Path

import numpy as np
import pandas as pd

from qlab import coinglass_substitution as substitution
from qlab import method_simulation


def test_truth_known_b06_runs_real_poly2_paths_through_joint_inference(monkeypatch):
    original_poly2 = substitution._poly2_registered_design
    poly2_calls = []

    def tracked_poly2(values):
        expanded = original_poly2(values)
        poly2_calls.append((values.shape, expanded.shape))
        return expanded

    monkeypatch.setattr(substitution, "_poly2_registered_design", tracked_poly2)
    registered = substitution.registered_replica_model_specs()
    design = (
        Path(__file__).resolve().parents[2]
        / "蓝图"
        / "ksv4_增量信息方法模拟设计清单.json"
    )
    artifacts = method_simulation.run_layer_b_c_simulation(
        design,
        scenario_id="B06",
        replicate=0,
        seed=8123,
        test_overrides={
            "day_count": 24,
            "object_count": 6,
            "feature_count": 6,
            "train_days": 12,
            "embargo_days": 1,
            "test_days": 5,
            "step_days": 5,
            "min_cross_section": 3,
            "block_length": 2,
            "n_bootstrap": 499,
            "alpha_grid": (1.0,),
            "model_specs": {
                "hist_gbm": (registered["hist_gbm"][0],),
                "random_forest": (registered["random_forest"][0],),
                "poly2_ridge": ({"alpha": 1.0},),
                "poly2_elastic_net": (
                    {
                        "alpha": 1.0,
                        "l1_ratio": 0.5,
                        "max_iter": 10_000,
                        "tol": 1e-6,
                    },
                ),
            },
            "allow_model_subset": True,
            "fit_workers": 1,
            "mcar_probability": 0.0,
            "missing_gap_objects": 1,
            "missing_gap_days": 4,
        },
    )

    assert poly2_calls
    assert {raw_shape[1] for raw_shape, _ in poly2_calls} == {6}
    assert {expanded_shape[1] for _, expanded_shape in poly2_calls} == {27}
    grid = artifacts.layer_c.comparison_grid
    assert len(grid) == 21
    assert set(grid["residual_method"]) == set(method_simulation.RESIDUAL_METHODS)
    assert grid["hypothesis_count"].eq(4).all()
    assert grid["true_positive_count"].eq(0).all()
    assert artifacts.layer_c.decision_moments["residual_method"].nunique() == 7
    assert artifacts.layer_c.observations["residual_method"].nunique() == 7

    observations = artifacts.layer_c.observations.copy()
    np.testing.assert_allclose(
        observations["signal_residual"],
        observations["signal_target"] - observations["signal_prediction"],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        observations["outcome_residual"],
        observations["outcome_target"] - observations["outcome_prediction"],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        observations["residual_product"],
        observations["signal_residual"] * observations["outcome_residual"],
        rtol=0.0,
        atol=1e-12,
    )
    hand_moments = (
        observations.groupby(
            [
                "hypothesis_id",
                "horizon",
                "fold_idx",
                "decision_ts",
                "residual_method",
            ],
            as_index=False,
            sort=True,
        )["residual_product"]
        .mean()
        .rename(columns={"residual_product": "hand_moment"})
    )
    program_moments = artifacts.layer_c.decision_moments.merge(
        hand_moments,
        on=[
            "hypothesis_id",
            "horizon",
            "fold_idx",
            "decision_ts",
            "residual_method",
        ],
        validate="one_to_one",
    )
    np.testing.assert_allclose(
        program_moments["double_residual_moment"],
        program_moments["hand_moment"],
        rtol=0.0,
        atol=1e-12,
    )
    for row in grid.itertuples(index=False):
        p_values = pd.Series(json.loads(row.p_values_json), dtype=float)
        assert int((p_values <= 0.05).sum()) == row.rejection_count
        assert row.true_positive_rejection_count == 0
        assert row.true_null_rejection_count == row.rejection_count
