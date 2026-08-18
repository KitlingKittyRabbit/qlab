from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from qlab import method_simulation
from qlab.workspace_paths import resolve_blueprint_root


def _blueprint_file(name: str) -> Path:
    try:
        root = resolve_blueprint_root(Path(__file__).resolve().parents[2])
    except RuntimeError as exc:
        pytest.skip(f"private blueprint repository is unavailable: {exc}")
    return root / name


def _empirical_family_fixture(day_count: int = 40):
    index = pd.date_range("2025-01-01", periods=day_count, freq="D", tz="UTC")
    angle = np.arange(day_count, dtype=float)
    null = pd.DataFrame(
        {
            "source_a": np.sin(angle / 3.0),
            "source_b": 0.6 * np.sin(angle / 3.0) + np.cos(angle / 5.0),
        },
        index=index,
    )
    null = null.subtract(null.mean(axis=0), axis="columns")
    effects = null + pd.Series({"source_a": 0.25, "source_b": 0.15})
    counts = pd.DataFrame(1, index=index, columns=null.columns)
    return effects, null, counts


def test_empirical_family_preflight_separates_null_noise_and_effect_pool():
    effects, centered, counts = _empirical_family_fixture(day_count=5)
    artifacts = method_simulation.prepare_empirical_block_family(
        effects,
        centered,
        counts,
        ["source_b", "source_a"],
        expected_day_count=5,
    )
    assert list(artifacts.null_daily_effects.columns) == ["H01", "H02"]
    np.testing.assert_allclose(artifacts.null_daily_effects.mean(), 0.0, atol=1e-15)
    expected = effects[["source_b", "source_a"]].mean() / effects[
        ["source_b", "source_a"]
    ].std(ddof=1)
    np.testing.assert_allclose(
        artifacts.empirical_standardized_effects.to_numpy(), expected.to_numpy()
    )
    assert artifacts.hypothesis_manifest["source_hypothesis_id"].tolist() == [
        "source_b", "source_a"
    ]


def test_circular_block_indices_are_hand_calculable_and_wrap_synchronously():
    indices = method_simulation.circular_block_indices_from_starts(
        5, 2, [4, 1, 3]
    )
    assert indices.tolist() == [4, 0, 1, 2, 3]
    values = np.column_stack([np.arange(5), 10 + np.arange(5)])
    sampled = values[indices]
    assert sampled[:, 0].tolist() == [4, 0, 1, 2, 3]
    assert (sampled[:, 1] - sampled[:, 0]).tolist() == [10] * 5


def test_synchronized_sampling_preserves_identical_columns_but_independent_does_not():
    values = np.arange(8, dtype=float)
    family = np.column_stack([values, values])
    shared = method_simulation.synchronized_circular_block_indices(
        8, 2, draw_count=1, seed=10
    )[0]
    np.testing.assert_array_equal(family[shared, 0], family[shared, 1])
    left = method_simulation.synchronized_circular_block_indices(
        8, 2, draw_count=1, seed=10
    )[0]
    right = method_simulation.synchronized_circular_block_indices(
        8, 2, draw_count=1, seed=11
    )[0]
    assert not np.array_equal(family[left, 0], family[right, 1])


def test_seed_namespaces_fail_closed_on_calibration_evaluation_collision():
    method_simulation.validate_disjoint_seed_namespaces({
        "calibration": [1, 2], "evaluation": [3, 4]
    })
    with pytest.raises(ValueError, match="shared"):
        method_simulation.validate_disjoint_seed_namespaces({
            "calibration": [1, 2], "evaluation": [2, 3]
        })


def test_empirical_calibration_and_paired_effect_injection_are_separate():
    effects, centered, counts = _empirical_family_fixture()
    family = method_simulation.prepare_empirical_block_family(
        effects, centered, counts, ["source_a", "source_b"], expected_day_count=40
    )
    calibration = method_simulation.calibrate_empirical_block_mean_standard_errors(
        family.null_daily_effects,
        block_length=4,
        n_draws=200,
        seed=17,
        batch_size=25,
    )
    reference = calibration.hypothesis_standard_errors.set_index(
        "hypothesis_id"
    )["reference_standard_error"]
    results = method_simulation.simulate_empirical_block_task_family(
        family.null_daily_effects,
        family.empirical_standardized_effects,
        family.null_daily_standard_deviations,
        reference,
        [
            {"scenario_id": "Z", "block_length": 4, "active_count": 0, "shrinkage_multiplier": 0.0},
            {"scenario_id": "R_HALF", "block_length": 4, "active_count": 1, "shrinkage_multiplier": 0.5},
            {"scenario_id": "R_FULL", "block_length": 4, "active_count": 1, "shrinkage_multiplier": 1.0},
        ],
        replicate=3,
        block_length=4,
        noise_seed=101,
        truth_seed=202,
    )
    half = results.loc[results["scenario_id"].eq("R_HALF")].set_index("hypothesis_id")
    full = results.loc[results["scenario_id"].eq("R_FULL")].set_index("hypothesis_id")
    null = results.loc[results["scenario_id"].eq("Z")].set_index("hypothesis_id")
    assert half["true_effect"].gt(0).sum() == 1
    np.testing.assert_allclose(full["true_effect"], 2.0 * half["true_effect"])
    np.testing.assert_allclose(
        half["observed_effect"] - half["true_effect"], null["observed_effect"]
    )
    np.testing.assert_allclose(
        results.groupby("hypothesis_id")["uncalibrated_standard_error"].nunique(), 1
    )
    np.testing.assert_allclose(
        results.groupby("hypothesis_id")["reference_standard_error"].nunique(), 1
    )
    np.testing.assert_allclose(
        full["observed_effect"] - null["observed_effect"], full["true_effect"]
    )


def test_empirical_reference_bh_and_randomized_truth_summary_use_full_family():
    fixture = _bh_fdr_fixture().copy()
    fixture["reference_standard_error"] = 1.0
    artifacts = method_simulation.evaluate_bh_fdr_variants(
        fixture,
        dataset_id="empirical_fixture",
        family_size=5,
        scenario_temporal_dependence=None,
        include_cross_scenario_summary=False,
    )
    assert set(artifacts.hypothesis_results["method_variant"]) == {
        "AR_BIC_1000_BH", "AR_BIC_1125_BH", "MC_REFERENCE_BH"
    }
    conditional = method_simulation.summarize_randomized_true_hypotheses(
        artifacts.hypothesis_results
    )
    assert conditional["assigned_task_count"].gt(0).all()
    assert conditional["conditional_discovery_rate"].between(0.0, 1.0).all()


def test_empirical_family_fails_closed_on_nonzero_centered_input():
    effects, centered, counts = _empirical_family_fixture(day_count=5)
    centered.iloc[:, 0] += 0.01
    with pytest.raises(ValueError, match="not zero mean"):
        method_simulation.prepare_empirical_block_family(
            effects, centered, counts, ["source_a", "source_b"], expected_day_count=5
        )


def test_empirical_family_fails_closed_on_wrong_days_and_nonfinite_values():
    effects, centered, counts = _empirical_family_fixture(day_count=5)
    with pytest.raises(ValueError, match="exactly 6 days"):
        method_simulation.prepare_empirical_block_family(
            effects, centered, counts, ["source_a", "source_b"], expected_day_count=6
        )
    effects.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite and complete"):
        method_simulation.prepare_empirical_block_family(
            effects, centered, counts, ["source_a", "source_b"], expected_day_count=5
        )


def test_load_design_fails_closed_until_approved(tmp_path):
    path = tmp_path / "design.json"
    path.write_text(json.dumps({"schema_version": "ksv4_method_simulation_design_v6", "lifecycle": "candidate", "approved": False, "layer_a": {}, "layer_b": {}}))
    with pytest.raises(ValueError, match="approved and frozen"):
        method_simulation.load_frozen_design(path)


def test_grouped_correlation_has_registered_blocks():
    matrix = method_simulation.grouped_correlation((2, 1), 0.4, 0.1)
    np.testing.assert_allclose(matrix, [[1, .4, .1], [.4, 1, .1], [.1, .1, 1]])


def test_layer_a_generator_reproducible_and_truth_is_fixed():
    scenario = {"hypothesis_dependence": "independent", "temporal_dependence": "iid", "effect": "first_5_positive_0.20"}
    first = method_simulation.generate_layer_a_dataset(scenario, day_count=20, group_sizes=(3, 3), seed=9)
    second = method_simulation.generate_layer_a_dataset(scenario, day_count=20, group_sizes=(3, 3), seed=9)
    pd.testing.assert_frame_equal(first.daily_values, second.daily_values)
    assert first.true_effects.tolist() == [0.2] * 5 + [0.0]


def test_layer_a_generator_supports_fresh_33_hypothesis_dense_family():
    scenario = {
        "hypothesis_dependence": "within_0.70_between_0.20",
        "temporal_dependence": "ar1_0.50",
        "effect": "first_17_positive_0.13",
    }
    dataset = method_simulation.generate_layer_a_dataset(
        scenario,
        day_count=494,
        group_sizes=(5, 4, 4, 4, 4, 4, 4, 4),
        seed=33,
    )
    assert dataset.daily_values.shape == (494, 33)
    assert dataset.daily_values.columns.tolist() == [
        f"H{index:02d}" for index in range(1, 34)
    ]
    assert dataset.true_effects.iloc[:17].eq(0.13).all()
    assert dataset.true_effects.iloc[17:].eq(0.0).all()


def test_exact_gaussian_mean_variance_uses_finite_pair_counts():
    iid = method_simulation.exact_gaussian_mean_variance("iid", day_count=5)
    assert iid["exact_mean_variance"] == pytest.approx(0.2)
    ar = method_simulation.exact_gaussian_mean_variance("ar1_0.50", day_count=3)
    expected_ar = (3 + 2 * (2 * 0.5 + 1 * 0.25)) / 9
    assert ar["exact_mean_variance"] == pytest.approx(expected_ar)
    assert ar["asymptotic_long_run_variance"] == pytest.approx(3.0)
    ma = method_simulation.exact_gaussian_mean_variance("ma_14", day_count=494)
    assert ma["asymptotic_long_run_variance"] == 14.0
    assert ma["exact_mean_variance"] != pytest.approx(14.0 / 494)
    expected_ma = (
        494
        + 2 * sum((494 - lag) * (14 - lag) / 14 for lag in range(1, 14))
    ) / 494**2
    assert ma["exact_mean_variance"] == pytest.approx(expected_ma)
    ma2 = method_simulation.exact_gaussian_mean_variance("ma_2", day_count=5)
    assert ma2["asymptotic_long_run_variance"] == pytest.approx(2.0)
    assert ma2["exact_mean_variance"] == pytest.approx((5 + 2 * 4 * 0.5) / 25)


def test_exact_gaussian_time_covariance_and_two_point_gls_are_hand_calculable():
    covariance = method_simulation.exact_gaussian_time_covariance(
        "ar1_0.50", day_count=2
    )
    np.testing.assert_allclose(covariance, [[1.0, 0.5], [0.5, 1.0]])
    contract = method_simulation.exact_gaussian_gls_contract(
        "ar1_0.50", day_count=2
    )
    np.testing.assert_allclose(contract["weights"], [0.5, 0.5])
    assert contract["estimator_variance"] == pytest.approx(0.75)
    dataset = method_simulation.LayerADataset(
        daily_values=pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]], columns=["H01", "H02"]
        ),
        true_effects=pd.Series([0.0, 0.0], index=["H01", "H02"]),
    )
    result = method_simulation.oracle_mean_gls_family(
        dataset, temporal_dependence="ar1_0.50"
    )
    np.testing.assert_allclose(result["gls_effect"], [2.0, 3.0])
    np.testing.assert_allclose(result["gls_standard_error"], np.sqrt(0.75))


def test_exact_gaussian_gls_rejects_invalid_covariance_specification():
    with pytest.raises(ValueError, match="strictly between -1 and 1"):
        method_simulation.exact_gaussian_gls_contract("ar1_1.0", day_count=10)


def test_real_effect_calibration_matches_hand_calculation_and_full_cycle_bootstrap():
    daily = pd.DataFrame(
        {
            "H01": [0.0, 1.0, 2.0, 3.0],
            "H02": [3.0, 1.0, -1.0, -3.0],
        },
        index=pd.date_range("2024-01-01", periods=4, tz="UTC"),
    )
    artifacts = method_simulation.calibrate_real_standardized_effects(
        daily,
        block_length=4,
        n_bootstrap=20,
        seed=7,
        distribution_quantiles=(0.0, 0.5, 1.0),
        simulation_grid=(0.0, 1.0),
        batch_size=3,
    )
    expected = daily.mean().to_numpy() / daily.std(ddof=1).to_numpy()
    np.testing.assert_allclose(
        artifacts.hypothesis_effects["standardized_effect"], expected
    )
    np.testing.assert_allclose(
        artifacts.hypothesis_effects["bootstrap_ci_lower"], expected
    )
    np.testing.assert_allclose(
        artifacts.hypothesis_effects["bootstrap_ci_upper"], expected
    )
    np.testing.assert_allclose(
        artifacts.distribution_quantiles["observed_standardized_effect"],
        np.quantile(expected, [0.0, 0.5, 1.0]),
    )
    assert artifacts.simulation_grid_alignment["empirical_percentile"].tolist() == [0.5, 0.5]


@pytest.mark.parametrize(
    "daily, error",
    [
        (pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [0.0, 1.0, 2.0]}), "constant"),
        (pd.DataFrame({"a": [0.0, np.nan, 1.0], "b": [0.0, 1.0, 2.0]}), "finite"),
    ],
)
def test_real_effect_calibration_fails_closed_on_invalid_daily_values(daily, error):
    with pytest.raises(ValueError, match=error):
        method_simulation.calibrate_real_standardized_effects(
            daily, block_length=2, n_bootstrap=10
        )


def test_real_effect_calibration_rejects_duplicate_keys_and_invalid_block():
    daily = pd.DataFrame(
        {"a": [0.0, 1.0, 2.0], "b": [1.0, 2.0, 4.0]},
        index=[0, 0, 1],
    )
    with pytest.raises(ValueError, match="index must be unique"):
        method_simulation.calibrate_real_standardized_effects(
            daily, block_length=2, n_bootstrap=10
        )
    daily.index = [0, 1, 2]
    with pytest.raises(ValueError, match="block_length"):
        method_simulation.calibrate_real_standardized_effects(
            daily, block_length=4, n_bootstrap=10
        )


def test_oracle_gls_equals_equal_weight_oracle_under_iid():
    dataset = method_simulation.LayerADataset(
        daily_values=pd.DataFrame(
            [[2.0, 0.0, -1.0], [0.0, 1.0, 1.0], [1.0, -1.0, 0.0], [3.0, 2.0, 2.0]],
            columns=["H01", "H02", "H03"],
        ),
        true_effects=pd.Series([0.2, 0.0, 0.0], index=["H01", "H02", "H03"]),
    )
    result = method_simulation.oracle_mean_gls_family(
        dataset, temporal_dependence="iid"
    )
    np.testing.assert_allclose(result["gls_effect"], result["mean_effect"])
    np.testing.assert_allclose(
        result["gls_standard_error"], result["mean_standard_error"]
    )
    np.testing.assert_allclose(result["gls_raw_p_value"], result["mean_raw_p_value"])
    np.testing.assert_allclose(result["gls_bh_q_value"], result["mean_bh_q_value"])
    legacy = method_simulation.oracle_gaussian_bh_family(
        dataset.daily_values.mean(axis=0),
        temporal_dependence="iid",
        day_count=4,
        alternative="greater",
    )
    np.testing.assert_allclose(
        result["mean_raw_p_value"], legacy["oracle_raw_p_value"]
    )
    np.testing.assert_allclose(result["mean_bh_q_value"], legacy["oracle_bh_q_value"])


def _power_attribution_fixture():
    rows = []
    for task_idx, shift in enumerate((0.0, 0.2)):
        dataset = method_simulation.LayerADataset(
            daily_values=pd.DataFrame(
                [
                    [2.0 + shift, 0.2, -0.1],
                    [1.8 + shift, -0.1, 0.2],
                    [2.2 + shift, 0.1, 0.0],
                    [2.0 + shift, 0.0, 0.1],
                ],
                columns=["H01", "H02", "H03"],
            ),
            true_effects=pd.Series([0.2, 0.0, 0.0], index=["H01", "H02", "H03"]),
        )
        result = method_simulation.oracle_mean_gls_family(
            dataset, temporal_dependence="iid"
        )
        result.insert(0, "replicate", task_idx)
        result.insert(0, "analysis_specification", "A07__right_tail_primary")
        result.insert(0, "scenario_id", "A07")
        result.insert(0, "registered_task_idx", task_idx)
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def test_power_attribution_summary_tracks_raw_bh_and_gls_by_hand():
    artifacts = method_simulation.summarize_discovery_power_attribution(
        _power_attribution_fixture(), family_size=3
    )
    assert set(artifacts.scenario_summary["method_variant"]) == {
        "MEAN_RAW", "MEAN_BH", "GLS_RAW", "GLS_BH"
    }
    attribution = artifacts.attribution_summary.iloc[0]
    assert attribution["mean_raw_tpr"] == pytest.approx(1.0)
    assert attribution["mean_bh_tpr"] == pytest.approx(1.0)
    assert attribution["gls_raw_tpr"] == pytest.approx(1.0)
    assert attribution["gls_bh_tpr"] == pytest.approx(1.0)
    assert attribution["bh_loss"] == pytest.approx(0.0)
    assert attribution["gls_bh_gain"] == pytest.approx(0.0)


def test_power_attribution_fails_closed_on_incomplete_or_duplicate_family():
    frame = _power_attribution_fixture()
    with pytest.raises(ValueError, match="complete, ordered fixed-size families"):
        method_simulation.summarize_discovery_power_attribution(
            frame.iloc[:-1], family_size=3
        )
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate task hypotheses"):
        method_simulation.summarize_discovery_power_attribution(
            duplicate, family_size=3
        )


def test_oracle_gaussian_holm_family_matches_two_hypothesis_hand_calculation():
    observed = pd.Series([2.0, 0.0], index=["h1", "h2"])
    result = method_simulation.oracle_gaussian_holm_family(
        observed,
        temporal_dependence="iid",
        day_count=4,
        alternative="greater",
    ).set_index("hypothesis_id")
    assert result.loc["h1", "oracle_se"] == pytest.approx(0.5)
    assert result.loc["h1", "oracle_z"] == pytest.approx(4.0)
    raw_h1 = float(method_simulation.normal_distribution.sf(4.0))
    assert result.loc["h1", "oracle_raw_p_value"] == pytest.approx(raw_h1)
    assert result.loc["h1", "oracle_holm_adjusted_p_value"] == pytest.approx(
        2.0 * raw_h1
    )
    assert result.loc["h2", "oracle_holm_adjusted_p_value"] == pytest.approx(0.5)


def _bh_fdr_fixture(*, invalid_p_input: bool = False):
    rows = []
    raw_probabilities = (
        (0.001, 0.010, 0.025, 0.200, 0.900),
        (0.001, 0.200, 0.300, 0.400, 0.900),
    )
    for replicate, probabilities in enumerate(raw_probabilities):
        for index, raw_probability in enumerate(probabilities, start=1):
            rows.append(
                {
                    "registered_task_idx": replicate,
                    "scenario_id": "A06",
                    "analysis_specification": "A06__right_tail_primary__fixture",
                    "replicate": replicate,
                    "hypothesis_id": f"H{index:02d}",
                    "observed_effect": float(
                        method_simulation.normal_distribution.isf(raw_probability)
                    ),
                    "uncalibrated_standard_error": (
                        np.nan if invalid_p_input and replicate == 0 and index == 5 else 1.0
                    ),
                    "alternative": "greater",
                    "true_effect": 0.2 if index <= 2 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_oracle_gaussian_bh_family_matches_hand_calculation():
    observed = pd.Series([2.0, 0.0], index=["H01", "H02"])
    result = method_simulation.oracle_gaussian_bh_family(
        observed,
        temporal_dependence="iid",
        day_count=4,
        alternative="greater",
    ).set_index("hypothesis_id")
    raw_h1 = float(method_simulation.normal_distribution.sf(4.0))
    assert result.loc["H01", "oracle_se"] == pytest.approx(0.5)
    assert result.loc["H01", "oracle_raw_p_value"] == pytest.approx(raw_h1)
    assert result.loc["H01", "oracle_bh_q_value"] == pytest.approx(2.0 * raw_h1)
    assert result.loc["H02", "oracle_bh_q_value"] == pytest.approx(0.5)


def test_bh_fdr_entry_computes_task_fdp_then_scenario_mean_by_hand():
    artifacts = method_simulation.evaluate_bh_fdr_variants(
        _bh_fdr_fixture(),
        dataset_id="fixture",
        scenario_temporal_dependence={"A06": "iid"},
        day_count=4,
        family_size=5,
    )
    tasks = artifacts.task_summary.loc[
        artifacts.task_summary["method_variant"].eq("AR_BIC_1000_BH")
    ].sort_values("replicate")
    assert tasks["discovery_count"].tolist() == [3, 1]
    assert tasks["false_discovery_count"].tolist() == [1, 0]
    assert tasks["true_discovery_count"].tolist() == [2, 1]
    assert tasks["false_discovery_proportion"].tolist() == pytest.approx([1 / 3, 0.0])
    assert tasks["true_positive_rate"].tolist() == pytest.approx([1.0, 0.5])
    scenario = artifacts.scenario_summary.loc[
        artifacts.scenario_summary["method_variant"].eq("AR_BIC_1000_BH")
    ].iloc[0]
    assert scenario["fdr"] == pytest.approx(1 / 6)
    assert scenario["mean_true_positive_rate"] == pytest.approx(0.75)
    assert scenario["true_positive_rate_monte_carlo_standard_error"] == pytest.approx(0.25)
    assert scenario["true_positive_rate_ci95_lower"] == pytest.approx(0.26)
    assert scenario["true_positive_rate_ci95_upper"] == pytest.approx(1.0)
    assert (1 / 6) != pytest.approx(1 / 4)  # pooled V/R is not the FDR estimand
    assert artifacts.cross_scenario_summary["scenario_count"].eq(1).all()

    without_cross = method_simulation.evaluate_bh_fdr_variants(
        _bh_fdr_fixture(),
        dataset_id="fixture_no_cross",
        scenario_temporal_dependence={"A06": "iid"},
        day_count=4,
        family_size=5,
        include_cross_scenario_summary=False,
    )
    assert without_cross.cross_scenario_summary.empty
    assert len(without_cross.scenario_summary) == 3


def _effect_retargeting_fixture():
    rows = []
    for replicate in range(2):
        for index in range(3):
            true_effect = 0.2 if index == 0 else 0.0
            rows.append(
                {
                    "registered_task_idx": 10 + replicate,
                    "scenario_id": "BASE",
                    "analysis_specification": "BASE__right_tail_primary__fixture",
                    "replicate": replicate,
                    "hypothesis_id": f"H{index + 1:02d}",
                    "observed_effect": true_effect + replicate + index / 10,
                    "uncalibrated_standard_error": 0.5 + index / 10,
                    "alternative": "greater",
                    "true_effect": true_effect,
                }
            )
    return pd.DataFrame(rows)


def test_effect_retargeting_is_paired_and_preserves_nulls_and_standard_errors():
    base = _effect_retargeting_fixture()
    transformed = method_simulation.retarget_additive_effect_scenarios(
        base,
        [
            {"scenario_id": "R01", "base_scenario_id": "BASE", "target_effect": 0.09, "active_count": 1},
            {"scenario_id": "R02", "base_scenario_id": "BASE", "target_effect": 0.13, "active_count": 1},
        ],
        family_size=3,
        expected_tasks_per_base=2,
    )
    r01 = transformed.loc[transformed["scenario_id"].eq("R01")].reset_index(drop=True)
    r02 = transformed.loc[transformed["scenario_id"].eq("R02")].reset_index(drop=True)
    active = r01["hypothesis_id"].eq("H01")
    np.testing.assert_allclose(
        r02.loc[active, "observed_effect"] - r01.loc[active, "observed_effect"],
        0.04,
    )
    np.testing.assert_allclose(
        r01.loc[~active, "observed_effect"],
        r02.loc[~active, "observed_effect"],
    )
    np.testing.assert_allclose(
        r01["uncalibrated_standard_error"],
        r02["uncalibrated_standard_error"],
    )
    assert r01.loc[active, "true_effect"].eq(0.09).all()
    assert r02.loc[active, "true_effect"].eq(0.13).all()
    assert transformed["registered_task_idx"].nunique() == 4


def test_effect_retargeting_creates_sparse_and_dense_families_from_all_null_noise():
    base = _effect_retargeting_fixture()
    base["observed_effect"] = base["observed_effect"] - base["true_effect"]
    base["true_effect"] = 0.0
    transformed = method_simulation.retarget_additive_effect_scenarios(
        base,
        [
            {"scenario_id": "NULL", "base_scenario_id": "BASE", "target_effect": 0.0, "active_count": 0},
            {"scenario_id": "SPARSE", "base_scenario_id": "BASE", "target_effect": 0.09, "active_count": 1},
            {"scenario_id": "DENSE", "base_scenario_id": "BASE", "target_effect": 0.18, "active_count": 2},
        ],
        family_size=3,
        expected_tasks_per_base=2,
    )
    sparse = transformed.loc[transformed["scenario_id"].eq("SPARSE")]
    dense = transformed.loc[transformed["scenario_id"].eq("DENSE")]
    null = transformed.loc[transformed["scenario_id"].eq("NULL")]
    assert null["true_effect"].eq(0.0).all()
    assert sparse.groupby("replicate")["true_effect"].apply(
        lambda values: values.gt(0).sum()
    ).eq(1).all()
    assert dense.groupby("replicate")["true_effect"].apply(
        lambda values: values.gt(0).sum()
    ).eq(2).all()
    np.testing.assert_allclose(
        (sparse["observed_effect"] - sparse["true_effect"]).to_numpy(),
        (dense["observed_effect"] - dense["true_effect"]).to_numpy(),
    )
    np.testing.assert_allclose(
        sparse["uncalibrated_standard_error"].to_numpy(),
        dense["uncalibrated_standard_error"].to_numpy(),
    )


def test_effect_retargeting_fails_closed_on_wrong_pattern_or_task_count():
    base = _effect_retargeting_fixture()
    spec = [{"scenario_id": "R01", "base_scenario_id": "BASE", "target_effect": 0.09, "active_count": 1}]
    with pytest.raises(ValueError, match="task count"):
        method_simulation.retarget_additive_effect_scenarios(
            base, spec, family_size=3, expected_tasks_per_base=3
        )
    malformed = base.copy()
    malformed.loc[
        (malformed["replicate"].eq(1)) & malformed["hypothesis_id"].eq("H02"),
        "true_effect",
    ] = 0.2
    with pytest.raises(ValueError, match="active pattern"):
        method_simulation.retarget_additive_effect_scenarios(
            malformed, spec, family_size=3, expected_tasks_per_base=2
        )


def test_realistic_effect_power_entry_is_end_to_end_and_paired():
    base = _effect_retargeting_fixture()
    specs = [
        {
            "scenario_id": scenario_id,
            "base_scenario_id": "BASE",
            "structure_id": "sparse_iid",
            "effect_label": label,
            "target_effect": effect,
            "active_count": 1,
            "temporal_dependence": "iid",
        }
        for scenario_id, label, effect in (
            ("R01", "p10", 0.09),
            ("R02", "median", 0.13),
            ("R03", "p90", 0.18),
        )
    ]
    artifacts = method_simulation.evaluate_realistic_effect_power(
        base,
        specs,
        day_count=4,
        family_size=3,
        expected_tasks_per_base=2,
    )
    assert len(artifacts.scenario_summary) == 9
    assert len(artifacts.paired_effect_contrasts) == 6
    assert len(artifacts.paired_method_contrasts) == 6
    assert artifacts.paired_effect_contrasts["task_count"].eq(2).all()
    assert artifacts.paired_method_contrasts["task_count"].eq(2).all()
    assert set(artifacts.scenario_summary["target_effect"]) == {0.09, 0.13, 0.18}


def test_realistic_effect_power_complete_47_family_matches_hand_counted_path():
    rows = []
    for replicate in range(2):
        for index in range(47):
            active = index == 0
            true_effect = 0.2 if active else 0.0
            noise_mean = 2.93 if active and replicate == 0 else 0.0 if active else -5.0
            rows.append(
                {
                    "registered_task_idx": replicate,
                    "scenario_id": "BASE47",
                    "analysis_specification": "BASE47__right_tail_primary__fixture",
                    "replicate": replicate,
                    "hypothesis_id": f"H{index + 1:02d}",
                    "observed_effect": noise_mean + true_effect,
                    "uncalibrated_standard_error": 1.0,
                    "alternative": "greater",
                    "true_effect": true_effect,
                }
            )
    base = pd.DataFrame(rows)
    specs = [
        {
            "scenario_id": scenario_id,
            "base_scenario_id": "BASE47",
            "structure_id": "complete_47",
            "effect_label": label,
            "target_effect": effect,
            "active_count": 1,
            "temporal_dependence": "iid",
        }
        for scenario_id, label, effect in (
            ("R01", "p10", 0.09),
            ("R02", "median", 0.13),
            ("R03", "p90", 0.18),
        )
    ]
    artifacts = method_simulation.evaluate_realistic_effect_power(
        base,
        specs,
        day_count=4,
        family_size=47,
        expected_tasks_per_base=2,
    )
    practical = artifacts.scenario_summary.loc[
        artifacts.scenario_summary["method_variant"].eq("AR_BIC_1000_BH")
    ].sort_values("target_effect")
    assert practical["fdr"].tolist() == [0.0, 0.0, 0.0]
    assert practical["mean_true_positive_rate"].tolist() == [0.0, 0.0, 0.5]
    assert practical["mean_true_discovery_count"].tolist() == [0.0, 0.0, 0.5]
    assert practical["mean_false_discovery_count"].tolist() == [0.0, 0.0, 0.0]
    high = practical.iloc[-1]
    assert high["true_positive_rate_monte_carlo_standard_error"] == pytest.approx(0.5)
    assert high["true_positive_rate_ci95_lower"] == pytest.approx(0.0)
    assert high["true_positive_rate_ci95_upper"] == pytest.approx(1.0)
    contrast = artifacts.paired_effect_contrasts.loc[
        artifacts.paired_effect_contrasts["method_variant"].eq("AR_BIC_1000_BH")
        & artifacts.paired_effect_contrasts["lower_effect"].eq(0.13)
    ].iloc[0]
    assert contrast["mean_true_positive_rate_difference"] == pytest.approx(0.5)
    assert contrast["monte_carlo_standard_error"] == pytest.approx(0.5)


def test_bh_fdr_entry_fails_closed_instead_of_shrinking_nan_family():
    with pytest.raises(ValueError, match="must not be missing"):
        method_simulation.evaluate_bh_fdr_variants(
            _bh_fdr_fixture(invalid_p_input=True),
            dataset_id="fixture",
            scenario_temporal_dependence={"A06": "iid"},
            day_count=4,
            family_size=5,
        )


def test_bh_fdr_entry_fails_closed_on_incomplete_family():
    frame = _bh_fdr_fixture().iloc[:-1]
    with pytest.raises(ValueError, match="complete, ordered fixed-size families"):
        method_simulation.evaluate_bh_fdr_variants(
            frame,
            dataset_id="fixture",
            scenario_temporal_dependence={"A06": "iid"},
            day_count=4,
            family_size=5,
        )


def test_bh_fdr_entry_fails_closed_on_out_of_order_family():
    frame = _bh_fdr_fixture()
    first_task = frame["registered_task_idx"].eq(0)
    frame.loc[first_task] = frame.loc[first_task].iloc[[1, 0, 2, 3, 4]].to_numpy()
    with pytest.raises(ValueError, match="complete, ordered fixed-size families"):
        method_simulation.evaluate_bh_fdr_variants(
            frame,
            dataset_id="fixture",
            scenario_temporal_dependence={"A06": "iid"},
            day_count=4,
            family_size=5,
        )


def test_bh_fdr_entry_fails_closed_on_cross_task_hypothesis_mix():
    frame = _bh_fdr_fixture()
    frame.loc[0, ["registered_task_idx", "replicate"]] = [1, 1]
    with pytest.raises(
        ValueError,
        match="duplicate task hypotheses|complete, ordered fixed-size families",
    ):
        method_simulation.evaluate_bh_fdr_variants(
            frame,
            dataset_id="fixture",
            scenario_temporal_dependence={"A06": "iid"},
            day_count=4,
            family_size=5,
        )


def test_bh_fdr_entry_uses_direction_specific_truth():
    frame = _bh_fdr_fixture()
    frame.loc[frame["hypothesis_id"].eq("H02"), "true_effect"] = -0.2
    greater = method_simulation.evaluate_bh_fdr_variants(
        frame,
        dataset_id="greater",
        scenario_temporal_dependence={"A06": "iid"},
        day_count=4,
        family_size=5,
    )
    assert not greater.hypothesis_results.loc[
        greater.hypothesis_results["hypothesis_id"].eq("H02"),
        "is_true_alternative",
    ].any()
    two_sided = frame.copy()
    two_sided["alternative"] = "two-sided"
    two_sided["analysis_specification"] = "A06__two_sided_supplement__fixture"
    bilateral = method_simulation.evaluate_bh_fdr_variants(
        two_sided,
        dataset_id="two_sided",
        scenario_temporal_dependence={"A06": "iid"},
        day_count=4,
        family_size=5,
    )
    assert bilateral.hypothesis_results.loc[
        bilateral.hypothesis_results["hypothesis_id"].eq("H02"),
        "is_true_alternative",
    ].all()


def test_bh_fdr_entry_all_null_fdp_equals_any_false_discovery():
    frame = _bh_fdr_fixture()
    frame["true_effect"] = 0.0
    artifacts = method_simulation.evaluate_bh_fdr_variants(
        frame,
        dataset_id="all_null",
        scenario_temporal_dependence={"A06": "iid"},
        day_count=4,
        family_size=5,
    )
    task = artifacts.task_summary
    np.testing.assert_array_equal(
        task["false_discovery_proportion"].to_numpy(),
        task["any_false_discovery"].astype(float).to_numpy(),
    )
    assert task["true_positive_rate"].isna().all()
    calibration = artifacts.raw_p_value_calibration_summary
    assert calibration["truth_class"].eq("true_null").all()
    assert calibration["observation_count"].eq(10).all()
    assert calibration["null_uniform_ks_distance"].notna().all()


def test_bh_fdr_entry_reports_raw_p_calibration_by_truth_class():
    artifacts = method_simulation.evaluate_bh_fdr_variants(
        _bh_fdr_fixture(),
        dataset_id="fixture",
        scenario_temporal_dependence={"A06": "iid"},
        day_count=4,
        family_size=5,
    )
    calibration = artifacts.raw_p_value_calibration_summary
    base = calibration.loc[calibration["method_variant"].eq("AR_BIC_1000_BH")]
    assert set(base["truth_class"]) == {"true_null", "true_alternative"}
    assert base.set_index("truth_class").loc["true_alternative", "observation_count"] == 4
    assert base.set_index("truth_class").loc["true_null", "observation_count"] == 6
    assert np.isnan(
        base.set_index("truth_class").loc[
            "true_alternative", "null_uniform_ks_distance"
        ]
    )


def test_bh_fdr_entry_validates_frozen_1125_raw_p_values():
    frame = _bh_fdr_fixture()
    statistic = frame["observed_effect"] / 1.125
    frame["expected_1125_raw_p_value"] = method_simulation.normal_distribution.sf(
        statistic
    )
    method_simulation.evaluate_bh_fdr_variants(
        frame,
        dataset_id="matched",
        scenario_temporal_dependence={"A06": "iid"},
        day_count=4,
        family_size=5,
    )
    frame.loc[0, "expected_1125_raw_p_value"] += 0.01
    with pytest.raises(ValueError, match="differ from frozen"):
        method_simulation.evaluate_bh_fdr_variants(
            frame,
            dataset_id="mismatch",
            scenario_temporal_dependence={"A06": "iid"},
            day_count=4,
            family_size=5,
        )


def test_bh_fdr_cross_scenario_summary_excludes_two_sided_supplement():
    right = _bh_fdr_fixture()
    a10_right = _bh_fdr_fixture()
    a10_right["registered_task_idx"] += 10
    a10_right["scenario_id"] = "A10"
    a10_right["analysis_specification"] = "A10__right_tail_primary__fixture"
    a10_two_sided = _bh_fdr_fixture()
    a10_two_sided["registered_task_idx"] += 20
    a10_two_sided["scenario_id"] = "A10"
    a10_two_sided["analysis_specification"] = "A10__two_sided_supplement__fixture"
    a10_two_sided["alternative"] = "two-sided"
    frame = pd.concat([right, a10_right, a10_two_sided], ignore_index=True)
    artifacts = method_simulation.evaluate_bh_fdr_variants(
        frame,
        dataset_id="mixed_specs",
        scenario_temporal_dependence={"A06": "iid", "A10": "iid"},
        day_count=4,
        family_size=5,
    )
    assert artifacts.cross_scenario_summary["scenario_count"].eq(2).all()
    assert set(artifacts.cross_scenario_summary["analysis_family"]) == {
        "right_tail_primary"
    }
    assert set(artifacts.scenario_summary["analysis_family"]) == {
        "right_tail_primary",
        "two_sided_supplement",
    }


def _fault_decomposition_frame(engine: str, *, observed_shift: float = 0.0):
    rows = []
    for replicate in range(2):
        for index in range(47):
            true_effect = 0.35 if index < 5 else 0.0
            observed = (2.0 if index < 5 else 0.0) + observed_shift
            rows.append(
                {
                    "joint_inference_engine": engine,
                    "scenario_id": "A08",
                    "analysis_specification": f"A08__{engine}",
                    "replicate": replicate,
                    "inference_variant": f"{engine}_14d",
                    "dependence_length": 14,
                    "hypothesis_id": f"H{index + 1:02d}",
                    "observed_effect": observed,
                    "bootstrap_se": 0.5,
                    "raw_one_sided_p_value": 0.001 if index < 5 else 1.0,
                    "raw_two_sided_p_value": np.nan,
                    "stepdown_max_t_adjusted_p_value": (
                        0.01 if engine == "E1F" and index < 5 else 0.10 if index < 5 else 1.0
                    ),
                    "alternative": "greater",
                    "true_effect": true_effect,
                }
            )
    return pd.DataFrame(rows)


def test_fault_decomposition_locates_e1s_joint_adjustment_power_loss():
    e1f = _fault_decomposition_frame("E1F")
    e1s = _fault_decomposition_frame("E1S")
    result = method_simulation.evaluate_joint_inference_fault_decomposition(
        e1f,
        e1s,
        scenarios=[
            {
                "id": "A08",
                "temporal_dependence": "iid",
            }
        ],
        day_count=4,
    )
    assert result.decision["frozen_gate_feasible"] is True
    assert result.decision["E1F"]["a08_adjusted_each_power_pass"] is True
    assert result.decision["E1S"]["a08_marginal_each_power_pass"] is True
    assert result.decision["E1S"]["a08_adjusted_each_power_pass"] is False
    assert result.decision["E1S"]["direct_failure_layer"] == "joint_adjustment_power_loss"


def test_fault_decomposition_fails_closed_on_cross_engine_identity_changes():
    e1f = _fault_decomposition_frame("E1F")
    e1s = _fault_decomposition_frame("E1S")
    e1s.loc[0, "observed_effect"] += 0.01
    with pytest.raises(ValueError, match="observed_effect values differ"):
        method_simulation.evaluate_joint_inference_fault_decomposition(
            e1f,
            e1s,
            scenarios=[{"id": "A08", "temporal_dependence": "iid"}],
            day_count=4,
        )
    duplicated = pd.concat([e1f, e1f.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate simulation identities"):
        method_simulation.evaluate_joint_inference_fault_decomposition(
            duplicated,
            _fault_decomposition_frame("E1S"),
            scenarios=[{"id": "A08", "temporal_dependence": "iid"}],
            day_count=4,
        )


def test_fault_decomposition_selects_two_sided_raw_p_values_by_alternative():
    def with_two_sided(engine: str):
        right_tail = _fault_decomposition_frame(engine)
        two_sided = right_tail.copy()
        two_sided["scenario_id"] = "A10"
        two_sided["analysis_specification"] = f"A10__two_sided__{engine}"
        two_sided["alternative"] = "two-sided"
        two_sided["raw_one_sided_p_value"] = np.nan
        two_sided["raw_two_sided_p_value"] = np.where(
            two_sided["true_effect"].ne(0.0), 0.001, 1.0
        )
        return pd.concat([right_tail, two_sided], ignore_index=True)

    result = method_simulation.evaluate_joint_inference_fault_decomposition(
        with_two_sided("E1F"),
        with_two_sided("E1S"),
        scenarios=[
            {"id": "A08", "temporal_dependence": "iid"},
            {"id": "A10", "temporal_dependence": "iid"},
        ],
        day_count=4,
    )
    a10 = result.engine_hypothesis.loc[
        result.engine_hypothesis["scenario_id"].eq("A10")
        & result.engine_hypothesis["hypothesis_id"].eq("H01")
    ]
    assert len(a10) == 2
    assert a10["marginal_detection_rate"].eq(1.0).all()

    invalid = with_two_sided("E1S")
    invalid.loc[invalid["scenario_id"].eq("A10"), "raw_two_sided_p_value"] = np.nan
    with pytest.raises(ValueError, match="selected raw p-values"):
        method_simulation.evaluate_joint_inference_fault_decomposition(
            with_two_sided("E1F"),
            invalid,
            scenarios=[
                {"id": "A08", "temporal_dependence": "iid"},
                {"id": "A10", "temporal_dependence": "iid"},
            ],
            day_count=4,
        )


def test_dependence_profile_matches_hand_computed_denominator_and_pairs():
    index = pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "a": [-2.0, -1.0, 0.0, 1.0, 2.0],
            "b": [-4.0, -2.0, 0.0, 2.0, 4.0],
            "c": [2.0, 1.0, 0.0, -1.0, -2.0],
        },
        index=index,
    )
    result = method_simulation.dependence_profile(
        frame, ("g1", "g1", "g2"), max_lag=2
    )
    standardized = (frame - frame.mean()) / np.sqrt(
        np.square(frame - frame.mean()).mean(axis=0)
    )
    expected_lag1 = float(
        np.sum(standardized["a"].to_numpy()[1:] * standardized["a"].to_numpy()[:-1])
        / 5
    )
    actual_lag1 = result.temporal_by_hypothesis.set_index(
        ["hypothesis_id", "lag_days"]
    ).loc[("a", 1), "autocorrelation"]
    assert actual_lag1 == pytest.approx(expected_lag1)
    pairs = result.cross_pairs.set_index(
        ["left_hypothesis_id", "right_hypothesis_id"]
    )
    assert pairs.loc[("a", "b"), "relation"] == "within"
    assert pairs.loc[("a", "b"), "correlation"] == pytest.approx(1.0)
    assert pairs.loc[("a", "c"), "correlation"] == pytest.approx(-1.0)


def test_dependence_profile_fails_closed_on_missing_day_and_constant_column():
    index = pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-04"], utc=True)
    with pytest.raises(ValueError, match="strictly consecutive"):
        method_simulation.dependence_profile(
            pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [1.0, 0.0, -1.0]}, index=index),
            ("g1", "g2"),
            max_lag=1,
        )
    index = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
    with pytest.raises(ValueError, match="constant column"):
        method_simulation.dependence_profile(
            pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [1.0, 0.0, -1.0]}, index=index),
            ("g1", "g2"),
            max_lag=1,
        )
    duplicated = pd.DatetimeIndex([index[0], index[0], index[2]])
    with pytest.raises(ValueError, match="duplicate"):
        method_simulation.dependence_profile(
            pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [1.0, 0.0, -1.0]}, index=duplicated),
            ("g1", "g2"),
            max_lag=1,
        )
    with pytest.raises(ValueError, match="wrong hypothesis count"):
        method_simulation.dependence_profile(
            pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [1.0, 0.0, -1.0]}, index=index),
            ("g1", "g2"),
            max_lag=1,
            expected_hypothesis_count=47,
        )


def test_theoretical_family_max_quantiles_are_reproducible():
    first = method_simulation.theoretical_family_max_quantiles(
        (2, 1), within=0.4, between=0.1, seed=19, draw_count=1_000, batch_size=100
    )
    second = method_simulation.theoretical_family_max_quantiles(
        (2, 1), within=0.4, between=0.1, seed=19, draw_count=1_000, batch_size=100
    )
    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "q90"] < first.loc[0, "q95"] < first.loc[0, "q99"]
    assert first.loc[0, "generator_algorithm"] == "PCG64DXSM"
    assert first.loc[0, "matrix_factorization"] == "numpy.linalg.cholesky"
    assert first.loc[0, "float_dtype"] == "float64"
    assert first.loc[0, "quantile_method"] == "linear"
    child_a = np.random.SeedSequence(123).spawn(1)[0]
    child_b = np.random.SeedSequence(123).spawn(1)[0]
    direct_a = method_simulation.theoretical_family_max_quantiles(
        (2, 1), within=0.4, between=0.1, seed=child_a,
        draw_count=1_000, batch_size=100,
    )
    direct_b = method_simulation.theoretical_family_max_quantiles(
        (2, 1), within=0.4, between=0.1, seed=child_b,
        draw_count=1_000, batch_size=100,
    )
    pd.testing.assert_frame_equal(direct_a, direct_b)
    assert direct_a.loc[0, "seed_spawn_key"] == "0"


def test_e1_failure_decomposition_keeps_four_counterfactuals_separate():
    inference = pd.DataFrame(
        {
            "hypothesis_id": ["h1", "h2"],
            "true_effect": [0.0, 0.0],
            "observed_effect": [0.4, 0.0],
            "bootstrap_se": [0.1, 0.1],
            "observed_t": [4.0, 0.0],
            "stepdown_max_t_adjusted_p_value": [0.04, 1.0],
        }
    )
    detail, summary = method_simulation.decompose_e1_failure(
        inference,
        np.array([0.0, 1.0, 2.0, 3.0]),
        exact_true_mean_variance=0.04,
        theoretical_q95=2.5,
        alpha=0.20,
    )
    assert detail["estimated_to_exact_variance_ratio"].tolist() == pytest.approx([0.25, 0.25])
    assert summary["original_family_rejected"] is True
    assert summary["formal_e1_final_family_rejected"] is True
    assert summary["formal_e1_final_rejected_hypothesis_ids"] == "h1"
    assert summary["true_se_only_family_rejected"] is False
    assert summary["theory_critical_only_family_rejected"] is True
    assert summary["both_oracle_family_rejected"] is False
    assert summary["original_rejected_hypothesis_ids"] == "h1"
    assert summary["true_se_only_rejected_hypothesis_ids"] == ""
    assert summary["theory_critical_only_rejected_hypothesis_ids"] == "h1"
    assert summary["both_oracle_rejected_hypothesis_ids"] == ""


def test_minimal_a05_chain_has_frozen_exact_outputs():
    scenario = {
        "id": "A05",
        "hypothesis_dependence": "within_0.70_between_0.20",
        "temporal_dependence": "ma_14",
        "effect": "all_null",
    }
    dataset = method_simulation.generate_layer_a_dataset(
        scenario, day_count=40, group_sizes=(2, 1), seed=101
    )
    inference = method_simulation.infer_layer_a_dataset_with_engine_artifacts(
        dataset,
        engine="E1",
        dependence_length=14,
        n_bootstrap=999,
        seed=202,
        alternative="greater",
        production_equivalent=False,
    )
    exact = method_simulation.exact_gaussian_mean_variance("ma_14", day_count=40)
    detail, summary = method_simulation.decompose_e1_failure(
        inference.results,
        inference.bootstrap_max_statistics["bootstrap_max_test_statistic"],
        exact_true_mean_variance=exact["exact_mean_variance"],
        theoretical_q95=2.1,
    )
    assert exact["exact_mean_variance"] == pytest.approx(0.309375)
    assert detail["observed_effect"].tolist() == pytest.approx(
        [0.177429083186, 0.301746477733, -0.803579939507], abs=2e-6
    )
    assert summary["median_estimated_to_exact_variance_ratio"] == pytest.approx(
        0.30330971215788083
    )
    assert summary["bootstrap_max_q95"] == pytest.approx(6.4411986871806315)
    assert summary["original_family_rejected"] is False
    assert summary["both_oracle_family_rejected"] is False


def test_e1_failure_task_rejects_registry_identity_mismatch():
    design = _blueprint_file("ksv4_增量信息方法模拟设计清单.json")
    if not design.exists():
        pytest.skip("private frozen design is outside the qlab-only checkout")
    task = dict(method_simulation.registered_e1_failure_diagnostic_tasks(design)[0])
    task["replicate"] = 999
    with pytest.raises(ValueError, match="frozen registry"):
        method_simulation.run_e1_failure_diagnostic_task(
            design, task, theoretical_q95=2.0
        )


def test_compare_real_and_simulated_dependence_reports_mixed_without_weighted_score():
    real_temporal = pd.DataFrame(
        {"lag_days": range(1, 29), "autocorrelation_median": [0.5**lag for lag in range(1, 29)]}
    )
    simulated_temporal = pd.concat(
        [
            pd.DataFrame({
                "scenario_id": scenario,
                "lag_days": range(1, 29),
                "profile_median": [value**lag for lag in range(1, 29)],
            })
            for scenario, value in (("A03", 0.5), ("A04", 0.7), ("A05", 0.8))
        ],
        ignore_index=True,
    )
    real_cross = pd.DataFrame(
        {"relation": ["between", "within"], "correlation_mean": [0.2, 0.7]}
    )
    simulated_cross = pd.DataFrame(
        {
            "scenario_id": ["A03", "A03", "A04", "A04", "A05", "A05"],
            "relation": ["between", "within"] * 3,
            "profile_median": [0.05, 0.25, 0.2, 0.7, 0.25, 0.75],
        }
    )
    distances, decision = method_simulation.compare_real_and_simulated_dependence(
        real_temporal, real_cross, simulated_temporal, simulated_cross
    )
    assert decision == {
        "temporal_closest_scenario": "A03",
        "cross_closest_scenario": "A04",
        "overall_closest_scenario": "mixed",
        "interpretation": "temporal/cross winners differ or contain a tie",
    }
    assert not (distances["is_temporal_winner"] & distances["is_cross_winner"]).any()


def test_compare_real_and_simulated_dependence_reports_same_winner_and_ties():
    real_temporal = pd.DataFrame({
        "lag_days": range(1, 29),
        "autocorrelation_median": [0.5**lag for lag in range(1, 29)],
    })
    simulated_temporal = pd.concat([
        pd.DataFrame({
            "scenario_id": scenario,
            "lag_days": range(1, 29),
            "profile_median": [value**lag for lag in range(1, 29)],
        })
        for scenario, value in (("A03", 0.5), ("A04", 0.7), ("A05", 0.8))
    ], ignore_index=True)
    real_cross = pd.DataFrame({
        "relation": ["between", "within"],
        "correlation_mean": [0.05, 0.25],
    })
    simulated_cross = pd.DataFrame({
        "scenario_id": ["A03", "A03", "A04", "A04", "A05", "A05"],
        "relation": ["between", "within"] * 3,
        "profile_median": [0.05, 0.25, 0.2, 0.7, 0.25, 0.75],
    })
    _, decision = method_simulation.compare_real_and_simulated_dependence(
        real_temporal, real_cross, simulated_temporal, simulated_cross
    )
    assert decision["overall_closest_scenario"] == "A03"

    tied = simulated_cross.copy()
    tied.loc[tied["scenario_id"].eq("A04"), "profile_median"] = [0.05, 0.25]
    distances, decision = method_simulation.compare_real_and_simulated_dependence(
        real_temporal, real_cross, simulated_temporal, tied
    )
    assert decision["cross_closest_scenario"] == "tie:A03;A04"
    assert decision["overall_closest_scenario"] == "mixed"
    assert distances["is_cross_winner"].sum() == 2


def test_e1_failure_summary_aggregates_each_counterfactual_without_interaction_claim():
    replicate_rows = []
    temporal_rows = []
    cross_rows = []
    for scenario in ("A03", "A04", "A05"):
        for replicate in range(1_000):
            replicate_rows.append({
                "scenario_id": scenario,
                "replicate": replicate,
                "median_estimated_to_exact_variance_ratio": 0.8,
                "bootstrap_to_theoretical_q95_ratio": 0.9,
                "original_family_rejected": replicate < 50,
                "true_se_only_family_rejected": replicate < 40,
                "theory_critical_only_family_rejected": replicate < 30,
                "both_oracle_family_rejected": replicate < 20,
            })
            temporal_rows.extend({
                "scenario_id": scenario,
                "replicate": replicate,
                "lag_days": lag,
                "autocorrelation_median": 0.0,
            } for lag in range(1, 29))
            cross_rows.extend({
                "scenario_id": scenario,
                "replicate": replicate,
                "relation": relation,
                "correlation_mean": 0.0,
            } for relation in ("within", "between"))
    mechanism, _, _ = method_simulation.summarize_e1_failure_diagnostics(
        pd.DataFrame(replicate_rows),
        pd.DataFrame(temporal_rows),
        pd.DataFrame(cross_rows),
    )
    row = mechanism.set_index("scenario_id").loc["A03"]
    assert row["original_family_rejected_rate"] == pytest.approx(0.05)
    assert row["true_se_only_family_rejected_rate"] == pytest.approx(0.04)
    assert row["theory_critical_only_family_rejected_rate"] == pytest.approx(0.03)
    assert row["both_oracle_family_rejected_rate"] == pytest.approx(0.02)
    assert row["conditional_variance_fix_rate_difference"] == pytest.approx(0.01)
    assert row["conditional_critical_fix_rate_difference"] == pytest.approx(0.02)
    assert not any("interaction" in column for column in mechanism.columns)


def test_b06_function_is_centered_unit_variance_and_linearly_orthogonal():
    rng = np.random.default_rng(31)
    x = rng.standard_normal((500_000, 6))
    q = method_simulation.shared_sparse_low_order_nonlinearity(x)
    assert abs(q.mean()) < 0.01
    assert q.var() == pytest.approx(1.0, abs=0.02)
    assert np.max(np.abs(np.corrcoef(np.column_stack([q, x]), rowvar=False)[0, 1:])) < 0.01


def test_monte_carlo_summary_matches_hand_counted_family_errors():
    rows = []
    for replicate, p_values in enumerate(((0.01, 0.9), (0.2, 0.8))):
        for hypothesis, p_value in zip(("a", "b"), p_values):
            rows.append({"scenario_id": "zero", "replicate": replicate, "hypothesis_id": hypothesis, "true_effect": 0.0, "observed_effect": 0.0, "stepdown_max_t_adjusted_p_value": p_value})
    summary = method_simulation.summarize_monte_carlo(pd.DataFrame(rows)).scenario_summary.iloc[0]
    assert summary["family_wise_error_rate"] == pytest.approx(0.5)
    assert summary["family_wise_error_rate_mcse"] == pytest.approx(np.sqrt(0.125))
    assert np.isnan(summary["any_power"])
    assert np.isnan(summary["true_positive_rate"])


def test_two_sided_monte_carlo_treats_negative_effect_as_nonzero_signal():
    frame = pd.DataFrame(
        {
            "scenario_id": ["mixed"] * 3,
            "replicate": [0] * 3,
            "hypothesis_id": ["positive", "negative", "null"],
            "true_effect": [0.2, -0.2, 0.0],
            "observed_effect": [0.2, -0.2, 0.0],
            "stepdown_max_t_adjusted_p_value": [0.01, 0.01, 1.0],
            "alternative": ["two-sided"] * 3,
        }
    )
    summary = method_simulation.summarize_monte_carlo(frame).scenario_summary.iloc[0]
    assert summary["family_wise_error_rate"] == 0.0
    assert summary["any_power"] == 1.0
    assert summary["true_positive_rate"] == 1.0


def test_layer_a_engine_dispatches_e1_without_changing_truth(monkeypatch):
    dataset = method_simulation.LayerADataset(
        daily_values=pd.DataFrame(
            {"H01": [0.0, 1.0, -1.0, 0.5, -0.5]},
            index=pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC"),
        ),
        true_effects=pd.Series({"H01": 0.0}, name="true_effect"),
    )
    calls = []

    def fake_entry(centered, counts, effects, **kwargs):
        calls.append(kwargs)
        return method_simulation.research_stats.StepdownMaxTBootstrapArtifacts(
            summary=pd.DataFrame(
                {
                    "hypothesis_id": ["H01"],
                    "observed_effect": [0.0],
                    "stepdown_max_t_adjusted_p_value": [1.0],
                    "alternative": ["greater"],
                }
            ),
            bootstrap_t_values=pd.DataFrame({"H01": [0.0]}),
            block_starts=pd.DataFrame(),
        )

    monkeypatch.setattr(
        method_simulation.research_stats,
        "simulation_calibration_restudentized_circular_block_stepdown_max_t",
        fake_entry,
    )
    result = method_simulation.infer_layer_a_dataset_with_engine(
        dataset,
        engine="E1",
        dependence_length=2,
        n_bootstrap=999,
        seed=17,
    )
    assert calls == [
        {
            "block_length": 2,
            "n_bootstrap": 999,
            "seed": 17,
            "alternative": "greater",
        }
    ]
    assert result["joint_inference_engine"].tolist() == ["E1"]
    assert result["true_effect"].tolist() == [0.0]


def test_layer_a_engine_dispatches_e1f_without_changing_truth(monkeypatch):
    dataset = method_simulation.LayerADataset(
        daily_values=pd.DataFrame(
            {"H01": np.linspace(-1.0, 1.0, 16)},
            index=pd.date_range("2025-01-01", periods=16, freq="D", tz="UTC"),
        ),
        true_effects=pd.Series({"H01": 0.0}, name="true_effect"),
    )
    calls = []

    def fake_entry(centered, counts, effects, **kwargs):
        calls.append(kwargs)
        return method_simulation.research_stats.StepdownMaxTBootstrapArtifacts(
            summary=pd.DataFrame(
                {
                    "hypothesis_id": ["H01"],
                    "observed_effect": [0.0],
                    "stepdown_max_t_adjusted_p_value": [1.0],
                    "alternative": ["greater"],
                }
            ),
            bootstrap_t_values=pd.DataFrame({"H01": [0.0]}),
            block_starts=pd.DataFrame(),
        )

    monkeypatch.setattr(
        method_simulation.research_stats,
        "simulation_calibration_adaptive_flat_top_restudentized_stepdown_max_t",
        fake_entry,
    )
    result = method_simulation.infer_layer_a_dataset_with_engine(
        dataset, engine="E1F", dependence_length=14, n_bootstrap=999, seed=17
    )
    assert calls == [
        {
            "block_length": 14,
            "n_bootstrap": 999,
            "seed": 17,
            "alternative": "greater",
        }
    ]
    assert result["joint_inference_engine"].tolist() == ["E1F"]
    assert result["true_effect"].tolist() == [0.0]


def test_layer_a_engine_dispatches_e1s_without_changing_truth(monkeypatch):
    dataset = method_simulation.LayerADataset(
        daily_values=pd.DataFrame(
            {"H01": [0.0, 1.0, -1.0, 0.5, -0.5]},
            index=pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC"),
        ),
        true_effects=pd.Series({"H01": 0.0}, name="true_effect"),
    )
    calls = []

    def fake_entry(centered, counts, effects, **kwargs):
        calls.append(kwargs)
        return method_simulation.research_stats.StepdownMaxTBootstrapArtifacts(
            summary=pd.DataFrame(
                {
                    "hypothesis_id": ["H01"],
                    "observed_effect": [0.0],
                    "stepdown_max_t_adjusted_p_value": [1.0],
                    "alternative": ["greater"],
                }
            ),
            bootstrap_t_values=pd.DataFrame({"H01": [0.0]}),
            block_starts=pd.DataFrame(),
        )

    monkeypatch.setattr(
        method_simulation.research_stats,
        "simulation_calibration_self_normalized_stepdown_max_t",
        fake_entry,
    )
    result = method_simulation.infer_layer_a_dataset_with_engine(
        dataset, engine="E1S", dependence_length=2, n_bootstrap=999, seed=17
    )
    assert calls == [
        {
            "block_length": 2,
            "n_bootstrap": 999,
            "seed": 17,
            "alternative": "greater",
        }
    ]
    assert result["joint_inference_engine"].tolist() == ["E1S"]
    assert result["true_effect"].tolist() == [0.0]


def test_e1s_mechanism_preflight_is_an_obvious_failure_stop_rule():
    rows = []
    for scenario in ("A03", "A04", "A05"):
        for replicate in range(100):
            reject = replicate < (15 if scenario == "A05" else 14)
            for hypothesis in range(1, 48):
                rows.append(
                    {
                        "joint_inference_engine": "E1S",
                        "scenario_id": scenario,
                        "replicate": replicate,
                        "hypothesis_id": f"H{hypothesis:02d}",
                        "true_effect": 0.0,
                        "stepdown_max_t_adjusted_p_value": (
                            0.01 if reject and hypothesis == 1 else 1.0
                        ),
                    }
                )
    summary, decision = method_simulation.evaluate_e1s_mechanism_preflight(
        pd.DataFrame(rows)
    )
    assert decision == {
        "engine": "E1S",
        "status": "mechanism_preflight_failed",
        "pass": False,
        "failed_scenarios": ["A05"],
        "promotes_method": False,
    }
    assert summary.set_index("scenario_id").loc["A03", "false_family_rejection_count"] == 14
    assert summary.set_index("scenario_id").loc["A05", "false_family_rejection_count"] == 15


def _joint_gate_fixture(*, engine: str, scenario: str, false_replicates: set[int]):
    rows = []
    for replicate in range(100):
        for hypothesis_id, true_effect in (("null", 0.0), ("signal", 0.35)):
            rejected = (
                replicate in false_replicates
                if hypothesis_id == "null"
                else replicate < 90
            )
            rows.append(
                {
                    "joint_inference_engine": engine,
                    "scenario_id": scenario,
                    "analysis_specification": f"{scenario}_greater",
                    "replicate": replicate,
                    "hypothesis_id": hypothesis_id,
                    "true_effect": true_effect,
                    "alternative": "greater",
                    "stepdown_max_t_adjusted_p_value": 0.01 if rejected else 1.0,
                }
            )
    return rows


def test_joint_inference_development_gate_uses_false_family_and_each_a08_power():
    frame = pd.DataFrame(
        _joint_gate_fixture(engine="E1", scenario="A08", false_replicates={0, 1})
        + _joint_gate_fixture(
            engine="E2", scenario="A08", false_replicates=set(range(8))
        )
    )
    specification, hypothesis, decision = (
        method_simulation.evaluate_joint_inference_calibration(
            frame, phase="development"
        )
    )
    assert len(specification) == 2
    assert set(hypothesis["hypothesis_id"]) == {"signal"}
    assert decision["engines"]["E1"]["pass"] is True
    assert decision["engines"]["E2"]["calibration_pass"] is False
    assert decision["engines"]["E2"]["pass"] is False


def test_joint_inference_confirmation_gate_requires_equal_replicate_counts():
    rows = _joint_gate_fixture(
        engine="E1", scenario="A01", false_replicates={0, 1}
    )
    rows += _joint_gate_fixture(
        engine="E1", scenario="A02", false_replicates={0, 1}
    )[:-2]
    with pytest.raises(ValueError, match="equal replicate counts"):
        method_simulation.evaluate_joint_inference_calibration(
            pd.DataFrame(rows), phase="confirmation"
        )


def test_joint_inference_confirmation_gate_also_requires_a08_power():
    rows = _joint_gate_fixture(
        engine="E1H_AIC", scenario="A08", false_replicates=set()
    )
    for row in rows:
        if row["hypothesis_id"] == "signal" and row["replicate"] >= 70:
            row["stepdown_max_t_adjusted_p_value"] = 1.0
    _, _, decision = method_simulation.evaluate_joint_inference_calibration(
        pd.DataFrame(rows), phase="confirmation"
    )
    outcome = decision["engines"]["E1H_AIC"]
    assert outcome["pooled_pass"] is True
    assert outcome["each_specification_pass"] is True
    assert outcome["a08_each_true_hypothesis_power_pass"] is False
    assert outcome["pass"] is False


def test_confirmation_precision_match_detects_changed_specification_status():
    main = pd.DataFrame(
        [
            {
                "joint_inference_engine": "E1",
                "scenario_id": "A01",
                "analysis_specification": "A01__right_tail_primary__E1_main",
                "replicate": replicate,
                "hypothesis_id": "H01",
                "true_effect": 0.0,
                "alternative": "greater",
                "stepdown_max_t_adjusted_p_value": (
                    0.01 if replicate < 2 else 1.0
                ),
            }
            for replicate in range(100)
        ]
    )
    precision = main.copy()
    precision["analysis_specification"] = (
        "A01__right_tail_primary__E1_precision"
    )
    precision["stepdown_max_t_adjusted_p_value"] = [
        0.01 if replicate < 20 else 1.0 for replicate in range(100)
    ]
    comparison, decision = method_simulation.evaluate_confirmation_precision_match(
        main,
        precision,
        expected_precision_replicates_per_specification=100,
    )
    assert comparison["specification_status_changed"].tolist() == [True]
    assert decision["pass"] is False


def test_confirmation_precision_match_allows_probability_changes_without_gate_change():
    main = pd.DataFrame(
        [
            {
                "joint_inference_engine": "E1",
                "scenario_id": "A01",
                "analysis_specification": "A01__right_tail_primary__E1_main",
                "replicate": replicate,
                "hypothesis_id": "H01",
                "true_effect": 0.0,
                "alternative": "greater",
                "stepdown_max_t_adjusted_p_value": 1.0,
            }
            for replicate in range(100)
        ]
    )
    precision = main.copy()
    precision["analysis_specification"] = (
        "A01__right_tail_primary__E1_precision"
    )
    precision["stepdown_max_t_adjusted_p_value"] = 0.9
    comparison, decision = method_simulation.evaluate_confirmation_precision_match(
        main,
        precision,
        expected_precision_replicates_per_specification=100,
    )
    assert comparison["specification_status_changed"].tolist() == [False]
    assert decision["pass"] is True


def test_joint_inference_engine_selection_applies_one_percentage_point_tie_rule():
    decision = {
        "phase": "development",
        "engines": {
            "E1": {"pass": True, "worst_specification_fwer": 0.051},
            "E2": {"pass": True, "worst_specification_fwer": 0.045},
        },
    }
    assert method_simulation.select_development_joint_inference_engine(decision) == "E1"
    decision["engines"]["E2"]["worst_specification_fwer"] = 0.040
    assert method_simulation.select_development_joint_inference_engine(decision) == "E2"


def test_e1h_development_selection_uses_frozen_tie_rule():
    decision = {
        "engines": {
            "E1H_AIC": {"pass": True, "worst_specification_fwer": 0.051},
            "E1H_BIC": {"pass": True, "worst_specification_fwer": 0.045},
        }
    }
    assert method_simulation.select_e1h_development_specification(decision) == "E1H_AIC"
    decision["engines"]["E1H_BIC"]["worst_specification_fwer"] = 0.040
    assert method_simulation.select_e1h_development_specification(decision) == "E1H_BIC"
    decision["engines"]["E1H_AIC"]["pass"] = False
    assert method_simulation.select_e1h_development_specification(decision) == "E1H_BIC"
    decision["engines"]["E1H_BIC"]["pass"] = False
    with pytest.raises(ValueError, match="no E1H specification"):
        method_simulation.select_e1h_development_specification(decision)


def test_layer_a_engine_dispatches_e1h_without_bootstrap():
    dataset = method_simulation.LayerADataset(
        daily_values=pd.DataFrame(
            np.random.default_rng(903).normal(size=(100, 47)),
            index=pd.date_range("2025-01-01", periods=100, freq="D", tz="UTC"),
            columns=[f"H{index + 1:02d}" for index in range(47)],
        ),
        true_effects=pd.Series(
            np.zeros(47), index=[f"H{index + 1:02d}" for index in range(47)]
        ),
    )
    artifacts = method_simulation.infer_layer_a_dataset_with_engine_artifacts(
        dataset,
        engine="E1H_BIC",
        dependence_length=0,
        n_bootstrap=0,
        seed=7,
    )
    assert len(artifacts.results) == 47
    assert artifacts.bootstrap_max_statistics.empty
    assert set(artifacts.results["joint_inference_engine"]) == {"E1H_BIC"}
    assert artifacts.results["family_adjusted_p_value"].between(0.0, 1.0).all()


def test_joint_inference_task_registries_are_frozen_and_complete():
    try:
        blueprint_root = resolve_blueprint_root(Path(__file__).resolve().parents[2])
    except RuntimeError as exc:
        pytest.skip(f"private blueprint repository is unavailable: {exc}")
    prior_path = blueprint_root / "ksv4_增量信息方法模拟设计清单.json"
    revision = json.loads(
        (blueprint_root / "ksv4_增量信息方法修订设计清单.json").read_text()
    )
    development = method_simulation.registered_joint_inference_development_tasks(
        prior_path
    )
    assert len(development) == 8_500
    assert sum(row["alternative"] == "two-sided" for row in development) == 500
    confirmation = method_simulation.registered_joint_inference_confirmation_tasks(
        prior_path,
        master_seed=revision["random_seeds"]["layer_a_confirmation_master"],
        replicates_per_specification=revision["joint_inference"][
            "confirmation_replicates_per_specification"
        ],
    )
    assert len(confirmation) == 24_000
    assert len({row["dataset_seed"] for row in confirmation}) == 24_000
    assert len({row["main_inference_seed"] for row in confirmation}) == 24_000
    assert sum(row["alternative"] == "two-sided" for row in confirmation) == 2_000


def test_revision_layer_a_task_registries_freeze_development_and_confirmation():
    prior = _blueprint_file("ksv4_增量信息方法模拟设计清单.json")
    development = method_simulation.registered_joint_inference_development_tasks(
        prior
    )
    diagnostic = method_simulation.registered_e0_diagnostic_tasks(prior)
    confirmation = method_simulation.registered_joint_inference_confirmation_tasks(
        prior, master_seed=2026081121, replicates_per_specification=2
    )
    assert len(development) == 8_500
    assert len(diagnostic) == 5_000
    assert len(confirmation) == 24
    assert sum(row["scenario_id"] == "A10" for row in development) == 1_000
    assert {
        row["alternative"] for row in development if row["scenario_id"] == "A10"
    } == {"greater", "two-sided"}
    assert {row["scenario_id"] for row in diagnostic} == {
        "A01", "A02", "A03", "A04", "A05"
    }
    assert confirmation == method_simulation.registered_joint_inference_confirmation_tasks(
        prior, master_seed=2026081121, replicates_per_specification=2
    )
    assert len({row["dataset_seed"] for row in confirmation}) == len(confirmation)


def test_e1j_development_validation_requires_exact_frozen_identity():
    specification = pd.DataFrame(
        {
            "joint_inference_engine": ["E1J_BIC_1125"],
            "family_wise_error_rate": [0.047],
        }
    )
    hypothesis = pd.DataFrame(
        {
            "joint_inference_engine": ["E1J_BIC_1125"] * 5,
            "scenario_id": ["A08"] * 5,
            "rejection_rate": [0.848, 0.852, 0.86, 0.87, 0.88],
        }
    )
    decision = {"engines": {"E1J_BIC_1125": {"pass": True}}}
    assert method_simulation.validate_e1j_development_result(
        specification, hypothesis, decision
    ) == "E1J_BIC_1125"
    bad = specification.copy()
    bad.loc[0, "family_wise_error_rate"] = 0.048
    with pytest.raises(ValueError, match="does not reproduce"):
        method_simulation.validate_e1j_development_result(bad, hypothesis, decision)


def test_e1j_multiplier_calibration_mechanically_selects_first_eligible_grid_value():
    rows = []
    for task_idx, scenario in enumerate(("A05", "A08")):
        for hypothesis_idx in range(47):
            is_true = scenario == "A08" and hypothesis_idx < 5
            observed_t = (
                10.0 if is_true else (3.4 if scenario == "A05" and hypothesis_idx == 0 else -10.0)
            )
            rows.append(
                {
                    "joint_inference_engine": "E1H_BIC",
                    "registered_task_idx": task_idx,
                    "observed_t": observed_t,
                    "alternative": "greater",
                    "scenario_id": scenario,
                    "analysis_specification": f"{scenario}__right_tail_primary__E1H_BIC",
                    "replicate": 0,
                    "hypothesis_id": f"H{hypothesis_idx + 1:02d}",
                    "true_effect": 0.35 if is_true else 0.0,
                }
            )
    summary, selected = method_simulation.calibrate_e1j_standard_error_multiplier(
        pd.DataFrame(rows)
    )
    assert selected == 1.125
    assert not summary.loc[
        summary["standard_error_multiplier"].eq(1.1), "eligible"
    ].item()
    assert summary.loc[
        summary["standard_error_multiplier"].eq(1.125), "eligible"
    ].item()


def test_revision_task_generates_dataset_once_and_runs_requested_engines(monkeypatch):
    prior = _blueprint_file("ksv4_增量信息方法模拟设计清单.json")
    task = method_simulation.registered_joint_inference_development_tasks(prior)[0]
    calls = []

    def fake_inference(dataset, **kwargs):
        calls.append((id(dataset), kwargs))
        return method_simulation.LayerAInferenceArtifacts(
            results=pd.DataFrame(
                {
                    "joint_inference_engine": [kwargs["engine"]],
                    "hypothesis_id": ["H01"],
                    "observed_effect": [0.0],
                    "stepdown_max_t_adjusted_p_value": [1.0],
                    "alternative": [kwargs["alternative"]],
                    "n_bootstrap": [kwargs["n_bootstrap"]],
                    "true_effect": [0.0],
                    "is_true_positive": [False],
                    "is_true_null": [True],
                }
            ),
            bootstrap_max_statistics=pd.DataFrame(
                {"bootstrap_idx": [0], "bootstrap_max_test_statistic": [0.0]}
            ),
        )

    monkeypatch.setattr(
        method_simulation, "infer_layer_a_dataset_with_engine_artifacts", fake_inference
    )
    result = method_simulation.run_joint_inference_revision_task(
        prior,
        task,
        engines=("E1", "E2"),
        dependence_length=14,
        n_bootstrap=999,
    )
    assert len(calls) == 2
    assert calls[0][0] == calls[1][0]
    assert [call[1]["engine"] for call in calls] == ["E1", "E2"]
    assert all(call[1]["seed"] == task["main_inference_seed"] for call in calls)
    assert result["joint_inference_engine"].tolist() == ["E1", "E2"]


def test_revision_specifications_reuse_one_dataset_across_e0_block_lengths(monkeypatch):
    prior = _blueprint_file("ksv4_增量信息方法模拟设计清单.json")
    task = method_simulation.registered_e0_diagnostic_tasks(prior)[0]
    dataset_ids = []

    def fake_inference(dataset, **kwargs):
        dataset_ids.append(id(dataset))
        return method_simulation.LayerAInferenceArtifacts(
            results=pd.DataFrame(
                {
                    "joint_inference_engine": [kwargs["engine"]],
                    "hypothesis_id": ["H01"],
                    "observed_effect": [0.0],
                    "stepdown_max_t_adjusted_p_value": [1.0],
                    "alternative": [kwargs["alternative"]],
                    "n_bootstrap": [kwargs["n_bootstrap"]],
                    "true_effect": [0.0],
                    "is_true_positive": [False],
                    "is_true_null": [True],
                }
            ),
            bootstrap_max_statistics=pd.DataFrame(
                {"bootstrap_idx": [0], "bootstrap_max_test_statistic": [0.0]}
            ),
        )

    monkeypatch.setattr(
        method_simulation, "infer_layer_a_dataset_with_engine_artifacts", fake_inference
    )
    specifications = [
        {
            "inference_variant": f"E0_{days}d_999",
            "engine": "E0",
            "dependence_length": days,
            "n_bootstrap": 999,
            "inference_seed": task["main_inference_seed"],
            "production_equivalent": False,
        }
        for days in (1, 7, 14, 28)
    ]
    result = method_simulation.run_joint_inference_revision_specifications(
        prior, task, specifications=specifications
    )
    assert len(set(dataset_ids)) == 1
    assert result["dependence_length"].tolist() == [1, 7, 14, 28]


def test_development_engine_selection_uses_pass_then_frozen_tie_rule():
    decision = {
        "engines": {
            "E1": {"pass": True, "worst_specification_fwer": 0.050},
            "E2": {"pass": True, "worst_specification_fwer": 0.055},
        }
    }
    assert method_simulation.select_development_joint_inference_engine(decision) == "E1"
    decision["engines"]["E2"]["worst_specification_fwer"] = 0.039
    assert method_simulation.select_development_joint_inference_engine(decision) == "E2"
    decision["engines"]["E1"]["pass"] = False
    assert method_simulation.select_development_joint_inference_engine(decision) == "E2"
    decision["engines"]["E2"]["pass"] = False
    with pytest.raises(RuntimeError, match="neither E1 nor E2"):
        method_simulation.select_development_joint_inference_engine(decision)


def test_e0_root_diagnostic_separates_marginal_family_and_max_distribution():
    results = pd.DataFrame(
        {
            "joint_inference_engine": ["E0"] * 4,
            "scenario_id": ["A01"] * 4,
            "analysis_specification": ["A01__E0_1d_999"] * 4,
            "replicate": [0, 0, 1, 1],
            "hypothesis_id": ["H01", "H02", "H01", "H02"],
            "raw_one_sided_p_value": [0.01, 0.50, 0.20, 0.30],
            "stepdown_max_t_adjusted_p_value": [0.04, 0.80, 0.40, 0.50],
        }
    )
    maxima = pd.DataFrame(
        {
            "scenario_id": ["A01"] * 4,
            "analysis_specification": ["A01__E0_1d_999"] * 4,
            "replicate": [0, 0, 1, 1],
            "bootstrap_idx": [0, 1, 0, 1],
            "bootstrap_max_test_statistic": [1.0, 2.0, 3.0, 4.0],
        }
    )
    marginal, family, maximum = method_simulation.summarize_e0_root_diagnostic(
        results, maxima
    )
    assert marginal.set_index("hypothesis_id").loc["H01", "marginal_rejection_rate"] == 0.5
    assert family.iloc[0]["family_rejection_rate"] == 0.5
    assert maximum.iloc[0]["bootstrap_draw_count"] == 4
    assert maximum.iloc[0]["max_stat_q50"] == pytest.approx(2.5)


def test_resource_projection_fails_if_either_budget_is_exceeded():
    result = method_simulation.resource_projection(
        measured_cpu_seconds=3600, measured_output_bytes=100,
        measured_model_fits=10, total_model_fits=100,
        available_disk_bytes=1000, cpu_hour_limit=9, disk_fraction_limit=0.5,
    )
    assert result["projected_cpu_hours"] == 10
    assert result["projected_output_bytes"] == 1000
    assert result["preflight_pass"] is False


def test_staged_resource_projection_sums_registered_multipliers():
    result = method_simulation.staged_resource_projection(
        pd.DataFrame({
            "stage": ["A", "B"],
            "measured_cpu_seconds": [2.0, 10.0],
            "measured_output_bytes": [100, 1000],
            "workload_multiplier": [5.0, 2.0],
        }),
        available_disk_bytes=10_000,
        cpu_hour_limit=1.0,
        disk_fraction_limit=0.5,
    )
    assert result["projected_cpu_hours"] == pytest.approx(30.0 / 3600.0)
    assert result["projected_output_bytes"] == 2500
    assert result["preflight_pass"] is True


def test_staged_resource_projection_treats_cpu_as_reporting_only_but_enforces_memory():
    measurements = pd.DataFrame({
        "stage": ["B"],
        "measured_cpu_seconds": [3600.0],
        "measured_output_bytes": [10],
        "workload_multiplier": [100.0],
    })
    accepted = method_simulation.staged_resource_projection(
        measurements,
        available_disk_bytes=10_000,
        cpu_hour_limit=72.0,
        disk_fraction_limit=0.5,
        cpu_hours_reporting_only=True,
        peak_rss_bytes=70,
        physical_memory_bytes=100,
        memory_fraction_limit=0.75,
    )
    assert accepted["projected_cpu_hours"] == 100.0
    assert accepted["cpu_within_legacy_limit"] is False
    assert accepted["preflight_pass"] is True

    rejected = method_simulation.staged_resource_projection(
        measurements,
        available_disk_bytes=10_000,
        cpu_hour_limit=72.0,
        disk_fraction_limit=0.5,
        cpu_hours_reporting_only=True,
        peak_rss_bytes=76,
        physical_memory_bytes=100,
        memory_fraction_limit=0.75,
    )
    assert rejected["memory_pass"] is False
    assert rejected["preflight_pass"] is False


def test_registered_preflight_workload_multipliers_cover_every_frozen_stage():
    manifest_path = _blueprint_file("ksv4_增量信息方法模拟设计清单.json")
    assert method_simulation.registered_preflight_workload_multipliers(
        manifest_path
    ) == {
        "A_main_499": 8_000,
        "A_production_10000": 100,
        "A_sensitivity_7d_499": 300,
        "A_sensitivity_28d_499": 300,
        "B_C_complete_499": 114,
        "B_C_production_inference_10000": 30,
        "temporal_falsification_complete": 1,
    }


def test_registered_layer_b_tasks_cover_frozen_114_in_manifest_order():
    manifest_path = _blueprint_file("ksv4_增量信息方法模拟设计清单.json")
    tasks = method_simulation.registered_layer_b_tasks(manifest_path)
    assert len(tasks) == 114
    assert [row["task_idx"] for row in tasks] == list(range(114))
    assert [row["case_id"] for row in tasks[:20]] == ["B01"] * 20
    assert [row["replicate"] for row in tasks[:20]] == list(range(20))
    assert [row["case_id"] for row in tasks[20:40]] == ["B06"] * 20
    assert [row["case_id"] for row in tasks[40:60]] == ["B07-M"] * 20
    assert sum(row["scenario_id"] == "B07-M" for row in tasks) == 44
    assert [row["case_id"] for row in tasks[-6:]] == [
        "B10-B01",
        "B10-B01",
        "B10-B01",
        "B10-B07-M",
        "B10-B07-M",
        "B10-B07-M",
    ]
    assert len({row["task_id"] for row in tasks}) == 114

    production = method_simulation.registered_layer_b_production_tasks(manifest_path)
    assert len(production) == 30
    assert {row["case_id"] for row in production} == {"B01", "B06", "B07-M"}
    assert all(row["replicate"] < 10 for row in production)
    assert len({row["production_inference_seed"] for row in production}) == 30
    manifest = method_simulation.load_frozen_design(manifest_path)
    children = np.random.SeedSequence(
        int(manifest["random_seeds"]["layer_b_c_master"])
    ).spawn(114)
    expected = {
        task["task_idx"]: int(children[task["task_idx"]].generate_state(1)[0])
        for task in production
    }
    assert {
        task["task_idx"]: task["production_inference_seed"] for task in production
    } == expected


def test_registered_layer_a_tasks_cover_frozen_variants_and_are_reproducible():
    manifest_path = _blueprint_file("ksv4_增量信息方法模拟设计清单.json")
    first = method_simulation.registered_layer_a_tasks(manifest_path)
    second = method_simulation.registered_layer_a_tasks(manifest_path)
    assert first == second
    assert len(first) == 8_000
    assert [row["task_idx"] for row in first] == list(range(8_000))
    assert sum(row["run_production_equivalent"] for row in first) == 100
    assert sum(row["run_block_sensitivity"] for row in first) == 300
    assert len({row["dataset_seed"] for row in first}) == 8_000


def test_layer_a_variant_entry_runs_only_the_requested_registered_variant(monkeypatch):
    manifest_path = _blueprint_file("ksv4_增量信息方法模拟设计清单.json")
    tasks = method_simulation.registered_layer_a_tasks(manifest_path)
    task = next(row for row in tasks if row["run_production_equivalent"])
    calls = []

    monkeypatch.setattr(
        method_simulation,
        "generate_layer_a_dataset",
        lambda *args, **kwargs: object(),
    )

    def fake_infer(dataset, **kwargs):
        del dataset
        calls.append(kwargs)
        return pd.DataFrame(
            {
                "hypothesis_id": ["H01"],
                "observed_effect": [0.0],
                "stepdown_max_t_adjusted_p_value": [1.0],
                "true_effect": [0.0],
            }
        )

    monkeypatch.setattr(method_simulation, "infer_layer_a_dataset", fake_infer)
    result = method_simulation.run_registered_layer_a_task_variant(
        manifest_path, task, "production_14d_10000"
    )
    assert result["analysis_variant"].tolist() == ["production_14d_10000"]
    assert len(calls) == 1
    assert calls[0]["n_bootstrap"] == 10_000
    assert calls[0]["production_equivalent"] is True


def test_a10_two_sided_variant_is_explicit_and_rejects_other_scenarios(monkeypatch):
    manifest_path = _blueprint_file("ksv4_增量信息方法模拟设计清单.json")
    tasks = method_simulation.registered_layer_a_tasks(manifest_path)
    a10_task = next(row for row in tasks if row["scenario_id"] == "A10")
    a09_task = next(row for row in tasks if row["scenario_id"] == "A09")
    calls = []
    dataset_seeds = []
    monkeypatch.setattr(
        method_simulation,
        "generate_layer_a_dataset",
        lambda *args, **kwargs: dataset_seeds.append(kwargs["seed"]) or object(),
    )

    def fake_infer(dataset, **kwargs):
        del dataset
        calls.append(kwargs)
        return pd.DataFrame(
            {
                "hypothesis_id": ["H01"],
                "observed_effect": [0.0],
                "stepdown_max_t_adjusted_p_value": [1.0],
                "true_effect": [0.0],
                "alternative": [kwargs["alternative"]],
                "raw_two_sided_p_value": [0.5],
            }
        )

    monkeypatch.setattr(method_simulation, "infer_layer_a_dataset", fake_infer)
    result = method_simulation.run_registered_layer_a_task_variant(
        manifest_path, a10_task, "a10_two_sided_14d_499"
    )
    assert calls[0]["alternative"] == "two-sided"
    assert calls[0]["block_length"] == 14
    assert calls[0]["n_bootstrap"] == 499
    assert calls[0]["seed"] == a10_task["main_inference_seed"]
    assert dataset_seeds == [a10_task["dataset_seed"]]
    assert result["alternative"].tolist() == ["two-sided"]
    assert result["raw_two_sided_p_value"].notna().all()
    with pytest.raises(ValueError, match="registered only for A10"):
        method_simulation.run_registered_layer_a_task_variant(
            manifest_path, a09_task, "a10_two_sided_14d_499"
        )


def test_matched_layer_a_summary_requires_identical_registered_replicates():
    rows = []
    for variant in ("v499", "v10000"):
        for replicate in (0, 1):
            rows.append(
                {
                    "scenario_id": "A01",
                    "analysis_variant": variant,
                    "replicate": replicate,
                    "hypothesis_id": "H01",
                    "true_effect": 0.0,
                    "observed_effect": 0.0,
                    "stepdown_max_t_adjusted_p_value": 1.0,
                }
            )
    frame = pd.DataFrame(rows)
    summary = method_simulation.summarize_matched_registered_layer_a_results(
        frame,
        scenario_ids=("A01",),
        analysis_variants=("v499", "v10000"),
        replicate_count=2,
    )
    assert set(summary.scenario_summary["analysis_variant"]) == {"v499", "v10000"}
    broken = frame.loc[
        ~(
            frame["analysis_variant"].eq("v10000")
            & frame["replicate"].eq(1)
        )
    ]
    with pytest.raises(ValueError, match="incomplete replicate support"):
        method_simulation.summarize_matched_registered_layer_a_results(
            broken,
            scenario_ids=("A01",),
            analysis_variants=("v499", "v10000"),
            replicate_count=2,
        )


def _write_federated_fixture(root, tasks, source_id, task):
    root.mkdir()
    runtime = {
        "status": "frozen",
        "identity": {
            "design_sha256": "design",
            "source_sha256": {"science.py": "same"},
            "environment": {"platform": source_id, "python": "3.12"},
            "data_root": f"/{source_id}",
        },
    }
    runtime_path = root / "runtime_manifest.json"
    runtime_path.write_text(json.dumps(runtime, sort_keys=True), encoding="utf-8")
    (root / "registered_task_manifest.csv").write_text(
        pd.DataFrame(tasks).to_csv(index=False), encoding="utf-8"
    )
    task_dir = root / task["task_id"]
    task_dir.mkdir()
    cells = []
    for residual in method_simulation.RESIDUAL_METHODS:
        for adjustment in method_simulation.FAMILY_ADJUSTMENTS:
            cells.append(
                {
                    "cell_id": f"{residual}-{adjustment}",
                    "residual_method": residual,
                    "family_adjustment": adjustment,
                    **method_simulation.correction_identity_for_code(adjustment),
                    "hypothesis_count": 4,
                    "true_positive_count": 0,
                    "rejection_count": 0,
                    "true_positive_rejection_count": 0,
                    "true_null_rejection_count": 0,
                    "mean_estimate": 0.0,
                    "mean_true_effect": 0.0,
                    "mean_bias": 0.0,
                    "mean_squared_error": 0.0,
                    "p_values_json": "{}",
                }
            )
    required = {
        "candidate_mapping.csv": pd.DataFrame(),
        "alias_mapping.csv": pd.DataFrame(),
        "fold_manifest.csv": pd.DataFrame(),
        "decision_moments.csv.gz": pd.DataFrame({"x": [1]}),
        "observations.csv.gz": pd.DataFrame({"x": [1]}),
        "comparison_grid.csv": pd.DataFrame(cells),
        "bootstrap_starts.csv.gz": pd.DataFrame({"x": [1]}),
    }
    hashes = {}
    for name, frame in required.items():
        path = task_dir / name
        frame.to_csv(path, index=False)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    runtime_sha = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    receipt = {
        "status": "complete",
        "design_sha256": "design",
        "runtime_manifest_sha256": runtime_sha,
        "file_sha256": hashes,
        **task,
    }
    (task_dir / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")


def test_federated_validation_accepts_different_runtimes_and_fails_on_tamper(tmp_path):
    tasks = (
        {"task_idx": 0, "task_id": "task_000", "case_id": "B01", "scenario_id": "B01", "replicate": 0, "base_state": None, "beta": None, "q_x": None},
        {"task_idx": 1, "task_id": "task_001", "case_id": "B01", "scenario_id": "B01", "replicate": 1, "base_state": None, "beta": None, "q_x": None},
    )
    local, remote = tmp_path / "local", tmp_path / "remote"
    _write_federated_fixture(local, tasks, "local", tasks[0])
    _write_federated_fixture(remote, tasks, "remote", tasks[1])
    artifacts = method_simulation.validate_federated_layer_bc_tasks(
        tasks, {"local": local, "remote": remote},
        design_sha256="design", baseline_source_id="local",
    )
    assert artifacts.receipt["status"] == "complete"
    assert artifacts.receipt["task_count"] == 2
    assert artifacts.receipt["runtime_count"] == 2
    assert artifacts.task_inventory["task_idx"].tolist() == [0, 1]
    (remote / "task_001" / "comparison_grid.csv").write_text("tampered\n")
    with pytest.raises(RuntimeError, match="file hash mismatch"):
        method_simulation.validate_federated_layer_bc_tasks(
            tasks, {"local": local, "remote": remote},
            design_sha256="design", baseline_source_id="local",
        )


def test_federated_validation_accepts_bound_external_observation_hash_audit(tmp_path):
    tasks = (
        {"task_idx": 0, "task_id": "task_000", "case_id": "B01", "scenario_id": "B01", "replicate": 0, "base_state": None, "beta": None, "q_x": None},
        {"task_idx": 1, "task_id": "task_001", "case_id": "B01", "scenario_id": "B01", "replicate": 1, "base_state": None, "beta": None, "q_x": None},
    )
    local, remote = tmp_path / "local", tmp_path / "remote"
    _write_federated_fixture(local, tasks, "local", tasks[0])
    _write_federated_fixture(remote, tasks, "remote", tasks[1])
    task_dir = remote / "task_001"
    task_receipt = json.loads((task_dir / "receipt.json").read_text(encoding="utf-8"))
    audit = pd.DataFrame(
        {
            "task_id": ["task_001"],
            "file_name": ["observations.csv.gz"],
            "sha256": [task_receipt["file_sha256"]["observations.csv.gz"]],
        }
    )
    audit_path = remote / "external_file_audit.csv"
    audit.to_csv(audit_path, index=False)
    raw_path = remote / "external_observation_sha256sum.txt"
    raw_line = (
        f"{task_receipt['file_sha256']['observations.csv.gz']}  "
        "/remote/formal_bc/task_001/observations.csv.gz\n"
    )
    raw_path.write_text(raw_line, encoding="utf-8")
    audit_receipt = {
        "status": "complete",
        "source_id": "remote",
        "design_sha256": "design",
        "runtime_manifest_sha256": hashlib.sha256(
            (remote / "runtime_manifest.json").read_bytes()
        ).hexdigest(),
        "registered_task_manifest_sha256": hashlib.sha256(
            (remote / "registered_task_manifest.csv").read_bytes()
        ).hexdigest(),
        "audit_table_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "raw_sha256sum_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "audited_file_name": "observations.csv.gz",
        "audited_file_count": 1,
    }
    (remote / "external_file_audit_receipt.json").write_text(
        json.dumps(audit_receipt), encoding="utf-8"
    )
    (task_dir / "observations.csv.gz").unlink()

    artifacts = method_simulation.validate_federated_layer_bc_tasks(
        tasks, {"local": local, "remote": remote},
        design_sha256="design", baseline_source_id="local",
    )
    assert artifacts.receipt["task_count"] == 2
    assert artifacts.runtime_inventory.loc[
        artifacts.runtime_inventory["source_id"] == "remote",
        "external_file_audit_receipt_sha256",
    ].notna().all()

    raw_path.unlink()
    with pytest.raises(RuntimeError, match="incomplete external audit"):
        method_simulation.validate_federated_layer_bc_tasks(
            tasks, {"local": local, "remote": remote},
            design_sha256="design", baseline_source_id="local",
        )
    raw_path.write_text(raw_line.replace("task_001", "task_999"), encoding="utf-8")
    with pytest.raises(RuntimeError, match="external audit is invalid"):
        method_simulation.validate_federated_layer_bc_tasks(
            tasks, {"local": local, "remote": remote},
            design_sha256="design", baseline_source_id="local",
        )

    audit_receipt["raw_sha256sum_sha256"] = hashlib.sha256(
        raw_path.read_bytes()
    ).hexdigest()
    (remote / "external_file_audit_receipt.json").write_text(
        json.dumps(audit_receipt), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="disagrees with audit table"):
        method_simulation.validate_federated_layer_bc_tasks(
            tasks, {"local": local, "remote": remote},
            design_sha256="design", baseline_source_id="local",
        )

    raw_path.write_text(raw_line, encoding="utf-8")
    audit_receipt["raw_sha256sum_sha256"] = hashlib.sha256(
        raw_path.read_bytes()
    ).hexdigest()
    (remote / "external_file_audit_receipt.json").write_text(
        json.dumps(audit_receipt), encoding="utf-8"
    )
    audit.loc[0, "sha256"] = "0" * 64
    audit.to_csv(audit_path, index=False)
    with pytest.raises(RuntimeError, match="external audit is invalid"):
        method_simulation.validate_federated_layer_bc_tasks(
            tasks, {"local": local, "remote": remote},
            design_sha256="design", baseline_source_id="local",
        )


def test_sha256sum_file_audit_parser_has_one_strict_contract():
    digest = "a" * 64
    parsed = method_simulation.parse_sha256sum_file_audit(
        f"{digest}  /remote/task_001/observations.csv.gz\n",
        allowed_file_names=("observations.csv.gz",),
    )
    assert parsed.to_dict("records") == [
        {
            "task_id": "task_001",
            "file_name": "observations.csv.gz",
            "sha256": digest,
        }
    ]
    with pytest.raises(ValueError, match="invalid task file hash"):
        method_simulation.parse_sha256sum_file_audit(
            f"{digest}  /remote/task_001/decision_moments.csv.gz\n",
            allowed_file_names=("observations.csv.gz",),
        )
    with pytest.raises(ValueError, match="empty or duplicated"):
        method_simulation.parse_sha256sum_file_audit(
            (
                f"{digest}  /remote/task_001/observations.csv.gz\n"
                f"{digest}  /remote/task_001/observations.csv.gz\n"
            ),
            allowed_file_names=("observations.csv.gz",),
        )


def test_layer_bc_summary_matches_hand_counted_family_rates():
    rows = []
    for replicate, false_count, true_count in ((0, 1, 1), (1, 0, 2)):
        rows.append(
            {
                "case_id": "B",
                "replicate": replicate,
                "cell_id": "R5-C2",
                "residual_method": "R5",
                "family_adjustment": "C2",
                **method_simulation.correction_identity_for_code("C2"),
                "hypothesis_count": 4,
                "true_positive_count": 2,
                "true_positive_rejection_count": true_count,
                "true_null_rejection_count": false_count,
                "mean_estimate": 0.2,
                "mean_bias": 0.0,
                "mean_squared_error": 0.1,
            }
        )
    summary = method_simulation.summarize_layer_bc_results(pd.DataFrame(rows))
    row = summary.scenario_summary.iloc[0]
    assert row["method_id"] == "legacy.family_adjustment.stepdown_maxT@v1"
    assert row["false_family_detection_rate"] == pytest.approx(0.5)
    assert row["true_positive_rejection_rate"] == pytest.approx(0.75)
    assert row["any_true_detection_rate"] == pytest.approx(1.0)

    null_rows = pd.DataFrame(rows).assign(
        case_id="zero",
        true_positive_count=0,
        true_positive_rejection_count=0,
    )
    null_summary = method_simulation.summarize_layer_bc_results(null_rows)
    assert np.isnan(null_summary.scenario_summary.iloc[0]["any_true_detection_rate"])

    incomplete = pd.DataFrame(rows).drop(columns=["method_id"])
    with pytest.raises(ValueError, match="Layer B/C results missing columns"):
        method_simulation.summarize_layer_bc_results(incomplete)


def test_correction_identity_contract_disambiguates_legacy_short_codes():
    assert method_simulation.FAMILY_ADJUSTMENTS == ("C0", "C1", "C2")
    assert method_simulation.correction_identity_for_code("C0") == {
        "identity_schema_version": "correction_identity_v1",
        "namespace": "legacy.family_adjustment",
        "method_id": "legacy.family_adjustment.raw@v1",
        "algorithm": "raw p-values, no family adjustment",
        "legacy_code": "C0",
    }
    assert method_simulation.correction_identity_for_code("C1")["method_id"] == (
        "legacy.family_adjustment.holm@v1"
    )
    assert method_simulation.correction_identity_for_code("C2")["method_id"] == (
        "legacy.family_adjustment.stepdown_maxT@v1"
    )
    with pytest.raises(ValueError, match="unknown correction identity"):
        method_simulation.correction_identity_for_code("C9")


@pytest.fixture(scope="module")
def reduced_layer_b_c_runs():
    original_estimator = method_simulation.substitution._registered_estimator

    def cheap_registered_estimator(model_class, parameters):
        del model_class, parameters
        return Ridge(alpha=0.5)

    method_simulation.substitution._registered_estimator = cheap_registered_estimator
    manifest_path = _blueprint_file("ksv4_增量信息方法模拟设计清单.json")
    overrides = {
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
                "hist_gbm": ({"max_depth": 2, "max_iter": 100, "learning_rate": 0.1},),
                "random_forest": ({"max_depth": 3, "n_estimators": 200},),
                "poly2_ridge": ({"alpha": 1.0},),
                "poly2_elastic_net": (
                    {"alpha": 1.0, "l1_ratio": 0.5, "max_iter": 10_000, "tol": 1e-6},
                ),
            },
        "allow_model_subset": True,
        "mcar_probability": 0.0,
        "missing_gap_objects": 1,
        "missing_gap_days": 4,
    }
    try:
        yield {
            scenario: method_simulation.run_layer_b_c_simulation(
                manifest_path,
                scenario_id=scenario,
                replicate=0,
                base_state="B07-M" if scenario == "B10" else "B01",
                seed=8123,
                test_overrides=overrides,
            )
            for scenario in ("B01", "B06", "B07-M", "B09", "B10")
        }
    finally:
        method_simulation.substitution._registered_estimator = original_estimator


def test_layer_b_c_entry_covers_fixed_21_cells_and_truth_states(reduced_layer_b_c_runs):
    zero = reduced_layer_b_c_runs["B01"]
    shared_zero = reduced_layer_b_c_runs["B06"]
    positive = reduced_layer_b_c_runs["B07-M"]
    expected_cells = {
        f"R{method}-C{adjustment}"
        for method in range(7)
        for adjustment in range(3)
    }
    for artifacts in (zero, shared_zero, positive):
        grid = artifacts.layer_c.comparison_grid
        assert len(grid) == 21
        assert set(grid["cell_id"]) == expected_cells
        assert grid.groupby("residual_method").size().eq(3).all()
        assert grid["hypothesis_count"].eq(4).all()
        assert artifacts.layer_c.decision_moments["residual_method"].nunique() == 7
        assert artifacts.layer_c.observations["residual_method"].nunique() == 7
        assert artifacts.layer_c.bootstrap_starts["residual_method"].nunique() == 7
    assert zero.layer_c.comparison_grid["true_positive_count"].eq(0).all()
    assert zero.layer_c.comparison_grid["mean_true_effect"].eq(0.0).all()
    assert shared_zero.layer_c.comparison_grid["true_positive_count"].eq(0).all()
    assert shared_zero.layer_c.comparison_grid["mean_true_effect"].eq(0.0).all()
    assert positive.layer_c.comparison_grid["true_positive_count"].eq(2).all()
    assert positive.layer_c.comparison_grid["mean_true_effect"].gt(0.0).all()
    starts = positive.layer_c.bootstrap_starts
    reference = starts.loc[starts["residual_method"] == "R0"].drop(
        columns="residual_method"
    ).reset_index(drop=True)
    for method in method_simulation.RESIDUAL_METHODS[1:]:
        pd.testing.assert_frame_equal(
            reference,
            starts.loc[starts["residual_method"] == method]
            .drop(columns="residual_method")
            .reset_index(drop=True),
        )


def test_layer_c_inference_can_be_repeated_without_refitting_models(
    reduced_layer_b_c_runs,
):
    artifacts = reduced_layer_b_c_runs["B01"]
    seed = int(np.random.SeedSequence([8123, 0, 91]).generate_state(1)[0])
    repeated = method_simulation.infer_layer_c_comparison(
        artifacts.layer_c.decision_moments,
        expected_decisions_per_day={
            horizon: 24 // hours
            for horizon, hours in method_simulation.HORIZON_HOURS.items()
        },
        block_length=2,
        n_bootstrap=499,
        seed=seed,
    )
    pd.testing.assert_frame_equal(
        repeated.comparison_grid, artifacts.layer_c.comparison_grid
    )
    pd.testing.assert_frame_equal(
        repeated.bootstrap_starts, artifacts.layer_c.bootstrap_starts
    )


def test_layer_c_residual_stream_to_final_grid_matches_hand_calculation():
    horizon_hours = {"4h": 4, "8h": 8, "12h": 12, "1d": 24}
    day_moments = {
        "H4": ("4h", 1.0, 3.0, 2.0),
        "H8": ("8h", 2.0, 4.0, 3.0),
        "H12": ("12h", -1.0, 1.0, 0.0),
        "H1D": ("1d", 0.0, 2.0, 1.0),
    }
    moments = []
    for hypothesis_id, (horizon, first, second, truth) in day_moments.items():
        signal_rows = []
        outcome_rows = []
        for day_offset, moment in enumerate((first, second)):
            day = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=day_offset)
            for hour in range(0, 24, horizon_hours[horizon]):
                decision_ts = day + pd.Timedelta(hours=hour)
                for symbol, sign in (("BTC", 1.0), ("ETH", -1.0)):
                    signal_rows.append(
                        {
                            "fold_idx": 0,
                            "decision_ts": decision_ts,
                            "symbol": symbol,
                            "target_signal": 0.5 + sign,
                            "replica_signal": 0.5,
                            "residual_signal": sign,
                            "source_model_class": "hand_fixture",
                        }
                    )
                    outcome_rows.append(
                        {
                            "fold_idx": 0,
                            "decision_ts": decision_ts,
                            "symbol": symbol,
                            "target_signal": sign * moment,
                            "replica_signal": 0.0,
                            "residual_signal": sign * moment,
                            "source_model_class": "hand_fixture",
                        }
                    )
        evaluated = method_simulation.substitution.evaluate_cross_fitted_double_residuals(
            pd.DataFrame(signal_rows),
            pd.DataFrame(outcome_rows),
            hypothesis_id=hypothesis_id,
            horizon=horizon,
            min_cross_section=2,
        )
        expected = [first] * (24 // horizon_hours[horizon]) + [second] * (
            24 // horizon_hours[horizon]
        )
        assert evaluated.decision_moments["double_residual_moment"].tolist() == pytest.approx(
            expected
        )
        for method in method_simulation.RESIDUAL_METHODS:
            moments.append(
                evaluated.decision_moments.assign(
                    residual_method=method,
                    true_effect=truth,
                )
            )

    result = method_simulation.infer_layer_c_comparison(
        pd.concat(moments, ignore_index=True),
        expected_decisions_per_day={"4h": 6, "8h": 3, "12h": 2, "1d": 1},
        block_length=1,
        n_bootstrap=499,
        seed=7,
        alpha=0.0,
    )
    grid = result.comparison_grid
    assert len(grid) == 21
    assert grid["hypothesis_count"].eq(4).all()
    assert grid["true_positive_count"].eq(3).all()
    assert grid["rejection_count"].eq(0).all()
    assert grid["true_positive_rejection_count"].eq(0).all()
    assert grid["true_null_rejection_count"].eq(0).all()
    assert grid["mean_estimate"].tolist() == pytest.approx([1.5] * 21)
    assert grid["mean_true_effect"].tolist() == pytest.approx([1.5] * 21)
    assert grid["mean_bias"].tolist() == pytest.approx([0.0] * 21, abs=1e-15)
    assert grid["mean_squared_error"].tolist() == pytest.approx(
        [0.0] * 21, abs=1e-15
    )


def test_layer_b_c_aliases_are_not_added_to_the_inference_family(reduced_layer_b_c_runs):
    artifacts = reduced_layer_b_c_runs["B10"]
    assert artifacts.dataset.alias_mapping.to_dict("records") == [
        {"alias_id": "A01", "source_signal_id": "S01"},
        {"alias_id": "A02", "source_signal_id": "S02"},
        {"alias_id": "A03", "source_signal_id": "S03"},
    ]
    assert artifacts.layer_c.comparison_grid["hypothesis_count"].eq(4).all()
    assert all(
        len(json.loads(value)) == 4
        for value in artifacts.layer_c.comparison_grid["p_values_json"]
    )


def test_layer_b_c_missing_features_use_only_complete_common_support(reduced_layer_b_c_runs):
    artifacts = reduced_layer_b_c_runs["B09"]
    assert any(frame.filter(regex=r"^X").isna().any(axis=None) for frame in artifacts.dataset.frames.values())
    support = artifacts.layer_c.decision_moments["symbol_count"]
    assert support.min() >= 3
    assert support.min() < 6
    numeric = artifacts.layer_c.observations.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy(dtype=float)).all()


@pytest.fixture(scope="module")
def reduced_temporal_dataset():
    manifest_path = _blueprint_file("ksv4_增量信息方法模拟设计清单.json")
    return method_simulation.generate_temporal_falsification_dataset(
        manifest_path,
        replicate=0,
        seed=91,
        test_overrides={
            "day_count": 68,
            "object_count": 4,
            "train_days": 5,
            "embargo_days": 1,
            "test_days": 31,
            "step_days": 31,
        },
    )


def test_temporal_falsification_covers_lifetimes_alignments_and_t05(reduced_temporal_dataset):
    main = reduced_temporal_dataset.observations
    assert set(main["effect_lifetime"]) == {"0.5H", "H", "2H", "7d"}
    assert set(main["alignment_label"]) == {"correct", "-2H", "-H", "+H", "+2H"}
    assert main["included_in_primary_summary"].all()
    assert (main["signal_source_ts"] - main["decision_ts"]).dt.total_seconds().eq(
        main["alignment_offset_hours"] * 3600
    ).all()
    grouped = main.groupby(["horizon", "hypothesis_id"])
    assert grouped["decision_ts"].nunique().groupby(level="horizon").nunique().eq(1).all()
    assert grouped["residual_product"].count().groupby(level="horizon").nunique().eq(1).all()

    t05 = reduced_temporal_dataset.t05_observations
    assert set(t05["alignment_label"]) == {"correct"}
    assert (t05["decision_interval_hours"] * 2 == t05["holding_hours"]).all()
    assert not t05["included_in_primary_summary"].any()


def test_temporal_true_alignment_uses_fixed_holding_window(reduced_temporal_dataset):
    metadata = reduced_temporal_dataset.observations.drop_duplicates("hypothesis_id")
    half = metadata.loc[
        (metadata["effect_lifetime"] == "0.5H")
        & (metadata["holding_hours"] == 4)
    ].set_index("alignment_label")
    assert half.loc["correct", "true_effect"] > 0
    assert half.loc["correct", "true_effect"] > half.loc["-H", "true_effect"]
    assert half.loc["correct", "true_effect"] > half.loc["+H", "true_effect"]
    long = metadata.loc[
        (metadata["effect_lifetime"] == "7d")
        & (metadata["holding_hours"] == 4)
    ].set_index("alignment_label")
    assert long.loc["-H", "true_effect"] > long.loc["+H", "true_effect"] > 0


def test_temporal_runner_reuses_signal_side_entry_with_formal_2000(monkeypatch):
    manifest_path = _blueprint_file("ksv4_增量信息方法模拟设计清单.json")
    calls = []

    def cheap_randomization(observations, daily_effects, **kwargs):
        calls.append(kwargs)
        hypotheses = list(daily_effects.columns)
        summary = pd.DataFrame(
            {
                "hypothesis_id": np.tile(hypotheses, 3),
                "guard_days": np.repeat([3, 7, 14], len(hypotheses)),
                "observed_effect": np.tile(daily_effects.mean().to_numpy(), 3),
                "n_randomizations": 2_000,
            }
        )
        return method_simulation.substitution.DoubleResidualTimeRandomizationArtifacts(
            schedule=pd.DataFrame(), null_effects=pd.DataFrame(), summary=summary
        )

    monkeypatch.setattr(
        method_simulation.substitution,
        "simulation_signal_time_randomization",
        cheap_randomization,
    )
    artifacts = method_simulation.run_temporal_falsification(
        manifest_path,
        test_overrides={
            "day_count": 37,
            "object_count": 3,
            "train_days": 3,
            "embargo_days": 1,
            "test_days": 31,
            "step_days": 31,
        },
    )
    assert calls == [
        {
            "guard_days": (3, 7, 14),
            "n_randomizations": 2_000,
            "seed": 2026080731,
            "alternative": "greater",
        }
    ]
    assert set(artifacts.primary_summary["guard_days"]) == {3, 7, 14}
    assert not artifacts.t05_summary["included_in_primary_summary"].any()
