import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from qlab import latent_effects, method_simulation


def _bounds(family="truncated_normal"):
    if family == "truncated_normal":
        return {"pi0": (0.0, 1.0), "location": (-0.2, 0.4), "scale": (0.01, 0.4)}
    if family == "gamma":
        return {"pi0": (0.0, 1.0), "shape": (0.2, 8.0), "scale": (0.01, 0.4)}
    return {"pi0": (0.0, 1.0), "log_scale": (-5.0, 0.0), "shape": (0.05, 2.0)}


def test_all_spike_likelihood_is_hand_calculable():
    observed = np.array([0.10, -0.20, 0.05])
    errors = np.array([0.05, 0.10, 0.20])
    actual = latent_effects.spike_slab_log_likelihood(
        observed,
        errors,
        family="truncated_normal",
        pi0=1.0,
        parameter_1=0.1,
        parameter_2=0.2,
    )
    expected = norm.logpdf(observed, loc=0.0, scale=errors).sum()
    assert actual == pytest.approx(expected, abs=1e-12)


def test_standardized_effect_bootstrap_is_synchronized_and_reproducible():
    base = np.arange(1.0, 11.0)
    daily = pd.DataFrame({"h1": base, "h2": 2.0 * base})
    first = latent_effects.standardized_effect_bootstrap(
        daily, block_length=2, n_draws=40, seed=123, batch_size=7
    )
    second = latent_effects.standardized_effect_bootstrap(
        daily, block_length=2, n_draws=40, seed=123, batch_size=7
    )
    pd.testing.assert_frame_equal(first.bootstrap_effects, second.bootstrap_effects)
    assert first.bootstrap_effects.corr().loc["h1", "h2"] == pytest.approx(1.0)
    assert tuple(first.hypothesis_summary["hypothesis_id"]) == ("h1", "h2")
    assert (first.hypothesis_summary["bootstrap_standard_error"] > 0.0).all()


@pytest.mark.parametrize("family", latent_effects.SUPPORTED_SLAB_FAMILIES)
def test_spike_slab_fit_is_finite_for_every_registered_family(family):
    observed = np.array([-0.01, 0.0, 0.02, 0.10, 0.14, 0.18])
    errors = np.full(len(observed), 0.03)
    fit = latent_effects.fit_spike_slab_measurement_model(
        observed,
        errors,
        family=family,
        parameter_bounds=_bounds(family),
        quadrature_nodes=48,
    )
    assert np.isfinite(fit.log_likelihood)
    assert 0.0 <= fit.pi0 <= 1.0
    assert fit.family == family
    assert len(fit.start_results) == 9


def test_profile_and_acceptance_grid_have_explicit_likelihood_ratios():
    observed = np.array([0.0, 0.03, 0.08, 0.12])
    errors = np.full(4, 0.04)
    profile = latent_effects.profile_spike_slab_pi0(
        observed,
        errors,
        family="truncated_normal",
        parameter_bounds=_bounds(),
        pi0_grid=[0.0, 0.5, 1.0],
        quadrature_nodes=48,
    )
    assert tuple(profile["pi0"]) == (0.0, 0.5, 1.0)
    assert profile["likelihood_ratio_from_profile_max"].min() == pytest.approx(0.0)
    grid = profile[["pi0", "location", "scale"]]
    accepted = latent_effects.evaluate_parameter_acceptance_grid(
        observed,
        errors,
        grid,
        family="truncated_normal",
        maximum_log_likelihood=float(profile["log_likelihood"].max()),
        likelihood_ratio_critical_value=2.0,
        quadrature_nodes=48,
    )
    assert accepted["accepted"].dtype == bool
    assert accepted["accepted"].any()


def test_acceptance_grid_exactly_applies_two_log_likelihood_difference(monkeypatch):
    def fixed_likelihood(
        observed_effects, standard_errors, *, family, pi0,
        parameter_1, parameter_2, quadrature_nodes=96,
    ):
        return 10.0 - float(pi0)

    monkeypatch.setattr(latent_effects, "spike_slab_log_likelihood", fixed_likelihood)
    grid = pd.DataFrame(
        {
            "pi0": [0.0, 0.5, 1.0],
            "location": [0.1, 0.1, 0.1],
            "scale": [0.1, 0.1, 0.1],
        }
    )
    result = latent_effects.evaluate_parameter_acceptance_grid(
        [0.0, 0.1],
        [0.1, 0.1],
        grid,
        family="truncated_normal",
        maximum_log_likelihood=10.0,
        likelihood_ratio_critical_value=1.0,
    )
    np.testing.assert_allclose(result["likelihood_ratio"], [0.0, 1.0, 2.0])
    assert tuple(result["accepted"]) == (True, True, False)


def test_identification_switches_when_bootstrap_interval_hits_domain_boundary():
    fits = pd.DataFrame(
        {
            "pi0": [0.0, 0.0, 0.2, 0.3],
            "location": [0.1, 0.1, 0.12, 0.13],
            "scale": [0.1, 0.1, 0.12, 0.13],
            "converged": [True] * 4,
            "log_likelihood": [-1.0] * 4,
        }
    )
    profile = pd.DataFrame(
        {
            "profile_parameter": ["pi0"] * 3,
            "profile_value": [0.0, 0.5, 1.0],
            "likelihood_ratio_from_profile_max": [0.0, 2.0, 8.0],
        }
    )
    decision = latent_effects.assess_identification(
        fits,
        profile,
        parameter_columns=["pi0", "location", "scale"],
        parameter_bounds=_bounds(),
        equality_tolerance=1e-8,
        profile_likelihood_ratio_critical_value=3.0,
    )
    assert decision.switch_to_partial_identification
    assert "pi0_bootstrap_interval_hits_boundary" in decision.reasons
    assert "pi0_profile_interval_open_at_lower_domain" in decision.reasons


def test_identification_uses_supplied_calibrated_profile_cutoff():
    fits = pd.DataFrame(
        {
            "pi0": [0.2, 0.3, 0.4, 0.5],
            "location": [0.1, 0.11, 0.12, 0.13],
            "scale": [0.1, 0.11, 0.12, 0.13],
            "converged": [True] * 4,
            "log_likelihood": [-1.0] * 4,
        }
    )
    profile = pd.DataFrame(
        {
            "profile_parameter": ["pi0"] * 3,
            "profile_value": [0.0, 0.5, 1.0],
            "likelihood_ratio_from_profile_max": [2.0, 0.0, 2.0],
        }
    )
    strict = latent_effects.assess_identification(
        fits,
        profile,
        parameter_columns=["pi0", "location", "scale"],
        parameter_bounds=_bounds(),
        equality_tolerance=1e-8,
        profile_likelihood_ratio_critical_value=1.0,
    )
    permissive = latent_effects.assess_identification(
        fits,
        profile,
        parameter_columns=["pi0", "location", "scale"],
        parameter_bounds=_bounds(),
        equality_tolerance=1e-8,
        profile_likelihood_ratio_critical_value=2.0,
    )
    assert not strict.switch_to_partial_identification
    assert permissive.switch_to_partial_identification
    assert "pi0_profile_interval_open_at_lower_domain" in permissive.reasons
    assert "pi0_profile_interval_open_at_upper_domain" in permissive.reasons


def test_multistart_varies_every_parameter_and_profiles_are_named():
    observed = np.array([0.0, 0.03, 0.08, 0.12])
    errors = np.full(4, 0.04)
    fit = latent_effects.fit_spike_slab_measurement_model(
        observed,
        errors,
        family="truncated_normal",
        parameter_bounds=_bounds(),
        quadrature_nodes=32,
    )
    assert fit.start_results["start_pi0"].nunique() > 1
    assert fit.start_results["start_location"].nunique() > 1
    assert fit.start_results["start_scale"].nunique() > 1
    profile = latent_effects.profile_spike_slab_parameter(
        observed,
        errors,
        family="truncated_normal",
        parameter_bounds=_bounds(),
        parameter_name="location",
        parameter_grid=[-0.2, 0.1, 0.4],
        quadrature_nodes=32,
    )
    assert profile["profile_parameter"].eq("location").all()
    assert tuple(profile["profile_value"]) == (-0.2, 0.1, 0.4)


def test_profile_chunk_combination_restores_global_likelihood_distance():
    left = pd.DataFrame(
        {
            "profile_parameter": ["pi0", "pi0"],
            "profile_value": [0.0, 0.5],
            "log_likelihood": [8.0, 10.0],
            "converged": [True, True],
        }
    )
    right = pd.DataFrame(
        {
            "profile_parameter": ["pi0"],
            "profile_value": [1.0],
            "log_likelihood": [9.0],
            "converged": [True],
        }
    )
    combined = latent_effects.combine_spike_slab_profile_chunks([left, right])
    assert tuple(combined["likelihood_ratio_from_profile_max"]) == (4.0, 0.0, 2.0)


def test_identification_detects_disconnected_profile_and_equivalent_solutions():
    fits = pd.DataFrame(
        {
            "pi0": [0.2, 0.3, 0.4, 0.5],
            "location": [0.1, 0.11, 0.12, 0.13],
            "scale": [0.1, 0.11, 0.12, 0.13],
            "converged": [True] * 4,
            "log_likelihood": [-1.0] * 4,
        }
    )
    profile = pd.DataFrame(
        {
            "profile_parameter": ["pi0"] * 3,
            "profile_value": [0.0, 0.5, 1.0],
            "likelihood_ratio_from_profile_max": [0.0, 5.0, 0.0],
        }
    )
    starts = pd.DataFrame(
        {
            "pi0": [0.2, 0.7],
            "location": [0.1, 0.3],
            "scale": [0.1, 0.2],
            "log_likelihood": [10.0, 10.0 - 1e-8],
            "optimizer_success": [True, True],
        }
    )
    decision = latent_effects.assess_identification(
        fits,
        profile,
        parameter_columns=["pi0", "location", "scale"],
        parameter_bounds=_bounds(),
        equality_tolerance=1e-7,
        profile_likelihood_ratio_critical_value=1.0,
        point_start_results=starts,
        equivalent_likelihood_tolerance=1e-6,
    )
    assert "pi0_profile_has_disconnected_regions" in decision.reasons
    assert "equivalent_likelihood_distinct_parameter_solutions" in decision.reasons


def test_exact_spike_draws_and_truth_ledger_are_reproducible():
    spike = latent_effects.SpikeSlabFit(
        family="truncated_normal",
        pi0=1.0,
        parameter_1=0.1,
        parameter_2=0.1,
        log_likelihood=0.0,
        converged=True,
        boundary_hit=True,
        start_results=pd.DataFrame(),
    )
    np.testing.assert_array_equal(
        latent_effects.sample_spike_slab_effects(spike, size=5, seed=1),
        np.zeros(5),
    )
    days = 40
    values = np.column_stack(
        [np.sin(np.arange(days) / 3.0), np.cos(np.arange(days) / 4.0)]
    )
    values -= values.mean(axis=0)
    null = pd.DataFrame(values, columns=["H01", "H02"])
    sd = null.std(axis=0, ddof=1)
    reference = pd.Series([0.1, 0.1], index=null.columns)
    first = latent_effects.simulate_latent_effect_block_task(
        null,
        sd,
        reference,
        spike,
        replicate=0,
        block_length=4,
        noise_seed=10,
        truth_seed=20,
        scenario_id="zero",
        all_null=True,
    )
    second = latent_effects.simulate_latent_effect_block_task(
        null,
        sd,
        reference,
        spike,
        replicate=0,
        block_length=4,
        noise_seed=10,
        truth_seed=20,
        scenario_id="zero",
        all_null=True,
    )
    pd.testing.assert_frame_equal(first, second)
    assert first["true_effect"].eq(0.0).all()
    evaluated = method_simulation.evaluate_bh_fdr_variants(
        first,
        dataset_id="hand_calculable_zero",
        scenario_temporal_dependence=None,
        day_count=days,
        family_size=2,
        alpha=0.05,
        include_cross_scenario_summary=False,
    )
    assert set(evaluated.hypothesis_results["method_variant"]) == {
        "AR_BIC_1000_BH", "AR_BIC_1125_BH", "MC_REFERENCE_BH"
    }
    assert evaluated.task_summary["false_discovery_proportion"].notna().all()


def test_likelihood_ratio_calibration_is_aligned_and_hand_calculable_at_reference():
    effects = pd.DataFrame(
        [[0.0, 0.1], [0.02, 0.08]], columns=["H01", "H02"]
    )
    errors = pd.Series([0.05, 0.05], index=effects.columns)
    reference = latent_effects.SpikeSlabFit(
        family="truncated_normal", pi0=0.4, parameter_1=0.1,
        parameter_2=0.05, log_likelihood=0.0, converged=True,
        boundary_hit=False, start_results=pd.DataFrame(),
    )
    fitted = []
    for idx, row in effects.iterrows():
        likelihood = latent_effects.spike_slab_log_likelihood(
            row, errors, family=reference.family, pi0=reference.pi0,
            parameter_1=reference.parameter_1,
            parameter_2=reference.parameter_2,
        )
        fitted.append(
            {"bootstrap_idx": idx, "log_likelihood": likelihood, "converged": True}
        )
    artifacts = latent_effects.calibrate_composite_likelihood_ratio(
        effects, errors, pd.DataFrame(fitted), reference
    )
    assert artifacts.replicate_statistics["likelihood_ratio_statistic"].eq(0.0).all()
    assert artifacts.summary.loc[0, "likelihood_ratio_critical_value"] == 0.0


def test_likelihood_ratio_calibration_rejects_incomplete_fit_identity():
    effects = pd.DataFrame([[0.0, 0.1], [0.02, 0.08]], columns=["H01", "H02"])
    errors = pd.Series([0.05, 0.05], index=effects.columns)
    reference = latent_effects.SpikeSlabFit(
        family="gamma", pi0=0.5, parameter_1=2.0, parameter_2=0.05,
        log_likelihood=0.0, converged=True, boundary_hit=False,
        start_results=pd.DataFrame(),
    )
    with pytest.raises(ValueError, match="cover every effect draw"):
        latent_effects.calibrate_composite_likelihood_ratio(
            effects,
            errors,
            pd.DataFrame(
                [{"bootstrap_idx": 0, "log_likelihood": -1.0, "converged": True}]
            ),
            reference,
        )


def test_bootstrap_refit_covers_every_draw_and_is_deterministic():
    effects = pd.DataFrame(
        [[0.0, 0.05, 0.10], [0.01, 0.06, 0.11]],
        columns=["H01", "H02", "H03"],
    )
    errors = pd.Series([0.03, 0.03, 0.03], index=effects.columns)
    first = latent_effects.refit_spike_slab_bootstrap(
        effects,
        errors,
        family="truncated_normal",
        parameter_bounds=_bounds(),
        quadrature_nodes=32,
    )
    second = latent_effects.refit_spike_slab_bootstrap(
        effects,
        errors,
        family="truncated_normal",
        parameter_bounds=_bounds(),
        quadrature_nodes=32,
    )
    pd.testing.assert_frame_equal(first, second)
    assert tuple(first["bootstrap_idx"]) == (0, 1)


def test_minimal_latent_distribution_end_to_end_is_reproducible():
    days = 32
    values = np.column_stack(
        [np.sin(np.arange(days) / 2.0), np.cos(np.arange(days) / 3.0)]
    )
    values -= values.mean(axis=0)
    null = pd.DataFrame(values, columns=["H01", "H02"])
    sd = null.std(axis=0, ddof=1)
    reference = pd.Series([0.08, 0.09], index=null.columns)
    fit = latent_effects.fit_spike_slab_measurement_model(
        [0.0, 0.0],
        [0.1, 0.1],
        family="truncated_normal",
        parameter_bounds={
            "pi0": (1.0, 1.0), "location": (0.1, 0.1), "scale": (0.1, 0.1),
        },
        quadrature_nodes=32,
    )
    first = latent_effects.evaluate_latent_effect_distribution(
        null,
        sd,
        reference,
        fit,
        dataset_id="minimal",
        scenario_id="main",
        replicate_count=2,
        block_length=days,
        noise_seed_base=100,
        truth_seed_base=200,
    )
    second = latent_effects.evaluate_latent_effect_distribution(
        null,
        sd,
        reference,
        fit,
        dataset_id="minimal",
        scenario_id="main",
        replicate_count=2,
        block_length=days,
        noise_seed_base=100,
        truth_seed_base=200,
    )
    pd.testing.assert_frame_equal(first.generated_tasks, second.generated_tasks)
    pd.testing.assert_frame_equal(first.task_summary, second.task_summary)
    assert len(first.generated_tasks) == 4
    assert first.generated_tasks["true_effect"].eq(0.0).all()
    assert first.generated_tasks["observed_effect"].abs().max() < 1e-12
    assert first.task_summary["discovery_count"].eq(0).all()
    assert first.task_summary["false_discovery_proportion"].eq(0.0).all()
    assert set(first.scenario_summary["method_variant"]) == {
        "AR_BIC_1000_BH", "AR_BIC_1125_BH", "MC_REFERENCE_BH"
    }


def test_reusable_null_base_matches_one_shot_simulation_exactly():
    days = 24
    values = np.column_stack(
        [np.sin(np.arange(days)), np.cos(np.arange(days))]
    )
    values -= values.mean(axis=0)
    null = pd.DataFrame(values, columns=["H01", "H02"])
    sd = null.std(axis=0, ddof=1)
    reference = pd.Series([0.1, 0.1], index=null.columns)
    fit = latent_effects.SpikeSlabFit(
        family="gamma", pi0=0.5, parameter_1=2.0, parameter_2=0.05,
        log_likelihood=0.0, converged=True, boundary_hit=False,
        start_results=pd.DataFrame(),
    )
    one_shot = latent_effects.simulate_latent_effect_block_task(
        null, sd, reference, fit, replicate=3, block_length=4,
        noise_seed=103, truth_seed=203, scenario_id="same",
    )
    base = latent_effects.prepare_latent_null_task_base(
        null, reference, replicate=3, block_length=4, noise_seed=103
    )
    reused = latent_effects.apply_latent_fit_to_null_task_bases(
        base, sd, fit, scenario_id="same", truth_seed_base=200
    )
    pd.testing.assert_frame_equal(one_shot, reused)


def test_envelope_selection_keeps_each_method_metric_extreme():
    frame = pd.DataFrame(
        {
            "parameter_id": ["a", "b", "a", "b"],
            "family": ["gamma"] * 4,
            "method_variant": ["m1", "m1", "m2", "m2"],
            "fdr": [0.01, 0.04, 0.02, 0.03],
            "task_mean_tpr": [0.8, 0.2, 0.4, 0.7],
            "fdr_monte_carlo_standard_error": [0.0] * 4,
            "task_mean_tpr_monte_carlo_standard_error": [0.0] * 4,
            "task_count": [1_000_000_000] * 4,
            "tpr_defined_task_count": [1_000_000_000] * 4,
        }
    )
    selected = latent_effects.select_performance_envelope_candidates(frame)
    assert len(selected) == 8
    row = selected.set_index(["method_variant", "metric", "bound"])
    assert row.loc[("m1", "fdr", "minimum"), "parameter_id"] == "a"
    assert row.loc[("m1", "task_mean_tpr", "minimum"), "parameter_id"] == "b"


def test_envelope_selection_retains_uncertain_challenger_and_excludes_separated_point():
    frame = pd.DataFrame(
        {
            "parameter_id": ["a", "b", "c"],
            "family": ["gamma"] * 3,
            "method_variant": ["m1"] * 3,
            "fdr": [0.01, 0.02, 0.80],
            "task_mean_tpr": [0.4, 0.5, 0.6],
            "fdr_monte_carlo_standard_error": [0.0, 0.01, 0.0],
            "task_mean_tpr_monte_carlo_standard_error": [0.0, 0.01, 0.0],
            "task_count": [500] * 3,
            "tpr_defined_task_count": [500] * 3,
        }
    )
    selected = latent_effects.select_performance_envelope_candidates(frame)
    fdr_min = selected.loc[
        selected["metric"].eq("fdr") & selected["bound"].eq("minimum")
    ].sort_values("boundary_rank")
    assert tuple(fdr_min["parameter_id"]) == ("a", "b")
    assert tuple(fdr_min["boundary_rank"]) == (1, 2)
    assert "c" not in set(fdr_min["parameter_id"])


def test_envelope_interval_stays_nonzero_when_sample_standard_error_is_zero():
    frame = pd.DataFrame(
        {
            "parameter_id": ["a", "b"],
            "family": ["gamma", "gamma"],
            "method_variant": ["m", "m"],
            "fdr": [0.0, 0.5],
            "task_mean_tpr": [0.0, 0.5],
            "fdr_monte_carlo_standard_error": [0.0, 0.0],
            "task_mean_tpr_monte_carlo_standard_error": [0.0, 0.0],
            "task_count": [500, 500],
            "tpr_defined_task_count": [500, 500],
        }
    )
    selected = latent_effects.select_performance_envelope_candidates(frame)
    zero = selected.loc[
        selected["parameter_id"].eq("a") & selected["metric"].eq("fdr")
        & selected["bound"].eq("minimum")
    ].iloc[0]
    assert zero["finite_sample_radius"] > 0.0
    assert zero["screening_ci_upper"] > 0.0


def test_performance_envelope_is_hand_calculable():
    frame = pd.DataFrame(
        {
            "parameter_id": ["a", "b"],
            "family": ["gamma", "gamma"],
            "method_variant": ["m", "m"],
            "fdr": [0.01, 0.03],
            "task_mean_tpr": [0.7, 0.4],
            "fdr_ci95_lower": [0.0, 0.02],
            "fdr_ci95_upper": [0.02, 0.04],
            "task_mean_tpr_ci95_lower": [0.6, 0.3],
            "task_mean_tpr_ci95_upper": [0.8, 0.5],
        }
    )
    envelope = latent_effects.summarize_performance_envelope(frame).set_index(
        ["metric", "bound"]
    )
    assert envelope.loc[("fdr", "minimum"), "parameter_id"] == "a"
    assert envelope.loc[("fdr", "maximum"), "value"] == 0.03
    assert envelope.loc[("task_mean_tpr", "minimum"), "parameter_id"] == "b"


def test_extended_task_metrics_and_envelope_are_hand_calculable():
    tasks = pd.DataFrame(
        {
            "dataset_id": ["d"] * 3,
            "method_variant": ["m"] * 3,
            "scenario_id": ["s"] * 3,
            "analysis_family": ["a"] * 3,
            "registered_task_idx": [0, 1, 2],
            "discovery_count": [2, 1, 0],
            "true_discovery_count": [1, 1, 0],
            "true_alternative_count": [2, 1, 0],
        }
    )
    extended = latent_effects.summarize_latent_task_level_metrics(
        tasks, family_size=3
    ).iloc[0]
    assert extended["mean_discovery_count"] == 1.0
    assert extended["no_discovery_rate"] == pytest.approx(1.0 / 3.0)
    assert extended["all_null_task_rate"] == pytest.approx(1.0 / 3.0)
    assert extended["pooled_true_effect_discovery_rate"] == pytest.approx(2.0 / 3.0)
    assert extended["mean_discovery_count_monte_carlo_standard_error"] > 0.0
    assert extended["no_discovery_rate_ci95_lower"] < 1.0 / 3.0
    assert extended["no_discovery_rate_ci95_upper"] > 1.0 / 3.0

    base = pd.DataFrame(
        {
            "parameter_id": ["p0", "p1"],
            "family": ["gamma", "gamma"],
            "method_variant": ["m", "m"],
            "fdr": [0.01, 0.02],
            "fdr_ci95_lower": [0.0, 0.01],
            "fdr_ci95_upper": [0.02, 0.03],
            "task_mean_tpr": [0.4, 0.8],
            "task_mean_tpr_ci95_lower": [0.3, 0.7],
            "task_mean_tpr_ci95_upper": [0.5, 0.9],
            "pooled_true_effect_discovery_rate": [0.5, 0.7],
            "pooled_true_effect_discovery_rate_ci95_lower": [0.4, 0.6],
            "pooled_true_effect_discovery_rate_ci95_upper": [0.6, 0.8],
            "mean_discovery_count": [1.0, 2.0],
            "mean_discovery_count_ci95_lower": [0.8, 1.8],
            "mean_discovery_count_ci95_upper": [1.2, 2.2],
            "no_discovery_rate": [0.6, 0.2],
            "no_discovery_rate_ci95_lower": [0.5, 0.1],
            "no_discovery_rate_ci95_upper": [0.7, 0.3],
            "all_null_task_rate": [0.1, 0.3],
            "all_null_task_rate_ci95_lower": [0.05, 0.2],
            "all_null_task_rate_ci95_upper": [0.15, 0.4],
        }
    )
    envelope = latent_effects.summarize_extended_performance_envelope(base).set_index(
        ["metric", "bound"]
    )
    assert envelope.loc[("mean_discovery_count", "minimum"), "parameter_id"] == "p0"
    assert envelope.loc[("no_discovery_rate", "minimum"), "parameter_id"] == "p1"
    assert envelope.loc[("all_null_task_rate", "maximum"), "value"] == 0.3


def test_true_effect_quantile_summary_is_hand_calculable():
    tasks = pd.DataFrame(
        {
            "registered_task_idx": [0, 0, 1, 1, 2],
            "hypothesis_id": ["a", "b", "a", "b", "a"],
            "standardized_true_effect": [0.1, 0.2, 0.3, 0.4, 0.0],
            "is_true_alternative": [True, True, True, True, False],
        }
    )
    results = pd.DataFrame(
        {
            "dataset_id": ["d"] * 5,
            "method_variant": ["m"] * 5,
            "scenario_id": ["s"] * 5,
            "analysis_family": ["a"] * 5,
            "registered_task_idx": [0, 0, 1, 1, 2],
            "hypothesis_id": ["a", "b", "a", "b", "a"],
            "discovered": [True, False, True, True, True],
        }
    )
    summary = latent_effects.summarize_discovery_by_true_effect_quantile(
        tasks, results, quantile_count=2
    ).set_index("effect_quantile_group")
    assert summary.loc[1, "conditional_discovery_rate"] == 0.5
    assert summary.loc[2, "conditional_discovery_rate"] == 1.0
    assert summary.loc[1, "effect_lower"] == 0.1
    assert summary.loc[2, "effect_upper"] == 0.4


def test_discovery_count_quantiles_are_hand_calculable():
    task = pd.DataFrame(
        {
            "dataset_id": ["d"] * 4,
            "method_variant": ["m"] * 4,
            "scenario_id": ["s"] * 4,
            "analysis_family": ["a"] * 4,
            "registered_task_idx": range(4),
            "false_discovery_proportion": [0.0] * 4,
            "true_positive_rate": [np.nan] * 4,
            "discovery_count": [0, 1, 2, 3],
        }
    )
    hypothesis = pd.DataFrame(
        {
            "dataset_id": ["d"] * 4,
            "method_variant": ["m"] * 4,
            "scenario_id": ["s"] * 4,
            "analysis_family": ["a"] * 4,
            "registered_task_idx": range(4),
            "discovered": [False] * 4,
            "is_true_alternative": [False] * 4,
        }
    )
    result = latent_effects.summarize_latent_truth_performance(task, hypothesis).iloc[0]
    assert result["discovery_count_q50"] == 1.5
    assert result["discovery_count_q90"] == pytest.approx(2.7)


def test_invalid_contracts_fail_closed():
    with pytest.raises(ValueError, match="parameter bounds"):
        latent_effects.fit_spike_slab_measurement_model(
            [0.0, 0.1], [0.1, 0.1], family="truncated_normal",
            parameter_bounds={"pi0": (0.0, 1.0), "location": (0.0, 1.0)},
        )
    with pytest.raises(ValueError, match="positive"):
        latent_effects.spike_slab_log_likelihood(
            [0.0, 0.1], [0.1, 0.0], family="gamma",
            pi0=0.5, parameter_1=1.0, parameter_2=0.1,
        )
