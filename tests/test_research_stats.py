from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from qlab import research_stats
from qlab.research_stats import (
    adaptive_flat_top_ratio_standard_error,
    adaptive_flat_top_restudentized_circular_block_bootstrap_stepdown_max_t,
    annualized_sharpe_from_periods,
    benjamini_hochberg_q_values,
    circular_block_bootstrap_stepdown_max_t,
    dependent_multiplier_bootstrap_stepdown_max_t,
    hac_t_stat,
    holm_adjusted_p_values,
    newey_west_max_lags,
    normal_one_sided_p_value,
    normal_two_sided_p_value,
    randomization_stepdown_max_t,
    restudentized_circular_block_bootstrap_stepdown_max_t,
    simulation_calibration_dependent_multiplier_bootstrap_stepdown_max_t,
    simulation_calibration_circular_block_stepdown_max_t,
    simulation_calibration_adaptive_flat_top_restudentized_stepdown_max_t,
    simulation_calibration_restudentized_circular_block_stepdown_max_t,
    simulation_calibration_self_normalized_stepdown_max_t,
    self_normalized_circular_block_bootstrap_stepdown_max_t,
    self_normalized_ratio_standard_error,
    simple_t_stat,
)


def test_simulation_calibration_entry_is_isolated_from_production_minimum():
    sums, counts, effects = _centered_daily_fixture()
    with pytest.raises(ValueError, match="at least 10000"):
        circular_block_bootstrap_stepdown_max_t(
            sums, counts, effects, block_length=2, n_bootstrap=499, seed=7
        )

    result = simulation_calibration_circular_block_stepdown_max_t(
        sums, counts, effects, block_length=2, n_bootstrap=499, seed=7
    )
    assert set(result.summary["n_bootstrap"]) == {499}
    assert result.summary["stepdown_adjusted_p_batch_mcse"].isna().all()

    with pytest.raises(ValueError, match="at least 499"):
        simulation_calibration_circular_block_stepdown_max_t(
            sums, counts, effects, block_length=2, n_bootstrap=498, seed=7
        )


def test_revised_simulation_entries_are_isolated_from_production_minimum():
    sums, counts, effects = _centered_daily_fixture()
    with pytest.raises(ValueError, match="at least 10000"):
        restudentized_circular_block_bootstrap_stepdown_max_t(
            sums, counts, effects, block_length=2, n_bootstrap=999, seed=7
        )
    with pytest.raises(ValueError, match="at least 10000"):
        dependent_multiplier_bootstrap_stepdown_max_t(
            sums, counts, effects, bandwidth=2, n_bootstrap=999, seed=7
        )
    long_sums, long_counts, long_effects = _long_centered_daily_fixture()
    with pytest.raises(ValueError, match="at least 10000"):
        adaptive_flat_top_restudentized_circular_block_bootstrap_stepdown_max_t(
            long_sums, long_counts, long_effects,
            block_length=14, n_bootstrap=999, seed=7,
        )
    with pytest.raises(ValueError, match="at least 10000"):
        self_normalized_circular_block_bootstrap_stepdown_max_t(
            sums, counts, effects, block_length=2, n_bootstrap=999, seed=7
        )
    assert set(
        simulation_calibration_restudentized_circular_block_stepdown_max_t(
            sums, counts, effects, block_length=2, n_bootstrap=999, seed=7
        ).summary["resampling_engine"]
    ) == {"restudentized_circular_block_bootstrap_t"}
    assert set(
        simulation_calibration_dependent_multiplier_bootstrap_stepdown_max_t(
            sums, counts, effects, bandwidth=2, n_bootstrap=999, seed=7
        ).summary["resampling_engine"]
    ) == {"dependent_gaussian_multiplier_bootstrap_t"}
    assert set(
        simulation_calibration_adaptive_flat_top_restudentized_stepdown_max_t(
            long_sums, long_counts, long_effects,
            block_length=14, n_bootstrap=999, seed=7,
        ).summary["resampling_engine"]
    ) == {"adaptive_flat_top_restudentized_circular_block_bootstrap_t"}
    assert set(
        simulation_calibration_self_normalized_stepdown_max_t(
            sums, counts, effects, block_length=2, n_bootstrap=999, seed=7
        ).summary["resampling_engine"]
    ) == {"self_normalized_circular_block_bootstrap_t"}


@pytest.mark.parametrize(
    "bad_index,match",
    [
        (pd.RangeIndex(4), "DatetimeIndex"),
        (pd.date_range("2025-01-01", periods=4, freq="D"), "UTC"),
        (
            pd.DatetimeIndex(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-04T00:00:00Z",
                    "2025-01-05T00:00:00Z",
                ]
            ),
            "consecutive",
        ),
        (pd.date_range("2025-01-01T01:00:00Z", periods=4, freq="D"), "midnight"),
    ],
)
def test_revised_inference_rejects_noncanonical_daily_index(bad_index, match):
    sums, counts, effects = _centered_daily_fixture()
    sums.index = bad_index
    counts.index = bad_index
    with pytest.raises((TypeError, ValueError), match=match):
        simulation_calibration_restudentized_circular_block_stepdown_max_t(
            sums, counts, effects, block_length=2, n_bootstrap=999, seed=7
        )


def test_newey_west_lag_count_is_bounded():
    assert newey_west_max_lags(1) == 0
    assert newey_west_max_lags(3) <= 2
    assert newey_west_max_lags(200) > 0


def test_newey_west_lag_count_respects_overlap_minimum():
    assert newey_west_max_lags(50, overlap_lags=7) >= 7
    assert newey_west_max_lags(5, overlap_lags=10) == 4
    with pytest.raises(ValueError, match="overlap_lags"):
        newey_west_max_lags(50, overlap_lags=-1)


def test_hac_t_stat_returns_finite_value_for_variable_series():
    values = pd.Series([0.01, 0.02, -0.01, 0.03, 0.00, 0.02])

    result = hac_t_stat(values)

    assert math.isfinite(result)


def test_hac_t_stat_accepts_overlap_lag_policy():
    values = pd.Series([0.01, 0.02, -0.01, 0.03, 0.00, 0.02])

    result = hac_t_stat(values, overlap_lags=2)

    assert math.isfinite(result)


def test_simple_t_stat_matches_manual_formula():
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    expected = values.mean() / values.std(ddof=1) * math.sqrt(len(values))

    assert simple_t_stat(values) == pytest.approx(expected)


def test_normal_one_sided_p_value_and_bh_q_values():
    assert normal_one_sided_p_value(0.0) == pytest.approx(0.5)
    assert normal_one_sided_p_value(2.0) < 0.05
    q_values = benjamini_hochberg_q_values([0.01, 0.04, float("nan"), 0.03])

    assert q_values[0] == pytest.approx(0.03)
    assert q_values[1] == pytest.approx(0.04)
    assert math.isnan(q_values[2])
    assert q_values[3] == pytest.approx(0.04)

    with pytest.raises(ValueError, match="between 0 and 1"):
        benjamini_hochberg_q_values([0.1, 1.2])


def test_normal_two_sided_p_value_is_symmetric():
    assert normal_two_sided_p_value(0.0) == pytest.approx(1.0)
    assert normal_two_sided_p_value(2.0) == pytest.approx(
        normal_two_sided_p_value(-2.0)
    )
    assert normal_two_sided_p_value(2.0) < 0.05
    assert math.isnan(normal_two_sided_p_value(float("nan")))
    assert math.isnan(normal_two_sided_p_value(float("inf")))


def test_holm_adjusted_p_values_preserve_order_and_nan():
    adjusted = holm_adjusted_p_values([0.01, 0.04, float("nan"), 0.03, 0.03])

    assert adjusted[0] == pytest.approx(0.04)
    assert adjusted[1] == pytest.approx(0.09)
    assert math.isnan(adjusted[2])
    assert adjusted[3] == pytest.approx(0.09)
    assert adjusted[4] == pytest.approx(0.09)
    with pytest.raises(ValueError, match="between 0 and 1"):
        holm_adjusted_p_values([-0.1, 0.2])


def test_ar_finite_sample_variance_matches_white_noise_and_ar1():
    white_mean_variance, white_lrv = (
        research_stats._stable_ar_finite_sample_variance(
            np.array([]), 2.0, observation_count=5
        )
    )
    assert white_mean_variance == pytest.approx(2.0 / 5.0)
    assert white_lrv == pytest.approx(2.0)

    mean_variance, lrv = research_stats._stable_ar_finite_sample_variance(
        np.array([0.5]), 0.75, observation_count=5
    )
    expected = (
        5.0 + 2.0 * sum((5 - lag) * 0.5**lag for lag in range(1, 5))
    ) / 25.0
    assert mean_variance == pytest.approx(expected)
    assert lrv == pytest.approx(3.0)


def test_ar_spectral_holm_end_to_end_uses_complete_family_and_common_tail():
    rng = np.random.default_rng(901)
    index = pd.date_range("2025-01-01", periods=100, freq="D", tz="UTC")
    raw = pd.DataFrame(
        rng.normal(size=(100, 2)), index=index, columns=["h1", "h2"]
    )
    centered = raw - raw.mean(axis=0)
    counts = pd.DataFrame(1, index=index, columns=raw.columns)
    effects = pd.Series({"h1": 0.30, "h2": 0.00})

    artifacts = research_stats.autoregressive_spectral_holm_test(
        centered,
        counts,
        effects,
        order_criterion="BIC",
        expected_hypothesis_count=2,
    )
    result = artifacts.summary.set_index("hypothesis_id")
    assert set(result["maximum_ar_order"]) == {10}
    assert set(result["common_fit_observation_count"]) == {90}
    assert (result["family_adjusted_p_value"] >= result["raw_one_sided_p_value"]).all()
    assert result.loc["h1", "family_adjusted_p_value"] < 0.05
    assert result.loc["h2", "family_adjusted_p_value"] >= 0.05
    assert artifacts.selected_coefficients["lag"].ge(1).all()


def test_ar_spectral_holm_fails_closed_on_incomplete_or_constant_family():
    index = pd.date_range("2025-01-01", periods=100, freq="D", tz="UTC")
    values = pd.DataFrame(
        {"h1": np.tile([-1.0, 1.0], 50), "h2": np.zeros(100)}, index=index
    )
    counts = pd.DataFrame(1, index=index, columns=values.columns)
    effects = pd.Series({"h1": 0.0, "h2": 0.0})
    with pytest.raises(ValueError, match="wrong hypothesis count"):
        research_stats.autoregressive_spectral_holm_test(
            values[["h1"]], counts[["h1"]], effects[["h1"]],
            order_criterion="AIC", expected_hypothesis_count=2,
        )
    with pytest.raises(ValueError, match="constant hypothesis"):
        research_stats.autoregressive_spectral_holm_test(
            values, counts, effects, order_criterion="AIC",
            expected_hypothesis_count=2,
        )


def test_ar_spectral_holm_is_deterministic_with_unequal_counts():
    rng = np.random.default_rng(902)
    index = pd.date_range("2025-01-01", periods=100, freq="D", tz="UTC")
    counts = pd.DataFrame(
        {"h1": np.tile([1, 2], 50), "h2": np.tile([3, 1], 50)}, index=index
    )
    influence = pd.DataFrame(
        rng.normal(size=(100, 2)), index=index, columns=counts.columns
    )
    influence -= influence.mean(axis=0)
    centered_sums = influence * counts.mean(axis=0)
    effects = pd.Series({"h1": 0.1, "h2": -0.1})
    first = research_stats.autoregressive_spectral_holm_test(
        centered_sums, counts, effects, order_criterion="AIC",
        expected_hypothesis_count=2,
    )
    second = research_stats.autoregressive_spectral_holm_test(
        centered_sums, counts, effects, order_criterion="AIC",
        expected_hypothesis_count=2,
    )
    pd.testing.assert_frame_equal(first.summary, second.summary)
    pd.testing.assert_frame_equal(
        first.selected_coefficients, second.selected_coefficients
    )


def test_ar_spectral_holm_applies_one_frozen_family_wide_se_multiplier():
    rng = np.random.default_rng(903)
    index = pd.date_range("2025-01-01", periods=100, freq="D", tz="UTC")
    raw = pd.DataFrame(rng.normal(size=(100, 2)), index=index, columns=["h1", "h2"])
    centered = raw - raw.mean(axis=0)
    counts = pd.DataFrame(1, index=index, columns=raw.columns)
    effects = pd.Series({"h1": 0.2, "h2": -0.1})
    base = research_stats.autoregressive_spectral_holm_test(
        centered, counts, effects, order_criterion="BIC", expected_hypothesis_count=2
    ).summary.set_index("hypothesis_id")
    calibrated = research_stats.autoregressive_spectral_holm_test(
        centered, counts, effects, order_criterion="BIC",
        standard_error_multiplier=1.125, expected_hypothesis_count=2,
    ).summary.set_index("hypothesis_id")
    np.testing.assert_allclose(calibrated["standard_error"], 1.125 * base["standard_error"])
    np.testing.assert_allclose(calibrated["uncalibrated_standard_error"], base["standard_error"])
    assert calibrated["standard_error_multiplier"].eq(1.125).all()
    with pytest.raises(ValueError, match="at least one"):
        research_stats.autoregressive_spectral_holm_test(
            centered, counts, effects, order_criterion="BIC",
            standard_error_multiplier=0.999, expected_hypothesis_count=2,
        )


def test_ar_spectral_bh_end_to_end_uses_complete_family_and_exact_bh_values():
    rng = np.random.default_rng(904)
    index = pd.date_range("2025-01-01", periods=100, freq="D", tz="UTC")
    raw = pd.DataFrame(
        rng.normal(size=(100, 3)), index=index, columns=["h1", "h2", "h3"]
    )
    centered = raw - raw.mean(axis=0)
    counts = pd.DataFrame(1, index=index, columns=raw.columns)
    effects = pd.Series({"h1": 0.25, "h2": 0.10, "h3": -0.05})

    artifacts = research_stats.autoregressive_spectral_bh_test(
        centered,
        counts,
        effects,
        order_criterion="BIC",
        expected_hypothesis_count=3,
    )
    summary = artifacts.summary
    expected = research_stats.benjamini_hochberg_q_values(
        summary["raw_one_sided_p_value"]
    )
    np.testing.assert_allclose(summary["bh_adjusted_q_value"], expected)
    np.testing.assert_allclose(summary["family_adjusted_p_value"], expected)
    assert set(summary["inference_engine"]) == {
        "autoregressive_spectral_normal_bh"
    }
    assert summary["discovered"].equals(summary["bh_adjusted_q_value"].le(0.05))
    assert artifacts.family_summary.iloc[0]["hypothesis_count"] == 3
    assert artifacts.family_summary.iloc[0]["discovery_count"] == summary[
        "discovered"
    ].sum()
    assert "holm_adjusted_p_value" not in summary

    with pytest.raises(ValueError, match="wrong hypothesis count"):
        research_stats.autoregressive_spectral_bh_test(
            centered,
            counts,
            effects,
            order_criterion="BIC",
            expected_hypothesis_count=2,
        )


def test_annualized_sharpe_uses_explicit_periods_per_year():
    values = pd.Series([0.01, 0.02, -0.01, 0.03])
    expected = values.mean() / values.std(ddof=1) * math.sqrt(365)

    assert annualized_sharpe_from_periods(
        values, 365) == pytest.approx(expected)


def test_annualized_sharpe_rejects_invalid_period_count():
    with pytest.raises(ValueError, match="periods_per_year"):
        annualized_sharpe_from_periods([0.01, 0.02], 0)


def _centered_daily_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    raw = pd.DataFrame(
        {
            "a": [1.0, -1.0, 2.0, -2.0],
            "b": [0.5, -0.5, 1.0, -1.0],
        },
        index=pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC"),
    )
    return raw, pd.DataFrame(1, index=raw.index, columns=raw.columns), pd.Series(
        {"a": 0.6, "b": 0.2}
    )


def _long_centered_daily_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(20260812)
    raw = rng.normal(size=(64, 2))
    raw -= raw.mean(axis=0)
    frame = pd.DataFrame(
        raw,
        columns=["a", "b"],
        index=pd.date_range("2025-01-01", periods=64, freq="D", tz="UTC"),
    )
    return frame, pd.DataFrame(1, index=frame.index, columns=frame.columns), pd.Series(
        {"a": 0.2, "b": 0.1}
    )


def test_adaptive_flat_top_selects_short_iid_and_retains_ma_dependence():
    iid = np.random.default_rng(0).normal(size=(494, 1))
    iid -= iid.mean(axis=0)
    iid_result = adaptive_flat_top_ratio_standard_error(iid, np.ones_like(iid))
    assert iid_result.bandwidth.tolist() == [2]

    innovations = np.random.default_rng(42).normal(size=520)
    moving_average = np.convolve(innovations, np.ones(14) / 14.0, mode="valid")[
        :494, None
    ]
    moving_average -= moving_average.mean(axis=0)
    ma_result = adaptive_flat_top_ratio_standard_error(
        moving_average, np.ones_like(moving_average)
    )
    assert ma_result.bandwidth[0] >= 28
    assert not ma_result.bartlett_fallback_applied[0]


def test_adaptive_flat_top_is_scale_equivariant_and_has_explicit_fallback():
    values = np.random.default_rng(28).normal(size=(16, 1))
    values -= values.mean(axis=0)
    result = adaptive_flat_top_ratio_standard_error(values, np.ones_like(values))
    scaled = adaptive_flat_top_ratio_standard_error(7.0 * values, np.ones_like(values))
    assert result.bartlett_fallback_applied.tolist() == [True]
    assert result.raw_long_run_variance[0] < 0.0
    assert result.long_run_variance[0] > 0.0
    np.testing.assert_array_equal(result.bandwidth, scaled.bandwidth)
    np.testing.assert_allclose(scaled.standard_error, 7.0 * result.standard_error)
    np.testing.assert_allclose(
        scaled.long_run_variance, 49.0 * result.long_run_variance
    )


def test_e1f_restudentizes_every_bootstrap_sample_and_preserves_stepdown(monkeypatch):
    sums, counts, effects = _long_centered_daily_fixture()
    original = research_stats.adaptive_flat_top_ratio_standard_error
    observed_shapes = []

    def recording_entry(influence, daily_counts):
        observed_shapes.append(np.asarray(influence).shape)
        return original(influence, daily_counts)

    monkeypatch.setattr(
        research_stats, "adaptive_flat_top_ratio_standard_error", recording_entry
    )
    result = simulation_calibration_adaptive_flat_top_restudentized_stepdown_max_t(
        sums, counts, effects, block_length=14, n_bootstrap=999, seed=31, batch_size=37
    )
    assert observed_shapes[0] == (64, 2)
    assert all(len(shape) == 3 and shape[1:] == (64, 2) for shape in observed_shapes[1:])
    assert sum(shape[0] for shape in observed_shapes[1:]) == 999
    summary = result.summary.sort_values("observed_t", ascending=False)
    assert summary["stepdown_max_t_adjusted_p_value"].is_monotonic_increasing
    assert (
        result.summary["stepdown_max_t_adjusted_p_value"]
        >= result.summary["raw_one_sided_p_value"]
    ).all()
    assert result.summary["bootstrap_median_adaptive_bandwidth_days"].notna().all()
    assert result.summary["bootstrap_bartlett_fallback_rate"].between(0.0, 1.0).all()


def test_self_normalized_ratio_standard_error_matches_unequal_count_hand_calculation():
    influence = np.asarray([[2.0], [-1.0], [-3.0], [2.0], [0.0]])
    counts = np.asarray([[1.0], [2.0], [1.0], [3.0], [2.0]])
    cumulative = np.asarray([2.0, 1.0, -2.0, 0.0, 0.0])
    expected_normalizer = np.sum(cumulative**2) / 25.0
    expected_se = math.sqrt(expected_normalizer / 5.0) / counts.mean()

    result = self_normalized_ratio_standard_error(influence, counts)

    assert result.self_normalizer[0] == pytest.approx(expected_normalizer)
    assert result.standard_error[0] == pytest.approx(expected_se)
    wrong_equal_count_se = math.sqrt(expected_normalizer / 5.0)
    assert result.standard_error[0] != pytest.approx(wrong_equal_count_se)


def test_self_normalized_ratio_standard_error_fails_closed():
    counts = np.ones((5, 1))
    with pytest.raises(ValueError, match="sum to zero"):
        self_normalized_ratio_standard_error(np.arange(5.0)[:, None], counts)
    with pytest.raises(ValueError, match="finite and positive"):
        self_normalized_ratio_standard_error(np.zeros((5, 1)), counts)


def test_e1s_recomputes_each_bootstrap_self_normalizer(monkeypatch):
    sums, counts, effects = _centered_daily_fixture()
    original = research_stats.self_normalized_ratio_standard_error
    observed_shapes = []

    def recording_entry(influence, daily_counts):
        observed_shapes.append(np.asarray(influence).shape)
        return original(influence, daily_counts)

    monkeypatch.setattr(
        research_stats, "self_normalized_ratio_standard_error", recording_entry
    )
    result = simulation_calibration_self_normalized_stepdown_max_t(
        sums, counts, effects, block_length=2, n_bootstrap=999, seed=31, batch_size=37
    )
    assert observed_shapes[0] == (4, 2)
    assert all(len(shape) == 3 and shape[1:] == (4, 2) for shape in observed_shapes[1:])
    assert sum(shape[0] for shape in observed_shapes[1:]) == 999
    summary = result.summary.sort_values("observed_t", ascending=False)
    assert summary["stepdown_max_t_adjusted_p_value"].is_monotonic_increasing
    assert result.summary["self_normalizer"].gt(0.0).all()


def _manual_bartlett_ratio_se(influence, counts, max_lag):
    influence = np.asarray(influence, dtype=float)
    counts = np.asarray(counts, dtype=float)
    day_axis = influence.ndim - 2
    day_count = influence.shape[day_axis]
    long_run = np.mean(np.square(influence), axis=day_axis)
    for lag in range(1, max_lag + 1):
        left = np.take(influence, np.arange(lag, day_count), axis=day_axis)
        right = np.take(influence, np.arange(day_count - lag), axis=day_axis)
        gamma = np.sum(left * right, axis=day_axis) / day_count
        long_run += 2.0 * (1.0 - lag / (max_lag + 1.0)) * gamma
    return np.sqrt(long_run / day_count) / counts.mean(axis=day_axis)


def test_restudentized_block_bootstrap_recomputes_each_draw_standard_error():
    sums, counts, effects = _centered_daily_fixture()
    result = simulation_calibration_restudentized_circular_block_stepdown_max_t(
        sums,
        counts,
        effects,
        block_length=2,
        n_bootstrap=999,
        seed=31,
        batch_size=37,
    )
    starts = result.block_starts.iloc[0].to_numpy(dtype=int)
    sampled_days = np.concatenate(
        [np.arange(start, start + 2, dtype=int) % len(sums) for start in starts]
    )[: len(sums)]
    sample_sums = sums.to_numpy(dtype=float)[sampled_days]
    sample_counts = counts.to_numpy(dtype=float)[sampled_days]
    sample_effect = sample_sums.sum(axis=0) / sample_counts.sum(axis=0)
    sample_influence = sample_sums - sample_counts * sample_effect
    sample_se = _manual_bartlett_ratio_se(sample_influence, sample_counts, 1)
    expected_t = sample_effect / sample_se
    np.testing.assert_allclose(
        result.bootstrap_t_values.iloc[0].to_numpy(dtype=float),
        expected_t,
        rtol=0.0,
        atol=1e-14,
    )
    fixed_observed_se = result.summary["bootstrap_se"].to_numpy(dtype=float)
    assert not np.allclose(sample_se, fixed_observed_se)


def test_restudentized_stepdown_two_hypothesis_five_day_hand_recalculation(monkeypatch):
    index = pd.date_range("2025-02-01", periods=5, freq="D", tz="UTC")
    sums = pd.DataFrame(
        {"first": [2.0, -1.0, 1.0, -2.0, 0.0], "second": [1.0, -2.0, 0.0, 2.0, -1.0]},
        index=index,
    )
    counts = pd.DataFrame(1, index=index, columns=sums.columns)
    effects = pd.Series({"first": 0.4, "second": 0.4381780460041329})
    patterns = np.asarray([[0, 0, 2], [2, 2, 4], [1, 1, 3]], dtype=int)
    frozen_starts = np.repeat(patterns, 333, axis=0)

    class FrozenRng:
        def integers(self, low, high, size, endpoint):
            assert (low, high, size, endpoint) == (0, 5, (999, 3), False)
            return frozen_starts.copy()

    monkeypatch.setattr(research_stats.np.random, "default_rng", lambda seed: FrozenRng())
    result = simulation_calibration_restudentized_circular_block_stepdown_max_t(
        sums, counts, effects, block_length=2, n_bootstrap=999, seed=101
    )

    expected_t = np.repeat(
        np.asarray(
            [
                [2.211629342120, -1.474419561549],
                [-1.474419561549, 1.920553198993],
                [-1.280368799329, -0.625000000000],
            ]
        ),
        333,
        axis=0,
    )
    np.testing.assert_array_equal(result.block_starts, frozen_starts)
    np.testing.assert_allclose(result.bootstrap_t_values, expected_t, atol=1e-11)
    summary = result.summary.set_index("hypothesis_id").loc[sums.columns]
    np.testing.assert_allclose(summary["bootstrap_se"], [0.447213595500, 0.489897948557])
    np.testing.assert_allclose(summary["observed_t"], [0.894427190999, 0.894427190999])
    np.testing.assert_allclose(summary["raw_one_sided_p_value"], [0.334, 0.334])
    np.testing.assert_allclose(summary["stepdown_max_t_adjusted_p_value"], [0.667, 0.667])
    assert summary["observed_t_descending_rank"].tolist() == [1, 2]


def test_dependent_multiplier_uses_one_synchronized_weight_path_for_family():
    sums, counts, effects = _centered_daily_fixture()
    result = simulation_calibration_dependent_multiplier_bootstrap_stepdown_max_t(
        sums,
        counts,
        effects,
        bandwidth=2,
        n_bootstrap=999,
        seed=37,
        batch_size=41,
    )
    weights = result.block_starts.iloc[0].to_numpy(dtype=float)
    observed_se = result.summary["bootstrap_se"].to_numpy(dtype=float)
    expected_effect = weights @ sums.to_numpy(dtype=float) / counts.to_numpy(
        dtype=float
    ).sum(axis=0)
    np.testing.assert_allclose(
        result.bootstrap_t_values.iloc[0].to_numpy(dtype=float),
        expected_effect / observed_se,
        rtol=0.0,
        atol=1e-14,
    )
    assert len(result.block_starts.columns) == len(sums)
    assert result.summary.sort_values("observed_t", ascending=False)[
        "stepdown_max_t_adjusted_p_value"
    ].is_monotonic_increasing


def test_circular_stepdown_matches_direct_fixed_block_recalculation():
    sums, counts, effects = _centered_daily_fixture()
    result = circular_block_bootstrap_stepdown_max_t(
        sums,
        counts,
        effects,
        block_length=2,
        n_bootstrap=10_000,
        seed=17,
        batch_size=73,
    )

    starts = result.block_starts.to_numpy(dtype=int)
    sampled = np.stack(
        [
            np.concatenate(
                [np.arange(start, start + 2, dtype=int) % 4 for start in row]
            )[:4]
            for row in starts
        ]
    )
    direct_effects = sums.to_numpy()[sampled].sum(axis=1) / 4.0
    direct_se = direct_effects.std(axis=0, ddof=1)
    direct_t = direct_effects / direct_se
    observed_t = effects.reindex(sums.columns).to_numpy() / direct_se

    np.testing.assert_allclose(
        result.bootstrap_t_values.to_numpy(), direct_t, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        result.summary["observed_t"], observed_t, rtol=0.0, atol=1e-15
    )
    raw_p = (1 + (direct_t >= observed_t).sum(axis=0)) / 10_001
    np.testing.assert_allclose(result.summary["raw_one_sided_p_value"], raw_p)
    batch_mcse = result.summary["stepdown_adjusted_p_batch_mcse"].to_numpy()
    assert np.isfinite(batch_mcse).all()
    assert (batch_mcse >= 0.0).all()
    assert (batch_mcse > 0.0).any()
    assert set(result.summary["monte_carlo_batch_count"]) == {20}
    assert set(result.summary["monte_carlo_batch_size"]) == {500}
    assert (
        result.summary["stepdown_max_t_adjusted_p_value"]
        >= result.summary["raw_one_sided_p_value"]
    ).all()


def test_circular_stepdown_duplicate_hypotheses_do_not_multiply_penalty():
    sums, counts, effects = _centered_daily_fixture()
    single = circular_block_bootstrap_stepdown_max_t(
        sums[["a"]],
        counts[["a"]],
        effects[["a"]],
        block_length=2,
        n_bootstrap=10_000,
        seed=19,
    )
    duplicate_sums = pd.concat([sums[["a"]]] * 5, axis=1)
    duplicate_sums.columns = [f"a_{index}" for index in range(5)]
    duplicate_counts = pd.DataFrame(
        1, index=sums.index, columns=duplicate_sums.columns
    )
    duplicate_effects = pd.Series(0.6, index=duplicate_sums.columns)
    duplicated = circular_block_bootstrap_stepdown_max_t(
        duplicate_sums,
        duplicate_counts,
        duplicate_effects,
        block_length=2,
        n_bootstrap=10_000,
        seed=19,
    )

    assert duplicated.summary["raw_one_sided_p_value"].nunique() == 1
    assert duplicated.summary["stepdown_max_t_adjusted_p_value"].nunique() == 1
    assert duplicated.summary.iloc[0][
        "stepdown_max_t_adjusted_p_value"
    ] == pytest.approx(single.summary.iloc[0]["raw_one_sided_p_value"])


def test_circular_stepdown_is_reproducible_monotone_and_sign_sensitive():
    sums, counts, effects = _centered_daily_fixture()
    first = circular_block_bootstrap_stepdown_max_t(
        sums,
        counts,
        effects,
        block_length=2,
        n_bootstrap=10_000,
        seed=23,
    )
    second = circular_block_bootstrap_stepdown_max_t(
        sums,
        counts,
        effects,
        block_length=2,
        n_bootstrap=10_000,
        seed=23,
    )
    pd.testing.assert_frame_equal(first.summary, second.summary)
    pd.testing.assert_frame_equal(
        first.bootstrap_t_values, second.bootstrap_t_values
    )
    ordered = first.summary.sort_values("observed_t", ascending=False)
    assert ordered["stepdown_max_t_adjusted_p_value"].is_monotonic_increasing

    negative = circular_block_bootstrap_stepdown_max_t(
        sums,
        counts,
        -effects,
        block_length=2,
        n_bootstrap=10_000,
        seed=23,
    )
    assert (negative.summary["raw_one_sided_p_value"] > 0.5).all()


def test_circular_stepdown_two_sided_is_sign_symmetric_and_detects_negative_effect():
    sums, counts, effects = _centered_daily_fixture()
    positive = circular_block_bootstrap_stepdown_max_t(
        sums,
        counts,
        effects,
        block_length=2,
        n_bootstrap=10_000,
        seed=29,
        alternative="two-sided",
    )
    negative = circular_block_bootstrap_stepdown_max_t(
        sums,
        counts,
        -effects,
        block_length=2,
        n_bootstrap=10_000,
        seed=29,
        alternative="two-sided",
    )

    np.testing.assert_allclose(
        positive.summary["raw_two_sided_p_value"],
        negative.summary["raw_two_sided_p_value"],
    )
    np.testing.assert_allclose(
        positive.summary["stepdown_max_t_adjusted_p_value"],
        negative.summary["stepdown_max_t_adjusted_p_value"],
    )
    assert positive.summary["raw_one_sided_p_value"].isna().all()
    assert set(positive.summary["alternative"]) == {"two-sided"}


def test_circular_stepdown_fails_closed_on_invalid_contracts():
    sums, counts, effects = _centered_daily_fixture()
    with pytest.raises(ValueError, match="at least 10000"):
        circular_block_bootstrap_stepdown_max_t(
            sums, counts, effects, block_length=2, n_bootstrap=9999, seed=1
        )
    with pytest.raises(ValueError, match="alternative"):
        circular_block_bootstrap_stepdown_max_t(
            sums,
            counts,
            effects,
            block_length=2,
            n_bootstrap=10_000,
            seed=1,
            alternative="less",
        )
    with pytest.raises(ValueError, match="divisible"):
        circular_block_bootstrap_stepdown_max_t(
            sums,
            counts,
            effects,
            block_length=2,
            n_bootstrap=10_001,
            seed=1,
        )
    with pytest.raises(ValueError, match="not centered"):
        circular_block_bootstrap_stepdown_max_t(
            sums + 1.0,
            counts,
            effects,
            block_length=2,
            n_bootstrap=10_000,
            seed=1,
        )
    bad_counts = counts.astype(float)
    bad_counts.iloc[0, 0] = 0.0
    with pytest.raises(ValueError, match="positive integers"):
        circular_block_bootstrap_stepdown_max_t(
            sums,
            bad_counts,
            effects,
            block_length=2,
            n_bootstrap=10_000,
            seed=1,
        )


def test_circular_stepdown_fixed_null_fwer_and_one_strong_effect():
    rng = np.random.default_rng(20260724)
    false_family_rejections = 0
    for trial in range(20):
        common = rng.normal(size=40)
        raw = np.column_stack(
            [0.8 * common + rng.normal(scale=0.6, size=40) for _ in range(8)]
        )
        effects = pd.Series(raw.mean(axis=0), index=[f"h{i}" for i in range(8)])
        centered = raw - raw.mean(axis=0)
        sums = pd.DataFrame(
            centered,
            columns=effects.index,
            index=pd.date_range("2025-01-01", periods=40, freq="D", tz="UTC"),
        )
        counts = pd.DataFrame(1, index=sums.index, columns=sums.columns)
        result = circular_block_bootstrap_stepdown_max_t(
            sums,
            counts,
            effects,
            block_length=4,
            n_bootstrap=10_000,
            seed=1000 + trial,
        )
        false_family_rejections += int(
            (result.summary["stepdown_max_t_adjusted_p_value"] <= 0.05).any()
        )
    assert false_family_rejections <= 1

    effects.iloc[:] = 0.0
    effects.iloc[0] = 2.0
    result = circular_block_bootstrap_stepdown_max_t(
        sums,
        counts,
        effects,
        block_length=4,
        n_bootstrap=10_000,
        seed=777,
    )
    summary = result.summary.set_index("hypothesis_id")
    assert summary.loc["h0", "stepdown_max_t_adjusted_p_value"] <= 0.05
    assert (
        summary.drop(index="h0")["stepdown_max_t_adjusted_p_value"] > 0.05
    ).all()


def test_randomization_stepdown_matches_direct_joint_null_calculation():
    rng = np.random.Generator(np.random.PCG64DXSM(20260727))
    null = pd.DataFrame(
        rng.normal(size=(9_999, 3)),
        columns=["a", "b", "c"],
        index=pd.RangeIndex(9_999, name="randomization_idx"),
    )
    observed = pd.Series({"a": 2.5, "b": 0.1, "c": -0.2})

    result = randomization_stepdown_max_t(null, observed)
    summary = result.summary.set_index("hypothesis_id")
    means = null.mean(axis=0)
    stds = null.std(axis=0, ddof=1)
    direct_t = (null - means) / stds
    observed_t = (observed - means) / stds

    pd.testing.assert_frame_equal(result.null_t_values, direct_t)
    np.testing.assert_allclose(
        summary.loc[null.columns, "observed_t"].to_numpy(),
        observed_t.to_numpy(),
    )
    raw = (1 + (null >= observed).sum(axis=0)) / 10_000
    np.testing.assert_allclose(
        summary.loc[null.columns, "raw_one_sided_p_value"].to_numpy(),
        raw.to_numpy(),
    )
    order = np.argsort(-observed_t.to_numpy(), kind="mergesort")
    ordered_null = direct_t.to_numpy()[:, order]
    suffix_max = np.maximum.accumulate(ordered_null[:, ::-1], axis=1)[:, ::-1]
    step_counts = (
        suffix_max >= observed_t.to_numpy()[order][None, :]
    ).sum(axis=0)
    step_raw = (1 + step_counts) / 10_000
    ordered_adjusted = np.maximum.accumulate(
        np.maximum(step_raw, raw.to_numpy()[order])
    )
    direct_adjusted = np.empty_like(ordered_adjusted)
    direct_adjusted[order] = ordered_adjusted
    np.testing.assert_allclose(
        summary.loc[
            null.columns, "stepdown_max_t_adjusted_p_value"
        ].to_numpy(),
        direct_adjusted,
    )
    assert (
        summary["stepdown_max_t_adjusted_p_value"]
        >= summary["raw_one_sided_p_value"]
    ).all()
    assert summary.loc["a", "stepdown_max_t_adjusted_p_value"] <= 0.05
    assert summary.loc["c", "raw_one_sided_p_value"] > 0.5


def test_randomization_stepdown_fails_closed_on_invalid_family():
    rng = np.random.Generator(np.random.PCG64DXSM(7))
    null = pd.DataFrame(rng.normal(size=(9_999, 2)), columns=["a", "b"])
    observed = pd.Series({"a": 1.0, "b": 0.0})

    with pytest.raises(ValueError, match="at least 9999"):
        randomization_stepdown_max_t(null.iloc[:-1], observed)
    with pytest.raises(ValueError, match="cover every hypothesis"):
        randomization_stepdown_max_t(null, observed.drop("b"))
    duplicated = null.copy()
    duplicated.columns = ["a", "a"]
    with pytest.raises(ValueError, match="duplicate hypothesis"):
        randomization_stepdown_max_t(duplicated, observed)
    constant = null.copy()
    constant["b"] = 1.0
    with pytest.raises(ValueError, match="standard deviations"):
        randomization_stepdown_max_t(constant, observed)
