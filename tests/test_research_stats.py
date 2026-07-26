from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from qlab.research_stats import (
    annualized_sharpe_from_periods,
    benjamini_hochberg_q_values,
    circular_block_bootstrap_stepdown_max_t,
    hac_t_stat,
    holm_adjusted_p_values,
    newey_west_max_lags,
    normal_one_sided_p_value,
    normal_two_sided_p_value,
    simple_t_stat,
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


def test_circular_stepdown_fails_closed_on_invalid_contracts():
    sums, counts, effects = _centered_daily_fixture()
    with pytest.raises(ValueError, match="at least 10000"):
        circular_block_bootstrap_stepdown_max_t(
            sums, counts, effects, block_length=2, n_bootstrap=9999, seed=1
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
        sums = pd.DataFrame(centered, columns=effects.index)
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
