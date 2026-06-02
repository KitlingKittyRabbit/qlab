from __future__ import annotations

import math

import pandas as pd
import pytest

from qlab.research_stats import (
    annualized_sharpe_from_periods,
    hac_t_stat,
    newey_west_max_lags,
    simple_t_stat,
)


def test_newey_west_lag_count_is_bounded():
    assert newey_west_max_lags(1) == 0
    assert newey_west_max_lags(3) <= 2
    assert newey_west_max_lags(200) > 0


def test_hac_t_stat_returns_finite_value_for_variable_series():
    values = pd.Series([0.01, 0.02, -0.01, 0.03, 0.00, 0.02])

    result = hac_t_stat(values)

    assert math.isfinite(result)


def test_simple_t_stat_matches_manual_formula():
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    expected = values.mean() / values.std(ddof=1) * math.sqrt(len(values))

    assert simple_t_stat(values) == pytest.approx(expected)


def test_annualized_sharpe_uses_explicit_periods_per_year():
    values = pd.Series([0.01, 0.02, -0.01, 0.03])
    expected = values.mean() / values.std(ddof=1) * math.sqrt(365)

    assert annualized_sharpe_from_periods(
        values, 365) == pytest.approx(expected)


def test_annualized_sharpe_rejects_invalid_period_count():
    with pytest.raises(ValueError, match="periods_per_year"):
        annualized_sharpe_from_periods([0.01, 0.02], 0)
