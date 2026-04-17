import warnings

import numpy as np
import pandas as pd
import pytest

from qlab.metrics import (
    sharpe, sortino, max_drawdown, calmar, win_rate, profit_factor,
)


class TestSharpe:

    def test_known_answer_daily(self):
        """Hand-verified: mean/std * sqrt(365)."""
        rng = np.random.RandomState(42)
        r = rng.normal(0.001, 0.02, 252)
        expected = r.mean() / r.std(ddof=1) * np.sqrt(365)
        result = sharpe(r, holding_days=1, trading_days_per_year=365)
        assert abs(result - expected) < 1e-10

    def test_holding_days_changes_annualization(self):
        """14-day returns must use sqrt(365/14), not sqrt(365)."""
        rng = np.random.RandomState(42)
        r = rng.normal(0.005, 0.03, 26)
        correct = sharpe(r, holding_days=14, trading_days_per_year=365)
        wrong = r.mean() / r.std(ddof=1) * np.sqrt(365)
        # Wrong inflates by sqrt(14)
        ratio = wrong / correct
        assert abs(ratio - np.sqrt(14)) < 0.01

    def test_fx_252_vs_crypto_365(self):
        rng = np.random.RandomState(42)
        r = rng.normal(0.001, 0.02, 252)
        sr_crypto = sharpe(r, holding_days=1, trading_days_per_year=365)
        sr_fx = sharpe(r, holding_days=1, trading_days_per_year=252)
        ratio = sr_crypto / sr_fx
        assert abs(ratio - np.sqrt(365 / 252)) < 0.01

    def test_overlapping_returns_warn(self):
        """Overlapping 14d returns sampled daily → autocorrelation warning."""
        rng = np.random.RandomState(42)
        prices = np.cumprod(1 + rng.normal(0, 0.01, 400))
        overlapping = pd.Series(prices).pct_change(14).dropna().values
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sharpe(overlapping, holding_days=14)
            assert any("autocorrelation" in str(x.message).lower() for x in w)

    def test_non_overlapping_no_warn(self):
        rng = np.random.RandomState(42)
        r = rng.normal(0.005, 0.03, 26)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sharpe(r, holding_days=14)
            ac_warns = [x for x in w if "autocorrelation" in str(
                x.message).lower()]
            assert len(ac_warns) == 0

    def test_constant_returns_nan(self):
        assert np.isnan(sharpe(np.array([0.01] * 100)))

    def test_empty(self):
        assert np.isnan(sharpe(np.array([])))

    def test_single(self):
        assert np.isnan(sharpe(np.array([0.01])))

    def test_nan_handling(self):
        r = np.array([0.01, np.nan, 0.02, 0.01, np.nan, 0.015] * 20)
        assert np.isfinite(sharpe(r, holding_days=1))


class TestSortino:

    def test_all_positive_inf(self):
        r = np.array([0.01, 0.02, 0.005, 0.01, 0.015])
        assert sortino(r) == np.inf

    def test_greater_than_sharpe(self):
        rng = np.random.RandomState(42)
        r = rng.normal(0.001, 0.02, 500)
        assert sortino(r) > sharpe(r)


class TestMaxDrawdown:

    def test_known_answer(self):
        eq = np.array([100, 120, 90, 110])
        assert abs(max_drawdown(eq) - (-0.25)) < 1e-10

    def test_monotonic_up(self):
        eq = np.array([100, 101, 102, 103])
        assert max_drawdown(eq) == 0.0

    def test_all_down(self):
        eq = np.array([100, 50, 25])
        assert abs(max_drawdown(eq) - (-0.75)) < 1e-10


class TestCalmar:

    def test_positive_returns(self):
        rng = np.random.RandomState(42)
        r = rng.normal(0.002, 0.01, 500)
        c = calmar(r)
        assert np.isfinite(c)
        assert c > 0


class TestWinRate:

    def test_basic(self):
        r = np.array([0.1, -0.05, 0.03, -0.01, 0.02])
        assert abs(win_rate(r) - 0.6) < 1e-10

    def test_zeros_excluded(self):
        r = np.array([0.1, 0.0, -0.05, 0.0, 0.03])
        assert abs(win_rate(r) - 2 / 3) < 1e-10


class TestProfitFactor:

    def test_basic(self):
        r = np.array([0.10, -0.05, 0.03, -0.01])
        assert abs(profit_factor(r) - 0.13 / 0.06) < 1e-10

    def test_no_losses(self):
        assert profit_factor(np.array([0.01, 0.02])) == np.inf
