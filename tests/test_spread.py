import numpy as np
import pandas as pd

from qlab.spread import build_spread, rolling_hedge_ratio, half_life


class TestBuildSpread:

    def test_perfect_cancel(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="D")
        a = pd.Series(np.arange(100, dtype=float), index=idx)
        b = pd.Series(np.arange(100, dtype=float) * 0.5, index=idx)
        spread = build_spread(a, b, hedge_ratio=2.0)
        np.testing.assert_array_almost_equal(spread.values, 0.0)

    def test_different_lengths(self):
        idx_a = pd.date_range("2020-01-01", periods=100, freq="D")
        idx_b = pd.date_range("2020-01-15", periods=100, freq="D")
        a = pd.Series(1.0, index=idx_a)
        b = pd.Series(0.5, index=idx_b)
        spread = build_spread(a, b, hedge_ratio=1.0)
        assert len(spread) == len(idx_a.intersection(idx_b))


class TestHalfLife:

    def test_known_ou_process(self):
        np.random.seed(42)
        n = 2000
        theta = 0.1
        s = np.zeros(n)
        for i in range(1, n):
            s[i] = (1 - theta) * s[i - 1] + np.random.normal(0, 1)
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        hl = half_life(pd.Series(s, index=idx))
        expected = -np.log(2) / np.log(1 - theta)
        assert abs(hl - expected) < 2

    def test_trending_not_mean_reverting(self):
        idx = pd.date_range("2020-01-01", periods=200, freq="D")
        rng = np.random.RandomState(42)
        s = pd.Series(np.cumsum(rng.normal(0.1, 1, 200)), index=idx)
        # Random walk with drift: half_life should be very long (> true OU)
        assert half_life(s) > 50

    def test_short_series(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        assert half_life(pd.Series([1, 2, 3, 4, 5], index=idx)) == np.inf


class TestRollingHedgeRatio:

    def test_linear_relationship(self):
        idx = pd.date_range("2020-01-01", periods=200, freq="D")
        b = pd.Series(np.arange(200, dtype=float) + 100, index=idx)
        a = 2 * b + 10
        ratios = rolling_hedge_ratio(a, b, window=60)
        assert abs(ratios.iloc[-1] - 2.0) < 0.01
