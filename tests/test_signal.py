import numpy as np
import pandas as pd
import pytest

from qlab.signal import zscore, zscore_fixed, ic, ic_direction, threshold_signal


class TestZscore:

    def test_constant_near_zero(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="D")
        s = pd.Series(5.0, index=idx)
        z = zscore(s, method="rolling", window=20)
        assert abs(z.iloc[-1]) < 1e-5

    def test_expanding_valid_range(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="D")
        rng = np.random.RandomState(42)
        s = pd.Series(rng.normal(0, 1, 100), index=idx)
        z = zscore(s, method="expanding", window=20)
        assert z.iloc[19:].notna().all()
        assert z.iloc[:19].isna().all()

    def test_invalid_method(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        s = pd.Series(1.0, index=idx)
        with pytest.raises(ValueError, match="method"):
            zscore(s, method="invalid")


class TestZscoreFixed:

    def test_basic(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        s = pd.Series([10, 12, 8, 11, 9], dtype=float, index=idx)
        z = zscore_fixed(s, mu=10, sd=2)
        expected = (s - 10) / 2
        pd.testing.assert_series_equal(z, expected)

    def test_zero_std_returns_nan(self):
        idx = pd.date_range("2020-01-01", periods=3, freq="D")
        s = pd.Series([1.0, 2.0, 3.0], index=idx)
        assert zscore_fixed(s, mu=2, sd=0).isna().all()


class TestIC:

    def test_perfect_positive(self):
        f = np.arange(100, dtype=float)
        r = np.arange(100, dtype=float)
        assert abs(ic(f, r) - 1.0) < 1e-10

    def test_perfect_negative(self):
        f = np.arange(100, dtype=float)
        r = -np.arange(100, dtype=float)
        assert abs(ic(f, r) - (-1.0)) < 1e-10

    def test_insufficient_data(self):
        assert np.isnan(ic(np.array([1, 2]), np.array([1, 2])))

    def test_nan_excluded(self):
        f = np.array([1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=float)
        r = np.array([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=float)
        assert np.isfinite(ic(f, r))


class TestICDirection:

    def test_positive(self):
        f = np.arange(100, dtype=float)
        r = np.arange(100, dtype=float)
        assert ic_direction(f, r) == 1

    def test_negative(self):
        f = np.arange(100, dtype=float)
        r = -np.arange(100, dtype=float)
        assert ic_direction(f, r) == -1


class TestThresholdSignal:

    def test_basic(self):
        comp = np.array([1.0, -1.0, 0.3, -0.3, 0.6, -0.7])
        result = threshold_signal(comp, threshold=0.5)
        expected = np.array([1, -1, 0, 0, 1, -1])
        np.testing.assert_array_equal(result, expected)

    def test_at_boundary_is_flat(self):
        comp = np.array([0.5, -0.5])
        result = threshold_signal(comp, threshold=0.5)
        np.testing.assert_array_equal(result, np.array([0, 0]))
