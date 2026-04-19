import numpy as np
import pandas as pd
import pytest

from qlab.diagnostics import forward_returns, ic_decay, quantile_returns


class TestForwardReturns:

    def test_basic_forward_returns(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        prices = pd.Series([100.0, 110.0, 121.0, 133.1, 146.41], index=idx)

        result = forward_returns(prices, horizons=[1, 2])

        expected_1 = pd.Series([0.1, 0.1, 0.1, 0.1, np.nan], index=idx, name="fwd_1")
        expected_2 = pd.Series([0.21, 0.21, 0.21, np.nan, np.nan], index=idx, name="fwd_2")

        pd.testing.assert_series_equal(result["fwd_1"], expected_1)
        pd.testing.assert_series_equal(result["fwd_2"], expected_2)

    def test_invalid_horizon_raises(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        prices = pd.Series(np.arange(5, dtype=float) + 100.0, index=idx)
        with pytest.raises(ValueError, match="positive"):
            forward_returns(prices, horizons=[0, 1])


class TestICDecay:

    def test_from_precomputed_forward_returns(self):
        idx = pd.date_range("2020-01-01", periods=20, freq="D")
        factor = pd.Series(np.arange(20, dtype=float), index=idx)
        ret = pd.DataFrame(
            {
                "fwd_1": np.arange(20, dtype=float),
                "fwd_3": -np.arange(20, dtype=float),
            },
            index=idx,
        )

        result = ic_decay(factor, forward_ret=ret)

        assert list(result["horizon"]) == [1, 3]
        assert np.isclose(result.loc[result["column"] == "fwd_1", "ic"].iloc[0], 1.0)
        assert np.isclose(result.loc[result["column"] == "fwd_3", "ic"].iloc[0], -1.0)
        assert (result["n_obs"] == 20).all()

    def test_from_prices_and_horizons(self):
        idx = pd.date_range("2020-01-01", periods=30, freq="D")
        prices = pd.Series(np.linspace(100, 160, len(idx)), index=idx)
        factor = pd.Series(np.linspace(1, 30, len(idx)), index=idx)

        result = ic_decay(factor, prices=prices, horizons=[1, 5])

        assert list(result["horizon"]) == [1, 5]
        assert list(result["column"]) == ["fwd_1", "fwd_5"]
        assert result["n_obs"].iloc[0] == len(idx) - 1
        assert result["n_obs"].iloc[1] == len(idx) - 5

    def test_exactly_one_input_source_required(self):
        idx = pd.date_range("2020-01-01", periods=20, freq="D")
        factor = pd.Series(np.arange(20, dtype=float), index=idx)
        prices = pd.Series(np.arange(20, dtype=float) + 100.0, index=idx)
        ret = pd.DataFrame({"fwd_1": np.arange(20, dtype=float)}, index=idx)

        with pytest.raises(ValueError, match="exactly one"):
            ic_decay(factor, prices=prices, forward_ret=ret, horizons=[1])

    def test_requested_horizon_missing_raises(self):
        idx = pd.date_range("2020-01-01", periods=20, freq="D")
        factor = pd.Series(np.arange(20, dtype=float), index=idx)
        ret = pd.DataFrame({"fwd_1": np.arange(20, dtype=float)}, index=idx)

        with pytest.raises(ValueError, match="missing"):
            ic_decay(factor, forward_ret=ret, horizons=[1, 3])

    def test_unparseable_forward_return_columns_raise(self):
        idx = pd.date_range("2020-01-01", periods=20, freq="D")
        factor = pd.Series(np.arange(20, dtype=float), index=idx)
        ret = pd.DataFrame({"next_week": np.arange(20, dtype=float)}, index=idx)

        with pytest.raises(ValueError, match="Could not infer horizon"):
            ic_decay(factor, forward_ret=ret)


class TestQuantileReturns:

    def test_from_precomputed_forward_series(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="D")
        factor = pd.Series(np.arange(100, dtype=float), index=idx)
        forward = pd.Series(np.arange(100, dtype=float), index=idx, name="fwd_7")

        result = quantile_returns(factor, forward_ret=forward, n_quantiles=5)

        assert list(result["quantile"]) == [1, 2, 3, 4, 5]
        assert result["count"].sum() == 100
        assert result["horizon"].nunique() == 1
        assert result["horizon"].iloc[0] == 7
        assert result["mean_return"].is_monotonic_increasing

    def test_from_prices_and_horizon(self):
        idx = pd.date_range("2020-01-01", periods=40, freq="D")
        prices = pd.Series(np.linspace(100, 140, len(idx)), index=idx)
        factor = pd.Series(np.sin(np.arange(len(idx))), index=idx)

        result = quantile_returns(factor, prices=prices, horizon=3, n_quantiles=4)

        assert list(result.columns) == ["horizon", "quantile", "mean_return", "count"]
        assert result["horizon"].iloc[0] == 3
        assert result["count"].sum() == len(idx) - 3

    def test_rejects_multiple_forward_columns(self):
        idx = pd.date_range("2020-01-01", periods=20, freq="D")
        factor = pd.Series(np.arange(20, dtype=float), index=idx)
        ret = pd.DataFrame(
            {
                "fwd_1": np.arange(20, dtype=float),
                "fwd_3": np.arange(20, dtype=float),
            },
            index=idx,
        )

        with pytest.raises(ValueError, match="exactly one column"):
            quantile_returns(factor, forward_ret=ret)

    def test_requires_enough_observations(self):
        idx = pd.date_range("2020-01-01", periods=3, freq="D")
        factor = pd.Series(np.arange(3, dtype=float), index=idx)
        forward = pd.Series(np.arange(3, dtype=float), index=idx, name="fwd_1")

        with pytest.raises(ValueError, match="not enough aligned observations"):
            quantile_returns(factor, forward_ret=forward, n_quantiles=5)

    def test_rejects_tied_factor_without_enough_unique_values(self):
        idx = pd.date_range("2020-01-01", periods=20, freq="D")
        factor = pd.Series([1.0] * 10 + [2.0] * 10, index=idx)
        forward = pd.Series(np.arange(20, dtype=float), index=idx, name="fwd_1")

        with pytest.raises(ValueError, match="enough unique values"):
            quantile_returns(factor, forward_ret=forward, n_quantiles=5)

    def test_requires_explicit_or_parseable_horizon(self):
        idx = pd.date_range("2020-01-01", periods=20, freq="D")
        factor = pd.Series(np.arange(20, dtype=float), index=idx)
        forward = pd.Series(np.arange(20, dtype=float), index=idx, name="next_week")

        with pytest.raises(ValueError, match="Could not infer horizon"):
            quantile_returns(factor, forward_ret=forward)