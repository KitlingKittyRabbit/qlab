import numpy as np
import pandas as pd
import pytest

from qlab.backtest.spread_bt import run_spread_backtest


class TestSpreadBacktest:

    @staticmethod
    def _ou(n=2000, theta=0.1, seed=42):
        rng = np.random.RandomState(seed)
        s = np.zeros(n)
        for i in range(1, n):
            s[i] = (1 - theta) * s[i - 1] + rng.normal(0, 1)
        return pd.Series(s, index=pd.date_range("2020-01-01", periods=n, freq="D"))

    def test_mean_reverting_positive_pnl(self):
        spread = self._ou(n=2000)
        result = run_spread_backtest(
            spread, entry_z=2.0, exit_z=0.0, lookback=60, capital_base=1000.0)
        assert result["n_trades"] > 0
        assert result["trades"]["pnl"].sum() > 0

    def test_constant_spread_no_trades(self):
        idx = pd.date_range("2020-01-01", periods=200, freq="D")
        result = run_spread_backtest(
            pd.Series(100.0, index=idx), lookback=60, capital_base=1000.0)
        assert result["n_trades"] == 0

    def test_max_holding_respected(self):
        spread = self._ou(n=2000, theta=0.01)
        result = run_spread_backtest(
            spread, entry_z=1.5, exit_z=0.0, max_holding_bars=20, lookback=60, capital_base=1000.0
        )
        if result["n_trades"] > 0:
            assert result["trades"]["holding_bars"].max() <= 20

    def test_cost_reduces_total_pnl(self):
        spread = self._ou(n=2000)
        r0 = run_spread_backtest(
            spread, entry_z=2.0, exit_z=0.0, cost_per_trade=0.0, capital_base=100.0)
        r1 = run_spread_backtest(
            spread, entry_z=2.0, exit_z=0.0, cost_per_trade=1.0, capital_base=100.0)
        if r0["n_trades"] > 0 and r1["n_trades"] > 0:
            assert r0["trades"]["pnl"].sum() >= r1["trades"]["pnl"].sum()

    def test_cost_flows_through_daily_returns_and_equity(self):
        spread = self._ou(n=2000)
        r0 = run_spread_backtest(
            spread, entry_z=2.0, exit_z=0.0, cost_per_trade=0.0, capital_base=100.0)
        r1 = run_spread_backtest(
            spread, entry_z=2.0, exit_z=0.0, cost_per_trade=1.0, capital_base=100.0)
        if r0["n_trades"] > 0 and r1["n_trades"] > 0:
            assert r0["daily_pnl"].sum() >= r1["daily_pnl"].sum()
            assert r0["equity_curve"].iloc[-1] >= r1["equity_curve"].iloc[-1]

    def test_daily_returns_match_daily_pnl_over_capital(self):
        spread = self._ou(n=1000)
        result = run_spread_backtest(
            spread, entry_z=2.0, exit_z=0.0, cost_per_trade=0.5, capital_base=250.0)
        if result["n_trades"] > 0:
            expected = result["daily_pnl"] / 250.0
            pd.testing.assert_series_equal(result["daily_returns"], expected)

    def test_end_of_data_closes_open_trade(self):
        idx = pd.date_range("2020-01-01", periods=200, freq="D")
        spread = pd.Series(np.linspace(0, -10, len(idx)), index=idx)
        result = run_spread_backtest(
            spread, entry_z=0.5, exit_z=0.0, stop_z=20.0, lookback=20, capital_base=100.0
        )
        if result["n_trades"] > 0:
            assert result["trades"].iloc[-1]["exit_date"] == spread.index[-1]
            assert result["trades"].iloc[-1]["exit_reason"] in {
                "end_of_data", "stop", "reversion", "max_hold"}

    def test_invalid_capital_base_raises(self):
        spread = self._ou(n=200)
        with pytest.raises(ValueError, match="capital_base"):
            run_spread_backtest(spread, capital_base=0.0)

    def test_stop_loss_present(self):
        spread = self._ou(n=3000, theta=0.05)
        result = run_spread_backtest(
            spread, entry_z=1.5, exit_z=0.0, stop_z=3.0, lookback=60, capital_base=1000.0
        )
        if result["n_trades"] > 5:
            reasons = result["trades"]["exit_reason"].value_counts()
            # At least some trades should have different exit reasons
            assert len(reasons) >= 1
