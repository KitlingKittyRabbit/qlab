import numpy as np
import pandas as pd

from qlab.backtest.signal_bt import run_signal_backtest


class TestSignalBacktest:

    def test_always_long_uptrend(self):
        idx = pd.date_range("2020-01-01", periods=200, freq="D")
        rng = np.random.RandomState(42)
        prices = pd.Series(
            100 * np.cumprod(1 + rng.normal(0.002, 0.005, 200)), index=idx
        )
        signals = pd.Series(1, index=idx)
        result = run_signal_backtest(
            signals, prices, holding_days=14, cost_bps=0)
        assert result["sharpe"] > 0
        assert result["n_trades"] > 0

    def test_non_overlapping_trades(self):
        idx = pd.date_range("2020-01-01", periods=200, freq="D")
        rng = np.random.RandomState(42)
        prices = pd.Series(
            100 * np.cumprod(1 + rng.normal(0, 0.01, 200)), index=idx
        )
        signals = pd.Series(np.where(rng.random(200) > 0.5, 1, -1), index=idx)
        result = run_signal_backtest(signals, prices, holding_days=14)
        trades = result["trades"]
        if len(trades) > 1:
            for i in range(1, len(trades)):
                assert trades.iloc[i]["entry_date"] >= trades.iloc[i - 1]["exit_date"]

    def test_flat_signal_no_trades(self):
        idx = pd.date_range("2020-01-01", periods=200, freq="D")
        prices = pd.Series(100.0, index=idx)
        signals = pd.Series(0, index=idx)
        result = run_signal_backtest(signals, prices, holding_days=14)
        assert result["n_trades"] == 0

    def test_cost_reduces_sharpe(self):
        idx = pd.date_range("2020-01-01", periods=500, freq="D")
        rng = np.random.RandomState(42)
        prices = pd.Series(
            100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 500)), index=idx
        )
        signals = pd.Series(1, index=idx)
        r0 = run_signal_backtest(signals, prices, holding_days=14, cost_bps=0)
        r5 = run_signal_backtest(signals, prices, holding_days=14, cost_bps=5)
        if np.isfinite(r0["sharpe"]) and np.isfinite(r5["sharpe"]):
            assert r0["sharpe"] >= r5["sharpe"]

    def test_insufficient_data(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        result = run_signal_backtest(
            pd.Series(1, index=idx), pd.Series(100.0, index=idx), holding_days=14
        )
        assert result["n_trades"] == 0
