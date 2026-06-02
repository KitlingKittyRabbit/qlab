import numpy as np
import pandas as pd
import pytest

from qlab.data.crypto.panel import (
    forward_returns_for_symbol,
    normalize_price_frame,
    panel_forward_returns,
    panel_with_forward_return,
)


def test_normalize_price_frame_localizes_and_deduplicates():
    index = pd.DatetimeIndex(["2024-01-01", "2024-01-01", "2024-01-02"])
    frame = pd.DataFrame({"c": [100.0, 101.0, 102.0]}, index=index)

    result = normalize_price_frame(frame)

    assert str(result.index.tz) == "UTC"
    assert len(result) == 2
    assert result.iloc[0]["c"] == 101.0


def test_forward_returns_for_symbol_uses_calendar_horizon():
    decision_index = pd.DatetimeIndex(
        ["2024-01-01 00:00", "2024-01-01 04:00"], tz="UTC")
    prices = pd.Series(
        [100.0, 110.0, 121.0],
        index=pd.DatetimeIndex(
            ["2024-01-01 00:00", "2024-01-01 04:00", "2024-01-01 08:00"], tz="UTC"),
    )

    result = forward_returns_for_symbol(
        decision_index, prices, pd.Timedelta(hours=4))

    assert result.iloc[0] == pytest.approx(0.1)
    assert result.iloc[1] == pytest.approx(0.1)


def test_panel_forward_return_join():
    dates = pd.DatetimeIndex(
        ["2024-01-01 00:00", "2024-01-01 04:00"], tz="UTC", name="decision_ts")
    index = pd.MultiIndex.from_product(
        [dates, ["BTC", "ETH"]], names=["decision_ts", "symbol"])
    panel = pd.DataFrame({"factor": [1.0, 2.0, 3.0, 4.0]}, index=index)
    price_index = pd.DatetimeIndex(
        ["2024-01-01 00:00", "2024-01-01 04:00", "2024-01-01 08:00"],
        tz="UTC",
    )
    price_payloads = {
        "BTC": pd.DataFrame({"c": [100.0, 110.0, 121.0]}, index=price_index),
        "ETH": pd.DataFrame({"c": [200.0, 220.0, 242.0]}, index=price_index),
    }

    returns = panel_forward_returns(
        panel, price_payloads, pd.Timedelta(hours=4))
    joined = panel_with_forward_return(
        panel, price_payloads, pd.Timedelta(hours=4))

    assert len(returns) == 4
    assert joined["forward_return"].notna().all()
