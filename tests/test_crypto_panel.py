import numpy as np
import pandas as pd
import pytest

from qlab.data.crypto.panel import (
    forward_returns_for_symbol,
    normalize_price_frame,
    panel_forward_returns,
    panel_with_forward_return,
    rank_standardize_grouped_series,
    relabel_open_indexed_bars_to_bar_end,
)


def test_normalize_price_frame_localizes_and_deduplicates():
    index = pd.DatetimeIndex(["2024-01-01", "2024-01-01", "2024-01-02"])
    frame = pd.DataFrame({"c": [100.0, 101.0, 102.0]}, index=index)

    result = normalize_price_frame(frame)

    assert str(result.index.tz) == "UTC"
    assert len(result) == 2
    assert result.iloc[0]["c"] == 101.0


def test_open_indexed_bar_is_only_available_at_bar_end() -> None:
    frame = pd.DataFrame(
        {"c": [101.0]},
        index=pd.DatetimeIndex(["2026-01-01 00:00"], tz="UTC"),
    )
    result = relabel_open_indexed_bars_to_bar_end(frame, pd.Timedelta(minutes=15))
    assert result.index[0] == pd.Timestamp("2026-01-01 00:15", tz="UTC")


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


def test_forward_returns_for_symbol_requires_exact_future_timestamp():
    decision_index = pd.DatetimeIndex(
        ["2024-01-01 00:00", "2024-01-01 04:00"], tz="UTC")
    prices = pd.Series(
        [100.0, 121.0],
        index=pd.DatetimeIndex(
            ["2024-01-01 00:00", "2024-01-01 08:00"], tz="UTC"),
    )

    result = forward_returns_for_symbol(
        decision_index, prices, pd.Timedelta(hours=4))

    assert result.empty


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


def test_rank_standardize_grouped_series_matches_cross_section_semantics():
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-01-01T00:15:00Z"), "A"),
            (pd.Timestamp("2026-01-01T00:15:00Z"), "B"),
            (pd.Timestamp("2026-01-01T00:15:00Z"), "C"),
            (pd.Timestamp("2026-01-02T00:15:00Z"), "A"),
            (pd.Timestamp("2026-01-02T00:15:00Z"), "B"),
        ],
        names=["decision_ts", "symbol"],
    )
    series = pd.Series([2.0, np.nan, 4.0, 10.0, np.nan], index=index)

    result = rank_standardize_grouped_series(series)

    assert result.loc[(pd.Timestamp("2026-01-01T00:15:00Z"), "A")] == pytest.approx(-1.0)
    assert np.isnan(result.loc[(pd.Timestamp("2026-01-01T00:15:00Z"), "B")])
    assert result.loc[(pd.Timestamp("2026-01-01T00:15:00Z"), "C")] == pytest.approx(1.0)
    assert result.loc[(pd.Timestamp("2026-01-02T00:15:00Z"), "A")] == pytest.approx(0.0)
    assert np.isnan(result.loc[(pd.Timestamp("2026-01-02T00:15:00Z"), "B")])
