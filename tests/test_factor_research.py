import numpy as np
import pandas as pd
import pytest

from qlab.factor_research import (
    ComboSpec,
    bucket_diagnostics_for_frame,
    combo_decision_frequency,
    fama_macbeth_diagnostics_for_frame_slice,
    features_decision_frequency,
    normalized_feature_weights,
    rank_ic_diagnostics_for_frame,
    summarize_fama_macbeth,
    train_feature_stats,
    validate_no_overlap_design,
)


def _panel_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    symbols = ["A", "B", "C", "D", "E", "F", "G", "H"]
    index = pd.MultiIndex.from_product(
        [dates, symbols], names=["decision_ts", "symbol"])
    values = np.tile(np.arange(1, 9, dtype=float), len(dates))
    size_control = np.tile(
        [0.125, -0.8, 0.33, 1.7, -1.4, 0.2, 0.95, -0.55], len(dates))
    momentum_control = np.tile(
        [-1.2, 0.4, 1.1, -0.3, 0.75, -0.9, 1.6, 0.05], len(dates))
    volatility_control = np.tile(
        [0.8, -1.1, -0.6, 0.9, 1.4, -0.2, -1.5, 0.35], len(dates))
    return pd.DataFrame(
        {
            "symbol": index.get_level_values("symbol"),
            "factor": values,
            "forward_return": values / 100.0 + size_control / 1000.0,
            "size_control": size_control,
            "momentum_control": momentum_control,
            "volatility_control": volatility_control,
        },
        index=index,
    )


def test_rank_ic_diagnostics_scores_each_decision():
    diagnostics = rank_ic_diagnostics_for_frame(
        _panel_frame(), "factor", min_cross_section=5)

    assert len(diagnostics) == 3
    assert {row["status"] for row in diagnostics} == {"ok"}
    assert all(row["raw_rank_ic"] == pytest.approx(1.0) for row in diagnostics)


def test_bucket_diagnostics_orients_spread():
    detail, diagnostics = bucket_diagnostics_for_frame(
        _panel_frame(), "factor", direction=1, n_buckets=3)

    assert {row["status"] for row in diagnostics} == {"ok"}
    pivot = detail.pivot_table(
        index="decision_ts", columns="bucket", values="bucket_return")
    assert (pivot[3] > pivot[1]).all()


def test_fama_macbeth_slice_and_summary_use_configurable_controls():
    diagnostics = fama_macbeth_diagnostics_for_frame_slice(
        _panel_frame(),
        "factor",
        direction=1,
        control_columns=("size_control", "momentum_control",
                         "volatility_control"),
        min_cross_section=6,
    )

    detail = pd.DataFrame(
        [row for row in diagnostics if row["status"] == "ok"])
    summary = summarize_fama_macbeth(
        "1d",
        "1d",
        "factor",
        detail,
        {"train_days": 30, "test_days": 10, "embargo_days": 1, "step_days": 10},
        diagnostics,
        ("size_control", "momentum_control", "volatility_control"),
    )

    assert summary["gamma_observation_count"] == 3
    assert summary["scored_decision_count"] == 3
    assert "mean_size_control_gamma" in summary


def test_train_stats_and_weights_are_generic():
    stats = train_feature_stats(
        _panel_frame(), ("factor",), min_cross_section=5)

    assert stats is not None
    scores, weights = normalized_feature_weights(stats, "ic_abs")
    assert scores["factor"] > 0
    assert weights["factor"] == pytest.approx(1.0)


def test_native_signal_timeframe_drives_combo_frequency():
    horizon_deltas = {
        "2h": pd.Timedelta(hours=2),
        "4h": pd.Timedelta(hours=4),
        "12h": pd.Timedelta(hours=12),
        "1d": pd.Timedelta(days=1),
    }
    supported = ("4h", "12h", "1d")
    spec = ComboSpec("x", "track", "2h", "12h", ("a__4h", "b__1d"))

    assert features_decision_frequency(
        spec.feature_names, spec.panel_frequency, horizon_deltas, supported) == "1d"
    assert combo_decision_frequency(spec, horizon_deltas, supported) == "1d"

    validate_no_overlap_design([spec], horizon_deltas, supported)
    with pytest.raises(ValueError, match="return_horizon <= decision_frequency"):
        validate_no_overlap_design(
            [ComboSpec("bad", "track", "2h", "1d", ("a__4h",))], horizon_deltas, supported)
