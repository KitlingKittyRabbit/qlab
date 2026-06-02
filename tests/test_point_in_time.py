from __future__ import annotations

import importlib
import sys

import pandas as pd

from qlab.point_in_time import (
    PointInTimeSemantics,
    audit_aggregation_semantics,
    validate_entry_timing_contract,
)


def test_crypto_package_can_be_imported_without_config_for_non_path_exports(monkeypatch, tmp_path):
    monkeypatch.delenv("QLAB_CRYPTO_DATA_DIR", raising=False)
    monkeypatch.delenv("COINGLASS_DATA_DIR", raising=False)
    monkeypatch.setenv("QLAB_TRADE_ENV_PATH", str(tmp_path / "missing.env"))
    sys.modules.pop("qlab.data.crypto", None)

    crypto = importlib.import_module("qlab.data.crypto")

    assert "BTC" in crypto.CORE_SYMBOLS
    assert "ETH" in crypto.RESEARCH_SYMBOLS_12


def test_audit_aggregation_semantics_identifies_start_labeled_sum_bars():
    component = pd.Series(
        [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        index=pd.DatetimeIndex(
            [
                "2026-05-23T00:00:00Z",
                "2026-05-23T04:00:00Z",
                "2026-05-23T08:00:00Z",
                "2026-05-23T12:00:00Z",
                "2026-05-23T16:00:00Z",
                "2026-05-23T20:00:00Z",
            ]
        ),
    )
    aggregate = pd.Series(
        [6.0, 60.0],
        index=pd.DatetimeIndex(
            ["2026-05-23T00:00:00Z", "2026-05-23T12:00:00Z"]),
    )

    result = audit_aggregation_semantics(
        aggregate=aggregate,
        component=component,
        aggregate_duration=pd.Timedelta(hours=12),
        component_duration=pd.Timedelta(hours=4),
        reducer="sum",
    )

    assert result.inferred_timestamp_kind == "bar_start"
    assert result.usable_rows == 2
    assert result.start_match_ratio == 1.0
    assert result.end_match_ratio == 0.0


def test_validate_entry_timing_contract_flags_unsafe_plus_15m_for_start_labeled_final_bar():
    semantics = PointInTimeSemantics(
        timestamp_kind="bar_start", value_status="final")
    signal_labels = pd.DatetimeIndex(["2026-05-23T12:00:00Z"])
    observed_entry_times = pd.DatetimeIndex(["2026-05-23T12:15:00Z"])

    result = validate_entry_timing_contract(
        signal_labels=signal_labels,
        observed_entry_times=observed_entry_times,
        semantics=semantics,
        bar_duration=pd.Timedelta(hours=12),
        decision_delay=pd.Timedelta(minutes=15),
    )

    assert result.passed is False
    assert len(result.violations) == 1
    assert result.violations[0].expected_time == pd.Timestamp(
        "2026-05-24T00:15:00Z")


def test_validate_entry_timing_contract_accepts_finalized_bar_plus_delay():
    semantics = PointInTimeSemantics(
        timestamp_kind="bar_start", value_status="final")
    signal_labels = pd.DatetimeIndex(["2026-05-23T12:00:00Z"])
    observed_entry_times = pd.DatetimeIndex(["2026-05-24T00:15:00Z"])

    result = validate_entry_timing_contract(
        signal_labels=signal_labels,
        observed_entry_times=observed_entry_times,
        semantics=semantics,
        bar_duration=pd.Timedelta(hours=12),
        decision_delay=pd.Timedelta(minutes=15),
    )

    assert result.passed is True
    assert result.violations == ()


def test_partial_semantics_require_explicit_publication_lag_for_contract_checks():
    semantics = PointInTimeSemantics(
        timestamp_kind="bar_start", value_status="partial")
    signal_labels = pd.DatetimeIndex(["2026-05-23T12:00:00Z"])
    observed_entry_times = pd.DatetimeIndex(["2026-05-23T12:15:00Z"])

    result = validate_entry_timing_contract(
        signal_labels=signal_labels,
        observed_entry_times=observed_entry_times,
        semantics=semantics,
        bar_duration=pd.Timedelta(hours=12),
        decision_delay=pd.Timedelta(minutes=15),
    )

    assert result.passed is False
    assert len(result.violations) == 1
    assert result.violations[0].reason == "semantics do not define an explicit availability time"
