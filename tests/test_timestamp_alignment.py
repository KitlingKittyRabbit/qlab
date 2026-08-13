from __future__ import annotations

import pandas as pd
import pytest

from qlab.data.crypto.timestamp_alignment import (
    aggregate_fine_series,
    compare_coarse_to_fine_aggregations,
    compare_timestamp_alignments,
)


def _series(values: list[float]) -> pd.Series:
    return pd.Series(
        values,
        index=pd.date_range("2026-01-01", periods=len(values), freq="1h", tz="UTC"),
    )


def test_alignment_audit_distinguishes_same_timestamp_from_previous_period() -> None:
    source = _series([1.0, 2.0, 3.0, 4.0])
    reference = _series([1.0, 2.0, 3.0, 4.0])

    audit = compare_timestamp_alignments(
        reference,
        source,
        source_offsets={"same_timestamp": "0h", "previous_period": "-1h"},
        rounded_decimals=2,
    )

    summary = audit.summary.set_index("offset_id")
    assert summary.loc["same_timestamp", "common_count"] == 3
    assert summary.loc["same_timestamp", "rounded_match_rate"] == 1.0
    assert summary.loc["same_timestamp", "mean_absolute_error"] == 0.0
    assert summary.loc["previous_period", "common_count"] == 3
    assert summary.loc["previous_period", "rounded_match_rate"] == 0.0
    previous = audit.observations.query("offset_id == 'previous_period'")
    assert (previous["source_ts"] == previous["reference_ts"] - pd.Timedelta(hours=1)).all()


def test_alignment_audit_can_show_previous_period_is_the_exact_mapping() -> None:
    source = _series([10.0, 20.0, 30.0, 40.0])
    reference = _series([99.0, 10.0, 20.0, 30.0])

    audit = compare_timestamp_alignments(
        reference,
        source,
        source_offsets={"same_timestamp": "0h", "previous_period": "-1h"},
        rounded_decimals=2,
    )

    summary = audit.summary.set_index("offset_id")
    assert summary.loc["previous_period", "rounded_match_rate"] == 1.0
    assert summary.loc["previous_period", "mean_absolute_error"] == 0.0
    assert summary.loc["same_timestamp", "mean_absolute_error"] > 0.0


def test_positive_offset_compares_reference_t_with_source_t_plus_period() -> None:
    source = _series([10.0, 20.0, 30.0, 40.0])
    reference = _series([20.0, 30.0, 40.0])

    audit = compare_timestamp_alignments(
        reference,
        source,
        source_offsets={"same_timestamp": "0h", "next_period_end": "1h"},
        rounded_decimals=2,
    )

    summary = audit.summary.set_index("offset_id")
    assert summary.loc["next_period_end", "rounded_match_rate"] == 1.0
    assert summary.loc["next_period_end", "mean_absolute_error"] == 0.0
    next_period = audit.observations.query("offset_id == 'next_period_end'")
    assert (next_period["source_ts"] == next_period["reference_ts"] + pd.Timedelta(hours=1)).all()


@pytest.mark.parametrize("defect", ["naive", "duplicate", "nonfinite"])
def test_alignment_audit_fails_closed_on_ambiguous_input(defect: str) -> None:
    reference = _series([1.0, 2.0])
    if defect == "naive":
        reference.index = reference.index.tz_localize(None)
    elif defect == "duplicate":
        reference.index = pd.DatetimeIndex([reference.index[0], reference.index[0]])
    else:
        reference.iloc[1] = float("nan")

    with pytest.raises(ValueError):
        compare_timestamp_alignments(
            reference,
            _series([1.0, 2.0]),
            source_offsets={"same_timestamp": "0h"},
            rounded_decimals=2,
        )


def test_alignment_audit_fails_when_declared_offsets_have_no_overlap() -> None:
    source = _series([1.0, 2.0])
    reference = _series([1.0, 2.0])
    reference.index = reference.index + pd.Timedelta(days=10)

    with pytest.raises(ValueError, match="no overlap"):
        compare_timestamp_alignments(
            reference,
            source,
            source_offsets={"same_timestamp": "0h"},
            rounded_decimals=2,
        )


def test_all_offsets_use_identical_reference_timestamp_support() -> None:
    audit = compare_timestamp_alignments(
        _series([1.0, 2.0, 3.0, 4.0]),
        _series([1.0, 2.0, 3.0, 4.0]),
        source_offsets={"same_timestamp": "0h", "next_period_end": "1h"},
        rounded_decimals=2,
    )

    supports = audit.observations.groupby("offset_id")["reference_ts"].apply(set)
    assert supports["same_timestamp"] == supports["next_period_end"]
    assert audit.summary.set_index("offset_id")["common_count"].nunique() == 1


def test_start_labelled_sum_reconstruction_is_hand_calculable() -> None:
    fine = _series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    result = aggregate_fine_series(
        fine,
        coarse_frequency="4h",
        label_semantics="bar_start",
        reducer="sum",
    )

    expected = pd.Series(
        [10.0, 26.0],
        index=pd.DatetimeIndex([fine.index[0], fine.index[4]]).as_unit("ns"),
    )
    pd.testing.assert_series_equal(result, expected)


def test_end_labelled_last_reconstruction_wins_against_start_hypothesis() -> None:
    fine = pd.Series(
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
        index=pd.date_range("2026-01-01", periods=9, freq="1h", tz="UTC"),
    )
    coarse = pd.Series(
        [10.0, 50.0, 90.0],
        index=pd.date_range("2026-01-01", periods=3, freq="4h", tz="UTC"),
    )

    audit = compare_coarse_to_fine_aggregations(
        coarse,
        fine,
        coarse_frequency="4h",
        reducer="last",
        rounded_decimals=2,
    )

    summary = audit.summary.set_index("offset_id")
    assert summary.loc["bar_end", "rounded_match_rate"] == 1.0
    assert summary.loc["bar_end", "mean_absolute_error"] == 0.0
    assert summary.loc["bar_start", "mean_absolute_error"] > 0.0
    assert summary["common_count"].nunique() == 1


@pytest.mark.parametrize(
    ("fine", "coarse_frequency", "message"),
    [
        (
            pd.Series(
                [1.0, 2.0, 3.0],
                index=pd.DatetimeIndex(
                    [
                        "2026-01-01T00:00:00Z",
                        "2026-01-01T01:00:00Z",
                        "2026-01-01T03:00:00Z",
                    ]
                ),
            ),
            "4h",
            "regular cadence",
        ),
        (_series([1.0, 2.0, 3.0]), "90min", "integer multiple"),
    ],
)
def test_coarse_reconstruction_fails_closed_on_bad_frequency_contract(
    fine: pd.Series, coarse_frequency: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        aggregate_fine_series(
            fine,
            coarse_frequency=coarse_frequency,
            label_semantics="bar_start",
            reducer="sum",
        )


def test_declared_fine_frequency_allows_gaps_but_drops_incomplete_periods() -> None:
    fine = _series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).drop(
        pd.Timestamp("2026-01-01T05:00:00Z")
    )

    result = aggregate_fine_series(
        fine,
        fine_frequency="1h",
        coarse_frequency="4h",
        label_semantics="bar_start",
        reducer="sum",
    )

    expected = pd.Series(
        [10.0],
        index=pd.DatetimeIndex(["2026-01-01T00:00:00Z"]).as_unit("ns"),
    )
    pd.testing.assert_series_equal(result, expected)


def test_declared_fine_frequency_rejects_wrong_grid_phase() -> None:
    fine = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=pd.date_range(
            "2026-01-01T00:30:00Z", periods=4, freq="1h", tz="UTC"
        ),
    )

    with pytest.raises(ValueError, match="UTC fine-frequency grid"):
        aggregate_fine_series(
            fine,
            fine_frequency="1h",
            coarse_frequency="4h",
            label_semantics="bar_start",
            reducer="sum",
        )
