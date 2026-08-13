"""Auditable timestamp-alignment comparisons for numeric time series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd


OBSERVATION_COLUMNS = (
    "reference_ts",
    "source_ts",
    "offset_id",
    "reference_value",
    "source_value",
    "signed_error",
    "absolute_error",
    "rounded_match",
)


@dataclass(frozen=True)
class TimestampAlignmentAudit:
    """Row-level evidence and summary metrics for declared timestamp offsets."""

    observations: pd.DataFrame
    summary: pd.DataFrame


AggregationLabel = Literal["bar_start", "bar_end"]
AggregationReducer = Literal["sum", "first", "last", "min", "max", "mean"]


def _numeric_utc_series(series: pd.Series, *, name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must use a DatetimeIndex")
    if series.index.tz is None:
        raise ValueError(f"{name} timestamps must be timezone-aware")
    if series.index.has_duplicates:
        raise ValueError(f"{name} timestamps must be unique")
    result = pd.to_numeric(series, errors="raise").astype(float).copy()
    result.index = result.index.tz_convert("UTC")
    result = result.sort_index()
    if result.empty:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError(f"{name} values must be finite")
    return result


def summarize_timestamp_alignment_observations(
    observations: pd.DataFrame,
    *,
    group_by: Sequence[str] = ("offset_id",),
) -> pd.DataFrame:
    """Summarize comparison rows without assigning a semantic winner.

    ``source_ts`` records the source timestamp compared with ``reference_ts``.
    The caller chooses the grouping columns; this function only reports sample
    counts, rounded equality, and numeric errors.
    """

    missing = set(OBSERVATION_COLUMNS).difference(observations.columns)
    if missing:
        raise ValueError(
            "timestamp alignment observations missing columns: "
            + ", ".join(sorted(missing))
        )
    groups = tuple(group_by)
    if not groups or any(column not in observations.columns for column in groups):
        raise ValueError("group_by must name one or more observation columns")
    if observations.empty:
        raise ValueError("timestamp alignment observations must not be empty")

    rows: list[dict[str, object]] = []
    grouper: str | list[str] = groups[0] if len(groups) == 1 else list(groups)
    for keys, frame in observations.groupby(grouper, sort=True, dropna=False):
        key_values = (keys,) if len(groups) == 1 else tuple(keys)
        absolute = pd.to_numeric(frame["absolute_error"], errors="raise")
        signed = pd.to_numeric(frame["signed_error"], errors="raise")
        matches = frame["rounded_match"].astype(bool)
        row = dict(zip(groups, key_values, strict=True))
        row.update(
            {
                "common_count": int(len(frame)),
                "rounded_match_count": int(matches.sum()),
                "rounded_match_rate": float(matches.mean()),
                "mean_signed_error": float(signed.mean()),
                "mean_absolute_error": float(absolute.mean()),
                "median_absolute_error": float(absolute.median()),
                "q95_absolute_error": float(absolute.quantile(0.95)),
                "reference_start_ts": frame["reference_ts"].min(),
                "reference_end_ts": frame["reference_ts"].max(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def compare_timestamp_alignments(
    reference: pd.Series,
    source: pd.Series,
    *,
    source_offsets: Mapping[str, str | pd.Timedelta],
    rounded_decimals: int,
) -> TimestampAlignmentAudit:
    """Compare ``reference[t]`` with ``source[t + offset]``.

    Offsets are explicit and signed.  For example, ``-1h`` compares a
    reference value labelled ``t`` with the source value labelled ``t-1h``.
    No offset is interpreted as correct by this function.
    """

    if not source_offsets:
        raise ValueError("source_offsets must not be empty")
    if rounded_decimals < 0:
        raise ValueError("rounded_decimals must be non-negative")
    reference_values = _numeric_utc_series(reference, name="reference")
    source_values = _numeric_utc_series(source, name="source")

    chunks: list[pd.DataFrame] = []
    for offset_id, raw_offset in source_offsets.items():
        if not str(offset_id).strip():
            raise ValueError("offset ids must not be empty")
        offset = pd.Timedelta(raw_offset)
        shifted = source_values.copy()
        shifted.index = shifted.index - offset
        joined = pd.concat(
            [reference_values.rename("reference_value"), shifted.rename("source_value")],
            axis=1,
            join="inner",
        ).dropna()
        if joined.empty:
            continue
        signed_error = joined["reference_value"] - joined["source_value"]
        frame = joined.reset_index(names="reference_ts")
        frame["source_ts"] = frame["reference_ts"] + offset
        frame["offset_id"] = str(offset_id)
        frame["signed_error"] = signed_error.to_numpy()
        frame["absolute_error"] = signed_error.abs().to_numpy()
        frame["rounded_match"] = (
            frame["reference_value"].round(rounded_decimals)
            == frame["source_value"].round(rounded_decimals)
        )
        chunks.append(frame.loc[:, OBSERVATION_COLUMNS])
    if not chunks:
        raise ValueError("reference and source have no overlap at declared offsets")
    common_reference_ts = set(chunks[0]["reference_ts"])
    for frame in chunks[1:]:
        common_reference_ts.intersection_update(frame["reference_ts"])
    if not common_reference_ts:
        raise ValueError("declared offsets have no common reference-timestamp support")
    observations = pd.concat(
        [frame.loc[frame["reference_ts"].isin(common_reference_ts)] for frame in chunks],
        ignore_index=True,
    )
    summary = summarize_timestamp_alignment_observations(observations)
    return TimestampAlignmentAudit(observations=observations, summary=summary)


def aggregate_fine_series(
    fine: pd.Series,
    *,
    fine_frequency: str | pd.Timedelta | None = None,
    coarse_frequency: str | pd.Timedelta,
    label_semantics: AggregationLabel,
    reducer: AggregationReducer,
) -> pd.Series:
    """Reconstruct complete coarse periods from a regular fine series.

    ``bar_start`` assigns a fine observation at ``t`` to ``[t0, t0 + period)``
    and labels the result ``t0``. ``bar_end`` assigns it to ``(t0-period, t0]``
    and labels the result ``t0``. Incomplete periods are removed rather than
    silently compared with complete source bars.
    """

    values = _numeric_utc_series(fine, name="fine")
    if len(values) < 2:
        raise ValueError("fine must contain at least two observations")
    differences = values.index.to_series().diff().dropna()
    if fine_frequency is None:
        if differences.nunique() != 1:
            raise ValueError(
                "fine timestamps must have one regular cadence when fine_frequency is omitted"
            )
        fine_period = pd.Timedelta(differences.iloc[0])
    else:
        fine_period = pd.Timedelta(fine_frequency)
        if fine_period <= pd.Timedelta(0):
            raise ValueError("fine_frequency must be positive")
        if any(difference % fine_period != pd.Timedelta(0) for difference in differences):
            raise ValueError("fine timestamp gaps must be multiples of fine_frequency")
    coarse = pd.Timedelta(coarse_frequency)
    if fine_period <= pd.Timedelta(0) or coarse <= fine_period:
        raise ValueError("coarse_frequency must be larger than the fine cadence")
    if coarse % fine_period != pd.Timedelta(0):
        raise ValueError("coarse_frequency must be an integer multiple of fine cadence")
    if label_semantics not in {"bar_start", "bar_end"}:
        raise ValueError(f"unknown label_semantics: {label_semantics}")
    if reducer not in {"sum", "first", "last", "min", "max", "mean"}:
        raise ValueError(f"unknown reducer: {reducer}")

    expected_count = int(coarse / fine_period)
    epoch_ns = pd.Timestamp("1970-01-01", tz="UTC").value
    values_ns = values.index.as_unit("ns").asi8
    coarse_ns = coarse.value
    if ((values_ns - epoch_ns) % fine_period.value != 0).any():
        raise ValueError("fine timestamps must align to the declared UTC fine-frequency grid")
    if label_semantics == "bar_start":
        label_ns = ((values_ns - epoch_ns) // coarse_ns) * coarse_ns + epoch_ns
    else:
        relative = values_ns - epoch_ns
        label_ns = ((relative + coarse_ns - 1) // coarse_ns) * coarse_ns + epoch_ns
    labels = pd.to_datetime(label_ns, utc=True)
    grouped = values.groupby(labels)
    counts = grouped.count()
    aggregated = getattr(grouped, reducer)()
    aggregated = aggregated.loc[counts == expected_count]
    aggregated.index.name = values.index.name
    return aggregated.astype(float).sort_index()


def compare_coarse_to_fine_aggregations(
    coarse: pd.Series,
    fine: pd.Series,
    *,
    fine_frequency: str | pd.Timedelta | None = None,
    coarse_frequency: str | pd.Timedelta,
    reducer: AggregationReducer,
    rounded_decimals: int,
) -> TimestampAlignmentAudit:
    """Compare a coarse series with start- and end-labelled fine reconstructions."""

    coarse_values = _numeric_utc_series(coarse, name="coarse")
    chunks: list[pd.DataFrame] = []
    for semantics in ("bar_start", "bar_end"):
        reconstructed = aggregate_fine_series(
            fine,
            fine_frequency=fine_frequency,
            coarse_frequency=coarse_frequency,
            label_semantics=semantics,
            reducer=reducer,
        )
        audit = compare_timestamp_alignments(
            coarse_values,
            reconstructed,
            source_offsets={semantics: pd.Timedelta(0)},
            rounded_decimals=rounded_decimals,
        )
        chunks.append(audit.observations)
    common_reference_ts = set(chunks[0]["reference_ts"])
    for frame in chunks[1:]:
        common_reference_ts.intersection_update(frame["reference_ts"])
    if not common_reference_ts:
        raise ValueError("start/end reconstructions have no common coarse support")
    observations = pd.concat(
        [frame.loc[frame["reference_ts"].isin(common_reference_ts)] for frame in chunks],
        ignore_index=True,
    )
    return TimestampAlignmentAudit(
        observations=observations,
        summary=summarize_timestamp_alignment_observations(observations),
    )


__all__ = [
    "AggregationLabel",
    "AggregationReducer",
    "OBSERVATION_COLUMNS",
    "TimestampAlignmentAudit",
    "aggregate_fine_series",
    "compare_coarse_to_fine_aggregations",
    "compare_timestamp_alignments",
    "summarize_timestamp_alignment_observations",
]
