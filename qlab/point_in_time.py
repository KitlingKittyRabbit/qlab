"""Point-in-time data semantics and safety contracts.

These helpers are intentionally generic so that strategy code can reason
about when a value becomes usable, rather than assuming a timestamp index is
already tradable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


TimestampKind = Literal[
    "bar_start",
    "bar_end",
    "publication_time",
    "fetch_time",
    "unknown",
]
ValueStatus = Literal["final", "partial", "snapshot", "unknown"]
Reducer = Literal["sum", "mean", "last", "first", "max", "min"]


def _to_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


@dataclass(frozen=True)
class PointInTimeSemantics:
    """Describe when a timestamped value becomes safe to use.

    ``timestamp_kind`` says what the label means.
    ``value_status`` says whether the labeled value is already final or still
    partial at that label.
    """

    timestamp_kind: TimestampKind = "unknown"
    value_status: ValueStatus = "unknown"
    publication_lag: pd.Timedelta | None = None
    notes: str = ""

    def availability_contract_is_explicit(self) -> bool:
        if self.timestamp_kind == "unknown" or self.value_status == "unknown":
            return False
        if self.timestamp_kind in {"publication_time", "fetch_time"}:
            return True
        if self.value_status == "final":
            return True
        return self.publication_lag is not None

    def resolved_publication_lag(self) -> pd.Timedelta:
        return pd.Timedelta(0) if self.publication_lag is None else pd.Timedelta(self.publication_lag)

    def nominal_bar_end_time(
        self,
        label,
        bar_duration: pd.Timedelta,
    ) -> pd.Timestamp | None:
        ts = _to_timestamp(label)
        if self.timestamp_kind == "bar_start":
            return ts + pd.Timedelta(bar_duration)
        if self.timestamp_kind in {"bar_end", "publication_time", "fetch_time"}:
            return ts
        return None

    def earliest_availability_time(
        self,
        label,
        bar_duration: pd.Timedelta,
    ) -> pd.Timestamp | None:
        if not self.availability_contract_is_explicit():
            return None
        ts = _to_timestamp(label)
        if self.timestamp_kind in {"publication_time", "fetch_time"}:
            base = ts
        elif self.value_status in {"partial", "snapshot"}:
            base = ts
        else:
            base = self.nominal_bar_end_time(ts, bar_duration)
        if base is None:
            return None
        return base + self.resolved_publication_lag()

    def earliest_safe_decision_time(
        self,
        label,
        bar_duration: pd.Timedelta,
        decision_delay: pd.Timedelta = pd.Timedelta(0),
    ) -> pd.Timestamp | None:
        available = self.earliest_availability_time(label, bar_duration)
        if available is None:
            return None
        return available + pd.Timedelta(decision_delay)

    def is_observable_at(
        self,
        label,
        decision_time,
        bar_duration: pd.Timedelta,
    ) -> bool:
        earliest = self.earliest_availability_time(label, bar_duration)
        if earliest is None:
            return False
        return _to_timestamp(decision_time) >= earliest


@dataclass(frozen=True)
class AggregationAuditRow:
    label: pd.Timestamp
    aggregate_value: float
    start_value: float | None
    end_value: float | None
    start_abs_error: float | None
    end_abs_error: float | None
    start_match: bool
    end_match: bool


@dataclass(frozen=True)
class AggregationAuditResult:
    reducer: Reducer
    aggregate_duration: pd.Timedelta
    component_duration: pd.Timedelta
    total_rows: int
    usable_rows: int
    start_match_ratio: float
    end_match_ratio: float
    start_mean_abs_error: float
    end_mean_abs_error: float
    inferred_timestamp_kind: TimestampKind
    rows: tuple[AggregationAuditRow, ...]


@dataclass(frozen=True)
class TimingContractViolation:
    label: pd.Timestamp
    expected_time: pd.Timestamp
    observed_time: pd.Timestamp
    delta: pd.Timedelta
    reason: str


@dataclass(frozen=True)
class TimingContractResult:
    passed: bool
    semantics: PointInTimeSemantics
    bar_duration: pd.Timedelta
    decision_delay: pd.Timedelta
    violations: tuple[TimingContractViolation, ...]


def _reduce_window(values: pd.Series, reducer: Reducer) -> float | None:
    if values.empty:
        return None
    if reducer == "sum":
        return float(values.sum())
    if reducer == "mean":
        return float(values.mean())
    if reducer == "last":
        return float(values.iloc[-1])
    if reducer == "first":
        return float(values.iloc[0])
    if reducer == "max":
        return float(values.max())
    if reducer == "min":
        return float(values.min())
    raise ValueError(f"unsupported reducer: {reducer}")


def audit_aggregation_semantics(
    aggregate: pd.Series,
    component: pd.Series,
    aggregate_duration: pd.Timedelta,
    component_duration: pd.Timedelta,
    reducer: Reducer = "sum",
    atol: float = 1e-9,
    rtol: float = 1e-6,
) -> AggregationAuditResult:
    """Infer whether aggregate timestamps behave like bar-start or bar-end labels.

    The audit compares each aggregate row against two candidate windows:

    - start-labeled window: ``[t, t + aggregate_duration)``
    - end-labeled window: ``[t - aggregate_duration, t)``
    """

    aggregate = aggregate.dropna().sort_index().astype(float)
    component = component.dropna().sort_index().astype(float)
    if aggregate.empty or component.empty:
        return AggregationAuditResult(
            reducer=reducer,
            aggregate_duration=pd.Timedelta(aggregate_duration),
            component_duration=pd.Timedelta(component_duration),
            total_rows=0,
            usable_rows=0,
            start_match_ratio=0.0,
            end_match_ratio=0.0,
            start_mean_abs_error=np.nan,
            end_mean_abs_error=np.nan,
            inferred_timestamp_kind="unknown",
            rows=(),
        )

    agg_duration = pd.Timedelta(aggregate_duration)
    comp_duration = pd.Timedelta(component_duration)
    expected_components = max(1, int(round(agg_duration / comp_duration)))

    rows: list[AggregationAuditRow] = []
    for label, aggregate_value in aggregate.items():
        label_ts = _to_timestamp(label)
        start_window = component[(component.index >= label_ts) & (
            component.index < label_ts + agg_duration)]
        end_window = component[(
            component.index >= label_ts - agg_duration) & (component.index < label_ts)]

        if len(start_window) != expected_components and len(end_window) != expected_components:
            continue

        start_value = _reduce_window(start_window, reducer) if len(
            start_window) == expected_components else None
        end_value = _reduce_window(end_window, reducer) if len(
            end_window) == expected_components else None
        start_abs_error = None if start_value is None else abs(
            start_value - float(aggregate_value))
        end_abs_error = None if end_value is None else abs(
            end_value - float(aggregate_value))
        start_match = False if start_value is None else bool(
            np.isclose(start_value, float(
                aggregate_value), atol=atol, rtol=rtol)
        )
        end_match = False if end_value is None else bool(
            np.isclose(end_value, float(aggregate_value), atol=atol, rtol=rtol)
        )
        rows.append(
            AggregationAuditRow(
                label=label_ts,
                aggregate_value=float(aggregate_value),
                start_value=start_value,
                end_value=end_value,
                start_abs_error=start_abs_error,
                end_abs_error=end_abs_error,
                start_match=start_match,
                end_match=end_match,
            )
        )

    start_errors = [
        row.start_abs_error for row in rows if row.start_abs_error is not None]
    end_errors = [
        row.end_abs_error for row in rows if row.end_abs_error is not None]
    usable_rows = len(rows)
    start_match_ratio = 0.0 if usable_rows == 0 else sum(
        row.start_match for row in rows) / usable_rows
    end_match_ratio = 0.0 if usable_rows == 0 else sum(
        row.end_match for row in rows) / usable_rows
    start_mean_abs_error = float(
        np.mean(start_errors)) if start_errors else np.nan
    end_mean_abs_error = float(np.mean(end_errors)) if end_errors else np.nan

    inferred = "unknown"
    if usable_rows > 0:
        if start_match_ratio > end_match_ratio:
            inferred = "bar_start"
        elif end_match_ratio > start_match_ratio:
            inferred = "bar_end"
        elif np.isfinite(start_mean_abs_error) and np.isfinite(end_mean_abs_error):
            if start_mean_abs_error < end_mean_abs_error:
                inferred = "bar_start"
            elif end_mean_abs_error < start_mean_abs_error:
                inferred = "bar_end"

    return AggregationAuditResult(
        reducer=reducer,
        aggregate_duration=agg_duration,
        component_duration=comp_duration,
        total_rows=len(aggregate),
        usable_rows=usable_rows,
        start_match_ratio=float(start_match_ratio),
        end_match_ratio=float(end_match_ratio),
        start_mean_abs_error=start_mean_abs_error,
        end_mean_abs_error=end_mean_abs_error,
        inferred_timestamp_kind=inferred,
        rows=tuple(rows),
    )


def validate_entry_timing_contract(
    signal_labels: pd.Index | pd.Series,
    observed_entry_times: pd.Index | pd.Series,
    semantics: PointInTimeSemantics,
    bar_duration: pd.Timedelta,
    decision_delay: pd.Timedelta = pd.Timedelta(0),
    tolerance: pd.Timedelta = pd.Timedelta(0),
) -> TimingContractResult:
    labels = pd.Index(signal_labels)
    observed = pd.Index(observed_entry_times)
    if len(labels) != len(observed):
        raise ValueError(
            "signal_labels and observed_entry_times must have the same length")

    violations: list[TimingContractViolation] = []
    tolerance = pd.Timedelta(tolerance)
    for label, observed_entry in zip(labels, observed):
        expected = semantics.earliest_safe_decision_time(
            label, bar_duration, decision_delay)
        if expected is None:
            violations.append(
                TimingContractViolation(
                    label=_to_timestamp(label),
                    expected_time=_to_timestamp(label),
                    observed_time=_to_timestamp(observed_entry),
                    delta=pd.Timedelta(0),
                    reason="semantics do not define an explicit availability time",
                )
            )
            continue
        observed_ts = _to_timestamp(observed_entry)
        delta = observed_ts - expected
        if abs(delta) > tolerance:
            violations.append(
                TimingContractViolation(
                    label=_to_timestamp(label),
                    expected_time=expected,
                    observed_time=observed_ts,
                    delta=delta,
                    reason="observed entry time does not match point-in-time safe decision time",
                )
            )

    return TimingContractResult(
        passed=not violations,
        semantics=semantics,
        bar_duration=pd.Timedelta(bar_duration),
        decision_delay=pd.Timedelta(decision_delay),
        violations=tuple(violations),
    )
