"""Schema and provenance validation for deployment candidate files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd


ColumnKind = Literal["string", "datetime", "float", "int", "bool"]


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: ColumnKind
    required: bool = True
    allow_null: bool = False


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    column: str | None = None


@dataclass(frozen=True)
class CandidateFileValidationResult:
    valid: bool
    issues: tuple[ValidationIssue, ...]
    row_count: int
    column_count: int


CURRENT_LIVE_CANDIDATE_SCHEMA = (
    ColumnSpec("candidate_id", "string"),
    ColumnSpec("signal_timeframe", "string"),
    ColumnSpec("family", "string"),
    ColumnSpec("source_name", "string"),
    ColumnSpec("transform_name", "string"),
    ColumnSpec("symbol", "string"),
    ColumnSpec("horizon", "string"),
    ColumnSpec("entry_rule", "string"),
)

CURRENT_LIVE_CANDIDATE_PROVENANCE = (
    ColumnSpec("generated_at", "datetime"),
    ColumnSpec("schema_version", "string"),
    ColumnSpec("generator_name", "string"),
    ColumnSpec("generator_commit", "string"),
    ColumnSpec("source_data_cutoff", "datetime"),
    ColumnSpec("selector_window_start", "datetime"),
    ColumnSpec("selector_window_end", "datetime"),
)


def _invalid_mask(series: pd.Series, kind: ColumnKind) -> pd.Series:
    if kind == "string":
        values = series.astype("string")
        return values.isna() | (values.str.strip() == "")
    if kind == "datetime":
        parsed = pd.to_datetime(series, utc=True, errors="coerce")
        return parsed.isna()
    if kind == "float":
        parsed = pd.to_numeric(series, errors="coerce")
        return parsed.isna()
    if kind == "int":
        parsed = pd.to_numeric(series, errors="coerce")
        return parsed.isna() | ((parsed % 1) != 0)
    if kind == "bool":
        normalized = series.astype("string").str.lower().str.strip()
        return ~normalized.isin({"true", "false", "1", "0", "yes", "no"})
    raise ValueError(f"unsupported column kind: {kind}")


def validate_candidate_frame(
    frame: pd.DataFrame,
    schema: tuple[ColumnSpec, ...] = CURRENT_LIVE_CANDIDATE_SCHEMA,
    provenance: tuple[ColumnSpec, ...] = CURRENT_LIVE_CANDIDATE_PROVENANCE,
    require_provenance: bool = True,
) -> CandidateFileValidationResult:
    issues: list[ValidationIssue] = []
    expected_columns = schema + (provenance if require_provenance else ())

    for spec in expected_columns:
        if spec.name not in frame.columns:
            if spec.required:
                issues.append(
                    ValidationIssue(
                        code="missing-column",
                        message=f"missing required column: {spec.name}",
                        column=spec.name,
                    )
                )
            continue

        series = frame[spec.name]
        if spec.allow_null:
            series = series[series.notna()]
        invalid_mask = _invalid_mask(series, spec.kind)
        if invalid_mask.any():
            issues.append(
                ValidationIssue(
                    code="invalid-column-values",
                    message=f"column {spec.name} contains invalid {spec.kind} values",
                    column=spec.name,
                )
            )

    if "candidate_id" in frame.columns and frame["candidate_id"].astype("string").duplicated().any():
        issues.append(
            ValidationIssue(
                code="duplicate-candidate-id",
                message="candidate_id must be unique",
                column="candidate_id",
            )
        )

    if {
        "selector_window_start",
        "selector_window_end",
    }.issubset(frame.columns):
        start = pd.to_datetime(
            frame["selector_window_start"], utc=True, errors="coerce")
        end = pd.to_datetime(
            frame["selector_window_end"], utc=True, errors="coerce")
        if ((start.notna()) & (end.notna()) & (end < start)).any():
            issues.append(
                ValidationIssue(
                    code="invalid-selector-window",
                    message="selector_window_end must be >= selector_window_start",
                    column="selector_window_end",
                )
            )

    return CandidateFileValidationResult(
        valid=not issues,
        issues=tuple(issues),
        row_count=len(frame),
        column_count=len(frame.columns),
    )


def read_candidate_csv(
    path: str | Path,
    **kwargs,
) -> CandidateFileValidationResult:
    frame = pd.read_csv(path)
    return validate_candidate_frame(frame, **kwargs)
