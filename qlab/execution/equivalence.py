"""Deterministic comparison helpers for execution-artifact evidence.

Lifecycle: formal qlab infrastructure.
Authority: qlab_research_private issue #16.
May be used for: comparing frozen reference and refactored execution artifacts.
Must not be used for: changing scientific definitions or tolerances.
"""

from __future__ import annotations

import hashlib

import pandas as pd


def canonical_frame_sha256(frame: pd.DataFrame) -> str:
    """Hash a tabular artifact with its schema, index, order, and values."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("execution equivalence requires pandas DataFrame artifacts")
    payload = frame.to_json(
        orient="table",
        date_format="iso",
        date_unit="ns",
        double_precision=15,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_frame_equivalent(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    *,
    artifact_name: str,
) -> str:
    """Assert exact artifact equality and return the common canonical digest."""
    if not isinstance(expected, pd.DataFrame) or not isinstance(actual, pd.DataFrame):
        raise TypeError("execution equivalence requires pandas DataFrame artifacts")
    try:
        pd.testing.assert_frame_equal(
            expected,
            actual,
            check_dtype=True,
            check_exact=True,
            check_like=False,
        )
    except AssertionError as error:
        raise AssertionError(f"scientific artifact differs: {artifact_name}") from error
    expected_digest = canonical_frame_sha256(expected)
    actual_digest = canonical_frame_sha256(actual)
    if expected_digest != actual_digest:  # pragma: no cover - defensive
        raise AssertionError(f"scientific artifact digest differs: {artifact_name}")
    return expected_digest
