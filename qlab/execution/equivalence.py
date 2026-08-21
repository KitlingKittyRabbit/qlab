"""Deterministic comparison helpers for execution-artifact evidence.

Lifecycle: formal qlab infrastructure.
Authority: qlab_research_private issue #16.
May be used for: comparing frozen reference and refactored execution artifacts.
Must not be used for: changing scientific definitions or tolerances.
"""

from __future__ import annotations

import hashlib
import pickle

import numpy as np
import pandas as pd


_CANONICAL_FRAME_SCHEMA = b"qlab.execution.frame.v2"


def _length_prefixed(value: bytes) -> bytes:
    return len(value).to_bytes(8, byteorder="big") + value


def _canonical_attr(value: object) -> object:
    if isinstance(value, dict):
        items = [(_canonical_attr(key), _canonical_attr(item)) for key, item in value.items()]
        return (
            "dict",
            tuple(
                sorted(
                    items,
                    key=lambda item: pickle.dumps(item[0], protocol=5),
                )
            ),
        )
    if isinstance(value, list):
        return ("list", tuple(_canonical_attr(item) for item in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_canonical_attr(item) for item in value))
    if isinstance(value, set):
        items = [_canonical_attr(item) for item in value]
        return (
            "set",
            tuple(sorted(items, key=lambda item: pickle.dumps(item, protocol=5))),
        )
    return value


def canonical_frame_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a frame without lossy float formatting.

    The payload records axes and dtypes separately from each column's contiguous
    values. Protocol-5 pickle is used only for these typed, in-memory components;
    unlike JSON formatting, it preserves every float64 bit pattern and null value.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("execution equivalence requires pandas DataFrame artifacts")
    metadata = {
        "schema": _CANONICAL_FRAME_SCHEMA.decode("ascii"),
        "index": frame.index,
        "columns": frame.columns,
        "dtypes": tuple(frame.dtypes),
        "allows_duplicate_labels": frame.flags.allows_duplicate_labels,
        "attrs": _canonical_attr(frame.attrs),
    }
    parts = [_length_prefixed(_CANONICAL_FRAME_SCHEMA)]
    parts.append(_length_prefixed(pickle.dumps(metadata, protocol=5)))
    for _, series in frame.items():
        values = np.ascontiguousarray(series.to_numpy(copy=False))
        parts.append(_length_prefixed(pickle.dumps(values, protocol=5)))
    return b"".join(parts)


def canonical_frame_sha256(frame: pd.DataFrame) -> str:
    """Hash a tabular artifact with lossless schema, order, and value bytes."""
    return hashlib.sha256(canonical_frame_bytes(frame)).hexdigest()


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
