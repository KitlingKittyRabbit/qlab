import numpy as np
import pandas as pd
import pytest

from qlab.execution.equivalence import (
    assert_frame_equivalent,
    canonical_frame_sha256,
)


def test_equivalence_digest_covers_schema_order_and_values():
    first = pd.DataFrame({"fold_idx": [0, 1], "score": [1.0, 2.0]})
    second = first.copy()
    assert canonical_frame_sha256(first) == canonical_frame_sha256(second)
    assert len(assert_frame_equivalent(first, second, artifact_name="inner_scores")) == 64
    with pytest.raises(AssertionError, match="inner_scores"):
        assert_frame_equivalent(
            first,
            pd.DataFrame({"fold_idx": [0, 1], "score": [1.0, 2.1]}),
            artifact_name="inner_scores",
        )


def test_equivalence_digest_preserves_adjacent_float64_values():
    first = pd.DataFrame({"score": [1.0]})
    adjacent = pd.DataFrame({"score": [np.nextafter(1.0, 2.0)]})
    assert canonical_frame_sha256(first) != canonical_frame_sha256(adjacent)
    with pytest.raises(AssertionError, match="float64"):
        assert_frame_equivalent(first, adjacent, artifact_name="float64_values")


def test_equivalence_digest_preserves_dtype_and_signed_zero():
    float32 = pd.DataFrame({"score": np.array([1.0], dtype=np.float32)})
    float64 = pd.DataFrame({"score": np.array([1.0], dtype=np.float64)})
    positive_zero = pd.DataFrame({"score": [0.0]})
    negative_zero = pd.DataFrame({"score": [-0.0]})
    assert canonical_frame_sha256(float32) != canonical_frame_sha256(float64)
    assert canonical_frame_sha256(positive_zero) != canonical_frame_sha256(negative_zero)
