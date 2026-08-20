import pandas as pd
import pytest

from qlab.execution.equivalence import assert_frame_equivalent, canonical_frame_sha256


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
