import hashlib
import json

import pytest

from qlab.execution.identity import (
    canonical_dict_sha256,
    scientific_identity,
    task_identity,
)


def test_task_identity_is_stable_and_field_order_independent():
    first = task_identity(
        schema="l5_5_inner_task_v1",
        hypothesis_id="hypothesis-1",
        fold_idx=3,
        model_class="hist_gbm",
        config_key="k",
        split_idx=1,
    )
    second = task_identity(
        split_idx=1,
        config_key="k",
        model_class="hist_gbm",
        fold_idx=3,
        hypothesis_id="hypothesis-1",
        schema="l5_5_inner_task_v1",
    )
    assert first == second
    assert len(first) == 64
    assert first == hashlib.sha256(
        json.dumps(
            {
                "identity_schema_version": "l5_5_inner_task_v1",
                "hypothesis_id": "hypothesis-1",
                "fold_idx": 3,
                "model_class": "hist_gbm",
                "config_key": "k",
                "split_idx": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def test_task_identity_never_depends_on_execution_environment():
    base = task_identity(
        schema="unit", hypothesis_id="h", fold_idx=0, model_class="c", config_key="k", split_idx=0
    )
    with_worker = task_identity(
        schema="unit",
        hypothesis_id="h",
        fold_idx=0,
        model_class="c",
        config_key="k",
        split_idx=0,
        worker_number=7,
    )
    assert base != with_worker
    assert task_identity(
        schema="unit",
        hypothesis_id="h",
        fold_idx=0,
        model_class="c",
        config_key="k",
        split_idx=0,
        worker_count=96,
    ) != base


def test_task_identity_changes_with_any_input():
    base = task_identity(schema="unit", hypothesis_id="h", fold_idx=0, config_key="k")
    for field in ("hypothesis_id", "fold_idx", "config_key"):
        variant = dict(hypothesis_id="h", fold_idx=0, config_key="k")
        variant[field] = "changed" if field == "hypothesis_id" else -1
        assert task_identity(schema="unit", **variant) != base


def test_scientific_identity_and_canonical_dict():
    first = scientific_identity(schema="sci_v1", alpha=1e-4, family="ridge")
    second = scientific_identity(schema="sci_v1", family="ridge", alpha=1e-4)
    assert first == second
    mapping = {"alpha": 1e-4, "family": "ridge"}
    assert canonical_dict_sha256(mapping, schema="sci_v1") == first
    assert canonical_dict_sha256(dict(reversed(mapping.items())), schema="sci_v1") == first
    with pytest.raises(TypeError):
        canonical_dict_sha256(123, schema="sci_v1")