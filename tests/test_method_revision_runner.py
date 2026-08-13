from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def _load_runner():
    root = Path(__file__).resolve().parents[2]
    path = (
        root
        / "qlab_research_private/research/crypto/ksv4_method_revision.py"
    )
    spec = importlib.util.spec_from_file_location("ksv4_method_revision_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_revision_runner_stage_tasks_and_specs_match_frozen_design():
    runner = _load_runner()
    e0 = runner._tasks("e0_diagnostic")
    development = runner._tasks("e1_e2_development")
    confirmation = runner._tasks("fresh_confirmation", selected_engine="E1")
    assert len(e0) == 5_000
    assert len(development) == 8_500
    assert len(confirmation) == 24_000

    e0_specs = runner._specifications(
        "e0_diagnostic", e0[0], selected_engine=None
    )
    assert [row["dependence_length"] for row in e0_specs] == [1, 7, 14, 28]
    assert {row["n_bootstrap"] for row in e0_specs} == {999}

    development_specs = runner._specifications(
        "e1_e2_development", development[0], selected_engine=None
    )
    assert [row["engine"] for row in development_specs] == ["E1", "E2"]
    assert {row["n_bootstrap"] for row in development_specs} == {999}

    first_confirmation = runner._specifications(
        "fresh_confirmation", confirmation[0], selected_engine="E1"
    )
    late_confirmation = runner._specifications(
        "fresh_confirmation", confirmation[100], selected_engine="E1"
    )
    assert [row["n_bootstrap"] for row in first_confirmation] == [1_999, 10_000]
    assert [row["n_bootstrap"] for row in late_confirmation] == [1_999]


def test_revision_runtime_identity_binds_generator_and_post_audit_sources():
    runner = _load_runner()
    identity = runner._runtime_identity("e0_diagnostic")
    assert len(identity["layer_a_generator_source_sha256"]) == 64
    assert len(identity["frozen_preimplementation_generator_file_sha256"]) == 64
    assert set(identity["runtime_source_sha256"]) == {
        "qlab/qlab/method_simulation.py",
        "qlab/qlab/research_stats.py",
        "qlab/qlab/coinglass_substitution.py",
        "qlab_research_private/research/crypto/ksv4_method_revision.py",
    }


def test_revision_runner_fails_closed_on_stage_skip_and_overlap(tmp_path, monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(runner, "RESULT_ROOT", tmp_path)
    monkeypatch.setattr(
        runner, "_ensure_runtime_manifest", lambda stage: ({}, f"{stage}-runtime")
    )
    with pytest.raises(RuntimeError, match="E0 diagnosis"):
        runner._require_preceding_stage("e1_e2_development", None)

    e0 = tmp_path / "e0_diagnostic"
    e0.mkdir()
    (e0 / "stage_receipt.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "stage": "e0_diagnostic",
                "runtime_manifest_sha256": "e0_diagnostic-runtime",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner, "_completed_e0_runtime_sha", lambda: "e0_diagnostic-runtime"
    )
    runner._require_preceding_stage("e1_e2_development", None)
    with pytest.raises(RuntimeError, match="development must complete"):
        runner._require_preceding_stage("fresh_confirmation", "E1")

    batches = e0 / "batches/batch_00000_00009"
    batches.mkdir(parents=True)
    (batches / "receipt.json").write_text(
        json.dumps({"batch_start": 0, "batch_stop": 10}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="overlaps existing"):
        runner._reject_overlapping_batches(
            "e0_diagnostic", 5, 15, e0 / "batches/batch_00005_00014"
        )


def test_revision_runner_requires_single_task_resource_measurement_first(
    tmp_path, monkeypatch
):
    runner = _load_runner()
    monkeypatch.setattr(runner, "RESULT_ROOT", tmp_path)
    runner._require_resource_preflight(
        "e0_diagnostic",
        batch_start=0,
        batch_stop=1,
        process_concurrency=1,
        runtime_sha="runtime",
    )
    with pytest.raises(RuntimeError, match="must first run task 0 alone"):
        runner._require_resource_preflight(
            "e0_diagnostic",
            batch_start=1,
            batch_stop=2,
            process_concurrency=1,
            runtime_sha="runtime",
        )


def test_revision_resource_preflight_records_cpu_time(tmp_path, monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(runner, "RESULT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "_require_preceding_stage", lambda *args: None)
    monkeypatch.setattr(
        runner, "_ensure_runtime_manifest", lambda stage: ({}, "runtime")
    )
    monkeypatch.setattr(
        runner, "_require_startup_audit", lambda *args, **kwargs: None
    )
    batch = tmp_path / "e1_e2_development/batches/batch_00000_00000"
    batch.mkdir(parents=True)
    (batch / "receipt.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "runtime_manifest_sha256": "runtime",
                "selected_engine": None,
                "task_count": 1,
                "process_concurrency": 1,
                "maximum_worker_peak_rss_bytes": 1024,
                "cpu_seconds": 0.75,
                "wall_seconds": 1.25,
                "output_bytes": 128,
            }
        ),
        encoding="utf-8",
    )
    preflight = runner.build_resource_preflight("e1_e2_development")
    assert preflight["measured_cpu_seconds"] == 0.75
    assert preflight["measured_wall_seconds"] == 1.25
    assert preflight["measured_peak_rss_bytes"] == 1024


def test_revision_runner_rejects_invalid_startup_audit(tmp_path, monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(runner, "RESULT_ROOT", tmp_path)
    root = tmp_path / "e0_diagnostic"
    root.mkdir()
    with pytest.raises(RuntimeError, match="missing independent startup audit"):
        runner._require_startup_audit(
            "e0_diagnostic", "runtime", operation="run_batch",
            batch_start=0, batch_stop=1, process_concurrency=1,
        )
    (root / "startup_audit.json").write_text(
        json.dumps(
            {
                "decision": "BLOCK",
                "stage": "e0_diagnostic",
                "runtime_manifest_sha256": "runtime",
                "authorized_operations": ["run_batch"],
                "authorized_batch_start": 0,
                "authorized_batch_stop": 1,
                "maximum_process_concurrency": 1,
                "allow_validate": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="invalid independent startup audit"):
        runner._require_startup_audit(
            "e0_diagnostic", "runtime", operation="run_batch",
            batch_start=0, batch_stop=1, process_concurrency=1,
        )


def test_revision_runner_enforces_startup_audit_scope(tmp_path, monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(runner, "RESULT_ROOT", tmp_path)
    root = tmp_path / "e1_e2_development"
    root.mkdir()
    (root / "startup_audit.json").write_text(
        json.dumps(
            {
                "decision": "ALLOW",
                "stage": "e1_e2_development",
                "runtime_manifest_sha256": "runtime",
                "authorized_operations": ["run_batch", "validate"],
                "authorized_batch_start": 1,
                "authorized_batch_stop": 101,
                "maximum_process_concurrency": 4,
                "allow_validate": True,
            }
        ),
        encoding="utf-8",
    )
    runner._require_startup_audit(
        "e1_e2_development", "runtime", operation="run_batch",
        batch_start=1, batch_stop=51, process_concurrency=4,
    )
    runner._require_startup_audit(
        "e1_e2_development", "runtime", operation="validate"
    )
    with pytest.raises(RuntimeError, match="does not authorize"):
        runner._require_startup_audit(
            "e1_e2_development", "runtime", operation="run_batch",
            batch_start=0, batch_stop=51, process_concurrency=4,
        )
    with pytest.raises(RuntimeError, match="does not authorize"):
        runner._require_startup_audit(
            "e1_e2_development", "runtime", operation="run_batch",
            batch_start=1, batch_stop=51, process_concurrency=5,
        )


def test_e0_runner_writes_task_lineage_and_diagnostic_not_selection(
    tmp_path, monkeypatch
):
    runner = _load_runner()
    monkeypatch.setattr(runner, "RESULT_ROOT", tmp_path)
    task = {
        "diagnostic_task_idx": 0,
        "diagnostic_task_id": "e0-task-0",
        "scenario_id": "A01",
        "replicate": 0,
        "dataset_seed": 1,
        "main_inference_seed": 2,
        "analysis_specification": "A01__right_tail_primary",
        "alternative": "greater",
    }
    monkeypatch.setattr(runner, "_tasks", lambda *args, **kwargs: (task,))
    monkeypatch.setattr(
        runner, "_ensure_runtime_manifest", lambda stage: ({}, "runtime-sha")
    )
    monkeypatch.setattr(
        runner, "_require_startup_audit", lambda *args, **kwargs: None
    )

    results = pd.DataFrame(
        {
            "registered_task_idx": [0],
            "registered_task_id": ["e0-task-0"],
            "joint_inference_engine": ["E0"],
            "scenario_id": ["A01"],
            "analysis_specification": ["A01__right_tail_primary__E0_1d_999"],
            "replicate": [0],
            "hypothesis_id": ["H01"],
            "raw_one_sided_p_value": [0.5],
            "stepdown_max_t_adjusted_p_value": [0.8],
            "inference_variant": ["E0_1d_999"],
        }
    )
    maxima = pd.DataFrame(
        {
            "registered_task_idx": [0],
            "registered_task_id": ["e0-task-0"],
            "scenario_id": ["A01"],
            "analysis_specification": ["A01__right_tail_primary__E0_1d_999"],
            "replicate": [0],
            "bootstrap_idx": [0],
            "bootstrap_max_test_statistic": [1.0],
        }
    )
    task_receipt = {
        "registered_task_idx": 0,
        "registered_task_id": "e0-task-0",
        "status": "complete",
        "prior_design_sha256": "prior",
        "revision_design_sha256": "revision",
        "task_input_sha256": "input",
        "seeds_json": '{"dataset_seed":1,"main_inference_seed":2}',
        "results_sha256": "result",
        "bootstrap_max_statistics_sha256": "bootstrap",
        "worker_peak_rss_bytes": 1024,
        "worker_cpu_seconds": 0.5,
    }
    output = {
        "results": results,
        "bootstrap_max_statistics": maxima,
        "task_receipt": task_receipt,
    }

    class SerialExecutor:
        def __init__(self, **kwargs):
            del kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            del args

        def map(self, function, payloads):
            del function, payloads
            return [output]

    monkeypatch.setattr(runner, "ProcessPoolExecutor", SerialExecutor)
    receipt = runner.run_batch(
        stage="e0_diagnostic",
        batch_start=0,
        batch_stop=1,
        process_concurrency=1,
    )
    assert receipt["task_receipts_sha256"]
    assert receipt["bootstrap_max_statistics_sha256"]
    assert receipt["maximum_worker_peak_rss_bytes"] == 1024
    assert receipt["cpu_seconds"] == 0.5
    _, stage_receipt = runner.validate_stage("e0_diagnostic")
    assert stage_receipt["status"] == "complete"
    summary = tmp_path / "e0_diagnostic/summary"
    assert (summary / "diagnostic.json").exists()
    assert not (summary / "decision.json").exists()
