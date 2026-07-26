from __future__ import annotations

import hashlib

import pandas as pd

from qlab.point_in_time import PointInTimeSemantics
from qlab.research_gate import (
    FactorContract,
    FormalRerunPrerequisites,
    GateArtifact,
    ReliabilityReview,
    collect_formal_rerun_artifacts,
    evaluate_formal_rerun_prerequisites,
)


def test_formal_rerun_prerequisites_pass_with_explicit_semantics(tmp_path):
    data_root = tmp_path / "data_root"
    data_root.mkdir()
    previous_archive = tmp_path / "archive" / "v1"
    previous_archive.mkdir(parents=True)
    support_entrypoint = tmp_path / "preflight.py"
    support_entrypoint.write_text("print('ok')\n", encoding="utf-8")
    report_root = tmp_path / "reports" / "candidate" / "v2"
    report_root.mkdir(parents=True)
    result_root = tmp_path / "results" / "candidate" / "v2"
    result_root.mkdir(parents=True)

    governance_doc = tmp_path / "governance.md"
    governance_doc.write_text("governance\n", encoding="utf-8")
    design_doc = tmp_path / "design.md"
    design_doc.write_text("design\n", encoding="utf-8")

    prerequisites = FormalRerunPrerequisites(
        route_name="coinglass_v2_reaudit_rebuild",
        version="v2_reaudit_rebuild",
        lifecycle_state="candidate",
        canonical_data_root=str(data_root),
        previous_version_archives=(str(previous_archive),),
        active_support_entrypoints=(str(support_entrypoint),),
        required_artifacts=(
            GateArtifact(
                label="governance",
                path=str(governance_doc),
                expected_sha256=hashlib.sha256(
                    governance_doc.read_bytes()).hexdigest(),
            ),
            GateArtifact(label="design", path=str(design_doc)),
        ),
        factor_contracts=(
            FactorContract(
                label="coinglass_12h_aggregates",
                signal_timeframe="12h",
                bar_duration=pd.Timedelta(hours=12),
                semantics=PointInTimeSemantics(
                    timestamp_kind="bar_start",
                    value_status="final",
                ),
                evidence_paths=(str(governance_doc), str(design_doc)),
            ),
        ),
        program_semantics_review=ReliabilityReview(
            label="program_semantics",
            verdict="pass",
            evidence_paths=(str(design_doc),),
        ),
        research_design_review=ReliabilityReview(
            label="research_design",
            verdict="pass",
            evidence_paths=(str(governance_doc), str(design_doc)),
        ),
        candidate_report_root=str(report_root),
        candidate_result_root=str(result_root),
    )

    result = evaluate_formal_rerun_prerequisites(prerequisites)
    artifacts = collect_formal_rerun_artifacts(prerequisites)

    assert result.passed is True
    assert result.issues == ()
    assert "governance" in artifacts
    assert "entrypoint_1" in artifacts


def test_formal_rerun_prerequisites_fail_closed_for_missing_reviews_and_implicit_semantics(tmp_path):
    missing_doc = tmp_path / "missing.md"

    prerequisites = FormalRerunPrerequisites(
        route_name="",
        version="",
        lifecycle_state="active",
        canonical_data_root=str(tmp_path / "missing_data_root"),
        previous_version_archives=(),
        active_support_entrypoints=(),
        required_artifacts=(
            GateArtifact(label="missing_doc", path=str(missing_doc)),
        ),
        factor_contracts=(
            FactorContract(
                label="coinglass_12h_aggregates",
                signal_timeframe="12h",
                bar_duration=pd.Timedelta(hours=0),
                semantics=PointInTimeSemantics(
                    timestamp_kind="unknown",
                    value_status="unknown",
                ),
                evidence_paths=(),
            ),
        ),
        program_semantics_review=ReliabilityReview(
            label="program_semantics",
            verdict="pending",
            evidence_paths=(),
        ),
        research_design_review=ReliabilityReview(
            label="research_design",
            verdict="fail",
            evidence_paths=(),
        ),
        candidate_report_root=str(tmp_path / "missing_reports"),
        candidate_result_root=str(tmp_path / "missing_results"),
    )

    result = evaluate_formal_rerun_prerequisites(prerequisites)

    assert result.passed is False
    issue_codes = {issue.code for issue in result.issues}
    assert "missing-route-name" in issue_codes
    assert "missing-version" in issue_codes
    assert "unexpected-lifecycle-state" in issue_codes
    assert "missing-data-root" in issue_codes
    assert "missing-previous-version-archives" in issue_codes
    assert "missing-active-support-entrypoints" in issue_codes
    assert "missing-required-artifact" in issue_codes
    assert "invalid-factor-bar-duration" in issue_codes
    assert "implicit-factor-availability" in issue_codes
    assert "missing-factor-evidence" in issue_codes
    assert "program-semantics-review-not-passed" in issue_codes
    assert "research-design-review-not-passed" in issue_codes


def test_formal_rerun_prerequisites_reject_hash_mismatch(tmp_path):
    data_root = tmp_path / "data_root"
    data_root.mkdir()
    previous_archive = tmp_path / "archive" / "v1"
    previous_archive.mkdir(parents=True)
    support_entrypoint = tmp_path / "preflight.py"
    support_entrypoint.write_text("print('ok')\n", encoding="utf-8")
    report_root = tmp_path / "reports" / "candidate" / "v2"
    report_root.mkdir(parents=True)
    result_root = tmp_path / "results" / "candidate" / "v2"
    result_root.mkdir(parents=True)

    governance_doc = tmp_path / "governance.md"
    governance_doc.write_text("governance\n", encoding="utf-8")
    design_doc = tmp_path / "design.md"
    design_doc.write_text("design\n", encoding="utf-8")

    prerequisites = FormalRerunPrerequisites(
        route_name="coinglass_v2_reaudit_rebuild",
        version="v2_reaudit_rebuild",
        lifecycle_state="candidate",
        canonical_data_root=str(data_root),
        previous_version_archives=(str(previous_archive),),
        active_support_entrypoints=(str(support_entrypoint),),
        required_artifacts=(
            GateArtifact(
                label="governance",
                path=str(governance_doc),
                expected_sha256="0" * 64,
            ),
            GateArtifact(label="design", path=str(design_doc)),
        ),
        factor_contracts=(
            FactorContract(
                label="coinglass_12h_aggregates",
                signal_timeframe="12h",
                bar_duration=pd.Timedelta(hours=12),
                semantics=PointInTimeSemantics(
                    timestamp_kind="bar_start",
                    value_status="final",
                ),
                evidence_paths=(str(governance_doc), str(design_doc)),
            ),
        ),
        program_semantics_review=ReliabilityReview(
            label="program_semantics",
            verdict="pass",
            evidence_paths=(str(design_doc),),
        ),
        research_design_review=ReliabilityReview(
            label="research_design",
            verdict="pass",
            evidence_paths=(str(governance_doc), str(design_doc)),
        ),
        candidate_report_root=str(report_root),
        candidate_result_root=str(result_root),
    )

    result = evaluate_formal_rerun_prerequisites(prerequisites)

    assert result.passed is False
    issue_codes = {issue.code for issue in result.issues}
    assert "unexpected-artifact-hash" in issue_codes
