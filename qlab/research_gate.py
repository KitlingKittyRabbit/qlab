"""Formal research rerun prerequisite gates.

These helpers are intentionally reusable and fail closed. They answer a narrow
question: is a versioned research route ready to begin a formal rerun under the
current reliability rules?
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from .point_in_time import PointInTimeSemantics


LifecycleState = Literal[
    "candidate",
    "active",
    "superseded",
    "archived",
    "invalidated",
    "quarantined",
]
ArtifactKind = Literal["file", "directory"]
ReviewVerdict = Literal["pass", "fail", "pending"]


@dataclass(frozen=True)
class GateArtifact:
    label: str
    path: str
    kind: ArtifactKind = "file"
    expected_sha256: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class FactorContract:
    label: str
    signal_timeframe: str
    bar_duration: pd.Timedelta
    semantics: PointInTimeSemantics
    evidence_paths: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class ReliabilityReview:
    label: str
    verdict: ReviewVerdict
    evidence_paths: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class FormalRerunPrerequisites:
    route_name: str
    version: str
    lifecycle_state: LifecycleState
    canonical_data_root: str
    previous_version_archives: tuple[str, ...]
    active_support_entrypoints: tuple[str, ...]
    required_artifacts: tuple[GateArtifact, ...]
    factor_contracts: tuple[FactorContract, ...]
    program_semantics_review: ReliabilityReview
    research_design_review: ReliabilityReview
    candidate_report_root: str | None = None
    candidate_result_root: str | None = None


@dataclass(frozen=True)
class GateIssue:
    code: str
    message: str
    path: str | None = None


@dataclass(frozen=True)
class FormalRerunGateResult:
    passed: bool
    issues: tuple[GateIssue, ...]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "issues": [asdict(issue) for issue in self.issues],
        }


def _resolve_path(raw_path: str) -> Path:
    return Path(raw_path).expanduser().resolve()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_exists(raw_path: str, kind: ArtifactKind) -> bool:
    path = _resolve_path(raw_path)
    if kind == "file":
        return path.is_file()
    if kind == "directory":
        return path.is_dir()
    raise ValueError(f"unsupported artifact kind: {kind}")


def _append_missing_path_issue(
    issues: list[GateIssue],
    *,
    code: str,
    message: str,
    raw_path: str,
    kind: ArtifactKind,
) -> None:
    if not _path_exists(raw_path, kind):
        issues.append(
            GateIssue(
                code=code,
                message=message,
                path=str(_resolve_path(raw_path)),
            )
        )


def collect_formal_rerun_artifacts(
    prerequisites: FormalRerunPrerequisites,
) -> dict[str, str]:
    """Collect distinct artifact paths that support the rerun gate."""

    artifacts: dict[str, str] = {}
    seen_paths: set[str] = set()

    def add(label: str, raw_path: str) -> None:
        resolved = str(_resolve_path(raw_path))
        if resolved in seen_paths:
            return
        artifacts[label] = resolved
        seen_paths.add(resolved)

    for artifact in prerequisites.required_artifacts:
        add(artifact.label, artifact.path)
    for index, raw_path in enumerate(prerequisites.previous_version_archives, start=1):
        add(f"archive_{index}", raw_path)
    for index, raw_path in enumerate(prerequisites.active_support_entrypoints, start=1):
        add(f"entrypoint_{index}", raw_path)
    for factor in prerequisites.factor_contracts:
        for index, raw_path in enumerate(factor.evidence_paths, start=1):
            add(f"factor_{factor.label}_{index}", raw_path)
    for review in (
        prerequisites.program_semantics_review,
        prerequisites.research_design_review,
    ):
        for index, raw_path in enumerate(review.evidence_paths, start=1):
            add(f"review_{review.label}_{index}", raw_path)
    return artifacts


def evaluate_formal_rerun_prerequisites(
    prerequisites: FormalRerunPrerequisites,
) -> FormalRerunGateResult:
    issues: list[GateIssue] = []

    if not prerequisites.route_name.strip():
        issues.append(GateIssue(code="missing-route-name",
                      message="route_name must be non-empty"))
    if not prerequisites.version.strip():
        issues.append(GateIssue(code="missing-version",
                      message="version must be non-empty"))
    if prerequisites.lifecycle_state != "candidate":
        issues.append(
            GateIssue(
                code="unexpected-lifecycle-state",
                message="formal rerun prerequisites must be evaluated on a candidate route",
            )
        )

    _append_missing_path_issue(
        issues,
        code="missing-data-root",
        message="canonical_data_root must exist before a formal rerun starts",
        raw_path=prerequisites.canonical_data_root,
        kind="directory",
    )

    if not prerequisites.previous_version_archives:
        issues.append(
            GateIssue(
                code="missing-previous-version-archives",
                message="at least one archived previous-version bundle is required",
            )
        )
    for raw_path in prerequisites.previous_version_archives:
        _append_missing_path_issue(
            issues,
            code="missing-previous-version-archive",
            message="a declared previous-version archive is missing",
            raw_path=raw_path,
            kind="directory",
        )

    if not prerequisites.active_support_entrypoints:
        issues.append(
            GateIssue(
                code="missing-active-support-entrypoints",
                message="formal rerun prerequisites require active support entrypoints",
            )
        )
    for raw_path in prerequisites.active_support_entrypoints:
        _append_missing_path_issue(
            issues,
            code="missing-active-support-entrypoint",
            message="an active support entrypoint is missing",
            raw_path=raw_path,
            kind="file",
        )

    for artifact in prerequisites.required_artifacts:
        _append_missing_path_issue(
            issues,
            code="missing-required-artifact",
            message=f"required artifact is missing: {artifact.label}",
            raw_path=artifact.path,
            kind=artifact.kind,
        )
        if artifact.expected_sha256 is not None and _path_exists(artifact.path, artifact.kind):
            if artifact.kind != "file":
                issues.append(
                    GateIssue(
                        code="invalid-hash-pinned-artifact",
                        message=f"artifact {artifact.label} pins a sha256 but is not a file",
                        path=str(_resolve_path(artifact.path)),
                    )
                )
            else:
                actual_hash = _hash_file(_resolve_path(artifact.path))
                if actual_hash.lower() != artifact.expected_sha256.lower():
                    issues.append(
                        GateIssue(
                            code="unexpected-artifact-hash",
                            message=(
                                f"artifact {artifact.label} does not match the pinned sha256 "
                                f"for this rerun route"
                            ),
                            path=str(_resolve_path(artifact.path)),
                        )
                    )

    if prerequisites.candidate_report_root is not None:
        _append_missing_path_issue(
            issues,
            code="missing-candidate-report-root",
            message="candidate report root must exist before formal rerun",
            raw_path=prerequisites.candidate_report_root,
            kind="directory",
        )
    if prerequisites.candidate_result_root is not None:
        _append_missing_path_issue(
            issues,
            code="missing-candidate-result-root",
            message="candidate result root must exist before formal rerun",
            raw_path=prerequisites.candidate_result_root,
            kind="directory",
        )

    if not prerequisites.factor_contracts:
        issues.append(
            GateIssue(
                code="missing-factor-contracts",
                message="factor_contracts must be declared before a formal rerun starts",
            )
        )
    for factor in prerequisites.factor_contracts:
        if pd.Timedelta(factor.bar_duration) <= pd.Timedelta(0):
            issues.append(
                GateIssue(
                    code="invalid-factor-bar-duration",
                    message=f"factor {factor.label} must declare a positive bar_duration",
                )
            )
        if not factor.semantics.availability_contract_is_explicit():
            issues.append(
                GateIssue(
                    code="implicit-factor-availability",
                    message=f"factor {factor.label} does not define an explicit availability contract",
                )
            )
        if not factor.evidence_paths:
            issues.append(
                GateIssue(
                    code="missing-factor-evidence",
                    message=f"factor {factor.label} must declare supporting evidence paths",
                )
            )
        for raw_path in factor.evidence_paths:
            _append_missing_path_issue(
                issues,
                code="missing-factor-evidence-path",
                message=f"factor {factor.label} references missing evidence",
                raw_path=raw_path,
                kind="file",
            )

    for review, code_prefix in (
        (prerequisites.program_semantics_review, "program-semantics-review"),
        (prerequisites.research_design_review, "research-design-review"),
    ):
        if review.verdict != "pass":
            issues.append(
                GateIssue(
                    code=f"{code_prefix}-not-passed",
                    message=f"{review.label} must pass before a formal rerun starts",
                )
            )
        if not review.evidence_paths:
            issues.append(
                GateIssue(
                    code=f"{code_prefix}-missing-evidence",
                    message=f"{review.label} must attach evidence paths",
                )
            )
        for raw_path in review.evidence_paths:
            _append_missing_path_issue(
                issues,
                code=f"{code_prefix}-missing-evidence-path",
                message=f"{review.label} references missing evidence",
                raw_path=raw_path,
                kind="file",
            )

    return FormalRerunGateResult(
        passed=not issues,
        issues=tuple(issues),
    )
