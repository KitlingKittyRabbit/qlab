from __future__ import annotations

import json

import pandas as pd

from qlab.evidence import build_evidence_bundle, write_evidence_bundle
from qlab.provenance import validate_candidate_frame


def test_validate_candidate_frame_accepts_valid_candidate_with_provenance():
    frame = pd.DataFrame(
        {
            "candidate_id": ["12h__liq__imbalance__ETH__12h"],
            "signal_timeframe": ["12h"],
            "family": ["liquidation"],
            "source_name": ["liq"],
            "transform_name": ["imbalance"],
            "symbol": ["ETH"],
            "horizon": ["12h"],
            "entry_rule": ["next_open"],
            "generated_at": ["2026-05-24T10:00:00Z"],
            "schema_version": ["1.0.0"],
            "generator_name": ["selector_pool_builder"],
            "generator_commit": ["abc123"],
            "source_data_cutoff": ["2026-05-24T09:45:00Z"],
            "selector_window_start": ["2026-04-24T00:00:00Z"],
            "selector_window_end": ["2026-05-24T00:00:00Z"],
        }
    )

    result = validate_candidate_frame(frame)

    assert result.valid is True
    assert result.issues == ()


def test_validate_candidate_frame_rejects_duplicate_ids_and_missing_provenance():
    frame = pd.DataFrame(
        {
            "candidate_id": ["dup", "dup"],
            "signal_timeframe": ["12h", "12h"],
            "family": ["liquidation", "liquidation"],
            "source_name": ["liq", "liq"],
            "transform_name": ["imbalance", "imbalance"],
            "symbol": ["ETH", "BTC"],
            "horizon": ["12h", "12h"],
            "entry_rule": ["next_open", "next_open"],
        }
    )

    result = validate_candidate_frame(frame)

    assert result.valid is False
    issue_codes = {issue.code for issue in result.issues}
    assert "missing-column" in issue_codes
    assert "duplicate-candidate-id" in issue_codes


def test_write_evidence_bundle_persists_hashes_and_metadata(tmp_path):
    report = tmp_path / "report.md"
    report.write_text("evidence report\n", encoding="utf-8")
    metrics = tmp_path / "metrics.csv"
    metrics.write_text("symbol,pnl\nETH,1.23\n", encoding="utf-8")

    bundle = build_evidence_bundle(
        artifacts={"report": report, "metrics": metrics},
        metadata={"strategy": "coinglass", "review_id": "2026-05-24"},
    )

    assert len(bundle.artifacts) == 2
    assert bundle.metadata["strategy"] == "coinglass"
    assert all(artifact.sha256 for artifact in bundle.artifacts)

    output_path = write_evidence_bundle(
        output_path=tmp_path / "bundle.json",
        artifacts={"report": report, "metrics": metrics},
        metadata={"strategy": "coinglass"},
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["metadata"]["strategy"] == "coinglass"
    assert {item["label"]
            for item in payload["artifacts"]} == {"report", "metrics"}
