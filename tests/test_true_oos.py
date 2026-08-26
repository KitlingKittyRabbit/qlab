from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from qlab.true_oos import (
    AsReceivedSnapshotStore,
    AppendOnlyEventLedger,
    BookQuote,
    CANONICAL_SYMBOLS,
    CONSISTENCY_DIMENSIONS,
    ExchangeRule,
    REQUIRED_ACTIVATION_PREFLIGHTS,
    SourceFreshness,
    TRUE_OOS_RUNTIME_CONTRACT_VERSION,
    aggregate_testnet_canary,
    apply_consistency_amendments,
    apply_fill_to_position,
    build_activation_intent,
    build_activation_manifest,
    build_epoch_training_panels,
    build_epoch_window,
    build_missed_decision_records,
    build_replay_live_consistency_record,
    build_revision_consistency_amendments,
    build_shadow_decision_artifacts,
    build_shadow_candidate_freeze_manifest,
    build_shadow_structural_equivalence_manifest,
    build_source_equivalence_record,
    build_source_equivalence_consistency_amendments,
    build_source_consistency_queue_records,
    build_source_reference_time_contract,
    build_source_reference_queue_records,
    build_source_value_equivalence_record,
    build_source_observation_rows,
    build_source_revision_event,
    classify_decision_readiness,
    classify_source_reference_action,
    classify_source_consistency_reference_action,
    consistency_evidence_sha256,
    book_quotes_from_binance,
    fit_epoch_candidate_parameters,
    fit_epoch_candidate_signals,
    first_eligible_decision_ts,
    due_candidate_rows,
    detect_source_observation_revisions,
    initial_shadow_state,
    marked_equity,
    persist_shadow_fill_events,
    plan_shadow_decision,
    plan_candidate_transitions,
    quantity_toward_zero,
    require_activation_intent_publishable,
    require_true_oos_runtime_contract,
    select_long_short_memberships,
    sha256_file,
    score_epoch_candidate_signals,
    safe_snapshot_source_id,
    source_values_at_exact_label,
    validate_source_freshness,
    validate_source_consistency_capture_contract,
    verify_sha256_manifest,
    verify_activation_intent,
    verify_activation_manifest,
    verify_freeze_source_manifest,
    verify_json_sha256_sidecar,
    write_immutable_input_snapshot,
    write_immutable_json_with_sha256,
    write_freeze_bundle,
)


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        {"symbol": CANONICAL_SYMBOLS, "signal_value": range(len(CANONICAL_SYMBOLS))}
    )


def _quotes(price: float = 100.0) -> dict[str, BookQuote]:
    return {
        symbol: BookQuote(
            symbol,
            price - 1.0,
            price + 1.0,
            "2026-08-01T00:00:02+00:00",
            "2026-08-01T00:00:01+00:00",
        )
        for symbol in CANONICAL_SYMBOLS
    }


def _evidence(live, replay) -> dict[str, str]:
    return {
        dimension: consistency_evidence_sha256(
            live[dimension], replay[dimension]
        )
        for dimension in CONSISTENCY_DIMENSIONS
    }


def _rules(step: float = 0.01, minimum: float = 5.0) -> dict[str, ExchangeRule]:
    return {
        symbol: ExchangeRule(
            symbol, "TRADING", step, step, minimum, "2026-08-01T00:00:01+00:00"
        )
        for symbol in CANONICAL_SYMBOLS
    }


def _structural_equivalence_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_counts = {"4h": 4, "8h": 28, "12h": 36, "1d": 32}
    signal_counts = {"4h": 1, "8h": 7, "12h": 21, "1d": 18}
    rows: list[dict[str, str]] = []
    registry_rows: list[dict[str, str]] = []
    schemes = ("equal", "family_alpha_0", "family_alpha_0p5", "family_alpha_1")
    for horizon, total_candidates in candidate_counts.items():
        groups = signal_counts[horizon]
        aliases_per_group = [1] * groups
        for index in range(total_candidates - groups):
            aliases_per_group[index % groups] += 1
        for group_index, alias_count in enumerate(aliases_per_group):
            feature = f"feature_{horizon}_{group_index:02d}"
            registry_rows.append({"feature_name": feature, "family": "synthetic"})
            for alias_index in range(alias_count):
                rows.append(
                    {
                        "combo_id": f"combo_{horizon}_{group_index:02d}",
                        "track": f"track_{horizon}_{group_index:02d}",
                        "weight_scheme": schemes[alias_index],
                        "return_horizon": horizon,
                        "component_features": feature,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(registry_rows)


def test_true_oos_runtime_contract_rejects_superseded_implementation() -> None:
    require_true_oos_runtime_contract(TRUE_OOS_RUNTIME_CONTRACT_VERSION)
    with pytest.raises(RuntimeError, match="runtime contract is superseded"):
        require_true_oos_runtime_contract("common_t0_v1")


def test_shadow_structural_equivalence_groups_only_identical_weight_rules() -> None:
    catalog, registry = _structural_equivalence_fixture()
    aliases, unique = build_shadow_structural_equivalence_manifest(catalog, registry)
    assert len(aliases) == 100
    assert len(unique) == 47
    assert unique.groupby("horizon").size().to_dict() == {
        "4h": 1, "8h": 7, "12h": 21, "1d": 18
    }
    funding = aliases.loc[
        aliases["candidate_id"].isin(
            [
                f"4h__track_4h_00__{scheme}"
                for scheme in (
                    "equal", "family_alpha_0", "family_alpha_0p5", "family_alpha_1"
                )
            ]
        )
    ]
    assert len(funding) == 4
    assert funding["signal_equivalence_id"].nunique() == 1
    freeze = build_shadow_candidate_freeze_manifest(
        unique,
        freeze_version="v4_test",
        data_source_contract="ksv4_shadow_sources_test",
    )
    assert len(freeze) == 47
    assert freeze["runtime_contract_version"].eq(
        TRUE_OOS_RUNTIME_CONTRACT_VERSION
    ).all()
    assert freeze["data_source_contract"].eq(
        "ksv4_shadow_sources_test"
    ).all()
    with pytest.raises(ValueError, match="data_source_contract must not be empty"):
        build_shadow_candidate_freeze_manifest(
            unique,
            freeze_version="v4_test",
            data_source_contract="",
        )


def test_shadow_structural_equivalence_rejects_conflicting_history() -> None:
    catalog, registry = _structural_equivalence_fixture()
    aliases, _ = build_shadow_structural_equivalence_manifest(catalog, registry)
    history = aliases[["candidate_id", "signal_equivalence_id"]].copy()
    history.loc[0, "signal_equivalence_id"] = "deliberately_wrong"
    with pytest.raises(ValueError, match="historical signal equivalence differs"):
        build_shadow_structural_equivalence_manifest(
            catalog, registry, historical_mapping=history
        )


def test_epoch_window_and_native_first_decisions_are_exact() -> None:
    epoch = build_epoch_window("2026-08-01T00:00:00Z", 0)
    assert epoch.train_start == "2026-02-01T00:00:00+00:00"
    assert epoch.train_end_exclusive == "2026-07-31T00:00:00+00:00"
    assert epoch.embargo_start == "2026-07-31T00:00:00+00:00"
    assert epoch.embargo_end_exclusive == "2026-08-01T00:00:00+00:00"
    assert epoch.run_end_exclusive == "2026-09-15T00:00:00+00:00"
    assert first_eligible_decision_ts(
        "4h", "2026-08-01T00:00:00Z"
    ) == pd.Timestamp("2026-08-01T04:00:00Z")
    assert first_eligible_decision_ts(
        "12h", "2026-08-01T10:15:00Z"
    ) == pd.Timestamp("2026-08-01T12:00:00Z")
    assert first_eligible_decision_ts(
        "1d", "2026-08-01T10:15:00Z"
    ) == pd.Timestamp("2026-08-02T00:00:00Z")


def test_non_midnight_shadow_start_uses_only_complete_calendar_days() -> None:
    epoch = build_epoch_window("2026-08-01T10:15:00Z", 0, train_days=2)
    assert epoch.shadow_start_ts == "2026-08-01T10:15:00+00:00"
    assert epoch.train_start == "2026-07-29T00:00:00+00:00"
    assert epoch.train_end_exclusive == "2026-07-31T00:00:00+00:00"
    assert epoch.embargo_start == "2026-07-31T00:00:00+00:00"
    assert epoch.embargo_end_exclusive == "2026-08-01T00:00:00+00:00"
    assert epoch.run_start == "2026-08-01T10:15:00+00:00"


def test_due_candidates_and_missed_records_follow_native_horizons() -> None:
    candidates = pd.DataFrame(
        {
            "freeze_version": ["v1"] * 4,
            "signal_equivalence_id": ["s4", "s8", "s12", "s1d"],
            "horizon": ["4h", "8h", "12h", "1d"],
        }
    )
    due = due_candidate_rows(
        candidates,
        shadow_start_ts="2026-08-01T01:00:00Z",
        decision_ts="2026-08-01T08:00:00Z",
    )
    assert due["signal_equivalence_id"].tolist() == ["s4", "s8"]
    missed = build_missed_decision_records(
        candidates,
        freeze_version="v1",
        shadow_start_ts="2026-08-01T01:00:00Z",
        decision_ts="2026-08-01T08:00:00Z",
        reason="service was offline",
    )
    assert {row["signal_equivalence_id"] for row in missed} == {"s4", "s8"}
    assert {row["overall_status"] for row in missed} == {"missed_decision"}
    assert {row["missed_reason"] for row in missed} == {"service was offline"}


def test_as_received_snapshot_is_immutable_and_revisions_are_explicit(tmp_path) -> None:
    store = AsReceivedSnapshotStore(tmp_path / "source")
    first = store.persist(
        b'{"value":1}',
        source_id="keystore_v4_basis",
        source_request_ts="2026-08-01T00:00:01Z",
        source_response_ts="2026-08-01T00:00:02Z",
        source_bar_label_ts="2026-08-01T00:00:00Z",
        native_bar_end_ts="2026-08-01T00:00:00Z",
    )
    retried = store.persist(
        b'{"value":1}',
        source_id="keystore_v4_basis",
        source_request_ts="2026-08-01T00:00:01Z",
        source_response_ts="2026-08-01T00:00:02Z",
        source_bar_label_ts="2026-08-01T00:00:00Z",
        native_bar_end_ts="2026-08-01T00:00:00Z",
    )
    assert retried == first
    assert store.load_verified(first) == b'{"value":1}'
    later = store.persist(
        b'{"value":2}',
        source_id="keystore_v4_basis",
        source_request_ts="2026-08-01T00:10:01Z",
        source_response_ts="2026-08-01T00:10:02Z",
        source_bar_label_ts="2026-08-01T00:00:00Z",
        native_bar_end_ts="2026-08-01T00:00:00Z",
    )
    revision = build_source_revision_event(first, later)
    assert revision is not None
    assert revision["earlier_payload_sha256"] != revision["later_payload_sha256"]
    Path(first.payload_path).write_bytes(b'{"value":999}')
    with pytest.raises(ValueError, match="payload SHA-256"):
        store.load_verified(first)


def test_as_received_snapshot_rejects_pre_bar_observation(tmp_path) -> None:
    store = AsReceivedSnapshotStore(tmp_path)
    with pytest.raises(ValueError, match="before native_bar_end"):
        store.persist(
            b"x",
            source_id="source",
            source_request_ts="2026-07-31T23:59:58Z",
            source_response_ts="2026-07-31T23:59:59Z",
            source_bar_label_ts="2026-08-01T00:00:00Z",
            native_bar_end_ts="2026-08-01T00:00:00Z",
        )


def test_snapshot_store_distinguishes_normalized_projection(tmp_path) -> None:
    store = AsReceivedSnapshotStore(tmp_path / "snapshots")
    receipt = store.persist(
        b'{"data":[]}',
        source_id="normalized_demo",
        source_request_ts="2026-01-01T00:00:01Z",
        source_response_ts="2026-01-01T00:00:02Z",
        source_bar_label_ts="2026-01-01T00:00:00Z",
        native_bar_end_ts="2026-01-01T00:00:00Z",
        evidence_role="normalized_projection",
    )
    metadata = json.loads(Path(receipt.receipt_path).read_text())
    assert metadata["evidence_role"] == "normalized_projection"
    assert metadata["Lifecycle"] == "authoritative normalized source projection"
    assert store.load_verified(receipt) == b'{"data":[]}'


def test_snapshot_source_id_is_safe_deterministic_and_role_separated() -> None:
    request_id = "keystore|futures/v2/net-position/history|BTC|snapshot"
    acquisition = safe_snapshot_source_id(request_id, role="acquisition")
    semantic = safe_snapshot_source_id(request_id, role="semantic_binding")
    assert acquisition == safe_snapshot_source_id(
        request_id, role="acquisition"
    )
    assert acquisition != semantic
    assert "|" not in acquisition and "/" not in acquisition
    assert acquisition.startswith("acquisition.keystore_futures_v2_")


def test_snapshot_source_id_rejects_empty_identity() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        safe_snapshot_source_id("", role="acquisition")
    with pytest.raises(ValueError, match="non-empty"):
        safe_snapshot_source_id("source", role="")


def test_source_observation_revisions_detect_changed_overlapping_bar() -> None:
    first = build_source_observation_rows(
        pd.DataFrame(
            {"value": [1.0, 2.0]},
            index=pd.to_datetime(
                ["2026-08-01T00:00:00Z", "2026-08-01T01:00:00Z"]
            ),
        ),
        source_id="keystore_v4_oi_1h_BTC",
        receipt_id="receipt-a",
        data_observed_ts="2026-08-01T02:00:01Z",
    )
    second = build_source_observation_rows(
        pd.DataFrame(
            {"value": [1.5, 2.0, 3.0]},
            index=pd.to_datetime(
                [
                    "2026-08-01T00:00:00Z",
                    "2026-08-01T01:00:00Z",
                    "2026-08-01T02:00:00Z",
                ]
            ),
        ),
        source_id="keystore_v4_oi_1h_BTC",
        receipt_id="receipt-b",
        data_observed_ts="2026-08-01T03:00:01Z",
    )
    revisions = detect_source_observation_revisions(first, second)
    assert len(revisions) == 1
    row = revisions.iloc[0]
    assert row["source_bar_label_ts"] == "2026-08-01T00:00:00+00:00"
    assert row["earlier_receipt_id"] == "receipt-a"
    assert row["later_receipt_id"] == "receipt-b"
    assert row["earlier_row_sha256"] != row["later_row_sha256"]


def test_source_observation_revisions_ignore_unchanged_and_new_bars() -> None:
    first = build_source_observation_rows(
        pd.DataFrame(
            {"value": [1.0]},
            index=pd.to_datetime(["2026-08-01T00:00:00Z"]),
        ),
        source_id="source",
        receipt_id="receipt-a",
        data_observed_ts="2026-08-01T01:00:01Z",
    )
    second = build_source_observation_rows(
        pd.DataFrame(
            {"value": [1.0, 2.0]},
            index=pd.to_datetime(
                ["2026-08-01T00:00:00Z", "2026-08-01T01:00:00Z"]
            ),
        ),
        source_id="source",
        receipt_id="receipt-b",
        data_observed_ts="2026-08-01T02:00:01Z",
    )
    assert detect_source_observation_revisions(first, second).empty


def test_source_revision_amends_every_candidate_that_used_old_receipt() -> None:
    records = [
        {
            "freeze_version": "v1",
            "signal_equivalence_id": candidate_id,
            "horizon": "4h",
            "decision_ts": "2026-08-01T04:00:00+00:00",
            "receipt_id": "receipt-a",
        }
        for candidate_id in ("signal_001", "signal_002")
    ]
    revisions = pd.DataFrame(
        [
            {
                "source_id": "source",
                "source_bar_label_ts": "2026-08-01T03:00:00+00:00",
                "earlier_receipt_id": "receipt-a",
                "later_receipt_id": "receipt-b",
                "earlier_row_sha256": "a" * 64,
                "later_row_sha256": "b" * 64,
                "revision_observed_ts": "2026-08-01T08:00:01+00:00",
            }
        ]
    )
    amendments = build_revision_consistency_amendments(records, revisions)
    assert [row["signal_equivalence_id"] for row in amendments] == [
        "signal_001",
        "signal_002",
    ]
    assert all(
        row["amended_overall_status"] == "replay_live_consistency_fail"
        for row in amendments
    )


def test_consistency_amendment_changes_effective_status_without_rewriting_original() -> None:
    original = {
        "freeze_version": "freeze",
        "signal_equivalence_id": "candidate",
        "horizon": "4h",
        "decision_ts": "2026-08-01T04:00:00+00:00",
        "overall_status": "pass",
        "dimensions": {
            dimension: {"status": "pass"} for dimension in CONSISTENCY_DIMENSIONS
        },
    }
    amendment = {
        "freeze_version": "freeze",
        "signal_equivalence_id": "candidate",
        "decision_ts": "2026-08-01T04:00:00+00:00",
        "dimension": "data_availability_consistency",
        "amended_dimension_status": "fail",
        "amended_overall_status": "replay_live_consistency_fail",
        "reason": "later source revision",
    }
    effective = apply_consistency_amendments([original], [amendment])
    assert original["overall_status"] == "pass"
    assert effective[0]["overall_status"] == "replay_live_consistency_fail"
    assert (
        effective[0]["dimensions"]["data_availability_consistency"]["status"]
        == "fail"
    )
    assert effective[0]["amended"] is True


def test_consistency_amendment_rejects_missing_original_record() -> None:
    with pytest.raises(ValueError, match="no original record"):
        apply_consistency_amendments(
            [],
            [
                {
                    "freeze_version": "freeze",
                    "signal_equivalence_id": "candidate",
                    "decision_ts": "2026-08-01T04:00:00+00:00",
                    "dimension": "data_availability_consistency",
                    "amended_dimension_status": "fail",
                    "amended_overall_status": "replay_live_consistency_fail",
                }
            ],
        )


def test_consistency_failure_amendment_is_sticky() -> None:
    original = {
        "freeze_version": "freeze",
        "signal_equivalence_id": "candidate",
        "horizon": "4h",
        "decision_ts": "2026-08-01T04:00:00+00:00",
        "overall_status": "replay_live_consistency_pass",
        "dimensions": {
            dimension: {"status": "pass"} for dimension in CONSISTENCY_DIMENSIONS
        },
    }
    identity = {
        "freeze_version": "freeze",
        "signal_equivalence_id": "candidate",
        "decision_ts": "2026-08-01T04:00:00+00:00",
        "dimension": "data_availability_consistency",
    }
    effective = apply_consistency_amendments(
        [original],
        [
            {
                **identity,
                "amended_dimension_status": "fail",
                "amended_overall_status": "replay_live_consistency_fail",
                "reason": "material source mismatch",
            },
            {
                **identity,
                "amended_dimension_status": "pass",
                "amended_overall_status": "replay_live_consistency_pass",
                "reason": "later reference matched",
            },
        ],
    )
    assert effective[0]["overall_status"] == "replay_live_consistency_fail"
    assert (
        effective[0]["dimensions"]["data_availability_consistency"]["status"]
        == "fail"
    )


def test_immutable_input_snapshot_binds_exact_available_bytes(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    first = source / "caches" / "a.pkl"
    first.parent.mkdir()
    first.write_bytes(b"alpha")
    destination = tmp_path / "snapshot"
    manifest = write_immutable_input_snapshot(
        [first],
        source_root=source,
        destination_root=destination,
        observed_ts="2026-08-01T00:00:01Z",
    )
    copied = Path(manifest.iloc[0]["snapshot_path"])
    assert copied.read_bytes() == b"alpha"
    assert manifest.iloc[0]["source_relative_path"] == "caches/a.pkl"
    assert manifest.iloc[0]["sha256"] == sha256_file(first)
    assert len(
        write_immutable_input_snapshot(
            [first],
            source_root=source,
            destination_root=destination,
            observed_ts="2026-08-01T00:00:01Z",
        )
    ) == 1
    first.write_bytes(b"changed")
    with pytest.raises(ValueError, match="different content"):
        write_immutable_input_snapshot(
            [first],
            source_root=source,
            destination_root=destination,
            observed_ts="2026-08-01T00:00:02Z",
        )


def test_immutable_json_sha_bundle_detects_partial_or_tampered_authority(
    tmp_path,
) -> None:
    path = tmp_path / "activation.json"
    output, sidecar = write_immutable_json_with_sha256({"freeze": "v1"}, path)
    assert output == path
    assert sidecar.is_file()
    assert verify_json_sha256_sidecar(path) == sha256_file(path)
    write_immutable_json_with_sha256({"freeze": "v1"}, path)
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="sidecar mismatch"):
        verify_json_sha256_sidecar(path)


def test_replay_live_consistency_compares_all_dimensions_and_fails_closed() -> None:
    live = {
        dimension: {"value": index}
        for index, dimension in enumerate(CONSISTENCY_DIMENSIONS)
    }
    evidence = _evidence(live, live)
    passed = build_replay_live_consistency_record(
        freeze_version="v2",
        signal_equivalence_id="signal_001",
        horizon="4h",
        decision_ts="2026-08-01T04:00:00Z",
        live_dimensions=live,
        replay_dimensions=live,
        evidence_sha256=evidence,
    )
    assert passed["overall_status"] == "replay_live_consistency_pass"
    assert all(
        row["status"] == "pass" for row in passed["dimensions"].values()
    )

    replay = {key: dict(value) for key, value in live.items()}
    replay["position_consistency"]["value"] = 999
    failed_evidence = _evidence(live, replay)
    failed = build_replay_live_consistency_record(
        freeze_version="v2",
        signal_equivalence_id="signal_001",
        horizon="4h",
        decision_ts="2026-08-01T04:00:00Z",
        live_dimensions=live,
        replay_dimensions=replay,
        evidence_sha256={
            key: value
            for key, value in failed_evidence.items()
            if key != "signal_consistency"
        },
    )
    assert failed["overall_status"] == "replay_live_consistency_fail"
    assert failed["dimensions"]["position_consistency"]["difference_paths"] == [
        "value"
    ]
    assert failed["dimensions"]["signal_consistency"]["status"] == "fail"


def test_replay_live_consistency_pending_reference_is_neither_pass_nor_fail() -> None:
    live = {
        dimension: {"value": index}
        for index, dimension in enumerate(CONSISTENCY_DIMENSIONS)
    }
    record = build_replay_live_consistency_record(
        freeze_version="v3",
        signal_equivalence_id="signal_001",
        horizon="4h",
        decision_ts="2026-08-01T04:00:00Z",
        live_dimensions=live,
        replay_dimensions=live,
        evidence_sha256=_evidence(live, live),
        dimension_overrides={
            "data_availability_consistency": "pending_reference"
        },
    )
    assert record["overall_status"] == "replay_live_consistency_pending_reference"
    assert (
        record["dimensions"]["data_availability_consistency"]["status"]
        == "pending_reference"
    )
    assert all(
        row["status"] == "pass"
        for name, row in record["dimensions"].items()
        if name != "data_availability_consistency"
    )


def test_source_equivalence_distinguishes_pending_exact_and_materiality() -> None:
    common = {
        "freeze_version": "v3",
        "signal_equivalence_id": "signal_001",
        "source_id": "keystore_v4_basis_12h_BTC",
        "symbol": "BTC",
        "native_bar_end_ts": "2026-08-01T12:00:00Z",
        "realtime_receipt_id": "real-receipt",
        "realtime_values": {"close_basis": 0.01},
        "realtime_decision_projection": {
            "rank": 3,
            "membership": "long",
            "order_quantity": 0.1,
        },
        "observed_ts": "2026-08-01T12:00:04Z",
    }
    pending = build_source_equivalence_record(**common)
    assert pending["status"] == "pending_reference"
    assert pending["reference_values_sha256"] is None

    exact = build_source_equivalence_record(
        **common,
        reference_receipt_id="history-initial",
        reference_values={"close_basis": 0.01},
        reference_decision_projection=common["realtime_decision_projection"],
        reference_role="initial",
    )
    assert exact["status"] == "exact_match"

    equivalent = build_source_equivalence_record(
        **common,
        reference_receipt_id="history-initial-2",
        reference_values={"close_basis": 0.011},
        reference_decision_projection=common["realtime_decision_projection"],
        reference_role="initial",
    )
    assert equivalent["status"] == "value_mismatch_decision_equivalent"
    assert equivalent["value_difference_paths"] == ["close_basis"]
    assert equivalent["decision_difference_paths"] == []

    material = build_source_equivalence_record(
        **common,
        reference_receipt_id="history-revision",
        reference_values={"close_basis": -0.02},
        reference_decision_projection={
            "rank": 18,
            "membership": "short",
            "order_quantity": -0.1,
        },
        reference_role="revision",
    )
    assert material["status"] == "decision_material_mismatch"
    assert material["decision_difference_paths"] == [
        "membership",
        "order_quantity",
        "rank",
    ]


def test_source_equivalence_rejects_partial_reference_evidence() -> None:
    with pytest.raises(ValueError, match="partial reference evidence"):
        build_source_equivalence_record(
            freeze_version="v3",
            signal_equivalence_id="signal_001",
            source_id="source",
            symbol="BTC",
            native_bar_end_ts="2026-08-01T12:00:00Z",
            realtime_receipt_id="real",
            realtime_values={"value": 1.0},
            realtime_decision_projection={"rank": 1},
            reference_receipt_id="history",
            observed_ts="2026-08-01T12:00:04Z",
        )


def test_source_equivalence_rejects_empty_candidate_identity() -> None:
    with pytest.raises(ValueError, match="identity must not be empty"):
        build_source_equivalence_record(
            freeze_version="v3",
            signal_equivalence_id="",
            source_id="source",
            symbol="BTC",
            native_bar_end_ts="2026-08-01T12:00:00Z",
            realtime_receipt_id="real",
            realtime_values={"value": 1.0},
            realtime_decision_projection={"rank": 1},
            observed_ts="2026-08-01T12:00:04Z",
        )


def test_source_reference_queue_freezes_initial_retry_revision_and_timeout() -> None:
    frame = pd.DataFrame(
        [{
            "source_scope": "ksv4_12h",
            "signal_timeframe": "12h",
            "endpoint": "basis",
            "symbol": "BTC",
            "receipt_id": "receipt-1",
            "target_label_ts": "2026-07-31T00:00:00Z",
        }]
    )
    records = build_source_reference_queue_records(
        frame,
        freeze_version="v1",
        decision_ts="2026-07-31T12:00:00Z",
        initial_query_delay_seconds=900,
        retry_interval_seconds=3600,
        revision_query_delay_seconds=86400,
        maximum_wait_seconds=259200,
    )
    assert records[0]["status"] == "pending_reference"
    assert records[0]["initial_query_due_ts"] == "2026-07-31T12:15:00+00:00"
    assert records[0]["revision_query_due_ts"] == "2026-08-01T12:00:00+00:00"
    assert records[0]["maximum_wait_ts"] == "2026-08-03T12:00:00+00:00"
    with pytest.raises(ValueError, match="revision query"):
        build_source_reference_queue_records(
            frame,
            freeze_version="v1",
            decision_ts="2026-07-31T12:00:00Z",
            initial_query_delay_seconds=900,
            retry_interval_seconds=3600,
            revision_query_delay_seconds=900,
            maximum_wait_seconds=259200,
        )


def test_source_reference_time_contract_aligns_label_and_query_window() -> None:
    for duration in ("1h", "12h", "1d"):
        target = pd.Timestamp("2026-08-16T15:00:00Z")
        start_contract = build_source_reference_time_contract(
            target_label_ts=target,
            timestamp_kind="bar_start",
            bar_duration=duration,
        )
        expected_end = target + pd.Timedelta(duration)
        assert start_contract["target_label_ts"] == target
        assert start_contract["native_bar_end_ts"] == expected_end
        assert start_contract["query_end_ts"] == expected_end
        assert start_contract["query_end_time_ms"] == int(
            expected_end.timestamp() * 1000
        )

        end_contract = build_source_reference_time_contract(
            target_label_ts=target,
            timestamp_kind="bar_end",
            bar_duration=duration,
        )
        assert end_contract["target_label_ts"] == target
        assert end_contract["native_bar_end_ts"] == target
        assert end_contract["query_end_ts"] == expected_end
        assert end_contract["query_end_time_ms"] == int(
            expected_end.timestamp() * 1000
        )
        assert start_contract["native_bar_end_ts"] != end_contract["native_bar_end_ts"]

    with pytest.raises(ValueError, match="positive"):
        build_source_reference_time_contract(
            target_label_ts="2026-08-16T15:00:00Z",
            timestamp_kind="bar_start",
            bar_duration="0s",
        )


def test_source_values_require_exact_label_and_do_not_use_nearest_row() -> None:
    target = pd.Timestamp("2026-08-16T15:00:00Z")
    frame = pd.DataFrame(
        {"close": [1.0, 2.0]},
        index=pd.DatetimeIndex(
            [target - pd.Timedelta(hours=1), target + pd.Timedelta(hours=1)]
        ),
    )
    with pytest.raises(RuntimeError, match="target label"):
        source_values_at_exact_label(frame, target)


def test_source_consistency_capture_contract_and_queue_are_candidate_independent() -> None:
    sources = (
        ["keystore"] * 26
        + ["binance_public"] * 58
        + ["bybit_public"] * 17
        + ["okx_public"] * 17
    )
    contract = pd.DataFrame(
        {
            "request_id": [f"request-{index:03d}" for index in range(118)],
            "source": sources,
            "source_contract_version": ["ksv4_shadow_sources_v4"] * 118,
        }
    )
    raw = pd.DataFrame(
        {
            "request_id": contract["request_id"],
            "receipt_id": [f"raw-{index:03d}" for index in range(118)],
            "payload_sha256": [f"sha-{index:03d}" for index in range(118)],
        }
    )
    normalized = pd.DataFrame(
        [
            {
                "source_scope": f"scope-{index // 20}",
                "signal_timeframe": "1h",
                "endpoint": f"endpoint-{index // 20}",
                "symbol": f"symbol-{index % 20}",
                "receipt_id": f"normalized-{index:03d}",
                "payload_sha256": f"normalized-sha-{index:03d}",
                "target_label_ts": "2026-08-12T00:00:00Z",
            }
            for index in range(360)
        ]
    )
    result = validate_source_consistency_capture_contract(contract, raw, normalized)
    assert result["request_count"] == 118
    assert result["normalized_count"] == 360
    assert result["source_counts"] == {
        "binance_public": 58,
        "bybit_public": 17,
        "keystore": 26,
        "okx_public": 17,
    }

    queue = build_source_consistency_queue_records(
        normalized.iloc[[0]],
        collector_id="collector-v1",
        capture_ts="2026-08-12T00:00:00Z",
        initial_query_delay_seconds=900,
        retry_interval_seconds=3600,
        revision_query_delay_seconds=86400,
        maximum_wait_seconds=259200,
    )[0]
    assert queue["collector_id"] == "collector-v1"
    assert "freeze_version" not in queue
    assert classify_source_consistency_reference_action(
        queue, [], observed_ts="2026-08-12T00:15:00Z"
    ) == "initial"

    with pytest.raises(ValueError, match="raw receipt count"):
        validate_source_consistency_capture_contract(
            contract, raw.iloc[:-1], normalized
        )


def test_source_value_equivalence_preserves_values_and_field_differences() -> None:
    common = {
        "collector_id": "collector-v1",
        "capture_ts": "2026-08-12T00:00:00Z",
        "source_scope": "ksv4_1h",
        "signal_timeframe": "1h",
        "endpoint": "fr",
        "symbol": "BTC",
        "target_label_ts": "2026-08-11T23:00:00Z",
        "realtime_native_bar_end_ts": "2026-08-12T00:00:00Z",
        "realtime_receipt_id": "realtime-1",
        "reference_receipt_id": "reference-1",
        "reference_native_bar_end_ts": "2026-08-12T00:00:00Z",
        "reference_role": "initial",
        "observed_ts": "2026-08-12T00:15:01Z",
    }
    exact = build_source_value_equivalence_record(
        **common,
        realtime_values={"close": 0.01},
        reference_values={"close": 0.01},
    )
    assert exact["status"] == "exact_match"
    assert exact["realtime_values"] == {"close": 0.01}

    changed = build_source_value_equivalence_record(
        **common,
        realtime_values={"close": 0.01, "open": 0.02},
        reference_values={"close": 0.011},
    )
    assert changed["status"] == "field_mismatch"
    assert changed["missing_from_reference"] == ["open"]
    assert changed["unequal_fields"] == ["close"]
    assert changed["absolute_differences"]["close"] == pytest.approx(0.001)


def test_source_consistency_two_sources_two_coins_initial_and_revision_hand_case(
    tmp_path,
) -> None:
    realtime = {
        ("fr", "BTC"): 1.0,
        ("fr", "ETH"): 2.0,
        ("oi", "BTC"): 3.0,
        ("oi", "ETH"): 4.0,
    }
    frame = pd.DataFrame(
        [
            {
                "source_scope": "ksv4_1h",
                "signal_timeframe": "1h",
                "endpoint": endpoint,
                "symbol": symbol,
                "receipt_id": f"real-{endpoint}-{symbol}",
                "target_label_ts": "2026-08-11T23:00:00Z",
            }
            for endpoint, symbol in realtime
        ]
    )
    queue = build_source_consistency_queue_records(
        frame,
        collector_id="collector-v1",
        capture_ts="2026-08-12T00:00:00Z",
        initial_query_delay_seconds=900,
        retry_interval_seconds=3600,
        revision_query_delay_seconds=86400,
        maximum_wait_seconds=259200,
    )
    assert all(
        classify_source_consistency_reference_action(
            row, [], observed_ts="2026-08-12T00:15:00Z"
        )
        == "initial"
        for row in queue
    )
    records = []
    for role, observed, adjustment in (
        ("initial", "2026-08-12T00:15:01Z", 0.0),
        ("revision", "2026-08-13T00:00:01Z", 0.5),
    ):
        for endpoint, symbol in realtime:
            value = realtime[(endpoint, symbol)]
            records.append(
                build_source_value_equivalence_record(
                    collector_id="collector-v1",
                    capture_ts="2026-08-12T00:00:00Z",
                    source_scope="ksv4_1h",
                    signal_timeframe="1h",
                    endpoint=endpoint,
                    symbol=symbol,
                    target_label_ts="2026-08-11T23:00:00Z",
                    realtime_native_bar_end_ts="2026-08-12T00:00:00Z",
                    realtime_receipt_id=f"real-{endpoint}-{symbol}",
                    realtime_values={"value": value},
                    reference_receipt_id=f"{role}-{endpoint}-{symbol}",
                    reference_native_bar_end_ts="2026-08-12T00:00:00Z",
                    reference_values={"value": value + adjustment},
                    reference_role=role,
                    observed_ts=observed,
                )
            )
    assert [row["status"] for row in records[:4]] == ["exact_match"] * 4
    assert [row["status"] for row in records[4:]] == ["value_mismatch"] * 4
    assert all(
        row["absolute_differences"]["value"] == pytest.approx(0.5)
        for row in records[4:]
    )

    ledger = AppendOnlyEventLedger(
        tmp_path / "comparisons.jsonl",
        tmp_path / "comparison_state.json",
        key_fields=("collector_id", "realtime_receipt_id", "reference_role"),
    )
    ledger.append_batch(records, {"comparison_count": 8})
    persisted = ledger.read_events()
    assert len(persisted) == 8
    assert all(
        classify_source_consistency_reference_action(
            row,
            persisted,
            observed_ts="2026-08-13T00:00:02Z",
        )
        == "complete"
        for row in queue
    )


def test_source_consistency_timeout_is_terminal() -> None:
    queue = build_source_consistency_queue_records(
        pd.DataFrame(
            [
                {
                    "source_scope": "ksv4_1h",
                    "signal_timeframe": "1h",
                    "endpoint": "fr",
                    "symbol": "BTC",
                    "receipt_id": "real-fr-BTC",
                    "target_label_ts": "2026-08-01T23:00:00Z",
                }
            ]
        ),
        collector_id="collector-v1",
        capture_ts="2026-08-02T00:00:00Z",
        initial_query_delay_seconds=900,
        retry_interval_seconds=3600,
        revision_query_delay_seconds=86400,
        maximum_wait_seconds=259200,
    )[0]
    assert classify_source_consistency_reference_action(
        queue, [], observed_ts="2026-08-05T00:00:00Z"
    ) == "timeout"
    timeout = {
        "collector_id": "collector-v1",
        "realtime_receipt_id": "real-fr-BTC",
        "reference_role": "timeout",
        "attempt_ts": "2026-08-05T00:00:00Z",
    }
    assert classify_source_consistency_reference_action(
        queue,
        [],
        observed_ts="2026-08-05T01:00:00Z",
        failed_attempts=[timeout],
    ) == "expired"


def test_source_reference_action_follows_the_frozen_schedule() -> None:
    queue = build_source_reference_queue_records(
        pd.DataFrame(
            [{
                "source_scope": "ksv4_4h",
                "signal_timeframe": "4h",
                "endpoint": "basis",
                "symbol": "BTC",
                "receipt_id": "real",
                "target_label_ts": "2026-07-31T00:00:00Z",
            }]
        ),
        freeze_version="v1",
        decision_ts="2026-07-31T04:00:00Z",
        initial_query_delay_seconds=900,
        retry_interval_seconds=3600,
        revision_query_delay_seconds=86400,
        maximum_wait_seconds=259200,
    )[0]
    assert classify_source_reference_action(
        queue, [], observed_ts="2026-07-31T04:14:59Z"
    ) == "not_due"
    assert classify_source_reference_action(
        queue, [], observed_ts="2026-07-31T04:15:00Z"
    ) == "initial"
    initial = build_source_equivalence_record(
        freeze_version="v1",
        signal_equivalence_id="signal_001",
        source_id="ksv4_4h:basis",
        symbol="BTC",
        native_bar_end_ts="2026-07-31T04:00:00Z",
        realtime_receipt_id="real",
        realtime_values={"close_basis": 0.1},
        realtime_decision_projection={"members": ["BTC"]},
        reference_receipt_id="initial-ref",
        reference_values={"close_basis": 0.1},
        reference_decision_projection={"members": ["BTC"]},
        reference_role="initial",
        observed_ts="2026-07-31T04:15:01Z",
    )
    assert classify_source_reference_action(
        queue, [initial], observed_ts="2026-08-01T03:59:59Z"
    ) == "not_due"
    assert classify_source_reference_action(
        queue, [initial], observed_ts="2026-08-01T04:00:00Z"
    ) == "revision"
    assert classify_source_reference_action(
        queue, [initial], observed_ts="2026-08-03T04:00:00Z"
    ) == "timeout"
    failed_attempt = {
        "freeze_version": "v1",
        "realtime_receipt_id": "real",
        "reference_role": "initial",
        "attempt_ts": "2026-07-31T04:15:00Z",
    }
    assert classify_source_reference_action(
        queue,
        [],
        observed_ts="2026-07-31T04:30:00Z",
        failed_attempts=[failed_attempt],
    ) == "not_due"
    assert classify_source_reference_action(
        queue,
        [],
        observed_ts="2026-07-31T05:15:00Z",
        failed_attempts=[failed_attempt],
    ) == "initial"


def test_source_equivalence_amendment_passes_only_after_all_sources_resolve() -> None:
    decision = "2026-07-31T04:00:00+00:00"
    original = build_replay_live_consistency_record(
        freeze_version="v1",
        signal_equivalence_id="signal_001",
        horizon="4h",
        decision_ts=decision,
        live_dimensions={name: {"value": 1} for name in CONSISTENCY_DIMENSIONS},
        replay_dimensions={name: {"value": 1} for name in CONSISTENCY_DIMENSIONS},
        evidence_sha256={name: "a" * 64 for name in CONSISTENCY_DIMENSIONS},
        dimension_overrides={"data_availability_consistency": "pending_reference"},
    )
    queue = [
        {
            "freeze_version": "v1",
            "decision_ts": decision,
            "source_scope": "ksv4_4h",
            "endpoint": endpoint,
            "symbol": "BTC",
            "realtime_receipt_id": receipt,
        }
        for endpoint, receipt in (("basis", "real-1"), ("oi", "real-2"))
    ]
    usage = [
        {
            "freeze_version": "v1",
            "signal_equivalence_id": "signal_001",
            "horizon": "4h",
            "decision_ts": decision,
            "receipt_id": receipt,
        }
        for receipt in ("real-1", "real-2", "unrelated-raw")
    ]
    def equivalence(receipt: str, reference: float) -> dict[str, object]:
        return build_source_equivalence_record(
            freeze_version="v1",
            signal_equivalence_id="signal_001",
            source_id="ksv4_4h:source",
            symbol="BTC",
            native_bar_end_ts=decision,
            realtime_receipt_id=receipt,
            realtime_values={"value": 1.0},
            realtime_decision_projection={"members": ["BTC"]},
            reference_receipt_id=f"ref-{receipt}",
            reference_values={"value": reference},
            reference_decision_projection={"members": ["BTC"]},
            reference_role="initial",
            observed_ts="2026-07-31T04:15:01Z",
        )
    first = equivalence("real-1", 1.0)
    assert build_source_equivalence_consistency_amendments(
        [original], usage, queue, [first]
    ) == []
    second = equivalence("real-2", 2.0)
    amendments = build_source_equivalence_consistency_amendments(
        [original], usage, queue, [first, second]
    )
    assert len(amendments) == 1
    assert amendments[0]["amended_dimension_status"] == "pass"
    assert amendments[0]["source_equivalence_statuses"] == [
        "exact_match",
        "value_mismatch_decision_equivalent",
    ]


def test_source_equivalence_amendment_fails_on_decision_material_mismatch() -> None:
    decision = "2026-07-31T04:00:00+00:00"
    original = build_replay_live_consistency_record(
        freeze_version="v1",
        signal_equivalence_id="signal_001",
        horizon="4h",
        decision_ts=decision,
        live_dimensions={name: {"value": 1} for name in CONSISTENCY_DIMENSIONS},
        replay_dimensions={name: {"value": 1} for name in CONSISTENCY_DIMENSIONS},
        evidence_sha256={name: "b" * 64 for name in CONSISTENCY_DIMENSIONS},
        dimension_overrides={"data_availability_consistency": "pending_reference"},
    )
    queue = [{
        "freeze_version": "v1",
        "decision_ts": decision,
        "source_scope": "ksv4_4h",
        "endpoint": "basis",
        "symbol": "BTC",
        "realtime_receipt_id": "real",
    }]
    usage = [{
        "freeze_version": "v1",
        "signal_equivalence_id": "signal_001",
        "horizon": "4h",
        "decision_ts": decision,
        "receipt_id": "real",
    }]
    material = build_source_equivalence_record(
        freeze_version="v1",
        signal_equivalence_id="signal_001",
        source_id="ksv4_4h:basis",
        symbol="BTC",
        native_bar_end_ts=decision,
        realtime_receipt_id="real",
        realtime_values={"value": 1.0},
        realtime_decision_projection={"members": ["BTC"]},
        reference_receipt_id="ref",
        reference_values={"value": 2.0},
        reference_decision_projection={"members": ["ETH"]},
        reference_role="initial",
        observed_ts="2026-07-31T04:15:01Z",
    )
    amendments = build_source_equivalence_consistency_amendments(
        [original], usage, queue, [material]
    )
    assert amendments[0]["amended_dimension_status"] == "fail"
    assert amendments[0]["amended_overall_status"] == "replay_live_consistency_fail"


def test_source_equivalence_amendment_is_candidate_specific() -> None:
    decision = "2026-07-31T04:00:00+00:00"
    originals = [
        build_replay_live_consistency_record(
            freeze_version="v1",
            signal_equivalence_id=candidate_id,
            horizon="4h",
            decision_ts=decision,
            live_dimensions={name: {"value": 1} for name in CONSISTENCY_DIMENSIONS},
            replay_dimensions={name: {"value": 1} for name in CONSISTENCY_DIMENSIONS},
            evidence_sha256={name: "c" * 64 for name in CONSISTENCY_DIMENSIONS},
            dimension_overrides={"data_availability_consistency": "pending_reference"},
        )
        for candidate_id in ("signal_a", "signal_b")
    ]
    queue = [{
        "freeze_version": "v1",
        "decision_ts": decision,
        "source_scope": "ksv4_4h",
        "endpoint": "basis",
        "symbol": "BTC",
        "realtime_receipt_id": "real",
    }]
    usage = [
        {
            "freeze_version": "v1",
            "signal_equivalence_id": candidate_id,
            "horizon": "4h",
            "decision_ts": decision,
            "receipt_id": "real",
        }
        for candidate_id in ("signal_a", "signal_b")
    ]
    equivalence = [
        build_source_equivalence_record(
            freeze_version="v1",
            signal_equivalence_id=candidate_id,
            source_id="ksv4_4h:basis",
            symbol="BTC",
            native_bar_end_ts=decision,
            realtime_receipt_id="real",
            realtime_values={"value": 1.0},
            realtime_decision_projection={"members": ["BTC"]},
            reference_receipt_id=f"ref-{candidate_id}",
            reference_values={"value": 2.0},
            reference_decision_projection={
                "members": ["ETH"] if candidate_id == "signal_a" else ["BTC"]
            },
            reference_role="initial",
            observed_ts="2026-07-31T04:15:01Z",
        )
        for candidate_id in ("signal_a", "signal_b")
    ]
    amendments = build_source_equivalence_consistency_amendments(
        originals, usage, queue, equivalence
    )
    status_by_candidate = {
        row["signal_equivalence_id"]: row["amended_dimension_status"]
        for row in amendments
    }
    assert status_by_candidate == {"signal_a": "fail", "signal_b": "pass"}


def test_replay_live_consistency_preserves_missed_and_not_scheduled_states() -> None:
    missed = build_replay_live_consistency_record(
        freeze_version="v2",
        signal_equivalence_id="signal_001",
        horizon="8h",
        decision_ts="2026-08-01T08:00:00Z",
        live_dimensions={},
        replay_dimensions={},
        evidence_sha256={},
        missed_decision=True,
    )
    assert missed["overall_status"] == "missed_decision"
    not_scheduled = build_replay_live_consistency_record(
        freeze_version="v2",
        signal_equivalence_id="signal_001",
        horizon="8h",
        decision_ts="2026-08-01T08:00:00Z",
        live_dimensions={},
        replay_dimensions={},
        evidence_sha256={},
        scheduled=False,
    )
    assert not_scheduled["overall_status"] == "not_scheduled"
    assert all(
        row["status"] == "not_applicable"
        for row in not_scheduled["dimensions"].values()
    )


def test_memberships_are_deterministic_four_by_four() -> None:
    result = select_long_short_memberships(_scores(), long_count=4, short_count=4)
    assert result.loc[result["leg"].eq("short"), "symbol"].tolist() == list(
        CANONICAL_SYMBOLS[:4]
    )
    assert result.loc[result["leg"].eq("long"), "symbol"].tolist() == list(
        CANONICAL_SYMBOLS[-4:]
    )


def test_unchanged_membership_holds_original_quantities_and_charges_nothing() -> None:
    membership = select_long_short_memberships(_scores(), long_count=4, short_count=4)
    current = {
        row.symbol: (0.7 if row.leg == "long" else -0.8)
        for row in membership.itertuples(index=False)
    }
    transitions = plan_candidate_transitions(
        current,
        membership,
        _quotes(),
        _rules(),
        virtual_submit_ts="2026-08-01T00:00:01Z",
        target_gross_notional=600.0,
    )
    assert set(transitions["status"]) == {"hold_unchanged"}
    assert transitions["executed_quantity"].abs().sum() == 0.0
    assert transitions["executed_notional"].sum() == 0.0
    assert transitions.set_index("symbol")["desired_signed_quantity"].to_dict() == current


def test_bid_ask_open_close_and_dynamic_minimum_are_hand_computable() -> None:
    membership = select_long_short_memberships(_scores(), long_count=4, short_count=4)
    transitions = plan_candidate_transitions(
        {},
        membership,
        _quotes(100.0),
        _rules(step=0.1),
        virtual_submit_ts="2026-08-01T00:00:01Z",
        target_gross_notional=600.0,
    )
    long_rows = transitions[transitions["symbol"].isin(CANONICAL_SYMBOLS[-4:])]
    short_rows = transitions[transitions["symbol"].isin(CANONICAL_SYMBOLS[:4])]
    assert (long_rows["execution_price"] == 101.0).all()
    assert (short_rows["execution_price"] == 99.0).all()
    assert (long_rows["desired_signed_quantity"] == 0.7).all()
    assert (short_rows["desired_signed_quantity"] == -0.7).all()

    expensive_rules = _rules(step=0.1, minimum=100.0)
    filtered = plan_candidate_transitions(
        {},
        membership,
        _quotes(100.0),
        expensive_rules,
        virtual_submit_ts="2026-08-01T00:00:01Z",
        target_gross_notional=600.0,
    )
    assert set(filtered["status"]) == {"filtered_keep_previous"}
    assert filtered["executed_quantity"].abs().sum() == 0.0


def test_partial_untradable_symbol_does_not_cancel_other_transitions() -> None:
    membership = select_long_short_memberships(_scores(), long_count=4, short_count=4)
    missing_symbol = CANONICAL_SYMBOLS[0]
    quotes = _quotes()
    del quotes[missing_symbol]
    transitions = plan_candidate_transitions(
        {},
        membership,
        quotes,
        _rules(),
        virtual_submit_ts="2026-08-01T00:00:01Z",
        target_gross_notional=600.0,
    )
    blocked = transitions.loc[transitions["symbol"].eq(missing_symbol)].iloc[0]
    assert blocked["status"] == "failed_missing_quote"
    assert blocked["executed_quantity"] == 0.0
    assert transitions["executed_quantity"].ne(0.0).sum() == 7

    stale = plan_candidate_transitions(
        {},
        membership,
        _quotes(),
        _rules(),
        virtual_submit_ts="2026-08-01T00:00:03Z",
        target_gross_notional=600.0,
    )
    assert set(stale["status"]) == {"failed_quote_requested_before_submit"}
    future_rules = {
        symbol: ExchangeRule(
            symbol,
            "TRADING",
            0.01,
            0.01,
            5.0,
            "2026-08-01T00:00:02+00:00",
        )
        for symbol in CANONICAL_SYMBOLS
    }
    late_rules = plan_candidate_transitions(
        {},
        membership,
        _quotes(),
        future_rules,
        virtual_submit_ts="2026-08-01T00:00:01Z",
        target_gross_notional=600.0,
    )
    assert set(late_rules["status"]) == {"blocked_rule_observed_after_submit"}
    assert stale["executed_quantity"].eq(0.0).all()


def test_fill_accounting_and_three_cost_ledgers() -> None:
    state = apply_fill_to_position(
        {}, signed_fill_quantity=2.0, fill_price=100.0, fee=0.1
    )
    state = apply_fill_to_position(
        state, signed_fill_quantity=-1.0, fill_price=110.0, fee=0.055
    )
    positions = {"BTC": state}
    assert state["quantity"] == 1.0
    assert state["realized_pnl"] == 10.0
    assert marked_equity(positions, {"BTC": 105.0}, initial_equity=272.0) == pytest.approx(
        286.845
    )
    assert marked_equity(
        positions, {"BTC": 105.0}, initial_equity=272.0, fee_multiplier=2.0
    ) == pytest.approx(286.69)


def test_canary_nets_candidates_without_erasing_virtual_transitions() -> None:
    frame = pd.DataFrame(
        [
            {"signal_equivalence_id": "a", "symbol": "BTC", "executed_quantity": 1.0},
            {"signal_equivalence_id": "b", "symbol": "BTC", "executed_quantity": -1.0},
            {"signal_equivalence_id": "a", "symbol": "ETH", "executed_quantity": 2.0},
        ]
    )
    result = aggregate_testnet_canary(
        frame, _quotes(100.0), _rules(step=0.01), canary_gross_notional=600.0
    )
    assert result["symbol"].tolist() == ["ETH"]
    assert result.loc[0, "reference_notional"] <= 600.0
    assert len(frame) == 3


def test_two_candidate_shadow_decision_is_hand_computable_and_isolated(tmp_path) -> None:
    decision = pd.Timestamp("2026-08-01T00:00:00Z")
    submit = pd.Timestamp("2026-08-01T00:00:01Z")
    candidates = pd.DataFrame(
        {
            "freeze_version": ["v1", "v1"],
            "signal_equivalence_id": ["a", "b"],
            "account_equity": [272.0, 272.0],
            "target_gross_notional": [600.0, 600.0],
            "taker_fee_rate": [0.0005, 0.0005],
        }
    )
    signal_rows = []
    for candidate_id, reverse in (("a", False), ("b", True)):
        raw = _scores()
        if reverse:
            raw["signal_value"] *= -1
        membership = select_long_short_memberships(
            raw, long_count=4, short_count=4
        ).set_index("symbol")["leg"].to_dict()
        for row in raw.itertuples(index=False):
            signal_rows.append(
                {
                    "freeze_version": "v1",
                    "signal_equivalence_id": candidate_id,
                    "decision_ts": decision,
                    "symbol": row.symbol,
                    "signal_value": row.signal_value,
                    "leg": membership.get(row.symbol, "flat"),
                }
            )
    signals = pd.DataFrame(signal_rows)
    initial = initial_shadow_state(candidates)
    transitions, events, state, equity = plan_shadow_decision(
        signals,
        candidates,
        initial,
        _quotes(),
        _rules(),
        decision_ts=decision,
        virtual_submit_ts=submit,
    )
    assert len(events) == 16
    assert transitions["executed_quantity"].ne(0.0).sum() == 16
    assert len(equity) == 6
    assert state["candidates"]["a"]["positions"] != state["candidates"]["b"]["positions"]
    assert equity["equity"].lt(272.0).all()

    ledger = AppendOnlyEventLedger(tmp_path / "events.jsonl", tmp_path / "state.json")
    persisted = persist_shadow_fill_events(
        ledger, events, initial_state=initial
    )
    assert persisted == state
    assert persist_shadow_fill_events(
        ledger, events, initial_state=initial
    ) == state
    assert len(ledger.read_events()) == 16


def test_shared_shadow_decision_artifacts_replay_exactly(tmp_path) -> None:
    decision = pd.Timestamp("2026-08-01T00:00:00Z")
    ready = decision + pd.Timedelta(seconds=2)
    submit = decision + pd.Timedelta(seconds=3)
    candidates = pd.DataFrame(
        {
            "freeze_version": ["v1"],
            "signal_equivalence_id": ["a"],
            "account_equity": [272.0],
            "target_gross_notional": [600.0],
            "taker_fee_rate": [0.0005],
            "horizon": ["4h"],
        }
    )
    raw = _scores()
    membership = select_long_short_memberships(
        raw, long_count=4, short_count=4
    ).set_index("symbol")["leg"].to_dict()
    signals = raw.assign(
        freeze_version="v1",
        signal_equivalence_id="a",
        decision_ts=decision,
        leg=raw["symbol"].map(membership).fillna("flat"),
    )
    receipt = AsReceivedSnapshotStore(tmp_path / "raw").persist(
        b'{"code":"0","data":[]}',
        source_id="keystore_v4",
        source_request_ts=decision,
        source_response_ts=decision + pd.Timedelta(seconds=1),
        source_bar_label_ts=decision - pd.Timedelta(hours=1),
        native_bar_end_ts=decision,
    )
    input_lineage = {
        "feature_cutoff_ts": decision.isoformat(),
        "fields": [
            {
                "field_name": "fixture_signal",
                "source_receipt_id": receipt.receipt_id,
            }
        ],
        "revision_receipt_ids": [],
    }
    signal_lineage = {
        "a": {
            "factor_values": {"fixture": 1.0},
            "directions": {"fixture": 1},
            "weights": {"fixture": 1.0},
            "parameter_sha256": "b" * 64,
        }
    }
    initial = initial_shadow_state(candidates)
    live = build_shadow_decision_artifacts(
        signals,
        candidates,
        initial,
        _quotes(),
        _rules(),
        [receipt],
        input_lineage,
        signal_lineage,
        horizon="4h",
        decision_ts=decision,
        signal_ready_ts=ready,
        virtual_submit_ts=submit,
    )
    replay = build_shadow_decision_artifacts(
        signals,
        candidates,
        initial,
        _quotes(),
        _rules(),
        [receipt],
        input_lineage,
        signal_lineage,
        horizon="4h",
        decision_ts=decision,
        signal_ready_ts=ready,
        virtual_submit_ts=submit,
    )
    assert live.transitions.equals(replay.transitions)
    assert live.state == replay.state
    assert live.dimensions == replay.dimensions
    evidence = _evidence(live.dimensions["a"], replay.dimensions["a"])
    record = build_replay_live_consistency_record(
        freeze_version="v1",
        signal_equivalence_id="a",
        horizon="4h",
        decision_ts=decision,
        live_dimensions=live.dimensions["a"],
        replay_dimensions=replay.dimensions["a"],
        evidence_sha256=evidence,
    )
    assert record["overall_status"] == "replay_live_consistency_pass"
    altered = json.loads(json.dumps(replay.dimensions["a"]))
    altered["position_consistency"]["before"]["positions"]["BTC"] = {
        "quantity": 1.0
    }
    failed_evidence = _evidence(live.dimensions["a"], altered)
    failed = build_replay_live_consistency_record(
        freeze_version="v1",
        signal_equivalence_id="a",
        horizon="4h",
        decision_ts=decision,
        live_dimensions=live.dimensions["a"],
        replay_dimensions=altered,
        evidence_sha256=failed_evidence,
    )
    assert failed["overall_status"] == "replay_live_consistency_fail"
    assert failed["dimensions"]["position_consistency"]["status"] == "fail"


def test_shared_shadow_decision_artifacts_rejects_wrong_identity(tmp_path) -> None:
    decision = pd.Timestamp("2026-08-01T00:00:00Z")
    candidates = pd.DataFrame(
        {
            "freeze_version": ["v1"],
            "signal_equivalence_id": ["a"],
            "account_equity": [272.0],
            "target_gross_notional": [600.0],
            "taker_fee_rate": [0.0005],
            "horizon": ["8h"],
        }
    )
    raw = _scores()
    membership = select_long_short_memberships(
        raw, long_count=4, short_count=4
    ).set_index("symbol")["leg"].to_dict()
    signals = raw.assign(
        freeze_version="wrong",
        signal_equivalence_id="a",
        decision_ts=decision,
        leg=raw["symbol"].map(membership).fillna("flat"),
    )
    receipt = AsReceivedSnapshotStore(tmp_path / "raw").persist(
        b'{"fixture":true}',
        source_id="fixture",
        source_request_ts=decision,
        source_response_ts=decision,
        source_bar_label_ts=decision,
        native_bar_end_ts=decision,
    )
    input_lineage = {
        "feature_cutoff_ts": decision.isoformat(),
        "fields": [
            {"field_name": "fixture", "source_receipt_id": receipt.receipt_id}
        ],
    }
    signal_lineage = {
        "a": {
            "factor_values": {},
            "directions": {},
            "weights": {},
            "parameter_sha256": "b" * 64,
        }
    }
    with pytest.raises(ValueError, match="candidate horizon mismatch"):
        build_shadow_decision_artifacts(
            signals,
            candidates,
            initial_shadow_state(candidates),
            _quotes(),
            _rules(),
            [receipt],
            input_lineage,
            signal_lineage,
            horizon="4h",
            decision_ts=decision,
            signal_ready_ts=decision,
            virtual_submit_ts=decision,
        )
    candidates["horizon"] = "4h"
    with pytest.raises(ValueError, match="signal freeze version mismatch"):
        build_shadow_decision_artifacts(
            signals,
            candidates,
            initial_shadow_state(candidates),
            _quotes(),
            _rules(),
            [receipt],
            input_lineage,
            signal_lineage,
            horizon="4h",
            decision_ts=decision,
            signal_ready_ts=decision,
            virtual_submit_ts=decision,
        )


def test_event_ledger_is_idempotent_and_restart_safe(tmp_path) -> None:
    ledger = AppendOnlyEventLedger(tmp_path / "events.jsonl", tmp_path / "state.json")
    event = {
        "freeze_version": "v1",
        "signal_equivalence_id": "signal_001",
        "decision_ts": "2026-08-01T00:00:00+00:00",
        "symbol": "BTC",
        "transition": "open",
        "quantity": 0.1,
    }
    assert ledger.append(event, {"positions": {"BTC": 0.1}})
    assert not ledger.append(event, {"positions": {"BTC": 0.1}})
    assert ledger.load_state() == {"positions": {"BTC": 0.1}}
    assert len(ledger.read_events()) == 1
    changed = dict(event, quantity=0.2)
    with pytest.raises(ValueError, match="different payload"):
        ledger.append(changed, {"positions": {"BTC": 0.2}})
    assert json.loads((tmp_path / "events.jsonl").read_text().splitlines()[0])["quantity"] == 0.1


def test_event_ledger_rebuilds_stale_snapshot_from_events(tmp_path) -> None:
    ledger = AppendOnlyEventLedger(tmp_path / "events.jsonl", tmp_path / "state.json")
    first = {
        "freeze_version": "v1",
        "signal_equivalence_id": "signal_001",
        "decision_ts": "2026-08-01T00:00:00+00:00",
        "symbol": "BTC",
        "transition": "open",
        "quantity": 0.1,
    }
    second = dict(
        first,
        decision_ts="2026-08-01T04:00:00+00:00",
        transition="close",
        quantity=-0.1,
    )
    ledger.append(first, {"quantity": 0.1})
    ledger.append(second, {"quantity": 0.0})
    ledger.write_state({"quantity": 999.0})

    def reducer(state, event):
        return {"quantity": float(state.get("quantity", 0.0)) + float(event["quantity"])}

    assert ledger.restore_state(reducer, initial_state={"quantity": 0.0}) == {
        "quantity": 0.0
    }
    assert ledger.load_state() == {"quantity": 0.0}


def test_event_ledger_rejects_duplicate_keys_already_on_disk(tmp_path) -> None:
    ledger = AppendOnlyEventLedger(tmp_path / "events.jsonl", tmp_path / "state.json")
    event = {
        "freeze_version": "v1",
        "signal_equivalence_id": "signal_001",
        "decision_ts": "2026-08-01T00:00:00+00:00",
        "symbol": "BTC",
        "transition": "open",
        "quantity": 0.1,
    }
    line = json.dumps(event, sort_keys=True)
    ledger.event_path.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate event"):
        ledger.read_events()


def test_event_ledger_appends_one_decision_batch_atomically_and_idempotently(
    tmp_path,
) -> None:
    ledger = AppendOnlyEventLedger(tmp_path / "events.jsonl", tmp_path / "state.json")
    events = [
        {
            "freeze_version": "v1",
            "signal_equivalence_id": "signal_001",
            "decision_ts": "2026-08-01T00:00:00+00:00",
            "symbol": symbol,
            "transition": "open",
        }
        for symbol in ("BTC", "ETH")
    ]
    final_state = {"positions": {"BTC": 0.1, "ETH": -1.0}}
    assert ledger.append_batch(events, final_state) == 2
    assert ledger.append_batch(events, final_state) == 0
    assert ledger.read_events() == events
    assert ledger.load_state() == final_state

    ledger.event_path.write_text(
        ledger.event_path.read_text(encoding="utf-8").splitlines()[0] + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="partially present"):
        ledger.append_batch(events, final_state)


def test_event_ledger_supports_consistency_record_identity(tmp_path) -> None:
    ledger = AppendOnlyEventLedger(
        tmp_path / "consistency.jsonl",
        tmp_path / "consistency-state.json",
        key_fields=("freeze_version", "signal_equivalence_id", "decision_ts"),
    )
    record = {
        "freeze_version": "v1",
        "signal_equivalence_id": "signal_001",
        "decision_ts": "2026-08-01T00:00:00+00:00",
        "overall_status": "replay_live_consistency_pass",
    }
    assert ledger.append(record, {"records": 1})
    assert not ledger.append(record, {"records": 1})
    with pytest.raises(ValueError, match="different payload"):
        ledger.append(
            dict(record, overall_status="replay_live_consistency_fail"),
            {"records": 1},
        )


def test_freshness_and_missed_decision_fail_closed() -> None:
    records = [
        SourceFreshness(
            "keystore",
            "2026-08-01T00:00:00Z",
            "2026-08-01T00:00:20Z",
            "2026-08-01T00:00:21Z",
            60.0,
            30.0,
        ),
        SourceFreshness(
            "binance",
            "2026-08-01T00:00:00Z",
            "2026-08-01T00:00:01Z",
            "2026-08-01T00:00:21Z",
            5.0,
            30.0,
        ),
    ]
    validated = validate_source_freshness(
        records, required_sources=("keystore", "binance")
    )
    assert validated["fresh"].all()
    stale = [
        SourceFreshness(
            "keystore",
            "2026-08-01T00:00:00Z",
            "2026-08-01T00:01:01Z",
            "2026-08-01T00:01:02Z",
            60.0,
            30.0,
        )
    ]
    with pytest.raises(RuntimeError, match="keystore"):
        validate_source_freshness(stale, required_sources=("keystore",))
    stale_age = [
        SourceFreshness(
            "binance",
            "2026-08-01T00:00:00Z",
            "2026-08-01T00:00:01Z",
            "2026-08-01T00:01:00Z",
            5.0,
            30.0,
        )
    ]
    with pytest.raises(RuntimeError, match="binance"):
        validate_source_freshness(stale_age, required_sources=("binance",))
    ready = classify_decision_readiness(
        horizon="4h",
        decision_ts="2026-08-01T00:00:00Z",
        signal_ready_ts="2026-08-01T00:00:20Z",
    )
    missed = classify_decision_readiness(
        horizon="4h",
        decision_ts="2026-08-01T00:00:00Z",
        signal_ready_ts="2026-08-01T04:00:00Z",
    )
    assert ready.status == "ready"
    assert missed.status == "missed_decision"
    with pytest.raises(ValueError, match="precede"):
        classify_decision_readiness(
            horizon="4h",
            decision_ts="2026-08-01T00:00:00Z",
            signal_ready_ts="2026-07-31T23:59:59Z",
        )
    with pytest.raises(ValueError, match="native 4h phase"):
        classify_decision_readiness(
            horizon="4h",
            decision_ts="2026-08-01T01:00:00Z",
            signal_ready_ts="2026-08-01T01:00:01Z",
        )


def test_activation_manifest_requires_every_preflight(tmp_path) -> None:
    paths = []
    for name in ("candidates.csv", "parameters.csv", "rules.json", "reference.json"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        paths.append(path)
    passing_checks = {
        name: True for name in REQUIRED_ACTIVATION_PREFLIGHTS
    }
    failed_checks = dict(passing_checks)
    failed_checks["testnet_connectivity_verified"] = False
    intent_path = tmp_path / "activation_intent.json"
    intent = build_activation_intent(
        freeze_version="v1",
        shadow_start_ts="2026-08-01T10:15:00Z",
        evidence_paths={"candidate": paths[0]},
        reviewed_code_sha="abc",
        config_sha="def",
        environment_id="hk-testnet-shadow",
    )
    write_immutable_json_with_sha256(intent, intent_path)
    with pytest.raises(RuntimeError, match="testnet_connectivity"):
        build_activation_manifest(
            freeze_version="v1",
            shadow_start_ts="2026-08-01T10:15:00Z",
            candidate_manifest_path=paths[0],
            parameter_manifest_path=paths[1],
            exchange_rules_path=paths[2],
            source_reference_plan_path=paths[3],
            activation_intent_path=intent_path,
            code_sha="abc",
            config_sha="def",
            environment_id="hk-testnet-shadow",
            manifest_generated_ts="2026-08-01T10:15:00Z",
            preflight_checks=failed_checks,
        )
    manifest = build_activation_manifest(
        freeze_version="v1",
        shadow_start_ts="2026-08-01T10:15:00Z",
        candidate_manifest_path=paths[0],
        parameter_manifest_path=paths[1],
        exchange_rules_path=paths[2],
        source_reference_plan_path=paths[3],
        activation_intent_path=intent_path,
        code_sha="abc",
        config_sha="def",
        environment_id="hk-testnet-shadow",
        manifest_generated_ts="2026-08-01T10:15:00Z",
        preflight_checks=passing_checks,
    )
    assert manifest["shadow_start_ts"] == "2026-08-01T10:15:00+00:00"
    assert manifest["first_eligible_decision_ts"] == {
        "4h": "2026-08-01T12:00:00+00:00",
        "8h": "2026-08-01T16:00:00+00:00",
        "12h": "2026-08-01T12:00:00+00:00",
        "1d": "2026-08-02T00:00:00+00:00",
    }
    assert set(manifest["input_sha256"]) == {
        "activation_intent",
        "candidate_manifest",
        "parameter_manifest",
        "exchange_rules",
        "source_reference_plan",
    }
    manifest["evidence_sha256"] = intent["evidence_sha256"]
    manifest["evidence_paths"] = intent["evidence_paths"]
    activation_path = tmp_path / "activation.json"
    activation_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_activation_manifest(
        activation_path, freeze_version="v1"
    )["shadow_start_ts"] == "2026-08-01T10:15:00+00:00"
    paths[0].write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="input SHA mismatch"):
        verify_activation_manifest(activation_path, freeze_version="v1")
    paths[0].write_text("candidates.csv", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot precede"):
        build_activation_manifest(
            freeze_version="v1",
            shadow_start_ts="2026-08-01T10:15:00Z",
            candidate_manifest_path=paths[0],
            parameter_manifest_path=paths[1],
            exchange_rules_path=paths[2],
            source_reference_plan_path=paths[3],
            activation_intent_path=intent_path,
            code_sha="abc",
            config_sha="def",
            environment_id="hk-testnet-shadow",
            manifest_generated_ts="2026-08-01T10:14:59Z",
            preflight_checks=passing_checks,
        )
    with pytest.raises(ValueError, match="preflight set mismatch"):
        build_activation_manifest(
            freeze_version="v1",
            shadow_start_ts="2026-08-01T10:15:00Z",
            candidate_manifest_path=paths[0],
            parameter_manifest_path=paths[1],
            exchange_rules_path=paths[2],
            source_reference_plan_path=paths[3],
            activation_intent_path=intent_path,
            code_sha="abc",
            config_sha="def",
            environment_id="hk-testnet-shadow",
            manifest_generated_ts="2026-08-01T10:15:00Z",
            preflight_checks={"arbitrary": True},
        )


def test_activation_intent_is_immutable_and_bound_to_exact_evidence(tmp_path) -> None:
    evidence = {}
    for name in ("canary.json", "review.json", "config.json"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        evidence[name] = path
    intent = build_activation_intent(
        freeze_version="v1",
        shadow_start_ts="2026-08-01T10:15:00Z",
        evidence_paths=evidence,
        reviewed_code_sha="reviewed-code",
        config_sha=sha256_file(evidence["config.json"]),
        environment_id="hk-testnet-shadow",
    )
    intent_path = tmp_path / "activation_intent.json"
    write_immutable_json_with_sha256(intent, intent_path)
    verified = verify_activation_intent(
        intent_path,
        freeze_version="v1",
        evidence_paths=evidence,
        reviewed_code_sha="reviewed-code",
        config_sha=sha256_file(evidence["config.json"]),
        environment_id="hk-testnet-shadow",
    )
    assert verified == intent
    write_immutable_json_with_sha256(intent, intent_path)
    evidence["review.json"].write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="current evidence identity"):
        verify_activation_intent(
            intent_path,
            freeze_version="v1",
            evidence_paths=evidence,
            reviewed_code_sha="reviewed-code",
            config_sha=sha256_file(evidence["config.json"]),
            environment_id="hk-testnet-shadow",
        )


def test_activation_intent_cannot_publish_at_or_near_first_decision() -> None:
    intent = {
        "first_eligible_decision_ts": {
            "4h": "2026-08-01T12:00:00Z",
            "8h": "2026-08-01T16:00:00Z",
            "12h": "2026-08-01T12:00:00Z",
            "1d": "2026-08-02T00:00:00Z",
        }
    }
    assert require_activation_intent_publishable(
        intent, observed_ts="2026-08-01T11:59:29Z"
    ) == pd.Timestamp("2026-08-01T12:00:00Z")
    with pytest.raises(RuntimeError, match="new freeze"):
        require_activation_intent_publishable(
            intent, observed_ts="2026-08-01T11:59:30Z"
        )
    with pytest.raises(RuntimeError, match="new freeze"):
        require_activation_intent_publishable(
            intent, observed_ts="2026-08-01T12:00:00Z"
        )


def test_activation_authority_requires_bound_publish_receipt(tmp_path) -> None:
    inputs = []
    for name in ("candidates.csv", "parameters.csv", "rules.csv", "reference.json"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        inputs.append(path)
    intent_path = tmp_path / "activation_intent.json"
    intent = build_activation_intent(
        freeze_version="v1",
        shadow_start_ts="2026-08-01T10:15:00Z",
        evidence_paths={"candidate": inputs[0]},
        reviewed_code_sha="abc",
        config_sha="def",
        environment_id="hk-testnet-shadow",
    )
    write_immutable_json_with_sha256(intent, intent_path)
    manifest = build_activation_manifest(
        freeze_version="v1",
        shadow_start_ts="2026-08-01T10:15:00Z",
        candidate_manifest_path=inputs[0],
        parameter_manifest_path=inputs[1],
        exchange_rules_path=inputs[2],
        source_reference_plan_path=inputs[3],
        activation_intent_path=intent_path,
        code_sha="abc",
        config_sha="def",
        environment_id="hk-testnet-shadow",
        manifest_generated_ts="2026-08-01T10:15:00Z",
        preflight_checks={
            name: True for name in REQUIRED_ACTIVATION_PREFLIGHTS
        },
    )
    manifest["evidence_sha256"] = intent["evidence_sha256"]
    manifest["evidence_paths"] = intent["evidence_paths"]
    activation_path = tmp_path / "activation.json"
    publish_path = tmp_path / "activation_publish_receipt.json"
    manifest["publish_receipt_path"] = str(publish_path)
    write_immutable_json_with_sha256(manifest, activation_path)
    with pytest.raises(FileNotFoundError, match="bundle is incomplete"):
        verify_activation_manifest(
            activation_path,
            freeze_version="v1",
            require_sha256_sidecar=True,
        )
    write_immutable_json_with_sha256(
        {
            "freeze_version": "v1",
            "activation_manifest_sha256": sha256_file(activation_path),
            "activation_publish_completed_ts": "2026-08-01T10:15:01Z",
        },
        publish_path,
    )
    assert verify_activation_manifest(
        activation_path,
        freeze_version="v1",
        require_sha256_sidecar=True,
    )["shadow_start_ts"] == "2026-08-01T10:15:00+00:00"


def test_freeze_sha_bundle_is_relocatable_and_detects_tampering(tmp_path) -> None:
    source_root = tmp_path / "workspace"
    source_root.mkdir()
    source = source_root / "source.csv"
    source.write_text("a\n1\n", encoding="utf-8")
    manifest = pd.DataFrame(
        {
            "freeze_version": ["v1"] * 47,
            "signal_equivalence_id": [f"signal_{index:03d}" for index in range(47)],
            **{
                column: ["x"] * 47
                for column in (
                    "candidate_id",
                    "canonical_candidate_id",
                    "horizon",
                    "track_id",
                    "alpha_id",
                    "track",
                    "weight_scheme",
                    "component_features",
                    "component_feature_order",
                    "historical_evidence_label",
                    "historical_evidence_group",
                    "universe",
                    "panel_frequency",
                    "data_source_contract",
                    "factor_transform_rule",
                    "direction_rule",
                    "weight_estimation_rule",
                    "decision_interval",
                    "decision_phase_utc",
                    "holding_interval",
                    "position_rule",
                    "execution_rule",
                    "cost_multipliers",
                    "minimum_notional_rule",
                    "runtime_contract_version",
                )
            },
            "long_count": [4] * 47,
            "short_count": [4] * 47,
            "account_equity": [272.0] * 47,
            "target_gross_notional": [600.0] * 47,
            "exchange_leverage": [5.0] * 47,
            "taker_fee_rate": [0.0005] * 47,
            "train_days": [180] * 47,
            "embargo_days": [1] * 47,
            "epoch_days": [45] * 47,
        }
    )
    output = tmp_path / "freeze"
    paths = write_freeze_bundle(
        manifest, output, [source], source_root=source_root
    )
    verify_sha256_manifest(paths["sha256"])
    verify_freeze_source_manifest(
        paths["source_manifest"], source_root=source_root
    )
    source_manifest = json.loads(paths["source_manifest"].read_text())
    assert source_manifest["runtime_contract_version"] == (
        TRUE_OOS_RUNTIME_CONTRACT_VERSION
    )
    assert source_manifest["sources"][0]["path"] == "source.csv"
    with pytest.raises(FileExistsError, match="already exists"):
        write_freeze_bundle(manifest, output, [source], source_root=source_root)
    paths["candidate_manifest"].write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="candidate_manifest"):
        verify_sha256_manifest(paths["sha256"])
    source.write_text("a\n2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source.csv"):
        verify_freeze_source_manifest(
            paths["source_manifest"], source_root=source_root
        )


def test_quantity_rounds_toward_zero() -> None:
    assert quantity_toward_zero(1.239, 0.01) == 1.23
    assert quantity_toward_zero(-1.239, 0.01) == -1.23


def test_book_ticker_parser_requires_complete_valid_universe() -> None:
    payload = [
        {
            "symbol": f"{symbol}USDT",
            "bidPrice": "99",
            "askPrice": "101",
        }
        for symbol in CANONICAL_SYMBOLS
    ]
    quotes = book_quotes_from_binance(
        payload,
        request_ts="2026-08-01T00:00:01+00:00",
        observed_ts="2026-08-01T00:00:02+00:00",
    )
    assert quotes["BTC"].bid == 99.0
    with pytest.raises(ValueError, match="cover"):
        book_quotes_from_binance(
            payload[:-1],
            request_ts="2026-08-01T00:00:01+00:00",
            observed_ts="2026-08-01T00:00:02+00:00",
        )


def test_live_signal_fit_uses_horizon_specific_training_returns() -> None:
    activation = pd.Timestamp("2026-08-01T00:00:00Z")
    epoch = build_epoch_window(activation, 0, train_days=2, embargo_days=1, epoch_days=45)
    train_times = pd.date_range("2026-07-29", periods=12, freq="4h", tz="UTC")
    rows = []
    for timestamp in train_times:
        for index, symbol in enumerate(CANONICAL_SYMBOLS):
            rows.append(
                {
                    "decision_ts": timestamp,
                    "symbol": symbol,
                    "feature__1h": float(index),
                    "forward_return": float(index),
                }
            )
    training = pd.DataFrame(rows).set_index("decision_ts")
    current = pd.DataFrame(
        {
            "decision_ts": activation,
            "symbol": CANONICAL_SYMBOLS,
            "feature__1h": range(len(CANONICAL_SYMBOLS)),
        }
    ).set_index("decision_ts")
    candidates = pd.DataFrame(
        {
            "freeze_version": ["v1"],
            "signal_equivalence_id": ["signal_001"],
            "horizon": ["4h"],
            "component_features": ["feature__1h"],
            "weight_scheme": ["family_alpha_0"],
        }
    )
    registry = pd.DataFrame(
        {"feature_name": ["feature__1h"], "family": ["fixture"]}
    )
    signals, parameters = fit_epoch_candidate_signals(
        {"4h": training},
        current,
        candidates,
        registry,
        epoch,
        activation,
    )
    assert len(signals) == 20
    assert signals.loc[signals["leg"].eq("long"), "symbol"].tolist() == list(
        CANONICAL_SYMBOLS[-4:]
    )
    assert signals["leg"].value_counts().to_dict() == {
        "flat": 12,
        "short": 4,
        "long": 4,
    }
    assert parameters.loc[0, "direction"] == 1
    assert parameters.loc[0, "weight"] == 1.0


def test_epoch_parameters_are_frozen_when_later_training_data_changes() -> None:
    activation = pd.Timestamp("2026-08-01T00:00:00Z")
    epoch = build_epoch_window(activation, 0, train_days=2, embargo_days=1)
    train_times = pd.date_range("2026-07-29", periods=12, freq="4h", tz="UTC")
    training = pd.DataFrame(
        [
            {
                "decision_ts": timestamp,
                "symbol": symbol,
                "feature__1h": float(index),
                "forward_return": float(index),
            }
            for timestamp in train_times
            for index, symbol in enumerate(CANONICAL_SYMBOLS)
        ]
    ).set_index("decision_ts")
    current = pd.DataFrame(
        {
            "decision_ts": activation,
            "symbol": CANONICAL_SYMBOLS,
            "feature__1h": range(len(CANONICAL_SYMBOLS)),
        }
    ).set_index("decision_ts")
    candidates = pd.DataFrame(
        {
            "freeze_version": ["v1"],
            "signal_equivalence_id": ["signal_001"],
            "horizon": ["4h"],
            "component_features": ["feature__1h"],
            "weight_scheme": ["family_alpha_0"],
        }
    )
    registry = pd.DataFrame(
        {"feature_name": ["feature__1h"], "family": ["fixture"]}
    )
    parameters = fit_epoch_candidate_parameters(
        {"4h": training}, candidates, registry, epoch
    )
    baseline = score_epoch_candidate_signals(
        current, candidates, parameters, epoch, activation
    )
    training["forward_return"] *= -1.0
    unchanged = score_epoch_candidate_signals(
        current, candidates, parameters, epoch, activation
    )
    pd.testing.assert_frame_equal(baseline, unchanged)


def test_epoch_training_panel_uses_exact_one_minute_execution_path() -> None:
    activation = pd.Timestamp("2026-08-01T00:00:00Z")
    epoch = build_epoch_window(activation, 0, train_days=2, embargo_days=1)
    feature_times = pd.date_range(
        "2026-07-29T00:00:00Z", "2026-08-01T23:00:00Z", freq="1h"
    )
    feature_panel = pd.DataFrame(
        [
            {
                "decision_ts": timestamp,
                "symbol": symbol,
                "feature__1h": float(index),
            }
            for timestamp in feature_times
            for index, symbol in enumerate(CANONICAL_SYMBOLS)
        ]
    ).set_index(["decision_ts", "symbol"])
    execution_times = pd.date_range(
        "2026-07-29T00:01:00Z", "2026-07-31T00:01:00Z", freq="4h"
    )
    execution = {
        symbol: pd.Series(
            [100.0 + index for index in range(len(execution_times))],
            index=execution_times,
            dtype=float,
        )
        for symbol in CANONICAL_SYMBOLS
    }
    candidates = pd.DataFrame(
        {
            "horizon": ["4h"],
            "component_features": ["feature__1h"],
        }
    )
    registry = pd.DataFrame(
        {
            "feature_name": ["feature__1h"],
            "signal_timeframe": ["1h"],
        }
    )
    panels = build_epoch_training_panels(
        feature_panel, execution, candidates, registry, epoch
    )
    training = panels["4h"]
    assert training.index.min() == pd.Timestamp("2026-07-29T00:00:00Z")
    assert training.index.max() == pd.Timestamp("2026-07-30T20:00:00Z")
    assert training.groupby(level=0)["symbol"].nunique().eq(20).all()
    first = training.iloc[0]
    assert first["execution_ts"] == pd.Timestamp("2026-07-29T00:01:00Z")
    assert first["next_execution_ts"] == pd.Timestamp("2026-07-29T04:01:00Z")
    assert first["forward_return"] == pytest.approx(0.01)


def test_epoch_training_panel_rejects_internal_decision_gap() -> None:
    activation = pd.Timestamp("2026-08-01T00:00:00Z")
    epoch = build_epoch_window(
        activation, 0, train_days=2, embargo_days=1
    )
    feature_times = pd.date_range(
        "2026-07-29T00:00:00Z", "2026-07-30T23:00:00Z", freq="1h"
    )
    feature_panel = pd.DataFrame(
        [
            {
                "decision_ts": timestamp,
                "symbol": symbol,
                "feature__1h": float(index),
            }
            for timestamp in feature_times
            for index, symbol in enumerate(CANONICAL_SYMBOLS)
            if timestamp != pd.Timestamp("2026-07-29T08:00:00Z")
        ]
    ).set_index(["decision_ts", "symbol"])
    execution_times = pd.date_range(
        "2026-07-29T00:01:00Z", "2026-07-31T00:01:00Z", freq="4h"
    )
    execution = {
        symbol: pd.Series(
            range(100, 100 + len(execution_times)),
            index=execution_times,
            dtype=float,
        )
        for symbol in CANONICAL_SYMBOLS
    }
    candidates = pd.DataFrame(
        {"horizon": ["4h"], "component_features": ["feature__1h"]}
    )
    registry = pd.DataFrame(
        {"feature_name": ["feature__1h"], "signal_timeframe": ["1h"]}
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        build_epoch_training_panels(
            feature_panel, execution, candidates, registry, epoch
        )


def test_scoring_accepts_native_intraday_decision_and_binds_parameters() -> None:
    decision = pd.Timestamp("2026-08-01T04:00:00Z")
    epoch = build_epoch_window("2026-08-01T00:00:00Z", 0)
    current = pd.DataFrame(
        {
            "decision_ts": decision,
            "symbol": CANONICAL_SYMBOLS,
            "feature__1h": range(len(CANONICAL_SYMBOLS)),
        }
    ).set_index("decision_ts")
    candidates = pd.DataFrame(
        {
            "freeze_version": ["freeze_A"],
            "signal_equivalence_id": ["signal_001"],
            "horizon": ["4h"],
            "component_features": ["feature__1h"],
        }
    )
    parameters = pd.DataFrame(
        {
            "freeze_version": ["freeze_A"],
            "signal_equivalence_id": ["signal_001"],
            "horizon": ["4h"],
            "epoch_index": [0],
            "feature_name": ["feature__1h"],
            "direction": [1],
            "weight": [1.0],
        }
    )
    signals = score_epoch_candidate_signals(
        current, candidates, parameters, epoch, decision
    )
    assert len(signals) == len(CANONICAL_SYMBOLS)

    wrong = parameters.assign(freeze_version="freeze_B", horizon="1d")
    with pytest.raises(ValueError, match="identity"):
        score_epoch_candidate_signals(
            current, candidates, wrong, epoch, decision
        )

    with pytest.raises(ValueError, match="exact UTC hour"):
        score_epoch_candidate_signals(
            current,
            candidates,
            parameters,
            epoch,
            decision + pd.Timedelta(minutes=1),
        )
