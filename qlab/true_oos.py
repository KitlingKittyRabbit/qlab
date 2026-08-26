"""Formal TRUE OOS candidate, execution, and event-ledger contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
from decimal import Decimal, ROUND_DOWN
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from . import factor_research
from .data.crypto import panel as crypto_panel
from .data.crypto.strategy_time_contract import ContinuousHoldingTimeContract


TRUE_OOS_RUNTIME_CONTRACT_VERSION = "replay_live_consistency_v4"
CONSISTENCY_DIMENSIONS = (
    "data_availability_consistency",
    "decision_clock_consistency",
    "signal_consistency",
    "membership_consistency",
    "position_consistency",
    "order_intent_consistency",
    "execution_accounting_consistency",
    "failure_recovery_consistency",
)
CONSISTENCY_DIMENSION_STATUSES = {
    "pass",
    "fail",
    "pending_reference",
    "not_applicable",
    "not_observable",
}
SOURCE_EQUIVALENCE_STATUSES = {
    "pending_reference",
    "exact_match",
    "value_mismatch_decision_equivalent",
    "decision_material_mismatch",
}
REQUIRED_ACTIVATION_PREFLIGHTS = {
    "freeze_sha_verified",
    "source_sha_verified",
    "epoch_parameters_verified",
    "production_market_snapshot_verified",
    "testnet_connectivity_verified",
    "testnet_one_way_mode_verified",
    "recovery_smoke_verified",
    "consistency_contract_smoke_verified",
    "source_reference_plan_verified",
    "main_review_passed",
    "independent_review_passed",
}


def require_true_oos_runtime_contract(implemented_contract_version: str) -> None:
    """Fail closed when a runtime still implements a superseded contract."""
    if implemented_contract_version != TRUE_OOS_RUNTIME_CONTRACT_VERSION:
        raise RuntimeError(
            "TRUE OOS runtime contract is superseded: "
            f"implemented={implemented_contract_version!r}, "
            f"required={TRUE_OOS_RUNTIME_CONTRACT_VERSION!r}"
        )


CANONICAL_SYMBOLS = (
    "ADA", "APT", "AVAX", "BCH", "BNB", "BTC", "DOGE", "DOT", "ETC", "ETH",
    "FET", "FIL", "LINK", "LTC", "NEAR", "SOL", "SUI", "TRX", "UNI", "XRP",
)
EXPECTED_HORIZON_COUNTS = {"4h": 1, "8h": 7, "12h": 21, "1d": 18}
EXPECTED_CANDIDATE_CONFIG_COUNTS = {"4h": 4, "8h": 28, "12h": 36, "1d": 32}
HORIZON_DELTAS = {
    "4h": pd.Timedelta(hours=4),
    "8h": pd.Timedelta(hours=8),
    "12h": pd.Timedelta(hours=12),
    "1d": pd.Timedelta(days=1),
}
SIGNAL_DELTAS = {"1h": pd.Timedelta(hours=1), **HORIZON_DELTAS}
FREEZE_COLUMNS = (
    "freeze_version",
    "signal_equivalence_id",
    "candidate_id",
    "canonical_candidate_id",
    "horizon",
    "track_id",
    "alpha_id",
    "track",
    "weight_scheme",
    "component_features",
    "component_feature_order",
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
    "long_count",
    "short_count",
    "execution_rule",
    "historical_evidence_label",
    "historical_evidence_group",
    "account_equity",
    "target_gross_notional",
    "exchange_leverage",
    "taker_fee_rate",
    "cost_multipliers",
    "minimum_notional_rule",
    "train_days",
    "embargo_days",
    "epoch_days",
    "runtime_contract_version",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_immutable_input_snapshot(
    paths: Sequence[str | Path],
    *,
    source_root: str | Path,
    destination_root: str | Path,
    observed_ts: str | pd.Timestamp,
) -> pd.DataFrame:
    """Copy exact available input bytes and fail if a source changes mid-copy."""
    source = Path(source_root).resolve()
    destination = Path(destination_root).resolve()
    observed = _utc_timestamp(observed_ts, field="observed_ts")
    rows = []
    resolved_paths = sorted({Path(path).resolve() for path in paths})
    if not resolved_paths:
        raise ValueError("input snapshot requires at least one file")
    for path in resolved_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            relative = path.relative_to(source)
        except ValueError as exc:
            raise ValueError(f"input path escapes source root: {path}") from exc
        before_sha = sha256_file(path)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256_file(target) != before_sha:
                raise ValueError(
                    f"immutable snapshot path has different content: {target}"
                )
        else:
            temporary = target.with_suffix(target.suffix + f".{uuid4().hex}.tmp")
            shutil.copyfile(path, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        after_sha = sha256_file(path)
        target_sha = sha256_file(target)
        if before_sha != after_sha or before_sha != target_sha:
            raise RuntimeError(f"input changed while being snapshotted: {path}")
        rows.append(
            {
                "source_relative_path": relative.as_posix(),
                "snapshot_path": str(target),
                "observed_ts": observed.isoformat(),
                "size_bytes": target.stat().st_size,
                "sha256": target_sha,
            }
        )
    return pd.DataFrame(rows)


def write_immutable_json_with_sha256(
    payload: object,
    path: str | Path,
) -> tuple[Path, Path]:
    """Atomically publish immutable JSON plus a bound SHA-256 sidecar."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            default=str,
        )
        + "\n"
    ).encode("utf-8")
    _write_immutable_bytes(output, encoded)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sha = hashlib.sha256(encoded).hexdigest()
    _write_immutable_bytes(sidecar, f"{sha}  {output.name}\n".encode("ascii"))
    return output, sidecar


def verify_json_sha256_sidecar(path: str | Path) -> str:
    output = Path(path)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if not output.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"JSON/SHA bundle is incomplete: {output}")
    fields = sidecar.read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[1] != output.name:
        raise RuntimeError(f"invalid JSON SHA sidecar: {sidecar}")
    actual = sha256_file(output)
    if fields[0] != actual:
        raise RuntimeError(f"JSON SHA sidecar mismatch: {output}")
    return actual


@dataclass(frozen=True)
class EpochWindow:
    epoch_index: int
    shadow_start_ts: str
    train_start: str
    train_end_exclusive: str
    embargo_start: str
    embargo_end_exclusive: str
    run_start: str
    run_end_exclusive: str


@dataclass(frozen=True)
class SourceFreshness:
    source_id: str
    native_bar_end_ts: str
    data_observed_ts: str
    checked_ts: str
    maximum_delay_seconds: float
    maximum_age_seconds: float


@dataclass(frozen=True)
class DecisionReadiness:
    horizon: str
    decision_ts: str
    next_decision_ts: str
    signal_ready_ts: str
    status: str
    delay_seconds: float


@dataclass(frozen=True)
class SourceReceipt:
    receipt_id: str
    source_id: str
    source_request_ts: str
    source_response_ts: str
    source_bar_label_ts: str
    native_bar_end_ts: str
    data_observed_ts: str
    payload_sha256: str
    payload_path: str
    receipt_path: str


@dataclass(frozen=True)
class ShadowDecisionArtifacts:
    transitions: pd.DataFrame
    events: tuple[dict[str, object], ...]
    state: dict[str, object]
    equity: pd.DataFrame
    dimensions: dict[str, object]


def build_epoch_window(
    shadow_start_ts: str | pd.Timestamp,
    epoch_index: int,
    *,
    train_days: int = 180,
    embargo_days: int = 1,
    epoch_days: int = 45,
) -> EpochWindow:
    if epoch_index < 0:
        raise ValueError("epoch_index must be non-negative")
    if min(train_days, embargo_days, epoch_days) <= 0:
        raise ValueError("train, embargo, and epoch days must be positive")
    shadow_start = _utc_timestamp(shadow_start_ts, field="shadow_start_ts")
    run_start = shadow_start + pd.Timedelta(days=epoch_index * epoch_days)
    embargo_end = run_start.normalize()
    embargo_start = embargo_end - pd.Timedelta(days=embargo_days)
    train_start = embargo_start - pd.Timedelta(days=train_days)
    return EpochWindow(
        epoch_index=epoch_index,
        shadow_start_ts=shadow_start.isoformat(),
        train_start=train_start.isoformat(),
        train_end_exclusive=embargo_start.isoformat(),
        embargo_start=embargo_start.isoformat(),
        embargo_end_exclusive=embargo_end.isoformat(),
        run_start=run_start.isoformat(),
        run_end_exclusive=(run_start + pd.Timedelta(days=epoch_days)).isoformat(),
    )


def first_eligible_decision_ts(
    horizon: str,
    shadow_start_ts: str | pd.Timestamp,
) -> pd.Timestamp:
    """Return the first native decision strictly after authoritative startup."""
    if horizon not in HORIZON_DELTAS:
        raise ValueError(f"unsupported horizon: {horizon}")
    start = _utc_timestamp(shadow_start_ts, field="shadow_start_ts")
    delta = HORIZON_DELTAS[horizon]
    epoch = pd.Timestamp("1970-01-01T00:00:00Z")
    elapsed = start - epoch
    completed = elapsed // delta
    return epoch + (completed + 1) * delta


def due_candidate_rows(
    candidate_manifest: pd.DataFrame,
    *,
    shadow_start_ts: str | pd.Timestamp,
    decision_ts: str | pd.Timestamp,
) -> pd.DataFrame:
    """Return candidates scheduled at one native boundary after activation."""
    required = {"signal_equivalence_id", "horizon"}
    if not required.issubset(candidate_manifest.columns):
        raise ValueError("candidate manifest missing scheduling columns")
    decision = _utc_timestamp(decision_ts, field="decision_ts")
    start = _utc_timestamp(shadow_start_ts, field="shadow_start_ts")
    epoch = pd.Timestamp("1970-01-01T00:00:00Z")
    due = []
    for row in candidate_manifest.itertuples(index=False):
        horizon = str(row.horizon)
        if horizon not in HORIZON_DELTAS:
            raise ValueError(f"unsupported horizon: {horizon}")
        scheduled = (
            decision >= first_eligible_decision_ts(horizon, start)
            and (decision - epoch) % HORIZON_DELTAS[horizon] == pd.Timedelta(0)
        )
        due.append(scheduled)
    return candidate_manifest.loc[due].copy()


def build_missed_decision_records(
    candidate_manifest: pd.DataFrame,
    *,
    freeze_version: str,
    shadow_start_ts: str | pd.Timestamp,
    decision_ts: str | pd.Timestamp,
    reason: str,
) -> list[dict[str, object]]:
    """Build candidate-level failure-closed records without backfilling."""
    if not reason.strip():
        raise ValueError("missed decision reason must not be empty")
    due = due_candidate_rows(
        candidate_manifest,
        shadow_start_ts=shadow_start_ts,
        decision_ts=decision_ts,
    )
    records = []
    for row in due.itertuples(index=False):
        record = build_replay_live_consistency_record(
            freeze_version=freeze_version,
            signal_equivalence_id=str(row.signal_equivalence_id),
            horizon=str(row.horizon),
            decision_ts=decision_ts,
            live_dimensions={},
            replay_dimensions={},
            evidence_sha256={},
            missed_decision=True,
        )
        record["missed_reason"] = reason
        records.append(record)
    return records


def _utc_timestamp(value: str | pd.Timestamp, *, field: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    timestamp = (
        timestamp.tz_localize("UTC")
        if timestamp.tz is None
        else timestamp.tz_convert("UTC")
    )
    if pd.isna(timestamp):
        raise ValueError(f"{field} must be a valid timestamp")
    return timestamp


def _write_immutable_bytes(path: Path, payload: bytes) -> None:
    """Create one immutable file, allowing only an identical idempotent retry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(
                    f"immutable path already has different content: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def safe_snapshot_source_id(source_id: str, *, role: str) -> str:
    """Return a deterministic path-safe identity without hiding the source label."""
    raw_source = str(source_id).strip()
    raw_role = str(role).strip()
    if not raw_source or not raw_role:
        raise ValueError("snapshot source id and role must be non-empty")
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_source).strip("_.-")
    role_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_role).strip("_.-")
    if not readable or not role_slug:
        raise ValueError("snapshot source id and role must contain safe characters")
    digest = hashlib.sha256(
        f"{raw_role}\0{raw_source}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{role_slug}.{readable}.{digest}"


class AsReceivedSnapshotStore:
    """Persist source bytes before parsing and retain each observation receipt."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def persist(
        self,
        payload: bytes,
        *,
        source_id: str,
        source_request_ts: str | pd.Timestamp,
        source_response_ts: str | pd.Timestamp,
        source_bar_label_ts: str | pd.Timestamp,
        native_bar_end_ts: str | pd.Timestamp,
        evidence_role: str = "as_received",
    ) -> SourceReceipt:
        if not payload:
            raise ValueError("as-received payload must not be empty")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", source_id):
            raise ValueError("source_id contains unsafe characters")
        if evidence_role not in {"as_received", "normalized_projection"}:
            raise ValueError("unsupported snapshot evidence role")
        requested = _utc_timestamp(source_request_ts, field="source_request_ts")
        responded = _utc_timestamp(source_response_ts, field="source_response_ts")
        bar_label = _utc_timestamp(source_bar_label_ts, field="source_bar_label_ts")
        native_end = _utc_timestamp(native_bar_end_ts, field="native_bar_end_ts")
        if responded < requested:
            raise ValueError("source_response_ts cannot precede source_request_ts")
        if responded < native_end:
            raise ValueError("source payload cannot be observed before native_bar_end_ts")

        payload_sha = hashlib.sha256(payload).hexdigest()
        object_path = self.root / "objects" / payload_sha[:2] / f"{payload_sha}.bin"
        _write_immutable_bytes(object_path, payload)
        normalized = evidence_role == "normalized_projection"
        metadata = {
            "Lifecycle": (
                "authoritative normalized source projection"
                if normalized
                else "authoritative as-received source evidence"
            ),
            "Authority": (
                "Deterministic projection bound to separately retained raw receipts."
                if normalized
                else "Source bytes and observed timing for replay-live audit."
            ),
            "Inputs": (
                "Canonical normalized values plus raw dependency receipt identities."
                if normalized
                else "Unparsed source response bytes and request/response timestamps."
            ),
            "May be used for": "Reconstructing decisions that cite this receipt.",
            "Must not be used for": "Backfilling an earlier decision or rewriting prior receipts.",
            "Archive condition": "Retain with the authoritative event ledger.",
            "source_id": source_id,
            "source_request_ts": requested.isoformat(),
            "source_response_ts": responded.isoformat(),
            "source_bar_label_ts": bar_label.isoformat(),
            "native_bar_end_ts": native_end.isoformat(),
            "data_observed_ts": responded.isoformat(),
            "payload_sha256": payload_sha,
            "payload_path": str(object_path),
            "evidence_role": evidence_role,
        }
        receipt_id = stable_json_sha256(metadata)
        receipt_path = (
            self.root / "receipts" / source_id / f"{receipt_id}.json"
        )
        metadata["receipt_id"] = receipt_id
        encoded = (
            json.dumps(metadata, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        _write_immutable_bytes(receipt_path, encoded)
        return SourceReceipt(
            receipt_id=receipt_id,
            source_id=source_id,
            source_request_ts=requested.isoformat(),
            source_response_ts=responded.isoformat(),
            source_bar_label_ts=bar_label.isoformat(),
            native_bar_end_ts=native_end.isoformat(),
            data_observed_ts=responded.isoformat(),
            payload_sha256=payload_sha,
            payload_path=str(object_path),
            receipt_path=str(receipt_path),
        )

    def load_verified(self, receipt: SourceReceipt) -> bytes:
        """Load one receipt only after path, metadata, identity, and bytes verify."""
        root = self.root.resolve()
        expected_payload = (
            self.root
            / "objects"
            / receipt.payload_sha256[:2]
            / f"{receipt.payload_sha256}.bin"
        ).resolve()
        expected_receipt = (
            self.root
            / "receipts"
            / receipt.source_id
            / f"{receipt.receipt_id}.json"
        ).resolve()
        for path in (expected_payload, expected_receipt):
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ValueError("as-received path escapes the snapshot root") from exc
        if Path(receipt.payload_path).resolve() != expected_payload:
            raise ValueError("source receipt payload path is not canonical")
        if Path(receipt.receipt_path).resolve() != expected_receipt:
            raise ValueError("source receipt metadata path is not canonical")
        metadata = json.loads(expected_receipt.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("source receipt metadata must be an object")
        expected_fields = asdict(receipt)
        for field, value in expected_fields.items():
            if field == "receipt_path":
                continue
            if str(metadata.get(field, "")) != str(value):
                raise ValueError(f"source receipt metadata mismatch: {field}")
        identity_payload = dict(metadata)
        identity_payload.pop("receipt_id", None)
        if stable_json_sha256(identity_payload) != receipt.receipt_id:
            raise ValueError("source receipt identity verification failed")
        payload = expected_payload.read_bytes()
        if hashlib.sha256(payload).hexdigest() != receipt.payload_sha256:
            raise ValueError("source payload SHA-256 verification failed")
        return payload


def build_source_revision_event(
    earlier: SourceReceipt,
    later: SourceReceipt,
) -> dict[str, object] | None:
    """Describe a changed payload for the same source bar without rewriting it."""
    if (
        earlier.source_id != later.source_id
        or earlier.source_bar_label_ts != later.source_bar_label_ts
    ):
        raise ValueError("revision receipts must identify the same source bar")
    if _utc_timestamp(
        later.data_observed_ts, field="later.data_observed_ts"
    ) < _utc_timestamp(earlier.data_observed_ts, field="earlier.data_observed_ts"):
        raise ValueError("revision receipts are not in observation order")
    if earlier.payload_sha256 == later.payload_sha256:
        return None
    return {
        "source_id": earlier.source_id,
        "source_bar_label_ts": earlier.source_bar_label_ts,
        "earlier_receipt_id": earlier.receipt_id,
        "later_receipt_id": later.receipt_id,
        "earlier_payload_sha256": earlier.payload_sha256,
        "later_payload_sha256": later.payload_sha256,
        "revision_observed_ts": later.data_observed_ts,
    }


def build_source_observation_rows(
    frame: pd.DataFrame,
    *,
    source_id: str,
    receipt_id: str,
    data_observed_ts: str | pd.Timestamp,
) -> pd.DataFrame:
    """Hash each source bar so later API responses can reveal revisions."""
    if frame.empty:
        raise ValueError("source observation frame must not be empty")
    if not source_id.strip() or not receipt_id.strip():
        raise ValueError("source observation identity must not be empty")
    observed = _utc_timestamp(data_observed_ts, field="data_observed_ts")
    index = pd.DatetimeIndex(frame.index)
    index = (
        index.tz_localize("UTC")
        if index.tz is None
        else index.tz_convert("UTC")
    )
    if index.has_duplicates:
        raise ValueError("source observation bars must be unique")
    normalized = frame.copy()
    normalized.index = index
    rows = []
    for bar_ts, row in normalized.sort_index().iterrows():
        values = {
            str(column): (
                None
                if pd.isna(value)
                else value.item()
                if isinstance(value, np.generic)
                else value
            )
            for column, value in row.items()
        }
        rows.append(
            {
                "source_id": source_id,
                "source_bar_label_ts": pd.Timestamp(bar_ts).isoformat(),
                "receipt_id": receipt_id,
                "data_observed_ts": observed.isoformat(),
                "row_sha256": stable_json_sha256(values),
            }
        )
    return pd.DataFrame(rows)


def detect_source_observation_revisions(
    prior_observations: pd.DataFrame,
    current_observations: pd.DataFrame,
) -> pd.DataFrame:
    """Return changed overlapping source bars, using latest prior observation."""
    required = {
        "source_id",
        "source_bar_label_ts",
        "receipt_id",
        "data_observed_ts",
        "row_sha256",
    }
    for name, frame in (
        ("prior_observations", prior_observations),
        ("current_observations", current_observations),
    ):
        if not required.issubset(frame.columns):
            raise ValueError(f"{name} missing source observation columns")
    if current_observations.empty:
        return pd.DataFrame(
            columns=[
                "source_id",
                "source_bar_label_ts",
                "earlier_receipt_id",
                "later_receipt_id",
                "earlier_row_sha256",
                "later_row_sha256",
                "revision_observed_ts",
            ]
        )
    prior = prior_observations.copy()
    current = current_observations.copy()
    for frame in (prior, current):
        frame["source_bar_label_ts"] = pd.to_datetime(
            frame["source_bar_label_ts"], utc=True
        )
        frame["data_observed_ts"] = pd.to_datetime(
            frame["data_observed_ts"], utc=True
        )
    prior = prior.sort_values("data_observed_ts").drop_duplicates(
        ["source_id", "source_bar_label_ts"], keep="last"
    )
    merged = current.merge(
        prior,
        on=["source_id", "source_bar_label_ts"],
        how="inner",
        suffixes=("_later", "_earlier"),
        validate="many_to_one",
    )
    changed = merged.loc[
        merged["row_sha256_later"].astype(str)
        != merged["row_sha256_earlier"].astype(str)
    ].copy()
    if changed.empty:
        return detect_source_observation_revisions(
            prior.iloc[0:0], current.iloc[0:0]
        )
    if (
        changed["data_observed_ts_later"]
        < changed["data_observed_ts_earlier"]
    ).any():
        raise ValueError("source observations are not in observation order")
    return pd.DataFrame(
        {
            "source_id": changed["source_id"].astype(str),
            "source_bar_label_ts": changed[
                "source_bar_label_ts"
            ].map(pd.Timestamp.isoformat),
            "earlier_receipt_id": changed["receipt_id_earlier"].astype(str),
            "later_receipt_id": changed["receipt_id_later"].astype(str),
            "earlier_row_sha256": changed["row_sha256_earlier"].astype(str),
            "later_row_sha256": changed["row_sha256_later"].astype(str),
            "revision_observed_ts": changed[
                "data_observed_ts_later"
            ].map(pd.Timestamp.isoformat),
        }
    ).sort_values(["source_id", "source_bar_label_ts"]).reset_index(drop=True)


def build_revision_consistency_amendments(
    decision_receipt_usage: Sequence[Mapping[str, object]],
    revisions: pd.DataFrame,
) -> list[dict[str, object]]:
    """Link later-discovered source revisions to decisions that used old receipts."""
    required = {
        "source_id",
        "source_bar_label_ts",
        "earlier_receipt_id",
        "later_receipt_id",
        "revision_observed_ts",
    }
    if not required.issubset(revisions.columns):
        raise ValueError("revisions missing consistency-amendment columns")
    usage_fields = {
        "freeze_version",
        "signal_equivalence_id",
        "horizon",
        "decision_ts",
        "receipt_id",
    }
    decisions_by_receipt: dict[str, list[Mapping[str, object]]] = {}
    for record in decision_receipt_usage:
        missing = sorted(usage_fields.difference(record))
        if missing:
            raise ValueError(
                "decision receipt usage is incomplete: " + ", ".join(missing)
            )
        receipt_id = str(record["receipt_id"])
        if not receipt_id:
            raise ValueError("decision receipt usage identity is missing")
        decisions_by_receipt.setdefault(receipt_id, []).append(record)
    amendments = []
    for revision in revisions.to_dict(orient="records"):
        earlier = str(revision["earlier_receipt_id"])
        for record in decisions_by_receipt.get(earlier, []):
            amendments.append(
                {
                    "freeze_version": str(record["freeze_version"]),
                    "signal_equivalence_id": str(record["signal_equivalence_id"]),
                    "decision_ts": str(record["decision_ts"]),
                    "source_id": str(revision["source_id"]),
                    "source_bar_label_ts": str(revision["source_bar_label_ts"]),
                    "earlier_receipt_id": earlier,
                    "later_receipt_id": str(revision["later_receipt_id"]),
                    "revision_observed_ts": str(revision["revision_observed_ts"]),
                    "dimension": "data_availability_consistency",
                    "amended_dimension_status": "fail",
                    "amended_overall_status": "replay_live_consistency_fail",
                    "reason": (
                        "a source bar used by this decision was later observed "
                        "with different values"
                    ),
                }
            )
    return sorted(
        amendments,
        key=lambda row: (
            row["decision_ts"],
            row["signal_equivalence_id"],
            row["source_id"],
            row["source_bar_label_ts"],
            row["later_receipt_id"],
        ),
    )


def apply_consistency_amendments(
    consistency_records: Sequence[Mapping[str, object]],
    amendments: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return current effective consistency records without rewriting history."""
    effective = [dict(record) for record in consistency_records]
    by_identity: dict[tuple[str, str, str], dict[str, object]] = {}
    for record in effective:
        identity = (
            str(record.get("freeze_version", "")),
            str(record.get("signal_equivalence_id", "")),
            str(record.get("decision_ts", "")),
        )
        if not all(identity):
            raise ValueError("consistency record identity is incomplete")
        if identity in by_identity:
            raise ValueError(f"duplicate consistency identity: {identity}")
        by_identity[identity] = record
    for amendment in amendments:
        identity = (
            str(amendment.get("freeze_version", "")),
            str(amendment.get("signal_equivalence_id", "")),
            str(amendment.get("decision_ts", "")),
        )
        if identity not in by_identity:
            raise ValueError(
                f"consistency amendment has no original record: {identity}"
            )
        dimension = str(amendment.get("dimension", ""))
        dimension_status = str(amendment.get("amended_dimension_status", ""))
        overall_status = str(amendment.get("amended_overall_status", ""))
        if dimension not in CONSISTENCY_DIMENSIONS:
            raise ValueError(f"unsupported amended dimension: {dimension}")
        if not dimension_status or not overall_status:
            raise ValueError("consistency amendment status is incomplete")
        target = by_identity[identity]
        if (
            str(target.get("overall_status")) == "replay_live_consistency_fail"
            and overall_status != "replay_live_consistency_fail"
        ):
            continue
        dimensions = {
            str(key): dict(value)
            for key, value in dict(target.get("dimensions", {})).items()
        }
        dimension_record = dict(dimensions.get(dimension, {}))
        dimension_record["status"] = dimension_status
        dimension_record["amended"] = True
        dimension_record["amendment_reason"] = str(amendment.get("reason", ""))
        dimensions[dimension] = dimension_record
        target["dimensions"] = dimensions
        target["overall_status"] = overall_status
        target["amended"] = True
    return effective


def _difference_paths(live: object, replay: object, prefix: str = "") -> list[str]:
    if isinstance(live, Mapping) and isinstance(replay, Mapping):
        paths: list[str] = []
        for key in sorted(set(live).union(replay), key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in live or key not in replay:
                paths.append(child)
            else:
                paths.extend(_difference_paths(live[key], replay[key], child))
        return paths
    if isinstance(live, (list, tuple)) and isinstance(replay, (list, tuple)):
        paths = []
        if len(live) != len(replay):
            paths.append(f"{prefix}.length" if prefix else "length")
        for index, (left, right) in enumerate(zip(live, replay)):
            child = f"{prefix}[{index}]" if prefix else f"[{index}]"
            paths.extend(_difference_paths(left, right, child))
        return paths
    return [] if stable_json_sha256(live) == stable_json_sha256(replay) else [prefix or "$"]


def consistency_evidence_sha256(live_value: object, replay_value: object) -> str:
    """Hash the exact pair of values used for one consistency comparison."""
    return stable_json_sha256({"live": live_value, "replay": replay_value})


def validate_source_consistency_capture_contract(
    request_contract: pd.DataFrame,
    raw_receipts: pd.DataFrame,
    normalized_receipts: pd.DataFrame,
    *,
    expected_request_count: int = 118,
    expected_normalized_count: int = 360,
    expected_source_counts: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Validate one candidate-independent realtime source capture."""
    source_counts = dict(
        expected_source_counts
        or {
            "keystore": 26,
            "binance_public": 58,
            "bybit_public": 17,
            "okx_public": 17,
        }
    )
    contract_required = {"request_id", "source", "source_contract_version"}
    raw_required = {"request_id", "receipt_id", "payload_sha256"}
    normalized_required = {
        "source_scope",
        "signal_timeframe",
        "endpoint",
        "symbol",
        "receipt_id",
        "payload_sha256",
        "target_label_ts",
    }
    for name, frame, required in (
        ("request contract", request_contract, contract_required),
        ("raw receipt manifest", raw_receipts, raw_required),
        ("normalized receipt manifest", normalized_receipts, normalized_required),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} missing: " + ", ".join(missing))
    if len(request_contract) != int(expected_request_count):
        raise ValueError("realtime request contract count differs from authority")
    if request_contract["request_id"].astype(str).duplicated().any():
        raise ValueError("realtime request contract contains duplicate identities")
    actual_source_counts = {
        str(key): int(value)
        for key, value in request_contract.groupby("source").size().items()
    }
    if actual_source_counts != source_counts:
        raise ValueError("realtime request source distribution differs from authority")
    if len(raw_receipts) != int(expected_request_count):
        raise ValueError("raw receipt count differs from request contract")
    if raw_receipts["request_id"].astype(str).duplicated().any():
        raise ValueError("raw receipt manifest contains duplicate request identities")
    if set(raw_receipts["request_id"].astype(str)) != set(
        request_contract["request_id"].astype(str)
    ):
        raise ValueError("raw receipt identities do not match request contract")
    if len(normalized_receipts) != int(expected_normalized_count):
        raise ValueError("normalized receipt count differs from authority")
    normalized_identity = normalized_receipts[
        ["source_scope", "endpoint", "symbol"]
    ].astype(str)
    if normalized_identity.duplicated().any():
        raise ValueError("normalized receipt manifest contains duplicate identities")
    if normalized_receipts["receipt_id"].astype(str).duplicated().any():
        raise ValueError("normalized receipt IDs are not unique")
    versions = sorted(
        request_contract["source_contract_version"].astype(str).unique()
    )
    if len(versions) != 1 or not versions[0]:
        raise ValueError("source contract version must be one non-empty value")
    result = {
        "request_count": len(request_contract),
        "normalized_count": len(normalized_receipts),
        "source_counts": actual_source_counts,
        "source_contract_version": versions[0],
        "request_contract_sha256": stable_json_sha256(
            request_contract.sort_values("request_id").to_dict(orient="records")
        ),
        "raw_receipts_sha256": stable_json_sha256(
            raw_receipts.sort_values("request_id").to_dict(orient="records")
        ),
        "normalized_receipts_sha256": stable_json_sha256(
            normalized_receipts.sort_values(
                ["source_scope", "endpoint", "symbol"]
            ).to_dict(orient="records")
        ),
    }
    result["capture_contract_sha256"] = stable_json_sha256(result)
    return result


def build_source_consistency_queue_records(
    request_manifest: pd.DataFrame,
    *,
    collector_id: str,
    capture_ts: str | pd.Timestamp,
    initial_query_delay_seconds: int,
    retry_interval_seconds: int,
    revision_query_delay_seconds: int,
    maximum_wait_seconds: int,
) -> list[dict[str, object]]:
    """Freeze delayed references without introducing a candidate identity."""
    identity = str(collector_id).strip()
    if not identity:
        raise ValueError("collector_id must not be empty")
    legacy = build_source_reference_queue_records(
        request_manifest,
        freeze_version=identity,
        decision_ts=capture_ts,
        initial_query_delay_seconds=initial_query_delay_seconds,
        retry_interval_seconds=retry_interval_seconds,
        revision_query_delay_seconds=revision_query_delay_seconds,
        maximum_wait_seconds=maximum_wait_seconds,
    )
    records: list[dict[str, object]] = []
    for item in legacy:
        record = dict(item)
        record["collector_id"] = record.pop("freeze_version")
        record["capture_ts"] = record.pop("decision_ts")
        record.pop("queue_record_sha256")
        record["queue_record_sha256"] = stable_json_sha256(record)
        records.append(record)
    return records


def build_source_reference_time_contract(
    *,
    target_label_ts: str | pd.Timestamp,
    timestamp_kind: str,
    bar_duration: str | pd.Timedelta,
) -> dict[str, object]:
    """Build the exact label/end-time contract for one delayed history query.

    The current KSV4 history API uses an exclusive ``end_time`` boundary.
    Therefore both a start label and an end label at ``t`` need a query end at
    ``t + duration`` to include that record.  The native end remains distinct:
    a start label at ``t`` has native end ``t + duration``, while an end label
    at ``t`` has native end ``t``.  The caller must still select the exact
    returned label; this helper never authorizes nearest-row matching.
    """
    target = _utc_timestamp(target_label_ts, field="target_label_ts")
    kind = str(timestamp_kind).strip()
    if kind not in {"bar_start", "bar_end"}:
        raise ValueError(f"unsupported source timestamp_kind: {kind}")
    duration = pd.Timedelta(bar_duration)
    if duration <= pd.Timedelta(0):
        raise ValueError("source bar duration must be positive")
    native_end = target + duration if kind == "bar_start" else target
    query_end = target + duration
    return {
        "target_label_ts": target,
        "native_bar_end_ts": native_end,
        "query_end_ts": query_end,
        "query_end_time_ms": int(query_end.timestamp() * 1000),
        "timestamp_kind": kind,
        "bar_duration": duration,
    }


def source_values_at_exact_label(
    frame: pd.DataFrame,
    target_label_ts: str | pd.Timestamp,
) -> dict[str, object]:
    """Return numeric values only for an exact canonical source label."""
    target = _utc_timestamp(target_label_ts, field="target_label_ts")
    if target not in frame.index:
        raise RuntimeError(f"reference does not contain target label {target}")
    row = frame.loc[target]
    if isinstance(row, pd.DataFrame):
        raise RuntimeError("reference target label is not unique")
    return {
        str(column): float(value)
        for column, value in row.items()
        if pd.notna(value)
    }


def classify_source_consistency_reference_action(
    queue_record: Mapping[str, object],
    comparison_records: Sequence[Mapping[str, object]],
    *,
    observed_ts: str | pd.Timestamp,
    failed_attempts: Sequence[Mapping[str, object]] = (),
) -> str:
    """Return the next delayed-reference action for a collector queue item."""
    required = {"collector_id", "capture_ts"}
    missing = sorted(required.difference(queue_record))
    if missing:
        raise ValueError("collector queue record is incomplete: " + ", ".join(missing))

    identity = (
        str(queue_record["collector_id"]),
        str(queue_record["realtime_receipt_id"]),
    )
    if any(
        (
            str(record.get("collector_id", "")),
            str(record.get("realtime_receipt_id", "")),
        )
        == identity
        and str(record.get("reference_role", "")) == "timeout"
        for record in failed_attempts
    ):
        return "expired"

    def legacy(record: Mapping[str, object]) -> dict[str, object]:
        result = dict(record)
        result["freeze_version"] = result.pop("collector_id")
        if "capture_ts" in result:
            result["decision_ts"] = result.pop("capture_ts")
        return result

    return classify_source_reference_action(
        legacy(queue_record),
        [legacy(record) for record in comparison_records],
        observed_ts=observed_ts,
        failed_attempts=[legacy(record) for record in failed_attempts],
    )


def build_source_value_equivalence_record(
    *,
    collector_id: str,
    capture_ts: str | pd.Timestamp,
    source_scope: str,
    signal_timeframe: str,
    endpoint: str,
    symbol: str,
    target_label_ts: str | pd.Timestamp,
    realtime_native_bar_end_ts: str | pd.Timestamp,
    realtime_receipt_id: str,
    realtime_values: Mapping[str, object],
    reference_receipt_id: str,
    reference_native_bar_end_ts: str | pd.Timestamp,
    reference_values: Mapping[str, object],
    reference_role: str,
    observed_ts: str | pd.Timestamp,
) -> dict[str, object]:
    """Compare one realtime projection with the same delayed history row."""
    identity = {
        "collector_id": str(collector_id).strip(),
        "source_scope": str(source_scope).strip(),
        "signal_timeframe": str(signal_timeframe).strip(),
        "endpoint": str(endpoint).strip(),
        "symbol": str(symbol).strip().upper(),
        "realtime_receipt_id": str(realtime_receipt_id).strip(),
        "reference_receipt_id": str(reference_receipt_id).strip(),
    }
    if not all(identity.values()):
        raise ValueError("source-value comparison identity must not be empty")
    if reference_role not in {"initial", "revision"}:
        raise ValueError("reference_role must be initial or revision")
    realtime = dict(realtime_values)
    reference = dict(reference_values)
    realtime_fields = set(realtime)
    reference_fields = set(reference)
    missing_from_reference = sorted(realtime_fields.difference(reference_fields))
    missing_from_realtime = sorted(reference_fields.difference(realtime_fields))
    common = sorted(realtime_fields.intersection(reference_fields))
    absolute_differences: dict[str, float] = {}
    relative_differences: dict[str, float] = {}
    unequal_fields: list[str] = []
    for field in common:
        left = realtime[field]
        right = reference[field]
        if stable_json_sha256(left) == stable_json_sha256(right):
            continue
        unequal_fields.append(field)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            absolute = abs(float(left) - float(right))
            absolute_differences[field] = absolute
            denominator = max(abs(float(right)), 1e-15)
            relative_differences[field] = absolute / denominator
    target_label = _utc_timestamp(target_label_ts, field="target_label_ts")
    realtime_native_end = _utc_timestamp(
        realtime_native_bar_end_ts, field="realtime_native_bar_end_ts"
    )
    reference_native_end = _utc_timestamp(
        reference_native_bar_end_ts, field="reference_native_bar_end_ts"
    )
    native_identity_equal = realtime_native_end == reference_native_end
    if not native_identity_equal:
        status = "native_identity_mismatch"
    elif missing_from_reference or missing_from_realtime:
        status = "field_mismatch"
    elif unequal_fields:
        status = "value_mismatch"
    else:
        status = "exact_match"
    record = {
        **identity,
        "capture_ts": _utc_timestamp(capture_ts, field="capture_ts").isoformat(),
        "target_label_ts": target_label.isoformat(),
        "realtime_native_bar_end_ts": realtime_native_end.isoformat(),
        "reference_native_bar_end_ts": reference_native_end.isoformat(),
        "native_identity_equal": native_identity_equal,
        "reference_role": reference_role,
        "observed_ts": _utc_timestamp(observed_ts, field="observed_ts").isoformat(),
        "status": status,
        "realtime_values": realtime,
        "reference_values": reference,
        "missing_from_reference": missing_from_reference,
        "missing_from_realtime": missing_from_realtime,
        "unequal_fields": unequal_fields,
        "absolute_differences": absolute_differences,
        "relative_differences": relative_differences,
        "realtime_values_sha256": stable_json_sha256(realtime),
        "reference_values_sha256": stable_json_sha256(reference),
    }
    record["record_sha256"] = stable_json_sha256(record)
    return record


def build_source_equivalence_record(
    *,
    freeze_version: str,
    signal_equivalence_id: str,
    source_id: str,
    symbol: str,
    native_bar_end_ts: str | pd.Timestamp,
    realtime_receipt_id: str,
    realtime_values: Mapping[str, object],
    realtime_decision_projection: Mapping[str, object],
    observed_ts: str | pd.Timestamp,
    reference_receipt_id: str | None = None,
    reference_values: Mapping[str, object] | None = None,
    reference_decision_projection: Mapping[str, object] | None = None,
    reference_role: str | None = None,
) -> dict[str, object]:
    """Classify one immutable realtime/reference comparison."""
    identity = {
        "freeze_version": str(freeze_version).strip(),
        "signal_equivalence_id": str(signal_equivalence_id).strip(),
        "source_id": str(source_id).strip(),
        "symbol": str(symbol).strip().upper(),
        "realtime_receipt_id": str(realtime_receipt_id).strip(),
    }
    if not all(identity.values()):
        raise ValueError("source-equivalence identity must not be empty")
    bar_end = _utc_timestamp(native_bar_end_ts, field="native_bar_end_ts")
    observed = _utc_timestamp(observed_ts, field="observed_ts")
    realtime_values_payload = dict(realtime_values)
    realtime_projection = dict(realtime_decision_projection)

    if reference_values is None:
        if (
            reference_receipt_id
            or reference_decision_projection is not None
            or reference_role
        ):
            raise ValueError(
                "pending source equivalence must not contain partial reference evidence"
            )
        status = "pending_reference"
        value_differences: list[str] = []
        decision_differences: list[str] = []
        reference_values_payload: dict[str, object] | None = None
        reference_projection_payload: dict[str, object] | None = None
    else:
        if not str(reference_receipt_id or "").strip():
            raise ValueError("reference receipt identity is required")
        if reference_role not in {"initial", "revision"}:
            raise ValueError("reference_role must be initial or revision")
        if reference_decision_projection is None:
            raise ValueError("reference decision projection is required")
        reference_values_payload = dict(reference_values)
        reference_projection_payload = dict(reference_decision_projection)
        value_differences = _difference_paths(
            realtime_values_payload, reference_values_payload
        )
        decision_differences = _difference_paths(
            realtime_projection, reference_projection_payload
        )
        if not value_differences:
            status = "exact_match"
        elif not decision_differences:
            status = "value_mismatch_decision_equivalent"
        else:
            status = "decision_material_mismatch"

    record = {
        **identity,
        "signal_equivalence_id": str(signal_equivalence_id).strip(),
        "native_bar_end_ts": bar_end.isoformat(),
        "observed_ts": observed.isoformat(),
        "reference_receipt_id": (
            None if reference_receipt_id is None else str(reference_receipt_id)
        ),
        "reference_role": reference_role,
        "status": status,
        "realtime_values_sha256": stable_json_sha256(realtime_values_payload),
        "reference_values_sha256": (
            None
            if reference_values_payload is None
            else stable_json_sha256(reference_values_payload)
        ),
        "realtime_decision_projection_sha256": stable_json_sha256(
            realtime_projection
        ),
        "reference_decision_projection_sha256": (
            None
            if reference_projection_payload is None
            else stable_json_sha256(reference_projection_payload)
        ),
        "value_difference_paths": value_differences,
        "decision_difference_paths": decision_differences,
    }
    record["record_sha256"] = stable_json_sha256(record)
    return record


def build_source_reference_queue_records(
    request_manifest: pd.DataFrame,
    *,
    freeze_version: str,
    decision_ts: str | pd.Timestamp,
    initial_query_delay_seconds: int,
    retry_interval_seconds: int,
    revision_query_delay_seconds: int,
    maximum_wait_seconds: int,
) -> list[dict[str, object]]:
    """Freeze delayed-reference work items for one committed realtime decision."""
    required = {
        "source_scope",
        "signal_timeframe",
        "endpoint",
        "symbol",
        "receipt_id",
        "target_label_ts",
    }
    missing = sorted(required.difference(request_manifest.columns))
    if missing:
        raise ValueError("reference queue input missing: " + ", ".join(missing))
    delays = (
        initial_query_delay_seconds,
        retry_interval_seconds,
        revision_query_delay_seconds,
        maximum_wait_seconds,
    )
    if any(int(value) <= 0 for value in delays):
        raise ValueError("reference queue delays must be positive")
    if revision_query_delay_seconds <= initial_query_delay_seconds:
        raise ValueError("revision query must follow the initial reference query")
    if maximum_wait_seconds <= revision_query_delay_seconds:
        raise ValueError("maximum wait must follow the revision query")
    decision = _utc_timestamp(decision_ts, field="decision_ts")
    records = []
    identities = set()
    for row in request_manifest.itertuples(index=False):
        identity = (
            str(row.source_scope),
            str(row.endpoint),
            str(row.symbol).upper(),
            str(row.receipt_id),
        )
        if identity in identities:
            raise ValueError("reference queue contains a duplicate source identity")
        identities.add(identity)
        record = {
            "freeze_version": str(freeze_version),
            "decision_ts": decision.isoformat(),
            "source_scope": identity[0],
            "signal_timeframe": str(row.signal_timeframe),
            "endpoint": identity[1],
            "symbol": identity[2],
            "realtime_receipt_id": identity[3],
            "target_label_ts": _utc_timestamp(
                row.target_label_ts, field="target_label_ts"
            ).isoformat(),
            "initial_query_due_ts": (
                decision + pd.Timedelta(seconds=initial_query_delay_seconds)
            ).isoformat(),
            "retry_interval_seconds": int(retry_interval_seconds),
            "revision_query_due_ts": (
                decision + pd.Timedelta(seconds=revision_query_delay_seconds)
            ).isoformat(),
            "maximum_wait_ts": (
                decision + pd.Timedelta(seconds=maximum_wait_seconds)
            ).isoformat(),
            "status": "pending_reference",
        }
        record["queue_record_sha256"] = stable_json_sha256(record)
        records.append(record)
    return records


def classify_source_reference_action(
    queue_record: Mapping[str, object],
    equivalence_records: Sequence[Mapping[str, object]],
    *,
    observed_ts: str | pd.Timestamp,
    failed_attempts: Sequence[Mapping[str, object]] = (),
) -> str:
    """Return the next immutable delayed-reference action for one queue item."""
    required = {
        "freeze_version",
        "decision_ts",
        "source_scope",
        "endpoint",
        "symbol",
        "realtime_receipt_id",
        "initial_query_due_ts",
        "revision_query_due_ts",
        "maximum_wait_ts",
    }
    missing = sorted(required.difference(queue_record))
    if missing:
        raise ValueError("reference queue record is incomplete: " + ", ".join(missing))
    now = _utc_timestamp(observed_ts, field="observed_ts")
    initial_due = _utc_timestamp(
        queue_record["initial_query_due_ts"], field="initial_query_due_ts"
    )
    revision_due = _utc_timestamp(
        queue_record["revision_query_due_ts"], field="revision_query_due_ts"
    )
    maximum_wait = _utc_timestamp(
        queue_record["maximum_wait_ts"], field="maximum_wait_ts"
    )
    if not initial_due < revision_due < maximum_wait:
        raise ValueError("reference queue schedule is not strictly ordered")
    identity = (
        str(queue_record["freeze_version"]),
        str(queue_record["realtime_receipt_id"]),
    )
    matching = [
        record
        for record in equivalence_records
        if (
            str(record.get("freeze_version", "")),
            str(record.get("realtime_receipt_id", "")),
        )
        == identity
    ]
    roles = [str(record.get("reference_role", "")) for record in matching]
    if len(roles) != len(set(roles)):
        raise ValueError("duplicate delayed-reference role for one realtime receipt")
    unexpected = sorted(set(roles).difference({"initial", "revision"}))
    if unexpected:
        raise ValueError("unexpected delayed-reference role: " + ", ".join(unexpected))
    if "revision" in roles:
        if "initial" not in roles:
            raise ValueError("revision reference cannot precede initial reference")
        return "complete"
    if now >= maximum_wait:
        return "timeout"
    retry_seconds = int(queue_record.get("retry_interval_seconds", 0))
    if retry_seconds <= 0:
        raise ValueError("reference queue retry interval must be positive")
    desired_role = "revision" if "initial" in roles else "initial"
    matching_failures = [
        record
        for record in failed_attempts
        if str(record.get("freeze_version", "")) == identity[0]
        and str(record.get("realtime_receipt_id", "")) == identity[1]
        and str(record.get("reference_role", "")) == desired_role
    ]
    if matching_failures:
        latest_attempt = max(
            _utc_timestamp(record["attempt_ts"], field="attempt_ts")
            for record in matching_failures
        )
        if now < latest_attempt + pd.Timedelta(seconds=retry_seconds):
            return "not_due"
    if "initial" in roles:
        return "revision" if now >= revision_due else "not_due"
    return "initial" if now >= initial_due else "not_due"


def build_source_equivalence_consistency_amendments(
    consistency_records: Sequence[Mapping[str, object]],
    decision_receipt_usage: Sequence[Mapping[str, object]],
    reference_queue_records: Sequence[Mapping[str, object]],
    equivalence_records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Resolve pending data consistency only after every used source is compared."""
    original_by_identity = {
        (
            str(record.get("freeze_version", "")),
            str(record.get("signal_equivalence_id", "")),
            str(record.get("decision_ts", "")),
        ): record
        for record in consistency_records
    }
    if len(original_by_identity) != len(consistency_records):
        raise ValueError("consistency records contain duplicate identities")
    queue_receipts_by_decision: dict[tuple[str, str], set[str]] = {}
    queue_identity_by_receipt: dict[tuple[str, str], tuple[str, str, str]] = {}
    for record in reference_queue_records:
        freeze_version = str(record.get("freeze_version", ""))
        decision_ts = str(record.get("decision_ts", ""))
        receipt_id = str(record.get("realtime_receipt_id", ""))
        if not freeze_version or not decision_ts or not receipt_id:
            raise ValueError("reference queue identity is incomplete")
        decision_key = (freeze_version, decision_ts)
        queue_receipts_by_decision.setdefault(decision_key, set()).add(receipt_id)
        queue_identity_by_receipt[(freeze_version, receipt_id)] = (
            str(record.get("source_scope", "")),
            str(record.get("endpoint", "")),
            str(record.get("symbol", "")),
        )
    latest_by_receipt: dict[tuple[str, str, str], Mapping[str, object]] = {}
    role_order = {"initial": 0, "revision": 1}
    for record in equivalence_records:
        key = (
            str(record.get("freeze_version", "")),
            str(record.get("realtime_receipt_id", "")),
            str(record.get("signal_equivalence_id", "")),
        )
        role = str(record.get("reference_role", ""))
        queue_key = (key[0], key[1])
        if queue_key not in queue_identity_by_receipt or role not in role_order:
            raise ValueError("source-equivalence record is not bound to the queue")
        prior = latest_by_receipt.get(key)
        if prior is None or role_order[role] > role_order[str(prior["reference_role"])]:
            latest_by_receipt[key] = record

    usage_by_candidate: dict[tuple[str, str, str], set[str]] = {}
    for record in decision_receipt_usage:
        identity = (
            str(record.get("freeze_version", "")),
            str(record.get("signal_equivalence_id", "")),
            str(record.get("decision_ts", "")),
        )
        receipt_id = str(record.get("receipt_id", ""))
        if not all(identity) or not receipt_id:
            raise ValueError("decision receipt usage identity is incomplete")
        usage_by_candidate.setdefault(identity, set()).add(receipt_id)

    amendments: list[dict[str, object]] = []
    for identity, original in sorted(original_by_identity.items()):
        dimensions = dict(original.get("dimensions", {}))
        availability = dict(dimensions.get("data_availability_consistency", {}))
        if availability.get("status") != "pending_reference":
            continue
        queue_receipts = queue_receipts_by_decision.get((identity[0], identity[2]), set())
        required_receipts = usage_by_candidate.get(identity, set()).intersection(
            queue_receipts
        )
        if not required_receipts:
            raise ValueError(f"pending candidate has no delayed-reference receipts: {identity}")
        resolved = {
            receipt_id: latest_by_receipt.get(
                (identity[0], receipt_id, identity[1])
            )
            for receipt_id in required_receipts
        }
        if any(record is None for record in resolved.values()):
            continue
        statuses = {str(record["status"]) for record in resolved.values() if record}
        allowed = {
            "exact_match",
            "value_mismatch_decision_equivalent",
            "decision_material_mismatch",
        }
        if not statuses.issubset(allowed):
            raise ValueError("unresolved source-equivalence status in consistency amendment")
        failed = "decision_material_mismatch" in statuses
        amendments.append(
            {
                "freeze_version": identity[0],
                "signal_equivalence_id": identity[1],
                "decision_ts": identity[2],
                "dimension": "data_availability_consistency",
                "amended_dimension_status": "fail" if failed else "pass",
                "amended_overall_status": (
                    "replay_live_consistency_fail"
                    if failed
                    else "replay_live_consistency_pass"
                ),
                "reason": (
                    "at least one delayed source reference changed the frozen decision"
                    if failed
                    else "all delayed source references resolved without changing the frozen decision"
                ),
                "resolved_realtime_receipt_ids": sorted(required_receipts),
                "source_equivalence_statuses": sorted(statuses),
                "reference_role": (
                    "revision"
                    if any(
                        str(record["reference_role"]) == "revision"
                        for record in resolved.values()
                        if record
                    )
                    else "initial"
                ),
            }
        )
    return amendments


def build_replay_live_consistency_record(
    *,
    freeze_version: str,
    signal_equivalence_id: str,
    horizon: str,
    decision_ts: str | pd.Timestamp,
    live_dimensions: Mapping[str, object],
    replay_dimensions: Mapping[str, object],
    evidence_sha256: Mapping[str, str],
    dimension_overrides: Mapping[str, str] | None = None,
    missed_decision: bool = False,
    scheduled: bool = True,
) -> dict[str, object]:
    """Compare one live decision with replay rebuilt from the same frozen inputs."""
    if horizon not in HORIZON_DELTAS:
        raise ValueError(f"unsupported horizon: {horizon}")
    if not freeze_version.strip() or not signal_equivalence_id.strip():
        raise ValueError("freeze and signal identities must not be empty")
    decision = _utc_timestamp(decision_ts, field="decision_ts")
    overrides = dict(dimension_overrides or {})
    unknown = sorted(set(overrides).difference(CONSISTENCY_DIMENSIONS))
    if unknown:
        raise ValueError("unknown consistency dimensions: " + ", ".join(unknown))
    invalid = sorted(
        status
        for status in overrides.values()
        if status not in CONSISTENCY_DIMENSION_STATUSES
    )
    if invalid:
        raise ValueError("invalid consistency status: " + ", ".join(invalid))

    dimension_rows: dict[str, dict[str, object]] = {}
    for dimension in CONSISTENCY_DIMENSIONS:
        override = overrides.get(dimension)
        if not scheduled or override == "not_applicable":
            dimension_rows[dimension] = {
                "status": "not_applicable",
                "reason": "decision is not scheduled" if not scheduled else "explicitly not applicable",
                "difference_paths": [],
            }
            continue
        if override == "not_observable":
            dimension_rows[dimension] = {
                "status": "not_observable",
                "reason": "required evidence is not observable",
                "difference_paths": [],
            }
            continue
        if override == "pending_reference":
            dimension_rows[dimension] = {
                "status": "pending_reference",
                "reason": "delayed historical reference is not yet available",
                "difference_paths": [],
            }
            continue
        live_present = dimension in live_dimensions
        replay_present = dimension in replay_dimensions
        evidence = str(evidence_sha256.get(dimension, ""))
        expected_evidence = (
            consistency_evidence_sha256(
                live_dimensions[dimension], replay_dimensions[dimension]
            )
            if live_present and replay_present
            else ""
        )
        evidence_valid = bool(
            re.fullmatch(r"[0-9a-f]{64}", evidence)
            and evidence == expected_evidence
        )
        if not live_present or not replay_present or not evidence_valid:
            missing = []
            if not live_present:
                missing.append("live")
            if not replay_present:
                missing.append("replay")
            if not evidence_valid:
                missing.append("matching_evidence_sha256")
            dimension_rows[dimension] = {
                "status": "fail",
                "reason": "missing required " + ", ".join(missing),
                "difference_paths": [],
                "evidence_sha256": evidence,
            }
            continue
        live_value = live_dimensions[dimension]
        replay_value = replay_dimensions[dimension]
        differences = _difference_paths(live_value, replay_value)
        dimension_rows[dimension] = {
            "status": "pass" if not differences else "fail",
            "reason": "" if not differences else "live and replay values differ",
            "difference_paths": differences,
            "live_sha256": stable_json_sha256(live_value),
            "replay_sha256": stable_json_sha256(replay_value),
            "evidence_sha256": evidence,
        }

    if not scheduled:
        overall = "not_scheduled"
    elif missed_decision:
        overall = "missed_decision"
    elif all(
        row["status"] in {"pass", "not_applicable"}
        for row in dimension_rows.values()
    ):
        overall = "replay_live_consistency_pass"
    elif all(
        row["status"] in {"pass", "pending_reference", "not_applicable"}
        for row in dimension_rows.values()
    ) and any(
        row["status"] == "pending_reference"
        for row in dimension_rows.values()
    ):
        overall = "replay_live_consistency_pending_reference"
    else:
        overall = "replay_live_consistency_fail"
    return {
        "freeze_version": freeze_version,
        "signal_equivalence_id": signal_equivalence_id,
        "horizon": horizon,
        "decision_ts": decision.isoformat(),
        "runtime_contract_version": TRUE_OOS_RUNTIME_CONTRACT_VERSION,
        "dimensions": dimension_rows,
        "overall_status": overall,
        "record_sha256": stable_json_sha256(
            {
                "freeze_version": freeze_version,
                "signal_equivalence_id": signal_equivalence_id,
                "horizon": horizon,
                "decision_ts": decision.isoformat(),
                "dimensions": dimension_rows,
                "overall_status": overall,
            }
        ),
    }


def validate_source_freshness(
    records: Sequence[SourceFreshness],
    *,
    required_sources: Sequence[str],
) -> pd.DataFrame:
    """Validate actual source observations without inferring availability."""
    if not records:
        raise ValueError("source freshness records must not be empty")
    required = set(map(str, required_sources))
    observed_ids = [str(record.source_id) for record in records]
    if set(observed_ids) != required or len(observed_ids) != len(set(observed_ids)):
        raise ValueError("freshness records must cover required sources exactly once")
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for record in records:
        native_end = _utc_timestamp(record.native_bar_end_ts, field="native_bar_end_ts")
        observed = _utc_timestamp(record.data_observed_ts, field="data_observed_ts")
        checked = _utc_timestamp(record.checked_ts, field="checked_ts")
        if record.maximum_delay_seconds < 0:
            raise ValueError("maximum_delay_seconds must be non-negative")
        if record.maximum_age_seconds < 0:
            raise ValueError("maximum_age_seconds must be non-negative")
        delay = (observed - native_end).total_seconds()
        age = (checked - observed).total_seconds()
        valid = (
            observed >= native_end
            and checked >= observed
            and delay <= float(record.maximum_delay_seconds)
            and age <= float(record.maximum_age_seconds)
        )
        if not valid:
            failures.append(str(record.source_id))
        rows.append(
            {
                **asdict(record),
                "observed_delay_seconds": delay,
                "observation_age_seconds": age,
                "fresh": valid,
            }
        )
    result = pd.DataFrame(rows).sort_values("source_id").reset_index(drop=True)
    if failures:
        raise RuntimeError(
            "source freshness preflight failed: " + ", ".join(sorted(failures))
        )
    return result


def classify_decision_readiness(
    *,
    horizon: str,
    decision_ts: str | pd.Timestamp,
    signal_ready_ts: str | pd.Timestamp,
) -> DecisionReadiness:
    """Classify an observed signal as actionable or irrecoverably missed."""
    if horizon not in HORIZON_DELTAS:
        raise ValueError(f"unsupported horizon: {horizon}")
    decision = _utc_timestamp(decision_ts, field="decision_ts")
    ready = _utc_timestamp(signal_ready_ts, field="signal_ready_ts")
    epoch = pd.Timestamp("1970-01-01T00:00:00Z")
    if (decision - epoch) % HORIZON_DELTAS[horizon] != pd.Timedelta(0):
        raise ValueError(f"decision_ts is not on the native {horizon} phase")
    if ready < decision:
        raise ValueError("signal_ready_ts cannot precede decision_ts")
    next_decision = decision + HORIZON_DELTAS[horizon]
    status = "ready" if decision <= ready < next_decision else "missed_decision"
    return DecisionReadiness(
        horizon=horizon,
        decision_ts=decision.isoformat(),
        next_decision_ts=next_decision.isoformat(),
        signal_ready_ts=ready.isoformat(),
        status=status,
        delay_seconds=(ready - decision).total_seconds(),
    )


def build_activation_manifest(
    *,
    freeze_version: str,
    shadow_start_ts: str | pd.Timestamp,
    candidate_manifest_path: str | Path,
    parameter_manifest_path: str | Path,
    exchange_rules_path: str | Path,
    source_reference_plan_path: str | Path,
    activation_intent_path: str | Path,
    code_sha: str,
    config_sha: str,
    environment_id: str,
    manifest_generated_ts: str | pd.Timestamp,
    preflight_checks: Mapping[str, bool],
) -> dict[str, object]:
    """Build an activation manifest only after every declared preflight passes."""
    if not preflight_checks:
        raise ValueError("activation requires explicit preflight checks")
    if set(preflight_checks) != REQUIRED_ACTIVATION_PREFLIGHTS:
        missing = sorted(REQUIRED_ACTIVATION_PREFLIGHTS.difference(preflight_checks))
        unknown = sorted(set(preflight_checks).difference(REQUIRED_ACTIVATION_PREFLIGHTS))
        raise ValueError(
            f"activation preflight set mismatch: missing={missing}, unknown={unknown}"
        )
    failed = sorted(name for name, passed in preflight_checks.items() if passed is not True)
    if failed:
        raise RuntimeError("activation preflight failed: " + ", ".join(failed))
    start = _utc_timestamp(shadow_start_ts, field="shadow_start_ts")
    generated = _utc_timestamp(manifest_generated_ts, field="manifest_generated_ts")
    if generated < start:
        raise ValueError("manifest_generated_ts cannot precede shadow_start_ts")
    files = {
        "candidate_manifest": Path(candidate_manifest_path),
        "parameter_manifest": Path(parameter_manifest_path),
        "exchange_rules": Path(exchange_rules_path),
        "source_reference_plan": Path(source_reference_plan_path),
        "activation_intent": Path(activation_intent_path),
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("activation inputs missing: " + ", ".join(missing))
    if not all(
        str(value).strip()
        for value in (freeze_version, code_sha, config_sha, environment_id)
    ):
        raise ValueError("activation identity fields must be non-empty")
    return {
        **lifecycle_metadata(),
        "freeze_version": freeze_version,
        "shadow_start_ts": start.isoformat(),
        "first_eligible_decision_ts": {
            horizon: first_eligible_decision_ts(horizon, start).isoformat()
            for horizon in HORIZON_DELTAS
        },
        "manifest_generated_ts": generated.isoformat(),
        "runtime_contract_version": TRUE_OOS_RUNTIME_CONTRACT_VERSION,
        "code_sha": code_sha,
        "config_sha": config_sha,
        "environment_id": environment_id,
        "activation_intent_path": str(Path(activation_intent_path).resolve()),
        "activation_intent_sha256": sha256_file(activation_intent_path),
        "input_sha256": {
            name: sha256_file(path)
            for name, path in sorted(files.items())
        },
        "input_paths": {
            name: str(path.resolve())
            for name, path in sorted(files.items())
        },
        "preflight_checks": dict(sorted(preflight_checks.items())),
    }


def build_activation_intent(
    *,
    freeze_version: str,
    shadow_start_ts: str | pd.Timestamp,
    evidence_paths: Mapping[str, str | Path],
    reviewed_code_sha: str,
    config_sha: str,
    environment_id: str,
) -> dict[str, object]:
    """Freeze activation identity before publishing any authoritative artifact."""
    if not evidence_paths:
        raise ValueError("activation intent requires evidence")
    if not all(
        str(value).strip()
        for value in (freeze_version, reviewed_code_sha, config_sha, environment_id)
    ):
        raise ValueError("activation intent identity fields must be non-empty")
    files = {name: Path(path) for name, path in evidence_paths.items()}
    missing = sorted(name for name, path in files.items() if not path.is_file())
    if missing:
        raise FileNotFoundError(
            "activation intent evidence missing: " + ", ".join(missing)
        )
    start = _utc_timestamp(shadow_start_ts, field="shadow_start_ts")
    return {
        **lifecycle_metadata(),
        "freeze_version": freeze_version,
        "shadow_start_ts": start.isoformat(),
        "first_eligible_decision_ts": {
            horizon: first_eligible_decision_ts(horizon, start).isoformat()
            for horizon in HORIZON_DELTAS
        },
        "reviewed_code_sha": reviewed_code_sha,
        "config_sha": config_sha,
        "environment_id": environment_id,
        "evidence_sha256": {
            name: sha256_file(path) for name, path in sorted(files.items())
        },
        "evidence_paths": {
            name: str(path.resolve()) for name, path in sorted(files.items())
        },
    }


def verify_activation_intent(
    intent_path: str | Path,
    *,
    freeze_version: str,
    evidence_paths: Mapping[str, str | Path],
    reviewed_code_sha: str,
    config_sha: str,
    environment_id: str,
) -> dict[str, object]:
    """Verify an immutable activation intent against the current evidence bundle."""
    verify_json_sha256_sidecar(intent_path)
    payload = json.loads(Path(intent_path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("activation intent must be an object")
    expected = build_activation_intent(
        freeze_version=freeze_version,
        shadow_start_ts=str(payload.get("shadow_start_ts", "")),
        evidence_paths=evidence_paths,
        reviewed_code_sha=reviewed_code_sha,
        config_sha=config_sha,
        environment_id=environment_id,
    )
    if dict(payload) != expected:
        raise RuntimeError("activation intent does not match current evidence identity")
    return dict(payload)


def require_activation_intent_publishable(
    intent: Mapping[str, object],
    *,
    observed_ts: str | pd.Timestamp,
    safety_margin_seconds: float = 30.0,
) -> pd.Timestamp:
    """Fail closed once an activation intent cannot publish before first use."""
    if safety_margin_seconds < 0:
        raise ValueError("activation safety margin must be non-negative")
    first_by_horizon = intent.get("first_eligible_decision_ts")
    if not isinstance(first_by_horizon, Mapping) or set(first_by_horizon) != set(
        HORIZON_DELTAS
    ):
        raise ValueError("activation intent first eligibility is invalid")
    first = min(
        _utc_timestamp(str(value), field="first_eligible_decision_ts")
        for value in first_by_horizon.values()
    )
    observed = _utc_timestamp(observed_ts, field="observed_ts")
    if first - observed <= pd.Timedelta(seconds=safety_margin_seconds):
        raise RuntimeError(
            "activation intent can no longer be published before first eligibility; "
            "create a new freeze"
        )
    return first


def verify_activation_manifest(
    manifest_path: str | Path,
    *,
    freeze_version: str,
    require_sha256_sidecar: bool = False,
) -> dict[str, object]:
    manifest_sha = None
    if require_sha256_sidecar:
        manifest_sha = verify_json_sha256_sidecar(manifest_path)
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("activation manifest must be an object")
    if payload.get("freeze_version") != freeze_version:
        raise RuntimeError("activation manifest freeze version mismatch")
    if payload.get("runtime_contract_version") != TRUE_OOS_RUNTIME_CONTRACT_VERSION:
        raise RuntimeError("activation manifest runtime contract is stale")
    checks = payload.get("preflight_checks")
    if not isinstance(checks, Mapping) or set(checks) != REQUIRED_ACTIVATION_PREFLIGHTS:
        raise RuntimeError("activation manifest preflight set is invalid")
    if any(value is not True for value in checks.values()):
        raise RuntimeError("activation manifest contains failed preflight")
    input_sha = payload.get("input_sha256")
    input_paths = payload.get("input_paths")
    if (
        not isinstance(input_sha, Mapping)
        or not isinstance(input_paths, Mapping)
        or set(input_sha) != set(input_paths)
    ):
        raise RuntimeError("activation input identity is invalid")
    for name, expected_sha in input_sha.items():
        path = Path(str(input_paths[name]))
        if not path.is_file() or sha256_file(path) != str(expected_sha):
            raise RuntimeError(f"activation input SHA mismatch: {name}")
    evidence_sha = payload.get("evidence_sha256")
    evidence_paths = payload.get("evidence_paths")
    if evidence_sha is not None or evidence_paths is not None:
        if (
            not isinstance(evidence_sha, Mapping)
            or not isinstance(evidence_paths, Mapping)
            or set(evidence_sha) != set(evidence_paths)
        ):
            raise RuntimeError("activation evidence identity is invalid")
        for name, expected_sha in evidence_sha.items():
            path = Path(str(evidence_paths[name]))
            if not path.is_file() or sha256_file(path) != str(expected_sha):
                raise RuntimeError(f"activation evidence SHA mismatch: {name}")
    intent_path_value = payload.get("activation_intent_path")
    intent_sha_value = payload.get("activation_intent_sha256")
    if not intent_path_value or not intent_sha_value:
        raise RuntimeError("activation intent identity is incomplete")
    intent_path = Path(str(intent_path_value))
    if str(input_paths.get("activation_intent", "")) != str(intent_path):
        raise RuntimeError("activation intent is not a declared manifest input")
    verify_json_sha256_sidecar(intent_path)
    if sha256_file(intent_path) != str(intent_sha_value):
        raise RuntimeError("activation intent SHA mismatch")
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    if (
        not isinstance(intent, Mapping)
        or str(intent.get("freeze_version")) != freeze_version
        or str(intent.get("shadow_start_ts"))
        != str(payload.get("shadow_start_ts"))
        or dict(intent.get("first_eligible_decision_ts", {}))
        != dict(payload.get("first_eligible_decision_ts", {}))
        or str(intent.get("reviewed_code_sha")) != str(payload.get("code_sha"))
        or str(intent.get("config_sha")) != str(payload.get("config_sha"))
        or str(intent.get("environment_id"))
        != str(payload.get("environment_id"))
        or dict(intent.get("evidence_sha256", {}))
        != dict(payload.get("evidence_sha256", {}))
        or dict(intent.get("evidence_paths", {}))
        != dict(payload.get("evidence_paths", {}))
    ):
        raise RuntimeError("activation manifest is not bound to its intent")
    _utc_timestamp(str(payload.get("shadow_start_ts", "")), field="shadow_start_ts")
    if require_sha256_sidecar:
        publish_path = Path(str(payload.get("publish_receipt_path", "")))
        verify_json_sha256_sidecar(publish_path)
        publish = json.loads(publish_path.read_text(encoding="utf-8"))
        if (
            not isinstance(publish, Mapping)
            or str(publish.get("activation_manifest_sha256")) != manifest_sha
            or str(publish.get("freeze_version")) != freeze_version
        ):
            raise RuntimeError("activation publish receipt is not bound to manifest")
        completed = _utc_timestamp(
            str(publish.get("activation_publish_completed_ts", "")),
            field="activation_publish_completed_ts",
        )
        start = _utc_timestamp(
            str(payload["shadow_start_ts"]), field="shadow_start_ts"
        )
        first_eligible = [
            _utc_timestamp(str(value), field="first_eligible_decision_ts")
            for value in payload["first_eligible_decision_ts"].values()
        ]
        if completed < start or completed >= min(first_eligible):
            raise RuntimeError(
                "activation was not completely published before first eligibility"
            )
    return payload


def build_candidate_freeze_manifest(
    unique_signal_frames: Sequence[pd.DataFrame],
    evidence_frame: pd.DataFrame,
    *,
    freeze_version: str,
    account_equity: float = 272.0,
    target_gross_notional: float = 600.0,
    exchange_leverage: float = 5.0,
    taker_fee_rate: float = 0.0005,
) -> pd.DataFrame:
    if not freeze_version.strip():
        raise ValueError("freeze_version must not be empty")
    required_unique = {
        "signal_equivalence_id", "candidate_id", "canonical_candidate_id", "horizon",
        "track", "weight_scheme", "component_features",
    }
    required_evidence = {"signal_equivalence_id", "main_7d_time_alignment_label", "test_status"}
    candidates = pd.concat([frame.copy() for frame in unique_signal_frames], ignore_index=True)
    missing = sorted(required_unique.difference(candidates.columns))
    if missing:
        raise ValueError("unique signal manifests missing columns: " + ", ".join(missing))
    missing = sorted(required_evidence.difference(evidence_frame.columns))
    if missing:
        raise ValueError("evidence frame missing columns: " + ", ".join(missing))
    if candidates["signal_equivalence_id"].astype(str).duplicated().any():
        raise ValueError("signal_equivalence_id must be unique across manifests")
    actual_counts = candidates.groupby("horizon").size().astype(int).to_dict()
    if actual_counts != EXPECTED_HORIZON_COUNTS:
        raise ValueError(
            f"candidate horizon counts mismatch: expected={EXPECTED_HORIZON_COUNTS}, actual={actual_counts}"
        )
    evidence = evidence_frame[list(required_evidence)].copy()
    if evidence["signal_equivalence_id"].astype(str).duplicated().any():
        raise ValueError("evidence signal_equivalence_id must be unique")
    merged = candidates.merge(
        evidence, on="signal_equivalence_id", how="left", validate="one_to_one"
    )
    if merged["main_7d_time_alignment_label"].isna().any():
        raise ValueError("evidence does not cover every frozen candidate")
    if not (merged["test_status"].astype(str) == "valid").all():
        raise ValueError("all frozen candidates must have valid evidence status")
    labels = merged["main_7d_time_alignment_label"].astype(str)
    allowed = {"time_alignment_detected", "time_alignment_not_detected"}
    if not set(labels).issubset(allowed):
        raise ValueError("unexpected historical evidence labels")
    if labels.value_counts().to_dict() != {
        "time_alignment_detected": 68,
        "time_alignment_not_detected": 4,
    }:
        raise ValueError("historical evidence group counts must be 68 detected and 4 not detected")

    manifest = pd.DataFrame(
        {
            "freeze_version": freeze_version,
            "signal_equivalence_id": merged["signal_equivalence_id"].astype(str),
            "candidate_id": merged["candidate_id"].astype(str),
            "canonical_candidate_id": merged["canonical_candidate_id"].astype(str),
            "horizon": merged["horizon"].astype(str),
            "track_id": merged["track"].astype(str),
            "alpha_id": merged["weight_scheme"].astype(str),
            "track": merged["track"].astype(str),
            "weight_scheme": merged["weight_scheme"].astype(str),
            "component_features": merged["component_features"].astype(str),
            "component_feature_order": merged["component_features"].astype(str),
            "universe": "|".join(CANONICAL_SYMBOLS),
            "panel_frequency": "1h",
            "data_source_contract": "keystore_v4_as_received_plus_binance_usdm",
            "factor_transform_rule": "frozen_ksv4_registry_non_price_transform",
            "direction_rule": "train_mean_ic_sign",
            "weight_estimation_rule": "frozen_family_alpha_train_only",
            "decision_interval": merged["horizon"].astype(str),
            "decision_phase_utc": merged["horizon"].map(
                {
                    "4h": "00:00|04:00|08:00|12:00|16:00|20:00",
                    "8h": "00:00|08:00|16:00",
                    "12h": "00:00|12:00",
                    "1d": "00:00",
                }
            ),
            "holding_interval": merged["horizon"].astype(str),
            "position_rule": "hold_quantity_when_membership_unchanged",
            "long_count": 4,
            "short_count": 4,
            "execution_rule": "first_production_book_ticker_at_or_after_virtual_submit",
            "historical_evidence_label": labels,
            "historical_evidence_group": np.where(
                labels.eq("time_alignment_detected"), "historical_detected", "natural_negative_control"
            ),
            "account_equity": float(account_equity),
            "target_gross_notional": float(target_gross_notional),
            "exchange_leverage": float(exchange_leverage),
            "taker_fee_rate": float(taker_fee_rate),
            "cost_multipliers": "1|1.5|2",
            "minimum_notional_rule": "dynamic_exchange_rule_at_order_intent",
            "train_days": 180,
            "embargo_days": 1,
            "epoch_days": 45,
            "runtime_contract_version": TRUE_OOS_RUNTIME_CONTRACT_VERSION,
        }
    )
    return manifest[list(FREEZE_COLUMNS)].sort_values(
        ["horizon", "signal_equivalence_id"], kind="mergesort"
    ).reset_index(drop=True)


def build_shadow_structural_equivalence_manifest(
    catalog: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    historical_mapping: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build future-stable signal aliases from candidate structure.

    Historical signal equality is accepted only as a cross-check. Group identity
    is derived from the ordered feature set and the deterministic normalized
    weight vector implied by the registered family-count weighting rule.
    """
    required_catalog = {
        "combo_id", "track", "weight_scheme", "return_horizon",
        "component_features",
    }
    required_registry = {"feature_name", "family"}
    missing = sorted(required_catalog.difference(catalog.columns))
    if missing:
        raise ValueError("candidate catalog missing columns: " + ", ".join(missing))
    missing = sorted(required_registry.difference(registry.columns))
    if missing:
        raise ValueError("factor registry missing columns: " + ", ".join(missing))
    if registry["feature_name"].astype(str).duplicated().any():
        raise ValueError("factor registry feature_name must be unique")
    family_by_feature = registry.set_index(
        registry["feature_name"].astype(str)
    )["family"].astype(str).to_dict()
    rows: list[dict[str, object]] = []
    for candidate in catalog.itertuples(index=False):
        horizon = str(candidate.return_horizon)
        features = split_component_features(str(candidate.component_features))
        if not features:
            raise ValueError("candidate component_features must not be empty")
        missing_features = sorted(set(features).difference(family_by_feature))
        if missing_features:
            raise ValueError(
                "candidate features absent from registry: "
                + ", ".join(missing_features)
            )
        scheme = str(candidate.weight_scheme)
        if scheme == "equal":
            weights = {feature: 1.0 / len(features) for feature in features}
        else:
            alpha = factor_research.family_alpha_from_weight_scheme(scheme)
            if alpha is None:
                raise ValueError(
                    "shadow structural equivalence does not support weight scheme: "
                    + scheme
                )
            weights = factor_research.family_count_feature_weights(
                features,
                family_by_feature,
                alpha=alpha,
            )
        weight_vector = tuple(round(float(weights[feature]), 15) for feature in features)
        structural_payload = {
            "horizon": horizon,
            "features": list(features),
            "weights": list(weight_vector),
        }
        structural_key = stable_json_sha256(structural_payload)
        candidate_id = f"{horizon}__{candidate.track}__{scheme}"
        rows.append(
            {
                "candidate_id": candidate_id,
                "combo_id": str(candidate.combo_id),
                "track": str(candidate.track),
                "weight_scheme": scheme,
                "horizon": horizon,
                "component_features": " | ".join(features),
                "structural_weight_vector": " | ".join(
                    f"{value:.15g}" for value in weight_vector
                ),
                "structural_key": structural_key,
            }
        )
    aliases = pd.DataFrame(rows)
    if aliases["candidate_id"].duplicated().any():
        raise ValueError("candidate_id must be unique")
    actual_candidates = aliases.groupby("horizon").size().astype(int).to_dict()
    if actual_candidates != EXPECTED_CANDIDATE_CONFIG_COUNTS:
        raise ValueError(
            "candidate configuration counts mismatch: "
            f"expected={EXPECTED_CANDIDATE_CONFIG_COUNTS}, actual={actual_candidates}"
        )
    aliases = aliases.sort_values(
        ["horizon", "structural_key", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)
    aliases["structural_group_index"] = (
        aliases.groupby("horizon", sort=False)["structural_key"]
        .transform(lambda values: pd.factorize(values, sort=False)[0] + 1)
        .astype(int)
    )
    aliases["signal_equivalence_id"] = aliases.apply(
        lambda row: f"{row['horizon']}_signal_{int(row['structural_group_index']):03d}",
        axis=1,
    )
    aliases["canonical_candidate_id"] = aliases.groupby(
        "signal_equivalence_id", sort=False
    )["candidate_id"].transform("first")
    aliases["alias_count"] = aliases.groupby(
        "signal_equivalence_id", sort=False
    )["candidate_id"].transform("size").astype(int)
    actual_signals = (
        aliases.drop_duplicates("signal_equivalence_id")
        .groupby("horizon")
        .size()
        .astype(int)
        .to_dict()
    )
    if actual_signals != EXPECTED_HORIZON_COUNTS:
        raise ValueError(
            "structural signal counts mismatch: "
            f"expected={EXPECTED_HORIZON_COUNTS}, actual={actual_signals}"
        )
    if historical_mapping is not None:
        required_history = {"candidate_id", "signal_equivalence_id"}
        missing = sorted(required_history.difference(historical_mapping.columns))
        if missing:
            raise ValueError(
                "historical equivalence mapping missing columns: " + ", ".join(missing)
            )
        history = historical_mapping[list(required_history)].copy()
        if history["candidate_id"].astype(str).duplicated().any():
            raise ValueError("historical candidate_id must be unique")
        if set(history["candidate_id"].astype(str)) != set(aliases["candidate_id"]):
            raise ValueError("historical mapping candidate set differs from catalog")
        structural_groups = {
            frozenset(group["candidate_id"].astype(str))
            for _, group in aliases.groupby("signal_equivalence_id")
        }
        historical_groups = {
            frozenset(group["candidate_id"].astype(str))
            for _, group in history.groupby("signal_equivalence_id")
        }
        if structural_groups != historical_groups:
            raise ValueError(
                "historical signal equivalence differs from structural equivalence"
            )
    unique = (
        aliases.sort_values("candidate_id", kind="mergesort")
        .groupby("signal_equivalence_id", sort=False, as_index=False)
        .first()
    )
    unique = unique[
        [
            "signal_equivalence_id", "canonical_candidate_id", "alias_count",
            "horizon", "track", "weight_scheme", "component_features",
            "structural_weight_vector", "structural_key",
        ]
    ].sort_values(["horizon", "signal_equivalence_id"], kind="mergesort")
    return aliases.reset_index(drop=True), unique.reset_index(drop=True)


def build_shadow_candidate_freeze_manifest(
    structural_unique: pd.DataFrame,
    *,
    freeze_version: str,
    data_source_contract: str,
    account_equity: float = 272.0,
    target_gross_notional: float = 600.0,
    exchange_leverage: float = 5.0,
    taker_fee_rate: float = 0.0005,
) -> pd.DataFrame:
    """Build the repaired 47-signal shadow freeze without historical labels."""
    required = {
        "signal_equivalence_id", "canonical_candidate_id", "horizon", "track",
        "weight_scheme", "component_features", "structural_key",
    }
    missing = sorted(required.difference(structural_unique.columns))
    if missing:
        raise ValueError("structural unique manifest missing columns: " + ", ".join(missing))
    if not freeze_version.strip():
        raise ValueError("freeze_version must not be empty")
    if not data_source_contract.strip():
        raise ValueError("data_source_contract must not be empty")
    if structural_unique["signal_equivalence_id"].astype(str).duplicated().any():
        raise ValueError("signal_equivalence_id must be unique")
    actual_counts = structural_unique.groupby("horizon").size().astype(int).to_dict()
    if actual_counts != EXPECTED_HORIZON_COUNTS:
        raise ValueError(
            f"shadow horizon counts mismatch: expected={EXPECTED_HORIZON_COUNTS}, actual={actual_counts}"
        )
    rows = structural_unique.copy()
    rows["freeze_version"] = freeze_version
    rows["candidate_id"] = rows["canonical_candidate_id"].astype(str)
    rows["track_id"] = rows["track"].astype(str)
    rows["alpha_id"] = rows["weight_scheme"].astype(str)
    rows["component_feature_order"] = rows["component_features"].astype(str)
    rows["universe"] = "|".join(CANONICAL_SYMBOLS)
    rows["panel_frequency"] = "1h"
    rows["data_source_contract"] = data_source_contract
    rows["factor_transform_rule"] = "repaired_endpoint_semantics_registry"
    rows["direction_rule"] = "train_mean_ic_sign"
    rows["weight_estimation_rule"] = "structurally_frozen_family_alpha"
    rows["decision_interval"] = rows["horizon"].astype(str)
    rows["decision_phase_utc"] = rows["horizon"].map(
        {
            "4h": "00:00|04:00|08:00|12:00|16:00|20:00",
            "8h": "00:00|08:00|16:00",
            "12h": "00:00|12:00",
            "1d": "00:00",
        }
    )
    rows["holding_interval"] = rows["horizon"].astype(str)
    rows["position_rule"] = "hold_quantity_when_membership_unchanged"
    rows["long_count"] = 4
    rows["short_count"] = 4
    rows["execution_rule"] = "first_production_book_ticker_at_or_after_virtual_submit"
    rows["historical_evidence_label"] = "not_used_for_shadow_freeze"
    rows["historical_evidence_group"] = "all_repaired_candidates"
    rows["account_equity"] = float(account_equity)
    rows["target_gross_notional"] = float(target_gross_notional)
    rows["exchange_leverage"] = float(exchange_leverage)
    rows["taker_fee_rate"] = float(taker_fee_rate)
    rows["cost_multipliers"] = "1|1.5|2"
    rows["minimum_notional_rule"] = "dynamic_exchange_rule_at_order_intent"
    rows["train_days"] = 180
    rows["embargo_days"] = 1
    rows["epoch_days"] = 45
    rows["runtime_contract_version"] = TRUE_OOS_RUNTIME_CONTRACT_VERSION
    return rows[list(FREEZE_COLUMNS)].sort_values(
        ["horizon", "signal_equivalence_id"], kind="mergesort"
    ).reset_index(drop=True)


def write_freeze_bundle(
    manifest: pd.DataFrame,
    output_dir: str | Path,
    source_paths: Sequence[str | Path],
    *,
    source_root: str | Path | None = None,
    candidate_aliases: pd.DataFrame | None = None,
) -> dict[str, Path]:
    missing = sorted(set(FREEZE_COLUMNS).difference(manifest.columns))
    if missing:
        raise ValueError("freeze manifest missing columns: " + ", ".join(missing))
    expected_signal_count = sum(EXPECTED_HORIZON_COUNTS.values())
    if (
        len(manifest) != expected_signal_count
        or manifest["signal_equivalence_id"].nunique() != expected_signal_count
    ):
        raise ValueError(
            f"freeze manifest must contain {expected_signal_count} unique signals"
        )
    freeze_version = str(manifest["freeze_version"].iloc[0])
    if manifest["freeze_version"].astype(str).nunique() != 1:
        raise ValueError("freeze manifest has multiple freeze versions")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"freeze bundle already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(
        f".{destination.name}.{uuid4().hex}.staged"
    )
    staging.mkdir()
    try:
        manifest_name = f"ksv4_true_oos_{freeze_version}_candidate_manifest.csv"
        aliases_name = f"ksv4_true_oos_{freeze_version}_candidate_aliases.csv"
        source_name = f"ksv4_true_oos_{freeze_version}_source_manifest.json"
        sha_name = f"ksv4_true_oos_{freeze_version}_sha256.csv"
        manifest_path = staging / manifest_name
        manifest.to_csv(manifest_path, index=False, float_format="%.17g")
        aliases_path: Path | None = None
        if candidate_aliases is not None:
            required_aliases = {"candidate_id", "signal_equivalence_id"}
            missing_aliases = sorted(required_aliases.difference(candidate_aliases.columns))
            if missing_aliases:
                raise ValueError(
                    "candidate aliases missing columns: " + ", ".join(missing_aliases)
                )
            if len(candidate_aliases) != sum(EXPECTED_CANDIDATE_CONFIG_COUNTS.values()):
                raise ValueError("candidate aliases must cover every candidate configuration")
            if candidate_aliases["candidate_id"].astype(str).duplicated().any():
                raise ValueError("candidate aliases contain duplicate candidate_id")
            if set(candidate_aliases["signal_equivalence_id"].astype(str)) != set(
                manifest["signal_equivalence_id"].astype(str)
            ):
                raise ValueError("candidate aliases do not match the freeze signal set")
            aliases_path = staging / aliases_name
            candidate_aliases.to_csv(aliases_path, index=False, float_format="%.17g")
        root = Path(source_root).resolve() if source_root is not None else None
        source_rows = []
        for path in source_paths:
            source = Path(path).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            try:
                label = source.relative_to(root) if root is not None else source
            except ValueError as exc:
                raise ValueError(
                    f"freeze source is outside source_root: {source}"
                ) from exc
            source_rows.append({"path": str(label), "sha256": sha256_file(source)})
        source_path = staging / source_name
        source_payload = {
            "Lifecycle": "candidate pre-activation freeze",
            "Authority": "Frozen candidate identity and source provenance only.",
            "Inputs": "Historical candidate manifests, evidence, contracts, and code.",
            "May be used for": "Pre-activation smoke and future activation preflight.",
            "Must not be used for": "Claiming shadow activation or submitting orders.",
            "Archive condition": "Archive when candidate semantics or runtime contract changes.",
            "freeze_version": freeze_version,
            "runtime_contract_version": TRUE_OOS_RUNTIME_CONTRACT_VERSION,
            "sources": source_rows,
        }
        source_path.write_text(
            json.dumps(source_payload, ensure_ascii=True, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        sha_path = staging / sha_name
        sha_rows = [
            {"path": manifest_name, "sha256": sha256_file(manifest_path)},
            {"path": source_name, "sha256": sha256_file(source_path)},
        ]
        if aliases_path is not None:
            sha_rows.append(
                {"path": aliases_name, "sha256": sha256_file(aliases_path)}
            )
        pd.DataFrame(sha_rows).to_csv(sha_path, index=False)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    outputs = {
        "candidate_manifest": destination / manifest_name,
        "source_manifest": destination / source_name,
        "sha256": destination / sha_name,
    }
    if candidate_aliases is not None:
        outputs["candidate_aliases"] = destination / aliases_name
    return outputs


def verify_sha256_manifest(
    sha_manifest_path: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> None:
    manifest_path = Path(sha_manifest_path)
    frame = pd.read_csv(manifest_path)
    if set(frame.columns) != {"path", "sha256"} or frame.empty:
        raise ValueError("SHA manifest must contain path and sha256 rows")
    root = Path(base_dir) if base_dir is not None else manifest_path.parent
    failures: list[str] = []
    for row in frame.itertuples(index=False):
        path = Path(str(row.path))
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file() or sha256_file(resolved) != str(row.sha256):
            failures.append(str(row.path))
    if failures:
        raise RuntimeError("SHA-256 verification failed: " + ", ".join(failures))


def verify_freeze_source_manifest(
    source_manifest_path: str | Path,
    *,
    source_root: str | Path,
) -> None:
    payload = json.loads(Path(source_manifest_path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("freeze source manifest must be an object")
    if payload.get("runtime_contract_version") != TRUE_OOS_RUNTIME_CONTRACT_VERSION:
        raise RuntimeError("freeze source manifest runtime contract is stale")
    rows = payload.get("sources")
    if not isinstance(rows, list) or not rows:
        raise ValueError("freeze source manifest has no sources")
    root = Path(source_root).resolve()
    failures: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("freeze source manifest row must be an object")
        relative = Path(str(row.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("freeze source path must be relative and contained")
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise ValueError("freeze source escapes source_root") from exc
        expected = str(row.get("sha256", ""))
        if not source.is_file() or sha256_file(source) != expected:
            failures.append(str(relative))
    if failures:
        raise RuntimeError(
            "freeze source SHA-256 verification failed: " + ", ".join(failures)
        )


def split_component_features(value: str) -> tuple[str, ...]:
    features = tuple(item.strip() for item in str(value).split("|") if item.strip())
    if not features:
        raise ValueError("candidate component_features must not be empty")
    return features


def build_epoch_training_panels(
    feature_panel: pd.DataFrame,
    execution_opens_by_symbol: Mapping[str, pd.DataFrame | pd.Series],
    candidate_manifest: pd.DataFrame,
    factor_registry: pd.DataFrame,
    epoch: EpochWindow,
) -> dict[str, pd.DataFrame]:
    """Build complete executable-return training panels for one TRUE OOS epoch."""
    if (
        not isinstance(feature_panel.index, pd.MultiIndex)
        or set(feature_panel.index.names) < {"decision_ts", "symbol"}
    ):
        raise ValueError("feature_panel must use a decision_ts/symbol MultiIndex")
    required_registry = {"feature_name", "signal_timeframe"}
    if not required_registry.issubset(factor_registry.columns):
        raise ValueError("factor_registry must contain feature_name and signal_timeframe")
    registry_timeframes = (
        factor_registry[list(required_registry)]
        .drop_duplicates("feature_name")
        .set_index("feature_name")["signal_timeframe"]
        .astype(str)
        .to_dict()
    )
    panel_symbols = set(feature_panel.index.get_level_values("symbol").astype(str))
    if panel_symbols != set(CANONICAL_SYMBOLS):
        raise ValueError("feature_panel must cover the canonical 20-symbol universe")
    if set(execution_opens_by_symbol) != set(CANONICAL_SYMBOLS):
        raise ValueError("execution opens must cover the canonical 20-symbol universe")
    train_start = pd.Timestamp(epoch.train_start)
    train_end = pd.Timestamp(epoch.train_end_exclusive)
    panel_times = pd.DatetimeIndex(feature_panel.index.get_level_values("decision_ts"))
    if (
        panel_times.min() > train_start
        or panel_times.max() < train_end - SIGNAL_DELTAS["1h"]
    ):
        raise RuntimeError("feature panel does not cover the complete epoch training window")

    result: dict[str, pd.DataFrame] = {}
    for horizon in sorted(
        candidate_manifest["horizon"].astype(str).unique(),
        key=lambda value: HORIZON_DELTAS[value],
    ):
        candidates = candidate_manifest.loc[
            candidate_manifest["horizon"].astype(str).eq(horizon)
        ]
        features = tuple(
            dict.fromkeys(
                feature
                for raw in candidates["component_features"].astype(str)
                for feature in split_component_features(raw)
            )
        )
        missing = sorted(set(features).difference(registry_timeframes))
        if missing:
            raise ValueError(
                "factor registry missing candidate features: " + ", ".join(missing)
            )
        timeframes = tuple(
            sorted(
                {registry_timeframes[feature] for feature in features},
                key=lambda value: SIGNAL_DELTAS[value],
            )
        )
        contract = ContinuousHoldingTimeContract(
            return_horizon=horizon,
            decision_interval=horizon,
            holding_interval=horizon,
            strategy_return_interval=horizon,
            signal_timeframes=timeframes,
        )
        route = factor_research.filter_frame_to_decision_frequency(
            feature_panel, horizon, SIGNAL_DELTAS
        )
        route_times = pd.DatetimeIndex(
            route.index.get_level_values("decision_ts")
        )
        route = route.loc[
            (route_times >= train_start) & (route_times < train_end)
        ].copy()
        labeled = crypto_panel.panel_with_executable_return(
            route,
            execution_opens_by_symbol,
            contract,
            SIGNAL_DELTAS,
        ).copy()
        labeled["forward_return"] = labeled["executable_return"]
        training = labeled.copy()
        expected_first = train_start
        expected_last = train_end - HORIZON_DELTAS[horizon]
        actual_times = pd.DatetimeIndex(training.index.unique()).sort_values()
        expected_times = pd.date_range(
            expected_first,
            expected_last,
            freq=HORIZON_DELTAS[horizon],
        )
        if (
            actual_times.empty
            or actual_times.min() != expected_first
            or actual_times.max() != expected_last
            or not actual_times.equals(expected_times)
            or training.groupby(level=0)["symbol"]
            .nunique()
            .ne(len(CANONICAL_SYMBOLS))
            .any()
        ):
            raise RuntimeError(
                f"{horizon} executable training panel is incomplete for the epoch"
            )
        result[horizon] = training
    return result


def fit_epoch_candidate_parameters(
    training_panels_by_horizon: Mapping[str, pd.DataFrame],
    candidate_manifest: pd.DataFrame,
    factor_registry: pd.DataFrame,
    epoch: EpochWindow,
    *,
    min_cross_section: int = 10,
) -> pd.DataFrame:
    """Fit train-only directions and weights once for a frozen live epoch."""
    required_registry = {"feature_name", "family"}
    if not required_registry.issubset(factor_registry.columns):
        raise ValueError("factor_registry must contain feature_name and family")
    family_map = dict(
        factor_registry[["feature_name", "family"]]
        .drop_duplicates("feature_name")
        .itertuples(index=False, name=None)
    )
    parameter_rows: list[dict[str, object]] = []
    for candidate in candidate_manifest.itertuples(index=False):
        horizon = str(candidate.horizon)
        if horizon not in HORIZON_DELTAS:
            raise ValueError(f"unsupported horizon: {horizon}")
        if horizon not in training_panels_by_horizon:
            raise ValueError(f"missing training panel for horizon: {horizon}")
        training_panel = training_panels_by_horizon[horizon]
        if not isinstance(training_panel.index, pd.DatetimeIndex) or training_panel.index.tz is None:
            raise ValueError(f"training panel for {horizon} must have a timezone-aware index")
        features = split_component_features(str(candidate.component_features))
        missing = sorted(set(features).difference(training_panel.columns))
        if missing:
            raise ValueError("training panel missing candidate features: " + ", ".join(missing))
        missing_families = sorted(set(features).difference(family_map))
        if missing_families:
            raise ValueError("factor registry missing families: " + ", ".join(missing_families))
        route = factor_research.filter_frame_to_decision_frequency(
            training_panel[["symbol", *features, "forward_return"]],
            horizon,
            HORIZON_DELTAS,
        )
        train = route.loc[
            (route.index >= pd.Timestamp(epoch.train_start))
            & (route.index < pd.Timestamp(epoch.train_end_exclusive))
        ]
        stats = factor_research.train_feature_stats(train, features, min_cross_section)
        if stats is None:
            raise ValueError(f"no train stats for {candidate.signal_equivalence_id}")
        directions = {name: stat.direction for name, stat in stats.items()}
        _, weights, diagnostics = factor_research.composite_weight_scores_weights_and_diagnostics(
            stats,
            str(candidate.weight_scheme),
            feature_families=family_map,
        )
        for feature in features:
            stat = stats[feature]
            parameter_rows.append(
                {
                    "freeze_version": candidate.freeze_version,
                    "signal_equivalence_id": candidate.signal_equivalence_id,
                    "horizon": horizon,
                    "epoch_index": epoch.epoch_index,
                    "train_start": epoch.train_start,
                    "train_end_exclusive": epoch.train_end_exclusive,
                    "feature_name": feature,
                    "direction": stat.direction,
                    "mean_ic": stat.mean_ic,
                    "icir": stat.icir,
                    "weight": float(weights[feature]),
                    "effective_factor_count": diagnostics["effective_factor_count"],
                }
            )
    return pd.DataFrame(parameter_rows)


def score_epoch_candidate_signals(
    current_panel: pd.DataFrame,
    candidate_manifest: pd.DataFrame,
    parameter_manifest: pd.DataFrame,
    epoch: EpochWindow,
    decision_ts: str | pd.Timestamp,
) -> pd.DataFrame:
    """Score one decision using parameters frozen for the current epoch."""
    if not isinstance(current_panel.index, pd.DatetimeIndex):
        raise TypeError("current_panel must use a DatetimeIndex")
    if current_panel.index.tz is None:
        raise ValueError("current_panel index must be timezone-aware")
    required_parameters = {
        "freeze_version", "signal_equivalence_id", "horizon", "epoch_index",
        "feature_name", "direction", "weight",
    }
    if not required_parameters.issubset(parameter_manifest.columns):
        raise ValueError("parameter manifest is incomplete")
    decision = pd.Timestamp(decision_ts)
    decision = decision.tz_localize("UTC") if decision.tz is None else decision.tz_convert("UTC")
    if not (pd.Timestamp(epoch.run_start) <= decision < pd.Timestamp(epoch.run_end_exclusive)):
        raise ValueError("decision_ts is outside the epoch run window")
    signal_rows: list[dict[str, object]] = []
    for candidate in candidate_manifest.itertuples(index=False):
        horizon = str(candidate.horizon)
        if horizon not in HORIZON_DELTAS:
            raise ValueError(f"unsupported horizon: {horizon}")
        if decision != decision.floor("h"):
            raise ValueError("decision_ts must be an exact UTC hour")
        if (decision - decision.normalize()) % HORIZON_DELTAS[horizon] != pd.Timedelta(0):
            continue
        features = split_component_features(str(candidate.component_features))
        missing = sorted(set(features).difference(current_panel.columns))
        if missing:
            raise ValueError("current panel missing candidate features: " + ", ".join(missing))
        candidate_parameters = parameter_manifest.loc[
            parameter_manifest["signal_equivalence_id"].astype(str).eq(
                str(candidate.signal_equivalence_id)
            )
            & parameter_manifest["epoch_index"].astype(int).eq(epoch.epoch_index)
        ]
        expected_freeze = str(candidate.freeze_version)
        if (
            candidate_parameters.empty
            or set(candidate_parameters["freeze_version"].astype(str))
            != {expected_freeze}
            or set(candidate_parameters["horizon"].astype(str)) != {horizon}
        ):
            raise ValueError(
                "parameter manifest identity does not match candidate freeze/horizon: "
                f"{candidate.signal_equivalence_id}"
            )
        selected = candidate_parameters
        if set(selected["feature_name"].astype(str)) != set(features):
            raise ValueError(
                f"parameter manifest does not cover {candidate.signal_equivalence_id}"
            )
        if selected.duplicated("feature_name").any():
            raise ValueError("parameter manifest has duplicate candidate features")
        directions = dict(
            selected[["feature_name", "direction"]].itertuples(index=False, name=None)
        )
        weights = dict(
            selected[["feature_name", "weight"]].itertuples(index=False, name=None)
        )
        current = current_panel.loc[
            current_panel.index == decision, ["symbol", *features]
        ]
        if current["symbol"].astype(str).nunique() != len(CANONICAL_SYMBOLS):
            raise ValueError(
                f"decision cross-section must contain all 20 symbols for {candidate.signal_equivalence_id}"
            )
        composite = factor_research.build_composite_frame(
            current, features, directions, weights, extra_columns=()
        )
        if composite["composite_signal"].isna().any():
            raise ValueError(f"incomplete live signal for {candidate.signal_equivalence_id}")
        memberships = select_long_short_memberships(
            composite.rename(columns={"composite_signal": "signal_value"}),
            long_count=4,
            short_count=4,
        )
        membership_map = memberships.set_index("symbol")["leg"].astype(str).to_dict()
        for row in composite[["symbol", "composite_signal"]].itertuples(index=False):
            signal_rows.append(
                {
                    "freeze_version": candidate.freeze_version,
                    "signal_equivalence_id": candidate.signal_equivalence_id,
                    "horizon": horizon,
                    "epoch_index": epoch.epoch_index,
                    "decision_ts": decision,
                    "symbol": row.symbol,
                    "signal_value": row.composite_signal,
                    "leg": membership_map.get(str(row.symbol), "flat"),
                }
            )
    return pd.DataFrame(signal_rows)


def fit_epoch_candidate_signals(
    training_panels_by_horizon: Mapping[str, pd.DataFrame],
    current_panel: pd.DataFrame,
    candidate_manifest: pd.DataFrame,
    factor_registry: pd.DataFrame,
    epoch: EpochWindow,
    decision_ts: str | pd.Timestamp,
    *,
    min_cross_section: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one epoch and score one decision through the two formal stages."""
    parameters = fit_epoch_candidate_parameters(
        training_panels_by_horizon,
        candidate_manifest,
        factor_registry,
        epoch,
        min_cross_section=min_cross_section,
    )
    signals = score_epoch_candidate_signals(
        current_panel, candidate_manifest, parameters, epoch, decision_ts
    )
    return signals, parameters


def select_long_short_memberships(
    scores: pd.DataFrame,
    *,
    long_count: int,
    short_count: int,
) -> pd.DataFrame:
    required = {"symbol", "signal_value"}
    if not required.issubset(scores.columns):
        raise ValueError("scores must contain symbol and signal_value")
    working = scores[list(required)].copy()
    working["symbol"] = working["symbol"].astype(str)
    working["signal_value"] = pd.to_numeric(working["signal_value"], errors="coerce")
    if working.isna().any(axis=None) or working["symbol"].duplicated().any():
        raise ValueError("scores must be complete and unique by symbol")
    if set(working["symbol"]) != set(CANONICAL_SYMBOLS):
        raise ValueError("scores must contain the canonical 20-symbol universe")
    if min(long_count, short_count) <= 0 or long_count + short_count > len(working):
        raise ValueError("invalid leg counts")
    ordered = working.sort_values(["signal_value", "symbol"], kind="mergesort").reset_index(drop=True)
    short = ordered.head(short_count).assign(leg="short")
    long = ordered.tail(long_count).assign(leg="long")
    return pd.concat([short, long], ignore_index=True)[["symbol", "signal_value", "leg"]]


@dataclass(frozen=True)
class ExchangeRule:
    symbol: str
    status: str
    min_qty: float
    step_size: float
    min_notional: float
    observed_ts: str


@dataclass(frozen=True)
class BookQuote:
    symbol: str
    bid: float
    ask: float
    observed_ts: str
    request_ts: str


def exchange_rules_from_binance(
    payload: Mapping[str, object],
    *,
    observed_ts: str,
) -> dict[str, ExchangeRule]:
    rows = payload.get("symbols")
    if not isinstance(rows, list):
        raise ValueError("exchangeInfo payload has no symbols list")
    result: dict[str, ExchangeRule] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        contract_symbol = str(row.get("symbol", ""))
        if not contract_symbol.endswith("USDT"):
            continue
        base = contract_symbol[:-4]
        if base not in CANONICAL_SYMBOLS:
            continue
        filters = {
            str(item.get("filterType")): item
            for item in row.get("filters", [])
            if isinstance(item, Mapping)
        }
        lot = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE")
        notional = filters.get("MIN_NOTIONAL") or filters.get("NOTIONAL")
        if not isinstance(lot, Mapping) or not isinstance(notional, Mapping):
            raise ValueError(f"incomplete Binance filters for {contract_symbol}")
        result[base] = ExchangeRule(
            symbol=base,
            status=str(row.get("status", "")),
            min_qty=float(lot["minQty"]),
            step_size=float(lot["stepSize"]),
            min_notional=float(notional.get("notional", notional.get("minNotional", 0.0))),
            observed_ts=observed_ts,
        )
    if set(result) != set(CANONICAL_SYMBOLS):
        raise ValueError("exchangeInfo does not cover canonical universe")
    return result


def book_quotes_from_binance(
    payload: Sequence[Mapping[str, object]],
    *,
    request_ts: str,
    observed_ts: str,
) -> dict[str, BookQuote]:
    result: dict[str, BookQuote] = {}
    for row in payload:
        contract_symbol = str(row.get("symbol", ""))
        if not contract_symbol.endswith("USDT"):
            continue
        symbol = contract_symbol[:-4]
        if symbol not in CANONICAL_SYMBOLS:
            continue
        bid = float(row.get("bidPrice", 0.0))
        ask = float(row.get("askPrice", 0.0))
        if bid <= 0 or ask <= 0 or bid > ask:
            raise ValueError(f"invalid Binance bookTicker for {contract_symbol}")
        result[symbol] = BookQuote(symbol, bid, ask, observed_ts, request_ts)
    if set(result) != set(CANONICAL_SYMBOLS):
        raise ValueError("bookTicker does not cover canonical universe")
    return result


def quantity_toward_zero(quantity: float, step_size: float) -> float:
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    value = Decimal(str(abs(quantity)))
    step = Decimal(str(step_size))
    units = (value / step).to_integral_value(rounding=ROUND_DOWN)
    rounded = units * step
    return float(rounded.copy_sign(Decimal(str(quantity))))


def plan_candidate_transitions(
    current_positions: Mapping[str, float],
    memberships: pd.DataFrame,
    quotes: Mapping[str, BookQuote],
    rules: Mapping[str, ExchangeRule],
    *,
    virtual_submit_ts: str | pd.Timestamp,
    target_gross_notional: float = 600.0,
    allow_partial: bool = True,
) -> pd.DataFrame:
    if target_gross_notional <= 0:
        raise ValueError("target_gross_notional must be positive")
    if set(current_positions).difference(CANONICAL_SYMBOLS):
        raise ValueError("current positions contain non-canonical symbols")
    desired = memberships.set_index("symbol")["leg"].astype(str).to_dict()
    if set(desired.values()).difference({"long", "short"}):
        raise ValueError("memberships contain invalid legs")
    long_count = sum(value == "long" for value in desired.values())
    short_count = sum(value == "short" for value in desired.values())
    if long_count != 4 or short_count != 4:
        raise ValueError("memberships must contain four long and four short symbols")
    rows: list[dict[str, object]] = []
    submit_ts = _utc_timestamp(virtual_submit_ts, field="virtual_submit_ts")
    slots = {"long": 0.5 * target_gross_notional / 4, "short": 0.5 * target_gross_notional / 4}
    for symbol in sorted(set(current_positions).union(desired)):
        previous = float(current_positions.get(symbol, 0.0))
        previous_side = "flat" if previous == 0 else ("long" if previous > 0 else "short")
        desired_side = desired.get(symbol, "flat")
        quote = quotes.get(symbol)
        rule = rules.get(symbol)
        quote_is_stale = (
            quote is not None
            and _utc_timestamp(quote.observed_ts, field="quote_observed_ts") < submit_ts
        )
        quote_requested_early = (
            quote is not None
            and _utc_timestamp(quote.request_ts, field="quote_request_ts") < submit_ts
        )
        quote_order_invalid = (
            quote is not None
            and _utc_timestamp(quote.observed_ts, field="quote_observed_ts")
            < _utc_timestamp(quote.request_ts, field="quote_request_ts")
        )
        rule_observed_late = (
            rule is not None
            and _utc_timestamp(rule.observed_ts, field="rule_observed_ts") > submit_ts
        )
        if (
            quote is None
            or rule is None
            or rule.status != "TRADING"
            or quote_is_stale
            or quote_requested_early
            or quote_order_invalid
            or rule_observed_late
        ):
            if not allow_partial:
                raise ValueError(f"missing valid quote or exchange rule for {symbol}")
            reason = (
                "failed_missing_quote"
                if quote is None
                else "failed_quote_requested_before_submit"
                if quote_requested_early
                else "failed_quote_timestamp_order"
                if quote_order_invalid
                else "failed_stale_quote"
                if quote_is_stale
                else "blocked_missing_rule"
                if rule is None
                else "blocked_rule_observed_after_submit"
                if rule_observed_late
                else "blocked_not_trading"
            )
            rows.append(
                {
                    "symbol": symbol,
                    "previous_signed_quantity": previous,
                    "desired_signed_quantity": previous,
                    "executed_quantity": 0.0,
                    "execution_side": "NONE",
                    "execution_price": np.nan,
                    "executed_notional": 0.0,
                    "status": reason,
                    "quote_observed_ts": None if quote is None else quote.observed_ts,
                    "rule_observed_ts": None if rule is None else rule.observed_ts,
                }
            )
            continue
        status = "hold_unchanged"
        target = previous
        execution_price = quote.ask if desired_side == "long" else quote.bid
        if desired_side == "flat":
            target = 0.0
            status = "close"
            execution_price = quote.bid if previous > 0 else quote.ask
        elif desired_side != previous_side:
            signed = 1.0 if desired_side == "long" else -1.0
            raw = signed * slots[desired_side] / execution_price
            rounded = quantity_toward_zero(raw, rule.step_size)
            valid = (
                abs(rounded) >= rule.min_qty
                and abs(rounded) * execution_price >= rule.min_notional
            )
            if valid:
                target = rounded
                status = "open" if previous_side == "flat" else "side_switch"
            else:
                target = previous
                status = "filtered_keep_previous"
        executed = target - previous
        rows.append(
            {
                "symbol": symbol,
                "previous_signed_quantity": previous,
                "desired_signed_quantity": target,
                "executed_quantity": executed,
                "execution_side": "BUY" if executed > 0 else ("SELL" if executed < 0 else "NONE"),
                "execution_price": float(execution_price),
                "executed_notional": abs(executed) * float(execution_price),
                "status": status,
                "quote_observed_ts": quote.observed_ts,
                "rule_observed_ts": rule.observed_ts,
            }
        )
    return pd.DataFrame(rows)


def aggregate_testnet_canary(
    candidate_transitions: pd.DataFrame,
    quotes: Mapping[str, BookQuote],
    rules: Mapping[str, ExchangeRule],
    *,
    canary_gross_notional: float = 600.0,
) -> pd.DataFrame:
    required = {"signal_equivalence_id", "symbol", "executed_quantity"}
    if not required.issubset(candidate_transitions.columns):
        raise ValueError("candidate transitions missing canary columns")
    if canary_gross_notional <= 0:
        raise ValueError("canary_gross_notional must be positive")
    net = candidate_transitions.groupby("symbol")["executed_quantity"].sum()
    raw_notional = {
        symbol: float(quantity) * (
            quotes[symbol].ask if quantity > 0 else quotes[symbol].bid
        )
        for symbol, quantity in net.items()
        if float(quantity) != 0.0
    }
    gross = sum(abs(value) for value in raw_notional.values())
    if gross == 0:
        return pd.DataFrame(
            columns=["symbol", "side", "quantity", "reference_price", "reference_notional"]
        )
    rows = []
    for symbol, notional in sorted(raw_notional.items()):
        price = quotes[symbol].ask if notional > 0 else quotes[symbol].bid
        quantity = quantity_toward_zero(
            np.sign(notional) * abs(notional) / gross * canary_gross_notional / price,
            rules[symbol].step_size,
        )
        if (
            abs(quantity) < rules[symbol].min_qty
            or abs(quantity) * price < rules[symbol].min_notional
        ):
            continue
        rows.append(
            {
                "symbol": symbol,
                "side": "BUY" if quantity > 0 else "SELL",
                "quantity": abs(quantity),
                "reference_price": price,
                "reference_notional": abs(quantity) * price,
            }
        )
    return pd.DataFrame(rows)


def apply_fill_to_position(
    state: Mapping[str, float],
    *,
    signed_fill_quantity: float,
    fill_price: float,
    fee: float,
) -> dict[str, float]:
    if fill_price <= 0 or fee < 0:
        raise ValueError("invalid fill price or fee")
    old_qty = float(state.get("quantity", 0.0))
    old_avg = float(state.get("average_entry_price", 0.0))
    realized = float(state.get("realized_pnl", 0.0))
    fees = float(state.get("fees", 0.0)) + fee
    new_qty = old_qty + signed_fill_quantity
    if old_qty == 0 or np.sign(old_qty) == np.sign(signed_fill_quantity):
        total = abs(old_qty) + abs(signed_fill_quantity)
        new_avg = (
            (abs(old_qty) * old_avg + abs(signed_fill_quantity) * fill_price) / total
            if total else 0.0
        )
    else:
        closed = min(abs(old_qty), abs(signed_fill_quantity))
        realized += closed * (fill_price - old_avg) * np.sign(old_qty)
        if new_qty == 0:
            new_avg = 0.0
        elif np.sign(new_qty) == np.sign(old_qty):
            new_avg = old_avg
        else:
            new_avg = fill_price
    return {
        "quantity": float(new_qty),
        "average_entry_price": float(new_avg),
        "realized_pnl": float(realized),
        "fees": float(fees),
    }


def marked_equity(
    positions: Mapping[str, Mapping[str, float]],
    marks: Mapping[str, float],
    *,
    initial_equity: float,
    fee_multiplier: float = 1.0,
) -> float:
    if initial_equity <= 0 or fee_multiplier <= 0:
        raise ValueError("equity and fee multiplier must be positive")
    equity = float(initial_equity)
    for symbol, state in positions.items():
        if symbol not in marks:
            raise ValueError(f"missing mark for {symbol}")
        quantity = float(state.get("quantity", 0.0))
        average = float(state.get("average_entry_price", 0.0))
        equity += float(state.get("realized_pnl", 0.0))
        equity += quantity * (float(marks[symbol]) - average)
        equity -= float(state.get("fees", 0.0)) * fee_multiplier
    return equity


def initial_shadow_state(candidate_manifest: pd.DataFrame) -> dict[str, object]:
    required = {"signal_equivalence_id", "account_equity"}
    if not required.issubset(candidate_manifest.columns):
        raise ValueError("candidate manifest missing shadow-state columns")
    if candidate_manifest["signal_equivalence_id"].astype(str).duplicated().any():
        raise ValueError("candidate state identities must be unique")
    return {
        "candidates": {
            str(row.signal_equivalence_id): {
                "initial_equity": float(row.account_equity),
                "positions": {},
            }
            for row in candidate_manifest.itertuples(index=False)
        }
    }


def reduce_shadow_fill_event(
    state: Mapping[str, object],
    event: Mapping[str, object],
) -> dict[str, object]:
    """Apply one authoritative virtual fill event to isolated candidate state."""
    required = {
        "signal_equivalence_id",
        "symbol",
        "signed_fill_quantity",
        "fill_price",
        "fee",
    }
    missing = sorted(required.difference(event))
    if missing:
        raise ValueError("virtual fill event missing fields: " + ", ".join(missing))
    result = deepcopy(dict(state))
    candidates = result.get("candidates")
    if not isinstance(candidates, dict):
        raise ValueError("shadow state has no candidates mapping")
    candidate_id = str(event["signal_equivalence_id"])
    if candidate_id not in candidates or not isinstance(candidates[candidate_id], dict):
        raise ValueError(f"unknown candidate in virtual fill event: {candidate_id}")
    candidate_state = candidates[candidate_id]
    positions = candidate_state.setdefault("positions", {})
    if not isinstance(positions, dict):
        raise ValueError("candidate positions state is not a mapping")
    symbol = str(event["symbol"])
    positions[symbol] = apply_fill_to_position(
        positions.get(symbol, {}),
        signed_fill_quantity=float(event["signed_fill_quantity"]),
        fill_price=float(event["fill_price"]),
        fee=float(event["fee"]),
    )
    return result


def plan_shadow_decision(
    signal_frame: pd.DataFrame,
    candidate_manifest: pd.DataFrame,
    state: Mapping[str, object],
    quotes: Mapping[str, BookQuote],
    rules: Mapping[str, ExchangeRule],
    *,
    decision_ts: str | pd.Timestamp,
    virtual_submit_ts: str | pd.Timestamp,
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, object], pd.DataFrame]:
    """Plan and apply one isolated virtual decision for every supplied candidate."""
    required_signals = {
        "freeze_version",
        "signal_equivalence_id",
        "decision_ts",
        "symbol",
        "signal_value",
        "leg",
    }
    if not required_signals.issubset(signal_frame.columns):
        raise ValueError("signal frame is incomplete")
    decision = _utc_timestamp(decision_ts, field="decision_ts")
    submit = _utc_timestamp(virtual_submit_ts, field="virtual_submit_ts")
    if submit < decision:
        raise ValueError("virtual_submit_ts cannot precede decision_ts")
    manifest = candidate_manifest.set_index(
        candidate_manifest["signal_equivalence_id"].astype(str), drop=False
    )
    result_state = deepcopy(dict(state))
    transition_frames: list[pd.DataFrame] = []
    events: list[dict[str, object]] = []
    for candidate_id, candidate_signals in signal_frame.groupby(
        signal_frame["signal_equivalence_id"].astype(str), sort=True
    ):
        if candidate_id not in manifest.index:
            raise ValueError(f"signal frame contains unknown candidate: {candidate_id}")
        row = manifest.loc[candidate_id]
        if isinstance(row, pd.DataFrame):
            raise ValueError(f"duplicate candidate manifest identity: {candidate_id}")
        signal_decisions = pd.to_datetime(
            candidate_signals["decision_ts"], utc=True
        ).unique()
        if len(signal_decisions) != 1 or pd.Timestamp(signal_decisions[0]) != decision:
            raise ValueError(f"signal decision timestamp mismatch: {candidate_id}")
        if (
            candidate_signals["symbol"].astype(str).nunique() != len(CANONICAL_SYMBOLS)
            or len(candidate_signals) != len(CANONICAL_SYMBOLS)
            or set(candidate_signals["symbol"].astype(str)) != set(CANONICAL_SYMBOLS)
        ):
            raise ValueError(f"candidate signal cross-section is incomplete: {candidate_id}")
        signal_freezes = candidate_signals["freeze_version"].astype(str).unique()
        if (
            len(signal_freezes) != 1
            or str(signal_freezes[0]) != str(row["freeze_version"])
        ):
            raise ValueError(f"signal freeze version mismatch: {candidate_id}")
        memberships = candidate_signals.loc[
            candidate_signals["leg"].astype(str).isin({"long", "short"}),
            ["symbol", "signal_value", "leg"],
        ]
        candidate_states = result_state.get("candidates")
        if not isinstance(candidate_states, dict) or candidate_id not in candidate_states:
            raise ValueError(f"candidate state is missing: {candidate_id}")
        candidate_state = candidate_states[candidate_id]
        positions = candidate_state.get("positions", {})
        if not isinstance(positions, Mapping):
            raise ValueError(f"candidate positions are invalid: {candidate_id}")
        current_quantities = {
            str(symbol): float(position.get("quantity", 0.0))
            for symbol, position in positions.items()
            if isinstance(position, Mapping)
        }
        transitions = plan_candidate_transitions(
            current_quantities,
            memberships,
            quotes,
            rules,
            virtual_submit_ts=submit,
            target_gross_notional=float(row["target_gross_notional"]),
        )
        transitions.insert(0, "decision_ts", decision)
        transitions.insert(0, "signal_equivalence_id", candidate_id)
        transitions.insert(0, "freeze_version", str(row["freeze_version"]))
        transition_frames.append(transitions)
        for transition in transitions.loc[
            transitions["executed_quantity"].ne(0.0)
        ].itertuples(index=False):
            fee = float(transition.executed_notional) * float(row["taker_fee_rate"])
            event = {
                "freeze_version": str(row["freeze_version"]),
                "signal_equivalence_id": candidate_id,
                "decision_ts": decision.isoformat(),
                "symbol": str(transition.symbol),
                "transition": str(transition.status),
                "signed_fill_quantity": float(transition.executed_quantity),
                "fill_price": float(transition.execution_price),
                "fill_ts": str(transition.quote_observed_ts),
                "fee": fee,
                "virtual_submit_ts": submit.isoformat(),
            }
            result_state = reduce_shadow_fill_event(result_state, event)
            events.append(event)

    equity_rows: list[dict[str, object]] = []
    candidate_states = result_state["candidates"]
    for candidate_id in sorted(signal_frame["signal_equivalence_id"].astype(str).unique()):
        candidate_state = candidate_states[candidate_id]
        positions = candidate_state["positions"]
        marks = {
            symbol: (
                quotes[symbol].bid
                if float(position.get("quantity", 0.0)) >= 0
                else quotes[symbol].ask
            )
            for symbol, position in positions.items()
        }
        for multiplier in (1.0, 1.5, 2.0):
            equity_rows.append(
                {
                    "freeze_version": str(manifest.loc[candidate_id, "freeze_version"]),
                    "signal_equivalence_id": candidate_id,
                    "decision_ts": decision,
                    "cost_multiplier": multiplier,
                    "equity": marked_equity(
                        positions,
                        marks,
                        initial_equity=float(candidate_state["initial_equity"]),
                        fee_multiplier=multiplier,
                    ),
                }
            )
    transitions = (
        pd.concat(transition_frames, ignore_index=True)
        if transition_frames
        else pd.DataFrame()
    )
    return transitions, events, result_state, pd.DataFrame(equity_rows)


def _json_compatible(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        timestamp = value
        timestamp = (
            timestamp.tz_localize("UTC")
            if timestamp.tz is None
            else timestamp.tz_convert("UTC")
        )
        return timestamp.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _frame_records(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    sort_by: Sequence[str],
) -> list[dict[str, object]]:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError("evidence frame missing columns: " + ", ".join(missing))
    selected = frame.loc[:, list(columns)].sort_values(
        list(sort_by), kind="stable"
    )
    return [
        _json_compatible(record)  # type: ignore[return-value]
        for record in selected.to_dict(orient="records")
    ]


def build_shadow_decision_artifacts(
    signal_frame: pd.DataFrame,
    candidate_manifest: pd.DataFrame,
    state: Mapping[str, object],
    quotes: Mapping[str, BookQuote],
    rules: Mapping[str, ExchangeRule],
    source_receipts: Sequence[SourceReceipt],
    input_lineage: Mapping[str, object],
    signal_lineage_by_candidate: Mapping[str, Mapping[str, object]],
    *,
    horizon: str,
    decision_ts: str | pd.Timestamp,
    signal_ready_ts: str | pd.Timestamp,
    virtual_submit_ts: str | pd.Timestamp,
) -> ShadowDecisionArtifacts:
    """Build the shared live/replay decision result and eight evidence bundles."""
    readiness = classify_decision_readiness(
        horizon=horizon,
        decision_ts=decision_ts,
        signal_ready_ts=signal_ready_ts,
    )
    if readiness.status != "ready":
        raise RuntimeError("missed_decision must not enter the decision execution path")
    ready = _utc_timestamp(signal_ready_ts, field="signal_ready_ts")
    submit = _utc_timestamp(virtual_submit_ts, field="virtual_submit_ts")
    if submit < ready:
        raise ValueError("virtual_submit_ts cannot precede signal_ready_ts")
    if not source_receipts:
        raise ValueError("source receipts must not be empty")
    late_receipts = sorted(
        receipt.receipt_id
        for receipt in source_receipts
        if _utc_timestamp(receipt.data_observed_ts, field="data_observed_ts") > ready
    )
    if late_receipts:
        raise ValueError(
            "signal uses source receipts observed after signal_ready_ts: "
            + ", ".join(late_receipts)
        )
    receipt_ids = {receipt.receipt_id for receipt in source_receipts}
    lineage_fields = input_lineage.get("fields")
    if not isinstance(lineage_fields, list) or not lineage_fields:
        raise ValueError("input_lineage must contain non-empty fields")
    for field in lineage_fields:
        if not isinstance(field, Mapping):
            raise ValueError("input lineage field must be an object")
        if not str(field.get("field_name", "")).strip():
            raise ValueError("input lineage field_name must not be empty")
        if str(field.get("source_receipt_id", "")) not in receipt_ids:
            raise ValueError("input lineage references an unknown source receipt")
    cutoff = _utc_timestamp(
        str(input_lineage.get("feature_cutoff_ts", "")),
        field="feature_cutoff_ts",
    )
    if cutoff > ready:
        raise ValueError("feature_cutoff_ts cannot exceed signal_ready_ts")

    candidate_ids = sorted(signal_frame["signal_equivalence_id"].astype(str).unique())
    manifest = candidate_manifest.assign(
        signal_equivalence_id=candidate_manifest["signal_equivalence_id"].astype(str)
    ).set_index("signal_equivalence_id", drop=False)
    if "horizon" not in manifest.columns:
        raise ValueError("candidate manifest is missing horizon")
    for candidate_id in candidate_ids:
        if candidate_id not in manifest.index:
            raise ValueError(f"candidate manifest is missing: {candidate_id}")
        candidate_row = manifest.loc[candidate_id]
        if isinstance(candidate_row, pd.DataFrame):
            raise ValueError(f"duplicate candidate manifest identity: {candidate_id}")
        if str(candidate_row["horizon"]) != horizon:
            raise ValueError(f"candidate horizon mismatch: {candidate_id}")
        lineage = signal_lineage_by_candidate.get(candidate_id)
        required_lineage = {
            "factor_values",
            "directions",
            "weights",
            "parameter_sha256",
        }
        if not isinstance(lineage, Mapping) or not required_lineage.issubset(lineage):
            raise ValueError(f"candidate signal lineage is incomplete: {candidate_id}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(lineage["parameter_sha256"])):
            raise ValueError(f"candidate parameter SHA is invalid: {candidate_id}")

    transitions, events, result_state, equity = plan_shadow_decision(
        signal_frame,
        candidate_manifest,
        state,
        quotes,
        rules,
        decision_ts=decision_ts,
        virtual_submit_ts=virtual_submit_ts,
    )
    before_candidates = state.get("candidates")
    after_candidates = result_state.get("candidates")
    if not isinstance(before_candidates, Mapping) or not isinstance(
        after_candidates, Mapping
    ):
        raise ValueError("shadow state has no candidates mapping")
    recovered_state = deepcopy(dict(state))
    for event in events:
        recovered_state = reduce_shadow_fill_event(recovered_state, event)
    if stable_json_sha256(recovered_state) != stable_json_sha256(result_state):
        raise RuntimeError("fill-event recovery does not reproduce planned state")
    recovered_candidates = recovered_state.get("candidates")
    if not isinstance(recovered_candidates, Mapping):
        raise ValueError("recovered state has no candidates mapping")

    receipt_records = [
        _json_compatible(asdict(receipt))
        for receipt in sorted(source_receipts, key=lambda item: item.receipt_id)
    ]
    dimensions: dict[str, object] = {}
    for candidate_id in candidate_ids:
        candidate_signals = signal_frame.loc[
            signal_frame["signal_equivalence_id"].astype(str).eq(candidate_id)
        ]
        candidate_transitions = transitions.loc[
            transitions["signal_equivalence_id"].astype(str).eq(candidate_id)
        ]
        candidate_events = [
            _json_compatible(event)
            for event in events
            if str(event["signal_equivalence_id"]) == candidate_id
        ]
        candidate_equity = equity.loc[
            equity["signal_equivalence_id"].astype(str).eq(candidate_id)
        ]
        signal_records = _frame_records(
            candidate_signals,
            columns=("symbol", "signal_value", "leg"),
            sort_by=("symbol",),
        )
        membership_records = [
            {"symbol": record["symbol"], "leg": record["leg"]}
            for record in signal_records
        ]
        transition_records = _frame_records(
            candidate_transitions,
            columns=(
                "symbol",
                "previous_signed_quantity",
                "desired_signed_quantity",
                "executed_quantity",
                "execution_side",
                "execution_price",
                "executed_notional",
                "status",
                "quote_observed_ts",
                "rule_observed_ts",
            ),
            sort_by=("symbol",),
        )
        for record in transition_records:
            symbol = str(record["symbol"])
            rule = rules.get(symbol)
            quote = quotes.get(symbol)
            record["exchange_rule"] = (
                None if rule is None else _json_compatible(asdict(rule))
            )
            record["quote_request_ts"] = (
                None if quote is None else quote.request_ts
            )
            record["reduce_only"] = record["status"] == "close"
            record["risk_result"] = str(record["status"])
        equity_records = _frame_records(
            candidate_equity,
            columns=("cost_multiplier", "equity"),
            sort_by=("cost_multiplier",),
        )
        before_state = deepcopy(before_candidates.get(candidate_id))
        after_state = deepcopy(after_candidates.get(candidate_id))
        if before_state is None or after_state is None:
            raise ValueError(f"candidate state is missing: {candidate_id}")
        recovered_candidate = deepcopy(recovered_candidates.get(candidate_id))
        if recovered_candidate is None:
            raise ValueError(f"recovered candidate state is missing: {candidate_id}")
        positions = after_state.get("positions", {})
        if not isinstance(positions, Mapping):
            raise ValueError(f"candidate positions are invalid: {candidate_id}")
        marked_notionals = {
            symbol: float(position.get("quantity", 0.0))
            * (
                quotes[symbol].bid
                if float(position.get("quantity", 0.0)) >= 0
                else quotes[symbol].ask
            )
            for symbol, position in positions.items()
            if isinstance(position, Mapping)
        }
        realized = sum(
            float(position.get("realized_pnl", 0.0))
            for position in positions.values()
            if isinstance(position, Mapping)
        )
        fees = sum(
            float(position.get("fees", 0.0))
            for position in positions.values()
            if isinstance(position, Mapping)
        )
        initial_equity = float(after_state["initial_equity"])
        dimensions[candidate_id] = {
            "data_availability_consistency": {
                "receipts": receipt_records,
                "input_lineage": _json_compatible(input_lineage),
                "feature_cutoff_ts": cutoff.isoformat(),
                "signal_ready_ts": ready.isoformat(),
            },
            "decision_clock_consistency": {
                **_json_compatible(asdict(readiness)),  # type: ignore[arg-type]
                "virtual_submit_ts": submit.isoformat(),
            },
            "signal_consistency": {
                "signals": signal_records,
                "lineage": _json_compatible(
                    signal_lineage_by_candidate[candidate_id]
                ),
            },
            "membership_consistency": {"memberships": membership_records},
            "position_consistency": {
                "before": _json_compatible(before_state),
                "after": _json_compatible(after_state),
            },
            "order_intent_consistency": {"transitions": transition_records},
            "execution_accounting_consistency": {
                "events": candidate_events,
                "equity": equity_records,
                "cash_balance_1x": initial_equity + realized - fees,
                "realized_pnl": realized,
                "fees_1x": fees,
                "gross_exposure": sum(abs(value) for value in marked_notionals.values()),
                "net_exposure": sum(marked_notionals.values()),
                "turnover_notional": float(
                    candidate_transitions["executed_notional"].sum()
                ),
            },
            "failure_recovery_consistency": {
                "pre_state_sha256": stable_json_sha256(before_state),
                "post_state_sha256": stable_json_sha256(after_state),
                "recovered_state_sha256": stable_json_sha256(recovered_candidate),
                "recovery_matches": (
                    stable_json_sha256(after_state)
                    == stable_json_sha256(recovered_candidate)
                ),
                "event_count": len(candidate_events),
            },
        }
    return ShadowDecisionArtifacts(
        transitions=transitions,
        events=tuple(events),
        state=result_state,
        equity=equity,
        dimensions=dimensions,
    )


def persist_shadow_fill_events(
    ledger: "AppendOnlyEventLedger",
    events: Sequence[Mapping[str, object]],
    *,
    initial_state: Mapping[str, object],
) -> dict[str, object]:
    """Recover from authority events, then idempotently append a fill batch."""
    state = ledger.restore_state(
        reduce_shadow_fill_event,
        initial_state=initial_state,
    )
    for event in events:
        next_state = reduce_shadow_fill_event(state, event)
        appended = ledger.append(event, next_state)
        if appended:
            state = next_state
    return state


class AppendOnlyEventLedger:
    """JSONL event source with strict idempotency and atomic state snapshots."""

    def __init__(
        self,
        event_path: str | Path,
        state_path: str | Path,
        *,
        key_fields: Sequence[str] = (
            "freeze_version",
            "signal_equivalence_id",
            "decision_ts",
            "symbol",
            "transition",
        ),
    ) -> None:
        if not key_fields or any(not str(field).strip() for field in key_fields):
            raise ValueError("event ledger key_fields must not be empty")
        self.event_path = Path(event_path)
        self.state_path = Path(state_path)
        self.lock_path = self.event_path.with_suffix(self.event_path.suffix + ".lock")
        self.key_fields = tuple(map(str, key_fields))

    def event_key(self, event: Mapping[str, object]) -> tuple[str, ...]:
        missing = [
            field
            for field in self.key_fields
            if not str(event.get(field, "")).strip()
        ]
        if missing:
            raise ValueError("event missing idempotency fields: " + ", ".join(missing))
        return tuple(str(event[field]) for field in self.key_fields)

    def _read_events_unlocked(self) -> list[dict[str, object]]:
        if not self.event_path.exists():
            return []
        events = []
        seen: dict[tuple[str, ...], dict[str, object]] = {}
        for line_number, line in enumerate(
            self.event_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid event JSON at line {line_number}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"event line {line_number} is not an object")
            key = self.event_key(event)
            if key in seen:
                raise ValueError(f"duplicate event idempotency key at line {line_number}")
            seen[key] = event
            events.append(event)
        return events

    def _write_events_unlocked(
        self, events: Sequence[Mapping[str, object]]
    ) -> None:
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.event_path.with_suffix(
            self.event_path.suffix + f".{uuid4().hex}.tmp"
        )
        block = "".join(
            json.dumps(dict(event), ensure_ascii=True, sort_keys=True) + "\n"
            for event in events
        )
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.event_path)

    def read_events(self) -> list[dict[str, object]]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH)
            try:
                return self._read_events_unlocked()
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def append(self, event: Mapping[str, object], state: Mapping[str, object]) -> bool:
        normalized = dict(event)
        key = self.event_key(normalized)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                existing = {
                    self.event_key(item): item for item in self._read_events_unlocked()
                }
                if key in existing:
                    if stable_json_sha256(existing[key]) != stable_json_sha256(normalized):
                        raise ValueError(
                            "idempotency key already exists with a different payload"
                        )
                    return False
                self._write_events_unlocked([*existing.values(), normalized])
                self.write_state(state)
                return True
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def append_batch(
        self,
        events: Sequence[Mapping[str, object]],
        state: Mapping[str, object],
    ) -> int:
        """Atomically append one decision batch and then replace its snapshot.

        Identical complete retries are accepted. A retry that overlaps only part
        of an existing batch fails closed because it indicates a broken decision
        boundary or an older non-atomic writer.
        """
        normalized = [dict(event) for event in events]
        keys = [self.event_key(event) for event in normalized]
        if len(keys) != len(set(keys)):
            raise ValueError("event batch contains duplicate idempotency keys")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                existing = {
                    self.event_key(item): item for item in self._read_events_unlocked()
                }
                present = [key in existing for key in keys]
                if any(present):
                    if not all(present):
                        raise RuntimeError(
                            "event batch is partially present; decision boundary is corrupt"
                        )
                    for key, event in zip(keys, normalized):
                        if stable_json_sha256(existing[key]) != stable_json_sha256(event):
                            raise ValueError(
                                "event batch key exists with a different payload"
                            )
                    return 0
                if normalized:
                    self._write_events_unlocked(
                        [*existing.values(), *normalized]
                    )
                self.write_state(state)
                return len(normalized)
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def write_state(self, state: Mapping[str, object]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(dict(state), ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)

    def load_state(self) -> dict[str, object]:
        if not self.state_path.exists():
            return {}
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("state snapshot is not an object")
        return payload

    def restore_state(
        self,
        reducer,
        *,
        initial_state: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Rebuild the disposable snapshot from the authoritative event stream."""
        state: dict[str, object] = dict(initial_state or {})
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                for event in self._read_events_unlocked():
                    reduced = reducer(state, event)
                    if not isinstance(reduced, Mapping):
                        raise TypeError("event reducer must return a mapping")
                    state = dict(reduced)
                self.write_state(state)
                return state
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def lifecycle_metadata() -> dict[str, str]:
    return {
        "Lifecycle": "candidate",
        "Authority": "TRUE OOS shadow infrastructure; no production trading authority.",
        "May be used for": "candidate freeze, virtual ledger, and testnet canary validation.",
        "Must not be used for": "production orders or candidate performance decisions.",
        "Archive condition": "archive when superseded or candidate semantics change.",
    }


__all__ = [
    "AsReceivedSnapshotStore",
    "AppendOnlyEventLedger",
    "BookQuote",
    "CANONICAL_SYMBOLS",
    "CONSISTENCY_DIMENSIONS",
    "SOURCE_EQUIVALENCE_STATUSES",
    "DecisionReadiness",
    "EXPECTED_HORIZON_COUNTS",
    "EpochWindow",
    "ExchangeRule",
    "REQUIRED_ACTIVATION_PREFLIGHTS",
    "ShadowDecisionArtifacts",
    "SourceReceipt",
    "SourceFreshness",
    "TRUE_OOS_RUNTIME_CONTRACT_VERSION",
    "aggregate_testnet_canary",
    "apply_fill_to_position",
    "build_activation_intent",
    "build_activation_manifest",
    "build_candidate_freeze_manifest",
    "build_epoch_window",
    "build_epoch_training_panels",
    "build_missed_decision_records",
    "build_replay_live_consistency_record",
    "build_revision_consistency_amendments",
    "build_shadow_decision_artifacts",
    "build_source_revision_event",
    "build_source_equivalence_record",
    "build_source_reference_time_contract",
    "build_source_reference_queue_records",
    "classify_source_reference_action",
    "build_source_equivalence_consistency_amendments",
    "build_source_observation_rows",
    "book_quotes_from_binance",
    "classify_decision_readiness",
    "consistency_evidence_sha256",
    "due_candidate_rows",
    "detect_source_observation_revisions",
    "exchange_rules_from_binance",
    "fit_epoch_candidate_parameters",
    "fit_epoch_candidate_signals",
    "first_eligible_decision_ts",
    "lifecycle_metadata",
    "marked_equity",
    "initial_shadow_state",
    "persist_shadow_fill_events",
    "plan_shadow_decision",
    "plan_candidate_transitions",
    "quantity_toward_zero",
    "require_activation_intent_publishable",
    "select_long_short_memberships",
    "score_epoch_candidate_signals",
    "reduce_shadow_fill_event",
    "require_true_oos_runtime_contract",
    "sha256_file",
    "stable_json_sha256",
    "source_values_at_exact_label",
    "validate_source_freshness",
    "verify_sha256_manifest",
    "verify_activation_intent",
    "verify_activation_manifest",
    "verify_freeze_source_manifest",
    "verify_json_sha256_sidecar",
    "write_freeze_bundle",
    "write_immutable_input_snapshot",
    "write_immutable_json_with_sha256",
]
