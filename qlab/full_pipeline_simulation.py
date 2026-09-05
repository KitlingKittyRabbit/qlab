"""Formal, pre-simulation diagnostics for the complete L0--L4 route.

Lifecycle: candidate implementation for the Issue #34 known-truth simulation
blueprint.  Authority: the sole qlab public entries for the observed-effect
inventory and its signal-level observed beta-total scale evidence.  Inputs: two
independently frozen B/C cache and price identities plus a pre-frozen registry,
windows, and candidate contract.  May be used for: retaining every unfiltered
candidate estimate and producing duplicate-aware observed beta-total scale
evidence.  Must not be used for: candidate selection, significance testing,
L2/L3/L4 discovery, G_beta_total_v1 simulation-grid construction, simulation
generation, or a research conclusion.  Archive condition: this v1 contract is
superseded by an approved, versioned successor.

The module deliberately reuses the formal qlab panel/rank and executable
return paths; it does not approximate them in a research-layer calculation.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import factor_research
from .data.crypto import keystore_coinglass_factors as factor_registry
from .data.crypto.keystore_coinglass_panel import (
    build_decision_grid_index,
    build_panel_from_payloads,
    signal_timeframe_from_scope,
)
from .data.crypto.panel import panel_with_executable_return
from .data.crypto.strategy_time_contract import (
    ContinuousHoldingTimeContract,
    validate_continuous_holding_contract,
    validate_decision_phase,
)


@dataclass(frozen=True)
class ObservedEffectCandidate:
    """One pre-frozen ``(feature, horizon)`` identity for the scale inventory.

    ``canonical_orientation`` is only the pre-frozen orientation used when
    checking an exact duplicate.  It never changes the candidate's own
    ``beta_obs`` estimate.  A negative value is how a sign alias is placed in
    the same duplicate group as its canonical expression.
    """

    candidate_id: str
    feature_name: str
    return_horizon: str
    canonical_orientation: int = 1
    declared_alias_of: str | None = None


@dataclass(frozen=True)
class DecisionWindow:
    """Closed UTC decision window frozen before any observed-effect run."""

    start: pd.Timestamp
    end: pd.Timestamp


def _json_content_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _frame_content_sha256(frame: pd.DataFrame | pd.Series) -> str:
    """Hash frame structure and ordered values without serialising to CSV."""
    if not isinstance(frame, (pd.DataFrame, pd.Series)):
        raise ValueError("frozen input values must be pandas DataFrame or Series")
    if isinstance(frame, pd.DataFrame):
        columns = [str(column) for column in frame.columns]
        dtypes = [str(dtype) for dtype in frame.dtypes]
        shape = list(frame.shape)
    else:
        columns = [str(frame.name)]
        dtypes = [str(frame.dtype)]
        shape = [len(frame), 1]
    metadata = {
        "kind": type(frame).__name__,
        "columns": columns,
        "dtypes": dtypes,
        "index_names": [str(name) for name in frame.index.names],
        "index_dtype": str(frame.index.dtype),
        "shape": shape,
    }
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8"))
    digest.update(
        pd.util.hash_pandas_object(frame, index=True)
        .to_numpy(dtype="<u8", copy=False)
        .tobytes()
    )
    return digest.hexdigest()


def _mapping_content_sha256(
    values: Mapping[str, object],
    *,
    nested: bool,
) -> str:
    if not isinstance(values, Mapping):
        raise ValueError("frozen input must be a mapping")
    digest = hashlib.sha256()
    for outer_key in sorted(values, key=str):
        digest.update(str(outer_key).encode("utf-8"))
        digest.update(b"\0")
        value = values[outer_key]
        if nested:
            if not isinstance(value, Mapping):
                raise ValueError("frozen cache scopes must be mappings")
            for inner_key in sorted(value, key=str):
                digest.update(str(inner_key).encode("utf-8"))
                digest.update(b"\0")
                digest.update(_frame_content_sha256(value[inner_key]).encode("ascii"))
                digest.update(b"\n")
        else:
            digest.update(_frame_content_sha256(value).encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


def _candidate_specs_content_sha256(
    candidate_specs: Sequence[ObservedEffectCandidate],
) -> str:
    payload = [
        {
            "candidate_id": str(spec.candidate_id),
            "feature_name": str(spec.feature_name),
            "return_horizon": str(spec.return_horizon),
            "canonical_orientation": int(spec.canonical_orientation),
            "declared_alias_of": spec.declared_alias_of,
        }
        for spec in candidate_specs
    ]
    return _json_content_sha256(payload)


def _symbols_content_sha256(symbols: Sequence[str]) -> str:
    return _json_content_sha256([str(symbol) for symbol in symbols])


def _decision_windows_content_sha256(
    decision_windows: Mapping[str, DecisionWindow],
) -> str:
    payload = []
    for horizon in sorted(decision_windows):
        window = decision_windows[horizon]
        start = _utc_timestamp(window.start, field=f"decision window {horizon} start")
        end = _utc_timestamp(window.end, field=f"decision window {horizon} end")
        payload.append(
            {
                "horizon": str(horizon),
                "start": start.isoformat(),
                "end": end.isoformat(),
            }
        )
    return _json_content_sha256(payload)


_DECISION_COVERAGE_POLICY = "per_candidate_horizon_complete_cross_section_v1"


def _horizon_contract_content_sha256(
    horizon_deltas: Mapping[str, pd.Timedelta],
    *,
    execution_delay_minutes: int,
    require_complete_cross_sections: bool,
    minimum_support_rows: int,
    min_common_panel_rows: int,
    decision_coverage_policy: str = _DECISION_COVERAGE_POLICY,
) -> str:
    payload = {
        "horizon_deltas": {
            str(horizon): int(pd.Timedelta(delta).value)
            for horizon, delta in sorted(horizon_deltas.items())
        },
        "execution_delay_minutes": int(execution_delay_minutes),
        "require_complete_cross_sections": bool(require_complete_cross_sections),
        "minimum_support_rows": int(minimum_support_rows),
        "min_common_panel_rows": int(min_common_panel_rows),
        "decision_coverage_policy": str(decision_coverage_policy),
    }
    return _json_content_sha256(payload)


def _registry_content_sha256(registry_frame: pd.DataFrame) -> str:
    """Hash the validated registry including its canonical row and column order."""
    return _frame_content_sha256(registry_frame.reset_index(drop=True))


@dataclass(frozen=True)
class ObservedEffectScaleInput:
    """One immutable input identity supplied to the formal v1 estimator.

    The caller must create two independently frozen instances, named ``B`` and
    ``C``.  Their equality is checked here, not reconstructed in research.  A
    production input also carries the independently recorded source-manifest
    digest; its identity must bind that digest and both in-memory content
    digests before the data can enter this entry point.
    """

    cache_payloads: Mapping[str, Mapping[str, pd.DataFrame]]
    minute_klines_by_symbol: Mapping[str, pd.DataFrame | pd.Series]
    input_identity: str
    source_manifest_sha256: str = ""
    cache_sha256: str = field(init=False)
    minute_klines_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cache_sha256",
            _mapping_content_sha256(self.cache_payloads, nested=True),
        )
        object.__setattr__(
            self,
            "minute_klines_sha256",
            _mapping_content_sha256(self.minute_klines_by_symbol, nested=False),
        )


@dataclass(frozen=True)
class ObservedEffectScaleContract:
    """Pre-frozen identity and coverage contract for one v1 scale inventory."""

    admitted_symbols: tuple[str, ...]
    candidate_specs: tuple[ObservedEffectCandidate, ...]
    decision_windows: Mapping[str, DecisionWindow]
    horizon_deltas: Mapping[str, pd.Timedelta]
    minimum_support_rows: int
    min_common_panel_rows: int
    registry_identity: str
    candidate_set_identity: str
    coverage_contract_identity: str
    registry_feature_count: int
    candidate_pair_count: int
    admitted_symbol_count: int
    decision_window_identity: str
    registry_content_sha256: str
    candidate_specs_sha256: str
    admitted_symbols_sha256: str
    decision_windows_sha256: str
    horizon_contract_sha256: str
    require_complete_cross_sections: bool = True
    execution_delay_minutes: int = 4
    decision_coverage_policy: str = _DECISION_COVERAGE_POLICY


@dataclass(frozen=True)
class ObservedEffectScaleArtifacts:
    """Unfiltered estimates, duplicate identities, and distribution summaries."""

    candidate_estimates: pd.DataFrame
    duplicate_mapping: pd.DataFrame
    distribution_summary: pd.DataFrame
    input_case_comparison: pd.DataFrame
    input_identity_manifest: pd.DataFrame


@dataclass(frozen=True)
class ObservedEffectBetaTotalArtifacts:
    """Signal-level observed ``beta_total`` scale evidence.

    This is deliberately an observed-scale mapping, not the user-frozen
    ``G_beta_total_v1`` simulation grid.  It consumes the already completed
    per-(candidate, horizon) inventory and never reads significance, L2/L3/L4
    selection, or discovery fields.
    """

    signal_level_scales: pd.DataFrame
    canonical_signal_mapping: pd.DataFrame
    distribution_summary: pd.DataFrame
    input_case_comparison: pd.DataFrame
    input_identity_manifest: pd.DataFrame


_ESTIMATE_COLUMNS = [
    "input_case",
    "candidate_id",
    "feature_name",
    "return_horizon",
    "declared_alias_of",
    "canonical_orientation",
    "status",
    "failure_reason",
    "support_rows",
    "support_asset_count",
    "support_decision_count",
    "support_start",
    "support_end",
    "signal_rows_on_window",
    "finite_signal_rows",
    "finite_return_rows",
    "alpha_obs",
    "beta_obs",
    "sigma_y",
    "delta_obs",
]

_DUPLICATE_COLUMNS = [
    "input_case",
    "candidate_id",
    "return_horizon",
    "duplicate_group_id",
    "canonical_candidate_id",
    "declared_alias_of",
    "canonical_orientation",
    "is_exact_duplicate",
    "signal_support_rows",
    "finite_signal_rows",
    "status",
]

_DISTRIBUTION_COLUMNS = [
    "input_case",
    "return_horizon",
    "distribution",
    "status",
    "candidate_count",
    "positive_count",
    "negative_count",
    "zero_count",
    "skewness",
    "p10",
    "p50",
    "p90",
]

_INPUT_COMPARISON_COLUMNS = [
    "candidate_id",
    "return_horizon",
    "signal_equal",
    "return_equal",
    "estimate_equal",
    "status",
]

_INPUT_IDENTITY_COLUMNS = [
    "input_case",
    "input_identity",
    "source_manifest_sha256",
    "cache_sha256",
    "minute_klines_sha256",
    "coverage_contract_identity",
    "registry_identity",
    "registry_content_sha256",
    "candidate_set_identity",
    "candidate_specs_sha256",
    "admitted_symbol_count",
    "admitted_symbols_sha256",
    "decision_window_identity",
    "decision_windows_sha256",
    "horizon_contract_sha256",
    "execution_delay_minutes",
    "decision_coverage_policy",
]


@dataclass(frozen=True)
class _SingleInputArtifacts:
    artifacts: ObservedEffectScaleArtifacts
    signal_by_candidate: Mapping[str, pd.Series]
    return_by_candidate: Mapping[str, pd.Series]


def _utc_timestamp(value: pd.Timestamp | str, *, field: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{field} must be an explicit UTC timestamp")
    return timestamp.tz_convert("UTC")


def _validate_candidate_specs(
    candidate_specs: Sequence[ObservedEffectCandidate],
    registry_frame: pd.DataFrame,
    horizon_deltas: Mapping[str, pd.Timedelta],
) -> tuple[ObservedEffectCandidate, ...]:
    specs = tuple(candidate_specs)
    if not specs:
        raise ValueError("candidate_specs must not be empty")
    candidate_ids = [str(spec.candidate_id) for spec in specs]
    if any(not value for value in candidate_ids) or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_ids must be non-empty and unique")
    candidate_pairs = [(str(spec.feature_name), str(spec.return_horizon)) for spec in specs]
    if len(set(candidate_pairs)) != len(candidate_pairs):
        raise ValueError(
            "candidate feature/horizon identities must be unique; "
            "register aliases as distinct feature identities"
        )
    registry_by_feature = registry_frame.set_index("feature_name", drop=False)
    for spec in specs:
        if spec.feature_name not in registry_by_feature.index:
            raise ValueError(f"candidate references a registry feature not present: {spec.feature_name}")
        if spec.return_horizon not in horizon_deltas:
            raise ValueError(f"candidate return_horizon missing from horizon_deltas: {spec.return_horizon}")
        if int(spec.canonical_orientation) not in {-1, 1}:
            raise ValueError("canonical_orientation must be either -1 or 1")
        if spec.declared_alias_of is not None and spec.declared_alias_of not in candidate_ids:
            raise ValueError(f"declared_alias_of is not a candidate_id: {spec.declared_alias_of}")
    by_id = {spec.candidate_id: spec for spec in specs}
    for spec in specs:
        if spec.declared_alias_of is None:
            continue
        target = by_id[spec.declared_alias_of]
        if target.return_horizon != spec.return_horizon:
            raise ValueError("an exact alias must use the same return_horizon as its canonical candidate")
        _root_alias(spec.candidate_id, by_id)
    return specs


_REQUIRED_OBSERVED_HORIZONS = ("4h", "8h", "12h", "1d")
_REQUIRED_HORIZON_DELTAS = {
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "8h": pd.Timedelta(hours=8),
    "12h": pd.Timedelta(hours=12),
    "1d": pd.Timedelta(days=1),
}
_REALITY_SCALE_COUNTS = {
    "registry_feature_count": 68,
    "candidate_pair_count": 159,
    "admitted_symbol_count": 20,
}

# These are the four decision windows frozen in the active Issue #34
# blueprint.  They are part of the production input identity, rather than a
# caller-defined convenience window.  Keep the values here in UTC so a
# production contract cannot silently replace them while recomputing its own
# digest.
_REALITY_SCALE_DECISION_WINDOWS = {
    "4h": (
        pd.Timestamp("2024-12-21T12:00:00Z"),
        pd.Timestamp("2026-04-30T08:00:00Z"),
    ),
    "8h": (
        pd.Timestamp("2024-12-21T16:00:00Z"),
        pd.Timestamp("2026-04-30T08:00:00Z"),
    ),
    "12h": (
        pd.Timestamp("2024-12-21T12:00:00Z"),
        pd.Timestamp("2026-04-30T00:00:00Z"),
    ),
    "1d": (
        pd.Timestamp("2024-12-22T00:00:00Z"),
        pd.Timestamp("2026-04-30T00:00:00Z"),
    ),
}


def _require_nonempty_identity(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty frozen identity")
    if value != value.strip():
        raise ValueError(f"{label} must not contain surrounding whitespace")
    return value.strip()


_REALITY_SCALE_COVERAGE_PREFIX = "reality_effect_scale_v1:"


def _is_reality_scale_coverage_identity(value: object) -> bool:
    """Return whether a coverage identity is the versioned production form."""
    return isinstance(value, str) and value.startswith(_REALITY_SCALE_COVERAGE_PREFIX)


def _validate_frozen_source_manifest_identity(
    input_case: str,
    input_data: ObservedEffectScaleInput,
    coverage_contract_identity: str,
) -> None:
    """Require production inputs to be bound to an external frozen manifest.

    The qlab entry cannot infer archive paths from in-memory DataFrames.  The
    outer input loader therefore supplies the independently hashed manifest;
    this gate prevents a caller from silently reusing one manifest identity
    for different B/C bytes.  A changed source manifest is a new input
    identity, not a mutation of the old one.
    """
    if not _is_reality_scale_coverage_identity(coverage_contract_identity):
        return
    source_manifest_sha256 = input_data.source_manifest_sha256
    if re.fullmatch(r"[0-9a-f]{64}", source_manifest_sha256 or "") is None:
        raise ValueError(
            f"frozen {input_case} source_manifest_sha256 must be a lowercase SHA-256 digest"
        )
    expected_identity = (
        f"reality_effect_scale_v1:{input_case}:{source_manifest_sha256}:"
        f"{input_data.cache_sha256}:{input_data.minute_klines_sha256}"
    )
    if input_data.input_identity != expected_identity:
        raise ValueError(
            f"frozen {input_case} input_identity must bind source manifest and content digests"
        )


def _expected_unfiltered_candidate_pairs(
    registry_frame: pd.DataFrame,
    horizon_deltas: Mapping[str, pd.Timedelta],
) -> set[tuple[str, str]]:
    expected: set[tuple[str, str]] = set()
    for row in registry_frame.itertuples(index=False):
        values = row._asdict()
        feature_name = str(values["feature_name"])
        timeframe = str(values["signal_timeframe"])
        if timeframe not in horizon_deltas:
            raise ValueError(f"horizon_deltas missing registry signal timeframe: {timeframe}")
        signal_delta = pd.Timedelta(horizon_deltas[timeframe])
        for horizon in _REQUIRED_OBSERVED_HORIZONS:
            if horizon not in horizon_deltas:
                raise ValueError(f"horizon_deltas missing required observed horizon: {horizon}")
            horizon_delta = pd.Timedelta(horizon_deltas[horizon])
            if signal_delta <= horizon_delta and horizon_delta % signal_delta == pd.Timedelta(0):
                expected.add((feature_name, horizon))
    return expected


def _validate_contract(contract: ObservedEffectScaleContract, registry_frame: pd.DataFrame) -> tuple[ObservedEffectCandidate, ...]:
    symbols = tuple(str(symbol) for symbol in contract.admitted_symbols)
    if not symbols or any(not symbol for symbol in symbols) or len(set(symbols)) != len(symbols):
        raise ValueError("contract admitted_symbols must be non-empty and unique")
    if int(contract.execution_delay_minutes) != 4:
        raise ValueError("observed-effect scale inventory requires execution_delay_minutes=4")
    if int(contract.minimum_support_rows) < 2:
        raise ValueError("minimum_support_rows must be at least two")
    if int(contract.min_common_panel_rows) < 1:
        raise ValueError("min_common_panel_rows must be positive")
    _require_nonempty_identity(contract.registry_identity, label="registry_identity")
    _require_nonempty_identity(contract.candidate_set_identity, label="candidate_set_identity")
    coverage_identity = _require_nonempty_identity(
        contract.coverage_contract_identity, label="coverage_contract_identity"
    )
    _require_nonempty_identity(
        contract.decision_window_identity, label="decision_window_identity"
    )
    for label in (
        "registry_content_sha256",
        "candidate_specs_sha256",
        "admitted_symbols_sha256",
        "decision_windows_sha256",
        "horizon_contract_sha256",
    ):
        value = getattr(contract, label)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    for horizon, expected_delta in _REQUIRED_HORIZON_DELTAS.items():
        if horizon not in contract.horizon_deltas:
            raise ValueError(f"horizon_deltas missing required horizon: {horizon}")
        if pd.Timedelta(contract.horizon_deltas[horizon]) != expected_delta:
            raise ValueError(
                f"horizon_deltas has non-canonical duration for {horizon}: "
                f"expected={expected_delta}"
            )
    is_reality_scale = _is_reality_scale_coverage_identity(coverage_identity)
    if is_reality_scale:
        coverage_horizon_sha256 = coverage_identity[len(_REALITY_SCALE_COVERAGE_PREFIX) :]
        if re.fullmatch(r"[0-9a-f]{64}", coverage_horizon_sha256) is None:
            raise ValueError(
                "reality_effect_scale_v1 coverage identity must bind a horizon-contract SHA"
            )
        declared_counts = {
            key: int(getattr(contract, key))
            for key in _REALITY_SCALE_COUNTS
        }
        if declared_counts != _REALITY_SCALE_COUNTS:
            raise ValueError(
                "reality_effect_scale_v1 requires the frozen 68/159/20 coverage contract"
            )
        if not contract.decision_window_identity.startswith("reality_effect_scale_v1:"):
            raise ValueError(
                "reality_effect_scale_v1 requires a versioned full decision-window identity"
            )
    elif not coverage_identity.startswith("test_fixture:"):
        raise ValueError("unknown observed-effect scale coverage contract identity")
    if contract.require_complete_cross_sections is not True:
        raise ValueError("observed-effect scale inventory requires complete cross-sections")
    if contract.decision_coverage_policy != _DECISION_COVERAGE_POLICY:
        raise ValueError(
            "observed-effect scale inventory requires the per-candidate complete-cross-section policy"
        )
    if is_reality_scale:
        raw_registry = registry_frame.reset_index(drop=True)
        canonical_registry = factor_registry.base_panel_registry("1h").reset_index(drop=True)
        try:
            # The validator canonicalises row order for panel use.  Identity
            # validation must happen before that operation so a permuted
            # production registry cannot receive the same identity.
            pd.testing.assert_frame_equal(
                raw_registry,
                canonical_registry,
                check_exact=True,
                check_dtype=True,
                check_like=False,
            )
        except AssertionError as error:
            raise ValueError(
                "reality_effect_scale_v1 registry content/order is not the formal qlab registry"
            ) from error
        registry_for_identity = raw_registry
    else:
        registry_for_identity = None
    # Validate the registry only after the production identity has been
    # checked.  The formal validator returns a sorted copy, so calling it
    # earlier would erase the order that belongs to the frozen identity.
    validated_registry = factor_registry.validate_factor_eligibility_registry(registry_frame)
    if registry_for_identity is None:
        registry_for_identity = validated_registry
    actual_registry_sha256 = _registry_content_sha256(registry_for_identity)
    actual_candidate_specs_sha256 = _candidate_specs_content_sha256(contract.candidate_specs)
    actual_symbols_sha256 = _symbols_content_sha256(symbols)
    actual_windows_sha256 = _decision_windows_content_sha256(contract.decision_windows)
    actual_horizon_contract_sha256 = _horizon_contract_content_sha256(
        contract.horizon_deltas,
        execution_delay_minutes=contract.execution_delay_minutes,
        require_complete_cross_sections=contract.require_complete_cross_sections,
        minimum_support_rows=contract.minimum_support_rows,
        min_common_panel_rows=contract.min_common_panel_rows,
        decision_coverage_policy=contract.decision_coverage_policy,
    )
    declared_digests = {
        "registry_content_sha256": contract.registry_content_sha256,
        "candidate_specs_sha256": contract.candidate_specs_sha256,
        "admitted_symbols_sha256": contract.admitted_symbols_sha256,
        "decision_windows_sha256": contract.decision_windows_sha256,
        "horizon_contract_sha256": contract.horizon_contract_sha256,
    }
    actual_digests = {
        "registry_content_sha256": actual_registry_sha256,
        "candidate_specs_sha256": actual_candidate_specs_sha256,
        "admitted_symbols_sha256": actual_symbols_sha256,
        "decision_windows_sha256": actual_windows_sha256,
        "horizon_contract_sha256": actual_horizon_contract_sha256,
    }
    if declared_digests != actual_digests:
        mismatched = [
            f"{key}: declared={declared_digests[key]} actual={actual_digests[key]}"
            for key in declared_digests
            if declared_digests[key] != actual_digests[key]
        ]
        raise ValueError("frozen observed-effect contract content changed: " + "; ".join(mismatched))
    if is_reality_scale:
        expected_coverage_identity = (
            f"{_REALITY_SCALE_COVERAGE_PREFIX}{actual_horizon_contract_sha256}"
        )
        if coverage_identity != expected_coverage_identity:
            raise ValueError(
                "reality_effect_scale_v1 coverage_contract_identity must bind the full horizon contract"
            )
        from .true_oos import CANONICAL_SYMBOLS

        if symbols != tuple(CANONICAL_SYMBOLS):
            raise ValueError(
                "reality_effect_scale_v1 admitted_symbols must equal qlab TRUE OOS canonical universe"
            )
        expected_window_identity = f"reality_effect_scale_v1:{actual_windows_sha256}"
        if contract.decision_window_identity != expected_window_identity:
            raise ValueError(
                "reality_effect_scale_v1 decision_window_identity must bind the window SHA"
            )
        expected_candidate_set_identity = (
            f"reality_effect_scale_v1:{actual_candidate_specs_sha256}"
        )
        if contract.candidate_set_identity != expected_candidate_set_identity:
            raise ValueError(
                "reality_effect_scale_v1 candidate_set_identity must bind candidate metadata SHA"
            )
        expected_windows = set(_REALITY_SCALE_DECISION_WINDOWS)
        if set(contract.decision_windows) != expected_windows:
            raise ValueError(
                "reality_effect_scale_v1 decision_windows must contain exactly the four frozen horizons"
            )
        for horizon, (expected_start, expected_end) in _REALITY_SCALE_DECISION_WINDOWS.items():
            actual_window = contract.decision_windows[horizon]
            actual_start = _utc_timestamp(
                actual_window.start, field=f"decision window {horizon} start"
            )
            actual_end = _utc_timestamp(
                actual_window.end, field=f"decision window {horizon} end"
            )
            if actual_start != expected_start or actual_end != expected_end:
                raise ValueError(
                    "reality_effect_scale_v1 decision_windows do not match the frozen blueprint windows"
                )
    actual_counts = {
        "registry_feature_count": int(len(validated_registry)),
        "candidate_pair_count": int(
            len(_expected_unfiltered_candidate_pairs(validated_registry, contract.horizon_deltas))
        ),
        "admitted_symbol_count": int(len(symbols)),
    }
    declared_counts = {
        key: int(getattr(contract, key)) for key in _REALITY_SCALE_COUNTS
    }
    if actual_counts != declared_counts:
        raise ValueError(
            "frozen observed-effect coverage counts do not match: "
            f"declared={declared_counts} actual={actual_counts}"
        )
    specs = _validate_candidate_specs(
        contract.candidate_specs, validated_registry, contract.horizon_deltas
    )
    expected_pairs = _expected_unfiltered_candidate_pairs(validated_registry, contract.horizon_deltas)
    actual_pairs = {(spec.feature_name, spec.return_horizon) for spec in specs}
    if actual_pairs != expected_pairs or len(specs) != len(expected_pairs):
        missing = sorted(expected_pairs.difference(actual_pairs))
        extra = sorted(actual_pairs.difference(expected_pairs))
        raise ValueError(
            "candidate_specs must cover the complete unfiltered registry grid; "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    if set(spec.return_horizon for spec in specs) != set(_REQUIRED_OBSERVED_HORIZONS):
        raise ValueError("candidate_specs must include all required observed horizons")
    return specs


def _validate_raw_input_identity(
    cache_payloads: Mapping[str, Mapping[str, pd.DataFrame]],
    registry_frame: pd.DataFrame,
) -> None:
    """Reject mutable/ambiguous cache identities before panel normalization can hide them."""
    for scope, payload in cache_payloads.items():
        if not isinstance(payload, Mapping):
            raise ValueError(f"cache scope is not a mapping: {scope}")
        for cache_key, frame in payload.items():
            if not isinstance(frame, pd.DataFrame):
                raise ValueError(f"cache entry is not a DataFrame: {scope}/{cache_key}")
            index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True, errors="raise"))
            if index.has_duplicates:
                raise ValueError(f"cache entry has duplicate source timestamps: {scope}/{cache_key}")
    for row in registry_frame.itertuples(index=False):
        values = row._asdict()
        scope = str(values["source_scope"])
        if scope not in cache_payloads:
            raise ValueError(f"cache scope missing for registry feature {values['feature_name']}: {scope}")


def _validate_minute_price_inputs(
    admitted_symbols: Sequence[str],
    minute_klines_by_symbol: Mapping[str, pd.DataFrame | pd.Series],
) -> None:
    if not isinstance(minute_klines_by_symbol, Mapping):
        raise ValueError("minute_klines_by_symbol must be a mapping")
    for symbol in admitted_symbols:
        if symbol not in minute_klines_by_symbol:
            raise ValueError(f"minute price input missing admitted symbol: {symbol}")
        frame = minute_klines_by_symbol[symbol]
        if isinstance(frame, pd.Series):
            values = pd.to_numeric(frame, errors="coerce")
        elif isinstance(frame, pd.DataFrame) and "open" in frame.columns:
            values = pd.to_numeric(frame["open"], errors="coerce")
        else:
            raise ValueError(f"minute price input has no open column: {symbol}")
        numeric_values = values.to_numpy(dtype=float, copy=False)
        if not len(numeric_values) or not np.isfinite(numeric_values).all():
            raise ValueError(f"minute price input contains non-finite open value: {symbol}")
        if (numeric_values <= 0.0).any():
            raise ValueError(f"minute price input contains non-positive open value: {symbol}")


def _validate_frozen_input_digests(
    input_case: str,
    input_data: ObservedEffectScaleInput,
) -> None:
    current_cache_sha256 = _mapping_content_sha256(input_data.cache_payloads, nested=True)
    current_price_sha256 = _mapping_content_sha256(
        input_data.minute_klines_by_symbol, nested=False
    )
    if current_cache_sha256 != input_data.cache_sha256:
        raise ValueError(f"frozen {input_case} cache input changed after construction")
    if current_price_sha256 != input_data.minute_klines_sha256:
        raise ValueError(f"frozen {input_case} minute price input changed after construction")


def _validate_required_raw_values(
    admitted_symbols: Sequence[str],
    cache_payloads: Mapping[str, Mapping[str, pd.DataFrame]],
    registry_frame: pd.DataFrame,
) -> None:
    """Reject infinities without changing historical NaN/missing-value semantics."""
    for row in registry_frame.itertuples(index=False):
        values = row._asdict()
        scope = str(values["source_scope"])
        endpoint = str(values["endpoint"])
        if scope not in cache_payloads:
            raise ValueError(f"cache scope missing for registry feature {values['feature_name']}: {scope}")
        columns = [token.strip() for token in str(values["required_columns"]).split(",") if token.strip()]
        for symbol in admitted_symbols:
            key = f"{symbol}_{endpoint}"
            if key not in cache_payloads[scope]:
                raise ValueError(f"cache entry missing for admitted symbol: {scope}/{key}")
            frame = cache_payloads[scope][key]
            for column in columns:
                if column not in frame.columns:
                    raise ValueError(
                        f"cache entry missing required column: {scope}/{key}/{column}"
                    )
                if pd.api.types.is_bool_dtype(frame[column]):
                    raise ValueError(
                        f"cache entry has boolean required value: {scope}/{key}/{column}"
                    )
                numeric = pd.to_numeric(frame[column], errors="coerce")
                raw_non_null = frame[column].notna().to_numpy()
                numeric_values = numeric.to_numpy(dtype=float, copy=False)
                if np.isnan(numeric_values[raw_non_null]).any() or np.isinf(numeric_values).any():
                    raise ValueError(
                        f"cache entry has invalid required value: {scope}/{key}/{column}"
                    )


def _complete_cross_section_decisions(
    route: pd.DataFrame,
    feature_name: str,
    admitted_symbols: Sequence[str],
) -> pd.DatetimeIndex:
    """Return only decisions with one finite row for every admitted symbol.

    The formal panel builder owns source extraction, timestamp alignment and
    cross-sectional standardisation.  This helper only applies the observed
    effect inventory's coverage rule *after* those formal operations: a
    missing/non-finite value for one symbol removes that decision for the
    candidate and horizon as a whole.  It never re-ranks a partial cross
    section and it does not alter the public L0--L4 panel semantics.
    """
    if not isinstance(route.index, pd.MultiIndex) or set(route.index.names) < {
        "decision_ts",
        "symbol",
    }:
        raise ValueError("candidate route must use a decision_ts/symbol MultiIndex")
    if feature_name not in route.columns:
        raise ValueError(f"candidate feature is absent from the formal route: {feature_name}")
    expected_symbols = {str(symbol) for symbol in admitted_symbols}
    if not expected_symbols:
        raise ValueError("admitted_symbols must not be empty")
    numeric = pd.to_numeric(route[feature_name], errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float, copy=False))
    finite_rows = route.loc[finite]
    if finite_rows.empty:
        return pd.DatetimeIndex([], tz="UTC", name="decision_ts")

    complete: list[pd.Timestamp] = []
    for decision_ts, group in finite_rows.groupby(level="decision_ts", sort=False):
        symbols = [str(value) for value in group.index.get_level_values("symbol")]
        if len(symbols) == len(expected_symbols) and set(symbols) == expected_symbols and len(set(symbols)) == len(symbols):
            complete.append(pd.Timestamp(decision_ts))
    if not complete:
        return pd.DatetimeIndex([], tz="UTC", name="decision_ts")
    return pd.DatetimeIndex(pd.to_datetime(complete, utc=True), name="decision_ts").sort_values().as_unit("ns")


def _validate_windows(
    decision_windows: Mapping[str, DecisionWindow],
    candidate_specs: Sequence[ObservedEffectCandidate],
    registry_frame: pd.DataFrame,
    horizon_deltas: Mapping[str, pd.Timedelta],
    *,
    execution_delay_minutes: int,
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    registry_by_feature = registry_frame.set_index("feature_name", drop=False)
    required_horizons = {spec.return_horizon for spec in candidate_specs}
    missing = required_horizons.difference(decision_windows)
    if missing:
        raise ValueError("decision_windows missing horizon(s): " + ", ".join(sorted(missing)))
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for horizon in sorted(required_horizons, key=lambda value: pd.Timedelta(horizon_deltas[value])):
        window = decision_windows[horizon]
        start = _utc_timestamp(window.start, field=f"decision window {horizon} start")
        end = _utc_timestamp(window.end, field=f"decision window {horizon} end")
        if end < start:
            raise ValueError(f"decision window end precedes start for {horizon}")
        horizon_specs = [spec for spec in candidate_specs if spec.return_horizon == horizon]
        timeframes = tuple(
            dict.fromkeys(
                str(registry_by_feature.loc[spec.feature_name, "signal_timeframe"])
                for spec in horizon_specs
            )
        )
        candidate_time_contract = ContinuousHoldingTimeContract(
            return_horizon=horizon,
            decision_interval=horizon,
            holding_interval=horizon,
            strategy_return_interval=horizon,
            signal_timeframes=timeframes,
            execution_delay_minutes=execution_delay_minutes,
            data_observed_rule="assumed_available_by_t_plus_4m",
        )
        validate_continuous_holding_contract(candidate_time_contract, horizon_deltas)
        validate_decision_phase([start, end], candidate_time_contract, horizon_deltas)
        windows[horizon] = (start, end)
    return windows


def _canonical_float64_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left_values = np.asarray(left, dtype="<f8")
    right_values = np.asarray(right, dtype="<f8")
    return left_values.shape == right_values.shape and left_values.tobytes() == right_values.tobytes()


def _signal_identity_equal(left: pd.Series, right: pd.Series) -> bool:
    if not left.index.equals(right.index):
        return False
    left_values = left.to_numpy(dtype=float, copy=False)
    right_values = right.to_numpy(dtype=float, copy=False)
    left_finite = np.isfinite(left_values)
    right_finite = np.isfinite(right_values)
    return np.array_equal(left_finite, right_finite) and _canonical_float64_equal(
        left_values[left_finite], right_values[right_finite]
    )


def _root_alias(candidate_id: str, by_id: Mapping[str, ObservedEffectCandidate]) -> str:
    seen: set[str] = set()
    current = candidate_id
    while by_id[current].declared_alias_of is not None:
        if current in seen:
            raise ValueError("declared alias mapping contains a cycle")
        seen.add(current)
        current = str(by_id[current].declared_alias_of)
    return current


def _exact_duplicate_mapping(
    candidate_specs: Sequence[ObservedEffectCandidate],
    signal_by_candidate: Mapping[str, pd.Series],
    *,
    input_case: str,
) -> pd.DataFrame:
    """Identify exact signal duplicates after only frozen sign canonicalization."""
    by_id = {spec.candidate_id: spec for spec in candidate_specs}
    rows: list[dict[str, object]] = []
    for horizon in sorted({spec.return_horizon for spec in candidate_specs}):
        members = [spec for spec in candidate_specs if spec.return_horizon == horizon]
        groups: list[list[ObservedEffectCandidate]] = []
        for spec in sorted(members, key=lambda value: value.candidate_id):
            signal = signal_by_candidate[spec.candidate_id] * int(spec.canonical_orientation)
            matching_group: list[ObservedEffectCandidate] | None = None
            for group in groups:
                first = group[0]
                first_signal = signal_by_candidate[first.candidate_id] * int(first.canonical_orientation)
                if _signal_identity_equal(signal, first_signal):
                    matching_group = group
                    break
            if matching_group is None:
                groups.append([spec])
            else:
                matching_group.append(spec)
        for ordinal, group in enumerate(groups, start=1):
            declared_roots = {_root_alias(spec.candidate_id, by_id) for spec in group}
            canonical = min(declared_roots) if len(declared_roots) == 1 else min(
                spec.candidate_id for spec in group
            )
            group_id = f"{horizon}:exact:{ordinal:03d}"
            for spec in group:
                signal = signal_by_candidate[spec.candidate_id]
                values = signal.to_numpy(dtype=float, copy=False)
                rows.append(
                    {
                        "input_case": input_case,
                        "candidate_id": spec.candidate_id,
                        "return_horizon": horizon,
                        "duplicate_group_id": group_id,
                        "canonical_candidate_id": canonical,
                        "declared_alias_of": spec.declared_alias_of,
                        "canonical_orientation": int(spec.canonical_orientation),
                        "is_exact_duplicate": spec.candidate_id != canonical,
                        "signal_support_rows": int(len(signal)),
                        "finite_signal_rows": int(np.isfinite(values).sum()),
                        "status": "exact_duplicate" if len(group) > 1 else "unique",
                    }
                )
    return pd.DataFrame(rows, columns=_DUPLICATE_COLUMNS).sort_values(
        ["return_horizon", "duplicate_group_id", "candidate_id"], kind="stable"
    ).reset_index(drop=True)


def _estimate_common_slope(
    candidate: ObservedEffectCandidate,
    signal: pd.Series,
    executable_rows: pd.DataFrame,
    *,
    minimum_support_rows: int,
    minimum_support_assets: int,
    horizon_sigma_y: float,
) -> dict[str, object]:
    values = executable_rows[[candidate.feature_name, "executable_return", "symbol"]].copy()
    values = values.rename(columns={candidate.feature_name: "signal_z", "executable_return": "return_y"})
    values["signal_z"] = pd.to_numeric(values["signal_z"], errors="coerce")
    values["return_y"] = pd.to_numeric(values["return_y"], errors="coerce")
    finite = np.isfinite(values["signal_z"].to_numpy(dtype=float)) & np.isfinite(
        values["return_y"].to_numpy(dtype=float)
    )
    support = values.loc[finite].copy()
    support_rows = int(len(support))
    base = {
        "input_case": "",
        "candidate_id": candidate.candidate_id,
        "feature_name": candidate.feature_name,
        "return_horizon": candidate.return_horizon,
        "declared_alias_of": candidate.declared_alias_of,
        "canonical_orientation": int(candidate.canonical_orientation),
        "support_rows": support_rows,
        "support_asset_count": int(support["symbol"].nunique()) if support_rows else 0,
        "support_decision_count": int(support.index.nunique()) if support_rows else 0,
        "support_start": support.index.min() if support_rows else pd.NaT,
        "support_end": support.index.max() if support_rows else pd.NaT,
        "signal_rows_on_window": int(len(signal)),
        "finite_signal_rows": int(np.isfinite(signal.to_numpy(dtype=float, copy=False)).sum()),
        "finite_return_rows": int(np.isfinite(values["return_y"].to_numpy(dtype=float)).sum()),
        "alpha_obs": float("nan"),
        "beta_obs": float("nan"),
        "sigma_y": float("nan"),
        "delta_obs": float("nan"),
    }
    if support_rows < minimum_support_rows:
        return {**base, "status": "insufficient_support", "failure_reason": "support_rows_below_minimum"}
    if base["support_asset_count"] < minimum_support_assets:
        return {**base, "status": "insufficient_support", "failure_reason": "support_assets_below_contract"}
    x = support["signal_z"].to_numpy(dtype=float)
    y = support["return_y"].to_numpy(dtype=float)
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= 0.0:
        return {**base, "status": "insufficient_support", "failure_reason": "constant_formal_signal"}
    beta = float(np.dot(x_centered, y - y.mean()) / denominator)
    alpha = float(y.mean() - beta * x.mean())
    sigma_y = float(horizon_sigma_y)
    if sigma_y <= 0.0:
        return {
            **base,
            "status": "insufficient_support",
            "failure_reason": "zero_return_volatility",
            "alpha_obs": alpha,
            "beta_obs": beta,
            "sigma_y": sigma_y,
        }
    return {
        **base,
        "status": "ok",
        "failure_reason": "",
        "alpha_obs": alpha,
        "beta_obs": beta,
        "sigma_y": sigma_y,
        "delta_obs": float(beta / sigma_y),
    }


def _distribution_summary(
    estimates: pd.DataFrame,
    duplicate_mapping: pd.DataFrame,
    *,
    input_case: str,
) -> pd.DataFrame:
    mapping = duplicate_mapping.set_index("candidate_id")
    rows: list[dict[str, object]] = []
    for horizon in sorted(estimates["return_horizon"].astype(str).unique()):
        horizon_estimates = estimates.loc[
            (estimates["return_horizon"] == horizon) & (estimates["status"] == "ok")
        ].copy()
        if not horizon_estimates.empty:
            horizon_estimates["canonical_candidate_id"] = horizon_estimates["candidate_id"].map(
                mapping["canonical_candidate_id"]
            )
            horizon_estimates = horizon_estimates.loc[
                horizon_estimates["candidate_id"] == horizon_estimates["canonical_candidate_id"]
            ]
        for distribution in ("signed", "absolute"):
            values = horizon_estimates["beta_obs"].to_numpy(dtype=float)
            if distribution == "absolute":
                values = np.abs(values)
            if len(values) == 0:
                rows.append(
                    {
                        "input_case": input_case,
                        "return_horizon": horizon,
                        "distribution": distribution,
                        "status": "no_eligible_nonduplicate_candidates",
                        "candidate_count": 0,
                        "positive_count": 0,
                        "negative_count": 0,
                        "zero_count": 0,
                        "skewness": float("nan"),
                        "p10": float("nan"),
                        "p50": float("nan"),
                        "p90": float("nan"),
                    }
                )
                continue
            skewness = float(pd.Series(values).skew()) if len(values) >= 3 else float("nan")
            rows.append(
                {
                    "input_case": input_case,
                    "return_horizon": horizon,
                    "distribution": distribution,
                    "status": "ok",
                    "candidate_count": int(len(values)),
                    "positive_count": int((values > 0.0).sum()),
                    "negative_count": int((values < 0.0).sum()),
                    "zero_count": int((values == 0.0).sum()),
                    "skewness": skewness,
                    "p10": float(np.quantile(values, 0.10)),
                    "p50": float(np.quantile(values, 0.50)),
                    "p90": float(np.quantile(values, 0.90)),
                }
            )
    return pd.DataFrame(rows, columns=_DISTRIBUTION_COLUMNS).sort_values(
        ["return_horizon", "distribution"], kind="stable"
    ).reset_index(drop=True)


def _estimate_one_frozen_input(
    *,
    input_case: str,
    input_data: ObservedEffectScaleInput,
    contract: ObservedEffectScaleContract,
    registry_frame: pd.DataFrame,
    specs: Sequence[ObservedEffectCandidate],
    windows: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> _SingleInputArtifacts:
    if not isinstance(registry_frame, pd.DataFrame) or registry_frame.empty:
        raise ValueError("registry_frame must be a non-empty DataFrame")
    if "feature_name" not in registry_frame.columns or "signal_timeframe" not in registry_frame.columns:
        raise ValueError("registry_frame must contain feature_name and signal_timeframe")
    _require_nonempty_identity(input_data.input_identity, label=f"input case {input_case} identity")
    _validate_frozen_source_manifest_identity(
        input_case, input_data, contract.coverage_contract_identity
    )
    _validate_frozen_input_digests(input_case, input_data)
    _validate_raw_input_identity(input_data.cache_payloads, registry_frame)
    _validate_required_raw_values(contract.admitted_symbols, input_data.cache_payloads, registry_frame)
    _validate_minute_price_inputs(contract.admitted_symbols, input_data.minute_klines_by_symbol)
    panel_window_start = min(start for start, _ in windows.values())
    panel_window_end = max(end for _, end in windows.values())
    panel_artifacts = build_panel_from_payloads(
        admitted_symbols=contract.admitted_symbols,
        cache_payloads={key: dict(value) for key, value in input_data.cache_payloads.items()},
        registry_frame=registry_frame,
        min_common_rows=int(contract.min_common_panel_rows),
        decision_index=build_decision_grid_index(panel_window_start, panel_window_end),
    )
    panel = panel_artifacts.panel
    registry_by_feature = registry_frame.set_index("feature_name", drop=False)
    signal_by_candidate: dict[str, pd.Series] = {}
    return_by_candidate: dict[str, pd.Series] = {}
    estimate_rows: list[dict[str, object]] = []

    route_by_horizon: dict[str, pd.DataFrame] = {}
    sigma_by_horizon: dict[str, float] = {}
    return_by_horizon: dict[str, pd.Series] = {}
    for horizon in sorted({spec.return_horizon for spec in specs}, key=lambda value: pd.Timedelta(contract.horizon_deltas[value])):
        start, end = windows[horizon]
        route = factor_research.filter_frame_to_decision_frequency(
            panel, horizon, contract.horizon_deltas
        )
        route_times = pd.DatetimeIndex(route.index.get_level_values("decision_ts"))
        route = route.loc[(route_times >= start) & (route_times <= end)].copy()
        if route.empty:
            raise ValueError(f"no formally aligned panel rows in frozen decision window for {horizon}")
        timeframes = tuple(
            dict.fromkeys(
                str(registry_by_feature.loc[spec.feature_name, "signal_timeframe"])
                for spec in specs if spec.return_horizon == horizon
            )
        )
        execution_contract = ContinuousHoldingTimeContract(
            return_horizon=horizon,
            decision_interval=horizon,
            holding_interval=horizon,
            strategy_return_interval=horizon,
            signal_timeframes=timeframes,
            execution_delay_minutes=4,
            data_observed_rule="assumed_available_by_t_plus_4m",
        )
        executable = panel_with_executable_return(
            route[["label_ts", "signal_bar_end_ts"]],
            input_data.minute_klines_by_symbol,
            execution_contract,
            contract.horizon_deltas,
        )
        return_index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex(executable.index), executable["symbol"].astype(str)],
            names=["decision_ts", "symbol"],
        )
        returns = pd.Series(
            executable["executable_return"].to_numpy(dtype=float), index=return_index, name="executable_return"
        ).sort_index()
        if not np.isfinite(returns.to_numpy(dtype=float)).all():
            raise ValueError(f"non-finite executable return in frozen {horizon} input")
        if contract.require_complete_cross_sections:
            expected_symbols = set(contract.admitted_symbols)
            counts = returns.groupby(level="decision_ts").size()
            actual_sets = returns.groupby(level="decision_ts").apply(
                lambda values: set(values.index.get_level_values("symbol"))
            )
            if (counts != len(expected_symbols)).any() or not all(value == expected_symbols for value in actual_sets):
                raise ValueError(f"incomplete executable return cross-section for {horizon}")
        route_by_horizon[horizon] = route
        return_by_horizon[horizon] = returns
        sigma_by_horizon[horizon] = float(np.std(returns.to_numpy(dtype=float), ddof=0))

    for spec in specs:
        route = route_by_horizon[spec.return_horizon]
        complete_decisions = _complete_cross_section_decisions(
            route,
            spec.feature_name,
            contract.admitted_symbols,
        )
        route_decisions = pd.DatetimeIndex(route.index.get_level_values("decision_ts"))
        candidate_route = route.loc[route_decisions.isin(complete_decisions)].copy()
        signal_by_candidate[spec.candidate_id] = candidate_route[spec.feature_name].sort_index()
        return_decisions = pd.DatetimeIndex(
            return_by_horizon[spec.return_horizon].index.get_level_values("decision_ts")
        )
        return_by_candidate[spec.candidate_id] = return_by_horizon[spec.return_horizon].loc[
            return_decisions.isin(complete_decisions)
        ]
        signal_values = signal_by_candidate[spec.candidate_id].to_numpy(dtype=float, copy=False)
        finite_signal = candidate_route
        if finite_signal.empty:
            estimate_rows.append(
                {
                    "input_case": input_case,
                    "candidate_id": spec.candidate_id,
                    "feature_name": spec.feature_name,
                    "return_horizon": spec.return_horizon,
                    "declared_alias_of": spec.declared_alias_of,
                    "canonical_orientation": int(spec.canonical_orientation),
                    "status": "insufficient_support",
                    "failure_reason": "no_complete_cross_section_decisions",
                    "support_rows": 0,
                    "support_asset_count": 0,
                    "support_decision_count": 0,
                    "support_start": pd.NaT,
                    "support_end": pd.NaT,
                    "signal_rows_on_window": int(len(candidate_route)),
                    "finite_signal_rows": 0,
                    "finite_return_rows": int(len(return_by_candidate[spec.candidate_id])),
                    "alpha_obs": float("nan"),
                    "beta_obs": float("nan"),
                    "sigma_y": float("nan"),
                    "delta_obs": float("nan"),
                }
            )
            continue
        timeframe = str(registry_by_feature.loc[spec.feature_name, "signal_timeframe"])
        candidate_execution_contract = ContinuousHoldingTimeContract(
            return_horizon=spec.return_horizon,
            decision_interval=spec.return_horizon,
            holding_interval=spec.return_horizon,
            strategy_return_interval=spec.return_horizon,
            signal_timeframes=(timeframe,),
            execution_delay_minutes=4,
            data_observed_rule="assumed_available_by_t_plus_4m",
        )
        labeled = panel_with_executable_return(
            finite_signal,
            input_data.minute_klines_by_symbol,
            candidate_execution_contract,
            contract.horizon_deltas,
        )
        row = _estimate_common_slope(
                spec,
                signal_by_candidate[spec.candidate_id],
                labeled,
                minimum_support_rows=int(contract.minimum_support_rows),
                minimum_support_assets=len(contract.admitted_symbols) if contract.require_complete_cross_sections else 1,
                horizon_sigma_y=sigma_by_horizon[spec.return_horizon],
            )
        row["input_case"] = input_case
        estimate_rows.append(row)

    estimates = pd.DataFrame(estimate_rows, columns=_ESTIMATE_COLUMNS).sort_values(
        ["input_case", "return_horizon", "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    duplicate_mapping = _exact_duplicate_mapping(specs, signal_by_candidate, input_case=input_case)
    duplicate_mapping["input_case"] = input_case
    for spec in specs:
        if spec.declared_alias_of is None:
            continue
        row = duplicate_mapping.set_index("candidate_id").loc[spec.candidate_id]
        target = duplicate_mapping.set_index("candidate_id").loc[spec.declared_alias_of]
        if row["duplicate_group_id"] != target["duplicate_group_id"]:
            raise ValueError(f"declared sign alias is not an exact canonical signal: {spec.candidate_id}")
    distribution_summary = _distribution_summary(estimates, duplicate_mapping, input_case=input_case)
    artifacts = ObservedEffectScaleArtifacts(
        candidate_estimates=estimates,
        duplicate_mapping=duplicate_mapping,
        distribution_summary=distribution_summary,
        input_case_comparison=pd.DataFrame(columns=_INPUT_COMPARISON_COLUMNS),
        input_identity_manifest=pd.DataFrame(columns=_INPUT_IDENTITY_COLUMNS),
    )
    return _SingleInputArtifacts(artifacts, signal_by_candidate, return_by_candidate)


def _frames_exact_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(left, right, check_exact=True, check_dtype=True)
    except AssertionError:
        return False
    return True


def estimate_l0_l4_observed_effect_scale_v1(
    *,
    registry_frame: pd.DataFrame,
    contract: ObservedEffectScaleContract,
    input_cases: Mapping[str, ObservedEffectScaleInput],
) -> ObservedEffectScaleArtifacts:
    """Estimate a frozen B/C inventory with formal L0--L4 signal and return paths.

    The public entry deliberately has no parameter for p-values, L2 gates, L3
    combinations, L4 selections, or a simulation effect mapping.  Both B and C
    must produce byte-exact formal inputs and result tables before this function
    returns an artifact usable as a v1 observed-effect scale inventory.
    """
    if set(input_cases) != {"B", "C"}:
        raise ValueError("observed-effect scale v1 requires exactly frozen input cases B and C")
    if not all(
        isinstance(input_cases[case], ObservedEffectScaleInput) for case in ("B", "C")
    ):
        raise TypeError("frozen input cases B and C must be ObservedEffectScaleInput instances")
    if input_cases["B"] is input_cases["C"]:
        raise ValueError("frozen B and C input cases must be distinct objects")
    if input_cases["B"].cache_payloads is input_cases["C"].cache_payloads:
        raise ValueError("frozen B and C cache payload mappings must be distinct")
    if input_cases["B"].minute_klines_by_symbol is input_cases["C"].minute_klines_by_symbol:
        raise ValueError("frozen B and C price mappings must be distinct")
    b_identity = _require_nonempty_identity(input_cases["B"].input_identity, label="B input identity")
    c_identity = _require_nonempty_identity(input_cases["C"].input_identity, label="C input identity")
    if b_identity == c_identity:
        raise ValueError("frozen B and C input identities must be distinct")
    if _is_reality_scale_coverage_identity(contract.coverage_contract_identity):
        b_manifest = _require_nonempty_identity(
            input_cases["B"].source_manifest_sha256,
            label="B source manifest identity",
        )
        c_manifest = _require_nonempty_identity(
            input_cases["C"].source_manifest_sha256,
            label="C source manifest identity",
        )
        if b_manifest == c_manifest:
            raise ValueError("frozen B and C source manifest identities must be distinct")
    specs = _validate_contract(contract, registry_frame)
    windows = _validate_windows(
        contract.decision_windows,
        specs,
        registry_frame,
        contract.horizon_deltas,
        execution_delay_minutes=contract.execution_delay_minutes,
    )
    cases = {
        input_case: _estimate_one_frozen_input(
            input_case=input_case,
            input_data=input_cases[input_case],
            contract=contract,
            registry_frame=registry_frame,
            specs=specs,
            windows=windows,
        )
        for input_case in ("B", "C")
    }
    comparison_rows: list[dict[str, object]] = []
    for spec in specs:
        signal_equal = _signal_identity_equal(
            cases["B"].signal_by_candidate[spec.candidate_id],
            cases["C"].signal_by_candidate[spec.candidate_id],
        )
        return_equal = _signal_identity_equal(
            cases["B"].return_by_candidate[spec.candidate_id],
            cases["C"].return_by_candidate[spec.candidate_id],
        )
        b_row = cases["B"].artifacts.candidate_estimates.set_index("candidate_id").loc[[spec.candidate_id]].drop(columns="input_case")
        c_row = cases["C"].artifacts.candidate_estimates.set_index("candidate_id").loc[[spec.candidate_id]].drop(columns="input_case")
        estimate_equal = _frames_exact_equal(b_row, c_row)
        comparison_rows.append(
            {
                "candidate_id": spec.candidate_id,
                "return_horizon": spec.return_horizon,
                "signal_equal": signal_equal,
                "return_equal": return_equal,
                "estimate_equal": estimate_equal,
                "status": "equal" if signal_equal and return_equal and estimate_equal else "mismatch",
            }
        )
    comparison = pd.DataFrame(comparison_rows, columns=_INPUT_COMPARISON_COLUMNS)
    if (comparison["status"] != "equal").any():
        raise ValueError("frozen B/C inputs do not produce identical formal observed-effect inventory")
    if not _frames_exact_equal(
        cases["B"].artifacts.duplicate_mapping.drop(columns="input_case"),
        cases["C"].artifacts.duplicate_mapping.drop(columns="input_case"),
    ) or not _frames_exact_equal(
        cases["B"].artifacts.distribution_summary.drop(columns="input_case"),
        cases["C"].artifacts.distribution_summary.drop(columns="input_case"),
    ):
        raise ValueError("frozen B/C inputs disagree on duplicate mapping or distribution summary")
    return ObservedEffectScaleArtifacts(
        candidate_estimates=pd.concat(
            [cases["B"].artifacts.candidate_estimates, cases["C"].artifacts.candidate_estimates],
            ignore_index=True,
        ),
        duplicate_mapping=pd.concat(
            [cases["B"].artifacts.duplicate_mapping, cases["C"].artifacts.duplicate_mapping],
            ignore_index=True,
        ),
        distribution_summary=pd.concat(
            [cases["B"].artifacts.distribution_summary, cases["C"].artifacts.distribution_summary],
            ignore_index=True,
        ),
        input_case_comparison=comparison,
        input_identity_manifest=pd.DataFrame(
            [
                {
                    "input_case": input_case,
                    "input_identity": input_cases[input_case].input_identity,
                    "source_manifest_sha256": input_cases[input_case].source_manifest_sha256,
                    "cache_sha256": input_cases[input_case].cache_sha256,
                    "minute_klines_sha256": input_cases[input_case].minute_klines_sha256,
                    "coverage_contract_identity": contract.coverage_contract_identity,
                    "registry_identity": contract.registry_identity,
                    "registry_content_sha256": contract.registry_content_sha256,
                    "candidate_set_identity": contract.candidate_set_identity,
                    "candidate_specs_sha256": contract.candidate_specs_sha256,
                    "admitted_symbol_count": contract.admitted_symbol_count,
                    "admitted_symbols_sha256": contract.admitted_symbols_sha256,
                    "decision_window_identity": contract.decision_window_identity,
                    "decision_windows_sha256": contract.decision_windows_sha256,
                    "horizon_contract_sha256": contract.horizon_contract_sha256,
                    "execution_delay_minutes": contract.execution_delay_minutes,
                    "decision_coverage_policy": contract.decision_coverage_policy,
                }
                for input_case in ("B", "C")
            ],
            columns=_INPUT_IDENTITY_COLUMNS,
        ),
    )


_BETA_TOTAL_HORIZON_ORDER = {horizon: ordinal for ordinal, horizon in enumerate(_REQUIRED_OBSERVED_HORIZONS)}
_BETA_TOTAL_ESTIMATE_COLUMNS = (
    "input_case",
    "candidate_id",
    "feature_name",
    "return_horizon",
    "status",
    "failure_reason",
    "beta_obs",
)
_BETA_TOTAL_DUPLICATE_COLUMNS = (
    "input_case",
    "candidate_id",
    "return_horizon",
    "duplicate_group_id",
    "canonical_candidate_id",
    "declared_alias_of",
    "canonical_orientation",
    "is_exact_duplicate",
)
_BETA_TOTAL_SIGNAL_COLUMNS = (
    "input_case",
    "canonical_signal_id",
    "signal_feature_names",
    "candidate_ids",
    "legal_horizons",
    "available_horizons",
    "excluded_horizons",
    "excluded_horizon_reasons",
    "excluded_candidate_ids",
    "exclusion_reasons",
    "selected_horizon",
    "selected_candidate_id",
    "selected_signed_beta_obs",
    "selected_canonical_signed_beta_obs",
    "selected_abs_beta_total_scale",
    "tie_candidate_ids",
    "status",
)
_BETA_TOTAL_CANONICAL_COLUMNS = (
    "input_case",
    "candidate_id",
    "feature_name",
    "return_horizon",
    "duplicate_group_id",
    "canonical_candidate_id",
    "declared_alias_of",
    "canonical_orientation",
    "is_exact_duplicate",
    "canonical_signal_id",
    "is_exact_duplicate_signal",
)
_BETA_TOTAL_DISTRIBUTION_COLUMNS = (
    "input_case",
    "distribution",
    "status",
    "signal_count",
    "positive_count",
    "negative_count",
    "zero_count",
    "p10",
    "p50",
    "p90",
    "quantile_method",
)
_BETA_TOTAL_COMPARISON_COLUMNS = (
    "canonical_signal_id",
    "signal_equal",
    "scale_equal",
    "status",
)
_BETA_TOTAL_BC_ESTIMATE_COMPARE_COLUMNS = (
    "candidate_id",
    "feature_name",
    "return_horizon",
    "status",
    "failure_reason",
    "beta_obs",
)
_BETA_TOTAL_BC_DUPLICATE_COMPARE_COLUMNS = (
    "candidate_id",
    "return_horizon",
    "duplicate_group_id",
    "canonical_candidate_id",
    "declared_alias_of",
    "canonical_orientation",
    "is_exact_duplicate",
    "status",
)
_BETA_TOTAL_UPSTREAM_COMPARISON_COLUMNS = (
    "candidate_id",
    "return_horizon",
    "signal_equal",
    "return_equal",
    "estimate_equal",
    "status",
)


def _beta_total_optional_text(value: object) -> str | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    return str(value)


def _beta_total_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _beta_total_bool(value: object, *, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    raise ValueError(f"{label} must be a boolean")


def _beta_total_case_tables(
    *,
    input_case: str,
    estimates: pd.DataFrame,
    duplicate_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(estimates, pd.DataFrame) or not isinstance(duplicate_mapping, pd.DataFrame):
        raise TypeError("beta-total mapping inputs must be pandas DataFrames")
    missing_estimates = sorted(set(_BETA_TOTAL_ESTIMATE_COLUMNS).difference(estimates.columns))
    missing_duplicates = sorted(set(_BETA_TOTAL_DUPLICATE_COLUMNS).difference(duplicate_mapping.columns))
    if missing_estimates:
        raise ValueError(f"candidate estimates are missing required columns: {missing_estimates}")
    if missing_duplicates:
        raise ValueError(f"duplicate mapping is missing required columns: {missing_duplicates}")
    case_estimates = estimates.loc[estimates["input_case"].astype(str) == input_case].copy()
    case_duplicates = duplicate_mapping.loc[
        duplicate_mapping["input_case"].astype(str) == input_case
    ].copy()
    if case_estimates.empty or case_duplicates.empty:
        raise ValueError(f"beta-total mapping has no complete input case {input_case}")
    case_estimates["candidate_id"] = case_estimates["candidate_id"].astype(str)
    case_estimates["feature_name"] = case_estimates["feature_name"].astype(str)
    case_estimates["return_horizon"] = case_estimates["return_horizon"].astype(str)
    case_duplicates["candidate_id"] = case_duplicates["candidate_id"].astype(str)
    case_duplicates["return_horizon"] = case_duplicates["return_horizon"].astype(str)
    case_duplicates["duplicate_group_id"] = case_duplicates["duplicate_group_id"].astype(str)
    case_duplicates["canonical_candidate_id"] = case_duplicates["canonical_candidate_id"].astype(str)
    if case_estimates["candidate_id"].duplicated().any():
        raise ValueError(f"input case {input_case} has duplicate candidate estimate identities")
    if case_duplicates["candidate_id"].duplicated().any():
        raise ValueError(f"input case {input_case} has duplicate duplicate-mapping identities")
    estimate_ids = set(case_estimates["candidate_id"])
    duplicate_ids = set(case_duplicates["candidate_id"])
    if estimate_ids != duplicate_ids:
        raise ValueError(
            f"input case {input_case} candidate/duplicate identities differ: "
            f"estimates_only={sorted(estimate_ids - duplicate_ids)[:5]} "
            f"duplicates_only={sorted(duplicate_ids - estimate_ids)[:5]}"
        )
    estimate_by_id = case_estimates.set_index("candidate_id", drop=False)
    duplicate_by_id = case_duplicates.set_index("candidate_id", drop=False)
    for candidate_id in sorted(estimate_ids):
        estimate = estimate_by_id.loc[candidate_id]
        duplicate = duplicate_by_id.loc[candidate_id]
        if str(estimate["return_horizon"]) != str(duplicate["return_horizon"]):
            raise ValueError(f"candidate/duplicate horizon differs for {input_case}/{candidate_id}")
        if _beta_total_optional_text(duplicate["canonical_candidate_id"]) is None:
            raise ValueError(f"canonical candidate is missing for {input_case}/{candidate_id}")
        if int(duplicate["canonical_orientation"]) not in {-1, 1}:
            raise ValueError(f"canonical orientation is not +/-1 for {input_case}/{candidate_id}")
        if not _beta_total_optional_text(duplicate["duplicate_group_id"]):
            raise ValueError(f"duplicate group is missing for {input_case}/{candidate_id}")
    return case_estimates, case_duplicates


def _beta_total_case_result(
    *,
    input_case: str,
    estimates: pd.DataFrame,
    duplicate_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    estimates, duplicate_mapping = _beta_total_case_tables(
        input_case=input_case,
        estimates=estimates,
        duplicate_mapping=duplicate_mapping,
    )
    estimate_by_id = estimates.set_index("candidate_id", drop=False)
    duplicate_by_id = duplicate_mapping.set_index("candidate_id", drop=False)
    feature_by_candidate = {
        candidate_id: str(row["feature_name"])
        for candidate_id, row in estimate_by_id.iterrows()
    }
    horizon_by_candidate = {
        candidate_id: str(row["return_horizon"])
        for candidate_id, row in estimate_by_id.iterrows()
    }
    if any(horizon not in _BETA_TOTAL_HORIZON_ORDER for horizon in horizon_by_candidate.values()):
        raise ValueError(f"input case {input_case} contains an unknown observed horizon")
    features = set(feature_by_candidate.values())
    parent = {feature: feature for feature in features}

    def find(feature: str) -> str:
        current = feature
        while parent[current] != current:
            parent[current] = parent[parent[current]]
            current = parent[current]
        return current

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    grouped = list(duplicate_mapping.groupby(["return_horizon", "duplicate_group_id"], sort=True))
    nontrivial_groups: list[tuple[str, str]] = []
    group_canonical_by_pair: dict[tuple[str, str], str] = {}
    for (horizon, group_id), group in grouped:
        members = [str(value) for value in group["candidate_id"]]
        canonical_ids = {str(value) for value in group["canonical_candidate_id"]}
        if len(canonical_ids) != 1:
            raise ValueError(f"duplicate group has multiple canonical candidates: {input_case}/{group_id}")
        canonical_id = next(iter(canonical_ids))
        if canonical_id not in members:
            raise ValueError(f"duplicate group canonical candidate is not a member: {input_case}/{group_id}")
        member_features = {feature_by_candidate[member] for member in members}
        if len(member_features) > 1:
            nontrivial_groups.append((str(horizon), str(group_id)))
            canonical_feature = feature_by_candidate[canonical_id]
            for feature in member_features:
                union(canonical_feature, feature)
        group_canonical_by_pair[(str(horizon), str(group_id))] = canonical_id
        for member in members:
            expected_duplicate = member != canonical_id
            actual_duplicate = _beta_total_bool(
                duplicate_by_id.loc[member, "is_exact_duplicate"],
                label=f"is_exact_duplicate {input_case}/{member}",
            )
            if actual_duplicate != expected_duplicate:
                raise ValueError(f"duplicate flag disagrees with canonical identity: {input_case}/{member}")
            alias = _beta_total_optional_text(duplicate_by_id.loc[member, "declared_alias_of"])
            if alias is not None:
                if alias not in feature_by_candidate:
                    raise ValueError(f"declared alias is not in the same input case: {input_case}/{member}")
                if horizon_by_candidate[alias] != horizon_by_candidate[member]:
                    raise ValueError(f"declared alias horizon differs: {input_case}/{member}")
                union(feature_by_candidate[member], feature_by_candidate[alias])
                if (str(horizon), str(group_id)) not in nontrivial_groups:
                    nontrivial_groups.append((str(horizon), str(group_id)))

    component_features: dict[str, set[str]] = {}
    for feature in sorted(features):
        component_features.setdefault(find(feature), set()).add(feature)
    component_authorities: dict[str, set[str]] = {root: set() for root in component_features}
    for horizon, group_id in nontrivial_groups:
        group = duplicate_mapping.loc[
            (duplicate_mapping["return_horizon"].astype(str) == horizon)
            & (duplicate_mapping["duplicate_group_id"].astype(str) == group_id)
        ]
        canonical_id = group_canonical_by_pair[(horizon, group_id)]
        component_authorities[find(feature_by_candidate[canonical_id])].add(
            feature_by_candidate[canonical_id]
        )
    canonical_signal_by_feature: dict[str, str] = {}
    for root, component in component_features.items():
        authorities = component_authorities[root]
        if not authorities:
            if len(component) != 1:
                raise ValueError(
                    f"canonical signal grouping is not determined for {input_case}: {sorted(component)}"
                )
            canonical_signal = next(iter(component))
        elif len(authorities) != 1:
            raise ValueError(
                f"ambiguous canonical signal grouping for {input_case}: {sorted(authorities)}"
            )
        else:
            canonical_signal = next(iter(authorities))
        for feature in component:
            canonical_signal_by_feature[feature] = canonical_signal

    # Exact duplicate groups must agree after the frozen sign orientation.  A
    # disagreement is an input-contract failure, not a reason to choose one
    # member or average the estimates.
    for (horizon, group_id), group in grouped:
        if len(group) <= 1:
            continue
        normalized_betas: list[float] = []
        statuses: list[str] = []
        for member in group["candidate_id"].astype(str):
            estimate = estimate_by_id.loc[member]
            status = str(estimate["status"])
            statuses.append(status)
            beta = pd.to_numeric(pd.Series([estimate["beta_obs"]]), errors="coerce").iloc[0]
            if status == "ok" and not np.isfinite(float(beta)):
                raise ValueError(f"exact duplicate group has non-finite beta: {input_case}/{group_id}")
            if status == "ok":
                normalized_betas.append(
                    float(beta) * int(duplicate_by_id.loc[member, "canonical_orientation"])
                )
        if len(set(statuses)) != 1:
            raise ValueError(f"exact duplicate group has inconsistent status: {input_case}/{group_id}")
        if normalized_betas:
            reference = np.asarray(normalized_betas[0], dtype="<f8").tobytes()
            if any(np.asarray(value, dtype="<f8").tobytes() != reference for value in normalized_betas[1:]):
                raise ValueError(f"exact duplicate group has inconsistent beta: {input_case}/{group_id}")

    canonical_rows_by_signal_horizon: dict[tuple[str, str], list[str]] = {}
    for (horizon, group_id), canonical_id in group_canonical_by_pair.items():
        canonical_signal = canonical_signal_by_feature[feature_by_candidate[canonical_id]]
        canonical_rows_by_signal_horizon.setdefault((canonical_signal, horizon), []).append(canonical_id)
    if any(len(set(values)) != 1 for values in canonical_rows_by_signal_horizon.values()):
        raise ValueError(
            f"canonical signal has multiple exact representatives at one horizon for {input_case}"
        )

    canonical_mapping_rows: list[dict[str, object]] = []
    for candidate_id in sorted(feature_by_candidate):
        duplicate = duplicate_by_id.loc[candidate_id]
        feature = feature_by_candidate[candidate_id]
        canonical_signal = canonical_signal_by_feature[feature]
        canonical_mapping_rows.append(
            {
                "input_case": input_case,
                "candidate_id": candidate_id,
                "feature_name": feature,
                "return_horizon": horizon_by_candidate[candidate_id],
                "duplicate_group_id": str(duplicate["duplicate_group_id"]),
                "canonical_candidate_id": str(duplicate["canonical_candidate_id"]),
                "declared_alias_of": _beta_total_optional_text(duplicate["declared_alias_of"]),
                "canonical_orientation": int(duplicate["canonical_orientation"]),
                "is_exact_duplicate": _beta_total_bool(
                    duplicate["is_exact_duplicate"],
                    label=f"is_exact_duplicate {input_case}/{candidate_id}",
                ),
                "canonical_signal_id": canonical_signal,
                "is_exact_duplicate_signal": feature != canonical_signal,
            }
        )
    canonical_mapping = pd.DataFrame(canonical_mapping_rows, columns=_BETA_TOTAL_CANONICAL_COLUMNS)

    component_candidates: dict[str, list[str]] = {}
    for candidate_id, feature in feature_by_candidate.items():
        component_candidates.setdefault(canonical_signal_by_feature[feature], []).append(candidate_id)
    signal_rows: list[dict[str, object]] = []
    for canonical_signal in sorted(component_candidates):
        candidate_ids = sorted(
            component_candidates[canonical_signal],
            key=lambda candidate_id: (
                _BETA_TOTAL_HORIZON_ORDER[horizon_by_candidate[candidate_id]],
                candidate_id,
            ),
        )
        legal_horizons = sorted(
            {horizon_by_candidate[candidate_id] for candidate_id in candidate_ids},
            key=_BETA_TOTAL_HORIZON_ORDER.get,
        )
        canonical_candidate_ids = {
            horizon: next(iter(set(values)))
            for (signal, horizon), values in canonical_rows_by_signal_horizon.items()
            if signal == canonical_signal
        }
        valid_canonical_rows: list[tuple[str, str, float, int]] = []
        exclusion_reasons: dict[str, str] = {}
        for candidate_id in candidate_ids:
            estimate = estimate_by_id.loc[candidate_id]
            status = str(estimate["status"])
            beta_value = pd.to_numeric(pd.Series([estimate["beta_obs"]]), errors="coerce").iloc[0]
            if status != "ok":
                failure_reason = _beta_total_optional_text(estimate["failure_reason"]) or "unspecified"
                exclusion_reasons[candidate_id] = f"status:{status};reason:{failure_reason}"
            elif not np.isfinite(float(beta_value)):
                exclusion_reasons[candidate_id] = "non_finite_beta_obs"
        for horizon, canonical_id in sorted(
            canonical_candidate_ids.items(), key=lambda item: _BETA_TOTAL_HORIZON_ORDER[item[0]]
        ):
            estimate = estimate_by_id.loc[canonical_id]
            status = str(estimate["status"])
            beta_value = pd.to_numeric(pd.Series([estimate["beta_obs"]]), errors="coerce").iloc[0]
            if status == "ok" and np.isfinite(float(beta_value)):
                valid_canonical_rows.append(
                    (
                        horizon,
                        canonical_id,
                        float(beta_value),
                        int(duplicate_by_id.loc[canonical_id, "canonical_orientation"]),
                    )
                )
        available_horizons = [row[0] for row in valid_canonical_rows]
        excluded_horizons = [horizon for horizon in legal_horizons if horizon not in available_horizons]
        excluded_horizon_reasons = {
            horizon: sorted(
                {
                    exclusion_reasons[candidate_id]
                    for candidate_id in candidate_ids
                    if horizon_by_candidate[candidate_id] == horizon and candidate_id in exclusion_reasons
                }
            )
            for horizon in excluded_horizons
        }
        if valid_canonical_rows:
            maximum = max(abs(row[2]) for row in valid_canonical_rows)
            tied_candidate_ids = sorted(
                [
                    candidate_id
                    for _, candidate_id, beta_value, _ in valid_canonical_rows
                    if abs(beta_value) == maximum
                ],
                key=lambda candidate_id: (
                    _BETA_TOTAL_HORIZON_ORDER[horizon_by_candidate[candidate_id]],
                    candidate_id,
                ),
            )
            selected = min(
                [row for row in valid_canonical_rows if abs(row[2]) == maximum],
                key=lambda row: (_BETA_TOTAL_HORIZON_ORDER[row[0]], row[1]),
            )
            signal_status = "ok"
            selected_horizon, selected_candidate_id, selected_beta, selected_orientation = selected
            selected_canonical_beta = selected_beta * selected_orientation
            selected_abs = abs(selected_beta)
        else:
            tied_candidate_ids = []
            signal_status = "no_valid_horizon"
            selected_horizon = ""
            selected_candidate_id = ""
            selected_beta = float("nan")
            selected_canonical_beta = float("nan")
            selected_abs = float("nan")
        signal_rows.append(
            {
                "input_case": input_case,
                "canonical_signal_id": canonical_signal,
                "signal_feature_names": _beta_total_json(
                    sorted({feature_by_candidate[candidate_id] for candidate_id in candidate_ids})
                ),
                "candidate_ids": _beta_total_json(candidate_ids),
                "legal_horizons": _beta_total_json(legal_horizons),
                "available_horizons": _beta_total_json(available_horizons),
                "excluded_horizons": _beta_total_json(excluded_horizons),
                "excluded_horizon_reasons": _beta_total_json(excluded_horizon_reasons),
                "excluded_candidate_ids": _beta_total_json(sorted(exclusion_reasons)),
                "exclusion_reasons": _beta_total_json(exclusion_reasons),
                "selected_horizon": selected_horizon,
                "selected_candidate_id": selected_candidate_id,
                "selected_signed_beta_obs": selected_beta,
                "selected_canonical_signed_beta_obs": selected_canonical_beta,
                "selected_abs_beta_total_scale": selected_abs,
                "tie_candidate_ids": _beta_total_json(tied_candidate_ids),
                "status": signal_status,
            }
        )
    signal_scales = pd.DataFrame(signal_rows, columns=_BETA_TOTAL_SIGNAL_COLUMNS)
    valid_scales = signal_scales.loc[signal_scales["status"] == "ok"]
    distribution_rows: list[dict[str, object]] = []
    for distribution in ("signed", "absolute"):
        values = valid_scales["selected_signed_beta_obs"].to_numpy(dtype=float)
        if distribution == "absolute":
            values = np.abs(values)
        if len(values) == 0:
            distribution_rows.append(
                {
                    "input_case": input_case,
                    "distribution": distribution,
                    "status": "no_valid_signal_scales",
                    "signal_count": 0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "zero_count": 0,
                    "p10": float("nan"),
                    "p50": float("nan"),
                    "p90": float("nan"),
                    "quantile_method": "linear",
                }
            )
            continue
        distribution_rows.append(
            {
                "input_case": input_case,
                "distribution": distribution,
                "status": "ok",
                "signal_count": int(len(values)),
                "positive_count": int((values > 0.0).sum()),
                "negative_count": int((values < 0.0).sum()),
                "zero_count": int((values == 0.0).sum()),
                "p10": float(np.quantile(values, 0.10, method="linear")),
                "p50": float(np.quantile(values, 0.50, method="linear")),
                "p90": float(np.quantile(values, 0.90, method="linear")),
                "quantile_method": "linear",
            }
        )
    distribution_summary = pd.DataFrame(
        distribution_rows, columns=_BETA_TOTAL_DISTRIBUTION_COLUMNS
    )
    return signal_scales, canonical_mapping, distribution_summary


def _beta_total_validate_frozen_b_c_inputs(
    artifacts: ObservedEffectScaleArtifacts,
) -> None:
    """Fail closed unless the complete upstream B/C inventory is identical."""

    def case_frame(frame: pd.DataFrame, case: str, columns: tuple[str, ...]) -> pd.DataFrame:
        missing = sorted(set(columns).difference(frame.columns))
        if missing:
            raise ValueError(f"B/C mapping gate is missing upstream columns: {missing}")
        selected = frame.loc[frame["input_case"].astype(str) == case, list(columns)].copy()
        if selected.empty:
            raise ValueError(f"B/C mapping gate has no upstream rows for case {case}")
        return selected.sort_values(list(columns[:3]), kind="stable").reset_index(drop=True)

    b_estimates = case_frame(
        artifacts.candidate_estimates,
        "B",
        _BETA_TOTAL_BC_ESTIMATE_COMPARE_COLUMNS,
    )
    c_estimates = case_frame(
        artifacts.candidate_estimates,
        "C",
        _BETA_TOTAL_BC_ESTIMATE_COMPARE_COLUMNS,
    )
    if not _frames_exact_equal(b_estimates, c_estimates):
        raise ValueError(
            "frozen B/C observed beta-total mapping differs in an upstream candidate/horizon row"
        )

    b_duplicates = case_frame(
        artifacts.duplicate_mapping,
        "B",
        _BETA_TOTAL_BC_DUPLICATE_COMPARE_COLUMNS,
    )
    c_duplicates = case_frame(
        artifacts.duplicate_mapping,
        "C",
        _BETA_TOTAL_BC_DUPLICATE_COMPARE_COLUMNS,
    )
    if not _frames_exact_equal(b_duplicates, c_duplicates):
        raise ValueError("frozen B/C observed beta-total mapping differs in duplicate identity")

    comparison = artifacts.input_case_comparison
    missing_comparison = sorted(
        set(_BETA_TOTAL_UPSTREAM_COMPARISON_COLUMNS).difference(comparison.columns)
    )
    if missing_comparison or comparison.empty:
        raise ValueError(
            "B/C mapping gate requires the complete upstream input_case_comparison: "
            f"missing={missing_comparison}"
        )
    try:
        comparison_flags = {
            column: [
                _beta_total_bool(value, label=f"upstream {column}")
                for value in comparison[column]
            ]
            for column in ("signal_equal", "return_equal", "estimate_equal")
        }
    except ValueError as exc:
        raise ValueError("upstream B/C input_case_comparison contains non-boolean flags") from exc
    if not (
        all(comparison_flags["signal_equal"])
        and all(comparison_flags["return_equal"])
        and all(comparison_flags["estimate_equal"])
        and comparison["status"].astype(str).eq("equal").all()
    ):
        raise ValueError("upstream B/C input_case_comparison is not fully equal")


def map_observed_effect_scale_to_beta_total_v1(
    artifacts: ObservedEffectScaleArtifacts,
) -> ObservedEffectBetaTotalArtifacts:
    """Map per-(candidate, horizon) observations to signal-level beta scales.

    The input must be the output of the formal observed-effect inventory for
    the independently materialised ``B`` and ``C`` cases.  Exact duplicate
    relationships are converted into signal-level connected components.  A
    component with conflicting canonical roots fails closed rather than
    selecting a convenient identity.  Within each component, exactly one
    canonical representative per legal horizon is considered, the maximum
    absolute ``beta_obs`` is selected, and ties use the fixed order
    ``4h < 8h < 12h < 1d`` followed by ``candidate_id``.  The returned
    quantiles are observed-scale evidence only; this function does not create
    the later user-frozen ``G_beta_total_v1`` simulation grid.
    """
    if not isinstance(artifacts, ObservedEffectScaleArtifacts):
        raise TypeError("artifacts must be ObservedEffectScaleArtifacts")
    if set(artifacts.candidate_estimates["input_case"].astype(str)) != {"B", "C"}:
        raise ValueError("beta-total mapping requires exactly input cases B and C")
    _beta_total_validate_frozen_b_c_inputs(artifacts)
    case_results = {
        input_case: _beta_total_case_result(
            input_case=input_case,
            estimates=artifacts.candidate_estimates,
            duplicate_mapping=artifacts.duplicate_mapping,
        )
        for input_case in ("B", "C")
    }
    comparison_rows: list[dict[str, object]] = []
    mismatches: list[str] = []
    b_scales, b_mapping, b_distribution = case_results["B"]
    c_scales, c_mapping, c_distribution = case_results["C"]
    signal_ids = sorted(
        set(b_scales["canonical_signal_id"]) | set(c_scales["canonical_signal_id"])
    )
    for signal_id in signal_ids:
        b_scale = b_scales.loc[b_scales["canonical_signal_id"] == signal_id].drop(columns="input_case")
        c_scale = c_scales.loc[c_scales["canonical_signal_id"] == signal_id].drop(columns="input_case")
        b_signal = b_mapping.loc[b_mapping["canonical_signal_id"] == signal_id].drop(columns="input_case")
        c_signal = c_mapping.loc[c_mapping["canonical_signal_id"] == signal_id].drop(columns="input_case")
        signal_equal = _frames_exact_equal(
            b_signal.sort_values(list(_BETA_TOTAL_CANONICAL_COLUMNS[1:]), kind="stable").reset_index(drop=True),
            c_signal.sort_values(list(_BETA_TOTAL_CANONICAL_COLUMNS[1:]), kind="stable").reset_index(drop=True),
        )
        scale_equal = _frames_exact_equal(
            b_scale.sort_values(["canonical_signal_id"], kind="stable").reset_index(drop=True),
            c_scale.sort_values(["canonical_signal_id"], kind="stable").reset_index(drop=True),
        )
        equal = signal_equal and scale_equal
        comparison_rows.append(
            {
                "canonical_signal_id": signal_id,
                "signal_equal": signal_equal,
                "scale_equal": scale_equal,
                "status": "equal" if equal else "mismatch",
            }
        )
        if not equal:
            mismatches.append(signal_id)
    b_dist_core = b_distribution.drop(columns="input_case").reset_index(drop=True)
    c_dist_core = c_distribution.drop(columns="input_case").reset_index(drop=True)
    if not _frames_exact_equal(b_dist_core, c_dist_core):
        mismatches.append("distribution_summary")
    if mismatches:
        raise ValueError(
            "frozen B/C observed beta-total mapping differs for: " + ", ".join(mismatches[:8])
        )
    comparison = pd.DataFrame(comparison_rows, columns=_BETA_TOTAL_COMPARISON_COLUMNS)
    return ObservedEffectBetaTotalArtifacts(
        signal_level_scales=pd.concat([b_scales, c_scales], ignore_index=True),
        canonical_signal_mapping=pd.concat([b_mapping, c_mapping], ignore_index=True),
        distribution_summary=pd.concat([b_distribution, c_distribution], ignore_index=True),
        input_case_comparison=comparison,
        input_identity_manifest=artifacts.input_identity_manifest.copy(deep=True),
    )


__all__ = [
    "DecisionWindow",
    "ObservedEffectCandidate",
    "ObservedEffectBetaTotalArtifacts",
    "ObservedEffectScaleContract",
    "ObservedEffectScaleArtifacts",
    "ObservedEffectScaleInput",
    "estimate_l0_l4_observed_effect_scale_v1",
    "map_observed_effect_scale_to_beta_total_v1",
]
