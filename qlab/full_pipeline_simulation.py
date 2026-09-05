"""Formal, pre-simulation diagnostics for the complete L0--L4 route.

Lifecycle: candidate implementation for the Issue #34 known-truth simulation
blueprint.  Authority: the sole qlab public entries for the observed-effect
inventory and the core G_beta_total_v1 observed-scale mapping.  Inputs: two
independently frozen B/C cache and price identities plus a pre-frozen registry,
windows, and candidate contract.  May be used for: retaining every unfiltered
candidate estimate and producing the frozen core G_beta_total_v1 weak/center/
strong mapping from duplicate-aware observed beta-total scale evidence.  Must
not be used for: candidate selection, significance testing, L2/L3/L4 discovery,
changing the frozen five-scale mapping, Monte Carlo N or append rules,
simulation generation, or a research conclusion.  Archive condition:
this v1 contract is superseded by an approved, versioned successor.

The module deliberately reuses the formal qlab panel/rank and executable
return paths; it does not approximate them in a research-layer calculation.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from numbers import Integral
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

    This is the formal core ``G_beta_total_v1`` weak/center/strong scale
    mapping.  It consumes the already completed per-(candidate, horizon)
    inventory and never reads significance, L2/L3/L4 selection, or discovery
    fields.  Outer very-weak/very-strong multipliers and Monte Carlo rules are
    outside this core mapping.
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
    absolute P10/P50/P90 are the frozen core ``G_beta_total_v1`` weak/center/
    strong scale mapping.  This remains separate from the per-(signal,
    horizon) real-analysis estimand and does not decide outer
    very-weak/very-strong multipliers, Monte Carlo N, append rules, or
    simulation generation.
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


# These are contract identities, not simulation outputs.  They deliberately
# live beside the existing observed-scale entry so that a future simulator can
# validate its frozen scenario before doing any data generation.  This section
# must remain pure: it must not construct prices, signals, returns, or call an
# L0--L4 entry point.
KNOWN_TRUTH_BETA_TOTAL_SCALES_V1 = (
    ("very_weak", 0.00011899372586593),
    ("weak", 0.00023798745173186),
    ("center", 0.0005372604401853),
    ("strong", 0.0011493639200697),
    ("very_strong", 0.0022987278401394),
)
KNOWN_TRUTH_EFFECT_CURVES_V1 = ("fast", "delayed", "persistent")
KNOWN_TRUTH_MIRROR_SIGNS_V1 = (-1, 1)
KNOWN_TRUTH_HORIZONS_V1 = ("4h", "8h", "12h", "1d")
KNOWN_TRUTH_ADMITTED_SYMBOLS_V1 = (
    "ADA",
    "APT",
    "AVAX",
    "BCH",
    "BNB",
    "BTC",
    "DOGE",
    "DOT",
    "ETC",
    "ETH",
    "FET",
    "FIL",
    "LINK",
    "LTC",
    "NEAR",
    "SOL",
    "SUI",
    "TRX",
    "UNI",
    "XRP",
)
KNOWN_TRUTH_REGISTRY_SOURCE_V1 = (
    "qlab_research_private/research/crypto/results/candidate/"
    "ksv4_endpoint_semantics_rebuild_l2_20260802_factor_registry.csv"
)
KNOWN_TRUTH_REGISTRY_SOURCE_SHA256_V1 = (
    "deb95f04dacb8f25a2abc3fecd0c21bac5f6fea6f674cf475dbe86084d0caa52"
)
KNOWN_TRUTH_CANDIDATE_IDENTITY_SOURCE_V1 = (
    "qlab_research_private/research/crypto/results/candidate/"
    "ksv4_endpoint_semantics_rebuild_l2_20260802_summary.csv"
)
KNOWN_TRUTH_CANDIDATE_IDENTITY_SOURCE_SHA256_V1 = (
    "2e721db0ebf39f2adc0084ab6c01e430c8f8bf233ba74c0f2c07951539f08bb6"
)
KNOWN_TRUTH_UNIVERSE_SOURCE_V1 = (
    "qlab_research_private/research/crypto/results/"
    "coinglass_universe_admission_audit.csv"
)
KNOWN_TRUTH_UNIVERSE_SOURCE_SHA256_V1 = (
    "fd86fefa275e51fcb79dcae08ff5538b691be5b947bd09f6670c490e4e268b0e"
)
KNOWN_TRUTH_REGISTRY_FEATURE_COUNT_V1 = 68
KNOWN_TRUTH_CANDIDATE_HORIZON_COUNTS_V1 = (
    ("4h", 23),
    ("8h", 23),
    ("12h", 45),
    ("1d", 68),
)
KNOWN_TRUTH_REGISTRY_IDENTITY_V1 = "ksv4_formal_registry_68_features_v1"
KNOWN_TRUTH_UNIVERSE_IDENTITY_V1 = "coinglass_admitted_universe_20_ordered_v1"
KNOWN_TRUTH_CORE_SCENARIO_ROLES_V1 = (
    "all_null",
    "direct_sparse",
    "proxy_and_alias",
    "rank_only",
)
KNOWN_TRUTH_OPTIONAL_SCENARIO_ROLES_V1 = ("mixed_null",)
KNOWN_TRUTH_CANDIDATE_ROLES_V1 = (
    "direct",
    "proxy",
    "alias",
    "near_alias",
    "null",
)
KNOWN_TRUTH_FORMAL_REPLICATES_V1 = 1100
KNOWN_TRUTH_APPEND_POLICY_V1 = "stop_and_report_uncertain_no_append_v1"
KNOWN_TRUTH_REALITY_SCOPE_V1 = "independent_per_signal_horizon_v1"
KNOWN_TRUTH_SIMULATION_EFFECT_SCOPE_V1 = "shared_beta_total_released_by_curve_v1"
KNOWN_TRUTH_SCALAR_EXPRESSION_V1 = "standardized_scalar_signal_v1"
KNOWN_TRUTH_RANK_ONLY_EXPRESSION_V1 = "cross_section_rank_only_v1"
KNOWN_TRUTH_NULL_EXPRESSION_V1 = "independent_null_v1"
KNOWN_TRUTH_FORMAL_AUTHORITY_V1 = "issue_34_issue_36_active_known_truth_blueprint_v1"
KNOWN_TRUTH_LIFECYCLE_V1 = "candidate_contract_validation_only_v1"
KNOWN_TRUTH_MAY_BE_USED_FOR_V1 = "pre_generation_contract_validation_only_v1"
KNOWN_TRUTH_MUST_NOT_BE_USED_FOR_V1 = (
    "no_generation_no_l0_l4_no_research_conclusion_v1"
)
KNOWN_TRUTH_ARCHIVE_CONDITION_V1 = (
    "superseded_by_approved_versioned_simulation_contract_v1"
)
KNOWN_TRUTH_INPUTS_V1 = (
    "frozen_registry_source",
    "frozen_candidate_identity_source",
    "frozen_ordered_universe_source",
    "four_formal_horizons",
    "five_beta_total_scales",
    "three_effect_curves_and_mirrors",
    "truth_manifest",
    "task_manifest",
)
KNOWN_TRUTH_SEED_PHASES_V1 = ("development", "formal")


DETERMINISTIC_RANDOM_ADDRESS_VERSION_V1 = "ksv4-deterministic-random-address/v1"
DETERMINISTIC_RANDOM_ALLOWED_PHASES_V1 = ("development", "formal")
DETERMINISTIC_RANDOM_ALLOWED_STREAM_KINDS_V1 = (
    "base",
    "measurement",
    "null",
    "price",
)
_DETERMINISTIC_RANDOM_MAX_TEXT_BYTES_V1 = 128
_DETERMINISTIC_RANDOM_MAX_TIME_INDEX_V1 = (1 << 63) - 1
_DETERMINISTIC_RANDOM_MAX_ASSET_INDEX_V1 = (1 << 32) - 1


@dataclass(frozen=True)
class DeterministicRandomAddressV1:
    """One stable address and uint64 seed for a registered simulation stream.

    This is an address derivation primitive only.  It does not sample a
    distribution or create a price, signal, innovation, or return.  The v1
    wire format is deliberately explicit and cross-process stable:

    ``SHA256(domain || LP(namespace) || LP(phase) || LP(stream_kind) ||
    LP(registered_stream_or_group_id) || U64BE(time_index) ||
    U32BE(asset_index))``.

    ``domain`` is the ASCII bytes
    ``b"ksv4-deterministic-random-address/v1\\0"`` and ``LP`` is a four-byte
    unsigned big-endian byte length followed by visible ASCII bytes.  The
    full digest, lower-case hexadecimal, is ``address_hex``; the first eight
    digest bytes interpreted as an unsigned big-endian integer are
    ``seed_uint64``.  The text fields are non-empty visible ASCII and at most
    128 bytes.  Indices are non-negative integers in their documented v1
    ranges; booleans and floating-point values are rejected.
    """

    address_hex: str
    seed_uint64: int


def _validate_deterministic_random_text_v1(value: object, *, label: str) -> bytes:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact str")
    if not value:
        raise ValueError(f"{label} must be non-empty")
    if not value.isascii() or any(not (0x21 <= ord(char) <= 0x7E) for char in value):
        raise ValueError(f"{label} must contain visible ASCII only")
    encoded = value.encode("ascii")
    if len(encoded) > _DETERMINISTIC_RANDOM_MAX_TEXT_BYTES_V1:
        raise ValueError(f"{label} exceeds 128 bytes")
    return encoded


def _validate_deterministic_random_index_v1(
    value: object,
    *,
    label: str,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    normalized = int(value)
    if normalized < 0 or normalized > maximum:
        raise ValueError(f"{label} must be in [0, {maximum}]")
    return normalized


def derive_deterministic_random_address_v1(
    seed_namespace: str,
    phase: str,
    stream_kind: str,
    registered_stream_or_group_id: str,
    time_index: int,
    asset_index: int,
) -> DeterministicRandomAddressV1:
    """Derive a deterministic, order-independent v1 random address and seed.

    Every identity field is serialized into the digest, so callers may
    reorder files, tasks, or parallel shards without changing an address.
    ``phase`` and ``stream_kind`` are closed vocabularies; namespace and
    registered stream/group identifiers use the visible-ASCII rule documented
    on :class:`DeterministicRandomAddressV1`.  This function only derives
    bytes and never touches a process-global RNG.
    """
    namespace_bytes = _validate_deterministic_random_text_v1(
        seed_namespace,
        label="seed_namespace",
    )
    if type(phase) is not str or phase not in DETERMINISTIC_RANDOM_ALLOWED_PHASES_V1:
        raise ValueError(
            "phase must be one of "
            + ", ".join(DETERMINISTIC_RANDOM_ALLOWED_PHASES_V1)
        )
    if (
        type(stream_kind) is not str
        or stream_kind not in DETERMINISTIC_RANDOM_ALLOWED_STREAM_KINDS_V1
    ):
        raise ValueError(
            "stream_kind must be one of "
            + ", ".join(DETERMINISTIC_RANDOM_ALLOWED_STREAM_KINDS_V1)
        )
    phase_bytes = _validate_deterministic_random_text_v1(phase, label="phase")
    stream_kind_bytes = _validate_deterministic_random_text_v1(
        stream_kind,
        label="stream_kind",
    )
    stream_id_bytes = _validate_deterministic_random_text_v1(
        registered_stream_or_group_id,
        label="registered_stream_or_group_id",
    )
    normalized_time = _validate_deterministic_random_index_v1(
        time_index,
        label="time_index",
        maximum=_DETERMINISTIC_RANDOM_MAX_TIME_INDEX_V1,
    )
    normalized_asset = _validate_deterministic_random_index_v1(
        asset_index,
        label="asset_index",
        maximum=_DETERMINISTIC_RANDOM_MAX_ASSET_INDEX_V1,
    )

    def length_prefix(value: bytes) -> bytes:
        return struct.pack(">I", len(value)) + value

    payload = DETERMINISTIC_RANDOM_ADDRESS_VERSION_V1.encode("ascii") + b"\0"
    payload += length_prefix(namespace_bytes)
    payload += length_prefix(phase_bytes)
    payload += length_prefix(stream_kind_bytes)
    payload += length_prefix(stream_id_bytes)
    payload += struct.pack(">Q", normalized_time)
    payload += struct.pack(">I", normalized_asset)
    digest = hashlib.sha256(payload).digest()
    return DeterministicRandomAddressV1(
        address_hex=digest.hex(),
        seed_uint64=int.from_bytes(digest[:8], byteorder="big", signed=False),
    )


RANDOM_STREAM_SPECIFICATION_VERSION_V1 = "ksv4-random-stream-specification/v1"
RANDOM_STREAM_CALIBRATION_STATUSES_V1 = ("frozen", "unavailable")
RANDOM_STREAM_SPECIFICATION_LIFECYCLE_V1 = (
    "candidate_random_stream_specification_validation_only_v1"
)
RANDOM_STREAM_SPECIFICATION_AUTHORITY_V1 = (
    "issue_34_known_truth_blueprint_random_stream_specification_v1"
)
RANDOM_STREAM_SPECIFICATION_MAY_BE_USED_FOR_V1 = (
    "random_stream_identity_validation_before_generation_v1"
)
RANDOM_STREAM_SPECIFICATION_MUST_NOT_BE_USED_FOR_V1 = (
    "no_sampling_no_generation_no_l0_l4_no_research_conclusion_v1"
)
RANDOM_STREAM_SPECIFICATION_ARCHIVE_CONDITION_V1 = (
    "superseded_by_approved_random_stream_specification_v1"
)


@dataclass(frozen=True)
class RandomTimeProcessSpecificationV1:
    """Identity-only specification for one registered time-process family.

    This object deliberately stores identities rather than AR/MA parameters or
    any sampled state.  ``calibration_status=unavailable`` is a valid explicit
    state: it records that no result-blind calibration was available without
    inventing a reality claim.
    """

    family_id: str
    parameter_identity: str
    initialization_identity: str
    burn_in_steps: int
    native_frequency: str
    time_order_identity: str
    calibration_identity: str
    calibration_status: str
    tail_rule_identity: str


@dataclass(frozen=True)
class RandomStreamSpecificationV1:
    """One base, measurement, null, or price stream identity.

    ``random_address_group_id`` is the registered stream/group identifier
    passed to :func:`derive_deterministic_random_address_v1`.  Reusing it is
    therefore an explicit declaration of shared addresses; the validator
    permits sharing only within one stream kind and requires a justification.
    This record never creates innovations, prices, signals, or returns.
    """

    stream_id: str
    stream_kind: str
    information_group_id: str | None
    random_address_group_id: str
    innovation_distribution_id: str
    asset_symbols: tuple[str, ...]
    r_identity: str | None
    r_decomposition_identity: str | None
    r_calibration_identity: str
    time_process: RandomTimeProcessSpecificationV1
    shared_random_group_justification_id: str | None = None


@dataclass(frozen=True)
class RandomInformationGroupStreamBindingV1:
    """The base stream identity shared by a group's proxy/near-alias roles.

    A later candidate map may contain many proxy or near-alias candidates, but
    all of them must use ``proxy_near_alias_base_stream_id``.  Keeping this
    relation in the stream registry makes a wrong base reference fail before
    any scenario or generator code exists.
    """

    information_group_id: str
    base_stream_id: str
    proxy_near_alias_base_stream_id: str
    measurement_stream_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RandomStreamSpecificationRegistryV1:
    """Immutable registry for random-stream and R/T identities only.

    The registry binds the already approved deterministic address primitive,
    phase, fixed 20-asset order, four stream kinds, R identities, T
    identities, initialization, burn-in, native frequency and calibration
    status.  It is a pre-generation validation object, not a scenario
    manifest and not a random-number, price, signal, or return generator.
    """

    specification_version: str
    seed_namespace: str
    phase: str
    address_derivation_version: str
    asset_symbols: tuple[str, ...]
    streams: tuple[RandomStreamSpecificationV1, ...]
    information_group_bindings: tuple[RandomInformationGroupStreamBindingV1, ...]
    lifecycle: str
    authority: str
    may_be_used_for: str
    must_not_be_used_for: str
    archive_condition: str


def _validate_random_stream_text_v1(value: object, *, label: str) -> str:
    _validate_deterministic_random_text_v1(value, label=label)
    return value  # type: ignore[return-value]


def _validate_random_time_process_specification_v1(
    process: object,
    *,
    label: str,
) -> RandomTimeProcessSpecificationV1:
    if not isinstance(process, RandomTimeProcessSpecificationV1):
        raise TypeError(f"{label} must be RandomTimeProcessSpecificationV1")
    for field_name in (
        "family_id",
        "parameter_identity",
        "initialization_identity",
        "native_frequency",
        "time_order_identity",
        "calibration_identity",
        "tail_rule_identity",
    ):
        _validate_random_stream_text_v1(
            getattr(process, field_name),
            label=f"{label}.{field_name}",
        )
    if (
        isinstance(process.burn_in_steps, bool)
        or not isinstance(process.burn_in_steps, Integral)
        or process.burn_in_steps < 0
    ):
        raise ValueError(f"{label}.burn_in_steps must be a non-negative integer")
    if process.calibration_status not in RANDOM_STREAM_CALIBRATION_STATUSES_V1:
        raise ValueError(
            f"{label}.calibration_status must be one of "
            + ", ".join(RANDOM_STREAM_CALIBRATION_STATUSES_V1)
        )
    return process


def validate_random_stream_specification_v1(
    registry: RandomStreamSpecificationRegistryV1,
) -> RandomStreamSpecificationRegistryV1:
    """Validate a closed random-stream/R/T identity registry.

    Validation is intentionally structural.  It does not select scientific
    parameters, construct an R matrix or T recursion, sample an innovation,
    or call any L0--L4 path.  The registry's ``random_address_group_id`` is
    checked through the approved pure address primitive at ``t=0, i=0`` only
    to prove that the recorded identity is consumable by that primitive.
    """
    if not isinstance(registry, RandomStreamSpecificationRegistryV1):
        raise TypeError(
            "registry must be RandomStreamSpecificationRegistryV1"
        )
    if registry.specification_version != RANDOM_STREAM_SPECIFICATION_VERSION_V1:
        raise ValueError("random-stream specification_version is not the frozen v1 version")
    _validate_random_stream_text_v1(registry.seed_namespace, label="seed_namespace")
    if registry.phase not in DETERMINISTIC_RANDOM_ALLOWED_PHASES_V1:
        raise ValueError(
            "random-stream phase must be one of "
            + ", ".join(DETERMINISTIC_RANDOM_ALLOWED_PHASES_V1)
        )
    if registry.address_derivation_version != DETERMINISTIC_RANDOM_ADDRESS_VERSION_V1:
        raise ValueError(
            "random-stream address_derivation_version does not match the approved primitive"
        )
    if type(registry.asset_symbols) is not tuple:
        raise ValueError("random-stream asset_symbols must be an immutable tuple")
    if registry.asset_symbols != KNOWN_TRUTH_ADMITTED_SYMBOLS_V1:
        raise ValueError(
            "random-stream asset_symbols must equal the frozen 20-asset order"
        )
    if type(registry.streams) is not tuple or not registry.streams:
        raise ValueError("random-stream streams must be a non-empty immutable tuple")
    if type(registry.information_group_bindings) is not tuple:
        raise ValueError(
            "random-stream information_group_bindings must be an immutable tuple"
        )
    for field_name, expected in (
        ("lifecycle", RANDOM_STREAM_SPECIFICATION_LIFECYCLE_V1),
        ("authority", RANDOM_STREAM_SPECIFICATION_AUTHORITY_V1),
        ("may_be_used_for", RANDOM_STREAM_SPECIFICATION_MAY_BE_USED_FOR_V1),
        ("must_not_be_used_for", RANDOM_STREAM_SPECIFICATION_MUST_NOT_BE_USED_FOR_V1),
        ("archive_condition", RANDOM_STREAM_SPECIFICATION_ARCHIVE_CONDITION_V1),
    ):
        actual = getattr(registry, field_name)
        _validate_random_stream_text_v1(actual, label=f"random-stream {field_name}")
        if actual != expected:
            raise ValueError(f"random-stream {field_name} is not the frozen boundary")

    by_stream_id: dict[str, RandomStreamSpecificationV1] = {}
    by_kind: dict[str, list[RandomStreamSpecificationV1]] = {
        kind: [] for kind in DETERMINISTIC_RANDOM_ALLOWED_STREAM_KINDS_V1
    }
    by_random_address_group: dict[str, list[RandomStreamSpecificationV1]] = {}
    base_by_information_group: dict[str, RandomStreamSpecificationV1] = {}

    for index, stream in enumerate(registry.streams):
        label = f"random-stream streams[{index}]"
        if not isinstance(stream, RandomStreamSpecificationV1):
            raise TypeError(f"{label} must be RandomStreamSpecificationV1")
        _validate_random_stream_text_v1(stream.stream_id, label=f"{label}.stream_id")
        if stream.stream_id in by_stream_id:
            raise ValueError(f"duplicate random stream_id: {stream.stream_id}")
        by_stream_id[stream.stream_id] = stream
        if stream.stream_kind not in DETERMINISTIC_RANDOM_ALLOWED_STREAM_KINDS_V1:
            raise ValueError(f"{label}.stream_kind is not registered")
        by_kind[stream.stream_kind].append(stream)
        _validate_random_stream_text_v1(
            stream.random_address_group_id,
            label=f"{label}.random_address_group_id",
        )
        by_random_address_group.setdefault(stream.random_address_group_id, []).append(stream)
        _validate_random_stream_text_v1(
            stream.innovation_distribution_id,
            label=f"{label}.innovation_distribution_id",
        )
        if type(stream.asset_symbols) is not tuple:
            raise ValueError(f"{label}.asset_symbols must be an immutable tuple")
        if stream.asset_symbols != registry.asset_symbols:
            raise ValueError(f"{label}.asset_symbols must equal the registry order")
        if stream.information_group_id is not None:
            _validate_random_stream_text_v1(
                stream.information_group_id,
                label=f"{label}.information_group_id",
            )
        if stream.stream_kind == "base":
            if stream.information_group_id is None:
                raise ValueError(f"{label}.base stream requires information_group_id")
            if stream.information_group_id in base_by_information_group:
                raise ValueError(
                    "each information group must have exactly one base stream: "
                    + stream.information_group_id
                )
            base_by_information_group[stream.information_group_id] = stream
        elif stream.stream_kind in ("null", "price"):
            if stream.information_group_id is not None:
                raise ValueError(
                    f"{label}.{stream.stream_kind} stream cannot bind an information group"
                )
        if stream.r_identity is None or stream.r_decomposition_identity is None:
            raise ValueError(
                f"{label} requires both r_identity and r_decomposition_identity"
            )
        _validate_random_stream_text_v1(stream.r_identity, label=f"{label}.r_identity")
        _validate_random_stream_text_v1(
            stream.r_decomposition_identity,
            label=f"{label}.r_decomposition_identity",
        )
        _validate_random_stream_text_v1(
            stream.r_calibration_identity,
            label=f"{label}.r_calibration_identity",
        )
        _validate_random_time_process_specification_v1(
            stream.time_process,
            label=f"{label}.time_process",
        )
        if stream.shared_random_group_justification_id is not None:
            _validate_random_stream_text_v1(
                stream.shared_random_group_justification_id,
                label=f"{label}.shared_random_group_justification_id",
            )
        derive_deterministic_random_address_v1(
            registry.seed_namespace,
            registry.phase,
            stream.stream_kind,
            stream.random_address_group_id,
            0,
            0,
        )

    for kind in DETERMINISTIC_RANDOM_ALLOWED_STREAM_KINDS_V1:
        if not by_kind[kind]:
            raise ValueError(f"random-stream registry requires at least one {kind} stream")
    if len(by_kind["price"]) != 1:
        raise ValueError("random-stream registry requires exactly one price stream")
    base_address_groups = [stream.random_address_group_id for stream in by_kind["base"]]
    if len(set(base_address_groups)) != len(base_address_groups):
        raise ValueError("base streams for different information groups must be independent")

    for group_id, streams in by_random_address_group.items():
        kinds = {stream.stream_kind for stream in streams}
        if len(kinds) > 1:
            raise ValueError(
                "random_address_group_id cannot be shared across stream kinds: " + group_id
            )
        if len(streams) > 1:
            justifications = {
                stream.shared_random_group_justification_id for stream in streams
            }
            if len(justifications) != 1 or None in justifications:
                raise ValueError(
                    "shared random address group requires one explicit justification: "
                    + group_id
                )

    binding_by_group: dict[str, RandomInformationGroupStreamBindingV1] = {}
    for index, binding in enumerate(registry.information_group_bindings):
        label = f"random-stream information_group_bindings[{index}]"
        if not isinstance(binding, RandomInformationGroupStreamBindingV1):
            raise TypeError(
                f"{label} must be RandomInformationGroupStreamBindingV1"
            )
        _validate_random_stream_text_v1(
            binding.information_group_id,
            label=f"{label}.information_group_id",
        )
        if binding.information_group_id in binding_by_group:
            raise ValueError(
                "duplicate random information-group binding: "
                + binding.information_group_id
            )
        binding_by_group[binding.information_group_id] = binding
        base = by_stream_id.get(binding.base_stream_id)
        proxy_base = by_stream_id.get(binding.proxy_near_alias_base_stream_id)
        if base is None or proxy_base is None:
            raise ValueError(f"{label} references an unregistered base stream")
        if base.stream_kind != "base" or proxy_base.stream_kind != "base":
            raise ValueError(f"{label} base references must be base streams")
        if base.information_group_id != binding.information_group_id:
            raise ValueError(f"{label}.base_stream_id has the wrong information group")
        if proxy_base.information_group_id != binding.information_group_id:
            raise ValueError(
                f"{label}.proxy_near_alias_base_stream_id has the wrong information group"
            )
        if binding.base_stream_id != binding.proxy_near_alias_base_stream_id:
            raise ValueError(
                f"{label} proxy/near_alias roles must share the group's base stream"
            )
        if type(binding.measurement_stream_ids) is not tuple:
            raise ValueError(f"{label}.measurement_stream_ids must be an immutable tuple")
        if len(set(binding.measurement_stream_ids)) != len(binding.measurement_stream_ids):
            raise ValueError(f"{label}.measurement_stream_ids contains duplicates")
        for measurement_id in binding.measurement_stream_ids:
            measurement = by_stream_id.get(measurement_id)
            if measurement is None or measurement.stream_kind != "measurement":
                raise ValueError(f"{label} references an unregistered measurement stream")
            if (
                measurement.information_group_id is not None
                and measurement.information_group_id != binding.information_group_id
            ):
                raise ValueError(
                    f"{label} measurement stream has the wrong information group"
                )

    if set(binding_by_group) != set(base_by_information_group):
        raise ValueError(
            "information-group bindings must cover exactly all registered base groups"
        )
    for stream in by_kind["measurement"]:
        if stream.information_group_id is not None:
            binding = binding_by_group.get(stream.information_group_id)
            if binding is None or stream.stream_id not in binding.measurement_stream_ids:
                raise ValueError(
                    "group-bound measurement stream is missing from its information-group binding"
                )
    return registry


def _known_truth_formal_candidate_ids_v1() -> tuple[str, ...]:
    """Derive the frozen candidate identities from qlab's formal registry."""
    timeframe_hours = {"1h": 1, "4h": 4, "8h": 8, "12h": 12, "1d": 24}
    formal_registry = factor_registry.base_panel_registry("1h")
    candidate_ids: list[str] = []
    for row in formal_registry.itertuples(index=False):
        signal_hours = timeframe_hours[str(row.signal_timeframe)]
        for horizon in KNOWN_TRUTH_HORIZONS_V1:
            horizon_hours = timeframe_hours[horizon]
            if signal_hours <= horizon_hours and horizon_hours % signal_hours == 0:
                candidate_ids.append(f"{row.feature_name}::{horizon}")
    return tuple(candidate_ids)


KNOWN_TRUTH_REGISTRY_CANDIDATE_IDS_V1 = _known_truth_formal_candidate_ids_v1()
KNOWN_TRUTH_REGISTRY_CANDIDATE_IDS_SHA256_V1 = (
    "0cc692c6ea79348592f5f667077902b79b20b521a57bda092931d81e3b1e255c"
)
if (
    len(KNOWN_TRUTH_REGISTRY_CANDIDATE_IDS_V1) != 159
    or _json_content_sha256(list(KNOWN_TRUTH_REGISTRY_CANDIDATE_IDS_V1))
    != KNOWN_TRUTH_REGISTRY_CANDIDATE_IDS_SHA256_V1
):
    raise RuntimeError("qlab formal registry no longer matches the frozen 159-candidate identity")
KNOWN_TRUTH_EFFECT_CASE_COVERAGE_V1 = tuple(
    (scale_label, curve_id, mirror_sign)
    for scale_label, _ in KNOWN_TRUTH_BETA_TOTAL_SCALES_V1
    for curve_id in KNOWN_TRUTH_EFFECT_CURVES_V1
    for mirror_sign in KNOWN_TRUTH_MIRROR_SIGNS_V1
)


@dataclass(frozen=True)
class KnownTruthSignalAssignmentV1:
    """One candidate's pre-registered truth identity in a scenario.

    The blueprint fixes the role vocabulary and the information-group
    relationship, but does not prescribe a particular candidate-to-group
    allocation for every future scenario.  This record therefore requires the
    allocation to be supplied and validates its identity without inventing an
    allocation.  Effect fields are required only when the candidate is a
    direct return-generating signal; the rank-only scenario changes the signal
    shape, not this role vocabulary.
    """

    candidate_id: str
    information_group: str | None
    base_signal_family: str | None
    role: str
    base_random_stream_id: str | None = None
    alias_of_candidate_id: str | None = None
    observation_variant_id: str | None = None
    measurement_noise_stream_id: str | None = None
    null_noise_stream_id: str | None = None
    standardization_id: str | None = None
    expression_type: str | None = None
    direction: int | None = None
    effect_scale_label: str | None = None
    effect_curve_id: str | None = None
    w_effect_id: str | None = None
    mirror_sign: int | None = None
    beta_id: str | None = None
    beta_total: float | None = None
    beta_rank: float | None = None
    w_rank: str | None = None
    analytic_truth_proof: str | None = None
    rho: float | None = None
    noise_scale: float | None = None
    return_inclusion: bool | None = None
    marginal_predictive_truth: int | None = None


@dataclass(frozen=True)
class KnownTruthScenarioV1:
    """A truth role plus a complete 159-candidate truth manifest slice."""

    scenario_id: str
    truth_role: str
    information_groups: tuple[str, ...]
    expression_id: str
    truth_assignments: tuple[KnownTruthSignalAssignmentV1, ...]


@dataclass(frozen=True)
class KnownTruthTaskV1:
    """Stable identity for one development or formal simulation task."""

    task_id: str
    scenario_id: str
    phase: str
    replicate_id: int
    seed_namespace: str
    seed: int


@dataclass(frozen=True)
class KnownTruthSimulationContractV1:
    """Pure, pre-generation contract for the Issue #34 simulation.

    This object freezes identities and validation rules only.  It is not a
    generator and has no permission to call the formal L0--L4 path.  The
    scenario-specific information-group allocation, signal-family allocation,
    noise streams, and direct members must be provided by a later approved
    scenario manifest; this validator checks that such a manifest is complete
    and internally coherent without selecting those scientific values.
    """

    contract_id: str
    registry_candidate_ids: tuple[str, ...]
    admitted_symbols: tuple[str, ...]
    registry_identity: str
    registry_source: str
    registry_source_sha256: str
    registry_feature_count: int
    candidate_identity_source: str
    candidate_identity_source_sha256: str
    candidate_horizon_counts: tuple[tuple[str, int], ...]
    universe_identity: str
    universe_source: str
    universe_source_sha256: str
    horizons: tuple[str, ...]
    beta_total_scales: tuple[tuple[str, float], ...]
    effect_curve_ids: tuple[str, ...]
    mirror_signs: tuple[int, ...]
    effect_case_coverage: tuple[tuple[str, str, int], ...]
    formal_replicates: int
    development_seed_namespace: str
    formal_seed_namespace: str
    allow_adaptive_append: bool
    append_policy: str
    reality_analysis_scope: str
    simulation_effect_scope: str
    scenarios: tuple[KnownTruthScenarioV1, ...]
    tasks: tuple[KnownTruthTaskV1, ...]
    lifecycle: str
    authority: str
    inputs: tuple[str, ...]
    may_be_used_for: str
    must_not_be_used_for: str
    archive_condition: str


def _known_truth_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"known-truth contract {label} must be a non-empty string")
    return value


def _known_truth_tuple(value: object, *, label: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"known-truth contract {label} must be an immutable tuple")
    return value


def _known_truth_decimal(value: object, *, label: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"known-truth contract {label} must be finite")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError(f"known-truth contract {label} must be finite") from None
    if not result.is_finite():
        raise ValueError(f"known-truth contract {label} must be finite")
    return result


def _validate_known_truth_scales_v1(
    actual_scales: object,
) -> None:
    scales = _known_truth_tuple(actual_scales, label="beta_total_scales")
    if len(scales) != len(KNOWN_TRUTH_BETA_TOTAL_SCALES_V1):
        raise ValueError("known-truth contract must contain exactly five beta_total scales")
    seen_labels: set[str] = set()
    for index, (actual, expected) in enumerate(
        zip(scales, KNOWN_TRUTH_BETA_TOTAL_SCALES_V1, strict=True)
    ):
        if not isinstance(actual, tuple) or len(actual) != 2:
            raise ValueError(f"known-truth beta_total scale {index} must be (label, value)")
        label, value = actual
        if label != expected[0] or label in seen_labels:
            raise ValueError("known-truth beta_total scale labels/order are not frozen")
        if _known_truth_decimal(value, label=f"beta_total_scales[{index}].value") != _known_truth_decimal(
            expected[1], label=f"expected beta_total_scales[{index}].value"
        ):
            raise ValueError("known-truth beta_total scale value changed")
        seen_labels.add(label)


def _validate_known_truth_assignment_v1(
    assignment: KnownTruthSignalAssignmentV1,
    *,
    candidate_ids: set[str],
    information_groups: set[str],
    scenario_role: str,
    scenario_expression_id: str,
) -> None:
    if not isinstance(assignment, KnownTruthSignalAssignmentV1):
        raise TypeError("known-truth scenario assignments must use the v1 data structure")
    candidate_id = _known_truth_text(assignment.candidate_id, label="candidate_id")
    if candidate_id not in candidate_ids:
        raise ValueError("known-truth assignment references an unknown candidate")
    if assignment.role not in KNOWN_TRUTH_CANDIDATE_ROLES_V1:
        raise ValueError(f"unknown known-truth candidate role: {assignment.role!r}")

    if assignment.role == "null":
        if any(
            value is not None
            for value in (
                assignment.information_group,
                assignment.base_signal_family,
                assignment.base_random_stream_id,
            )
        ):
            raise ValueError("known-truth null candidates cannot bind a base signal")
    else:
        information_group = _known_truth_text(
            assignment.information_group,
            label="information_group",
        )
        if information_group not in information_groups:
            raise ValueError("known-truth assignment references an unregistered information group")
        _known_truth_text(assignment.base_signal_family, label="base_signal_family")
        _known_truth_text(assignment.base_random_stream_id, label="base_random_stream_id")

    _known_truth_text(assignment.standardization_id, label="standardization_id")
    expression_type = _known_truth_text(assignment.expression_type, label="expression_type")
    _known_truth_text(assignment.analytic_truth_proof, label="analytic_truth_proof")
    if assignment.role == "null":
        if expression_type != KNOWN_TRUTH_NULL_EXPRESSION_V1:
            raise ValueError("known-truth null expression type is not the frozen null expression")
    elif scenario_expression_id == KNOWN_TRUTH_RANK_ONLY_EXPRESSION_V1:
        if expression_type != KNOWN_TRUTH_RANK_ONLY_EXPRESSION_V1:
            raise ValueError("rank_only candidates require the independent rank-only expression")
    elif expression_type != KNOWN_TRUTH_SCALAR_EXPRESSION_V1:
        raise ValueError("non-rank known-truth candidates require the scalar expression")

    if assignment.return_inclusion not in {True, False}:
        raise ValueError("known-truth return_inclusion must be an explicit boolean")
    if (
        not isinstance(assignment.marginal_predictive_truth, Integral)
        or isinstance(assignment.marginal_predictive_truth, bool)
        or assignment.marginal_predictive_truth not in {0, 1}
    ):
        raise ValueError("known-truth marginal_predictive_truth must be 0 or 1")

    if assignment.role == "null":
        if assignment.observation_variant_id is not None:
            raise ValueError("known-truth null candidates cannot use an observation variant")
        _known_truth_text(assignment.null_noise_stream_id, label="null_noise_stream_id")
        if assignment.measurement_noise_stream_id is not None:
            raise ValueError("known-truth null candidates cannot use measurement noise")
    else:
        _known_truth_text(assignment.observation_variant_id, label="observation_variant_id")
        if assignment.null_noise_stream_id is not None:
            raise ValueError("known-truth non-null candidates cannot use null noise")

    if assignment.role == "alias":
        alias_of = _known_truth_text(assignment.alias_of_candidate_id, label="alias_of_candidate_id")
        if alias_of not in candidate_ids or alias_of == candidate_id:
            raise ValueError("known-truth alias must reference a different registered candidate")
    elif assignment.alias_of_candidate_id is not None:
        raise ValueError("only alias candidates may set alias_of_candidate_id")

    if assignment.role in {"proxy", "near_alias"}:
        _known_truth_text(
            assignment.measurement_noise_stream_id,
            label="measurement_noise_stream_id",
        )
        if assignment.null_noise_stream_id is not None:
            raise ValueError("proxy and near_alias candidates cannot use null noise")
    elif assignment.measurement_noise_stream_id is not None:
        raise ValueError("only proxy and near_alias candidates may use measurement noise")

    if assignment.role in {"proxy", "near_alias", "null"}:
        noise_scale = _known_truth_decimal(assignment.noise_scale, label="noise_scale")
        if noise_scale <= 0:
            raise ValueError("known-truth noise_scale must be positive")
    elif assignment.noise_scale is not None:
        raise ValueError("direct and alias candidates cannot set noise_scale")

    if assignment.role in {"proxy", "near_alias"}:
        rho = _known_truth_decimal(assignment.rho, label="rho")
        if rho == 0 or abs(rho) >= 1:
            raise ValueError("known-truth proxy rho must satisfy 0 < abs(rho) < 1")
        if assignment.direction != (1 if rho > 0 else -1):
            raise ValueError("proxy direction must match the sign of rho")
    elif assignment.rho is not None:
        raise ValueError("only proxy and near_alias candidates may set rho")

    if assignment.role == "null":
        if assignment.direction is not None:
            raise ValueError("null candidates cannot declare a direction")
        if assignment.return_inclusion is not False or assignment.marginal_predictive_truth != 0:
            raise ValueError("null candidates must be excluded from returns and have M=0")
    else:
        if (
            not isinstance(assignment.direction, Integral)
            or isinstance(assignment.direction, bool)
            or assignment.direction not in KNOWN_TRUTH_MIRROR_SIGNS_V1
        ):
            raise ValueError("predictive known-truth candidates require a direction of -1 or 1")
        if assignment.return_inclusion is not (assignment.role == "direct"):
            raise ValueError("only direct candidates may enter the return path")
        if assignment.marginal_predictive_truth != 1:
            raise ValueError("predictive known-truth candidates must have M=1")

    effect_fields = (
        assignment.effect_scale_label,
        assignment.effect_curve_id,
        assignment.w_effect_id,
        assignment.mirror_sign,
        assignment.beta_id,
        assignment.beta_total,
    )
    rank_effect_fields = (
        assignment.effect_scale_label,
        assignment.effect_curve_id,
        assignment.mirror_sign,
        assignment.beta_id,
        assignment.beta_rank,
        assignment.w_rank,
    )
    if assignment.role == "direct":
        if scenario_expression_id == KNOWN_TRUTH_RANK_ONLY_EXPRESSION_V1:
            if any(value is None for value in rank_effect_fields):
                raise ValueError(
                    "rank-only direct candidates require scale, curve, w_rank, sign, beta, and beta_rank"
                )
            if assignment.w_effect_id is not None or assignment.beta_total is not None:
                raise ValueError("rank-only direct candidates cannot bind scalar beta_total or w_effect")
        else:
            if any(value is None for value in effect_fields):
                raise ValueError(
                    "direct known-truth candidates require scale, curve, w_effect, sign, beta, and beta_total"
                )
            if assignment.beta_rank is not None or assignment.w_rank is not None:
                raise ValueError("scalar direct candidates cannot bind rank-only effect fields")
        if assignment.effect_scale_label not in {
            label for label, _ in KNOWN_TRUTH_BETA_TOTAL_SCALES_V1
        }:
            raise ValueError("known-truth effect scale label is not frozen")
        if assignment.effect_curve_id not in KNOWN_TRUTH_EFFECT_CURVES_V1:
            raise ValueError("known-truth effect curve is not frozen")
        if scenario_expression_id == KNOWN_TRUTH_RANK_ONLY_EXPRESSION_V1:
            if assignment.w_rank != assignment.effect_curve_id:
                raise ValueError("known-truth w_rank must bind the declared rank-only effect curve")
        elif assignment.w_effect_id != assignment.effect_curve_id:
            raise ValueError("known-truth w_effect must bind the declared effect curve")
        if assignment.mirror_sign not in KNOWN_TRUTH_MIRROR_SIGNS_V1:
            raise ValueError("known-truth mirror sign is not frozen")
        if assignment.direction != assignment.mirror_sign:
            raise ValueError("direct direction must equal its effect mirror sign")
        expected_beta = dict(KNOWN_TRUTH_BETA_TOTAL_SCALES_V1)[assignment.effect_scale_label]
        if scenario_expression_id == KNOWN_TRUTH_RANK_ONLY_EXPRESSION_V1:
            if _known_truth_decimal(assignment.beta_rank, label="beta_rank") != _known_truth_decimal(
                expected_beta,
                label="expected beta_rank",
            ):
                raise ValueError("rank-only beta_rank does not match its frozen scale label")
        elif _known_truth_decimal(assignment.beta_total, label="beta_total") != _known_truth_decimal(
            expected_beta,
            label="expected beta_total",
        ):
            raise ValueError("direct beta_total does not match its frozen scale label")
        _known_truth_text(assignment.beta_id, label="beta_id")
    elif any(value is not None for value in effect_fields + rank_effect_fields):
        raise ValueError("only direct candidates may bind effect fields")

    if scenario_role == "all_null" and assignment.role != "null":
        raise ValueError("all_null scenario must mark every candidate null")


def _validate_known_truth_scenarios_v1(
    scenarios: object,
    *,
    candidate_ids: tuple[str, ...],
) -> tuple[set[str], set[tuple[str, str, int]]]:
    scenario_rows = _known_truth_tuple(scenarios, label="scenarios")
    if len(scenario_rows) < len(KNOWN_TRUTH_CORE_SCENARIO_ROLES_V1):
        raise ValueError("known-truth contract is missing a core scenario")
    expected_roles = set(KNOWN_TRUTH_CORE_SCENARIO_ROLES_V1)
    allowed_roles = expected_roles | set(KNOWN_TRUTH_OPTIONAL_SCENARIO_ROLES_V1)
    seen_ids: set[str] = set()
    seen_roles: set[str] = set()
    actual_effect_cases: set[tuple[str, str, int]] = set()
    candidate_id_set = set(candidate_ids)
    for scenario in scenario_rows:
        if not isinstance(scenario, KnownTruthScenarioV1):
            raise TypeError("known-truth scenarios must use the v1 data structure")
        scenario_id = _known_truth_text(scenario.scenario_id, label="scenario_id")
        if scenario_id in seen_ids:
            raise ValueError("known-truth scenario identity is duplicated")
        if scenario.truth_role not in allowed_roles or scenario.truth_role in seen_roles:
            raise ValueError("known-truth scenario role is missing, duplicated, or unknown")
        seen_ids.add(scenario_id)
        seen_roles.add(scenario.truth_role)
        expression_id = _known_truth_text(scenario.expression_id, label="expression_id")
        expected_expression_id = (
            KNOWN_TRUTH_NULL_EXPRESSION_V1
            if scenario.truth_role == "all_null"
            else KNOWN_TRUTH_RANK_ONLY_EXPRESSION_V1
            if scenario.truth_role == "rank_only"
            else KNOWN_TRUTH_SCALAR_EXPRESSION_V1
        )
        if expression_id != expected_expression_id:
            raise ValueError("known-truth scenario expression does not match its truth role")
        groups = _known_truth_tuple(scenario.information_groups, label="information_groups")
        if not groups or any(
            not isinstance(group, str) or not group.strip() for group in groups
        ) or len(set(groups)) != len(groups):
            raise ValueError("known-truth information groups must be non-empty and unique")
        assignments = _known_truth_tuple(scenario.truth_assignments, label="truth_assignments")
        if len(assignments) != len(candidate_ids):
            raise ValueError("known-truth scenario must cover all 159 registry candidates")
        assignment_ids: list[str] = []
        for assignment in assignments:
            _validate_known_truth_assignment_v1(
                assignment,
                candidate_ids=candidate_id_set,
                information_groups=set(groups),
                scenario_role=scenario.truth_role,
                scenario_expression_id=expression_id,
            )
            assignment_ids.append(assignment.candidate_id)
        if set(assignment_ids) != candidate_id_set or len(set(assignment_ids)) != len(assignment_ids):
            raise ValueError("known-truth assignments must contain each registry candidate exactly once")
        role_set = {assignment.role for assignment in assignments}
        if scenario.truth_role == "direct_sparse" and "direct" not in role_set:
            raise ValueError("direct_sparse scenario must contain a direct candidate")
        if scenario.truth_role == "proxy_and_alias" and not {"proxy", "alias"}.issubset(role_set):
            raise ValueError("proxy_and_alias scenario must contain proxy and alias candidates")
        if scenario.truth_role == "rank_only" and "direct" not in role_set:
            raise ValueError("rank_only scenario must contain a predictive candidate")

        for assignment in assignments:
            if assignment.role == "direct":
                actual_effect_cases.add(
                    (
                        assignment.effect_scale_label,
                        assignment.effect_curve_id,
                        int(assignment.mirror_sign),
                    )
                )

        assignment_by_id = {assignment.candidate_id: assignment for assignment in assignments}
        direct_by_group: dict[str, list[KnownTruthSignalAssignmentV1]] = {}
        for assignment in assignments:
            if assignment.role == "direct":
                direct_by_group.setdefault(assignment.information_group, []).append(assignment)
        for assignment in assignments:
            if assignment.role == "alias":
                target = assignment_by_id.get(assignment.alias_of_candidate_id)
                if target is None or target.role != "direct":
                    raise ValueError("known-truth alias must point directly to a direct candidate")
                if target.information_group != assignment.information_group:
                    raise ValueError("known-truth alias and direct target must share information_group")
                if target.alias_of_candidate_id is not None:
                    raise ValueError("known-truth alias chain/cycle is not permitted")
                if target.base_signal_family != assignment.base_signal_family:
                    raise ValueError("known-truth alias must preserve the direct signal family")
                if target.base_random_stream_id != assignment.base_random_stream_id:
                    raise ValueError("known-truth alias must preserve the direct random stream")
            if assignment.role in {"proxy", "near_alias"}:
                direct_candidates = direct_by_group.get(assignment.information_group, [])
                if not direct_candidates:
                    raise ValueError("proxy and near_alias candidates require a direct candidate in their group")
                if any(
                    direct.base_signal_family != assignment.base_signal_family
                    or direct.base_random_stream_id != assignment.base_random_stream_id
                    for direct in direct_candidates
                ):
                    raise ValueError(
                        "proxy and near_alias candidates must share base family and random stream with their direct group"
                    )
            if assignment.role == "null":
                occupied_signal_streams = {
                    other_stream
                    for other in assignments
                    for other_stream in (
                        other.base_random_stream_id,
                        other.measurement_noise_stream_id,
                    )
                    if other_stream is not None
                }
                if assignment.null_noise_stream_id in occupied_signal_streams:
                    raise ValueError(
                        "null noise stream must be independent of base and measurement streams"
                    )
    if not expected_roles.issubset(seen_roles):
        raise ValueError("known-truth contract must include all four core scenario roles")
    return seen_ids, actual_effect_cases


def _validate_known_truth_tasks_v1(
    tasks: object,
    *,
    scenario_ids: set[str],
    development_namespace: str,
    formal_namespace: str,
) -> None:
    task_rows = _known_truth_tuple(tasks, label="tasks")
    if not task_rows:
        raise ValueError("known-truth contract must contain formal tasks")
    seen_task_ids: set[str] = set()
    seen_identities: set[tuple[object, ...]] = set()
    seen_seed_keys: set[tuple[str, str, int]] = set()
    formal_counts: dict[str, int] = {scenario_id: 0 for scenario_id in scenario_ids}
    formal_replicates: dict[str, set[int]] = {scenario_id: set() for scenario_id in scenario_ids}
    for task in task_rows:
        if not isinstance(task, KnownTruthTaskV1):
            raise TypeError("known-truth tasks must use the v1 data structure")
        task_id = _known_truth_text(task.task_id, label="task_id")
        if task_id in seen_task_ids:
            raise ValueError("known-truth task identity is duplicated")
        if task.scenario_id not in scenario_ids:
            raise ValueError("known-truth task references an unknown scenario")
        if task.phase not in KNOWN_TRUTH_SEED_PHASES_V1:
            raise ValueError("known-truth task phase must be development or formal")
        expected_namespace = (
            formal_namespace if task.phase == "formal" else development_namespace
        )
        if task.seed_namespace != expected_namespace:
            raise ValueError("known-truth task uses the wrong seed namespace")
        if not isinstance(task.replicate_id, Integral) or isinstance(task.replicate_id, bool):
            raise ValueError("known-truth replicate_id must be an integer")
        if not isinstance(task.seed, Integral) or isinstance(task.seed, bool):
            raise ValueError("known-truth seed must be an integer")
        seed_key = (task.phase, task.seed_namespace, int(task.seed))
        if seed_key in seen_seed_keys:
            raise ValueError("known-truth seed must be unique within its phase and namespace")
        identity = (
            task.scenario_id,
            task.phase,
            int(task.replicate_id),
            task.seed_namespace,
            int(task.seed),
        )
        if identity in seen_identities:
            raise ValueError("known-truth task identity tuple is duplicated")
        seen_task_ids.add(task_id)
        seen_identities.add(identity)
        seen_seed_keys.add(seed_key)
        if task.phase == "formal":
            if not 0 <= int(task.replicate_id) < KNOWN_TRUTH_FORMAL_REPLICATES_V1:
                raise ValueError("known-truth formal replicate_id is outside the closed range")
            formal_counts[task.scenario_id] += 1
            formal_replicates[task.scenario_id].add(int(task.replicate_id))
    for scenario_id, count in formal_counts.items():
        if count != KNOWN_TRUTH_FORMAL_REPLICATES_V1:
            raise ValueError(
                f"known-truth formal task count for {scenario_id} must be "
                f"{KNOWN_TRUTH_FORMAL_REPLICATES_V1}, got {count}"
            )
        if formal_replicates[scenario_id] != set(range(KNOWN_TRUTH_FORMAL_REPLICATES_V1)):
            raise ValueError("known-truth formal replicate set is not the closed 0..1099 set")


def validate_known_truth_simulation_contract_v1(
    contract: KnownTruthSimulationContractV1,
) -> KnownTruthSimulationContractV1:
    """Validate a frozen simulation contract without executing any pipeline.

    The validator is the sole public contract gate for this phase.  It checks
    only identities and frozen design invariants: the formal qlab registry and
    its 20-asset universe, four horizons, five shared absolute ``beta_total``
    scales, all curve/sign cases, explicit truth-role fields, per-scenario
    ``N=1100``, separated seed namespaces, no result-driven append, complete
    core truth roles, and exact lifecycle metadata.
    It deliberately does not generate data, estimate an effect, call L0--L4,
    or inspect any discovery result.
    """
    if not isinstance(contract, KnownTruthSimulationContractV1):
        raise TypeError("contract must be KnownTruthSimulationContractV1")
    _known_truth_text(contract.contract_id, label="contract_id")

    candidate_ids = _known_truth_tuple(
        contract.registry_candidate_ids,
        label="registry_candidate_ids",
    )
    if candidate_ids != KNOWN_TRUTH_REGISTRY_CANDIDATE_IDS_V1:
        raise ValueError(
            "known-truth registry candidate identities must match the formal qlab registry"
        )
    if _json_content_sha256(list(candidate_ids)) != KNOWN_TRUTH_REGISTRY_CANDIDATE_IDS_SHA256_V1:
        raise ValueError("known-truth formal qlab registry identity digest is not frozen")

    symbols = _known_truth_tuple(contract.admitted_symbols, label="admitted_symbols")
    if symbols != KNOWN_TRUTH_ADMITTED_SYMBOLS_V1:
        raise ValueError("known-truth universe symbols/order do not match the frozen universe")
    for label, actual, expected in (
        ("registry_identity", contract.registry_identity, KNOWN_TRUTH_REGISTRY_IDENTITY_V1),
        ("registry_source", contract.registry_source, KNOWN_TRUTH_REGISTRY_SOURCE_V1),
        (
            "registry_source_sha256",
            contract.registry_source_sha256,
            KNOWN_TRUTH_REGISTRY_SOURCE_SHA256_V1,
        ),
        (
            "candidate_identity_source",
            contract.candidate_identity_source,
            KNOWN_TRUTH_CANDIDATE_IDENTITY_SOURCE_V1,
        ),
        (
            "candidate_identity_source_sha256",
            contract.candidate_identity_source_sha256,
            KNOWN_TRUTH_CANDIDATE_IDENTITY_SOURCE_SHA256_V1,
        ),
        ("universe_identity", contract.universe_identity, KNOWN_TRUTH_UNIVERSE_IDENTITY_V1),
        ("universe_source", contract.universe_source, KNOWN_TRUTH_UNIVERSE_SOURCE_V1),
        (
            "universe_source_sha256",
            contract.universe_source_sha256,
            KNOWN_TRUTH_UNIVERSE_SOURCE_SHA256_V1,
        ),
    ):
        if _known_truth_text(actual, label=label) != expected:
            raise ValueError(f"known-truth {label} is not the frozen source identity")
    if (
        not isinstance(contract.registry_feature_count, Integral)
        or isinstance(contract.registry_feature_count, bool)
        or contract.registry_feature_count != KNOWN_TRUTH_REGISTRY_FEATURE_COUNT_V1
    ):
        raise ValueError("known-truth registry feature count is not the frozen 68")
    candidate_horizon_counts = _known_truth_tuple(
        contract.candidate_horizon_counts,
        label="candidate_horizon_counts",
    )
    if (
        len(candidate_horizon_counts) != len(KNOWN_TRUTH_CANDIDATE_HORIZON_COUNTS_V1)
        or any(
            not isinstance(row, tuple)
            or len(row) != 2
            or not isinstance(row[0], str)
            or not isinstance(row[1], Integral)
            or isinstance(row[1], bool)
            for row in candidate_horizon_counts
        )
        or candidate_horizon_counts != KNOWN_TRUTH_CANDIDATE_HORIZON_COUNTS_V1
    ):
        raise ValueError("known-truth candidate horizon counts must be 23/23/45/68")
    horizons = _known_truth_tuple(contract.horizons, label="horizons")
    if horizons != KNOWN_TRUTH_HORIZONS_V1:
        raise ValueError("known-truth contract horizons must be 4h, 8h, 12h, 1d")
    _validate_known_truth_scales_v1(contract.beta_total_scales)
    effect_curve_ids = _known_truth_tuple(
        contract.effect_curve_ids,
        label="effect_curve_ids",
    )
    if (
        len(effect_curve_ids) != len(KNOWN_TRUTH_EFFECT_CURVES_V1)
        or any(not isinstance(curve_id, str) for curve_id in effect_curve_ids)
        or effect_curve_ids != KNOWN_TRUTH_EFFECT_CURVES_V1
    ):
        raise ValueError("known-truth contract must contain fast, delayed, persistent curves")
    mirror_signs = _known_truth_tuple(contract.mirror_signs, label="mirror_signs")
    if (
        len(mirror_signs) != len(KNOWN_TRUTH_MIRROR_SIGNS_V1)
        or any(
            not isinstance(sign, Integral) or isinstance(sign, bool)
            for sign in mirror_signs
        )
        or mirror_signs != KNOWN_TRUTH_MIRROR_SIGNS_V1
    ):
        raise ValueError("known-truth contract must contain both negative and positive mirrors")
    effect_case_coverage = _known_truth_tuple(
        contract.effect_case_coverage,
        label="effect_case_coverage",
    )
    if (
        len(effect_case_coverage) != len(KNOWN_TRUTH_EFFECT_CASE_COVERAGE_V1)
        or any(
            not isinstance(row, tuple)
            or len(row) != 3
            or not isinstance(row[0], str)
            or not isinstance(row[1], str)
            or not isinstance(row[2], Integral)
            or isinstance(row[2], bool)
            for row in effect_case_coverage
        )
        or effect_case_coverage != KNOWN_TRUTH_EFFECT_CASE_COVERAGE_V1
    ):
        raise ValueError("known-truth effect coverage must contain all five-by-three-by-two cases")
    if (
        not isinstance(contract.formal_replicates, Integral)
        or isinstance(contract.formal_replicates, bool)
        or contract.formal_replicates != KNOWN_TRUTH_FORMAL_REPLICATES_V1
    ):
        raise ValueError("known-truth formal replicate count must be exactly 1100")
    development_namespace = _known_truth_text(
        contract.development_seed_namespace,
        label="development_seed_namespace",
    )
    formal_namespace = _known_truth_text(
        contract.formal_seed_namespace,
        label="formal_seed_namespace",
    )
    if development_namespace == formal_namespace:
        raise ValueError("known-truth development and formal seed namespaces must differ")
    if contract.allow_adaptive_append is not False:
        raise ValueError("known-truth contract must forbid result-driven append")
    if contract.append_policy != KNOWN_TRUTH_APPEND_POLICY_V1:
        raise ValueError("known-truth append policy is not the frozen stop-and-report rule")
    if contract.reality_analysis_scope != KNOWN_TRUTH_REALITY_SCOPE_V1:
        raise ValueError("reality analysis must remain independent per signal and horizon")
    if contract.simulation_effect_scope != KNOWN_TRUTH_SIMULATION_EFFECT_SCOPE_V1:
        raise ValueError("simulation effect scope is not the frozen shared-beta rule")

    scenario_ids, actual_effect_cases = _validate_known_truth_scenarios_v1(
        contract.scenarios,
        candidate_ids=tuple(str(candidate_id) for candidate_id in candidate_ids),
    )
    if actual_effect_cases != set(KNOWN_TRUTH_EFFECT_CASE_COVERAGE_V1):
        raise ValueError(
            "known-truth effect coverage must be realized by direct truth assignments"
        )
    _validate_known_truth_tasks_v1(
        contract.tasks,
        scenario_ids=scenario_ids,
        development_namespace=development_namespace,
        formal_namespace=formal_namespace,
    )

    if _known_truth_text(contract.lifecycle, label="lifecycle") != KNOWN_TRUTH_LIFECYCLE_V1:
        raise ValueError("known-truth lifecycle is not the frozen candidate-contract lifecycle")
    if _known_truth_text(contract.authority, label="authority") != KNOWN_TRUTH_FORMAL_AUTHORITY_V1:
        raise ValueError("known-truth authority is not the frozen Issue #34/#36 authority")
    inputs = _known_truth_tuple(contract.inputs, label="inputs")
    if inputs != KNOWN_TRUTH_INPUTS_V1:
        raise ValueError("known-truth contract inputs are incomplete or reordered")
    if _known_truth_text(contract.may_be_used_for, label="may_be_used_for") != KNOWN_TRUTH_MAY_BE_USED_FOR_V1:
        raise ValueError("known-truth may_be_used_for is not the frozen boundary")
    if _known_truth_text(contract.must_not_be_used_for, label="must_not_be_used_for") != KNOWN_TRUTH_MUST_NOT_BE_USED_FOR_V1:
        raise ValueError("known-truth must_not_be_used_for is not the frozen boundary")
    if _known_truth_text(contract.archive_condition, label="archive_condition") != KNOWN_TRUTH_ARCHIVE_CONDITION_V1:
        raise ValueError("known-truth archive_condition is not the frozen boundary")
    return contract


__all__ = [
    "DecisionWindow",
    "DETERMINISTIC_RANDOM_ADDRESS_VERSION_V1",
    "DETERMINISTIC_RANDOM_ALLOWED_PHASES_V1",
    "DETERMINISTIC_RANDOM_ALLOWED_STREAM_KINDS_V1",
    "DeterministicRandomAddressV1",
    "RANDOM_STREAM_CALIBRATION_STATUSES_V1",
    "RANDOM_STREAM_SPECIFICATION_ARCHIVE_CONDITION_V1",
    "RANDOM_STREAM_SPECIFICATION_AUTHORITY_V1",
    "RANDOM_STREAM_SPECIFICATION_LIFECYCLE_V1",
    "RANDOM_STREAM_SPECIFICATION_MAY_BE_USED_FOR_V1",
    "RANDOM_STREAM_SPECIFICATION_MUST_NOT_BE_USED_FOR_V1",
    "RANDOM_STREAM_SPECIFICATION_VERSION_V1",
    "KNOWN_TRUTH_ADMITTED_SYMBOLS_V1",
    "KNOWN_TRUTH_ARCHIVE_CONDITION_V1",
    "KNOWN_TRUTH_BETA_TOTAL_SCALES_V1",
    "KNOWN_TRUTH_CANDIDATE_HORIZON_COUNTS_V1",
    "KNOWN_TRUTH_CANDIDATE_IDENTITY_SOURCE_SHA256_V1",
    "KNOWN_TRUTH_CANDIDATE_IDENTITY_SOURCE_V1",
    "KNOWN_TRUTH_CORE_SCENARIO_ROLES_V1",
    "KNOWN_TRUTH_EFFECT_CURVES_V1",
    "KNOWN_TRUTH_EFFECT_CASE_COVERAGE_V1",
    "KNOWN_TRUTH_FORMAL_AUTHORITY_V1",
    "KNOWN_TRUTH_FORMAL_REPLICATES_V1",
    "KNOWN_TRUTH_HORIZONS_V1",
    "KNOWN_TRUTH_INPUTS_V1",
    "KNOWN_TRUTH_LIFECYCLE_V1",
    "KNOWN_TRUTH_MAY_BE_USED_FOR_V1",
    "KNOWN_TRUTH_MUST_NOT_BE_USED_FOR_V1",
    "KNOWN_TRUTH_MIRROR_SIGNS_V1",
    "KNOWN_TRUTH_NULL_EXPRESSION_V1",
    "KNOWN_TRUTH_RANK_ONLY_EXPRESSION_V1",
    "KNOWN_TRUTH_REGISTRY_CANDIDATE_IDS_SHA256_V1",
    "KNOWN_TRUTH_REGISTRY_CANDIDATE_IDS_V1",
    "KNOWN_TRUTH_REGISTRY_FEATURE_COUNT_V1",
    "KNOWN_TRUTH_REGISTRY_SOURCE_SHA256_V1",
    "KNOWN_TRUTH_REGISTRY_SOURCE_V1",
    "KNOWN_TRUTH_SCALAR_EXPRESSION_V1",
    "KNOWN_TRUTH_SEED_PHASES_V1",
    "KNOWN_TRUTH_SIMULATION_EFFECT_SCOPE_V1",
    "KNOWN_TRUTH_UNIVERSE_IDENTITY_V1",
    "KNOWN_TRUTH_UNIVERSE_SOURCE_SHA256_V1",
    "KNOWN_TRUTH_UNIVERSE_SOURCE_V1",
    "KnownTruthScenarioV1",
    "KnownTruthSignalAssignmentV1",
    "KnownTruthSimulationContractV1",
    "KnownTruthTaskV1",
    "RandomInformationGroupStreamBindingV1",
    "RandomStreamSpecificationRegistryV1",
    "RandomStreamSpecificationV1",
    "RandomTimeProcessSpecificationV1",
    "ObservedEffectCandidate",
    "ObservedEffectBetaTotalArtifacts",
    "ObservedEffectScaleContract",
    "ObservedEffectScaleArtifacts",
    "ObservedEffectScaleInput",
    "derive_deterministic_random_address_v1",
    "estimate_l0_l4_observed_effect_scale_v1",
    "map_observed_effect_scale_to_beta_total_v1",
    "validate_random_stream_specification_v1",
    "validate_known_truth_simulation_contract_v1",
]
