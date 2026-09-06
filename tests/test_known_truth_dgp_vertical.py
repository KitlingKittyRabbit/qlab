from dataclasses import replace
import inspect
import math
import pickle

import numpy as np
import pandas as pd
import pytest

from qlab.data.crypto import keystore_coinglass_factors as factor_registry

from qlab.full_pipeline_simulation import (
    DETERMINISTIC_RANDOM_ADDRESS_VERSION_V1,
    KNOWN_TRUTH_ADMITTED_SYMBOLS_V1,
    KNOWN_TRUTH_BETA_TOTAL_SCALES_V1,
    KNOWN_TRUTH_DGP_MARKET_SOURCE_V1,
    KNOWN_TRUTH_DGP_RANK_STANDARDIZATION_V1,
    KNOWN_TRUTH_DGP_RANK_CURVE_DISCRETIZATION_V1,
    KNOWN_TRUTH_DGP_RANK_CURVE_FAMILY_V1,
    KNOWN_TRUTH_DGP_RANK_CURVE_NORMALIZATION_V1,
    KNOWN_TRUTH_DGP_RANK_CURVE_TAIL_RULE_V1,
    KNOWN_TRUTH_DGP_SIGNAL_RECORD_COLUMNS_V1,
    KNOWN_TRUTH_DGP_STANDARDIZATION_V1,
    KNOWN_TRUTH_DGP_VALUE_STATUS_V1,
    KNOWN_TRUTH_DGP_VARIANT_AVAILABILITY_RULE_V1,
    KNOWN_TRUTH_DGP_VARIANT_LABEL_ALIGNMENT_V1,
    KNOWN_TRUTH_DGP_VARIANT_OPERATOR_IDENTITY_V1,
    KNOWN_TRUTH_DGP_VARIANT_TIMEFRAME_V1,
    KNOWN_TRUTH_DGP_VERTICAL_ARCHIVE_CONDITION_V1,
    KNOWN_TRUTH_DGP_VERTICAL_AUTHORITY_V1,
    KNOWN_TRUTH_DGP_VERTICAL_LIFECYCLE_V1,
    KNOWN_TRUTH_DGP_VERTICAL_MAY_BE_USED_FOR_V1,
    KNOWN_TRUTH_DGP_VERTICAL_MUST_NOT_BE_USED_FOR_V1,
    KNOWN_TRUTH_EFFECT_CURVES_V1,
    KNOWN_TRUTH_HORIZONS_V1,
    KNOWN_TRUTH_NULL_EXPRESSION_V1,
    KNOWN_TRUTH_RANK_ONLY_EXPRESSION_V1,
    KNOWN_TRUTH_REGISTRY_CANDIDATE_IDS_V1,
    KNOWN_TRUTH_SCALAR_EXPRESSION_V1,
    MARKET_INPUT_SPINE_ARCHIVE_CONDITION_V1,
    MARKET_INPUT_SPINE_AUTHORITY_V1,
    MARKET_INPUT_SPINE_INNOVATION_DISTRIBUTION_V1,
    MARKET_INPUT_SPINE_LIFECYCLE_V1,
    MARKET_INPUT_SPINE_MAY_BE_USED_FOR_V1,
    MARKET_INPUT_SPINE_MUST_NOT_BE_USED_FOR_V1,
    MARKET_INPUT_SPINE_NATIVE_FREQUENCY_V1,
    MARKET_INPUT_SPINE_RECORD_COLUMNS_V1,
    MARKET_INPUT_SPINE_SCHEMA_V1,
    MARKET_INPUT_SPINE_TIME_ORDER_V1,
    RANDOM_STREAM_SPECIFICATION_ARCHIVE_CONDITION_V1,
    RANDOM_STREAM_SPECIFICATION_AUTHORITY_V1,
    RANDOM_STREAM_SPECIFICATION_LIFECYCLE_V1,
    RANDOM_STREAM_SPECIFICATION_MAY_BE_USED_FOR_V1,
    RANDOM_STREAM_SPECIFICATION_MUST_NOT_BE_USED_FOR_V1,
    RANDOM_STREAM_SPECIFICATION_VERSION_V1,
    KnownTruthDgpEffectCurveV1,
    KnownTruthDgpFaultInjectionV1,
    KnownTruthDgpObservationVariantV1,
    KnownTruthDgpRankEffectCurveV1,
    KnownTruthDgpSignalStreamV1,
    KnownTruthDgpVerticalSpecificationV1,
    KnownTruthScenarioV1,
    KnownTruthSignalAssignmentV1,
    MarketInputSpineSpecificationV1,
    MarketInputSpineStreamV1,
    RandomInformationGroupStreamBindingV1,
    RandomStreamSpecificationRegistryV1,
    RandomStreamSpecificationV1,
    RandomTimeProcessSpecificationV1,
    _known_truth_dgp_curve_cdf_v1,
    _known_truth_dgp_effect_weight_v1,
    _known_truth_dgp_rank_effect_weight_v1,
    generate_known_truth_dgp_vertical_slice_v1,
    KnownTruthL0L4RawInputV1,
    KnownTruthL0L4PipelineDiscoveryArtifactsV1,
    KnownTruthL0L4TruthBlindEvaluationInputV1,
    KNOWN_TRUTH_L0_L4_TRUTH_BLIND_INPUT_SCHEMA_V1,
    KNOWN_TRUTH_L0_L4_TRUTH_BLIND_INPUT_LIFECYCLE_V1,
    KNOWN_TRUTH_L0_L4_TRUTH_BLIND_INPUT_AUTHORITY_V1,
    KNOWN_TRUTH_L4_ACTIVATION_COLUMNS_V1,
    KNOWN_TRUTH_PIPELINE_DISCOVERY_COLUMNS_V1,
    bind_known_truth_l0_l4_truth_blind_evaluation_input_v1,
    evaluate_known_truth_pipeline_terminal_v1,
    run_known_truth_l0_l4_pipeline_discovery_micro_e2e_v1,
    run_known_truth_l0_l4_micro_e2e_v1,
    validate_known_truth_dgp_vertical_specification_v1,
)
from qlab.walkforward import WalkForwardFold

ASSETS = KNOWN_TRUTH_ADMITTED_SYMBOLS_V1
FIXTURE_ASSETS = ASSETS[:2]
CANDIDATES = KNOWN_TRUTH_REGISTRY_CANDIDATE_IDS_V1
START = pd.Timestamp("2026-01-01T00:00:00Z")
R_CORRELATED = ((1.0, 0.5), (0.5, 1.0))
L_CORRELATED = ((1.0, 0.0), (0.5, math.sqrt(0.75)))


BASE_PROCESS = RandomTimeProcessSpecificationV1(
    family_id="fixture-base-iid-v1",
    parameter_identity="fixture-base-iid-parameters-v1",
    initialization_identity="fixture-base-zero-init-v1",
    burn_in_steps=1,
    native_frequency=MARKET_INPUT_SPINE_NATIVE_FREQUENCY_V1,
    time_order_identity=MARKET_INPUT_SPINE_TIME_ORDER_V1,
    calibration_identity="fixture-base-calibration-unavailable-v1",
    calibration_status="unavailable",
    tail_rule_identity="fixture-base-finite-tail-v1",
)
MEASUREMENT_PROCESS = replace(
    BASE_PROCESS,
    family_id="fixture-measurement-iid-v1",
    parameter_identity="fixture-measurement-iid-parameters-v1",
    initialization_identity="fixture-measurement-zero-init-v1",
    calibration_identity="fixture-measurement-calibration-unavailable-v1",
    tail_rule_identity="fixture-measurement-finite-tail-v1",
)
NULL_PROCESS = replace(
    BASE_PROCESS,
    family_id="fixture-null-iid-v1",
    parameter_identity="fixture-null-iid-parameters-v1",
    initialization_identity="fixture-null-zero-init-v1",
    calibration_identity="fixture-null-calibration-unavailable-v1",
    tail_rule_identity="fixture-null-finite-tail-v1",
)
PRICE_PROCESS = replace(
    BASE_PROCESS,
    family_id="fixture-price-iid-v1",
    parameter_identity="fixture-price-iid-parameters-v1",
    initialization_identity="fixture-price-zero-init-v1",
    calibration_identity="fixture-price-calibration-unavailable-v1",
    tail_rule_identity="fixture-price-finite-tail-v1",
)
VOLUME_PROCESS = replace(
    BASE_PROCESS,
    family_id="fixture-volume-ar1-v1",
    parameter_identity="fixture-volume-ar1-phi-half-v1",
    initialization_identity="fixture-volume-zero-init-v1",
    calibration_identity="fixture-volume-calibration-unavailable-v1",
    tail_rule_identity="fixture-volume-finite-tail-v1",
)


def _identity_matrix(size: int) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(index == other) for other in range(size))
        for index in range(size)
    )


def _registered_stream(
    stream_id: str,
    stream_kind: str,
    process: RandomTimeProcessSpecificationV1,
    *,
    information_group_id: str | None = None,
) -> RandomStreamSpecificationV1:
    return RandomStreamSpecificationV1(
        stream_id=stream_id,
        stream_kind=stream_kind,
        information_group_id=information_group_id,
        random_address_group_id=f"{stream_id}-address-v1",
        innovation_distribution_id=MARKET_INPUT_SPINE_INNOVATION_DISTRIBUTION_V1,
        asset_symbols=ASSETS,
        r_identity=f"{stream_id}-r-v1",
        r_decomposition_identity=f"{stream_id}-decomposition-v1",
        r_calibration_identity=f"{stream_id}-r-calibration-v1",
        time_process=process,
    )


def _registry() -> RandomStreamSpecificationRegistryV1:
    return RandomStreamSpecificationRegistryV1(
        specification_version=RANDOM_STREAM_SPECIFICATION_VERSION_V1,
        seed_namespace="issue34-known-truth-dgp-fixture-v1",
        phase="formal",
        address_derivation_version=DETERMINISTIC_RANDOM_ADDRESS_VERSION_V1,
        asset_symbols=ASSETS,
        streams=(
            _registered_stream("base-alpha", "base", BASE_PROCESS, information_group_id="alpha"),
            _registered_stream(
                "measurement-alpha",
                "measurement",
                MEASUREMENT_PROCESS,
                information_group_id="alpha",
            ),
            _registered_stream("null-main", "null", NULL_PROCESS),
            _registered_stream("price-main", "price", PRICE_PROCESS),
            _registered_stream("volume-main", "volume", VOLUME_PROCESS),
        ),
        information_group_bindings=(
            RandomInformationGroupStreamBindingV1(
                information_group_id="alpha",
                base_stream_id="base-alpha",
                proxy_near_alias_base_stream_id="base-alpha",
                measurement_stream_ids=("measurement-alpha",),
            ),
        ),
        lifecycle=RANDOM_STREAM_SPECIFICATION_LIFECYCLE_V1,
        authority=RANDOM_STREAM_SPECIFICATION_AUTHORITY_V1,
        may_be_used_for=RANDOM_STREAM_SPECIFICATION_MAY_BE_USED_FOR_V1,
        must_not_be_used_for=RANDOM_STREAM_SPECIFICATION_MUST_NOT_BE_USED_FOR_V1,
        archive_condition=RANDOM_STREAM_SPECIFICATION_ARCHIVE_CONDITION_V1,
    )


def _market_stream(
    stream_id: str,
    process: RandomTimeProcessSpecificationV1,
    assets: tuple[str, ...],
    *,
    level_values: tuple[float, ...],
    state_scale: float,
    correlation_matrix: tuple[tuple[float, ...], ...],
    correlation_decomposition: tuple[tuple[float, ...], ...],
) -> MarketInputSpineStreamV1:
    return MarketInputSpineStreamV1(
        stream_id=stream_id,
        process_family="ar1" if stream_id == "volume-main" else "iid",
        process_family_id=process.family_id,
        parameter_identity=process.parameter_identity,
        initialization_identity=process.initialization_identity,
        process_parameters=(0.5,) if stream_id == "volume-main" else (),
        initial_state=(0.0,) * len(assets),
        level_values=level_values,
        state_scale=state_scale,
        correlation_matrix=correlation_matrix,
        correlation_decomposition=correlation_decomposition,
    )


def _assignment_null(candidate_id: str) -> KnownTruthSignalAssignmentV1:
    return KnownTruthSignalAssignmentV1(
        candidate_id=candidate_id,
        information_group=None,
        base_signal_family=None,
        role="null",
        null_noise_stream_id="null-main",
        standardization_id=KNOWN_TRUTH_DGP_STANDARDIZATION_V1,
        role_standardization_center=0.0,
        role_standardization_scale=1.0,
        expression_type=KNOWN_TRUTH_NULL_EXPRESSION_V1,
        analytic_truth_proof="independent null stream is disconnected from return path",
        noise_scale=1.0,
        return_inclusion=False,
        marginal_predictive_truth=0,
    )


def _assignment_direct(
    candidate_id: str,
    *,
    expression_type: str = KNOWN_TRUTH_SCALAR_EXPRESSION_V1,
    standardization_id: str = KNOWN_TRUTH_DGP_STANDARDIZATION_V1,
    mirror_sign: int = 1,
) -> KnownTruthSignalAssignmentV1:
    values: dict[str, object] = {
        "candidate_id": candidate_id,
        "information_group": "alpha",
        "base_signal_family": "fixture-base-family-v1",
        "role": "direct",
        "base_random_stream_id": "base-alpha",
        "observation_variant_id": "variant-direct",
        "standardization_id": standardization_id,
        "expression_type": expression_type,
        "direction": mirror_sign,
        "effect_scale_label": "weak",
        "effect_curve_id": "fast",
        "mirror_sign": mirror_sign,
        "beta_id": "beta-weak-v1",
        "analytic_truth_proof": "direct signal is the only return-generating candidate",
        "return_inclusion": True,
        "marginal_predictive_truth": 1,
    }
    if expression_type == KNOWN_TRUTH_RANK_ONLY_EXPRESSION_V1:
        values.update(
            beta_rank=dict(KNOWN_TRUTH_BETA_TOTAL_SCALES_V1)["weak"],
            effect_curve_id=None,
            w_rank=RANK_CURVE.curve_id,
        )
    else:
        values.update(
            w_effect_id="fast",
            beta_total=dict(KNOWN_TRUTH_BETA_TOTAL_SCALES_V1)["weak"],
            role_standardization_center=0.0,
            role_standardization_scale=1.0,
        )
    return KnownTruthSignalAssignmentV1(**values)


def _assignment_proxy(candidate_id: str) -> KnownTruthSignalAssignmentV1:
    return KnownTruthSignalAssignmentV1(
        candidate_id=candidate_id,
        information_group="alpha",
        base_signal_family="fixture-base-family-v1",
        role="proxy",
        base_random_stream_id="base-alpha",
        observation_variant_id="variant-proxy",
        measurement_noise_stream_id="measurement-alpha",
        standardization_id=KNOWN_TRUTH_DGP_STANDARDIZATION_V1,
        role_standardization_center=0.0,
        role_standardization_scale=1.0,
        expression_type=KNOWN_TRUTH_SCALAR_EXPRESSION_V1,
        direction=1,
        analytic_truth_proof="proxy observes the direct base through registered measurement noise",
        rho=0.5,
        noise_scale=1.0,
        return_inclusion=False,
        marginal_predictive_truth=1,
    )


def _assignment_alias(candidate_id: str, target_id: str) -> KnownTruthSignalAssignmentV1:
    return KnownTruthSignalAssignmentV1(
        candidate_id=candidate_id,
        information_group="alpha",
        base_signal_family="fixture-base-family-v1",
        role="alias",
        base_random_stream_id="base-alpha",
        alias_of_candidate_id=target_id,
        observation_variant_id="variant-alias",
        standardization_id=KNOWN_TRUTH_DGP_STANDARDIZATION_V1,
        role_standardization_center=0.0,
        role_standardization_scale=1.0,
        expression_type=KNOWN_TRUTH_SCALAR_EXPRESSION_V1,
        direction=1,
        analytic_truth_proof="exact alias is a deterministic identity of the direct candidate",
        return_inclusion=False,
        marginal_predictive_truth=1,
    )


def _assignment_near_alias(candidate_id: str) -> KnownTruthSignalAssignmentV1:
    return KnownTruthSignalAssignmentV1(
        candidate_id=candidate_id,
        information_group="alpha",
        base_signal_family="fixture-base-family-v1",
        role="near_alias",
        base_random_stream_id="base-alpha",
        observation_variant_id="variant-near",
        measurement_noise_stream_id="measurement-alpha",
        standardization_id=KNOWN_TRUTH_DGP_STANDARDIZATION_V1,
        role_standardization_center=0.0,
        role_standardization_scale=1.0,
        expression_type=KNOWN_TRUTH_SCALAR_EXPRESSION_V1,
        direction=1,
        analytic_truth_proof="near alias is a noisy proxy with a separately registered rho",
        rho=0.25,
        noise_scale=1.0,
        return_inclusion=False,
        marginal_predictive_truth=1,
    )


def _scenario_scalar() -> KnownTruthScenarioV1:
    assignments = [_assignment_direct(CANDIDATES[0]), _assignment_proxy(CANDIDATES[1])]
    assignments.extend([
        _assignment_alias(CANDIDATES[2], CANDIDATES[0]),
        _assignment_near_alias(CANDIDATES[3]),
    ])
    assignments.extend(_assignment_null(candidate_id) for candidate_id in CANDIDATES[4:])
    return KnownTruthScenarioV1(
        scenario_id="fixture-scalar-roles-v1",
        truth_role="direct_sparse",
        information_groups=("alpha",),
        expression_id=KNOWN_TRUTH_SCALAR_EXPRESSION_V1,
        truth_assignments=tuple(assignments),
    )


def _scenario_rank_only() -> KnownTruthScenarioV1:
    assignments = [_assignment_direct(
        CANDIDATES[0],
        expression_type=KNOWN_TRUTH_RANK_ONLY_EXPRESSION_V1,
        standardization_id=KNOWN_TRUTH_DGP_RANK_STANDARDIZATION_V1,
    )]
    assignments.extend(_assignment_null(candidate_id) for candidate_id in CANDIDATES[1:])
    return KnownTruthScenarioV1(
        scenario_id="fixture-rank-only-v1",
        truth_role="rank_only",
        information_groups=("alpha",),
        expression_id=KNOWN_TRUTH_RANK_ONLY_EXPRESSION_V1,
        truth_assignments=tuple(assignments),
    )


def _scenario_pipeline_discovery() -> KnownTruthScenarioV1:
    """One true information group, an exact alias, and registered nulls."""
    direct_id = CANDIDATES[67]  # 1h signal, 4h discovery unit
    alias_id = CANDIDATES[71]   # another 1h signal, same true base
    assignments = []
    for candidate_id in CANDIDATES:
        if candidate_id == direct_id:
            assignments.append(_assignment_direct(candidate_id))
        elif candidate_id == alias_id:
            assignments.append(_assignment_alias(candidate_id, direct_id))
        else:
            assignments.append(_assignment_null(candidate_id))
    return KnownTruthScenarioV1(
        scenario_id="fixture-pipeline-discovery-v1",
        truth_role="direct_sparse",
        information_groups=("alpha",),
        expression_id=KNOWN_TRUTH_SCALAR_EXPRESSION_V1,
        truth_assignments=tuple(assignments),
    )


def _signal_stream(
    stream_id: str,
    process: RandomTimeProcessSpecificationV1,
    size: int,
    *,
    correlation_matrix: tuple[tuple[float, ...], ...],
    correlation_decomposition: tuple[tuple[float, ...], ...],
) -> KnownTruthDgpSignalStreamV1:
    return KnownTruthDgpSignalStreamV1(
        stream_id=stream_id,
        process_family="iid",
        process_family_id=process.family_id,
        parameter_identity=process.parameter_identity,
        initialization_identity=process.initialization_identity,
        process_parameters=(),
        initial_state=(0.0,) * size,
        standardization_id=KNOWN_TRUTH_DGP_STANDARDIZATION_V1,
        standardization_center=(0.0,) * size,
        standardization_scale=(1.0,) * size,
        correlation_matrix=correlation_matrix,
        correlation_decomposition=correlation_decomposition,
    )


def _variant(variant_id: str, role: str, input_type: str, input_key: str) -> KnownTruthDgpObservationVariantV1:
    return KnownTruthDgpObservationVariantV1(
        variant_id=variant_id,
        role=role,
        input_type=input_type,
        input_key=input_key,
        output_timeframe=KNOWN_TRUTH_DGP_VARIANT_TIMEFRAME_V1,
        operator_id=KNOWN_TRUTH_DGP_VARIANT_OPERATOR_IDENTITY_V1,
        input_window=(0,),
        weights=(1.0,),
        label_alignment=KNOWN_TRUTH_DGP_VARIANT_LABEL_ALIGNMENT_V1,
        availability_rule=KNOWN_TRUTH_DGP_VARIANT_AVAILABILITY_RULE_V1,
    )


RANK_CURVE = KnownTruthDgpRankEffectCurveV1(
    curve_id="rank-exp-fixture-v1",
    family_id=KNOWN_TRUTH_DGP_RANK_CURVE_FAMILY_V1,
    lambda_rank_minutes=2.0,
    epsilon_rank=1e-12,
    discretization_identity=KNOWN_TRUTH_DGP_RANK_CURVE_DISCRETIZATION_V1,
    normalization_identity=KNOWN_TRUTH_DGP_RANK_CURVE_NORMALIZATION_V1,
    tail_rule_identity=KNOWN_TRUTH_DGP_RANK_CURVE_TAIL_RULE_V1,
)


def _market_spec(
    assets: tuple[str, ...],
    *,
    processing_order: tuple[int, ...] | None = None,
    periods: int = 5,
) -> MarketInputSpineSpecificationV1:
    size = len(assets)
    identity = _identity_matrix(size)
    price_r, price_l = (R_CORRELATED, L_CORRELATED) if size == 2 else (identity, identity)
    return MarketInputSpineSpecificationV1(
        schema_version=MARKET_INPUT_SPINE_SCHEMA_V1,
        generation_batch=f"fixture-known-truth-dgp-v1-{size}",
        seed_namespace="issue34-known-truth-dgp-fixture-v1",
        phase="formal",
        asset_symbols=assets,
        start_time=START,
        periods=periods,
        asset_processing_order=processing_order or tuple(range(size)),
        price=_market_stream(
            "price-main",
            PRICE_PROCESS,
            assets,
            level_values=tuple(100.0 + 100.0 * index for index in range(size)),
            state_scale=0.01,
            correlation_matrix=price_r,
            correlation_decomposition=price_l,
        ),
        volume=_market_stream(
            "volume-main",
            VOLUME_PROCESS,
            assets,
            level_values=tuple(10.0 + index for index in range(size)),
            state_scale=0.1,
            correlation_matrix=identity,
            correlation_decomposition=identity,
        ),
        lifecycle=MARKET_INPUT_SPINE_LIFECYCLE_V1,
        authority=MARKET_INPUT_SPINE_AUTHORITY_V1,
        may_be_used_for=MARKET_INPUT_SPINE_MAY_BE_USED_FOR_V1,
        must_not_be_used_for=MARKET_INPUT_SPINE_MUST_NOT_BE_USED_FOR_V1,
        archive_condition=MARKET_INPUT_SPINE_ARCHIVE_CONDITION_V1,
    )


def _spec(
    *,
    assets: tuple[str, ...] = FIXTURE_ASSETS,
    scenario: KnownTruthScenarioV1 | None = None,
    faults: tuple[KnownTruthDgpFaultInjectionV1, ...] = (),
    processing_order: tuple[int, ...] | None = None,
    periods: int = 5,
) -> KnownTruthDgpVerticalSpecificationV1:
    size = len(assets)
    identity = _identity_matrix(size)
    base_r, base_l = (R_CORRELATED, L_CORRELATED) if size == 2 else (identity, identity)
    market = _market_spec(assets, processing_order=processing_order, periods=periods)
    scenario_value = scenario or _scenario_scalar()
    roles = {assignment.role for assignment in scenario_value.truth_assignments}
    signal_streams = [
        _signal_stream("base-alpha", BASE_PROCESS, size, correlation_matrix=base_r, correlation_decomposition=base_l),
        _signal_stream("null-main", NULL_PROCESS, size, correlation_matrix=identity, correlation_decomposition=identity),
    ]
    if roles.intersection({"proxy", "near_alias"}):
        signal_streams.insert(
            1,
            _signal_stream("measurement-alpha", MEASUREMENT_PROCESS, size, correlation_matrix=identity, correlation_decomposition=identity),
        )
    variants = [_variant("variant-direct", "direct", "base_signal", "alpha")]
    if "proxy" in roles:
        variants.append(_variant("variant-proxy", "proxy", "base_signal", "alpha"))
    if "near_alias" in roles:
        variants.append(_variant("variant-near", "near_alias", "base_signal", "alpha"))
    if "alias" in roles:
        alias_target = next(
            assignment.alias_of_candidate_id
            for assignment in scenario_value.truth_assignments
            if assignment.role == "alias"
        )
        variants.append(_variant("variant-alias", "alias", "direct_candidate", alias_target))
    curves = (
        KnownTruthDgpEffectCurveV1(
            "fast", "exponential_cdf_v1", (-4.0 / math.log(0.20),),
            "adjacent_cdf_difference_infinite_mass_v1",
        ),
        KnownTruthDgpEffectCurveV1(
            "delayed", "gamma_shape_3_cdf_v1", (4.0,),
            "adjacent_cdf_difference_infinite_mass_v1",
        ),
        KnownTruthDgpEffectCurveV1(
            "persistent", "exponential_cdf_v1", (12.0 / math.log(2.0),),
            "adjacent_cdf_difference_infinite_mass_v1",
        ),
    )
    rank_curves = (
        RANK_CURVE,
    ) if scenario_value.expression_id == KNOWN_TRUTH_RANK_ONLY_EXPRESSION_V1 else ()
    return KnownTruthDgpVerticalSpecificationV1(
        schema_version="ksv4-known-truth-dgp-vertical-slice/v1",
        generation_batch=market.generation_batch,
        market=market,
        scenario=scenario_value,
        signal_streams=tuple(signal_streams),
        observation_variants=tuple(variants),
        effect_curves=curves,
        rank_effect_curves=rank_curves,
        execution_delay_minutes=4,
        faults=faults,
        lifecycle=KNOWN_TRUTH_DGP_VERTICAL_LIFECYCLE_V1,
        authority=KNOWN_TRUTH_DGP_VERTICAL_AUTHORITY_V1,
        may_be_used_for=KNOWN_TRUTH_DGP_VERTICAL_MAY_BE_USED_FOR_V1,
        must_not_be_used_for=KNOWN_TRUTH_DGP_VERTICAL_MUST_NOT_BE_USED_FOR_V1,
        archive_condition=KNOWN_TRUTH_DGP_VERTICAL_ARCHIVE_CONDITION_V1,
    )


def _injected(size: int = 2, periods: int = 5) -> dict[str, np.ndarray]:
    rows = periods + 1
    price = np.zeros((rows, size), dtype=float)
    volume = np.zeros((rows, size), dtype=float)
    base = np.zeros((rows, size), dtype=float)
    measurement = np.zeros((rows, size), dtype=float)
    null = np.zeros((rows, size), dtype=float)
    if size == 2:
        base[1:6] = (
            (1.0, -1.0),
            (2.0, -2.0),
            (3.0, -3.0),
            (4.0, -4.0),
            (5.0, -5.0),
        )
        null[1:6] = (
            (0.25, 0.75),
            (0.5, 0.25),
            (0.75, 0.5),
            (1.0, 0.75),
            (1.25, 1.0),
        )
    return {
        "price-main": price,
        "volume-main": volume,
        "base-alpha": base,
        "measurement-alpha": measurement,
        "null-main": null,
    }


def _values(artifacts, candidate_id: str, period_index: int) -> np.ndarray:
    decision = START + pd.Timedelta(minutes=period_index)
    frame = artifacts.signal_records.loc[
        (artifacts.signal_records["candidate_id"] == candidate_id)
        & (artifacts.signal_records["decision_time"] == decision)
    ].sort_values("symbol")
    return frame["signal_value"].to_numpy(dtype=float)


def test_vertical_slice_manual_two_asset_five_minute_scalar_chain():
    specification = _spec()
    artifacts = generate_known_truth_dgp_vertical_slice_v1(
        specification,
        _registry(),
        injected_standard_innovations=_injected(),
    )
    assert tuple(artifacts.market_records.columns) == MARKET_INPUT_SPINE_RECORD_COLUMNS_V1
    assert tuple(artifacts.signal_records.columns) == KNOWN_TRUTH_DGP_SIGNAL_RECORD_COLUMNS_V1
    assert artifacts.market_records.shape == (10, len(MARKET_INPUT_SPINE_RECORD_COLUMNS_V1))
    assert artifacts.signal_records.shape == (5 * 2 * 159, len(KNOWN_TRUTH_DGP_SIGNAL_RECORD_COLUMNS_V1))
    assert set(artifacts.market_records["source"]) == {KNOWN_TRUTH_DGP_MARKET_SOURCE_V1}
    assert set(artifacts.market_records["value_status"]) == {KNOWN_TRUTH_DGP_VALUE_STATUS_V1}
    assert set(artifacts.signal_records.columns).isdisjoint(
        {"truth", "truth_role", "future_return", "beta_total", "discovery_result"}
    )

    expected_base = np.asarray((1.0, -1.0)) @ np.asarray(L_CORRELATED).T
    np.testing.assert_allclose(_values(artifacts, CANDIDATES[0], 0), expected_base, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(_values(artifacts, CANDIDATES[1], 0), 0.5 * expected_base, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(_values(artifacts, CANDIDATES[2], 0), expected_base, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(_values(artifacts, CANDIDATES[3], 0), 0.25 * expected_base, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(_values(artifacts, CANDIDATES[4], 0), (0.25, 0.75), rtol=0.0, atol=0.0)

    trace = artifacts.truth_sidecar.effect_trace
    assert set(trace["candidate_id"]) == {CANDIDATES[0]}
    assert set(trace["lag_minutes"]) == {1}
    assert set(trace["price_time"]) == {START + pd.Timedelta(minutes=4)}
    beta = dict(KNOWN_TRUTH_BETA_TOTAL_SCALES_V1)["weak"]
    weight = _known_truth_dgp_effect_weight_v1("fast", 1)
    np.testing.assert_allclose(
        trace.sort_values("symbol")["log_return_contribution"].to_numpy(dtype=float),
        beta * expected_base * weight,
        rtol=0.0,
        atol=1e-18,
    )
    price_rows = artifacts.market_records.loc[
        artifacts.market_records["open_time"] == START + pd.Timedelta(minutes=4)
    ].sort_values("symbol")
    np.testing.assert_allclose(
        price_rows["close"].to_numpy(dtype=float),
        np.asarray((100.0, 200.0)) * np.exp(beta * expected_base * weight),
        rtol=0.0,
        atol=1e-12,
    )
    assert (artifacts.market_records["volume"].to_numpy(dtype=float) > 0.0).all()


def test_vertical_slice_applies_role_standardization_and_alias_direction():
    injected = _injected()
    baseline = generate_known_truth_dgp_vertical_slice_v1(
        _spec(),
        _registry(),
        injected_standard_innovations=injected,
    )
    assignments = list(_spec().scenario.truth_assignments)
    assignments[0] = replace(
        assignments[0],
        role_standardization_center=1.0,
        role_standardization_scale=2.0,
    )
    assignments[2] = replace(assignments[2], direction=-1)
    specification = _spec(
        scenario=replace(
            _spec().scenario,
            truth_assignments=tuple(assignments),
        )
    )
    artifacts = generate_known_truth_dgp_vertical_slice_v1(
        specification,
        _registry(),
        injected_standard_innovations=injected,
    )
    expected_direct = (_values(baseline, CANDIDATES[0], 0) - 1.0) / 2.0
    np.testing.assert_allclose(
        _values(artifacts, CANDIDATES[0], 0),
        expected_direct,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        _values(artifacts, CANDIDATES[2], 0),
        -expected_direct,
        rtol=0.0,
        atol=0.0,
    )


def test_vertical_slice_requires_registry_group_and_role_standardization_binding():
    registry = _registry()
    streams = list(registry.streams)
    for index, stream in enumerate(streams):
        if stream.stream_id in {"base-alpha", "measurement-alpha"}:
            streams[index] = replace(stream, information_group_id="beta")
    binding = replace(registry.information_group_bindings[0], information_group_id="beta")
    bad_registry = replace(
        registry,
        streams=tuple(streams),
        information_group_bindings=(binding,),
    )
    with pytest.raises(ValueError, match="wrong information group"):
        validate_known_truth_dgp_vertical_specification_v1(_spec(), bad_registry)

    assignments = list(_spec().scenario.truth_assignments)
    assignments[0] = replace(assignments[0], role_standardization_scale=None)
    with pytest.raises(ValueError, match="role standardization"):
        validate_known_truth_dgp_vertical_specification_v1(
            replace(
                _spec(),
                scenario=replace(_spec().scenario, truth_assignments=tuple(assignments)),
            ),
            _registry(),
        )


def test_vertical_slice_rank_only_is_separate_and_uses_formal_rank():
    injected = _injected()
    injected.pop("measurement-alpha")
    artifacts = generate_known_truth_dgp_vertical_slice_v1(
        _spec(scenario=_scenario_rank_only()),
        _registry(),
        injected_standard_innovations=injected,
    )
    np.testing.assert_allclose(_values(artifacts, CANDIDATES[0], 0), (1.0, -1.0), rtol=0.0, atol=0.0)
    trace = artifacts.truth_sidecar.effect_trace
    assert trace["beta_total"].isna().all()
    assert trace["beta_rank"].notna().all()
    assert trace["effect_curve_id"].isna().all()
    assert set(trace["rank_curve_id"]) == {RANK_CURVE.curve_id}
    assert (trace["effect_coefficient"] == dict(KNOWN_TRUTH_BETA_TOTAL_SCALES_V1)["weak"]).all()
    rank_weight = _known_truth_dgp_rank_effect_weight_v1(RANK_CURVE, 1)
    np.testing.assert_allclose(
        trace["log_return_contribution"].to_numpy(dtype=float),
        dict(KNOWN_TRUTH_BETA_TOTAL_SCALES_V1)["weak"]
        * trace["signal_value"].to_numpy(dtype=float)
        * trace["mirror_sign"].to_numpy(dtype=float)
        * rank_weight,
        rtol=0.0,
        atol=1e-18,
    )


def test_vertical_slice_rank_curve_formula_is_independent_from_scalar_fast():
    ratio = math.exp(-1.0 / RANK_CURVE.lambda_rank_minutes)
    assert _known_truth_dgp_rank_effect_weight_v1(RANK_CURVE, 1) == pytest.approx(
        1.0 - ratio,
        abs=1e-15,
    )
    assert _known_truth_dgp_rank_effect_weight_v1(RANK_CURVE, 2) == pytest.approx(
        (1.0 - ratio) * ratio,
        abs=1e-15,
    )
    assert _known_truth_dgp_rank_effect_weight_v1(RANK_CURVE, 1) != pytest.approx(
        _known_truth_dgp_effect_weight_v1("fast", 1),
        abs=1e-12,
    )
    cutoff = next(
        lag for lag in range(1, 1000) if ratio**lag <= RANK_CURVE.epsilon_rank
    )
    assert 1.0 - sum(
        _known_truth_dgp_rank_effect_weight_v1(RANK_CURVE, lag)
        for lag in range(1, cutoff + 1)
    ) == pytest.approx(ratio**cutoff, abs=1e-15)
    assert _known_truth_dgp_rank_effect_weight_v1(RANK_CURVE, cutoff + 1) > 0.0


def test_vertical_slice_rank_curve_contract_rejects_scalar_mixing_or_missing_curve():
    rank_spec = _spec(scenario=_scenario_rank_only())
    assignments = list(rank_spec.scenario.truth_assignments)
    with pytest.raises(ValueError, match="rank-only direct candidates cannot bind scalar"):
        validate_known_truth_dgp_vertical_specification_v1(
            replace(
                rank_spec,
                scenario=replace(
                    rank_spec.scenario,
                    truth_assignments=(
                        replace(assignments[0], effect_curve_id="fast"),
                        *assignments[1:],
                    ),
                ),
            ),
            _registry(),
        )
    with pytest.raises(ValueError, match="rank-only direct candidates require"):
        validate_known_truth_dgp_vertical_specification_v1(
            replace(
                rank_spec,
                scenario=replace(
                    rank_spec.scenario,
                    truth_assignments=(
                        replace(assignments[0], w_rank=None),
                        *assignments[1:],
                    ),
                ),
            ),
            _registry(),
        )
    with pytest.raises(ValueError, match="rank_effect_curves"):
        validate_known_truth_dgp_vertical_specification_v1(
            replace(rank_spec, rank_effect_curves=()),
            _registry(),
        )
    with pytest.raises(ValueError, match="rank_effect_curves are not allowed"):
        validate_known_truth_dgp_vertical_specification_v1(
            replace(_spec(), rank_effect_curves=(RANK_CURVE,)),
            _registry(),
        )


def test_vertical_slice_default_path_is_deterministic_and_order_invariant():
    registry = _registry()
    first = generate_known_truth_dgp_vertical_slice_v1(_spec(), registry)
    second = generate_known_truth_dgp_vertical_slice_v1(_spec(), registry)
    reordered = generate_known_truth_dgp_vertical_slice_v1(
        _spec(processing_order=(1, 0)),
        registry,
    )
    pd.testing.assert_frame_equal(first.market_records, second.market_records)
    pd.testing.assert_frame_equal(first.signal_records, second.signal_records)
    pd.testing.assert_frame_equal(first.market_records, reordered.market_records)
    pd.testing.assert_frame_equal(first.signal_records, reordered.signal_records)


def test_vertical_slice_overlapping_decisions_add_at_the_same_price_minute():
    periods = 6
    artifacts = generate_known_truth_dgp_vertical_slice_v1(
        _spec(periods=periods),
        _registry(),
        injected_standard_innovations=_injected(periods=periods),
    )
    price_time = START + pd.Timedelta(minutes=5)
    trace = artifacts.truth_sidecar.effect_trace.loc[
        (artifacts.truth_sidecar.effect_trace["candidate_id"] == CANDIDATES[0])
        & (artifacts.truth_sidecar.effect_trace["price_time"] == price_time)
    ]
    assert len(trace) == 2 * 2
    assert set(trace["lag_minutes"]) == {1, 2}
    for symbol in FIXTURE_ASSETS:
        price_row = artifacts.market_records.loc[
            (artifacts.market_records["symbol"] == symbol)
            & (artifacts.market_records["open_time"] == price_time)
        ].iloc[0]
        observed_log_move = math.log(float(price_row["close"]) / float(price_row["open"]))
        expected_log_move = float(
            trace.loc[trace["symbol"] == symbol, "log_return_contribution"].sum()
        )
        assert observed_log_move == pytest.approx(expected_log_move, abs=1e-15)


def test_vertical_slice_supports_the_twenty_asset_production_shape():
    artifacts = generate_known_truth_dgp_vertical_slice_v1(_spec(assets=ASSETS), _registry())
    assert artifacts.asset_symbols == ASSETS
    assert len(artifacts.market_records) == 20 * 5
    assert len(artifacts.signal_records) == 20 * 5 * 159
    assert set(artifacts.signal_records["candidate_id"]) == set(CANDIDATES)


def test_vertical_slice_curves_are_continuous_and_have_positive_minute_weights():
    assert _known_truth_dgp_curve_cdf_v1("fast", 4.0) == pytest.approx(0.8, abs=1e-15)
    assert _known_truth_dgp_curve_cdf_v1("persistent", 12.0) == pytest.approx(0.5, abs=1e-15)
    for curve_id in KNOWN_TRUTH_EFFECT_CURVES_V1:
        weights = [_known_truth_dgp_effect_weight_v1(curve_id, lag) for lag in range(1, 6)]
        assert all(weight > 0.0 for weight in weights)
        assert all(
            _known_truth_dgp_curve_cdf_v1(curve_id, lag / 60.0)
            >= _known_truth_dgp_curve_cdf_v1(curve_id, (lag - 1) / 60.0)
            for lag in range(1, 6)
        )


def test_vertical_slice_tail_tolerance_is_numeric_only_and_not_a_hard_cutoff():
    epsilon = 1e-12
    for curve_id in KNOWN_TRUTH_EFFECT_CURVES_V1:
        cutoff = next(
            lag
            for lag in range(1, 200_001)
            if 1.0 - _known_truth_dgp_curve_cdf_v1(curve_id, lag / 60.0) <= epsilon
        )
        cumulative = sum(
            _known_truth_dgp_effect_weight_v1(curve_id, lag)
            for lag in range(1, cutoff + 1)
        )
        assert 1.0 - _known_truth_dgp_curve_cdf_v1(curve_id, cutoff / 60.0) <= epsilon
        assert cumulative == pytest.approx(
            _known_truth_dgp_curve_cdf_v1(curve_id, cutoff / 60.0),
            abs=1e-12,
        )
        assert _known_truth_dgp_effect_weight_v1(curve_id, cutoff + 1) > 0.0


@pytest.mark.parametrize(
    "fault",
    (
        KnownTruthDgpFaultInjectionV1("drop-signal", "drop_signal_row", 0, FIXTURE_ASSETS[0], CANDIDATES[0]),
        KnownTruthDgpFaultInjectionV1("nan-signal", "nan_signal_value", 0, FIXTURE_ASSETS[0], CANDIDATES[0]),
        KnownTruthDgpFaultInjectionV1("shift-signal", "shift_signal_time", 0, FIXTURE_ASSETS[0], CANDIDATES[0]),
        KnownTruthDgpFaultInjectionV1("drop-market", "drop_market_row", 0, FIXTURE_ASSETS[0]),
        KnownTruthDgpFaultInjectionV1("bad-volume", "nonpositive_volume", 0, FIXTURE_ASSETS[0]),
    ),
)
def test_vertical_slice_registered_faults_fail_closed(fault):
    with pytest.raises(ValueError):
        generate_known_truth_dgp_vertical_slice_v1(
            _spec(faults=(fault,)),
            _registry(),
            injected_standard_innovations=_injected(),
        )


def test_vertical_slice_rejects_invalid_role_identity_and_variant_bindings():
    specification = _spec()
    assignments = list(specification.scenario.truth_assignments)
    assignments[1] = replace(assignments[1], rho=1.0)
    with pytest.raises(ValueError, match="rho"):
        validate_known_truth_dgp_vertical_specification_v1(
            replace(specification, scenario=replace(specification.scenario, truth_assignments=tuple(assignments))),
            _registry(),
        )

    assignments = list(specification.scenario.truth_assignments)
    assignments[4] = replace(assignments[4], null_noise_stream_id="base-alpha")
    with pytest.raises(ValueError, match="null noise"):
        validate_known_truth_dgp_vertical_specification_v1(
            replace(specification, scenario=replace(specification.scenario, truth_assignments=tuple(assignments))),
            _registry(),
        )

    assignments = list(specification.scenario.truth_assignments)
    assignments[2] = replace(assignments[2], alias_of_candidate_id=CANDIDATES[3])
    with pytest.raises(ValueError, match="alias"):
        validate_known_truth_dgp_vertical_specification_v1(
            replace(specification, scenario=replace(specification.scenario, truth_assignments=tuple(assignments))),
            _registry(),
        )

    with pytest.raises(ValueError, match="observation_variants"):
        validate_known_truth_dgp_vertical_specification_v1(
            replace(specification, observation_variants=specification.observation_variants[:-1]),
            _registry(),
        )


def test_vertical_slice_rejects_shared_signal_kind_identity_and_wrong_variant():
    specification = _spec()
    signal_streams = list(specification.signal_streams)
    signal_streams[0] = replace(signal_streams[0], stream_id="price-main")
    with pytest.raises(ValueError, match="exactly cover"):
        validate_known_truth_dgp_vertical_specification_v1(
            replace(specification, signal_streams=tuple(signal_streams)),
            _registry(),
        )

    variants = list(specification.observation_variants)
    variants[0] = replace(variants[0], output_timeframe="5m")
    with pytest.raises(ValueError, match="timeframe"):
        validate_known_truth_dgp_vertical_specification_v1(
            replace(specification, observation_variants=tuple(variants)),
            _registry(),
        )


def test_vertical_slice_rejects_missing_formal_grid_or_schema_identity():
    specification = _spec()
    with pytest.raises(ValueError, match="159"):
        validate_known_truth_dgp_vertical_specification_v1(
            replace(
                specification,
                scenario=replace(
                    specification.scenario,
                    truth_assignments=specification.scenario.truth_assignments[:-1],
                ),
            ),
            _registry(),
        )
    with pytest.raises(ValueError, match="schema_version"):
        validate_known_truth_dgp_vertical_specification_v1(
            replace(specification, schema_version="other-v1"),
            _registry(),
        )


@pytest.fixture(scope="module")
def _pipeline_discovery_case():
    """A full registered 4h scan with one true group and registered nulls."""
    periods = 1445  # through t=20h plus the 4h executable return at t+4m
    injected = _injected(periods=periods)
    injected["base-alpha"][:] = 0.0
    for decision_number in range(6):
        signal = float(decision_number + 1)
        injected["base-alpha"][1 + 240 * decision_number] = (signal, -signal)
    # One middle-of-holding price shock makes exactly one test rank-IC
    # observation negative while leaving the adjacent four-hour windows
    # unchanged.  This supplies finite variance to the existing formal HAC
    # gate without changing any production threshold or statistic.
    injected["price-main"][1 + 240 * 3 + 120] = (0.0, 0.3)
    injected["null-main"][:] = 0.0
    injected.pop("measurement-alpha")
    scenario = _scenario_pipeline_discovery()
    dgp = generate_known_truth_dgp_vertical_slice_v1(
        _spec(scenario=scenario, periods=periods),
        _registry(),
        injected_standard_innovations=injected,
    )
    decision_times = tuple(
        START + pd.Timedelta(hours=4 * index) for index in range(6)
    )
    horizon = "4h"
    candidate_registry = []
    formal_registry = factor_registry.base_panel_registry("1h").reset_index(drop=True)
    by_feature = formal_registry.set_index("feature_name", drop=False)
    for candidate_id in CANDIDATES:
        feature_name, candidate_horizon = candidate_id.rsplit("::", 1)
        if candidate_horizon != horizon:
            continue
        row = by_feature.loc[feature_name]
        candidate_registry.append(
            {
                "candidate_id": candidate_id,
                "feature_name": candidate_id,
                "base_feature_name": feature_name,
                "return_horizon": horizon,
                "family": row["family"],
                "signal_timeframe": row["signal_timeframe"],
            }
        )
    scan_registry = pd.DataFrame(
        candidate_registry,
        columns=[
            "candidate_id", "feature_name", "base_feature_name",
            "return_horizon", "family", "signal_timeframe",
        ],
    )
    selected_signals = dgp.signal_records.loc[
        dgp.signal_records["decision_time"].isin(decision_times)
        & dgp.signal_records["candidate_id"].isin(scan_registry["candidate_id"])
    ].copy()
    raw = KnownTruthL0L4RawInputV1(
        market_records=dgp.market_records,
        signal_records=selected_signals,
        schema_version=dgp.schema_version,
        generation_batch=dgp.generation_batch,
        asset_symbols=dgp.asset_symbols,
    )
    fold = WalkForwardFold(
        fold_idx=0,
        train_start=START,
        train_end=START,
        test_start=START + pd.Timedelta(hours=4),
        test_end=START + pd.Timedelta(hours=20),
    )
    horizon_deltas = {
        "1m": pd.Timedelta(minutes=1),
        "1h": pd.Timedelta(hours=1),
        "4h": pd.Timedelta(hours=4),
        "8h": pd.Timedelta(hours=8),
        "12h": pd.Timedelta(hours=12),
        "1d": pd.Timedelta(days=1),
    }
    truth_rows = []
    for candidate_id in scan_registry["candidate_id"]:
        if candidate_id == CANDIDATES[67]:
            truth_rows.append(
                {
                    "candidate_id": candidate_id,
                    "return_horizon": horizon,
                    "truth_role": "direct",
                    "information_group": "alpha",
                    "marginal_predictive_truth": 1,
                    "expected_direction": 1,
                }
            )
        elif candidate_id == CANDIDATES[71]:
            truth_rows.append(
                {
                    "candidate_id": candidate_id,
                    "return_horizon": horizon,
                    "truth_role": "alias",
                    "information_group": "alpha",
                    "marginal_predictive_truth": 1,
                    "expected_direction": 1,
                }
            )
        else:
            truth_rows.append(
                {
                    "candidate_id": candidate_id,
                    "return_horizon": horizon,
                    "truth_role": "null",
                    "information_group": "",
                    "marginal_predictive_truth": 0,
                    "expected_direction": 0,
                }
            )
    return {
        "dgp": dgp,
        "raw": raw,
        "scan_registry": scan_registry,
        "folds": (fold,),
        "horizon_deltas": horizon_deltas,
        "truth_manifest": pd.DataFrame(truth_rows),
        "run_kwargs": {
            "folds": (fold,),
            "walk_forward_spec": {
                "train_days": 1,
                "test_days": 1,
                "embargo_days": 0,
                "step_days": 1,
            },
            "horizon_deltas": horizon_deltas,
            "frequency_periods_per_year": {"4h": 2190},
            "supported_signal_timeframes": ("1h", "4h", "8h", "12h", "1d"),
            "cost_multipliers": (1.0,),
            "taker_fee_rate": 0.001,
        },
    }


@pytest.fixture(scope="module")
def _pipeline_discovery_result(_pipeline_discovery_case):
    case = _pipeline_discovery_case
    return run_known_truth_l0_l4_pipeline_discovery_micro_e2e_v1(
        case["raw"],
        scan_registry=case["scan_registry"],
        **case["run_kwargs"],
    )


def test_pipeline_discovery_uses_complete_registry_and_terminal_truth_blind_boundary(
    _pipeline_discovery_case,
    _pipeline_discovery_result,
):
    case = _pipeline_discovery_case
    result = _pipeline_discovery_result
    direct_id, alias_id = CANDIDATES[67], CANDIDATES[71]

    assert isinstance(result, KnownTruthL0L4PipelineDiscoveryArtifactsV1)
    assert tuple(result.registered_candidate_ids) == tuple(case["scan_registry"]["candidate_id"])
    assert len(result.l2_gate_summary) == 23
    supported = set(
        result.l2_gate_summary.loc[
            result.l2_gate_summary["two_gate_support"], "feature_name"
        ]
    )
    assert supported == {direct_id, alias_id}
    assert len(result.l3_catalog) == 1
    assert set(result.l3_catalog["component_features"]) == {
        f"{direct_id} | {alias_id}"
    }
    discovery = result.pipeline_discovery.set_index("candidate_id")
    assert discovery.loc[[direct_id, alias_id], "pipeline_discovery"].tolist() == [True, True]
    assert discovery.loc[[direct_id, alias_id], "first_loss_layer"].tolist() == ["none", "none"]
    null_ids = [candidate_id for candidate_id in case["scan_registry"]["candidate_id"]
                if candidate_id not in {direct_id, alias_id}]
    assert not discovery.loc[null_ids, "pipeline_discovery"].any()
    assert set(discovery.loc[null_ids, "first_loss_layer"]) == {"l2"}

    # L4 activation is derived from the formal holdings' signed_quantity, not
    # from detail counts or orders.  The two accepted units share one actual
    # formal combo and therefore both expose the same holdings evidence.
    assert set(result.l4_activation["candidate_id"]) == {direct_id, alias_id}
    assert result.l4_activation["l4_exposure"].tolist() == [True, True]
    assert (result.l4_holdings["signed_quantity"].abs() > 0.0).any()
    for frame_name in ("pipeline_discovery", "l4_activation"):
        assert not any("truth" in str(column).lower() for column in getattr(result, frame_name).columns)

    parameters = inspect.signature(run_known_truth_l0_l4_pipeline_discovery_micro_e2e_v1).parameters
    assert "candidate_ids" not in parameters
    assert "truth_manifest" not in parameters

    truth_blind = bind_known_truth_l0_l4_truth_blind_evaluation_input_v1(
        result,
        persistence_reference="fixture://atomic/truth-blind-output-v1",
    )
    evaluated = evaluate_known_truth_pipeline_terminal_v1(
        truth_blind,
        case["truth_manifest"],
    )
    summary = evaluated.summary.iloc[0]
    assert int(summary["total_units"]) == 23
    assert int(summary["discovered_units"]) == 2
    assert int(summary["tp"]) == 2
    assert int(summary["fp"]) == 0
    assert int(summary["fn"]) == 0
    assert float(summary["tpr"]) == pytest.approx(1.0)
    assert float(summary["fdp"]) == pytest.approx(0.0)
    assert int(summary["false_activation"]) == 0
    assert int(summary["end_to_end_recovery"]) == 2
    assert int(summary["information_group_tp"]) == 1
    assert int(summary["information_group_fp"]) == 0
    candidate = evaluated.candidate_results.set_index("candidate_id")
    assert candidate.loc[[direct_id, alias_id], "end_to_end_recovery"].tolist() == [True, True]
    assert not candidate.loc[null_ids, "false_activation"].any()


def test_pipeline_discovery_rejects_preselected_or_truth_contaminated_inputs(
    _pipeline_discovery_case,
):
    case = _pipeline_discovery_case
    with pytest.raises(ValueError, match="complete registered horizon"):
        run_known_truth_l0_l4_pipeline_discovery_micro_e2e_v1(
            case["raw"],
            scan_registry=case["scan_registry"].head(2),
            **case["run_kwargs"],
        )

    contaminated = case["raw"].signal_records.copy(deep=True)
    contaminated["truth_role"] = "direct"
    with pytest.raises(ValueError, match="do not match the DGP signal schema"):
        run_known_truth_l0_l4_pipeline_discovery_micro_e2e_v1(
            replace(case["raw"], signal_records=contaminated),
            scan_registry=case["scan_registry"],
            **case["run_kwargs"],
        )


def test_pipeline_discovery_requires_l3_acceptance_and_not_l4_alone(
    _pipeline_discovery_case,
    _pipeline_discovery_result,
):
    case = _pipeline_discovery_case
    result = _pipeline_discovery_result
    from qlab.full_pipeline_simulation import _known_truth_pipeline_build_discovery_frames_v1

    # An L2 qualification without a catalog/spec acceptance is not a discovery.
    l2_only, no_activation = _known_truth_pipeline_build_discovery_frames_v1(
        gate_summary=result.l2_gate_summary,
        catalog=pd.DataFrame(columns=result.l3_catalog.columns),
        l3_weights=pd.DataFrame(columns=result.l3_weights.columns),
        l4_orders=pd.DataFrame(),
        l4_holdings=pd.DataFrame(),
        candidate_registry=case["scan_registry"],
        return_horizon="4h",
    )
    direct_id, alias_id = CANDIDATES[67], CANDIDATES[71]
    assert l2_only.set_index("candidate_id").loc[
        [direct_id, alias_id], "l2_two_gate_support"
    ].tolist() == [True, True]
    assert not l2_only["pipeline_discovery"].any()
    assert set(l2_only.loc[
        l2_only["l2_two_gate_support"], "first_loss_layer"
    ]) == {"l3"}
    assert no_activation.empty

    # Holdings alone are execution evidence, not a discovery substitute.
    l4_only, activation = _known_truth_pipeline_build_discovery_frames_v1(
        gate_summary=result.l2_gate_summary.assign(two_gate_support=False),
        catalog=result.l3_catalog,
        l3_weights=result.l3_weights,
        l4_orders=result.l4_orders,
        l4_holdings=result.l4_holdings,
        candidate_registry=case["scan_registry"],
        return_horizon="4h",
    )
    assert not l4_only["pipeline_discovery"].any()
    assert set(activation["candidate_id"]) == {direct_id, alias_id}
    assert activation["l4_exposure"].all()


def test_pipeline_discovery_truth_blind_identity_and_l4_holdings_fail_closed(
    _pipeline_discovery_case,
    _pipeline_discovery_result,
):
    case = _pipeline_discovery_case
    result = _pipeline_discovery_result
    with pytest.raises(ValueError, match="complete persistence"):
        bind_known_truth_l0_l4_truth_blind_evaluation_input_v1(
            result,
            persistence_reference="fixture://incomplete",
            persistence_status="partial",
        )
    truth_blind = bind_known_truth_l0_l4_truth_blind_evaluation_input_v1(
        result,
        persistence_reference="fixture://atomic/truth-blind-output-v1",
    )
    with pytest.raises(ValueError, match="output identity"):
        evaluate_known_truth_pipeline_terminal_v1(
            replace(truth_blind, output_identity="0" * 64),
            case["truth_manifest"],
        )

    from qlab.full_pipeline_simulation import _known_truth_pipeline_build_discovery_frames_v1

    missing_exposure = result.l4_holdings.drop(columns=["signed_quantity"])
    with pytest.raises(ValueError, match="signed_quantity exposure evidence"):
        _known_truth_pipeline_build_discovery_frames_v1(
            gate_summary=result.l2_gate_summary,
            catalog=result.l3_catalog,
            l3_weights=result.l3_weights,
            l4_orders=result.l4_orders,
            l4_holdings=missing_exposure,
            candidate_registry=case["scan_registry"],
            return_horizon="4h",
        )


@pytest.fixture(scope="module")
def _micro_l0_l4_case():
    """A complete enough two-asset case for the formal 4h execution path."""
    periods = 485
    injected = _injected(periods=periods)
    injected["base-alpha"][:] = 0.0
    injected["base-alpha"][1] = (1.0, -1.0)
    injected["base-alpha"][241] = (2.0, -2.0)
    injected["null-main"][:] = 0.0
    injected["null-main"][1] = (0.25, 0.75)
    injected["null-main"][241] = (0.75, 0.25)
    dgp = generate_known_truth_dgp_vertical_slice_v1(
        _spec(periods=periods),
        _registry(),
        injected_standard_innovations=injected,
    )
    decision_times = {START, START + pd.Timedelta(hours=4)}
    selected_signals = dgp.signal_records.loc[
        dgp.signal_records["decision_time"].isin(decision_times)
        & dgp.signal_records["candidate_id"].isin((CANDIDATES[0], CANDIDATES[4]))
    ].copy()
    raw = KnownTruthL0L4RawInputV1(
        market_records=dgp.market_records,
        signal_records=selected_signals,
        schema_version=dgp.schema_version,
        generation_batch=dgp.generation_batch,
        asset_symbols=dgp.asset_symbols,
    )
    fold = WalkForwardFold(
        fold_idx=0,
        train_start=START,
        train_end=START,
        test_start=START + pd.Timedelta(hours=4),
        test_end=START + pd.Timedelta(hours=4),
    )
    horizon_deltas = {
        "1m": pd.Timedelta(minutes=1),
        "4h": pd.Timedelta(hours=4),
        "8h": pd.Timedelta(hours=8),
        "12h": pd.Timedelta(hours=12),
        "1d": pd.Timedelta(days=1),
    }
    run_kwargs = {
        "candidate_ids": (CANDIDATES[0], CANDIDATES[4]),
        "folds": (fold,),
        "walk_forward_spec": {
            "train_days": 1,
            "test_days": 1,
            "embargo_days": 0,
            "step_days": 1,
        },
        "horizon_deltas": horizon_deltas,
        "frequency_periods_per_year": {"4h": 2190},
        "supported_signal_timeframes": ("1m",),
        "cost_multipliers": (1.0,),
        "taker_fee_rate": 0.001,
    }
    return {"dgp": dgp, "raw": raw, "run_kwargs": run_kwargs}


def _run_micro_l0_l4(case, raw=None):
    return run_known_truth_l0_l4_micro_e2e_v1(
        case["raw"] if raw is None else raw,
        **case["run_kwargs"],
    )


def test_known_truth_l0_l4_micro_manual_formal_chain(_micro_l0_l4_case):
    """Manually check raw -> executable L0--L4 values without truth leakage."""
    case = _micro_l0_l4_case
    dgp = case["dgp"]
    raw = case["raw"]
    result = _run_micro_l0_l4(case)
    direct_id, null_id = CANDIDATES[0], CANDIDATES[4]

    # The public raw contract contains exactly two decisions, two assets and
    # one direct plus one null candidate.  Truth is held only by the fixture.
    assert set(raw.signal_records["candidate_id"]) == {direct_id, null_id}
    assert raw.signal_records.groupby("candidate_id").size().to_dict() == {
        direct_id: 4,
        null_id: 4,
    }
    np.testing.assert_allclose(
        _values(dgp, direct_id, 0), [1.0, 0.5 - math.sqrt(0.75)]
    )
    np.testing.assert_allclose(
        _values(dgp, direct_id, 240), [2.0, 1.0 - math.sqrt(3.0)]
    )
    np.testing.assert_allclose(_values(dgp, null_id, 0), [0.25, 0.75])
    np.testing.assert_allclose(_values(dgp, null_id, 240), [0.75, 0.25])
    assert tuple(raw.market_records.columns) == MARKET_INPUT_SPINE_RECORD_COLUMNS_V1
    assert set(raw.market_records["symbol"]) == set(FIXTURE_ASSETS)
    assert np.isfinite(
        raw.market_records[["open", "high", "low", "close", "volume"]]
        .to_numpy(dtype=float)
    ).all()
    assert (raw.market_records["volume"] > 0).all()
    market_by_key = raw.market_records.set_index(["symbol", "open_time"])
    expected_market = {
        ("ADA", START + pd.Timedelta(minutes=244)): 100.01904080867055,
        ("APT", START + pd.Timedelta(minutes=244)): 199.98606297312566,
        ("ADA", START + pd.Timedelta(minutes=484)): 100.06094335056180,
        ("APT", START + pd.Timedelta(minutes=484)): 199.95540493254370,
    }
    for key, expected_open in expected_market.items():
        assert float(market_by_key.loc[key, "open"]) == pytest.approx(expected_open)
        assert float(market_by_key.loc[key, "volume"]) == pytest.approx(
            10.0 if key[0] == "ADA" else 11.0
        )
        row = market_by_key.loc[key]
        assert row["high"] == max(row["open"], row["close"])
        assert row["low"] == min(row["open"], row["close"])
    assert not any("truth" in str(column).lower() for column in raw.signal_records.columns)

    # L1 canonical decision/availability and the exact 4-minute executable
    # ledger are checked at the sole test decision t=04:00.
    test_ts = START + pd.Timedelta(hours=4)
    l1_test = result.l1_panel.loc[test_ts].sort_values("symbol")
    assert list(l1_test["symbol"]) == list(sorted(FIXTURE_ASSETS))
    assert set(l1_test["canonical_period_end_ts"]) == {test_ts}
    assert set(l1_test["availability_ts"]) == {test_ts + pd.Timedelta(minutes=4)}
    assert set(l1_test["execution_ts"]) == {test_ts + pd.Timedelta(minutes=4)}
    assert set(l1_test["next_execution_ts"]) == {
        test_ts + pd.Timedelta(hours=4, minutes=4)
    }
    assert (l1_test["exit_price"] / l1_test["entry_price"] - 1.0).to_numpy() == pytest.approx(
        l1_test["executable_return"].to_numpy()
    )

    # With two observations, the hand rank correlations are +1 for the direct
    # candidate and -1 for the null candidate in the training decision.
    train_ic = result.l2_rank_ic.loc[
        result.l2_rank_ic["decision_ts"].eq(START)
    ].set_index("feature_name")["raw_rank_ic"]
    assert train_ic.loc[direct_id] == pytest.approx(1.0)
    assert train_ic.loc[null_id] == pytest.approx(-1.0)
    directions = result.l2_directions.set_index("feature_name")
    assert directions.loc[direct_id, "train_mean_ic"] == pytest.approx(1.0)
    assert directions.loc[direct_id, "direction"] == 1
    assert directions.loc[null_id, "train_mean_ic"] == pytest.approx(-1.0)
    assert directions.loc[null_id, "direction"] == -1

    # Equal formal combo weights are 1/2 each; target membership is one long
    # and one short with +/-1/2 weights after the learned directions.
    np.testing.assert_allclose(result.l3_weights["feature_weight"].to_numpy(), [0.5, 0.5])
    composite_by_symbol = result.l3_composite.set_index("symbol")["combo_signal"]
    assert composite_by_symbol.loc["ADA"] == pytest.approx(0.625)
    assert composite_by_symbol.loc["APT"] == pytest.approx(-0.4910254037844386)
    targets = result.l3_targets.loc[result.l3_targets["decision_ts"].eq(test_ts)]
    assert set(targets["leg"]) == {"long", "short"}
    assert sorted(targets["target_weight"].to_numpy()) == pytest.approx([-0.5, 0.5])
    assert len(targets) == 2
    target_by_symbol = targets.set_index("symbol")
    assert target_by_symbol.loc["ADA", "signal_value"] == pytest.approx(0.625)
    assert target_by_symbol.loc["APT", "signal_value"] == pytest.approx(-0.4910254037844386)
    assert set(result.l4_detail["execution_ts"]) == {test_ts + pd.Timedelta(minutes=4)}
    assert set(result.l4_orders["status"]) == {"open", "terminal_close"}

    # These are independently hand-calculated fixture constants, not values
    # derived from the returned orders/holdings.  They pin the complete L4
    # ledger for entry, terminal close, fee and net result.
    expected_ledger = {
        "ADA": {
            "entry_price": 100.01904080867055,
            "exit_price": 100.0609433505618,
            "signed_quantity": 0.00499904814080816,
            "terminal_close_notional": 0.5002094728241366,
        },
        "APT": {
            "entry_price": 199.98606297312566,
            "exit_price": 199.9554049325437,
            "signed_quantity": -0.00250017422497682,
            "terminal_close_notional": 0.4999233495571487,
        },
    }
    holdings_by_symbol = result.l4_holdings.set_index("symbol")
    orders_by_key = result.l4_orders.set_index(["symbol", "status"])
    for symbol, expected in expected_ledger.items():
        holding = holdings_by_symbol.loc[symbol]
        assert holding["entry_price"] == pytest.approx(expected["entry_price"])
        assert holding["exit_price"] == pytest.approx(expected["exit_price"])
        assert holding["signed_quantity"] == pytest.approx(expected["signed_quantity"])
        open_order = orders_by_key.loc[(symbol, "open")]
        close_order = orders_by_key.loc[(symbol, "terminal_close")]
        assert open_order["execution_price"] == pytest.approx(expected["entry_price"])
        assert open_order["executed_quantity"] == pytest.approx(expected["signed_quantity"])
        assert open_order["executed_order_notional"] == pytest.approx(0.5)
        assert close_order["execution_price"] == pytest.approx(expected["exit_price"])
        assert close_order["executed_quantity"] == pytest.approx(-expected["signed_quantity"])
        assert close_order["executed_order_notional"] == pytest.approx(
            expected["terminal_close_notional"]
        )

    # L4 arithmetic is the hand ledger: quantity*(exit-entry), both entry and
    # terminal-close notional charged once, then the explicit 0.1% fee.
    detail = result.l4_detail.iloc[0]
    holdings = result.l4_holdings
    gross_pnl = float(
        (holdings["signed_quantity"] * (holdings["exit_price"] - holdings["entry_price"]))
        .sum()
    )
    charged_notional = float(result.l4_orders["executed_order_notional"].abs().sum())
    assert detail["gross_pnl_usd"] == pytest.approx(gross_pnl)
    assert detail["charged_order_notional"] == pytest.approx(charged_notional)
    expected_gross_pnl = 0.0002861232669878698
    expected_charged_notional = 2.0001328223812855
    expected_cost = 0.0020001328223812854
    expected_net_return = -0.0017140095553934156
    assert gross_pnl == pytest.approx(expected_gross_pnl)
    assert charged_notional == pytest.approx(expected_charged_notional)
    assert detail["gross_pnl_usd"] == pytest.approx(expected_gross_pnl)
    assert detail["charged_order_notional"] == pytest.approx(expected_charged_notional)
    assert detail["cost_1x"] == pytest.approx(expected_cost)
    assert detail["net_return_1x"] == pytest.approx(expected_net_return)
    assert expected_cost == pytest.approx(expected_charged_notional * 0.001)
    assert expected_net_return == pytest.approx(expected_gross_pnl - expected_cost)
    assert set(result.l4_holdings["symbol"]) == set(FIXTURE_ASSETS)

    # The returned formal artifact has no truth sidecar; the fixture sidecar is
    # only available here for a terminal assertion and never entered the call.
    assert not hasattr(result, "truth_sidecar")
    assert set(dgp.truth_sidecar.effect_trace["candidate_id"]) == {direct_id}
    assert not any("truth" in str(column).lower() for column in result.l1_panel.columns)


def test_known_truth_l0_l4_micro_row_reordering_is_invariant(_micro_l0_l4_case):
    case = _micro_l0_l4_case
    raw = case["raw"]
    reordered = KnownTruthL0L4RawInputV1(
        market_records=raw.market_records.sample(frac=1.0, random_state=17).reset_index(drop=True),
        signal_records=raw.signal_records.sample(frac=1.0, random_state=23).reset_index(drop=True),
        schema_version=raw.schema_version,
        generation_batch=raw.generation_batch,
        asset_symbols=raw.asset_symbols,
    )
    first = _run_micro_l0_l4(case)
    second = _run_micro_l0_l4(case, reordered)
    for field_name in (
        "l0_market_records", "l0_signal_records", "l1_panel", "l2_rank_ic",
        "l2_directions", "l3_summary", "l3_composite", "l3_targets", "l3_ic",
        "l3_bucket", "l3_weights", "l3_diagnostics", "l4_summary", "l4_detail",
        "l4_orders", "l4_holdings",
    ):
        pd.testing.assert_frame_equal(
            getattr(first, field_name).reset_index(drop=True),
            getattr(second, field_name).reset_index(drop=True),
            check_dtype=True,
        )


def test_known_truth_l0_l4_micro_serialization_is_invariant(_micro_l0_l4_case):
    case = _micro_l0_l4_case
    raw = case["raw"]
    serialized = KnownTruthL0L4RawInputV1(
        market_records=pickle.loads(pickle.dumps(raw.market_records, protocol=5)),
        signal_records=pickle.loads(pickle.dumps(raw.signal_records, protocol=5)),
        schema_version=raw.schema_version,
        generation_batch=raw.generation_batch,
        asset_symbols=raw.asset_symbols,
    )
    first = _run_micro_l0_l4(case)
    second = _run_micro_l0_l4(case, serialized)
    for field_name in (
        "l0_market_records", "l0_signal_records", "l1_panel", "l2_rank_ic",
        "l2_directions", "l3_summary", "l3_composite", "l3_targets", "l3_ic",
        "l3_bucket", "l3_weights", "l3_diagnostics", "l4_summary", "l4_detail",
        "l4_orders", "l4_holdings",
    ):
        pd.testing.assert_frame_equal(
            getattr(first, field_name).reset_index(drop=True),
            getattr(second, field_name).reset_index(drop=True),
            check_dtype=True,
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing_null", "candidate grid is incomplete"),
        ("bad_identity", "content identity"),
        ("bad_timestamp", "UTC-aware"),
        ("naive_signal_time", "UTC-aware"),
        ("zero_volume", "OHLCV"),
        ("zero_price", "OHLCV"),
        ("truth_column", "schema"),
        ("missing_market", "minute grid"),
        ("grid_gap", "1-minute grid"),
    ],
)
def test_known_truth_l0_l4_micro_fail_closed_inputs(_micro_l0_l4_case, mutation, match):
    case = _micro_l0_l4_case
    raw = case["raw"]
    market = raw.market_records.copy(deep=True)
    signals = raw.signal_records.copy(deep=True)
    if mutation == "missing_null":
        signals = signals.loc[signals["candidate_id"].ne(CANDIDATES[4])].copy()
    elif mutation == "bad_identity":
        signals.loc[signals.index[0], "content_identity"] = "0" * 64
    elif mutation == "bad_timestamp":
        signals.loc[signals.index[0], "actual_observed_availability"] = pd.NaT
    elif mutation == "naive_signal_time":
        signals["decision_time"] = signals["decision_time"].astype(object)
        signals.loc[signals.index[0], "decision_time"] = pd.Timestamp(
            "2026-01-01T00:00:00"
        )
    elif mutation == "zero_volume":
        market.loc[market.index[0], "volume"] = 0.0
    elif mutation == "zero_price":
        market.loc[market.index[0], "open"] = 0.0
    elif mutation == "truth_column":
        signals["truth_role"] = "direct"
    elif mutation == "missing_market":
        market = market.drop(index=market.index[-1]).reset_index(drop=True)
    elif mutation == "grid_gap":
        market = market.drop(index=market.index[1]).reset_index(drop=True)
    mutated = KnownTruthL0L4RawInputV1(
        market_records=market,
        signal_records=signals,
        schema_version=raw.schema_version,
        generation_batch=raw.generation_batch,
        asset_symbols=raw.asset_symbols,
    )
    with pytest.raises((KeyError, ValueError), match=match):
        _run_micro_l0_l4(case, mutated)
