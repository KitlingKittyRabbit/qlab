from dataclasses import replace
import math

import numpy as np
import pandas as pd
import pytest

from qlab.data.crypto.binance_um_klines import KLINE_COLUMNS, normalize_klines
from qlab.full_pipeline_simulation import (
    DETERMINISTIC_RANDOM_ADDRESS_VERSION_V1,
    KNOWN_TRUTH_ADMITTED_SYMBOLS_V1,
    MARKET_INPUT_SPINE_ARCHIVE_CONDITION_V1,
    MARKET_INPUT_SPINE_AUTHORITY_V1,
    MARKET_INPUT_SPINE_INNOVATION_DISTRIBUTION_V1,
    MARKET_INPUT_SPINE_LIFECYCLE_V1,
    MARKET_INPUT_SPINE_MAY_BE_USED_FOR_V1,
    MARKET_INPUT_SPINE_MUST_NOT_BE_USED_FOR_V1,
    MARKET_INPUT_SPINE_NATIVE_FREQUENCY_V1,
    MARKET_INPUT_SPINE_RECORD_COLUMNS_V1,
    MARKET_INPUT_SPINE_SCHEMA_V1,
    MARKET_INPUT_SPINE_SOURCE_V1,
    MARKET_INPUT_SPINE_TIME_ORDER_V1,
    MARKET_INPUT_SPINE_VALUE_STATUS_V1,
    RANDOM_STREAM_SPECIFICATION_ARCHIVE_CONDITION_V1,
    RANDOM_STREAM_SPECIFICATION_AUTHORITY_V1,
    RANDOM_STREAM_SPECIFICATION_LIFECYCLE_V1,
    RANDOM_STREAM_SPECIFICATION_MAY_BE_USED_FOR_V1,
    RANDOM_STREAM_SPECIFICATION_MUST_NOT_BE_USED_FOR_V1,
    RANDOM_STREAM_SPECIFICATION_VERSION_V1,
    RandomInformationGroupStreamBindingV1,
    RandomStreamSpecificationRegistryV1,
    RandomStreamSpecificationV1,
    RandomTimeProcessSpecificationV1,
    MarketInputSpineSpecificationV1,
    MarketInputSpineStreamV1,
    generate_market_input_spine_v1,
    validate_market_input_spine_specification_v1,
)


ASSETS = KNOWN_TRUTH_ADMITTED_SYMBOLS_V1
FIXTURE_ASSETS = ASSETS[:2]
R_CORRELATED = ((1.0, 0.5), (0.5, 1.0))
L_CORRELATED = ((1.0, 0.0), (0.5, math.sqrt(0.75)))
R_INDEPENDENT = ((1.0, 0.0), (0.0, 1.0))
L_INDEPENDENT = ((1.0, 0.0), (0.0, 1.0))


GENERIC_PROCESS = RandomTimeProcessSpecificationV1(
    family_id="generic-iid-v1",
    parameter_identity="generic-iid-parameters-v1",
    initialization_identity="generic-zero-init-v1",
    burn_in_steps=0,
    native_frequency=MARKET_INPUT_SPINE_NATIVE_FREQUENCY_V1,
    time_order_identity=MARKET_INPUT_SPINE_TIME_ORDER_V1,
    calibration_identity="generic-calibration-v1",
    calibration_status="unavailable",
    tail_rule_identity="generic-finite-tail-v1",
)
PRICE_PROCESS = RandomTimeProcessSpecificationV1(
    family_id="fixture-price-iid-v1",
    parameter_identity="fixture-price-iid-parameters-v1",
    initialization_identity="fixture-price-zero-init-v1",
    burn_in_steps=1,
    native_frequency=MARKET_INPUT_SPINE_NATIVE_FREQUENCY_V1,
    time_order_identity=MARKET_INPUT_SPINE_TIME_ORDER_V1,
    calibration_identity="fixture-price-calibration-unavailable-v1",
    calibration_status="unavailable",
    tail_rule_identity="fixture-price-finite-tail-v1",
)
VOLUME_PROCESS = RandomTimeProcessSpecificationV1(
    family_id="fixture-volume-ar1-v1",
    parameter_identity="fixture-volume-ar1-phi-half-v1",
    initialization_identity="fixture-volume-zero-init-v1",
    burn_in_steps=1,
    native_frequency=MARKET_INPUT_SPINE_NATIVE_FREQUENCY_V1,
    time_order_identity=MARKET_INPUT_SPINE_TIME_ORDER_V1,
    calibration_identity="fixture-volume-calibration-unavailable-v1",
    calibration_status="unavailable",
    tail_rule_identity="fixture-volume-finite-tail-v1",
)


def _registered_stream(
    stream_id: str,
    stream_kind: str,
    address_group: str,
    *,
    information_group_id: str | None = None,
    process: RandomTimeProcessSpecificationV1 = GENERIC_PROCESS,
) -> RandomStreamSpecificationV1:
    return RandomStreamSpecificationV1(
        stream_id=stream_id,
        stream_kind=stream_kind,
        information_group_id=information_group_id,
        random_address_group_id=address_group,
        innovation_distribution_id=MARKET_INPUT_SPINE_INNOVATION_DISTRIBUTION_V1,
        asset_symbols=ASSETS,
        r_identity=f"{stream_id}-r-v1",
        r_decomposition_identity=f"{stream_id}-decomposition-v1",
        r_calibration_identity=f"{stream_id}-calibration-v1",
        time_process=process,
    )


def _registry() -> RandomStreamSpecificationRegistryV1:
    return RandomStreamSpecificationRegistryV1(
        specification_version=RANDOM_STREAM_SPECIFICATION_VERSION_V1,
        seed_namespace="issue34-market-input-spine-fixture-v1",
        phase="formal",
        address_derivation_version=DETERMINISTIC_RANDOM_ADDRESS_VERSION_V1,
        asset_symbols=ASSETS,
        streams=(
            _registered_stream("base-alpha", "base", "base-alpha", information_group_id="alpha"),
            _registered_stream(
                "measurement-alpha",
                "measurement",
                "measurement-alpha",
                information_group_id="alpha",
            ),
            _registered_stream("null-main", "null", "null-main"),
            _registered_stream("price-main", "price", "price-main", process=PRICE_PROCESS),
            _registered_stream("volume-main", "volume", "volume-main", process=VOLUME_PROCESS),
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


def _stream_spec(
    stream_id: str,
    process_family: str,
    process_family_id: str,
    parameter_identity: str,
    initialization_identity: str,
    process_parameters: tuple[float, ...],
    level_values: tuple[float, ...],
    state_scale: float,
    correlation_matrix: tuple[tuple[float, ...], ...],
    correlation_decomposition: tuple[tuple[float, ...], ...],
) -> MarketInputSpineStreamV1:
    return MarketInputSpineStreamV1(
        stream_id=stream_id,
        process_family=process_family,
        process_family_id=process_family_id,
        parameter_identity=parameter_identity,
        initialization_identity=initialization_identity,
        process_parameters=process_parameters,
        initial_state=(0.0, 0.0),
        level_values=level_values,
        state_scale=state_scale,
        correlation_matrix=correlation_matrix,
        correlation_decomposition=correlation_decomposition,
    )


def _spec(
    *,
    asset_processing_order: tuple[int, ...] = (0, 1),
    start_time: pd.Timestamp = pd.Timestamp("2026-01-01T00:00:00Z"),
    price: MarketInputSpineStreamV1 | None = None,
    volume: MarketInputSpineStreamV1 | None = None,
) -> MarketInputSpineSpecificationV1:
    return MarketInputSpineSpecificationV1(
        schema_version=MARKET_INPUT_SPINE_SCHEMA_V1,
        generation_batch="fixture-market-input-spine-v1",
        seed_namespace="issue34-market-input-spine-fixture-v1",
        phase="formal",
        asset_symbols=FIXTURE_ASSETS,
        start_time=start_time,
        periods=5,
        asset_processing_order=asset_processing_order,
        price=price
        or _stream_spec(
            "price-main",
            "iid",
            PRICE_PROCESS.family_id,
            PRICE_PROCESS.parameter_identity,
            PRICE_PROCESS.initialization_identity,
            (),
            (100.0, 200.0),
            0.01,
            R_CORRELATED,
            L_CORRELATED,
        ),
        volume=volume
        or _stream_spec(
            "volume-main",
            "ar1",
            VOLUME_PROCESS.family_id,
            VOLUME_PROCESS.parameter_identity,
            VOLUME_PROCESS.initialization_identity,
            (0.5,),
            (10.0, 20.0),
            0.1,
            R_INDEPENDENT,
            L_INDEPENDENT,
        ),
        lifecycle=MARKET_INPUT_SPINE_LIFECYCLE_V1,
        authority=MARKET_INPUT_SPINE_AUTHORITY_V1,
        may_be_used_for=MARKET_INPUT_SPINE_MAY_BE_USED_FOR_V1,
        must_not_be_used_for=MARKET_INPUT_SPINE_MUST_NOT_BE_USED_FOR_V1,
        archive_condition=MARKET_INPUT_SPINE_ARCHIVE_CONDITION_V1,
    )


FIXED_STANDARD_INNOVATIONS = {
    "price": np.asarray(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (-1.0, 0.0),
            (0.0, -1.0),
            (1.0, 1.0),
        ),
        dtype=float,
    ),
    "volume": np.asarray(
        (
            (0.0, 0.0),
            (1.0, 2.0),
            (0.0, 0.0),
            (-1.0, 1.0),
            (0.0, -1.0),
            (1.0, 0.0),
        ),
        dtype=float,
    ),
}


def test_market_input_spine_manual_two_asset_five_minute_fixture():
    specification = _spec()
    artifacts = generate_market_input_spine_v1(
        specification,
        _registry(),
        injected_standard_innovations=FIXED_STANDARD_INNOVATIONS,
    )
    records = artifacts.records

    assert list(records.columns) == list(MARKET_INPUT_SPINE_RECORD_COLUMNS_V1)
    assert records.shape == (10, len(MARKET_INPUT_SPINE_RECORD_COLUMNS_V1))
    assert set(records.columns).isdisjoint(
        {"truth", "truth_role", "future_return", "discovery_result"}
    )
    assert set(records["source"]) == {MARKET_INPUT_SPINE_SOURCE_V1}
    assert set(records["value_status"]) == {MARKET_INPUT_SPINE_VALUE_STATUS_V1}

    rho = 0.5
    chol_tail = math.sqrt(0.75)
    expected_price_state = np.asarray(
        (
            (1.0, rho),
            (0.0, chol_tail),
            (-1.0, -rho),
            (0.0, -chol_tail),
            (1.0, rho + chol_tail),
        )
    )
    expected_volume_state = np.asarray(
        (
            (chol_tail, 2.0 * chol_tail),
            (0.5 * chol_tail, chol_tail),
            (0.5 * 0.5 * chol_tail - chol_tail, 0.5 * chol_tail + chol_tail),
            (0.5 * (0.5 * 0.5 * chol_tail - chol_tail),
             0.5 * (0.5 * chol_tail + chol_tail) - chol_tail),
            (0.5 * (0.5 * (0.5 * 0.5 * chol_tail - chol_tail)) + chol_tail,
             0.5 * (0.5 * (0.5 * chol_tail + chol_tail) - chol_tail)),
        )
    )
    expected_prices = np.asarray((100.0, 200.0))
    expected_volumes = np.asarray((10.0, 20.0))
    expected_rows = []
    for period_index in range(5):
        for asset_index in range(2):
            opening = expected_prices[asset_index]
            closing = opening * math.exp(0.01 * expected_price_state[period_index, asset_index])
            expected_rows.append(
                (
                    opening,
                    closing,
                    expected_volumes[asset_index]
                    * math.exp(0.1 * expected_volume_state[period_index, asset_index]),
                )
            )
            expected_prices[asset_index] = closing
    np.testing.assert_allclose(
        records[["open", "close", "volume"]].to_numpy(dtype=float),
        np.asarray(expected_rows),
        rtol=0.0,
        atol=1e-12,
    )
    assert all(len(identity) == 64 for identity in records["content_identity"])
    for symbol in FIXTURE_ASSETS:
        normalize_klines(
            records.loc[records["symbol"] == symbol, list(KLINE_COLUMNS)],
            "1m",
        )


def test_market_input_spine_default_path_is_stable_and_order_invariant():
    registry = _registry()
    first = generate_market_input_spine_v1(_spec(), registry).records
    second = generate_market_input_spine_v1(_spec(), registry).records
    reordered = generate_market_input_spine_v1(
        _spec(asset_processing_order=(1, 0)),
        registry,
    ).records
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first, reordered)
    assert first.iloc[0]["content_identity"] == (
        "fd6604ef2537933749d4688c65b34857bc3c823072cabee3b594aa6f0c0e3ac7"
    )


@pytest.mark.parametrize(
    "changed",
    (
        lambda spec: replace(
            spec,
            price=replace(
                spec.price,
                correlation_decomposition=((1.0, 0.0), (0.4, 0.8)),
            ),
        ),
        lambda spec: replace(spec, volume=replace(spec.volume, level_values=(0.0, 20.0))),
        lambda spec: replace(
            spec,
            start_time=pd.Timestamp("2026-01-01T00:00:30Z"),
        ),
        lambda spec: replace(spec, asset_processing_order=(0, 0)),
        lambda spec: replace(spec, price=replace(spec.price, stream_id="missing-price")),
    ),
)
def test_market_input_spine_rejects_invalid_r_volume_time_order_or_stream(changed):
    with pytest.raises((TypeError, ValueError)):
        validate_market_input_spine_specification_v1(changed(_spec()), _registry())


def test_market_input_spine_rejects_missing_volume_stream_from_registry():
    registry = _registry()
    with pytest.raises(ValueError, match="requires at least one volume stream"):
        validate_market_input_spine_specification_v1(
            _spec(),
            replace(registry, streams=registry.streams[:-1]),
        )


def test_market_input_spine_rejects_missing_formal_kline_field():
    records = generate_market_input_spine_v1(_spec(), _registry()).records
    with pytest.raises(ValueError, match="missing columns"):
        normalize_klines(
            records.loc[records["symbol"] == FIXTURE_ASSETS[0], list(KLINE_COLUMNS)].drop(
                columns=["volume"]
            ),
            "1m",
        )


def test_market_input_spine_rejects_injected_innovation_identity_or_shape():
    with pytest.raises(ValueError, match="exactly price and volume"):
        generate_market_input_spine_v1(
            _spec(),
            _registry(),
            injected_standard_innovations={"price": FIXED_STANDARD_INNOVATIONS["price"]},
        )
    with pytest.raises(ValueError, match="wrong shape"):
        generate_market_input_spine_v1(
            _spec(),
            _registry(),
            injected_standard_innovations={
                "price": FIXED_STANDARD_INNOVATIONS["price"][:-1],
                "volume": FIXED_STANDARD_INNOVATIONS["volume"],
            },
        )


def test_market_input_spine_rejects_noncanonical_cholesky_and_nonpositive_generated_volume():
    bad_cholesky = replace(
        _spec().price,
        correlation_decomposition=((-1.0, 0.0), (0.5, math.sqrt(0.75))),
    )
    with pytest.raises(ValueError, match="canonical Cholesky"):
        validate_market_input_spine_specification_v1(
            replace(_spec(), price=bad_cholesky),
            _registry(),
        )
    with pytest.raises(ValueError, match="strictly positive"):
        generate_market_input_spine_v1(
            _spec(volume=replace(_spec().volume, level_values=(-1.0, 20.0))),
            _registry(),
        )


def test_market_input_spine_supports_ma_process_with_zero_history():
    ma_process = replace(
        PRICE_PROCESS,
        family_id="fixture-price-ma-v1",
        parameter_identity="fixture-price-ma-parameters-v1",
    )
    registry = replace(
        _registry(),
        streams=(
            *_registry().streams[:3],
            replace(_registry().streams[3], time_process=ma_process),
            _registry().streams[4],
        ),
    )
    ma_price = replace(
        _spec().price,
        process_family="ma",
        process_family_id=ma_process.family_id,
        parameter_identity=ma_process.parameter_identity,
        process_parameters=(1.0,),
    )
    artifacts = generate_market_input_spine_v1(
        replace(_spec(), price=ma_price),
        registry,
        injected_standard_innovations=FIXED_STANDARD_INNOVATIONS,
    )
    assert len(artifacts.records) == 10
