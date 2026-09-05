from dataclasses import replace

import pytest

from qlab.full_pipeline_simulation import (
    DETERMINISTIC_RANDOM_ADDRESS_VERSION_V1,
    KNOWN_TRUTH_ADMITTED_SYMBOLS_V1,
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
    validate_random_stream_specification_v1,
)


PROCESS = RandomTimeProcessSpecificationV1(
    family_id="iid-v1",
    parameter_identity="iid-parameters-v1",
    initialization_identity="zero-init-v1",
    burn_in_steps=12,
    native_frequency="1min",
    time_order_identity="utc-minute-order-v1",
    calibration_identity="calibration-signal-v1",
    calibration_status="unavailable",
    tail_rule_identity="finite-tail-v1",
)
VOLUME_PROCESS = replace(
    PROCESS,
    family_id="volume-iid-v1",
    parameter_identity="volume-iid-parameters-v1",
    initialization_identity="volume-zero-init-v1",
    calibration_identity="volume-calibration-v1",
    tail_rule_identity="volume-finite-tail-v1",
)


def _stream(
    stream_id: str,
    stream_kind: str,
    random_address_group_id: str,
    *,
    information_group_id: str | None = None,
) -> RandomStreamSpecificationV1:
    return RandomStreamSpecificationV1(
        stream_id=stream_id,
        stream_kind=stream_kind,
        information_group_id=information_group_id,
        random_address_group_id=random_address_group_id,
        innovation_distribution_id=f"{stream_kind}-innovation-v1",
        asset_symbols=KNOWN_TRUTH_ADMITTED_SYMBOLS_V1,
        r_identity=f"{stream_kind}-r-v1",
        r_decomposition_identity=f"{stream_kind}-decomposition-v1",
        r_calibration_identity=f"{stream_kind}-r-calibration-v1",
        time_process=VOLUME_PROCESS if stream_kind == "volume" else PROCESS,
    )


def _registry(
    *,
    streams: tuple[RandomStreamSpecificationV1, ...] | None = None,
    bindings: tuple[RandomInformationGroupStreamBindingV1, ...] | None = None,
    **changes: object,
) -> RandomStreamSpecificationRegistryV1:
    if streams is None:
        streams = (
            _stream("base-alpha", "base", "base-alpha", information_group_id="alpha"),
            _stream(
                "measurement-alpha",
                "measurement",
                "measurement-alpha",
                information_group_id="alpha",
            ),
            _stream("null-main", "null", "null-main"),
            _stream("price-main", "price", "price-main"),
            _stream("volume-main", "volume", "volume-main"),
        )
    if bindings is None:
        bindings = (
            RandomInformationGroupStreamBindingV1(
                information_group_id="alpha",
                base_stream_id="base-alpha",
                proxy_near_alias_base_stream_id="base-alpha",
                measurement_stream_ids=("measurement-alpha",),
            ),
        )
    values = dict(
        specification_version=RANDOM_STREAM_SPECIFICATION_VERSION_V1,
        seed_namespace="issue34-random-streams-v1",
        phase="formal",
        address_derivation_version=DETERMINISTIC_RANDOM_ADDRESS_VERSION_V1,
        asset_symbols=KNOWN_TRUTH_ADMITTED_SYMBOLS_V1,
        streams=streams,
        information_group_bindings=bindings,
        lifecycle=RANDOM_STREAM_SPECIFICATION_LIFECYCLE_V1,
        authority=RANDOM_STREAM_SPECIFICATION_AUTHORITY_V1,
        may_be_used_for=RANDOM_STREAM_SPECIFICATION_MAY_BE_USED_FOR_V1,
        must_not_be_used_for=RANDOM_STREAM_SPECIFICATION_MUST_NOT_BE_USED_FOR_V1,
        archive_condition=RANDOM_STREAM_SPECIFICATION_ARCHIVE_CONDITION_V1,
    )
    values.update(changes)
    return RandomStreamSpecificationRegistryV1(**values)


def test_random_stream_specification_accepts_five_kinds_and_explicit_group_binding():
    registry = _registry()

    assert validate_random_stream_specification_v1(registry) is registry
    assert {stream.stream_kind for stream in registry.streams} == {
        "base",
        "measurement",
        "null",
        "price",
        "volume",
    }
    assert registry.asset_symbols == KNOWN_TRUTH_ADMITTED_SYMBOLS_V1


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("specification_version", "random-stream/v0"),
        ("seed_namespace", ""),
        ("phase", "test"),
        ("address_derivation_version", "other-address/v1"),
        ("lifecycle", "active forever"),
        ("authority", "guessed authority"),
        ("may_be_used_for", "everything"),
        ("must_not_be_used_for", "nothing"),
        ("archive_condition", "never"),
    ),
)
def test_random_stream_specification_rejects_unfrozen_registry_identity(field, value):
    with pytest.raises((TypeError, ValueError)):
        validate_random_stream_specification_v1(_registry(**{field: value}))


def test_random_stream_specification_rejects_duplicate_stream_id():
    registry = _registry()
    duplicate = replace(registry.streams[1], stream_id=registry.streams[0].stream_id)

    with pytest.raises(ValueError, match="duplicate random stream_id"):
        validate_random_stream_specification_v1(
            replace(registry, streams=(registry.streams[0], duplicate, *registry.streams[2:]))
        )


def test_random_stream_specification_rejects_unregistered_binding_reference():
    registry = _registry()
    bad_binding = replace(registry.information_group_bindings[0], base_stream_id="missing")

    with pytest.raises(ValueError, match="unregistered base stream"):
        validate_random_stream_specification_v1(
            replace(registry, information_group_bindings=(bad_binding,))
        )


def test_random_stream_specification_rejects_non_frozen_asset_order():
    registry = _registry(asset_symbols=tuple(reversed(KNOWN_TRUTH_ADMITTED_SYMBOLS_V1)))

    with pytest.raises(ValueError, match="frozen 20-asset order"):
        validate_random_stream_specification_v1(registry)


def test_random_stream_specification_rejects_only_r_identity():
    broken_stream = replace(
        _stream("broken", "base", "broken", information_group_id="alpha"),
        r_decomposition_identity=None,
    )
    registry = _registry(
        streams=(
            broken_stream,
            _stream("measurement-alpha", "measurement", "measurement-alpha",
                    information_group_id="alpha"),
            _stream("null-main", "null", "null-main"),
            _stream("price-main", "price", "price-main"),
            _stream("volume-main", "volume", "volume-main"),
        )
    )

    with pytest.raises(ValueError, match="both r_identity and r_decomposition_identity"):
        validate_random_stream_specification_v1(registry)


def test_random_stream_specification_rejects_only_r_decomposition_identity():
    broken_stream = replace(
        _stream("broken", "base", "broken", information_group_id="alpha"),
        r_identity=None,
    )
    registry = _registry(
        streams=(
            broken_stream,
            _stream("measurement-alpha", "measurement", "measurement-alpha",
                    information_group_id="alpha"),
            _stream("null-main", "null", "null-main"),
            _stream("price-main", "price", "price-main"),
            _stream("volume-main", "volume", "volume-main"),
        )
    )

    with pytest.raises(ValueError, match="both r_identity and r_decomposition_identity"):
        validate_random_stream_specification_v1(registry)


@pytest.mark.parametrize(
    "broken_stream",
    (
        replace(_stream("broken", "base", "broken", information_group_id="alpha"),
                r_identity=None, r_decomposition_identity=None),
        replace(_stream("broken", "base", "broken", information_group_id="alpha"),
                r_calibration_identity=""),
        replace(_stream("broken", "base", "broken", information_group_id="alpha"),
                time_process=replace(PROCESS, family_id="")),
        replace(_stream("broken", "base", "broken", information_group_id="alpha"),
                time_process=replace(PROCESS, parameter_identity="")),
        replace(_stream("broken", "base", "broken", information_group_id="alpha"),
                time_process=replace(PROCESS, initialization_identity="")),
        replace(_stream("broken", "base", "broken", information_group_id="alpha"),
                time_process=replace(PROCESS, burn_in_steps=-1)),
        replace(_stream("broken", "base", "broken", information_group_id="alpha"),
                time_process=replace(PROCESS, native_frequency="")),
        replace(_stream("broken", "base", "broken", information_group_id="alpha"),
                time_process=replace(PROCESS, time_order_identity="")),
        replace(_stream("broken", "base", "broken", information_group_id="alpha"),
                time_process=replace(PROCESS, calibration_identity="")),
        replace(_stream("broken", "base", "broken", information_group_id="alpha"),
                time_process=replace(PROCESS, tail_rule_identity="")),
    ),
)
def test_random_stream_specification_rejects_missing_r_t_or_process_identity(broken_stream):
    registry = _registry(
        streams=(
            broken_stream,
            _stream("measurement-alpha", "measurement", "measurement-alpha",
                    information_group_id="alpha"),
            _stream("null-main", "null", "null-main"),
            _stream("price-main", "price", "price-main"),
            _stream("volume-main", "volume", "volume-main"),
        )
    )

    with pytest.raises((TypeError, ValueError)):
        validate_random_stream_specification_v1(registry)


def test_random_stream_specification_rejects_cross_kind_random_address_sharing():
    registry = _registry()
    shared_measurement = replace(
        registry.streams[1],
        random_address_group_id=registry.streams[0].random_address_group_id,
    )

    with pytest.raises(ValueError, match="across stream kinds"):
        validate_random_stream_specification_v1(
            replace(registry, streams=(registry.streams[0], shared_measurement, *registry.streams[2:]))
        )


@pytest.mark.parametrize(
    "field",
    (
        "r_identity",
        "r_decomposition_identity",
        "r_calibration_identity",
        "time_process.family_id",
        "time_process.parameter_identity",
        "time_process.initialization_identity",
        "time_process.calibration_identity",
        "time_process.tail_rule_identity",
    ),
)
def test_random_stream_specification_rejects_volume_cross_kind_r_t_identity_reuse(field):
    registry = _registry()
    volume = registry.streams[-1]
    other = registry.streams[0]
    if field.startswith("time_process."):
        process_field = field.split(".", 1)[1]
        volume = replace(
            volume,
            time_process=replace(
                volume.time_process,
                **{process_field: getattr(other.time_process, process_field)},
            ),
        )
    else:
        volume = replace(volume, **{field: getattr(other, field)})

    with pytest.raises(ValueError, match="volume stream identity cannot be shared"):
        validate_random_stream_specification_v1(
            replace(registry, streams=(*registry.streams[:-1], volume))
        )


def test_random_stream_specification_rejects_null_sharing_base_or_measurement():
    registry = _registry()
    shared_null = replace(
        registry.streams[2],
        random_address_group_id=registry.streams[0].random_address_group_id,
    )

    with pytest.raises(ValueError, match="across stream kinds"):
        validate_random_stream_specification_v1(
            replace(
                registry,
                streams=(*registry.streams[:2], shared_null, *registry.streams[3:]),
            )
        )


def test_random_stream_specification_rejects_proxy_near_alias_base_mismatch():
    registry = _registry()
    second_base = _stream("base-beta", "base", "base-beta", information_group_id="beta")
    beta_binding = RandomInformationGroupStreamBindingV1(
        information_group_id="beta",
        base_stream_id="base-beta",
        proxy_near_alias_base_stream_id="base-beta",
    )
    bad_alpha_binding = replace(
        registry.information_group_bindings[0],
        proxy_near_alias_base_stream_id="base-beta",
    )

    with pytest.raises(ValueError, match="wrong information group"):
        validate_random_stream_specification_v1(
            replace(
                registry,
                streams=(*registry.streams, second_base),
                information_group_bindings=(bad_alpha_binding, beta_binding),
            )
        )


def test_random_stream_specification_rejects_missing_group_binding():
    registry = _registry(information_group_bindings=())

    with pytest.raises(ValueError, match="cover exactly all registered base groups"):
        validate_random_stream_specification_v1(registry)


def test_random_stream_specification_rejects_wrong_measurement_kind_or_group_reference():
    registry = _registry()
    bad_binding = replace(
        registry.information_group_bindings[0],
        measurement_stream_ids=("null-main",),
    )

    with pytest.raises(ValueError, match="unregistered measurement stream"):
        validate_random_stream_specification_v1(
            replace(registry, information_group_bindings=(bad_binding,))
        )


def test_random_stream_specification_requires_explicit_justification_for_same_kind_sharing():
    registry = _registry()
    second_measurement = _stream(
        "measurement-beta",
        "measurement",
        registry.streams[1].random_address_group_id,
    )

    with pytest.raises(ValueError, match="explicit justification"):
        validate_random_stream_specification_v1(
            replace(registry, streams=(*registry.streams[:2], second_measurement, *registry.streams[2:]))
        )
