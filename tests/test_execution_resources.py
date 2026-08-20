import json
import os
from pathlib import Path

import pytest

from qlab.execution.resources import (
    NATIVE_THREAD_ENVIRONMENT_VARIABLES,
    ExecutionProfile,
    MachineTopology,
    apply_native_thread_environment,
    detect_machine_topology,
    native_thread_limits,
)


def test_native_thread_environment_is_idempotent():
    previous = {name: os.environ.get(name) for name in NATIVE_THREAD_ENVIRONMENT_VARIABLES}
    try:
        apply_native_thread_environment(3)
        for name in NATIVE_THREAD_ENVIRONMENT_VARIABLES:
            assert os.environ.get(name) == "3"
        apply_native_thread_environment(3)
        for name in NATIVE_THREAD_ENVIRONMENT_VARIABLES:
            assert os.environ.get(name) == "3"
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_native_thread_limits_context_restores_environment():
    previous = {name: os.environ.get(name) for name in NATIVE_THREAD_ENVIRONMENT_VARIABLES}
    try:
        with native_thread_limits(2):
            for name in NATIVE_THREAD_ENVIRONMENT_VARIABLES:
                assert os.environ.get(name) == "2"
        assert all(os.environ.get(name) == previous[name] for name in NATIVE_THREAD_ENVIRONMENT_VARIABLES)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_native_thread_counts_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        apply_native_thread_environment(0)
    with pytest.raises(ValueError, match="at least 1"):
        with native_thread_limits(0):
            pass


def test_detect_machine_topology_reports_positive_values():
    topology = detect_machine_topology()
    assert topology.logical_cpus >= 1
    assert topology.available_ram_bytes > 0


def test_profile_rejects_invalid_fields():
    with pytest.raises(ValueError, match="at least 1"):
        ExecutionProfile(workers=0)
    with pytest.raises(ValueError, match="at least 1"):
        ExecutionProfile(workers=1, native_threads=0)
    with pytest.raises(ValueError, match="oversubscription_factor"):
        ExecutionProfile(workers=1, oversubscription_factor=0.0)
    with pytest.raises(ValueError, match="oversubscription_factor"):
        ExecutionProfile(workers=1, oversubscription_factor=4.5)
    with pytest.raises(ValueError, match="per_task_ram_bytes"):
        ExecutionProfile(workers=1, per_task_ram_bytes=0)


def test_profile_validate_fails_closed_on_oversubscription():
    topology = MachineTopology(
        logical_cpus=4,
        physical_cpus=4,
        available_ram_bytes=8 * 1024**3,
    )
    profile = ExecutionProfile(
        workers=4,
        native_threads=2,
        oversubscription_factor=1.0,
    )
    profile.validate(topology)
    assert profile.concurrent_capacity(topology) == 2
    pathological = ExecutionProfile(
        workers=2,
        native_threads=5,
        oversubscription_factor=1.0,
    )
    with pytest.raises(ValueError, match="oversubscribes"):
        pathological.validate(topology)
    with pytest.raises(ValueError, match="oversubscribes"):
        ExecutionProfile(
            workers=8,
            native_threads=2,
            oversubscription_factor=0.25,
        ).validate(
            MachineTopology(logical_cpus=16, physical_cpus=None, available_ram_bytes=1024)
        )


def test_concurrent_capacity_respects_cpu_thread_and_ram_budgets():
    topology = MachineTopology(
        logical_cpus=16,
        physical_cpus=8,
        available_ram_bytes=32 * 1024**3,
    )
    profile = ExecutionProfile(
        workers=16,
        native_threads=2,
        per_task_ram_bytes=4 * 1024**3,
        ram_budget_bytes=32 * 1024**3,
        oversubscription_factor=1.0,
    )
    assert profile.concurrent_capacity(topology) == 8
    small_ram = ExecutionProfile(
        workers=16,
        native_threads=1,
        per_task_ram_bytes=16 * 1024**3,
        ram_budget_bytes=32 * 1024**3,
        oversubscription_factor=1.0,
    )
    assert small_ram.concurrent_capacity(topology) == 2
    no_budget = ExecutionProfile(workers=16, native_threads=1, oversubscription_factor=1.0)
    assert no_budget.concurrent_capacity(topology) == 16


def test_profile_round_trip_json_and_dict():
    profile = ExecutionProfile(
        workers=12,
        native_threads=1,
        per_task_ram_bytes=1024**3,
        ram_budget_bytes=64 * 1024**3,
        oversubscription_factor=1.5,
    )
    restored = ExecutionProfile.from_dict(json.loads(profile.to_json()))
    assert restored.to_dict() == profile.to_dict()
    with pytest.raises(ValueError, match="schema is unsupported"):
        ExecutionProfile.from_dict({"schema": "other", "workers": 1})