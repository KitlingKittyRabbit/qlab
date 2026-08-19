import os
import time

import pytest

from qlab.execution.pool import WorkPool, current_context, set_pool_context
from qlab.execution.resources import ExecutionProfile, MachineTopology


def _sleep_unit(seconds: float, value: int) -> int:
    time.sleep(seconds)
    return value


def _context_unit() -> str:
    context = current_context()
    return str(context.get("tag"))


def _cpu_unit(value: int) -> int:
    return value * 2


def test_work_pool_returns_results_in_submission_order():
    profile = ExecutionProfile(workers=4, native_threads=1, oversubscription_factor=1.0)
    topology = MachineTopology(
        logical_cpus=max(8, os.cpu_count() or 8),
        physical_cpus=None,
        available_ram_bytes=4 * 1024**3,
    )
    units = [
        (_sleep_unit, (0.2, 1)),
        (_sleep_unit, (0.05, 2)),
        (_sleep_unit, (0.3, 3)),
        (_sleep_unit, (0.1, 4)),
        (_sleep_unit, (0.0, 5)),
    ]
    with WorkPool(profile=profile, topology=topology) as pool:
        results = pool.map(units)
    assert results == [1, 2, 3, 4, 5]
    with WorkPool(profile=profile, topology=topology) as pool:
        assert pool.map_ordered(units) == [1, 2, 3, 4, 5]


def test_work_pool_context_is_inherited_by_forked_workers():
    profile = ExecutionProfile(workers=2, native_threads=1, oversubscription_factor=1.0)
    topology = MachineTopology(
        logical_cpus=max(4, os.cpu_count() or 4),
        physical_cpus=None,
        available_ram_bytes=2 * 1024**3,
    )
    previous = current_context()
    try:
        with WorkPool(
            profile=profile, topology=topology, context={"tag": "shared"}
        ) as pool:
            assert pool.map([(_context_unit, ())] * 3) == ["shared"] * 3
    finally:
        set_pool_context(previous)


def test_work_pool_requires_context_manager():
    profile = ExecutionProfile(workers=1, native_threads=1)
    topology = MachineTopology(logical_cpus=2, physical_cpus=None, available_ram_bytes=1024)
    pool = WorkPool(profile=profile, topology=topology)
    with pytest.raises(RuntimeError, match="context manager"):
        pool.map([(_cpu_unit, (1,))])


def test_work_pool_concurrency_respects_ram_budget():
    profile = ExecutionProfile(
        workers=4,
        native_threads=1,
        per_task_ram_bytes=2 * 1024**3,
        ram_budget_bytes=4 * 1024**3,
        oversubscription_factor=1.0,
    )
    topology = MachineTopology(
        logical_cpus=8,
        physical_cpus=None,
        available_ram_bytes=8 * 1024**3,
    )
    pool = WorkPool(profile=profile, topology=topology)
    assert pool.concurrency == 2


def test_work_pool_empty_map_is_deterministic():
    profile = ExecutionProfile(workers=1, native_threads=1)
    topology = MachineTopology(logical_cpus=2, physical_cpus=None, available_ram_bytes=1024)
    with WorkPool(profile=profile, topology=topology) as pool:
        assert pool.map([]) == []