"""Resource-aware deterministic work pool for CPU-heavy estimator tasks.

Lifecycle: formal qlab infrastructure.
Authority: qlab_research_private issue #14.
May be used for: executing independent estimator-fit work units with a CPU/RAM/native-thread
budget and deterministic result collection.
Must not be used for: changing statistical definitions.

Design notes
------------
- Processes (fork on Linux) provide the outer parallelism; each worker runs with the
  parent's controlled native-thread state (threadpoolctl limits applied before fork,
  environment variables set at import).
- Read-only context data is built once in the parent and inherited by fork copy-on-write;
  task arguments carry only small descriptors, never full panels.
- Result collection is deterministic: results are returned in canonical submission order,
  independent of completion order.
- Concurrency = min(workers, logical_cpus // native_threads, ram_budget // per_task_ram).
"""

from __future__ import annotations

import multiprocessing
import os
import queue
from dataclasses import dataclass
from typing import Callable, Sequence

from qlab.execution.resources import (
    ExecutionProfile,
    MachineTopology,
    detect_machine_topology,
    native_thread_limits,
)

_POOL_CONTEXT: dict[str, object] = {}
POOL_START_METHOD = "fork"


@dataclass(frozen=True)
class DependencyTask:
    """One deterministic work unit and its explicit completion prerequisites."""

    task_id: str
    dependencies: tuple[str, ...]
    unit: tuple[Callable[..., object], tuple]


def set_pool_context(context: dict[str, object] | None) -> None:
    """Replace the read-only context inherited by forked workers (parent side only)."""
    global _POOL_CONTEXT
    _POOL_CONTEXT = dict(context) if context is not None else {}


def current_context() -> dict[str, object]:
    """Return the context dict available inside a worker (inherited via fork)."""
    return _POOL_CONTEXT


def _worker_initializer(native_threads: int) -> None:
    # Environment is set for any library imported later inside the worker; the
    # already-imported BLAS/OpenMP state was limited in the parent before fork.
    from qlab.execution.resources import apply_native_thread_environment

    apply_native_thread_environment(int(native_threads))


class WorkPool:
    """Deterministic resource-budgeted process pool."""

    def __init__(
        self,
        *,
        profile: ExecutionProfile,
        topology: MachineTopology | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        self._topology = topology or detect_machine_topology()
        self._profile = profile
        self._concurrency = profile.concurrent_capacity(self._topology)
        # The context inheritance and copy-on-write contract requires Linux fork.
        self._mp_context = multiprocessing.get_context(POOL_START_METHOD)
        set_pool_context(context)
        self._pool: multiprocessing.pool.Pool | None = None

    @property
    def concurrency(self) -> int:
        return self._concurrency

    def __enter__(self) -> "WorkPool":
        with native_thread_limits(self._profile.native_threads):
            self._pool = self._mp_context.Pool(
                processes=self._concurrency,
                initializer=_worker_initializer,
                initargs=(self._profile.native_threads,),
            )
        return self

    def __exit__(self, *exc: object) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def map(self, units: Sequence[tuple[Callable[..., object], tuple]]) -> list[object]:
        """Execute units and return results in canonical submission order."""
        if self._pool is None:
            raise RuntimeError("WorkPool must be used as a context manager")
        if not units:
            return []
        return list(self._pool.map(_dispatch_unit, units, chunksize=1))

    def map_ordered(self, units: Sequence[tuple[Callable[..., object], tuple]]) -> list[object]:
        """Alias of map: deterministic, order-preserving collection."""
        return self.map(units)

    def run_dag(self, tasks: Sequence[DependencyTask]) -> list[object]:
        """Run dependency-ready tasks without introducing a global phase barrier.

        Tasks are submitted in deterministic task-id order whenever capacity is
        available.  Completion order affects only readiness; returned values always
        follow the caller's canonical task sequence.
        """
        if self._pool is None:
            raise RuntimeError("WorkPool must be used as a context manager")
        ordered = list(tasks)
        if not ordered:
            return []
        by_id: dict[str, DependencyTask] = {}
        for task in ordered:
            task_id = str(task.task_id)
            if not task_id or task_id in by_id:
                raise ValueError("dependency task ids must be unique and non-empty")
            dependencies = tuple(str(value) for value in task.dependencies)
            if len(set(dependencies)) != len(dependencies):
                raise ValueError(f"dependency task has duplicate prerequisites: {task_id}")
            if task_id in dependencies:
                raise ValueError(f"dependency task depends on itself: {task_id}")
            by_id[task_id] = DependencyTask(task_id, dependencies, task.unit)
        for task in by_id.values():
            missing = sorted(set(task.dependencies).difference(by_id))
            if missing:
                raise ValueError(
                    f"dependency task {task.task_id} has missing prerequisites: {missing}"
                )

        dependents: dict[str, list[str]] = {task_id: [] for task_id in by_id}
        remaining: dict[str, set[str]] = {}
        for task in by_id.values():
            remaining[task.task_id] = set(task.dependencies)
            for dependency in task.dependencies:
                dependents[dependency].append(task.task_id)
        ready = [task_id for task_id, dependencies in remaining.items() if not dependencies]
        ready.sort()
        submitted: set[str] = set()
        completed: dict[str, object] = {}
        running: set[str] = set()
        completions: queue.Queue[tuple[str, bool, object]] = queue.Queue()

        def submit(task_id: str) -> None:
            task = by_id[task_id]
            submitted.add(task_id)
            running.add(task_id)
            self._pool.apply_async(
                _dispatch_unit,
                (task.unit,),
                callback=lambda value, task_id=task_id: completions.put(
                    (task_id, True, value)
                ),
                error_callback=lambda error, task_id=task_id: completions.put(
                    (task_id, False, error)
                ),
            )

        try:
            while len(completed) < len(by_id):
                while ready and len(running) < self._concurrency:
                    submit(ready.pop(0))
                if not running:
                    raise ValueError("dependency graph contains a cycle")
                task_id, succeeded, value = completions.get()
                running.remove(task_id)
                if not succeeded:
                    raise RuntimeError(f"dependency task failed: {task_id}") from value
                completed[task_id] = value
                for dependent in sorted(dependents[task_id]):
                    remaining[dependent].discard(task_id)
                    if not remaining[dependent] and dependent not in submitted:
                        ready.append(dependent)
                ready.sort()
        except BaseException:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
            raise
        return [completed[str(task.task_id)] for task in ordered]


def _dispatch_unit(unit: tuple[Callable[..., object], tuple]) -> object:
    function, arguments = unit
    return function(*arguments)
