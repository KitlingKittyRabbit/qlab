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
from typing import Callable, Sequence

from qlab.execution.resources import (
    ExecutionProfile,
    MachineTopology,
    native_thread_limits,
)

_POOL_CONTEXT: dict[str, object] = {}


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
        self._topology = topology or MachineTopology(
            logical_cpus=max(1, int(os.cpu_count() or 1)),
            physical_cpus=None,
            available_ram_bytes=0,
        )
        self._profile = profile
        self._concurrency = profile.concurrent_capacity(self._topology)
        set_pool_context(context)
        self._pool: multiprocessing.pool.Pool | None = None

    @property
    def concurrency(self) -> int:
        return self._concurrency

    def __enter__(self) -> "WorkPool":
        with native_thread_limits(self._profile.native_threads):
            self._pool = multiprocessing.Pool(
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


def _dispatch_unit(unit: tuple[Callable[..., object], tuple]) -> object:
    function, arguments = unit
    return function(*arguments)