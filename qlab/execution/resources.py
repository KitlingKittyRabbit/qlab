"""Machine topology, native-thread control, and the execution resource profile.

Lifecycle: formal qlab infrastructure.
Authority: qlab_research_private issue #14.
May be used for: resource-aware scheduling and oversubscription control.
Must not be used for: tuning any scientific parameter from measured performance.
"""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

NATIVE_THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class MachineTopology:
    logical_cpus: int
    physical_cpus: int | None
    available_ram_bytes: int


def detect_machine_topology() -> "MachineTopology":
    """Detect logical/physical CPU count and available RAM without privileged calls."""
    logical_cpus = max(1, int(os.cpu_count() or 1))
    physical_cpus: int | None = None
    sysconf_cores = getattr(os, "sysconf", lambda _: None)("SC_NPROCESSORS_ONLN")
    if isinstance(sysconf_cores, int) and sysconf_cores > 0:
        physical_cpus = sysconf_cores
    available_ram_bytes = _available_ram_bytes()
    return MachineTopology(
        logical_cpus=logical_cpus,
        physical_cpus=physical_cpus,
        available_ram_bytes=available_ram_bytes,
    )


def _available_ram_bytes() -> int:
    try:
        with Path("/proc/meminfo").open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    kib = int(line.split(":", 1)[1].strip().split()[0])
                    return max(1, kib * 1024)
    except (OSError, ValueError, IndexError):
        pass
    return max(1, int(getattr(os, "sysconf", lambda _: None)("SC_AVPHYS_PAGES") or 0) * 4096)


def apply_native_thread_environment(threads: int) -> None:
    """Set process-level native thread environment variables (idempotent)."""
    count = int(threads)
    if count < 1:
        raise ValueError("native thread count must be at least 1")
    for name in NATIVE_THREAD_ENVIRONMENT_VARIABLES:
        os.environ[name] = str(count)


@contextlib.contextmanager
def native_thread_limits(threads: int) -> Iterator[None]:
    """Apply threadpoolctl limits plus environment variables for the current process.

    On Linux with fork start method, pool children created inside this context inherit
    the limited BLAS/OpenMP state, which is the mechanism that prevents nested
    oversubscription inside worker processes.  Environment variables are restored on
    exit.
    """
    count = int(threads)
    if count < 1:
        raise ValueError("native thread count must be at least 1")
    previous = {name: os.environ.get(name) for name in NATIVE_THREAD_ENVIRONMENT_VARIABLES}
    apply_native_thread_environment(count)
    try:
        from threadpoolctl import threadpool_limits  # type: ignore
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("threadpoolctl is required for native thread control") from error
    try:
        with threadpool_limits(limits=count, user_api="blas"), threadpool_limits(
            limits=count, user_api="openmp"
        ):
            yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class ExecutionProfile:
    """Calibration-selected resource profile for one execution run.

    Parameters are always selected by performance-only calibration on the actual
    machine; they are never hard-coded per machine size.
    """

    def __init__(
        self,
        *,
        workers: int,
        native_threads: int = 1,
        per_task_ram_bytes: int | None = None,
        ram_budget_bytes: int | None = None,
        oversubscription_factor: float = 1.0,
    ) -> None:
        self.workers = int(workers)
        self.native_threads = int(native_threads)
        self.per_task_ram_bytes = (
            None if per_task_ram_bytes is None else int(per_task_ram_bytes)
        )
        self.ram_budget_bytes = None if ram_budget_bytes is None else int(ram_budget_bytes)
        self.oversubscription_factor = float(oversubscription_factor)
        if self.workers < 1 or self.native_threads < 1:
            raise ValueError("workers and native_threads must be at least 1")
        if not 0.0 < self.oversubscription_factor <= 4.0:
            raise ValueError("oversubscription_factor must be in (0, 4]")
        if self.per_task_ram_bytes is not None and self.per_task_ram_bytes < 1:
            raise ValueError("per_task_ram_bytes must be positive")
        if self.ram_budget_bytes is not None and self.ram_budget_bytes < 1:
            raise ValueError("ram_budget_bytes must be positive")

    def _effective_capacity(self, topology: MachineTopology) -> int:
        capacity = self.workers
        capacity = min(capacity, max(1, topology.logical_cpus // self.native_threads))
        if self.per_task_ram_bytes is not None and self.ram_budget_bytes is not None:
            capacity = min(
                capacity,
                max(1, self.ram_budget_bytes // self.per_task_ram_bytes),
            )
        return capacity

    def validate(self, topology: MachineTopology) -> None:
        """Fail closed when the effective concurrency would oversubscribe the machine."""
        allowed_threads = int(topology.logical_cpus * self.oversubscription_factor)
        effective_threads = self._effective_capacity(topology) * self.native_threads
        if effective_threads > allowed_threads:
            raise ValueError(
                "execution profile oversubscribes the machine: "
                f"effective concurrency {self._effective_capacity(topology)} workers "
                f"x {self.native_threads} native threads = {effective_threads} threads "
                f"but only {allowed_threads} allowed on {topology.logical_cpus} logical CPUs"
            )

    def concurrent_capacity(self, topology: MachineTopology) -> int:
        """Effective concurrency after CPU, thread, and RAM budgets."""
        self.validate(topology)
        return self._effective_capacity(topology)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "qlab_execution_profile_v1",
            "workers": self.workers,
            "native_threads": self.native_threads,
            "per_task_ram_bytes": self.per_task_ram_bytes,
            "ram_budget_bytes": self.ram_budget_bytes,
            "oversubscription_factor": self.oversubscription_factor,
        }

    @classmethod
    def from_dict(cls, raw: object) -> "ExecutionProfile":
        if not isinstance(raw, dict) or raw.get("schema") != "qlab_execution_profile_v1":
            raise ValueError("execution profile schema is unsupported")
        return cls(
            workers=int(raw["workers"]),
            native_threads=int(raw["native_threads"]),
            per_task_ram_bytes=(
                None if raw.get("per_task_ram_bytes") is None else int(raw["per_task_ram_bytes"])
            ),
            ram_budget_bytes=(
                None if raw.get("ram_budget_bytes") is None else int(raw["ram_budget_bytes"])
            ),
            oversubscription_factor=float(raw.get("oversubscription_factor", 1.0)),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n"