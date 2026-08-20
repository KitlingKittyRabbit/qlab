"""Execution-layer primitives for resource-bounded deterministic task execution.

Lifecycle: formal qlab infrastructure.
Authority: qlab_research_private issue #14 (L5.5 execution layer).
May be used for: scheduling, task identity, resource control, and artifact lifecycle
for deterministic scientific workloads.
Must not be used for: changing statistical definitions or interpreting research results.
"""

from qlab.execution.artifacts import (
    atomic_directory_finalize,
    checksum_manifest_frame,
    finalize_checksum_manifest,
    verify_artifact_checksums,
    verify_checksum_frame,
    write_text_atomic,
)
from qlab.execution.equivalence import assert_frame_equivalent, canonical_frame_sha256
from qlab.execution.identity import scientific_identity, task_identity
from qlab.execution.pool import DependencyTask, WorkPool, current_context, set_pool_context
from qlab.execution.resources import (
    ExecutionProfile,
    MachineTopology,
    apply_native_thread_environment,
    detect_machine_topology,
    machine_topology_identity,
    native_thread_limits,
)

__all__ = [
    "ExecutionProfile",
    "DependencyTask",
    "MachineTopology",
    "WorkPool",
    "apply_native_thread_environment",
    "atomic_directory_finalize",
    "assert_frame_equivalent",
    "canonical_frame_sha256",
    "checksum_manifest_frame",
    "current_context",
    "detect_machine_topology",
    "machine_topology_identity",
    "finalize_checksum_manifest",
    "native_thread_limits",
    "scientific_identity",
    "set_pool_context",
    "task_identity",
    "verify_artifact_checksums",
    "verify_checksum_frame",
    "write_text_atomic",
]
