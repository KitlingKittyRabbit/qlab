"""Stable deterministic task identity derived from scientific/execution inputs.

Lifecycle: formal qlab infrastructure.
Authority: qlab_research_private issue #14.
May be used for: task ids, claim files, artifact receipts, and deterministic reduction.
Must not be used for: scientific identification or interpretation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Mapping


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def task_identity(*, schema: str, **fields: object) -> str:
    """Return the SHA-256 identity of a canonical task descriptor.

    The identity depends only on the supplied scientific/execution fields. It never
    depends on worker number, process id, hostname, completion order, wall clock,
    or the number of concurrent workers.
    """
    payload = {"identity_schema_version": str(schema), **fields}
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def scientific_identity(*, schema: str, **fields: object) -> str:
    """Return the SHA-256 identity of a frozen scientific configuration."""
    payload = {"identity_schema_version": str(schema), **fields}
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def canonical_dict_sha256(mapping: Mapping[str, object], *, schema: str) -> str:
    """Stable digest of an ordered mapping of identifiers (order-independent)."""
    payload = dict(mapping)
    if "schema" in payload:
        payload["mapping_schema"] = payload.pop("schema")
    return scientific_identity(schema=schema, **payload)
