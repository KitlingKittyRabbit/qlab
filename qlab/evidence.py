"""Evidence bundle generation for reproducible reviews and restart gates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class EvidenceArtifact:
    label: str
    path: str
    size_bytes: int
    sha256: str
    modified_at: str


@dataclass(frozen=True)
class EvidenceBundle:
    created_at: str
    metadata: dict
    artifacts: tuple[EvidenceArtifact, ...]

    def to_dict(self) -> dict:
        return {
            "created_at": self.created_at,
            "metadata": self.metadata,
            "artifacts": [asdict(artifact) for artifact in self.artifacts],
        }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_evidence_bundle(
    artifacts: Mapping[str, str | Path],
    metadata: Mapping[str, object] | None = None,
) -> EvidenceBundle:
    items: list[EvidenceArtifact] = []
    for label, raw_path in artifacts.items():
        path = Path(raw_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"evidence artifact not found: {path}")
        stat = path.stat()
        items.append(
            EvidenceArtifact(
                label=label,
                path=str(path),
                size_bytes=stat.st_size,
                sha256=_hash_file(path),
                modified_at=datetime.fromtimestamp(
                    stat.st_mtime, tz=UTC).isoformat(),
            )
        )

    return EvidenceBundle(
        created_at=datetime.now(tz=UTC).isoformat(),
        metadata=dict(metadata or {}),
        artifacts=tuple(items),
    )


def write_evidence_bundle(
    output_path: str | Path,
    artifacts: Mapping[str, str | Path],
    metadata: Mapping[str, object] | None = None,
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle = build_evidence_bundle(artifacts=artifacts, metadata=metadata)
    destination.write_text(
        json.dumps(bundle.to_dict(), ensure_ascii=False,
                   indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return destination
