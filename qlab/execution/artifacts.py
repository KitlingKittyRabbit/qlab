"""Atomic artifact lifecycle: temp -> validate -> checksum -> rename.

Lifecycle: formal qlab infrastructure.
Authority: qlab_research_private issue #14.
May be used for: crash-safe task artifacts and deterministic checksum manifests.
Must not be used for: scientific interpretation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

from qlab.true_oos import sha256_file


def checksum_manifest_frame(files: Sequence[Path]) -> "object":
    """Return the manifest records for files (sorted, deduplicated, absolute)."""
    import pandas as pd

    unique = sorted({value.expanduser().resolve() for value in files})
    missing = [value for value in unique if not value.is_file()]
    if missing:
        raise FileNotFoundError(f"artifact member is missing: {missing[0]}")
    return pd.DataFrame(
        {
            "path": [str(value) for value in unique],
            "sha256": [sha256_file(value) for value in unique],
            "size_bytes": [value.stat().st_size for value in unique],
        }
    )


def verify_checksum_frame(frame: "object", base_dir: Path) -> None:
    """Verify every record of a checksum manifest frame (fail closed)."""
    import pandas as pd

    required = {"path", "sha256", "size_bytes"}
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns):
        raise ValueError("artifact checksum manifest columns are incomplete")
    for row in frame.itertuples(index=False):
        file_path = Path(str(row.path)).expanduser()
        if not file_path.is_absolute():
            file_path = base_dir / file_path
        file_path = file_path.resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"artifact member missing: {file_path}")
        if file_path.stat().st_size != int(row.size_bytes) or sha256_file(file_path) != str(
            row.sha256
        ):
            raise ValueError(f"artifact checksum verification failed: {file_path}")


def verify_artifact_checksums(artifact_dir: Path, manifest_name: str = "sha256.csv") -> None:
    """Verify an already-finalized artifact directory against its checksum manifest."""
    artifact_dir = artifact_dir.expanduser().resolve()
    manifest_path = artifact_dir / manifest_name
    if not manifest_path.is_file():
        raise FileNotFoundError(f"artifact checksum manifest missing: {manifest_path}")
    manifest = _read_manifest(manifest_path)
    for row in manifest.itertuples(index=False):
        file_path = artifact_dir / str(row.path)
        if not file_path.is_file():
            raise FileNotFoundError(f"artifact member missing: {file_path}")
        if file_path.stat().st_size != int(row.size_bytes) or sha256_file(file_path) != str(
            row.sha256
        ):
            raise ValueError(f"artifact checksum verification failed: {file_path}")
    listed = {str(row.path) for row in manifest.itertuples(index=False)}
    present = {
        value.name for value in artifact_dir.iterdir() if value.is_file()
    } - {manifest_name}
    if listed != present:
        raise ValueError("artifact directory contains unexpected or missing members")


def atomic_directory_finalize(temp_dir: Path, final_dir: Path, *, manifest_name: str = "sha256.csv") -> None:
    """Validate the temp directory and atomically rename it into place.

    Never leaves a half-written artifact looking complete: the temp directory is
    only renamed after every member is present and checksum-valid.
    """
    temp_dir = temp_dir.expanduser().resolve()
    final_dir = final_dir.expanduser().resolve()
    if temp_dir == final_dir:
        raise ValueError("artifact temp and final directories must differ")
    if final_dir.exists():
        raise FileExistsError(f"refusing to overwrite finalized artifact: {final_dir}")
    if not temp_dir.is_dir():
        raise FileNotFoundError(f"artifact temp directory is missing: {temp_dir}")
    verify_artifact_checksums(temp_dir, manifest_name=manifest_name)
    temp_dir.rename(final_dir)


def _read_manifest(manifest_path: Path) -> "object":
    import pandas as pd

    manifest = pd.read_csv(manifest_path, low_memory=False, float_precision="round_trip")
    required = {"path", "sha256", "size_bytes"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"invalid artifact checksum manifest: {manifest_path}")
    return manifest


def write_text_atomic(file_path: Path, text: str) -> None:
    """Write text with fsync and atomic rename."""
    file_path = file_path.expanduser().resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp = file_path.with_name(f".{file_path.name}.tmp")
    temp.write_text(text, encoding="utf-8")
    with temp.open("rb") as handle:
        os.fsync(handle.fileno())
    temp.rename(file_path)


def finalize_checksum_manifest(artifact_dir: Path, files: Sequence[Path], *, manifest_name: str = "sha256.csv") -> None:
    """Write the per-file checksum manifest into an artifact directory."""
    artifact_dir = artifact_dir.expanduser().resolve()
    frame = checksum_manifest_frame(files)
    frame["path"] = frame["path"].map(lambda value: str(Path(value).name))
    manifest_path = artifact_dir / manifest_name
    frame.to_csv(manifest_path, index=False)
    verify_artifact_checksums(artifact_dir, manifest_name=manifest_name)