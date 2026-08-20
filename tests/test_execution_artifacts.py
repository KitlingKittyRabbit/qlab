import os
import time
from pathlib import Path

import pytest

from qlab.execution.artifacts import (
    atomic_directory_finalize,
    checksum_manifest_frame,
    finalize_checksum_manifest,
    verify_artifact_checksums,
    verify_checksum_frame,
    write_text_atomic,
)


def test_checksum_manifest_frame_is_sorted_and_deduplicated(tmp_path: Path):
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    first.write_text("alpha")
    second.write_text("beta")
    frame = checksum_manifest_frame([second, first, first])
    assert list(frame["path"]) == [str(first), str(second)]
    assert frame["size_bytes"].tolist() == [5, 4]
    assert len(frame["sha256"].iloc[0]) == 64


def test_checksum_manifest_frame_fails_closed_on_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="missing"):
        checksum_manifest_frame([tmp_path / "missing.csv"])


def test_verify_checksum_frame_rejects_mutation(tmp_path: Path):
    member = tmp_path / "member.csv"
    member.write_text("original")
    frame = checksum_manifest_frame([member])
    verify_checksum_frame(frame, tmp_path)
    mutated = frame.copy()
    mutated.loc[0, "size_bytes"] = 999
    with pytest.raises(ValueError, match="checksum verification failed"):
        verify_checksum_frame(mutated, tmp_path)
    member.write_text("changed")
    with pytest.raises(ValueError, match="checksum verification failed"):
        verify_checksum_frame(frame, tmp_path)


def test_verify_checksum_frame_rejects_incomplete_columns(tmp_path: Path):
    member = tmp_path / "member.csv"
    member.write_text("data")
    frame = checksum_manifest_frame([member]).drop(columns=["size_bytes"])
    with pytest.raises(ValueError, match="columns are incomplete"):
        verify_checksum_frame(frame, tmp_path)


def test_atomic_directory_finalize_never_overwrites(tmp_path: Path):
    temp_dir = tmp_path / "task_tmp"
    final_dir = tmp_path / "task_done"
    temp_dir.mkdir()
    (temp_dir / "output.csv").write_text("result")
    finalize_checksum_manifest(temp_dir, [temp_dir / "output.csv"])
    atomic_directory_finalize(temp_dir, final_dir)
    assert not temp_dir.exists()
    assert (final_dir / "output.csv").read_text() == "result"
    verify_artifact_checksums(final_dir)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        atomic_directory_finalize(tmp_path / "other_tmp", final_dir)
    with pytest.raises(ValueError, match="must differ"):
        atomic_directory_finalize(final_dir, final_dir)


def test_atomic_directory_finalize_rejects_incomplete_temp(tmp_path: Path):
    temp_dir = tmp_path / "bad_tmp"
    temp_dir.mkdir()
    (temp_dir / "output.csv").write_text("result")
    manifest = checksum_manifest_frame([temp_dir / "output.csv"])
    manifest.to_csv(temp_dir / "sha256.csv", index=False)
    (temp_dir / "output.csv").write_text("tampered")
    with pytest.raises(ValueError, match="checksum verification failed"):
        atomic_directory_finalize(temp_dir, tmp_path / "bad_done")


def test_write_text_atomic_and_fsync(tmp_path: Path):
    target = tmp_path / "nested" / "file.txt"
    write_text_atomic(target, "hello")
    assert target.read_text(encoding="utf-8") == "hello"
    write_text_atomic(target, "world")
    assert target.read_text(encoding="utf-8") == "world"
    assert not list((target.parent).glob(".file.txt.tmp"))


def test_finalize_checksum_manifest_round_trip(tmp_path: Path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    member = artifact_dir / "member.csv"
    member.write_text("payload")
    finalize_checksum_manifest(artifact_dir, [member])
    verify_artifact_checksums(artifact_dir)
    extra = artifact_dir / "unexpected.csv"
    extra.write_text("extra")
    with pytest.raises(ValueError, match="unexpected or missing members"):
        verify_artifact_checksums(artifact_dir)

def test_execution_package_exports_artifact_surface():
    import qlab.execution as execution

    assert execution.finalize_checksum_manifest is finalize_checksum_manifest
    assert execution.atomic_directory_finalize is atomic_directory_finalize
    assert execution.verify_artifact_checksums is verify_artifact_checksums
    assert execution.checksum_manifest_frame is checksum_manifest_frame
    assert execution.verify_checksum_frame is verify_checksum_frame
