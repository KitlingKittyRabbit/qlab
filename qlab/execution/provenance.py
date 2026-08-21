"""Selective tracked-file provenance for execution contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Iterable


def _git(repository_root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def _relative_path(repository_root: Path, relative_path: str) -> Path:
    value = Path(relative_path)
    if value.is_absolute() or not value.parts or ".." in value.parts:
        raise ValueError(f"provenance path must be relative to its repository: {relative_path}")
    resolved = (repository_root / value).resolve()
    if resolved != repository_root and repository_root not in resolved.parents:
        raise ValueError(f"provenance path escapes its repository: {relative_path}")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tracked_file_provenance(
    repository_root: Path,
    relative_paths: Iterable[str],
) -> dict[str, object]:
    """Return clean, tracked worktree content identities for selected files.

    The check intentionally ignores unrelated dirty files.  Every selected file must be
    tracked and byte-identical to `HEAD`; this prevents an execution report from silently
    binding an uncommitted change in an execution or authority path.
    """
    root = repository_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"provenance repository is missing: {root}")
    try:
        actual_root = Path(_git(root, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"provenance repository is not a Git checkout: {root}") from error
    if actual_root != root:
        raise ValueError(f"provenance repository root is not canonical: {root}")

    paths = sorted({_relative_path(root, str(value)) for value in relative_paths}, key=str)
    if not paths:
        raise ValueError("selective provenance requires at least one file")
    files: dict[str, object] = {}
    for relative in paths:
        label = relative.as_posix()
        tracked = _git(root, "ls-files", "--error-unmatch", "--", label, check=False)
        if tracked.returncode != 0 or tracked.stdout.strip() != label:
            raise ValueError(f"provenance file is not Git tracked: {root}/{label}")
        current_path = root / relative
        if not current_path.is_file():
            raise FileNotFoundError(f"provenance file is missing: {current_path}")
        clean = _git(root, "diff", "--quiet", "HEAD", "--", label, check=False)
        if clean.returncode != 0:
            raise ValueError(f"provenance file is dirty relative to HEAD: {root}/{label}")
        current = current_path.read_bytes()
        try:
            head = _git(root, "show", f"HEAD:{label}").stdout.encode("utf-8")
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError(f"provenance file cannot be read from HEAD: {root}/{label}") from error
        files[label] = {
            "tracked": True,
            "content_sha256": _sha256_bytes(current),
            "head_content_sha256": _sha256_bytes(head),
        }
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    return {
        "repository_head": head,
        "files": files,
    }
