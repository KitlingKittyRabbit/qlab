"""Verified command-run receipts for research infrastructure checks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Mapping, Sequence
import xml.etree.ElementTree as ET


UTC = timezone.utc


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def junit_summary(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise ValueError("JUnit evidence contains no test suites")
    return {
        key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def run_command_with_receipt(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    junit_path: Path,
    receipt_path: Path,
    env_overrides: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Run one command and bind its exit status, log, and JUnit evidence."""
    if not command:
        raise ValueError("verified command must not be empty")
    destinations = (log_path, junit_path, receipt_path)
    if any(path.exists() for path in destinations):
        raise FileExistsError("verified test evidence refuses overwrite")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in (env_overrides or {}).items()})
    started = datetime.now(UTC)
    with log_path.open("wb") as log_handle:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_handle.flush()
        os.fsync(log_handle.fileno())
    finished = datetime.now(UTC)
    if not junit_path.is_file():
        raise FileNotFoundError("verified command did not produce its declared JUnit file")
    summary = junit_summary(junit_path)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    warning_matches = re.findall(r"(\d+) warnings? in ", log_text)
    warning_count = int(warning_matches[-1]) if warning_matches else 0
    status = (
        "pass"
        if completed.returncode == 0 and summary["failures"] == 0 and summary["errors"] == 0
        else "fail"
    )
    receipt: dict[str, object] = {
        "status": status,
        "command": list(command),
        "cwd": str(cwd.resolve()),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "exit_code": int(completed.returncode),
        "junit_summary": summary,
        "warning_count": warning_count,
        "junit_path": str(junit_path.resolve()),
        "junit_sha256": file_sha256(junit_path),
        "log_path": str(log_path.resolve()),
        "log_sha256": file_sha256(log_path),
    }
    temporary = receipt_path.with_name(f".{receipt_path.name}.tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, receipt_path)
    if status != "pass":
        raise RuntimeError(f"verified command failed; receipt={receipt_path}")
    return receipt


def verify_test_run_receipt(receipt_path: Path) -> dict[str, object]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "pass" or int(receipt.get("exit_code", -1)) != 0:
        raise ValueError("test-run receipt is not successful")
    if not isinstance(receipt.get("command"), list) or not receipt["command"]:
        raise ValueError("test-run receipt has no command")
    junit_path = Path(str(receipt["junit_path"]))
    log_path = Path(str(receipt["log_path"]))
    for path, key in ((junit_path, "junit_sha256"), (log_path, "log_sha256")):
        if not path.is_file() or file_sha256(path) != str(receipt[key]):
            raise ValueError(f"test-run receipt member verification failed: {path}")
    summary = junit_summary(junit_path)
    if summary != receipt.get("junit_summary"):
        raise ValueError("test-run receipt JUnit summary changed")
    if summary["failures"] or summary["errors"]:
        raise ValueError("test-run receipt JUnit contains failures")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--junit", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    overrides = {}
    for item in args.env:
        if "=" not in item:
            raise ValueError("--env values must use KEY=VALUE")
        key, value = item.split("=", 1)
        overrides[key] = value
    receipt = run_command_with_receipt(
        command,
        cwd=Path(args.cwd),
        log_path=Path(args.log),
        junit_path=Path(args.junit),
        receipt_path=Path(args.receipt),
        env_overrides=overrides,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
