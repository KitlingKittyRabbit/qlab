import json
from pathlib import Path

from qlab.verification import junit_summary, verify_test_run_receipt


def test_verified_test_receipt_binds_exit_log_and_junit(tmp_path: Path) -> None:
    junit = tmp_path / "tests.xml"
    junit.write_text(
        '<testsuites tests="3" failures="0" errors="0" skipped="1">'
        '<testsuite tests="3" failures="0" errors="0" skipped="1" />'
        "</testsuites>",
        encoding="utf-8",
    )
    log = tmp_path / "tests.log"
    log.write_text("2 passed, 1 skipped, 4 warnings in 1.00s\n", encoding="utf-8")
    from qlab.verification import file_sha256

    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "pass",
                "command": ["pytest", "-q"],
                "exit_code": 0,
                "junit_summary": junit_summary(junit),
                "warning_count": 4,
                "junit_path": str(junit),
                "junit_sha256": file_sha256(junit),
                "log_path": str(log),
                "log_sha256": file_sha256(log),
            }
        ),
        encoding="utf-8",
    )

    verified = verify_test_run_receipt(receipt)
    assert verified["warning_count"] == 4

    log.write_text("changed", encoding="utf-8")
    try:
        verify_test_run_receipt(receipt)
    except ValueError as exc:
        assert "member verification failed" in str(exc)
    else:
        raise AssertionError("mutated test log must fail receipt verification")
