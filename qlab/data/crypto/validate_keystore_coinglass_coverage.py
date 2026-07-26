from __future__ import annotations

"""Lifecycle: candidate.

Generate coverage manifests for candidate KeyStore/CoinGlass v4 raw history.
This is a data-quality artifact, not a research conclusion.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in (None, ""):
    PACKAGE_ROOT = Path(__file__).resolve().parents[3]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))

from qlab.data.crypto.keystore_coinglass_endpoints import ENDPOINTS_BY_NAME  # noqa: E402
from qlab.data.crypto.paths import RAW_HISTORY_ROOT, manifest_path  # noqa: E402


METADATA_COLUMNS = {
    "ts",
    "api_version",
    "source",
    "interval",
    "symbol",
    "endpoint",
    "path",
    "parser",
    "migration_type",
    "fetched_at",
}


def infer_time_step_ok(index: pd.Series, interval: str) -> bool:
    if len(index) < 3:
        return True
    expected = pd.to_timedelta(interval.replace("d", "D"))
    diffs = index.sort_values().diff().dropna()
    if diffs.empty:
        return True
    return bool((diffs == expected).mean() >= 0.95)


def summarize_file(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    if "ts" not in frame.columns:
        return {"file": str(path), "rows": len(frame), "error": "missing ts"}
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    endpoint = str(frame["endpoint"].dropna().iloc[-1]) if "endpoint" in frame.columns and frame["endpoint"].notna().any() else path.stem.split("_", 1)[-1]
    symbol = str(frame["symbol"].dropna().iloc[-1]) if "symbol" in frame.columns and frame["symbol"].notna().any() else path.stem.split("_", 1)[0]
    interval = str(frame["interval"].dropna().iloc[-1]) if "interval" in frame.columns and frame["interval"].notna().any() else path.parent.name
    spec = ENDPOINTS_BY_NAME.get(endpoint)
    data_columns = [column for column in frame.columns if column not in METADATA_COLUMNS]
    null_rate = float(frame[data_columns].isna().mean().mean()) if data_columns else 1.0
    required = list(spec.required_columns) if spec else []
    required_present = all(column in frame.columns for column in required)
    return {
        "file": str(path),
        "endpoint": endpoint,
        "symbol": symbol,
        "interval": interval,
        "rows": len(frame),
        "start": frame["ts"].min(),
        "end": frame["ts"].max(),
        "required_columns_present": required_present,
        "required_columns": ",".join(required),
        "null_rate": null_rate,
        "duplicate_ts_count": int(frame["ts"].duplicated().sum()),
        "monotonic_ts": bool(frame["ts"].is_monotonic_increasing),
        "time_step_consistency": infer_time_step_ok(frame["ts"], interval),
        "native_interval_supported": bool(spec.supports_interval(interval)) if spec else False,
        "error": "",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate KeyStore raw-history coverage.")
    parser.add_argument("--raw-root", default=str(RAW_HISTORY_ROOT / "keystore_v4"))
    parser.add_argument("--output", default="keystore_coinglass_v4_endpoint_coverage.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_root = Path(args.raw_root)
    rows = [summarize_file(path) for path in sorted(raw_root.glob("*/*.csv"))]
    output_path = manifest_path(args.output)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
