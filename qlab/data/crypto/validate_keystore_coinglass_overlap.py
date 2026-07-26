from __future__ import annotations

"""Lifecycle: candidate.

Compare candidate KeyStore/CoinGlass v4 cache payloads with frozen official
CoinGlass cache payloads. This validates source replacement; it does not select
or approve factors.
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in (None, ""):
    PACKAGE_ROOT = Path(__file__).resolve().parents[3]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))

from qlab.data.crypto.paths import cache_path, manifest_path  # noqa: E402


OLD_CACHE_BY_INTERVAL = {
    "1h": [],
    "2h": [],
    "4h": ["coinglass_4h_cache.pkl", "coinglass_v4_4h_cache.pkl"],
    "6h": ["coinglass_6h_cache.pkl", "coinglass_v4_6h_cache.pkl"],
    "8h": [],
    "12h": ["coinglass_12h_cache.pkl", "coinglass_v4_12h_cache.pkl"],
    "1d": ["coinglass_daily_cache.pkl", "coinglass_v4_1d_cache.pkl"],
}


def load_pickle(path: Path) -> dict[str, pd.DataFrame]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload if isinstance(payload, dict) else {}


def numeric_first(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="float64")
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        numeric = frame.apply(pd.to_numeric, errors="coerce").select_dtypes(include="number")
    if numeric.empty:
        return pd.Series(dtype="float64")
    return numeric.iloc[:, 0]


def compare_frames(old: pd.DataFrame, new: pd.DataFrame) -> dict[str, Any]:
    old_series = numeric_first(old)
    new_series = numeric_first(new)
    common = old_series.dropna().index.intersection(new_series.dropna().index)
    old_aligned = old_series.reindex(common)
    new_aligned = new_series.reindex(common)
    diff = (old_aligned - new_aligned).abs()
    return {
        "overlap_start": common.min() if len(common) else pd.NaT,
        "overlap_end": common.max() if len(common) else pd.NaT,
        "old_rows": len(old),
        "new_rows": len(new),
        "matched_ts": len(common),
        "missing_old_ts": len(new_series.dropna().index.difference(old_series.dropna().index)),
        "missing_new_ts": len(old_series.dropna().index.difference(new_series.dropna().index)),
        "pearson_corr": old_aligned.corr(new_aligned, method="pearson") if len(common) >= 3 else float("nan"),
        "spearman_corr": old_aligned.corr(new_aligned, method="spearman") if len(common) >= 3 else float("nan"),
        "median_abs_diff": diff.median() if not diff.empty else float("nan"),
        "max_abs_diff": diff.max() if not diff.empty else float("nan"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate KeyStore cache overlap against frozen official caches.")
    parser.add_argument("--intervals", default="4h,6h,12h,1d")
    parser.add_argument("--output", default="keystore_coinglass_v4_overlap_validation.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    intervals = [item.strip() for item in args.intervals.split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    for interval in intervals:
        new_payload = load_pickle(cache_path(f"keystore_coinglass_v4_{interval}_cache.pkl"))
        if not new_payload:
            continue
        old_payloads = {
            filename: load_pickle(cache_path(filename))
            for filename in OLD_CACHE_BY_INTERVAL.get(interval, [])
        }
        for new_key, new_frame in new_payload.items():
            for old_filename, old_payload in old_payloads.items():
                old_frame = old_payload.get(new_key)
                if old_frame is None:
                    continue
                rows.append(
                    {
                        "interval": interval,
                        "old_cache_file": old_filename,
                        "new_cache_file": f"keystore_coinglass_v4_{interval}_cache.pkl",
                        "cache_key": new_key,
                        **compare_frames(old_frame, new_frame),
                    }
                )

    output_path = manifest_path(args.output)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
