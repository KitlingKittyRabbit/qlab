from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    PACKAGE_ROOT = Path(__file__).resolve().parents[3]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))

from qlab.data.crypto.paths import cache_path, ensure_data_dirs, manifest_path


CACHE_PATH = cache_path("crypto_15m_cache.pkl")
SUMMARY_PATH = manifest_path("crypto_liquidity_summary.csv")
LOOKBACK_DAYS = 90


def normalize_symbol_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = pd.DataFrame(frame).copy()
    if normalized.empty or "c" not in normalized.columns or "v" not in normalized.columns:
        return pd.DataFrame(columns=["c", "v"])
    index = pd.DatetimeIndex(normalized.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    normalized.index = index
    normalized["c"] = pd.to_numeric(normalized["c"], errors="coerce")
    normalized["v"] = pd.to_numeric(normalized["v"], errors="coerce")
    return normalized[["c", "v"]].dropna().sort_index()


def summarize_symbol(symbol: str, frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "symbol": symbol,
            "rows": 0,
            "start": pd.NaT,
            "end": pd.NaT,
            "median_daily_dollar_volume_90d": 0.0,
            "mean_daily_dollar_volume_90d": 0.0,
            "min_daily_dollar_volume_90d": 0.0,
        }

    daily = (frame["c"] * frame["v"]).resample("1D").sum().dropna()
    cutoff = daily.index.max() - pd.Timedelta(days=LOOKBACK_DAYS)
    recent = daily[daily.index >= cutoff]
    if recent.empty:
        recent = daily

    return {
        "symbol": symbol,
        "rows": int(len(frame)),
        "start": frame.index.min(),
        "end": frame.index.max(),
        "median_daily_dollar_volume_90d": float(recent.median()),
        "mean_daily_dollar_volume_90d": float(recent.mean()),
        "min_daily_dollar_volume_90d": float(recent.min()),
    }


def main() -> None:
    ensure_data_dirs()
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing price cache: {CACHE_PATH}")

    with open(CACHE_PATH, "rb") as file_handle:
        raw = pickle.load(file_handle)

    rows = [
        summarize_symbol(str(symbol), normalize_symbol_frame(frame))
        for symbol, frame in sorted(raw.items())
    ]
    summary = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    summary.to_csv(SUMMARY_PATH, index=False)

    print(f"Saved {SUMMARY_PATH}")
    if summary.empty:
        print("No liquidity rows generated")
    else:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
