from __future__ import annotations

from pathlib import Path
from typing import Iterable


CORE_SYMBOLS = ["BTC", "ETH", "BNB", "SOL"]

# 第一轮多家族统一扫描直接使用 12 个 Binance USDT 永续主力品种。
RESEARCH_SYMBOLS_12 = [
    "BTC",
    "ETH",
    "BNB",
    "SOL",
    "XRP",
    "ADA",
    "DOGE",
    "LINK",
    "AVAX",
    "TRX",
    "LTC",
    "SUI",
]


def normalize_symbol_list(values: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        symbol = str(raw_value).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(symbol)
    return normalized


def parse_symbol_csv(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return normalize_symbol_list(raw_value.replace("\n", ",").split(","))


def load_symbol_list_file(file_value: str | Path | None) -> list[str]:
    if not file_value:
        return []

    file_path = Path(file_value).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"Target symbol file not found: {file_path}")

    tokens: list[str] = []
    for raw_line in file_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        tokens.extend(line.replace(",", " ").split())
    return normalize_symbol_list(tokens)


def resolve_target_symbols(
    *csv_overrides: str | None,
    file_value: str | Path | None = None,
    default: Iterable[str] | None = None,
) -> list[str]:
    for raw_value in csv_overrides:
        symbols = parse_symbol_csv(raw_value)
        if symbols:
            return symbols

    file_symbols = load_symbol_list_file(file_value)
    if file_symbols:
        return file_symbols

    baseline = default if default is not None else RESEARCH_SYMBOLS_12
    return normalize_symbol_list(baseline)
