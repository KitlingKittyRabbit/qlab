"""Canonical crypto data infrastructure for qlab.

Path resolution is intentionally lazy so that utility modules under
``qlab.data.crypto`` can be imported in tests or diagnostics without first
configuring a crypto data root.
"""

from .symbol_universe import (
    CORE_SYMBOLS,
    RESEARCH_SYMBOLS_12,
    load_symbol_list_file,
    normalize_symbol_list,
    parse_symbol_csv,
    resolve_target_symbols,
)
from .panel import (
    build_control_panel,
    forward_returns_for_symbol,
    normalize_price_frame,
    panel_forward_returns,
    panel_with_forward_return,
    price_controls_for_symbol,
    rank_standardize_with_nans,
)

_PATH_EXPORTS = {
    "CACHE_DIR",
    "DATA_ROOT",
    "MANIFEST_DIR",
    "RAW_HISTORY_ROOT",
    "TRADE_ENV_PATH",
    "cache_path",
    "ensure_data_dirs",
    "manifest_path",
}

__all__ = sorted(
    _PATH_EXPORTS
    | {
        "CORE_SYMBOLS",
        "RESEARCH_SYMBOLS_12",
        "load_symbol_list_file",
        "normalize_symbol_list",
        "parse_symbol_csv",
        "build_control_panel",
        "forward_returns_for_symbol",
        "normalize_price_frame",
        "panel_forward_returns",
        "panel_with_forward_return",
        "price_controls_for_symbol",
        "rank_standardize_with_nans",
        "resolve_target_symbols",
    }
)


def __getattr__(name: str):
    if name in _PATH_EXPORTS:
        from . import paths

        value = getattr(paths, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
