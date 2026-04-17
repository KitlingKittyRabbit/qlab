"""qlab — Quantitative research infrastructure."""

from .metrics import sharpe, sortino, max_drawdown, calmar, win_rate, profit_factor
from .signal import zscore, zscore_fixed, ic, ic_direction, threshold_signal
from .walkforward import walk_forward_splits, select_dates

__version__ = "0.1.0"
