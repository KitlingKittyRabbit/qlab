"""qlab — Quantitative research infrastructure."""

from .diagnostics import forward_returns, ic_decay, quantile_returns
from .metrics import sharpe, sortino, max_drawdown, calmar, win_rate, profit_factor
from .point_in_time import (
    PointInTimeSemantics,
    audit_aggregation_semantics,
    validate_entry_timing_contract,
)
from .provenance import (
    CandidateFileValidationResult,
    validate_candidate_frame,
)
from .research_gate import (
    FactorContract,
    FormalRerunGateResult,
    FormalRerunPrerequisites,
    GateArtifact,
    GateIssue,
    ReliabilityReview,
    collect_formal_rerun_artifacts,
    evaluate_formal_rerun_prerequisites,
)
from .signal import (
    ic,
    ic_direction,
    rank_standardize_cross_section,
    rank_standardize_panel_cross_section,
    threshold_signal,
    zscore,
    zscore_fixed,
)
from .walkforward import walk_forward_splits, select_dates

__version__ = "0.1.0"
