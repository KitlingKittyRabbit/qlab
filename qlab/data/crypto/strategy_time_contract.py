from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class ContinuousHoldingTimeContract:
    return_horizon: str
    decision_interval: str
    holding_interval: str
    strategy_return_interval: str
    signal_timeframes: tuple[str, ...]
    decision_anchor_minute: int = 0
    execution_delay_minutes: int = 1
    exit_rule: str = "rebalance_at_next_decision"
    data_observed_rule: str = "assumed_available_by_t_plus_1m"
    execution_price_field: str = "binance_um_1m_open"
    score_order: str = "high_score_long_low_score_short"

    def to_manifest_row(self) -> dict[str, object]:
        row = asdict(self)
        row["signal_timeframes"] = ",".join(self.signal_timeframes)
        return row


def _delta(name: str, horizon_deltas: Mapping[str, pd.Timedelta]) -> pd.Timedelta:
    if name not in horizon_deltas:
        raise ValueError(f"Timeframe missing from horizon_deltas: {name}")
    value = pd.Timedelta(horizon_deltas[name])
    if value <= pd.Timedelta(0):
        raise ValueError(f"Timeframe must be positive: {name}")
    return value


def validate_continuous_holding_contract(
    contract: ContinuousHoldingTimeContract,
    horizon_deltas: Mapping[str, pd.Timedelta],
) -> None:
    names = {
        contract.return_horizon,
        contract.decision_interval,
        contract.holding_interval,
        contract.strategy_return_interval,
    }
    if len(names) != 1:
        raise ValueError(
            "Continuous holding requires return_horizon == decision_interval == "
            "holding_interval == strategy_return_interval"
        )
    if contract.exit_rule != "rebalance_at_next_decision":
        raise ValueError("Continuous holding requires rebalance_at_next_decision")
    if contract.execution_delay_minutes != 1:
        raise ValueError("Historical execution contract requires a one-minute delay")
    decision_delta = _delta(contract.decision_interval, horizon_deltas)
    if not contract.signal_timeframes:
        raise ValueError("signal_timeframes must not be empty")
    for timeframe in contract.signal_timeframes:
        signal_delta = _delta(timeframe, horizon_deltas)
        if signal_delta > decision_delta or decision_delta % signal_delta != pd.Timedelta(0):
            raise ValueError(
                f"Signal timeframe {timeframe} is not an exact divisor of decision interval "
                f"{contract.decision_interval}"
            )
    if not 0 <= contract.decision_anchor_minute < 60:
        raise ValueError("decision_anchor_minute must be in [0, 59]")


def validate_decision_phase(
    decision_timestamps: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    contract: ContinuousHoldingTimeContract,
    horizon_deltas: Mapping[str, pd.Timedelta],
) -> None:
    validate_continuous_holding_contract(contract, horizon_deltas)
    index = pd.DatetimeIndex(pd.to_datetime(decision_timestamps, utc=True))
    if index.empty:
        raise ValueError("decision_timestamps must not be empty")
    delta = _delta(contract.decision_interval, horizon_deltas)
    anchor = index.normalize() + pd.Timedelta(minutes=contract.decision_anchor_minute)
    if (((index - anchor) % delta) != pd.Timedelta(0)).any():
        raise ValueError("Decision timestamps are not aligned to the approved UTC phase")


def execution_timestamps(
    signal_bar_end_timestamps: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    contract: ContinuousHoldingTimeContract,
    horizon_deltas: Mapping[str, pd.Timedelta],
) -> pd.DataFrame:
    validate_decision_phase(signal_bar_end_timestamps, contract, horizon_deltas)
    signal_end = pd.DatetimeIndex(pd.to_datetime(signal_bar_end_timestamps, utc=True))
    delay = pd.Timedelta(minutes=contract.execution_delay_minutes)
    horizon = _delta(contract.return_horizon, horizon_deltas)
    return pd.DataFrame(
        {
            "signal_timeframes": ",".join(contract.signal_timeframes),
            "native_bar_end_ts": signal_end,
            "signal_bar_end_ts": signal_end,
            "availability_ts": signal_end + delay,
            "data_observed_ts": signal_end + delay,
            "decision_ts": signal_end,
            "order_submit_ts": signal_end + delay,
            "execution_ts": signal_end + delay,
            "execution_open_time": signal_end + delay,
            "next_execution_ts": signal_end + horizon + delay,
            "return_horizon": contract.return_horizon,
            "decision_interval": contract.decision_interval,
            "holding_interval": contract.holding_interval,
            "strategy_return_interval": contract.strategy_return_interval,
            "exit_rule": contract.exit_rule,
            "execution_price_field": contract.execution_price_field,
            "data_observed_rule": contract.data_observed_rule,
            "score_order": contract.score_order,
        }
    )


def factor_eligibility_manifest(
    registry: pd.DataFrame,
    *,
    horizon: str,
    horizon_deltas: Mapping[str, pd.Timedelta],
    feature_column: str = "feature_name",
    timeframe_column: str = "signal_timeframe",
) -> pd.DataFrame:
    required = {feature_column, timeframe_column}
    missing = required.difference(registry.columns)
    if missing:
        raise ValueError(f"Factor registry missing columns: {sorted(missing)}")
    horizon_delta = _delta(horizon, horizon_deltas)
    rows = []
    for row in registry.itertuples(index=False):
        values = row._asdict()
        timeframe = str(values[timeframe_column])
        signal_delta = _delta(timeframe, horizon_deltas)
        admitted = signal_delta <= horizon_delta and horizon_delta % signal_delta == pd.Timedelta(0)
        rows.append(
            {
                feature_column: values[feature_column],
                timeframe_column: timeframe,
                "return_horizon": horizon,
                "decision_interval": horizon,
                "availability_delay_minutes": 1,
                "common_release_phase": "utc_horizon_boundary",
                "admitted": admitted,
                "reason": "admitted_exact_divisor" if admitted else "excluded_not_exact_divisor",
            }
        )
    return pd.DataFrame(rows)
