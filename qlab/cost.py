"""
Cost models for different asset classes.

Usage:
    cost = CryptoCost(taker_bps=5)
    rt_cost = cost.round_trip_cost(notional=10000)  # $10

    cost = FxCost(spread_pips=1.5, pip_value=0.0001)
    total = cost.total_cost(notional=100000, holding_days=5)
"""

from dataclasses import dataclass


@dataclass
class CostModel:
    """Base cost model. Subclass and override round_trip_cost()."""

    def round_trip_cost(self, notional: float = 1.0) -> float:
        raise NotImplementedError


@dataclass
class CryptoCost(CostModel):
    """Crypto futures: taker fee on entry + exit."""
    taker_bps: float = 5.0

    def round_trip_cost(self, notional: float = 1.0) -> float:
        return notional * self.taker_bps * 2 / 10_000


@dataclass
class FxCost(CostModel):
    """
    FX cost: bid-ask spread + daily swap.

    Parameters
    ----------
    spread_pips : float
        Typical bid-ask spread in pips.
    pip_value : float
        Value of 1 pip in price terms (0.0001 for most pairs, 0.01 for JPY).
    swap_per_day : float
        Daily swap cost per unit notional (absolute value).
    """
    spread_pips: float = 1.5
    pip_value: float = 0.0001
    swap_per_day: float = 0.0

    def round_trip_cost(self, notional: float = 1.0) -> float:
        """Spread cost only (entry + exit)."""
        return notional * self.spread_pips * self.pip_value

    def swap_cost(self, notional: float, holding_days: int) -> float:
        return notional * abs(self.swap_per_day) * holding_days

    def total_cost(self, notional: float, holding_days: int) -> float:
        """Spread + swap."""
        return self.round_trip_cost(notional) + self.swap_cost(notional, holding_days)
