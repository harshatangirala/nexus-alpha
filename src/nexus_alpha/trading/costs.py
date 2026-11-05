"""Transaction-cost scenarios (TriFrame conventions, data DECISIONS_AND_CONVENTIONS #6): round trip 10 / 15 / 20 bps
(base / stressed / severe) of entry value covering commission, fees, half-spread and slippage; shorts also pay 0.5 % a
year borrow. There are no quote data, so spreads are not observed; the scenarios bound them. Fees are not broken out."""
from __future__ import annotations

from dataclasses import asdict, dataclass

SCENARIOS = {"base": 10.0, "stressed": 15.0, "severe": 20.0}


@dataclass(frozen=True)
class CostModel:
    round_trip_bps: float = 10.0
    borrow_annual: float = 0.005
    scenario: str = "base"

    @classmethod
    def from_scenario(cls, name: str) -> "CostModel":
        return cls(round_trip_bps=SCENARIOS[name], scenario=name)

    def per_share_round_trip(self, price: float) -> float:
        return price * self.round_trip_bps / 1e4

    def borrow(self, price: float, sessions: int) -> float:
        return price * self.borrow_annual * sessions / 252

    def to_dict(self) -> dict:
        return asdict(self)
