"""Typed inference output contract (master spec section 16). Unavailable values stay None; probabilities are validated.

Every forecast's three class probabilities must be finite, in [0, 1] and sum to 1 within PROB_TOL. Nothing in a result
may be invented: fields the pipeline cannot compute are None with a reason in `limitations` or `missing_information`.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

PROB_TOL = 1e-6
HORIZON_KEYS = ("1D", "2D", "5D", "10D", "20D", "1M")


class Instrument(BaseModel):
    symbol: str
    company: Optional[str] = None
    exchange: Optional[str] = None
    forecast_timestamp: str
    asof_session: Optional[str] = None
    reference_price: Optional[float] = None
    reference_price_source: Optional[str] = None
    data_quality: Literal["validated", "partial", "insufficient"]


class ModelInfo(BaseModel):
    architecture_version: str
    checkpoint_ids: list[str] = Field(default_factory=list)
    calibration_status: Literal["calibrated", "uncalibrated", "unavailable"]
    bundle_id: Optional[str] = None
    trained_through: Optional[str] = None


class HorizonForecast(BaseModel):
    up_probability: Optional[float] = None
    down_probability: Optional[float] = None
    sideways_probability: Optional[float] = None
    predicted_direction: Optional[Literal["UP", "DOWN", "SIDEWAYS"]] = None
    expected_return: Optional[float] = None
    return_q10: Optional[float] = None
    return_q90: Optional[float] = None
    uncertainty: Optional[float] = None
    direction_skill_validated: Optional[bool] = None

    @model_validator(mode="after")
    def _probs(self):
        ps = [self.up_probability, self.down_probability, self.sideways_probability]
        if any(p is None for p in ps):
            if any(p is not None for p in ps):
                raise ValueError("either all three class probabilities are given or none")
            return self
        for p in ps:
            if not (0.0 - PROB_TOL <= p <= 1.0 + PROB_TOL) or p != p:
                raise ValueError(f"probability out of range: {p}")
        if abs(sum(ps) - 1.0) > PROB_TOL:
            raise ValueError(f"class probabilities sum to {sum(ps)}, not 1")
        return self


class TradePlan(BaseModel):
    decision: Literal["LONG", "SHORT", "NO_TRADE", "NO_SIGNAL"]
    reasons: list[str] = Field(default_factory=list)
    setup_family: Optional[str] = None
    entry_reference: Optional[float] = None
    entry_rule: Optional[str] = None
    max_chase_price: Optional[float] = None
    stop: Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    risk_per_share: Optional[float] = None
    reward_per_share: Optional[float] = None
    gross_rr: Optional[float] = None
    net_rr: Optional[float] = None
    quantity: Optional[int] = None
    position_value: Optional[float] = None
    planned_loss_cash: Optional[float] = None
    holding_sessions: Optional[int] = None
    invalidation: Optional[str] = None
    cost_assumptions: Optional[dict] = None
    rule_ids: list[str] = Field(default_factory=list)
    levels_source: Optional[str] = None

    @model_validator(mode="after")
    def _geometry(self):
        if self.decision in ("LONG", "SHORT"):
            need = [self.entry_reference, self.stop, self.target_1, self.quantity]
            if any(v is None for v in need):
                raise ValueError("a LONG/SHORT plan needs entry, stop, target and quantity")
            if self.decision == "LONG" and not (self.stop < self.entry_reference < self.target_1):
                raise ValueError("long plan must satisfy stop < entry < target")
            if self.decision == "SHORT" and not (self.target_1 < self.entry_reference < self.stop):
                raise ValueError("short plan must satisfy target < entry < stop")
            if self.quantity <= 0:
                raise ValueError("quantity must be positive")
        return self


class InferenceResult(BaseModel):
    instrument: Instrument
    model: ModelInfo
    forecasts: dict[str, HorizonForecast]
    evidence: dict
    confluence: dict
    trade_plans: dict[str, TradePlan]
    limitations: list[str] = Field(default_factory=list)
    mode: Literal["research_paper_trading_only"] = "research_paper_trading_only"

    @model_validator(mode="after")
    def _keys(self):
        if set(self.forecasts) != set(HORIZON_KEYS):
            raise ValueError(f"forecasts must cover exactly {HORIZON_KEYS}")
        return self
