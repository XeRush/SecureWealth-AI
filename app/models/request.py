"""
Pydantic request models for SecureWealth Twin AI API.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, field_validator, model_validator


class AssetEntry(BaseModel):
    asset_type: Literal["property", "gold", "vehicle", "financial_instrument"]
    value_inr: float  # required; 422 if missing
    description: Optional[str] = None


class TransactionRecord(BaseModel):
    date: date
    amount_inr: float
    category: str
    direction: Literal["credit", "debit"]


class AccountAggregatorPayload(BaseModel):
    bank_name: str
    transactions: List[TransactionRecord]


class UserProfile(BaseModel):
    user_id: str
    consent_token: str
    income_monthly_inr: float
    risk_appetite: Literal["conservative", "moderate", "aggressive"]
    goals: List[str]
    assets: List[AssetEntry]
    liabilities_inr: float
    transactions: List[TransactionRecord]
    account_aggregator_data: Optional[AccountAggregatorPayload] = None


class AnalyzeUserRequest(BaseModel):
    profile: UserProfile


class WhatIfScenario(BaseModel):
    label: str
    monthly_savings_rate_inr: Optional[float] = None
    investment_allocation: Optional[Dict[str, float]] = None
    time_horizon_months: Optional[int] = None


class SimulateRequest(BaseModel):
    consent_token: str
    current_assets_inr: float
    monthly_savings_rate_inr: float  # must be >= 0
    investment_allocation: Dict[str, float]  # asset class → fraction, must sum to 1.0
    target_goal_inr: float
    time_horizon_months: int  # must be > 0
    what_if_scenarios: Optional[List[WhatIfScenario]] = None

    @field_validator("monthly_savings_rate_inr")
    @classmethod
    def savings_rate_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("monthly_savings_rate_inr must be >= 0")
        return v

    @field_validator("time_horizon_months")
    @classmethod
    def time_horizon_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("time_horizon_months must be > 0")
        return v

    @model_validator(mode="after")
    def allocation_sums_to_one(self) -> "SimulateRequest":
        total = sum(self.investment_allocation.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"investment_allocation values must sum to 1.0, got {total:.6f}"
            )
        return self


class BehavioralFlags(BaseModel):
    retry_loop_detected: bool = False
    cancel_retry_pattern: bool = False


class RiskCheckRequest(BaseModel):
    consent_token: str
    user_id: str
    action_type: str
    action_amount_inr: float
    device_trust_status: Optional[bool] = None
    login_to_action_seconds: Optional[float] = None
    amount_90day_avg_inr: Optional[float] = None
    otp_retry_count: Optional[int] = None
    first_time_action: Optional[bool] = None
    behavioral_flags: Optional[BehavioralFlags] = None


class DecisionRequest(BaseModel):
    consent_token: str
    user_id: str
    action_type: str
    action_amount_inr: float
    device_trust_status: Optional[bool] = None
    login_to_action_seconds: Optional[float] = None
    amount_90day_avg_inr: Optional[float] = None
    otp_retry_count: Optional[int] = None
    first_time_action: Optional[bool] = None
    behavioral_flags: Optional[BehavioralFlags] = None


class ChatRequest(BaseModel):
    consent_token: str
    user_id: str
    message: str
    profile_context: Optional[UserProfile] = None
