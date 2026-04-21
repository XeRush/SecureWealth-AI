"""
Internal Pydantic models used between engines and agents (not exposed in the API).
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel

from app.models.response import SignalContribution


class RuleEngineInput(BaseModel):
    device_trust_status: bool = False       # default: False (conservative)
    login_to_action_seconds: float = 0.0    # default: 0 (conservative)
    action_amount_inr: float
    amount_90day_avg_inr: float = 0.0       # default: 0 (conservative)
    otp_retry_count: int = 3                # default: 3 (conservative)
    first_time_action: bool = True          # default: True (conservative)
    retry_loop_detected: bool = True        # default: True (conservative)
    cancel_retry_pattern: bool = True       # default: True (conservative)


class RuleEngineOutput(BaseModel):
    risk_score: float
    risk_label: Literal["Low", "Medium", "High"]
    signal_contributions: List[SignalContribution]
    missing_signals: List[str]


class MLClassificationResult(BaseModel):
    spending_category: Literal["conservative", "moderate", "aggressive"]
    confidence: float
    shap_values: Dict[str, float]  # feature_name → shap_value


class MarketDataSnapshot(BaseModel):
    fetched_at: datetime
    equity_index: Optional[float] = None
    interest_rate: Optional[float] = None
    inflation_indicator: Optional[float] = None
    is_stale: bool
