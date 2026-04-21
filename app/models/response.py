"""
Pydantic response models for SecureWealth Twin AI API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel


class ShapFeature(BaseModel):
    feature_name: str
    shap_value: float
    direction: Literal["positive", "negative"]


class ShapExplanation(BaseModel):
    top_features: List[ShapFeature]  # top 5 by |shap_value|


class SignalContribution(BaseModel):
    signal_name: str
    contribution: float
    description: str


class DocumentSource(BaseModel):
    title: str
    origin: str
    similarity_score: float


class ScenarioResult(BaseModel):
    label: str
    goal_achievement_probability: float
    projected_wealth_inr: float


class AnalyzeUserResponse(BaseModel):
    trace_id: str
    user_id: str
    spending_category: Literal["conservative", "moderate", "aggressive"]
    net_worth_inr: float
    income_trend: str
    expense_trend: str
    recommendations: List[str]
    shap_explanation: ShapExplanation
    explanation: str
    market_data_stale: bool
    data_quality_warning: Optional[str] = None
    disclaimer: str


class SimulateResponse(BaseModel):
    trace_id: str
    goal_achievement_probability: float  # 0–100
    projected_wealth_inr: float
    scenario_results: Optional[List[ScenarioResult]] = None
    simulation_label: str


class RiskCheckResponse(BaseModel):
    trace_id: str
    risk_score: float  # 0–100
    risk_label: Literal["Low", "Medium", "High"]
    reasons: List[SignalContribution]
    missing_signals: Optional[List[str]] = None


class DecisionResponse(BaseModel):
    trace_id: str
    risk_score: float
    risk_label: Literal["Low", "Medium", "High"]
    decision: Literal["allow", "warn", "block"]
    reasons: List[SignalContribution]
    risk_explanation: Optional[str] = None
    remediation_steps: Optional[List[str]] = None
    missing_signals: Optional[List[str]] = None


class ChatResponse(BaseModel):
    trace_id: str
    answer: str
    sources: List[DocumentSource]
    risk_score: Optional[float] = None
    disclaimer: str


class SubsystemStatus(BaseModel):
    status: Literal["up", "down"]
    latency_ms: Optional[float] = None


class ProviderStatus(BaseModel):
    status: Literal["up", "down", "stale"]
    last_successful_fetch: Optional[datetime] = None


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    subsystems: Dict[str, SubsystemStatus]
    market_data_providers: Dict[str, ProviderStatus]
