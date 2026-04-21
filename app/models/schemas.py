"""
Pydantic schemas for all request/response models across the AI backend.

Organized by domain:
  - User & Profile
  - Transaction
  - Wealth Analysis
  - Risk & Decision
  - Simulation
  - Chat
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ════════════════════════════════════════════
#  Enums
# ════════════════════════════════════════════

class RiskAppetite(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DecisionAction(str, Enum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"


class TransactionCategory(str, Enum):
    FOOD = "food"
    TRANSPORT = "transport"
    SHOPPING = "shopping"
    BILLS = "bills"
    ENTERTAINMENT = "entertainment"
    HEALTH = "health"
    EDUCATION = "education"
    INVESTMENT = "investment"
    SALARY = "salary"
    TRANSFER = "transfer"
    OTHER = "other"


class GoalType(str, Enum):
    EDUCATION = "education"
    HOME = "home"
    RETIREMENT = "retirement"
    EMERGENCY = "emergency"
    TRAVEL = "travel"
    CUSTOM = "custom"


# ════════════════════════════════════════════
#  User & Profile
# ════════════════════════════════════════════

class Asset(BaseModel):
    """A single asset owned by the user."""
    asset_type: str = Field(..., examples=["property", "gold", "vehicle", "stocks", "fd", "ppf"])
    name: str = Field(..., examples=["Flat in Mumbai"])
    value_inr: float = Field(..., ge=0)


class Goal(BaseModel):
    """A financial goal."""
    goal_type: GoalType
    name: str = Field(..., examples=["Daughter's education"])
    target_amount_inr: float = Field(..., gt=0)
    target_date: Optional[str] = Field(None, examples=["2030-06-01"])
    current_savings_inr: float = Field(default=0, ge=0)


class UserProfile(BaseModel):
    """Complete user financial profile sent to the AI."""
    user_id: str
    name: str
    age: int = Field(..., ge=18, le=100)
    monthly_income_inr: float = Field(..., ge=0)
    monthly_expenses_inr: float = Field(default=0, ge=0)
    risk_appetite: Optional[RiskAppetite] = None
    assets: list[Asset] = Field(default_factory=list)
    goals: list[Goal] = Field(default_factory=list)
    liabilities_inr: float = Field(default=0, ge=0)


# ════════════════════════════════════════════
#  Transactions
# ════════════════════════════════════════════

class Transaction(BaseModel):
    """A single financial transaction."""
    txn_id: str
    user_id: str
    amount_inr: float
    category: TransactionCategory
    description: str = ""
    timestamp: datetime
    is_credit: bool = False  # True = income, False = expense


# ════════════════════════════════════════════
#  Wealth Analysis (Response)
# ════════════════════════════════════════════

class SpendingInsight(BaseModel):
    category: TransactionCategory
    total_spent_inr: float
    percentage_of_income: float
    trend: str = Field(..., examples=["increasing", "stable", "decreasing"])
    is_overspending: bool = False


class Recommendation(BaseModel):
    title: str
    description: str
    priority: str = Field(..., examples=["high", "medium", "low"])
    category: str = Field(..., examples=["savings", "investment", "tax", "spending", "protection"])


class WealthAnalysisRequest(BaseModel):
    user_profile: UserProfile
    transactions: list[Transaction] = Field(default_factory=list)


class WealthAnalysisResponse(BaseModel):
    user_id: str
    net_worth_inr: float
    monthly_savings_inr: float
    savings_rate_pct: float
    risk_profile: RiskAppetite
    spending_insights: list[SpendingInsight] = Field(default_factory=list)
    income_trend: str = Field(default="stable")
    recommendations: list[Recommendation] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)


# ════════════════════════════════════════════
#  Risk Check & Decision
# ════════════════════════════════════════════

class DeviceInfo(BaseModel):
    device_id: str
    is_trusted: bool = True
    is_new_device: bool = False


class OTPInfo(BaseModel):
    attempts: int = Field(default=1, ge=1)
    time_since_last_attempt_sec: Optional[float] = None


class BehaviorFlags(BaseModel):
    repeated_clicks: bool = False
    cancel_retry_loops: int = Field(default=0, ge=0)
    sudden_corrections: bool = False


class RiskCheckRequest(BaseModel):
    user_id: str
    action_type: str = Field(..., examples=["sip_start", "large_transfer", "portfolio_rebalance", "fd_break"])
    amount_inr: float = Field(..., ge=0)
    device_info: DeviceInfo
    otp_info: Optional[OTPInfo] = None
    login_to_action_seconds: Optional[float] = None
    is_first_time_action: bool = False
    behavior_flags: Optional[BehaviorFlags] = None
    historical_avg_amount_inr: Optional[float] = None


class RiskSignalDetail(BaseModel):
    signal_name: str
    score_contribution: int
    reason: str


class RiskCheckResponse(BaseModel):
    user_id: str
    risk_score: int = Field(..., ge=0, le=100)
    risk_level: RiskLevel
    signals: list[RiskSignalDetail] = Field(default_factory=list)
    evaluated_at: datetime = Field(default_factory=datetime.now)


class DecisionRequest(BaseModel):
    user_id: str
    risk_score: int = Field(..., ge=0, le=100)
    risk_level: RiskLevel
    action_type: str
    amount_inr: float


class DecisionResponse(BaseModel):
    user_id: str
    action: DecisionAction
    message: str
    cooloff_seconds: Optional[int] = None
    requires_additional_auth: bool = False


# ════════════════════════════════════════════
#  Simulation
# ════════════════════════════════════════════

class SimulationRequest(BaseModel):
    current_savings_inr: float = Field(..., ge=0)
    monthly_contribution_inr: float = Field(default=0, ge=0)
    expected_annual_return_pct: float = Field(default=8.0)
    years: int = Field(default=10, ge=1, le=50)
    goal_amount_inr: Optional[float] = None
    goal_name: Optional[str] = None


class YearlyProjection(BaseModel):
    year: int
    corpus_inr: float
    total_invested_inr: float
    gains_inr: float


class SimulationResponse(BaseModel):
    final_corpus_inr: float
    total_invested_inr: float
    total_gains_inr: float
    goal_name: Optional[str] = None
    goal_achievement_pct: Optional[float] = None
    yearly_projections: list[YearlyProjection] = Field(default_factory=list)
    suggestion: Optional[str] = None


# ════════════════════════════════════════════
#  Chat
# ════════════════════════════════════════════

class ChatRequest(BaseModel):
    user_id: str
    message: str
    context: Optional[dict] = None  # optional user profile / analysis context


class ChatResponse(BaseModel):
    reply: str
    sources: list[str] = Field(default_factory=list)
    disclaimer: str = Field(
        default="This is for informational purposes only and not financial advice. "
                "Past performance does not guarantee future results."
    )
