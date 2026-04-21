"""
FastAPI router for SecureWealth Twin AI.

Endpoints:
  POST /analyze-user  — Wealth Advisor Agent (10s SLA)
  POST /simulate      — Simulation Agent (15s SLA)
  POST /risk-check    — Risk Assessment Agent (3s SLA)
  POST /decision      — Risk Assessment Agent (3s SLA)
  POST /chat          — RAG Pipeline + optional Risk Assessment (8s SLA)
  GET  /health        — subsystem and provider status

SLA enforcement uses asyncio.wait_for with run_in_executor so that
synchronous agent calls don't block the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from app.models.request import (
    AnalyzeUserRequest,
    ChatRequest,
    DecisionRequest,
    RiskCheckRequest,
    SimulateRequest,
)
from app.models.response import (
    AnalyzeUserResponse,
    ChatResponse,
    DecisionResponse,
    HealthResponse,
    ProviderStatus,
    RiskCheckResponse,
    ScenarioResult,
    SimulateResponse,
    SubsystemStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Lazy singleton orchestrator
# ---------------------------------------------------------------------------

_orchestrator: Optional[object] = None  # type: OrchestratorAgent at runtime


def get_orchestrator():
    """Return the module-level OrchestratorAgent, building it on first call."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = _build_orchestrator()
    return _orchestrator


def _build_orchestrator():
    """Instantiate and wire all components into an OrchestratorAgent."""
    from app.agents.orchestrator import OrchestratorAgent
    from app.agents.risk_assessment import RiskAssessmentAgent
    from app.agents.simulation import SimulationAgent
    from app.agents.wealth_advisor import WealthAdvisorAgent
    from app.engines.market_data import MarketDataClient
    from app.engines.rule_engine import RuleEngine
    from app.engines.wealth_intelligence import (
        BehaviorClassifier,
        NetWorthCalculator,
        TrendDetector,
    )
    from app.rag.knowledge_base import build_default_knowledge_base
    from app.rag.pipeline import RAGPipeline

    rule_engine = RuleEngine()
    behavior_classifier = BehaviorClassifier.load_or_train()
    trend_detector = TrendDetector()
    net_worth_calculator = NetWorthCalculator()
    market_data_client = MarketDataClient()

    kb = build_default_knowledge_base()
    rag_pipeline = RAGPipeline(vectorstore=kb.get_store())

    risk_agent = RiskAssessmentAgent(rule_engine=rule_engine)
    simulation_agent = SimulationAgent()
    wealth_advisor = WealthAdvisorAgent(
        behavior_classifier=behavior_classifier,
        trend_detector=trend_detector,
        net_worth_calculator=net_worth_calculator,
        rag_pipeline=rag_pipeline,
        market_data_client=market_data_client,
    )

    return OrchestratorAgent(
        wealth_advisor=wealth_advisor,
        risk_assessment=risk_agent,
        simulation=simulation_agent,
        rag_pipeline=rag_pipeline,
    )


# ---------------------------------------------------------------------------
# Helper: run a synchronous callable with a timeout
# ---------------------------------------------------------------------------

async def _run_with_timeout(fn, timeout: float, trace_id: str):
    """
    Execute *fn* in a thread-pool executor and raise HTTP 504 on timeout.

    Parameters
    ----------
    fn:
        Zero-argument callable wrapping the synchronous agent call.
    timeout:
        SLA in seconds.
    trace_id:
        Included in the 504 error body for traceability.
    """
    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, fn),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "SLA exceeded trace_id=%s timeout=%.1fs", trace_id, timeout
        )
        raise HTTPException(
            status_code=504,
            detail={"error": "timeout", "trace_id": trace_id},
        )


# ---------------------------------------------------------------------------
# POST /analyze-user
# ---------------------------------------------------------------------------

@router.post("/analyze-user", response_model=AnalyzeUserResponse)
async def analyze_user(request_body: AnalyzeUserRequest, request: Request):
    """
    Analyze a user's financial profile and return personalized recommendations.

    SLA: 10 seconds.
    Includes disclaimer and trace_id in every response (Requirements 2.1, 2.5).
    """
    trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
    orchestrator = get_orchestrator()

    result = await _run_with_timeout(
        lambda: orchestrator.route_to_wealth_advisor(request_body, trace_id),
        timeout=10.0,
        trace_id=trace_id,
    )

    return AnalyzeUserResponse(**result)


# ---------------------------------------------------------------------------
# POST /simulate
# ---------------------------------------------------------------------------

@router.post("/simulate", response_model=SimulateResponse)
async def simulate(request_body: SimulateRequest, request: Request):
    """
    Run a wealth simulation and return goal achievement projections.

    SLA: 15 seconds.
    Includes simulation_label in every response (Requirement 5.4).
    """
    trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
    orchestrator = get_orchestrator()

    result = await _run_with_timeout(
        lambda: orchestrator.route_to_simulation(request_body, trace_id),
        timeout=15.0,
        trace_id=trace_id,
    )

    # Convert scenario_results dicts → ScenarioResult objects
    scenario_results = None
    if result.get("scenario_results"):
        scenario_results = [
            ScenarioResult(**s) if isinstance(s, dict) else s
            for s in result["scenario_results"]
        ]

    return SimulateResponse(
        trace_id=result["trace_id"],
        goal_achievement_probability=result["goal_achievement_probability"],
        projected_wealth_inr=result["projected_wealth_inr"],
        scenario_results=scenario_results,
        simulation_label=result["simulation_label"],
    )


# ---------------------------------------------------------------------------
# POST /risk-check
# ---------------------------------------------------------------------------

@router.post("/risk-check", response_model=RiskCheckResponse)
async def risk_check(request_body: RiskCheckRequest, request: Request):
    """
    Score a financial action for risk.

    SLA: 3 seconds (Requirement 6.1).
    """
    trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
    orchestrator = get_orchestrator()

    result = await _run_with_timeout(
        lambda: orchestrator.route_to_risk_assessment(request_body, trace_id),
        timeout=3.0,
        trace_id=trace_id,
    )

    return RiskCheckResponse(
        trace_id=result["trace_id"],
        risk_score=result["risk_score"],
        risk_label=result["risk_label"],
        reasons=result["reasons"],
        missing_signals=result.get("missing_signals"),
    )


# ---------------------------------------------------------------------------
# POST /decision
# ---------------------------------------------------------------------------

@router.post("/decision", response_model=DecisionResponse)
async def decision(request_body: DecisionRequest, request: Request):
    """
    Return an allow / warn / block decision for a financial action.

    SLA: 3 seconds (Requirement 7.1).
    Includes risk_explanation on warn, remediation_steps on block (Requirements 7.5, 7.6).
    """
    trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
    orchestrator = get_orchestrator()

    result = await _run_with_timeout(
        lambda: orchestrator.route_to_decision(request_body, trace_id),
        timeout=3.0,
        trace_id=trace_id,
    )

    dec = result.get("decision")

    return DecisionResponse(
        trace_id=result["trace_id"],
        risk_score=result["risk_score"],
        risk_label=result["risk_label"],
        decision=dec,
        reasons=result["reasons"],
        risk_explanation=result.get("risk_explanation") if dec == "warn" else None,
        remediation_steps=result.get("remediation_steps") if dec == "block" else None,
        missing_signals=result.get("missing_signals"),
    )


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def chat(request_body: ChatRequest, request: Request):
    """
    Answer a natural-language financial question using the RAG pipeline.

    SLA: 8 seconds (Requirement 8.1).
    Includes sources, disclaimer, and risk_score when query relates to a
    financial action (Requirements 8.3, 8.4, 8.5).
    """
    trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
    orchestrator = get_orchestrator()

    result = await _run_with_timeout(
        lambda: orchestrator.route_to_conversational(request_body, trace_id),
        timeout=8.0,
        trace_id=trace_id,
    )

    return ChatResponse(
        trace_id=result["trace_id"],
        answer=result["answer"],
        sources=result["sources"],
        risk_score=result.get("risk_score"),
        disclaimer=result["disclaimer"],
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
async def health():
    """
    Ping all subsystems and return operational status.

    Returns per-subsystem status (up/down) and per-provider connectivity
    status (Requirement 12.4).
    """
    import time

    subsystems: dict = {}
    providers: dict = {}

    # --- Wealth Intelligence Engine ---
    try:
        start = time.monotonic()
        from app.engines.wealth_intelligence import NetWorthCalculator
        NetWorthCalculator()  # lightweight instantiation check
        latency = (time.monotonic() - start) * 1000
        subsystems["wealth_intelligence"] = SubsystemStatus(
            status="up", latency_ms=round(latency, 2)
        )
    except Exception as exc:
        logger.warning("health: wealth_intelligence down: %s", exc)
        subsystems["wealth_intelligence"] = SubsystemStatus(status="down")

    # --- Cyber Protection Engine (Rule Engine) ---
    try:
        start = time.monotonic()
        from app.engines.rule_engine import RuleEngine
        RuleEngine()
        latency = (time.monotonic() - start) * 1000
        subsystems["cyber_protection"] = SubsystemStatus(
            status="up", latency_ms=round(latency, 2)
        )
    except Exception as exc:
        logger.warning("health: cyber_protection down: %s", exc)
        subsystems["cyber_protection"] = SubsystemStatus(status="down")

    # --- Conversational Layer (RAG Pipeline) ---
    try:
        start = time.monotonic()
        orchestrator = get_orchestrator()
        rag_ok = orchestrator._rag_pipeline.ping()
        latency = (time.monotonic() - start) * 1000
        subsystems["conversational_layer"] = SubsystemStatus(
            status="up" if rag_ok else "down",
            latency_ms=round(latency, 2),
        )
    except Exception as exc:
        logger.warning("health: conversational_layer down: %s", exc)
        subsystems["conversational_layer"] = SubsystemStatus(status="down")

    # --- RAG Pipeline (vector store) ---
    try:
        start = time.monotonic()
        rag_ok = orchestrator._rag_pipeline.ping()
        latency = (time.monotonic() - start) * 1000
        subsystems["rag_pipeline"] = SubsystemStatus(
            status="up" if rag_ok else "down",
            latency_ms=round(latency, 2),
        )
    except Exception as exc:
        logger.warning("health: rag_pipeline down: %s", exc)
        subsystems["rag_pipeline"] = SubsystemStatus(status="down")

    # --- Market Data Providers ---
    try:
        from app.engines.market_data import MarketDataClient
        client = MarketDataClient()
        snapshot = client.fetch()
        status = "stale" if snapshot.is_stale else "up"
        providers["yahoo_finance"] = ProviderStatus(
            status=status,
            last_successful_fetch=snapshot.fetched_at if not snapshot.is_stale else None,
        )
        providers["alpha_vantage"] = ProviderStatus(status="up")
    except Exception as exc:
        logger.warning("health: market_data_providers down: %s", exc)
        providers["yahoo_finance"] = ProviderStatus(status="down")
        providers["alpha_vantage"] = ProviderStatus(status="down")

    # Determine overall status
    all_up = all(s.status == "up" for s in subsystems.values())
    any_down = any(s.status == "down" for s in subsystems.values())

    if all_up:
        overall = "healthy"
    elif any_down:
        overall = "degraded"
    else:
        overall = "degraded"

    return HealthResponse(
        status=overall,
        subsystems=subsystems,
        market_data_providers=providers,
    )
