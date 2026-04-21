"""
Orchestrator Agent — routes incoming requests to the appropriate sub-agent
or engine and assembles the final response.

Design notes:
- Plain Python class (no LangChain AgentExecutor).
- Logs routing path with trace_id for observability (Requirement 12.2).
- The LLM / Rule Engine authority is preserved inside each sub-agent;
  the Orchestrator never makes financial decisions itself.
"""

from __future__ import annotations

import logging
from typing import Any

from app.agents.risk_assessment import RiskAssessmentAgent
from app.agents.simulation import SimulationAgent
from app.agents.wealth_advisor import WealthAdvisorAgent
from app.models.request import (
    AnalyzeUserRequest,
    ChatRequest,
    DecisionRequest,
    RiskCheckRequest,
    SimulateRequest,
)
from app.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

_CHAT_DISCLAIMER = (
    "This response is for informational purposes only and does not constitute "
    "financial advice. Please consult a certified financial advisor."
)

FINANCIAL_ACTION_KEYWORDS = {
    "invest",
    "transfer",
    "withdraw",
    "buy",
    "sell",
    "purchase",
    "redeem",
}


def _is_financial_action_query(message: str) -> bool:
    """Return True if the message contains a financial action keyword."""
    return any(kw in message.lower() for kw in FINANCIAL_ACTION_KEYWORDS)


class OrchestratorAgent:
    """
    Top-level orchestrator that routes requests to sub-agents and assembles
    final responses.

    Parameters
    ----------
    wealth_advisor:
        Instantiated ``WealthAdvisorAgent``.
    risk_assessment:
        Instantiated ``RiskAssessmentAgent``.
    simulation:
        Instantiated ``SimulationAgent``.
    rag_pipeline:
        Instantiated ``RAGPipeline`` for conversational queries.
    """

    def __init__(
        self,
        wealth_advisor: WealthAdvisorAgent,
        risk_assessment: RiskAssessmentAgent,
        simulation: SimulationAgent,
        rag_pipeline: RAGPipeline,
    ) -> None:
        self._wealth_advisor = wealth_advisor
        self._risk_assessment = risk_assessment
        self._simulation = simulation
        self._rag_pipeline = rag_pipeline

    # ------------------------------------------------------------------
    # Router methods
    # ------------------------------------------------------------------

    def route_to_wealth_advisor(
        self, request: AnalyzeUserRequest, trace_id: str
    ) -> dict:
        """
        Route to the Wealth Advisor Agent.

        Logs: trace_id={trace_id} route=wealth_advisor
        Returns the agent result enriched with ``trace_id`` and ``user_id``.
        """
        logger.info("trace_id=%s route=%s", trace_id, "wealth_advisor")
        result = self._wealth_advisor.analyze(request)
        result["trace_id"] = trace_id
        result["user_id"] = request.profile.user_id
        return result

    def route_to_risk_assessment(
        self, request: RiskCheckRequest, trace_id: str
    ) -> dict:
        """
        Route to the Risk Assessment Agent (risk-check path).

        Logs: trace_id={trace_id} route=risk_assessment
        Returns the agent result enriched with ``trace_id``.
        """
        logger.info("trace_id=%s route=%s", trace_id, "risk_assessment")
        result = self._risk_assessment.assess(request)
        result["trace_id"] = trace_id
        return result

    def route_to_decision(
        self, request: DecisionRequest, trace_id: str
    ) -> dict:
        """
        Route to the Risk Assessment Agent (decision path).

        Logs: trace_id={trace_id} route=decision
        Returns the agent result enriched with ``trace_id``.
        """
        logger.info("trace_id=%s route=%s", trace_id, "decision")
        result = self._risk_assessment.decide(request)
        result["trace_id"] = trace_id
        return result

    def route_to_simulation(
        self, request: SimulateRequest, trace_id: str
    ) -> dict:
        """
        Route to the Simulation Agent.

        Logs: trace_id={trace_id} route=simulation
        Returns the agent result enriched with ``trace_id``.
        """
        logger.info("trace_id=%s route=%s", trace_id, "simulation")
        result = self._simulation.simulate(request)
        result["trace_id"] = trace_id
        return result

    def route_to_conversational(
        self, request: ChatRequest, trace_id: str
    ) -> dict:
        """
        Route to the Conversational Layer (RAG pipeline).

        If the message contains a financial action keyword, also invokes the
        Risk Assessment Agent with a minimal ``RiskCheckRequest`` and includes
        the resulting ``risk_score`` in the response.

        Logs: trace_id={trace_id} route=conversational
        Returns dict with: answer, sources, risk_score (optional), disclaimer,
        trace_id.
        """
        logger.info("trace_id=%s route=%s", trace_id, "conversational")

        rag_result = self._rag_pipeline.answer(request.message)

        risk_score: Any = None
        if _is_financial_action_query(request.message):
            logger.info(
                "trace_id=%s route=%s financial_action_detected=True",
                trace_id,
                "conversational",
            )
            minimal_risk_req = RiskCheckRequest(
                consent_token=request.consent_token,
                user_id=request.user_id,
                action_type="chat_financial_query",
                action_amount_inr=0.0,
            )
            risk_result = self._risk_assessment.assess(minimal_risk_req)
            risk_score = risk_result.get("risk_score")

        return {
            "answer": rag_result["answer"],
            "sources": rag_result["sources"],
            "risk_score": risk_score,
            "disclaimer": _CHAT_DISCLAIMER,
            "trace_id": trace_id,
        }
