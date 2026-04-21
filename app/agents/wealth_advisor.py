"""
Wealth Advisor Agent — orchestrates ML classification, trend detection,
net worth computation, RAG retrieval, market data, and SHAP explanation
to produce personalized wealth recommendations.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.engines.market_data import MarketDataClient
from app.engines.wealth_intelligence import (
    BehaviorClassifier,
    NetWorthCalculator,
    TrendDetector,
    build_shap_explanation,
)
from app.models.request import AnalyzeUserRequest
from app.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

_DISCLAIMER = (
    "This analysis is for simulation and informational purposes only "
    "and does not constitute financial advice."
)


def _explain_shap_with_llm(shap_exp, category: str, llm: Optional[Any]) -> str:
    """
    Generate a plain-language SHAP explanation via LLM.

    Falls back to a template string when LLM is None or the call fails.
    """
    fallback = f"Spending behavior classified as {category} based on transaction patterns."

    if llm is None:
        return fallback

    features_summary = ", ".join(
        f"{f.feature_name} ({'+' if f.direction == 'positive' else '-'}{abs(f.shap_value):.4f})"
        for f in shap_exp.top_features
    )
    prompt = (
        f"The user's spending behavior is classified as {category}. "
        f"The top contributing features are: {features_summary}. "
        "Explain in 2-3 sentences what this means for their financial health."
    )

    try:
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        logger.warning("LLM SHAP explanation failed: %s", exc)
        return fallback


class WealthAdvisorAgent:
    """
    Plain Python agent (no LangChain AgentExecutor) that combines:
      - BehaviorClassifier  (classify_behavior)
      - TrendDetector       (detect_trends)
      - NetWorthCalculator  (compute_net_worth)
      - MarketDataClient    (fetch_market_data)
      - RAGPipeline         (retrieve_knowledge)
      - LLM                 (explain_shap — optional)

    Exposes a single ``analyze(request)`` method that returns a dict
    matching the fields of ``AnalyzeUserResponse``.
    """

    def __init__(
        self,
        behavior_classifier: BehaviorClassifier,
        trend_detector: TrendDetector,
        net_worth_calculator: NetWorthCalculator,
        rag_pipeline: RAGPipeline,
        market_data_client: MarketDataClient,
        llm: Optional[Any] = None,
    ) -> None:
        self._behavior_classifier = behavior_classifier
        self._trend_detector = trend_detector
        self._net_worth_calculator = net_worth_calculator
        self._rag_pipeline = rag_pipeline
        self._market_data_client = market_data_client
        self._llm = llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, request: AnalyzeUserRequest) -> dict:
        """
        Run the full wealth advisory pipeline and return a result dict
        whose keys match ``AnalyzeUserResponse`` fields (excluding
        ``trace_id`` and ``user_id``, which are injected by the caller).

        Steps
        -----
        1. Classify spending behavior (BehaviorClassifier)
        2. Detect income/expense trends (TrendDetector)
        3. Compute net worth (NetWorthCalculator)
        4. Fetch market data snapshot (MarketDataClient)
        5. Build SHAP explanation object (build_shap_explanation)
        6. Generate plain-language SHAP explanation (LLM or fallback)
        7. Retrieve RAG knowledge grounding (RAGPipeline)
        8. Build recommendations list
        9. Return assembled result dict
        """
        profile = request.profile

        # 1. Classify behavior
        clf_result = self._behavior_classifier.classify(profile.transactions)

        # 2. Detect trends
        trend_result = self._trend_detector.detect(profile.transactions)

        # 3. Compute net worth
        net_worth = self._net_worth_calculator.compute(
            profile.assets, profile.liabilities_inr
        )

        # 4. Fetch market data
        market_snapshot = self._market_data_client.fetch()

        # 5. Build SHAP explanation object
        shap_exp = build_shap_explanation(clf_result.shap_values)

        # 6. Generate plain-language SHAP explanation
        explanation = _explain_shap_with_llm(
            shap_exp, clf_result.spending_category, self._llm
        )

        # 7. Retrieve RAG knowledge
        rag_result = self._rag_pipeline.answer(
            f"financial advice for {clf_result.spending_category} investor"
        )

        # 8. Build recommendations
        market_recommendations = self._build_market_recommendations(market_snapshot)
        recommendations = [rag_result["answer"]] + market_recommendations

        # 9. Assemble result
        return {
            "spending_category": clf_result.spending_category,
            "net_worth_inr": net_worth,
            "income_trend": trend_result["income_trend"],
            "expense_trend": trend_result["expense_trend"],
            "recommendations": recommendations,
            "shap_explanation": shap_exp,
            "explanation": explanation,
            "market_data_stale": market_snapshot.is_stale,
            "data_quality_warning": trend_result.get("data_quality_warning"),
            "disclaimer": _DISCLAIMER,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_market_recommendations(market_snapshot) -> list:
        """
        Derive plain-language market signal strings from a MarketDataSnapshot.

        Rules:
        - equity_index present and not stale → add index value message
        - interest_rate present → add rate message
        - snapshot is stale → add staleness notice
        """
        recs = []

        if (
            market_snapshot.equity_index is not None
            and not market_snapshot.is_stale
        ):
            recs.append(
                f"Current equity market index: {market_snapshot.equity_index:.2f}"
            )

        if market_snapshot.interest_rate is not None:
            recs.append(
                f"Current interest rate: {market_snapshot.interest_rate:.2f}%"
            )

        if market_snapshot.is_stale:
            recs.append("Note: Market data may be outdated")

        return recs
