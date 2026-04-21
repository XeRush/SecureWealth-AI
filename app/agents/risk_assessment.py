"""
Risk Assessment Agent — orchestrates the Rule Engine and optional LLM
to produce risk scores, decisions, explanations, and remediation steps.

Design notes:
- Does NOT use LangChain AgentExecutor; it is a plain Python class.
- The Rule Engine is the sole authority for allow/warn/block decisions.
- The LLM is used ONLY to generate a plain-language explanation on `warn`.
- All financial decisions are deterministic and never delegated to the LLM.
"""

from __future__ import annotations

from typing import Any, List, Optional

from app.engines.rule_engine import RuleEngine
from app.models.internal import RuleEngineInput
from app.models.request import DecisionRequest, RiskCheckRequest


class RiskAssessmentAgent:
    """
    Orchestrates risk scoring for financial actions.

    Parameters
    ----------
    rule_engine:
        A ``RuleEngine`` instance used for all deterministic scoring.
    llm:
        Optional LangChain-compatible chat model (e.g. ``ChatOpenAI``).
        Used only to generate a plain-language explanation when the
        decision is ``warn``.  If ``None``, ``risk_explanation`` is
        returned as ``None``.
    """

    def __init__(self, rule_engine: RuleEngine, llm: Optional[Any] = None) -> None:
        self._rule_engine = rule_engine
        self._llm = llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(
        self,
        request: RiskCheckRequest,
        missing_signals: Optional[List[str]] = None,
    ) -> dict:
        """
        Compute risk score and label for a financial action.

        Builds ``RuleEngineInput`` from the request (using conservative
        defaults for missing fields), calls the Rule Engine, and returns
        a plain dict suitable for the ``/risk-check`` response.

        Parameters
        ----------
        request:
            Validated ``RiskCheckRequest`` from the API layer.
        missing_signals:
            Pre-populated list of missing signal names.  Any additional
            missing signals detected here are appended.

        Returns
        -------
        dict with keys: risk_score, risk_label, decision, reasons, missing_signals
        """
        if missing_signals is None:
            missing_signals = []

        rule_input, missing_signals = self._build_rule_engine_input(
            request, missing_signals
        )
        output = self._rule_engine.compute(rule_input, missing_signals)
        decision = RuleEngine.score_to_decision(output.risk_score)

        return {
            "risk_score": output.risk_score,
            "risk_label": output.risk_label,
            "decision": decision,
            "reasons": output.signal_contributions,
            "missing_signals": output.missing_signals,
        }

    def decide(
        self,
        request: DecisionRequest,
        missing_signals: Optional[List[str]] = None,
    ) -> dict:
        """
        Compute risk score, decision, and enriched output for ``/decision``.

        Extends ``assess()`` with:
        - ``risk_explanation`` (LLM-generated) when decision is ``warn``
        - ``remediation_steps`` (deterministic lookup) when decision is ``block``

        Parameters
        ----------
        request:
            Validated ``DecisionRequest`` from the API layer.
        missing_signals:
            Pre-populated list of missing signal names.

        Returns
        -------
        dict with keys: risk_score, risk_label, decision, reasons,
        missing_signals, risk_explanation (optional), remediation_steps (optional)
        """
        if missing_signals is None:
            missing_signals = []

        rule_input, missing_signals = self._build_rule_engine_input(
            request, missing_signals
        )
        output = self._rule_engine.compute(rule_input, missing_signals)
        decision = RuleEngine.score_to_decision(output.risk_score)

        result: dict = {
            "risk_score": output.risk_score,
            "risk_label": output.risk_label,
            "decision": decision,
            "reasons": output.signal_contributions,
            "missing_signals": output.missing_signals,
            "risk_explanation": None,
            "remediation_steps": None,
        }

        if decision == "warn":
            result["risk_explanation"] = self._explain_risk(
                score=output.risk_score,
                reasons=[c.description for c in output.signal_contributions if c.contribution > 0],
            )

        if decision == "block":
            result["remediation_steps"] = self._rule_engine.get_remediation_steps()

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_rule_engine_input(
        request: RiskCheckRequest | DecisionRequest,
        missing_signals: List[str],
    ) -> tuple[RuleEngineInput, List[str]]:
        """
        Map a ``RiskCheckRequest`` / ``DecisionRequest`` to a ``RuleEngineInput``,
        substituting conservative defaults for absent optional fields and
        recording each substituted field in ``missing_signals``.
        """
        # device_trust_status — default False (conservative)
        if request.device_trust_status is not None:
            device_trust_status = request.device_trust_status
        else:
            device_trust_status = False
            if "device_trust_status" not in missing_signals:
                missing_signals.append("device_trust_status")

        # login_to_action_seconds — default 0.0 (conservative: very short delta)
        if request.login_to_action_seconds is not None:
            login_to_action_seconds = request.login_to_action_seconds
        else:
            login_to_action_seconds = 0.0
            if "login_to_action_seconds" not in missing_signals:
                missing_signals.append("login_to_action_seconds")

        # action_amount_inr — always required in request
        action_amount_inr = request.action_amount_inr

        # amount_90day_avg_inr — default 0.0 (conservative: unknown avg → max deviation)
        if request.amount_90day_avg_inr is not None:
            amount_90day_avg_inr = request.amount_90day_avg_inr
        else:
            amount_90day_avg_inr = 0.0
            if "amount_90day_avg_inr" not in missing_signals:
                missing_signals.append("amount_90day_avg_inr")

        # otp_retry_count — default 3 (conservative: max retries)
        if request.otp_retry_count is not None:
            otp_retry_count = request.otp_retry_count
        else:
            otp_retry_count = 3
            if "otp_retry_count" not in missing_signals:
                missing_signals.append("otp_retry_count")

        # first_time_action — default True (conservative)
        if request.first_time_action is not None:
            first_time_action = request.first_time_action
        else:
            first_time_action = True
            if "first_time_action" not in missing_signals:
                missing_signals.append("first_time_action")

        # behavioral_flags — default True for both (conservative)
        if request.behavioral_flags is not None:
            retry_loop_detected = request.behavioral_flags.retry_loop_detected
            cancel_retry_pattern = request.behavioral_flags.cancel_retry_pattern
        else:
            retry_loop_detected = True
            cancel_retry_pattern = True

        rule_input = RuleEngineInput(
            device_trust_status=device_trust_status,
            login_to_action_seconds=login_to_action_seconds,
            action_amount_inr=action_amount_inr,
            amount_90day_avg_inr=amount_90day_avg_inr,
            otp_retry_count=otp_retry_count,
            first_time_action=first_time_action,
            retry_loop_detected=retry_loop_detected,
            cancel_retry_pattern=cancel_retry_pattern,
        )

        return rule_input, missing_signals

    def _explain_risk(self, score: float, reasons: List[str]) -> Optional[str]:
        """
        Generate a plain-language explanation for a ``warn`` decision.

        Uses the LLM if available; returns ``None`` otherwise.
        """
        if self._llm is None:
            return None

        reasons_text = "; ".join(reasons) if reasons else "multiple risk signals"
        prompt = (
            f"The financial action has a medium risk score of {score:.1f}. "
            f"The contributing factors are: {reasons_text}. "
            "Explain in 2-3 sentences why this action is flagged as medium risk."
        )

        try:
            response = self._llm.invoke(prompt)
            # Support both plain string responses and LangChain message objects
            if hasattr(response, "content"):
                return str(response.content)
            return str(response)
        except Exception:
            return None
