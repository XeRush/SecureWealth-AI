"""
Rule Engine — deterministic, weighted-signal cyber-protection risk scorer.

The Rule Engine is the sole authority for allow/warn/block decisions.
The LLM never overrides its output.
"""

from __future__ import annotations

from typing import List, Optional

from app.models.internal import RuleEngineInput, RuleEngineOutput
from app.models.response import SignalContribution

# ---------------------------------------------------------------------------
# Signal weights (must sum to 100)
# ---------------------------------------------------------------------------
SIGNAL_WEIGHTS = {
    "device_trust_status": 20,
    "login_to_action_delta": 15,
    "amount_vs_90day_avg": 25,
    "otp_retry_count": 15,
    "first_time_action_flag": 10,
    "behavioral_consistency": 15,
}

# ---------------------------------------------------------------------------
# Remediation steps returned on every "block" decision (deterministic lookup)
# ---------------------------------------------------------------------------
REMEDIATION_STEPS: List[str] = [
    "Verify your device by completing device registration",
    "Complete OTP verification with your registered mobile number",
    "Contact customer support if you believe this is an error",
    "Wait 24 hours before retrying if you have exceeded retry limits",
]


class RuleEngine:
    """
    Deterministic weighted-signal risk scorer.

    Usage::

        engine = RuleEngine()
        output = engine.compute(rule_input, missing_signals=["otp_retry_count"])
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        inp: RuleEngineInput,
        missing_signals: Optional[List[str]] = None,
    ) -> RuleEngineOutput:
        """
        Compute risk score, label, signal contributions, and (on block) remediation steps.

        Parameters
        ----------
        inp:
            A ``RuleEngineInput`` whose fields have already been populated with
            conservative defaults for any missing signals by the caller (or by
            ``compute_with_defaults``).
        missing_signals:
            Names of signals that were absent from the original request and
            substituted with conservative defaults.  Passed through verbatim
            into ``RuleEngineOutput.missing_signals``.
        """
        if missing_signals is None:
            missing_signals = []

        contributions: List[SignalContribution] = [
            self._score_device_trust(inp),
            self._score_login_delta(inp),
            self._score_amount_deviation(inp),
            self._score_otp_retry(inp),
            self._score_first_time_action(inp),
            self._score_behavioral_consistency(inp),
        ]

        raw_score = sum(c.contribution for c in contributions)
        risk_score = max(0.0, min(100.0, raw_score))

        risk_label = self._score_to_label(risk_score)

        return RuleEngineOutput(
            risk_score=risk_score,
            risk_label=risk_label,
            signal_contributions=contributions,
            missing_signals=list(missing_signals),
        )

    def compute_with_defaults(
        self,
        inp: RuleEngineInput,
        missing_signals: Optional[List[str]] = None,
    ) -> RuleEngineOutput:
        """
        Convenience wrapper — identical to ``compute`` but makes the intent
        explicit that conservative defaults have already been applied.
        """
        return self.compute(inp, missing_signals=missing_signals)

    @staticmethod
    def get_remediation_steps() -> List[str]:
        """Return the deterministic remediation lookup table for block decisions."""
        return list(REMEDIATION_STEPS)

    # ------------------------------------------------------------------
    # Score → label / decision helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_label(score: float) -> str:
        if score >= 70:
            return "High"
        if score >= 40:
            return "Medium"
        return "Low"

    @staticmethod
    def score_to_decision(score: float) -> str:
        """Map a numeric risk score to an allow/warn/block decision string."""
        if score >= 70:
            return "block"
        if score >= 40:
            return "warn"
        return "allow"

    # ------------------------------------------------------------------
    # Individual signal scorers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_device_trust(inp: RuleEngineInput) -> SignalContribution:
        """
        device_trust_status (weight 20).
        False (new / untrusted device) → 20 pts (minimum enforced).
        True (trusted device) → 0 pts.
        """
        if not inp.device_trust_status:
            contribution = 20.0
            description = "Device is not trusted; full weight applied (minimum 20 pts enforced)."
        else:
            contribution = 0.0
            description = "Device is trusted; no contribution."

        return SignalContribution(
            signal_name="device_trust_status",
            contribution=contribution,
            description=description,
        )

    @staticmethod
    def _score_login_delta(inp: RuleEngineInput) -> SignalContribution:
        """
        login_to_action_delta (weight 15).
        < 5 s → 15 pts; < 30 s → 10 pts; < 120 s → 5 pts; else → 0 pts.
        """
        seconds = inp.login_to_action_seconds
        if seconds < 5:
            contribution = 15.0
            description = f"Login-to-action delta {seconds:.1f}s is extremely short (<5s); full weight."
        elif seconds < 30:
            contribution = 10.0
            description = f"Login-to-action delta {seconds:.1f}s is very short (<30s)."
        elif seconds < 120:
            contribution = 5.0
            description = f"Login-to-action delta {seconds:.1f}s is short (<120s)."
        else:
            contribution = 0.0
            description = f"Login-to-action delta {seconds:.1f}s is within normal range."

        return SignalContribution(
            signal_name="login_to_action_delta",
            contribution=contribution,
            description=description,
        )

    @staticmethod
    def _score_amount_deviation(inp: RuleEngineInput) -> SignalContribution:
        """
        amount_vs_90day_avg (weight 25).
        avg == 0 → 25 pts (unknown avg is conservative).
        ratio > 2.5 → 25 pts (minimum 25 enforced).
        ratio > 1.5 → 15 pts.
        ratio > 1.0 → 5 pts.
        else → 0 pts.
        """
        avg = inp.amount_90day_avg_inr
        amount = inp.action_amount_inr

        if avg == 0:
            contribution = 25.0
            description = "90-day average is unknown (0); conservative maximum applied."
        else:
            ratio = amount / avg
            if ratio > 2.5:
                contribution = 25.0  # minimum 25 enforced
                description = (
                    f"Action amount is {ratio:.2f}× the 90-day average (>2.5×); "
                    "minimum contribution of 25 pts enforced."
                )
            elif ratio > 1.5:
                contribution = 15.0
                description = f"Action amount is {ratio:.2f}× the 90-day average (>1.5×)."
            elif ratio > 1.0:
                contribution = 5.0
                description = f"Action amount is {ratio:.2f}× the 90-day average (>1.0×)."
            else:
                contribution = 0.0
                description = f"Action amount is {ratio:.2f}× the 90-day average; within normal range."

        return SignalContribution(
            signal_name="amount_vs_90day_avg",
            contribution=contribution,
            description=description,
        )

    @staticmethod
    def _score_otp_retry(inp: RuleEngineInput) -> SignalContribution:
        """
        otp_retry_count (weight 15).
        >= 3 → 15 pts; == 2 → 10 pts; == 1 → 5 pts; else → 0 pts.
        """
        count = inp.otp_retry_count
        if count >= 3:
            contribution = 15.0
            description = f"OTP retry count {count} ≥ 3; full weight applied."
        elif count == 2:
            contribution = 10.0
            description = "OTP retry count is 2."
        elif count == 1:
            contribution = 5.0
            description = "OTP retry count is 1."
        else:
            contribution = 0.0
            description = "No OTP retries detected."

        return SignalContribution(
            signal_name="otp_retry_count",
            contribution=contribution,
            description=description,
        )

    @staticmethod
    def _score_first_time_action(inp: RuleEngineInput) -> SignalContribution:
        """
        first_time_action_flag (weight 10).
        True → 10 pts; False → 0 pts.
        """
        if inp.first_time_action:
            contribution = 10.0
            description = "First-time action or investment type detected."
        else:
            contribution = 0.0
            description = "Action type has been performed before."

        return SignalContribution(
            signal_name="first_time_action_flag",
            contribution=contribution,
            description=description,
        )

    @staticmethod
    def _score_behavioral_consistency(inp: RuleEngineInput) -> SignalContribution:
        """
        behavioral_consistency (weight 15).
        retry_loop AND cancel_retry → 15 pts.
        either → 8 pts.
        neither → 0 pts.
        """
        retry_loop = inp.retry_loop_detected
        cancel_retry = inp.cancel_retry_pattern

        if retry_loop and cancel_retry:
            contribution = 15.0
            description = "Both retry loop and cancel-retry pattern detected; full weight applied."
        elif retry_loop or cancel_retry:
            contribution = 8.0
            description = "One behavioral anomaly detected (retry loop or cancel-retry pattern)."
        else:
            contribution = 0.0
            description = "No behavioral anomalies detected."

        return SignalContribution(
            signal_name="behavioral_consistency",
            contribution=contribution,
            description=description,
        )
