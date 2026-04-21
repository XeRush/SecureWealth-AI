"""
Simulation Agent for SecureWealth Twin AI.

Runs compound growth projections and what-if scenario analyses.
All outputs are for simulation and demo purposes only.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from app.models.request import SimulateRequest, WhatIfScenario

SIMULATION_LABEL = (
    "For simulation and demo purposes only. Not a guarantee of future returns."
)

# Annual return rates per asset class (converted to monthly by dividing by 12)
ANNUAL_RETURN_RATES: Dict[str, float] = {
    "equity": 0.12,
    "debt": 0.07,
    "gold": 0.08,
    "real_estate": 0.10,
}
DEFAULT_ANNUAL_RATE = 0.08


def _monthly_rate(asset_class: str) -> float:
    """Return the monthly return rate for a given asset class."""
    annual = ANNUAL_RETURN_RATES.get(asset_class, DEFAULT_ANNUAL_RATE)
    return annual / 12


def project_wealth(
    assets: float,
    savings_rate: float,
    allocation: Dict[str, float],
    horizon: int,
) -> float:
    """
    Project future wealth using the compound growth formula.

    FV = PV * (1 + r)^n + PMT * ((1 + r)^n - 1) / r
    where r is the weighted average monthly return rate.

    If r == 0: FV = PV + PMT * n

    Args:
        assets: Current assets in INR (PV).
        savings_rate: Monthly savings rate in INR (PMT).
        allocation: Dict mapping asset class to fraction (must sum to 1.0).
        horizon: Time horizon in months (n).

    Returns:
        Projected future value in INR.

    Raises:
        ValueError: If savings_rate < 0 or horizon <= 0.
    """
    if savings_rate < 0:
        raise ValueError("savings_rate must be >= 0")
    if horizon <= 0:
        raise ValueError("horizon must be > 0")

    # Weighted average monthly return rate
    r = sum(fraction * _monthly_rate(asset) for asset, fraction in allocation.items())
    n = horizon
    pv = assets
    pmt = savings_rate

    if r == 0:
        return pv + pmt * n

    growth_factor = (1 + r) ** n
    fv = pv * growth_factor + pmt * (growth_factor - 1) / r
    return fv


def compute_goal_probability(projected_wealth: float, target: float) -> float:
    """
    Compute goal achievement probability as a percentage in [0, 100].

    Heuristic: probability = min(100, (projected_wealth / target) * 100)

    Args:
        projected_wealth: Projected future wealth in INR.
        target: Target goal amount in INR.

    Returns:
        Probability as a float in [0.0, 100.0].
    """
    if target <= 0:
        return 100.0
    probability = (projected_wealth / target) * 100.0
    return max(0.0, min(100.0, probability))


class SimulationAgent:
    """
    Agent that runs wealth simulations and what-if scenario analyses.

    This is a plain Python class (no LangChain AgentExecutor) that exposes
    a single `simulate` method accepting a SimulateRequest.
    """

    def simulate(self, request: SimulateRequest) -> dict:
        """
        Run a wealth simulation for the given request.

        Args:
            request: A validated SimulateRequest instance.

        Returns:
            Dict with keys:
                - goal_achievement_probability: float in [0, 100]
                - projected_wealth_inr: float
                - scenario_results: Optional[List[dict]] for what-if scenarios
                - simulation_label: str (always the standard disclaimer)

        Raises:
            ValueError: If savings_rate < 0 or time_horizon_months <= 0.
        """
        # Agent-level validation (Pydantic handles API layer, but guard here too)
        if request.monthly_savings_rate_inr < 0:
            raise ValueError(
                "monthly_savings_rate_inr must be >= 0"
            )
        if request.time_horizon_months <= 0:
            raise ValueError(
                "time_horizon_months must be > 0"
            )

        # Base projection
        projected_wealth = project_wealth(
            assets=request.current_assets_inr,
            savings_rate=request.monthly_savings_rate_inr,
            allocation=request.investment_allocation,
            horizon=request.time_horizon_months,
        )

        goal_probability = compute_goal_probability(
            projected_wealth=projected_wealth,
            target=request.target_goal_inr,
        )

        # What-if scenarios
        scenario_results: Optional[List[dict]] = None
        if request.what_if_scenarios:
            scenario_results = self._run_scenarios(request)

        return {
            "goal_achievement_probability": goal_probability,
            "projected_wealth_inr": projected_wealth,
            "scenario_results": scenario_results,
            "simulation_label": SIMULATION_LABEL,
        }

    def _run_scenarios(self, request: SimulateRequest) -> List[dict]:
        """Run what-if scenario projections, overriding base values per scenario."""
        results = []
        for scenario in request.what_if_scenarios:  # type: WhatIfScenario
            savings = (
                scenario.monthly_savings_rate_inr
                if scenario.monthly_savings_rate_inr is not None
                else request.monthly_savings_rate_inr
            )
            allocation = (
                scenario.investment_allocation
                if scenario.investment_allocation is not None
                else request.investment_allocation
            )
            horizon = (
                scenario.time_horizon_months
                if scenario.time_horizon_months is not None
                else request.time_horizon_months
            )

            projected = project_wealth(
                assets=request.current_assets_inr,
                savings_rate=savings,
                allocation=allocation,
                horizon=horizon,
            )
            probability = compute_goal_probability(
                projected_wealth=projected,
                target=request.target_goal_inr,
            )

            results.append(
                {
                    "label": scenario.label,
                    "goal_achievement_probability": probability,
                    "projected_wealth_inr": projected,
                }
            )
        return results
