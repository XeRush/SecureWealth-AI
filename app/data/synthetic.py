"""
Synthetic data generator for SecureWealth Twin AI.

Generates labeled transaction records, user profiles, and fraud scenario
signal sets for ML training and testing.

Run directly to regenerate data files:
    python -m app.data.synthetic
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

# ── Output paths ─────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent
PROFILES_PATH = DATA_DIR / "synthetic_profiles.json"
TRANSACTIONS_PATH = DATA_DIR / "synthetic_transactions.csv"

# ── Category definitions per spending label ──────────────────────────────────
CONSERVATIVE_CATEGORIES = [
    "groceries",
    "utilities",
    "rent",
    "public_transport",
    "medical",
    "insurance",
]

MODERATE_CATEGORIES = [
    "groceries",
    "utilities",
    "rent",
    "dining",
    "shopping",
    "travel",
    "entertainment",
    "medical",
]

AGGRESSIVE_CATEGORIES = [
    "luxury",
    "investments",
    "entertainment",
    "travel",
    "dining",
    "shopping",
    "electronics",
    "subscriptions",
]

# Amount ranges in INR per label
AMOUNT_RANGES = {
    "conservative": (200, 8_000),
    "moderate": (500, 25_000),
    "aggressive": (2_000, 150_000),
}

# ── Transaction generation ────────────────────────────────────────────────────

def _random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=int(rng.integers(0, delta + 1)))


def generate_transactions(
    n: int,
    label: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[dict[str, Any]]:
    """Return *n* transaction dicts with the given spending_label."""
    if start_date is None:
        start_date = date(2023, 1, 1)
    if end_date is None:
        end_date = date(2024, 12, 31)

    categories = {
        "conservative": CONSERVATIVE_CATEGORIES,
        "moderate": MODERATE_CATEGORIES,
        "aggressive": AGGRESSIVE_CATEGORIES,
    }[label]

    lo, hi = AMOUNT_RANGES[label]
    records = []
    for _ in range(n):
        amount = float(round(rng.uniform(lo, hi), 2))
        category = random.choice(categories)
        # Credits are rare (salary / refund) — ~10 % of records
        direction = "credit" if rng.random() < 0.10 else "debit"
        records.append(
            {
                "date": str(_random_date(start_date, end_date)),
                "amount_inr": amount,
                "category": category,
                "direction": direction,
                "spending_label": label,
            }
        )
    return records


def generate_transaction_dataset(total: int = 600) -> pd.DataFrame:
    """
    Generate a balanced, labeled transaction dataset.

    Args:
        total: Total number of records (split evenly across 3 labels).

    Returns:
        DataFrame with columns: date, amount_inr, category, direction, spending_label
    """
    per_label = total // 3
    remainder = total - per_label * 3  # distribute leftover to first label

    rows: list[dict[str, Any]] = []
    rows += generate_transactions(per_label + remainder, "conservative")
    rows += generate_transactions(per_label, "moderate")
    rows += generate_transactions(per_label, "aggressive")

    df = pd.DataFrame(rows, columns=["date", "amount_inr", "category", "direction", "spending_label"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    # Shuffle so labels are interleaved
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


# ── User profile generation ───────────────────────────────────────────────────

_GOALS_POOL = [
    "retirement",
    "home_purchase",
    "child_education",
    "emergency_fund",
    "travel",
    "car_purchase",
    "business_investment",
    "wealth_growth",
]

_ASSET_TYPES = ["property", "gold", "vehicle", "financial_instrument"]


def _generate_assets(label: str) -> list[dict[str, Any]]:
    """Generate a small list of asset entries appropriate for the spending label."""
    n_assets = int(rng.integers(1, 4))
    assets = []
    for _ in range(n_assets):
        asset_type = random.choice(_ASSET_TYPES)
        if label == "conservative":
            value = float(round(rng.uniform(50_000, 500_000), 2))
        elif label == "moderate":
            value = float(round(rng.uniform(200_000, 2_000_000), 2))
        else:
            value = float(round(rng.uniform(1_000_000, 20_000_000), 2))
        assets.append({"asset_type": asset_type, "value_inr": value})
    return assets


def generate_user_profiles(n: int = 50) -> list[dict[str, Any]]:
    """
    Generate *n* simplified UserProfile dicts for testing.

    Each profile includes a subset of transactions (30–90 records) drawn
    from the matching spending label so that the BehaviorClassifier can
    be validated end-to-end.
    """
    labels = ["conservative", "moderate", "aggressive"]
    profiles = []

    for i in range(n):
        label = labels[i % 3]
        income_ranges = {
            "conservative": (20_000, 60_000),
            "moderate": (60_000, 150_000),
            "aggressive": (150_000, 500_000),
        }
        lo, hi = income_ranges[label]
        income = float(round(rng.uniform(lo, hi), 2))
        liabilities = float(round(rng.uniform(0, income * 12), 2))
        n_tx = int(rng.integers(30, 91))
        transactions = generate_transactions(n_tx, label)
        goals = random.sample(_GOALS_POOL, k=int(rng.integers(1, 4)))

        profiles.append(
            {
                "user_id": f"synthetic_user_{i:04d}",
                "consent_token": f"consent_{i:04d}",
                "income_monthly_inr": income,
                "risk_appetite": label,
                "goals": goals,
                "assets": _generate_assets(label),
                "liabilities_inr": liabilities,
                "transactions": transactions,
            }
        )

    return profiles


# ── Fraud scenario signal sets ────────────────────────────────────────────────

def generate_fraud_signals(n: int = 100) -> list[dict[str, Any]]:
    """
    Generate *n* fraud scenario signal dicts for Rule Engine testing.

    Each dict maps directly to the fields of ``RuleEngineInput``.
    Scenarios are split across low / medium / high risk profiles.
    """
    signals = []
    for i in range(n):
        risk_tier = i % 3  # 0=low, 1=medium, 2=high

        if risk_tier == 0:  # low risk
            device_trust = True
            login_delta = float(round(rng.uniform(60, 600), 1))
            amount = float(round(rng.uniform(500, 5_000), 2))
            avg_90 = float(round(rng.uniform(amount * 0.8, amount * 1.2), 2))
            otp_retries = int(rng.integers(0, 2))
            first_time = False
            retry_loop = False
            cancel_retry = False
        elif risk_tier == 1:  # medium risk
            device_trust = bool(rng.integers(0, 2))
            login_delta = float(round(rng.uniform(5, 120), 1))
            amount = float(round(rng.uniform(5_000, 50_000), 2))
            avg_90 = float(round(rng.uniform(amount * 0.4, amount * 0.9), 2))
            otp_retries = int(rng.integers(1, 4))
            first_time = bool(rng.integers(0, 2))
            retry_loop = bool(rng.integers(0, 2))
            cancel_retry = False
        else:  # high risk
            device_trust = False
            login_delta = float(round(rng.uniform(0, 10), 1))
            amount = float(round(rng.uniform(50_000, 500_000), 2))
            avg_90 = float(round(rng.uniform(100, amount * 0.3), 2))
            otp_retries = int(rng.integers(3, 6))
            first_time = True
            retry_loop = True
            cancel_retry = True

        signals.append(
            {
                "device_trust_status": device_trust,
                "login_to_action_seconds": login_delta,
                "action_amount_inr": amount,
                "amount_90day_avg_inr": avg_90,
                "otp_retry_count": otp_retries,
                "first_time_action": first_time,
                "retry_loop_detected": retry_loop,
                "cancel_retry_pattern": cancel_retry,
            }
        )

    return signals


# ── Export helpers ────────────────────────────────────────────────────────────

def export_transactions(df: pd.DataFrame, path: Path = TRANSACTIONS_PATH) -> None:
    """Write transaction DataFrame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Exported {len(df)} transaction records → {path}")


def export_profiles(profiles: list[dict[str, Any]], path: Path = PROFILES_PATH) -> None:
    """Write user profiles list to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(profiles, fh, indent=2, default=str)
    print(f"Exported {len(profiles)} user profiles → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Generating synthetic data …")

    # Transactions (600 records, balanced across 3 labels)
    df = generate_transaction_dataset(total=600)
    label_counts = df["spending_label"].value_counts().to_dict()
    print(f"  Label distribution: {label_counts}")
    export_transactions(df)

    # User profiles (50 profiles)
    profiles = generate_user_profiles(n=50)
    export_profiles(profiles)

    print("Done.")


if __name__ == "__main__":
    main()
