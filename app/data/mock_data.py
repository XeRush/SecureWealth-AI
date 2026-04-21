"""
Mock data generator — produces realistic demo users, transactions, and profiles.

Usage:
    from app.data.mock_data import generate_mock_user, generate_mock_transactions
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta

from app.models.schemas import (
    Asset,
    Goal,
    GoalType,
    Transaction,
    TransactionCategory,
    UserProfile,
    RiskAppetite,
)

# ────────────────────────────────────────────
#  Constants
# ────────────────────────────────────────────

_NAMES = [
    "Aarav Sharma", "Priya Patel", "Rohan Gupta", "Sneha Iyer",
    "Vikram Singh", "Ananya Reddy", "Karthik Nair", "Meera Joshi",
    "Arjun Mehta", "Diya Kapoor",
]

_EXPENSE_CATEGORIES: list[tuple[TransactionCategory, float, float]] = [
    # (category, min_amount, max_amount)
    (TransactionCategory.FOOD, 100, 3000),
    (TransactionCategory.TRANSPORT, 50, 2000),
    (TransactionCategory.SHOPPING, 200, 15000),
    (TransactionCategory.BILLS, 500, 8000),
    (TransactionCategory.ENTERTAINMENT, 100, 5000),
    (TransactionCategory.HEALTH, 200, 10000),
    (TransactionCategory.EDUCATION, 500, 20000),
    (TransactionCategory.OTHER, 50, 5000),
]

_ASSET_TEMPLATES = [
    ("property", "Flat in Mumbai", 4_500_000),
    ("property", "Plot in Pune", 2_000_000),
    ("gold", "Gold Jewellery", 350_000),
    ("gold", "Gold Coins (50g)", 300_000),
    ("vehicle", "Maruti Baleno", 800_000),
    ("vehicle", "Honda Activa", 95_000),
    ("stocks", "Equity Portfolio", 200_000),
    ("fd", "SBI Fixed Deposit", 500_000),
    ("ppf", "PPF Account", 250_000),
    ("mutual_fund", "HDFC Flexi-Cap Fund", 150_000),
]


# ────────────────────────────────────────────
#  Generators
# ────────────────────────────────────────────

def _uid() -> str:
    return uuid.uuid4().hex[:12]


def generate_mock_user(user_id: str | None = None) -> UserProfile:
    """Generate a single realistic mock user."""
    uid = user_id or f"USR-{_uid()}"
    name = random.choice(_NAMES)
    age = random.randint(22, 55)
    income = round(random.uniform(30_000, 300_000), 2)
    expenses = round(income * random.uniform(0.4, 0.85), 2)

    # Random assets (1-4)
    asset_count = random.randint(1, 4)
    assets = []
    for tmpl in random.sample(_ASSET_TEMPLATES, min(asset_count, len(_ASSET_TEMPLATES))):
        variation = random.uniform(0.7, 1.4)
        assets.append(Asset(
            asset_type=tmpl[0],
            name=tmpl[1],
            value_inr=round(tmpl[2] * variation, 2),
        ))

    # Random goals (1-3)
    goal_types = random.sample(list(GoalType), random.randint(1, 3))
    goals = []
    for gt in goal_types:
        target = round(random.uniform(200_000, 5_000_000), 2)
        saved = round(target * random.uniform(0.05, 0.5), 2)
        years_out = random.randint(2, 15)
        goals.append(Goal(
            goal_type=gt,
            name=f"{gt.value.title()} Goal",
            target_amount_inr=target,
            target_date=(datetime.now() + timedelta(days=365 * years_out)).strftime("%Y-%m-%d"),
            current_savings_inr=saved,
        ))

    return UserProfile(
        user_id=uid,
        name=name,
        age=age,
        monthly_income_inr=income,
        monthly_expenses_inr=expenses,
        risk_appetite=random.choice(list(RiskAppetite)),
        assets=assets,
        goals=goals,
        liabilities_inr=round(random.uniform(0, income * 6), 2),
    )


def generate_mock_transactions(
    user_id: str,
    months: int = 6,
    txns_per_month: int = 20,
) -> list[Transaction]:
    """Generate realistic transaction history for a user."""
    transactions: list[Transaction] = []
    now = datetime.now()

    for month_offset in range(months, 0, -1):
        month_start = now - timedelta(days=30 * month_offset)

        # One salary credit per month
        salary_day = month_start + timedelta(days=random.randint(0, 2))
        transactions.append(Transaction(
            txn_id=f"TXN-{_uid()}",
            user_id=user_id,
            amount_inr=round(random.uniform(30_000, 300_000), 2),
            category=TransactionCategory.SALARY,
            description="Monthly Salary Credit",
            timestamp=salary_day,
            is_credit=True,
        ))

        # Expense transactions
        for _ in range(txns_per_month):
            cat, lo, hi = random.choice(_EXPENSE_CATEGORIES)
            day = month_start + timedelta(days=random.randint(0, 29))
            transactions.append(Transaction(
                txn_id=f"TXN-{_uid()}",
                user_id=user_id,
                amount_inr=round(random.uniform(lo, hi), 2),
                category=cat,
                description=f"{cat.value.title()} expense",
                timestamp=day,
                is_credit=False,
            ))

    # Sort chronologically
    transactions.sort(key=lambda t: t.timestamp)
    return transactions


def generate_demo_dataset(num_users: int = 5) -> list[dict]:
    """Generate a complete demo dataset with users and their transactions."""
    dataset = []
    for _ in range(num_users):
        user = generate_mock_user()
        txns = generate_mock_transactions(user.user_id)
        dataset.append({"user": user, "transactions": txns})
    return dataset
