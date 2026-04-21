"""
Wealth Intelligence Engine — ML-based behavior classification and analysis.

Sub-components (implemented incrementally across tasks):
  - BehaviorClassifier  (task 4.1) — Random Forest spending-category classifier
  - SHAPExplainer       (task 4.2) — SHAP feature-contribution explainer
  - TrendDetector       (task 4.4) — rolling-window income/expense trend detector
  - NetWorthCalculator  (task 4.5) — pure-arithmetic net worth computation
"""

from __future__ import annotations

import logging
from pathlib import Path
from statistics import mode as stat_mode
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from app.models.internal import MLClassificationResult
from app.models.request import AssetEntry, TransactionRecord
from app.models.response import ShapExplanation, ShapFeature

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent.parent / "data"
_CSV_PATH = _DATA_DIR / "synthetic_transactions.csv"
_MODELS_DIR = _DATA_DIR / "models"
_PKL_PATH = _MODELS_DIR / "behavior_classifier.pkl"

# ---------------------------------------------------------------------------
# SHAPExplainer
# ---------------------------------------------------------------------------

class SHAPExplainer:
    """
    Wraps ``shap.TreeExplainer`` to compute per-feature SHAP values for a
    trained ``RandomForestClassifier``.

    Usage::

        explainer = SHAPExplainer(model, feature_names)
        shap_dict = explainer.explain(feature_array, predicted_class_idx)
    """

    def __init__(
        self,
        model: RandomForestClassifier,
        feature_names: List[str],
    ) -> None:
        self._feature_names = feature_names
        self._explainer = shap.TreeExplainer(model)

    def explain(
        self,
        feature_array: np.ndarray,
        predicted_class_idx: int,
    ) -> Dict[str, float]:
        """
        Compute SHAP values for a single sample and return a mapping of
        feature_name → shap_value for the predicted class.

        Parameters
        ----------
        feature_array:
            Shape (1, n_features) array of input features.
        predicted_class_idx:
            Index of the predicted class (used to select the correct SHAP slice).

        Returns
        -------
        Dict mapping each feature name to its SHAP value for the predicted class.
        """
        # shap_values shape varies by shap version:
        #   - Old (list): list of (n_samples, n_features) arrays, one per class
        #   - New (ndarray): (n_samples, n_features, n_classes)
        raw = self._explainer.shap_values(feature_array)

        if isinstance(raw, list):
            # Old multi-class format: raw[class_idx] has shape (n_samples, n_features)
            values = raw[predicted_class_idx][0]
        elif isinstance(raw, np.ndarray) and raw.ndim == 3:
            # New multi-class format: (n_samples, n_features, n_classes)
            values = raw[0, :, predicted_class_idx]
        else:
            # Binary or single output: shape (n_samples, n_features)
            values = raw[0]

        return {
            name: float(val)
            for name, val in zip(self._feature_names, values)
        }


# ---------------------------------------------------------------------------
# build_shap_explanation helper
# ---------------------------------------------------------------------------

_PADDING_FEATURE_PREFIX = "feature_"


def build_shap_explanation(shap_values: Dict[str, float]) -> ShapExplanation:
    """
    Build a ``ShapExplanation`` from a feature_name → shap_value dict.

    Steps:
    1. Sort features by |shap_value| descending.
    2. Take the top 5 (or all if fewer than 5).
    3. Pad with zero-contribution synthetic features to always reach exactly 5.

    Parameters
    ----------
    shap_values:
        Mapping of feature_name → shap_value.

    Returns
    -------
    ``ShapExplanation`` with exactly 5 ``ShapFeature`` objects.
    """
    sorted_features = sorted(
        shap_values.items(),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )

    top_features: List[ShapFeature] = [
        ShapFeature(
            feature_name=name,
            shap_value=value,
            direction="positive" if value >= 0 else "negative",
        )
        for name, value in sorted_features[:5]
    ]

    # Pad to exactly 5 if fewer features exist
    pad_idx = 0
    while len(top_features) < 5:
        pad_name = f"{_PADDING_FEATURE_PREFIX}{pad_idx}"
        # Avoid name collisions with real features
        while pad_name in shap_values:
            pad_idx += 1
            pad_name = f"{_PADDING_FEATURE_PREFIX}{pad_idx}"
        top_features.append(
            ShapFeature(
                feature_name=pad_name,
                shap_value=0.0,
                direction="positive",
            )
        )
        pad_idx += 1

    return ShapExplanation(top_features=top_features)


# ---------------------------------------------------------------------------
# BehaviorClassifier
# ---------------------------------------------------------------------------

class BehaviorClassifier:
    """
    Random Forest classifier that maps a list of TransactionRecords to a
    spending category (conservative / moderate / aggressive) with a confidence
    score.

    Usage::

        clf = BehaviorClassifier.load_or_train()
        result = clf.classify(transactions)
    """

    # Feature names used during training and inference
    FEATURE_NAMES = ["mean_amount", "std_amount", "category_encoded", "credit_ratio"]

    def __init__(
        self,
        model: RandomForestClassifier,
        category_encoder: LabelEncoder,
        label_encoder: LabelEncoder,
    ) -> None:
        self._model = model
        self._category_encoder = category_encoder  # encodes transaction category strings
        self._label_encoder = label_encoder         # encodes spending_label target
        self._shap_explainer = SHAPExplainer(model, self.FEATURE_NAMES)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, transactions: List[TransactionRecord]) -> MLClassificationResult:
        """
        Aggregate transaction features and return a spending category + confidence.

        Parameters
        ----------
        transactions:
            One or more TransactionRecord objects representing the user's history.

        Returns
        -------
        MLClassificationResult with spending_category, confidence, and an empty
        shap_values dict (populated in task 4.2).
        """
        features = self._aggregate_features(transactions)
        feature_array = np.array([[
            features["mean_amount"],
            features["std_amount"],
            features["category_encoded"],
            features["credit_ratio"],
        ]])

        proba = self._model.predict_proba(feature_array)[0]
        predicted_idx = int(np.argmax(proba))
        confidence = float(proba[predicted_idx])
        spending_category = self._label_encoder.inverse_transform([predicted_idx])[0]

        shap_values = self._shap_explainer.explain(feature_array, predicted_idx)

        return MLClassificationResult(
            spending_category=spending_category,
            confidence=confidence,
            shap_values=shap_values,
        )

    # ------------------------------------------------------------------
    # Class-level factory
    # ------------------------------------------------------------------

    @classmethod
    def load_or_train(cls, csv_path: Optional[Path] = None) -> "BehaviorClassifier":
        """
        Load a persisted model from disk if available; otherwise train from CSV
        and persist the result.

        Parameters
        ----------
        csv_path:
            Path to the labeled transaction CSV.  Defaults to the bundled
            ``app/data/synthetic_transactions.csv``.
        """
        if _PKL_PATH.exists():
            logger.info("Loading BehaviorClassifier from %s", _PKL_PATH)
            return cls._load(_PKL_PATH)

        logger.info("No persisted model found — training from CSV …")
        return cls._train_and_save(csv_path or _CSV_PATH)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _aggregate_features(self, transactions: List[TransactionRecord]) -> dict:
        """Compute aggregate features from a list of TransactionRecords."""
        amounts = [t.amount_inr for t in transactions]
        mean_amount = float(np.mean(amounts))
        std_amount = float(np.std(amounts)) if len(amounts) > 1 else 0.0

        categories = [t.category for t in transactions]
        # Use the most common category; fall back to first if all unique
        try:
            common_cat = stat_mode(categories)
        except Exception:
            common_cat = categories[0]

        # Encode the category — handle unseen categories gracefully
        try:
            category_encoded = float(
                self._category_encoder.transform([common_cat])[0]
            )
        except ValueError:
            # Unseen category → use mean of known codes
            category_encoded = float(
                np.mean(self._category_encoder.transform(self._category_encoder.classes_))
            )

        credit_count = sum(1 for t in transactions if t.direction == "credit")
        credit_ratio = credit_count / len(transactions)

        return {
            "mean_amount": mean_amount,
            "std_amount": std_amount,
            "category_encoded": category_encoded,
            "credit_ratio": credit_ratio,
        }

    @classmethod
    def _train_and_save(cls, csv_path: Path) -> "BehaviorClassifier":
        """Train the Random Forest on the CSV and persist to disk."""
        df = pd.read_csv(csv_path)

        # --- Feature engineering ---
        category_encoder = LabelEncoder()
        df["category_encoded"] = category_encoder.fit_transform(df["category"])
        df["direction_encoded"] = (df["direction"] == "credit").astype(int)

        # Aggregate per spending_label group to build training rows
        # Each row = one "user session" aggregated from all transactions of that label
        # For training we treat each individual transaction as a sample with its own
        # aggregate context (mean/std computed over the full label group).
        label_stats = (
            df.groupby("spending_label")["amount_inr"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "mean_amount", "std": "std_amount"})
        )

        rows = []
        for _, row in df.iterrows():
            label = row["spending_label"]
            stats = label_stats.loc[label]
            credit_ratio = (
                df[df["spending_label"] == label]["direction_encoded"].mean()
            )
            rows.append({
                "mean_amount": stats["mean_amount"],
                "std_amount": stats["std_amount"],
                "category_encoded": row["category_encoded"],
                "credit_ratio": credit_ratio,
                "spending_label": label,
            })

        train_df = pd.DataFrame(rows)

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(train_df["spending_label"])
        X = train_df[["mean_amount", "std_amount", "category_encoded", "credit_ratio"]].values

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        instance = cls(
            model=model,
            category_encoder=category_encoder,
            label_encoder=label_encoder,
        )
        instance._save(_PKL_PATH)
        return instance

    def _save(self, path: Path) -> None:
        """Persist model + encoders to a joblib pickle."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "category_encoder": self._category_encoder,
            "label_encoder": self._label_encoder,
        }
        joblib.dump(payload, path)
        logger.info("BehaviorClassifier persisted to %s", path)

    @classmethod
    def _load(cls, path: Path) -> "BehaviorClassifier":
        """Load model + encoders from a joblib pickle."""
        payload = joblib.load(path)
        return cls(
            model=payload["model"],
            category_encoder=payload["category_encoder"],
            label_encoder=payload["label_encoder"],
        )


# ---------------------------------------------------------------------------
# TrendDetector
# ---------------------------------------------------------------------------

class TrendDetector:
    """
    Detects income and expense trends from a list of TransactionRecords using
    rolling-window linear regression (numpy polyfit, degree=1).

    Slope classification thresholds:
      - slope >  100  → "increasing"
      - slope < -100  → "decreasing"
      - otherwise     → "stable"

    A data quality warning is appended when total transaction count < 30.

    Usage::

        detector = TrendDetector()
        result = detector.detect(transactions)
        # {
        #   "income_trend": "increasing" | "stable" | "decreasing" | "insufficient data",
        #   "expense_trend": "increasing" | "stable" | "decreasing" | "insufficient data",
        #   "data_quality_warning": Optional[str],
        # }
    """

    _SLOPE_INCREASING_THRESHOLD = 100.0
    _SLOPE_DECREASING_THRESHOLD = -100.0
    _MIN_TRANSACTIONS_FOR_QUALITY = 30
    _DATA_QUALITY_WARNING = (
        "Fewer than 30 transactions; trends may be less accurate"
    )

    def detect(self, transactions: List[TransactionRecord]) -> dict:
        """
        Analyse income and expense trends.

        Parameters
        ----------
        transactions:
            Full list of TransactionRecord objects for the user.

        Returns
        -------
        dict with keys:
          - ``income_trend``  : str
          - ``expense_trend`` : str
          - ``data_quality_warning`` : Optional[str]  (None when count >= 30)
        """
        income_txns = sorted(
            [t for t in transactions if t.direction == "credit"],
            key=lambda t: t.date,
        )
        expense_txns = sorted(
            [t for t in transactions if t.direction == "debit"],
            key=lambda t: t.date,
        )

        income_trend = self._classify_trend(income_txns)
        expense_trend = self._classify_trend(expense_txns)

        warning: Optional[str] = None
        if len(transactions) < self._MIN_TRANSACTIONS_FOR_QUALITY:
            warning = self._DATA_QUALITY_WARNING

        return {
            "income_trend": income_trend,
            "expense_trend": expense_trend,
            "data_quality_warning": warning,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify_trend(self, sorted_txns: List[TransactionRecord]) -> str:
        """
        Fit a degree-1 polynomial (linear regression) over the time series and
        classify the slope.

        Parameters
        ----------
        sorted_txns:
            Transactions sorted by date (ascending), all of the same direction.

        Returns
        -------
        "increasing", "stable", "decreasing", or "insufficient data" when the
        list is empty.
        """
        if not sorted_txns:
            return "insufficient data"

        x = np.arange(len(sorted_txns), dtype=float)
        y = np.array([t.amount_inr for t in sorted_txns], dtype=float)

        # polyfit returns [slope, intercept]
        slope, _ = np.polyfit(x, y, 1)

        if slope > self._SLOPE_INCREASING_THRESHOLD:
            return "increasing"
        if slope < self._SLOPE_DECREASING_THRESHOLD:
            return "decreasing"
        return "stable"


# ---------------------------------------------------------------------------
# NetWorthCalculator
# ---------------------------------------------------------------------------

class NetWorthCalculator:
    """
    Pure-arithmetic net worth calculator.

    Computes ``net_worth = sum(asset.value_inr for asset in assets) - liabilities_inr``.

    Usage::

        calculator = NetWorthCalculator()
        net_worth = calculator.compute(assets, liabilities_inr=50000.0)
    """

    def compute(self, assets: List[AssetEntry], liabilities_inr: float) -> float:
        """
        Compute net worth from a list of AssetEntry objects and a liabilities value.

        Parameters
        ----------
        assets:
            List of AssetEntry objects, each with a ``value_inr`` field.
        liabilities_inr:
            Total declared liabilities in INR.

        Returns
        -------
        Net worth as a float (may be negative if liabilities exceed assets).
        """
        total_assets = sum(asset.value_inr for asset in assets)
        return total_assets - liabilities_inr
