"""
StudyTrade — Machine Learning Models and Probability Calibration
==================================================================

Model wrappers for systematic option trading:
  - Logistic Regression (L2 regularized baseline)
  - Random Forest Classifier
  - HistGradientBoostingClassifier (LightGBM-style)
  - Probability calibration via CalibratedClassifierCV
  - Permutation feature importance analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from studytrade.ml.cv import PurgedWalkForwardCV


class ModelType(StrEnum):
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"


@dataclass(frozen=True)
class CVFoldResult:
    """Metrics for a single walk-forward cross-validation fold."""

    fold: int
    train_size: int
    test_size: int
    accuracy: float
    precision: float
    recall: float
    roc_auc: float
    brier_score: float


def build_estimator(model_type: ModelType | str, random_state: int = 42) -> object:
    """Factory creating uncalibrated base estimator."""
    m_type = ModelType(model_type) if isinstance(model_type, str) else model_type

    if m_type == ModelType.LOGISTIC_REGRESSION:
        return LogisticRegression(C=0.1, max_iter=500, random_state=random_state)
    elif m_type == ModelType.RANDOM_FOREST:
        return RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_leaf=15, random_state=random_state
        )
    elif m_type == ModelType.GRADIENT_BOOSTING:
        return HistGradientBoostingClassifier(
            max_depth=4, min_samples_leaf=15, random_state=random_state
        )
    raise ValueError(f"Unknown model type: {model_type}")


class CalibratedModelWrapper:
    """
    Wraps an estimator with probability calibration and strict walk-forward evaluation.
    """

    def __init__(
        self,
        model_type: ModelType | str = ModelType.GRADIENT_BOOSTING,
        calibrate: bool = True,
        random_state: int = 42,
    ) -> None:
        self.model_type = model_type
        self.calibrate = calibrate
        self.random_state = random_state
        self.model: object = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> CalibratedModelWrapper:
        """Fit model with optional probability calibration."""
        base_est = build_estimator(self.model_type, self.random_state)

        if self.calibrate:
            # Calibrate probabilities using isotonic regression or sigmoid
            calibrated = CalibratedClassifierCV(estimator=base_est, method="sigmoid", cv=3)
            calibrated.fit(x, y)
            self.model = calibrated
        else:
            base_est.fit(x, y)
            self.model = base_est

        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return calibrated class probabilities."""
        if self.model is None:
            raise ValueError("Model has not been fitted")
        return self.model.predict_proba(x)

    def evaluate_walk_forward(
        self,
        x: np.ndarray,
        y: np.ndarray,
        cv: PurgedWalkForwardCV,
    ) -> list[CVFoldResult]:
        """
        Run purged walk-forward cross-validation.
        """
        results: list[CVFoldResult] = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(x, y)):
            x_train, y_train = x[train_idx], y[train_idx]
            x_test, y_test = x[test_idx], y[test_idx]

            # Check class balance
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            base_est = build_estimator(self.model_type, self.random_state)
            base_est.fit(x_train, y_train)

            y_pred = base_est.predict(x_test)
            y_proba = base_est.predict_proba(x_test)[:, 1]

            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0))
            try:
                auc = float(roc_auc_score(y_test, y_proba))
            except ValueError:
                auc = 0.5
            brier = float(brier_score_loss(y_test, y_proba))

            results.append(
                CVFoldResult(
                    fold=fold_idx,
                    train_size=len(train_idx),
                    test_size=len(test_idx),
                    accuracy=acc,
                    precision=prec,
                    recall=rec,
                    roc_auc=auc,
                    brier_score=brier,
                )
            )

        return results

    def compute_feature_importance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ) -> pd.DataFrame:
        """
        Compute permutation feature importances on holdout data.
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")

        res = permutation_importance(self.model, x, y, n_repeats=10, random_state=self.random_state)
        return (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance_mean": res.importances_mean,
                    "importance_std": res.importances_std,
                }
            )
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
