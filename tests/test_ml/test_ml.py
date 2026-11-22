"""
Tests for studytrade.ml — Walk-forward purged CV, target labeling, and calibrated models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from studytrade.ml.cv import PurgedWalkForwardCV
from studytrade.ml.models import CalibratedModelWrapper, ModelType
from studytrade.ml.targets import (
    create_direction_target,
    create_triple_barrier_labels,
    create_volatility_expansion_target,
)


class TestPurgedWalkForwardCV:
    """Validate zero-leakage cross-validation properties."""

    def test_embargo_and_monotonicity(self):
        n = 200
        x = np.arange(n).reshape(-1, 1)
        y = np.random.randint(0, 2, n)

        embargo = 5
        cv = PurgedWalkForwardCV(n_splits=3, min_train_size=80, embargo_size=embargo)

        for train_idx, test_idx in cv.split(x, y):
            # 1. Train must strictly precede Test
            assert np.max(train_idx) < np.min(test_idx)
            # 2. Embargo buffer must be respected
            gap = np.min(test_idx) - np.max(train_idx)
            assert gap >= embargo
            # 3. No overlap
            assert len(set(train_idx).intersection(set(test_idx))) == 0


class TestTargetEngineering:
    """Validate target labeling rules."""

    def test_create_direction_target(self):
        # 10 bars: price doubles over 3 bars
        prices = pd.Series([100, 102, 104, 150, 155, 160, 120, 110, 105, 100])
        target = create_direction_target(prices, horizon=3, threshold_pct=5.0)
        # Bar 0: price goes from 100 to 150 (+50%) -> Target should be 1
        assert target.iloc[0] == 1
        # Last 3 bars should be NaN (future unknown)
        assert target.iloc[-3:].isna().all()

    def test_volatility_expansion_target(self):
        atr = pd.Series([10.0, 10.0, 10.0, 25.0, 30.0, 30.0])
        target = create_volatility_expansion_target(atr, horizon=3, expansion_threshold=1.5)
        # Bar 0: ATR 10 -> ATR 25 (+150%) -> Target should be 1.0
        assert target.iloc[0] == 1.0

    def test_triple_barrier_labeling(self):
        closes = pd.Series([100.0, 101.0, 103.0, 102.0, 99.0])
        highs = closes + 1.0
        lows = closes - 1.0
        labels = create_triple_barrier_labels(
            closes, highs, lows, horizon=2, profit_take_pct=2.0, stop_loss_pct=1.0
        )
        assert len(labels) == len(closes)


class TestCalibratedModels:
    """Validate ML model training and calibration."""

    @pytest.mark.parametrize(
        "m_type",
        [ModelType.LOGISTIC_REGRESSION, ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING],
    )
    def test_model_fit_and_calibrate(self, m_type):
        rng = np.random.default_rng(42)
        n = 150
        x = rng.normal(0, 1, size=(n, 4))
        # Logistic probability relation
        logits = x[:, 0] * 1.5 - x[:, 1] * 0.8
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.uniform(0, 1, n) < probs).astype(int)

        model = CalibratedModelWrapper(model_type=m_type, calibrate=True, random_state=42)
        model.fit(x, y)

        pred_probs = model.predict_proba(x)
        assert pred_probs.shape == (n, 2)
        # Probabilities sum to 1
        np.testing.assert_allclose(pred_probs.sum(axis=1), np.ones(n), atol=1e-5)

    def test_walk_forward_evaluation(self):
        rng = np.random.default_rng(42)
        n = 180
        x = rng.normal(0, 1, size=(n, 3))
        y = rng.choice([0, 1], size=n)

        cv = PurgedWalkForwardCV(n_splits=3, min_train_size=60, embargo_size=5)
        model = CalibratedModelWrapper(model_type=ModelType.LOGISTIC_REGRESSION, calibrate=False)
        results = model.evaluate_walk_forward(x, y, cv)

        assert len(results) > 0
        for r in results:
            assert 0.0 <= r.accuracy <= 1.0
            assert 0.0 <= r.brier_score <= 1.0
