"""
StudyTrade — Purged Walk-Forward Time-Series Cross-Validation
==============================================================

Prevents serial correlation leakage and look-ahead bias in financial ML models.
Implements the Purged K-Fold with Embargo mechanism (Marcos Lopez de Prado, 2018).

Key Rules:
  1. Train timestamps must strictly precede Test timestamps: max(Train) < min(Test).
  2. Embargo gap: An embargo buffer of E bars separates train and test sets
     to prevent autocorrelation leakage from overlapping rolling indicators (e.g. 20-day ATR).
  3. No random shuffling ever.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


class PurgedWalkForwardCV:
    """
    Expanding or rolling walk-forward time-series splitter with embargo.
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_size: int = 100,
        test_size: int | None = None,
        embargo_size: int = 5,
        rolling: bool = False,
    ) -> None:
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.embargo_size = embargo_size
        self.rolling = rolling

    def split(
        self,
        x: np.ndarray,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate train and test indices with embargo.
        """
        n_samples = len(x)
        if n_samples < self.min_train_size + self.embargo_size + 10:
            raise ValueError(
                f"Insufficient samples ({n_samples}) for {self.min_train_size} min train size"
            )

        avail_samples = n_samples - self.min_train_size - self.embargo_size
        fold_test_size = self.test_size or max(10, avail_samples // self.n_splits)

        for i in range(self.n_splits):
            test_start = self.min_train_size + self.embargo_size + (i * fold_test_size)
            test_end = min(n_samples, test_start + fold_test_size)

            if test_start >= n_samples:
                break

            train_end = test_start - self.embargo_size
            train_start = max(0, train_end - self.min_train_size) if self.rolling else 0

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            if len(train_indices) == 0 or len(test_indices) == 0:
                continue

            yield train_indices, test_indices

    def get_n_splits(self, x: np.ndarray | None = None) -> int:
        return self.n_splits
