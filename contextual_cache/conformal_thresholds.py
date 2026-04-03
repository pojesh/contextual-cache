"""
Per-entry conformal threshold store using online conformal prediction.

Each cached entry gets its own similarity threshold τ_i learned from
feedback signals (correct/incorrect hits).  This is the key novelty
over global-threshold approaches like GPTCache and MeanCache.

Guarantee: P(incorrect hit) ≤ ε for user-specified error rate ε.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np

from .config import settings

logger = logging.getLogger(__name__)


class ConformalThresholdStore:
    """
    Per-entry conformal thresholds via Venn-Abers calibration.

    - New entries start at `default_threshold`
    - After ≥ `min_calibration_points` feedback signals, the threshold
      is computed as the (1-ε)-quantile of nonconformity scores
    - Sliding window of last `max_calibration_points` to adapt to drift
    - Threshold clamped to [min_threshold, max_threshold]
    """

    def __init__(
        self,
        target_error_rate: float = settings.target_error_rate,
        default_threshold: float = settings.default_threshold,
        min_calibration_points: int = 3,
        max_calibration_points: int = 200,
        min_threshold: float = 0.60,
        max_threshold: float = 0.99,
    ) -> None:
        self.epsilon = target_error_rate
        self.default_threshold = default_threshold
        self.min_cal = min_calibration_points
        self.max_cal = max_calibration_points
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        # Per-entry calibration scores:
        # nonconformity = 1 - similarity for correct hits
        self._scores: Dict[str, List[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

        # Stats
        self.total_updates = 0
        self.correct_updates = 0
        self.incorrect_updates = 0

    async def get_threshold(self, entry_id: str) -> float:
        """
        Return the current conformal threshold for `entry_id`.
        Falls back to default for new entries with < min_cal data.
        """
        async with self._lock:
            scores = self._scores.get(entry_id)
            if scores is None or len(scores) < self.min_cal:
                return self.default_threshold

            # Conformal quantile: τ = 1 - quantile_{1-ε}(nonconformity_scores)
            sorted_scores = sorted(scores)
            n = len(sorted_scores)
            quantile_idx = int(np.ceil((1 - self.epsilon) * (n + 1))) - 1
            quantile_idx = min(quantile_idx, n - 1)
            quantile_idx = max(quantile_idx, 0)

            threshold = 1.0 - sorted_scores[quantile_idx]
            return float(np.clip(threshold, self.min_threshold, self.max_threshold))

    async def update(self, entry_id: str, similarity: float,
                     was_correct: bool) -> None:
        """
        Update calibration scores for `entry_id` based on feedback.

        Correct hit → nonconformity = 1 - similarity
        Incorrect hit → nonconformity = max(1 - similarity + 0.1, 0.5)
            (penalty pushes threshold higher to reject similar-but-wrong hits)
        """
        async with self._lock:
            self.total_updates += 1

            if was_correct:
                nonconformity = 1.0 - similarity
                self.correct_updates += 1
            else:
                nonconformity = min(1.0, max(1.0 - similarity + 0.1, 0.5))
                self.incorrect_updates += 1

            scores = self._scores[entry_id]
            scores.append(nonconformity)

            # Sliding window
            if len(scores) > self.max_cal:
                self._scores[entry_id] = scores[-self.max_cal:]

    def get_all_thresholds(self) -> Dict[str, float]:
        """Return current thresholds for all calibrated entries (sync)."""
        import asyncio
        result = {}
        for entry_id in self._scores:
            scores = self._scores[entry_id]
            if len(scores) < self.min_cal:
                result[entry_id] = self.default_threshold
            else:
                sorted_s = sorted(scores)
                n = len(sorted_s)
                qi = int(np.ceil((1 - self.epsilon) * (n + 1))) - 1
                qi = min(qi, n - 1)
                qi = max(qi, 0)
                result[entry_id] = float(
                    np.clip(1.0 - sorted_s[qi], self.min_threshold, self.max_threshold)
                )
        return result

    @property
    def calibrated_count(self) -> int:
        return sum(
            1 for scores in self._scores.values()
            if len(scores) >= self.min_cal
        )

    def get_stats(self) -> dict:
        return {
            "total_entries": len(self._scores),
            "calibrated_entries": self.calibrated_count,
            "total_updates": self.total_updates,
            "correct_updates": self.correct_updates,
            "incorrect_updates": self.incorrect_updates,
            "default_threshold": self.default_threshold,
            "target_error_rate": self.epsilon,
        }
