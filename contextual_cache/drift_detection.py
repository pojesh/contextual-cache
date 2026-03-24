"""
ADWIN-based drift detection for concept shift in query distributions.

Triggers bandit posterior reset when the reward distribution changes,
preventing stale threshold arms from dominating.

Simplified ADWIN (Adaptive Windowing) — maintains a window of observations
and splits it into two sub-windows; detects change when their means differ
by more than a bound determined by the Hoeffding inequality.
"""

from __future__ import annotations

import math
from collections import deque


class ADWINDriftDetector:
    """
    Lightweight ADWIN implementation.

    - Maintains a sliding window of reward observations
    - On each new observation, checks if the mean of recent observations
      differs significantly from the mean of older observations
    - Significance threshold is controlled by `delta` (lower = more sensitive)
    """

    def __init__(self, delta: float = 0.002, max_window: int = 5000) -> None:
        self.delta = delta
        self.max_window = max_window
        self._window: deque[float] = deque(maxlen=max_window)
        self._change_detected = False

    def add_element(self, value: float) -> None:
        """Add a new observation and run drift detection."""
        self._window.append(value)
        self._change_detected = False

        if len(self._window) < 20:
            return  # need minimum observations

        self._check_drift()

    def detected_change(self) -> bool:
        """Returns True if drift was detected on the last add_element call."""
        return self._change_detected

    def reset(self) -> None:
        self._window.clear()
        self._change_detected = False

    def _check_drift(self) -> None:
        """
        Split window at every point and check if means differ significantly.
        Uses Hoeffding bound for the comparison.
        """
        n = len(self._window)
        total_sum = sum(self._window)

        left_sum = 0.0
        for split in range(1, n):
            left_sum += self._window[split - 1]
            if split < 10 or (n - split) < 10:
                continue  # skip edges for statistical stability

            left_mean = left_sum / split
            right_n = n - split
            right_mean = (total_sum - left_sum) / right_n

            # Hoeffding bound
            m = 1.0 / split + 1.0 / right_n
            epsilon_cut = math.sqrt(
                0.5 * m * math.log(4.0 * n / self.delta)
            )

            if abs(left_mean - right_mean) >= epsilon_cut:
                self._change_detected = True
                # Trim window to the more recent sub-window
                for _ in range(split):
                    self._window.popleft()
                return

    @property
    def window_size(self) -> int:
        return len(self._window)
