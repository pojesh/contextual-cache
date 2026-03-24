"""
Shard-local Thompson Sampling bandit for global threshold prior learning.

Each shard runs independent Thompson Sampling; shard posteriors are
periodically merged via FedAvg.  ADWIN triggers resync on distribution
shift.  Per-entry conformal thresholds are the actual decision mechanism;
the bandit learns a *prior* for new entries.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

import numpy as np

from .config import settings
from .drift_detection import ADWINDriftDetector

logger = logging.getLogger(__name__)


class ShardLocalBanditAdaptor:
    """
    Thompson Sampling over threshold arms with drift detection.
    
    Arms: linearly spaced thresholds in [0.65, 0.95]
    Posterior: Beta(α_i, β_i) per arm
    
    FedAvg sync: average α, β across shards every `sync_interval_s`.
    Drift detection: ADWIN resets posteriors to weakly informative prior
    on detected distribution change.
    """

    def __init__(
        self,
        shard_id: str = "main",
        n_arms: int = settings.bandit_n_arms,
        sync_interval_s: float = settings.bandit_sync_interval_s,
        drift_delta: float = settings.drift_delta,
    ) -> None:
        self.shard_id = shard_id
        self.n_arms = n_arms
        self.sync_interval = sync_interval_s

        # Threshold arms
        self.threshold_arms = np.linspace(0.65, 0.95, n_arms)

        # Beta distribution parameters (Thompson Sampling)
        self.alpha = np.ones(n_arms, dtype=np.float64)  # successes
        self.beta_params = np.ones(n_arms, dtype=np.float64)   # failures

        # ADWIN drift detector
        self.drift_detector = ADWINDriftDetector(delta=drift_delta)

        # Sync tracking
        self.last_sync = time.time()
        self.total_updates = 0
        self.drift_resets = 0

    def sample_threshold(self) -> Tuple[int, float]:
        """
        Thompson Sampling: sample from each arm's Beta posterior,
        pick the arm with highest sample.
        
        Returns (arm_index, threshold_value).
        """
        samples = np.random.beta(self.alpha, self.beta_params)
        best_arm = int(np.argmax(samples))
        return best_arm, float(self.threshold_arms[best_arm])

    def update(self, arm: int, reward: float) -> None:
        """
        Update arm posterior.

        reward = 1.0: correct cache hit
        reward = 0.0: incorrect hit (false positive)
        reward = 0.5: uncertain (default when no feedback) — skipped
        """
        if reward == 0.5:
            return  # uncertain feedback — don't pollute posterior

        self.alpha[arm] += reward
        self.beta_params[arm] += (1.0 - reward)
        self.total_updates += 1

        # Feed to drift detector
        self.drift_detector.add_element(reward)

        if self.drift_detector.detected_change():
            logger.warning(
                "Distribution drift detected on shard %s — resetting posteriors.",
                self.shard_id,
            )
            self.alpha = np.ones(self.n_arms) * 2
            self.beta_params = np.ones(self.n_arms) * 2
            self.drift_resets += 1

    def get_current_best(self) -> Tuple[int, float]:
        """Return the arm with highest expected reward (α / (α + β))."""
        expected = self.alpha / (self.alpha + self.beta_params)
        best = int(np.argmax(expected))
        return best, float(self.threshold_arms[best])

    def get_sync_params(self) -> Dict:
        """Export parameters for FedAvg sync."""
        return {
            "shard_id": self.shard_id,
            "alpha": self.alpha.tolist(),
            "beta": self.beta_params.tolist(),
            "timestamp": time.time(),
        }

    def apply_fedavg_update(self, peer_params: List[Dict]) -> None:
        """
        FedAvg: average α, β across shards.
        Mathematically equivalent to pooling observations under uniform
        Dirichlet prior.
        """
        all_alphas = [self.alpha] + [np.array(p["alpha"]) for p in peer_params]
        all_betas = [self.beta_params] + [np.array(p["beta"]) for p in peer_params]

        self.alpha = np.mean(all_alphas, axis=0)
        self.beta_params = np.mean(all_betas, axis=0)
        self.last_sync = time.time()

    def get_stats(self) -> dict:
        expected = self.alpha / (self.alpha + self.beta_params)
        best_arm, best_threshold = self.get_current_best()
        return {
            "shard_id": self.shard_id,
            "n_arms": self.n_arms,
            "total_updates": self.total_updates,
            "drift_resets": self.drift_resets,
            "best_arm": best_arm,
            "best_threshold": round(best_threshold, 4),
            "arm_thresholds": self.threshold_arms.tolist(),
            "arm_expected_rewards": [round(v, 4) for v in expected.tolist()],
            "arm_alphas": [round(v, 2) for v in self.alpha.tolist()],
            "arm_betas": [round(v, 2) for v in self.beta_params.tolist()],
        }
