"""Tests for CircuitBreaker, ADWINDriftDetector, and ShardLocalBandit."""

import asyncio

import numpy as np
import pytest

from contextual_cache.circuit_breaker import CircuitBreaker, CircuitBreakerError, CircuitState
from contextual_cache.drift_detection import ADWINDriftDetector
from contextual_cache.bandit import ShardLocalBanditAdaptor


def run(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.run(coro)


class TestCircuitBreaker:
    def test_closed_state_passes(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        result = run(cb.call(self._success))
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_failures(self):
        cb = CircuitBreaker("test", failure_threshold=2, reset_timeout_s=100)
        for _ in range(2):
            with pytest.raises(ValueError):
                run(cb.call(self._fail))
        assert cb.state == CircuitState.OPEN

    def test_open_rejects_calls(self):
        cb = CircuitBreaker("test", failure_threshold=1, reset_timeout_s=100)
        with pytest.raises(ValueError):
            run(cb.call(self._fail))
        with pytest.raises(CircuitBreakerError):
            run(cb.call(self._success))

    def test_half_open_on_timeout(self):
        cb = CircuitBreaker("test", failure_threshold=1, reset_timeout_s=0)
        with pytest.raises(ValueError):
            run(cb.call(self._fail))
        # Reset timeout = 0 → immediately half-open
        assert cb.state == CircuitState.HALF_OPEN

    def test_recovery(self):
        cb = CircuitBreaker("test", failure_threshold=1, reset_timeout_s=0)
        with pytest.raises(ValueError):
            run(cb.call(self._fail))
        # Now half-open; success should close it
        result = run(cb.call(self._success))
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED

    def test_stats(self):
        cb = CircuitBreaker("test")
        stats = cb.get_stats()
        assert stats["name"] == "test"
        assert stats["state"] == "closed"

    @staticmethod
    async def _success():
        return "ok"

    @staticmethod
    async def _fail():
        raise ValueError("fail")


class TestADWINDriftDetector:
    def test_no_drift_on_stable(self):
        d = ADWINDriftDetector(delta=0.01)
        for _ in range(100):
            d.add_element(0.5)
        assert not d.detected_change()

    def test_drift_on_shift(self):
        d = ADWINDriftDetector(delta=0.1, max_window=200)
        for _ in range(80):
            d.add_element(0.9)
        detected = False
        for _ in range(80):
            d.add_element(0.1)
            if d.detected_change():
                detected = True
                break
        assert detected, "Should detect drift when mean shifts"

    def test_reset(self):
        d = ADWINDriftDetector()
        d.add_element(1.0)
        d.reset()
        assert d.window_size == 0


class TestShardLocalBandit:
    def test_sample_threshold(self):
        b = ShardLocalBanditAdaptor(shard_id="s1", n_arms=5)
        arm, threshold = b.sample_threshold()
        assert 0 <= arm < 5
        assert 0.65 <= threshold <= 0.95

    def test_update_changes_posterior(self):
        b = ShardLocalBanditAdaptor(n_arms=5)
        initial_alpha = b.alpha.copy()
        b.update(0, 1.0)
        assert b.alpha[0] > initial_alpha[0]

    def test_get_current_best(self):
        b = ShardLocalBanditAdaptor(n_arms=5)
        # Give arm 2 many successes
        for _ in range(50):
            b.update(2, 1.0)
        best_arm, _ = b.get_current_best()
        assert best_arm == 2

    def test_fedavg_sync(self):
        b = ShardLocalBanditAdaptor(n_arms=3)
        b.alpha = np.array([10.0, 1.0, 1.0])
        peer = {"alpha": [1.0, 10.0, 1.0], "beta": [1.0, 1.0, 1.0]}
        b.apply_fedavg_update([peer])
        # After averaging, alpha[0] and alpha[1] should be closer
        assert abs(b.alpha[0] - b.alpha[1]) < abs(10.0 - 1.0)

    def test_stats(self):
        b = ShardLocalBanditAdaptor(shard_id="test")
        stats = b.get_stats()
        assert stats["shard_id"] == "test"
        assert stats["n_arms"] == 10
