"""Tests for ConformalThresholdStore."""

import asyncio

import numpy as np
import pytest

from contextual_cache.conformal_thresholds import ConformalThresholdStore


def run(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.run(coro)


class TestConformalThresholdStore:
    def test_default_threshold_for_new_entry(self):
        store = ConformalThresholdStore(default_threshold=0.80)
        tau = run(store.get_threshold("new_entry"))
        assert tau == 0.80

    def test_threshold_after_calibration(self):
        store = ConformalThresholdStore(
            target_error_rate=0.05,
            min_calibration_points=5,
            default_threshold=0.80,
        )

        async def _run():
            for _ in range(20):
                await store.update("e1", similarity=0.92, was_correct=True)
            return await store.get_threshold("e1")

        tau = run(_run())
        assert 0.60 <= tau <= 0.99

    def test_incorrect_hits_affect_threshold(self):
        store = ConformalThresholdStore(
            min_calibration_points=5,
            target_error_rate=0.05,
        )

        async def _run():
            for _ in range(10):
                await store.update("e1", similarity=0.85, was_correct=True)
            tau_before = await store.get_threshold("e1")

            for _ in range(5):
                await store.update("e1", similarity=0.82, was_correct=False)
            tau_after = await store.get_threshold("e1")
            return tau_before, tau_after

        tau_before, tau_after = run(_run())
        # Both thresholds should be within valid bounds
        assert 0.60 <= tau_before <= 0.99
        assert 0.60 <= tau_after <= 0.99

    def test_threshold_clamped(self):
        store = ConformalThresholdStore(min_threshold=0.60, max_threshold=0.99)

        async def _run():
            for _ in range(30):
                await store.update("e1", similarity=0.99, was_correct=True)
            return await store.get_threshold("e1")

        tau = run(_run())
        assert 0.60 <= tau <= 0.99

    def test_sliding_window(self):
        store = ConformalThresholdStore(max_calibration_points=20)

        async def _run():
            for _ in range(50):
                await store.update("e1", similarity=0.90, was_correct=True)

        run(_run())
        assert len(store._scores["e1"]) == 20

    def test_stats(self):
        store = ConformalThresholdStore()

        async def _run():
            await store.update("e1", 0.9, True)
            await store.update("e1", 0.8, False)

        run(_run())
        stats = store.get_stats()
        assert stats["total_updates"] == 2
        assert stats["correct_updates"] == 1
        assert stats["incorrect_updates"] == 1
