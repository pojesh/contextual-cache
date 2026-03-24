"""Tests for CostAwareEvictor and SemanticWTinyLFUAdmission."""

import numpy as np
import pytest

from contextual_cache.eviction import CostAwareEvictor
from contextual_cache.models import CacheEntry
from contextual_cache.admission_policy import SemanticWTinyLFUAdmission


class TestCostAwareEvictor:
    def _make_entry(self, entry_id: str, freq: int = 1,
                    cost: float = 0.001, embed_ms: float = 50.0,
                    size: int = 1000) -> CacheEntry:
        return CacheEntry(
            entry_id=entry_id,
            query_text=f"query {entry_id}",
            response_text="response " * 20,
            embedding=np.random.randn(384).astype(np.float32),
            frequency=freq,
            llm_cost_usd=cost,
            embed_cost_ms=embed_ms,
            storage_bytes=size,
        )

    def test_evict_lowest_priority(self):
        evictor = CostAwareEvictor()
        # Expensive entry (should survive)
        e1 = self._make_entry("expensive", freq=10, cost=0.01)
        # Cheap entry (should be evicted)
        e2 = self._make_entry("cheap", freq=1, cost=0.0001)

        evictor.register(e1)
        evictor.register(e2)

        evicted = evictor.evict_one()
        assert evicted == "cheap"

    def test_frequency_matters(self):
        evictor = CostAwareEvictor()
        e1 = self._make_entry("popular", freq=100, cost=0.001)
        e2 = self._make_entry("rare", freq=1, cost=0.001)

        evictor.register(e1)
        evictor.register(e2)

        evicted = evictor.evict_one()
        assert evicted == "rare"

    def test_inflation_factor_updates(self):
        evictor = CostAwareEvictor()
        e1 = self._make_entry("a", freq=1, cost=0.001)
        evictor.register(e1)

        assert evictor.L == 0.0
        evictor.evict_one()
        assert evictor.L > 0.0  # updated after eviction

    def test_record_access(self):
        evictor = CostAwareEvictor()
        e = self._make_entry("x", freq=1)
        evictor.register(e)
        initial_h = evictor._entries["x"].H

        evictor.record_access("x")
        updated_h = evictor._entries["x"].H
        assert updated_h > initial_h

    def test_empty_eviction(self):
        evictor = CostAwareEvictor()
        assert evictor.evict_one() is None

    def test_stats(self):
        evictor = CostAwareEvictor()
        e = self._make_entry("a")
        evictor.register(e)
        stats = evictor.get_stats()
        assert stats["tracked_entries"] == 1


class TestSemanticWTinyLFUAdmission:
    def _make_entry(self, entry_id: str) -> CacheEntry:
        return CacheEntry(
            entry_id=entry_id,
            query_text=f"query {entry_id}",
            response_text="response",
            embedding=np.random.randn(384).astype(np.float32),
        )

    def test_window_admits_freely(self):
        policy = SemanticWTinyLFUAdmission(
            capacity=100, window_pct=0.1, embed_dim=384
        )
        # Window size = 10, should admit first entries freely
        for i in range(5):
            entry = self._make_entry(f"e{i}")
            assert policy.should_admit(entry) is True

    def test_tracks_stats(self):
        policy = SemanticWTinyLFUAdmission(capacity=50, embed_dim=384)
        e = self._make_entry("x")
        policy.should_admit(e)
        stats = policy.get_stats()
        assert stats["total_checks"] == 1
