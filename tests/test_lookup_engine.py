"""Tests for the two-tier lookup engine including FAISS IndexIDMap2,
normalization consistency, TTL, and index rebuild."""

import time

import numpy as np
import pytest

from contextual_cache.conformal_thresholds import ConformalThresholdStore
from contextual_cache.lookup_engine import TwoTierLookupEngine
from contextual_cache.models import CacheEntry


def _make_entry(entry_id: str, query: str, dim: int = 384,
                expires_at=None) -> CacheEntry:
    vec = np.random.randn(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return CacheEntry(
        entry_id=entry_id,
        query_text=query,
        response_text=f"Response for {query}",
        embedding=vec,
        expires_at=expires_at,
    )


class TestTwoTierLookupEngine:
    """Core store/remove/search operations."""

    @pytest.fixture
    def engine(self):
        return TwoTierLookupEngine(embedding_dim=384)

    @pytest.fixture
    def threshold_store(self):
        return ConformalThresholdStore()

    async def test_store_and_exact_lookup(self, engine, threshold_store):
        entry = _make_entry("e1", "What is Python?")
        await engine.store(entry)

        result = await engine.lookup(
            "What is Python?", query_embedding=None,
            session=None, threshold_store=threshold_store,
        )
        assert result.hit
        assert result.tier == 1
        assert result.entry_id == "e1"

    async def test_store_increments_faiss_size(self, engine):
        assert engine.faiss_index_size == 0
        entry = _make_entry("e1", "test")
        await engine.store(entry)
        assert engine.faiss_index_size == 1
        assert engine.size == 1

    async def test_remove_decrements_size(self, engine):
        entry = _make_entry("e1", "test")
        await engine.store(entry)
        assert engine.size == 1
        await engine.remove("e1")
        assert engine.size == 0

    async def test_remove_soft_deletes_from_faiss(self, engine):
        entry = _make_entry("e1", "test")
        await engine.store(entry)
        assert engine.faiss_index_size == 1
        await engine.remove("e1")
        # HNSW doesn't support true deletion; vector remains until rebuild
        assert engine.faiss_index_size == 1
        assert engine.size == 0  # but logical size is 0

    async def test_remove_clears_exact_store(self, engine, threshold_store):
        entry = _make_entry("e1", "What is Python?")
        await engine.store(entry)
        await engine.remove("e1")

        result = await engine.lookup(
            "What is Python?", query_embedding=None,
            session=None, threshold_store=threshold_store,
        )
        assert not result.hit

    async def test_semantic_search_finds_similar(self, engine, threshold_store):
        vec = np.random.randn(384).astype(np.float32)
        vec /= np.linalg.norm(vec)

        entry = CacheEntry(
            entry_id="e1", query_text="original",
            response_text="Answer", embedding=vec.copy(),
        )
        await engine.store(entry)

        # Search with same embedding should hit
        result = await engine.lookup(
            "different text", query_embedding=vec.copy(),
            session=None, threshold_store=threshold_store,
        )
        assert result.hit
        assert result.tier == 2
        assert result.similarity > 0.99

    async def test_multiple_store_remove_cycle(self, engine):
        for i in range(10):
            entry = _make_entry(f"e{i}", f"query {i}")
            await engine.store(entry)

        assert engine.size == 10
        assert engine.faiss_index_size == 10

        for i in range(5):
            await engine.remove(f"e{i}")

        assert engine.size == 5
        # FAISS vectors are soft-deleted; cleaned up during rebuild
        assert engine.faiss_index_size == 10

    async def test_rebuild_index_cleans_up_faiss(self, engine):
        for i in range(10):
            entry = _make_entry(f"e{i}", f"query {i}")
            await engine.store(entry)

        # Remove some entries (soft-delete)
        for i in range(5):
            await engine.remove(f"e{i}")

        assert engine.size == 5
        assert engine.faiss_index_size == 10  # stale vectors remain

        # Rebuild should compact the index
        engine.rebuild_index()
        assert engine.size == 5
        assert engine.faiss_index_size == 5  # now cleaned up


class TestNormalizationConsistency:
    """Ensure lookup engine and embedding service use same normalization."""

    def test_normalize_strips_punctuation(self):
        engine = TwoTierLookupEngine()
        assert engine.normalize_query("What's up?") == "whats up"

    def test_normalize_collapses_whitespace(self):
        engine = TwoTierLookupEngine()
        assert engine.normalize_query("  hello   world  ") == "hello world"

    def test_embedding_cache_key_consistency(self):
        from contextual_cache.embedding_service import EmbeddingService
        from contextual_cache.utils import normalize_text

        # Both should produce the same normalized text
        engine = TwoTierLookupEngine()
        query = "What's the meaning of life?"
        normalized = engine.normalize_query(query)
        assert normalized == normalize_text(query)


class TestTTLExpiry:
    """TTL integration in lookup engine."""

    @pytest.fixture
    def engine(self):
        return TwoTierLookupEngine(embedding_dim=384)

    @pytest.fixture
    def threshold_store(self):
        return ConformalThresholdStore()

    async def test_expired_entry_not_returned(self, engine, threshold_store):
        vec = np.random.randn(384).astype(np.float32)
        vec /= np.linalg.norm(vec)

        entry = CacheEntry(
            entry_id="e1", query_text="test",
            response_text="Answer", embedding=vec.copy(),
            expires_at=time.time() - 10,  # already expired
        )
        await engine.store(entry)

        result = await engine.lookup(
            "different", query_embedding=vec.copy(),
            session=None, threshold_store=threshold_store,
        )
        assert not result.hit

    async def test_non_expired_entry_returned(self, engine, threshold_store):
        vec = np.random.randn(384).astype(np.float32)
        vec /= np.linalg.norm(vec)

        entry = CacheEntry(
            entry_id="e1", query_text="test",
            response_text="Answer", embedding=vec.copy(),
            expires_at=time.time() + 3600,  # expires in 1 hour
        )
        await engine.store(entry)

        result = await engine.lookup(
            "different", query_embedding=vec.copy(),
            session=None, threshold_store=threshold_store,
        )
        assert result.hit

    async def test_get_expired_entry_ids(self, engine):
        e1 = _make_entry("e1", "fresh", expires_at=time.time() + 3600)
        e2 = _make_entry("e2", "stale", expires_at=time.time() - 10)
        await engine.store(e1)
        await engine.store(e2)

        expired = engine.get_expired_entry_ids()
        assert "e2" in expired
        assert "e1" not in expired
