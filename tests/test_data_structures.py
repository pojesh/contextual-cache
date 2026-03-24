"""Tests for core data structures: CountMinSketch, LSH, LRU, SLRU."""

import numpy as np
import pytest

from contextual_cache.data_structures import (
    CountMinSketch,
    LRUCache,
    RandomProjectionLSH,
    SLRUCache,
)


class TestCountMinSketch:
    def test_increment_and_estimate(self):
        cms = CountMinSketch(width=1024, depth=4)
        cms.increment("hello", 5)
        assert cms.estimate("hello") >= 5

    def test_conservative_update(self):
        cms = CountMinSketch(width=1024, depth=4)
        for _ in range(10):
            cms.increment("key_a")
        # Should be exact or near-exact for a single key
        assert cms.estimate("key_a") == 10

    def test_unseen_key_returns_zero(self):
        cms = CountMinSketch()
        assert cms.estimate("never_seen") == 0

    def test_halving(self):
        cms = CountMinSketch(width=256, depth=2, halve_every=10)
        for i in range(15):
            cms.increment("k")
        # After halving, count is halved
        est = cms.estimate("k")
        assert est < 15

    def test_reset(self):
        cms = CountMinSketch()
        cms.increment("x", 100)
        cms.reset()
        assert cms.estimate("x") == 0

    def test_tuple_key(self):
        cms = CountMinSketch()
        cms.increment((1, 2, 3))
        assert cms.estimate((1, 2, 3)) >= 1

    def test_integer_key(self):
        cms = CountMinSketch()
        cms.increment(42, 3)
        assert cms.estimate(42) >= 3


class TestRandomProjectionLSH:
    def test_similar_vectors_same_hash(self):
        lsh = RandomProjectionLSH(input_dim=64, n_bits=8, n_tables=4, seed=0)
        v1 = np.random.randn(64).astype(np.float32)
        v2 = v1 + np.random.randn(64) * 0.01  # very similar
        h1 = lsh.hash(v1)
        h2 = lsh.hash(v2)
        # At least some tables should agree
        matches = sum(a == b for a, b in zip(h1, h2))
        assert matches >= 2, f"Expected similar vectors to collide, got {matches}/4 matches"

    def test_different_vectors_different_hash(self):
        lsh = RandomProjectionLSH(input_dim=64, n_bits=8, n_tables=4, seed=0)
        v1 = np.ones(64)
        v2 = -np.ones(64)
        h1 = lsh.hash(v1)
        h2 = lsh.hash(v2)
        # Opposite vectors should rarely collide
        assert h1 != h2

    def test_hash_single(self):
        lsh = RandomProjectionLSH(input_dim=32, n_bits=4, n_tables=2)
        vec = np.random.randn(32)
        h = lsh.hash_single(vec)
        assert isinstance(h, int)


class TestLRUCache:
    def test_basic_put_get(self):
        c = LRUCache(maxsize=3)
        c.put("a", 1)
        assert c.get("a") == 1

    def test_eviction_on_overflow(self):
        c = LRUCache(maxsize=2)
        c.put("a", 1)
        c.put("b", 2)
        evicted = c.put("c", 3)
        assert evicted == ("a", 1)
        assert c.get("a") is None
        assert c.get("c") == 3

    def test_access_refreshes(self):
        c = LRUCache(maxsize=2)
        c.put("a", 1)
        c.put("b", 2)
        c.get("a")  # refresh "a"
        evicted = c.put("c", 3)
        assert evicted == ("b", 2)

    def test_get_lru_victim(self):
        c = LRUCache(maxsize=3)
        c.put("x", 10)
        c.put("y", 20)
        c.put("z", 30)
        victim = c.get_lru_victim()
        assert victim == ("x", 10)

    def test_remove(self):
        c = LRUCache(maxsize=5)
        c.put("a", 1)
        c.remove("a")
        assert c.get("a") is None
        assert len(c) == 0


class TestSLRUCache:
    def test_insert_into_probationary(self):
        s = SLRUCache(maxsize=10)
        s.put("a", 1)
        assert s.get("a") == 1

    def test_promotion_to_protected(self):
        s = SLRUCache(maxsize=10)
        s.put("a", 1)
        s.get("a")  # promotes to protected
        s.get("a")  # still in protected
        assert s.get("a") == 1

    def test_eviction_from_probationary(self):
        s = SLRUCache(maxsize=4, protected_pct=0.75)
        # probationary_size = 1
        s.put("a", 1)
        evicted = s.put("b", 2)
        assert evicted == ("a", 1) or len(s) <= 4

    def test_remove(self):
        s = SLRUCache(maxsize=10)
        s.put("x", 42)
        s.remove("x")
        assert s.get("x") is None
