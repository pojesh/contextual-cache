"""Tests for SessionContextAccumulator and SessionManager."""

import asyncio
import time

import numpy as np
import pytest

from contextual_cache.session_context import SessionContextAccumulator, SessionManager


class TestSessionContextAccumulator:
    def test_first_turn_returns_raw(self):
        acc = SessionContextAccumulator("s1", alpha=0.85, embed_dim=64)
        vec = np.random.randn(64).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        result = acc.update(vec)
        np.testing.assert_array_almost_equal(result, vec)
        assert acc.turn_count == 1

    def test_fusion_is_query_dominant(self):
        acc = SessionContextAccumulator("s1", alpha=0.85, embed_dim=64)
        v1 = np.random.randn(64).astype(np.float32)
        v1 = v1 / np.linalg.norm(v1)
        acc.update(v1)  # first turn

        v2 = np.random.randn(64).astype(np.float32)
        v2 = v2 / np.linalg.norm(v2)
        fused = acc.update(v2)

        # Fused should be closer to v2 than v1
        sim_to_v2 = np.dot(fused, v2)
        sim_to_v1 = np.dot(fused, v1)
        assert sim_to_v2 > sim_to_v1, "Fused should be query-dominant"

    def test_fused_is_unit_norm(self):
        acc = SessionContextAccumulator("s1", alpha=0.85, embed_dim=64)
        for _ in range(5):
            vec = np.random.randn(64).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            fused = acc.update(vec)
            assert abs(np.linalg.norm(fused) - 1.0) < 1e-5

    def test_expiry(self):
        acc = SessionContextAccumulator("s1")
        acc.last_active = time.time() - 2000
        assert acc.is_expired

    def test_not_expired(self):
        acc = SessionContextAccumulator("s1")
        assert not acc.is_expired

    def test_reset(self):
        acc = SessionContextAccumulator("s1", embed_dim=64)
        vec = np.random.randn(64).astype(np.float32)
        acc.update(vec)
        acc.reset()
        assert acc.turn_count == 0
        assert acc.context_vec is None


class TestSessionManager:
    def test_get_or_create(self):
        sm = SessionManager()
        s1 = asyncio.run(sm.get_or_create("session-1"))
        assert s1.session_id == "session-1"
        assert sm.active_count == 1

    def test_same_session_returns_same(self):
        sm = SessionManager()
        s1 = asyncio.run(sm.get_or_create("x"))
        s2 = asyncio.run(sm.get_or_create("x"))
        assert s1 is s2

    def test_different_sessions(self):
        sm = SessionManager()
        s1 = asyncio.run(sm.get_or_create("a"))
        s2 = asyncio.run(sm.get_or_create("b"))
        assert s1 is not s2
        assert sm.active_count == 2

    def test_expired_session_resets(self):
        sm = SessionManager()
        s = asyncio.run(sm.get_or_create("x"))
        v = np.random.randn(384).astype(np.float32)
        s.update(v)
        assert s.turn_count == 1

        # Force expiry
        s.last_active = time.time() - 3600

        s2 = asyncio.run(sm.get_or_create("x"))
        # Session was reset on expiry
        assert s2.context_vec is None
        assert s2.turn_count == 0
