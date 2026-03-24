"""Tests for the SQLite persistence layer."""

import numpy as np
import pytest

from contextual_cache.models import CacheEntry
from contextual_cache.persistence import PersistenceLayer


class TestPersistenceLayer:

    @pytest.fixture
    async def persistence(self, tmp_path):
        db_path = str(tmp_path / "test_cache.db")
        p = PersistenceLayer(db_path=db_path)
        await p.initialize()
        yield p
        await p.close()

    def _make_entry(self, entry_id="e1", dim=384):
        vec = np.random.randn(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return CacheEntry(
            entry_id=entry_id,
            query_text="What is Python?",
            response_text="Python is a programming language.",
            embedding=vec,
        )

    async def test_save_and_load_entry(self, persistence):
        entry = self._make_entry()
        await persistence.save_entry(entry)

        loaded = await persistence.load_all_entries()
        assert len(loaded) == 1
        assert loaded[0].entry_id == "e1"
        assert loaded[0].query_text == "What is Python?"
        assert loaded[0].response_text == "Python is a programming language."

    async def test_embedding_round_trip(self, persistence):
        entry = self._make_entry()
        original_embedding = entry.embedding.copy()
        await persistence.save_entry(entry)

        loaded = await persistence.load_all_entries()
        np.testing.assert_allclose(
            loaded[0].embedding, original_embedding, rtol=1e-6
        )

    async def test_delete_entry(self, persistence):
        entry = self._make_entry()
        await persistence.save_entry(entry)
        await persistence.delete_entry("e1")

        loaded = await persistence.load_all_entries()
        assert len(loaded) == 0

    async def test_flush_multiple_entries(self, persistence):
        entries = [self._make_entry(f"e{i}") for i in range(5)]
        await persistence.flush_entries(entries)

        loaded = await persistence.load_all_entries()
        assert len(loaded) == 5

    async def test_save_and_load_bandit_state(self, persistence):
        alpha = np.array([1.0, 2.0, 3.0])
        beta = np.array([4.0, 5.0, 6.0])

        await persistence.save_bandit_state("main", alpha, beta,
                                             total_updates=100,
                                             drift_resets=2)

        state = await persistence.load_bandit_state("main")
        assert state is not None
        np.testing.assert_allclose(state["alpha"], alpha)
        np.testing.assert_allclose(state["beta"], beta)
        assert state["total_updates"] == 100
        assert state["drift_resets"] == 2

    async def test_load_nonexistent_bandit_state(self, persistence):
        state = await persistence.load_bandit_state("nonexistent")
        assert state is None

    async def test_conformal_scores_round_trip(self, persistence):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        await persistence.save_conformal_scores("e1", scores)

        loaded = await persistence.load_conformal_scores()
        assert "e1" in loaded
        np.testing.assert_allclose(loaded["e1"], scores, rtol=1e-10)

    async def test_expired_entries_not_loaded(self, persistence):
        import time
        entry = self._make_entry()
        entry.expires_at = time.time() - 100  # already expired
        await persistence.save_entry(entry)

        loaded = await persistence.load_all_entries()
        assert len(loaded) == 0
