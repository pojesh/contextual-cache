"""
Semantic-W-TinyLFU admission policy.

Adapts W-TinyLFU (Einziger et al., 2017) to the semantic cache domain:
- Frequency estimation via Count-Min Sketch over LSH-bucketed embeddings
- Window cache (1 % capacity) admits freely; main cache (99 %) is
  frequency-gated so one-hit wonders are rejected.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .config import settings
from .data_structures import CountMinSketch, LRUCache, RandomProjectionLSH, SLRUCache
from .models import CacheEntry

logger = logging.getLogger(__name__)


class SemanticWTinyLFUAdmission:
    """
    Two-stage admission:
      1. If window has space → admit immediately (cold-start friendly)
      2. If window full → admit only if freq(new) > freq(evict_candidate)

    Frequency is estimated via Count-Min Sketch keyed by LSH bucket
    of the embedding vector, so semantically similar queries contribute
    to the same frequency bucket.
    """

    def __init__(
        self,
        capacity: int = settings.cache_capacity,
        window_pct: float = settings.window_pct,
        cms_width: int = settings.cms_width,
        cms_depth: int = settings.cms_depth,
        lsh_bits: int = settings.lsh_bits,
        lsh_tables: int = settings.lsh_tables,
        embed_dim: int = settings.embedding_dim,
    ) -> None:
        self.capacity = capacity
        self.window_size = max(1, int(capacity * window_pct))
        self.main_size = capacity - self.window_size

        # Count-Min Sketch over LSH buckets
        self.sketch = CountMinSketch(width=cms_width, depth=cms_depth)

        # LSH for semantic bucketing
        self.lsh = RandomProjectionLSH(
            input_dim=embed_dim, n_bits=lsh_bits, n_tables=lsh_tables
        )

        # Window (LRU) and main (SLRU) caches
        self.window_cache = LRUCache(maxsize=self.window_size)
        self.main_cache = SLRUCache(maxsize=self.main_size)

        # Stats
        self.total_checks = 0
        self.total_admitted = 0
        self.total_rejected = 0

    def should_admit(self, entry: CacheEntry) -> bool:
        """
        Decide whether to admit `entry` into the cache.

        Standard W-TinyLFU flow:
        1. Record access in the frequency sketch.
        2. If window has space → admit freely, store in window.
        3. If window full → compare new entry frequency against window
           LRU victim. Winner stays in window; loser is rejected (or
           victim is promoted to main cache).
        """
        self.total_checks += 1

        # Record access frequency in the sketch
        self.record_access(entry.embedding)

        # Window has space → admit freely
        if len(self.window_cache) < self.window_size:
            self.window_cache.put(entry.entry_id, entry)
            self.total_admitted += 1
            return True

        # Window full → frequency gate
        victim = self.window_cache.get_lru_victim()
        if victim is None:
            self.window_cache.put(entry.entry_id, entry)
            self.total_admitted += 1
            return True

        _victim_key, victim_entry = victim
        new_freq = self._estimate_frequency(entry.embedding)

        if isinstance(victim_entry, CacheEntry):
            victim_freq = self._estimate_frequency(victim_entry.embedding)
        else:
            victim_freq = 0

        if new_freq > victim_freq:
            # Evict window LRU victim → promote to main cache
            self.window_cache.remove(_victim_key)
            if isinstance(victim_entry, CacheEntry):
                self.main_cache.put(victim_entry.entry_id, victim_entry)
            # New entry takes a window slot
            self.window_cache.put(entry.entry_id, entry)
            self.total_admitted += 1
            return True

        self.total_rejected += 1
        return False

    def on_access(self, entry_id: str, embedding: np.ndarray) -> None:
        """Record a cache hit — update frequency and promote within policy caches."""
        self.record_access(embedding)
        # If the entry is in probationary, SLRU.get() promotes it
        self.main_cache.get(entry_id)

    def on_evict(self, entry_id: str) -> None:
        """Remove an entry from the policy caches when it is evicted."""
        self.window_cache.remove(entry_id)
        self.main_cache.remove(entry_id)

    def record_access(self, embedding: np.ndarray) -> None:
        """Increment frequency estimate for this embedding's LSH bucket."""
        lsh_key = self.lsh.hash(embedding)
        self.sketch.increment(lsh_key)

    def _estimate_frequency(self, embedding: np.ndarray) -> int:
        lsh_key = self.lsh.hash(embedding)
        return self.sketch.estimate(lsh_key)

    def get_stats(self) -> dict:
        return {
            "total_checks": self.total_checks,
            "total_admitted": self.total_admitted,
            "total_rejected": self.total_rejected,
            "admission_rate": self.total_admitted / max(self.total_checks, 1),
            "window_size": len(self.window_cache),
            "window_capacity": self.window_size,
            "main_size": len(self.main_cache),
            "main_capacity": self.main_size,
        }
