"""
Baseline cache implementations for benchmarking.

Each baseline implements the same BaseCache interface:
  - async query(text, embedding) -> CacheResult
  - stats() -> dict
  - reset() -> None

All share the same EmbeddingService for fair comparison.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Shared types ──────────────────────────────────────────────────

@dataclass
class CacheResult:
    """Result from a cache query."""
    hit: bool
    response: str
    tier: int = 0            # 0=miss, 1=exact, 2=semantic
    latency_ms: float = 0.0
    similarity: float = 0.0
    entry_id: Optional[str] = None


@dataclass
class CachedItem:
    """Single cached entry."""
    entry_id: str
    key: str
    query_text: str
    response_text: str
    embedding: Optional[np.ndarray]
    frequency: int = 1
    llm_cost_usd: float = 0.0
    embed_cost_ms: float = 0.0
    storage_bytes: int = 0
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)


# ── Abstract base ─────────────────────────────────────────────────

class BaseCache(ABC):
    """Interface all benchmark caches implement."""

    def __init__(self, name: str, capacity: int = 200):
        self.name = name
        self.capacity = capacity
        self.total_queries = 0
        self.total_hits = 0
        self.total_misses = 0
        self.hit_latencies: List[float] = []
        self.miss_latencies: List[float] = []

    @abstractmethod
    async def query(self, text: str, embedding: Optional[np.ndarray],
                    response: str, llm_latency_ms: float) -> CacheResult:
        """
        Query the cache. If miss, store the response.
        
        Args:
            text: query string
            embedding: precomputed embedding (shared across all strategies)
            response: the ground-truth / LLM response to store on miss
            llm_latency_ms: simulated LLM latency for misses
        
        Returns CacheResult.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    def stats(self) -> dict:
        return {
            "name": self.name,
            "total_queries": self.total_queries,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": round(self.total_hits / max(self.total_queries, 1), 4),
            "avg_hit_latency_ms": round(
                sum(self.hit_latencies) / max(len(self.hit_latencies), 1), 2
            ),
            "avg_miss_latency_ms": round(
                sum(self.miss_latencies) / max(len(self.miss_latencies), 1), 2
            ),
            "avg_overall_latency_ms": round(
                (sum(self.hit_latencies) + sum(self.miss_latencies))
                / max(self.total_queries, 1), 2
            ),
            "llm_calls": self.total_misses,
        }

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().strip())

    @staticmethod
    def _hash_key(text: str) -> str:
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ═══════════════════════════════════════════════════════════════════
# 1. Exact-Match LRU — no semantics at all
# ═══════════════════════════════════════════════════════════════════

class ExactMatchLRUCache(BaseCache):
    """Traditional exact-hash cache with LRU eviction."""

    def __init__(self, capacity: int = 200):
        super().__init__("Exact-Match LRU", capacity)
        self._store: OrderedDict[str, CachedItem] = OrderedDict()

    async def query(self, text: str, embedding: Optional[np.ndarray],
                    response: str, llm_latency_ms: float) -> CacheResult:
        t0 = time.monotonic()
        self.total_queries += 1
        key = self._hash_key(text)

        if key in self._store:
            self._store.move_to_end(key)
            item = self._store[key]
            item.frequency += 1
            item.last_access = time.time()
            ms = (time.monotonic() - t0) * 1000
            self.total_hits += 1
            self.hit_latencies.append(ms)
            return CacheResult(hit=True, response=item.response_text,
                               tier=1, latency_ms=ms, similarity=1.0, entry_id=item.entry_id)

        # Miss — store
        self.total_misses += 1
        ms = (time.monotonic() - t0) * 1000 + llm_latency_ms
        self.miss_latencies.append(ms)

        item = CachedItem(entry_id=str(uuid.uuid4()), key=key, query_text=text, response_text=response,
                          embedding=None, storage_bytes=len(response.encode()))
        self._store[key] = item
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)

        return CacheResult(hit=False, response=response, tier=0, latency_ms=ms)

    def reset(self):
        self._store.clear()
        self.total_queries = self.total_hits = self.total_misses = 0
        self.hit_latencies.clear()
        self.miss_latencies.clear()


# ═══════════════════════════════════════════════════════════════════
# 2. GPTCache-style — static threshold 0.80, LRU eviction
# ═══════════════════════════════════════════════════════════════════

class GPTCacheStyleCache(BaseCache):
    """FAISS-less semantic cache with static threshold and LRU eviction."""

    def __init__(self, capacity: int = 200, threshold: float = 0.75):
        super().__init__("GPTCache", capacity)
        self.threshold = threshold
        self._store: OrderedDict[str, CachedItem] = OrderedDict()
        self._embeddings: Dict[str, np.ndarray] = {}

    async def query(self, text: str, embedding: Optional[np.ndarray],
                    response: str, llm_latency_ms: float) -> CacheResult:
        t0 = time.monotonic()
        self.total_queries += 1

        # Exact check first
        key = self._hash_key(text)
        if key in self._store:
            self._store.move_to_end(key)
            item = self._store[key]
            ms = (time.monotonic() - t0) * 1000
            self.total_hits += 1
            self.hit_latencies.append(ms)
            return CacheResult(hit=True, response=item.response_text,
                               tier=1, latency_ms=ms, similarity=1.0, entry_id=item.entry_id)

        # Semantic check
        if embedding is not None and self._embeddings:
            best_sim = -1.0
            best_key = None
            for ek, ev in self._embeddings.items():
                sim = float(np.dot(embedding, ev))
                if sim > best_sim:
                    best_sim = sim
                    best_key = ek

            if best_sim >= self.threshold and best_key in self._store:
                self._store.move_to_end(best_key)
                item = self._store[best_key]
                ms = (time.monotonic() - t0) * 1000
                self.total_hits += 1
                self.hit_latencies.append(ms)
                return CacheResult(hit=True, response=item.response_text,
                                   tier=2, latency_ms=ms, similarity=best_sim, entry_id=item.entry_id)

        # Miss
        self.total_misses += 1
        ms = (time.monotonic() - t0) * 1000 + llm_latency_ms
        self.miss_latencies.append(ms)

        item = CachedItem(entry_id=str(uuid.uuid4()), key=key, query_text=text, response_text=response,
                          embedding=embedding, storage_bytes=len(response.encode()))
        self._store[key] = item
        if embedding is not None:
            self._embeddings[key] = embedding.copy()
        if len(self._store) > self.capacity:
            evicted_key, _ = self._store.popitem(last=False)
            self._embeddings.pop(evicted_key, None)

        return CacheResult(hit=False, response=response, tier=0, latency_ms=ms)

    def reset(self):
        self._store.clear()
        self._embeddings.clear()
        self.total_queries = self.total_hits = self.total_misses = 0
        self.hit_latencies.clear()
        self.miss_latencies.clear()


# ═══════════════════════════════════════════════════════════════════
# 3. MeanCache-style — adaptive global τ + PCA compression
# ═══════════════════════════════════════════════════════════════════

class MeanCacheStyleCache(BaseCache):
    """
    Approximates MeanCache (Gill et al., 2024):
    - Adaptive global threshold (running mean of hit similarities)
    - PCA-compressed embeddings (128d from 384d)
    - LRU eviction
    """

    def __init__(self, capacity: int = 200, pca_dim: int = 128,
                 initial_threshold: float = 0.82):
        super().__init__("MeanCache", capacity)
        self.pca_dim = pca_dim
        self.threshold = initial_threshold
        self._similarity_history: List[float] = []
        self._pca_matrix: Optional[np.ndarray] = None
        self._store: OrderedDict[str, CachedItem] = OrderedDict()
        self._embeddings: Dict[str, np.ndarray] = {}

    def _compress(self, vec: np.ndarray) -> np.ndarray:
        """PCA-like random projection to pca_dim dimensions."""
        if self._pca_matrix is None:
            rng = np.random.RandomState(42)
            self._pca_matrix = rng.randn(len(vec), self.pca_dim).astype(np.float32)
            # Orthogonalize
            q, _ = np.linalg.qr(self._pca_matrix)
            self._pca_matrix = q[:, :self.pca_dim]
        projected = vec @ self._pca_matrix
        norm = np.linalg.norm(projected)
        return projected / norm if norm > 0 else projected

    def _update_threshold(self, sim: float):
        """Adaptive threshold: running mean of recent hit similarities."""
        self._similarity_history.append(sim)
        if len(self._similarity_history) > 200:
            self._similarity_history = self._similarity_history[-200:]
        # Threshold = mean - 1 * std (allows more hits while staying precise)
        arr = np.array(self._similarity_history)
        self.threshold = max(0.65, float(np.mean(arr) - np.std(arr)))

    async def query(self, text: str, embedding: Optional[np.ndarray],
                    response: str, llm_latency_ms: float) -> CacheResult:
        t0 = time.monotonic()
        self.total_queries += 1

        # Exact check
        key = self._hash_key(text)
        if key in self._store:
            self._store.move_to_end(key)
            item = self._store[key]
            ms = (time.monotonic() - t0) * 1000
            self.total_hits += 1
            self.hit_latencies.append(ms)
            self._update_threshold(1.0)
            return CacheResult(hit=True, response=item.response_text,
                               tier=1, latency_ms=ms, similarity=1.0, entry_id=item.entry_id)

        # Semantic check with compressed embeddings
        if embedding is not None and self._embeddings:
            compressed = self._compress(embedding)
            best_sim = -1.0
            best_key = None
            for ek, ev in self._embeddings.items():
                sim = float(np.dot(compressed, ev))
                if sim > best_sim:
                    best_sim = sim
                    best_key = ek

            if best_sim >= self.threshold and best_key in self._store:
                self._store.move_to_end(best_key)
                item = self._store[best_key]
                ms = (time.monotonic() - t0) * 1000
                self.total_hits += 1
                self.hit_latencies.append(ms)
                self._update_threshold(best_sim)
                return CacheResult(hit=True, response=item.response_text,
                                   tier=2, latency_ms=ms, similarity=best_sim, entry_id=item.entry_id)

        # Miss
        self.total_misses += 1
        ms = (time.monotonic() - t0) * 1000 + llm_latency_ms
        self.miss_latencies.append(ms)

        compressed_emb = self._compress(embedding) if embedding is not None else None
        item = CachedItem(entry_id=str(uuid.uuid4()), key=key, query_text=text, response_text=response,
                          embedding=compressed_emb, storage_bytes=len(response.encode()))
        self._store[key] = item
        if compressed_emb is not None:
            self._embeddings[key] = compressed_emb.copy()
        if len(self._store) > self.capacity:
            evicted_key, _ = self._store.popitem(last=False)
            self._embeddings.pop(evicted_key, None)

        return CacheResult(hit=False, response=response, tier=0, latency_ms=ms)

    def reset(self):
        self._store.clear()
        self._embeddings.clear()
        self._similarity_history.clear()
        self._pca_matrix = None
        self.threshold = 0.82
        self.total_queries = self.total_hits = self.total_misses = 0
        self.hit_latencies.clear()
        self.miss_latencies.clear()


# ═══════════════════════════════════════════════════════════════════
# 4. vCache-style — per-entry conformal thresholds, LRU eviction
# ═══════════════════════════════════════════════════════════════════

class VCacheStyleCache(BaseCache):
    """
    Approximates vCache (Schroeder et al., 2025):
    - Per-entry conformal thresholds with online calibration
    - No admission gate, LRU eviction
    """

    def __init__(self, capacity: int = 200, target_error: float = 0.05,
                 default_threshold: float = 0.80):
        super().__init__("vCache", capacity)
        self.target_error = target_error
        self.default_threshold = default_threshold
        self._store: OrderedDict[str, CachedItem] = OrderedDict()
        self._embeddings: Dict[str, np.ndarray] = {}
        self._thresholds: Dict[str, float] = {}
        self._scores: Dict[str, List[float]] = {}  # calibration scores per entry

    def _get_threshold(self, key: str) -> float:
        scores = self._scores.get(key)
        if not scores or len(scores) < 3:
            return self.default_threshold
        arr = sorted(scores)
        idx = max(0, int((1.0 - self.target_error) * len(arr)) - 1)
        tau = arr[idx]
        return max(0.60, min(0.99, tau))

    def _update_threshold(self, key: str, similarity: float, correct: bool):
        if key not in self._scores:
            self._scores[key] = []
        score = similarity if correct else similarity + 0.1
        self._scores[key].append(score)
        if len(self._scores[key]) > 50:
            self._scores[key] = self._scores[key][-50:]
        self._thresholds[key] = self._get_threshold(key)

    async def query(self, text: str, embedding: Optional[np.ndarray],
                    response: str, llm_latency_ms: float) -> CacheResult:
        t0 = time.monotonic()
        self.total_queries += 1

        # Exact check
        key = self._hash_key(text)
        if key in self._store:
            self._store.move_to_end(key)
            item = self._store[key]
            ms = (time.monotonic() - t0) * 1000
            self.total_hits += 1
            self.hit_latencies.append(ms)
            self._update_threshold(key, 1.0, True)
            return CacheResult(hit=True, response=item.response_text,
                               tier=1, latency_ms=ms, similarity=1.0, entry_id=item.entry_id)

        # Semantic check with per-entry thresholds
        if embedding is not None and self._embeddings:
            best_sim = -1.0
            best_key = None
            for ek, ev in self._embeddings.items():
                sim = float(np.dot(embedding, ev))
                entry_tau = self._thresholds.get(ek, self.default_threshold)
                if sim >= entry_tau and sim > best_sim:
                    best_sim = sim
                    best_key = ek

            if best_key is not None and best_key in self._store:
                self._store.move_to_end(best_key)
                item = self._store[best_key]
                ms = (time.monotonic() - t0) * 1000
                self.total_hits += 1
                self.hit_latencies.append(ms)
                # Simulate correctness check via text overlap
                correct = self._check_correctness(response, item.response_text)
                self._update_threshold(best_key, best_sim, correct)
                return CacheResult(hit=True, response=item.response_text,
                                   tier=2, latency_ms=ms, similarity=best_sim, entry_id=item.entry_id)

        # Miss
        self.total_misses += 1
        ms = (time.monotonic() - t0) * 1000 + llm_latency_ms
        self.miss_latencies.append(ms)

        item = CachedItem(entry_id=str(uuid.uuid4()), key=key, query_text=text, response_text=response,
                          embedding=embedding, storage_bytes=len(response.encode()))
        self._store[key] = item
        self._thresholds[key] = self.default_threshold
        if embedding is not None:
            self._embeddings[key] = embedding.copy()
        if len(self._store) > self.capacity:
            evicted_key, _ = self._store.popitem(last=False)
            self._embeddings.pop(evicted_key, None)
            self._thresholds.pop(evicted_key, None)
            self._scores.pop(evicted_key, None)

        return CacheResult(hit=False, response=response, tier=0, latency_ms=ms)

    @staticmethod
    def _check_correctness(expected: str, cached: str) -> bool:
        """Simple ROUGE-L-like overlap check."""
        e_words = set(expected.lower().split())
        c_words = set(cached.lower().split())
        if not e_words:
            return True
        overlap = len(e_words & c_words) / max(len(e_words), 1)
        return overlap >= 0.3

    def reset(self):
        self._store.clear()
        self._embeddings.clear()
        self._thresholds.clear()
        self._scores.clear()
        self.total_queries = self.total_hits = self.total_misses = 0
        self.hit_latencies.clear()
        self.miss_latencies.clear()


# ═══════════════════════════════════════════════════════════════════
# 5. RAGCache-style — static threshold + cost-aware GDSF eviction
# ═══════════════════════════════════════════════════════════════════

class RAGCacheStyleCache(BaseCache):
    """
    Approximates RAGCache (Jin et al., 2024):
    - Static similarity threshold (τ=0.85)
    - Cost-aware GDSF eviction (frequency × regen_cost / size + L)
    """

    def __init__(self, capacity: int = 200, threshold: float = 0.85):
        super().__init__("RAGCache", capacity)
        self.threshold = threshold
        self._store: Dict[str, CachedItem] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._priorities: Dict[str, float] = {}
        self._inflation_L: float = 0.0

    def _priority(self, item: CachedItem) -> float:
        regen_cost = (item.llm_cost_usd * 1e6) + item.embed_cost_ms + 50
        return (item.frequency * regen_cost) / max(item.storage_bytes, 1) + self._inflation_L

    def _evict_lowest(self):
        if not self._priorities:
            return
        victim_key = min(self._priorities, key=lambda k: self._priorities[k])
        self._inflation_L = self._priorities[victim_key]
        self._store.pop(victim_key, None)
        self._embeddings.pop(victim_key, None)
        self._priorities.pop(victim_key, None)

    async def query(self, text: str, embedding: Optional[np.ndarray],
                    response: str, llm_latency_ms: float) -> CacheResult:
        t0 = time.monotonic()
        self.total_queries += 1

        key = self._hash_key(text)
        if key in self._store:
            item = self._store[key]
            item.frequency += 1
            item.last_access = time.time()
            self._priorities[key] = self._priority(item)
            ms = (time.monotonic() - t0) * 1000
            self.total_hits += 1
            self.hit_latencies.append(ms)
            return CacheResult(hit=True, response=item.response_text,
                               tier=1, latency_ms=ms, similarity=1.0, entry_id=item.entry_id)

        if embedding is not None and self._embeddings:
            best_sim = -1.0
            best_key = None
            for ek, ev in self._embeddings.items():
                sim = float(np.dot(embedding, ev))
                if sim > best_sim:
                    best_sim = sim
                    best_key = ek

            if best_sim >= self.threshold and best_key in self._store:
                item = self._store[best_key]
                item.frequency += 1
                item.last_access = time.time()
                self._priorities[best_key] = self._priority(item)
                ms = (time.monotonic() - t0) * 1000
                self.total_hits += 1
                self.hit_latencies.append(ms)
                return CacheResult(hit=True, response=item.response_text,
                                   tier=2, latency_ms=ms, similarity=best_sim, entry_id=item.entry_id)

        # Miss
        self.total_misses += 1
        ms = (time.monotonic() - t0) * 1000 + llm_latency_ms
        self.miss_latencies.append(ms)

        while len(self._store) >= self.capacity:
            self._evict_lowest()

        item = CachedItem(entry_id=str(uuid.uuid4()), key=key, query_text=text, response_text=response,
                          embedding=embedding, llm_cost_usd=llm_latency_ms * 0.00001,
                          embed_cost_ms=5.0, storage_bytes=len(response.encode()))
        self._store[key] = item
        self._priorities[key] = self._priority(item)
        if embedding is not None:
            self._embeddings[key] = embedding.copy()

        return CacheResult(hit=False, response=response, tier=0, latency_ms=ms)

    def reset(self):
        self._store.clear()
        self._embeddings.clear()
        self._priorities.clear()
        self._inflation_L = 0.0
        self.total_queries = self.total_hits = self.total_misses = 0
        self.hit_latencies.clear()
        self.miss_latencies.clear()


# ═══════════════════════════════════════════════════════════════════
# 6. No-Admission Semantic — our system minus W-TinyLFU
# ═══════════════════════════════════════════════════════════════════

class NoAdmissionSemanticCache(BaseCache):
    """
    Ablation: conformal thresholds + CostGDSF eviction but NO admission gate.
    Tests the contribution of the W-TinyLFU admission policy.
    """

    def __init__(self, capacity: int = 200, target_error: float = 0.05,
                 default_threshold: float = 0.80):
        super().__init__("No-Admission", capacity)
        self.target_error = target_error
        self.default_threshold = default_threshold
        self._store: Dict[str, CachedItem] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._thresholds: Dict[str, float] = {}
        self._scores: Dict[str, List[float]] = {}
        self._priorities: Dict[str, float] = {}
        self._inflation_L: float = 0.0

    def _get_threshold(self, key: str) -> float:
        scores = self._scores.get(key)
        if not scores or len(scores) < 3:
            return self.default_threshold
        arr = sorted(scores)
        idx = max(0, int((1.0 - self.target_error) * len(arr)) - 1)
        tau = arr[idx]
        return max(0.60, min(0.99, tau))

    def _priority(self, item: CachedItem) -> float:
        regen_cost = (item.llm_cost_usd * 1e6) + item.embed_cost_ms + 50
        return (item.frequency * regen_cost) / max(item.storage_bytes, 1) + self._inflation_L

    def _evict_lowest(self):
        if not self._priorities:
            return
        victim_key = min(self._priorities, key=lambda k: self._priorities[k])
        self._inflation_L = self._priorities[victim_key]
        self._store.pop(victim_key, None)
        self._embeddings.pop(victim_key, None)
        self._thresholds.pop(victim_key, None)
        self._scores.pop(victim_key, None)
        self._priorities.pop(victim_key, None)

    async def query(self, text: str, embedding: Optional[np.ndarray],
                    response: str, llm_latency_ms: float) -> CacheResult:
        t0 = time.monotonic()
        self.total_queries += 1

        key = self._hash_key(text)
        if key in self._store:
            item = self._store[key]
            item.frequency += 1
            item.last_access = time.time()
            self._priorities[key] = self._priority(item)
            ms = (time.monotonic() - t0) * 1000
            self.total_hits += 1
            self.hit_latencies.append(ms)
            return CacheResult(hit=True, response=item.response_text,
                               tier=1, latency_ms=ms, similarity=1.0, entry_id=item.entry_id)

        if embedding is not None and self._embeddings:
            best_sim = -1.0
            best_key = None
            for ek, ev in self._embeddings.items():
                sim = float(np.dot(embedding, ev))
                entry_tau = self._thresholds.get(ek, self.default_threshold)
                if sim >= entry_tau and sim > best_sim:
                    best_sim = sim
                    best_key = ek

            if best_key is not None and best_key in self._store:
                item = self._store[best_key]
                item.frequency += 1
                item.last_access = time.time()
                self._priorities[best_key] = self._priority(item)
                ms = (time.monotonic() - t0) * 1000
                self.total_hits += 1
                self.hit_latencies.append(ms)
                correct = VCacheStyleCache._check_correctness(response, item.response_text)
                if best_key not in self._scores:
                    self._scores[best_key] = []
                score = best_sim if correct else best_sim + 0.1
                self._scores[best_key].append(score)
                if len(self._scores[best_key]) > 50:
                    self._scores[best_key] = self._scores[best_key][-50:]
                self._thresholds[best_key] = self._get_threshold(best_key)
                return CacheResult(hit=True, response=item.response_text,
                                   tier=2, latency_ms=ms, similarity=best_sim, entry_id=item.entry_id)

        # Miss — ALWAYS admit (no admission gate)
        self.total_misses += 1
        ms = (time.monotonic() - t0) * 1000 + llm_latency_ms
        self.miss_latencies.append(ms)

        while len(self._store) >= self.capacity:
            self._evict_lowest()

        item = CachedItem(entry_id=str(uuid.uuid4()), key=key, query_text=text, response_text=response,
                          embedding=embedding, llm_cost_usd=llm_latency_ms * 0.00001,
                          embed_cost_ms=5.0, storage_bytes=len(response.encode()))
        self._store[key] = item
        self._thresholds[key] = self.default_threshold
        self._priorities[key] = self._priority(item)
        if embedding is not None:
            self._embeddings[key] = embedding.copy()

        return CacheResult(hit=False, response=response, tier=0, latency_ms=ms)

    def reset(self):
        self._store.clear()
        self._embeddings.clear()
        self._thresholds.clear()
        self._scores.clear()
        self._priorities.clear()
        self._inflation_L = 0.0
        self.total_queries = self.total_hits = self.total_misses = 0
        self.hit_latencies.clear()
        self.miss_latencies.clear()


# ═══════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════

def create_all_baselines(capacity: int = 200) -> List[BaseCache]:
    """Create all baseline cache instances."""
    return [
        ExactMatchLRUCache(capacity),
        GPTCacheStyleCache(capacity),
        MeanCacheStyleCache(capacity),
        VCacheStyleCache(capacity),
        RAGCacheStyleCache(capacity),
        NoAdmissionSemanticCache(capacity),
    ]
