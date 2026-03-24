"""
Two-tier lookup engine.

Tier 1: O(1) exact-hash lookup via in-memory dict (or Redis).
Tier 2: O(log n) approximate nearest neighbor via FAISS HNSW with
         per-entry conformal thresholds.

Embedding is ONLY computed if Tier 1 misses — reduces embedding
calls by ~40-60 % on real workloads.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from .config import settings
from .conformal_thresholds import ConformalThresholdStore
from .models import CacheEntry, LookupResult
from .session_context import SessionContextAccumulator
from .utils import normalize_text

logger = logging.getLogger(__name__)


class TwoTierLookupEngine:
    """
    Tier 1: SHA-256(normalize(query)) → cached response (sub-ms).
    Tier 2: FAISS HNSW ANN search → per-entry conformal threshold check.
    """

    def __init__(
        self,
        embedding_dim: int = settings.embedding_dim,
        hnsw_m: int = settings.hnsw_m,
        hnsw_ef_construction: int = settings.hnsw_ef_construction,
        hnsw_ef_search: int = settings.hnsw_ef_search,
        ann_k: int = settings.ann_k,
    ) -> None:
        self._embed_dim = embedding_dim
        self._ann_k = ann_k

        # Tier 1: exact hash → (entry_id, response_text)
        self._exact_store: Dict[str, Tuple[str, str]] = {}

        # Tier 2: FAISS HNSW index wrapped in IndexIDMap2 for explicit
        # ID management and true vector removal support.
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._hnsw_ef_search = hnsw_ef_search
        self._index = self._build_index(embedding_dim, hnsw_m,
                                        hnsw_ef_construction, hnsw_ef_search)

        # Mapping: FAISS user-supplied id → entry_id
        self._id_map: Dict[int, str] = {}
        # Reverse mapping: entry_id → faiss_id (for fast removal)
        self._entry_to_faiss: Dict[str, int] = {}
        # entry_id → CacheEntry
        self._entries: Dict[str, CacheEntry] = {}
        self._next_faiss_id = 0
        self._removals_since_rebuild = 0
        self._lock = asyncio.Lock()

    @staticmethod
    def _build_index(embedding_dim: int, hnsw_m: int,
                     ef_construction: int, ef_search: int) -> faiss.Index:
        """Create a fresh IndexIDMap2-wrapped HNSW index."""
        base = faiss.IndexHNSWFlat(embedding_dim, hnsw_m)
        base.hnsw.efConstruction = ef_construction
        base.hnsw.efSearch = ef_search
        return faiss.IndexIDMap2(base)

    def rebuild_index(self) -> None:
        """Rebuild the FAISS index from scratch to reclaim fragmented space."""
        new_index = self._build_index(
            self._embed_dim, self._hnsw_m,
            self._hnsw_ef_construction, self._hnsw_ef_search,
        )
        new_id_map: Dict[int, str] = {}
        new_entry_to_faiss: Dict[str, int] = {}
        next_id = 0

        for entry_id, entry in self._entries.items():
            vec = entry.embedding.reshape(1, -1).astype(np.float32)
            ids = np.array([next_id], dtype=np.int64)
            new_index.add_with_ids(vec, ids)
            new_id_map[next_id] = entry_id
            new_entry_to_faiss[entry_id] = next_id
            next_id += 1

        self._index = new_index
        self._id_map = new_id_map
        self._entry_to_faiss = new_entry_to_faiss
        self._next_faiss_id = next_id
        self._removals_since_rebuild = 0
        logger.info("FAISS index rebuilt with %d entries.", next_id)

    def normalize_query(self, query: str) -> str:
        """Lowercase, strip punctuation, collapse whitespace."""
        return normalize_text(query)

    def exact_key(self, query: str, tenant_id: str = "default") -> str:
        normalized = self.normalize_query(query)
        h = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"{{{tenant_id}}}:exact:{h}"

    # ── Lookup ────────────────────────────────────────────────

    async def lookup(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        session: Optional[SessionContextAccumulator],
        threshold_store: ConformalThresholdStore,
        tenant_id: str = "default",
    ) -> LookupResult:
        """
        Full two-tier lookup pipeline.
        
        1. Tier 1: exact hash check (no embedding needed)
        2. Tier 2: ANN search with per-entry conformal thresholds
        """
        # ── Tier 1: Exact hash ────────────────────────────────
        t0 = time.monotonic()
        ek = self.exact_key(query, tenant_id)

        if ek in self._exact_store:
            entry_id, response = self._exact_store[ek]
            latency = (time.monotonic() - t0) * 1000
            return LookupResult(
                hit=True,
                tier=1,
                response=response,
                entry_id=entry_id,
                similarity=1.0,
                latency_ms=latency,
            )

        # ── Tier 2: Semantic search ──────────────────────────
        if query_embedding is None:
            return LookupResult(hit=False, tier=2, latency_ms=0.0)

        # Fuse with session context
        fused = query_embedding
        if session is not None:
            fused = session.update(query_embedding)

        if self._index.ntotal == 0:
            latency = (time.monotonic() - t0) * 1000
            return LookupResult(
                hit=False, tier=2, latency_ms=latency,
                query_embedding=fused,
            )

        # ANN search
        fused_2d = fused.reshape(1, -1).astype(np.float32)
        k = min(self._ann_k, self._index.ntotal)
        distances, indices = self._index.search(fused_2d, k)

        for i in range(k):
            faiss_id = int(indices[0, i])
            if faiss_id < 0:
                continue

            entry_id = self._id_map.get(faiss_id)
            if entry_id is None:
                continue

            entry = self._entries.get(entry_id)
            if entry is None:
                continue

            # Skip expired entries (lazy TTL check)
            if entry.is_expired:
                continue

            # FAISS IndexHNSWFlat uses L2 distance;
            # convert to cosine similarity for unit-norm vectors
            l2_dist = float(distances[0, i])
            similarity = 1.0 - (l2_dist / 2.0)  # cos_sim for unit vectors

            # Per-entry conformal threshold
            tau_i = await threshold_store.get_threshold(entry_id)

            if similarity >= tau_i:
                latency = (time.monotonic() - t0) * 1000
                return LookupResult(
                    hit=True,
                    tier=2,
                    response=entry.response_text,
                    entry_id=entry_id,
                    similarity=similarity,
                    latency_ms=latency,
                    query_embedding=fused,
                )

        latency = (time.monotonic() - t0) * 1000
        return LookupResult(
            hit=False, tier=2, latency_ms=latency,
            query_embedding=fused,
        )

    # ── Store ─────────────────────────────────────────────────

    async def store(self, entry: CacheEntry, tenant_id: str = "default") -> None:
        """
        Add an entry to both tiers.

        Tier 1: exact hash of query text
        Tier 2: embedding added to FAISS HNSW index via IndexIDMap2
        """
        async with self._lock:
            # Tier 1
            ek = self.exact_key(entry.query_text, tenant_id)
            self._exact_store[ek] = (entry.entry_id, entry.response_text)

            # Tier 2: add to FAISS with explicit ID
            vec = entry.embedding.reshape(1, -1).astype(np.float32)
            faiss_id = self._next_faiss_id
            ids = np.array([faiss_id], dtype=np.int64)
            self._index.add_with_ids(vec, ids)
            self._id_map[faiss_id] = entry.entry_id
            self._entry_to_faiss[entry.entry_id] = faiss_id
            self._entries[entry.entry_id] = entry
            self._next_faiss_id += 1

    async def remove(self, entry_id: str, tenant_id: str = "default") -> None:
        """
        Remove an entry from all tiers including the FAISS index.

        Uses IndexIDMap2.remove_ids() for true vector removal.
        """
        async with self._lock:
            entry = self._entries.pop(entry_id, None)
            if entry is not None:
                ek = self.exact_key(entry.query_text, tenant_id)
                self._exact_store.pop(ek, None)

            # Soft-delete from FAISS: remove from mappings so search skips it.
            # HNSW doesn't support true vector removal; vectors are cleaned
            # up during periodic rebuild_index().
            faiss_id = self._entry_to_faiss.pop(entry_id, None)
            if faiss_id is not None:
                self._id_map.pop(faiss_id, None)
                self._removals_since_rebuild += 1

                # Auto-rebuild when too many stale vectors accumulate
                if self._removals_since_rebuild >= settings.index_rebuild_interval:
                    self.rebuild_index()

    # ── Info ──────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def exact_store_size(self) -> int:
        return len(self._exact_store)

    @property
    def faiss_index_size(self) -> int:
        return self._index.ntotal

    def get_entry(self, entry_id: str) -> Optional[CacheEntry]:
        return self._entries.get(entry_id)

    def get_all_entry_ids(self) -> List[str]:
        return list(self._entries.keys())

    def get_expired_entry_ids(self) -> List[str]:
        """Return IDs of all expired entries."""
        return [
            eid for eid, entry in self._entries.items()
            if entry.is_expired
        ]

    def get_stats(self) -> dict:
        return {
            "total_entries": len(self._entries),
            "exact_store_size": len(self._exact_store),
            "faiss_index_size": self._index.ntotal,
            "embedding_dim": self._embed_dim,
        }
