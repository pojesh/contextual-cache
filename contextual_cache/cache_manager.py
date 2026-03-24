"""
Central orchestrator — ContextualCacheManager.

Coordinates: embedding service → session context → two-tier lookup →
admission policy → eviction → conformal threshold updates → bandit
adaptation → metrics collection.

This is the single entry point for all cache operations.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

import numpy as np

from .admission_policy import SemanticWTinyLFUAdmission
from .bandit import ShardLocalBanditAdaptor
from .circuit_breaker import CircuitBreaker, CircuitBreakerError
from .config import settings
from .conformal_thresholds import ConformalThresholdStore
from .embedding_service import EmbeddingService
from .eviction import CostAwareEvictor
from .llm_provider import LLMProvider, LLMResponse
from .lookup_engine import TwoTierLookupEngine
from .metrics import MetricsCollector
from .models import CacheEntry, LookupResult, QueryMetrics
from .session_context import SessionManager

logger = logging.getLogger(__name__)


class ContextualCacheManager:
    """
    Single entry point for all cache operations.
    
    query()    → lookup + optional LLM fallback + store
    feedback() → conformal threshold update
    stats()    → aggregate metrics for dashboard
    """

    def __init__(self) -> None:
        # Core components
        self.embedding_service = EmbeddingService()
        self.session_manager = SessionManager()
        self.threshold_store = ConformalThresholdStore()
        self.admission_policy = SemanticWTinyLFUAdmission()
        self.evictor = CostAwareEvictor()
        self.lookup_engine = TwoTierLookupEngine()
        self.bandit = ShardLocalBanditAdaptor()
        self.llm_provider = LLMProvider()
        self.metrics = MetricsCollector()

        # Circuit breaker for embedding service
        self._embed_circuit = CircuitBreaker(name="embedding")

        self._capacity = settings.cache_capacity

    async def query(
        self,
        query_text: str,
        session_id: Optional[str] = None,
        tenant_id: str = "default",
    ) -> dict:
        """
        Main query endpoint.
        
        1. Tier 1 exact-hash check (no embedding needed)
        2. Tier 2 semantic search (embedding required)
        3. On miss: call LLM → admit to cache if policy allows
        
        Returns dict with response, hit/miss info, latency.
        """
        overall_t0 = time.monotonic()
        session = None
        if session_id:
            session = await self.session_manager.get_or_create(session_id)

        # ── Tier 1: exact hash (no embedding needed) ──────────
        result = await self.lookup_engine.lookup(
            query=query_text,
            query_embedding=None,
            session=None,
            threshold_store=self.threshold_store,
            tenant_id=tenant_id,
        )

        if result.hit:
            self._on_hit(result)
            overall_ms = (time.monotonic() - overall_t0) * 1000
            self._record_metrics(result, overall_ms)
            return self._format_response(result, overall_ms)

        # ── Compute embedding (needed for Tier 2) ─────────────
        embed_t0 = time.monotonic()
        try:
            query_embedding = await self._embed_circuit.call(
                self.embedding_service.encode, query_text
            )
        except CircuitBreakerError:
            logger.warning("Embedding circuit open — falling back to LLM only.")
            return await self._llm_fallback(query_text, overall_t0)
        embed_ms = (time.monotonic() - embed_t0) * 1000

        # ── Tier 2: semantic search ───────────────────────────
        result = await self.lookup_engine.lookup(
            query=query_text,
            query_embedding=query_embedding,
            session=session,
            threshold_store=self.threshold_store,
            tenant_id=tenant_id,
        )

        if result.hit:
            self._on_hit(result)
            overall_ms = (time.monotonic() - overall_t0) * 1000
            self._record_metrics(result, overall_ms)
            return self._format_response(result, overall_ms)

        # ── Cache miss: call LLM ──────────────────────────────
        try:
            llm_resp = await self.llm_provider.generate(query_text)
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            overall_ms = (time.monotonic() - overall_t0) * 1000
            miss_result = LookupResult(hit=False, tier=2, latency_ms=overall_ms)
            self._record_metrics(miss_result, overall_ms)
            return {
                "response": f"Error: LLM unavailable — {e}",
                "hit": False,
                "tier": 0,
                "latency_ms": round(overall_ms, 2),
                "error": True,
            }

        # ── Admission + Store ─────────────────────────────────
        fused_embedding = result.query_embedding if result.query_embedding is not None else query_embedding
        expires_at = None
        if settings.default_ttl_s > 0:
            expires_at = time.time() + settings.default_ttl_s

        entry = CacheEntry(
            entry_id=str(uuid.uuid4()),
            query_text=query_text,
            response_text=llm_resp.text,
            embedding=fused_embedding,
            session_id=session_id,
            llm_cost_usd=llm_resp.cost_usd,
            embed_cost_ms=embed_ms,
            output_tokens=llm_resp.output_tokens,
            expires_at=expires_at,
        )

        admitted = self.admission_policy.should_admit(entry)

        if admitted:
            # Evict if at capacity
            eviction_triggered = False
            while self.lookup_engine.size >= self._capacity:
                evicted_id = self.evictor.evict_one()
                if evicted_id:
                    await self.lookup_engine.remove(evicted_id, tenant_id)
                    self.admission_policy.on_evict(evicted_id)
                    eviction_triggered = True
                else:
                    break

            await self.lookup_engine.store(entry, tenant_id)
            self.evictor.register(entry)
            self.metrics.record_admission()

        overall_ms = (time.monotonic() - overall_t0) * 1000
        miss_result = LookupResult(hit=False, tier=2, latency_ms=overall_ms)
        self._record_metrics(
            miss_result, overall_ms,
            was_admission_rejected=not admitted,
            was_eviction_triggered=eviction_triggered if admitted else False,
        )

        return {
            "response": llm_resp.text,
            "hit": False,
            "tier": 0,
            "latency_ms": round(overall_ms, 2),
            "entry_id": entry.entry_id if admitted else None,
            "admitted": admitted,
            "llm_latency_ms": round(llm_resp.latency_ms, 2),
        }

    async def feedback(self, entry_id: str, was_correct: bool,
                       similarity: float = 0.0) -> dict:
        """
        User feedback for conformal threshold calibration.
        
        Also updates the bandit reward signal.
        """
        entry = self.lookup_engine.get_entry(entry_id)
        if entry is None:
            return {"status": "error", "message": "Entry not found."}

        await self.threshold_store.update(entry_id, similarity, was_correct)
        self.metrics.record_feedback(was_correct)

        # Bandit update
        reward = 1.0 if was_correct else 0.0
        arm, _ = self.bandit.get_current_best()
        self.bandit.update(arm, reward)

        return {
            "status": "ok",
            "entry_id": entry_id,
            "was_correct": was_correct,
            "new_threshold": await self.threshold_store.get_threshold(entry_id),
        }

    def stats(self) -> dict:
        """Return comprehensive stats for the dashboard."""
        self.metrics.update_cache_size(
            self.lookup_engine.size, self._capacity
        )
        return {
            "aggregate": self.metrics.get_aggregate(),
            "time_series": self.metrics.get_time_series(200),
            "latency_distribution": self.metrics.get_latency_distribution(),
            "threshold_distribution": self.metrics.get_threshold_distribution(),
            "similarity_distribution": self.metrics.get_similarity_distribution(),
            "eviction": self.evictor.get_stats(),
            "admission": self.admission_policy.get_stats(),
            "bandit": self.bandit.get_stats(),
            "thresholds": self.threshold_store.get_stats(),
            "lookup_engine": self.lookup_engine.get_stats(),
            "embedding_service": {
                "encode_calls": self.embedding_service.encode_calls,
                "cache_hits": self.embedding_service.cache_hits,
                "avg_encode_ms": round(self.embedding_service.avg_encode_ms, 2),
            },
            "llm": self.llm_provider.get_stats(),
            "sessions": {
                "active_count": self.session_manager.active_count,
                "sessions": self.session_manager.get_session_info()[:20],
            },
        }

    async def cleanup_expired(self, tenant_id: str = "default") -> int:
        """Remove expired entries. Returns the number removed."""
        expired_ids = self.lookup_engine.get_expired_entry_ids()
        for eid in expired_ids:
            await self.lookup_engine.remove(eid, tenant_id)
            self.evictor.remove(eid)
            self.admission_policy.on_evict(eid)
        if expired_ids:
            logger.info("Cleaned up %d expired cache entries.", len(expired_ids))
        return len(expired_ids)

    async def close(self) -> None:
        """Cleanup resources."""
        await self.llm_provider.close()

    # ── Internal helpers ──────────────────────────────────────

    def _on_hit(self, result: LookupResult) -> None:
        """Update eviction stats and admission frequency on cache hit."""
        if result.entry_id:
            self.evictor.record_access(result.entry_id)
            entry = self.lookup_engine.get_entry(result.entry_id)
            if entry:
                entry.frequency += 1
                entry.last_access = time.time()
                self.admission_policy.on_access(result.entry_id, entry.embedding)

    def _record_metrics(
        self,
        result: LookupResult,
        overall_ms: float,
        was_admission_rejected: bool = False,
        was_eviction_triggered: bool = False,
    ) -> None:
        qm = QueryMetrics(
            timestamp=time.time(),
            hit=result.hit,
            tier=result.tier,
            latency_ms=overall_ms,
            similarity=result.similarity,
            was_admission_rejected=was_admission_rejected,
            was_eviction_triggered=was_eviction_triggered,
            threshold_used=0.0,
        )
        self.metrics.record_query(qm)

    async def _llm_fallback(self, query: str, t0: float) -> dict:
        """Direct LLM call when embedding service is unavailable."""
        try:
            llm_resp = await self.llm_provider.generate(query)
            ms = (time.monotonic() - t0) * 1000
            return {
                "response": llm_resp.text,
                "hit": False,
                "tier": 0,
                "latency_ms": round(ms, 2),
                "fallback": True,
            }
        except Exception as e:
            ms = (time.monotonic() - t0) * 1000
            return {
                "response": f"Error: Both embedding and LLM services unavailable — {e}",
                "hit": False,
                "tier": 0,
                "latency_ms": round(ms, 2),
                "error": True,
            }

    def _format_response(self, result: LookupResult, overall_ms: float) -> dict:
        return {
            "response": result.response,
            "hit": True,
            "tier": result.tier,
            "latency_ms": round(overall_ms, 2),
            "entry_id": result.entry_id,
            "similarity": round(result.similarity, 4),
        }
