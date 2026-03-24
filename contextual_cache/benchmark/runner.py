"""
Benchmark runner — orchestrates running all cache strategies on the dataset.

Runs each strategy sequentially on the same query list.
Stores results as JSON in benchmarks/results/.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..embedding_service import EmbeddingService
from ..llm_provider import LLMProvider
from .baselines import BaseCache, CacheResult, create_all_baselines
from .dataset import BenchmarkQuery, load_benchmark_dataset

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "benchmarks" / "results"


@dataclass
class BenchmarkProgress:
    """Live progress tracking for a benchmark run."""
    run_id: str
    status: str = "pending"  # pending | running | done | error
    total_strategies: int = 0
    completed_strategies: int = 0
    current_strategy: str = ""
    total_queries: int = 0
    completed_queries: int = 0
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0


@dataclass
class StrategyResult:
    """Results for a single strategy."""
    name: str
    total_queries: int = 0
    total_hits: int = 0
    total_misses: int = 0
    hit_rate: float = 0.0
    precision: float = 0.0
    f1_score: float = 0.0
    avg_hit_latency_ms: float = 0.0
    avg_miss_latency_ms: float = 0.0
    avg_overall_latency_ms: float = 0.0
    llm_calls: int = 0
    correct_hits: int = 0
    incorrect_hits: int = 0
    tier1_hits: int = 0
    tier2_hits: int = 0
    total_time_s: float = 0.0


class BenchmarkRunner:
    """
    Runs all cache strategies on the same NQ dataset.
    
    - Shared EmbeddingService (embeddings computed once, reused)
    - Shared LLMProvider (LLM calls only on first miss per unique question)
    - Deterministic ordering
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        llm_provider: LLMProvider,
        cache_manager_factory: Optional[Callable] = None,
    ):
        self.embedding_service = embedding_service
        self.llm_provider = llm_provider
        self.cache_manager_factory = cache_manager_factory
        self._active_runs: Dict[str, BenchmarkProgress] = {}
        self._results_cache: Dict[str, dict] = {}

    async def run_benchmark(
        self,
        num_questions: int = 500,
        capacity: int = 200,
        progress_callback: Optional[Callable[[BenchmarkProgress], None]] = None,
    ) -> dict:
        """
        Execute a full benchmark. Returns the results dict.
        """
        run_id = str(uuid.uuid4())[:8]
        progress = BenchmarkProgress(
            run_id=run_id,
            status="running",
            started_at=time.time(),
        )
        self._active_runs[run_id] = progress

        try:
            # Load dataset
            logger.info("[Benchmark %s] Loading dataset…", run_id)
            queries = load_benchmark_dataset(num_questions=num_questions)
            progress.total_queries = len(queries)

            # Pre-compute ALL embeddings once (shared across strategies)
            logger.info("[Benchmark %s] Pre-computing %d embeddings…", run_id, len(queries))
            embeddings: Dict[str, np.ndarray] = {}
            for i, q in enumerate(queries):
                embeddings[q.id] = await self.embedding_service.encode(q.question)

            # Pre-compute LLM responses for all unique questions (avoids redundant calls)
            logger.info("[Benchmark %s] Pre-generating LLM responses for unique questions…", run_id)
            llm_responses: Dict[str, str] = {}
            llm_latencies: Dict[str, float] = {}
            unique_answers = {}
            for q in queries:
                answer_key = q.original_id if q.is_paraphrase else q.id
                if answer_key not in unique_answers:
                    unique_answers[answer_key] = q.answer

            for i, q in enumerate(queries):
                answer_key = q.original_id if q.is_paraphrase else q.id
                if answer_key not in llm_responses:
                    try:
                        t0 = time.monotonic()
                        resp = await self.llm_provider.generate(q.question)
                        latency = (time.monotonic() - t0) * 1000
                        llm_responses[answer_key] = resp.text
                        llm_latencies[answer_key] = latency
                    except Exception as e:
                        logger.warning("LLM call failed for query %s: %s", q.id, e)
                        llm_responses[answer_key] = q.answer
                        llm_latencies[answer_key] = 500.0

                if i % 50 == 0:
                    logger.info("[Benchmark %s] Generated %d/%d LLM responses",
                                run_id, len(llm_responses), len(unique_answers))

            # Create baselines
            baselines = create_all_baselines(capacity)

            # Strategy 7: ContextualCache (our system)
            # We wrap our CacheManager to implement the same interface
            our_cache = _ContextualCacheWrapper(
                self.embedding_service, self.llm_provider, capacity
            )

            all_strategies: List[BaseCache] = baselines + [our_cache]
            progress.total_strategies = len(all_strategies)

            # Run each strategy
            strategy_results: List[StrategyResult] = []
            for si, strategy in enumerate(all_strategies):
                progress.current_strategy = strategy.name
                progress.completed_queries = 0
                logger.info("[Benchmark %s] Running strategy: %s", run_id, strategy.name)

                strategy.reset()
                t_start = time.monotonic()
                correct_hits = 0
                incorrect_hits = 0
                tier1_hits = 0
                tier2_hits = 0

                for qi, q in enumerate(queries):
                    answer_key = q.original_id if q.is_paraphrase else q.id
                    llm_resp = llm_responses.get(answer_key, q.answer)
                    llm_lat = llm_latencies.get(answer_key, 500.0)
                    emb = embeddings.get(q.id)

                    result = await strategy.query(
                        text=q.question,
                        embedding=emb,
                        response=llm_resp,
                        llm_latency_ms=llm_lat,
                    )

                    if result.hit:
                        # Check correctness: filtered word overlap (stopwords removed)
                        _stopwords = {
                            "the", "a", "an", "is", "are", "was", "were",
                            "in", "on", "at", "to", "of", "and", "or",
                            "for", "it", "its", "this", "that", "with",
                            "as", "by", "be", "has", "had", "have", "from",
                        }
                        expected_words = set(q.answer.lower().split()) - _stopwords
                        cached_words = set(result.response.lower().split()) - _stopwords
                        if expected_words:
                            overlap = len(expected_words & cached_words) / len(expected_words)
                            if overlap >= 0.4:
                                correct_hits += 1
                            else:
                                incorrect_hits += 1
                        else:
                            correct_hits += 1

                        if result.tier == 1:
                            tier1_hits += 1
                        elif result.tier == 2:
                            tier2_hits += 1

                    progress.completed_queries = qi + 1

                t_elapsed = time.monotonic() - t_start
                stats = strategy.stats()

                total_hits = stats["total_hits"]
                precision = correct_hits / max(total_hits, 1)
                recall = correct_hits / max(len(queries), 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-9)

                sr = StrategyResult(
                    name=strategy.name,
                    total_queries=stats["total_queries"],
                    total_hits=total_hits,
                    total_misses=stats["total_misses"],
                    hit_rate=stats["hit_rate"],
                    precision=round(precision, 4),
                    f1_score=round(f1, 4),
                    avg_hit_latency_ms=stats["avg_hit_latency_ms"],
                    avg_miss_latency_ms=stats["avg_miss_latency_ms"],
                    avg_overall_latency_ms=stats["avg_overall_latency_ms"],
                    llm_calls=stats["llm_calls"],
                    correct_hits=correct_hits,
                    incorrect_hits=incorrect_hits,
                    tier1_hits=tier1_hits,
                    tier2_hits=tier2_hits,
                    total_time_s=round(t_elapsed, 2),
                )
                strategy_results.append(sr)

                progress.completed_strategies = si + 1
                logger.info("[Benchmark %s] %s — hit_rate=%.2f%%, precision=%.2f%%",
                            run_id, strategy.name, sr.hit_rate * 100, sr.precision * 100)

            # Save results
            result_data = {
                "run_id": run_id,
                "timestamp": time.time(),
                "num_questions": len(queries),
                "num_paraphrases": sum(1 for q in queries if q.is_paraphrase),
                "cache_capacity": capacity,
                "strategies": [vars(sr) for sr in strategy_results],
            }

            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            result_file = RESULTS_DIR / f"benchmark_{run_id}.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2)
            logger.info("[Benchmark %s] Results saved to %s", run_id, result_file)

            progress.status = "done"
            progress.finished_at = time.time()
            self._results_cache[run_id] = result_data
            return result_data

        except Exception as e:
            logger.error("[Benchmark %s] Failed: %s", run_id, e, exc_info=True)
            progress.status = "error"
            progress.error = str(e)
            progress.finished_at = time.time()
            raise

    def get_progress(self, run_id: str) -> Optional[dict]:
        p = self._active_runs.get(run_id)
        if p is None:
            return None
        return {
            "run_id": p.run_id,
            "status": p.status,
            "total_strategies": p.total_strategies,
            "completed_strategies": p.completed_strategies,
            "current_strategy": p.current_strategy,
            "total_queries": p.total_queries,
            "completed_queries": p.completed_queries,
            "error": p.error,
            "elapsed_s": round(
                (p.finished_at or time.time()) - p.started_at, 1
            ) if p.started_at else 0,
        }

    def list_results(self) -> List[dict]:
        """List all saved benchmark results."""
        results = []
        if not RESULTS_DIR.exists():
            return results
        for f in sorted(RESULTS_DIR.glob("benchmark_*.json"), reverse=True):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    results.append({
                        "run_id": data["run_id"],
                        "timestamp": data["timestamp"],
                        "num_questions": data["num_questions"],
                        "cache_capacity": data["cache_capacity"],
                        "strategies": len(data["strategies"]),
                    })
            except Exception:
                continue
        return results

    def get_result(self, run_id: str) -> Optional[dict]:
        """Get a specific benchmark result."""
        if run_id in self._results_cache:
            return self._results_cache[run_id]
        result_file = RESULTS_DIR / f"benchmark_{run_id}.json"
        if result_file.exists():
            with open(result_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._results_cache[run_id] = data
            return data
        return None


# ═══════════════════════════════════════════════════════════════════
# ContextualCache wrapper for benchmark interface
# ═══════════════════════════════════════════════════════════════════

class _ContextualCacheWrapper(BaseCache):
    """
    Wraps our full ContextualCacheManager into the BaseCache interface
    for fair benchmarking comparison.
    """

    def __init__(self, embedding_service: EmbeddingService,
                 llm_provider: LLMProvider, capacity: int = 200):
        super().__init__("ContextualCache (Ours)", capacity)
        # Import here to avoid circular deps
        from ..admission_policy import SemanticWTinyLFUAdmission
        from ..bandit import ShardLocalBanditAdaptor
        from ..conformal_thresholds import ConformalThresholdStore
        from ..eviction import CostAwareEvictor
        from ..lookup_engine import TwoTierLookupEngine
        from ..models import CacheEntry

        self._lookup = TwoTierLookupEngine()
        self._threshold_store = ConformalThresholdStore()
        self._admission = SemanticWTinyLFUAdmission()
        self._evictor = CostAwareEvictor()
        self._bandit = ShardLocalBanditAdaptor()
        self._embedding_service = embedding_service
        self._capacity = capacity
        self._CacheEntry = CacheEntry

    async def query(self, text: str, embedding: Optional[np.ndarray],
                    response: str, llm_latency_ms: float) -> CacheResult:
        import uuid as _uuid
        t0 = time.monotonic()
        self.total_queries += 1

        # Tier 1: exact hash
        result = await self._lookup.lookup(
            query=text, query_embedding=None,
            session=None, threshold_store=self._threshold_store,
        )
        if result.hit:
            if result.entry_id:
                self._evictor.record_access(result.entry_id)
            ms = (time.monotonic() - t0) * 1000
            self.total_hits += 1
            self.hit_latencies.append(ms)
            return CacheResult(hit=True, response=result.response,
                               tier=1, latency_ms=ms, similarity=1.0)

        # Tier 2: semantic
        if embedding is not None:
            result = await self._lookup.lookup(
                query=text, query_embedding=embedding,
                session=None, threshold_store=self._threshold_store,
            )
            if result.hit:
                if result.entry_id:
                    self._evictor.record_access(result.entry_id)
                ms = (time.monotonic() - t0) * 1000
                self.total_hits += 1
                self.hit_latencies.append(ms)
                return CacheResult(hit=True, response=result.response,
                                   tier=2, latency_ms=ms, similarity=result.similarity)

        # Miss
        self.total_misses += 1
        ms = (time.monotonic() - t0) * 1000 + llm_latency_ms
        self.miss_latencies.append(ms)

        entry = self._CacheEntry(
            entry_id=str(_uuid.uuid4()),
            query_text=text,
            response_text=response,
            embedding=embedding,
            llm_cost_usd=llm_latency_ms * 0.00001,
            embed_cost_ms=5.0,
            output_tokens=len(response.split()),
        )

        if self._admission.should_admit(entry):
            while self._lookup.size >= self._capacity:
                evicted = self._evictor.evict_one()
                if evicted:
                    await self._lookup.remove(evicted, "default")
                else:
                    break
            await self._lookup.store(entry, "default")
            self._evictor.register(entry)

        return CacheResult(hit=False, response=response, tier=0, latency_ms=ms)

    def reset(self):
        from ..admission_policy import SemanticWTinyLFUAdmission
        from ..bandit import ShardLocalBanditAdaptor
        from ..conformal_thresholds import ConformalThresholdStore
        from ..eviction import CostAwareEvictor
        from ..lookup_engine import TwoTierLookupEngine

        self._lookup = TwoTierLookupEngine()
        self._threshold_store = ConformalThresholdStore()
        self._admission = SemanticWTinyLFUAdmission()
        self._evictor = CostAwareEvictor()
        self._bandit = ShardLocalBanditAdaptor()
        self.total_queries = self.total_hits = self.total_misses = 0
        self.hit_latencies.clear()
        self.miss_latencies.clear()
