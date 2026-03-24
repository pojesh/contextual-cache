"""
FastAPI route definitions.

All endpoints are async and use the global CacheManager instance.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Request / Response models ─────────────────────────────────


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000,
                       description="The user query text")
    session_id: Optional[str] = Field(None,
                                       description="Session ID for multi-turn context")
    tenant_id: str = Field("default",
                           description="Tenant namespace for isolation")


class QueryResponse(BaseModel):
    response: str
    hit: bool
    tier: int
    latency_ms: float
    entry_id: Optional[str] = None
    similarity: float = 0.0
    admitted: Optional[bool] = None
    llm_latency_ms: Optional[float] = None
    error: Optional[bool] = None
    fallback: Optional[bool] = None


class FeedbackRequest(BaseModel):
    entry_id: str = Field(..., description="ID of the cache entry")
    was_correct: bool = Field(..., description="Whether the cached response was correct")
    similarity: float = Field(0.0, ge=0.0, le=1.0)


class FeedbackResponse(BaseModel):
    status: str
    entry_id: str
    was_correct: bool
    new_threshold: float
    message: Optional[str] = None


class BulkQueryRequest(BaseModel):
    queries: list[str] = Field(..., min_length=1, max_length=100)
    session_id: Optional[str] = None
    tenant_id: str = "default"


# ── Endpoints ─────────────────────────────────────────────────

# The cache_manager is injected by the main app on startup
_cache_manager = None


def set_cache_manager(cm) -> None:
    global _cache_manager
    _cache_manager = cm


def _get_cm():
    if _cache_manager is None:
        raise HTTPException(503, "Cache manager not initialized")
    return _cache_manager


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """
    Main cache-fronted LLM query endpoint.
    
    - Checks Tier 1 (exact hash) and Tier 2 (semantic ANN)
    - Falls back to LLM on cache miss
    - Automatically admits new entries via W-TinyLFU
    """
    cm = _get_cm()
    result = await cm.query(
        query_text=req.query,
        session_id=req.session_id,
        tenant_id=req.tenant_id,
    )
    return QueryResponse(**result)


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback_endpoint(req: FeedbackRequest):
    """
    Submit correctness feedback for conformal threshold calibration.
    """
    cm = _get_cm()
    result = await cm.feedback(
        entry_id=req.entry_id,
        was_correct=req.was_correct,
        similarity=req.similarity,
    )
    return FeedbackResponse(**result)


@router.post("/bulk-query")
async def bulk_query_endpoint(req: BulkQueryRequest):
    """Send multiple queries at once. Useful for benchmarking."""
    cm = _get_cm()
    results = []
    for q in req.queries:
        r = await cm.query(
            query_text=q,
            session_id=req.session_id,
            tenant_id=req.tenant_id,
        )
        results.append(r)
    return {"results": results}


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    cm = _get_cm()
    return {
        "status": "healthy",
        "cache_size": cm.lookup_engine.size,
        "cache_capacity": cm._capacity,
    }


@router.get("/api/stats")
async def get_stats():
    """Full stats for the analytics dashboard."""
    cm = _get_cm()
    return cm.stats()


@router.get("/api/analytics")
async def get_analytics():
    """Detailed analytics data (time series, distributions)."""
    cm = _get_cm()
    return cm.metrics.get_full_analytics()


@router.post("/api/clear-cache")
async def clear_cache():
    """Clear all cache entries. Useful for testing."""
    cm = _get_cm()
    entry_ids = cm.lookup_engine.get_all_entry_ids()
    for eid in entry_ids:
        await cm.lookup_engine.remove(eid)
        cm.evictor.remove(eid)
    return {"status": "ok", "cleared": len(entry_ids)}


# ── Benchmark endpoints ───────────────────────────────────────

import asyncio
from ..benchmark.runner import BenchmarkRunner

_benchmark_runner: Optional[BenchmarkRunner] = None


def set_benchmark_runner(runner: BenchmarkRunner) -> None:
    global _benchmark_runner
    _benchmark_runner = runner


def _get_br():
    if _benchmark_runner is None:
        raise HTTPException(503, "Benchmark runner not initialized")
    return _benchmark_runner


class BenchmarkRunRequest(BaseModel):
    num_questions: int = Field(500, ge=10, le=2000,
                               description="Number of NQ questions to benchmark")
    capacity: int = Field(200, ge=10, le=5000,
                          description="Cache capacity for all strategies")


@router.post("/api/benchmark/run")
async def start_benchmark(req: BenchmarkRunRequest):
    """Start a benchmark run in the background."""
    br = _get_br()

    # Check if there's already a running benchmark
    for run_id, progress in br._active_runs.items():
        if progress.status == "running":
            return {"status": "already_running", "run_id": run_id}

    run_id = None

    async def _run():
        nonlocal run_id
        result = await br.run_benchmark(
            num_questions=req.num_questions,
            capacity=req.capacity,
        )
        return result

    # Start as background task
    task = asyncio.create_task(_run())

    # Brief wait to get the run_id from the runner
    await asyncio.sleep(0.2)
    for rid, p in br._active_runs.items():
        if p.status == "running":
            run_id = rid
            break

    return {"status": "started", "run_id": run_id}


@router.get("/api/benchmark/status/{run_id}")
async def benchmark_status(run_id: str):
    """Poll benchmark progress."""
    br = _get_br()
    progress = br.get_progress(run_id)
    if progress is None:
        raise HTTPException(404, f"Benchmark run {run_id} not found")
    return progress


@router.get("/api/benchmark/results")
async def list_benchmark_results():
    """List all saved benchmark results."""
    br = _get_br()
    return {"results": br.list_results()}


@router.get("/api/benchmark/results/{run_id}")
async def get_benchmark_result(run_id: str):
    """Get a specific benchmark result."""
    br = _get_br()
    result = br.get_result(run_id)
    if result is None:
        raise HTTPException(404, f"Benchmark result {run_id} not found")
    return result
