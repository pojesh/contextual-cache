"""
FastAPI application entry point.

- Initializes all components via lifespan
- Configures CORS for the Tauri frontend
- Mounts API routes
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router, set_cache_manager, set_benchmark_runner, set_persistence_layer
from .benchmark.runner import BenchmarkRunner
from .cache_manager import ContextualCacheManager
from .config import settings
from .persistence import PersistenceLayer
from .rate_limiter import RateLimitMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

class EndpointFilter(logging.Filter):
    """Filter out spammy access logs for polling endpoints."""
    def filter(self, record: logging.LogRecord) -> bool:
        # record.args contains (client_addr, method, path, http_version, status_code)
        if record.args and len(record.args) >= 3:
            path = record.args[2]
            if path in ["/api/stats", "/health"] or (isinstance(path, str) and path.startswith("/api/benchmark/status")):
                return False
        return True

logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

# Suppress spammy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and teardown the cache manager."""
    logger.info("=" * 60)
    logger.info("  ContextualCache v1.0 — Starting …")
    logger.info("=" * 60)
    logger.info("  LLM backend:    %s (%s)", settings.llm_backend.value, settings.llm_model)
    logger.info("  Embedding:      %s (%dd)", settings.embedding_model, settings.embedding_dim)
    logger.info("  Cache capacity:  %s entries", f"{settings.cache_capacity:,}")
    logger.info("  Error rate (ε):  %s", settings.target_error_rate)
    logger.info("=" * 60)

    cm = ContextualCacheManager()
    set_cache_manager(cm)
    app.state.cache_manager = cm

    # Eagerly load the embedding model so first query isn't slow
    logger.info("  Loading embedding model …")
    await cm.embedding_service._ensure_model()
    logger.info("  ✓ Embedding model ready.")

    # Always initialize persistence for conversation history;
    # cache entry persistence is gated by persistence_enabled.
    persistence = PersistenceLayer()
    await persistence.initialize()
    set_persistence_layer(persistence)
    cm.set_persistence(persistence)

    if settings.persistence_enabled:
        entries = await persistence.load_all_entries()
        for entry in entries:
            await cm.lookup_engine.store(entry)
            cm.evictor.register(entry)
        logger.info("  ✓ Persistence: loaded %d entries.", len(entries))

        # Load bandit state
        bandit_state = await persistence.load_bandit_state(cm.bandit.shard_id)
        if bandit_state:
            cm.bandit.alpha = bandit_state["alpha"]
            cm.bandit.beta_params = bandit_state["beta"]
            cm.bandit.total_updates = bandit_state["total_updates"]
            cm.bandit.drift_resets = bandit_state["drift_resets"]
            logger.info("  ✓ Persistence: restored bandit state.")

    # Initialize benchmark runner (shares embedding service + LLM)
    br = BenchmarkRunner(
        embedding_service=cm.embedding_service,
        llm_provider=cm.llm_provider,
    )
    set_benchmark_runner(br)
    logger.info("  ✓ Benchmark runner ready.")

    # Background tasks
    _background_tasks: list[asyncio.Task] = []

    # TTL cleanup task
    async def _ttl_cleanup_loop():
        while True:
            await asyncio.sleep(settings.ttl_cleanup_interval_s)
            try:
                await cm.cleanup_expired()
            except Exception:
                logger.exception("TTL cleanup error")

    if settings.default_ttl_s > 0:
        _background_tasks.append(asyncio.create_task(_ttl_cleanup_loop()))
        logger.info("  ✓ TTL cleanup task started (every %.0fs).",
                     settings.ttl_cleanup_interval_s)

    # Periodic persistence flush (only for cache entries when enabled)
    async def _persistence_flush_loop():
        while True:
            await asyncio.sleep(30.0)
            try:
                entries = list(cm.lookup_engine._entries.values())
                if entries:
                    await persistence.flush_entries(entries)
                    await persistence.save_bandit_state(
                        cm.bandit.shard_id,
                        cm.bandit.alpha,
                        cm.bandit.beta_params,
                        cm.bandit.total_updates,
                        cm.bandit.drift_resets,
                    )
            except Exception:
                logger.exception("Persistence flush error")

    if settings.persistence_enabled:
        _background_tasks.append(asyncio.create_task(_persistence_flush_loop()))
        logger.info("  ✓ Persistence flush task started (every 30s).")

    logger.info("=" * 60)
    logger.info("  Server ready at http://%s:%d", settings.host, settings.port)
    logger.info("=" * 60)

    yield

    # Cancel background tasks
    for task in _background_tasks:
        task.cancel()
    for task in _background_tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Final persistence flush
    if settings.persistence_enabled:
        entries = list(cm.lookup_engine._entries.values())
        if entries:
            await persistence.flush_entries(entries)
        await persistence.save_bandit_state(
            cm.bandit.shard_id,
            cm.bandit.alpha,
            cm.bandit.beta_params,
            cm.bandit.total_updates,
            cm.bandit.drift_resets,
        )
        logger.info("Persistence: cache entries flushed.")
    await persistence.close()
    logger.info("Persistence closed.")

    logger.info("Shutting down ContextualCache…")
    await cm.close()


app = FastAPI(
    title="ContextualCache",
    description="A Fault-Tolerant Adaptive Semantic Cache for LLM Serving",
    version="1.0.0",
    lifespan=lifespan,
)

# Rate limiting middleware (applied before CORS)
app.add_middleware(
    RateLimitMiddleware,
    rate=settings.rate_limit_rps,
    burst=settings.rate_limit_burst,
)

# CORS for Tauri frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(router)


def main():
    """Run the server."""
    uvicorn.run(
        "contextual_cache.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
