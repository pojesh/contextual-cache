"""
Embedding service with local sentence-transformers backend
and an in-memory embedding cache to avoid redundant computation.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from typing import Optional

import numpy as np

from .config import settings
from .utils import normalize_text

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Async-compatible embedding service.

    - Wraps the synchronous sentence-transformers encode() in asyncio.to_thread()
    - Maintains an LRU embedding cache keyed by SHA-256 of normalized text
    - Avoids importing the heavy model until first use (lazy init)
    """

    def __init__(
        self,
        model_name: str = settings.embedding_model,
        embed_dim: int = settings.embedding_dim,
        cache_size: int = settings.embedding_cache_size,
    ) -> None:
        self._model_name = model_name
        self._embed_dim = embed_dim
        self._model = None          # lazy load
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        self._lock = asyncio.Lock()
        self._load_lock = asyncio.Lock()

        # Stats
        self.encode_calls = 0
        self.cache_hits = 0
        self.total_encode_ms = 0.0

    # ── Lazy model loading ────────────────────────────────────

    async def _ensure_model(self) -> None:
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:
                return
            logger.info("Loading embedding model '%s' …", self._model_name)
            self._model = await asyncio.to_thread(self._load_model)
            logger.info("Embedding model loaded (dim=%d).", self._embed_dim)

    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self._model_name)
        return model

    # ── Public API ────────────────────────────────────────────

    async def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text string into a unit-norm embedding vector.

        Returns a cached result if available; otherwise delegates to the
        model in a thread pool.
        """
        await self._ensure_model()

        cache_key = self._cache_key(text)

        # Check embedding cache
        async with self._lock:
            if cache_key in self._cache:
                self.cache_hits += 1
                self._cache.move_to_end(cache_key)
                return self._cache[cache_key].copy()

        # Compute embedding in thread pool (non-blocking)
        self.encode_calls += 1
        t0 = time.monotonic()
        vec = await asyncio.to_thread(self._encode_sync, text)
        elapsed_ms = (time.monotonic() - t0) * 1000
        self.total_encode_ms += elapsed_ms

        # L2-normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        # Store in cache
        async with self._lock:
            self._cache[cache_key] = vec.copy()
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

        return vec

    async def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Encode multiple texts; uses cache for known texts."""
        results: list[Optional[np.ndarray]] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        async with self._lock:
            for i, text in enumerate(texts):
                ck = self._cache_key(text)
                if ck in self._cache:
                    self.cache_hits += 1
                    self._cache.move_to_end(ck)
                    results[i] = self._cache[ck].copy()
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)

        if uncached_texts:
            await self._ensure_model()
            self.encode_calls += len(uncached_texts)
            t0 = time.monotonic()
            vecs = await asyncio.to_thread(self._encode_batch_sync, uncached_texts)
            self.total_encode_ms += (time.monotonic() - t0) * 1000

            async with self._lock:
                for idx, vec, text in zip(uncached_indices, vecs, uncached_texts):
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    results[idx] = vec
                    ck = self._cache_key(text)
                    self._cache[ck] = vec.copy()
                    if len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)

        return results  # type: ignore[return-value]

    @property
    def avg_encode_ms(self) -> float:
        return self.total_encode_ms / max(self.encode_calls, 1)

    # ── Private helpers ───────────────────────────────────────

    def _encode_sync(self, text: str) -> np.ndarray:
        return self._model.encode(text, normalize_embeddings=False)

    def _encode_batch_sync(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=False,
                                  batch_size=32)

    @staticmethod
    def _cache_key(text: str) -> str:
        """SHA-256 of canonically normalized text (consistent with lookup engine)."""
        return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
