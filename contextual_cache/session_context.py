"""
Session-aware context accumulation via exponential moving average.

Each user session maintains a lightweight O(d) context vector that
is fused with the current query embedding to enable context-aware
cache hits in multi-turn conversations.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Optional

import numpy as np

from .config import settings

logger = logging.getLogger(__name__)


class SessionContextAccumulator:
    """
    Maintains an exponential moving average of turn embeddings.
    
    Space: O(d) per session (NOT O(T×d) like full attention history).
    
    Fusion: q_fused = α * embed(current_query) + (1-α) * context_vec
    α = 0.85 ensures current query dominates (avoids context hijacking).
    
    Session expiry: configurable inactivity timeout clears context.
    """

    __slots__ = ("alpha", "context_decay", "context_vec", "turn_count",
                 "last_active", "embed_dim", "session_id")

    def __init__(self, session_id: str, alpha: float = settings.context_alpha,
                 context_decay: float = settings.context_decay,
                 embed_dim: int = settings.embedding_dim) -> None:
        self.session_id = session_id
        self.alpha = alpha
        self.context_decay = context_decay
        self.context_vec: Optional[np.ndarray] = None
        self.turn_count: int = 0
        self.last_active: float = time.time()
        self.embed_dim = embed_dim

    def update(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Fuse query embedding with session context; return fused vector.
        
        First turn: returns raw embedding (no context to fuse with).
        Subsequent turns: α-weighted blend with EMA context.
        """
        self.last_active = time.time()
        self.turn_count += 1

        if self.context_vec is None or self.turn_count == 1:
            self.context_vec = query_embedding.copy()
            return query_embedding

        # EMA fusion
        fused = self.alpha * query_embedding + (1 - self.alpha) * self.context_vec
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm

        # Update context with configurable decay toward current query
        self.context_vec = self.context_decay * query_embedding + (1 - self.context_decay) * self.context_vec
        ctx_norm = np.linalg.norm(self.context_vec)
        if ctx_norm > 0:
            self.context_vec = self.context_vec / ctx_norm

        return fused

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.last_active) > settings.session_timeout_s

    def reset(self) -> None:
        self.context_vec = None
        self.turn_count = 0


class SessionManager:
    """
    Manages per-session context accumulators with automatic cleanup.
    
    Creates accumulators on first access; garbage-collects expired ones
    periodically (every `cleanup_interval_s` seconds).
    """

    def __init__(self, cleanup_interval_s: float = 300.0) -> None:
        self._sessions: Dict[str, SessionContextAccumulator] = {}
        self._cleanup_interval = cleanup_interval_s
        self._last_cleanup = time.time()
        self._lock = asyncio.Lock()

    async def get_or_create(self, session_id: str) -> SessionContextAccumulator:
        async with self._lock:
            self._maybe_cleanup()
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionContextAccumulator(session_id)
                logger.debug("Created session context: %s", session_id)
            else:
                acc = self._sessions[session_id]
                if acc.is_expired:
                    logger.debug("Session expired, resetting: %s", session_id)
                    acc.reset()
            return self._sessions[session_id]

    def _maybe_cleanup(self) -> None:
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        expired = [
            sid for sid, acc in self._sessions.items() if acc.is_expired
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.info("Cleaned up %d expired sessions.", len(expired))
        self._last_cleanup = now

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    def get_session_info(self) -> list[dict]:
        """Return info about active sessions for the dashboard."""
        return [
            {
                "session_id": sid,
                "turn_count": acc.turn_count,
                "last_active": acc.last_active,
                "is_expired": acc.is_expired,
            }
            for sid, acc in self._sessions.items()
        ]
