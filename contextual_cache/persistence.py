"""
SQLite-backed persistence layer for crash recovery.

Stores cache entries, conformal calibration scores, and bandit state
so the cache survives restarts. Uses aiosqlite for async I/O.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import settings
from .models import CacheEntry

logger = logging.getLogger(__name__)

# aiosqlite is optional — fall back to sync sqlite3 if unavailable
try:
    import aiosqlite
    _HAS_AIOSQLITE = True
except ImportError:
    _HAS_AIOSQLITE = False


class PersistenceLayer:
    """
    Async SQLite persistence for cache state.

    Tables:
      cache_entries  — full cache entries with serialized embeddings
      conformal_scores — per-entry calibration score windows
      bandit_state — Thompson Sampling posterior parameters
    """

    def __init__(self, db_path: str = settings.persistence_path) -> None:
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    async def initialize(self) -> None:
        """Create tables if they don't exist."""
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                entry_id TEXT PRIMARY KEY,
                query_text TEXT NOT NULL,
                response_text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                session_id TEXT,
                llm_cost_usd REAL DEFAULT 0.0,
                embed_cost_ms REAL DEFAULT 0.0,
                output_tokens INTEGER DEFAULT 0,
                storage_bytes INTEGER DEFAULT 0,
                frequency INTEGER DEFAULT 1,
                last_access REAL,
                created_at REAL,
                expires_at REAL,
                priority_score REAL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS conformal_scores (
                entry_id TEXT PRIMARY KEY,
                scores BLOB NOT NULL
            );

            CREATE TABLE IF NOT EXISTS bandit_state (
                shard_id TEXT PRIMARY KEY,
                alpha BLOB NOT NULL,
                beta BLOB NOT NULL,
                total_updates INTEGER DEFAULT 0,
                drift_resets INTEGER DEFAULT 0
            );
        """)
        self._conn.commit()
        logger.info("Persistence layer initialized at %s", self._db_path)

    async def save_entry(self, entry: CacheEntry) -> None:
        """Persist a single cache entry."""
        if self._conn is None:
            return
        self._conn.execute(
            """INSERT OR REPLACE INTO cache_entries
               (entry_id, query_text, response_text, embedding, session_id,
                llm_cost_usd, embed_cost_ms, output_tokens, storage_bytes,
                frequency, last_access, created_at, expires_at, priority_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.entry_id,
                entry.query_text,
                entry.response_text,
                entry.embedding.tobytes(),
                entry.session_id,
                entry.llm_cost_usd,
                entry.embed_cost_ms,
                entry.output_tokens,
                entry.storage_bytes,
                entry.frequency,
                entry.last_access,
                entry.created_at,
                entry.expires_at,
                entry.priority_score,
            ),
        )
        self._conn.commit()

    async def load_all_entries(self, embed_dim: int = settings.embedding_dim
                               ) -> List[CacheEntry]:
        """Load all non-expired cache entries."""
        if self._conn is None:
            return []
        now = time.time()
        rows = self._conn.execute(
            """SELECT entry_id, query_text, response_text, embedding,
                      session_id, llm_cost_usd, embed_cost_ms, output_tokens,
                      storage_bytes, frequency, last_access, created_at,
                      expires_at, priority_score
               FROM cache_entries
               WHERE expires_at IS NULL OR expires_at > ?""",
            (now,),
        ).fetchall()

        entries: List[CacheEntry] = []
        for row in rows:
            embedding = np.frombuffer(row[3], dtype=np.float32).reshape(embed_dim)
            entry = CacheEntry(
                entry_id=row[0],
                query_text=row[1],
                response_text=row[2],
                embedding=embedding,
                session_id=row[4],
                llm_cost_usd=row[5],
                embed_cost_ms=row[6],
                output_tokens=row[7],
                storage_bytes=row[8],
                frequency=row[9],
                last_access=row[10],
                created_at=row[11],
                expires_at=row[12],
                priority_score=row[13],
            )
            entries.append(entry)

        logger.info("Loaded %d cache entries from persistence.", len(entries))
        return entries

    async def delete_entry(self, entry_id: str) -> None:
        """Remove an entry from persistence."""
        if self._conn is None:
            return
        self._conn.execute("DELETE FROM cache_entries WHERE entry_id = ?",
                           (entry_id,))
        self._conn.execute("DELETE FROM conformal_scores WHERE entry_id = ?",
                           (entry_id,))
        self._conn.commit()

    async def save_conformal_scores(self, entry_id: str,
                                     scores: List[float]) -> None:
        """Persist calibration scores for an entry."""
        if self._conn is None:
            return
        blob = np.array(scores, dtype=np.float64).tobytes()
        self._conn.execute(
            "INSERT OR REPLACE INTO conformal_scores (entry_id, scores) VALUES (?, ?)",
            (entry_id, blob),
        )
        self._conn.commit()

    async def load_conformal_scores(self) -> Dict[str, List[float]]:
        """Load all calibration scores."""
        if self._conn is None:
            return {}
        rows = self._conn.execute(
            "SELECT entry_id, scores FROM conformal_scores"
        ).fetchall()
        result: Dict[str, List[float]] = {}
        for entry_id, blob in rows:
            scores = np.frombuffer(blob, dtype=np.float64).tolist()
            result[entry_id] = scores
        return result

    async def save_bandit_state(self, shard_id: str, alpha: np.ndarray,
                                 beta: np.ndarray, total_updates: int = 0,
                                 drift_resets: int = 0) -> None:
        """Persist bandit posterior parameters."""
        if self._conn is None:
            return
        self._conn.execute(
            """INSERT OR REPLACE INTO bandit_state
               (shard_id, alpha, beta, total_updates, drift_resets)
               VALUES (?, ?, ?, ?, ?)""",
            (shard_id, alpha.tobytes(), beta.tobytes(),
             total_updates, drift_resets),
        )
        self._conn.commit()

    async def load_bandit_state(self, shard_id: str
                                 ) -> Optional[Dict]:
        """Load bandit state for a shard."""
        if self._conn is None:
            return None
        row = self._conn.execute(
            "SELECT alpha, beta, total_updates, drift_resets FROM bandit_state WHERE shard_id = ?",
            (shard_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "alpha": np.frombuffer(row[0], dtype=np.float64),
            "beta": np.frombuffer(row[1], dtype=np.float64),
            "total_updates": row[2],
            "drift_resets": row[3],
        }

    async def flush_entries(self, entries: List[CacheEntry]) -> None:
        """Batch-persist multiple entries."""
        if self._conn is None:
            return
        data = [
            (
                e.entry_id, e.query_text, e.response_text,
                e.embedding.tobytes(), e.session_id,
                e.llm_cost_usd, e.embed_cost_ms, e.output_tokens,
                e.storage_bytes, e.frequency, e.last_access,
                e.created_at, e.expires_at, e.priority_score,
            )
            for e in entries
        ]
        self._conn.executemany(
            """INSERT OR REPLACE INTO cache_entries
               (entry_id, query_text, response_text, embedding, session_id,
                llm_cost_usd, embed_cost_ms, output_tokens, storage_bytes,
                frequency, last_access, created_at, expires_at, priority_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            data,
        )
        self._conn.commit()
        logger.debug("Flushed %d entries to persistence.", len(data))

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
