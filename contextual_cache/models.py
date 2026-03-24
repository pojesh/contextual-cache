"""
Domain models – immutable data carriers used across every module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── Lookup ────────────────────────────────────────────────────


@dataclass(slots=True)
class LookupResult:
    """Returned by TwoTierLookupEngine.lookup()."""

    hit: bool
    tier: int = 0                       # 1 = exact hash, 2 = semantic ANN
    response: Optional[str] = None
    entry_id: Optional[str] = None
    similarity: float = 0.0
    latency_ms: float = 0.0
    query_embedding: Optional[np.ndarray] = None


# ── Cache Entry ───────────────────────────────────────────────


@dataclass
class CacheEntry:
    """A single item stored in the semantic cache."""

    entry_id: str
    query_text: str
    response_text: str
    embedding: np.ndarray
    session_id: Optional[str] = None

    # Cost signals
    llm_cost_usd: float = 0.0
    embed_cost_ms: float = 0.0
    output_tokens: int = 0
    storage_bytes: int = 0

    # Access tracking
    frequency: int = 1
    last_access: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None  # None = no expiry

    # Eviction score (set by CostAwareEvictor)
    priority_score: float = 0.0

    @property
    def is_expired(self) -> bool:
        return self.expires_at is not None and time.time() > self.expires_at

    def __post_init__(self) -> None:
        if self.storage_bytes == 0:
            self.storage_bytes = len(self.response_text.encode("utf-8"))


# ── Eviction ──────────────────────────────────────────────────


@dataclass(slots=True)
class CostEntry:
    """Lightweight view used by the evictor."""

    entry_id: str
    frequency: int
    llm_cost_usd: float
    embed_cost_ms: float
    storage_bytes: int
    last_access: float
    H: float = 0.0  # priority


# ── Gossip Metadata ──────────────────────────────────────────


@dataclass(slots=True)
class EntryMetadata:
    """What travels over the gossip protocol – NO embedding vectors."""

    entry_id: str
    key_hash: int
    access_count: int
    eviction_score: float
    last_access: float


# ── Metrics ───────────────────────────────────────────────────


@dataclass
class QueryMetrics:
    """Per-query metrics snapshot."""

    timestamp: float = field(default_factory=time.time)
    hit: bool = False
    tier: int = 0
    latency_ms: float = 0.0
    similarity: float = 0.0
    was_admission_rejected: bool = False
    was_eviction_triggered: bool = False
    session_id: Optional[str] = None
    threshold_used: float = 0.0


@dataclass
class AggregateMetrics:
    """Rolling aggregate metrics for the dashboard."""

    total_queries: int = 0
    total_hits: int = 0
    tier1_hits: int = 0
    tier2_hits: int = 0
    total_misses: int = 0
    total_evictions: int = 0
    total_admissions: int = 0
    admission_rejections: int = 0
    avg_latency_ms: float = 0.0
    avg_hit_latency_ms: float = 0.0
    avg_miss_latency_ms: float = 0.0
    cache_size: int = 0
    cache_capacity: int = 0

    # Correctness
    correct_hits: int = 0
    incorrect_hits: int = 0

    @property
    def hit_rate(self) -> float:
        return self.total_hits / max(self.total_queries, 1)

    @property
    def tier1_rate(self) -> float:
        return self.tier1_hits / max(self.total_hits, 1)

    @property
    def tier2_rate(self) -> float:
        return self.tier2_hits / max(self.total_hits, 1)

    @property
    def false_hit_rate(self) -> float:
        return self.incorrect_hits / max(self.total_queries, 1)

    @property
    def precision(self) -> float:
        tp = self.correct_hits
        fp = self.incorrect_hits
        return tp / max(tp + fp, 1)

    def to_dict(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "total_hits": self.total_hits,
            "tier1_hits": self.tier1_hits,
            "tier2_hits": self.tier2_hits,
            "total_misses": self.total_misses,
            "total_evictions": self.total_evictions,
            "total_admissions": self.total_admissions,
            "admission_rejections": self.admission_rejections,
            "hit_rate": round(self.hit_rate, 4),
            "tier1_rate": round(self.tier1_rate, 4),
            "tier2_rate": round(self.tier2_rate, 4),
            "false_hit_rate": round(self.false_hit_rate, 4),
            "precision": round(self.precision, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "avg_hit_latency_ms": round(self.avg_hit_latency_ms, 2),
            "avg_miss_latency_ms": round(self.avg_miss_latency_ms, 2),
            "cache_size": self.cache_size,
            "cache_capacity": self.cache_capacity,
            "correct_hits": self.correct_hits,
            "incorrect_hits": self.incorrect_hits,
        }
