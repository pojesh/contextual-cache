"""
Cost-aware eviction using an extended Greedy-Dual-Size-Frequency (GDSF).

Eviction score H(e) = freq(e) × (llm_cost + embed_cost) / storage_bytes + L

Entries expensive to regenerate (long responses, slow LLMs) are favored
for retention.  One-hit wonders with cheap regeneration are evicted first.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from .models import CacheEntry

logger = logging.getLogger(__name__)


class CostAwareEvictor:
    """
    CostGDSF: extends GDSF with LLM inference cost as the "cost" signal.

    Priority H = freq × regen_cost / storage_bytes + L
    L (inflation factor) is updated on each eviction to ensure
    recently inserted entries aren't immediately evicted.

    Higher H → more valuable → keep.  Lowest H → evict first.
    """

    def __init__(self) -> None:
        self.L: float = 0.0  # inflation factor
        self._entries: Dict[str, _EvictorEntry] = {}
        self.total_evictions: int = 0

    def register(self, entry: CacheEntry) -> None:
        """Register an entry and compute its initial priority."""
        e = _EvictorEntry(
            entry_id=entry.entry_id,
            frequency=entry.frequency,
            llm_cost_usd=entry.llm_cost_usd,
            embed_cost_ms=entry.embed_cost_ms,
            storage_bytes=max(entry.storage_bytes, 1),
            last_access=entry.last_access,
        )
        e.H = self._compute_priority(e)
        self._entries[entry.entry_id] = e

    def record_access(self, entry_id: str) -> None:
        """Update frequency and re-compute priority on cache hit."""
        e = self._entries.get(entry_id)
        if e is None:
            return
        e.frequency += 1
        e.H = self._compute_priority(e)

    def evict_one(self) -> Optional[str]:
        """Evict the entry with the lowest priority score. Returns entry_id."""
        if not self._entries:
            return None

        victim_id = min(self._entries, key=lambda eid: self._entries[eid].H)
        victim = self._entries.pop(victim_id)
        self.L = victim.H  # update inflation factor
        self.total_evictions += 1
        logger.debug(
            "Evicted entry %s (H=%.4f, freq=%d)", victim_id, victim.H, victim.frequency
        )
        return victim_id

    def remove(self, entry_id: str) -> None:
        """Remove an entry from the evictor (e.g. on explicit invalidation)."""
        self._entries.pop(entry_id, None)

    def _compute_priority(self, e: _EvictorEntry) -> float:
        # Regeneration cost in normalized units
        regen_cost = (e.llm_cost_usd * 1e6) + e.embed_cost_ms
        return (e.frequency * regen_cost) / e.storage_bytes + self.L

    @property
    def size(self) -> int:
        return len(self._entries)

    def get_stats(self) -> dict:
        if not self._entries:
            return {
                "total_evictions": self.total_evictions,
                "tracked_entries": 0,
                "inflation_factor": self.L,
                "min_priority": 0.0,
                "max_priority": 0.0,
                "avg_priority": 0.0,
            }
        priorities = [e.H for e in self._entries.values()]
        return {
            "total_evictions": self.total_evictions,
            "tracked_entries": len(self._entries),
            "inflation_factor": round(self.L, 4),
            "min_priority": round(min(priorities), 4),
            "max_priority": round(max(priorities), 4),
            "avg_priority": round(sum(priorities) / len(priorities), 4),
        }

    def get_priority_distribution(self) -> list[float]:
        """Return all priority scores for dashboard histogram."""
        return [round(e.H, 4) for e in self._entries.values()]


class _EvictorEntry:
    """Internal eviction tracking entry."""
    __slots__ = ("entry_id", "frequency", "llm_cost_usd", "embed_cost_ms",
                 "storage_bytes", "last_access", "H")

    def __init__(self, entry_id: str, frequency: int, llm_cost_usd: float,
                 embed_cost_ms: float, storage_bytes: int,
                 last_access: float) -> None:
        self.entry_id = entry_id
        self.frequency = frequency
        self.llm_cost_usd = llm_cost_usd
        self.embed_cost_ms = embed_cost_ms
        self.storage_bytes = storage_bytes
        self.last_access = last_access
        self.H: float = 0.0
