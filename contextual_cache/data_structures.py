"""
Probabilistic data structures for the admission policy.

- CountMinSketch: space-efficient frequency estimator
- RandomProjectionLSH: locality-sensitive hashing for semantic bucketing
- OrderedLRUCache / SLRUCache: deterministic eviction-ordered caches
"""

from __future__ import annotations

import hashlib
import struct
from collections import OrderedDict
from typing import Any, Optional, Tuple

import numpy as np


# ── Count-Min Sketch ──────────────────────────────────────────


class CountMinSketch:
    """
    A conservative-update Count-Min Sketch for frequency estimation.

    Width × Depth matrix of counters.  Each item is hashed by `depth`
    independent hash functions; its estimated count is the *minimum*
    across all rows (conservative estimate reduces over-counting).

    Periodic halving prevents stale counts from dominating.
    """

    __slots__ = ("_width", "_depth", "_table", "_total", "_halve_every", "_ops")

    def __init__(self, width: int = 4096, depth: int = 4,
                 halve_every: int = 100_000) -> None:
        self._width = width
        self._depth = depth
        self._table = np.zeros((depth, width), dtype=np.int32)
        self._total = 0
        self._halve_every = halve_every
        self._ops = 0

    # ── Hash helpers ──────────────────────────────────────────

    def _hashes(self, key: bytes) -> list[int]:
        """Return `depth` independent hash indices via double-hashing."""
        h = hashlib.md5(key).digest()
        h1 = struct.unpack_from("<I", h, 0)[0]
        h2 = struct.unpack_from("<I", h, 4)[0]
        return [(h1 + i * h2) % self._width for i in range(self._depth)]

    @staticmethod
    def _to_bytes(key: Any) -> bytes:
        if isinstance(key, bytes):
            return key
        if isinstance(key, str):
            return key.encode("utf-8")
        if isinstance(key, (int, np.integer)):
            return struct.pack("<q", int(key))
        if isinstance(key, tuple):
            return b"|".join(
                struct.pack("<q", int(k)) if isinstance(k, (int, np.integer))
                else str(k).encode() for k in key
            )
        return str(key).encode("utf-8")

    # ── Public API ────────────────────────────────────────────

    def increment(self, key: Any, count: int = 1) -> int:
        """Increment the count for *key* and return the new estimate."""
        kb = self._to_bytes(key)
        indices = self._hashes(kb)
        min_val = min(self._table[row, col] for row, col in enumerate(indices))
        # Conservative update: only increment rows that equal current min
        for row, col in enumerate(indices):
            if self._table[row, col] == min_val:
                self._table[row, col] += count
        self._total += count
        self._ops += 1
        if self._ops >= self._halve_every:
            self._halve()
        return min_val + count

    def estimate(self, key: Any) -> int:
        """Return the estimated count for *key*."""
        kb = self._to_bytes(key)
        indices = self._hashes(kb)
        return int(min(self._table[row, col] for row, col in enumerate(indices)))

    def _halve(self) -> None:
        """Aging: halve all counters to prevent staleness."""
        self._table >>= 1
        self._total >>= 1
        self._ops = 0

    def reset(self) -> None:
        self._table[:] = 0
        self._total = 0
        self._ops = 0


# ── Random Projection LSH ────────────────────────────────────


class RandomProjectionLSH:
    """
    Locality-Sensitive Hashing via random hyperplane projections.

    Maps a d-dimensional vector to an n_bits-wide binary hash.
    Multiple tables increase recall (union of hash collisions).

    Used to bucket semantically similar embeddings for the
    Count-Min Sketch frequency estimation.
    """

    __slots__ = ("_planes", "_n_bits", "_n_tables", "_input_dim")

    def __init__(self, input_dim: int = 384, n_bits: int = 8,
                 n_tables: int = 4, seed: int = 42) -> None:
        self._input_dim = input_dim
        self._n_bits = n_bits
        self._n_tables = n_tables
        rng = np.random.RandomState(seed)
        # Each table has n_bits random hyperplanes of dimension input_dim
        self._planes = rng.randn(n_tables, n_bits, input_dim).astype(np.float32)

    def hash(self, vector: np.ndarray) -> Tuple[int, ...]:
        """
        Return a tuple of `n_tables` integer hashes for `vector`.
        
        Each hash is an n_bits-wide integer from the sign of
        dot products with the random planes.
        """
        v = vector.astype(np.float32).ravel()
        hashes: list[int] = []
        for table_idx in range(self._n_tables):
            projections = self._planes[table_idx] @ v  # (n_bits,)
            bits = (projections > 0).astype(np.uint8)
            h = 0
            for bit in bits:
                h = (h << 1) | int(bit)
            hashes.append(h)
        return tuple(hashes)

    def hash_single(self, vector: np.ndarray) -> int:
        """Return a single combined hash via XOR of all tables."""
        hashes = self.hash(vector)
        combined = 0
        for h in hashes:
            combined ^= h
        return combined


# ── LRU Cache ─────────────────────────────────────────────────


class LRUCache:
    """
    Simple ordered-dict-backed LRU cache with O(1) ops.
    Stores (key → value) pairs; evicts least-recently-used on overflow.
    """

    __slots__ = ("_maxsize", "_data")

    def __init__(self, maxsize: int = 1000) -> None:
        self._maxsize = max(1, maxsize)
        self._data: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def put(self, key: str, value: Any) -> Optional[Tuple[str, Any]]:
        """Insert/update. Returns (evicted_key, evicted_val) or None."""
        evicted = None
        if key in self._data:
            self._data.move_to_end(key)
            self._data[key] = value
        else:
            if len(self._data) >= self._maxsize:
                evicted = self._data.popitem(last=False)
            self._data[key] = value
        return evicted

    def get_lru_victim(self) -> Optional[Tuple[str, Any]]:
        """Peek at the LRU victim without removing it."""
        if not self._data:
            return None
        key = next(iter(self._data))
        return key, self._data[key]

    def remove(self, key: str) -> Optional[Any]:
        return self._data.pop(key, None)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()


# ── SLRU Cache ────────────────────────────────────────────────


class SLRUCache:
    """
    Segmented LRU: protected (80 %) + probationary (20 %) segments.
    
    Items enter probationary; on second access they promote to protected.
    Eviction always targets the probationary segment first.
    """

    __slots__ = ("_maxsize", "_protected", "_probationary",
                 "_protected_size", "_probationary_size")

    def __init__(self, maxsize: int = 1000,
                 protected_pct: float = 0.80) -> None:
        self._maxsize = max(2, maxsize)
        self._protected_size = max(1, int(maxsize * protected_pct))
        self._probationary_size = max(1, maxsize - self._protected_size)
        self._protected: OrderedDict[str, Any] = OrderedDict()
        self._probationary: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        # Check protected first
        if key in self._protected:
            self._protected.move_to_end(key)
            return self._protected[key]
        # Check probationary → promote
        if key in self._probationary:
            val = self._probationary.pop(key)
            self._promote(key, val)
            return val
        return None

    def put(self, key: str, value: Any) -> Optional[Tuple[str, Any]]:
        """Insert into probationary. Returns evicted entry or None."""
        if key in self._protected:
            self._protected.move_to_end(key)
            self._protected[key] = value
            return None
        if key in self._probationary:
            self._probationary.move_to_end(key)
            self._probationary[key] = value
            return None

        evicted = None
        if len(self._probationary) >= self._probationary_size:
            evicted = self._probationary.popitem(last=False)
        self._probationary[key] = value
        return evicted

    def _promote(self, key: str, value: Any) -> None:
        """Move from probationary to protected; demote if protected full."""
        if len(self._protected) >= self._protected_size:
            demoted_key, demoted_val = self._protected.popitem(last=False)
            # Demote to probationary
            if len(self._probationary) >= self._probationary_size:
                self._probationary.popitem(last=False)
            self._probationary[demoted_key] = demoted_val
        self._protected[key] = value

    def get_min_score_entry(self) -> Optional[Tuple[str, Any]]:
        """Peek at the probationary LRU victim."""
        if self._probationary:
            key = next(iter(self._probationary))
            return key, self._probationary[key]
        if self._protected:
            key = next(iter(self._protected))
            return key, self._protected[key]
        return None

    def remove(self, key: str) -> Optional[Any]:
        if key in self._protected:
            return self._protected.pop(key)
        return self._probationary.pop(key, None)

    def __len__(self) -> int:
        return len(self._protected) + len(self._probationary)

    def __contains__(self, key: str) -> bool:
        return key in self._protected or key in self._probationary
