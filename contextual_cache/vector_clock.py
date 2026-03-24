"""
Vector clock implementation for causal ordering in distributed mode.

Each node maintains a logical clock; vector clocks track the
happens-before relation between events across nodes. Used for
conflict resolution when distributed cache entries are updated
concurrently on different nodes.
"""

from __future__ import annotations

from typing import Dict


class VectorClock:
    """
    Vector clock for causal event ordering.

    Operations:
      increment()     — tick the local node's clock
      merge(other)    — merge with a received clock (element-wise max)
      happens_before(other) — check causal ordering
      is_concurrent(other)  — check if events are concurrent

    Space: O(N) where N = number of nodes that have ever participated.
    """

    def __init__(self, node_id: str, clock: Dict[str, int] | None = None) -> None:
        self.node_id = node_id
        self._clock: Dict[str, int] = dict(clock) if clock else {}
        if node_id not in self._clock:
            self._clock[node_id] = 0

    def increment(self) -> VectorClock:
        """Increment the local node's clock and return self."""
        self._clock[self.node_id] = self._clock.get(self.node_id, 0) + 1
        return self

    def merge(self, other: VectorClock) -> VectorClock:
        """Merge with another vector clock (element-wise max) and return self."""
        for node, ts in other._clock.items():
            self._clock[node] = max(self._clock.get(node, 0), ts)
        return self

    def happens_before(self, other: VectorClock) -> bool:
        """
        Check if this clock happens-before `other`.

        A ≤ B iff ∀k: A[k] ≤ B[k], and ∃k: A[k] < B[k].
        """
        all_leq = True
        at_least_one_lt = False
        all_keys = set(self._clock.keys()) | set(other._clock.keys())
        for k in all_keys:
            a = self._clock.get(k, 0)
            b = other._clock.get(k, 0)
            if a > b:
                all_leq = False
                break
            if a < b:
                at_least_one_lt = True
        return all_leq and at_least_one_lt

    def is_concurrent(self, other: VectorClock) -> bool:
        """Check if events are concurrent (neither happens-before the other)."""
        return not self.happens_before(other) and not other.happens_before(self)

    def get(self, node_id: str) -> int:
        """Get the clock value for a specific node."""
        return self._clock.get(node_id, 0)

    def to_dict(self) -> Dict[str, int]:
        """Serialize for transmission or storage."""
        return dict(self._clock)

    @classmethod
    def from_dict(cls, node_id: str, data: Dict[str, int]) -> VectorClock:
        """Deserialize from dict."""
        return cls(node_id, clock=data)

    def __repr__(self) -> str:
        return f"VectorClock({self.node_id}, {self._clock})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorClock):
            return NotImplemented
        return self._clock == other._clock
