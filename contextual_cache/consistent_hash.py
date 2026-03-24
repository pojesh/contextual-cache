"""
Consistent hashing ring for distributed cache sharding.

Maps cache keys to nodes using SHA-256 with virtual nodes for
even distribution. Supports dynamic node addition/removal with
minimal key redistribution (only ~1/N keys move on rebalance).
"""

from __future__ import annotations

import hashlib
import logging
from bisect import bisect_right, insort
from typing import Dict, List, Optional, Set

from .config import settings

logger = logging.getLogger(__name__)


class ConsistentHashRing:
    """
    Consistent hashing ring with virtual nodes.

    Each physical node is mapped to `virtual_nodes` positions on the ring
    to ensure even distribution. Keys are hashed to a ring position and
    routed to the next node clockwise.

    Time complexity:
      get_node: O(log V) where V = total virtual nodes
      add_node: O(V log V) amortized
      remove_node: O(V log V) amortized
    """

    def __init__(
        self,
        nodes: Optional[List[str]] = None,
        virtual_nodes: int = settings.virtual_nodes
        if hasattr(settings, "virtual_nodes") else 150,
    ) -> None:
        self._virtual_nodes = virtual_nodes
        self._ring: List[int] = []            # sorted ring positions
        self._ring_to_node: Dict[int, str] = {}  # position → physical node
        self._nodes: Set[str] = set()

        for node in (nodes or []):
            self.add_node(node)

    @staticmethod
    def _hash(key: str) -> int:
        """SHA-256 → 64-bit integer position on the ring."""
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big")

    def add_node(self, node: str) -> None:
        """Add a physical node with its virtual replicas to the ring."""
        if node in self._nodes:
            return
        self._nodes.add(node)
        for i in range(self._virtual_nodes):
            vnode_key = f"{node}#vn{i}"
            pos = self._hash(vnode_key)
            self._ring_to_node[pos] = node
            insort(self._ring, pos)
        logger.info("Added node '%s' (%d vnodes) to hash ring.", node,
                     self._virtual_nodes)

    def remove_node(self, node: str) -> Set[str]:
        """
        Remove a physical node and all its virtual replicas.
        Returns the set of ring positions that were removed (for rebalancing).
        """
        if node not in self._nodes:
            return set()
        self._nodes.discard(node)
        removed_positions: Set[str] = set()
        new_ring: List[int] = []
        for pos in self._ring:
            if self._ring_to_node.get(pos) == node:
                del self._ring_to_node[pos]
                removed_positions.add(str(pos))
            else:
                new_ring.append(pos)
        self._ring = new_ring
        logger.info("Removed node '%s' from hash ring.", node)
        return removed_positions

    def get_node(self, key: str) -> Optional[str]:
        """
        Determine which node owns `key`.

        Returns None if the ring is empty.
        """
        if not self._ring:
            return None
        pos = self._hash(key)
        idx = bisect_right(self._ring, pos) % len(self._ring)
        return self._ring_to_node[self._ring[idx]]

    def get_nodes_for_replication(self, key: str, n: int = 3) -> List[str]:
        """
        Return up to `n` distinct physical nodes for replicating `key`.
        Walks clockwise from the key's position, skipping duplicate nodes.
        """
        if not self._ring:
            return []
        pos = self._hash(key)
        idx = bisect_right(self._ring, pos) % len(self._ring)
        result: List[str] = []
        seen: Set[str] = set()
        for _ in range(len(self._ring)):
            node = self._ring_to_node[self._ring[idx]]
            if node not in seen:
                seen.add(node)
                result.append(node)
                if len(result) >= n:
                    break
            idx = (idx + 1) % len(self._ring)
        return result

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def ring_size(self) -> int:
        return len(self._ring)

    def get_stats(self) -> dict:
        return {
            "nodes": sorted(self._nodes),
            "node_count": len(self._nodes),
            "ring_size": len(self._ring),
            "virtual_nodes_per_node": self._virtual_nodes,
        }
