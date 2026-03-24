"""Tests for the consistent hashing ring."""

from contextual_cache.consistent_hash import ConsistentHashRing


class TestConsistentHashRing:

    def test_empty_ring_returns_none(self):
        ring = ConsistentHashRing()
        assert ring.get_node("some-key") is None

    def test_single_node(self):
        ring = ConsistentHashRing(nodes=["node-1"])
        assert ring.get_node("any-key") == "node-1"
        assert ring.get_node("another-key") == "node-1"

    def test_deterministic_routing(self):
        ring = ConsistentHashRing(nodes=["node-1", "node-2", "node-3"])
        node1 = ring.get_node("test-key")
        node2 = ring.get_node("test-key")
        assert node1 == node2

    def test_distribution_across_nodes(self):
        ring = ConsistentHashRing(
            nodes=["node-1", "node-2", "node-3"],
            virtual_nodes=150,
        )
        counts = {"node-1": 0, "node-2": 0, "node-3": 0}
        for i in range(3000):
            node = ring.get_node(f"key-{i}")
            counts[node] += 1

        # Each node should get roughly 1000 keys (within 30% tolerance)
        for node, count in counts.items():
            assert 500 < count < 1500, f"{node} got {count} keys"

    def test_add_node_minimal_redistribution(self):
        ring = ConsistentHashRing(nodes=["node-1", "node-2"])
        keys = [f"key-{i}" for i in range(1000)]
        before = {k: ring.get_node(k) for k in keys}

        ring.add_node("node-3")
        after = {k: ring.get_node(k) for k in keys}

        # At most ~1/3 of keys should move
        moved = sum(1 for k in keys if before[k] != after[k])
        assert moved < 500  # conservative bound

    def test_remove_node(self):
        ring = ConsistentHashRing(nodes=["node-1", "node-2", "node-3"])
        assert ring.node_count == 3

        ring.remove_node("node-2")
        assert ring.node_count == 2

        # All keys should route to remaining nodes
        for i in range(100):
            node = ring.get_node(f"key-{i}")
            assert node in ("node-1", "node-3")

    def test_replication_nodes(self):
        ring = ConsistentHashRing(nodes=["node-1", "node-2", "node-3"])
        replicas = ring.get_nodes_for_replication("test-key", n=3)
        assert len(replicas) == 3
        assert len(set(replicas)) == 3  # all distinct

    def test_replication_limited_by_node_count(self):
        ring = ConsistentHashRing(nodes=["node-1", "node-2"])
        replicas = ring.get_nodes_for_replication("test-key", n=5)
        assert len(replicas) == 2  # only 2 physical nodes

    def test_stats(self):
        ring = ConsistentHashRing(nodes=["a", "b"], virtual_nodes=100)
        stats = ring.get_stats()
        assert stats["node_count"] == 2
        assert stats["ring_size"] == 200
        assert stats["virtual_nodes_per_node"] == 100

    def test_add_duplicate_node_is_noop(self):
        ring = ConsistentHashRing(nodes=["node-1"])
        initial_size = ring.ring_size
        ring.add_node("node-1")
        assert ring.ring_size == initial_size
