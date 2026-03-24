"""Tests for vector clock causal ordering."""

from contextual_cache.vector_clock import VectorClock


class TestVectorClock:

    def test_initial_state(self):
        vc = VectorClock("node-1")
        assert vc.get("node-1") == 0

    def test_increment(self):
        vc = VectorClock("node-1")
        vc.increment()
        assert vc.get("node-1") == 1
        vc.increment()
        assert vc.get("node-1") == 2

    def test_happens_before(self):
        a = VectorClock("node-1")
        a.increment()  # {node-1: 1}

        b = VectorClock("node-2", clock=a.to_dict())
        b.increment()  # {node-1: 1, node-2: 1}

        assert a.happens_before(b)
        assert not b.happens_before(a)

    def test_concurrent_events(self):
        a = VectorClock("node-1")
        a.increment()  # {node-1: 1}

        b = VectorClock("node-2")
        b.increment()  # {node-2: 1}

        assert a.is_concurrent(b)
        assert b.is_concurrent(a)

    def test_merge(self):
        a = VectorClock("node-1")
        a.increment()
        a.increment()  # {node-1: 2}

        b = VectorClock("node-2")
        b.increment()  # {node-2: 1}

        a.merge(b)
        assert a.get("node-1") == 2
        assert a.get("node-2") == 1

    def test_merge_takes_max(self):
        a = VectorClock("node-1", clock={"node-1": 3, "node-2": 1})
        b = VectorClock("node-2", clock={"node-1": 1, "node-2": 5})

        a.merge(b)
        assert a.get("node-1") == 3
        assert a.get("node-2") == 5

    def test_to_dict_and_from_dict(self):
        vc = VectorClock("node-1")
        vc.increment()
        vc.increment()

        data = vc.to_dict()
        restored = VectorClock.from_dict("node-1", data)
        assert restored == vc

    def test_equality(self):
        a = VectorClock("node-1", clock={"node-1": 1, "node-2": 2})
        b = VectorClock("node-2", clock={"node-1": 1, "node-2": 2})
        assert a == b

    def test_not_happens_before_equal(self):
        a = VectorClock("node-1", clock={"node-1": 1})
        b = VectorClock("node-2", clock={"node-1": 1})
        # Equal clocks: neither happens-before
        assert not a.happens_before(b)
        assert not b.happens_before(a)
        # But they're not concurrent either (they're equal)
        # is_concurrent returns True since neither dominates
        # Actually equal clocks ARE concurrent by definition
        assert a.is_concurrent(b)
