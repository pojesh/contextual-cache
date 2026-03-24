"""Tests for the token bucket rate limiter."""

import time

from contextual_cache.rate_limiter import PerTenantRateLimiter, TokenBucket


class TestTokenBucket:

    def test_allows_within_burst(self):
        bucket = TokenBucket(rate=10.0, burst=5)
        for _ in range(5):
            assert bucket.allow()

    def test_rejects_after_burst(self):
        bucket = TokenBucket(rate=10.0, burst=3)
        for _ in range(3):
            bucket.allow()
        assert not bucket.allow()

    def test_refills_over_time(self):
        bucket = TokenBucket(rate=1000.0, burst=5)
        # Drain all tokens
        for _ in range(5):
            bucket.allow()
        assert not bucket.allow()

        # Manually advance time by manipulating last_refill
        bucket._last_refill -= 0.01  # 10ms ago
        assert bucket.allow()  # refilled ~10 tokens

    def test_wait_time_zero_when_available(self):
        bucket = TokenBucket(rate=10.0, burst=5)
        assert bucket.wait_time() == 0.0

    def test_wait_time_positive_when_empty(self):
        bucket = TokenBucket(rate=10.0, burst=1)
        bucket.allow()
        wt = bucket.wait_time()
        assert wt > 0.0

    def test_available_tokens(self):
        bucket = TokenBucket(rate=10.0, burst=5)
        assert bucket.available_tokens == 5.0
        bucket.allow()
        assert bucket.available_tokens < 5.0


class TestPerTenantRateLimiter:

    def test_independent_tenants(self):
        limiter = PerTenantRateLimiter(rate=10.0, burst=2)

        # Tenant A
        assert limiter.allow("tenant-a")
        assert limiter.allow("tenant-a")
        assert not limiter.allow("tenant-a")

        # Tenant B should still have tokens
        assert limiter.allow("tenant-b")

    def test_default_tenant(self):
        limiter = PerTenantRateLimiter(rate=10.0, burst=1)
        assert limiter.allow()
        assert not limiter.allow()
