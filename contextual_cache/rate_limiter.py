"""
Token bucket rate limiter with per-tenant support.

Prevents overload by limiting requests per second. Returns HTTP 429
when the bucket is empty. Supports per-tenant isolation so one noisy
tenant cannot starve others.
"""

from __future__ import annotations

import time
from typing import Dict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from .config import settings


class TokenBucket:
    """
    Token bucket rate limiter.

    Tokens are added at `rate` per second up to `burst` capacity.
    Each request consumes one token. If empty, the request is rejected.

    Thread-safe for single-threaded async (no lock needed).
    """

    __slots__ = ("_rate", "_burst", "_tokens", "_last_refill")

    def __init__(self, rate: float, burst: int) -> None:
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now

    def allow(self) -> bool:
        """Try to consume a token. Returns True if allowed."""
        self._refill()
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def wait_time(self) -> float:
        """Seconds until at least one token is available."""
        self._refill()
        if self._tokens >= 1.0:
            return 0.0
        return (1.0 - self._tokens) / self._rate

    @property
    def available_tokens(self) -> float:
        self._refill()
        return self._tokens


class PerTenantRateLimiter:
    """Rate limiter with per-tenant token buckets."""

    def __init__(self, rate: float, burst: int) -> None:
        self._rate = rate
        self._burst = burst
        self._buckets: Dict[str, TokenBucket] = {}

    def allow(self, tenant_id: str = "default") -> bool:
        if tenant_id not in self._buckets:
            self._buckets[tenant_id] = TokenBucket(self._rate, self._burst)
        return self._buckets[tenant_id].allow()

    def wait_time(self, tenant_id: str = "default") -> float:
        if tenant_id not in self._buckets:
            return 0.0
        return self._buckets[tenant_id].wait_time()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that enforces rate limits.

    Returns 429 Too Many Requests when the rate limit is exceeded.
    Adds Retry-After header indicating when the client should retry.

    Skips rate limiting for internal/health endpoints.
    """

    _SKIP_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

    def __init__(self, app, rate: float = 100.0, burst: int = 200) -> None:
        super().__init__(app)
        self._limiter = PerTenantRateLimiter(rate=rate, burst=burst)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip rate limiting for health/docs/internal endpoints
        if path in self._SKIP_PATHS or path.startswith("/internal/"):
            return await call_next(request)

        # Extract tenant from header or default
        tenant_id = request.headers.get("x-tenant-id", "default")

        if not self._limiter.allow(tenant_id):
            retry_after = self._limiter.wait_time(tenant_id)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after_seconds": round(retry_after, 2),
                },
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

        return await call_next(request)
