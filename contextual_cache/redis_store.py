"""
Optional Redis-backed exact-match store (enhanced Tier 1).

When enabled, exact-hash lookups check Redis first, falling back
to the in-memory dict if Redis is unavailable.  Uses circuit breaker
for fault tolerance.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Tuple

from .circuit_breaker import CircuitBreaker, CircuitBreakerError
from .config import settings

logger = logging.getLogger(__name__)


class RedisStore:
    """Redis-backed key-value store for exact-match cache entries."""

    def __init__(self) -> None:
        self._client = None
        self._circuit = CircuitBreaker(name="redis")
        self._enabled = settings.redis_enabled
        if self._enabled:
            self._connect()

    def _connect(self) -> None:
        try:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
            )
            logger.info("Redis store connected: %s", settings.redis_url)
        except Exception as e:
            logger.warning(
                "Redis connection failed: %s — falling back to in-memory", e
            )
            self._client = None

    @property
    def available(self) -> bool:
        return self._enabled and self._client is not None

    async def get(self, key: str) -> Optional[Tuple[str, str]]:
        """Return (entry_id, response_text) or None."""
        if not self.available:
            return None
        try:

            async def _get():
                raw = await self._client.get(key)
                if raw is None:
                    return None
                data = json.loads(raw)
                return (data["entry_id"], data["response"])

            return await self._circuit.call(_get)
        except CircuitBreakerError:
            return None
        except Exception as e:
            logger.debug("Redis GET failed: %s", e)
            return None

    async def set(
        self, key: str, entry_id: str, response: str, ttl_s: int = 0
    ) -> None:
        """Store an entry in Redis with optional TTL."""
        if not self.available:
            return
        try:

            async def _set():
                value = json.dumps({"entry_id": entry_id, "response": response})
                if ttl_s > 0:
                    await self._client.setex(key, ttl_s, value)
                else:
                    await self._client.set(key, value)

            await self._circuit.call(_set)
        except (CircuitBreakerError, Exception) as e:
            logger.debug("Redis SET failed: %s", e)

    async def delete(self, key: str) -> None:
        """Remove an entry from Redis."""
        if not self.available:
            return
        try:

            async def _del():
                await self._client.delete(key)

            await self._circuit.call(_del)
        except (CircuitBreakerError, Exception):
            pass

    async def close(self) -> None:
        """Shutdown the Redis connection."""
        if self._client:
            await self._client.aclose()
            self._client = None
