"""
Circuit breaker for fault-tolerant dependency management.

Three states: CLOSED (normal) → OPEN (failing) → HALF_OPEN (probing).
Isolates failures in Redis, embedding service, and LLM backends
so a single dependency outage doesn't cascade.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from typing import Any, Callable, Optional

from .config import settings

logger = logging.getLogger(__name__)


class CircuitState(enum.Enum):
    CLOSED = "closed"       # normal operation
    OPEN = "open"           # failing, reject calls
    HALF_OPEN = "half_open" # probing with limited calls


class CircuitBreakerError(Exception):
    """Raised when the circuit is open and calls are rejected."""
    pass


class CircuitBreaker:
    """
    Per-dependency circuit breaker.

    CLOSED:    All calls pass through. Failures count up.
    OPEN:      All calls immediately raise CircuitBreakerError.
               After `reset_timeout_s`, transitions to HALF_OPEN.
    HALF_OPEN: One call is allowed through.
               Success → CLOSED.  Failure → OPEN.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = settings.cb_failure_threshold,
        reset_timeout_s: float = settings.cb_reset_timeout_s,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout_s

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0  # monotonic
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            # Check if reset timeout has elapsed
            if time.monotonic() - self._last_failure_time >= self.reset_timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info("Circuit '%s' transitioning to HALF_OPEN.", self.name)
        return self._state

    async def call(self, coro_func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute an async function through the circuit breaker.
        
        Raises CircuitBreakerError if the circuit is OPEN.
        """
        state = self.state
        self._total_calls += 1

        if state == CircuitState.OPEN:
            raise CircuitBreakerError(
                f"Circuit '{self.name}' is OPEN — dependency unavailable."
            )

        try:
            result = await coro_func(*args, **kwargs)
            self._on_success()
            return result
        except CircuitBreakerError:
            raise
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        self._total_successes += 1
        if self._state == CircuitState.HALF_OPEN:
            logger.info("Circuit '%s' recovered → CLOSED.", self.name)
        self._state = CircuitState.CLOSED
        self._failure_count = 0

    def _on_failure(self) -> None:
        self._total_failures += 1
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                "Circuit '%s' → OPEN after %d failures.",
                self.name,
                self._failure_count,
            )

    def reset(self) -> None:
        self._state = CircuitState.CLOSED
        self._failure_count = 0

    def get_stats(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
        }
