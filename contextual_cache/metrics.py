"""
In-memory metrics collector with rolling history for the dashboard.

Collects per-query metrics and maintains aggregate statistics
with a configurable rolling window for time-series visualization.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Dict, List

from .config import settings
from .models import AggregateMetrics, QueryMetrics


class MetricsCollector:
    """
    Thread-safe metrics collector.
    
    - Records per-query metrics
    - Maintains rolling aggregate
    - Provides time-series data for the analytics dashboard
    """

    def __init__(self, history_size: int = settings.metrics_history_size) -> None:
        self._lock = threading.Lock()
        self._history: deque[QueryMetrics] = deque(maxlen=history_size)
        self._aggregate = AggregateMetrics()
        self._time_series: deque[dict] = deque(maxlen=history_size)

        # Latency tracking
        self._hit_latencies: deque[float] = deque(maxlen=history_size)
        self._miss_latencies: deque[float] = deque(maxlen=history_size)
        self._all_latencies: deque[float] = deque(maxlen=history_size)

        # Threshold history
        self._threshold_history: deque[float] = deque(maxlen=history_size)

        # Similarity history
        self._similarity_history: deque[float] = deque(maxlen=history_size)

        # Production metrics
        self.ttl_expirations = 0
        self.index_rebuilds = 0
        self.rate_limit_rejections = 0

    def record_query(self, metrics: QueryMetrics) -> None:
        """Record a single query's metrics."""
        with self._lock:
            self._history.append(metrics)
            agg = self._aggregate

            agg.total_queries += 1
            self._all_latencies.append(metrics.latency_ms)

            if metrics.hit:
                agg.total_hits += 1
                if metrics.tier == 1:
                    agg.tier1_hits += 1
                elif metrics.tier == 2:
                    agg.tier2_hits += 1
                self._hit_latencies.append(metrics.latency_ms)
                if metrics.similarity > 0:
                    self._similarity_history.append(metrics.similarity)
            else:
                agg.total_misses += 1
                self._miss_latencies.append(metrics.latency_ms)

            if metrics.was_admission_rejected:
                agg.admission_rejections += 1

            if metrics.was_eviction_triggered:
                agg.total_evictions += 1

            if metrics.threshold_used > 0:
                self._threshold_history.append(metrics.threshold_used)

            # Update latency averages
            if self._all_latencies:
                agg.avg_latency_ms = sum(self._all_latencies) / len(self._all_latencies)
            if self._hit_latencies:
                agg.avg_hit_latency_ms = sum(self._hit_latencies) / len(self._hit_latencies)
            if self._miss_latencies:
                agg.avg_miss_latency_ms = sum(self._miss_latencies) / len(self._miss_latencies)

            # Time series data point
            self._time_series.append({
                "timestamp": metrics.timestamp,
                "hit": metrics.hit,
                "tier": metrics.tier,
                "latency_ms": round(metrics.latency_ms, 2),
                "similarity": round(metrics.similarity, 4),
                "hit_rate": round(agg.hit_rate, 4),
            })

    def record_feedback(self, was_correct: bool) -> None:
        """Record correctness feedback for a cache hit."""
        with self._lock:
            if was_correct:
                self._aggregate.correct_hits += 1
            else:
                self._aggregate.incorrect_hits += 1

    def update_cache_size(self, size: int, capacity: int) -> None:
        with self._lock:
            self._aggregate.cache_size = size
            self._aggregate.cache_capacity = capacity

    def record_admission(self) -> None:
        with self._lock:
            self._aggregate.total_admissions += 1

    def record_ttl_expiration(self, count: int = 1) -> None:
        self.ttl_expirations += count

    def record_index_rebuild(self) -> None:
        self.index_rebuilds += 1

    def record_rate_limit_rejection(self) -> None:
        self.rate_limit_rejections += 1

    def get_aggregate(self) -> dict:
        with self._lock:
            return self._aggregate.to_dict()

    def get_time_series(self, last_n: int = 100) -> List[dict]:
        with self._lock:
            data = list(self._time_series)
            return data[-last_n:]

    def get_latency_distribution(self) -> dict:
        """Return latency percentiles for the dashboard."""
        import numpy as np
        with self._lock:
            if not self._all_latencies:
                return {"p50": 0, "p90": 0, "p95": 0, "p99": 0}
            arr = np.array(list(self._all_latencies))
            return {
                "p50": round(float(np.percentile(arr, 50)), 2),
                "p90": round(float(np.percentile(arr, 90)), 2),
                "p95": round(float(np.percentile(arr, 95)), 2),
                "p99": round(float(np.percentile(arr, 99)), 2),
            }

    def get_threshold_distribution(self) -> List[float]:
        with self._lock:
            return [round(t, 4) for t in self._threshold_history]

    def get_similarity_distribution(self) -> List[float]:
        with self._lock:
            return [round(s, 4) for s in self._similarity_history]

    def get_full_analytics(self) -> dict:
        """Return everything the dashboard needs in a single call."""
        return {
            "aggregate": self.get_aggregate(),
            "time_series": self.get_time_series(200),
            "latency_distribution": self.get_latency_distribution(),
            "threshold_distribution": self.get_threshold_distribution(),
            "similarity_distribution": self.get_similarity_distribution(),
        }
