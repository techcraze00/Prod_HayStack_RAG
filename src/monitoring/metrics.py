"""
RAG System Metrics Collector

Tracks and aggregates operational metrics:
- Query latency (P50, P90, P99)
- Cache hit/miss rates
- Retrieval quality scores
- Fallback rates
- Error counts
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class QueryMetrics:
    """Metrics for a single query"""
    query: str
    strategy: str = ""
    latency_ms: float = 0
    retrieval_latency_ms: float = 0
    generation_latency_ms: float = 0
    num_results: int = 0
    avg_score: float = 0
    cache_hit: bool = False
    fallback_used: bool = False
    fallback_strategy: str = ""
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Collects and aggregates RAG system metrics."""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._queries: List[QueryMetrics] = []
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._strategy_counts: Dict[str, int] = defaultdict(int)
        self._total_queries = 0
        self._total_cache_hits = 0
        self._total_fallbacks = 0

    def record_query(self, metrics: QueryMetrics):
        self._total_queries += 1
        self._queries.append(metrics)

        if metrics.cache_hit:
            self._total_cache_hits += 1
        if metrics.fallback_used:
            self._total_fallbacks += 1
        if metrics.strategy:
            self._strategy_counts[metrics.strategy] += 1
        if metrics.error:
            self._error_counts[metrics.error] += 1

        if len(self._queries) > self.max_history:
            self._queries = self._queries[-self.max_history:]

    def record_error(self, error_type: str):
        self._error_counts[error_type] += 1

    def timer(self, name: str = "operation") -> "Timer":
        return Timer(name)

    def get_summary(self) -> Dict[str, Any]:
        if not self._queries:
            return {"total_queries": 0, "message": "No queries recorded"}

        latencies = [q.latency_ms for q in self._queries if q.latency_ms > 0]
        retrieval_latencies = [
            q.retrieval_latency_ms for q in self._queries
            if q.retrieval_latency_ms > 0
        ]
        scores = [q.avg_score for q in self._queries if q.avg_score > 0]

        return {
            "total_queries": self._total_queries,
            "latency": self._percentiles(latencies) if latencies else {},
            "retrieval_latency": (
                self._percentiles(retrieval_latencies)
                if retrieval_latencies else {}
            ),
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "cache_hit_rate": (
                self._total_cache_hits / self._total_queries
                if self._total_queries else 0
            ),
            "fallback_rate": (
                self._total_fallbacks / self._total_queries
                if self._total_queries else 0
            ),
            "strategy_distribution": dict(self._strategy_counts),
            "error_counts": dict(self._error_counts),
        }

    def export_json(self, path: str):
        summary = self.get_summary()
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    @staticmethod
    def _percentiles(values: List[float]) -> Dict[str, float]:
        if not values:
            return {}
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        return {
            "p50": sorted_vals[int(n * 0.5)],
            "p90": sorted_vals[int(n * 0.9)],
            "p99": sorted_vals[min(int(n * 0.99), n - 1)],
            "avg": sum(sorted_vals) / n,
            "min": sorted_vals[0],
            "max": sorted_vals[-1],
        }


class Timer:
    """Simple context manager timer."""

    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time: float = 0
        self.elapsed_ms: float = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.time() - self.start_time) * 1000
