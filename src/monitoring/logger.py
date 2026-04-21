"""
RAG System Structured Logger

Provides structured logging for all RAG operations:
- Query processing
- Retrieval results
- Fallback events
- Cache hits/misses
- Errors with context
"""

import logging
import json
import time
from typing import Dict, Any, Optional, List


class RAGLogger:
    """Structured logger for RAG system events."""

    def __init__(self, name: str = "rag", log_level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _log_structured(self, level: int, event: str, data: Dict[str, Any]):
        log_entry = {
            "event": event,
            "timestamp": time.time(),
            **data,
        }
        self.logger.log(level, json.dumps(log_entry, default=str))

    def log_query(self, query: str, strategy: str = "", query_type: str = ""):
        self._log_structured(logging.INFO, "query", {
            "query": query[:200],
            "strategy": strategy,
            "query_type": query_type,
        })

    def log_retrieval(
        self, results: List[Dict[str, Any]], latency_ms: float = 0, method: str = "vector",
    ):
        scores = [r.get("score", 0) for r in results]
        self._log_structured(logging.INFO, "retrieval", {
            "method": method,
            "num_results": len(results),
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "latency_ms": round(latency_ms, 2),
        })

    def log_fallback(self, strategy: str, success: bool, message: str = ""):
        self._log_structured(logging.WARNING, "fallback", {
            "strategy": strategy,
            "success": success,
            "message": message,
        })

    def log_cache(self, cache_type: str, hit: bool, key: str = ""):
        self._log_structured(logging.DEBUG, "cache", {
            "cache_type": cache_type,
            "hit": hit,
            "key": key[:50] if key else "",
        })

    def log_generation(self, latency_ms: float = 0, token_count: int = 0):
        self._log_structured(logging.INFO, "generation", {
            "latency_ms": round(latency_ms, 2),
            "token_count": token_count,
        })

    def log_error(self, error_type: str, details: Optional[Dict[str, Any]] = None):
        self._log_structured(logging.ERROR, "error", {
            "error_type": error_type,
            "details": details or {},
        })

    def log_routing(
        self, query: str, classification: Dict[str, Any], strategy: Dict[str, Any],
    ):
        self._log_structured(logging.INFO, "routing", {
            "query": query[:200],
            "classification": classification,
            "strategy": strategy,
        })
