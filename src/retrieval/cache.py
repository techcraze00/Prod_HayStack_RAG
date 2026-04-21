"""
Multi-Level Caching Layer

Provides LRU caching with TTL for:
- Retrieval results (query → search results)
- Embeddings (text → embedding vector)
- LLM responses (prompt → response)

Reduces latency and API calls for repeated or similar queries.
"""

import hashlib
import time
from typing import Any, Optional, Dict
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class CacheEntry:
    """A single cache entry with TTL"""
    value: Any
    created_at: float
    ttl: float
    hits: int = 0

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl


class LRUCache:
    """
    Thread-safe LRU cache with TTL support.

    Features:
    - Fixed capacity with LRU eviction
    - Per-entry TTL (time-to-live)
    - Hit/miss statistics
    """

    def __init__(self, capacity: int = 100, default_ttl: float = 3600):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache, returns None if missing or expired."""
        if key not in self._cache:
            self._misses += 1
            return None

        entry = self._cache[key]

        if entry.is_expired:
            del self._cache[key]
            self._misses += 1
            return None

        self._cache.move_to_end(key)
        entry.hits += 1
        self._hits += 1
        return entry.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Store value in cache."""
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl=ttl or self.default_ttl,
            )
        else:
            if len(self._cache) >= self.capacity:
                self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl=ttl or self.default_ttl,
            )

    def invalidate(self, key: str):
        """Remove a specific key from cache."""
        self._cache.pop(key, None)

    def clear(self):
        """Clear all entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "capacity": self.capacity,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


class RAGCache:
    """
    Multi-level cache for the RAG pipeline.

    Caches:
    - retrieval: query results
    - embeddings: text embeddings
    - responses: LLM responses
    """

    def __init__(
        self,
        retrieval_capacity: int = 200,
        retrieval_ttl: float = 1800,
        embedding_capacity: int = 500,
        embedding_ttl: float = 7200,
        response_capacity: int = 100,
        response_ttl: float = 3600,
    ):
        self.retrieval = LRUCache(retrieval_capacity, retrieval_ttl)
        self.embeddings = LRUCache(embedding_capacity, embedding_ttl)
        self.responses = LRUCache(response_capacity, response_ttl)

    @staticmethod
    def make_key(*args, **kwargs) -> str:
        """Create a deterministic cache key from arguments."""
        key_str = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_str.encode()).hexdigest()

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        return {
            "retrieval": self.retrieval.get_stats(),
            "embeddings": self.embeddings.get_stats(),
            "responses": self.responses.get_stats(),
        }

    def clear_all(self):
        self.retrieval.clear()
        self.embeddings.clear()
        self.responses.clear()
