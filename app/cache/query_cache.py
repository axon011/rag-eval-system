import os
import hashlib
import time
from typing import Optional, Any, Dict, List
from dataclasses import dataclass
from cachetools import LRUCache


RESPONSE_CACHE_TTL = int(os.getenv("RESPONSE_CACHE_TTL", "900"))
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))


@dataclass
class CachedResponse:
    """Cached query response."""

    answer: str
    sources: List[Dict[str, Any]]
    retrieved_chunks: int
    retrieval_mode: str
    rewritten_query: Optional[str]
    timestamp: float


class QueryCache:
    """In-memory cache for query responses with TTL support."""

    def __init__(self, maxsize: int = CACHE_MAX_SIZE, ttl: int = RESPONSE_CACHE_TTL):
        self.cache: LRUCache = LRUCache(maxsize=maxsize)
        self.ttl = ttl
        self.timestamps: Dict[str, float] = {}
        self.hits = 0
        self.misses = 0

    def _hash(
        self,
        question: str,
        provider: str = "",
        model: str = "",
        retrieval_mode: str = "",
    ) -> str:
        """Hash question plus config so different models don't share cache entries."""
        payload = f"{question.lower().strip()}|{provider}|{model}|{retrieval_mode}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(
        self,
        question: str,
        provider: str = "",
        model: str = "",
        retrieval_mode: str = "",
    ) -> Optional[CachedResponse]:
        """Get response from cache if not expired."""
        key = self._hash(question, provider, model, retrieval_mode)

        if key in self.cache:
            if time.time() - self.timestamps.get(key, 0) < self.ttl:
                self.hits += 1
                return self.cache[key]
            else:
                del self.cache[key]
                if key in self.timestamps:
                    del self.timestamps[key]

        self.misses += 1
        return None

    def set(
        self,
        question: str,
        response: CachedResponse,
        provider: str = "",
        model: str = "",
        retrieval_mode: str = "",
    ):
        """Store response in cache."""
        key = self._hash(question, provider, model, retrieval_mode)
        self.cache[key] = response
        self.timestamps[key] = time.time()

    def clear(self):
        """Clear all cached responses."""
        self.cache.clear()
        self.timestamps.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.cache.maxsize,
            "ttl_seconds": self.ttl,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": round(hit_rate, 2),
        }


# Global instance
query_cache = QueryCache()
