import os
import hashlib
import time
from typing import Optional, Any, Dict
from cachetools import LRUCache


EMBEDDING_CACHE_TTL = int(os.getenv("EMBEDDING_CACHE_TTL", "3600"))
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))


class EmbeddingCache:
    """In-memory cache for embeddings with TTL support."""

    def __init__(self, maxsize: int = CACHE_MAX_SIZE, ttl: int = EMBEDDING_CACHE_TTL):
        self.cache: LRUCache = LRUCache(maxsize=maxsize)
        self.ttl = ttl
        self.timestamps: Dict[str, float] = {}

    def _hash(self, text: str) -> str:
        """Generate hash for text."""
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[list]:
        """Get embedding from cache if not expired."""
        key = self._hash(text)

        if key in self.cache:
            if time.time() - self.timestamps.get(key, 0) < self.ttl:
                return self.cache[key]
            else:
                del self.cache[key]
                if key in self.timestamps:
                    del self.timestamps[key]

        return None

    def set(self, text: str, embedding: list):
        """Store embedding in cache."""
        key = self._hash(text)
        self.cache[key] = embedding
        self.timestamps[key] = time.time()

    def clear(self):
        """Clear all cached embeddings."""
        self.cache.clear()
        self.timestamps.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.cache.maxsize,
            "ttl_seconds": self.ttl,
        }


# Global instance
embedding_cache = EmbeddingCache()
