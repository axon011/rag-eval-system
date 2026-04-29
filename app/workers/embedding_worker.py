"""Background worker for asynchronous embedding pre-computation.

Synchronous embedding inside an ingest request handler blocks the event loop
for the duration of `embedder.embed_documents(chunks)`, which can be tens of
seconds for a multi-page upload. By running the embed call in a thread-pool
executor we keep the FastAPI process responsive to concurrent /query and
/health traffic while indexing proceeds in the background.

The worker is a process-level singleton with a small bounded pool so several
concurrent ingest requests serialize at the executor instead of fighting for
the event loop. This is the "async embedding pre-computation" path used by
the ingest route when INGEST_ASYNC_EMBED is enabled (default on).
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Awaitable, Callable, List, Optional

logger = logging.getLogger(__name__)


class EmbeddingWorker:
    def __init__(self, max_workers: Optional[int] = None):
        workers = max_workers or int(os.getenv("EMBED_WORKER_POOL", "2"))
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="embed-worker",
        )

    async def precompute(
        self,
        chunks: List[str],
        embed_fn: Callable[[List[str]], List[List[float]]],
        on_complete: Optional[Callable[[List[List[float]]], Awaitable[None]]] = None,
    ) -> List[List[float]]:
        """Embed `chunks` off the event loop and return the resulting vectors.

        `embed_fn` is the synchronous embedder callable (e.g.
        `embedder.embed_documents`). `on_complete`, if provided, is awaited
        with the vectors after embedding finishes — useful for chained
        upsert-into-vector-db steps.
        """
        if not chunks:
            return []
        loop = asyncio.get_running_loop()
        logger.info("embedding %d chunks in worker pool", len(chunks))
        vectors = await loop.run_in_executor(self._executor, embed_fn, chunks)
        if on_complete is not None:
            await on_complete(vectors)
        return vectors

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)


_worker: Optional[EmbeddingWorker] = None


def get_worker() -> EmbeddingWorker:
    """Return the process-wide embedding worker, creating it on first use."""
    global _worker
    if _worker is None:
        _worker = EmbeddingWorker()
    return _worker


def is_async_ingest_enabled() -> bool:
    """Whether the ingest route should dispatch embedding to the worker."""
    return os.getenv("INGEST_ASYNC_EMBED", "true").lower() == "true"
