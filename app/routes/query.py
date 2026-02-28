import os
import time
from fastapi import APIRouter, HTTPException

from app.models.schemas import QueryRequest, QueryResponse, Source
from app.core.pipeline import RAGPipeline

router = APIRouter(prefix="/query", tags=["Query"])

QUERY_CACHE_ENABLED = os.getenv("QUERY_CACHE_ENABLED", "true").lower() == "true"


@router.post("/", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        if QUERY_CACHE_ENABLED:
            from app.cache import query_cache, CachedResponse

            cached = query_cache.get(request.question)
            if cached:
                return QueryResponse(
                    answer=cached.answer,
                    sources=[Source(**s) for s in cached.sources],
                    retrieved_chunks=cached.retrieved_chunks,
                    retrieval_mode=cached.retrieval_mode,
                    rewritten_query=cached.rewritten_query,
                    latency_ms=0,
                    cached=True,
                )

        pipeline = RAGPipeline()

        start_time = time.time()

        result = pipeline.query(
            question=request.question, rewrite_query=request.rewrite_query
        )

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        sources = [Source(**source) for source in result["sources"]]

        if QUERY_CACHE_ENABLED:
            from app.cache import query_cache, CachedResponse

            query_cache.set(
                request.question,
                CachedResponse(
                    answer=result["answer"],
                    sources=result["sources"],
                    retrieved_chunks=result["retrieved_chunks"],
                    retrieval_mode=result["retrieval_mode"],
                    rewritten_query=result.get("rewritten_query"),
                    timestamp=time.time(),
                ),
            )

        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            retrieved_chunks=result["retrieved_chunks"],
            retrieval_mode=result["retrieval_mode"],
            rewritten_query=result.get("rewritten_query"),
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.get("/config")
async def get_config():
    try:
        pipeline = RAGPipeline()
        return pipeline.get_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting config: {str(e)}")


@router.get("/health")
async def query_health():
    return {"status": "query service healthy"}


@router.get("/cache/stats")
async def cache_stats():
    from app.cache import query_cache

    return query_cache.stats()
