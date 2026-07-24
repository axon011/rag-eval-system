import asyncio
import os
import time
from fastapi import APIRouter, HTTPException

from app.models.schemas import QueryRequest, QueryResponse, Source
from app.core.pipeline import RAGPipeline
from app.core.generator import detect_provider_from_key, validate_provider_model

router = APIRouter(prefix="/query", tags=["Query"])

QUERY_CACHE_ENABLED = os.getenv("QUERY_CACHE_ENABLED", "true").lower() == "true"


@router.post("/", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    # Resolve the provider/model the pipeline will actually use (same logic as
    # the generator: an API-key prefix wins, else the request, else env), so
    # both the up-front validation and any error message name the real values.
    eff_provider = detect_provider_from_key(
        request.api_key,
        default=request.provider or os.getenv("LLM_PROVIDER", "ollama"),
    )
    eff_model = request.model or os.getenv("LLM_MODEL", "llama3.2")

    # Fail fast on an incoherent provider/model pair with an actionable message,
    # rather than letting it surface deep in the SDK as "Connection error."
    try:
        validate_provider_model(eff_provider, eff_model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

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

        pipeline = RAGPipeline(
            retrieval_mode=request.retrieval_mode,
            model=request.model,
            provider=request.provider,
            api_key=request.api_key,
            max_chunks=request.max_chunks
        )

        start_time = time.time()

        # The Claude CLI wrapper refuses sync .invoke() calls inside a running
        # event loop. Push the whole sync pipeline to a thread so any LLM
        # backend (Ollama, OpenAI, Claude CLI) works the same way from here.
        result = await asyncio.to_thread(
            pipeline.query,
            request.question,
            request.rewrite_query,
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

    except HTTPException:
        raise
    except Exception as e:
        # Surface the REAL failure — provider, model, exception type and message —
        # instead of a bare "Connection error." A terse SDK message like that is
        # almost always a provider/model/key mismatch; name the moving parts so
        # it's fixable from the UI.
        msg = str(e).strip() or "no detail"
        detail = (
            f"Query failed [provider={eff_provider}, model={eff_model}]: "
            f"{type(e).__name__}: {msg}"
        )
        if "connection error" in msg.lower():
            detail += (
                " — this usually means the model isn't valid for this provider, "
                "or the API key/base URL is wrong. Check the Model and API Key in Settings."
            )
        raise HTTPException(status_code=500, detail=detail)


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
