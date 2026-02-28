import time
from fastapi import APIRouter, HTTPException

from app.models.schemas import QueryRequest, QueryResponse, Source
from app.core.pipeline import RAGPipeline

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        pipeline = RAGPipeline()
        
        start_time = time.time()
        
        result = pipeline.query(
            question=request.question,
            rewrite_query=request.rewrite_query
        )
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        sources = [Source(**source) for source in result["sources"]]
        
        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            retrieved_chunks=result["retrieved_chunks"],
            retrieval_mode=result["retrieval_mode"],
            rewritten_query=result.get("rewritten_query"),
            latency_ms=round(latency_ms, 2)
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
