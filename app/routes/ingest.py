import os
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.models.schemas import IngestResponse
from app.core.loaders import get_loader

router = APIRouter(prefix="/ingest", tags=["Ingest"])


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}


@router.post("/", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """Ingest PDF, Markdown, or text files."""

    file_ext = os.path.splitext(file.filename.lower())[1]

    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    try:
        contents = await file.read()

        loader = get_loader(file_ext)
        chunks = loader.load_and_chunk(contents)

        if not chunks or not chunks[0].strip():
            raise HTTPException(
                status_code=400, detail="No text could be extracted from file"
            )

        from app.core.pipeline import RAGPipeline

        pipeline = RAGPipeline()

        result = pipeline.ingest_documents(chunks)

        # New corpus = old query answers may be wrong. Invalidate query cache.
        from app.cache import query_cache
        query_cache.clear()

        return IngestResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.post("/pdf", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    """Ingest PDF files only."""

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    return await ingest_document(file)


@router.post("/markdown", response_model=IngestResponse)
async def ingest_markdown(file: UploadFile = File(...)):
    """Ingest Markdown files only."""

    file_ext = os.path.splitext(file.filename.lower())[1]
    if file_ext not in [".md", ".markdown", ".txt"]:
        raise HTTPException(status_code=400, detail="Only Markdown files are supported")

    return await ingest_document(file)


@router.get("/health")
async def ingest_health():
    return {"status": "ingest service healthy"}


@router.get("/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    from app.cache import embedding_cache, query_cache

    return {
        "embedding_cache": embedding_cache.stats(),
        "query_cache": query_cache.stats(),
    }


@router.post("/cache/clear")
async def clear_cache():
    """Clear all caches."""
    from app.cache import embedding_cache, query_cache

    embedding_cache.clear()
    query_cache.clear()

    return {"status": "success", "message": "All caches cleared"}
