from pydantic import BaseModel, Field
from typing import List, Optional, Any


class IngestRequest(BaseModel):
    pass


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
    embed_model: str


class Source(BaseModel):
    text: str
    score: float
    methods: List[str] = []


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask")
    rewrite_query: bool = Field(
        default=True, description="Whether to rewrite the query for better retrieval"
    )
    model: Optional[str] = Field(default=None, description="Model; falls back to LLM_MODEL env var")
    provider: Optional[str] = Field(default=None, description="LLM provider (ollama/openai/anthropic); falls back to LLM_PROVIDER env var")
    api_key: Optional[str] = Field(default=None, description="API key; falls back to LLM_API_KEY env var")
    retrieval_mode: Optional[str] = Field(default=None, description="Retrieval mode (hybrid/dense/sparse); falls back to RETRIEVAL_MODE env var")
    max_chunks: Optional[int] = Field(default=5, description="Maximum number of chunks to retrieve")


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    retrieved_chunks: int
    retrieval_mode: str
    rewritten_query: Optional[str] = None
    latency_ms: Optional[float] = None
    cached: Optional[bool] = False


class HealthResponse(BaseModel):
    status: str
    services: dict
