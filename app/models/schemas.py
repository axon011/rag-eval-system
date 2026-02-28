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
    rewrite_query: bool = Field(default=True, description="Whether to rewrite the query for better retrieval")


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    retrieved_chunks: int
    retrieval_mode: str
    rewritten_query: Optional[str] = None
    latency_ms: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    services: dict
