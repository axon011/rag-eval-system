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
    model: Optional[str] = Field(default="llama3.2", description="Model to use for generation")
    provider: Optional[str] = Field(default="ollama", description="LLM provider: ollama, openai, anthropic")
    api_key: Optional[str] = Field(default=None, description="API key for external providers")
    retrieval_mode: Optional[str] = Field(default="hybrid", description="Retrieval mode: hybrid, dense, sparse")
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
