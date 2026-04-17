import os
from typing import List, Dict, Any, Optional
from .embedder import Embedder
from .retriever import Retriever
from .generator import Generator


class RAGPipeline:
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        retriever: Optional[Retriever] = None,
        generator: Optional[Generator] = None,
        retrieval_mode: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        max_chunks: int = 5
    ):
        self.embedder = embedder or Embedder()
        self.retriever = retriever or Retriever.get_instance()
        self.generator = generator or Generator()

        # Align Qdrant collection dimension to the embedder so mismatched
        # EMBED_DIM env vars don't silently corrupt inserts.
        try:
            self.retriever.ensure_dimension(self.embedder.get_dimension())
        except Exception:
            # Retriever may be a mock in tests; safe to skip.
            pass
        self.retrieval_mode = retrieval_mode or os.getenv("RETRIEVAL_MODE", "hybrid")
        # Fall back to env-driven defaults so a caller that passes nothing
        # still uses LLM_PROVIDER / LLM_MODEL / LLM_API_KEY instead of
        # silently forcing Ollama.
        self.model = model or os.getenv("LLM_MODEL", "llama3.2")
        self.provider = provider or os.getenv("LLM_PROVIDER", "ollama")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.max_chunks = max_chunks

    def ingest_documents(self, chunks: List[str]) -> Dict[str, Any]:
        vectors = self.embedder.embed_documents(chunks)
        self.retriever.add_documents(chunks, vectors)
        
        return {
            "status": "success",
            "chunks_indexed": len(chunks),
            "embed_model": self.embedder.get_model_name()
        }

    def query(self, question: str, rewrite_query: bool = True) -> Dict[str, Any]:
        original_question = question
        
        if rewrite_query:
            question = self.generator.rewrite_query(question)
        
        query_vector = self.embedder.embed_query(question)
        
        retrieved_docs = self.retriever.retrieve(
            query_vector=query_vector,
            query=question,
            mode=self.retrieval_mode,
            top_k=self.max_chunks
        )
        
        answer = self.generator.generate(
            original_question, 
            retrieved_docs,
            model=self.model,
            provider=self.provider,
            api_key=self.api_key
        )
        
        sources = [
            {
                "text": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"],
                "score": doc["score"],
                "methods": doc.get("methods", [])
            }
            for doc in retrieved_docs
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(retrieved_docs),
            "retrieval_mode": self.retrieval_mode,
            "rewritten_query": question if rewrite_query else None
        }

    def get_config(self) -> Dict[str, Any]:
        return {
            "embed_model": self.embedder.get_model_name(),
            "llm_model": self.generator.get_model_name(),
            "retrieval_mode": self.retrieval_mode,
            "top_k": self.retriever.top_k
        }
