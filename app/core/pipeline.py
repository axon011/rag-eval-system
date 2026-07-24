import os
from typing import List, Dict, Any, Optional
from .embedder import Embedder
from .retriever import Retriever
from .generator import Generator, detect_provider_from_key
from . import tracing


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
        # Auto-detect the provider from the API key prefix so tracing metadata
        # and the generate() call below use the same resolved provider.
        self.provider = detect_provider_from_key(self.api_key, default=self.provider)
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

        # Root observation. Every stage below nests inside it automatically —
        # nesting follows Python scope, no parent IDs are threaded through.
        with tracing.observe(
            as_type="span",
            name="rag-query",
            input={"question": original_question},
            metadata={
                "retrieval_mode": self.retrieval_mode,
                "llm_model": self.model,
                "provider": self.provider,
                "top_k": self.max_chunks,
            },
        ) as root:

            # Query rewriting is itself an LLM call, so type it as a generation
            # rather than a plain span — otherwise its tokens go uncounted.
            if rewrite_query:
                with tracing.observe(
                    as_type="generation",
                    name="query-rewrite",
                    model=self.model,
                    input=question,
                ) as rw:
                    question = self.generator.rewrite_query(question)
                    rw.update(output=question)

            with tracing.observe(
                as_type="span", name="embed-query", input={"text": question}
            ) as emb:
                query_vector = self.embedder.embed_query(question)
                emb.update(output={"dim": len(query_vector)})

            # "retriever" is a first-class observation type in v4 and is the
            # correct one for RAG lookups.
            with tracing.observe(
                as_type="retriever",
                name=f"retrieve-{self.retrieval_mode}",
                input={"query": question, "top_k": self.max_chunks},
            ) as ret:
                retrieved_docs = self.retriever.retrieve(
                    query_vector=query_vector,
                    query=question,
                    mode=self.retrieval_mode,
                    top_k=self.max_chunks
                )
                ret.update(output={
                    "hits": len(retrieved_docs),
                    "scores": [d.get("score") for d in retrieved_docs],
                    "methods": [d.get("methods", []) for d in retrieved_docs],
                })

            with tracing.observe(
                as_type="generation",
                name="generate-answer",
                model=self.model,
                input={"question": original_question, "n_contexts": len(retrieved_docs)},
            ) as gen:
                answer = self.generator.generate(
                    original_question,
                    retrieved_docs,
                    model=self.model,
                    provider=self.provider,
                    api_key=self.api_key
                )
                gen.update(output=answer)

            root.update(output={"answer": answer, "retrieved_chunks": len(retrieved_docs)})

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
