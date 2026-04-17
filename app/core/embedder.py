import os
from typing import List
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings


class SentenceTransformerEmbeddings:
    """Wrapper to match LangChain embeddings interface using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        try:
            self.dim = self.model.get_embedding_dimension()
        except AttributeError:
            self.dim = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()


class Embedder:
    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        provider: str = None,
        api_key: str = None,
        use_cache: bool = True,
    ):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        self.model = model or os.getenv("EMBED_MODEL", "nomic-embed-text")
        self.provider = provider or os.getenv("EMBED_PROVIDER", "ollama")
        self.api_key = api_key or os.getenv("EMBED_API_KEY")
        self.use_cache = (
            use_cache and os.getenv("QUERY_CACHE_ENABLED", "true").lower() == "true"
        )

        if self.provider == "sentence-transformers":
            self.embeddings = SentenceTransformerEmbeddings(model_name=self.model)
            self._dim = self.embeddings.dim
        elif self.provider == "openai":
            self.embeddings = OpenAIEmbeddings(
                model=self.model,
                api_key=self.api_key,
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
            self._dim = None  # determined at runtime
        else:
            self.embeddings = OllamaEmbeddings(base_url=self.base_url, model=self.model)
            self._dim = None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        if self.use_cache:
            from app.cache import embedding_cache

            # Include model name so swapping providers doesn't return stale vectors.
            cache_key = f"{self.provider}:{self.model}:{text}"
            cached = embedding_cache.get(cache_key)
            if cached is not None:
                return cached

            embedding = self.embeddings.embed_query(text)
            embedding_cache.set(cache_key, embedding)
            return embedding

        return self.embeddings.embed_query(text)

    def get_model_name(self) -> str:
        return self.model

    def get_provider(self) -> str:
        return self.provider

    def get_dimension(self) -> int:
        """Return embedding dimension, using model metadata when possible.

        SentenceTransformer exposes it directly; OpenAI/Ollama fall back to a
        single probe (uncached so we don't pollute the cache with 'test')."""
        if self._dim:
            return self._dim
        # Try SBERT's native attribute first.
        if hasattr(self.embeddings, "dim"):
            self._dim = self.embeddings.dim
            return self._dim
        # Last resort: uncached probe — bypass the cache path intentionally.
        test = self.embeddings.embed_query("__dim_probe__")
        self._dim = len(test)
        return self._dim
