import os
from typing import List
from langchain_community.embeddings import OllamaEmbeddings


class Embedder:
    def __init__(
        self,
        base_url: str = None,
        model: str = None
    ):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        self.model = model or os.getenv("EMBED_MODEL", "nomic-embed-text")
        self.embeddings = OllamaEmbeddings(
            base_url=self.base_url,
            model=self.model
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

    def get_model_name(self) -> str:
        return self.model
