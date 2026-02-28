import os
from typing import List
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings


class Embedder:
    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        provider: str = None,
        api_key: str = None,
    ):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        self.model = model or os.getenv("EMBED_MODEL", "nomic-embed-text")
        self.provider = provider or os.getenv("EMBED_PROVIDER", "ollama")
        self.api_key = api_key or os.getenv("EMBED_API_KEY")

        if self.provider == "openai":
            self.embeddings = OpenAIEmbeddings(
                model=self.model,
                api_key=self.api_key,
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
        else:
            self.embeddings = OllamaEmbeddings(base_url=self.base_url, model=self.model)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

    def get_model_name(self) -> str:
        return self.model

    def get_provider(self) -> str:
        return self.provider
