import os
from typing import List, Dict, Any
from langchain_community.llms import Ollama


class Generator:
    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        temperature: float = 0.0
    ):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
        self.temperature = temperature
        
        self.llm = Ollama(
            base_url=self.base_url,
            model=self.model,
            temperature=self.temperature
        )

    def generate(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        context = "\n\n".join([
            f"[Source {i+1}]: {chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        prompt = f"""You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Instructions:
- Only use information from the provided context
- If the context doesn't contain enough information to answer, say so
- Cite sources using [Source X] format
- Be concise and direct

Answer:"""

        response = self.llm.invoke(prompt)
        return response

    def rewrite_query(self, question: str) -> str:
        prompt = f"""Rewrite the following question to be more effective for document retrieval. 
Make it clearer and more specific without changing the meaning.

Original question: {question}

Rewritten question:"""

        response = self.llm.invoke(prompt)
        return response.strip()

    def get_model_name(self) -> str:
        return self.model
