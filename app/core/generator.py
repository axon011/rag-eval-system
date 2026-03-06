import os
from typing import List, Dict, Any
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


class Generator:
    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        temperature: float = 0.0,
        provider: str = None,
        api_key: str = None,
    ):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        self.model = model or os.getenv("LLM_MODEL", "llama3.2")
        self.temperature = temperature
        self.provider = provider or os.getenv("LLM_PROVIDER", "ollama")
        self.api_key = api_key or os.getenv("LLM_API_KEY")

        if self.provider == "openai":
            self.llm = ChatOpenAI(
                model=self.model,
                api_key=self.api_key,
                temperature=self.temperature,
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
        elif self.provider == "anthropic":
            self.llm = ChatAnthropic(
                model=self.model, api_key=self.api_key, temperature=self.temperature
            )
        else:
            self.llm = Ollama(
                base_url=self.base_url, model=self.model, temperature=self.temperature
            )

    def generate(self, question: str, context_chunks: List[Dict[str, Any]], model: str = None, provider: str = None, api_key: str = None) -> str:
        # Use provided values or fall back to instance values
        model = model or self.model
        provider = provider or self.provider
        api_key = api_key or self.api_key
        
        # Create LLM instance with provided settings if different
        if provider != self.provider or model != self.model or api_key != self.api_key:
            if provider == "openai":
                llm = ChatOpenAI(model=model, api_key=api_key, temperature=self.temperature)
            elif provider == "anthropic":
                llm = ChatAnthropic(model=model, api_key=api_key, temperature=self.temperature)
            else:
                llm = Ollama(base_url=self.base_url, model=model, temperature=self.temperature)
            current_llm = llm
        else:
            current_llm = self.llm
        context = "\n\n".join(
            [
                f"[Source {i + 1}]: {chunk['text']}"
                for i, chunk in enumerate(context_chunks)
            ]
        )

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

        if self.provider in ["openai", "anthropic"]:
            response = current_llm.invoke(prompt)
            return response.content
        else:
            response = current_llm.invoke(prompt)
            return response

    def rewrite_query(self, question: str) -> str:
        prompt = f"""Rewrite the following question to be more effective for document retrieval. 
Make it clearer and more specific without changing the meaning.

Original question: {question}

Rewritten question:"""

        if self.provider in ["openai", "anthropic"]:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        else:
            response = self.llm.invoke(prompt)
            return response.strip()

    def get_model_name(self) -> str:
        return self.model

    def get_provider(self) -> str:
        return self.provider
