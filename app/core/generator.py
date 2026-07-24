import os
from typing import List, Dict, Any, Optional
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


def detect_provider_from_key(api_key: Optional[str], default: Optional[str] = None) -> Optional[str]:
    """Infer the LLM provider from an API key prefix. Mirrors the UI auto-detect.

    Returns the detected provider, or ``default`` when the key is empty or its
    prefix is unrecognized (e.g. Ollama, or the keyless Claude CLI path).
    """
    if not api_key:
        return default
    key = api_key.strip()
    if key.startswith("sk-ant-"):
        return "anthropic"   # Anthropic
    if key.startswith("sk-or-"):
        return "openrouter"  # OpenRouter
    if key.startswith("sk-"):
        return "openai"      # OpenAI
    return default


# A sensible default model per provider, used when the UI switches providers and
# needs to replace a now-invalid model, or as a server-side fallback.
PROVIDER_DEFAULT_MODELS = {
    "ollama": "llama3.2",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
    "openrouter": "openai/gpt-4o-mini",
    "claude": "sonnet",
}


def default_model_for_provider(provider: Optional[str]) -> str:
    """Return a valid default model id for a provider."""
    return PROVIDER_DEFAULT_MODELS.get((provider or "").strip().lower(), "llama3.2")


def validate_provider_model(provider: Optional[str], model: Optional[str]) -> None:
    """Raise ValueError on a provider/model pair that cannot possibly work.

    High-precision on purpose — it only rejects combinations that are certain to
    fail (which otherwise surface as an opaque "Connection error"), so valid
    setups are never blocked. The common trap: an OpenRouter model id left in the
    field after switching the provider to Anthropic/OpenAI.
    """
    if not provider or not model:
        return
    p = provider.strip().lower()
    m = model.strip().lower()
    if p == "anthropic" and "claude" not in m:
        raise ValueError(
            f"Provider 'anthropic' cannot use model '{model}'. Use a Claude model "
            f"(e.g. 'claude-3-5-haiku-20241022'). A '.../...' id like this is usually "
            f"an OpenRouter model — switch the provider to 'openrouter' to use it."
        )
    if p == "openai" and "/" in model and "openrouter" not in (os.getenv("OPENAI_BASE_URL") or ""):
        raise ValueError(
            f"Provider 'openai' cannot use model '{model}'. Models with a '/' are "
            f"OpenRouter ids — switch the provider to 'openrouter', or use a plain "
            f"OpenAI model such as 'gpt-4o-mini'."
        )


def _build_claude_code_llm(model: str = "sonnet"):
    """ChatClaudeCode wrapper. Lazy-imports so the module isn't required
    unless `claude` is the selected provider. Runs the Claude CLI as a
    subprocess against the user's logged-in subscription — no API key.
    """
    from langchain_claude_code import ChatClaudeCode
    return ChatClaudeCode(model=model, permission_mode="default")


class Generator:
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        self.model = model or os.getenv("LLM_MODEL", "llama3.2")
        self.temperature = temperature
        self.provider = provider or os.getenv("LLM_PROVIDER", "ollama")
        self.api_key = api_key or os.getenv("LLM_API_KEY")

        # Auto-detect the provider from the API key prefix (mirrors the UI).
        self.provider = detect_provider_from_key(self.api_key, default=self.provider)

        if self.provider == "claude":
            # Claude CLI subscription path — no API key, model ∈ {sonnet,opus,haiku}.
            self.llm = _build_claude_code_llm(model=self.model or "sonnet")
        elif self.provider == "openrouter":
            self.llm = ChatOpenAI(
                model=self.model,
                api_key=self.api_key,
                temperature=self.temperature,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://github.com/axon011/rag-eval-system",
                    "X-Title": "RAG Eval System"
                }
            )
        elif self.provider == "openai":
            openai_base_url = os.getenv("OPENAI_BASE_URL")
            # Support OpenRouter (api.openrouter.ai)
            if openai_base_url and "openrouter" in openai_base_url:
                self.llm = ChatOpenAI(
                    model=self.model,
                    api_key=self.api_key,
                    temperature=self.temperature,
                    base_url=openai_base_url,
                    default_headers={
                        "HTTP-Referer": "https://github.com/axon011/rag-eval-system",
                        "X-Title": "RAG Eval System"
                    }
                )
            else:
                self.llm = ChatOpenAI(
                    model=self.model,
                    api_key=self.api_key,
                    temperature=self.temperature,
                    base_url=openai_base_url if openai_base_url else None,
                )
        elif self.provider == "anthropic":
            self.llm = ChatAnthropic(
                model=self.model, api_key=self.api_key, temperature=self.temperature
            )
        else:
            self.llm = Ollama(
                base_url=self.base_url, model=self.model, temperature=self.temperature
            )

    def generate(
        self, 
        question: str, 
        context_chunks: List[Dict[str, Any]], 
        model: Optional[str] = None, 
        provider: Optional[str] = None, 
        api_key: Optional[str] = None
    ) -> str:
        # Use provided values or fall back to instance values
        model = model or self.model
        provider = provider or self.provider
        api_key = api_key or self.api_key
        
        # Auto-detect the provider from the API key prefix (mirrors the UI).
        provider = detect_provider_from_key(api_key, default=provider)
        
        # Create LLM instance with provided settings if different
        if provider != self.provider or model != self.model or api_key != self.api_key:
            if provider == "claude":
                llm = _build_claude_code_llm(model=model or "sonnet")
            elif provider == "openrouter":
                llm = ChatOpenAI(
                    model=model, 
                    api_key=api_key, 
                    temperature=self.temperature,
                    base_url="https://openrouter.ai/api/v1",
                    default_headers={
                        "HTTP-Referer": "https://github.com/axon011/rag-eval-system",
                        "X-Title": "RAG Eval System"
                    }
                )
            elif provider == "openai":
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

        if provider in ["openai", "anthropic", "openrouter", "claude"]:
            response = current_llm.invoke(prompt)
            return response.content
        else:
            response = current_llm.invoke(prompt)
            return str(response)

    def rewrite_query(self, question: str) -> str:
        prompt = f"""Rewrite the following question to be more effective for document retrieval. 
Make it clearer and more specific without changing the meaning.

Original question: {question}

Rewritten question:"""

        if self.provider in ["openai", "anthropic", "openrouter", "claude"]:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        else:
            response = self.llm.invoke(prompt)
            return str(response).strip()

    def get_model_name(self) -> str:
        return self.model

    def get_provider(self) -> str:
        return self.provider
