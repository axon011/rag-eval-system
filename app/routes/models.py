"""List available models for a provider so the UI can offer a real dropdown
instead of a free-text field (which is what let the P1 provider/model mismatch
happen). Given a provider + optional API key, query that provider's models API
and return the model ids.

The API key is read from the request body (never the URL/query string) and is
never logged or echoed back.
"""
import os
from typing import List, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/models", tags=["Models"])

_TIMEOUT = httpx.Timeout(10.0)


class ModelsRequest(BaseModel):
    provider: str = Field(..., description="ollama | openai | anthropic | openrouter | claude")
    api_key: Optional[str] = Field(default=None, description="Provider API key (never logged)")


class ModelsResponse(BaseModel):
    provider: str
    models: List[str]


async def _ollama_models() -> List[str]:
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(f"{base}/api/tags")
        resp.raise_for_status()
        data = resp.json()
    return sorted(m["name"] for m in data.get("models", []) if m.get("name"))


async def _openai_models(api_key: str) -> List[str]:
    if not api_key:
        raise HTTPException(status_code=400, detail="An OpenAI API key is required to list models.")
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        data = resp.json()
    # Keep chat/generation-capable ids; drop embeddings/audio/image utility models.
    ids = [m["id"] for m in data.get("data", []) if m.get("id")]
    keep = [i for i in ids if i.startswith(("gpt-", "o1", "o3", "chatgpt"))]
    return sorted(keep or ids)


async def _openrouter_models(api_key: str) -> List[str]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get("https://openrouter.ai/api/v1/models", headers=headers)
        resp.raise_for_status()
        data = resp.json()
    return sorted(m["id"] for m in data.get("data", []) if m.get("id"))


async def _anthropic_models(api_key: str) -> List[str]:
    if not api_key:
        raise HTTPException(status_code=400, detail="An Anthropic API key is required to list models.")
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(
            "https://api.anthropic.com/v1/models",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        )
        resp.raise_for_status()
        data = resp.json()
    return sorted(m["id"] for m in data.get("data", []) if m.get("id"))


@router.post("/", response_model=ModelsResponse)
async def list_models(request: ModelsRequest) -> ModelsResponse:
    provider = (request.provider or "").strip().lower()
    # Fall back to the server's configured key when the UI didn't send one.
    api_key = (request.api_key or "").strip() or os.getenv("LLM_API_KEY", "")

    try:
        if provider == "ollama":
            models = await _ollama_models()
        elif provider == "claude":
            # Claude CLI subscription path — fixed aliases, no models API.
            models = ["sonnet", "opus", "haiku"]
        elif provider == "openai":
            models = await _openai_models(api_key)
        elif provider == "openrouter":
            models = await _openrouter_models(api_key)
        elif provider == "anthropic":
            models = await _anthropic_models(api_key)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown provider '{provider}'.")
    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        # Surface the provider's status without leaking the key or body.
        raise HTTPException(
            status_code=502,
            detail=f"The {provider} models API returned HTTP {e.response.status_code}. "
                   f"Check the API key and try again.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Could not fetch {provider} models: {type(e).__name__}: {e}",
        )

    return ModelsResponse(provider=provider, models=models)


@router.get("/health")
async def models_health():
    return {"status": "models service healthy"}
