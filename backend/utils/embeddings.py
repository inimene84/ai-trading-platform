"""Shared text embedding helper — OpenRouter first, LiteLLM fallback."""

from __future__ import annotations

import logging
import math
import os
from typing import Optional

import httpx

from backend.services.provider_credit_circuit import credit_circuit, looks_like_credit_error

logger = logging.getLogger(__name__)

OPENROUTER_EMBED_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
).rstrip("/") + "/embeddings"

DEFAULT_EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    os.getenv("TRADE_MEMORY_EMBED_MODEL", "openai/text-embedding-3-small"),
)
DEFAULT_VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "1536"))


def _fit_vector(vec: list[float], vector_size: int, normalize: bool) -> list[float]:
    if len(vec) >= vector_size:
        out = vec[:vector_size]
    else:
        out = vec + [0.0] * (vector_size - len(vec))
    if not normalize:
        return out
    norm = math.sqrt(sum(x * x for x in out)) or 1.0
    return [x / norm for x in out]


async def _openrouter_embed(text: str, model: str) -> Optional[list[float]]:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return None
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    title = os.getenv("OPENROUTER_APP_TITLE", "ai-trading-platform").strip()
    if title:
        headers["X-Title"] = title

    payload: dict = {"model": model, "input": text}
    dimensions = os.getenv("EMBED_DIMENSIONS", "").strip()
    if dimensions.isdigit():
        payload["dimensions"] = int(dimensions)

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(OPENROUTER_EMBED_URL, headers=headers, json=payload)
        if resp.status_code == 402:
            credit_circuit.note_failure("openrouter", "embedding 402")
            resp.raise_for_status()
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]


async def _litellm_embed(text: str, model: str) -> Optional[list[float]]:
    base = os.getenv("LITELLM_BASE_URL", "http://litellm:4000/v1").rstrip("/")
    key = os.getenv("LITELLM_API_KEY", "") or os.getenv("KIE_API_KEY", "")
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{base}/embeddings",
            headers={"Authorization": f"Bearer {key}"} if key else {},
            json={"model": model, "input": text},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]


async def generate_text_embedding(
    text: str,
    *,
    vector_size: int = DEFAULT_VECTOR_SIZE,
    normalize: bool = True,
    model: Optional[str] = None,
) -> Optional[list[float]]:
    """Return a vector for *text*, or None if all providers fail."""
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    embed_model = model or DEFAULT_EMBED_MODEL
    # OpenRouter model ids use provider/model; bare OpenAI names still work there.
    if "/" not in embed_model and os.getenv("OPENROUTER_API_KEY"):
        embed_model = f"openai/{embed_model}"

    providers = []
    if os.getenv("OPENROUTER_API_KEY", "").strip():
        providers.append(("openrouter", _openrouter_embed))
    providers.append(("litellm", _litellm_embed))

    for name, fn in providers:
        if credit_circuit.is_open(name):
            logger.info(
                "[%s] embedding skipped — credit circuit open (%.0fs left)",
                name,
                credit_circuit.remaining_seconds(name),
            )
            continue
        try:
            vec = await fn(cleaned, embed_model)
            if vec:
                credit_circuit.note_success(name)
                return _fit_vector(vec, vector_size, normalize)
        except Exception as exc:
            if looks_like_credit_error(exc):
                credit_circuit.note_failure(name, str(exc))
            logger.warning("[%s] embedding failed (%s): %s", name, embed_model, exc)

    return None
