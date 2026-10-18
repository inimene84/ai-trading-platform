"""Shared text embedding helper — OpenRouter, OpenAI, then LiteLLM.

Qdrant ``finmem-memory`` (and the rest of the news/memory collections) uses
``QDRANT_VECTOR_SIZE`` (default 1536), matching ``text-embedding-3-small``.
OpenAI is the fail-soft backend when OpenRouter/LiteLLM are missing or down.
A missing ``OPENAI_API_KEY`` or provider error returns ``None`` — callers must
never let that raise into a trading loop.
"""

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

_OPENAI_DEFAULT_BASE = "https://api.openai.com/v1"

DEFAULT_EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    os.getenv("TRADE_MEMORY_EMBED_MODEL", "openai/text-embedding-3-small"),
)
DEFAULT_VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "1536"))
_PLACEHOLDER_KEYS = frozenset({"", "your_openai_api_key_here", "changeme", "changeme-openai"})


def _fit_vector(vec: list[float], vector_size: int, normalize: bool) -> list[float]:
    if len(vec) >= vector_size:
        out = vec[:vector_size]
    else:
        out = vec + [0.0] * (vector_size - len(vec))
    if not normalize:
        return out
    norm = math.sqrt(sum(x * x for x in out)) or 1.0
    return [x / norm for x in out]


def _usable_api_key(raw: Optional[str]) -> bool:
    key = (raw or "").strip()
    if not key:
        return False
    return key.lower() not in _PLACEHOLDER_KEYS


def _openai_model_id(model: str) -> str:
    """OpenAI expects ``text-embedding-3-small``; OpenRouter uses ``openai/...``."""
    name = (model or "").strip()
    if name.startswith("openai/"):
        return name.split("/", 1)[1]
    return name or "text-embedding-3-small"


def _embed_dimensions() -> Optional[int]:
    raw = os.getenv("EMBED_DIMENSIONS", "").strip()
    if raw.isdigit():
        return int(raw)
    return None


def _openai_embed_url() -> str:
    """Resolve the embeddings URL per call so late dotenv / tests can override it."""
    base = (os.getenv("OPENAI_BASE_URL") or _OPENAI_DEFAULT_BASE).strip().rstrip("/")
    if not base:
        base = _OPENAI_DEFAULT_BASE
    return f"{base}/embeddings"


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
    dimensions = _embed_dimensions()
    if dimensions is not None:
        payload["dimensions"] = dimensions

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(OPENROUTER_EMBED_URL, headers=headers, json=payload)
        if resp.status_code == 402:
            credit_circuit.note_failure("openrouter", "embedding 402")
            resp.raise_for_status()
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]


async def _openai_embed(text: str, model: str) -> Optional[list[float]]:
    """Direct OpenAI embeddings API. Returns None when the key is missing."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not _usable_api_key(api_key):
        return None
    payload: dict = {"model": _openai_model_id(model), "input": text}
    dimensions = _embed_dimensions()
    if dimensions is not None:
        payload["dimensions"] = dimensions
    # Distinct circuit name so embedding 401/429 never trips the OpenAI *chat* fallback.
    async with httpx.AsyncClient(timeout=15.0, trust_env=False, follow_redirects=False) as client:
        resp = await client.post(
            _openai_embed_url(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if resp.status_code in (401, 402, 403, 429):
            credit_circuit.trip("openai-embed", f"embedding {resp.status_code}")
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
    if _usable_api_key(os.getenv("OPENROUTER_API_KEY")):
        providers.append(("openrouter", _openrouter_embed))
    if _usable_api_key(os.getenv("OPENAI_API_KEY")):
        providers.append(("openai-embed", _openai_embed))
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
