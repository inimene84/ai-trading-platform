"""Provider selection for Jev calls.

The default provider is the direct TypeSafe client. LiteLLM is used only
when JEV_PROVIDER=litellm. The mock provider is explicit and cannot be a
silent fallback, because a stub answer must not look like a live decision.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from backend.jev.client import JevClient, JevUnavailable
from backend.jev.config import jev_fallback_provider, jev_pause_seconds, jev_provider, jev_timeout_seconds
from backend.jev.schema import validate_system_one

_paused_until = 0.0


def reset_circuit() -> None:
    global _paused_until
    _paused_until = 0.0


def circuit_open() -> bool:
    return time.time() < _paused_until


def note_credit_failure(message: str) -> None:
    global _paused_until
    text = message.lower()
    if "402" in text or "insufficient credit" in text or "insufficient credits" in text:
        _paused_until = time.time() + jev_pause_seconds()


class MockJevClient:
    """Deterministic HOLD. Marked mock so the book ignores it."""

    name = "mock"

    async def system_one(self, _state: dict[str, Any], _questions: dict[str, Any]) -> dict[str, Any]:
        if circuit_open():
            raise JevUnavailable("Jev provider paused after credit errors")
        payload = {
            "model": "mock",
            "answers": {
                "trade_action": {
                    "choice": "HOLD",
                    "confidence": 0.5,
                    "probabilities": {
                        "STRONG_BUY": 0.05,
                        "BUY": 0.1,
                        "HOLD": 0.6,
                        "TAKE_PROFIT": 0.1,
                        "SELL": 0.1,
                        "STRONG_SELL": 0.05,
                    },
                },
                "sentiment_spectrum": {"score": 2.0},
                "is_short_squeeze_risk": {"noul": 0.2},
                "catalyst_impact": {"score": 0.0},
                "price_direction": {
                    "choice": "FLAT",
                    "probabilities": {"UP": 0.2, "FLAT": 0.6, "DOWN": 0.2},
                },
            },
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        return validate_system_one(payload)


class LiteLLMJevClient:
    name = "litellm"

    def __init__(self, transport: httpx.BaseTransport | None = None) -> None:
        self._transport = transport

    async def system_one(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        if circuit_open():
            raise JevUnavailable("Jev provider paused after credit errors")
        base = os.getenv("LITELLM_BASE_URL", "http://litellm:4000").rstrip("/")
        key = os.getenv("LITELLM_API_KEY", "").strip()
        if not key:
            raise JevUnavailable("LITELLM_API_KEY not configured")
        url = f"{base}/typesafe/v1/systemone"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        body = {"model": os.getenv("JEV_MODEL", "jev-1.13.0"), "state": state, "questions": questions}
        try:
            async with httpx.AsyncClient(timeout=jev_timeout_seconds(), transport=self._transport) as client:
                response = await client.post(url, headers=headers, json=body)
                if response.status_code == 402:
                    note_credit_failure("402")
                    raise JevUnavailable("Jev provider returned 402")
                response.raise_for_status()
                return validate_system_one(response.json())
        except JevUnavailable:
            raise
        except httpx.HTTPError as exc:
            note_credit_failure(str(exc))
            raise JevUnavailable(f"LiteLLM Jev HTTP error: {exc}") from exc
        except ValueError as exc:
            raise JevUnavailable("LiteLLM Jev response was not JSON") from exc


def build_client(name: str | None = None, transport: httpx.BaseTransport | None = None) -> Any:
    provider = (name or jev_provider()).strip().lower()
    if provider == "mock":
        return MockJevClient()
    if provider == "litellm":
        return LiteLLMJevClient(transport=transport)
    client = JevClient(transport=transport)
    client.name = "typesafe"  # type: ignore[attr-defined]
    return client


async def evaluate_providers(state: dict[str, Any], questions: dict[str, Any], client: Any | None = None) -> tuple[dict[str, Any], str]:
    primary = client or build_client()
    name = getattr(primary, "name", jev_provider())
    try:
        return await primary.system_one(state, questions), name
    except JevUnavailable as exc:
        note_credit_failure(str(exc))
        fallback = jev_fallback_provider()
        if not fallback or fallback == name or fallback == "mock":
            raise
        secondary = build_client(fallback)
        return await secondary.system_one(state, questions), fallback
