"""Direct TypeSafe System One client.

Calls POST {JEV_BASE_URL}/v1/systemone. JEV_BASE_URL can point at a gateway,
but this process does not invent a LiteLLM route. Missing credentials,
timeouts, HTTP errors, and invalid JSON never become a trade.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import httpx

from backend.jev.config import jev_base_url, jev_model, jev_timeout_seconds, typesafe_api_key
from backend.jev.schema import JevSchemaError, validate_system_one

logger = logging.getLogger(__name__)


class JevUnavailable(RuntimeError):
    """Jev could not produce a validated evaluation."""


class JevClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = typesafe_api_key() if api_key is None else api_key.strip()
        self.base_url = (base_url or jev_base_url()).rstrip("/")
        self.model = model or jev_model()
        self.timeout = jev_timeout_seconds() if timeout is None else timeout
        self._transport = transport

    async def system_one(
        self,
        state: dict[str, Any],
        questions: dict[str, Any],
        validator: Callable[[Any], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise JevUnavailable("TYPESAFE_API_KEY not configured")
        url = f"{self.base_url}/v1/systemone"
        payload = {"model": self.model, "state": state, "questions": questions}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                body = response.json()
        except httpx.TimeoutException as exc:
            raise JevUnavailable("Jev request timed out") from exc
        except httpx.HTTPError as exc:
            raise JevUnavailable(f"Jev HTTP error: {exc}") from exc
        except ValueError as exc:
            raise JevUnavailable("Jev response was not JSON") from exc
        try:
            checker = validator or validate_system_one
            return checker(body)
        except JevSchemaError as exc:
            raise JevUnavailable(f"Jev schema rejected: {exc}") from exc


def log_unavailable(symbol: str, exc: Exception) -> None:
    logger.warning("Jev evaluation unavailable for %s: %s", symbol, exc)
