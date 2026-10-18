"""Shared credit / HTTP-402 circuit for LLM, embedding, and Jev calls.

OpenRouter (and similar prepaid providers) return 402 when the account is out
of credits. Without a circuit those calls retry every cycle — about 1000
errors/hour on embeddings + Jev opinions — and drown the logs while the
router already falls through to gpt-4o-mini.

This module is process-local and fail-open: an unknown provider or a clock
glitch never blocks a working fallback.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_COOLDOWN_SEC = 1800.0  # 30 minutes
_CREDIT_HINTS = ("402", "insufficient credit", "insufficient credits", "payment required")


def _cooldown_seconds() -> float:
    raw = os.getenv("LLM_CREDIT_CIRCUIT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_COOLDOWN_SEC
    try:
        return max(60.0, float(raw))
    except ValueError:
        return DEFAULT_COOLDOWN_SEC


class ProviderCreditCircuit:
    """Trip a named provider for ``LLM_CREDIT_CIRCUIT_SECONDS`` after a 402."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._open_until: dict[str, float] = {}
        self._last_reason: dict[str, str] = {}

    def _norm(self, provider: str) -> str:
        name = (provider or "").strip().lower()
        if name.startswith("openrouter"):
            return "openrouter"
        return name

    def is_open(self, provider: str) -> bool:
        key = self._norm(provider)
        if not key:
            return False
        with self._lock:
            until = self._open_until.get(key, 0.0)
            if until <= time.monotonic():
                if key in self._open_until:
                    self._open_until.pop(key, None)
                    self._last_reason.pop(key, None)
                return False
            return True

    def remaining_seconds(self, provider: str) -> float:
        key = self._norm(provider)
        with self._lock:
            until = self._open_until.get(key, 0.0)
        return max(0.0, until - time.monotonic())

    def note_success(self, provider: str) -> None:
        key = self._norm(provider)
        with self._lock:
            self._open_until.pop(key, None)
            self._last_reason.pop(key, None)

    def trip(self, provider: str, reason: str = "402") -> None:
        key = self._norm(provider)
        if not key:
            return
        cooldown = _cooldown_seconds()
        with self._lock:
            already = self._open_until.get(key, 0.0) > time.monotonic()
            self._open_until[key] = time.monotonic() + cooldown
            self._last_reason[key] = reason
        if not already:
            logger.warning(
                "Credit circuit OPEN for %s (%.0fs): %s",
                key,
                cooldown,
                reason,
            )

    def note_failure(self, provider: str, message: str) -> bool:
        """Trip when *message* looks like a credit / 402 failure. Return True if tripped."""
        text = (message or "").lower()
        if any(hint in text for hint in _CREDIT_HINTS):
            self.trip(provider, message)
            return True
        return False

    def snapshot(self) -> dict[str, dict[str, float | str]]:
        now = time.monotonic()
        with self._lock:
            return {
                name: {
                    "remaining_sec": round(max(0.0, until - now), 1),
                    "reason": self._last_reason.get(name, ""),
                }
                for name, until in self._open_until.items()
                if until > now
            }

    def reset(self) -> None:
        with self._lock:
            self._open_until.clear()
            self._last_reason.clear()


credit_circuit = ProviderCreditCircuit()


def looks_like_credit_error(exc: BaseException | str) -> bool:
    text = str(exc or "").lower()
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 402:
        return True
    return any(hint in text for hint in _CREDIT_HINTS)


def provider_from_config(provider: Optional[str]) -> str:
    return (provider or "").strip().lower()
