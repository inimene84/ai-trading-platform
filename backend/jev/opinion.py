"""Opinion-layer adapter. Jev stays out of the book unless it is enabled."""

from __future__ import annotations

import logging
from typing import Any

from backend.jev.config import jev_analysis_enabled, jev_replace_personas
from backend.jev.service import evaluate_symbol

logger = logging.getLogger(__name__)


def should_skip_personas(include_personas: bool, jev_status: str | None) -> bool:
    """Drop LLM personas only after Jev returned a validated evaluation.

    A missing key, timeout, or rejected schema leaves the existing agents in
    place. An uncertainty veto still counts as validated: there is an answer,
    and that answer is "no directional signal".
    """
    if not include_personas or not jev_replace_personas():
        return False
    return jev_status == "ok"


async def evaluate_opinion(
    symbol: str,
    bars: list,
    metrics: dict | None = None,
) -> dict[str, Any] | None:
    """Return a Jev result for the opinion layer, or None when Jev is off."""
    if not jev_analysis_enabled():
        return None
    try:
        result = await evaluate_symbol(
            symbol,
            bars=list(bars or []),
            metrics=metrics,
            fetch_bars=False,
        )
    except Exception as exc:
        logger.warning("Jev opinion path failed for %s: %s", symbol, exc)
        return None
    if result.get("status") != "ok":
        logger.info("Jev produced no signal for %s: %s", symbol, result.get("reason"))
        return None
    return result
