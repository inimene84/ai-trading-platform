"""Batch LLM sentiment classifier for the QuantumTrade Pro Research Plane.

Classifies bundles of headlines on a timer to enrich OpenSearch qt-news documents.
Fail-soft: timeouts or parse failures default to 'unclear' with should_block_entry=False.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import List, Literal
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def is_research_sentiment_enabled() -> bool:
    """Check if periodic LLM sentiment classification is enabled.

    Default is False to prevent background news classification from competing
    with live trading persona generation budgets on OmniRoute/LiteLLM.
    """
    return os.getenv("RESEARCH_SENTIMENT_ENABLED", "false").strip().lower() in ("true", "1", "yes")


def get_sentiment_timeout_sec() -> float:
    try:
        return float(os.getenv("RESEARCH_SENTIMENT_TIMEOUT_SEC", "10.0"))
    except ValueError:
        return 10.0


class SentimentLabel(BaseModel):
    symbol: str = "UNKNOWN"
    horizon: str = "intraday"
    stance: Literal["bullish", "bearish", "neutral", "unclear"] = "unclear"
    event_type: Literal["etf", "hack", "macro", "listing", "none"] = "none"
    severity: int = Field(default=0, ge=0, le=5)
    confidence: int = Field(default=0, ge=0, le=100)
    should_block_entry: bool = False
    summary: str = "No analysis available"


def create_default_sentiment(
    symbol: str, summary: str = "Analysis unavailable or timed out"
) -> SentimentLabel:
    return SentimentLabel(
        symbol=symbol,
        horizon="intraday",
        stance="unclear",
        event_type="none",
        severity=0,
        confidence=0,
        should_block_entry=False,
        summary=summary,
    )


async def classify_bundle(symbol: str, headlines: List[str]) -> SentimentLabel:
    """Classify a cluster of news headlines for a symbol.

    Fail-soft: catches all exceptions and returns default unclear.
    Disabled by default if RESEARCH_SENTIMENT_ENABLED is not set to true.
    """
    if not is_research_sentiment_enabled():
        return create_default_sentiment(symbol, "LLM sentiment labeling disabled by configuration")

    if not headlines:
        return create_default_sentiment(symbol, "No headlines provided")

    # Filter empty lines
    cleaned_headlines = [h.strip() for h in headlines if h and h.strip()]
    if not cleaned_headlines:
        return create_default_sentiment(symbol, "Empty headlines")

    prompt = (
        f"You are a quantitative crypto market news analyst.\n"
        f"Analyze these recent headlines for {symbol} and classify market stance and risk:\n"
        + "\n".join(f"- {h}" for h in cleaned_headlines[:15])
        + "\n\nGuidelines:\n"
        "- Stance: bullish | bearish | neutral | unclear\n"
        "- Event type: etf | hack | macro | listing | none\n"
        "- Severity: 0 (benign) to 5 (critical catastrophe)\n"
        "- Confidence: 0 to 100\n"
        "- should_block_entry: true ONLY for catastrophic events (major exchange/bridge hack, trading halt).\n"
        "- summary: A single concise sentence summarizing the sentiment.\n"
    )

    try:
        from backend.llm.router import call_llm_resilient

        system_prompt = (
            "You output one JSON object matching the schema: "
            '{"symbol": str, "horizon": str, "stance": "bullish"|"bearish"|"neutral"|"unclear", '
            '"event_type": "etf"|"hack"|"macro"|"listing"|"none", "severity": int(0..5), '
            '"confidence": int(0..100), "should_block_entry": bool, "summary": str}. '
            "Output JSON only."
        )

        timeout_sec = get_sentiment_timeout_sec()
        res_str = await asyncio.wait_for(
            call_llm_resilient(
                task_type="research",
                prompt=prompt,
                system=system_prompt,
                temperature=0.1,
                max_tokens=200,
                response_json=True,
            ),
            timeout=timeout_sec,
        )

        data = json.loads(res_str)
        if isinstance(data, dict):
            # Enforce symbol and valid field defaults
            data.setdefault("symbol", symbol)
            return SentimentLabel(**data)
        return create_default_sentiment(symbol, "Invalid JSON structure from LLM")
    except Exception as exc:
        logger.warning(f"LLM sentiment classification failed for {symbol}: {exc}")
        return create_default_sentiment(symbol, f"Classification failed: {exc}")
