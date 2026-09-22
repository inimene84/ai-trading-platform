"""Five-category evidence gatherer.

A failed category becomes an empty list. The request fails closed only when
every category is empty, which the route surfaces as 404. This module does
not place orders and does not invent evidence when search is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from backend.services.searxng_client import searxng_client

logger = logging.getLogger(__name__)

EVIDENCE_CATEGORIES = (
    "market_data",
    "company_news",
    "industry_news",
    "analyst_views",
    "macro_risks",
)

_QUERY = {
    "market_data": "{symbol} price volume returns",
    "company_news": "{symbol} company news",
    "industry_news": "{symbol} industry sector news",
    "analyst_views": "{symbol} analyst rating",
    "macro_risks": "{symbol} macro risk rates inflation",
}

SearchFn = Callable[[str], Awaitable[list[dict[str, Any]]]]


def _clip_item(raw: dict[str, Any], credibility: float) -> dict[str, Any]:
    snippet = str(raw.get("snippet") or raw.get("content") or raw.get("title") or "")[:900]
    return {
        "title": str(raw.get("title") or "")[:240],
        "url": str(raw.get("url") or ""),
        "source": str(raw.get("source") or raw.get("engine") or "searxng"),
        "date": str(raw.get("date") or raw.get("publishedDate") or ""),
        "snippet": snippet,
        "source_credibility": credibility,
        "untrusted_text": True,
    }


def _fingerprint(item: dict[str, Any]) -> str:
    text = " ".join(str(item.get("snippet") or "").lower().split())[:80]
    return text


async def _searxng_search(query: str) -> list[dict[str, Any]]:
    try:
        rows = await searxng_client.search(query, categories="news")
    except Exception as exc:
        logger.warning("Evidence search failed: %s", exc)
        return []
    return list(rows or [])


async def gather_evidence(symbol: str, search: SearchFn | None = None) -> dict[str, Any]:
    finder = search or _searxng_search
    buckets: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for category in EVIDENCE_CATEGORIES:
        query = _QUERY[category].format(symbol=symbol)
        try:
            rows = await finder(query)
        except Exception as exc:
            logger.warning("Evidence category %s failed: %s", category, exc)
            rows = []
        items: list[dict[str, Any]] = []
        for raw in rows or []:
            if not isinstance(raw, dict):
                continue
            item = _clip_item(raw, 0.5)
            fingerprint = _fingerprint(item)
            if not fingerprint or fingerprint in seen:
                continue
            seen.add(fingerprint)
            items.append(item)
            if len(items) >= 5:
                break
        buckets[category] = items
    filled = sum(1 for items in buckets.values() if items)
    return {
        "symbol": symbol,
        "categories": buckets,
        "filled_categories": filled,
        "empty": filled == 0,
    }
