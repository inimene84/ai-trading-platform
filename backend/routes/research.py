"""Research Plane API (read-only) for QuantumTrade Pro.

Exposes search and retrieval endpoints for:
- OpenSearch, SearXNG, and MT5 health status
- Scrapling fetch proxy
- News hits (qt-news)
- Economic calendar events (qt-events)
- Trading decisions and promotion verdicts (qt-decisions)
- MetaTrader 5 closed bars (qt-mt-bars)

All data endpoints are protected by admin authentication (validate_admin_request).
All queries fail soft, returning [] + HTTP 200 when downstream search engines are offline.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Body, HTTPException, Query, Request
from pydantic import BaseModel, Field

from backend.security import validate_admin_request
from backend.services.opensearch_client import opensearch_client
from backend.services.searxng_client import searxng_client
from backend.services.mt_bridge_client import mt_bridge_client
from scrapling_sidecar.url_guard import is_public_http_url

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["research"])

SCRAPLING_RESEARCH_URL = os.getenv("SCRAPLING_RESEARCH_URL", "http://ai-trading-scrapling:8080").rstrip("/")
ADMIN_API_KEY = (os.getenv("ADMIN_API_KEY") or "").strip()


class ResearchFetchRequest(BaseModel):
    url: str = Field(..., min_length=8, max_length=2000)
    mode: str = Field(default="http", description="http or stealth")
    extraction_type: str = Field(default="markdown", description="markdown, html, or text")
    css_selector: Optional[str] = None


def _sidecar_headers() -> Dict[str, str]:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if ADMIN_API_KEY:
        headers["Authorization"] = f"Bearer {ADMIN_API_KEY}"
        headers["X-API-Key"] = ADMIN_API_KEY
    return headers


@router.get("/health")
async def research_health() -> Dict[str, Any]:
    """Check connectivity to Research Plane services (OpenSearch, SearXNG, MT5 bridge)."""
    os_ok = await opensearch_client.ping(timeout=2.0)
    searx_ok = await searxng_client.ping(timeout=2.0)
    mt_ok = await mt_bridge_client.ping(timeout=2.0)

    # Optional check on Scrapling
    scrapling_ok = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            res = await client.get(f"{SCRAPLING_RESEARCH_URL}/health")
            scrapling_ok = res.status_code == 200
    except Exception:
        pass

    return {
        "status": "ok" if (os_ok or searx_ok or mt_ok) else "down",
        "opensearch": os_ok,
        "searxng": searx_ok,
        "mt5_bridge": mt_ok,
        "scrapling": scrapling_ok,
    }


@router.post("/fetch")
async def research_fetch(request: Request, payload: ResearchFetchRequest = Body(...)) -> Dict[str, Any]:
    """Fetch a public page through Scrapling and return extracted content."""
    validate_admin_request(request)
    if not is_public_http_url(payload.url):
        raise HTTPException(status_code=400, detail="url must be a public http(s) host")
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                f"{SCRAPLING_RESEARCH_URL}/fetch",
                json=payload.model_dump(),
                headers=_sidecar_headers(),
            )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Scrapling sidecar unreachable: {exc}") from exc

    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail="fetch failed")
    return response.json()


@router.get("/news")
async def get_research_news(
    request: Request,
    q: Optional[str] = Query(None, description="Keyword search in title or snippet"),
    symbol: Optional[str] = Query(None, description="Filter by symbol (e.g. BTCUSDC)"),
    from_: Optional[str] = Query(None, alias="from", description="ISO datetime start filter"),
    limit: int = Query(50, ge=1, le=500),
) -> List[Dict[str, Any]]:
    """Retrieve news articles from qt-news."""
    validate_admin_request(request)

    must_clauses: List[Dict[str, Any]] = []
    if symbol:
        must_clauses.append({"term": {"symbol": symbol.upper()}})
    if q:
        must_clauses.append({"multi_match": {"query": q, "fields": ["title", "snippet"]}})
    if from_:
        must_clauses.append({"range": {"ts": {"gte": from_}}})

    query_body: Dict[str, Any] = {
        "size": limit,
        "sort": [{"ts": {"order": "desc"}}],
    }
    if must_clauses:
        query_body["query"] = {"bool": {"must": must_clauses}}
    else:
        query_body["query"] = {"match_all": {}}

    return await opensearch_client.search("qt-news", query_body)


@router.get("/events")
async def get_research_events(
    request: Request,
    currency: Optional[str] = Query(None, description="Filter by currency (e.g. USD, EUR)"),
    from_: Optional[str] = Query(None, alias="from", description="ISO datetime start filter"),
    limit: int = Query(50, ge=1, le=500),
) -> List[Dict[str, Any]]:
    """Retrieve macro economic calendar events from qt-events."""
    validate_admin_request(request)

    must_clauses: List[Dict[str, Any]] = []
    if currency:
        must_clauses.append({"term": {"currency": currency.upper()}})
    if from_:
        must_clauses.append({"range": {"ts": {"gte": from_}}})

    query_body: Dict[str, Any] = {
        "size": limit,
        "sort": [{"ts": {"order": "desc"}}],
    }
    if must_clauses:
        query_body["query"] = {"bool": {"must": must_clauses}}
    else:
        query_body["query"] = {"match_all": {}}

    return await opensearch_client.search("qt-events", query_body)


@router.get("/decisions")
async def get_research_decisions(
    request: Request,
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    verdict: Optional[str] = Query(None, description="Filter by promotion_verdict or signal"),
    from_: Optional[str] = Query(None, alias="from", description="ISO datetime start filter"),
    limit: int = Query(50, ge=1, le=500),
) -> List[Dict[str, Any]]:
    """Retrieve decision and veto logs from qt-decisions."""
    validate_admin_request(request)

    must_clauses: List[Dict[str, Any]] = []
    if symbol:
        must_clauses.append({"term": {"symbol": symbol.upper()}})
    if verdict:
        v_upper = verdict.upper()
        must_clauses.append({
            "bool": {
                "should": [
                    {"term": {"promotion_verdict": v_upper}},
                    {"term": {"signal": v_upper}},
                ]
            }
        })
    if from_:
        must_clauses.append({"range": {"ts": {"gte": from_}}})

    query_body: Dict[str, Any] = {
        "size": limit,
        "sort": [{"ts": {"order": "desc"}}],
    }
    if must_clauses:
        query_body["query"] = {"bool": {"must": must_clauses}}
    else:
        query_body["query"] = {"match_all": {}}

    return await opensearch_client.search("qt-decisions", query_body)


@router.get("/mt/bars")
async def get_research_mt_bars(
    request: Request,
    symbol: Optional[str] = Query(None, description="Filter by symbol (e.g. XAUUSD)"),
    tf: Optional[str] = Query(None, description="Timeframe (e.g. H1, M15)"),
    from_: Optional[str] = Query(None, alias="from", description="ISO datetime start filter"),
    limit: int = Query(100, ge=1, le=1000),
) -> List[Dict[str, Any]]:
    """Retrieve MetaTrader bars from qt-mt-bars."""
    validate_admin_request(request)

    must_clauses: List[Dict[str, Any]] = []
    if symbol:
        must_clauses.append({"term": {"symbol": symbol.upper()}})
    if tf:
        must_clauses.append({"term": {"tf": tf.upper()}})
    if from_:
        must_clauses.append({"range": {"ts": {"gte": from_}}})

    query_body: Dict[str, Any] = {
        "size": limit,
        "sort": [{"ts": {"order": "desc"}}],
    }
    if must_clauses:
        query_body["query"] = {"bool": {"must": must_clauses}}
    else:
        query_body["query"] = {"match_all": {}}

    return await opensearch_client.search("qt-mt-bars", query_body)
