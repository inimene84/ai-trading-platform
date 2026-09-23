"""Phase 2 JEV ingest / evaluate / orchestrate routes.

Mounted under the existing /jev and /api/jev prefixes. Collect aliases live
under /data so n8n scanners can POST zip-shaped payloads without a second app.
All endpoints require the same admin auth as legacy /api/jev.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database.connection import get_db
from backend.database.models import JevMarketScan
from backend.jev.config import jev_execution_mode, jev_live_execution_allowed, jev_may_call_execute
from backend.security import validate_admin_request
from backend.services.jev_pipeline import (
    calibration_snapshot,
    evaluate_and_log,
    ingest_market_scan,
    ingest_news_sentiment,
    orchestrate_scan,
    recent_news_for_symbol,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jev", tags=["jev-pipeline"])
ingest_alias_router = APIRouter(prefix="/data", tags=["jev-ingest"])


class MarketScanIn(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=32)
    timeframe: str = Field(default="M5", max_length=10)
    signal_direction: str | None = Field(default=None, max_length=10)
    confidence: float | None = Field(default=None, ge=-1.0, le=1.0)
    signal_strength: float | None = None
    indicators: dict[str, Any] = Field(default_factory=dict)
    price_data: dict[str, Any] | None = None
    volume_data: dict[str, Any] | None = None
    scan_source: str = Field(default="n8n_market_scanner", max_length=50)
    scan_timestamp: str | None = None


class NewsSentimentIn(BaseModel):
    source: str = Field(..., min_length=1, max_length=50)
    headline: str | None = None
    url: str | None = None
    published_at: str | None = None
    sentiment_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    sentiment_label: str | None = Field(default=None, max_length=20)
    impact_rating: str | None = Field(default=None, max_length=10)
    categories: dict[str, Any] | list[Any] | None = None
    entities: dict[str, Any] | list[Any] | None = None
    full_data: dict[str, Any] | None = None
    market_scan_id: int | None = None


class PipelineEvaluateIn(BaseModel):
    symbol: str | None = Field(default=None, max_length=32)
    timeframe: str = Field(default="1H", max_length=10)
    scan_id: int | None = None
    signal_direction: str | None = Field(default=None, max_length=10)
    confidence: float | None = Field(default=None, ge=-1.0, le=1.0)
    market_context: list[dict[str, Any]] | None = None
    news_sentiment_context: list[dict[str, Any]] | None = None


def _require_admin(request: Request) -> None:
    validate_admin_request(request)


@router.get("/pipeline/status")
async def pipeline_status(request: Request):
    _require_admin(request)
    mode = jev_execution_mode()
    return {
        "execution_mode": mode,
        "may_call_execute": jev_may_call_execute(),
        "live_execution_allowed": jev_live_execution_allowed(),
        "sizing_allowed": False,
        "note": (
            "Default is off. Enable shadow to record hypothetical fills. "
            "Enable paper only after reviewing risk gates. "
            "Do not set JEV_EXECUTION_MODE=live without owner confirm — this PR blocks live."
        ),
    }


@router.post("/ingest/market")
async def ingest_market(body: MarketScanIn, request: Request, db: Session = Depends(get_db)):
    _require_admin(request)
    try:
        row = ingest_market_scan(db, body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "success", "id": row.id, "symbol": row.symbol, "timeframe": row.timeframe}


@router.post("/ingest/news")
async def ingest_news(body: NewsSentimentIn, request: Request, db: Session = Depends(get_db)):
    _require_admin(request)
    try:
        row = ingest_news_sentiment(db, body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "success", "id": row.id, "source": row.source}


@ingest_alias_router.post("/collect-market")
async def collect_market_alias(body: MarketScanIn, request: Request, db: Session = Depends(get_db)):
    return await ingest_market(body, request, db)


@ingest_alias_router.post("/collect-news")
async def collect_news_alias(body: NewsSentimentIn, request: Request, db: Session = Depends(get_db)):
    return await ingest_news(body, request, db)


@router.post("/pipeline/evaluate")
async def pipeline_evaluate(body: PipelineEvaluateIn, request: Request, db: Session = Depends(get_db)):
    _require_admin(request)
    scan: JevMarketScan | None = None
    if body.scan_id is not None:
        scan = db.query(JevMarketScan).filter(JevMarketScan.id == body.scan_id).first()
        if scan is None:
            raise HTTPException(status_code=404, detail="scan not found")
    symbol = (body.symbol or (scan.symbol if scan else "")).strip()
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol or scan_id is required")
    logger.info("JEV pipeline evaluate requested for %s mode=%s", symbol, jev_execution_mode())
    return await evaluate_and_log(
        db,
        symbol=symbol,
        timeframe=body.timeframe or (scan.timeframe if scan else "1H"),
        source_scan=scan,
        market_context=body.market_context,
        news_context=body.news_sentiment_context,
        fallback_direction=body.signal_direction or (scan.signal_direction if scan else None),
        fallback_confidence=body.confidence if body.confidence is not None else (scan.confidence if scan else None),
    )


@router.post("/pipeline/orchestrate")
async def pipeline_orchestrate(
    request: Request,
    limit: int = Query(default=1, ge=1, le=5),
    db: Session = Depends(get_db),
):
    _require_admin(request)
    logger.info("JEV pipeline orchestrate requested limit=%s mode=%s", limit, jev_execution_mode())
    return await orchestrate_scan(db, limit=limit)


@router.get("/pipeline/calibration")
async def pipeline_calibration(request: Request, db: Session = Depends(get_db)):
    _require_admin(request)
    return calibration_snapshot(db)


@router.get("/pipeline/news/{symbol}")
async def pipeline_news(
    symbol: str,
    request: Request,
    days: int = Query(default=3, ge=1, le=14),
    db: Session = Depends(get_db),
):
    _require_admin(request)
    rows = recent_news_for_symbol(db, symbol, days=days, limit=40)
    return [
        {
            "headline": row.headline,
            "sentiment_score": row.sentiment_score,
            "impact_rating": row.impact_rating,
            "published_at": row.published_at.isoformat() if row.published_at else None,
            "source": row.source,
        }
        for row in rows
    ]
