"""
Desk UI Live Read Model API (WP5)
=================================
Authoritative, read-only surface projecting backend trading engine state,
open positions, fills tape, live decisions, and outbox delivery health directly to Desk UI.

Endpoints:
  GET /desk/status
  GET /desk/positions
  GET /desk/tape
  GET /desk/decisions
  GET /desk/opinions
  GET /desk/outbox-health
(Also dual-mounted under /api/desk/...)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.database.connection import get_db
from backend.database.models import (
    DecisionOpinion,
    DecisionRecord,
    IngestionOutbox,
    Trade,
)
from backend.routes.control_plane import _get_control_plane_status_data
from backend.services.trading_mode import get_trading_mode

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/desk", tags=["desk-ui"])


class DeskPositionModel(BaseModel):
    id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    mark_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    opened_at: Optional[str] = None
    broker: str = "binance"
    mode: str = "shadow"


class DeskTapeFillModel(BaseModel):
    id: str
    ts: int
    symbol: str
    side: str
    qty: float
    entry_price: float
    exit_price: Optional[float] = None
    realized_pnl: float = 0.0
    closed_at: Optional[str] = None


class DeskDecisionModel(BaseModel):
    decision_id: str
    created_at: str
    symbol: str
    timeframe: str
    baseline_action: str
    baseline_confidence: float
    with_jev_action: str
    with_jev_confidence: float
    divergence: str
    jev_signal: Optional[str] = None
    jev_confidence: Optional[float] = None
    jev_vetoed: bool = False
    execution_mode: str = "shadow"
    executed: bool = False


class DeskOutboxHealthModel(BaseModel):
    pending_count: int
    processing_count: int
    completed_count: int
    failed_count: int
    oldest_pending_age_sec: float
    healthy: bool


@router.get("/status")
def get_desk_status(db: Session = Depends(get_db)):
    """Return effective trading state, safety telemetry, and high-level book stats."""
    status_data = _get_control_plane_status_data(db)
    open_count = db.query(Trade).filter(Trade.status == "open").count()
    return {
        **status_data,
        "open_positions_count": open_count,
        "is_read_only": True,
    }


@router.get("/positions", response_model=List[DeskPositionModel])
def get_desk_positions(db: Session = Depends(get_db)):
    """Authoritative ledger open positions."""
    trades = (
        db.query(Trade)
        .filter(Trade.status == "open")
        .order_by(Trade.timestamp.desc())
        .limit(100)
        .all()
    )

    out = []
    for t in trades:
        entry_px = float(t.entry_price or 0.0)
        cur_px = float(t.exit_price or entry_px)
        pnl_val = float(t.pnl or 0.0)
        out.append(
            DeskPositionModel(
                id=str(t.id),
                symbol=str(t.symbol or ""),
                side=str(t.direction or "BUY").upper(),
                quantity=float(t.quantity or 0.0),
                entry_price=entry_px,
                mark_price=cur_px,
                stop_loss=float(t.stop_loss) if t.stop_loss is not None else None,
                take_profit=float(t.take_profit) if t.take_profit is not None else None,
                unrealized_pnl=pnl_val,
                opened_at=t.timestamp.isoformat() if t.timestamp else None,
                broker=getattr(t, "broker", "binance") or "binance",
                mode=str(get_trading_mode().value if hasattr(get_trading_mode(), "value") else get_trading_mode()),
            )
        )
    return out


@router.get("/tape", response_model=List[DeskTapeFillModel])
def get_desk_tape(
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Authoritative closed trades / fills tape."""
    trades = (
        db.query(Trade)
        .filter(Trade.status == "closed")
        .order_by(Trade.closed_at.desc())
        .limit(limit)
        .all()
    )

    out = []
    for t in trades:
        ts_ms = int(t.closed_at.timestamp() * 1000) if t.closed_at else int(datetime.now(timezone.utc).timestamp() * 1000)
        out.append(
            DeskTapeFillModel(
                id=str(t.id),
                ts=ts_ms,
                symbol=str(t.symbol or ""),
                side=str(t.direction or "BUY").upper(),
                qty=float(t.quantity or 0.0),
                entry_price=float(t.entry_price or 0.0),
                exit_price=float(t.exit_price or 0.0) if t.exit_price is not None else None,
                realized_pnl=float(t.pnl or 0.0),
                closed_at=t.closed_at.isoformat() if t.closed_at else None,
            )
        )
    return out


@router.get("/decisions", response_model=List[DeskDecisionModel])
def get_desk_decisions(
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Recent dual-decision traces (baseline vs with-JEV)."""
    records = (
        db.query(DecisionRecord)
        .order_by(DecisionRecord.created_at.desc())
        .limit(limit)
        .all()
    )

    out = []
    for r in records:
        out.append(
            DeskDecisionModel(
                decision_id=r.decision_id,
                created_at=r.created_at.isoformat() if r.created_at else "",
                symbol=r.symbol,
                timeframe=r.timeframe,
                baseline_action=r.baseline_action,
                baseline_confidence=float(r.baseline_confidence or 0.0),
                with_jev_action=r.with_jev_action,
                with_jev_confidence=float(r.with_jev_confidence or 0.0),
                divergence=r.divergence,
                jev_signal=r.jev_signal,
                jev_confidence=float(r.jev_confidence or 0.0) if r.jev_confidence is not None else None,
                jev_vetoed=bool(r.jev_vetoed),
                execution_mode=r.execution_mode,
                executed=bool(r.executed),
            )
        )
    return out


@router.get("/opinions")
def get_desk_opinions(
    symbol: Optional[str] = None,
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Latest advisory opinions from JEV and sub-analysts."""
    query = db.query(DecisionOpinion)
    if symbol:
        query = query.filter(DecisionOpinion.symbol == symbol.strip().upper())
    opinions = query.order_by(DecisionOpinion.created_at.desc()).limit(limit).all()

    return [
        {
            "opinion_id": o.opinion_id,
            "created_at": o.created_at.isoformat() if o.created_at else "",
            "symbol": o.symbol,
            "source": o.source,
            "signal": o.signal,
            "confidence": o.confidence,
            "reasoning": o.reasoning,
            "vetoed": o.vetoed,
            "conflict": o.conflict,
        }
        for o in opinions
    ]


@router.get("/outbox-health", response_model=DeskOutboxHealthModel)
def get_outbox_health(db: Session = Depends(get_db)):
    """Health and backlog telemetry of the transactional outbox queue."""
    counts = dict(
        db.query(IngestionOutbox.status, func.count(IngestionOutbox.id))
        .group_by(IngestionOutbox.status)
        .all()
    )

    pending_cnt = counts.get("pending", 0)
    processing_cnt = counts.get("processing", 0)
    completed_cnt = counts.get("completed", 0)
    failed_cnt = counts.get("dead", 0) + counts.get("failed", 0)

    oldest_pending = (
        db.query(IngestionOutbox.created_at)
        .filter(IngestionOutbox.status == "pending")
        .order_by(IngestionOutbox.created_at.asc())
        .first()
    )

    age_sec = 0.0
    if oldest_pending and oldest_pending[0]:
        now = datetime.now(timezone.utc)
        dt = oldest_pending[0]
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_sec = max(0.0, (now - dt).total_seconds())

    healthy = failed_cnt == 0 and age_sec < 300.0

    return DeskOutboxHealthModel(
        pending_count=pending_cnt,
        processing_count=processing_cnt,
        completed_count=completed_cnt,
        failed_count=failed_cnt,
        oldest_pending_age_sec=round(age_sec, 2),
        healthy=healthy,
    )
