"""Read-only Jev evaluation endpoint.

GET /jev/evaluate is advisory. It does not place orders. Admin auth matches
the rest of the sensitive API surface.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from backend.jev.calibration import calibration_report
from backend.jev.evidence import gather_evidence
from backend.jev.journal import get_journal, reset_journal_cache
from backend.jev.research import classify_research, signing_key_path
from backend.jev.revalue import revalue_tape
from backend.jev.service import evaluate_symbol
from backend.jev.trust import replay_hysteresis, sign_jsonl, trust_report
from backend.security import validate_admin_request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jev", tags=["jev"])
crypto_router = APIRouter(prefix="/signals/jev", tags=["jev-signals"])


class RevalueBody(BaseModel):
    symbols: list[str] = Field(..., min_length=1, max_length=8)
    loss_frac: float = Field(default=0.0, ge=-1.0, le=1.0)
    gross_frac: float = Field(default=0.0, ge=0.0, le=5.0)


@router.post("/revalue")
async def revalue(body: RevalueBody, request: Request):
    """Fast tape revalue. Advisory only: no social pull, no order, no sizing."""
    validate_admin_request(request)
    logger.info("Jev revalue requested for %s", ",".join(body.symbols))
    return await revalue_tape(
        body.symbols,
        loss_frac=body.loss_frac,
        gross_frac=body.gross_frac,
        fetch_bars=True,
    )


@router.get("/evaluate")
async def evaluate(
    request: Request,
    symbol: str = Query(..., min_length=1, max_length=32, description="Asset or pair, e.g. BTC or BTCUSDT"),
    include_social: bool = Query(False, description="Pull cached social posts when a TwitterAPI key is set"),
):
    """Return trade stance, sentiment, squeeze risk, catalyst, and direction probabilities."""
    validate_admin_request(request)
    logger.info("Jev evaluate requested for %s social=%s", symbol, include_social)
    return await evaluate_symbol(symbol, include_social=include_social, fetch_bars=True)


class OutcomeBody(BaseModel):
    outcome: int = Field(..., ge=-1, le=1)
    horizon: str = Field(default="t+1", max_length=32)


@router.get("/journal")
async def journal(
    request: Request,
    symbol: str | None = Query(default=None, max_length=32),
    limit: int = Query(default=50, ge=1, le=200),
):
    """Recent Jev decisions with state hashes. Does not place orders."""
    validate_admin_request(request)
    reset_journal_cache()
    return {"decisions": get_journal().recent(symbol=symbol, limit=limit)}


@router.post("/journal/{decision_id}/outcome")
async def label_outcome(decision_id: int, body: OutcomeBody, request: Request):
    """Store a later triple-barrier or direction label. Does not size a trade."""
    validate_admin_request(request)
    reset_journal_cache()
    updated = get_journal().label(decision_id, body.outcome, body.horizon)
    if not updated:
        raise HTTPException(status_code=404, detail="decision not found")
    return {"decision_id": decision_id, "outcome": body.outcome, "sizing_allowed": False}


@router.get("/calibration")
async def calibration(request: Request):
    """Whether enough labels exist to display a calibrated probability."""
    validate_admin_request(request)
    reset_journal_cache()
    report = calibration_report(get_journal().labeled_choices())
    report["sizing_allowed"] = False
    return report


@router.post("/evidence")
async def evidence(
    request: Request,
    symbol: str = Query(..., min_length=1, max_length=32),
):
    """Research evidence by category. Empty evidence is a 404, not a buy."""
    validate_admin_request(request)
    started = time.perf_counter()
    gathered = await gather_evidence(symbol)
    evidence_ms = (time.perf_counter() - started) * 1000.0
    if gathered["empty"]:
        raise HTTPException(status_code=404, detail="no evidence in any category")
    classified = await classify_research(symbol, gathered["categories"])
    decision_ms = float(classified.get("decision_ms") or 0.0)
    return {
        "symbol": classified["symbol"],
        "decision_engine": "jev" if classified.get("status") == "ok" else "research",
        "advisory": True,
        "influence_book": False,
        "sizing_allowed": False,
        "order_size_fraction": 0.0,
        "evidence": gathered["categories"],
        "passages": classified.get("passages") or [],
        "classification": classified.get("classification"),
        "status": classified.get("status"),
        "reason": classified.get("reason"),
        "timing": {
            "evidence_ms": round(evidence_ms, 2),
            "decision_ms": round(decision_ms, 2),
            "total_ms": round(evidence_ms + decision_ms, 2),
        },
    }


class ReplayBody(BaseModel):
    scores: list[float] = Field(..., min_length=1, max_length=500)
    enter: float = Field(..., ge=0.0, le=1.0)
    exit_below: float = Field(..., ge=0.0, le=1.0)


@router.get("/trust")
async def trust(request: Request):
    """Brier, ECE, and a research verdict. Does not enable sizing."""
    validate_admin_request(request)
    reset_journal_cache()
    report = trust_report(get_journal().labeled_choices())
    report["sizing_allowed"] = False
    report["export_signing"] = sign_jsonl("trust-report", signing_key_path()) if signing_key_path() else {"signed": False}
    return report


@router.post("/replay")
async def replay(body: ReplayBody, request: Request):
    """Replay a threshold band over recorded scores without calling Jev."""
    validate_admin_request(request)
    try:
        path = replay_hysteresis(body.scores, body.enter, body.exit_below)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"path": path, "called_jev": False, "sizing_allowed": False}


@crypto_router.get("/crypto")
async def crypto_sentiment(
    request: Request,
    symbol: str = Query(..., min_length=1, max_length=32),
    sample_size: int = Query(default=100, ge=1, le=1000),
    include_social: bool = Query(default=False),
):
    """Read-only crypto sentiment evaluation. Social pulls stay opt-in."""
    validate_admin_request(request)
    return await evaluate_symbol(
        symbol,
        include_social=include_social,
        fetch_bars=True,
        social_sample=sample_size,
    )
