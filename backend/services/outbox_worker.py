"""
Outbox Consumer & JEV Shadow Decision Trace Worker (WP2)
========================================================
Consumes canonical `market.scan.received` events from `ingestion_outbox`,
evaluates observations through both the baseline engine and the JEV opinion adapter,
records dual-decision traces (baseline vs with-JEV) in `decision_records`,
and guarantees zero autonomous broker order execution in shadow mode.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session

from backend.database.models import DecisionOpinion, DecisionRecord, IngestionOutbox
from backend.jev.config import jev_analysis_enabled
from backend.jev.service import evaluate_symbol
from backend.services.jev_pipeline import normalize_symbol
from backend.services.trading_mode import get_trading_mode, is_shadow_mode

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def classify_divergence(baseline_action: str, with_jev_action: str) -> str:
    """
    Classify the divergence between baseline trading decision and counterfactual (with-JEV) decision.
    Values: AGREED, UPGRADED, DOWNGRADED, FLIPPED, OPPOSED
    """
    base = (baseline_action or "HOLD").upper()
    with_j = (with_jev_action or "HOLD").upper()

    if base == with_j:
        return "AGREED"
    if base == "HOLD" and with_j in ("BUY", "SELL"):
        return "UPGRADED"
    if base in ("BUY", "SELL") and with_j == "HOLD":
        return "DOWNGRADED"
    if base in ("BUY", "SELL") and with_j in ("BUY", "SELL") and base != with_j:
        return "FLIPPED"
    return "OPPOSED"


async def evaluate_jev_for_observation(
    symbol: str,
    price: float,
    indicators: dict[str, Any],
    fallback_direction: Optional[str] = None,
    fallback_confidence: float = 0.5,
) -> dict[str, Any]:
    """
    Query JEV opinion adapter for a given market observation.
    Falls back gracefully if JEV evaluation fails or is disabled.
    """
    if not jev_analysis_enabled():
        return {
            "status": "disabled",
            "signal": "HOLD",
            "confidence": 0.0,
            "vetoed": False,
            "conflict": False,
            "reason": "JEV analysis disabled in config",
            "answers": {},
        }

    try:
        metrics = {
            k: indicators[k]
            for k in ("funding_rate", "open_interest", "oi_change_pct_1h", "rsi", "macd", "atr")
            if k in indicators
        }
        res = await evaluate_symbol(
            symbol=symbol,
            bars=None,
            metrics=metrics or None,
            fetch_bars=False,
            include_social=False,
        )
        if isinstance(res, dict) and res.get("status") == "ok":
            action = str(res.get("action") or res.get("signal") or "HOLD").upper()
            if action in ("BULLISH", "LONG", "STRONG_BUY"):
                action = "BUY"
            elif action in ("BEARISH", "SHORT", "STRONG_SELL"):
                action = "SELL"
            elif action not in ("BUY", "SELL"):
                action = "HOLD"

            return {
                "status": "ok",
                "signal": action,
                "confidence": float(res.get("confidence") or 0.0),
                "vetoed": bool(res.get("vetoed")),
                "conflict": bool(res.get("conflict")),
                "reason": str(res.get("reason") or ""),
                "answers": res.get("answers") or {},
                "raw": res,
            }
    except Exception as exc:
        logger.warning(f"[{symbol}] JEV evaluation failed in outbox worker: {exc}")

    # Fallback response
    return {
        "status": "fallback",
        "signal": (fallback_direction or "HOLD").upper(),
        "confidence": float(fallback_confidence or 0.0),
        "vetoed": False,
        "conflict": False,
        "reason": "Fallback opinion due to service unavailability or timeout",
        "answers": {},
    }


def claim_pending_events(
    db: Session,
    batch_size: int = 10,
    worker_id: str = "outbox_worker_default",
) -> list[IngestionOutbox]:
    """
    Claim PENDING events with row locking (SKIP LOCKED on PostgreSQL, standard on SQLite).
    """
    query = (
        db.query(IngestionOutbox)
        .filter(IngestionOutbox.status == "pending")
        .order_by(IngestionOutbox.created_at.asc())
        .limit(batch_size)
    )

    # Apply SKIP LOCKED if supported (PostgreSQL)
    bind = db.get_bind()
    dialect_name = getattr(bind.dialect, "name", "")
    if dialect_name == "postgresql":
        query = query.with_for_update(skip_locked=True)

    rows = query.all()
    now = utc_now()
    for row in rows:
        row.status = "processing"
        row.locked_at = now
        row.locked_by = worker_id
        row.attempt_count += 1
    db.commit()
    return rows


async def process_outbox_event(
    db: Session,
    event: IngestionOutbox,
    jev_override_eval: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Process a single outbox event.
    Parses canonical MarketScan, processes observations, builds dual decisions,
    stores records, and transitions event status.
    """
    if event.event_type != "market.scan.received":
        logger.info(f"Ignoring non-market scan event: {event.event_type} (id={event.event_id})")
        event.status = "completed"
        event.processed_at = utc_now()
        db.commit()
        return {"status": "skipped", "event_id": event.event_id}

    payload = event.payload or {}
    scan_id = payload.get("scan_id", "")
    trace_id = payload.get("trace_id")
    observations = payload.get("observations") or []

    global_mode = get_trading_mode()
    mode_str = global_mode.value if hasattr(global_mode, "value") else str(global_mode)
    is_shadow = is_shadow_mode()

    decisions_created = []

    for obs in observations:
        instrument = obs.get("instrument") or {}
        raw_symbol = instrument.get("symbol", "UNKNOWN")
        symbol = normalize_symbol(raw_symbol)
        venue = instrument.get("venue", "binance_futures")
        timeframe = obs.get("timeframe", "M5")
        observed_price = float(obs.get("price") or 0.0)
        signal_dir = obs.get("signal_direction")
        obs_confidence = float(obs.get("confidence") or 0.5)
        indicators = obs.get("indicators") or {}

        # 1. Baseline Decision (Canonical Engine baseline without JEV)
        if signal_dir in ("BUY", "LONG", "STRONG_BUY"):
            baseline_action = "BUY"
        elif signal_dir in ("SELL", "SHORT", "STRONG_SELL"):
            baseline_action = "SELL"
        else:
            baseline_action = "HOLD"

        baseline_conf = obs_confidence
        baseline_reason = f"Baseline scanner input: {signal_dir} (conf={baseline_conf:.2f})"

        # 2. JEV Advisory Opinion
        if jev_override_eval is not None:
            jev_eval = jev_override_eval
        else:
            jev_eval = await evaluate_jev_for_observation(
                symbol=symbol,
                price=observed_price,
                indicators=indicators,
                fallback_direction=signal_dir,
                fallback_confidence=obs_confidence,
            )

        jev_sig = jev_eval.get("signal", "HOLD")
        jev_conf = float(jev_eval.get("confidence") or 0.0)
        jev_veto = bool(jev_eval.get("vetoed"))
        jev_conflict = bool(jev_eval.get("conflict"))
        jev_reason_text = str(jev_eval.get("reason") or "")

        # Record DecisionOpinion
        opinion_id = f"op_{uuid.uuid4().hex[:12]}"
        opinion_row = DecisionOpinion(
            opinion_id=opinion_id,
            trace_id=trace_id,
            scan_id=scan_id,
            symbol=symbol,
            source="jev_analyst",
            signal=jev_sig,
            confidence=jev_conf,
            reasoning=jev_reason_text,
            vetoed=jev_veto,
            conflict=jev_conflict,
            raw_payload=jev_eval,
        )
        db.add(opinion_row)

        # 3. Counterfactual Decision (Canonical Engine WITH JEV opinion)
        if jev_veto:
            with_jev_action = "HOLD"
            with_jev_conf = 0.0
            with_jev_reason = f"Vetoed by JEV advisor: {jev_reason_text}"
        elif jev_conflict:
            with_jev_action = "HOLD"
            with_jev_conf = 0.0
            with_jev_reason = f"Conflict flagged by JEV advisor: {jev_reason_text}"
        elif jev_sig in ("BUY", "SELL") and jev_sig == baseline_action:
            # Mutual confirmation -> confidence boost
            with_jev_action = baseline_action
            with_jev_conf = min(1.0, (baseline_conf * 0.55) + (jev_conf * 0.45))
            with_jev_reason = f"JEV confirmed {baseline_action} (boosted conf={with_jev_conf:.2f})"
        elif jev_sig in ("BUY", "SELL") and baseline_action in ("BUY", "SELL") and jev_sig != baseline_action:
            # Opposing signals -> fail closed to HOLD
            with_jev_action = "HOLD"
            with_jev_conf = 0.0
            with_jev_reason = f"Signal opposition: baseline={baseline_action} vs jev={jev_sig}"
        elif baseline_action == "HOLD" and jev_sig in ("BUY", "SELL") and jev_conf >= 0.70:
            # Upgrade from neutral if JEV confidence clears high bar
            with_jev_action = jev_sig
            with_jev_conf = jev_conf
            with_jev_reason = f"Upgraded to {jev_sig} by high-confidence JEV opinion ({jev_conf:.2f})"
        else:
            with_jev_action = baseline_action
            with_jev_conf = baseline_conf
            with_jev_reason = f"Baseline action retained: {baseline_reason}"

        # 4. Divergence Classification
        divergence = classify_divergence(baseline_action, with_jev_action)

        # 5. Safety & Execution Check
        # In shadow mode, executed is strictly False and no broker call is ever made.
        executed = False
        order_intent_id = None

        decision_id = f"dec_{uuid.uuid4().hex[:12]}"
        decision_row = DecisionRecord(
            decision_id=decision_id,
            trace_id=trace_id,
            scan_id=scan_id,
            event_id=event.event_id,
            symbol=symbol,
            venue=venue,
            timeframe=timeframe,
            observed_price=observed_price,
            baseline_action=baseline_action,
            baseline_confidence=baseline_conf,
            baseline_reason=baseline_reason,
            with_jev_action=with_jev_action,
            with_jev_confidence=with_jev_conf,
            with_jev_reason=with_jev_reason,
            divergence=divergence,
            jev_signal=jev_sig,
            jev_confidence=jev_conf,
            jev_vetoed=jev_veto,
            jev_conflict=jev_conflict,
            jev_reason=jev_reason_text,
            execution_mode=mode_str,
            executed=executed,
            order_intent_id=order_intent_id,
            details={
                "indicators": indicators,
                "is_shadow": is_shadow,
                "opinion_id": opinion_id,
            },
        )
        db.add(decision_row)
        decisions_created.append(decision_id)

        # WP4: Non-blocking Supabase Dual-Write Projection
        try:
            from backend.services.supabase_projector import default_supabase_projector
            default_supabase_projector.project_decision(decision_row, opinion_row)
        except Exception as p_err:
            logger.debug(f"Non-blocking Supabase projection error: {p_err}")

    # Mark outbox event completed
    event.status = "completed"
    event.processed_at = utc_now()
    event.last_error = None
    db.commit()

    logger.info(
        f"[OUTBOX WORKER] Processed event_id={event.event_id} (scan_id={scan_id}). "
        f"Generated {len(decisions_created)} decision records."
    )
    return {
        "event_id": event.event_id,
        "scan_id": scan_id,
        "status": "completed",
        "decisions_count": len(decisions_created),
        "decision_ids": decisions_created,
    }


async def run_outbox_worker_iteration(
    db: Session,
    batch_size: int = 10,
    worker_id: str = "outbox_worker_1",
    jev_override_eval: Optional[dict[str, Any]] = None,
) -> int:
    """
    Run one iteration of the outbox consumer worker.
    Returns the number of successfully processed events.
    """
    events = claim_pending_events(db, batch_size=batch_size, worker_id=worker_id)
    if not events:
        return 0

    processed_count = 0
    for event in events:
        try:
            res = await process_outbox_event(db, event, jev_override_eval=jev_override_eval)
            if res.get("status") == "completed":
                processed_count += 1
        except Exception as exc:
            logger.error(f"[OUTBOX WORKER] Error processing event {event.event_id}: {exc}", exc_info=True)
            db.rollback()
            event.last_error = str(exc)
            if event.attempt_count >= event.max_attempts:
                event.status = "dead"
            else:
                event.status = "pending"
            db.commit()

    return processed_count
