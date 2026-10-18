"""
Outcome Closure & Deterministic Labeling Worker (WP3)
=====================================================
Evaluates forward returns at strictly defined horizons (1H, 4H, 24H),
applies standardized Triple-Barrier policy labels (WIN, LOSS, SCRATCH),
attributes realized trade PnL, and idempotently projects closed-trade
features to Qdrant semantic memory.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from sqlalchemy.orm import Session

from backend.database.models import DecisionOutcome, DecisionRecord, Trade
from backend.services.symbol_aliases import normalize_side

logger = logging.getLogger(__name__)

HORIZON_DELTAS = {
    "1H": timedelta(hours=1),
    "4H": timedelta(hours=4),
    "24H": timedelta(hours=24),
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def compute_triple_barrier_label(
    direction: str,
    reference_price: float,
    horizon_price: float,
    high_price: Optional[float] = None,
    low_price: Optional[float] = None,
    atr: Optional[float] = None,
    sl_mult: float = 1.75,
    tp_mult: float = 5.5,
) -> tuple[str, float, float, float]:
    """
    Compute forward return and Triple-Barrier label.
    Returns: (label, return_pct, mfe_pct, mae_pct)
    """
    if reference_price <= 0:
        return "SCRATCH", 0.0, 0.0, 0.0

    dir_norm = normalize_side(direction)
    high_px = high_price if high_price is not None else max(reference_price, horizon_price)
    low_px = low_price if low_price is not None else min(reference_price, horizon_price)

    if dir_norm == "BUY":
        return_pct = ((horizon_price - reference_price) / reference_price) * 100.0
        mfe_pct = ((high_px - reference_price) / reference_price) * 100.0
        mae_pct = ((low_px - reference_price) / reference_price) * 100.0

        if atr and atr > 0:
            tp_price = reference_price + (atr * tp_mult)
            sl_price = reference_price - (atr * sl_mult)
            if high_px >= tp_price:
                return "WIN", return_pct, mfe_pct, mae_pct
            if low_px <= sl_price:
                return "LOSS", return_pct, mfe_pct, mae_pct

        # Default vertical threshold (+0.5% / -0.5%)
        if return_pct >= 0.50:
            label = "WIN"
        elif return_pct <= -0.50:
            label = "LOSS"
        else:
            label = "SCRATCH"

    elif dir_norm == "SELL":
        return_pct = ((reference_price - horizon_price) / reference_price) * 100.0
        mfe_pct = ((reference_price - low_px) / reference_price) * 100.0
        mae_pct = ((reference_price - high_px) / reference_price) * 100.0

        if atr and atr > 0:
            tp_price = reference_price - (atr * tp_mult)
            sl_price = reference_price + (atr * sl_mult)
            if low_px <= tp_price:
                return "WIN", return_pct, mfe_pct, mae_pct
            if high_px >= sl_price:
                return "LOSS", return_pct, mfe_pct, mae_pct

        if return_pct >= 0.50:
            label = "WIN"
        elif return_pct <= -0.50:
            label = "LOSS"
        else:
            label = "SCRATCH"

    else:
        # Direction neutral / HOLD
        return_pct = abs(((horizon_price - reference_price) / reference_price) * 100.0)
        mfe_pct = 0.0
        mae_pct = 0.0
        label = "SCRATCH"

    return label, round(return_pct, 4), round(mfe_pct, 4), round(mae_pct, 4)


def label_decision_record(
    db: Session,
    record: DecisionRecord,
    horizon: str = "1H",
    horizon_price: Optional[float] = None,
    high_price: Optional[float] = None,
    low_price: Optional[float] = None,
    policy_version: str = "v1_triple_barrier",
) -> Optional[DecisionOutcome]:
    """
    Deterministically label an individual decision record. Idempotent across re-runs.
    """
    # 1. Check existing outcome
    existing = (
        db.query(DecisionOutcome)
        .filter(
            DecisionOutcome.decision_id == record.decision_id,
            DecisionOutcome.horizon == horizon,
            DecisionOutcome.policy_version == policy_version,
        )
        .first()
    )
    if existing:
        return existing

    ref_price = record.observed_price or 0.0
    if ref_price <= 0:
        logger.debug(f"Decision {record.decision_id} has invalid reference price {ref_price}")
        return None

    # Determine horizon price (caller or fallback)
    target_price = horizon_price if horizon_price is not None else ref_price

    # Determine direction: active engine decision preferred
    action = record.with_jev_action
    if action == "HOLD":
        action = record.baseline_action

    indicators = (record.details or {}).get("indicators") or {}
    atr = float(indicators.get("atr") or 0.0)

    label, ret_pct, mfe_pct, mae_pct = compute_triple_barrier_label(
        direction=action,
        reference_price=ref_price,
        horizon_price=target_price,
        high_price=high_price,
        low_price=low_price,
        atr=atr if atr > 0 else None,
    )

    # Correlate with executed trade if applicable
    trade_id = None
    realized_pnl = None
    fees_paid = 0.0
    if record.order_intent_id:
        trade_row = db.query(Trade).filter(Trade.id == record.order_intent_id).first()
        if trade_row and trade_row.status == "closed":
            trade_id = trade_row.id
            realized_pnl = float(trade_row.realized_pnl or 0.0)

    outcome_id = f"out_{uuid.uuid4().hex[:12]}"
    outcome = DecisionOutcome(
        outcome_id=outcome_id,
        decision_id=record.decision_id,
        symbol=record.symbol,
        horizon=horizon,
        reference_price=ref_price,
        horizon_price=target_price,
        forward_return_pct=ret_pct,
        mfe_pct=mfe_pct,
        mae_pct=mae_pct,
        label=label,
        policy_version=policy_version,
        trade_id=trade_id,
        realized_pnl=realized_pnl,
        fees_paid=fees_paid,
        qdrant_synced=False,
        details={
            "action_evaluated": action,
            "divergence": record.divergence,
            "jev_signal": record.jev_signal,
            "baseline_action": record.baseline_action,
        },
    )
    db.add(outcome)
    db.commit()
    return outcome


def run_outcome_worker_iteration(
    db: Session,
    horizon: str = "1H",
    batch_size: int = 50,
    price_provider: Optional[Callable[[str, datetime], float]] = None,
    policy_version: str = "v1_triple_barrier",
) -> int:
    """
    Find mature decision records lacking outcomes for the given horizon,
    and compute labels.
    """
    delta = HORIZON_DELTAS.get(horizon, timedelta(hours=1))
    cutoff = utc_now() - delta

    # Find candidates
    existing_subquery = (
        db.query(DecisionOutcome.decision_id)
        .filter(
            DecisionOutcome.horizon == horizon,
            DecisionOutcome.policy_version == policy_version,
        )
        .subquery()
    )

    records = (
        db.query(DecisionRecord)
        .filter(
            DecisionRecord.created_at <= cutoff,
            ~DecisionRecord.decision_id.in_(existing_subquery.select()),
        )
        .order_by(DecisionRecord.created_at.asc())
        .limit(batch_size)
        .all()
    )

    count = 0
    for record in records:
        try:
            hp = None
            if price_provider is not None:
                hp = price_provider(record.symbol, record.created_at + delta)
            label_decision_record(
                db=db,
                record=record,
                horizon=horizon,
                horizon_price=hp,
                policy_version=policy_version,
            )
            count += 1
        except Exception as exc:
            logger.warning(f"Error labeling decision {record.decision_id}: {exc}")
            db.rollback()

    return count
