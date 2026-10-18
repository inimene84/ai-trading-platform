"""cTrader break-even + trailing exit manager.

The FX candidate path attaches hard SL/TP at entry and then never manages
the trade — winners that revert give back everything (2026-09-08 live book:
8 full stop-outs, several after going positive first). This manager ratchets
stops only in the profit direction:

  +CTRADER_BE_TRIGGER_R (1.0R)  -> move SL to entry + small lock (break-even)
  +CTRADER_TRAIL_START_R (1.5R) -> trail SL at max(0.75R, broker min) behind price

R is the ORIGINAL entry->stop distance, persisted in Trade.broker_metadata
so restarts and prior amends can never shrink it. The ratchet only ever
tightens stops, never loosens them.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from backend.database.connection import SessionLocal
from backend.database.models import Trade
from backend.services.ctrader_service import CTraderService, ctrader_service
from backend.services.ctrader_weekend_protect import apply_swap_holds, apply_weekend_protect

logger = logging.getLogger(__name__)

ENABLED = os.getenv("CTRADER_EXIT_MANAGER_ENABLED", "true").strip().lower() == "true"
BE_TRIGGER_R = float(os.getenv("CTRADER_BE_TRIGGER_R", "1.0"))
BE_LOCK_R = float(os.getenv("CTRADER_BE_LOCK_R", "0.1"))
TRAIL_START_R = float(os.getenv("CTRADER_TRAIL_START_R", "1.5"))
TRAIL_DIST_R = float(os.getenv("CTRADER_TRAIL_DIST_R", "0.75"))


def _find_trade_row(db, position: Dict[str, Any]) -> Optional[Trade]:
    pid = str(position.get("position_id") or "")
    if pid:
        row = (
            db.query(Trade)
            .filter(
                Trade.broker == "ctrader",
                Trade.status == "open",
                Trade.broker_position_id == pid,
            )
            .first()
        )
        if row:
            return row
    sym = str(position.get("symbol") or "").upper()
    if sym:
        return (
            db.query(Trade)
            .filter(
                Trade.broker == "ctrader",
                Trade.status == "open",
                Trade.symbol == sym,
            )
            .order_by(Trade.timestamp.desc())
            .first()
        )
    return None


def _original_r_distance(db, position: Dict[str, Any], entry: float) -> Optional[float]:
    """Original entry->stop distance; persisted so amends never shrink R.

    Positions without a DB row (manual broker trades) are skipped entirely:
    with nothing to persist R into, a restart would recompute R from the
    already-ratcheted stop and glue the trail to the price.
    """
    row = _find_trade_row(db, position)
    if row is None:
        return None
    try:
        meta = dict(row.broker_metadata or {})
    except Exception:
        meta = {}
    r = meta.get("r_distance")
    if r and float(r) > 0:
        return float(r)

    # First sight: derive from the trade row's stop, else the live position's.
    candidates = []
    if row.stop_loss:
        candidates.append(float(row.stop_loss))
    if position.get("stop_loss"):
        candidates.append(float(position["stop_loss"]))
    for sl in candidates:
        r = abs(entry - sl)
        if r > 0:
            meta["r_distance"] = r
            row.broker_metadata = meta  # reassign so SQLAlchemy tracks JSON
            return r
    return None


def _manage_position(db, pos: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    symbol = str(pos.get("symbol") or "")
    side = str(pos.get("side") or pos.get("direction") or "").upper()
    pid = pos.get("position_id")
    current = float(pos.get("current_price") or 0)
    cur_sl = float(pos["stop_loss"]) if pos.get("stop_loss") else None

    if not symbol or not current or not pid or side not in ("BUY", "SELL"):
        return None

    # Prefer DB entry_price: the broker cache can drift on reconnect (spread
    # included vs not, or Spotware rounding) and produces wrong-side SL geometry.
    row = _find_trade_row(db, pos)
    broker_entry = float(pos.get("entry_price") or 0)
    entry = float(row.entry_price) if row and row.entry_price else broker_entry
    if entry <= 0:
        return None

    r = _original_r_distance(db, pos, entry)
    if not r or r <= 0:
        return None

    favorable = (current - entry) if side == "BUY" else (entry - current)
    if favorable < BE_TRIGGER_R * r:
        return None

    min_dist = CTraderService.min_protective_distance(symbol, current)
    pip = max(float(CTraderService.pip_size_for(symbol)), 1e-9)

    # After BE trigger, SL may lock at/through entry (profit side).
    # Broker legality is vs *current* price only — clamping back to the
    # loss side of entry defeats break-even.
    if side == "BUY":
        be_level = entry + BE_LOCK_R * r
        target = be_level
        if favorable >= TRAIL_START_R * r:
            target = max(be_level, current - max(TRAIL_DIST_R * r, min_dist))
        target = min(target, current - min_dist)  # broker legality vs mark
        improves = cur_sl is None or target > cur_sl + pip
    else:
        be_level = entry - BE_LOCK_R * r
        target = be_level
        if favorable >= TRAIL_START_R * r:
            target = min(be_level, current + max(TRAIL_DIST_R * r, min_dist))
        target = max(target, current + min_dist)
        improves = cur_sl is None or target < cur_sl - pip

    if not improves:
        return None

    clamped_sl, _ = CTraderService.clamp_protective_prices(
        symbol, current, target, None, direction=side
    )
    if clamped_sl is None:
        return None
    # Re-check improvement after clamp rounding
    if side == "BUY" and cur_sl is not None and clamped_sl <= cur_sl + pip:
        return None
    if side == "SELL" and cur_sl is not None and clamped_sl >= cur_sl - pip:
        return None

    res = ctrader_service.amend_position_sltp(pid, stop_loss=clamped_sl)
    if res.get("status") in ("sent", "simulated"):
        logger.info(
            "Exit manager %s %s: SL %s -> %s (favorable=%.1f pips, R=%.1f pips)",
            side, symbol, cur_sl, clamped_sl, favorable / pip, r / pip,
        )
        return {
            "symbol": symbol,
            "side": side,
            "position_id": str(pid),
            "old_sl": cur_sl,
            "new_sl": clamped_sl,
            "favorable_pips": round(favorable / pip, 1),
            "status": res.get("status"),
        }
    logger.warning("Exit manager amend failed for %s: %s", symbol, res.get("error"))
    return None


def run_exit_manager_once() -> Dict[str, Any]:
    """Single pass over the live cTrader book. Called from the poller."""
    if not ctrader_service.has_credentials():
        return {"skipped": True, "reason": "no_credentials"}
    if not ctrader_service.is_connected:
        return {"skipped": True, "reason": "not_connected"}

    positions = ctrader_service.get_positions() or []
    weekend = apply_weekend_protect(positions, ctrader_service)
    swap_holds = apply_swap_holds(positions, ctrader_service)
    if not ENABLED:
        return {
            "skipped": True,
            "reason": "disabled",
            "weekend_protect": weekend,
            "swap_holds": swap_holds,
        }
    if not positions:
        return {"managed": 0, "weekend_protect": weekend, "swap_holds": swap_holds}

    db = SessionLocal()
    try:
        amends = []
        for pos in positions:
            try:
                res = _manage_position(db, pos)
                if res:
                    amends.append(res)
            except Exception as exc:
                logger.warning(
                    "Exit manager error for %s: %s", pos.get("symbol"), exc
                )
        db.commit()
        return {
            "managed": len(amends),
            "amends": amends,
            "weekend_protect": weekend,
            "swap_holds": swap_holds,
        }
    finally:
        db.close()
