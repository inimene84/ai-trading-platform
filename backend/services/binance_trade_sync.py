"""Mirror live Binance futures positions into the SQL Trade table the dashboard reads."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from backend.database.models import Trade
from backend.services.ledger import (
    binance_position_key,
    is_binance_paper_fill,
)
from backend.services.symbol_aliases import perp_alias_set
from backend.services.trade_dedupe import find_open_leg

logger = logging.getLogger(__name__)

_OPEN_STATUSES = ("open", "filled")
_DEAD_STATUSES = ("closed", "orphaned")


def _norm_side(raw: Any) -> str:
    side = str(raw or "BUY").upper()
    if side in {"LONG", "BUY"}:
        return "BUY"
    return "SELL"


def upsert_binance_live_trades(db: Any, live_positions: List[Dict[str, Any]]) -> Dict[str, int]:
    """Ensure each live USDT-M/USDC-M leg has an open dashboard Trade row.

    Revives live rows that cTrader reconcile previously closed as FX ghosts.
    Does not quarantine or flatten. Paper/unverified rows stay untouched.
    """
    created = 0
    revived = 0
    updated = 0
    if not live_positions:
        return {"created": 0, "revived": 0, "updated": 0}

    for raw in live_positions:
        symbol = str(raw.get("symbol") or "").upper()
        qty = abs(float(raw.get("quantity") or raw.get("positionAmt") or 0))
        if not symbol or qty <= 0:
            continue
        direction = _norm_side(raw.get("side") or raw.get("direction"))
        entry = float(raw.get("entry_price") or raw.get("entryPrice") or 0)
        key = binance_position_key(symbol, direction)

        open_row = (
            db.query(Trade)
            .filter(
                Trade.status.in_(_OPEN_STATUSES),
                Trade.broker_position_id == key,
            )
            .order_by(Trade.id.desc())
            .first()
        )
        if open_row is None:
            aliases = perp_alias_set(symbol) or {symbol}
            candidates = (
                db.query(Trade)
                .filter(
                    Trade.status.in_(_OPEN_STATUSES),
                    Trade.symbol.in_(aliases),
                    Trade.direction == direction,
                )
                .order_by(Trade.id.desc())
                .all()
            )
            open_row = next(
                (
                    t
                    for t in candidates
                    if not is_binance_paper_fill(t)
                    and "binance" in str(getattr(t, "broker", None) or getattr(t, "exchange", None) or "binance").lower()
                ),
                None,
            )
            if open_row is None:
                open_row = find_open_leg(
                    [t for t in candidates if not is_binance_paper_fill(t)],
                    symbol=symbol,
                    direction=direction,
                    broker="binance_futures",
                )

        if open_row is not None:
            open_row.quantity = qty
            if entry > 0:
                open_row.entry_price = entry
                open_row.filled_price = entry
            open_row.broker = "binance_futures"
            open_row.exchange = "binance_futures"
            open_row.mode = "live"
            open_row.broker_position_id = key
            open_row.status = "open"
            if symbol and str(open_row.symbol or "").upper() != symbol:
                open_row.symbol = symbol
            updated += 1
            continue

        dead = (
            db.query(Trade)
            .filter(
                Trade.broker_position_id == key,
                Trade.status.in_(_DEAD_STATUSES),
            )
            .order_by(Trade.id.desc())
            .first()
        )
        if dead is not None and not is_binance_paper_fill(dead):
            dead.status = "open"
            dead.closed_at = None
            dead.exit_price = None
            dead.pnl = None
            dead.quantity = qty
            if entry > 0:
                dead.entry_price = entry
                dead.filled_price = entry
            dead.broker = "binance_futures"
            dead.exchange = "binance_futures"
            dead.mode = "live"
            dead.broker_position_id = key
            dead.notes = (dead.notes or "") + " | Reopened: still live on Binance book"
            revived += 1
            continue

        db.add(
            Trade(
                symbol=symbol,
                direction=direction,
                quantity=qty,
                entry_price=entry or 0.0,
                filled_price=entry or 0.0,
                status="open",
                strategy="exchange_reconciliation",
                notes="Synced from Binance live book",
                exchange="binance_futures",
                broker="binance_futures",
                broker_position_id=key,
                mode="live",
            )
        )
        created += 1

    if created or revived or updated:
        db.commit()
        logger.info(
            "Binance live-book upsert: created=%s revived=%s updated=%s",
            created, revived, updated,
        )
    return {"created": created, "revived": revived, "updated": updated}
