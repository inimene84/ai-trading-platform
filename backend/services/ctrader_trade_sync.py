"""Mirror and reconcile live cTrader positions into the SQL Trade table used by the dashboard."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.database.connection import SessionLocal
from backend.database.models import Trade
from backend.services.ctrader_crypto import ctrader_trade_mode
from backend.services.trade_dedupe import find_open_leg

logger = logging.getLogger(__name__)


def ctrader_close_grace_sec() -> float:
    """Fill grace before an absent cTrader row may be closed (CTRADER_SYNC_CLOSE_GRACE_SEC, default 90)."""
    import os

    try:
        return max(0.0, float(os.getenv("CTRADER_SYNC_CLOSE_GRACE_SEC", "90") or 90))
    except (TypeError, ValueError):
        return 90.0


def _row_age_sec(row: Any, now: datetime) -> Optional[float]:
    ts = getattr(row, "timestamp", None)
    if not isinstance(ts, datetime):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (now - ts).total_seconds()

_OPEN_STATUSES = ("open", "filled")

_CTRADER_EXTRA_SYMBOLS = {
    "XAUUSD", "XAGUSD", "GOLD", "SILVER", "OIL", "WTI", "BRENT",
    "XTIUSD", "XBRUSD", "US30", "NAS100", "US500", "GER40", "UK100",
    "JP225", "SPX500", "WS30", "NDX", "DAX",
    "BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD", "BNBUSD",
    "LNKUSD", "AVXUSD", "DOGUSD",
}


def is_ctrader_trade(trade: Any) -> bool:
    """True for trades originating from or belonging to cTrader.

    Live Binance rows stamp broker_position_id as SYMBOL:LONG/SHORT. Treating
    any PID as cTrader made GET /positions close those legs as FX ghosts.
    """
    from backend.services.ledger import is_binance_position_key

    broker = (getattr(trade, "broker", None) or getattr(trade, "exchange", None) or "").lower()
    if "binance" in broker:
        return False
    if "ctrader" in broker or broker in {"ic", "icmarkets"}:
        return True
    pid = getattr(trade, "broker_position_id", None)
    if pid and not is_binance_position_key(pid):
        return True
    sym = str(getattr(trade, "symbol", "") or "").upper().strip()
    clean_sym = sym.split(".")[0].split("_")[0].replace("/", "").replace("-", "")
    if clean_sym in _CTRADER_EXTRA_SYMBOLS or sym in _CTRADER_EXTRA_SYMBOLS:
        return True
    return len(clean_sym) == 6 and clean_sym.isalpha() and not clean_sym.endswith("USDT")


def persist_ctrader_execution(
    *,
    symbol: str,
    direction: str,
    quantity: float,
    entry_price: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    strategy: Optional[str] = None,
    order_id: Optional[str] = None,
    position_id: Optional[str] = None,
    notes: Optional[str] = None,
) -> Optional[int]:
    """Insert an open Trade row for a successful cTrader dispatch."""
    db = SessionLocal()
    try:
        trade = Trade(
            symbol=str(symbol).upper(),
            direction=str(direction).upper(),
            quantity=float(quantity or 0),
            entry_price=float(entry_price or 0),
            stop_loss=stop_loss,
            take_profit=take_profit,
            status="open",
            strategy=strategy or "signal_candidate",
            broker="ctrader",
            exchange="ctrader",
            broker_order_id=str(order_id) if order_id else None,
            broker_position_id=str(position_id) if position_id else None,
            notes=notes or "cTrader live execution",
            mode=ctrader_trade_mode(),
        )
        db.add(trade)
        db.commit()
        db.refresh(trade)
        return int(trade.id)
    except Exception as exc:
        logger.warning("Failed to persist cTrader execution to Trade table: %s", exc)
        db.rollback()
        return None
    finally:
        db.close()




def count_open_ctrader_db_trades() -> int:
    """Open cTrader Trade rows (live + simulated). Used by execution caps."""
    db = SessionLocal()
    try:
        from backend.database.models import Trade
        return int(
            db.query(Trade)
            .filter(
                Trade.broker == "ctrader",
                Trade.status.in_(_OPEN_STATUSES),
                Trade.closed_at.is_(None),
            )
            .count()
        )
    except Exception as exc:
        logger.warning("Could not count open cTrader DB trades: %s", exc)
        return 0
    finally:
        db.close()


def open_ctrader_db_symbols() -> set:
    """Symbols with an open cTrader Trade row (live + simulated)."""
    db = SessionLocal()
    try:
        from backend.database.models import Trade
        rows = (
            db.query(Trade.symbol)
            .filter(
                Trade.broker == "ctrader",
                Trade.status.in_(_OPEN_STATUSES),
                Trade.closed_at.is_(None),
            )
            .all()
        )
        return {str(r[0]).upper() for r in rows if r and r[0]}
    except Exception as exc:
        logger.warning("Could not list open cTrader DB symbols: %s", exc)
        return set()
    finally:
        db.close()


def try_open_ctrader_db_positions() -> Optional[list]:
    """Open cTrader Trade rows, or None if the DB cannot be read."""
    db = SessionLocal()
    try:
        from backend.database.models import Trade
        rows = (
            db.query(Trade.symbol, Trade.direction)
            .filter(
                Trade.broker == "ctrader",
                Trade.status.in_(_OPEN_STATUSES),
                Trade.closed_at.is_(None),
            )
            .all()
        )
        return [
            {"symbol": str(r[0]).upper(), "direction": str(r[1] or "").upper()}
            for r in rows if r and r[0]
        ]
    except Exception as exc:
        logger.warning("Could not list open cTrader DB positions: %s", exc)
        return None
    finally:
        db.close()


def open_ctrader_db_positions() -> list:
    """Open cTrader Trade rows as {symbol, direction} dicts (live + simulated).

    Used for net currency exposure caps, which need direction, not just symbols.
    """
    found = try_open_ctrader_db_positions()
    return found if found is not None else []


def close_simulated_open_ctrader_trades(*, reason: str = "simulated ghost cleanup") -> int:
    """Mark open simulated cTrader trades closed (no broker position)."""
    db = SessionLocal()
    try:
        from backend.database.models import Trade
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        rows = (
            db.query(Trade)
            .filter(
                Trade.broker == "ctrader",
                Trade.status == "open",
                Trade.closed_at.is_(None),
                Trade.broker_position_id.is_(None),
            )
            .all()
        )
        closed = 0
        for t in rows:
            notes = (t.notes or "")
            order_id = (t.broker_order_id or "")
            is_sim = (
                "simulated" in notes.lower()
                or str(order_id).startswith("sim_ct_")
            )
            if not is_sim:
                continue
            t.status = "closed"
            t.closed_at = now
            t.exit_price = t.entry_price
            t.pnl = 0.0
            t.notes = (notes + f" | {reason}").strip(" |")
            closed += 1
        db.commit()
        return closed
    except Exception as exc:
        logger.warning("Failed closing simulated cTrader opens: %s", exc)
        db.rollback()
        return 0
    finally:
        db.close()


def is_ctrader_close_deal(deal: Optional[Dict[str, Any]]) -> bool:
    """True only when the cached deal is a close, not the opening fill.

    record_deal used to default close_type to CLOSED for every execution,
    so an entry fill must not confirm an empty-book mass-close.
    """
    if not deal:
        return False
    closed_volume = deal.get("closed_volume")
    try:
        if closed_volume is not None and float(closed_volume) > 0:
            return True
    except (TypeError, ValueError):
        pass
    if deal.get("gross_profit") is not None:
        return True
    close_type = str(deal.get("close_type") or "").upper()
    if close_type in {"SL_TP"}:
        return True
    if deal.get("is_close") or deal.get("closePositionDetail"):
        return True
    return False


def reconcile_ctrader_positions(
    db: Any,
    live_positions: Optional[List[Dict[str, Any]]] = None,
    broker: Optional[Any] = None,
) -> Dict[str, int]:
    """Reconcile DB open trades against live cTrader positions.

    - Creates missing DB rows for positions active on cTrader.
    - Updates entry prices / quantities for existing open rows.
    - Marks DB rows as 'closed' with accurate exit prices and P&L when they are
      absent from the live book (e.g. hit SL/TP or were closed).
    """
    if broker is None:
        try:
            from backend.services.ctrader_service import ctrader_broker
            broker = ctrader_broker
        except Exception:
            broker = None

    if live_positions is None:
        if broker is not None:
            try:
                live_positions = broker.get_positions()
            except Exception as exc:
                logger.warning("Failed to fetch cTrader live positions for reconcile: %s", exc)
                return {"created": 0, "updated": 0, "closed": 0}
        else:
            return {"created": 0, "updated": 0, "closed": 0}

    live_positions_list = list(live_positions or [])

    open_rows = (
        db.query(Trade)
        .filter(Trade.status.in_(_OPEN_STATUSES))
        .all()
    )
    ctrader_open_rows = [row for row in open_rows if is_ctrader_trade(row)]

    by_pid = {
        str(row.broker_position_id): row
        for row in ctrader_open_rows
        if row.broker_position_id
    }
    unmatched_by_symbol: Dict[str, List[Trade]] = {}
    for row in ctrader_open_rows:
        if row.broker_position_id:
            continue
        unmatched_by_symbol.setdefault(str(row.symbol).upper(), []).append(row)

    matched_db_trade_ids = set()
    created = 0
    updated = 0

    for raw in live_positions_list:
        pid = str(raw.get("position_id") or "").strip()
        symbol = str(raw.get("symbol") or "").upper()
        side = str(raw.get("side") or raw.get("direction") or "BUY").upper()
        qty = float(raw.get("quantity") or 0)
        entry = float(raw.get("entry_price") or 0)
        if not symbol:
            continue

        if pid and pid in by_pid:
            row = by_pid[pid]
            matched_db_trade_ids.add(row.id)
            row.quantity = qty or row.quantity
            if entry > 0:
                row.entry_price = entry
            row.direction = side
            live_sl = raw.get("stop_loss")
            live_tp = raw.get("take_profit")
            if live_sl:
                row.stop_loss = float(live_sl)
            if live_tp:
                row.take_profit = float(live_tp)
            updated += 1
            continue

        pending = unmatched_by_symbol.get(symbol) or []
        if pending:
            row = pending.pop(0)
            matched_db_trade_ids.add(row.id)
            if pid:
                row.broker_position_id = pid
                by_pid[pid] = row
            row.quantity = qty or row.quantity
            if entry > 0:
                row.entry_price = entry
            row.direction = side
            live_sl = raw.get("stop_loss")
            live_tp = raw.get("take_profit")
            if live_sl:
                row.stop_loss = float(live_sl)
            if live_tp:
                row.take_profit = float(live_tp)
            updated += 1
            continue

        # Persist-then-reconcile used to insert a second open row when the
        # execution row already had a PID that did not string-match (or the
        # snapshot raced the insert). Attach to that leg instead of twinning.
        existing_leg = find_open_leg(
            [r for r in ctrader_open_rows if r.id not in matched_db_trade_ids],
            symbol=symbol,
            direction=side,
            broker="ctrader",
        )
        if existing_leg is not None:
            matched_db_trade_ids.add(existing_leg.id)
            if pid:
                existing_leg.broker_position_id = pid
                by_pid[pid] = existing_leg
            existing_leg.quantity = qty or existing_leg.quantity
            if entry > 0:
                existing_leg.entry_price = entry
            existing_leg.direction = side
            updated += 1
            continue

        db.add(
            Trade(
                symbol=symbol,
                direction=side,
                quantity=qty,
                entry_price=entry or 0.0,
                status="open",
                strategy="ctrader_live",
                broker="ctrader",
                exchange="ctrader",
                broker_position_id=pid or None,
                notes="Synced from cTrader live book",
                mode=ctrader_trade_mode(),
            )
        )
        created += 1

    # Reconcile absent trades (e.g. hit SL/TP or closed on cTrader)
    closed = 0
    now = datetime.now(timezone.utc)
    unmatched_rows = [row for row in ctrader_open_rows if row.id not in matched_db_trade_ids]
    # An entirely empty live book while SQL still has opens is the same
    # ambiguous case as Binance sync: rate-limit / reconnect / permission
    # blip. Only close a row when the broker confirms a recent close deal.
    if not live_positions_list and unmatched_rows:
        confirmed = []
        for row in unmatched_rows:
            deal = None
            if broker is not None and row.broker_position_id:
                try:
                    deal = broker.get_recent_deal(
                        position_id=row.broker_position_id,
                        symbol=row.symbol,
                    )
                except Exception:
                    deal = None
            if is_ctrader_close_deal(deal):
                confirmed.append(row)
        if not confirmed:
            logger.error(
                "cTrader returned an empty positions snapshot while DB has "
                "%s open cTrader row(s) and no close deals; refusing bulk close",
                len(unmatched_rows),
            )
            db.commit()
            return {"created": created, "updated": updated, "closed": 0}
        unmatched_rows = confirmed

    grace_sec = ctrader_close_grace_sec()
    fresh_pids: Optional[set] = None
    for row in unmatched_rows:
        if row.id in matched_db_trade_ids:
            continue

        # Fill grace: a just-dispatched order is persisted (with its PID from
        # the ack) before the reconcile snapshot that contains the new position
        # arrives. Closing it here wrote a false "Closed externally" row and a
        # 45-minute post-loss cooldown on every fresh fill. Inside the grace
        # window only a broker-confirmed close deal may close the row.
        age = _row_age_sec(row, now)
        if age is not None and age < grace_sec:
            deal = None
            if broker is not None and row.broker_position_id:
                try:
                    deal = broker.get_recent_deal(
                        position_id=row.broker_position_id,
                        symbol=row.symbol,
                    )
                except Exception:
                    deal = None
            if not is_ctrader_close_deal(deal):
                logger.info(
                    "cTrader sync: keeping trade %s (%s pos %s) open, %.0fs old inside %.0fs "
                    "fill grace and no close deal",
                    row.id, row.symbol, row.broker_position_id, age, grace_sec,
                )
                continue

        # Re-check the broker's current book before closing anything.
        if row.broker_position_id:
            if fresh_pids is None:
                fresh_pids = set()
                if broker is not None:
                    try:
                        fresh_pids = {
                            str(p.get("position_id") or "").strip()
                            for p in (broker.get_positions() or [])
                        }
                    except Exception:
                        fresh_pids = set()
            if str(row.broker_position_id).strip() in fresh_pids:
                logger.info(
                    "cTrader sync: trade %s (%s pos %s) still in refreshed live book; not closing",
                    row.id, row.symbol, row.broker_position_id,
                )
                continue

        row.status = "closed"
        row.closed_at = now

        # Best effort exit price and P&L resolution
        exit_px = None
        pnl = None
        recent_deal = None
        if broker is not None:
            try:
                recent_deal = broker.get_recent_deal(
                    position_id=row.broker_position_id,
                    symbol=row.symbol,
                )
            except Exception:
                recent_deal = None

            if recent_deal:
                if recent_deal.get("execution_price"):
                    exit_px = float(recent_deal["execution_price"])
                if recent_deal.get("gross_profit") is not None:
                    pnl = float(recent_deal["gross_profit"])

            if exit_px is None:
                try:
                    exit_px = broker.get_exit_price(
                        row.symbol, row.direction, row.broker_position_id
                    )
                except Exception:
                    exit_px = None

        if pnl is None and exit_px and row.entry_price and row.quantity and broker is not None:
            try:
                lots = float(row.quantity or 0)
                units_fn = getattr(broker, "units_per_lot", None)
                units = lots * (
                    float(units_fn(row.symbol))
                    if callable(units_fn)
                    else float(getattr(broker, "CONTRACT_UNITS_PER_LOT", 100_000))
                )
                direction_mult = 1 if str(row.direction or "BUY").upper() in ("BUY", "LONG") else -1
                quote_pnl = (exit_px - float(row.entry_price)) * units * direction_mult
                rate = broker.quote_to_usd_rate(row.symbol, exit_px)
                pnl = round(quote_pnl * rate, 2) if rate else round(quote_pnl, 2)
            except Exception:
                pnl = None

        row.exit_price = exit_px
        row.pnl = pnl
        close_reason = (recent_deal.get("close_type") if recent_deal else None) or "SL/TP hit / closed on cTrader"
        current_notes = getattr(row, "notes", "") or ""
        marker = f"Closed externally ({close_reason})"
        if marker not in current_notes and "Closed externally" not in current_notes:
            try:
                row.notes = current_notes + f" | {marker}"
            except AttributeError:
                pass
        closed += 1
        if (pnl is not None and pnl < 0) or "SL" in str(close_reason).upper():
            try:
                from backend.services.signal_candidate_engine import signal_candidate_engine
                signal_candidate_engine.set_symbol_cooldown(row.symbol)
            except Exception as cd_err:
                logger.debug("Failed to set symbol cooldown on %s: %s", row.symbol, cd_err)
        logger.info(
            "cTrader ghost trade reconciled: id=%s symbol=%s pos_id=%s exit=%s pnl=%s reason=%s",
            row.id, row.symbol, row.broker_position_id, exit_px, pnl, close_reason,
        )

    db.commit()
    return {"created": created, "updated": updated, "closed": closed}


def upsert_ctrader_live_trades(db, live_positions: List[Dict[str, Any]]) -> int:
    """Create missing open Trade rows and reconcile closed cTrader positions.

    Maintains backwards compatibility by returning the count of newly created rows.
    """
    res = reconcile_ctrader_positions(db, live_positions)
    return res.get("created", 0)


def overlay_live_mark(
    payload: Dict[str, Any],
    live_by_pid: Dict[str, Dict[str, Any]],
    live_by_symbol: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Prefer live-book PnL/qty when the dashboard row is a cTrader trade."""
    broker = (payload.get("broker") or "").lower()
    sym = str(payload.get("symbol") or "").upper()
    if "binance" in broker:
        return payload
    from backend.services.ledger import is_binance_position_key
    is_ctrader = (
        broker == "ctrader"
        or (
            payload.get("broker_position_id")
            and not is_binance_position_key(payload.get("broker_position_id"))
        )
        or (len(sym) == 6 and sym.isalpha() and not sym.endswith("USDT"))
        or sym in ("XAUUSD", "XAGUSD")
    )
    if not is_ctrader:
        return payload
    if broker != "ctrader":
        payload["broker"] = "ctrader"

    pid = str(payload.get("broker_position_id") or "")
    live = live_by_pid.get(pid) or live_by_symbol.get(sym)
    if not live:
        entry = float(payload.get("entry_price") or 0)
        if not payload.get("current_price") and entry > 0:
            payload["current_price"] = entry
        return payload

    entry = float(live.get("entry_price") or payload.get("entry_price") or 0)
    qty = float(live.get("quantity") or payload.get("quantity") or 0)
    pnl = float(live.get("unrealized_pnl") or payload.get("unrealized_pnl") or 0)
    payload["entry_price"] = entry
    payload["quantity"] = qty
    payload["unrealized_pnl"] = round(pnl, 2)
    live_mark = live.get("current_price")
    payload["current_price"] = _positive_price(live_mark, payload.get("current_price"), entry)
    # Show the protection the broker actually holds, not what was requested.
    for level in ("stop_loss", "take_profit"):
        if live.get(level) is not None:
            payload[level] = float(live[level])
    payload["unrealized_pnl_pct"] = position_pnl_pct(
        entry=entry,
        mark=float(payload.get("current_price") or entry),
        direction=str(payload.get("direction") or "BUY"),
        is_ctrader=True,
    )
    return payload


def _positive_price(*candidates: Any) -> float:
    for raw in candidates:
        if raw is None:
            continue
        val = float(raw)
        if val > 0:
            return val
    return 0.0


def position_pnl_pct(
    *,
    entry: float,
    mark: float,
    direction: str,
    unrealized_pnl: float = 0.0,
    quantity: float = 0.0,
    is_ctrader: bool = False,
) -> float:
    """Return position return % for the dashboard.

    cTrader FX quantity is lots (0.1 = 10k units). Dividing dollar P&L by
    ``entry * lots`` (~0.12 on EURUSD) produced -1980% on a -$2 move. FX rows
    use mark vs entry price change instead, matching crypto semantics.
    """
    entry_f = float(entry or 0)
    mark_f = float(mark or 0)
    if entry_f <= 0:
        return 0.0
    if is_ctrader:
        if mark_f <= 0:
            return 0.0
        if str(direction or "BUY").upper() in ("BUY", "LONG"):
            return round((mark_f - entry_f) / entry_f * 100, 2)
        return round((entry_f - mark_f) / entry_f * 100, 2)
    qty = float(quantity or 0)
    if qty <= 0:
        return 0.0
    return round(float(unrealized_pnl or 0) / (entry_f * qty) * 100, 2)
