"""Dedupe closed Trade rows so P&L and expectancy use one fill per close.

Live week (Oct 3–13): Binance income history was +4.91 USDC net, but the
local DB summed closed rows to about -16.59. Two writers created extra
``exchange_reconciliation`` / estimated-exit rows for the same fill:

* ``upsert_binance_live_trades`` + ``BrokerPositionSyncService`` both adopt
  orphans when ``broker_position_id`` does not match the strategy row.
* Dashboard + loop both call cTrader reconcile; persist-then-reconcile
  created a second open row (no PID yet) that later closed as a twin.

Those twins poisoned the expectancy gate (15 net-negative closes locked
every USDC pair) and the dashboard lifetime P&L. This module is the
read-side truth: strategy rows win, twins collapse, estimated P&L loses
to a sibling with a venue id.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from backend.services.ledger import is_binance_position_key
from backend.services.pnl_accounting import is_reconciliation_trade

_CLOSE_WINDOW_SEC = 15 * 60  # same-symbol closes within 15 min are one event
_QTY_REL_TOL = 0.02
_PRICE_REL_TOL = 0.005


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _closed_at(trade: Any) -> Optional[datetime]:
    raw = getattr(trade, "closed_at", None) or getattr(trade, "timestamp", None)
    if raw is None:
        return None
    if isinstance(raw, datetime):
        if raw.tzinfo is None:
            return raw.replace(tzinfo=timezone.utc)
        return raw
    return None


def _norm_side(trade: Any) -> str:
    side = str(getattr(trade, "direction", "") or "").upper()
    if side in {"LONG", "BUY"}:
        return "BUY"
    if side in {"SHORT", "SELL"}:
        return "SELL"
    return side


def _norm_symbol(trade: Any) -> str:
    return str(getattr(trade, "symbol", "") or "").upper()


def _close_key(trade: Any) -> tuple[str, str, str]:
    """Bucket key: symbol + side + venue id when present.

    Hedge-mode keys like ``ETHUSDC:SHORT`` are reused for every later short
    on that perp, so they must not identify a single close. Numeric cTrader
    position ids and Binance order ids do.
    """
    pid = str(getattr(trade, "broker_position_id", "") or "").strip()
    oid = str(
        getattr(trade, "binance_order_id", None)
        or getattr(trade, "broker_order_id", None)
        or ""
    ).strip()
    venue = pid if pid and not is_binance_position_key(pid) else oid
    return (_norm_symbol(trade), _norm_side(trade), venue)


def _qty_close(a: Any, b: Any) -> bool:
    qa, qb = abs(_as_float(getattr(a, "quantity", 0))), abs(_as_float(getattr(b, "quantity", 0)))
    if qa <= 0 or qb <= 0:
        return True
    return abs(qa - qb) / max(qa, qb) <= _QTY_REL_TOL


def _entry_close(a: Any, b: Any) -> bool:
    ea, eb = _as_float(getattr(a, "entry_price", 0)), _as_float(getattr(b, "entry_price", 0))
    if ea <= 0 or eb <= 0:
        return True
    return abs(ea - eb) / max(ea, eb) <= _PRICE_REL_TOL


def _same_close_event(a: Any, b: Any) -> bool:
    if _norm_symbol(a) != _norm_symbol(b) or _norm_side(a) != _norm_side(b):
        return False
    key_a, key_b = _close_key(a)[2], _close_key(b)[2]
    if key_a and key_b:
        return key_a == key_b
    # Without a shared venue id, only collapse a strategy row against its
    # reconciliation twin, or a venue-id row against a missing-id sibling.
    # Two strategy rows of the same size stay distinct — collapsing them
    # would make the expectancy gate fail-open on a genuinely bad book.
    a_recon = is_reconciliation_trade(a)
    b_recon = is_reconciliation_trade(b)
    if not (a_recon != b_recon or bool(key_a) != bool(key_b)):
        return False
    ta, tb = _closed_at(a), _closed_at(b)
    if ta is None or tb is None:
        return _qty_close(a, b) and _entry_close(a, b)
    if abs((ta - tb).total_seconds()) > _CLOSE_WINDOW_SEC:
        return False
    return _qty_close(a, b) and _entry_close(a, b)


def _prefer(a: Any, b: Any) -> Any:
    """Keep the row that looks like the real fill, not the reconcile twin."""
    a_recon = is_reconciliation_trade(a)
    b_recon = is_reconciliation_trade(b)
    if a_recon != b_recon:
        return b if a_recon else a
    a_pid = bool(getattr(a, "broker_position_id", None) or getattr(a, "binance_order_id", None))
    b_pid = bool(getattr(b, "broker_position_id", None) or getattr(b, "binance_order_id", None))
    if a_pid != b_pid:
        return a if a_pid else b
    a_exit = _as_float(getattr(a, "exit_price", 0))
    b_exit = _as_float(getattr(b, "exit_price", 0))
    if (a_exit > 0) != (b_exit > 0):
        return a if a_exit > 0 else b
    # Prefer the earlier close (the first writer) so re-reconcile cannot
    # refresh closed_at and pin the expectancy window forever.
    ta, tb = _closed_at(a), _closed_at(b)
    if ta and tb and ta != tb:
        return a if ta <= tb else b
    id_a = getattr(a, "id", 0) or 0
    id_b = getattr(b, "id", 0) or 0
    return a if id_a <= id_b else b


def dedupe_closed_trades(trades: Iterable[Any]) -> list[Any]:
    """Collapse twin closed rows that describe the same fill."""
    kept: list[Any] = []
    for trade in trades:
        merged = False
        for idx, existing in enumerate(kept):
            if _same_close_event(existing, trade):
                kept[idx] = _prefer(existing, trade)
                merged = True
                break
        if not merged:
            kept.append(trade)
    return kept


def strategy_closed_trades(trades: Iterable[Any]) -> list[Any]:
    """Dedupe then drop leftover exchange_reconciliation rows."""
    return [t for t in dedupe_closed_trades(trades) if not is_reconciliation_trade(t)]


def aggregate_symbol_pnl(trades: Iterable[Any]) -> list[tuple[str, int, float]]:
    """Return (symbol, count, pnl_sum) for expectancy / reporting."""
    buckets: dict[str, list[Any]] = {}
    for trade in strategy_closed_trades(trades):
        symbol = _norm_symbol(trade)
        if not symbol:
            continue
        if getattr(trade, "pnl", None) is None:
            continue
        buckets.setdefault(symbol, []).append(trade)
    rows = []
    for symbol, group in buckets.items():
        pnl_sum = sum(_as_float(getattr(t, "pnl", 0.0)) for t in group)
        rows.append((symbol, len(group), pnl_sum))
    return rows


def find_open_leg(open_rows: Iterable[Any], *, symbol: str, direction: str, broker: str) -> Any | None:
    """Existing open row for this venue leg, if any (prevents a second insert)."""
    want_sym = str(symbol or "").upper()
    want_dir = str(direction or "").upper()
    if want_dir in {"LONG", "BUY"}:
        want_dir = "BUY"
    elif want_dir in {"SHORT", "SELL"}:
        want_dir = "SELL"
    want_broker = (broker or "").strip().lower()
    for row in open_rows:
        if _norm_symbol(row) != want_sym:
            continue
        if _norm_side(row) != want_dir:
            continue
        row_broker = str(getattr(row, "broker", None) or getattr(row, "exchange", None) or "").lower()
        if want_broker and row_broker and want_broker not in row_broker and row_broker not in want_broker:
            continue
        return row
    return None
