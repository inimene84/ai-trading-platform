"""Strategy vs exchange-reconciliation PnL — Phase C accounting truth.

Lifetime ``Trade.pnl`` sums are a local DB artifact (they can include
adopted orphans, AIN/ONE/BULLA/SYN reconciles, and years of paper rows).
They are not cash and must not be presented as live performance.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Optional

RECONCILIATION_STRATEGIES = frozenset({"exchange_reconciliation"})

LIFETIME_DB_PNL_NOTE = (
    "lifetime_db_pnl is the sum of closed-row Trade.pnl in the local DB. "
    "It is not cash, not union equity, and includes historical / reconcile "
    "artifacts. Do not report it as live performance."
)

UNION_LABEL = "union = cTrader + Binance"


def _strategy_of(trade: Any) -> str:
    if isinstance(trade, Mapping):
        return str(trade.get("strategy") or "")
    return str(getattr(trade, "strategy", "") or "")


def _pnl_of(trade: Any) -> float:
    if isinstance(trade, Mapping):
        return float(trade.get("pnl") or 0.0)
    return float(getattr(trade, "pnl", 0.0) or 0.0)


def _closed_at(trade: Any) -> Optional[datetime]:
    raw = None
    if isinstance(trade, Mapping):
        raw = trade.get("closed_at") or trade.get("timestamp")
    else:
        raw = getattr(trade, "closed_at", None) or getattr(trade, "timestamp", None)
    if raw is None:
        return None
    if isinstance(raw, datetime):
        if raw.tzinfo is None:
            return raw.replace(tzinfo=timezone.utc)
        return raw
    return None


def is_reconciliation_trade(trade: Any) -> bool:
    return _strategy_of(trade).strip().lower() in RECONCILIATION_STRATEGIES


def strategy_trades(trades: Iterable[Any]) -> list[Any]:
    """Drop exchange_reconciliation rows so they cannot enter strategy PnL."""
    return [trade for trade in trades if not is_reconciliation_trade(trade)]


def _sum_pnl(trades: Iterable[Any]) -> float:
    return float(sum(_pnl_of(trade) for trade in trades))


def realized_windows(
    trades: Iterable[Any],
    *,
    now: Optional[datetime] = None,
    equity: float = 0.0,
) -> dict[str, Any]:
    """Realized today / 7d from *strategy* trades only."""
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=timezone.utc)
    start_today = clock.replace(hour=0, minute=0, second=0, microsecond=0)
    start_7d = clock - timedelta(days=7)
    strategy = strategy_trades(trades)
    recon = [trade for trade in trades if is_reconciliation_trade(trade)]

    def _in_window(start: datetime) -> list[Any]:
        rows = []
        for trade in strategy:
            closed = _closed_at(trade)
            if closed is None or closed >= start:
                if closed is not None:
                    rows.append(trade)
        return rows

    today_rows = _in_window(start_today)
    week_rows = _in_window(start_7d)
    today = _sum_pnl(today_rows)
    week = _sum_pnl(week_rows)
    strategy_lifetime = _sum_pnl(strategy)
    recon_lifetime = _sum_pnl(recon)
    db_lifetime = strategy_lifetime + recon_lifetime
    return {
        "realized_today": round(today, 4),
        "realized_7d": round(week, 4),
        "strategy_lifetime_pnl": round(strategy_lifetime, 4),
        "reconciliation_pnl": round(recon_lifetime, 4),
        "reconciliation_trade_count": len(recon),
        "strategy_trade_count": len(strategy),
        "lifetime_db_pnl": round(db_lifetime, 4),
        "lifetime_db_pnl_pct": round((db_lifetime / equity * 100.0) if equity > 0 else 0.0, 2),
        "lifetime_db_pnl_is_not_cash": True,
        "lifetime_db_pnl_note": LIFETIME_DB_PNL_NOTE,
        "excluded_strategies": sorted(RECONCILIATION_STRATEGIES),
    }


def attach_portfolio_accounting(
    payload: dict[str, Any],
    *,
    closed_trades: Iterable[Any],
    unrealized_pnl: float,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Rewrite portfolio fields so lifetime DB PnL is not presented as cash."""
    equity = float(payload.get("equity") or 0.0)
    windows = realized_windows(closed_trades, now=now, equity=equity)
    out = dict(payload)
    out.update(windows)
    out["open_unrealized_pnl"] = round(float(unrealized_pnl), 4)
    # Keep the old keys for one release but tag them as the DB artifact.
    out["total_pnl"] = windows["lifetime_db_pnl"]
    out["total_pnl_pct"] = windows["lifetime_db_pnl_pct"]
    out["total_pnl_is_not_cash"] = True
    out["total_pnl_source"] = "lifetime_db_artifact"
    out["union_label"] = UNION_LABEL
    return out
