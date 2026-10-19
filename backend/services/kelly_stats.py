"""Venue- and symbol-scoped closed-trade stats for the empirical Kelly payoff b.

Oct 19 review: the crypto loop computed b from cTrader FX/metal rows (the
risk broker was cTrader while the loop traded Binance), including a corrupt
gold row, so b hit the 8.0 cap and Kelly collapsed to 0. Stats now come only
from the venue that will take the order, for the same instrument (USDT/USDC
twins merged), falling back to the same asset class on that venue, and finally
to a conservative geometric b when neither sample is large enough.

Rows that are not real outcomes are dropped before anything is averaged:

* test / simulated rows: ``test-*`` ids, ``sim_ct_*`` ids, broker
  ``ctrader:paper``, notes containing "placed (simulated)";
* void / reconciliation rows (``exchange_reconciliation`` adoptions);
* null or zero P&L;
* corrupt prices: entry more than 50% away from that symbol's median entry.

b is computed on return-on-notional (pnl / (entry * qty)) when size is known,
so a handful of oversized paper fills cannot dominate the ratio.
"""

from __future__ import annotations

import logging
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from backend.services.multi_asset_bars import classify_symbol
from backend.services.symbol_aliases import crypto_perp_base

logger = logging.getLogger(__name__)

# Real broker books only. ``ctrader:paper`` is the local cTrader simulator.
VENUE_BROKERS: dict[str, frozenset[str]] = {
    "ctrader": frozenset({"ctrader", "ic", "icmarkets"}),
    "binance_futures": frozenset({"binance_futures", "binance", "binanceusdm"}),
}
TEST_BROKERS = frozenset({"ctrader:paper"})
EXCLUDED_STRATEGIES = frozenset({"exchange_reconciliation"})
EXCLUDED_STATUSES = frozenset({"void", "failed", "orphaned", "cancelled", "canceled"})
_TEST_PREFIXES = ("test-", "test_", "sim_ct_")
CORRUPT_PRICE_DEVIATION = 0.50


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default)) or default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def kelly_stats_min_sample() -> int:
    """Closed trades needed before an empirical b is trusted (symbol or class)."""
    return max(1, _env_int("KELLY_STATS_MIN_SAMPLE", 30))


def kelly_fallback_payoff_cap() -> float:
    """Upper bound on the thin-book b. Geometry alone (~2.5 net) is optimistic."""
    return max(0.5, _env_float("KELLY_FALLBACK_PAYOFF_CAP", 1.5))


def venue_for_broker(broker: Optional[str]) -> str:
    b = str(broker or "").strip().lower()
    if b.startswith("ctrader") or b in {"ic", "icmarkets"}:
        return "ctrader"
    return "binance_futures"


def instrument_key(symbol: Optional[str]) -> str:
    """ETHUSDT / ETHUSDC / ETH-USD → ETH; FX and metals stay as-is."""
    s = str(symbol or "").upper()
    return crypto_perp_base(s) or s


def _get(row: Any, name: str) -> Any:
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name, None)


def is_test_row(row: Any) -> bool:
    broker = str(_get(row, "broker") or "").strip().lower()
    if broker in TEST_BROKERS:
        return True
    for attr in ("signal_id", "broker_order_id", "broker_position_id", "binance_order_id", "symbol", "strategy"):
        val = str(_get(row, attr) or "").strip().lower()
        if val.startswith(_TEST_PREFIXES):
            return True
    notes = str(_get(row, "notes") or "").lower()
    if "placed (simulated)" in notes or "sim_ct_" in notes or notes.startswith("test-"):
        return True
    return False


@dataclass
class ExclusionCounts:
    test: int = 0
    void: int = 0
    reconciliation: int = 0
    null_or_zero_pnl: int = 0
    corrupt_price: int = 0
    other_venue_or_mode: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass
class ScopedPnlStats:
    closed_count: int
    avg_win: float
    avg_loss_abs: float
    scope: str  # "symbol" | "asset_class" | "fallback"
    venue: str
    symbol: str
    asset_class: str
    symbol_count: int = 0
    class_count: int = 0
    excluded: dict[str, int] = field(default_factory=dict)

    def describe(self) -> str:
        return (
            f"venue={self.venue} scope={self.scope} symbol={self.symbol} "
            f"class={self.asset_class} n_symbol={self.symbol_count} "
            f"n_class={self.class_count} used={self.closed_count}"
        )


def _row_return(row: Any) -> Optional[float]:
    """Return on notional when size is known, else raw P&L."""
    pnl = _get(row, "pnl")
    if pnl is None:
        return None
    try:
        pnl = float(pnl)
    except (TypeError, ValueError):
        return None
    try:
        entry = float(_get(row, "entry_price") or 0.0)
        qty = abs(float(_get(row, "quantity") or 0.0))
    except (TypeError, ValueError):
        entry, qty = 0.0, 0.0
    notional = entry * qty
    if notional > 0 and venue_for_broker(_get(row, "broker")) == "binance_futures":
        return pnl / notional
    # cTrader quantity is in lots; P&L in account currency is already
    # size-comparable within the small demo lot range.
    return pnl


def clean_rows(
    rows: Iterable[Any],
    *,
    venue: str,
    modes: Optional[Sequence[str]] = None,
) -> tuple[list[Any], ExclusionCounts]:
    """Keep real closed outcomes on ``venue`` (and ``modes``)."""
    excl = ExclusionCounts()
    brokers = VENUE_BROKERS.get(venue, frozenset({venue}))
    mode_set = {m.lower() for m in modes} if modes else None
    kept: list[Any] = []
    for row in rows:
        if is_test_row(row):
            excl.test += 1
            continue
        status = str(_get(row, "status") or "").strip().lower()
        if status in EXCLUDED_STATUSES:
            excl.void += 1
            continue
        if str(_get(row, "strategy") or "") in EXCLUDED_STRATEGIES:
            excl.reconciliation += 1
            continue
        broker = str(_get(row, "broker") or "").strip().lower()
        mode = str(_get(row, "mode") or "").strip().lower()
        if broker not in brokers or (mode_set is not None and mode not in mode_set):
            excl.other_venue_or_mode += 1
            continue
        pnl = _get(row, "pnl")
        try:
            if pnl is None or float(pnl) == 0.0:
                excl.null_or_zero_pnl += 1
                continue
        except (TypeError, ValueError):
            excl.null_or_zero_pnl += 1
            continue
        kept.append(row)

    # Corrupt-price filter: entry vs that instrument's median entry.
    by_inst: dict[str, list[float]] = {}
    for row in kept:
        try:
            ep = float(_get(row, "entry_price") or 0.0)
        except (TypeError, ValueError):
            ep = 0.0
        if ep > 0:
            by_inst.setdefault(instrument_key(_get(row, "symbol")), []).append(ep)
    medians = {k: statistics.median(v) for k, v in by_inst.items() if v}
    out: list[Any] = []
    for row in kept:
        try:
            ep = float(_get(row, "entry_price") or 0.0)
        except (TypeError, ValueError):
            ep = 0.0
        med = medians.get(instrument_key(_get(row, "symbol")))
        if ep <= 0 or not med or abs(ep / med - 1.0) > CORRUPT_PRICE_DEVIATION:
            excl.corrupt_price += 1
            continue
        out.append(row)
    return out, excl


def _win_loss(rows: Sequence[Any]) -> tuple[int, float, float]:
    rets = [r for r in (_row_return(x) for x in rows) if r is not None and r != 0]
    wins = [r for r in rets if r > 0]
    losses = [abs(r) for r in rets if r < 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return len(rets), avg_win, avg_loss


def compute_scoped_stats(
    rows: Iterable[Any],
    *,
    symbol: str,
    venue: str,
    modes: Optional[Sequence[str]] = None,
    min_sample: Optional[int] = None,
) -> ScopedPnlStats:
    """Symbol stats on ``venue``; same-asset-class fallback; else thin book.

    A thin book returns ``closed_count`` below ``min_sample`` so
    ``empirical_payoff_ratio`` uses its conservative fallback b.
    """
    need = min_sample if min_sample is not None else kelly_stats_min_sample()
    cleaned, excl = clean_rows(rows, venue=venue, modes=modes)
    key = instrument_key(symbol)
    asset = classify_symbol(symbol)
    sym_rows = [r for r in cleaned if instrument_key(_get(r, "symbol")) == key]
    cls_rows = [r for r in cleaned if classify_symbol(str(_get(r, "symbol") or "")) == asset]
    base = dict(venue=venue, symbol=symbol.upper(), asset_class=asset,
                symbol_count=len(sym_rows), class_count=len(cls_rows),
                excluded=excl.as_dict())
    n, w, l = _win_loss(sym_rows)
    if n >= need and w > 0 and l > 0:
        return ScopedPnlStats(closed_count=n, avg_win=w, avg_loss_abs=l, scope="symbol", **base)
    n, w, l = _win_loss(cls_rows)
    if n >= need and w > 0 and l > 0:
        return ScopedPnlStats(closed_count=n, avg_win=w, avg_loss_abs=l, scope="asset_class", **base)
    # Thin book: report the (small) sample size so callers fall back.
    small = min(n, max(need - 1, 0))
    return ScopedPnlStats(closed_count=small, avg_win=0.0, avg_loss_abs=0.0, scope="fallback", **base)


# ── DB loader with a short TTL cache (one query per venue per minute) ──────
_CACHE: dict[tuple, tuple[float, list[dict]]] = {}
_CACHE_TTL_SEC = 60.0
_COLUMNS = (
    "symbol", "broker", "mode", "status", "strategy", "pnl", "entry_price",
    "quantity", "notes", "signal_id", "broker_order_id", "broker_position_id",
    "binance_order_id",
)


def clear_cache() -> None:
    _CACHE.clear()


def load_closed_rows(venue: str) -> list[dict]:
    key = (venue,)
    now = time.monotonic()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < _CACHE_TTL_SEC:
        return hit[1]
    from backend.database.connection import SessionLocal
    from backend.database.models import Trade

    rows: list[dict] = []
    db = SessionLocal()
    try:
        brokers = sorted(VENUE_BROKERS.get(venue, frozenset({venue})) | TEST_BROKERS)
        q = (
            db.query(*[getattr(Trade, c) for c in _COLUMNS])
            .filter(Trade.status.in_(["closed", "exit"]))
            .filter(Trade.broker.in_(brokers))
        )
        for rec in q.all():
            rows.append(dict(zip(_COLUMNS, rec)))
    finally:
        db.close()
    _CACHE[key] = (now, rows)
    return rows
