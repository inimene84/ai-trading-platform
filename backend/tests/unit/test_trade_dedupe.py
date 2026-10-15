"""Closed-row dedupe and expectancy aggregates must ignore reconcile twins."""

from datetime import datetime, timedelta, timezone

import pytest

from backend.database.models import Trade
from backend.services.trade_dedupe import (
    aggregate_symbol_pnl,
    dedupe_closed_trades,
    find_open_leg,
    strategy_closed_trades,
)


def _t(**kwargs):
    now = datetime(2026, 10, 10, 12, tzinfo=timezone.utc)
    defaults = dict(
        symbol="ETHUSDC",
        direction="SELL",
        quantity=0.05,
        entry_price=4000.0,
        exit_price=4010.0,
        status="closed",
        pnl=-0.50,
        strategy="combined",
        closed_at=now,
        timestamp=now,
    )
    defaults.update(kwargs)
    return Trade(**defaults)


def test_dedupe_collapses_reconciliation_twin():
    strategy = _t(id=1, strategy="combined", pnl=-0.50)
    twin = _t(
        id=2,
        strategy="exchange_reconciliation",
        pnl=-4.00,
        closed_at=strategy.closed_at + timedelta(minutes=2),
        notes="Synced from Binance live book | Closed externally (sync)",
    )
    kept = dedupe_closed_trades([strategy, twin])
    assert len(kept) == 1
    assert kept[0].strategy == "combined"
    assert kept[0].pnl == -0.50


def test_dedupe_keeps_distinct_symbols_and_sides():
    rows = [
        _t(symbol="ETHUSDC", direction="SELL", pnl=-0.4),
        _t(symbol="ETHUSDC", direction="BUY", pnl=0.8),
        _t(symbol="SOLUSDC", direction="SELL", pnl=-0.2),
    ]
    assert len(dedupe_closed_trades(rows)) == 3


def test_aggregate_excludes_reconciliation_and_sums_strategy():
    rows = [
        _t(symbol="ADAUSDC", strategy="combined", pnl=4.22, quantity=40),
        _t(symbol="ADAUSDC", strategy="exchange_reconciliation", pnl=-8.0, quantity=40),
        _t(symbol="LINKUSDC", strategy="combined", pnl=-1.43),
        _t(symbol="LINKUSDC", strategy="combined", pnl=-0.20, closed_at=datetime(2026, 10, 11, tzinfo=timezone.utc)),
    ]
    agg = {sym: (n, pnl) for sym, n, pnl in aggregate_symbol_pnl(rows)}
    assert agg["ADAUSDC"] == (1, 4.22)
    assert agg["LINKUSDC"][0] == 2
    assert agg["LINKUSDC"][1] == pytest.approx(-1.63)


def test_dedupe_does_not_collapse_distinct_strategy_losses():
    """Same size/entry within 15 min is normal for a losing streak, not a twin."""
    now = datetime(2026, 10, 10, 12, tzinfo=timezone.utc)
    rows = [
        _t(id=i, symbol="SOLUSDC", pnl=-0.4, closed_at=now, quantity=1, entry_price=10)
        for i in range(1, 16)
    ]
    kept = dedupe_closed_trades(rows)
    assert len(kept) == 15
    agg = {sym: (n, pnl) for sym, n, pnl in aggregate_symbol_pnl(rows)}
    assert agg["SOLUSDC"] == (15, pytest.approx(-6.0))


def test_dedupe_ignores_reusable_binance_position_keys():
    """ETHUSDC:SHORT is reused for every later short — not a unique close id."""
    now = datetime(2026, 10, 10, 12, tzinfo=timezone.utc)
    rows = [
        _t(
            id=i,
            symbol="ETHUSDC",
            direction="SELL",
            pnl=-0.5,
            quantity=0.02,
            entry_price=4000 + i,
            closed_at=now + timedelta(hours=i),
            broker_position_id="ETHUSDC:SHORT",
            binance_order_id=str(1000 + i),
        )
        for i in range(15)
    ]
    assert len(dedupe_closed_trades(rows)) == 15
    agg = {sym: (n, pnl) for sym, n, pnl in aggregate_symbol_pnl(rows)}
    assert agg["ETHUSDC"] == (15, pytest.approx(-7.5))


def test_dedupe_still_collapses_recon_twin_with_synthetic_key():
    strategy = _t(id=1, strategy="combined", pnl=-0.50, broker_position_id="ETHUSDC:SHORT")
    twin = _t(
        id=2,
        strategy="exchange_reconciliation",
        pnl=-4.00,
        broker_position_id="ETHUSDC:SHORT",
        closed_at=strategy.closed_at + timedelta(minutes=2),
    )
    kept = dedupe_closed_trades([strategy, twin])
    assert len(kept) == 1
    assert kept[0].strategy == "combined"


def test_strategy_closed_trades_drops_orphan_recon():
    rows = [_t(strategy="exchange_reconciliation", pnl=-16.0)]
    assert strategy_closed_trades(rows) == []


def test_find_open_leg_matches_symbol_direction_broker():
    open_rows = [
        _t(status="open", symbol="GBPUSD", direction="SELL", broker="ctrader", pnl=None),
        _t(status="open", symbol="ETHUSDC", direction="SELL", broker="binance_futures", pnl=None),
    ]
    assert find_open_leg(open_rows, symbol="ETHUSDC", direction="SHORT", broker="binance_futures") is open_rows[1]
    assert find_open_leg(open_rows, symbol="GBPUSD", direction="SELL", broker="ctrader") is open_rows[0]
    assert find_open_leg(open_rows, symbol="ETHUSDC", direction="BUY", broker="binance_futures") is None
