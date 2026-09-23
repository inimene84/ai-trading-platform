"""Exit-price sanity on external-close reconciliation.

Regression tests for a live data-corruption bug: broker.get_exit_price()
returned a bogus price (~0.00075 vs ~$0.08 entries) for 35 ARBUSDT closes and
the sync path trusted it, fabricating ~+$856 of phantom P&L.
"""
import types
from unittest.mock import MagicMock

import pytest

from backend.services.trading_loop_helpers import (
    BrokerPositionSyncService,
    is_plausible_exit_price,
)


def test_plausible_exit_price_accepts_normal_closes():
    assert is_plausible_exit_price(0.08, 0.081)      # tiny move
    assert is_plausible_exit_price(0.08, 0.11)       # +37%
    assert is_plausible_exit_price(1789.27, 1700.0)  # -5%


def test_plausible_exit_price_rejects_corrupt_values():
    # The exact live corruption: ARBUSDT entry ~$0.076, exit 0.000757
    assert not is_plausible_exit_price(0.076, 0.0007570616435259581)
    assert not is_plausible_exit_price(0.08, 0.2)    # +150%
    assert not is_plausible_exit_price(0.08, 0.0)
    assert not is_plausible_exit_price(0.08, None)
    assert not is_plausible_exit_price(0.0, 0.08)
    assert not is_plausible_exit_price(None, 0.08)


def _trade(symbol="ARBUSDT", direction="SELL", entry=0.076, qty=300.0, **extra):
    t = types.SimpleNamespace(
        symbol=symbol, direction=direction, entry_price=entry, quantity=qty,
        exit_price=None, pnl=None, status="open", closed_at=None, notes="",
        binance_order_id="555", broker_position_id=None, mode="live",
        broker="binance_futures", broker_order_id=None,
    )
    for k, v in extra.items():
        setattr(t, k, v)
    return t


def _db_with(trades):
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = trades
    return db


def _broker(exit_price):
    broker = MagicMock()
    # Non-empty, authoritative snapshot where the tested symbol is absent.
    broker.get_positions.return_value = [
        {"symbol": "OTHERUSDT", "quantity": 1.0},
    ]
    broker.get_exit_price.return_value = exit_price
    return broker


@pytest.mark.asyncio
async def test_sync_rejects_corrupt_exit_price():
    """A corrupt broker exit price must not fabricate P&L."""
    trade = _trade()
    db = _db_with([trade])
    broker = _broker(exit_price=0.0007570616435259581)

    updated = await BrokerPositionSyncService.sync_positions(db, broker, {}, {})

    assert updated == 1
    assert trade.status == "closed"
    assert trade.exit_price is None
    assert trade.pnl is None
    assert "implausible" in trade.notes


@pytest.mark.asyncio
async def test_sync_accepts_plausible_exit_price():
    trade = _trade(direction="SELL", entry=0.076, qty=300.0)
    db = _db_with([trade])
    broker = _broker(exit_price=0.074)

    updated = await BrokerPositionSyncService.sync_positions(db, broker, {}, {})

    assert updated == 1
    assert trade.exit_price == 0.074
    # SELL: pnl = (entry - exit) * qty = (0.076 - 0.074) * 300 = 0.6
    assert trade.pnl == pytest.approx(0.6)
    assert "Closed externally (sync)" in trade.notes


@pytest.mark.asyncio
async def test_sync_handles_missing_exit_price():
    trade = _trade()
    db = _db_with([trade])
    broker = _broker(exit_price=None)

    updated = await BrokerPositionSyncService.sync_positions(db, broker, {}, {})

    assert updated == 1
    assert trade.status == "closed"
    assert trade.pnl is None
    assert "unavailable" in trade.notes or "implausible" in trade.notes


@pytest.mark.asyncio
async def test_sync_refuses_bulk_close_on_empty_exchange_snapshot():
    """One empty snapshot must not flatten live venue rows or cancel protection."""
    trade = _trade()
    db = _db_with([trade])
    broker = _broker(exit_price=0.074)
    broker.get_positions.return_value = []

    updated = await BrokerPositionSyncService.sync_positions(db, broker, {}, {})

    assert updated == 0
    assert trade.status == "open"
    assert trade.closed_at is None
    broker.get_exit_price.assert_not_called()
    broker.cancel_all_orders.assert_not_called()
    db.commit.assert_not_called()


@pytest.mark.asyncio
async def test_empty_snapshot_closes_synthetic_binance_key_ghosts():
    """Flat exchange book closes rows keyed as BTCUSDT:LONG (our fill path)."""
    trade = _trade(
        broker_position_id="BTCUSDC:SHORT",
        broker="binance_futures",
        mode="live",
        binance_order_id=None,
    )
    db = _db_with([trade])
    broker = _broker(exit_price=76500.0)
    broker.get_positions.return_value = []

    updated = await BrokerPositionSyncService.sync_positions(db, broker, {}, {})

    assert updated == 1
    assert trade.status == "closed"
    assert "flat book sync" in trade.notes
    broker.cancel_all_orders.assert_not_called()
    db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_empty_snapshot_quarantines_paper_ghost_rows():
    trade = _trade(
        binance_order_id="paper_000007",
        mode="paper",
        broker_position_id=None,
    )
    db = _db_with([trade])
    broker = _broker(exit_price=0.074)
    broker.get_positions.return_value = []

    updated = await BrokerPositionSyncService.sync_positions(db, broker, {}, {})

    assert updated == 1
    assert trade.status == "orphaned"
    assert "Quarantined" in trade.notes
    broker.cancel_all_orders.assert_not_called()
    db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_sync_adopts_and_protects_orphan_exchange_leg():
    db = _db_with([])
    broker = MagicMock()
    broker.get_positions.return_value = [{
        "symbol": "ETHUSDT",
        "side": "BUY",
        "quantity": 0.1,
        "entry_price": 100.0,
    }]
    broker.get_open_orders.return_value = []
    broker.ensure_protective_orders.return_value = {"status": "restored"}

    updated = await BrokerPositionSyncService.sync_positions(db, broker, {}, {})

    assert updated == 1
    adopted = db.add.call_args.args[0]
    assert adopted.symbol == "ETHUSDT"
    assert adopted.status == "filled"
    assert adopted.strategy == "exchange_reconciliation"
    assert adopted.stop_loss == pytest.approx(97.0)
    assert adopted.take_profit == pytest.approx(104.0)
    broker.ensure_protective_orders.assert_called_once_with(
        "ETHUSDT", "BUY", pytest.approx(97.0), pytest.approx(104.0),
    )
    db.commit.assert_called_once()
