"""Position reconciliation must stay inside the broker it is syncing.

A cTrader forex row is invisible to a Binance positions snapshot. Before the
broker filter, a Binance sync would mark every open FX row "Closed externally"
and cancel orders for symbols Binance has never heard of.
"""

from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database.models import Base, Trade
from backend.services.trading_loop import get_active_broker_name
from backend.services.trading_loop_helpers import BrokerPositionSyncService


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _seed(db):
    db.add_all(
        [
            Trade(
                symbol="EURUSD",
                direction="BUY",
                quantity=0.1,
                entry_price=1.16179,
                status="open",
                broker="ctrader",
                broker_position_id="667118176",
            ),
            Trade(
                symbol="ETHUSDT",
                direction="BUY",
                quantity=0.1,
                entry_price=3000.0,
                status="open",
                broker="binance_futures",
                mode="live",
                binance_order_id="1001",
            ),
        ]
    )
    db.commit()


@pytest.mark.asyncio
async def test_binance_sync_does_not_close_live_ctrader_rows():
    db = _session()
    _seed(db)
    broker = MagicMock()
    # Binance reports one unrelated position; EURUSD simply is not its market.
    broker.get_positions.return_value = [{"symbol": "BTCUSDT", "quantity": 1.0}]
    broker.get_exit_price.return_value = 3000.0

    await BrokerPositionSyncService.sync_positions(
        db, broker, {}, {}, broker_name="binance_futures"
    )

    fx = db.query(Trade).filter(Trade.symbol == "EURUSD").one()
    assert fx.status == "open", "a Binance snapshot must not close a cTrader position"
    assert fx.exit_price is None

    perp = db.query(Trade).filter(Trade.symbol == "ETHUSDT").one()
    assert perp.status == "closed", "Binance rows absent from the snapshot still reconcile"

    # Never cancel orders on a venue that does not own the symbol.
    cancelled = [c.args[0] for c in broker.cancel_all_orders.call_args_list]
    assert "EURUSD" not in cancelled


@pytest.mark.asyncio
async def test_ctrader_sync_only_touches_ctrader_rows():
    db = _session()
    _seed(db)
    broker = MagicMock()
    broker.get_positions.return_value = [{"symbol": "GBPUSD", "quantity": 0.1}]
    broker.get_exit_price.return_value = 1.16179

    await BrokerPositionSyncService.sync_positions(
        db, broker, {}, {}, broker_name="ctrader"
    )

    assert db.query(Trade).filter(Trade.symbol == "ETHUSDT").one().status == "open"
    assert db.query(Trade).filter(Trade.symbol == "EURUSD").one().status == "closed"


@pytest.mark.asyncio
async def test_legacy_null_broker_rows_reconcile_with_binance():
    """Rows written before the broker column existed default to Binance."""
    db = _session()
    db.add(
        Trade(
            symbol="SOLUSDT",
            direction="BUY",
            quantity=1.0,
            entry_price=100.0,
            status="open",
            broker=None,
            mode="live",
            binance_order_id="2002",
        )
    )
    db.commit()
    broker = MagicMock()
    broker.get_positions.return_value = [{"symbol": "BTCUSDT", "quantity": 1.0}]
    broker.get_exit_price.return_value = 101.0

    await BrokerPositionSyncService.sync_positions(
        db, broker, {}, {}, broker_name="binance_futures"
    )

    assert db.query(Trade).filter(Trade.symbol == "SOLUSDT").one().status == "closed"


@pytest.mark.asyncio
async def test_binance_sync_attaches_usdc_row_to_live_usdt_leg():
    """A strategy row on ETHUSDC must stay open when the live book is ETHUSDT."""
    db = _session()
    db.add(
        Trade(
            symbol="ETHUSDC",
            direction="SELL",
            quantity=0.02,
            entry_price=4000.0,
            status="open",
            broker="binance_futures",
            strategy="combined",
            mode="live",
        )
    )
    db.commit()
    broker = MagicMock()
    broker.get_positions.return_value = [
        {"symbol": "ETHUSDT", "side": "SELL", "quantity": 0.02, "entry_price": 4001.0}
    ]

    await BrokerPositionSyncService.sync_positions(
        db, broker, {}, {}, broker_name="binance_futures"
    )

    row = db.query(Trade).one()
    assert row.status == "open"
    assert row.symbol == "ETHUSDT"
    assert row.strategy == "combined"
    assert row.broker_position_id == "ETHUSDT:SHORT"
    broker.cancel_all_orders.assert_not_called()


def test_active_broker_name_matches_selector(monkeypatch):
    from backend.services import trading_loop

    monkeypatch.setenv("ACTIVE_BROKER", "binance_futures")
    assert get_active_broker_name() == "binance_futures"
    assert trading_loop.get_active_broker() is trading_loop.binance_futures_broker

    monkeypatch.setenv("ACTIVE_BROKER", "ctrader")
    assert get_active_broker_name() == "ctrader"
    assert trading_loop.get_active_broker() is trading_loop.ctrader_broker

    # An unset value must resolve the name and the object consistently.
    monkeypatch.delenv("ACTIVE_BROKER", raising=False)
    name = get_active_broker_name()
    resolved = trading_loop.get_active_broker()
    expected = (
        trading_loop.binance_futures_broker
        if name == "binance_futures"
        else trading_loop.ctrader_broker
    )
    assert resolved is expected
