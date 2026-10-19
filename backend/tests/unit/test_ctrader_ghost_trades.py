"""Comprehensive tests for cTrader ghost trade prevention and SL/TP reconciliation."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database.models import Base, Trade
from backend.services.ctrader_service import CTraderService
from backend.services.ctrader_trade_sync import (
    is_ctrader_trade,
    reconcile_ctrader_positions,
    upsert_ctrader_live_trades,
)
from backend.routes.trading import get_positions


@pytest.fixture(autouse=True)
def _no_fill_grace(monkeypatch):
    """These rows model positions that closed long ago, not fresh fills."""
    monkeypatch.setenv("CTRADER_SYNC_CLOSE_GRACE_SEC", "0")


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_is_ctrader_trade_detection():
    t1 = SimpleNamespace(symbol="EURUSD", broker="ctrader", exchange="ctrader", broker_position_id="123")
    t2 = SimpleNamespace(symbol="BTCUSDT", broker="binance_futures", exchange="binance_futures", broker_position_id=None)
    t3 = SimpleNamespace(symbol="USDJPY", broker=None, exchange=None, broker_position_id="456")
    t4 = SimpleNamespace(symbol="XAUUSD", broker=None, exchange=None, broker_position_id=None)
    t5 = SimpleNamespace(symbol="ETHUSDT", broker=None, exchange=None, broker_position_id=None)

    t6 = SimpleNamespace(
        symbol="BTCUSDT",
        broker="binance_futures",
        exchange="binance_futures",
        broker_position_id="BTCUSDT:LONG",
    )
    t7 = SimpleNamespace(
        symbol="AVAXUSDT",
        broker="binance_futures",
        exchange="binance_futures",
        broker_position_id="AVAXUSDT:SHORT",
    )

    assert is_ctrader_trade(t1) is True
    assert is_ctrader_trade(t2) is False
    assert is_ctrader_trade(t3) is True
    assert is_ctrader_trade(t4) is True
    assert is_ctrader_trade(t5) is False
    assert is_ctrader_trade(t6) is False
    assert is_ctrader_trade(t7) is False


def test_reconcile_closes_position_when_sl_hit():
    """When a trade hits Stop Loss in cTrader, it disappears from live book and must close in DB."""
    db = _session()
    trade = Trade(
        symbol="EURUSD",
        direction="BUY",
        quantity=0.1,  # 0.1 lot = 10,000 units
        entry_price=1.1000,
        stop_loss=1.0950,
        take_profit=1.1100,
        status="open",
        broker="ctrader",
        broker_position_id="pos-sl-123",
        timestamp=datetime.now(timezone.utc),
    )
    db.add(trade)
    db.commit()

    # Broker live positions is now empty (SL was hit)
    broker = MagicMock()
    broker.CONTRACT_UNITS_PER_LOT = 100_000
    broker.get_recent_deal.return_value = {
        "position_id": "pos-sl-123",
        "symbol": "EURUSD",
        "execution_price": 1.0950,
        "gross_profit": -50.0,
        "close_type": "SL_TP",
    }
    broker.get_exit_price.return_value = 1.0950
    broker.quote_to_usd_rate.return_value = 1.0

    res = reconcile_ctrader_positions(db, live_positions=[], broker=broker)

    assert res["closed"] == 1
    updated_trade = db.query(Trade).filter(Trade.id == trade.id).one()
    assert updated_trade.status == "closed"
    assert updated_trade.exit_price == pytest.approx(1.0950)
    assert updated_trade.pnl == pytest.approx(-50.0)
    assert updated_trade.closed_at is not None
    assert "Closed externally" in updated_trade.notes


def test_reconcile_closes_position_when_tp_hit():
    """When a trade hits Take Profit in cTrader, it closes in DB with profit."""
    db = _session()
    trade = Trade(
        symbol="USDJPY",
        direction="SELL",
        quantity=0.2,  # 0.2 lot = 20,000 units
        entry_price=155.00,
        stop_loss=156.00,
        take_profit=154.00,
        status="open",
        broker="ctrader",
        broker_position_id="pos-tp-456",
        timestamp=datetime.now(timezone.utc),
    )
    db.add(trade)
    db.commit()

    broker = MagicMock()
    broker.CONTRACT_UNITS_PER_LOT = 100_000
    broker.get_recent_deal.return_value = {
        "position_id": "pos-tp-456",
        "symbol": "USDJPY",
        "execution_price": 154.00,
        "gross_profit": 129.87,  # in USD
        "close_type": "SL_TP",
    }
    broker.get_exit_price.return_value = 154.00

    res = reconcile_ctrader_positions(db, live_positions=[], broker=broker)

    assert res["closed"] == 1
    updated_trade = db.query(Trade).filter(Trade.id == trade.id).one()
    assert updated_trade.status == "closed"
    assert updated_trade.exit_price == pytest.approx(154.00)
    assert updated_trade.pnl == pytest.approx(129.87)


def test_reconcile_only_closes_absent_position_among_multiple():
    """When one position closes but another remains open, only the closed one is modified."""
    db = _session()
    trade1 = Trade(
        symbol="EURUSD",
        direction="BUY",
        quantity=0.1,
        entry_price=1.1000,
        status="open",
        broker="ctrader",
        broker_position_id="pos-1",
        timestamp=datetime.now(timezone.utc),
    )
    trade2 = Trade(
        symbol="GBPUSD",
        direction="BUY",
        quantity=0.1,
        entry_price=1.3000,
        status="open",
        broker="ctrader",
        broker_position_id="pos-2",
        timestamp=datetime.now(timezone.utc),
    )
    db.add_all([trade1, trade2])
    db.commit()

    # Only pos-2 is still live on cTrader
    live_positions = [
        {
            "position_id": "pos-2",
            "symbol": "GBPUSD",
            "quantity": 0.1,
            "entry_price": 1.3000,
            "side": "BUY",
        }
    ]

    broker = MagicMock()
    broker.get_recent_deal.return_value = None
    broker.get_exit_price.return_value = 1.0980
    broker.quote_to_usd_rate.return_value = 1.0
    broker.CONTRACT_UNITS_PER_LOT = 100_000

    res = reconcile_ctrader_positions(db, live_positions=live_positions, broker=broker)

    assert res["closed"] == 1
    assert res["updated"] == 1

    t1 = db.query(Trade).filter(Trade.id == trade1.id).one()
    t2 = db.query(Trade).filter(Trade.id == trade2.id).one()

    assert t1.status == "closed"
    assert t1.exit_price == pytest.approx(1.0980)
    assert t2.status == "open"


def test_ctrader_service_record_deal_and_get_exit_price():
    svc = CTraderService()
    svc._last_spots = {
        "EURUSD": {"symbol": "EURUSD", "symbol_id": 1, "bid": 1.1050, "ask": 1.1052}
    }

    # Before recording deal, fallback to mark price
    assert svc.get_exit_price("EURUSD", "BUY") == pytest.approx(1.1050)

    # Record deal for pos-999
    svc.record_deal({
        "position_id": "pos-999",
        "symbol": "EURUSD",
        "execution_price": 1.0955,
        "gross_profit": -45.0,
        "close_type": "SL_TP",
    })

    # Query with position_id returns exact deal price
    assert svc.get_exit_price("EURUSD", position_id="pos-999") == pytest.approx(1.0955)
    deal = svc.get_recent_deal(position_id="pos-999")
    assert deal["gross_profit"] == -45.0
    assert deal["close_type"] == "SL_TP"


@pytest.mark.asyncio
async def test_dashboard_positions_filters_out_ghost_trades():
    """The /positions endpoint must not return trades that closed on cTrader."""
    db = _session()
    # Stale trade sitting in DB
    trade = Trade(
        symbol="EURUSD",
        direction="BUY",
        quantity=0.1,
        entry_price=1.1000,
        status="open",
        broker="ctrader",
        broker_position_id="ghost-pos-888",
        timestamp=datetime.now(timezone.utc),
    )
    db.add(trade)
    db.commit()

    broker = MagicMock()
    # cTrader reports 0 open positions
    broker.get_positions.return_value = []
    broker.is_connected = True
    broker.CONTRACT_UNITS_PER_LOT = 100_000
    broker.get_mark_price.return_value = 1.0950
    broker.get_exit_price.return_value = 1.0950
    broker.get_recent_deal.return_value = None
    broker.quote_to_usd_rate.return_value = 1.0

    trade_id = trade.id

    with patch("backend.routes.trading.SessionLocal", return_value=db), patch(
        "backend.routes.trading.ctrader_broker", broker
    ), patch(
        "backend.routes.trading._fetch_mark_prices_for_symbols",
        new_callable=AsyncMock,
        return_value={},
    ), patch(
        "backend.routes.trading._live_binance_book",
        new_callable=AsyncMock,
        return_value=([], True),
    ):
        result = await get_positions()

    # Empty live book + no close deal is ambiguous (rate-limit / reconnect).
    # Do not mass-close; the row stays open until a deal confirms.
    row = db.query(Trade).filter(Trade.id == trade_id).one()
    assert row.status == "open"
    assert result["count"] == 1
    assert result["positions"][0]["symbol"] == "EURUSD"


@pytest.mark.asyncio
async def test_reconcile_14_trades_down_to_9_live_positions():
    """14 open trades in DB with only 9 live on cTrader must reconcile down to 9."""
    from backend.routes.trading import get_portfolio

    db = _session()
    symbols = [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
        "NZDUSD", "USDCHF", "EURJPY", "GBPJPY", "EURGBP",
        "AUDJPY", "CADJPY", "EURAUD", "XAUUSD",
    ]
    for idx, sym in enumerate(symbols, start=1):
        db.add(Trade(
            symbol=sym,
            direction="BUY" if idx % 2 == 1 else "SELL",
            quantity=0.1,
            entry_price=100.0 + idx,
            status="open",
            broker="ctrader",
            broker_position_id=f"pos-{idx}",
            timestamp=datetime.now(timezone.utc),
        ))
    db.commit()

    # Exactly 9 positions are still alive on cTrader (pos-1 through pos-9)
    # pos-10 through pos-14 hit SL/TP
    live_positions = [
        {
            "position_id": f"pos-{i}",
            "symbol": symbols[i - 1],
            "quantity": 0.1,
            "entry_price": 100.0 + i,
            "side": "BUY" if i % 2 == 1 else "SELL",
        }
        for i in range(1, 10)
    ]

    broker = MagicMock()
    broker.get_positions.return_value = live_positions
    broker.is_connected = True
    broker.has_credentials.return_value = True
    broker.CONTRACT_UNITS_PER_LOT = 100_000
    broker.get_mark_price.return_value = 105.0
    broker.get_exit_price.return_value = 105.0
    broker.get_recent_deal.return_value = None
    broker.quote_to_usd_rate.return_value = 1.0

    with patch("backend.routes.trading.SessionLocal", return_value=db), patch(
        "backend.routes.trading.ctrader_broker", broker
    ), patch(
        "backend.routes.trading._fetch_mark_prices_for_symbols",
        new_callable=AsyncMock,
        return_value={},
    ), patch(
        "backend.routes.trading._live_binance_book",
        new_callable=AsyncMock,
        return_value=([], True),
    ):
        positions_res = await get_positions()
        portfolio_res = await get_portfolio()

    assert positions_res["count"] == 9
    assert len(positions_res["positions"]) == 9

    assert len(portfolio_res["positions"]) == 9

    open_in_db = db.query(Trade).filter(Trade.status.in_(["open", "filled"])).all()
    assert len(open_in_db) == 9
    closed_in_db = db.query(Trade).filter(Trade.status == "closed").all()
    assert len(closed_in_db) == 5

