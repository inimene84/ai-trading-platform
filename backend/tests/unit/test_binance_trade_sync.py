"""Binance live-book upsert + cTrader classifier must not hide dual-live crypto."""

from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database.models import Base, Trade
from backend.services.binance_trade_sync import upsert_binance_live_trades
from backend.services.ctrader_trade_sync import is_ctrader_trade, overlay_live_mark, reconcile_ctrader_positions
from backend.services.ledger import is_binance_position_key


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_binance_position_key_detection():
    assert is_binance_position_key("BTCUSDT:LONG") is True
    assert is_binance_position_key("AVAXUSDT:SHORT") is True
    assert is_binance_position_key("667118033") is False
    assert is_binance_position_key(None) is False


def test_reconcile_ctrader_does_not_close_live_binance_row():
    db = _session()
    row = Trade(
        symbol="BTCUSDT",
        direction="BUY",
        quantity=0.01,
        entry_price=77320.0,
        status="open",
        broker="binance_futures",
        exchange="binance_futures",
        broker_position_id="BTCUSDT:LONG",
        mode="live",
        strategy="exchange_reconciliation",
        timestamp=datetime.now(timezone.utc),
    )
    db.add(row)
    db.commit()

    res = reconcile_ctrader_positions(db, live_positions=[], broker=None)
    assert res["closed"] == 0
    assert db.query(Trade).one().status == "open"
    assert is_ctrader_trade(row) is False


def test_upsert_revives_falsely_closed_live_binance_row():
    db = _session()
    db.add(
        Trade(
            symbol="AVAXUSDT",
            direction="SELL",
            quantity=46.0,
            entry_price=7.482,
            status="closed",
            broker="binance_futures",
            exchange="binance_futures",
            broker_position_id="AVAXUSDT:SHORT",
            mode="live",
            notes="Adopted orphan | Closed externally (SL/TP hit / closed on cTrader)",
        )
    )
    db.commit()

    res = upsert_binance_live_trades(
        db,
        [{"symbol": "AVAXUSDT", "side": "SELL", "quantity": 46.0, "entry_price": 7.482}],
    )
    assert res["revived"] == 1
    row = db.query(Trade).one()
    assert row.status == "open"
    assert row.closed_at is None
    assert row.mode == "live"
    assert "still live on Binance book" in (row.notes or "")


def test_upsert_creates_missing_live_row_and_skips_paper():
    db = _session()
    db.add(
        Trade(
            symbol="SUIUSDT",
            direction="SELL",
            quantity=481.0,
            entry_price=0.72,
            status="orphaned",
            broker="binance_futures",
            mode="paper",
            binance_order_id="paper_abc",
            notes="Quarantined: absent from live Binance book",
        )
    )
    db.commit()

    res = upsert_binance_live_trades(
        db,
        [
            {"symbol": "DOTUSDT", "side": "SELL", "quantity": 335.6, "entry_price": 1.0393},
        ],
    )
    assert res["created"] == 1
    paper = db.query(Trade).filter(Trade.symbol == "SUIUSDT").one()
    assert paper.status == "orphaned"
    live = db.query(Trade).filter(Trade.symbol == "DOTUSDT").one()
    assert live.status == "open"
    assert live.broker_position_id == "DOTUSDT:SHORT"
    assert live.mode == "live"


def test_upsert_attaches_to_existing_strategy_row_instead_of_twinning():
    db = _session()
    db.add(
        Trade(
            symbol="ETHUSDC",
            direction="SELL",
            quantity=0.02,
            entry_price=4000.0,
            status="open",
            broker="binance_futures",
            exchange="binance_futures",
            strategy="combined",
            mode="live",
        )
    )
    db.commit()

    res = upsert_binance_live_trades(
        db,
        [{"symbol": "ETHUSDC", "side": "SELL", "quantity": 0.02, "entry_price": 4001.0}],
    )
    assert res["created"] == 0
    assert res["updated"] == 1
    assert db.query(Trade).count() == 1
    row = db.query(Trade).one()
    assert row.strategy == "combined"
    assert row.broker_position_id == "ETHUSDC:SHORT"


def test_upsert_aligns_usdc_row_to_live_usdt_symbol():
    db = _session()
    db.add(
        Trade(
            symbol="ETHUSDC",
            direction="SELL",
            quantity=0.02,
            entry_price=4000.0,
            status="open",
            broker="binance_futures",
            exchange="binance_futures",
            strategy="combined",
            mode="live",
        )
    )
    db.commit()

    res = upsert_binance_live_trades(
        db,
        [{"symbol": "ETHUSDT", "side": "SELL", "quantity": 0.02, "entry_price": 4001.0}],
    )
    assert res["created"] == 0
    assert res["updated"] == 1
    row = db.query(Trade).one()
    assert row.symbol == "ETHUSDT"
    assert row.strategy == "combined"
    assert row.broker_position_id == "ETHUSDT:SHORT"


def test_overlay_live_mark_does_not_relabel_binance():
    payload = {
        "symbol": "SOLUSDT",
        "broker": "binance_futures",
        "broker_position_id": "SOLUSDT:LONG",
        "quantity": 0.2,
        "entry_price": 102.97,
    }
    out = overlay_live_mark(payload, {}, {})
    assert out["broker"] == "binance_futures"
