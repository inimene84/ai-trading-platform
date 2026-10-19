"""False 'Closed externally' race: a fresh fill must not be closed by a stale snapshot."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database.models import Base, Trade
from backend.services.ctrader_trade_sync import ctrader_close_grace_sec, reconcile_ctrader_positions


class FakeBroker:
    def __init__(self, deal=None, positions=None):
        self.deal = deal
        self.positions = positions or []
        self.deal_calls = 0

    def get_recent_deal(self, position_id=None, symbol=None):
        self.deal_calls += 1
        return self.deal

    def get_positions(self):
        return list(self.positions)

    def get_exit_price(self, *a, **k):
        return None

    def quote_to_usd_rate(self, *a, **k):
        return 1.0


def _db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


OTHER = {"symbol": "USDCAD", "side": "SELL", "quantity": 0.04, "entry_price": 1.41, "position_id": "1"}


def _row(db, age_sec, pid="298071890", symbol="EURAUD"):
    t = Trade(
        symbol=symbol, direction="BUY", quantity=0.01, entry_price=1.6107, status="open",
        broker="ctrader", exchange="ctrader", strategy="MOMENTUM_TREND_PULSE",
        broker_position_id=pid,
        timestamp=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=age_sec),
    )
    db.add(t)
    db.commit()
    return t


@pytest.fixture
def cooldown_spy():
    with patch(
        "backend.services.signal_candidate_engine.signal_candidate_engine.set_symbol_cooldown"
    ) as spy:
        yield spy


def test_fresh_fill_absent_from_stale_snapshot_stays_open(cooldown_spy):
    db = _db()
    row = _row(db, age_sec=1)
    res = reconcile_ctrader_positions(db, live_positions=[OTHER], broker=FakeBroker())
    db.refresh(row)
    assert row.status == "open"
    assert res["closed"] == 0
    assert "Closed externally" not in (row.notes or "")
    cooldown_spy.assert_not_called()


def test_fresh_fill_with_confirmed_close_deal_is_closed(cooldown_spy):
    db = _db()
    row = _row(db, age_sec=5)
    deal = {"closed_volume": 100000, "execution_price": 1.607, "gross_profit": -0.25, "close_type": "SL"}
    res = reconcile_ctrader_positions(db, live_positions=[OTHER], broker=FakeBroker(deal=deal))
    db.refresh(row)
    assert row.status == "closed"
    assert res["closed"] == 1


def test_old_row_absent_everywhere_is_closed(cooldown_spy):
    db = _db()
    row = _row(db, age_sec=600)
    res = reconcile_ctrader_positions(db, live_positions=[OTHER], broker=FakeBroker())
    db.refresh(row)
    assert row.status == "closed"
    assert res["closed"] == 1


def test_old_row_still_in_refreshed_book_is_not_closed(cooldown_spy):
    db = _db()
    row = _row(db, age_sec=600)
    live_now = [OTHER, {"symbol": "EURAUD", "side": "BUY", "quantity": 0.01, "entry_price": 1.6106, "position_id": "298071890"}]
    res = reconcile_ctrader_positions(db, live_positions=[OTHER], broker=FakeBroker(positions=live_now))
    db.refresh(row)
    assert row.status == "open"
    assert res["closed"] == 0
    cooldown_spy.assert_not_called()


def test_opening_fill_deal_does_not_confirm_close(cooldown_spy):
    db = _db()
    row = _row(db, age_sec=2)
    opening = {"closed_volume": 0, "execution_price": 1.6106}
    reconcile_ctrader_positions(db, live_positions=[OTHER], broker=FakeBroker(deal=opening))
    db.refresh(row)
    assert row.status == "open"


def test_grace_env(monkeypatch):
    monkeypatch.delenv("CTRADER_SYNC_CLOSE_GRACE_SEC", raising=False)
    assert ctrader_close_grace_sec() == 90.0
    monkeypatch.setenv("CTRADER_SYNC_CLOSE_GRACE_SEC", "0")
    assert ctrader_close_grace_sec() == 0.0
    monkeypatch.setenv("CTRADER_SYNC_CLOSE_GRACE_SEC", "bad")
    assert ctrader_close_grace_sec() == 90.0


# --- pytest production-DB guard ----------------------------------------

def test_app_engine_is_the_temp_test_db():
    from backend.database import connection

    url = str(connection.engine.url)
    assert "pytest-db-" in url
    assert "/app/data/hedge_fund.db" not in url


def test_guard_refuses_production_sqlite_file():
    from sqlalchemy import text

    eng = create_engine("sqlite:////app/data/hedge_fund.db")
    with pytest.raises(Exception) as exc:
        with eng.connect() as conn:
            conn.execute(text("select 1"))
    assert "pytest DB guard" in str(exc.value)
    eng.dispose()
