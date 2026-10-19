"""Regression: SL-only cTrader amends must not strip the broker TP.

ProtoOAAmendPositionSLTPReq replaces both legs; an omitted takeProfit is
removed at the broker. 2026-10-19: the exit manager trailed GBPAUD SELL
298227173 to SL 1.90827 with tp=None and the broker TP (1.90469) vanished.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from backend.services import ctrader_exit_manager as em
from backend.services.ctrader_service import CTraderService
from backend.services.ctrader_weekend_protect import apply_weekend_protect


def _svc(monkeypatch, pos, db_levels=None, mark=1.90608):
    monkeypatch.setenv("TRADING_MODE", "live")
    svc = CTraderService()
    svc._dry_run = False
    svc._connected = True
    svc._authenticated = True
    svc._protocol = MagicMock()
    svc._positions = [pos] if pos else []
    monkeypatch.setattr(svc, "_orders_live_blocked", lambda: False)
    monkeypatch.setattr(svc, "get_mark_price", lambda *a, **k: mark)
    # Live GBPAUD has 5 digits; the offline default grid is coarser.
    monkeypatch.setattr(svc, "round_price_to_grid", lambda sym, px: px)
    db_calls = []

    def _db(pid):
        db_calls.append(pid)
        return dict(db_levels or {})

    monkeypatch.setattr(CTraderService, "_db_protection_for", staticmethod(_db))
    import twisted.internet.reactor as reactor
    monkeypatch.setattr(reactor, "callFromThread", lambda fn, *a, **k: fn(*a, **k), raising=False)
    return svc, db_calls


def _sent_req(svc):
    assert svc._protocol._send.called, "amend was not sent"
    return svc._protocol._send.call_args.args[0]


def _gbpaud(tp=1.90469, sl=1.91095):
    return {
        "position_id": "298227173", "symbol": "GBPAUD", "side": "SELL",
        "entry_price": 1.90851, "quantity": 0.02, "stop_loss": sl, "take_profit": tp,
    }


def test_sl_only_amend_keeps_broker_tp(monkeypatch):
    svc, db_calls = _svc(monkeypatch, _gbpaud())
    res = svc.amend_position_sltp("298227173", stop_loss=1.90827)
    assert res["status"] == "sent"
    req = _sent_req(svc)
    assert req.stopLoss == pytest.approx(1.90827)
    assert req.HasField("takeProfit")
    assert req.takeProfit == pytest.approx(1.90469)
    assert res["kept"] == {"take_profit": "broker"}
    assert db_calls == []  # broker cache wins, DB not consulted


def test_sl_only_amend_falls_back_to_db_tp(monkeypatch):
    svc, db_calls = _svc(monkeypatch, _gbpaud(tp=None), {"take_profit": 1.90469, "stop_loss": 1.90827})
    res = svc.amend_position_sltp("298227173", stop_loss=1.90827)
    req = _sent_req(svc)
    assert req.takeProfit == pytest.approx(1.90469)
    assert res["kept"] == {"take_profit": "db"}
    assert db_calls == ["298227173"]


def test_db_tp_through_mark_is_not_resent(monkeypatch):
    # SELL TP above the mark would make the broker reject the whole amend.
    svc, _ = _svc(monkeypatch, _gbpaud(tp=None), {"take_profit": 1.90700}, mark=1.90608)
    res = svc.amend_position_sltp("298227173", stop_loss=1.90827)
    req = _sent_req(svc)
    assert not req.HasField("takeProfit")
    assert req.stopLoss == pytest.approx(1.90827)
    assert res["kept"] == {}


def test_explicit_clear_take_profit_still_clears(monkeypatch):
    svc, db_calls = _svc(monkeypatch, _gbpaud(), {"take_profit": 1.90469})
    res = svc.amend_position_sltp("298227173", stop_loss=1.90827, clear_take_profit=True)
    req = _sent_req(svc)
    assert not req.HasField("takeProfit")
    assert res["take_profit"] is None


def test_no_tp_known_behaves_as_before(monkeypatch):
    svc, _ = _svc(monkeypatch, _gbpaud(tp=None), {})
    res = svc.amend_position_sltp("298227173", stop_loss=1.90827)
    req = _sent_req(svc)
    assert not req.HasField("takeProfit")
    assert req.stopLoss == pytest.approx(1.90827)
    assert res["kept"] == {}


def test_tp_only_amend_keeps_broker_sl(monkeypatch):
    svc, _ = _svc(monkeypatch, _gbpaud(sl=1.90827, tp=None))
    res = svc.amend_position_sltp("298227173", take_profit=1.90469)
    req = _sent_req(svc)
    assert req.HasField("stopLoss") and req.stopLoss == pytest.approx(1.90827)
    assert req.takeProfit == pytest.approx(1.90469)
    assert res["kept"] == {"stop_loss": "broker"}


def test_both_legs_sent_unchanged(monkeypatch):
    svc, db_calls = _svc(monkeypatch, _gbpaud())
    res = svc.amend_position_sltp("298227173", stop_loss=1.90827, take_profit=1.90400)
    req = _sent_req(svc)
    assert req.takeProfit == pytest.approx(1.90400)
    assert res["kept"] == {}
    assert db_calls == []


def test_unknown_position_tp_only_keeps_old_behaviour(monkeypatch):
    svc, db_calls = _svc(monkeypatch, None, {"stop_loss": 1.90827})
    res = svc.amend_position_sltp("999", take_profit=1.90469)
    req = _sent_req(svc)
    assert not req.HasField("stopLoss")
    assert res["kept"] == {}
    assert db_calls == []


def test_exit_manager_trail_passes_live_tp(monkeypatch):
    monkeypatch.setattr(em, "BE_TRIGGER_R", 1.0)
    monkeypatch.setattr(em, "BE_LOCK_R", 0.1)
    monkeypatch.setattr(em, "TRAIL_START_R", 1.5)
    entry = 1.10000
    r = CTraderService.min_protective_distance("EURUSD", entry)
    row = SimpleNamespace(entry_price=entry, stop_loss=entry + r, broker_metadata={"r_distance": r})
    pos = {
        "symbol": "EURUSD", "side": "SELL", "position_id": "pos-1",
        "current_price": entry - r * 1.05, "stop_loss": entry + r,
        "entry_price": entry, "take_profit": 1.09000,
    }
    amend = MagicMock(return_value={"status": "sent"})
    with patch.object(em, "_find_trade_row", return_value=row), \
         patch.object(em, "_original_r_distance", return_value=r), \
         patch.object(em.ctrader_service, "amend_position_sltp", amend):
        res = em._manage_position(MagicMock(), pos)
    assert res is not None
    assert amend.call_args.kwargs["take_profit"] == 1.09000


def test_weekend_tighten_passes_live_tp(monkeypatch):
    from backend.tests.unit.test_ctrader_weekend_protect import FRIDAY_20_30

    monkeypatch.setenv("CTRADER_WEEKEND_PROTECT_MODE", "tighten")
    pos = {
        "symbol": "EURUSD", "side": "BUY", "position_id": "p3",
        "entry_price": 1.1000, "current_price": 1.1050, "quantity": 0.01,
        "take_profit": 1.1200,
    }
    broker = MagicMock()
    broker.amend_position_sltp.return_value = {"status": "sent"}
    apply_weekend_protect([pos], broker, now=FRIDAY_20_30)
    broker.amend_position_sltp.assert_called_once_with("p3", stop_loss=1.1000, take_profit=1.1200)


def test_route_rejects_value_plus_clear_and_forwards_flags(monkeypatch):
    from backend.routes import trading as tr
    from backend.services import ctrader_service as cs

    broker = MagicMock()
    broker.is_connected = True
    broker.dry_run = False
    broker.get_positions.return_value = [dict(_gbpaud(), current_price=1.90608)]
    broker.amend_position_sltp.return_value = {"status": "sent"}
    monkeypatch.setattr(cs, "ctrader_broker", broker)

    with pytest.raises(HTTPException) as ei:
        asyncio.run(tr.amend_ctrader_live_position(
            "298227173", tr.ModifyPositionRequest(take_profit=1.90469, clear_take_profit=True)))
    assert ei.value.status_code == 400

    asyncio.run(tr.amend_ctrader_live_position(
        "298227173", tr.ModifyPositionRequest(stop_loss=1.90827)))
    kw = broker.amend_position_sltp.call_args.kwargs
    assert kw["stop_loss"] == 1.90827 and kw["take_profit"] is None
    assert kw["clear_take_profit"] is False and kw["clear_stop_loss"] is False
