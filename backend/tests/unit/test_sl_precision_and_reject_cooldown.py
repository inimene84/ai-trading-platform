"""SL/TP must sit on the broker price grid; permanent rejects must not loop."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import backend.services.ctrader_service as cts
from backend.services.ctrader_service import CTraderService
from backend.services.signal_candidate_engine import (
    CandidateStatus,
    TimingMode,
    is_permanent_broker_reject,
    signal_candidate_engine,
    signal_reject_cooldown_sec,
)


@pytest.fixture
def live_specs(monkeypatch):
    specs: dict = {}
    monkeypatch.setattr(cts.ctrader_service, "_symbol_specs", specs)
    return specs


@pytest.fixture(autouse=True)
def isolate_ctrader_db_positions(monkeypatch):
    mod = "backend.services.signal_candidate_engine."
    monkeypatch.setattr(mod + "open_ctrader_db_symbols", lambda: set())
    monkeypatch.setattr(mod + "count_open_ctrader_db_trades", lambda: 0)
    monkeypatch.setattr(mod + "open_ctrader_db_positions", lambda: [])
    monkeypatch.setattr(mod + "try_open_ctrader_db_positions", lambda: [])


def _modules(reactor, msgs):
    twisted_internet = MagicMock()
    twisted_internet.reactor = reactor
    messages_mod = MagicMock()
    messages_mod.OpenApiMessages_pb2 = msgs
    return {
        "twisted": MagicMock(),
        "twisted.internet": twisted_internet,
        "ctrader_open_api": MagicMock(),
        "ctrader_open_api.messages": messages_mod,
    }


def _live_svc(symbol, symbol_id, protocol):
    svc = CTraderService()
    svc._dry_run = False
    svc._connected = True
    svc._authenticated = True
    svc._account_id = 49222055
    svc._symbol_ids[symbol] = symbol_id
    svc.PLACE_ORDER_SEND_TIMEOUT_SEC = 0.05
    svc.PLACE_ORDER_ACK_TIMEOUT_SEC = 0.05
    svc._protocol = protocol
    return svc


def _order_msgs():
    msgs = MagicMock()
    order_req = MagicMock()
    order_req.relativeStopLoss = 0
    order_req.relativeTakeProfit = 0
    msgs.ProtoOANewOrderReq.return_value = order_req
    return msgs, order_req


class _BareProtocol:
    def _send(self, req, ptype):
        pass


# --- grid helpers -------------------------------------------------------

def test_order_price_digits_prefers_live_spec(live_specs):
    live_specs["EURAUD"] = {"digits": 4, "pip_position": 4}
    assert CTraderService.order_price_digits("EURAUD") == 4
    assert CTraderService.relative_price_step_units(4) == 10


def test_order_price_digits_without_spec_is_whole_pip(live_specs):
    # The generic 5-digit fallback sent sub-pip stops (151 units) that
    # Spotware rejected; without a live spec only whole pips are safe.
    assert CTraderService.order_price_digits("EURAUD") == 4
    assert CTraderService.order_price_digits("GBPJPY") == 2
    assert CTraderService.relative_price_step_units(2) == 1000


def test_relative_units_euraud_case_is_on_grid(live_specs):
    live_specs["EURAUD"] = {"digits": 4, "pip_position": 4}
    digits = CTraderService.order_price_digits("EURAUD")
    units = CTraderService.relative_stop_units(1.6114, 1.61291, digits)
    assert units == 150
    assert units % CTraderService.relative_price_step_units(digits) == 0


def test_round_price_to_grid(live_specs):
    live_specs["EURAUD"] = {"digits": 4, "pip_position": 4}
    assert CTraderService.round_price_to_grid("EURAUD", 1.61291) == pytest.approx(1.6129)
    live_specs["USDJPY"] = {"digits": 3, "pip_position": 2}
    assert CTraderService.round_price_to_grid("USDJPY", 150.12345) == pytest.approx(150.123)
    assert CTraderService.round_price_to_grid("USDJPY", None) is None


# --- place_order --------------------------------------------------------

def test_place_order_rounds_relative_sl_tp_to_live_digits(monkeypatch, live_specs):
    monkeypatch.setenv("TRADING_MODE", "live")
    live_specs["EURAUD"] = {"digits": 4, "pip_position": 4}
    svc = _live_svc("EURAUD", 14, _BareProtocol())
    msgs, order_req = _order_msgs()
    reactor = MagicMock()
    reactor.callFromThread.side_effect = lambda fn, *a: fn(*a)
    with patch.dict("sys.modules", _modules(reactor, msgs)):
        svc.place_order(
            symbol="EURAUD", direction="SELL", volume=0.03,
            price=1.6114, stop_loss=1.61291, take_profit=1.5887,
        )
    assert order_req.relativeStopLoss > 0
    assert order_req.relativeStopLoss % 10 == 0
    assert order_req.relativeTakeProfit % 10 == 0


def test_place_order_without_spec_falls_back_to_whole_pips(monkeypatch, live_specs):
    monkeypatch.setenv("TRADING_MODE", "live")
    svc = _live_svc("EURAUD", 14, _BareProtocol())  # cannot fetch a spec
    msgs, order_req = _order_msgs()
    reactor = MagicMock()
    reactor.callFromThread.side_effect = lambda fn, *a: fn(*a)
    with patch.dict("sys.modules", _modules(reactor, msgs)):
        svc.place_order(
            symbol="EURAUD", direction="SELL", volume=0.03,
            price=1.6114, stop_loss=1.61291, take_profit=1.5887,
        )
    assert order_req.relativeStopLoss % 10 == 0
    assert order_req.relativeTakeProfit % 10 == 0


def test_ensure_symbol_spec_fetches_on_demand(live_specs):
    requested = []

    class Proto(_BareProtocol):
        def _request_symbol_specs(self, ids):
            requested.extend(ids)
            live_specs["EURAUD"] = {"digits": 4, "pip_position": 4}

    svc = _live_svc("EURAUD", 14, Proto())
    reactor = MagicMock()
    reactor.callFromThread.side_effect = lambda fn, *a: fn(*a)
    with patch.dict("sys.modules", _modules(reactor, MagicMock())):
        assert svc.ensure_symbol_spec("EURAUD", 14, timeout=0.5) is True
    assert requested == [14]
    # Already loaded: no second request.
    with patch.dict("sys.modules", _modules(reactor, MagicMock())):
        assert svc.ensure_symbol_spec("EURAUD", 14, timeout=0.5) is True
    assert requested == [14]


def test_ensure_symbol_spec_times_out_quietly(live_specs):
    class Proto(_BareProtocol):
        def _request_symbol_specs(self, ids):
            pass

    svc = _live_svc("EURAUD", 14, Proto())
    reactor = MagicMock()
    reactor.callFromThread.side_effect = lambda fn, *a: fn(*a)
    with patch.dict("sys.modules", _modules(reactor, MagicMock())):
        t0 = time.time()
        assert svc.ensure_symbol_spec("EURAUD", 14, timeout=0.2) is False
        assert time.time() - t0 < 1.0


# --- permanent reject cooldown -----------------------------------------

def test_permanent_reject_classification():
    assert is_permanent_broker_reject("INVALID_REQUEST — Relative stop loss has invalid precision")
    assert is_permanent_broker_reject("cTrader order rejected: TRADING_BAD_STOPS")
    assert not is_permanent_broker_reject("MARKET_CLOSED")
    assert not is_permanent_broker_reject("dispatch timeout")
    assert not is_permanent_broker_reject("")


def test_reject_cooldown_env(monkeypatch):
    monkeypatch.delenv("SIGNAL_REJECT_COOLDOWN_SEC", raising=False)
    assert signal_reject_cooldown_sec() == 900
    monkeypatch.setenv("SIGNAL_REJECT_COOLDOWN_SEC", "120")
    assert signal_reject_cooldown_sec() == 120
    monkeypatch.setenv("SIGNAL_REJECT_COOLDOWN_SEC", "junk")
    assert signal_reject_cooldown_sec() == 900


@pytest.mark.asyncio
async def test_permanent_reject_pauses_candidate(monkeypatch):
    monkeypatch.setenv("SIGNAL_REJECT_COOLDOWN_SEC", "600")
    now_ts = int(time.time())
    sig_id = "test-slp-reject-01"
    previous = signal_candidate_engine.candidates
    signal_candidate_engine.candidates = {}
    signal_candidate_engine.candidates[sig_id] = {
        "id": sig_id,
        "symbol": "BTCUSDT",
        "broker": "binance_futures",
        "strategy": "MOMENTUM_TREND_PULSE",
        "direction": "BUY",
        "entry_price": 0,
        "stop_loss": 90000,
        "take_profit": 110000,
        "timing_mode": TimingMode.BAR_CLOSE,
        "status": CandidateStatus.READY,
        "earliest_exec_at": now_ts - 5,
        "latest_exec_at": now_ts + 600,
        "sizing": {"lots": 0.1, "quantity": 0.01, "risk_usd": 50.0},
    }
    resp = MagicMock(
        success=False, order_id=None,
        message="INVALID_REQUEST — Relative stop loss has invalid precision",
    )
    try:
        with patch.dict(
            signal_candidate_engine.execution_config,
            {"forex_only": False, "max_portfolio_risk_pct": 0},
        ), patch.object(
            signal_candidate_engine, "_resolve_mark_price", new_callable=AsyncMock, return_value=0.0
        ), patch.object(
            signal_candidate_engine, "_try_open_binance_positions", return_value=[]
        ), patch("backend.services.signal_candidate_engine.UnifiedTrading") as ut_cls:
            ut_cls.return_value.list_sessions.return_value = [
                {"id": "binance_futures_paper", "broker": "binance_futures", "mode": "paper"}
            ]
            ut_cls.return_value.place_order.return_value = resp
            res = await signal_candidate_engine.execute_candidate(sig_id, force=True)
            assert res["success"] is False
            cand = signal_candidate_engine.candidates[sig_id]
            assert cand["reject_count"] == 1
            assert cand["reject_cooldown_until"] >= now_ts + 590
            assert cand["status"] == CandidateStatus.READY

            ready_ids = {c["id"] for c in signal_candidate_engine.get_ready_signals(now_ts)}
            assert sig_id not in ready_ids

            again = await signal_candidate_engine.execute_candidate(sig_id)
            assert again["success"] is False
            assert "REJECT_COOLDOWN" in again["error"]
            assert ut_cls.return_value.place_order.call_count == 1

            # Cooldown elapsed: eligible again.
            cand["reject_cooldown_until"] = now_ts - 1
            ready_ids = {c["id"] for c in signal_candidate_engine.get_ready_signals(now_ts)}
            assert sig_id in ready_ids
    finally:
        signal_candidate_engine.candidates = previous


@pytest.mark.asyncio
async def test_transient_failure_does_not_pause_candidate():
    now_ts = int(time.time())
    sig_id = "test-slp-transient-01"
    previous = signal_candidate_engine.candidates
    signal_candidate_engine.candidates = {}
    signal_candidate_engine.candidates[sig_id] = {
        "id": sig_id, "symbol": "BTCUSDT", "broker": "binance_futures",
        "strategy": "MOMENTUM_TREND_PULSE", "direction": "BUY", "entry_price": 0,
        "stop_loss": 90000, "take_profit": 110000, "timing_mode": TimingMode.BAR_CLOSE,
        "status": CandidateStatus.READY, "earliest_exec_at": now_ts - 5,
        "latest_exec_at": now_ts + 600,
        "sizing": {"lots": 0.1, "quantity": 0.01, "risk_usd": 50.0},
    }
    resp = MagicMock(success=False, order_id=None, message="Paper market entry requires an explicit price")
    try:
        with patch.dict(
            signal_candidate_engine.execution_config,
            {"forex_only": False, "max_portfolio_risk_pct": 0},
        ), patch.object(
            signal_candidate_engine, "_resolve_mark_price", new_callable=AsyncMock, return_value=0.0
        ), patch.object(
            signal_candidate_engine, "_try_open_binance_positions", return_value=[]
        ), patch("backend.services.signal_candidate_engine.UnifiedTrading") as ut_cls:
            ut_cls.return_value.list_sessions.return_value = [
                {"id": "binance_futures_paper", "broker": "binance_futures", "mode": "paper"}
            ]
            ut_cls.return_value.place_order.return_value = resp
            await signal_candidate_engine.execute_candidate(sig_id, force=True)
        assert not signal_candidate_engine.candidates[sig_id].get("reject_cooldown_until")
    finally:
        signal_candidate_engine.candidates = previous
