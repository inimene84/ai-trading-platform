"""
Unit tests for cTrader Open API Service & Token Lifecycle Manager.
Verifies BrokerService protocol adherence, symbol mapping, lot conversions, and paper simulation.
"""
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from backend.brokers.base import BrokerService
from backend.services.ctrader_service import CTraderProtocol, CTraderService, ctrader_broker
from backend.services.ctrader_tokens import CTraderTokenStore


def test_ctrader_implements_broker_service_protocol():
    """Verify CTraderService strictly satisfies the BrokerService protocol."""
    svc = CTraderService()
    assert isinstance(svc, BrokerService)
    assert hasattr(svc, "connect")
    assert hasattr(svc, "disconnect")
    assert hasattr(svc, "get_balance")
    assert hasattr(svc, "get_positions")
    assert hasattr(svc, "place_order")
    assert hasattr(svc, "cancel_order")
    assert hasattr(svc, "close_position")
    assert hasattr(svc, "status")


def test_symbol_normalization():
    """Test mapping of generic/yfinance tickers to canonical cTrader pairs."""
    svc = CTraderService()
    assert svc._normalize_symbol("EURUSD=X") == "EURUSD"
    assert svc._normalize_symbol("GBPUSD=X") == "GBPUSD"
    assert svc._normalize_symbol("BTC-USD") == "BTCUSD"
    assert svc._normalize_symbol("ETH-USD") == "ETHUSD"
    assert svc._normalize_symbol("USDJPY=X") == "USDJPY"
    assert svc._normalize_symbol("EURUSD") == "EURUSD"


def test_paper_order_execution():
    """Verify order placement in default paper/dry-run mode."""
    svc = CTraderService()
    assert svc.dry_run is True

    result = svc.place_order(
        symbol="EURUSD=X",
        direction="BUY",
        volume=0.5,
        price=1.0850,
        stop_loss=1.0800,
        take_profit=1.0950,
    )
    assert result["status"] == "simulated"
    assert result["broker"] == "ctrader:paper"
    assert result["symbol"] == "EURUSD"
    assert result["direction"] == "BUY"
    assert result["quantity"] == 0.5
    assert "order_id" in result


def test_get_balance_structure():
    """Verify balance and equity dictionary structure."""
    svc = CTraderService()
    svc.balance = 50000.0
    svc.equity = 50250.0
    svc.margin = 1200.0

    balance = svc.get_balance()
    assert balance["balance"] == 50000.0
    assert balance["equity"] == 50250.0
    assert balance["margin_used"] == 1200.0
    assert balance["available"] == 48800.0
    assert balance["broker"] == "ctrader"


def test_positions_tracking_and_close():
    """Verify position reporting and closing."""
    svc = CTraderService()
    svc._positions = [
        {
            "symbol": "EURUSD",
            "side": "BUY",
            "quantity": 1.0,
            "entry_price": 1.0850,
            "unrealized_pnl": 150.0,
            "position_id": "pos_123",
            "broker": "ctrader",
        }
    ]

    positions = svc.get_positions()
    assert len(positions) == 1
    assert positions[0]["symbol"] == "EURUSD"
    assert positions[0]["position_id"] == "pos_123"

    close_res = svc.close_position("pos_123")
    assert close_res["status"] == "closed"
    assert len(svc.get_positions()) == 0


def test_cancel_order_paper():
    """Verify order cancellation in paper mode."""
    svc = CTraderService()
    res = svc.cancel_order("ord_456")
    assert res["success"] is True
    assert res["order_id"] == "ord_456"


def test_lots_to_protocol_volume_uses_spotware_cents():
    """0.01 lot must be 100_000 protocol units (cents), not 1_000."""
    assert CTraderService.lots_to_protocol_volume(0.01) == 100_000
    assert CTraderService.lots_to_protocol_volume(0.10) == 1_000_000
    assert CTraderService.lots_to_protocol_volume(1.0) == 10_000_000
    assert CTraderService.protocol_volume_to_lots(100_000) == pytest.approx(0.01)


def test_relative_stop_units_align_to_symbol_digits():
    """Gold 2-digit prices must snap to 1000-unit steps; EURUSD 5-digit to 1."""
    eurusd = CTraderService.relative_stop_units(1.16161, 1.16061, 5)  # 10 pips = 0.0010
    assert eurusd == 100
    gold = CTraderService.relative_stop_units(2500.88, 2500.00, 2)
    assert gold == 88000
    assert gold % 1000 == 0


def test_decode_spotware_price_always_divides_by_100000():
    """JPY digits=3 must not be used as the protocol scale (that produced 18558 USDJPY)."""
    assert CTraderService.decode_spotware_price(9_454_000, "NZDJPY") == pytest.approx(94.54)
    assert CTraderService.decode_spotware_price(15_420_000, "USDJPY") == pytest.approx(154.2)
    assert CTraderService.decode_spotware_price(116_172, "EURUSD") == pytest.approx(1.16172)


def test_nzdjpy_pip_and_digits_are_jpy_not_eurusd_defaults():
    assert CTraderService.digits_for("NZDJPY") == 3
    assert CTraderService.pip_size_for("NZDJPY") == Decimal("0.01")
    spec = CTraderService().get_symbol_specification("NZDJPY")
    assert spec["digits"] == 3
    assert spec["pip_size"] == 0.01
    assert spec["base_price"] == pytest.approx(94.5)


def test_clamp_nzdjpy_take_profit_rejects_historically_impossible_level():
    """Reproduce IC Markets NZDJPY TP 142.020 (~4748 pips) and cap it."""
    sl, tp = CTraderService.clamp_protective_prices("NZDJPY", 94.540, 70.800, 142.020)
    max_dist = CTraderService.max_protective_distance("NZDJPY", 94.540)
    assert max_dist == pytest.approx(max(1.20, 94.540 * 0.012))
    assert tp == pytest.approx(94.540 + max_dist)
    assert sl == pytest.approx(94.540 - max_dist)
    assert tp < 100.0
    assert sl > 90.0


def test_merge_symbol_catalog_drops_default_ids_reused_by_broker():
    defaults = {"XAUUSD": 21, "XAGUSD": 22, "EURUSD": 1, "GBPUSD": 2}
    catalog = {"NZDJPY": 21, "USDNOK": 22, "EURUSD": 101}
    merged = CTraderService.merge_symbol_catalog(defaults, catalog)
    assert merged["NZDJPY"] == 21
    assert merged["USDNOK"] == 22
    assert merged["EURUSD"] == 101
    assert merged["GBPUSD"] == 2
    assert "XAUUSD" not in merged
    assert "XAGUSD" not in merged


def test_merge_symbol_catalog_normalizes_slashed_broker_names():
    catalog = {
        CTraderService.normalize_catalog_name("NZD/JPY"): 21,
        CTraderService.normalize_catalog_name("EUR/USD"): 101,
    }
    assert "NZDJPY" in catalog
    merged = CTraderService.merge_symbol_catalog(dict(CTraderService.DEFAULT_SYMBOL_IDS), catalog)
    assert merged["NZDJPY"] == 21
    assert merged["EURUSD"] == 101
    assert "XAUUSD" not in merged


def test_symbol_name_for_id_uses_catalog_after_id_collision():
    svc = CTraderService()
    svc._symbol_ids = dict(CTraderService.DEFAULT_SYMBOL_IDS)
    assert svc.symbol_name_for_id(21) == "XAUUSD"
    svc._symbol_ids = CTraderService.merge_symbol_catalog(
        dict(CTraderService.DEFAULT_SYMBOL_IDS),
        {"NZDJPY": 21, "USDNOK": 22, "EURUSD": 101},
    )
    assert svc.symbol_name_for_id(21) == "NZDJPY"
    assert svc.symbol_name_for_id(22) == "USDNOK"
    assert svc.symbol_name_for_id(101) == "EURUSD"
    assert svc.symbol_name_for_id(99999) == "99999"


def test_relabel_cached_symbols_fixes_reconcile_before_catalog():
    svc = CTraderService()
    svc._symbol_ids = dict(CTraderService.DEFAULT_SYMBOL_IDS)
    svc._positions = [
        {
            "symbol": "XAUUSD",
            "symbol_id": 21,
            "side": "BUY",
            "quantity": 0.01,
            "entry_price": 94.54,
            "position_id": "667094381",
            "broker": "ctrader",
        }
    ]
    svc._last_spots = {
        "XAUUSD": {"symbol": "XAUUSD", "symbol_id": 21, "bid": 94.526, "ask": 94.527}
    }
    svc._symbol_ids = CTraderService.merge_symbol_catalog(
        dict(CTraderService.DEFAULT_SYMBOL_IDS),
        {"NZDJPY": 21},
    )
    svc.relabel_cached_symbols()
    assert svc._positions[0]["symbol"] == "NZDJPY"
    assert "NZDJPY" in svc._last_spots
    assert "XAUUSD" not in svc._last_spots


def test_close_position_live_sends_protocol_volume():
    svc = CTraderService()
    svc._dry_run = False
    svc._connected = True
    svc._authenticated = True
    svc._account_id = 46756268
    svc._positions = [
        {
            "symbol": "NZDJPY",
            "side": "BUY",
            "quantity": 0.01,
            "entry_price": 94.54,
            "unrealized_pnl": 0.0,
            "position_id": "667094381",
            "broker": "ctrader",
        }
    ]
    sent: dict = {}

    class FakeProtocol:
        def _send(self, req, ptype):
            sent["ptype"] = ptype
            sent["position_id"] = req.positionId
            sent["volume"] = req.volume
            sent["account_id"] = req.ctidTraderAccountId

    svc._protocol = FakeProtocol()
    reactor = MagicMock()
    reactor.callFromThread.side_effect = lambda fn: fn()
    msgs = MagicMock()
    close_req = MagicMock()
    msgs.ProtoOAClosePositionReq.return_value = close_req
    messages_mod = MagicMock()
    messages_mod.OpenApiMessages_pb2 = msgs
    twisted_mod = MagicMock()
    twisted_internet = MagicMock()
    twisted_internet.reactor = reactor

    with patch.dict(
        "sys.modules",
        {
            "twisted": twisted_mod,
            "twisted.internet": twisted_internet,
            "ctrader_open_api": MagicMock(),
            "ctrader_open_api.messages": messages_mod,
        },
    ):
        res = svc.close_position("667094381")

    assert res["status"] == "sent"
    assert res["symbol"] == "NZDJPY"
    assert sent["ptype"] == 2111
    assert sent["position_id"] == 667094381
    assert sent["volume"] == 100_000
    assert sent["account_id"] == 46756268


def test_close_position_live_errors_without_volume():
    svc = CTraderService()
    svc._dry_run = False
    svc._connected = True
    svc._authenticated = True
    svc._protocol = MagicMock()
    svc._positions = []
    res = svc.close_position("667094381")
    assert res["status"] == "error"
    assert "volume" in res["error"]


def test_status_report():
    """Verify status snapshot contains all required keys."""
    svc = CTraderService()
    st = svc.status()
    assert "connected" in st
    assert "authenticated" in st
    assert "env" in st
    assert "dry_run" in st
    assert "symbols_supported" in st
    assert "balance" in st
    assert "equity" in st


def test_token_store_save_and_load(tmp_path):
    """Verify token store reads and writes persistent token data."""
    store_file = tmp_path / "test_tokens.json"
    store = CTraderTokenStore(file_path=store_file)

    sample_tokens = {
        "access_token": "acc_123",
        "refresh_token": "ref_456",
        "client_id": "cid_789",
        "client_secret": "sec_000",
        "account_id": 46756268,
        "updated_at": 1700000000,
        "expires_in": 2592000,
    }
    store.save_tokens(sample_tokens)

    loaded = store.get_tokens()
    assert loaded["access_token"] == "acc_123"
    assert loaded["refresh_token"] == "ref_456"
    assert loaded["account_id"] == 46756268


def test_get_trendbars_registers_waiter_before_send():
    """Live trendbar fetch must wait on a waiter that exists before the request is sent."""
    svc = CTraderService()
    svc._dry_run = False
    svc._connected = True
    svc._authenticated = True
    order: list[str] = []

    class FakeProtocol:
        def _send(self, req, ptype):
            order.append("send")
            waiter = svc._trendbar_events.get("EURUSD_5")
            assert waiter is not None, "waiter must be registered before send"
            svc._trendbar_cache["EURUSD_5"] = [
                {
                    "timestamp": 1_700_000_000_000,
                    "time": 1_700_000_000,
                    "open": 1.10,
                    "high": 1.12,
                    "low": 1.09,
                    "close": 1.11,
                    "volume": 10.0,
                }
            ]
            waiter.set()
            order.append("acked")

    svc._protocol = FakeProtocol()
    reactor = MagicMock()
    reactor.callFromThread.side_effect = lambda fn: fn()
    msgs = MagicMock()

    twisted_mod = MagicMock()
    twisted_internet = MagicMock()
    twisted_internet.reactor = reactor
    messages_mod = MagicMock()
    messages_mod.OpenApiMessages_pb2 = msgs

    with patch.dict(
        "sys.modules",
        {
            "twisted": twisted_mod,
            "twisted.internet": twisted_internet,
            "ctrader_open_api": MagicMock(),
            "ctrader_open_api.messages": messages_mod,
        },
    ):
        bars = svc.get_trendbars("EURUSD", "M5", count=1)

    assert order == ["send", "acked"]
    assert bars[0]["close"] == 1.11
    assert bars[0]["time"] == 1_700_000_000


def test_get_trendbars_live_timeout_returns_empty_not_synthetic():
    """Live sessions must not fabricate bars when the broker response times out."""
    svc = CTraderService()
    svc._dry_run = False
    svc._connected = True
    svc._authenticated = True
    svc._symbol_ids["EURUSD"] = 1
    svc._trendbar_cache.clear()
    svc._trendbar_last_req.clear()

    class SlowProtocol:
        def _send(self, req, ptype):
            pass  # never signals the waiter → 2s timeout

    svc._protocol = SlowProtocol()
    reactor = MagicMock()
    reactor.callFromThread.side_effect = lambda fn: fn()
    msgs = MagicMock()

    twisted_mod = MagicMock()
    twisted_internet = MagicMock()
    twisted_internet.reactor = reactor
    messages_mod = MagicMock()
    messages_mod.OpenApiMessages_pb2 = msgs

    with patch.dict(
        "sys.modules",
        {
            "twisted": twisted_mod,
            "twisted.internet": twisted_internet,
            "ctrader_open_api": MagicMock(),
            "ctrader_open_api.messages": messages_mod,
        },
    ):
        bars = svc.get_trendbars("EURUSD", "H1", count=40)

    assert bars == []


def test_get_trendbars_live_missing_symbol_id_returns_empty():
    """Live sessions without a broker symbol id must not synthesize bars."""
    svc = CTraderService()
    svc._dry_run = False
    svc._connected = True
    svc._authenticated = True
    svc._protocol = MagicMock()
    svc._symbol_ids.pop("EURUSD", None)
    svc._trendbar_cache.clear()

    bars = svc.get_trendbars("EURUSD", "H1", count=40)
    assert bars == []


def test_trendbar_lookback_is_count_bounded_for_m5():
    """M5 requests must not ask for a 7-day payload."""
    ms = CTraderService.trendbar_lookback_ms(5, 40)
    seven_days = 7 * 86400 * 1000
    assert ms < seven_days
    assert ms >= 40 * 300 * 1000
    assert ms <= (40 + 2) * 300 * 1000


def test_trendbar_lookback_caps_long_periods_at_seven_days():
    ms = CTraderService.trendbar_lookback_ms(11, 500)
    assert ms == 7 * 86400 * 1000


def test_rate_limit_does_not_drop_the_session():
    assert CTraderService.protocol_error_should_disconnect("BLOCKED_PAYLOAD_TYPE", "You are being rate limited") is False
    assert CTraderService.protocol_error_should_disconnect("OA_ERROR", "rate limit exceeded") is False
    assert CTraderService.protocol_error_should_disconnect("CANT_ROUTE_REQUEST", "no route") is True


def test_get_trendbars_live_window_follows_count():
    """The protobuf window is the count lookback, not a flat week."""
    svc = CTraderService()
    svc._dry_run = False
    svc._connected = True
    svc._authenticated = True
    svc._symbol_ids["EURUSD"] = 1
    svc._trendbar_cache.clear()
    svc._trendbar_last_req.clear()
    captured = {}

    class FakeProtocol:
        def _send(self, req, ptype):
            captured["from"] = req.fromTimestamp
            captured["to"] = req.toTimestamp
            waiter = svc._trendbar_events.get("EURUSD_5")
            svc._trendbar_cache["EURUSD_5"] = [{
                "timestamp": 1, "time": 1, "open": 1.0, "high": 1.0,
                "low": 1.0, "close": 1.0, "volume": 1.0,
            }]
            waiter.set()

    svc._protocol = FakeProtocol()
    reactor = MagicMock()
    reactor.callFromThread.side_effect = lambda fn: fn()
    twisted_mod = MagicMock()
    twisted_internet = MagicMock()
    twisted_internet.reactor = reactor
    messages_mod = MagicMock()
    with patch.dict(
        "sys.modules",
        {
            "twisted": twisted_mod,
            "twisted.internet": twisted_internet,
            "ctrader_open_api": MagicMock(),
            "ctrader_open_api.messages": messages_mod,
        },
    ):
        svc.get_trendbars("EURUSD", "M5", count=40)

    span = captured["to"] - captured["from"]
    assert span == CTraderService.trendbar_lookback_ms(5, 40)
    assert span < 7 * 86400 * 1000


def test_pick_trader_account_prefers_configured_id():
    rows = [
        {"id": 111, "is_live": True},
        {"id": 46756268, "is_live": True},
    ]
    assert CTraderService.pick_trader_account_id(rows, 46756268, True) == 46756268


def test_demo_account_is_not_selected_as_live():
    rows = [{"id": 46756268, "is_live": False, "login": 9937385}]
    assert CTraderService.pick_trader_account_id(rows, 46756268, True) == 46756268


def test_pick_trader_account_falls_back_to_live_flag():
    rows = [
        {"id": 11, "is_live": False},
        {"id": 22, "is_live": True},
    ]
    assert CTraderService.pick_trader_account_id(rows, 999, True) == 22
    assert CTraderService.pick_trader_account_id(rows, 999, False) == 11


def test_failed_auth_connection_lost_does_not_reconnect():
    svc = CTraderService()
    svc._authenticated = False
    svc._dry_run = False
    svc._intentional_disconnect = False
    svc._schedule_reconnect = MagicMock()
    proto = CTraderProtocol.__new__(CTraderProtocol)
    proto._service = svc
    proto.connectionLost("auth failed")
    svc._schedule_reconnect.assert_not_called()


def _live_place_order_harness(svc, reactor_side_effect):
    svc._dry_run = False
    svc._connected = True
    svc._authenticated = True
    svc._account_id = 46756268
    svc._symbol_ids["EURUSD"] = 1
    svc.PLACE_ORDER_SEND_TIMEOUT_SEC = 0.05
    svc.PLACE_ORDER_ACK_TIMEOUT_SEC = 0.05
    sent: dict = {}

    class FakeProtocol:
        def _send(self, req, ptype):
            sent["ptype"] = ptype
            sent["symbol_id"] = req.symbolId

    svc._protocol = FakeProtocol()
    reactor = MagicMock()
    reactor.callFromThread.side_effect = reactor_side_effect
    msgs = MagicMock()
    order_req = MagicMock()
    order_req.relativeStopLoss = 0
    order_req.relativeTakeProfit = 0
    msgs.ProtoOANewOrderReq.return_value = order_req
    twisted_mod = MagicMock()
    twisted_internet = MagicMock()
    twisted_internet.reactor = reactor
    messages_mod = MagicMock()
    messages_mod.OpenApiMessages_pb2 = msgs
    return sent, order_req, {
        "twisted": twisted_mod,
        "twisted.internet": twisted_internet,
        "ctrader_open_api": MagicMock(),
        "ctrader_open_api.messages": messages_mod,
    }


def test_place_order_persists_venue_ids_on_per_request_ack(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    svc = CTraderService()

    def run_and_ack(fn):
        fn()
        svc.signal_place_order_ack(order_id="9911", position_id="41416076")

    sent, _req, modules = _live_place_order_harness(svc, run_and_ack)
    with patch.dict("sys.modules", modules):
        res = svc.place_order(
            symbol="EURUSD",
            direction="BUY",
            volume=0.01,
            price=1.1000,
            stop_loss=1.0980,
            take_profit=1.1040,
        )
    assert res["status"] == "sent"
    assert res["order_id"] == "9911"
    assert res["position_id"] == "41416076"
    assert sent["ptype"] == 2106


def test_place_order_timeout_is_sent_not_retryable_error(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    svc = CTraderService()
    _sent, _req, modules = _live_place_order_harness(svc, lambda fn: None)
    with patch.dict("sys.modules", modules):
        res = svc.place_order(
            symbol="EURUSD",
            direction="BUY",
            volume=0.01,
            price=1.1000,
            stop_loss=1.0980,
            take_profit=1.1040,
        )
    assert res["status"] == "sent"
    assert res.get("ack") == "dispatch_timeout_do_not_retry"
    assert "error" not in res or not res.get("error")


def test_signal_place_order_ack_is_per_request():
    svc = CTraderService()
    first = {"event": __import__("threading").Event(), "error": None, "order_id": None, "position_id": None}
    second = {"event": __import__("threading").Event(), "error": None, "order_id": None, "position_id": None}
    svc._order_ack_waiters = [first, second]
    svc.signal_place_order_ack(order_id="1")
    assert first["event"].is_set()
    assert first["order_id"] == "1"
    assert not second["event"].is_set()
    svc.signal_place_order_ack(order_id="2")
    assert second["event"].is_set()
    assert second["order_id"] == "2"


def test_authenticated_drop_does_reconnect():
    svc = CTraderService()
    svc._authenticated = True
    svc._intentional_disconnect = False
    svc._schedule_reconnect = MagicMock()
    proto = CTraderProtocol.__new__(CTraderProtocol)
    proto._service = svc
    proto.connectionLost("socket closed")
    svc._schedule_reconnect.assert_called_once()
