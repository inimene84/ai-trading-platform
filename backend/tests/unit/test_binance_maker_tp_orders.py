"""Maker TP stays TAKE_PROFIT_MARKET unless BINANCE_MAKER_TP_ENABLED."""
import os
from unittest.mock import MagicMock, patch

os.environ["MAKER_ENTRY_ENABLED"] = "false"

from backend.services.binance_futures_service import BinanceFuturesService
from backend.services.binance_maker import maker_tp_params


def _broker():
    svc = BinanceFuturesService.__new__(BinanceFuturesService)
    svc.leverage = 10
    svc.margin_type = "CROSSED"
    svc.dry_run = False
    svc._leverage_set = set()
    svc._lot_step = {}
    svc._lot_min = {}
    svc._qty_precision = {}
    return svc


def test_default_tp_is_take_profit_market():
    broker = _broker()
    client = MagicMock()
    client.futures_account.return_value = {"availableBalance": "1000"}
    created = []

    def capture(_client, params):
        created.append(params)
        return {"orderId": len(created), "avgPrice": "100"}

    with patch.dict(os.environ, {"MAKER_ENTRY_ENABLED": "false", "BINANCE_MAKER_TP_ENABLED": "false"}), \
         patch.object(broker, "get_positions", return_value=[]), \
         patch.object(broker, "_get_client", return_value=client), \
         patch.object(broker, "_setup_symbol"), \
         patch.object(broker, "_round_qty", side_effect=lambda s, q, **kw: q), \
         patch.object(broker, "_round_price", side_effect=lambda s, p: p), \
         patch.object(broker, "_has_exchange_stop", return_value=True), \
         patch.object(broker, "_has_exchange_take_profit", return_value=False), \
         patch.object(broker, "_safe_create_order", side_effect=capture), \
         patch.object(broker, "_native_trailing_enabled", return_value=False), \
         patch("backend.services.binance_futures_service.fetch_commission_rates", return_value={"maker_rate": 0.0002, "taker_rate": 0.0005}), \
         patch("backend.services.binance_futures_service.summarize_fills", return_value={"commission": 0.0, "role": "taker"}):
        result = broker.place_order(
            symbol="BTCUSDC", direction="BUY", quantity=0.01,
            price=100.0, stop_loss=95.0, take_profit=110.0,
        )
    assert result["status"] == "sent"
    tp = [p for p in created if p.get("type") == "TAKE_PROFIT_MARKET"]
    assert tp and tp[0]["stopPrice"] == 110.0
    assert not any(p.get("timeInForce") == "GTX" and p.get("type") == "LIMIT" for p in created)


def test_maker_tp_places_gtx_limit(monkeypatch):
    broker = _broker()
    client = MagicMock()
    created = []

    def capture(_client, params):
        created.append(params)
        return {"orderId": 9}

    monkeypatch.setenv("BINANCE_MAKER_TP_ENABLED", "true")
    with patch.object(broker, "_live_position_qty", return_value=0.01), \
         patch.object(broker, "_round_qty", side_effect=lambda s, q, **kw: q), \
         patch.object(broker, "_round_price", side_effect=lambda s, p: p), \
         patch.object(broker, "_safe_create_order", side_effect=capture):
        broker._place_take_profit_order(client, "BTCUSDC", "SELL", "LONG", 110.0, 0.01)
    assert created[0]["type"] == "LIMIT"
    assert created[0]["timeInForce"] == "GTX"
    assert created[0]["price"] == 110.0
    assert str(created[0]["newClientOrderId"]).startswith("qtpMTP")


def test_maker_tp_fallback_when_gtx_would_take(monkeypatch):
    broker = _broker()
    client = MagicMock()
    created = []

    def capture(_client, params):
        created.append(params)
        if params.get("type") == "LIMIT":
            raise Exception("APIError(code=-5022): Post Only order will be rejected")
        return {"orderId": 11, "algoId": 11}

    monkeypatch.setenv("BINANCE_MAKER_TP_ENABLED", "true")
    with patch.object(broker, "_live_position_qty", return_value=0.01), \
         patch.object(broker, "_round_qty", side_effect=lambda s, q, **kw: q), \
         patch.object(broker, "_round_price", side_effect=lambda s, p: p), \
         patch.object(broker, "_mark_or_last_price", return_value=110.0), \
         patch.object(broker, "_safe_create_order", side_effect=capture):
        broker._place_take_profit_order(client, "BTCUSDC", "SELL", "LONG", 110.0, 0.01)
    assert created[0]["type"] == "LIMIT"
    assert created[1]["type"] == "MARKET"
    assert created[1]["quantity"] == 0.01
    assert created[1]["positionSide"] == "LONG"
    assert not any(p.get("type") == "TAKE_PROFIT_MARKET" for p in created)


def test_reconcile_maker_tps_noops_when_flag_off(monkeypatch):
    monkeypatch.setenv("BINANCE_MAKER_TP_ENABLED", "false")
    broker = _broker()
    assert broker.reconcile_maker_tps()["skipped"] is True


def test_reconcile_markets_through_resting_maker_tp(monkeypatch):
    broker = _broker()
    client = MagicMock()
    monkeypatch.setenv("BINANCE_MAKER_TP_ENABLED", "true")
    resting = {
        "symbol": "BTCUSDC",
        "side": "SELL",
        "type": "LIMIT",
        "price": 110.0,
        "quantity": 0.01,
        "order_id": "77",
        "clientOrderId": "qtpMTPBTCUSDC1",
        "timeInForce": "GTX",
        "updateTime": 1,
    }
    with patch.object(broker, "_get_client", return_value=client), \
         patch.object(broker, "get_open_orders", return_value=[resting]), \
         patch.object(broker, "_mark_or_last_price", return_value=110.5), \
         patch.object(broker, "_live_position_qty", return_value=0.02), \
         patch.object(broker, "_round_qty", side_effect=lambda s, q, **kw: q), \
         patch.object(broker, "_safe_create_order", return_value={"orderId": 8}) as mkt:
        res = broker.reconcile_maker_tps()
    assert res["skipped"] is False
    assert res["fallbacks"][0]["reason"] == "traded_through"
    assert res["fallbacks"][0]["qty"] == 0.02
    client.futures_cancel_order.assert_called_once()
    flatten = mkt.call_args.args[1]
    assert flatten["type"] == "MARKET"
    assert flatten["quantity"] == 0.02
    assert flatten["positionSide"] == "LONG"


def test_reconcile_does_not_flatten_on_timeout_alone(monkeypatch):
    broker = _broker()
    client = MagicMock()
    monkeypatch.setenv("BINANCE_MAKER_TP_ENABLED", "true")
    monkeypatch.setenv("BINANCE_MAKER_TP_TIMEOUT_SEC", "1")
    resting = {
        "symbol": "BTCUSDC",
        "side": "SELL",
        "type": "LIMIT",
        "price": 110.0,
        "quantity": 0.01,
        "order_id": "77",
        "clientOrderId": "qtpMTPBTCUSDC1",
        "timeInForce": "GTX",
        "updateTime": 1,
    }
    with patch.object(broker, "_get_client", return_value=client), \
         patch.object(broker, "get_open_orders", return_value=[resting]), \
         patch.object(broker, "_mark_or_last_price", return_value=105.0), \
         patch.object(broker, "_safe_create_order") as mkt:
        res = broker.reconcile_maker_tps()
    assert res["skipped"] is False
    assert res["fallbacks"] == []
    client.futures_cancel_order.assert_not_called()
    mkt.assert_not_called()


def test_reconcile_skips_market_when_cancel_fails(monkeypatch):
    broker = _broker()
    client = MagicMock()
    client.futures_cancel_order.side_effect = Exception("cancel rejected")
    monkeypatch.setenv("BINANCE_MAKER_TP_ENABLED", "true")
    resting = {
        "symbol": "BTCUSDC",
        "side": "SELL",
        "type": "LIMIT",
        "price": 110.0,
        "quantity": 0.01,
        "order_id": "77",
        "clientOrderId": "qtpMTPBTCUSDC1",
        "timeInForce": "GTX",
        "updateTime": 1,
    }
    with patch.object(broker, "_get_client", return_value=client), \
         patch.object(broker, "get_open_orders", return_value=[resting]), \
         patch.object(broker, "_mark_or_last_price", return_value=110.5), \
         patch.object(broker, "_safe_create_order") as mkt:
        res = broker.reconcile_maker_tps()
    assert res["fallbacks"] == []
    mkt.assert_not_called()


def test_maker_tp_params_helper_shape():
    params = maker_tp_params(
        futures_sym="ETHUSDC", close_side="BUY", position_side="SHORT",
        quantity=0.1, tp_price=2000.0,
    )
    assert params["positionSide"] == "SHORT"
    assert params["side"] == "BUY"
