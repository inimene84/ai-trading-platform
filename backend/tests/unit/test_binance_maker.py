from backend.services.binance_maker import (
    book_walked_away,
    gtx_rejected_as_taker,
    is_maker_tp_order,
    maker_close_enabled,
    maker_requote_enabled,
    maker_tp_enabled,
    maker_tp_market_flatten_params,
    maker_tp_params,
    price_traded_through_tp,
    take_profit_market_params,
)


def test_maker_flags_default_off(monkeypatch):
    monkeypatch.delenv("BINANCE_MAKER_TP_ENABLED", raising=False)
    monkeypatch.delenv("BINANCE_MAKER_CLOSE_ENABLED", raising=False)
    monkeypatch.delenv("BINANCE_MAKER_REQUOTE_ENABLED", raising=False)
    assert maker_tp_enabled() is False
    assert maker_close_enabled() is False
    assert maker_requote_enabled() is False


def test_price_traded_through_long_and_short_tp():
    assert price_traded_through_tp(side="SELL", tp_price=100.0, last_price=100.0) is True
    assert price_traded_through_tp(side="SELL", tp_price=100.0, last_price=99.9) is False
    assert price_traded_through_tp(side="BUY", tp_price=100.0, last_price=99.9) is True
    assert price_traded_through_tp(side="BUY", tp_price=100.0, last_price=100.1) is False


def test_book_walked_away_buy_and_sell():
    assert book_walked_away(side="BUY", rest_price=100.0, best_bid=100.1, best_ask=100.2, tick=0.1)
    assert not book_walked_away(side="BUY", rest_price=100.0, best_bid=100.0, best_ask=100.1, tick=0.1)
    assert book_walked_away(side="SELL", rest_price=100.0, best_bid=99.8, best_ask=99.9, tick=0.1)


def test_gtx_rejection_codes():
    assert gtx_rejected_as_taker(Exception("APIError(code=-5022): Post Only"))
    assert gtx_rejected_as_taker("-2021 would immediately match")
    assert not gtx_rejected_as_taker("APIError(code=-2010)")


def test_maker_tp_params_are_gtx_limit_not_market():
    params = maker_tp_params(
        futures_sym="BTCUSDC",
        close_side="SELL",
        position_side="LONG",
        quantity=0.01,
        tp_price=70000.0,
        client_order_id="qtpMTPBTCUSDC1",
    )
    assert params["type"] == "LIMIT"
    assert params["timeInForce"] == "GTX"
    assert params["price"] == 70000.0
    assert "closePosition" not in params


def test_take_profit_market_params_unchanged_shape():
    params = take_profit_market_params(
        futures_sym="BTCUSDC",
        close_side="SELL",
        position_side="LONG",
        tp_price=70000.0,
    )
    assert params["type"] == "TAKE_PROFIT_MARKET"
    assert params["closePosition"] == "true"


def test_maker_tp_market_flatten_is_market_not_tpm():
    params = maker_tp_market_flatten_params(
        futures_sym="BTCUSDC",
        close_side="SELL",
        position_side="LONG",
        quantity=0.01,
    )
    assert params["type"] == "MARKET"
    assert params["quantity"] == 0.01
    assert params["positionSide"] == "LONG"
    assert "stopPrice" not in params
    assert "closePosition" not in params


def test_is_maker_tp_order_by_client_id_and_gtx():
    assert is_maker_tp_order({"clientOrderId": "qtpMTPBTCUSDC1", "type": "LIMIT"})
    assert is_maker_tp_order({"type": "LIMIT", "timeInForce": "GTX", "reduceOnly": True})
    assert not is_maker_tp_order({"type": "LIMIT", "timeInForce": "GTC"})
    assert not is_maker_tp_order({"type": "TAKE_PROFIT_MARKET"})
