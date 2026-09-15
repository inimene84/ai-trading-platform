"""Trading loop safety fixes: status balance, pyramid reconstruct, live SL/TP skip."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.services.trading_loop import TradingLoopService
from backend.services.trading_loop_helpers import remove_closed_pyramid_layer
from backend.services.trading_mode import TradingMode


def _trade(symbol, trade_id, entry_price, notes=None):
    return SimpleNamespace(
        symbol=symbol, id=trade_id, entry_price=entry_price, notes=notes
    )


def test_pyramid_prices_from_trades_ignores_base_entry():
    trades = [
        _trade("DOGEUSDT", 1, 0.10, notes=None),
        _trade("DOGEUSDT", 2, 0.11, notes="pyramid_layer_1"),
        _trade("DOGEUSDT", 3, 0.12, notes="pyramid_layer_2"),
    ]
    layers = TradingLoopService._pyramid_prices_from_trades(trades)
    assert layers == {"DOGEUSDT": [0.11, 0.12]}


def test_pyramid_prices_legacy_multi_row_without_notes():
    trades = [
        _trade("ARBUSDT", 10, 1.0, notes=None),
        _trade("ARBUSDT", 11, 1.05, notes=None),
    ]
    layers = TradingLoopService._pyramid_prices_from_trades(trades)
    assert layers == {"ARBUSDT": [1.05]}


def test_pyramid_prices_single_base_not_counted():
    trades = [_trade("ETHUSDT", 5, 3000.0, notes=None)]
    layers = TradingLoopService._pyramid_prices_from_trades(trades)
    assert layers == {}


def test_closing_one_pyramid_row_removes_only_that_layer():
    layers = {"ETHUSDT": [101.0, 102.0, 103.0]}
    closed = _trade(
        "ETHUSDT", 2, 102.0, notes="pyramid_layer_2",
    )
    remove_closed_pyramid_layer(layers, closed)
    assert layers == {"ETHUSDT": [101.0, 103.0]}


def test_closing_base_row_does_not_clear_pyramid_history():
    layers = {"ETHUSDT": [101.0, 102.0]}
    base = _trade("ETHUSDT", 1, 100.0, notes=None)
    remove_closed_pyramid_layer(layers, base)
    assert layers == {"ETHUSDT": [101.0, 102.0]}


def test_chunk_symbols_splits_into_batches():
    symbols = [f"S{i}USDT" for i in range(20)]
    batches = TradingLoopService._chunk_symbols(symbols, 5)
    assert len(batches) == 4
    assert all(len(b) == 5 for b in batches)
    assert [s for batch in batches for s in batch] == symbols


def test_chunk_symbols_handles_remainder():
    symbols = ["AUSDT", "BUSDT", "CUSDT", "DUSDT", "EUSDT", "FUSDT", "GUSDT"]
    batches = TradingLoopService._chunk_symbols(symbols, 3)
    assert batches == [
        ["AUSDT", "BUSDT", "CUSDT"],
        ["DUSDT", "EUSDT", "FUSDT"],
        ["GUSDT"],
    ]


def test_status_uses_broker_balance():
    loop = TradingLoopService()
    mock_balance = {
        "available": 44.0,
        "equity": 91.5,
        "margin_used": 47.0,
        "balance": 91.5,
        "broker": "ctrader",
    }
    with patch(
        "backend.services.trading_loop.get_trading_mode",
        return_value=TradingMode.LIVE,
    ), patch.object(
        loop, "_is_live_binance", return_value=False,
    ), patch(
        "backend.services.trading_loop.ctrader_broker.get_balance",
        return_value=mock_balance,
    ), patch(
        "backend.services.trading_loop.get_active_broker_name",
        return_value="ctrader",
    ):
        st = loop.status
    assert st["cash"] == 44.0
    assert st["equity"] == 91.5
    assert st["margin_used"] == 47.0
    assert st["split_book"] is False
    assert st["risk_broker"] in {"ctrader", "cTrader"}


def test_status_uses_binance_equity_on_split_book(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("ACTIVE_BROKER", "ctrader")
    monkeypatch.setenv("CTRADER_ENV", "sandbox")
    monkeypatch.setenv("CTRADER_PAPER_MODE", "false")
    monkeypatch.setenv("BINANCE_TESTNET", "false")
    monkeypatch.setenv("BINANCE_PAPER_PARALLEL", "false")
    loop = TradingLoopService()
    with patch(
        "backend.services.trading_loop.ctrader_broker.get_balance",
        return_value={"broker": "ctrader", "equity": 150.0, "balance": 150.0, "available": 150.0},
    ), patch(
        "backend.services.trading_loop.binance_futures_broker.get_balance",
        return_value={
            "broker": "binance_futures",
            "equity": 918.0,
            "balance": 900.0,
            "available": 800.0,
        },
    ):
        st = loop.status
    assert st["equity"] == 918.0
    assert st["split_book"] is True
    assert st["risk_broker"] == "binance_futures"


def test_status_uses_paper_portfolio_in_paper_mode():
    loop = TradingLoopService()
    mock_broker = MagicMock()
    mock_broker.get_balance.return_value = {
        "available": 44.0,
        "equity": 91.5,
        "margin_used": 47.0,
    }
    with patch(
        "backend.services.trading_loop.get_trading_mode",
        return_value=TradingMode.PAPER,
    ), patch(
        "backend.services.trading_loop.get_active_broker",
        return_value=mock_broker,
    ), patch(
        "backend.services.unified_trading.UnifiedTrading.get_paper_portfolio",
        return_value={"cash": 12.0, "equity": 15.0, "margin_used": 3.0},
    ):
        st = loop.status
    assert st["cash"] == 12.0
    assert st["equity"] == 15.0
    assert st["margin_used"] == 3.0
    mock_broker.get_balance.assert_not_called()


def test_check_sl_tp_skips_software_close_on_live_binance():
    loop = TradingLoopService()
    db = MagicMock()
    bars = [{"close": 100.0}]

    with patch.object(loop, "_ensure_exchange_protection"), \
         patch.object(loop, "_apply_trailing_stop"), \
         patch.object(loop, "_apply_partial_tp_live") as live_ptp, \
         patch.object(loop, "_apply_partial_tp") as paper_ptp, \
         patch.object(loop, "_is_live_binance", return_value=True), \
         patch("backend.services.trading_loop.UnifiedTrading") as ut_cls:
        loop._check_sl_tp(db, "BTCUSDT", bars)
        ut_cls.assert_not_called()  # no software full close
        live_ptp.assert_called_once()
        paper_ptp.assert_not_called()


def test_check_sl_tp_runs_software_close_in_paper_mode():
    loop = TradingLoopService()
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = []
    bars = [{"close": 100.0}]

    with patch.object(loop, "_ensure_exchange_protection"), \
         patch.object(loop, "_apply_trailing_stop"), \
         patch.object(loop, "_is_live_binance", return_value=False):
        loop._check_sl_tp(db, "BTCUSDT", bars)
    db.query.assert_called()
