"""Regression tests for bugs found in the full-codebase audit.

Each test pins a behaviour that previously produced a wrong trade rather than
an error, which is why none of them were caught by the existing suite.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.binance_futures_service import BinanceFuturesService
from backend.services.ctrader_service import CTraderService
from backend.services.multi_asset_bars import fetch_bars, tf_to_binance_interval
from backend.services.signal_candidate_engine import signal_candidate_engine
from backend.services.trading_loop_helpers import EmergencyExitManager


def _bars(n, base=1.10, step=0.0001):
    return [
        {
            "close": base + i * step,
            "high": base + i * step + 0.00005,
            "low": base + i * step - 0.00005,
            "volume": 100,
        }
        for i in range(n)
    ]


# ── Fabricated indicators on empty data ──────────────────────────────────────

def test_features_return_none_instead_of_placeholder_indicators():
    """Empty data used to yield price 1.0 / RSI 50 / volatility 0.1%."""
    assert signal_candidate_engine._compute_features([]) is None
    assert signal_candidate_engine._compute_features(None) is None
    assert signal_candidate_engine._compute_features(_bars(5)) is None
    assert signal_candidate_engine._compute_features(_bars(25)) is not None


def test_placeholder_features_would_have_fired_a_straddle():
    """Proves the old defaults were inside the straddle trigger window."""
    stale_placeholder = {
        "last_close": 1.0,
        "atr": 0.0010,
        "rsi": 50.0,
        "trend": "NEUTRAL",
        "volatility_pct": 0.1,
        "recent_high": 1.0,
        "recent_low": 1.0,
    }
    fired = signal_candidate_engine._evaluate_straddle(
        "EURUSD", stale_placeholder, "ctrader"
    )
    assert fired is not None, "placeholder features sit inside the straddle window"
    # Which is exactly why _compute_features must never return them.
    assert signal_candidate_engine._compute_features([]) is None


@pytest.mark.asyncio
async def test_scan_markets_skips_symbols_without_enough_bars():
    with patch(
        "backend.services.signal_candidate_engine.ctrader_service.get_trendbars",
        return_value=[],
    ):
        await signal_candidate_engine.scan_markets(universe=["EURUSD"], timeframe="M5")

    fabricated = [
        c
        for c in signal_candidate_engine.candidates.values()
        if c.get("symbol") == "EURUSD" and c.get("entry_price") == 1.0
    ]
    assert not fabricated, "a symbol with no bars must not produce a candidate"


# ── Inverted stop geometry on short macro trades ─────────────────────────────

@pytest.mark.asyncio
async def test_macro_sell_candidate_has_stop_above_entry():
    """A short with its stop below entry never caps the loss."""
    # Volatility must clear the broker minimum stop (10 pips) after the
    # 1.2x ATR multiplier, else the geometry veto correctly rejects the trade.
    bearish = [
        {"close": 1.10 - i * 0.001, "high": 1.10 - i * 0.001 + 0.0002,
         "low": 1.10 - i * 0.001 - 0.0002, "volume": 100}
        for i in range(30)
    ]
    calendar = {"events": [{"event": "NFP", "currency": "USD", "impact": "high"}]}
    previous = dict(signal_candidate_engine.candidates)
    signal_candidate_engine.candidates.clear()

    with patch(
        "backend.services.signal_candidate_engine.ctrader_service.get_trendbars",
        return_value=bearish,
    ), patch(
        "backend.routes.news.get_economic_calendar",
        new=AsyncMock(return_value=calendar),
    ), patch(
        "backend.routes.news.get_news_feed", new=AsyncMock(return_value={})
    ), patch(
        "backend.routes.news.get_market_sentiment", new=AsyncMock(return_value={})
    ), patch.object(
        signal_candidate_engine, "_has_open_position", return_value=False
    ), patch.object(
        signal_candidate_engine, "_portfolio_risk_breach", return_value=None
    ):
        try:
            created = await signal_candidate_engine.scan_news_and_events()
        finally:
            signal_candidate_engine.candidates = previous

    assert created, "a falling market should produce a macro candidate"
    for c in created:
        # CTRADER_INVERT_SIDE default ON: strategy SELL is stored as BUY.
        # Geometry must still match the *stored* side.
        if c["direction"] == "SELL":
            assert c["stop_loss"] > c["entry_price"], "SELL stop must sit above entry"
            assert c["take_profit"] < c["entry_price"], "SELL target must sit below entry"
        else:
            assert c["direction"] == "BUY"
            assert c.get("original_direction") == "SELL"
            assert c["stop_loss"] < c["entry_price"], "BUY stop must sit below entry"
            assert c["take_profit"] > c["entry_price"], "BUY target must sit above entry"


# ── Binance / cTrader timeframe mapping ──────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_bars_maps_ctrader_timeframe_for_binance():
    """Binance rejects 'M5'; the empty result looked like a flat market."""
    market = MagicMock()
    market.get_klines = AsyncMock(return_value=[])
    with patch("backend.services.binance_market_data.binance_market_data", market):
        await fetch_bars("BTCUSDT", timeframe="M5", limit=10)

    assert market.get_klines.call_args.kwargs["interval"] == "5m"
    assert tf_to_binance_interval("M5") == "5m"
    assert tf_to_binance_interval("H1") == "1h"


# ── Closes that used to open positions ───────────────────────────────────────

def test_ctrader_close_action_does_not_open_a_new_position():
    svc = CTraderService()
    svc._positions = [
        {
            "symbol": "EURUSD",
            "side": "BUY",
            "quantity": 0.1,
            "entry_price": 1.16,
            "position_id": "667118176",
        }
    ]
    with patch.object(svc, "close_position", return_value={"status": "sent"}) as closer:
        res = svc.place_order(symbol="EURUSD", direction="BUY", action="close", quantity=0.1)

    closer.assert_called_once()
    assert closer.call_args.kwargs["position_id"] == "667118176"
    assert res["status"] == "sent"


def test_ctrader_close_action_on_flat_book_is_a_noop():
    svc = CTraderService()
    svc._positions = []
    res = svc.place_order(symbol="EURUSD", direction="BUY", action="close", quantity=0.1)
    assert res["status"] == "already_flat"


def test_binance_close_position_targets_the_held_side():
    """Hardcoding SELL aimed every close at the SHORT leg."""
    svc = BinanceFuturesService.__new__(BinanceFuturesService)
    svc._to_futures_symbol = lambda s: s
    svc.get_positions = lambda *a, **k: [
        {"symbol": "ETHUSDT", "side": "LONG", "quantity": 0.5}
    ]
    with patch.object(svc, "_live_position_qty", return_value=0.5), \
         patch.object(svc, "place_order", return_value={"status": "sent"}) as order:
        svc.close_position(symbol="ETHUSDT")

    assert order.call_args.kwargs["direction"] == "BUY", "closing a long must target the LONG leg"
    assert order.call_args.kwargs["quantity"] == 0.5

    with patch.object(svc, "_live_position_qty", return_value=0.5), \
         patch.object(svc, "place_order", return_value={"status": "sent"}) as order:
        svc.close_position(symbol="ETHUSDT", direction="SHORT")
    assert order.call_args.kwargs["direction"] == "SELL"


def _emergency_env(monkeypatch):
    monkeypatch.setenv("ACTIVE_BROKER", "binance_futures")
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("PAPER_TRADING", "false")
    monkeypatch.setenv("DRY_RUN_ALL", "false")
    monkeypatch.setenv("BINANCE_DRY_RUN", "false")


def test_emergency_halt_leaves_row_open_when_close_is_rejected(monkeypatch):
    """Marking a row closed after a rejected order hides live exchange risk."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from backend.database.models import Base, Trade
    from backend.services import sentry_emergency

    _emergency_env(monkeypatch)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(
        Trade(
            symbol="ETHUSDT", direction="BUY", quantity=0.1, entry_price=3000.0,
            status="open", broker="binance_futures",
        )
    )
    db.commit()

    broker = MagicMock()
    broker.place_order.return_value = {"status": "error", "message": "margin insufficient"}
    broker.get_positions.return_value = []

    monkeypatch.setattr(
        "backend.database.connection.SessionLocal", lambda: db, raising=False
    )
    monkeypatch.setattr(
        "backend.services.binance_futures_service.binance_futures_broker", broker
    )
    result = sentry_emergency._close_all_positions_sync()

    row = db.query(Trade).one()
    assert row.status == "open", "a rejected close must not mark the DB row closed"
    assert result["closed_trades"] == 0
    assert any("margin insufficient" in e for e in result["errors"])


def test_emergency_halt_does_not_send_forex_rows_to_binance(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from backend.database.models import Base, Trade
    from backend.services import sentry_emergency

    _emergency_env(monkeypatch)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(
        Trade(
            symbol="EURUSD", direction="BUY", quantity=0.1, entry_price=1.16,
            status="open", broker="ctrader",
        )
    )
    db.commit()

    broker = MagicMock()
    broker.place_order.return_value = {"status": "sent"}
    broker.get_positions.return_value = []

    monkeypatch.setattr(
        "backend.database.connection.SessionLocal", lambda: db, raising=False
    )
    monkeypatch.setattr(
        "backend.services.binance_futures_service.binance_futures_broker", broker
    )
    sentry_emergency._close_all_positions_sync()

    broker.place_order.assert_not_called()
    assert db.query(Trade).one().status == "open"


def test_binance_close_position_on_flat_book_does_not_order():
    svc = BinanceFuturesService.__new__(BinanceFuturesService)
    svc._to_futures_symbol = lambda s: s
    svc.get_positions = lambda *a, **k: []
    with patch.object(svc, "place_order") as order:
        res = svc.close_position(symbol="ETHUSDT")
    order.assert_not_called()
    assert res["status"] == "already_flat"


def test_binance_place_order_skips_forex_instead_of_inventing_usdt_pair():
    """EURUSD used to become EURUSDUSDT and hit the Binance API."""
    svc = BinanceFuturesService.__new__(BinanceFuturesService)
    svc.dry_run = False
    res = svc.place_order(symbol="EURUSD", direction="BUY", quantity=0.1)
    assert res["status"] == "skipped"
    assert "EURUSDUSDT" not in str(res)


@pytest.mark.asyncio
async def test_emergency_exit_skips_ctrader_rows():
    fx = SimpleNamespace(
        symbol="EURUSD",
        direction="BUY",
        quantity=0.1,
        entry_price=1.16,
        broker="ctrader",
        notes="",
        status="open",
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [fx]
    broker = MagicMock()
    broker.get_positions.return_value = [
        {"symbol": "EURUSD", "mark_price": 1.10, "entry_price": 1.16},
    ]

    with patch(
        "backend.services.trading_loop_helpers.UnifiedTrading"
    ) as ut_cls, patch(
        "backend.services.trading_loop_helpers.get_position_manager"
    ) as pm_fn:
        pm_fn.return_value.emergency_drawdown_pct = -1.0
        closed = await EmergencyExitManager.run_emergency_exits(
            db, broker, pyramid_layers={}, sl_cooldown={},
        )

    assert closed == 0
    ut_cls.return_value.place_order.assert_not_called()
    assert fx.status == "open"
