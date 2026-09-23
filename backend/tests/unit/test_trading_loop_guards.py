"""Expanded safety coverage for TradingLoopService.

Covers entry-bar gating, market-hours, symbol conversion, price extraction,
the symbol-quality gate (blacklist + liquidity floor + fail-open), the
expectancy gate, SL cooldown reconstruction, and SL/TP hit detection + PnL.
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backend.services.trading_loop import TradingLoopService


def _fake_risk_config(**overrides):
    """A plain mutable stand-in for the pydantic RiskConfig — several of its
    attributes (symbol_blacklist, min_24h_quote_volume_usdt, expectancy_*) are
    read-only @property getters, so they can't be reassigned on the real model
    in tests. This namespace exposes the same surface, mutably."""
    base = dict(
        symbol_blacklist=set(),
        min_24h_quote_volume_usdt=0,
        symbol_expectancy_gate_enabled=False,
        symbol_expectancy_lookback_days=30,
        symbol_expectancy_min_trades=3,
        sl_cooldown_minutes=30,
        opinion_close_cooldown_min=30,
        eval_on_new_bar_only=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _loop(risk=None):
    loop = TradingLoopService()
    loop._pyramid_mode = False  # set lazily at runtime; not in __init__
    if risk is not None:
        loop.risk_config = risk
    return loop


def test_regime_detector_is_reused_per_symbol():
    loop = _loop()
    eth_a = loop._regime_detector_for("ETHUSDT")
    eth_b = loop._regime_detector_for("ETHUSDT")
    btc = loop._regime_detector_for("BTCUSDT")
    assert eth_a is eth_b
    assert eth_a is not btc
    eth_a._history.append("TRENDING")
    assert eth_b._history[-1] == "TRENDING"
    assert btc._history == []


# --------------------------------------------------------------------------- #
# _should_evaluate_bar — entry pipeline runs at most once per bar
# --------------------------------------------------------------------------- #
def test_should_evaluate_bar_returns_true_for_new_bar():
    loop = _loop()
    bars = [{"date": "2026-08-28T00:00:00", "close": 100.0}]
    assert loop._should_evaluate_bar("BTCUSDT", bars) is True
    # Same bar again -> already evaluated -> False
    assert loop._should_evaluate_bar("BTCUSDT", bars) is False


def test_should_evaluate_bar_fails_open_when_no_timestamp():
    loop = _loop()
    # Bars without a 'date' key must NOT block trading (fail-open).
    assert loop._should_evaluate_bar("BTCUSDT", [{"close": 100.0}]) is True
    assert loop._should_evaluate_bar("BTCUSDT", []) is True


def test_entry_bar_interval_follows_loop_cadence():
    loop = _loop()
    loop._interval_minutes = 5
    assert loop._entry_bar_interval() == "5m"
    loop._interval_minutes = 15
    assert loop._entry_bar_interval() == "15m"
    loop._interval_minutes = 60
    assert loop._entry_bar_interval() == "1h"


def test_bar_interval_max_age_scales_with_timeframe(monkeypatch):
    monkeypatch.delenv("WS_CANDLE_MAX_AGE_SEC", raising=False)
    assert TradingLoopService._bar_interval_max_age_sec("5m") == 750.0
    assert TradingLoopService._bar_interval_max_age_sec("15m") == 2250.0
    assert TradingLoopService._bar_interval_max_age_sec("1h") == 9000.0


def test_should_evaluate_bar_disabled_gate_always_true():
    loop = _loop(_fake_risk_config(eval_on_new_bar_only=False))
    bars = [{"date": "2026-08-28T00:00:00", "close": 100.0}]
    assert loop._should_evaluate_bar("BTCUSDT", bars) is True
    assert loop._should_evaluate_bar("BTCUSDT", bars) is True


# --------------------------------------------------------------------------- #
# _is_market_open — crypto is always open (deterministic branch)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("symbol", [
    "BTCUSDT", "ETHUSDC", "BNBBUSD", "BTC-USD", "ETH-USD", "SOL-USD",
])
def test_is_market_open_crypto_always_open(symbol):
    assert _loop()._is_market_open(symbol) is True


# --------------------------------------------------------------------------- #
# _to_yfinance_symbol — Binance-native -> yfinance format
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("symbol,expected", [
    ("BTCUSDT", "BTC-USD"),
    ("ETHUSDC", "ETH-USD"),
    ("BNBBUSD", "BNB-USD"),
    ("BTC-USD", "BTC-USD"),   # already yfinance format -> passthrough
    ("foo", "FOO"),           # unknown -> uppercased, NOT suffixed
])
def test_to_yfinance_symbol(symbol, expected):
    assert TradingLoopService._to_yfinance_symbol(symbol) == expected


# --------------------------------------------------------------------------- #
# _get_current_price — last close, empty -> None
# --------------------------------------------------------------------------- #
def test_get_current_price_returns_last_close():
    loop = _loop()
    bars = [{"close": 100.0}, {"close": 105.5}]
    assert loop._get_current_price(bars) == 105.5


def test_get_current_price_empty_bars_returns_none():
    loop = _loop()
    assert loop._get_current_price([]) is None


# --------------------------------------------------------------------------- #
# _filter_tradeable_symbols — blacklist + liquidity floor + fail-open
# --------------------------------------------------------------------------- #
def _no_open_positions_db():
    """A SessionLocal() that returns no open positions (empty query chain)."""
    db = MagicMock()
    q = MagicMock()
    q.filter.return_value.distinct.return_value.all.return_value = []
    db.query.return_value = q
    db.close.return_value = None
    return db


def _filter_symbols_sync(loop, symbols):
    """_filter_tradeable_symbols is async; run it from a sync test."""
    import asyncio
    return asyncio.run(loop._filter_tradeable_symbols(symbols))


def test_filter_symbols_blacklist_skips_new_but_keeps_open_legs():
    """A blacklisted symbol with an OPEN position must still flow through
    management (never abandon an in-flight SL/TP/trail)."""
    loop = _loop(_fake_risk_config(symbol_blacklist={"BADUSDT"}, min_24h_quote_volume_usdt=0))

    open_db = MagicMock()
    oq = MagicMock()
    oq.filter.return_value.distinct.return_value.all.return_value = [("BADUSDT",)]
    open_db.query.return_value = oq

    with patch("backend.services.trading_loop.SessionLocal", return_value=open_db):
        result = _filter_symbols_sync(loop, ["BTCUSDT", "BADUSDT"])

    assert "BTCUSDT" in result
    assert "BADUSDT" in result  # kept for management despite blacklist


def test_filter_symbols_blacklist_removes_new_entries():
    loop = _loop(_fake_risk_config(symbol_blacklist={"BADUSDT"}, min_24h_quote_volume_usdt=0))

    with patch("backend.services.trading_loop.SessionLocal", return_value=_no_open_positions_db()):
        result = _filter_symbols_sync(loop, ["BTCUSDT", "BADUSDT"])

    assert result == ["BTCUSDT"]


def test_filter_symbols_drops_unlisted_futures_for_new_entries():
    """exchangeInfo TRADING set must drop unknown perps before place_order."""
    loop = _loop(_fake_risk_config(min_24h_quote_volume_usdt=0))
    with patch("backend.services.trading_loop.SessionLocal", return_value=_no_open_positions_db()), \
         patch(
             "backend.services.trading_loop.binance_futures_broker.futures_listing_status",
             side_effect=lambda s: True if s == "AVAXUSDT" else False,
         ):
        result = _filter_symbols_sync(loop, ["AVAXUSDT", "NOTAREALUSDT"])
    assert result == ["AVAXUSDT"]


def test_filter_symbols_fails_closed_on_empty_volume_snapshot():
    """Empty ticker map must drop new-entry candidates; open legs stay."""
    loop = _loop(_fake_risk_config(min_24h_quote_volume_usdt=10_000_000))

    with patch("backend.services.trading_loop.SessionLocal", return_value=_no_open_positions_db()), \
         patch("backend.services.trading_loop.binance_market_data.get_all_tickers_24h",
               return_value=[]):
        result = _filter_symbols_sync(loop, ["BTCUSDT", "ETHUSDT"])

    assert result == []


def test_filter_symbols_empty_snapshot_keeps_open_legs_only():
    loop = _loop(_fake_risk_config(min_24h_quote_volume_usdt=10_000_000))

    open_db = MagicMock()
    oq = MagicMock()
    oq.filter.return_value.distinct.return_value.all.return_value = [("ETHUSDT",)]
    open_db.query.return_value = oq

    with patch("backend.services.trading_loop.SessionLocal", return_value=open_db), \
         patch("backend.services.trading_loop.binance_market_data.get_all_tickers_24h",
               return_value=[]):
        result = _filter_symbols_sync(loop, ["BTCUSDT", "ETHUSDT"])

    assert result == ["ETHUSDT"]


def test_filter_symbols_fails_open_on_volume_error():
    """If the 24h volume snapshot raises, keep ALL candidates (fail-open)
    rather than halting trading."""
    loop = _loop(_fake_risk_config(min_24h_quote_volume_usdt=50_000_000))

    with patch("backend.services.trading_loop.SessionLocal", return_value=_no_open_positions_db()), \
         patch("backend.services.trading_loop.binance_market_data.get_all_tickers_24h",
               side_effect=RuntimeError("timeout")):
        result = _filter_symbols_sync(loop, ["BTCUSDT", "ETHUSDT"])

    assert result == ["BTCUSDT", "ETHUSDT"]  # fail-open: nothing dropped


def test_filter_symbols_rejects_illiquid_and_unknown():
    loop = _loop(_fake_risk_config(min_24h_quote_volume_usdt=10_000_000))  # $10M floor

    tickers = [
        {"symbol": "BTCUSDT", "quoteVolume": 5_000_000_000},   # passes
        {"symbol": "ILLIQUSDT", "quoteVolume": 1_000},          # illiquid -> reject
        # GHOSTUSDT absent from snapshot -> unknown/delisted -> reject
    ]
    with patch("backend.services.trading_loop.SessionLocal", return_value=_no_open_positions_db()), \
         patch("backend.services.trading_loop.binance_market_data.get_all_tickers_24h",
               return_value=tickers):
        result = _filter_symbols_sync(loop, ["BTCUSDT", "ILLIQUSDT", "GHOSTUSDT"])

    assert result == ["BTCUSDT"]


def test_filter_symbols_keeps_open_legs_without_binance_ticker():
    """Open FX / unknown symbols must stay in the manage set even when
    Binance has no 24h ticker (otherwise SL/TP maintenance is abandoned)."""
    loop = _loop(_fake_risk_config(min_24h_quote_volume_usdt=10_000_000))

    open_db = MagicMock()
    oq = MagicMock()
    oq.filter.return_value.distinct.return_value.all.return_value = [("EURUSD",), ("ILLIQUSDT",)]
    open_db.query.return_value = oq

    tickers = [
        {"symbol": "BTCUSDT", "quoteVolume": 5_000_000_000},
        {"symbol": "ILLIQUSDT", "quoteVolume": 1_000},
    ]
    with patch("backend.services.trading_loop.SessionLocal", return_value=open_db), \
         patch("backend.services.trading_loop.binance_market_data.get_all_tickers_24h",
               return_value=tickers):
        result = _filter_symbols_sync(loop, ["BTCUSDT", "ILLIQUSDT", "EURUSD", "GHOSTUSDT"])

    assert "BTCUSDT" in result
    assert "ILLIQUSDT" in result  # open, even if illiquid
    assert "EURUSD" in result     # open, even if no Binance ticker
    assert "GHOSTUSDT" not in result


# --------------------------------------------------------------------------- #
# _apply_expectancy_gate — disabled + fail-open + positive blocking
# --------------------------------------------------------------------------- #
def test_expectancy_gate_disabled_returns_candidates_unchanged():
    loop = _loop(_fake_risk_config(symbol_expectancy_gate_enabled=False))
    assert loop._apply_expectancy_gate(["BTCUSDT", "ETHUSDT"]) == ["BTCUSDT", "ETHUSDT"]


def test_expectancy_gate_fails_open_on_db_error():
    loop = _loop(_fake_risk_config(symbol_expectancy_gate_enabled=True))
    with patch("backend.services.trading_loop.SessionLocal", side_effect=RuntimeError("db down")):
        result = loop._apply_expectancy_gate(["BTCUSDT", "ETHUSDT"])
    assert result == ["BTCUSDT", "ETHUSDT"]  # fail-open


def test_expectancy_gate_blocks_negative_pnl_symbols():
    """Symbols with >= min_trades closes and net-negative P&L are dropped for
    NEW entries — but never blocked if they have an open position."""
    loop = _loop(_fake_risk_config(
        symbol_expectancy_gate_enabled=True,
        symbol_expectancy_min_trades=3,
        symbol_expectancy_lookback_days=30,
    ))

    db = MagicMock()
    agg_q = MagicMock()
    agg_q.filter.return_value.group_by.return_value.all.return_value = [
        ("BTCUSDT", 5, -120.0),   # 5 trades, negative -> blocked
        ("ETHUSDT", 2, -50.0),     # < min_trades -> NOT blocked
        ("SOLUSDT", 4, 80.0),     # positive -> not blocked
    ]
    open_q = MagicMock()
    open_q.filter.return_value.distinct.return_value.all.return_value = []  # no open positions

    # First db.query() = aggregate expectancy rows; second = open positions.
    db.query.side_effect = [agg_q, open_q]

    with patch("backend.services.trading_loop.SessionLocal", return_value=db):
        result = loop._apply_expectancy_gate(["BTCUSDT", "ETHUSDT", "SOLUSDT"])

    assert "BTCUSDT" not in result
    assert "ETHUSDT" in result
    assert "SOLUSDT" in result


# --------------------------------------------------------------------------- #
# _reconstruct_cooldowns — restore SL cooldowns, tz-naive -> UTC
# --------------------------------------------------------------------------- #
def test_reconstruct_cooldowns_restores_from_recent_closes():
    loop = _loop()
    now = datetime.now(timezone.utc)
    recent_close = now - timedelta(minutes=5)

    trade = SimpleNamespace(
        symbol="BTCUSDT",
        status="closed",
        closed_at=recent_close.replace(tzinfo=None),  # tz-naive (DB default)
    )

    db = MagicMock()
    q = MagicMock()
    q.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [trade]
    db.query.return_value = q
    db.close.return_value = None

    with patch("backend.services.trading_loop.SessionLocal", return_value=db):
        loop._reconstruct_cooldowns()

    assert "BTCUSDT" in loop._sl_cooldown
    restored = loop._sl_cooldown["BTCUSDT"]
    assert restored.tzinfo is not None  # tz-naive closed_at got UTC


# --------------------------------------------------------------------------- #
# _check_sl_tp — SL/TP hit detection + PnL + already-flat
# --------------------------------------------------------------------------- #
def _open_trade(symbol="BTCUSDT", direction="BUY", qty=0.1, entry=100.0,
                stop_loss=95.0, take_profit=110.0, trade_id=1):
    t = MagicMock()
    t.symbol = symbol
    t.direction = direction
    t.quantity = qty
    t.entry_price = entry
    t.stop_loss = stop_loss
    t.take_profit = take_profit
    t.notes = ""
    t.status = "open"
    t.id = trade_id
    return t


def _run_check_sl_tp(loop, trade, close_price, fake_res):
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [trade]
    bars = [{"close": close_price}]

    with patch.object(loop, "_ensure_exchange_protection"), \
         patch.object(loop, "_apply_trailing_stop"), \
         patch.object(loop, "_apply_partial_tp"), \
         patch.object(loop, "_is_live_binance", return_value=False), \
         patch("backend.services.trading_loop.UnifiedTrading") as UT:
        UT.return_value.place_order.return_value = fake_res
        loop._check_sl_tp(db, "BTCUSDT", bars)
    return UT


def test_check_sl_tp_buy_stop_loss_hit():
    """BUY: price <= stop_loss -> SL hit, position closed via reduce-only SELL."""
    loop = _loop()
    trade = _open_trade(direction="BUY", stop_loss=95.0)
    fake_res = SimpleNamespace(
        success=True, filled_price=94.0, realized_pnl=None,
        commission=0.5, mode="paper", message="ok",
    )
    UT = _run_check_sl_tp(loop, trade, close_price=94.0, fake_res=fake_res)

    assert trade.status == "closed"
    assert trade.closed_at is not None
    # BUY PnL = (exit - entry) * qty - commission = (94-100)*0.1 - 0.5 = -1.1
    assert trade.pnl == pytest.approx(-1.1)
    UT.return_value.place_order.assert_called_once()


def test_check_sl_tp_buy_take_profit_hit():
    loop = _loop()
    trade = _open_trade(direction="BUY", take_profit=110.0)
    fake_res = SimpleNamespace(
        success=True, filled_price=111.0, realized_pnl=None,
        commission=0.2, mode="paper", message="ok",
    )
    UT = _run_check_sl_tp(loop, trade, close_price=111.0, fake_res=fake_res)

    assert trade.status == "closed"
    # (111-100)*0.1 - 0.2 = 0.9
    assert trade.pnl == pytest.approx(0.9)


def test_check_sl_tp_short_take_profit_hit():
    """SHORT: price <= take_profit -> TP hit (short profits when price falls)."""
    loop = _loop()
    trade = _open_trade(direction="SELL", entry=100.0, stop_loss=110.0, take_profit=90.0)
    fake_res = SimpleNamespace(
        success=True, filled_price=88.0, realized_pnl=None,
        commission=0.1, mode="paper", message="ok",
    )
    UT = _run_check_sl_tp(loop, trade, close_price=88.0, fake_res=fake_res)

    assert trade.status == "closed"
    # SHORT PnL = (entry - exit) * qty - commission = (100-88)*0.1 - 0.1 = 1.1
    assert trade.pnl == pytest.approx(1.1)


def test_check_sl_tp_failed_close_leaves_trade_open():
    """A failed close (including unverified -2022) must not close the DB row."""
    loop = _loop()
    trade = _open_trade(direction="BUY", stop_loss=95.0)
    fake_res = SimpleNamespace(
        success=False, filled_price=None, realized_pnl=None,
        commission=0.0, mode="paper", message="-2022: ReduceOnly Order Rejected",
    )
    UT = _run_check_sl_tp(loop, trade, close_price=94.0, fake_res=fake_res)

    UT.return_value.place_order.assert_called_once()
    assert trade.status == "open"
    assert "SL/TP close FAILED" in (trade.notes or "")


def test_check_sl_tp_no_hit_leaves_trade_open():
    """Price between SL and TP -> nothing closed."""
    loop = _loop()
    trade = _open_trade(direction="BUY", stop_loss=95.0, take_profit=110.0)
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [trade]
    bars = [{"close": 102.0}]

    with patch.object(loop, "_ensure_exchange_protection"), \
         patch.object(loop, "_apply_trailing_stop"), \
         patch.object(loop, "_apply_partial_tp"), \
         patch.object(loop, "_is_live_binance", return_value=False), \
         patch("backend.services.trading_loop.UnifiedTrading") as UT:
        loop._check_sl_tp(db, "BTCUSDT", bars)

    assert trade.status != "closed"
    UT.return_value.place_order.assert_not_called()
