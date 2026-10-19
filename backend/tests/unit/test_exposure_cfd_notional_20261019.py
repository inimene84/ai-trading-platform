"""Regression: cTrader crypto/metal CFDs must not be sized as 100k-unit FX lots.

Incident 2026-10-19 13:55:43 EEST (cycle 17): "[RISK BREACH] Max directional
exposure exceeded: $805,651.99 > $209,611.57. LOOP STOPPED." The live book was
USDCAD BUY 0.05, EURUSD SELL 0.02, EURAUD SELL 0.01 and ETHUSD BUY 0.1 (IC
demo, volume=10 cents = 0.1 ETH). ETHUSD 0.1 lot was read as 10,000 ETH.
"""

from datetime import datetime, timezone

import pytest

from backend.database.models import Trade, PortfolioSnapshot
from backend.services import risk_guard, shared_risk
from backend.services.ctrader_notional import (
    ctrader_notional_usd,
    ctrader_risk_exposure_usd,
    ctrader_units_per_lot,
    dedupe_open_ctrader_rows,
)
from backend.services.risk_config import RiskConfig
from backend.services.risk_guard import RiskBreach, enforce_risk_limits


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("DISABLE_RISK_GUARD", "false")
    monkeypatch.setenv("ACTIVE_BROKER", "ctrader")
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.delenv("RISK_FX_LEVERAGE", raising=False)
    monkeypatch.delenv("CTRADER_LEVERAGE", raising=False)
    monkeypatch.delenv("RISK_CTRADER_CRYPTO_LEVERAGE", raising=False)
    monkeypatch.setattr(risk_guard, "get_trading_mode", lambda: risk_guard.TradingMode.LIVE)
    monkeypatch.setattr(risk_guard, "is_split_book", lambda: False)


def _ct(tid, symbol, direction, lots, px, pid=None):
    return Trade(
        id=tid, symbol=symbol, direction=direction, quantity=lots, entry_price=px,
        status="open", broker="ctrader", exchange="ctrader", mode="demo",
        broker_position_id=pid,
    )


def _incident_book():
    return [
        _ct(11810, "USDCAD", "BUY", 0.05, 1.40928, "298162690"),
        _ct(11813, "EURUSD", "SELL", 0.02, 1.1214, "298163033"),
        _ct(11814, "EURAUD", "SELL", 0.01, 1.607, "298163034"),
        _ct(11815, "ETHUSD", "BUY", 0.1, 2416.071, "298186080"),
    ]


def _old_formula(lots, px, quote_usd=True, base_usd=False):
    units = lots * 100_000.0
    if base_usd:
        return units / 30.0
    return units * px / 30.0


def test_incident_number_reproduces_with_legacy_formula():
    old = (
        _old_formula(0.05, 1.40928, base_usd=True)
        + _old_formula(0.02, 1.1214)
        + _old_formula(0.01, 1.607)
        + _old_formula(0.1, 2416.071)
    )
    assert round(old, 2) == 805651.99


def test_units_per_lot_per_symbol():
    assert ctrader_units_per_lot("EURUSD") == 100_000.0
    assert ctrader_units_per_lot("USDCAD") == 100_000.0
    assert ctrader_units_per_lot("ETHUSD") == 1.0
    assert ctrader_units_per_lot("BTCUSD") == 1.0
    assert ctrader_units_per_lot("XAUUSD") == 100.0
    assert ctrader_units_per_lot("XAGUSD") == 1_000.0


def test_live_lot_size_spec_wins(monkeypatch):
    from backend.services.ctrader_service import CTraderService

    monkeypatch.setattr(
        CTraderService, "_live_spec", classmethod(lambda cls, s: {"lot_size": 1000} if s == "ETHUSD" else {})
    )
    assert ctrader_units_per_lot("ETHUSD") == 10.0  # 1000 cents = 10 coins/lot
    assert ctrader_units_per_lot("EURUSD") == 100_000.0


def test_ethusd_cfd_notional_is_coin_notional():
    # 0.1 lot = 0.1 ETH (Spotware volume 10 cents)
    assert ctrader_notional_usd("ETHUSD", 0.1, 2416.071) == pytest.approx(241.6071)
    assert ctrader_risk_exposure_usd("ETHUSD", 0.1, 2416.071) == pytest.approx(241.6071)


def test_metals_notional():
    assert ctrader_notional_usd("XAUUSD", 0.01, 4182.91) == pytest.approx(4182.91)
    assert ctrader_notional_usd("XAGUSD", 0.01, 57.71) == pytest.approx(577.1)
    assert ctrader_risk_exposure_usd("XAUUSD", 0.01, 4182.91) == pytest.approx(4182.91 / 30)


def test_fx_notional_unchanged():
    assert ctrader_notional_usd("USDCAD", 0.05, 1.40928) == pytest.approx(5000.0)
    assert ctrader_notional_usd("EURUSD", 0.02, 1.1214) == pytest.approx(2242.8)
    assert risk_guard._fx_notional_usdt("EURUSD", 0.02, 1.1214) == pytest.approx(2242.8)


def test_crypto_leverage_override(monkeypatch):
    monkeypatch.setenv("RISK_CTRADER_CRYPTO_LEVERAGE", "2")
    assert ctrader_risk_exposure_usd("ETHUSD", 0.1, 2416.071) == pytest.approx(120.80355)


def test_incident_book_exposure_is_small():
    exp = risk_guard._directional_exposure_usdt(_incident_book())
    # 166.67 + 74.76 + 53.57 (EURAUD legacy cross fallback, no live marks) + 241.61
    assert exp == pytest.approx(536.60, abs=0.05)
    assert exp < 1_000


def test_incident_book_no_breach_live():
    cfg = RiskConfig(
        max_portfolio_drawdown_pct=100.0, max_daily_loss_pct=100.0,
        max_positions=10, max_open_positions=10,
    )
    cfg.max_directional_exposure_usdt = 5000.0
    cfg.equity_sizing_enabled = True
    cfg.max_direction_notional_equity_mult = 2.0
    snap = PortfolioSnapshot(
        total_value=104805.785, cash=104805.785, timestamp=datetime.now(timezone.utc),
        broker="ctrader", mode="live",
    )
    enforce_risk_limits(None, cfg, _incident_book(), snap)  # must not raise


def test_oversized_cfd_still_breaches():
    cfg = RiskConfig(
        max_portfolio_drawdown_pct=100.0, max_daily_loss_pct=100.0,
        max_positions=10, max_open_positions=10,
    )
    cfg.max_directional_exposure_usdt = 500.0
    cfg.equity_sizing_enabled = False
    book = [_ct(1, "ETHUSD", "BUY", 1.0, 2416.0, "p1")]  # 1 ETH = $2,416
    with pytest.raises(RiskBreach, match="Max directional exposure exceeded"):
        enforce_risk_limits(None, cfg, book, None)


def test_duplicate_open_rows_for_one_position_count_once():
    a = _ct(1, "USDCAD", "BUY", 0.05, 1.40928, "298162690")
    b = _ct(2, "USDCAD", "BUY", 0.05, 1.40925, "298162690")
    assert len(dedupe_open_ctrader_rows([a, b])) == 1
    assert risk_guard._directional_exposure_usdt([a, b]) == pytest.approx(5000 / 30)


def test_binance_crypto_unchanged():
    t = Trade(symbol="ETHUSDC", direction="BUY", quantity=0.02, entry_price=2400.0,
              status="open", broker="binance_futures", exchange="binance_futures")
    assert risk_guard._trade_exposure_usdt(t) == pytest.approx(48.0)
    # brokerless ETHUSD is crypto, not a 6-letter FX pair
    t2 = Trade(symbol="ETHUSD", direction="BUY", quantity=0.1, entry_price=2400.0, status="open")
    assert risk_guard._is_fx_trade(t2) is False
    assert risk_guard._trade_exposure_usdt(t2) == pytest.approx(240.0)


def test_shared_risk_uses_same_notional():
    book = _incident_book() + [_ct(9, "XAUUSD", "SELL", 0.01, 4182.91, "p9")]
    exp = shared_risk._directional_exposure_usdt(book)
    assert exp == pytest.approx(536.60 + 4182.91 / 30, abs=0.05)
