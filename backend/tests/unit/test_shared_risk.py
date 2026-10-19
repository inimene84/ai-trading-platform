from types import SimpleNamespace

import pytest

from backend.services.risk_guard import RiskBreach, enforce_risk_limits
from backend.services.risk_config import RiskConfig
from backend.services.shared_risk import (
    books_combinable,
    evaluate_shared_risk,
    net_usd_direction,
    shared_risk_mode,
)


def _trade(**kwargs):
    defaults = {
        "symbol": "BTCUSDC",
        "broker": "binance_futures",
        "direction": "BUY",
        "quantity": 0.01,
        "entry_price": 70000.0,
        "stop_loss": 69300.0,
        "status": "open",
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_default_mode_is_shadow(monkeypatch):
    monkeypatch.delenv("SHARED_RISK_MODE", raising=False)
    assert shared_risk_mode() == "shadow"


def test_split_book_is_not_combinable(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("PAPER_TRADING", "false")
    monkeypatch.setenv("DRY_RUN_ALL", "false")
    monkeypatch.setenv("CTRADER_ENV", "demo")
    monkeypatch.setenv("CTRADER_PAPER_MODE", "false")
    monkeypatch.setenv("BINANCE_TESTNET", "false")
    monkeypatch.setenv("BINANCE_PAPER_PARALLEL", "false")
    assert books_combinable() is False
    result = evaluate_shared_risk([
        _trade(),
        _trade(symbol="EURUSD", broker="ctrader", quantity=0.01, entry_price=1.1, stop_loss=1.09),
    ])
    assert result["combinable"] is False
    assert result["action"] == "allow"
    assert result["combined"] is None
    assert "different money modes" in (result["warning"] or "")
    assert result["books"]["binance_futures"]["positions"] == 1
    assert result["books"]["ctrader"]["positions"] == 1


def test_shadow_would_breach_does_not_block(monkeypatch):
    monkeypatch.setenv("SHARED_RISK_MODE", "shadow")
    monkeypatch.setenv("SHARED_RISK_MAX_POSITIONS", "1")
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.setenv("PAPER_TRADING", "true")
    result = evaluate_shared_risk([
        _trade(symbol="BTCUSDC"),
        _trade(symbol="ETHUSDC", entry_price=3000, stop_loss=2970, quantity=0.1),
    ], combined_equity=10_000)
    assert result["would_breaches"]
    assert result["action"] == "allow"
    assert result["breaches"] == []


def test_enforce_blocks_shared_position_cap(monkeypatch):
    monkeypatch.setenv("SHARED_RISK_MODE", "enforce")
    monkeypatch.setenv("SHARED_RISK_MAX_POSITIONS", "1")
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.setenv("PAPER_TRADING", "true")
    result = evaluate_shared_risk([
        _trade(symbol="BTCUSDC"),
        _trade(symbol="ETHUSDC", entry_price=3000, stop_loss=2970, quantity=0.1),
    ], combined_equity=10_000)
    assert result["action"] == "block"
    assert any("max positions" in b for b in result["breaches"])


def test_enforce_blocks_open_risk_and_net_usd(monkeypatch):
    monkeypatch.setenv("SHARED_RISK_MODE", "enforce")
    monkeypatch.setenv("SHARED_RISK_MAX_POSITIONS", "10")
    monkeypatch.setenv("SHARED_RISK_MAX_OPEN_RISK_PCT", "0.01")
    monkeypatch.setenv("SHARED_RISK_MAX_NET_USD", "1")
    monkeypatch.setenv("TRADING_MODE", "paper")
    result = evaluate_shared_risk(
        [_trade(quantity=1.0, entry_price=100.0, stop_loss=90.0, direction="BUY")],
        combined_equity=100.0,
    )
    assert result["action"] == "block"
    assert result["breaches"]


def test_net_usd_crypto_long_is_short_dollar():
    net = net_usd_direction([_trade(direction="BUY", quantity=0.01, entry_price=10000)])
    assert net < 0
    net_short = net_usd_direction([_trade(direction="SELL", quantity=0.01, entry_price=10000)])
    assert net_short > 0


def test_per_venue_cap_still_fires_when_shared_is_shadow(db_session, monkeypatch):
    """MAX_POSITIONS=2 is unchanged; shared shadow must not swallow it."""
    from backend.database.models import Trade

    monkeypatch.setenv("SHARED_RISK_MODE", "shadow")
    monkeypatch.setenv("SHARED_RISK_MAX_POSITIONS", "99")
    cfg = RiskConfig(
        max_positions=2,
        max_open_positions=2,
        max_portfolio_drawdown_pct=99,
        max_daily_loss_pct=99,
        max_directional_exposure_usdt=0,
    )
    trades = [
        Trade(symbol="BTCUSDT", direction="BUY", quantity=0.1, status="open"),
        Trade(symbol="ETHUSDT", direction="BUY", quantity=1.0, status="open"),
        Trade(symbol="SOLUSDT", direction="BUY", quantity=5.0, status="open"),
    ]
    with __import__("pytest").raises(RiskBreach, match="Max open positions exceeded"):
        enforce_risk_limits(db_session, cfg, trades, None)


def test_ctrader_crypto_exposure_is_coin_notional_not_fx_lots(monkeypatch):
    monkeypatch.setenv("SHARED_RISK_MODE", "shadow")
    monkeypatch.setenv("TRADING_MODE", "paper")
    result = evaluate_shared_risk([
        _trade(symbol="ETHUSD", broker="ctrader", quantity=0.1, entry_price=3000.0, stop_loss=2970.0),
    ], combined_equity=10_000)
    # 0.1 coin * $3000 = $300, not 0.1 * 100_000 FX units.
    assert result["ctrader_crypto_exposure_usd"] == pytest.approx(300.0)
    assert result["books"]["ctrader"]["exposure_usd"] == pytest.approx(300.0)


def test_same_base_coin_blocks_in_enforce_logs_in_shadow(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.setenv("SHARED_RISK_MODE", "shadow")
    trades = [
        _trade(symbol="ETHUSDC", broker="binance_futures", quantity=0.1, entry_price=3000),
        _trade(symbol="ETHUSD", broker="ctrader", quantity=0.1, entry_price=3000, stop_loss=2970),
    ]
    shadow = evaluate_shared_risk(trades, combined_equity=10_000)
    assert shadow["overlapping_bases"] == ["ETH"]
    assert shadow["action"] == "allow"
    assert any("ETH" in b for b in shadow["would_breaches"])

    monkeypatch.setenv("SHARED_RISK_MODE", "enforce")
    enforced = evaluate_shared_risk(trades, combined_equity=10_000)
    assert enforced["action"] == "block"
    assert any("same base coin" in b for b in enforced["breaches"])


def test_off_mode_skips_combined_math(monkeypatch):
    monkeypatch.setenv("SHARED_RISK_MODE", "off")
    result = evaluate_shared_risk([_trade(), _trade(symbol="ETHUSDC")], combined_equity=1)
    assert result["combined"] is None
    assert result["warning"] == "shared_risk_off"
