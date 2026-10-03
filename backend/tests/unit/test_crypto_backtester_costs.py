"""Unit tests for cost-aware and partial-TP-aware CryptoBacktestEngine."""

import pytest

from backend.backtesting.crypto_backtester import CryptoBacktestEngine


def _make_history(n: int = 20, base: float = 100.0, tr: float = 2.0):
    bars = []
    for i in range(n):
        bars.append(
            {
                "date": f"2026-01-01T00:{i:02d}:00Z",
                "open": base,
                "high": base + tr / 2.0,
                "low": base - tr / 2.0,
                "close": base,
                "volume": 1000.0,
            }
        )
    return bars


def test_crypto_backtester_charges_roundtrip_costs():
    engine = CryptoBacktestEngine(symbols=["BTCUSDT"], initial_capital=1000.0)
    engine.risk_config.taker_fee_rate = 0.0004
    engine.risk_config.slippage_rate = 0.0002
    # roundtrip_cost_rate = 0.0012 (0.0006 per leg)

    engine._execute_trade("BTCUSDT", "BUY", qty=1.0, price=100.0, sl=95.0, tp=110.0, timestamp="t0")
    engine._close_trade("BTCUSDT", exit_price=100.0, timestamp="t1", reason="SL")

    report = engine._generate_report()
    assert report["cost_aware"] is True
    assert report["total_gross_pnl"] == pytest.approx(0.0)
    assert report["total_fees"] == pytest.approx(0.12)
    assert report["total_pnl"] == pytest.approx(-0.12)
    assert report["final_capital"] == pytest.approx(1000.0 - 0.12)


def test_crypto_backtester_partial_tp_and_be_fees_ratchet():
    engine = CryptoBacktestEngine(symbols=["BTCUSDT"], initial_capital=1000.0)
    engine.risk_config.partial_tp_enabled = True
    engine.risk_config.partial_tp_atr_mult = 1.0
    engine.risk_config.partial_tp_close_pct = 0.50
    engine.risk_config.trailing_stop_enabled = True
    engine.risk_config.trail_activation_atr = 2.0
    engine.risk_config.trail_atr_mult = 0.8
    engine.risk_config.step_trail_enabled = False
    engine.risk_config.taker_fee_rate = 0.0004
    engine.risk_config.slippage_rate = 0.0002

    history = _make_history(20, base=100.0, tr=2.0)  # ATR = 2.0
    engine._execute_trade("BTCUSDT", "BUY", qty=2.0, price=100.0, sl=98.0, tp=110.0, timestamp="t0")

    # Move +1.0 ATR (close = 102.0): partial TP closes 50% (qty=1.0) and ratchets SL to BE + roundtrip fees (100.12)
    candle_partial = {
        "date": "2026-01-01T01:00:00Z",
        "open": 100.0,
        "high": 102.0,
        "low": 100.0,
        "close": 102.0,
        "volume": 1000.0,
    }
    engine._check_exits("BTCUSDT", candle_partial, history + [candle_partial])

    assert "BTCUSDT" in engine.positions
    pos = engine.positions["BTCUSDT"]
    assert pos["partial_tp_done"] is True
    assert pos["qty"] == pytest.approx(1.0)
    assert pos["sl"] == pytest.approx(100.0 * (1.0 + engine.risk_config.roundtrip_cost_rate))
    assert len(engine.trade_history) == 1
    assert engine.trade_history[0]["reason"] == "PARTIAL_TP"
